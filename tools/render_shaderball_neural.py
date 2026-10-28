"""Render the existing neural shaderball at ell = sigma = 0.1 beside the scale sweep."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
from pathlib import Path
import subprocess
import time

import numpy as np

from compare_shaderball_fixed_ratio import (
    ROOT, block_mean, build_report, filmic, load, read_pfm, render_rows,
    roi_stats, save, signed_color, verified, write_pfm, write_png,
)


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def prepare(main, executable):
    reference = load(main / 'manifest.json')
    assert reference['ratio'] == 1 and .1 in reference['lengths']
    base = load(main / 'configs/ell0p1_classic_global.json')
    source = ROOT / 'gpis-scenes/shaderball/knob-neural-l0.02-s0.02.json'
    scene = load(source)
    medium = scene['media'][0]
    medium['model'] = str((source.parent / medium['model']).resolve())
    medium['field']['file'] = str((source.parent / medium['field']['file']).resolve())
    medium['covariance'].update(sigma=.1, lengthScale=.1)
    medium['max_bounces'] = base['media'][0]['max_bounces']
    assert medium['material'] == base['media'][0]['material']
    scene['integrator'] = copy.deepcopy(base['integrator'])
    scene['renderer'] = copy.deepcopy(base['renderer'])
    scene['camera']['resolution'] = base['camera']['resolution']
    assert scene['bsdfs'] == base['bsdfs']
    bake = load(source.parent / 'knob-l0.02-s0.02-dx0.01-bake-source.json')
    assert bake['media'][0]['gaussian_process']['mean'] == base['media'][0]['gaussian_process']['mean']
    sidecar = load(source.parent / 'knob-l0.02-s0.02-dx0.01.nvdb.json')
    transform = sidecar['nvdb_to_scene_transform']
    def to_scene(position):
        return np.asarray(position) * transform['scale'] + transform['position']
    for key in ['position', 'look_at']:
        assert np.allclose(to_scene(scene['camera']['transform'][key]),
                           base['camera']['transform'][key], rtol=0, atol=1e-12)
    assert scene['camera']['transform']['scale'] == [-1, 1, 1]
    assert scene['camera']['transform']['up'] == base['camera']['transform']['up']
    for key in ['fov', 'tonemap', 'reconstruction_filter']:
        assert scene['camera'][key] == base['camera'][key]
    original = {p.get('name', p['type']): p for p in base['primitives']}
    for primitive in scene['primitives']:
        matched = original[primitive.get('name', primitive['type'])]
        assert np.allclose(to_scene(primitive['transform']['position']),
                           matched['transform']['position'], rtol=0, atol=1e-12)
        for key in ['type', 'power', 'bsdf']:
            assert primitive.get(key) == matched.get(key)
        assert primitive['transform'].get('scale') == matched['transform'].get('scale')
    assert original['light']['transform']['rotation'] == [160, 0, 0]
    assert next(p for p in scene['primitives'] if p.get('name') == 'light')['transform']['rotation'] == [-160, 0, 0]
    hashes = dict(executable_sha256=sha256(executable), model_sha256=sha256(Path(medium['model'])),
                  field_sha256=sha256(Path(medium['field']['file'])))
    assert hashes['executable_sha256'] == reference['executable_sha256']
    assert hashes['field_sha256'] == sidecar['nvdb_sha256']
    model = load(Path(medium['model']))
    manifest = dict(ell=.1, sigma=.1, ratio=1, resolution=scene['camera']['resolution'],
                    spp_per_seed=scene['renderer']['spp'], seed_ids=reference['seed_ids'],
                    seed_offset=2000000, mode='neural_causal_renewal',
                    source_scene=str(source), reference_manifest=reference,
                    model=medium['model'], field=medium['field']['file'], **hashes,
                    profile_maximum_step=medium['profile_maximum_step'],
                    nvdb_to_scene_transform=transform, voxel_size=sidecar['dx'],
                    model_config=model['model_config'], gradient_config=model['gradient_config'],
                    checkpoint_sha256=model['checkpoint_sha256'],
                    note='Existing model, unchanged evaluator. Neural uses the existing trilinear SDF '
                         'and native active-node domain; reference modes use the procedural mean and forward cube. '
                         'The existing camera/light coordinate conversion is retained. '
                         'The model was trained with the shader_ball_se_sigma002_r05 dataset, not retrained at sigma=0.1.')
    return scene, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--main', type=Path, default=ROOT / 'outputs/shaderball_fixed_ratio/main')
    parser.add_argument('--threads', type=int, default=28)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    main_path = args.main.resolve()
    output = main_path / 'neural_ell0p1'
    if args.report_only:
        report(main_path)
        return
    executable = ROOT / 'build/Release/tungsten.exe'
    scene, manifest = prepare(main_path, executable)
    if (output / 'manifest.json').exists() and load(output / 'manifest.json') != manifest:
        raise ValueError('Existing neural experiment has different inputs; refusing to overwrite it.')
    save(output / 'manifest.json', manifest)
    save(main_path / 'configs/ell0p1_neural.json', scene)
    rows = []
    for seed in manifest['seed_ids']:
        folder = output / f'seed{seed}'
        render_seed = seed + manifest['seed_offset']
        cached = verified(folder, scene, render_seed, manifest['executable_sha256'])
        if cached:
            record = load(folder / 'result.json')
            cached = all(record.get(key) == manifest[key] for key in ['model_sha256', 'field_sha256'])
        if cached:
            print(f'Reuse neural seed={seed}', flush=True)
        elif args.render:
            save(folder / 'scene.json', scene)
            command = [str(executable), '-t', str(args.threads), '-s', str(render_seed),
                       '-d', str(folder), str(folder / 'scene.json')]
            print(f'Render neural ell=sigma=0.1 seed={seed}, {manifest["spp_per_seed"]} spp', flush=True)
            start = time.monotonic()
            with (folder / 'render.log').open('w', encoding='utf-8') as log:
                process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            if process.returncode:
                raise RuntimeError(f'Render failed: {folder / "render.log"}')
            record = dict(ell=.1, sigma=.1, mode=manifest['mode'], seed_id=seed,
                          render_seed=render_seed, spp=manifest['spp_per_seed'],
                          seconds=time.monotonic() - start, folder=str(folder), command=command,
                          **{key: manifest[key] for key in ['executable_sha256', 'model_sha256', 'field_sha256']})
            save(folder / 'result.json', record)
            if not verified(folder, scene, render_seed, manifest['executable_sha256']):
                raise RuntimeError(f'Invalid render: {folder / "render.log"}')
            print(f'Finished neural seed={seed} in {record["seconds"]:.2f}s', flush=True)
        else:
            continue
        rows.append(record)
        save(output / 'render_rows.json', rows)
    if rows:
        report(main_path)


def report(main_path):
    output = main_path / 'neural_ell0p1'
    scene, expected = prepare(main_path, ROOT / 'build/Release/tungsten.exe')
    manifest = load(output / 'manifest.json')
    if manifest != expected:
        raise ValueError('Neural input provenance differs from the saved manifest.')
    rows = load(output / 'render_rows.json')
    seeds = [row['seed_id'] for row in rows]
    if len(set(seeds)) != len(seeds) or not set(seeds).issubset(manifest['seed_ids']):
        raise ValueError('Invalid or duplicate neural seeds.')
    for row in rows:
        assert row['spp'] == manifest['spp_per_seed']
        assert all(row[key] == manifest[key] for key in ['executable_sha256', 'model_sha256', 'field_sha256'])
        assert verified(Path(row['folder']), scene, row['seed_id'] + manifest['seed_offset'],
                        manifest['executable_sha256'])
    selected = {'neural': rows}
    for mode in ['classic_global', 'gpis_global']:
        selected[mode] = [row for row in render_rows(main_path) if row['ell'] == .1 and row['mode'] == mode]
        assert sorted(row['seed_id'] for row in selected[mode]) == sorted(manifest['seed_ids'])
        reference_scene = load(main_path / f'configs/ell0p1_{mode}.json')
        for row in selected[mode]:
            assert row['spp'] == manifest['spp_per_seed']
            offset = 1000000 if mode == 'gpis_global' else 0
            assert verified(Path(row['folder']), reference_scene, row['seed_id'] + offset,
                            manifest['executable_sha256'])
    stacks = {mode: np.stack([read_pfm(Path(r['folder']) / 'render.pfm') for r in entries])
              for mode, entries in selected.items()}
    assert all(a.shape[1:] == (*manifest['resolution'][::-1], 3) for a in stacks.values())
    summary = dict(manifest=manifest, output_directory=str(output), completed_renders=len(rows), modes={}, comparisons={},
                   statistics='Linear Rec.709 luminance; independent render seeds between modes. '
                              'Welch 95% intervals describe sampling uncertainty only.',
                   display='Linear radiance average before the common filmic map; no denoising.',
                   difference_scale=.15)
    prefix = 'neural_ell0p1'
    images, differences, table, csv_rows = [], [], [], []
    labels = {'classic_global': 'classic_global', 'neural': 'neural', 'gpis_global': 'GPIS global'}
    def image_card(title, filename, pfm=None):
        raw = f'{prefix}/averages/{filename}.png'
        pooled = f'{prefix}/averages/{filename}_block2.png'
        link = f' · <a href="{prefix}/averages/{pfm}.pfm">线性 PFM</a>' if pfm else ''
        return (f'<figure><figcaption>{title}{link}</figcaption><a href="{raw}">'
                f'<img class="switchable" src="{raw}" data-raw="{raw}" data-block="{pooled}" '
                f'alt="{title}"></a></figure>')
    for mode in ['classic_global', 'neural', 'gpis_global']:
        average = stacks[mode].mean(axis=0)
        write_pfm(output / 'averages' / f'{mode}.pfm', average)
        write_png(output / 'averages' / f'{mode}.png', filmic(average))
        write_png(output / 'averages' / f'{mode}_block2.png', filmic(block_mean(average)))
        spp = sum(row['spp'] for row in selected[mode])
        summary['modes'][mode] = dict(seed_ids=[row['seed_id'] for row in selected[mode]],
                                     spp_total=spp, seconds=sum(row['seconds'] for row in selected[mode]),
                                     image=f'averages/{mode}.png', pfm=f'averages/{mode}.pfm')
        images.append(image_card(f'{labels[mode]} · {spp} spp', mode, mode))
    for reference in ['classic_global', 'gpis_global']:
        rois = {}
        for name, roi in {'full': [0, 0, 1, 1], **manifest['reference_manifest']['rois']}.items():
            stats = roi_stats(stacks[reference], stacks['neural'], roi)
            # The shared statistic routine uses historical classic/gpis argument names.
            stats = {key.replace('classic', 'reference').replace('gpis', 'neural'): value
                     for key, value in stats.items()}
            rois[name] = stats
            ci = stats.get('difference_ci95')
            missing_ci = '单个 seed；未估计' if len(manifest['seed_ids']) == 1 else '等待至少 2 个 neural seed'
            confidence = f'[{ci[0]:+.5f}, {ci[1]:+.5f}]' if ci else missing_ci
            if name != 'full':
                table.append(f'<tr><td>neural / {labels[reference]}</td><td>{name}</td>'
                             f'<td>{stats["reference_mean"]:.5f}</td><td>{stats["neural_mean"]:.5f}</td>'
                             f'<td>{stats["relative_difference"]:+.2%}</td><td>{confidence}</td></tr>')
            csv_rows.append([reference, name, stats['reference_mean'], stats['neural_mean'],
                             stats['relative_difference'], stats['difference'], *(ci or ['', ''])])
        summary['comparisons'][reference] = rois
        delta = (stacks['neural'].mean(axis=0) - stacks[reference].mean(axis=0)) @ np.array([.2126, .7152, .0722])
        filename = f'neural_minus_{reference}'
        write_png(output / 'averages' / f'{filename}.png', signed_color(delta))
        write_png(output / 'averages' / f'{filename}_block2.png', signed_color(block_mean(delta[..., None])[..., 0]))
        differences.append(image_card(f'neural − {labels[reference]} · 线性亮度 ±0.15', filename))
    save(output / 'summary.json', summary)
    with (output / 'roi_statistics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['reference', 'roi', 'reference_mean', 'neural_mean', 'relative_difference',
                         'difference', 'difference_ci95_low', 'difference_ci95_high'])
        writer.writerows(csv_rows)
    fragment = (f'<section id="neural-ell0p1"><h2>Neural 对照 · ℓ = σ = 0.1</h2>'
                f'<p>neural 已完成 {len(rows)} / {len(manifest["seed_ids"])} 个 seed；'
                f'{manifest["resolution"][0]}×{manifest["resolution"][1]}，每个 seed {manifest["spp_per_seed"]} spp。相同相机视图、光照、绿色镜面材质、'
                '20 次反射上限与采样设置。各模式使用独立随机种子，先在线性空间平均，再统一 filmic 映射。</p>'
                '<div class="images neural-images">' + ''.join(images) + '</div>'
                '<p class="muted">neural 复用现有模型和 dx=0.01 的三线性 SDF；相机、地板、灯光已按原有坐标转换对齐。'
                'classic / GPIS 使用 procedural mean 与包围盒，neural 使用 SDF 活跃节点范围。'
                '训练记录为 shader_ball_se_sigma002_r05；本次只将运行时 ℓ、σ 设为 0.1，未重新训练。'
                '随机场求值与模型权重均未修改。该对照同时包含模型近似和均值场离散化的影响。</p>'
                f'<p><a href="{prefix}/comparison.html">独立对比页</a> · '
                f'<a href="{prefix}/manifest.json">配置与资源哈希</a> · '
                f'<a href="{prefix}/summary.json">完整统计</a> · '
                f'<a href="{prefix}/roi_statistics.csv">区域统计 CSV</a></p>'
                '<div class="images neural-differences">' + ''.join(differences) + '</div>'
                '<p>蓝色为 neural 更暗，橙色为 neural 更亮。百分比为 neural / reference − 1；'
                '95% 区间针对绝对线性亮度差，只反映重复采样不确定性。</p>'
                '<table><thead><tr><th>对比</th><th>区域</th><th>reference</th><th>neural</th>'
                '<th>相对差</th><th>差值 95% 区间</th></tr></thead><tbody>' + ''.join(table) + '</tbody></table></section>'
                '<style>.images.neural-images{grid-template-columns:repeat(3,minmax(0,1fr))}'
                '.images.neural-differences{grid-template-columns:repeat(2,minmax(0,1fr));margin-top:20px}'
                '@media(max-width:650px){.images.neural-images{grid-template-columns:1fr}}</style>')
    (output / 'section.html').write_text(fragment, encoding='utf-8')
    page = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><base href="../">'
            '<title>Shaderball neural · ℓ = σ = 0.1</title><style>'
            'body{max-width:1200px;margin:28px auto;padding:0 20px;font:15px/1.6 system-ui;color:#233145;background:#f4f6f9}'
            'section{background:white;padding:20px;border-radius:8px}.images{display:grid;gap:16px}'
            'figure{margin:0}figcaption{font-size:14px}img{width:100%}table{border-collapse:collapse;width:100%}'
            'th,td{border-bottom:1px solid #d9e0e8;padding:8px;text-align:right}.muted{color:#56677b}a{color:#225ca0}'
            '</style><p><a href="comparison.html#neural-ell0p1">返回固定 σ/ℓ 的完整实验</a></p>'
            '<label><input id="block" type="checkbox"> 以 2×2 像素均值显示</label>' + fragment +
            '<script>document.getElementById("block").addEventListener("change",e=>{'
            'document.querySelectorAll(".switchable").forEach(img=>img.src=e.target.checked?img.dataset.block:img.dataset.raw)})'
            '</script></html>')
    (output / 'comparison.html').write_text(page, encoding='utf-8')
    build_report(main_path)
    print(f'Neural report: {output / "comparison.html"} ({len(rows)} seeds)', flush=True)


if __name__ == '__main__':
    main()
