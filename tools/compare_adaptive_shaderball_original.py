"""Compare adaptive function-space GPIS on the archived full Renewal+ shaderball.

Preserves the original floor, area emitter, environment, mirror material,
camera and integrator. Uses only the Python standard library.
"""
import argparse
import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

from compare_adaptive_gpis import ROOT, pfm

SOURCE = ROOT/'outputs/gpis_shaderball_compare/ell0p2_sigma0p02/renewal_3d_shadow_conditioned/sparse_3d_renewal_plus/seed20261024/scene.json'
OUTPUT = ROOT/'outputs/adaptive_gpis_double_20261114/experiment'
LEGACY_KEYS = ('sample_points', 'step_size_cov', 'step_size', 'seed', 'impulse_density',
               'single_realization', '1D_sampling', 'isotropic_3D_sampling',
               '1D_sampling_scheme', '1D_gradient_correlationXY', 'additional_phase_functions')
FATAL = re.compile(r'Caught an internal error|Unable to load|unrecoverable|Gradient invalid|'
    r'Wrong normal direction|Failed to sample gradient|Gradient zero|Conditioning .* (?:error|failed)|'
    r'GPContext does not exist|Clamp one (?:inf|nan) sample|invalid FP|terminate called|nonfinite', re.I)
LABELS = {'renewal_3d': '原 GPIS 3D Renewal+', 'adaptive': '自适应 function-space GPIS',
          'refined': '自适应：更严格精度'}


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def make_scene(source, mode, size, spp):
    scene = copy.deepcopy(source)
    medium = scene['media'][0]
    if mode != 'renewal_3d':
        # Standard id=0 always selects the first phase. The archived second
        # phase is unreferenced; the single-process adaptive type rejects it.
        assert medium['gaussian_process']['type'] == 'standard'
        assert medium['gaussian_process'].get('id', 0) == 0
        step = medium['step_size']
        for key in LEGACY_KEYS:
            medium.pop(key, None)
        medium['type'] = 'adaptive_function_space_gaussian_process'
        medium['adaptive'] = dict(probability_tolerance=.001, minimum_step=step,
            root_tolerance=1e-5, rank_tolerance=1e-10, max_samples=512,
            max_steps=100000, statistics=True)
        if mode == 'refined':
            medium['adaptive'].update(minimum_step=step*.5, root_tolerance=5e-6,
                                      rank_tolerance=1e-12)
    scene['camera']['resolution'] = [size, size]
    scene['renderer'].pop('output_directory', None)
    scene['renderer'].update(spp=spp, spp_step=min(2, spp), enable_resume_render=False)
    # These equality checks define what "the original scene" means here.
    for key in ('primitives', 'bsdfs', 'integrator'):
        assert scene[key] == source[key], key
    camera = copy.deepcopy(scene['camera'])
    camera['resolution'] = source['camera']['resolution']
    assert camera == source['camera']
    for key in ('gaussian_process', 'phase_function', 'max_bounces', 'sigma_a', 'sigma_s',
                'density', 'correlation_context'):
        assert medium[key] == source['media'][0][key], key
    return scene


def differences(before, after, path=''):
    if isinstance(before, dict) and isinstance(after, dict):
        changes = []
        for key in sorted(before.keys() | after.keys()):
            changes += differences(before.get(key), after.get(key), path+'/'+key)
        return changes
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        return [c for i, (a, b) in enumerate(zip(before, after))
                for c in differences(a, b, path+'/'+str(i))]
    return [] if before == after else [dict(path=path, before=before, after=after)]


def image_stats(folder, size):
    w, h, rgb = pfm(folder/'render.pfm')
    wa, ha, a = pfm(folder/'samplesA.pfm')
    wb, hb, b = pfm(folder/'samplesB.pfm')
    assert (w, h) == (wa, ha) == (wb, hb) == (size, size)
    assert all(math.isclose(x, (y+z)*.5, rel_tol=2e-5, abs_tol=2e-6)
               for x, y, z in zip(rgb, a, b)), 'Half-buffer normalization'
    weights = (.2126, .7152, .0722)
    lum = [sum(rgb[i+k]*weights[k] for k in range(3)) for i in range(0, len(rgb), 3)]
    noise = [sum((a[i+k]-b[i+k])*.5*weights[k] for k in range(3))
             for i in range(0, len(rgb), 3)]
    # PFM rows run bottom to top; keep the original shaderball bounding box.
    roi = [y*w+x for y in range(h) for x in range(w)
           if .14 <= (x+.5)/w <= .86 and .10 <= 1-(y+.5)/h <= .94]
    return dict(mean_luminance=sum(lum)/len(lum), maximum_rgb=max(rgb),
                noise_rms=math.sqrt(sum(x*x for x in noise)/len(noise)),
                object_box_mean_luminance=sum(lum[i] for i in roi)/len(roi),
                object_box_noise_rms=math.sqrt(sum(noise[i]**2 for i in roi)/len(roi)))


def report(output, manifest):
    rows = []
    for mode in LABELS:
        path = output/mode/'result.json'
        if not path.exists():
            continue
        row = read(path)
        for name, digest in row['sha256'].items():
            assert sha(path.parent/name) == digest, path.parent/name
        log = (path.parent/'render.log').read_text(encoding='utf-8', errors='replace')
        row['warnings'] = dict(zero_length_path_terminations=log.count(
            'Path tracing bounces ended with farT == 0'),
            nee_sanity_checks=log.count('1D NEE sanity check failed'))
        rows.append(row)
    save(output/'report.json', dict(manifest=manifest, runs=rows))
    cards = []
    for row in rows:
        mode = row['mode']
        stats = row['statistics']
        details = ''
        if stats:
            details = (f'<p>查询/射线：{stats["queries"]/stats["rays"]:.2f}；'
                       f'异常梯度：{stats["invalid_gradients"]}；'
                       f'不透明起点：{stats["invalid_births"]}；')
            if 'arithmetic' in stats:
                details += f'计算精度：{html.escape(stats["arithmetic"])}</p>'
            else:
                details += f'高精度射线：{stats["high_precision_rays"]}/{stats["rays"]}</p>'
        warnings = row['warnings']
        details += (f'<p>日志提示：零长度片段终止 {warnings["zero_length_path_terminations"]}；'
                    f'NEE 一致性告警 {warnings["nee_sanity_checks"]}</p>')
        cards.append(f'<article><h2>{LABELS[mode]}</h2><a href="{mode}/render.png">'
            f'<img src="{mode}/render.png" alt="{LABELS[mode]}"></a>'
            f'<p>渲染 {row["render_seconds"]:.2f} s；平均亮度 {row["mean_luminance"]:.6f}<br>'
            f'半样本噪声 RMS {row["noise_rms"]:.6f}；物体区域 {row["object_box_noise_rms"]:.6f}</p>'
            f'{details}<p><a href="{mode}/scene.json">配置</a> · '
            f'<a href="{mode}/render.log">日志</a> · <a href="{mode}/render.pfm">线性 PFM</a> · '
            f'<a href="{mode}/changes.json">相对存档的全部配置差异</a></p></article>')
    source_link = Path(os.path.relpath(Path(manifest['source']).parent/'render.png', output)).as_posix()
    refined_note = ('更严格设置将步长和根容差减半，数值秩容差缩小 100 倍。'
                    if any(row['mode'] == 'refined' for row in rows) else '')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>原始 Shaderball · Adaptive GPIS / Renewal+</title>'
        '<style>body{max-width:1500px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;'
        'color:#203044;background:#f5f7fa}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(340px,1fr));gap:24px}'
        'article{background:white;padding:20px;border-radius:8px}img{width:100%;image-rendering:pixelated}'
        'h2{font-size:20px}a{color:#2166b6}.note{background:#e8eef7;padding:18px}code{overflow-wrap:anywhere}</style>'
        '<h1>原始 Shaderball：自适应 GPIS 与 GPIS 3D Renewal+</h1>'
        '<div class="note">沿用已跑过的完整 Renewal+ 场景：绿色镜面 shaderball、白色地面、'
        '大面积矩形灯、环境光。ℓ=0.2，σ=0.02，原相机、filmic 显示、20 次路径反射上限。</div>'
        f'<p>本轮统一 {manifest["size"]}×{manifest["size"]}，{manifest["spp"]} spp，'
        f'seed {manifest["seed"]}，{manifest["threads"]} 线程，同一程序。'
        '原方法按存档使用 3D Renewal+、step=0.01；自适应主设置 minimum_step=0.01，'
        f'根容差 1e-5，数值秩容差 1e-10。{refined_note}</p>'
        '<p>相机、几何、灯光、材质、GP 参数和积分器已逐项检查。只更换介质实现与必要配置，'
        '并调整输出路径、分辨率和采样数。原始 id=0 不使用的备用相位仅在新介质配置中移除。</p>'
        '<p>两者都是采样到法线后的镜面反射，此处未引入 1D MIS。有限脉冲的 sparse-convolution '
        '与在线高斯条件采样仍有模型和数值差异；低 spp 图片、半样本噪声与耗时不构成精确真值或论文加速复现。'
        '计算精度在每个方法的统计中标明。不透明起点计数包含初始采样落入实体的路径，'
        '应与异常梯度和内部数值错误分别看待。</p>'
        f'<p>来源：<code>{html.escape(manifest["source"])}</code><br>'
        f'<a href="source.json">原始配置快照</a> · <a href="{source_link}">原先 256×256 / 128 spp 的记录图</a> · '
        '<a href="report.json">完整数据</a> · <a href="manifest.json">来源、程序与运行参数</a></p>'
        '<main>'+''.join(cards)+'</main></html>')
    (output/'comparison.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--executable', type=Path, default=ROOT/'build/Release/tungsten.exe')
    parser.add_argument('--modes', nargs='+', choices=list(LABELS), default=['renewal_3d', 'adaptive'])
    parser.add_argument('--size', type=int, default=64)
    parser.add_argument('--spp', type=int, default=16)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--timeout', type=int, default=3600)
    args = parser.parse_args()
    if min(args.size, args.spp, args.threads, args.timeout) <= 0 or args.spp % 2:
        parser.error('Require positive settings and an even spp for half-buffer diagnostics')
    source_path, output, exe = args.source.resolve(), args.output.resolve(), args.executable.resolve()
    source = read(source_path)
    archived = read(source_path.parent/'result.json')
    assert sha(source_path) == archived['sha256']['scene.json'], 'Archived scene hash mismatch'
    medium = source['media'][0]
    assert medium['type'] == 'sparse_conv_noise' and medium['correlation_context'] == 'renewal+'
    assert medium['1D_sampling'] is False and medium['1D_sampling_scheme'] == 'uni'
    assert medium['phase_function']['bsdf']['type'] == 'mirror'
    assert [p['type'] for p in source['primitives']] == ['quad', 'cube', 'quad', 'infinite_sphere']
    manifest = dict(source=str(source_path), source_sha256=sha(source_path),
        archived_result_sha256=sha(source_path.parent/'result.json'), executable=str(exe),
        runtime_sha256={p.name:sha(p) for p in [exe, *sorted(exe.parent.glob('*.dll'))]},
        size=args.size, spp=args.spp, threads=args.threads, seed=args.seed)
    if (output/'manifest.json').exists():
        assert read(output/'manifest.json') == manifest, 'Inputs changed; select a new output directory'
    save(output/'manifest.json', manifest)
    save(output/'source.json', source)
    for mode in args.modes:
        folder = output/mode
        scene = make_scene(source, mode, args.size, args.spp)
        identity = dict(scene=scene, manifest=manifest)
        if (folder/'result.json').exists():
            assert read(folder/'result.json')['identity'] == identity
            print('REUSE', mode, flush=True)
            continue
        save(folder/'scene.json', scene)
        save(folder/'changes.json', differences(source, scene))
        command = [str(exe), '-t', str(args.threads), '-s', str(args.seed),
                   '-d', str(folder), str(folder/'scene.json')]
        save(folder/'command.json', command)
        print('RENDER', mode, args.size, args.spp, flush=True)
        begin = time.perf_counter()
        with (folder/'render.log').open('wb') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        wall = time.perf_counter()-begin
        assert sha(exe) == manifest['runtime_sha256'][exe.name], 'Renderer changed during run'
        log = (folder/'render.log').read_text(encoding='utf-8', errors='replace')
        if completed.returncode or FATAL.search(log) or not re.search(rf'Completed {args.spp}/{args.spp} spp', log):
            raise RuntimeError(f'Render failed: {folder / "render.log"}')
        counters = re.findall(r'^Adaptive GPIS statistics: (.+)$', log, re.M)
        stats = json.loads(counters[-1]) if counters else None
        if mode != 'renewal_3d':
            assert stats and stats['hits'] > 0, 'Adaptive GPIS was not exercised'
        row = dict(identity=identity, mode=mode, wall_seconds=wall,
            render_seconds=float(re.findall(r'Completed \d+/\d+ spp, ([\d.]+)/', log)[-1]),
            statistics=stats, **image_stats(folder, args.size),
            sha256={name:sha(folder/name) for name in ('scene.json', 'changes.json', 'command.json',
                'render.png', 'render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm', 'render.log')})
        save(folder/'result.json', row)
        report(output, manifest)
        print('DONE', mode, f'{row["render_seconds"]:.2f} s', flush=True)
    report(output, manifest)
    print(output/'comparison.html', flush=True)


if __name__ == '__main__':
    main()
