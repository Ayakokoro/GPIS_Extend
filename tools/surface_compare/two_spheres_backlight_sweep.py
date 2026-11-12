"""Scan covariance parameters without changing the two-sphere lighting setup.

python -B tools/surface_compare/two_spheres_backlight_sweep.py run
python -B tools/surface_compare/two_spheres_backlight_sweep.py report

Four matched methods, including the original dark baseline:
python -B tools/surface_compare/two_spheres_backlight_sweep.py run --stage methods --size 64 --spp 16 --pairs 0.08:0.2 0.32:0.2 0.32:0.8 --include-classic

Each stage has immutable render settings. New settings require a new stage.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import subprocess
import time

import two_spheres_backlight as base
import numpy as np
from PIL import Image, ImageDraw

OUT = base.OUT / 'sweep'
SIGMAS = (.08, .16, .32)
LENGTHS = (.05, .2, .8, 1.6)
WEIGHTS = np.array([.2126, .7152, .0722])
CLASSIC_MODES = ('classic_global', 'global_conditional')
LABELS = dict(base.LABELS, classic_global='classic_global', global_conditional='global_conditional')


def method_scene(mode):
    scene = base.scene(mode if mode in base.MODES else 'global', 'preview')
    if mode in CLASSIC_MODES:
        native = scene['media'][0]
        scene['media'][0] = dict(name=native['name'], type=mode,
            gaussian_process=copy.deepcopy(native['gaussian_process']),
            material=copy.deepcopy(native['phase_function']['bsdf']),
            max_bounces=native['max_bounces'], sdf_lipschitz=1.001, gradient_bound=1.001)
    return scene


def tag(sigma, ell):
    return f's{sigma:g}_l{ell:g}'.replace('.', 'p')


def prepare(args):
    folder = OUT / args.stage
    pairs = [tuple(map(float, value.split(':'))) for value in args.pairs] if args.pairs else [
        (s, ell) for s in args.sigmas for ell in args.lengths]
    assert all(len(pair)==2 and all(v>0 and math.isfinite(v) for v in pair) for pair in pairs)
    cases = [dict(name=tag(s, ell), sigma=s, lengthScale=ell) for s, ell in pairs]
    manifest = dict(size=args.size, spp=args.spp, seeds=args.seeds, step=args.step,
        cases=cases, renderer_sha256=base.sha(base.EXE),
        base_scene_sha256=base.sha(base.OUT/'preview/global/scene.json'),
        unchanged='Geometry, camera, light, material, impulse density and integrator.',
        pair_difference='correlation_context only', step_convergence_established=False)
    modes = tuple(args.models) if args.models else (base.MODES + CLASSIC_MODES if args.include_classic else base.MODES)
    assert modes and len(set(modes))==len(modes)
    has_classic = any(mode in CLASSIC_MODES for mode in modes)
    if has_classic:
        manifest['pair_difference'] = 'medium type, transport and gradient law'
    if has_classic or modes != base.MODES:
        manifest.update(modes=list(modes), shared_analytic_sdf=True,
            method_difference='Transport/gradient laws differ; shared mean, covariance, material and non-medium settings.',
            source_sha256={str(p.relative_to(base.ROOT)):base.sha(p) for p in [
                base.ROOT/'src/core/math/SdfFunctionsQueries.cpp', base.ROOT/'src/core/math/ProceduralMeanSdf.cpp']})
    path = folder/'manifest.json'
    if path.exists():
        existing = base.read(path)
        # Correct the description of mixed-method comparisons without changing
        # any immutable render settings or render provenance.
        if has_classic and existing.get('pair_difference') == 'correlation_context only':
            existing['pair_difference'] = manifest['pair_difference']
            assert existing == manifest, 'Stage settings changed; use a new --stage'
            base.save(path, existing)
        assert existing == manifest, 'Stage settings changed; use a new --stage'
        return manifest
    for case in cases:
        pair = []
        for mode in modes:
            scene = method_scene(mode)
            gp = scene['media'][0]['gaussian_process']
            gp['covariance'].update(sigma=case['sigma'], lengthScale=case['lengthScale'])
            if mode in base.MODES:
                scene['media'][0]['step_size'] = args.step
            scene['camera']['resolution'] = [args.size, args.size]
            scene['renderer'].update(spp=args.spp, spp_step=min(8, args.spp))
            pair.append(scene)
            for seed in args.seeds:
                base.save(folder/case['name']/mode/f'seed{seed}'/'scene.json', scene)
        by_mode = dict(zip(modes, pair))
        if all(mode in by_mode for mode in base.MODES):
            matched = copy.deepcopy(by_mode['renewal_plus'])
            matched['media'][0]['correlation_context'] = 'global'
            assert by_mode['global'] == matched
        first_material = pair[0]['media'][0].get('material', pair[0]['media'][0].get('phase_function', {}).get('bsdf'))
        for compared in pair[1:]:
            assert {k:v for k,v in compared.items() if k!='media'} == {k:v for k,v in pair[0].items() if k!='media'}
            assert compared['media'][0]['gaussian_process'] == pair[0]['media'][0]['gaussian_process']
            material = compared['media'][0].get('material', compared['media'][0].get('phase_function', {}).get('bsdf'))
            assert material == first_material
        restored = copy.deepcopy(pair[0])
        reference = method_scene(modes[0])
        restored['media'][0]['gaussian_process']['covariance'] = reference['media'][0]['gaussian_process']['covariance']
        if modes[0] in base.MODES:
            restored['media'][0]['step_size'] = reference['media'][0]['step_size']
        restored['camera']['resolution'] = reference['camera']['resolution']
        restored['renderer'] = reference['renderer']
        assert restored == reference, 'Unexpected change beyond parameters and render quality'
    base.save(path, manifest)
    if has_classic:
        for mode in CLASSIC_MODES:
            base.save(base.ROOT/'data/example-scenes/two-spheres-backlight'/(mode.replace('_','-')+'.json'), method_scene(mode))
    print('Prepared', len(cases), 'parameter cases;', len(cases)*len(modes)*len(args.seeds), 'renders', flush=True)
    return manifest


def render(args):
    manifest = prepare(args)
    folder = OUT/args.stage
    if args.reuse_stage:
        source_stage = OUT/args.reuse_stage
        source_manifest = base.read(source_stage/'manifest.json')
        assert source_manifest['renderer_sha256']==manifest['renderer_sha256'], 'Reuse requires the same renderer'
        for case in manifest['cases']:
            for mode in manifest.get('modes', base.MODES):
                for seed in manifest['seeds']:
                    source = source_stage/case['name']/mode/f'seed{seed}'
                    dest = folder/case['name']/mode/f'seed{seed}'
                    if (dest/'run.json').exists() or not (source/'run.json').exists():
                        continue
                    record = base.read(source/'run.json')
                    assert record.get('finite') and record['seed']==seed
                    assert record['renderer_sha256']==manifest['renderer_sha256']
                    assert record['scene_sha256']==base.sha(source/'scene.json')==base.sha(dest/'scene.json'), 'Reuse scene mismatch'
                    assert record['image_sha256']==base.sha(source/'render.exr'), 'Reuse image mismatch'
                    for name in ('render.exr','render.pfm','render.png','render.log'):
                        shutil.copy2(source/name, dest/name)
                    record['reused_from'] = str(source.relative_to(base.ROOT))
                    base.save(dest/'run.json', record)
    for case in manifest['cases']:
        for seed in manifest['seeds']:
            for mode in manifest.get('modes', base.MODES):
                dest = folder/case['name']/mode/f'seed{seed}'
                if (dest/'run.json').exists():
                    old = base.read(dest/'run.json')
                    if (old.get('finite') and old['scene_sha256'] == base.sha(dest/'scene.json')
                            and old['renderer_sha256']==manifest['renderer_sha256'] and (dest/'render.pfm').exists()):
                        print('REUSE', case['name'], mode, seed, flush=True)
                        continue
                assert base.sha(base.EXE) == manifest['renderer_sha256']
                cmd = [str(base.EXE), '-t', str(args.threads), '-s', str(seed), '-d', str(dest), str(dest/'scene.json')]
                print('START', args.stage, case['name'], mode, 'seed', seed, flush=True)
                start = time.monotonic()
                with (dest/'render.log').open('wb') as log:
                    proc = subprocess.Popen(cmd, cwd=base.ROOT, stdout=log, stderr=subprocess.STDOUT,
                        creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
                    while proc.poll() is None:
                        try:
                            proc.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            print('RUNNING', case['name'], mode, round(time.monotonic()-start), 's', flush=True)
                assert proc.returncode == 0, 'Renderer failed: ' + str(dest/'render.log')
                log = (dest/'render.log').read_text(encoding='utf-8', errors='replace')
                assert f'Completed {args.spp}/{args.spp} spp' in log and 'Finished render.' in log
                assert 'Caught an internal error' not in log
                subprocess.run([str(base.ROOT/'build/Release/hdrmanip.exe'), '-o', str(dest/'render.pfm'),
                    str(dest/'render.exr')], check=True, capture_output=True)
                rgb = base.pfm(dest/'render.pfm')
                assert rgb.shape == (args.size, args.size, 3)
                y = rgb @ WEIGHTS
                # A fixed camera-space crop avoids choosing a bright ROI after
                # seeing the result. The central half lies inside the mean front sphere.
                a, b = args.size//4, 3*args.size//4
                central = y[a:b, a:b]
                record = dict(seconds=time.monotonic()-start, exit_code=proc.returncode,
                    seed=seed, sigma=case['sigma'], lengthScale=case['lengthScale'],
                    mean_luminance=float(y.mean()), max_luminance=float(y.max()),
                    central_mean_luminance=float(central.mean()), central_nonzero_pixels=int((central>0).sum()),
                    nonzero_pixels=int((y>0).sum()), finite=True,
                    scene_sha256=base.sha(dest/'scene.json'), image_sha256=base.sha(dest/'render.exr'),
                    renderer_sha256=base.sha(base.EXE), command=cmd)
                discarded = re.search(r'GPIS conditional discarded surface births: (\{[^\n]+\})', log)
                if discarded:
                    record['discarded_surface_births'] = json.loads(discarded.group(1))
                base.save(dest/'run.json', record)
                print('DONE', case['name'], mode, seed, round(record['seconds'], 1),
                    's; Y', f'{record["mean_luminance"]:.7g}', 'central', f'{record["central_mean_luminance"]:.7g}',
                    'lit', record['nonzero_pixels'], flush=True)
                report(args.stage)
    report(args.stage)


def report(stage):
    folder = OUT/stage
    manifest = base.read(folder/'manifest.json')
    modes = tuple(manifest.get('modes', base.MODES))
    reference_mode = 'global' if 'global' in modes else ('classic_global' if 'classic_global' in modes else modes[0])
    comparison_mode = 'renewal_plus' if 'renewal_plus' in modes else ('global_conditional' if 'global_conditional' in modes else modes[-1])
    coarse_stage = 'methods_smoke' if 'modes' in manifest else 'coarse'
    coarse_file = OUT/coarse_stage/'results.json'
    coarse_document = base.read(coarse_file) if stage != coarse_stage and coarse_file.exists() else None
    coarse = {r['case']['name']: r for r in coarse_document['ranking']} if (
        coarse_document and coarse_document['manifest']['renderer_sha256']==manifest['renderer_sha256']) else {}
    rows, ranking, overview = [], [], []
    for case in manifest['cases']:
        panels, stats = [], {}
        for mode in modes:
            images, runs = [], []
            for seed in manifest['seeds']:
                dest = folder/case['name']/mode/f'seed{seed}'
                if not (dest/'run.json').exists():
                    continue
                run = base.read(dest/'run.json')
                assert run['finite'] and run['scene_sha256'] == base.sha(dest/'scene.json')
                assert run['renderer_sha256']==manifest['renderer_sha256']
                assert run['image_sha256']==base.sha(dest/'render.exr')
                runs.append(run)
                images.append(base.pfm(dest/'render.pfm').astype(np.float64))
            if not images:
                panels.append('<div class="pending">等待 '+mode+'</div>')
                continue
            rgb = np.mean(images, axis=0)
            y = rgb @ WEIGHTS
            means = np.array([r['mean_luminance'] for r in runs])
            stats[mode] = dict(mean_luminance=float(y.mean()), nonzero_pixels=int((y>0).sum()),
                central_mean_luminance=float(np.mean([r['central_mean_luminance'] for r in runs])),
                seeds=len(runs), seed_means=means.tolist(),
                seed_mean_standard_error=float(means.std(ddof=1)/math.sqrt(len(means))) if len(means)>1 else None,
                records=runs)
            dest = folder/case['name']/mode
            for name, gain in [('display', 1), ('gain10', 10), ('gain100', 100)]:
                display = np.clip(rgb*gain, 0, 1)**(1/2.2)
                Image.fromarray(np.round(display*255).astype(np.uint8)).save(dest/(name+'.png'))
            overview.append((case, mode, dest/'display.png', dest/'gain100.png'))
            rel = f'{case["name"]}/{mode}'
            discarded = [r['discarded_surface_births'] for r in runs if 'discarded_surface_births' in r]
            discarded_note = ('<p>丢弃出生：路径 ' + str(sum(v['paths'] for v in discarded))
                + '，阴影连接 ' + str(sum(v['shadow_connections'] for v in discarded)) + '</p>') if discarded else ''
            error_note = (f' · seed 均值 SE={stats[mode]["seed_mean_standard_error"]:.3g}' if len(runs)>1 else '')
            panels.append(f'<div><h3>{LABELS[mode]}</h3><img class="render" data-base="{rel}" src="{rel}/gain100.png">'
                f'<p>Y={float(y.mean()):.6g} · 非零 {int((y>0).sum())} · 中心 Y={stats[mode]["central_mean_luminance"]:.6g}</p>'
                f'<p>{len(runs)} seed{error_note} · <a href="{rel}/seed{runs[0]["seed"]}/scene.json">JSON</a> · '
                f'<a href="{rel}/seed{runs[0]["seed"]}/render.exr">EXR（首个 seed）</a></p>{discarded_note}</div>')
        if len(stats) == len(modes):
            g, r = (stats[m]['mean_luminance'] for m in (reference_mode, comparison_mode))
            central_delta = stats[comparison_mode]['central_mean_luminance']-stats[reference_mode]['central_mean_luminance']
            seed_values = {mode: {r['seed']: r['mean_luminance'] for r in stats[mode]['records']}
                           for mode in (reference_mode, comparison_mode)}
            paired_seeds = sorted(set(seed_values[reference_mode]) & set(seed_values[comparison_mode]))
            differences = np.array([seed_values[comparison_mode][s]-seed_values[reference_mode][s] for s in paired_seeds])
            row = dict(case=case, modes=stats, delta=r-g, central_delta=central_delta,
                reference_mode=reference_mode, comparison_mode=comparison_mode,
                ratio=r/g if g>0 else None, paired_seeds=paired_seeds, paired_seed_differences=differences.tolist(),
                paired_difference_standard_error=float(differences.std(ddof=1)/math.sqrt(len(differences))) if len(differences)>1 else None)
            row['deltas_to_reference'] = {mode: stats[mode]['mean_luminance']-g for mode in modes}
            row['ratios_to_reference'] = {mode: stats[mode]['mean_luminance']/g if g>0 else None for mode in modes}
            if reference_mode=='global':
                row['deltas_to_global'] = row['deltas_to_reference']
                row['ratios_to_global'] = row['ratios_to_reference']
            if case['name'] in coarse and all(mode in coarse[case['name']]['modes'] for mode in (reference_mode, comparison_mode)):
                row['coarse_stage'] = coarse_stage
                previous_modes = coarse[case['name']]['modes']
                row['coarse_delta'] = previous_modes[comparison_mode]['mean_luminance']-previous_modes[reference_mode]['mean_luminance']
                row['delta_sign_changed_from_coarse'] = row['delta']*row['coarse_delta'] < 0
            ranking.append(row)
            detail = f'ΔY（{LABELS[comparison_mode]} − {LABELS[reference_mode]}）={r-g:.6g} · 中心 ΔY={central_delta:.6g}'
            if row['paired_difference_standard_error'] is not None:
                detail += f' · 配对 seed 差值 SE={row["paired_difference_standard_error"]:.3g}'
        else:
            detail = '渲染进行中'
        rows.append(f'<article><h2>σ={case["sigma"]:g} · ℓ={case["lengthScale"]:g} · σ/ℓ={case["sigma"]/case["lengthScale"]:g}</h2>'
            f'<div class="pair">{"".join(panels)}</div><p>{detail}</p></article>')
    complete = len(ranking) == len(manifest['cases']) and all(
        all(m['seeds']==len(manifest['seeds']) for m in row['modes'].values()) for row in ranking)
    base.save(folder/'results.json', dict(complete=complete, manifest=manifest,
        ranking=sorted(ranking, key=lambda r: abs(r['delta']), reverse=True),
        interpretation='Exploratory parameter scan; brightness can reflect actual geometry changes. No history-bias proof.'))
    if overview:
        tile = 192 if manifest['size']>=128 else 144
        columns = len(modes)
        row_height = tile+30
        for filename, image_index in [('overview.png', 2), ('overview100.png', 3)]:
            canvas = Image.new('RGB', (tile*columns, row_height*len(manifest['cases'])), 'white')
            draw = ImageDraw.Draw(canvas)
            for item in overview:
                case, mode = item[:2]
                case_index = next(i for i, candidate in enumerate(manifest['cases']) if candidate['name']==case['name'])
                x, y = modes.index(mode)*tile, case_index*row_height
                draw.text((x+3, y+2), f's={case["sigma"]:g} l={case["lengthScale"]:g}\n{mode}', fill='black')
                canvas.paste(Image.open(item[image_index]).resize((tile, tile), Image.Resampling.NEAREST), (x, y+30))
            canvas.save(folder/filename)
    stages = ''.join(f'<a href="../{p.parent.name}/comparison.html">{p.parent.name}</a>'
                     for p in sorted(OUT.glob('*/comparison.html')) if p.parent.name != stage)
    tests_link = '<a href="sdf-query-tests.log">SDF 查询回归测试</a>' if (folder/'sdf-query-tests.log').exists() else ''
    table = ''.join(f'<tr><td>{r["case"]["sigma"]:g}</td><td>{r["case"]["lengthScale"]:g}</td>'
        + ''.join(f'<td>{r["modes"][mode]["mean_luminance"]:.6g}</td>' for mode in modes)
        + f'<td>{r["delta"]:.6g}</td></tr>'
        for r in sorted(ranking, key=lambda r: abs(r['delta']), reverse=True))
    table_head = ''.join(f'<th>{LABELS[mode]} Y</th>' for mode in modes)
    method_note = (f'<p>本页 {len(modes)} 种方法使用同一解析双球 SDF，以及支持认证查询的同一个渲染器；'
        'classic_global 和 global_conditional 使用原始默认梯度规则（未启用 fixed5 修正）。'
        '原生 3D 使用 sparse convolution，其他两种使用各自的 Gaussian 输运近似，因此差异包含各方法完整的近似。</p>') if len(modes)>2 else ''
    if len(modes)>2:
        method_note += '<p>global_conditional 保留最新一次观测进行条件输运；它没有保留整条路径的全部历史。</p>'
        if 'global' in modes:
            method_note += '<p>3D global 在一条路径上使用同一个实现。</p>'
        if stage == 'renewal_high':
            method_note += ('<p>新增原生 3D GPIS renewal+；classic_global 与 global_conditional 复用'
                '<a href="../classic_high/comparison.html">同分辨率、同 spp、同 seed=41 的结果</a>，'
                '复用前核对场景、渲染器和图像哈希。'
                '<a href="../classic_high_stats/comparison.html">三种子复核</a>另页保留；本页全部为单种子结果。</p>')
    elif all(mode in modes for mode in CLASSIC_MODES):
        method_note = ('<p>本页重点比较 classic_global 与 global_conditional，采用同一解析 SDF、Gaussian 协方差和 Lambert 材质，均未启用 fixed5 修正。'
            'global_conditional 保留最新观测进行条件输运。原生 3D 参考图保留在<a href="../methods/comparison.html">上一轮四方法页面（64×64、16 spp）</a>，本页不将旧参考图标为本轮高采样结果。</p>')
    quality_table = ''
    if stage.startswith('classic_high') and all(mode in modes for mode in CLASSIC_MODES):
        quality_rows = []
        quality_records = []
        for source_name in dict.fromkeys(('methods', 'classic_high', stage)):
            source_path = OUT/source_name/'results.json'
            if not source_path.exists():
                continue
            source = base.read(source_path)
            settings = source['manifest']
            if settings['renderer_sha256']!=manifest['renderer_sha256']:
                continue
            for row in source['ranking']:
                if not all(mode in row['modes'] for mode in CLASSIC_MODES):
                    continue
                c, g = (row['modes'][mode] for mode in CLASSIC_MODES)
                quality_records.append(dict(stage=source_name, case=row['case'], size=settings['size'],
                    spp_per_seed=settings['spp'], seeds=min(c['seeds'],g['seeds']),
                    classic_mean=c['mean_luminance'], conditional_mean=g['mean_luminance'],
                    classic_central_mean=c['central_mean_luminance'], conditional_central_mean=g['central_mean_luminance']))
                quality_rows.append(f'<tr><td>{row["case"]["sigma"]:g} / {row["case"]["lengthScale"]:g}</td>'
                    f'<td>{settings["size"]}² / {settings["spp"]} spp / {min(c["seeds"],g["seeds"])} seed</td>'
                    f'<td>{c["mean_luminance"]:.6g}</td><td>{g["mean_luminance"]:.6g}</td>'
                    f'<td>{c["central_mean_luminance"]:.6g}</td><td>{g["central_mean_luminance"]:.6g}</td></tr>')
        base.save(folder/'quality-comparison.json', dict(complete=complete, records=quality_records))
        quality_table = ('<details><summary>与 64×64 / 16 spp 的对照</summary><table>'
            '<tr><th>σ / ℓ</th><th>分辨率 / 每 seed spp / seed 数</th><th>classic Y</th><th>conditional Y</th>'
            '<th>classic 中心 Y</th><th>conditional 中心 Y</th></tr>'+''.join(quality_rows)+'</table></details>')
    if stage == 'renewal_high':
        quality_records, quality_rows = [], []
        for source_name in ('methods', stage):
            source_path = OUT/source_name/'results.json'
            if not source_path.exists():
                continue
            source = base.read(source_path)
            settings = source['manifest']
            if settings['renderer_sha256'] != manifest['renderer_sha256']:
                continue
            for row in source['ranking']:
                if not all(mode in row['modes'] for mode in modes):
                    continue
                quality_records.append(dict(stage=source_name, case=row['case'], size=settings['size'],
                    spp_per_seed=settings['spp'], seeds=settings['seeds'],
                    means={mode: row['modes'][mode]['mean_luminance'] for mode in modes},
                    central_means={mode: row['modes'][mode]['central_mean_luminance'] for mode in modes}))
                quality_rows.append(f'<tr><td>{row["case"]["sigma"]:g} / {row["case"]["lengthScale"]:g}</td>'
                    f'<td>{settings["size"]}² / {settings["spp"]} spp / seed {settings["seeds"]}</td>'
                    + ''.join(f'<td>{row["modes"][mode]["mean_luminance"]:.6g}</td>' for mode in modes)+'</tr>')
        base.save(folder/'quality-comparison.json', dict(complete=complete, records=quality_records))
        quality_table = ('<details><summary>与 64×64 / 16 spp 的同种子对照</summary><table>'
            '<tr><th>σ / ℓ</th><th>分辨率 / spp / seed</th>'+table_head+'</tr>'
            + ''.join(quality_rows)+'</table></details>')
    tracking_note = (f'原生 3D 步长 {manifest["step"]}' if any(mode in base.MODES for mode in modes)
                     else 'Gaussian 方法保持默认跟踪设置')
    limitation_note = ('这轮未完成步长收敛和多种子统计，不能作为历史偏差的证明。' if any(mode in base.MODES for mode in modes)
                       else '本页检验分辨率、采样和种子稳定性；这些近似模型之间的比较不能直接作为完整历史真值。')
    method_style = ('.grid{grid-template-columns:1fr}.pair{grid-template-columns:repeat('+str(len(modes))+',1fr)}'
                    '@media(max-width:800px){.pair{grid-template-columns:repeat(2,1fr)}}') if len(modes)>2 else ''
    reversed_cases = [r for r in ranking if r.get('delta_sign_changed_from_coarse')]
    reversal_note = ('<p><strong>加采样后差异方向反转：</strong>' + '、'.join(
        f'σ={r["case"]["sigma"]:g}, ℓ={r["case"]["lengthScale"]:g}' for r in reversed_cases)
        + '。初扫的亮度排序不稳定，不能据此认定历史偏差。</p>') if reversed_cases else ''
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>双球背光 / σ 与 ℓ 扫描</title>
<style>body{{max-width:1500px;margin:28px auto;padding:0 20px;background:#f4f5f7;color:#20303d;font:15px/1.6 system-ui}}h1{{font-size:25px}}h2{{font-size:19px}}h3{{font-size:16px}}.intro,article{{padding:16px;background:white;border:1px solid #d9dfe4;border-radius:8px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px;margin-top:18px}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:12px}}img{{width:100%;display:block;image-rendering:pixelated}}p{{margin:8px 0}}a{{color:#17659a}}nav{{display:flex;gap:18px;align-items:center;flex-wrap:wrap}}table{{border-collapse:collapse}}th,td{{padding:4px 16px;border-bottom:1px solid #ddd;text-align:right}}@media(max-width:1050px){{.grid{{grid-template-columns:repeat(2,1fr)}}}}@media(max-width:650px){{.grid{{grid-template-columns:1fr}}}}</style>
<style>{method_style}</style><h1>双球背光：σ / lengthScale 参数扫描 · {stage}</h1><div class="intro">
<p>固定双球 SDF、相机、小圆盘光源、白色 Lambert 材质、最大 20 次反弹。</p>
{method_note}
<p>{manifest['size']}×{manifest['size']}，每个 seed {manifest['spp']} spp，seed={manifest['seeds']}，合计 {manifest['spp']*len(manifest['seeds'])} spp，{tracking_note}。
所有图片默认统一 ×100 线性曝光和 gamma 2.2，可切换到 ×10 或 ×1。Y 是原始线性亮度；中心区域固定为画面中间一半宽高。</p>
<p>多 seed 图片在原始线性 RGB 上平均；SE 表示独立 seed 的整图均值标准误，未把像素当作独立重复实验。</p>
<p>σ 改变场的振幅，ℓ 改变空间相关尺度。增大 σ 可能让后球真正露出，因此非零亮度本身不能证明忽略历史造成漏光；
大 σ 时随机表面也可能接触光源或过程边界。{limitation_note}</p>{reversal_note}
<nav><label>统一曝光 <select id="gain"><option value="gain100" selected>×100</option><option value="gain10">×10</option><option value="display">×1</option></select></label>
<a href="../../preview/comparison.html">原始布置和控制图</a><a href="manifest.json">配置</a><a href="results.json">线性数据</a>{tests_link}{stages}</nav>
<details open><summary>线性亮度（末列 ΔY 为 {LABELS[comparison_mode]} − {LABELS[reference_mode]}）</summary><table><tr><th>σ</th><th>ℓ</th>{table_head}<th>ΔY</th></tr>{table}</table></details></div>
{quality_table}
<div class="grid">{''.join(rows)}</div>
<script>document.querySelector('#gain').onchange=e=>document.querySelectorAll('.render').forEach(i=>i.src=i.dataset.base+'/'+e.target.value+'.png');</script></html>'''
    (folder/'comparison.html').write_text(page, encoding='utf-8')
    print('REPORT', stage, len(ranking), '/', len(manifest['cases']), 'pairs; complete', complete, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'run', 'report'))
    parser.add_argument('--stage', default='coarse')
    parser.add_argument('--size', type=int, default=32)
    parser.add_argument('--spp', type=int, default=4)
    parser.add_argument('--seeds', nargs='+', type=int, default=[41])
    parser.add_argument('--sigmas', nargs='+', type=float, default=SIGMAS)
    parser.add_argument('--lengths', nargs='+', type=float, default=LENGTHS)
    parser.add_argument('--pairs', nargs='+', help='Explicit sigma:lengthScale pairs instead of a Cartesian grid')
    parser.add_argument('--step', type=float, default=base.STEP)
    parser.add_argument('--threads', type=int, default=16)
    parser.add_argument('--include-classic', action='store_true', help='Add classic_global and global_conditional; rerender all four methods')
    parser.add_argument('--models', nargs='+', choices=base.MODES+CLASSIC_MODES, help='Render an explicit subset of methods')
    parser.add_argument('--reuse-stage', help='Reuse matching finished renders from another stage after checking scene, image and renderer hashes')
    args = parser.parse_args()
    if args.action == 'run':
        render(args)
    elif args.action == 'prepare':
        prepare(args)
    else:
        report(args.stage)


if __name__ == '__main__':
    main()
