"""Matched 1D Renewal Half+ / MIS additions to the existing plane experiment.

python -B tools/surface_compare/plane_renewal_mis.py render
python -B tools/surface_compare/plane_renewal_mis.py render --component single
python -B tools/surface_compare/plane_renewal_mis.py render --stage renewal_1d_mis_pilot --size 32 --spp 8
python -B tools/surface_compare/plane_renewal_mis.py render --stage renewal_1d_mis_pilot --size 32 --spp 8 --component single
python -B tools/surface_compare/plane_renewal_mis.py render --stage renewal_1d_mis_pilot --size 32 --spp 8 --component multiple_check
python -B tools/surface_compare/plane_renewal_mis.py check --stage renewal_1d_mis_pilot
<plot-python> -B tools/surface_compare/plane_renewal_mis.py report

Copy the original scenes and use their exact renderer binary. Only the sparse
sampling/memory scheme and output destination change in the full-size run.
Scattering decomposition additionally changes the integrator's bounce limits.
"""
from __future__ import annotations

import argparse
import copy
import csv
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from plane import OUT, ROOT, CASES, SEED, RENDER_MODES, folder
from run import load, save, sha, pfm

MODE = 'sparse_1d_renewal_mis'
STAGE = 'renewal_1d_mis'
CHANGES = {'correlation_context': 'renewal+', '1D_sampling': True,
           '1D_sampling_scheme': 'mis', '1D_gradient_correlationXY': False}
LABELS = ('classic_local', 'classic_global', 'global_conditional',
          'GPIS 3D global', 'GPIS 1D renewal+ MIS')
ERRORS = re.compile(r'Caught an internal error|unrecoverable|Gradient invalid|'
                    r'Unable to load|Wrong normal direction|Failed to sample gradient|'
                    r'Gradient zero|Conditioning .* error|sanity check failed|'
                    r'GPContext does not exist', re.I)


def reference(case, mode='sparse_3d_global', component='total'):
    return folder('final', case, mode, component, SEED)


def destination(stage, case, component='total'):
    return folder(stage, case, MODE, component, SEED)


def renderer():
    expected = load(reference(CASES[0])/'result.json')['provenance']['renderer_sha256']
    for name in ('tungsten.exe', 'tungsten_before_endpoint_fix.exe'):
        candidate = ROOT/'build/Release'/name
        if candidate.exists() and sha(candidate) == expected:
            return candidate, expected
    raise RuntimeError('The original plane renderer binary is unavailable')


def scene(case, stage, size, spp, component='total'):
    source = reference(case)
    original = load(source/'scene.json')
    if sha(source/'scene.json') != load(source/'result.json')['scene_sha256']:
        raise RuntimeError('Original scene hash mismatch: '+case)
    s = copy.deepcopy(original)
    s['media'][0].update(CHANGES)
    s['camera']['resolution'] = [size, size]
    s['integrator'].update(min_bounces=2 if component == 'multiple_check' else 1,
                           max_bounces=2 if component == 'single' else 128)
    s['renderer'].update(spp=spp, spp_step=min(16, spp),
                         output_directory=str(destination(stage, case, component)))
    check = copy.deepcopy(s)
    for key in CHANGES:
        check['media'][0][key] = original['media'][0][key]
    check['camera']['resolution'] = original['camera']['resolution']
    check['integrator'] = original['integrator']
    check['renderer'] = original['renderer']
    assert check == original
    assert s['integrator']['enable_volume_light_sampling']
    assert not s['media'][0]['single_realization']
    return s


def measure(d):
    a = pfm(d/'render.pfm')[:, :, 0]
    v = pfm(d/'samplesVariance.pfm')[:, :, 0]
    halves = [pfm(d/f'samples{c}.pfm')[:, :, 0] for c in 'AB']
    if not np.allclose((halves[0]+halves[1])/2, a, rtol=2e-6, atol=2e-7):
        raise RuntimeError('Split sample buffers do not match the render')
    return dict(mean=float(a.mean()), standard_error=float(np.sqrt(v.sum())/a.size),
                rms_pixel_standard_error=float(np.sqrt(v.mean())),
                zero_fraction=float((a == 0).mean()), maximum=float(a.max()))


def render(args):
    exe, exe_hash = renderer()
    manifest = dict(stage=args.stage, component=args.component, size=args.size, spp=args.spp, seed=SEED,
        threads=args.threads, executable=str(exe), renderer_sha256=exe_hash,
        changes=CHANGES, runner_sha256=sha(Path(__file__)), sources=[],
        memory='Renewal Half+: renewal+ with 1D_gradient_correlationXY=false',
        note='Memory and sampling both change relative to 3D global; this is not solely a variance-reduction switch.')
    for case in args.cases:
        s = scene(case, args.stage, args.size, args.spp, args.component)
        d = destination(args.stage, case, args.component)
        old = d/'result.json'
        provenance = dict(load(reference(case)/'result.json')['provenance'])
        assert provenance['renderer_sha256'] == exe_hash
        if old.exists():
            row = load(old)
            if load(d/'scene.json') != s or row['provenance'] != provenance:
                raise RuntimeError('Existing incompatible run: '+str(d))
            for name in ('scene.json', 'render.pfm', 'render.png'):
                key = {'scene.json':'scene_sha256', 'render.pfm':'pfm_sha256',
                       'render.png':'png_sha256'}[name]
                assert sha(d/name) == row[key]
            print('Reuse', case, args.component, flush=True)
        else:
            if (d/'render.pfm').exists():
                raise RuntimeError('Existing unvalidated render: '+str(d))
            save(d/'scene.json', s)
            print('Render', case, args.component, args.size, args.spp, flush=True)
            start = time.monotonic()
            with (d/'render.log').open('w', encoding='utf-8') as log:
                p = subprocess.run([str(exe), '-t', str(args.threads), '-s', str(SEED),
                    '-d', str(d), str(d/'scene.json')], cwd=ROOT, stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
            log = (d/'render.log').read_text(encoding='utf-8', errors='replace')
            if (p.returncode or 'Finished render.' not in log or
                    f'Completed {args.spp}/{args.spp} spp' not in log or ERRORS.search(log)):
                raise RuntimeError('Render failed; inspect '+str(d/'render.log'))
            assert pfm(d/'render.pfm').shape == (args.size, args.size, 3)
            row = dict(seconds=time.monotonic()-start, provenance=provenance,
                completed_spp=args.spp, size=[args.size, args.size],
                scene_sha256=sha(d/'scene.json'), pfm_sha256=sha(d/'render.pfm'),
                png_sha256=sha(d/'render.png'), statistics=measure(d))
            save(old, row)
            print('Done', case, args.component, round(row['seconds'], 2), row['statistics'], flush=True)
        manifest['sources'].append(dict(case=case, folder=str(d), result=row,
            original_scene_sha256=sha(reference(case)/'scene.json')))
        name = 'run.json' if args.component == 'total' else f'run_{args.component}.json'
        save(OUT/args.stage/name, manifest)


def check_decomposition(args):
    checks = []
    for case in args.cases:
        paths = [destination(args.stage, case, c) for c in ('total', 'single', 'multiple_check')]
        results = [load(d/'result.json') for d in paths]
        assert len({(tuple(r['size']), r['completed_spp'], r['provenance']['renderer_sha256'])
                    for r in results}) == 1
        arrays = []
        for d, r in zip(paths, results):
            assert sha(d/'render.pfm') == r['pfm_sha256']
            assert sha(d/'scene.json') == r['scene_sha256']
            assert load(d/'scene.json') == scene(case, args.stage, r['size'][0], r['completed_spp'], d.name)
            measure(d)
            arrays.append(pfm(d/'render.pfm'))
        a, b, c = arrays
        tolerance = 2e-5*max(1., float(a.max()))
        errors = [float(np.abs(a-b-c).max())]
        for letter in 'AB':
            x, y, z = [pfm(d/f'samples{letter}.pfm') for d in paths]
            errors.append(float(np.abs(x-y-z).max()))
        if max(errors) > tolerance or float((a-b).min()) < -tolerance:
            raise RuntimeError(f'Unpaired scattering decomposition: {case}, errors={errors}')
        checks.append(dict(case=case, size=results[0]['size'], spp=results[0]['completed_spp'],
            max_absolute_errors=dict(zip(('full', 'half_A', 'half_B'), errors)),
            tolerance=tolerance, minimum_paired_difference=float((a-b).min()),
            total=float(a.mean()), single=float(b.mean()), multiple=float(c.mean()),
            sources=[dict(folder=str(d), result=r) for d, r in zip(paths, results)]))
    save(OUT/args.stage/'decomposition_checks.json', dict(passed=True, checks=checks,
        definition='Same-seed total(min=1,max=128) = single(min=1,max=2) + depth-filtered multiple(min=2,max=128); also checked on even/odd halves.'))
    print('Verified scattering decomposition:', len(checks), 'cases', flush=True)


def scattering_report(args, totals, colors, renderer_hash):
    from figures import plt, Image
    from plane_figures import filmic, GAINS, write_pfm

    dest = OUT/args.stage
    validation_path = OUT/(args.stage+'_pilot')/'decomposition_checks.json'
    validation = load(validation_path)
    assert validation['passed'] and {c['case'] for c in validation['checks']} == set(CASES)
    for check in validation['checks']:
        for source in check['sources']:
            d, r = Path(source['folder']), source['result']
            assert r['provenance']['renderer_sha256'] == renderer_hash
            assert sha(d/'render.pfm') == r['pfm_sha256']
            assert sha(d/'scene.json') == r['scene_sha256']

    modes = (*RENDER_MODES, MODE)
    parts = ('total', 'single', 'multiple')
    part_labels = ('完整结果', '单次散射', '两次及以上散射')
    data, stats, sources, summaries, checks = {}, [], [], [], []
    for case in CASES:
        data[case] = {}
        for mode in modes:
            paths = [destination(args.stage, case, part) if mode == MODE else reference(case, mode, part)
                     for part in ('total', 'single')]
            raw = {'total': totals[case][mode]}
            halves, variances, scenes = {}, {}, []
            for part, d in zip(('total', 'single'), paths):
                result, s = load(d/'result.json'), load(d/'scene.json')
                assert result['provenance']['renderer_sha256'] == renderer_hash
                assert result['size'] == [256, 256] and result['completed_spp'] == 128
                for name, key in (('scene.json', 'scene_sha256'), ('render.pfm', 'pfm_sha256'),
                                  ('render.png', 'png_sha256')):
                    assert sha(d/name) == result[key]
                if mode == MODE:
                    assert s == scene(case, args.stage, 256, 128, part)
                raw[part] = pfm(d/'render.pfm')
                halves[part] = [pfm(d/f'samples{letter}.pfm') for letter in 'AB']
                variances[part] = pfm(d/'samplesVariance.pfm')
                assert np.allclose((halves[part][0]+halves[part][1])/2, raw[part], rtol=2e-6, atol=2e-7)
                assert np.max(np.ptp(raw[part], axis=2)) < 1e-7
                scenes.append(s)
                sources.append(dict(case=case, mode=mode, component=part, folder=str(d), result=result,
                    variance_sha256=sha(d/'samplesVariance.pfm'),
                    split_sha256=[sha(d/f'samples{c}.pfm') for c in 'AB']))
            paired = copy.deepcopy(scenes[1])
            assert paired['integrator']['max_bounces'] == 2
            assert scenes[0]['integrator']['max_bounces'] == 128
            assert paired['integrator']['min_bounces'] == scenes[0]['integrator']['min_bounces'] == 1
            paired['integrator']['max_bounces'] = 128
            paired['renderer']['output_directory'] = scenes[0]['renderer']['output_directory']
            assert paired == scenes[0], (case, mode)

            delta = raw['total']-raw['single']
            tolerance = 2e-5*max(1., float(raw['total'].max()))
            if float(delta.min()) < -tolerance:
                raise RuntimeError(f'Unpaired full-size subtraction: {case} {mode}')
            raw['multiple'] = np.maximum(delta, 0)
            halves['multiple'] = [a-b for a, b in zip(halves['total'], halves['single'])]
            # Retain covariance between the paired total/single estimates.
            variances['multiple'] = .25*(halves['multiple'][0]-halves['multiple'][1])**2
            error = float(np.abs(raw['total']-raw['single']-raw['multiple']).max())
            checks.append(dict(case=case, mode=mode, minimum_paired_difference=float(delta.min()),
                max_reconstruction_error=error, tolerance=tolerance))
            assert error <= tolerance
            data[case][mode] = raw
            if mode == MODE:
                target = dest/'derived'/case/mode
                write_pfm(target/'multiple.pfm', raw['multiple'])
                Image.fromarray(np.rint(255*filmic(raw['multiple'])).astype('uint8')).save(target/'multiple.png')
            else:
                target = OUT/'final/derived'/case/mode
                assert np.allclose(pfm(target/'multiple.pfm'), raw['multiple'], rtol=2e-6, atol=2e-7)
            for part in parts:
                a, v = raw[part][:, :, 0], variances[part][:, :, 0]
                stats.append(dict(case=case, mode=mode, component=part, region='full',
                    mean=float(a.mean()), standard_error=float(np.sqrt(v.sum())/a.size),
                    pixels=a.size, nonzero_pixels=int(np.count_nonzero(a)), gain=GAINS[case],
                    variance_method='paired_even_odd_halves' if part == 'multiple' else 'per_sample_Welford'))
            summaries.append(dict(case=case, mode=mode,
                total=float(raw['total'].mean()), single=float(raw['single'].mean()),
                multiple=float(raw['multiple'].mean()),
                multiple_fraction=float(raw['multiple'].mean()/raw['total'].mean()),
                multiple_standard_error=stats[-1]['standard_error'],
                multiple_pfm=Path(os.path.relpath(target/'multiple.pfm', dest)).as_posix(),
                single_png=Path(os.path.relpath(paths[1]/'render.png', dest)).as_posix()))

    lookup = {(s['case'], s['mode'], s['component']): s for s in stats}
    for case in CASES:
        fig, axes = plt.subplots(3, 5, figsize=(17.5, 11))
        for row, part in enumerate(parts):
            for col, mode in enumerate(modes):
                ax, st = axes[row, col], lookup[case, mode, part]
                ax.imshow(filmic(data[case][mode][part], GAINS[case]), interpolation='nearest')
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f'{LABELS[col]}\nL = {st["mean"]:.6f} ± {1.96*st["standard_error"]:.6f}',
                             fontsize=9, color=colors[col])
                if col == 0:
                    ax.set_ylabel(part_labels[row], fontsize=11)
        side, angle = case.split('_')
        title = '同侧' if side == 'same' else '对侧'
        fig.suptitle(f'{title}照明 · {angle}° · 按散射次数分解 · 五种模式\n'
                     f'256×256，128 spp；全部面板使用显示增益 ×{GAINS[case]:g}', fontsize=16)
        fig.text(.035, .025,
            '上排 = 中排 + 下排（在线性辐亮度中）；透明边界不计入散射次数。多次 = 同种子总光照 − 单次。\n'
            '小图已用独立深度筛选渲染验证分解；多次误差通过配对奇偶样本估计。最后一列为 Renewal Half+ / MIS。',
            fontsize=10, color='#4b535e')
        fig.subplots_adjust(top=.88, bottom=.10, left=.055, right=.99, hspace=.27, wspace=.055)
        fig.savefig(dest/f'scattering_{case}.png', dpi=150)
        plt.close(fig)

    with (dest/'scattering_metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    save(dest/'scattering_manifest.json', dict(statistics=stats, summaries=summaries,
        sources=sources, full_size_checks=checks, validation=validation,
        renderer_sha256=renderer_hash, report_script_sha256=sha(Path(__file__)),
        validation_sha256=sha(validation_path),
        decomposition='total(min=1,max=128) minus single(min=1,max=2), paired by pixel/sample seed; multiple means two or more scattering vertices.',
        multiple_variance='(delta_A-delta_B)^2/4, preserving covariance between total and single.',
        display_gains=GAINS, approximate_95pct_sampling_only=True))
    rows = []
    for s in summaries:
        rows.append(f'<tr><td>{s["case"]}</td><td>{LABELS[modes.index(s["mode"])]}</td>'
            f'<td>{s["total"]:.6f}</td><td>{s["single"]:.6f}</td>'
            f'<td>{s["multiple"]:.6f} ± {1.96*s["multiple_standard_error"]:.6f}</td>'
            f'<td>{100*s["multiple_fraction"]:.2f}%</td>'
            f'<td><a href="{s["single_png"]}">单次 PNG</a> · <a href="{s["multiple_pfm"]}">多次 PFM</a></td></tr>')
    cards = ''.join(f'<a href="scattering_{case}.png"><img loading="lazy" src="scattering_{case}.png" '
                    f'alt="{case} 总光照、单次与多次散射"><span>{case}</span></a>' for case in CASES)
    section = '''<section id="scattering"><h2>按 bounce 次数分解：总光照／单次／两次及以上</h2>
<p>每张图三行分别为总光照、单次散射、两次及以上散射，五列对应五种模型。同一张图共用显示增益与 filmic。
多次图由原始线性 PFM 做配对相减得到；显示后的 PNG 不能直接相减。透明计算域边界不计入散射次数。</p>
<p>新增模式六组 32×32、8 spp 小图已通过独立深度筛选验证：总光照 = 单次 + 两次及以上，奇偶样本也分别满足。
正式图为 256×256、128 spp；下表多次误差区间使用配对样本，保留总光照与单次之间的协方差。</p>
<p><a href="scattering_metrics.csv">散射统计 CSV</a> · <a href="scattering_manifest.json">分解来源与验证记录</a></p>
<div class="scattering-grid">'''+cards+'''</div><div class="scroll"><table><thead><tr>
<th>条件</th><th>模式</th><th>总光照</th><th>单次</th><th>两次及以上 ±95%</th><th>多次占比</th><th>原始文件</th>
</tr></thead><tbody>'''+''.join(rows)+'</tbody></table></div></section>'
    for s in summaries:
        if s['mode'] == MODE:
            print(s['case'], 'single', round(s['single'], 7), 'multiple', round(s['multiple'], 7),
                  'multiple %', round(100*s['multiple_fraction'], 2), flush=True)
    return section


def report(args):
    from figures import plt
    from plane_figures import filmic, GAINS

    dest = OUT/args.stage
    run = load(dest/'run.json')
    modes = (*RENDER_MODES, MODE)
    colors = ('#1675b9', '#df822d', '#9a2b85', '#168469', '#b84d30')
    data, stats, sources = {}, [], []
    for case in CASES:
        data[case] = {}
        original = load(reference(case)/'scene.json')
        for mode in modes:
            d = destination(args.stage, case) if mode == MODE else reference(case, mode)
            result, s = load(d/'result.json'), load(d/'scene.json')
            assert result['provenance']['renderer_sha256'] == run['renderer_sha256']
            assert result['completed_spp'] == run['spp'] == 128
            assert result['size'] == [256, 256]
            for key in ('camera', 'primitives', 'integrator'):
                assert s[key] == original[key], (case, mode, key)
            if mode == MODE:
                assert s == scene(case, args.stage, 256, 128)
            for name, key in (('scene.json', 'scene_sha256'), ('render.pfm', 'pfm_sha256'),
                              ('render.png', 'png_sha256')):
                assert sha(d/name) == result[key]
            data[case][mode] = pfm(d/'render.pfm')
            row = dict(case=case, mode=mode, **measure(d),
                       gain=GAINS[case], seconds=result['seconds'])
            stats.append(row)
            sources.append(dict(case=case, mode=mode, folder=str(d), result=result,
                variance_sha256=sha(d/'samplesVariance.pfm'),
                split_sha256=[sha(d/f'samples{c}.pfm') for c in 'AB']))

    def find(case, mode):
        return next(r for r in stats if r['case'] == case and r['mode'] == mode)

    for side in ('same', 'opposite'):
        for uniform in (False, True):
            fig, axes = plt.subplots(3, 5, figsize=(17.5, 11))
            for row, angle in enumerate((30, 60, 80)):
                case = f'{side}_{angle}'
                gain = 1 if uniform else GAINS[case]
                for col, mode in enumerate(modes):
                    ax, st = axes[row, col], find(case, mode)
                    ax.imshow(filmic(data[case][mode], gain), interpolation='nearest')
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_title(f'{LABELS[col]}\nL = {st["mean"]:.6f} ± {1.96*st["standard_error"]:.6f}',
                                 fontsize=9, color=colors[col])
                    if col == 0:
                        ax.set_ylabel(f'{angle}° · 显示增益 ×{gain:g}', fontsize=11)
            title = '同侧照明' if side == 'same' else '对侧照明'
            fig.suptitle(f'平面 · {title} · 新增 1D renewal+ / MIS\n'
                         '相同相机、光源、材质与渲染器；256×256，128 spp', fontsize=16)
            fig.text(.035, .025,
                '同一行使用相同增益与 filmic；原始像素，无降噪。L 与 ±95% 采样误差来自线性图的整图平均。\n'
                '1D renewal+ 使用 correlationXY=false（Renewal Half+）；同时改变记忆模型和采样方式，不能仅归因于 MIS。',
                fontsize=10, color='#4b535e')
            fig.subplots_adjust(top=.87, bottom=.10, left=.055, right=.99, hspace=.27, wspace=.055)
            suffix = '_uniform' if uniform else ''
            fig.savefig(dest/f'appearance_{side}{suffix}.png', dpi=150)
            plt.close(fig)

    comparisons = []
    for case in CASES:
        old, new = find(case, 'sparse_3d_global'), find(case, MODE)
        comparisons.append(dict(case=case,
            mean_change_percent=100*(new['mean']/old['mean']-1),
            rms_pixel_se_reduction_factor=old['rms_pixel_standard_error']/new['rms_pixel_standard_error'],
            old_zero_fraction=old['zero_fraction'], new_zero_fraction=new['zero_fraction']))
    scattering = ''
    scattering_ready = all((destination(args.stage, c, 'single')/'result.json').exists() for c in CASES)
    if scattering_ready:
        scattering = scattering_report(args, data, colors, run['renderer_sha256'])
    with (dest/'metrics.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    save(dest/'manifest.json', dict(render=run, statistics=stats, sources=sources,
        display_gains=GAINS, comparisons=comparisons,
        verification=dict(matched_cameras=True, matched_primitives=True, matched_integrators=True,
            matched_renderer_binary=True, original_render_hashes_verified=True,
            split_sample_buffers_verified=True, finite_nonnegative_linear_outputs=True),
        uncertainty='Sample-mean variance from Welford buffers. Approximate 95% intervals exclude model and discretization error.',
        baseline='Original final renders, unchanged. All five columns use the same renderer SHA256.',
        memory='New mode is 1D Renewal Half+ (correlationXY=false), not 3D global.',
        scattering_manifest='scattering_manifest.json' if scattering_ready else None,
        report_script_sha256=sha(Path(__file__))))
    table = []
    for row in stats:
        case, mode = row['case'], row['mode']
        d = destination(args.stage, case) if mode == MODE else reference(case, mode)
        link = Path(os.path.relpath(d, dest)).as_posix()
        table.append(f'<tr><td>{case}</td><td><a href="{link}/render.png">{LABELS[modes.index(mode)]}</a></td>'
            f'<td>{row["mean"]:.6f} ± {1.96*row["standard_error"]:.6f}</td>'
            f'<td>{row["rms_pixel_standard_error"]:.6f}</td><td>{100*row["zero_fraction"]:.2f}%</td>'
            f'<td><a href="{link}/scene.json">配置</a> · <a href="{link}/render.pfm">PFM</a> · '
            f'<a href="{link}/render.log">日志</a></td></tr>')
    body = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>平面：1D renewal+ MIS 对照</title>
<style>body{max-width:1650px;margin:32px auto;padding:0 24px;color:#253145;background:#f5f6f8;font:16px/1.65 system-ui}img{width:100%;background:white}a{color:#235e91}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:8px;border-bottom:1px solid #ddd;text-align:right}td:first-child,td:nth-child(2){text-align:left}.note{padding:18px;background:#e7edf5;border-radius:10px}.scroll{overflow-x:auto}.scattering-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}.scattering-grid a{padding:10px;background:white}.scattering-grid span{display:block}@media(max-width:900px){.scattering-grid{grid-template-columns:1fr}}</style>
<h1>平面：新增 1D renewal+ / MIS</h1>
<p><a href="../comparison.html">原平面实验</a> · <a href="metrics.csv">线性统计 CSV</a> · <a href="manifest.json">参数与验证记录</a></p>
<p>六组原相机和布光：同侧／对侧，各 30°、60°、80°；256×256，128 spp，seed=20261024。
均值 μ=y，σ=0.01，ℓ=0.025，镜面反射率 0.9，面光源 0.45×0.45、辐亮度 30，无环境光。</p>
<div class="note">最后一列为本轮新渲染，前四列复用原始结果；已核对相机、光源、积分器配置与渲染器哈希一致。
新模式：1D_sampling=true，correlation_context=renewal+，1D_sampling_scheme=mis，single_realization=false，correlationXY=false（Renewal Half+）。
步长 0.0025，每格 10 个脉冲。记忆模型与采样方式同时改变，亮度差不能只归因于 MIS。<br>
同侧 30°／60°／80° 显示增益分别为 ×1／×8／×32，对侧均为 ×1；没有降噪或逐面板调亮。
误差区间只包含蒙特卡洛采样误差。RMS 像素标准误差由每像素样本方差估计，不是相对真值的图像误差。</div>
<h2>对侧照明</h2><a href="appearance_opposite.png"><img src="appearance_opposite.png"></a>
<h2>同侧照明</h2><a href="appearance_same.png"><img src="appearance_same.png"></a>
<p>所有角度统一增益 ×1：<a href="appearance_same_uniform.png">同侧</a> · <a href="appearance_opposite_uniform.png">对侧</a></p>
<h2>线性统计与原始文件</h2><div class="scroll"><table><thead><tr><th>条件</th><th>模式</th><th>平均亮度 ±95%</th><th>RMS 像素标准误差</th><th>零值像素</th><th>原始文件</th></tr></thead><tbody>'''
    if scattering:
        body = body.replace('<h2>线性统计与原始文件</h2>', scattering+'\n<h2>线性统计与原始文件</h2>')
        body = body.replace('<h1>平面：新增 1D renewal+ / MIS</h1>',
            '<h1>平面：新增 1D renewal+ / MIS</h1><p><a href="#scattering">查看单次／多次 bounce 对照</a></p>')
    (dest/'comparison.html').write_text(body+''.join(table)+'</tbody></table></div></html>\n', encoding='utf-8')
    # Add a discoverable link without changing any original experiment figures/data.
    if args.stage == STAGE:
        index = OUT/'comparison.html'
        original = index.read_text(encoding='utf-8')
        marker = 'id="renewal-1d-mis-link"'
        if marker not in original:
            link = f'<p {marker}><a href="{STAGE}/comparison.html">新增：相同配置下的 1D renewal+ / MIS 平面对照</a></p>'
            index.write_text(original.replace('</h1>', '</h1>\n'+link, 1), encoding='utf-8')
    print(dest/'comparison.html', flush=True)
    for row in comparisons:
        print(row, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('render', 'report', 'check'))
    parser.add_argument('--component', choices=('total', 'single', 'multiple_check'), default='total')
    parser.add_argument('--stage', default=STAGE)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--threads', type=int, default=24)
    parser.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    args = parser.parse_args()
    if min(args.size, args.spp, args.threads) <= 0:
        parser.error('Resolution, spp and threads must be positive')
    {'render': render, 'report': report, 'check': check_decomposition}[args.command](args)
