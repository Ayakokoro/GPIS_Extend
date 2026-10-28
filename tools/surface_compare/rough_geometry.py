"""Matched rough-plane / rounded-U renders using the production integrator.

python -B tools/surface_compare/rough_geometry.py render --stage preview --size 64 --spp 32
python -B tools/surface_compare/rough_geometry.py render --stage final --size 128 --spp 128
<plot-python> tools/surface_compare/rough_geometry.py report --stage final
python -B tools/surface_compare/rough_geometry.py render --stage highres_256 --size 256 --spp 128

The three modes share analytic means, isotropic SE covariance and micro-BSDF.
Lambert micro-reflection is the default to expose correlated visibility using
NEE/MIS. --material mirror retains the more expensive specular comparison.
The U is an upward-open channel, extruded along Z, with a semicircular bottom
tangent to vertical walls. The process box truncates the channel away from the
central comparison region. No grid approximation or artificial crease is used.
"""
from __future__ import annotations

import argparse
import copy
import html
import math
import os
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from run import ROOT, load, save, sha, pfm, light

OUT = ROOT / 'outputs/rough_geometry_compare'
MODES = ('classic_global', 'global_conditional', 'sparse_3d_global')
LABELS = ('classic_global', 'global_conditional', 'GPIS global')
CASES = ('plane_same_75', 'plane_opposite_75', 'u_groove_side')
TITLES = ('粗糙平面 · 75° 同侧光', '粗糙平面 · 75° 对侧光', '圆底 U 槽 · 侧上方照明')
# Fixed per-case display gains; never normalized separately by rendering mode.
GAINS = dict(lambert=dict(plane_same_75=4., plane_opposite_75=4., u_groove_side=2.),
             mirror=dict(plane_same_75=16., plane_opposite_75=1., u_groove_side=1.))
SIGMA, ELL, RADIUS, ALBEDO = .04, .1, .15, .9
SEED = 20261025


def folder(args, case, mode):
    return OUT / args.stage / case / mode / f'seed{args.seed}'


def mean_field(case):
    if case.startswith('plane'):
        return dict(type='linear', reference_point=[0, 0, 0], direction=[0, 1, 0], scale=1)
    # Procedural transforms change coordinates; scale also restores world-space
    # signed distances and unit gradients after the uniform spatial scaling.
    return dict(type='procedural', func='rounded_u_groove',
                transform=dict(scale=RADIUS), scale=RADIUS)


def medium(args, case, mode):
    gp = dict(type='standard', mean=mean_field(case),
              covariance=dict(type='squared_exponential', sigma=SIGMA, lengthScale=ELL))
    material = dict(type=args.material, albedo=ALBEDO)
    if mode == 'sparse_3d_global':
        return dict(name='surface', type='sparse_conv_noise', gaussian_process=gp,
                    phase_function=dict(type='brdf', bsdf=material),
                    sigma_a=0, sigma_s=1, density=1, max_bounces=128,
                    step_size=args.step, min_step=8, seed=17429,
                    impulse_density=10, single_realization=False,
                    correlation_context='global', isotropic_3D_sampling=False,
                    **{'1D_sampling': False, '1D_sampling_scheme': 'uni',
                       '1D_gradient_correlationXY': False})
    result = dict(name='surface', type=mode, gaussian_process=gp, material=material,
                  max_bounces=128, sdf_lipschitz=1.001, gradient_bound=1.001)
    if mode == 'global_conditional':
        result.update(tracking_step_scale=.5, tracking_age_scale=.25,
                      tracking_max_expected_events=64.)
    return result


def scene(args, case, mode):
    if case.startswith('plane'):
        theta = math.radians(75)
        view = np.array([math.sin(theta), math.cos(theta), 0.])
        illumination = view.copy()
        if 'opposite' in case:
            illumination[0] *= -1
        position = (2 * view).tolist()
        target = [0, 0, 0]
        # Moderate FOV makes the observer/light angular separation visible.
        fov = 12
        key = light('key', (3 * illumination).tolist(), [.45, 1, .45], 30)
        lo, hi = np.array([-2., -.3, -1.]), np.array([2., .3, 1.])
    else:
        position, target, fov = [0, .65, 1.4], [0, -.04, 0], 21
        key = light('key', [.08, .85, .1], [.12, 1, .3], 30)
        lo, hi = np.array([-.5, -.45, -1.8]), np.array([.5, .6, 1.8])
    # Enlarge lateral/depth boundaries without moving the upper boundary over
    # the camera. This is a finite-domain check, not a change in the mean shape.
    lo *= np.array([args.domain_scale, args.domain_scale, args.domain_scale])
    hi *= np.array([args.domain_scale, 1., args.domain_scale])
    box = dict(name='processBox', type='cube', int_medium='surface',
               bsdf=dict(type='forward', albedo=1),
               transform=dict(position=((lo + hi) / 2).tolist(), scale=(hi - lo).tolist()))
    return dict(media=[medium(args, case, mode)],
        camera=dict(type='pinhole', resolution=[args.size, args.size], fov=fov,
                    tonemap='filmic', reconstruction_filter='box',
                    transform=dict(position=position, look_at=target, up=[0, 1, 0], scale=[-1, 1, 1])),
        integrator=dict(type='path_tracer', min_bounces=1, max_bounces=64,
                        enable_light_sampling=True, enable_volume_light_sampling=True,
                        enable_consistency_checks=False, enable_two_sided_shading=True,
                        use_cuda_gpis=False, independent_path_sampling=True, wavefront_gpis=False),
        primitives=[box, key],
        renderer=dict(output_directory=str(folder(args, case, mode)),
                      output_file='render.png', hdr_output_file='render.pfm',
                      overwrite_output_files=True, spp=args.spp, spp_step=min(16, args.spp),
                      adaptive_sampling=False, stratified_sampler=False, enable_resume_render=False,
                      output_buffers=[dict(type='color', hdr_output_file='samples.pfm',
                                           sample_variance=True, two_buffer_variance=True)]))


def render(args):
    exe = ROOT / 'build/Release/tungsten.exe'
    version = sha(exe)
    sources = []
    for case in args.cases:
        for mode in args.modes:
            d = folder(args, case, mode)
            s = scene(args, case, mode)
            if (d / 'result.json').exists():
                r = load(d / 'result.json')
                assert r['renderer_sha256'] == version and load(d / 'scene.json') == s
                for name, digest in r['sha256'].items():
                    assert sha(d / name) == digest
                print('Reuse', args.stage, case, mode, flush=True)
            else:
                if (d / 'render.pfm').exists():
                    raise RuntimeError('Unvalidated existing output: ' + str(d))
                save(d / 'scene.json', s)
                start = time.monotonic()
                print('Render', args.stage, case, mode, args.size, args.spp, flush=True)
                with (d / 'render.log').open('w', encoding='utf-8') as log:
                    p = subprocess.run([str(exe), '-t', str(args.threads), '-s', str(args.seed),
                                        '-d', str(d), str(d / 'scene.json')], cwd=ROOT,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
                log = (d / 'render.log').read_text(encoding='utf-8', errors='replace')
                if (p.returncode or 'Finished render.' not in log
                        or not re.search(rf'Completed {args.spp}/{args.spp} spp', log)
                        or re.search(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|'
                                     r'Wrong normal direction|Failed to sample gradient|Gradient zero', log, re.I)):
                    save(d / 'failure.json', dict(returncode=p.returncode, seconds=time.monotonic()-start))
                    raise RuntimeError('Render failed: ' + str(d / 'render.log'))
                rgb = pfm(d / 'render.pfm')
                assert rgb.shape == (args.size, args.size, 3)
                assert np.max(np.ptp(rgb, axis=2)) < 1e-6
                assert np.allclose((pfm(d / 'samplesA.pfm') + pfm(d / 'samplesB.pfm')) / 2,
                                   rgb, rtol=3e-6, atol=3e-7)
                pfm(d / 'samplesVariance.pfm')
                discarded = re.search(r'discarded surface births: (\{.*\})', log)
                import json
                r = dict(seconds=time.monotonic()-start, renderer_sha256=version,
                         resolution=[args.size, args.size], spp=args.spp, seed=args.seed,
                         mean_luminance=float(rgb[..., 0].mean()),
                         discarded_surface_births=json.loads(discarded[1]) if discarded else None,
                         zero_length_messages=log.count('zero-length'),
                         sha256={n: sha(d / n) for n in ('scene.json', 'render.png', 'render.pfm',
                                 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm', 'render.log')})
                assert sha(exe) == version, 'Renderer changed during run'
                save(d / 'result.json', r)
                print('Done', case, mode, round(r['seconds'], 2), 'mean', r['mean_luminance'], flush=True)
            sources.append(dict(case=case, mode=mode, folder=str(d), result=r))
    save(OUT / args.stage / ('run_' + '_'.join(args.cases) + '_' + '_'.join(args.modes) + '.json'),
         dict(arguments=vars(args), sources=sources))


def report(args):
    from figures import plt, Image
    from plane_figures import filmic, blocks
    dest = OUT / args.stage
    data, sources, statistics, pairs = {}, [], [], []
    versions, settings, materials = set(), set(), set()
    for case in args.cases:
        data[case] = {}
        reference_scene = None
        reference_material = None
        for mode in MODES:
            d = folder(args, case, mode)
            r, s = load(d / 'result.json'), load(d / 'scene.json')
            for name, digest in r['sha256'].items():
                assert sha(d / name) == digest
            versions.add(r['renderer_sha256'])
            settings.add((tuple(r['resolution']), r['spp']))
            if reference_scene is not None:
                for k in ('camera', 'integrator', 'primitives'):
                    assert s[k] == reference_scene[k]
                assert s['media'][0]['gaussian_process'] == reference_scene['media'][0]['gaussian_process']
            reference_scene = s
            mat = s['media'][0].get('material', s['media'][0].get('phase_function', {}).get('bsdf'))
            if reference_material is not None: assert mat == reference_material
            reference_material = mat
            materials.add(mat['type'])
            cov=s['media'][0]['gaussian_process']['covariance']
            assert cov==dict(type='squared_exponential',sigma=SIGMA,lengthScale=ELL)
            assert mat['albedo']==ALBEDO
            if case.startswith('u_'):
                assert s['media'][0]['gaussian_process']['mean']==mean_field(case)
            if mode=='sparse_3d_global':
                m=s['media'][0]
                assert m['correlation_context']=='global' and not m['single_realization'] and not m['1D_sampling']
            rgb = pfm(d / 'render.pfm')
            halves = [pfm(d / f'samples{c}.pfm') for c in ('A', 'B')]
            v = pfm(d / 'samplesVariance.pfm')[..., 0]
            y = rgb[..., 0]
            h, w = y.shape
            roi = (slice(h // 4, 3 * h // 4), slice(w // 4, 3 * w // 4))
            for region, sl in (('full', (slice(None), slice(None))), ('center_half', roi)):
                statistics.append(dict(case=case, mode=mode, region=region, mean=float(y[sl].mean()),
                                       standard_error=float(np.sqrt(v[sl].sum()) / y[sl].size)))
            gain = GAINS[mat['type']][case]
            preview = f'{case}_{mode}.png'
            Image.fromarray(np.rint(filmic(rgb, gain) * 255).astype('uint8')).save(dest / preview)
            data[case][mode] = dict(rgb=rgb, halves=halves, folder=d, result=r, preview=preview, gain=gain)
            sources.append(dict(case=case, mode=mode, folder=str(d), result=r))
        g = data[case]['sparse_3d_global']
        norm = float(g['rgb'][roi][..., 0].mean())
        for mode in MODES[:-1]:
            a = data[case][mode]
            delta = blocks(a['rgb']-g['rgb'])[..., 0][4:12, 4:12]
            split_delta = blocks(a['halves'][0]-a['halves'][1]-g['halves'][0]+g['halves'][1])[..., 0][4:12, 4:12]
            pairs.append(dict(case=case, mode=mode, reference='sparse_3d_global', region='center_half',
                              block_grid=[16, 16], normalization_mean=norm,
                              relative_rms=float(np.sqrt(np.mean(delta**2))/norm),
                              relative_noise_rms=float(np.sqrt(np.mean(split_delta**2)) / (2*norm))))
    assert len(versions) == len(settings) == len(materials) == 1
    material = next(iter(materials))
    gains = GAINS[material]
    size, spp = next(iter(settings))
    fig, axes = plt.subplots(len(args.cases), 3, figsize=(13, 4.3*len(args.cases)), squeeze=False)
    for row, case in enumerate(args.cases):
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            ax.imshow(filmic(data[case][mode]['rgb'], gains[case]))
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0: ax.set_title(LABELS[col])
            if col == 0: ax.set_ylabel(TITLES[CASES.index(case)] + f'\n统一增益 ×{gains[case]:g}')
    fig.suptitle(f'粗糙平面与圆底 U 槽 · 三模式平均外观\n{material} 微材质 · σ={SIGMA:g} · ℓ={ELL:g} · σ/ℓ={SIGMA/ELL:g} · {size[0]}² / {spp} spp', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, .95))
    fig.savefig(dest / 'comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(len(args.cases), 3, figsize=(13, 4.3*len(args.cases)), squeeze=False)
    for row, case in enumerate(args.cases):
        for col, mode in enumerate(MODES):
            axes[row,col].imshow(filmic(blocks(data[case][mode]['rgb']), gains[case]), interpolation='nearest')
            axes[row,col].set_xticks([]); axes[row,col].set_yticks([])
            if row==0: axes[row,col].set_title(LABELS[col])
            if col==0: axes[row,col].set_ylabel(TITLES[CASES.index(case)])
    fig.suptitle('16×16 分区平均：先在线性 RGB 中平均，再应用相同显示增益',fontsize=16)
    fig.tight_layout(rect=(0,0,1,.96)); fig.savefig(dest/'block_means.png',dpi=150); plt.close(fig)
    fig, axes = plt.subplots(len(args.cases), 2, figsize=(10, 4*len(args.cases)), squeeze=False)
    for row, case in enumerate(args.cases):
        g = data[case]['sparse_3d_global']['rgb']
        h, w = g.shape[:2]
        norm = g[h//4:3*h//4, w//4:3*w//4, 0].mean()
        for col, mode in enumerate(MODES[:-1]):
            delta = blocks(data[case][mode]['rgb']-g)[..., 0]/norm
            im = axes[row, col].imshow(delta, cmap='RdBu_r', vmin=-.3, vmax=.3)
            axes[row, col].set_title(TITLES[CASES.index(case)] + '\n' + mode + ' − GPIS')
            axes[row, col].set_xticks([]); axes[row, col].set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), label='ΔY / GPIS 中央区域平均亮度', shrink=.75, extend='both')
    fig.suptitle('统一色标 ±30%；超出范围的差值饱和显示', fontsize=14)
    fig.savefig(dest / 'difference.png', dpi=150); plt.close(fig)
    # Mean cross-sections disclose the actual analytic geometry and finite box.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot([-.5, .5], [0, 0], lw=3); axes[0].set_title('平面：m = y')
    a = np.linspace(0, np.pi, 200)
    axes[1].plot(RADIUS*np.cos(a), -RADIUS*np.sin(a), lw=3)
    for x in (-RADIUS, RADIUS): axes[1].plot([x, x], [0, .6], lw=3)
    axes[1].set_title('U 槽：圆底与直壁相切，沿 z 延伸')
    for ax in axes:
        ax.set_aspect('equal'); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(dest / 'geometry.png', dpi=150); plt.close(fig)
    cards = ''
    for case in args.cases:
        cards += '<h2>' + TITLES[CASES.index(case)] + f' · 统一显示增益 ×{gains[case]:g}</h2><div class="grid">'
        for mode, label in zip(MODES, LABELS):
            d = data[case][mode]
            rel = d['folder'].relative_to(dest).as_posix()
            cards += f'<div><b>{label}</b><a href="{d["preview"]}"><img src="{d["preview"]}"></a><p><a href="{rel}/render.png">原始 PNG</a> · <a href="{rel}/render.pfm">线性 PFM</a> · <a href="{rel}/scene.json">场景</a> · <a href="{rel}/render.log">日志</a></p></div>'
        cards += '</div>'
    rows = ''
    for case in args.cases:
        g = next(r for r in statistics if r['case']==case and r['mode']=='sparse_3d_global' and r['region']=='center_half')
        for mode in MODES:
            s = next(r for r in statistics if r['case']==case and r['mode']==mode and r['region']=='center_half')
            rows += f'<tr><td>{TITLES[CASES.index(case)]}</td><td>{mode}</td><td>{s["mean"]:.6f} ± {1.96*s["standard_error"]:.6f}</td><td>{100*(s["mean"]/g["mean"]-1):+.2f}%</td></tr>'
    validation = load(dest / 'validation.json') if (dest / 'validation.json').exists() else None
    material_validation = load(OUT/'material_check'/'validation.json') if (OUT/'material_check'/'validation.json').exists() else None
    material_validation_link = Path(os.path.relpath(OUT/'material_check'/'validation.json', dest)).as_posix()
    check_table = ''
    if validation:
        check_rows=''
        for check in validation['checks']:
            label='步长减半' if check['check']=='half_step' else '扩大侧向与下方计算域'
            status='未超过采样区间' if check['within_approximate_95_percent_sampling_interval'] else '超过采样区间，存在可测敏感性'
            check_rows += f'<tr><td>{TITLES[CASES.index(check["case"])]}</td><td>{label}</td><td>{100*check["relative_change"]:+.2f}%</td><td>{status}</td></tr>'
        check_table='<table><tr><th>场景</th><th>检查</th><th>中央区域亮度变化</th><th>约 95% 采样区间</th></tr>'+check_rows+'</table>'
    notes = [f'三种模式使用完全相同的相机、照明、均值场、协方差和 {material} 微材质；无环境光。',
             'GPIS 每条路径保持同一个 3D 稀疏卷积随机场，不同路径独立采样；展示随机场平均外观。',
             'U 槽 m=R−sqrt(x²+min(y,0)²)，R=0.15；曲面法线在圆底与直壁连接处连续。计算域截断无限通道，统计使用中央半幅。',
             '无降噪、无逐模式曝光匹配。同一行的三种模式使用相同显示增益，具体数值标在各行标题中；统计始终使用未经映射的线性亮度。',
             'GPIS 是有限密度稀疏卷积和离散步进的数值参考；步长检查与采样误差分别记录。']
    body = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>粗糙平面与圆底 U 槽对照</title>
<style>body{{max-width:1450px;margin:35px auto;padding:0 24px;font:16px/1.6 system-ui;color:#233044;background:#f5f6f8}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px}}.grid>div{{background:white;padding:12px}}img{{width:100%}}a{{color:#24699b}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}.note{{background:#e7edf5;padding:16px}}@media(max-width:800px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>粗糙平面与圆底 U 槽：classic_global / conditional / GPIS global</h1>
<p>σ={SIGMA:g}，ℓ={ELL:g}，每个梯度扰动分量的标准差={SIGMA/ELL:g}；{material} 微材质，反照率 {ALBEDO:g}；{size[0]}×{size[1]}，{spp} spp。</p>
<div class="note">{'<br>'.join(map(html.escape, notes))}</div>
<p><a href="comparison.png">完整对比图</a> · <a href="manifest.json">参数与来源</a> · <a href="geometry.png">几何截面</a></p>
<p>{'<a href="../pilot/comparison.html">镜面微材质预览（64×64、16 spp，噪声较大）</a>' if args.stage=='final' and (OUT/'pilot/comparison.html').exists() else ''}</p>
{cards}<h2>中央半幅线性亮度</h2><p>± 为约 95% 蒙特卡洛采样区间，不包含离散步长偏差。</p>
<table><tr><th>场景</th><th>模式</th><th>平均亮度</th><th>相对 GPIS</th></tr>{rows}</table>
<h2>16×16 分区亮度差</h2><p>差值统一除以 GPIS 中央半幅的平均亮度。色标固定为 ±30%，超出范围的值饱和显示；准确的区域均值见上表。</p><a href="difference.png"><img src="difference.png"></a>
<details><summary>查看分区平均外观</summary><a href="block_means.png"><img src="block_means.png"></a></details>
<h2>几何截面</h2><img src="geometry.png">
<h2>数值核验</h2><p>{'<a href="validation.json">查看步长与计算域检查</a>' if validation else '步长与计算域检查尚未汇总。'}</p>{check_table}
<p>{f'<a href="{material_validation_link}">Lambert 微材质白炉检查：NEE 开／关均通过</a>' if material_validation and material_validation['passed'] else ''}</p></html>'''
    (dest / 'comparison.html').write_text(body, encoding='utf-8')
    save(dest / 'manifest.json', dict(complete=True, stage=args.stage, resolution=list(size), spp=spp,
         sigma=SIGMA, ell=ELL, radius=RADIUS, albedo=ALBEDO, material=material, display_gains=gains,
         renderer_sha256=next(iter(versions)), sources=sources, statistics=statistics,
         pair_metrics=pairs, notes=notes, validation=validation, material_validation=material_validation,
         figures={n: sha(dest/n) for n in ('comparison.png','difference.png','geometry.png','block_means.png')}))
    print(dest / 'comparison.html', flush=True)


def validate(args):
    """Small paired sensitivity renders; do not present GPIS as converged truth."""
    checks = []
    for suffix, step, scale in (('base', args.step, 1.), ('half_step', args.step/2, 1.),
                                ('larger_domain', args.step, 1.5)):
        job = copy.copy(args)
        job.stage = args.stage + '_check_' + suffix
        job.size, job.spp = 64, 64
        job.step, job.domain_scale = step, scale
        job.modes = ['sparse_3d_global']
        render(job)
    for case in args.cases:
        base_job = copy.copy(args); base_job.stage = args.stage + '_check_base'
        base = folder(base_job, case, 'sparse_3d_global')
        a = pfm(base/'render.pfm')[..., 0]
        ah = [pfm(base/f'samples{c}.pfm')[..., 0] for c in ('A','B')]
        sl = (slice(16,48),slice(16,48))
        for suffix in ('half_step','larger_domain'):
            other_job = copy.copy(args); other_job.stage = args.stage + '_check_' + suffix
            other = folder(other_job, case, 'sparse_3d_global')
            b = pfm(other/'render.pfm')[..., 0]
            bh = [pfm(other/f'samples{c}.pfm')[..., 0] for c in ('A','B')]
            delta = b[sl]-a[sl]
            half_diff = (bh[0][sl]-ah[0][sl])-(bh[1][sl]-ah[1][sl])
            se = float(.5*np.sqrt(np.sum(half_diff**2))/delta.size)
            checks.append(dict(case=case, check=suffix, baseline=str(base), refined=str(other),
                region='center_half', size=[64,64], spp=64, baseline_mean=float(a[sl].mean()),
                refined_mean=float(b[sl].mean()), difference=float(delta.mean()),
                relative_change=float(delta.mean()/a[sl].mean()), paired_half_standard_error=se,
                within_approximate_95_percent_sampling_interval=bool(abs(delta.mean())<=1.96*se)))
    save(OUT/args.stage/'validation.json', dict(checks=checks,
         interpretation='Paired even/odd half estimates measure sampling uncertainty. A finite step-halving check does not establish convergence.',
         domain_check='Expand x/z extents and lower y; upper y is held fixed to keep the camera outside the process box.'))
    for c in checks:
        print(c['case'],c['check'],f'{100*c["relative_change"]:+.2f}%',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('render', 'report', 'validate'))
    p.add_argument('--stage', default='final'); p.add_argument('--size', type=int, default=128)
    p.add_argument('--spp', type=int, default=128); p.add_argument('--threads', type=int, default=32)
    p.add_argument('--seed', type=int, default=SEED); p.add_argument('--step', type=float, default=.004)
    p.add_argument('--domain-scale', type=float, default=1.)
    p.add_argument('--material', choices=('lambert','mirror'), default='lambert')
    p.add_argument('--cases', nargs='+', choices=CASES, default=list(CASES))
    p.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    args = p.parse_args()
    if min(args.size, args.spp, args.threads) <= 0 or args.size % 16 or args.spp % 2:
        p.error('positive size divisible by 16, even spp and positive threads required')
    if not math.isfinite(args.step) or args.step <= 0: p.error('step must be positive and finite')
    if not math.isfinite(args.domain_scale) or args.domain_scale < 1: p.error('domain scale must be finite and >= 1')
    {'render':render, 'report':report, 'validate':validate}[args.command](args)
