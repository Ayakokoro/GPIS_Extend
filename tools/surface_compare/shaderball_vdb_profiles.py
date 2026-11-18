"""Reproduce VDB shaderball ray profiles; run from repository root.

Build gpis_shaderball_bake, ray_extinction_probe, tungsten, and the standalone
tools/surface_compare/nvdb_to_vdb converter first. Use --setup, --run, --report
or any combination. All output is local and no original scene is modified.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/shaderball_vdb_ray_profiles_20261117'
SOURCE = ROOT/'gpis-scenes/shaderball/shaderball-stationary-270.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(out):
    text = SOURCE.read_text(encoding='utf-8-sig')
    text = re.sub(r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*[\s\S]*?\*/',
                  lambda m: m[0] if m[0].startswith('"') else ' ', text)
    source = json.loads(text)
    source['media'][0]['gaussian_process']['covariance'].update(lengthScale=.05, sigma=.04)
    save(out/'source.json', source)
    scene = copy.deepcopy(source)
    gp = scene['media'][0]['gaussian_process']
    gp['mean'] = dict(type='tabulated', volume=False, scale=1., offset=0.,
        grid=dict(type='vdb', file=str(out/'fields/shaderball_sdf.vdb'), density_name='sdf',
                  density_scale=1., emission_name='', interpolate='linear', normalize_size=False,
                  request_sdf=False, integration_method='raymarching'))
    scene['media'] = [dict(name='gp', type='global_conditional',
        material=source['media'][0]['phase_function']['bsdf'], max_bounces=1024,
        gaussian_process=gp, tracking_step=0., tracking_fixed_step_limit=False,
        tracking_local_far_start_scale=4., tracking_transition_point_bounds=False)]
    scene['camera']['resolution'] = [256, 256]
    scene['integrator'].update(use_cuda_gpis=False, independent_path_sampling=True, wavefront_gpis=False)
    scene['renderer'].update(output_file='render.png', hdr_output_file='render.pfm',
        output_directory='', overwrite_output_files=True, spp=4, spp_step=4, output_buffers=[])
    save(out/'scene.json', scene)
    smoke = copy.deepcopy(scene)
    smoke['media'][0]['type'] = 'classic_global'
    smoke['camera']['resolution'] = [32, 32]
    save(out/'classic_smoke/scene.json', smoke)
    # Match the float transform used by the baker and renderer.
    import struct
    f32 = lambda x: struct.unpack('f', struct.pack('f', x))[0]
    translation = [f32(x) for x in source['media'][0]['gaussian_process']['mean']['transform']['position']]
    box = next(p['transform'] for p in source['primitives'] if p.get('name') == 'processBox')
    box_min = [f32(f32(x)-f32(box['scale']*.5)) for x in box['position']]
    box_max = [f32(f32(x)+f32(box['scale']*.5)) for x in box['position']]
    for name, pixel in [('center', [128, 128]), ('left_lower', [102, 152]), ('right_upper', [156, 105])]:
        save(out/name/'spec.json', dict(scene=str(out/'scene.json'), pixel=pixel, sample=0,
            seed=20261024, box_min=box_min, box_max=box_max, length_scales=[.05],
            denominator=16384, slice_denominator=192, slice_extent=1.5,
            export_majorants=True, reuse_loaded_mean=True, procedural_translation=translation))


def execute(command, log):
    print('Running', ' '.join(str(x) for x in command), flush=True)
    with log.open('w', encoding='utf-8') as stream:
        subprocess.run([str(x) for x in command], cwd=ROOT, stdout=stream,
                       stderr=subprocess.STDOUT, check=True)


def run(out):
    field = out/'fields/shaderball.nvdb'
    vdb = out/'fields/shaderball_sdf.vdb'
    if not field.exists():
        execute([ROOT/'build/Release/gpis_shaderball_bake.exe', out/'source.json', field, '.005'], out/'bake.log')
    if not vdb.exists():
        execute([ROOT/'build/nvdb-to-vdb/Release/gpis_nvdb_to_vdb.exe', field, vdb, 'sdf'], out/'convert.log')
    baked = read(Path(str(field)+'.json'))
    assert baked['dimensions'] == [609]*3 and baked['active_nodes'] == 225866529
    assert baked['dx'] == .005 and baked['sigma'] == .04
    assert baked['source_mean'] == read(out/'source.json')['media'][0]['gaussian_process']['mean']
    verification = read(Path(str(vdb)+'.json'))
    assert verification['verified_nodes'] == 609**3 == 225866529
    assert verification['bitwise_value_mismatches'] == verification['active_state_mismatches'] == 0
    assert verification['value_type'] == 'float32'
    assert all(abs(x-.005) < 1e-14 for x in verification['voxel_size'])
    assert verification['transform_max_abs_error'] <= 1e-12
    smoke = out/'classic_smoke'
    if not (smoke/'render.pfm').exists():
        execute([ROOT/'build/Release/tungsten.exe', '-t', '4', '--spp', '4', '-d', smoke,
                 smoke/'scene.json'], smoke/'render.log')
    def probe(spec):
        if not (spec.parent/'curves.json').exists():
            execute([ROOT/'build/Release/ray_extinction_probe.exe', spec, spec.parent/'curves.json'], spec.parent/'probe.log')
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(probe, sorted(out.glob('*/spec.json'))))


def report(out):
    os.environ['MPLCONFIGDIR'] = str(ROOT/'build/ray_curve_matplotlib')
    deps = ROOT/'build/fp-gradient-validation-deps'
    if deps.exists():
        sys.path.insert(0, str(deps))
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    chinese_font = Path('C:/Windows/Fonts/msyh.ttc')
    if chinese_font.exists():
        font_manager.fontManager.addfont(str(chinese_font))
        font_family = [font_manager.FontProperties(fname=str(chinese_font)).get_name(), 'DejaVu Sans']
    else:
        font_family = ['DejaVu Sans']
    plt.rcParams.update({'font.family': font_family, 'font.size': 9, 'svg.fonttype': 'none'})
    blue, orange, purple = '#2465bd', '#d75b24', '#8a439f'
    all_summaries, cards, overview = [], [], []
    verification = read(out/'fields/shaderball_sdf.vdb.json')
    with (out/'classic_smoke/render.pfm').open('rb') as stream:
        assert stream.readline().strip() == b'PF'
        w, h = map(int, stream.readline().split())
        scale = float(stream.readline())
        rgb = np.frombuffer(stream.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)
    assert (w, h) == (32, 32) and np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() > 0
    smoke_summary = dict(width=w, height=h, spp=4, minimum=float(rgb.min()),
                         maximum=float(rgb.max()), mean=float(rgb.mean()), finite=True)

    def stairs(ax, rows, color, label, lw=1.2):
        xs, ys = [], []
        for r in rows:
            xs.extend([r['begin'], r['end']]); ys.extend([r['maximum']]*2)
        ax.plot(xs, ys, color=color, label=label, lw=lw)

    def phase_bands(ax, bound, old, end):
        splits = ([0, min(bound['birth_split'], end), min(bound['far_split'], end), end]
                  if old else [0, min(bound['far_split'], end), end])
        names = ['近起点段', '过渡段', '远段'] if old else ['近起点段（自适应）', '远段']
        colors = ['#83c5be', '#ffdf91', '#b8bde8'] if old else ['#a7d5c8', '#b8bde8']
        for a, b, name, color in zip(splits, splits[1:], names, colors):
            if b <= a:
                continue
            ax.axvspan(a, b, color=color, alpha=.22)
            left, right = ax.get_xlim()
            a, b = max(a, left), min(b, right)
            if b <= a:
                continue
            x = np.sqrt(a*b) if ax.get_xscale() == 'log' else .5*(a+b)
            # Avoid illegible labels on a micron-size prefix in linear view.
            fraction = (np.log(b/a)/np.log(right/left) if ax.get_xscale() == 'log'
                        else (b-a)/(right-left))
            if fraction > .055:
                ax.text(x, .97, name, ha='center', va='top', transform=ax.get_xaxis_transform(), fontsize=8)

    for file in sorted(out.glob('*/curves.json')):
        data = read(file)
        group = file.parent.name
        for ray in data['rays']:
            p = ray['profiles'][0]; bound = p['majorants']; rows = p['points']
            a = {k: np.array([r.get(k, np.nan) for r in rows], dtype=float) for k in
                 ['t', 'mean', 'classic_mean', 'procedural_mean', 'conditional_mean',
                  'conditional_sd', 'classic', 'conditional']}
            t = a['t']; end = ray['length']; event = ray['sampled_event'].get('distance')
            check = {}
            sampled_bounds = {}
            for method in ['classic', 'old_three', 'current_two']:
                intervals = bound[method]
                starts = np.array([b['begin'] for b in intervals])
                ends = np.array([b['end'] for b in intervals])
                maxima = np.array([b['maximum'] for b in intervals])
                assert starts[0] == 0 and ends[-1] == end
                assert np.all(ends[:-1] == starts[1:]) and np.all(ends > starts)
                assert np.isfinite(maxima).all() and (maxima >= 0).all()
                selected = np.clip(np.searchsorted(starts, t, side='right')-1, 0, len(starts)-1)
                sampled_bounds[method] = maxima[selected]
                hazard = a['classic' if method == 'classic' else 'conditional']
                excess = hazard - sampled_bounds[method]
                scaled = excess/(1+sampled_bounds[method])
                check[method] = dict(intervals=len(intervals), max_scaled_excess=float(max(0., scaled.max())),
                    violations=int(np.sum(scaled > 1e-8)),
                    integrated_majorant=float(np.dot(ends-starts, maxima)))
                assert check[method]['violations'] == 0, (group, ray['kind'], method, check[method])
            assert p['max_mean_error'] <= 1e-10 and p['max_hazard_formula_error'] <= 1e-8
            stem = group+'_'+ray['kind']
            fig, axes = plt.subplots(4, 2, figsize=(15, 13), constrained_layout=True)
            for column in range(2):
                for row, ax in enumerate(axes[:, column]):
                    if column:
                        ax.set_xscale('log'); ax.set_xlim(5e-9, min(end, .25))
                    else:
                        ax.set_xlim(0, end)
                    ax.grid(alpha=.2)
                    ax.set_xlabel('Distance from conditioning birth t [world units]')
                    if event is not None:
                        ax.axvline(event, color='#66717b', ls='--', lw=.8,
                                   label='Sampled next collision' if row == 0 else None)
                    if bound['far_split'] < end:
                        ax.axvline(bound['far_split'], color='#8463a8', ls=':', lw=.9)
                ax = axes[0, column]
                ax.plot(t, a['procedural_mean'], 'k--', lw=1., label='Original procedural m(t)')
                ax.plot(t, a['classic_mean'], color=blue, lw=1.8, label='VDB mean: classic')
                ax.plot(t, a['mean'], color=orange, ls=':', lw=1.6, label='VDB mean: conditional')
                ax.plot(t, a['conditional_mean'], color=purple, lw=1., label='Conditioned mean E[F(t) | F0,G0]')
                ax.axhline(0, color='#888', lw=.6)
                ax.set_ylabel('Mean field value')
                ax.set_title('Full frozen flight' if column == 0 else 'Near-birth magnification (log distance)')
                ax.legend(fontsize=7, loc='best')
                ax = axes[1, column]
                ax.plot(t, a['classic'], color=blue, label='classic_global')
                ax.plot(t, a['conditional'], color=orange, label='global_conditional')
                ax.set_ylabel('Extinction [1 / world unit]'); ax.legend(fontsize=8)
                if column:
                    ax.set_yscale('symlog', linthresh=1e-8)
                    ax.set_ylabel('Extinction (symlog) [1 / world unit]')
                for row, method, old in [(2, 'old_three', True), (3, 'current_two', False)]:
                    ax = axes[row, column]
                    phase_bands(ax, bound, old, end)
                    ax.plot(t, a['classic'], color=blue, lw=.65, alpha=.6, label='Classic extinction')
                    ax.plot(t, a['conditional'], color=orange, lw=.65, alpha=.6, label='Conditional extinction')
                    stairs(ax, bound['classic'], blue, 'Classic majorant')
                    stairs(ax, bound[method], purple, 'Conditional majorant')
                    ax.set_yscale('symlog', linthresh=1e-3)
                    ax.set_ylabel(('Old 3 phases' if old else 'Current 2 phases')+'\nMajorant [1 / world unit]')
                    ax.set_ylim(bottom=0); ax.legend(fontsize=7, loc='lower right')
                # Match extinction scales to the data in each visible window.
                mask = (t >= axes[1, column].get_xlim()[0]) & (t <= axes[1, column].get_xlim()[1])
                top = max(a['classic'][mask].max(), a['conditional'][mask].max())
                axes[1, column].set_ylim(0, max(top*1.12, 1e-8))
                means = np.concatenate([a[k][mask] for k in ['mean','procedural_mean','conditional_mean']])
                means = means[np.isfinite(means)]
                margin = max(np.ptp(means)*.08, .002)
                axes[0, column].set_ylim(means.min()-margin, means.max()+margin)
                left, right = axes[2, column].get_xlim()
                ceiling = max(r['maximum'] for method in sampled_bounds for r in bound[method]
                              if r['end'] >= left and r['begin'] <= right)
                for row in [2,3]:
                    axes[row,column].set_ylim(0,max(ceiling*1.2,.01))
            fig.suptitle(f'{group} pixel {data["spec"]["pixel"]} | {ray["kind"]} | ell=0.05, sigma=0.04\n'
                f'F0={ray["birth_value"]:.6g}, w.G0={ray["birth_axial_gradient"]:.6g}; '
                f'birth split={bound["birth_split"]:.6g}, far split={bound["far_split"]:.6g}', fontsize=12)
            for ext in ['png', 'svg']:
                fig.savefig(out/f'{stem}.{ext}', dpi=150)
            plt.close(fig)
            columns = list(a)+[name+'_majorant' for name in sampled_bounds]
            with (out/f'{stem}.csv').open('w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f); writer.writerow(columns)
                writer.writerows(zip(*a.values(), *sampled_bounds.values()))
            for method in sampled_bounds:
                with (out/f'{stem}_{method}_intervals.csv').open('w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=['begin', 'end', 'maximum', 'phase'])
                    writer.writeheader(); writer.writerows(bound[method])
            summary = dict(name=stem, pixel=data['spec']['pixel'], kind=ray['kind'],
                birth_value=ray['birth_value'], birth_axial_gradient=ray['birth_axial_gradient'],
                length=end, sampled_event=event, birth_split=bound['birth_split'], far_split=bound['far_split'],
                mean_shared_max_error=p['max_mean_error'],
                procedural_vdb_mean_max_error=float(np.max(abs(a['mean']-a['procedural_mean']))),
                hazard_formula_max_scaled_error=p['max_hazard_formula_error'], sampled_points=len(t), bounds=check,
                replay_checks={k:bound[k] for k in ['old_three_replay_check','current_two_replay_check'] if k in bound})
            far = t >= bound['far_split']
            summary['old_current_far_majorant_max_difference'] = (
                float(np.max(abs(sampled_bounds['old_three'][far]-sampled_bounds['current_two'][far]))) if far.any() else None)
            all_summaries.append(summary)
            overview.append((stem, a, bound))
            cards.append(f'<section><h2>{html.escape(stem)} · 像素 {data["spec"]["pixel"]}</h2>'
                f'<p>F₀={ray["birth_value"]:.7g}，w·G₀={ray["birth_axial_gradient"]:.7g}；'
                f'旧近起点截止 {bound["birth_split"]:.8g}；4ℓ 截止 {bound["far_split"]:.8g}。</p>'
                f'<a href="{stem}.svg"><img src="{stem}.png"></a>'
                f'<p><a href="{stem}.csv">全部采样 CSV</a> · <a href="{stem}.svg">SVG</a> · '
                + ' · '.join(f'<a href="{stem}_{m}_intervals.csv">{m} 区间</a>' for m in sampled_bounds)
                + '</p></section>')
        # Locate both frozen rays in a real VDB section.
        section = data['slice']; n = section['denominator']; e = section['extent']
        axis = np.linspace(-e, e, n+1); values = np.array(section['values']).reshape(n+1, n+1)
        fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
        ax.contourf(axis, axis, values, levels=[-10, 0, .04, .12, 10], colors=['#b9c4cf','#f7dfb2','#fdf3df','#fff'])
        ax.contour(axis, axis, values, levels=[0], colors=['#384c5a'])
        basis = np.array([section['basis_u'], section['basis_v']]); center = np.array(section['center'])
        for ray, color in zip(data['rays'], [blue, orange]):
            origin, direction = np.array(ray['origin']), np.array(ray['direction'])
            for length, style in [(ray['length'], ':'), (ray['sampled_event'].get('distance', ray['length']), '-')]:
                xy = (np.array([origin, origin+length*direction])-center)@basis.T
                ax.plot(*xy.T, style, color=color, lw=1.6, label=ray['kind'] if style=='-' else None)
        ax.scatter(0, 0, marker='*', s=80, color='k'); ax.legend()
        ax.set(xlim=(-e,e), ylim=(-e,e), aspect='equal', xlabel='u [world units]', ylabel='v [world units]',
               title=f'{group}: actual VDB section; solid = sampled flight, dotted = virtual extension')
        fig.savefig(out/f'{group}_location.png', dpi=150); plt.close(fig)

    # One compact figure for all rays, in the exact order requested.
    fig, axes = plt.subplots(len(overview), 3, figsize=(16, 2.8*len(overview)), constrained_layout=True)
    for axs, (name, a, b) in zip(axes, overview):
        end = min(.5, a['t'][-1]); t = a['t']; mask = t <= end
        axs[0].plot(t, a['mean'], color=blue); axs[0].plot(t, a['conditional_mean'], color=purple, lw=1.)
        axs[1].plot(t, a['classic'], color=blue); axs[1].plot(t, a['conditional'], color=orange)
        for method, color in [('classic', blue), ('old_three', orange), ('current_two', purple)]:
            stairs(axs[2], b[method], color, method)
        axs[2].set_yscale('symlog', linthresh=1e-3); axs[2].set_ylim(bottom=0)
        axs[2].axvline(.2, color='#666', ls=':'); axs[2].legend(fontsize=7)
        for ax in axs:
            ax.set_xlim(0,end); ax.grid(alpha=.2); ax.set_xlabel('t [world units]')
        for ax, series in [(axs[0], [a['mean'], a['conditional_mean']]), (axs[1], [a['classic'],a['conditional']])]:
            vals = np.concatenate([s[mask] for s in series]); vals = vals[np.isfinite(vals)]
            margin = max(np.ptp(vals)*.1,1e-5); ax.set_ylim(vals.min()-margin, vals.max()+margin)
        axs[0].set_ylabel(name, fontsize=8)
    for ax, title in zip(axes[0], ['Mean: VDB (blue), conditioned (purple)', 'Extinction: classic / conditional', 'Majorants: classic / old 3 / current 2']):
        ax.set_title(title)
    fig.savefig(out/'overview.png', dpi=150); fig.savefig(out/'overview.svg'); plt.close(fig)
    summary = dict(conversion=verification, smoke=smoke_summary, rays=all_summaries,
        settings=dict(ell=.05, sigma=.04, old_three=dict(tracking_transition_point_bounds=True,
            tracking_local_far_start_scale=4), current_two=dict(tracking_transition_point_bounds=False,
            tracking_local_far_start_scale=4), shared=dict(tracking_fixed_step_limit=False,
            tracking_macrocell_size=0, tracking_step=0, tracking_age_scale=.25,
            tracking_max_expected_events=64, tracking_tighten_expected_events=4)),
        limitations=['Bounds are production formula evaluations on a deterministic complete-flight proposal replay; no sampled null events.',
            'Extensions beyond the marked sampled collision are virtual frozen observations, not continuing physical paths.',
            'Dense enclosure checks are diagnostics, not a replacement for the analytic certificates.'])
    save(out/'summary.json', summary)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball VDB 光线曲线</title>
<style>body{max-width:1500px;margin:36px auto;padding:0 24px;font:16px/1.7 system-ui;color:#243444;background:#f8fafc}
img{width:100%;background:white}section{margin:36px 0;padding:24px;background:white;border:1px solid #d9e2ec;border-radius:10px}
a{color:#2465bd}code{background:#eef2f6;padding:2px 5px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>
<h1>原大光源 shaderball · ℓ=0.05，σ=0.04</h1>
<p>609³、float32、0.005 体素间距，原坐标变换。NanoVDB → OpenVDB 由 <code>convert.cpp</code> 编译工具直接转换；
225,866,529 个节点写盘重载后逐位匹配，激活状态匹配。classic_global 32×32、4 spp 验证通过。</p>
<p>VDB 保存均值 SDF；ℓ 和 σ 保存在场景的协方差参数中。原均值的世界坐标变换已烘焙进网格，不重复施加平移。</p>
<p>三组相机像素各取入射线和第一次反射线，共六条固定光线。两种模型使用同一个 VDB；条件模型固定实际采样的 F₀、G₀。
均值图同时显示共同的 VDB 均值、原程序化均值以及条件后验均值；共同均值重合是预期结果。</p>
<p><b>旧三段：</b>近起点 [0,t_birth)、过渡 [t_birth,0.20)、远段 [0.20,end]，过渡采用 midpoint/full conditional bounds。
<b>现两段：</b>近段 [0,0.20) 自适应，远段 [0.20,end] 局部 SE 上界。近段内部仍包含起点证书。
<code>tracking_transition_point_bounds</code> 旧=true、现=false；远段均为 <code>tracking_local_far_start_scale=4</code>。
两段指算法阶段数量，阶段内仍有多个常数上界区间。</p>
<p>上界来自生产公式，按固定设置重放完整 proposal/refinement 分区，不包含随机 null events。
虚线标记实际采样的下一次碰撞，之后曲线是保持起点观测的虚拟延长。主图完整显示到几何边界；右列对数横轴放大近起点。
上界纵轴使用 symlog（线性阈值 10⁻³）；右列消光纵轴使用 symlog（线性阈值 10⁻⁸），保留精确零并显示微小消光。
采样点覆盖检查通过不能替代解析证明。</p>
<p><a href="summary.json">汇总与检查</a> · <a href="fields/shaderball_sdf.vdb.json">逐节点转换核验</a> ·
<a href="scene.json">VDB 场景</a> · <a href="manifest.json">复现记录与哈希</a> · <a href="overview.svg">总览 SVG</a></p>
<img src="overview.png"><h2>光线位置</h2>LOCATIONS CARDS</html>'''
    locations = ''.join(f'<img style="max-width:32%" src="{p.parent.name}_location.png">' for p in sorted(out.glob('*/curves.json')))
    (out/'comparison.html').write_text(page.replace('LOCATIONS',locations).replace('CARDS',''.join(cards)), encoding='utf-8')
    files = [SOURCE, Path(__file__), ROOT/'tools/surface_compare/ray_extinction.cpp',
        ROOT/'tools/surface_compare/ray_majorants.hpp', ROOT/'tools/surface_compare/nvdb_to_vdb/convert.cpp',
        ROOT/'tools/surface_compare/knob_bake.cpp', ROOT/'src/core/media/ClassicGlobalMedium.cpp',
        ROOT/'src/core/math/conditional/transport/ConditionalSegment.cpp',
        ROOT/'src/core/math/conditional/transport/ConditionalFlightKernel.cpp',
        ROOT/'build/Release/ray_extinction_probe.exe', ROOT/'build/Release/tungsten.exe',
        ROOT/'build/Release/gpis_shaderball_bake.exe', ROOT/'build/nvdb-to-vdb/Release/gpis_nvdb_to_vdb.exe',
        out/'fields/shaderball.nvdb', out/'fields/shaderball_sdf.vdb']
    files.extend(out.glob('*/curves.json')); files.extend(out.glob('*/spec.json')); files.extend(out.glob('*.json'))
    files = [p for p in files if p.name != 'manifest.json']
    save(out/'manifest.json', dict(command=f'python -B tools/surface_compare/shaderball_vdb_profiles.py --setup --run --report --output {out}',
        sha256={str(p.relative_to(ROOT)):digest(p) for p in files}, python=sys.executable))
    print(json.dumps(dict(rays=len(all_summaries), smoke=smoke_summary,
                         violations=sum(c['violations'] for r in all_summaries for c in r['bounds'].values())), indent=2))
    print(out/'comparison.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--setup', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report', action='store_true')
    args = parser.parse_args(); out = args.output.resolve()
    if not any((args.setup,args.run,args.report)):
        args.setup = args.run = args.report = True
    if args.setup: setup(out)
    if args.run: run(out)
    if args.report: report(out)
