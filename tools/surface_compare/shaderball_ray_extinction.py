"""Plot production extinction laws on a frozen shaderball camera/reflection path.

Build: cmake --build build --config Release --target ray_extinction_probe
Run:   python -B tools/surface_compare/shaderball_ray_extinction.py
Use --pixel X Y / --sample N to select another path, or --plot-only to redraw.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
# Optional workspace-only plotting dependencies; installed environments work too.
if (ROOT / 'build/ray_curve_python').exists():
    sys.path.insert(0, str(ROOT / 'build/ray_curve_python'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/ray_curve_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(ell):
    return f'{ell:g}'.replace('.', 'p')


def cumulative(t, h):
    return np.r_[0., np.cumsum((h[:-1] + h[1:]) * np.diff(t) / 2)]


def plot_profile(out, ray, profile):
    points = profile['points']
    arrays = {key: np.array([p.get(key, np.nan) for p in points]) for key in
              ['t', 'mean', 'mean_derivative', 'mean_gradient_norm', 'classic',
               'conditional', 'conditional_mean', 'conditional_sd', 'crossing_mean', 'crossing_sd']}
    t, classic, conditional = (arrays[k] for k in ['t', 'classic', 'conditional'])
    tau = {k: cumulative(t, arrays[k]) for k in ['classic', 'conditional']}
    ell = profile['length_scale']
    event = ray['sampled_event'].get('distance')
    # Keep the plotted interval physical. Full virtual extensions remain in CSV.
    end = min(ray['length'], event if event is not None else ray['length'])
    visible = t <= end
    if not np.any(visible[1:]):
        raise RuntimeError('Empty plotted flight')
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8), constrained_layout=True)
    blue, orange = '#2465c0', '#d45a22'
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.set(xlim=(0, end), xlabel='Distance from birth t [world units]')
        for q in [1, 2, 4]:
            if q * ell < end:
                ax.axvline(q * ell, color='#8b96a2', lw=.7, ls=':', alpha=.55)
        if event is not None and event <= end:
            ax.axvline(event, color='#52616c', lw=1, ls='--')
    axes[0, 0].plot(t, classic, color=blue, label='classic_global', lw=1.8)
    axes[0, 0].plot(t, conditional, color=orange, label='global_conditional', lw=1.5)
    upper = max(float(classic[visible].max()), float(conditional[visible].max()))
    axes[0, 0].set(ylim=(0, max(upper * 1.08, 1e-12)), ylabel='Extinction [1 / world unit]', title='Directional extinction')
    axes[0, 0].legend()
    floor = 1e-12
    axes[0, 1].semilogy(t, np.maximum(classic, floor), color=blue, lw=1.6)
    axes[0, 1].semilogy(t, np.maximum(conditional, floor), color=orange, lw=1.4)
    axes[0, 1].set(ylim=(floor, max(upper * 1.5, floor * 100)), ylabel='Extinction, logarithmic scale',
                   title='Display floor 1e-12; exact zero is preserved in data')
    mu, sd = arrays['conditional_mean'], arrays['conditional_sd']
    axes[1, 0].plot(t, arrays['mean'], color='#475867', label='Mean SDF m(t)', lw=1.8)
    axes[1, 0].plot(t, mu, color=orange, label='Conditional mean', lw=1.3)
    axes[1, 0].fill_between(t, mu - sd, mu + sd, color=orange, alpha=.18, label='Conditional mean +/- sd')
    axes[1, 0].axhline(0, color='#77818b', lw=.7)
    envelope = np.r_[arrays['mean'][visible], (mu-sd)[visible], (mu+sd)[visible]]
    low, high = np.nanmin(envelope), np.nanmax(envelope)
    margin = max((high-low)*.1, profile['sigma']*.25)
    axes[1, 0].set(ylim=(low-margin, high+margin), ylabel='Field value', title='SDF and conditional value statistics')
    axes[1, 0].legend(fontsize=9)
    for name, color in [('classic', blue), ('conditional', orange)]:
        axes[1, 1].plot(t, np.exp(-tau[name]), color=color, lw=1.6)
    axes[1, 1].set(ylim=(-.02, 1.02), ylabel='T(t) = exp(- integral extinction)', title='Transmittance along this frozen flight')
    label = 'Camera entry' if ray['kind'] == 'camera_entry' else 'After first mirror reflection'
    fig.suptitle(f'Shaderball | {label} | sigma={profile["sigma"]:g}, ell={ell:g}\n'
                 f'F0={ray["birth_value"]:.6g}, w dot G0={ray["birth_axial_gradient"]:.6g}; '
                 'dotted lines: t / ell = 1, 2, 4', fontsize=13)
    stem = f'{ray["kind"]}_ell{tag(ell)}'
    for extension in ['png', 'svg']:
        fig.savefig(out / f'{stem}.{extension}', dpi=150)
    plt.close(fig)
    with (out / f'{stem}.csv').open('w', newline='', encoding='utf-8') as f:
        columns = list(arrays) + ['classic_T', 'conditional_T']
        writer = csv.writer(f)
        writer.writerow(columns)
        writer.writerows(zip(*arrays.values(), np.exp(-tau['classic']), np.exp(-tau['conditional'])))
    # Refinement is assessed on the physical plotted segment, not deep virtual
    # solid regions where both transmittances have already underflowed.
    physical_t = np.r_[t[t < end], end]
    checks = {}
    for name in tau:
        physical_h = np.interp(physical_t, t, arrays[name])
        fine = cumulative(physical_t, physical_h)[-1]
        indices = np.unique(np.r_[np.arange(0, len(physical_t), 2), len(physical_t)-1])
        coarse = cumulative(physical_t[indices], physical_h[indices])[-1]
        checks[name] = dict(optical_depth=float(fine), transmittance=float(np.exp(-fine)),
                            half_grid_T_difference=float(abs(np.exp(-fine)-np.exp(-coarse))))
    delta = np.abs(classic[visible] - conditional[visible])
    summary = dict(kind=ray['kind'], length_scale=ell, figure=stem+'.png', csv=stem+'.csv',
                   visible_end=end, actual_boundary_distance=ray['length'],
                   birth_value=ray['birth_value'], birth_axial_gradient=ray['birth_axial_gradient'],
                   max_classic=float(classic[visible].max()), max_conditional=float(conditional[visible].max()),
                   max_absolute_difference=float(delta.max()), integral_checks=checks,
                   max_mean_error=profile['max_mean_error'], max_hazard_formula_error=profile['max_hazard_formula_error'])
    for name in ['classic', 'conditional']:
        index = np.flatnonzero(visible)[np.argmax(arrays[name][visible])]
        summary[name+'_peak_distance'] = float(t[index])
    far = t >= 6 * ell
    summary['far_age_max_scaled_difference'] = (float(np.max(abs(classic[far]-conditional[far])/(1+classic[far]))) if far.any() else None)
    # Compact, evenly indexed data for a local-file-compatible point inspector.
    indices = np.unique(np.r_[0, np.arange(0, len(t), 4), len(t)-1])
    inspector = dict(summary=summary, t=t[indices].tolist(), classic=classic[indices].tolist(),
                     conditional=conditional[indices].tolist(), mean=arrays['mean'][indices].tolist())
    return summary, inspector


def geometry(out, data):
    section = data['slice']
    n, extent = section['denominator'], section['extent']
    axis = np.linspace(-extent, extent, n+1)
    values = np.array(section['values']).reshape(n+1, n+1)
    fig, ax = plt.subplots(figsize=(9, 7), constrained_layout=True)
    ax.contourf(axis, axis, values, levels=[-10, -.2, 0, .02, .1, .3, 10],
                colors=['#9ba9b4', '#d1dbe3', '#fff0cf', '#fff8e9', '#f5f8fb', '#ffffff'])
    ax.contour(axis, axis, values, levels=[0], colors=['#394b59'], linewidths=1.3)
    center = np.array(section['center'])
    basis = np.array([section['basis_u'], section['basis_v']])
    for ray, color, label in zip(data['rays'], ['#316fba', '#da6028'], ['Incoming camera ray', 'First reflected ray']):
        start = np.array(ray['origin'])
        direction = np.array(ray['direction'])
        length = min(ray['length'], ray['sampled_event'].get('distance', ray['length']))
        vertices = (np.array([start, start+length*direction])-center) @ basis.T
        ax.plot(*vertices.T, color=color, lw=2.2, label=label)
        end = vertices[-1]
        if ray['sampled_event'].get('distance') is not None:
            ax.scatter(*end, s=45, color=color, zorder=5)
    ax.scatter(0, 0, c='#222d36', marker='*', s=120, zorder=6, label='First GP collision / reflected-ray birth')
    ax.set(xlim=(-extent, extent), ylim=(-extent, extent), aspect='equal', xlabel='Section coordinate u [world units]',
           ylabel='Section coordinate v [world units]', title='Actual procedural shaderball SDF in the plane of both rays\nDark contour: m=0; gray: m<0')
    ax.legend(fontsize=9, loc='upper left')
    for extension in ['png', 'svg']:
        fig.savefig(out/f'ray_location.{extension}', dpi=150)
    plt.close(fig)


def report(out, data):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'font.size': 10})
    summaries, inspector = [], {}
    for ray in data['rays']:
        for profile in ray['profiles']:
            entry, points = plot_profile(out, ray, profile)
            summaries.append(entry)
            inspector[ray['kind']+'_'+tag(profile['length_scale'])] = points
    geometry(out, data)
    checks = dict(data['checks'])
    checks['shared_sdf_max_error'] = max(v['max_mean_error'] for v in summaries)
    checks['hazard_formula_max_scaled_error'] = max(v['max_hazard_formula_error'] for v in summaries)
    checks['half_grid_T_max_difference'] = max(q['half_grid_T_difference'] for v in summaries for q in v['integral_checks'].values())
    write(out/'summary.json', dict(curves=summaries, checks=checks, note=data['note']))
    rows = ''
    for v in summaries:
        rows += (f'<tr><td>{v["kind"]}</td><td>{v["length_scale"]}</td><td>{v["max_classic"]:.6g}</td>'
                 f'<td>{v["max_conditional"]:.6g}</td><td>{v["integral_checks"]["classic"]["transmittance"]:.6g}</td>'
                 f'<td>{v["integral_checks"]["conditional"]["transmittance"]:.6g}</td>'
                 f'<td><a href="{v["csv"]}">CSV</a> · <a href="{Path(v["figure"]).stem}.svg">SVG</a></td></tr>')
    ell_options = ''.join(f'<option value="{tag(p["length_scale"])}">ℓ={p["length_scale"]}</option>' for p in data['rays'][0]['profiles'])
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball 单条光线的消光系数</title>
<style>body{font:16px/1.65 system-ui,sans-serif;color:#233445;background:#f5f7fa;max-width:1350px;margin:30px auto;padding:0 22px}
img{width:100%;height:auto;background:white;border-radius:8px}select,input{font:inherit;padding:5px}input[type=range]{width:70%}
table{width:100%;border-collapse:collapse;background:white}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right}
th:first-child,td:first-child{text-align:left}a{color:#2164ad}pre{white-space:pre-wrap;background:#fff;padding:15px;font-size:13px}
.controls{position:sticky;top:0;background:#f5f7faee;padding:10px 0}code{background:#e7edf5;padding:2px 4px}</style>
<h1>Shaderball：沿固定射线比较消光系数</h1>
<p>直接调用当前 C++ 实现：classic_global 的 extinction(p,w) 与 global_conditional 的 ConditionalFlightKernel::evaluate(t)。
使用此前性能实验的解析 knob SDF、σ=0.02，相同位置和方向；此处曲线是真实消光系数，不是 tracking majorant。</p>
<p>相机像素 PIXEL，种子 SEED，样本 SAMPLE；相机分辨率与来源场景一致。射线的位置、方向和观测梯度见页面末尾的精确参数。</p>
<p>先用 ℓ=0.2 的真实相机、介质和镜面采样获得一条路径的入射段及第一次反射段，再冻结起点值 F₀、完整梯度 G₀和方向。
ℓ=0.05 也使用相同射线和观测，单独考察相关长度的影响；它不是重新采样的一条 ℓ=0.05 路径。诊断省略 NEE 和 roulette 的随机数消耗。</p>
<div class="controls"><label>射线 <select id="ray"><option value="first_reflection">第一次反射后</option><option value="camera_entry">相机入射段</option></select></label>
<label>相关长度 <select id="ell">ELL_OPTIONS</select></label></div>
<img id="curve" alt="消光系数、对数消光、SDF 和透射率曲线"><p id="state"></p>
<p>点值检查器（包含到包围盒／地面之前的完整直线延伸）：<input id="point" type="range" min="0" value="0"><br><output id="values"></output></p>
<p>图中虚线标出 ℓ=0.2 路径采样到的下一次 GP 碰撞；主图只画到该事件，逃逸段显示到包围盒／地面边界。CSV 和点值检查器还保存事件之后的虚拟直线延伸，
其中起点观测保持冻结，不能把该延伸当作真实多次碰撞路径。点线标出 t/ℓ=1、2、4。对数图将低于 10⁻¹² 的数值压到显示下限，原始数据未截断。
t=0 的 conditional=0 是当前实现的精确零约定；数据也密集包含了 t>0 的近起点值。</p>
<table><thead><tr><th>射线</th><th>ℓ</th><th>classic 峰值</th><th>conditional 峰值</th><th>classic T</th><th>conditional T</th><th>下载</th></tr></thead><tbody>ROWS</tbody></table>
<p>峰值和 T 对应主图显示的物理射线段。T 使用导出曲线的梯形积分；summary.json 同时记录半网格检查。</p>
<p>阅读曲线时，可以先看 SDF 的下降段，再比较 conditional mean 与原始 mean 的偏移。消光峰值还依赖 crossing 的负向梯度通量，
因此不是只由 SDF 是否接近零决定。已知 F₀=0 且 w·G₀&gt;0 时，起点附近确定地向外走，conditional 会抑制立刻再次碰撞；
距离超过若干相关长度后，起点记忆逐渐消退。</p>
<h2>射线在 shaderball 上的位置</h2><img src="ray_location.png" alt="真实 SDF 截面与入射、反射射线">
<p>截面通过入射线和反射线，星号是第一次 GP 碰撞，也是反射射线的条件化起点。灰色区域均值 SDF 为负。
这条曲线只描述给定最近观测后的实现 hazard；不会额外施加整段此前始终为正的首次穿零约束。</p>
<details><summary>精确射线参数</summary><pre>PARAMETERS</pre></details>
<p><a href="curves.json">全部原始采样</a> · <a href="summary.json">汇总与数值检查</a> · <a href="spec.json">输入配置</a> ·
<a href="manifest.json">源码、程序与结果哈希及复现命令</a> · <a href="../point_lipschitz/comparison.html">性能实验</a></p>
<script>const data=INSPECTOR;
const ray=document.getElementById('ray'),ell=document.getElementById('ell'),point=document.getElementById('point');
function selected(){return data[ray.value+'_'+ell.value]}
function values(){const d=selected(),i=Number(point.value);document.getElementById('values').textContent=
`t=${d.t[i].toPrecision(7)}, t/ℓ=${(d.t[i]/d.summary.length_scale).toPrecision(5)}, classic=${d.classic[i].toExponential(6)}, conditional=${d.conditional[i].toExponential(6)}, m=${d.mean[i].toPrecision(7)}`}
function update(){const d=selected(),s=d.summary;document.getElementById('curve').src=s.figure;
document.getElementById('state').textContent=`F₀=${s.birth_value.toPrecision(8)}, w·G₀=${s.birth_axial_gradient.toPrecision(8)}, 主图终点=${s.visible_end.toPrecision(7)}, 几何边界距离=${s.actual_boundary_distance.toPrecision(7)}`;
point.max=d.t.length-1;point.value=0;values()}ray.onchange=ell.onchange=update;point.oninput=values;update();</script></html>'''
    parameters = [dict((k, v) for k, v in r.items() if k != 'profiles') for r in data['rays']]
    page = page.replace('ELL_OPTIONS', ell_options).replace('ROWS', rows).replace('PARAMETERS', html.escape(json.dumps(parameters, indent=2)))
    page = page.replace('PIXEL', str(data['spec']['pixel'])).replace('SEED', str(data['spec']['seed'])).replace('SAMPLE', str(data['spec']['sample']))
    page = page.replace('INSPECTOR', json.dumps(inspector, separators=(',', ':')))
    (out/'comparison.html').write_text(page, encoding='utf-8')
    for target, href in [(out.parent/'comparison.html', 'ray_extinction/comparison.html'),
                         (out.parent/'point_lipschitz/comparison.html', '../ray_extinction/comparison.html')]:
        if target.exists():
            text = target.read_text(encoding='utf-8')
            if 'id="ray-extinction"' not in text:
                link = f'<p id="ray-extinction"><a href="{href}">新增：同一条 shaderball 射线的 classic / conditional 消光系数曲线</a></p>'
                target.write_text(text.replace('</h1>', '</h1>'+link, 1), encoding='utf-8')
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene', type=Path, default=ROOT/'outputs/global_conditional_shaderball_profile_20261113/point_statistics_reuse/sources/ell0p2.json')
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/global_conditional_shaderball_profile_20261113/ray_extinction')
    parser.add_argument('--executable', type=Path, default=ROOT/'build/Release/ray_extinction_probe.exe')
    parser.add_argument('--pixel', type=int, nargs=2, default=[128, 128])
    parser.add_argument('--sample', type=int, default=0)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--length-scales', type=float, nargs='+', default=[.2, .05])
    parser.add_argument('--denominator', type=int, default=16384)
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve();out.mkdir(parents=True, exist_ok=True)
    if not args.plot_only:
        scene = read(args.scene)
        if scene['media'][0]['gaussian_process']['mean'].get('func') != 'knob':
            raise RuntimeError('This diagnostic expects the procedural shaderball scene')
        box = next(p for p in scene['primitives'] if p.get('name') == 'processBox')['transform']
        if box.get('rotation', [0, 0, 0]) != [0, 0, 0]:
            raise RuntimeError('Expected axis-aligned processBox')
        # The production unit cube spans [-1/2,1/2] before its transform.
        position = np.array(box['position']);half = np.array(box['scale'])/2
        spec = dict(scene=str(args.scene.resolve()), pixel=args.pixel, sample=args.sample, seed=args.seed,
                    box_min=(position-half).tolist(), box_max=(position+half).tolist(),
                    length_scales=args.length_scales, denominator=args.denominator,
                    slice_denominator=256, slice_extent=1.5)
        write(out/'spec.json', spec)
        command = [str(args.executable.resolve()), str(out/'spec.json'), str(out/'curves.json')]
        with (out/'probe.log').open('w', encoding='utf-8') as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        files = [ROOT/'CMakeLists.txt', Path(__file__).resolve(), ROOT/'tools/surface_compare/ray_extinction.cpp', args.scene.resolve()]
        files.extend(p for p in (ROOT/'src/core').rglob('*') if p.suffix in ['.h', '.hpp', '.cpp'])
        write(out/'manifest.json', dict(command=command, python=sys.executable,
              invocation=sys.argv, executable_sha256=sha(args.executable),
              source_sha256={str(p.relative_to(ROOT)): sha(p) for p in files},
              curves_sha256=sha(out/'curves.json')))
    data = read(out/'curves.json')
    manifest = read(out/'manifest.json')
    if sha(out/'curves.json') != manifest['curves_sha256']:
        raise RuntimeError('Curve hash changed')
    summaries = report(out, data)
    for entry in summaries:
        for name, check in entry['integral_checks'].items():
            if check['half_grid_T_difference'] > 2e-4:
                raise RuntimeError(f'Curve grid insufficient for {entry["kind"]}, ell={entry["length_scale"]}, {name}')
    manifest['artifact_sha256'] = {p.name: sha(p) for p in out.iterdir()
                                 if p.suffix in ['.png', '.svg', '.csv', '.html'] or p.name == 'summary.json'}
    write(out/'manifest.json', manifest)
    print(json.dumps(summaries, indent=2))
    print(out/'comparison.html')


if __name__ == '__main__':
    main()
