"""Export F1-shaped JSON and SDF figures for the three-slab experiment.

cmake --build build --config Release --target slabs_ray_profiles
python -B tools/surface_compare/slabs_ray_profiles.py --run
python -B tools/surface_compare/slabs_ray_profiles.py  # regenerate report only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'build/fp-gradient-validation-deps'))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/ray_curve_matplotlib'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patheffects
from PIL import Image
from run import pfm

BASE = ROOT / 'outputs/gpis_three_slabs_compare_20261116'
OUT = BASE / 'sdf_ray_profiles_20261117'
SCENE = BASE / 'gauss_legendre5/seed20261024/scene.json'
TEMPLATE = ROOT / 'outputs/F1-ray-input.json'
EXE = ROOT / 'build/Release/slabs_ray_profiles.exe'
CASES = [
    dict(id='three-overlap', label='三重叠', pixel=[32, 32], expected_layers=3),
    dict(id='two-overlap', label='正常二重叠', pixel=[6, 32], expected_layers=2),
    dict(id='two-overlap-dark', label='暗弧二重叠', pixel=[6, 12], expected_layers=2),
]
COLORS = ['#0879b8', '#17835d', '#c83e4d']
METHODS = ['numeric', 'gauss_legendre5', 'gpis_renewal_plus']
NAMES = ['Single history / numeric', 'Single history / CDF5', 'Renewal+']
SEEDS = [20261024, 20261113]


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def template():
    # The supplied example has a trailing comma. Read its shape without editing it.
    s = TEMPLATE.read_text(encoding='utf-8-sig')
    return json.loads(re.sub(r',\s*([}\]])', r'\1', s))


def configure():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family': [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans'],
                         'axes.unicode_minus': False, 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})


def figure_save(fig, name):
    for extension in ['png', 'svg']:
        fig.savefig(OUT / f'{name}.{extension}', dpi=170, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def source_evidence():
    evidence, hashes = {}, {}
    shared = load(SCENE)
    for method in METHODS:
        evidence[method] = {}
        for seed in SEEDS:
            d = BASE / method / f'seed{seed}'
            scene = load(d / 'scene.json')
            assert scene['camera'] == shared['camera']
            assert scene['primitives'] == shared['primitives']
            assert scene['media'][0]['gaussian_process'] == shared['media'][0]['gaussian_process']
            assert scene['renderer']['spp'] == 256
            record = load(d / 'result.json')
            for name in ['render.png', 'render.pfm']:
                digest = sha(d / name)
                assert digest == record['sha256'][name]
                hashes[str((d / name).relative_to(ROOT))] = digest
            evidence[method][seed] = pfm(d / 'render.pfm')[:, :, 0]
    return evidence, hashes


def check_record(record, shape):
    for key, value in shape.items():
        assert key in record, key
        if isinstance(value, dict):
            assert set(value) <= set(record[key]), key
    p = np.array(record['sdf_curve']['samples'])
    ray = record['ray']
    origin, w = np.array(ray['origin']), np.array(ray['direction'])
    xyz = origin + p[:, :1] * w
    centers = np.array(record['geometry']['centers'])
    half = np.array(record['geometry']['half_size'])
    components = np.max(np.abs(xyz[:, None, :] - centers) - half, axis=2)
    error = float(np.max(np.abs(components.min(axis=1) - p[:, 2])))
    assert p.shape == (6657, 4) and np.isfinite(p).all()
    assert abs(np.linalg.norm(w) - 1) < 1e-14
    assert np.allclose(p[:, 0], np.linspace(0, ray['t'], len(p)), atol=1e-15, rtol=1e-14)
    assert np.allclose(p[:, 1], p[:, 0] / .2, atol=1e-14)
    assert np.allclose(p[:, 3], p[:, 2] / .02, atol=1e-14)
    assert np.allclose(xyz[-1], ray['endpoint_position'], atol=1e-14)
    assert error < 1e-12
    assert record['start']['gp_observation']['f0'] is None
    assert record['endpoint']['conditioned_gp_field_value'] is None
    # Piecewise-linear knots must reconstruct all independently exported samples.
    knots = record['geometry']['piecewise_linear_knots']
    knot_error = float(np.max(np.abs(np.interp(p[:, 0], [q['s'] for q in knots], [q['mean'] for q in knots]) - p[:, 2])))
    assert knot_error < 1e-12
    hits = record['geometry']['slab_intersections']
    assert len(hits) == record['provenance']['case']['expected_layers']
    for hit in hits:
        assert hit['midpoint_mean'] < 0
        for s in [hit['begin'], hit['end']]:
            value = np.max(np.abs(origin + s*w - centers[hit['slab']-1]) - half)
            assert abs(value) < 1e-12
    record['checks'].update(independent_python_mean_error=error, piecewise_reconstruction_error=knot_error,
                            f1_shape_keys_present=True, normalized_columns_passed=True)
    return p, components


def locate(records):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    for ax, method, name in zip(axes, METHODS, NAMES):
        source = BASE / method / 'seed20261024/render.png'
        ax.imshow(np.asarray(Image.open(source).convert('RGB')), interpolation='nearest', origin='upper')
        for i, record in enumerate(records):
            x, y = record['provenance']['case']['pixel']
            ax.plot(x, y, '+', color=COLORS[i], markersize=13, markeredgewidth=2)
            ax.annotate(f'R{i+1}', (x, y), xytext=(7, 6), textcoords='offset points', color=COLORS[i],
                        weight='bold', path_effects=[patheffects.withStroke(linewidth=3, foreground='white')])
        ax.set(title=name, xlabel='像素 x', ylabel='像素 y', xlim=(-.5, 63.5), ylim=(63.5, -.5))
    fig.suptitle('原渲染上的光线位置 · 64×64 / 256 spp / seed 20261024', fontsize=15)
    fig.text(.5, .015, 'R1 三重叠 (32,32)     R2 正常二重叠 (6,32)     R3 暗弧二重叠 (6,12) · 坐标从 0 开始', ha='center')
    fig.tight_layout(rect=(0, .12, 1, .94))
    figure_save(fig, 'ray_locations')


def profiles(records):
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    for i, (record, row) in enumerate(zip(records, axes)):
        p = np.array(record['sdf_curve']['samples'])
        ell = record['kernel']['configured']['length_scale_ell']
        hits = record['geometry']['slab_intersections']
        for j, ax in enumerate(row):
            ax.axhline(0, color='#343b40', lw=.9)
            ax.plot(p[:, 0], p[:, 2], color=COLORS[i], lw=2, label='生产代码均值 m(s)')
            for g in record['selection_groups']['boundaries']:
                ax.axvline(g, color='#8b7fa5', ls=':', lw=.8, alpha=.7)
            for h in hits:
                ax.axvspan(h['begin'], h['end'], color=COLORS[i], alpha=.12)
                ax.scatter([h['midpoint']], [h['midpoint_mean']], s=25, color=COLORS[i], zorder=5)
                if j:
                    ax.annotate(f"{['A1','B2','A3'][h['slab']-1]}\ns={h['midpoint']:.5f}",
                                (h['midpoint'], h['midpoint_mean']), xytext=(h['midpoint'], .065), textcoords='data',
                                fontsize=9, ha='center', va='bottom',
                                arrowprops=dict(arrowstyle='-', color=COLORS[i], lw=.6, alpha=.55))
            ax.set(xlabel='距介质入口的距离 s', ylabel='SDF 均值 m(s)',
                   xlim=(.24, .43) if j else (0, .66), ylim=(-.035, .10) if j else (-.035, .30))
            ax.grid(alpha=.15)
            ax.secondary_xaxis('top', functions=(lambda s:s/ell, lambda q:q*ell)).set_xlabel('s / ℓ')
            ax.set_title(('接近零值区域' if j else f"R{i+1} {record['provenance']['case']['label']} · 完整介质段"), pad=50)
        row[0].legend(loc='upper center', fontsize=9)
    fig.suptitle('Three slabs · 相同场景的三条 SDF 曲线 · ℓ=0.2，σ=0.02', fontsize=16, y=.995)
    fig.text(.5, .008, '阴影：均值 slab 内部；点：层内最低点；竖虚线：原 single_history 的选择组边界（约每 0.1）。\n'
             '曲线是确定性先验均值，不是随机 GP 实现；三条光线均延伸至介质出口。', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .97), h_pad=2.4)
    figure_save(fig, 'sdf_profiles')
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, record in enumerate(records):
        p = np.array(record['sdf_curve']['samples'])
        ax.plot(p[:, 0], p[:, 2], lw=2, color=COLORS[i], label=f"R{i+1} {record['provenance']['case']['label']}")
    ax.axvline(records[0]['selection_groups']['boundaries'][2], color='#786791', ls=':', label='第三个选择组边界 ≈ 0.30')
    ax.axhline(0, color='#454545', lw=.8)
    ax.set(xlim=(.24, .43), ylim=(-.017, .065), xlabel='距介质入口的距离 s', ylabel='SDF 均值 m(s)',
           title='三条光线叠加 · 暗弧光线的第一个谷值位于 s≈0.30 之后')
    ax.legend(fontsize=9);ax.grid(alpha=.17)
    fig.tight_layout()
    figure_save(fig, 'sdf_overlay')


def report():
    configure()
    data = load(OUT / 'raw_profiles.json')
    shape = template()
    records = data['records']
    evidence, hashes = source_evidence()
    summary = []
    for record in records:
        record['mean_field_scene_definition'].setdefault('transform', dict(position=[0, 0, 0], scale=1, rotation=[0, 0, 0]))
        record['provenance']['mean_transform_note'] = 'Identity transform was implicit in the source scene; made explicit to retain the F1 field structure.'
        p, components = check_record(record, shape)
        case = record['provenance']['case'];name = case['id'];x, y = case['pixel']
        values = {m: {str(seed): float(evidence[m][seed][y, x]) for seed in SEEDS} for m in METHODS}
        record['render_evidence'] = dict(pixel=case['pixel'], linear_radiance=values,
                                        spp_per_pixel_per_seed=256, note='Existing PFM pixel averages, not the contribution of the selected sample-0 ray.')
        record['provenance'].update(scene_sha256=sha(SCENE), template_sha256=sha(TEMPLATE), probe_sha256=sha(EXE))
        save(OUT / f'{name}-ray-input.json', record)
        np.savetxt(OUT / f'{name}.csv', np.column_stack((p, components)), delimiter=',', fmt='%.17g',
                   header=','.join(record['sdf_curve']['columns'] + ['slab_A1_mean', 'slab_B2_mean', 'slab_A3_mean']), comments='')
        summary.append(dict(id=name, label=case['label'], pixel=case['pixel'], length=record['ray']['t'],
                            layers=record['geometry']['intersected_layers'],
                            slab_intersections=record['geometry']['slab_intersections'],
                            linear_radiance=values, checks=record['checks']))
    locate(records)
    profiles(records)
    rows = ''
    links = ''
    for i, r in enumerate(summary):
        means = {m: np.mean(list(r['linear_radiance'][m].values())) for m in METHODS}
        rows += f"<tr><td>R{i+1} {r['label']}</td><td>{r['pixel']}</td><td>{r['layers']}</td>" + ''.join(f'<td>{means[m]:.6f}</td>' for m in METHODS) + '</tr>'
        hits = ', '.join(f"{['A1','B2','A3'][h['slab']-1]}: {h['midpoint']:.6f}" for h in r['slab_intersections'])
        links += f"<li>R{i+1} {r['label']}：<a href=\"{r['id']}-ray-input.json\">F1 格式 JSON</a> · <a href=\"{r['id']}.csv\">CSV</a>。谷值位置 s：{hits}。</li>"
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Three slabs · SDF 光线曲线</title><style>body{max-width:1280px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui;color:#22313c;background:#fbfcfe}h1{font-size:28px}img{max-width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;text-align:left;border-bottom:1px solid #dde3e8}a{color:#096baf}.note{padding:16px 20px;background:#edf3f8;border-radius:9px}code{background:#eef1f5;padding:2px 5px}</style>
<h1>Three slabs：三重叠、二重叠、暗弧二重叠的 SDF 曲线</h1>
<p>沿用原实验 ℓ=0.2、σ=0.02、A/B/A 三层布局、原相机和介质边界。三条光线均由生产相机采样器生成，seed 20261024、sample 0；曲线从介质入口一直延伸至出口，每条 6,657 个均匀样本。</p>
<div class="note">JSON 保留 <code>F1-ray-input.json</code> 的字段结构及四列曲线数据。这里展示确定性先验均值 m(s)，不是 GP 随机场实现。原 single_history 的入口仅条件化 <code>F(0)&gt;0</code>，没有观测到 f0、梯度、g0，所以这三项为 null；末端是介质出口，<code>conditioned_gp_field_value</code> 也为 null，不能当作 F(t)=0 的碰撞输入。</div>
<h2>光线位置与原渲染</h2><p>坐标为左上角起算的 0-based (x,y)。图像使用原 PNG，坐标标记仅用于定位。</p><img src="ray_locations.png" alt="三条光线在三种原渲染中的位置">
<h2>原始像素的线性值</h2><p>下表是两个已有 seed 各 256 spp 的像素均值；并非选中 sample 0 光线的单次贡献。</p>
<table><tr><th>案例</th><th>像素</th><th>穿过层数</th><th>Numeric</th><th>CDF5</th><th>Renewal+</th></tr>''' + rows + '''</table>
<p>“黑色”案例在 seed 20261024 的 CDF5 图中为 0.04296875、Renewal+ 为 0.27734375；它是明显变暗的弧形区域，原始数值不等于零。</p>
<h2>曲线下载</h2><ul>''' + links + '''</ul><p><a href="summary.json">摘要与核验</a> · <a href="manifest.json">来源和复现命令</a> · <a href="../comparison.html">原实验对比</a></p>
<h2>三条曲线叠加</h2><img src="sdf_overlay.png" alt="三条 SDF 曲线叠加">
<p>三重叠有 A1/B2/A3 三个负值谷；两条二重叠光线穿过 A1/A3，均有两个负值谷。正常二重叠的第一个谷值在 s≈0.30 前，暗弧光线的第一个谷值在其后。竖线标记原 single_history 的选择组边界，这与已有的暗弧位置诊断一致；仅凭均值曲线不能断定消光或历史选择的误差原因。<a href="../edge_diagnostic_20261116/comparison.html">已有的选择组 / 入口位置实验</a>。</p>
<h2>完整区间与零值附近放大</h2><img src="sdf_profiles.png" alt="三条完整 SDF 曲线和零值区域放大"><p><a href="sdf_profiles.svg">矢量 SVG</a> · <a href="sdf_overlay.svg">叠加图 SVG</a> · <a href="ray_locations.svg">像素定位 SVG</a></p>
<p>数值来自生产 SingleHistoryGpisMedium 的 mean.evaluate；所有 19,971 个样本与独立 min(max(face-distance)) 公式核对，层交点及分段线性重建也已验证。相机方向保持同 seed/pixel/sample，再归一化并以双精度解析裁切介质盒；这份曲线导出不宣称复现渲染器边界交点的 float32 舍入或整条吸收路径。</p></html>'''
    (OUT / 'comparison.html').write_text(page, encoding='utf-8')
    targets = re.findall(r'(?:href|src)="([^"]+)"', page)
    assert all((OUT / target).exists() for target in targets)
    summary_data = dict(complete=True, rays=summary, total_samples=sum(len(r['sdf_curve']['samples']) for r in records),
                        original_images_verified=True, html_local_links_verified=len(targets),
                        template_unchanged_sha256=sha(TEMPLATE))
    save(OUT / 'summary.json', summary_data)
    sources = [SCENE, TEMPLATE, EXE, ROOT/'tools/surface_compare/slabs_ray_profiles.cpp', ROOT/'tools/surface_compare/slabs_ray_profiles.py']
    hashes.update({str(p.relative_to(ROOT)):sha(p) for p in sources})
    save(OUT / 'manifest.json', dict(spec=data['spec'], source_sha256=hashes,
         commands=['cmake --build build --config Release --target slabs_ray_profiles',
                   'python -B tools/surface_compare/slabs_ray_profiles.py --run'],
         scope='SDF-only diagnostic; original scene and render files retained.',
         format_notes='F1 key structure retained. Null values identify absent GP observations and absent collision endpoint.'))
    print(json.dumps(summary_data, ensure_ascii=True), flush=True)
    print(OUT / 'comparison.html', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.run:
        save(OUT / 'spec.json', dict(scene=str(SCENE), seed=20261024, sample=0,
                                    interval_count=template()['sdf_curve']['interval_count'], rays=CASES))
        with (OUT / 'probe.log').open('w', encoding='utf-8') as log:
            subprocess.run([str(EXE), str(OUT/'spec.json'), str(OUT/'raw_profiles.json')], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Create the two linked records before HTML local-link validation.
    for name in ['summary.json', 'manifest.json']:
        if not (OUT/name).exists():
            save(OUT/name, {'complete': False})
    report()


if __name__ == '__main__':
    main()
