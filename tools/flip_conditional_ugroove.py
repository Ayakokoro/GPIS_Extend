"""Compute LDR-FLIP for displayed U-groove PNGs against matched GPIS renewal+.

Run again after rebuilding the original reports to refresh their FLIP sections.
Only existing, matched images are used; no resizing or rerendering is performed.
"""
from __future__ import annotations

import argparse
import csv
import html
from importlib.metadata import version
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

import flip_evaluator as flip
import numpy as np

from flip_shaderball_fixed_ratio import Images, STYLE, digest, write_png

ROOT = Path(__file__).resolve().parents[1] / 'outputs/conditional_ugroove'
REFERENCE = 'gpis_renewal_plus'
MODES = ('classic_global', 'global_conditional', REFERENCE, 'classic_local', 'single_realization')
LABELS = {m: m for m in MODES} | {REFERENCE: 'GPIS renewal+ (3D sparse convolution)'}
BEGIN, END = '<!-- ugroove-flip-begin -->', '<!-- ugroove-flip-end -->'


def strip_report(page):
    return re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', page, flags=re.S)


def collect(folder, kind):
    page = strip_report((folder / 'comparison.html').read_text(encoding='utf-8'))
    parser = Images()
    parser.feed(page)
    groups = {}
    for image in parser.images:
        src = unquote(urlsplit(image['src']).path)
        parts = Path(src).parts
        if kind == 'overview':
            if len(parts) != 6 or parts[1] != 'averages' or Path(src).stem not in MODES:
                continue
            stage, _, group, view, component, _ = parts
            key, mode = (stage, group, view, component), Path(src).stem
        elif kind == 'formal':
            if len(parts) != 6 or parts[-1] != 'exposure025.png' or parts[3] not in MODES:
                continue
            group, view, component, mode, _, _ = parts
            key = ('res256_spp256', group, view, component)
        else:
            if Path(src).name != 'exposure025.png':
                continue
            mode = REFERENCE if REFERENCE in parts else 'single_realization'
            key = ('single_realization', 'A_strong', 'view45', 'total')
        if not (folder / src).is_file():
            raise FileNotFoundError(folder / src)
        previous = groups.setdefault(key, {}).get(mode)
        if previous and digest(folder / previous) != digest(folder / src):
            raise ValueError(f'Conflicting displayed images: {key}, {mode}')
        groups[key][mode] = src
    unmatched = [{'stage': k[0], 'group': k[1], 'view': k[2], 'component': k[3],
                  'modes': list(v), 'reason': 'No matching GPIS renewal+ image at this configuration'}
                 for k, v in groups.items() if REFERENCE not in v]
    matched = {k: v for k, v in groups.items() if REFERENCE in v}
    if not matched:
        raise ValueError(f'No matched GPIS renewal+ images in {folder}')
    if kind in ('overview', 'formal'):
        expected = {'A_strong', 'B_medium', 'C_short', 'D_small_sigma'}
        if {k[1] for k in matched} != expected or any(set(v) != set(MODES[:4]) for v in matched.values()):
            raise ValueError('Expected four completed groups with four modes each')
    return page, matched, unmatched


def evaluate(folder, groups, exposure, ppd, parameters):
    rows = []
    for (stage, group, view, component), modes in sorted(groups.items()):
        reference_path = folder / modes[REFERENCE]
        reference = flip.load(str(reference_path))
        for mode in MODES:
            if mode not in modes:
                continue
            test_path = folder / modes[mode]
            candidate = flip.load(str(test_path))
            if reference.shape != candidate.shape or reference.shape != (256, 256, 3):
                raise ValueError(f'Unmatched image resolution: {test_path}')
            if not all(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
                       for a in (reference, candidate)):
                raise ValueError('Invalid sRGB input')
            errors, mean, _ = flip.evaluate(reference, candidate, 'LDR',
                applyMagma=False, parameters={'ppd': ppd})
            colors, color_mean, _ = flip.evaluate(reference, candidate, 'LDR',
                applyMagma=True, parameters={'ppd': ppd})
            errors = errors[..., 0]
            # Pool in float64: FLIP's native float accumulator loses precision
            # for maps with many large errors (notably the fixed-field image).
            map_mean = float(errors.mean(dtype=np.float64))
            if (not np.isfinite(errors).all() or errors.min() < 0 or errors.max() > 1
                    or not np.isclose(mean, map_mean, rtol=2e-5, atol=1e-6)
                    or not np.isclose(mean, color_mean, atol=1e-7)):
                raise ValueError('Invalid FLIP output')
            if mode == REFERENCE and np.any(errors):
                raise ValueError('Reference self comparison must be exactly zero')
            relative = Path('flip') / group / mode
            write_png(folder / relative.with_suffix('.png'), colors)
            np.save(folder / relative.with_suffix('.npy'), errors)
            row = dict(stage=stage, group=group, sigma=parameters[group]['sigma'],
                ell=parameters[group]['ell'], view=view, component=component, mode=mode,
                exposure=exposure, ppd=ppd, width=256, height=256,
                reference=modes[REFERENCE], test=modes[mode],
                reference_sha256=digest(reference_path), test_sha256=digest(test_path),
                mean=map_mean, evaluator_mean=float(mean), minimum=float(errors.min()), maximum=float(errors.max()),
                p95=float(np.quantile(errors, .95)), image=relative.with_suffix('.png').as_posix(),
                values=relative.with_suffix('.npy').as_posix())
            rows.append(row)
            print(f'{folder.name} {group} {mode}: mean FLIP={map_mean:.6f}', flush=True)
    return rows


def fragment(rows, exposure, ppd, kind, unmatched):
    parts = [STYLE, '<section class="flip-section" id="flip"><h2>FLIP vs GPIS renewal+ (3D sparse convolution)</h2>',
        f'<p>NVIDIA FLIP {html.escape(version("flip-evaluator"))} · LDR-FLIP · {ppd:g} pixels/degree。'
        f'直接比较当前展示的 256×256 PNG（曝光 {exposure:g} + filmic）；'
        '参考图为相同参数组、45° 视角与 total 分量的 GPIS renewal+。'
        '统一使用原生 magma 色标 [0,1]，暗色误差小，亮色误差大，未逐图归一化。</p>',
        '<p>mean FLIP 为全图均值（包含背景）；单 seed 的差异包含渲染采样噪声。参考图自比较为 0。</p>',
        '<p><a href="flip_comparison.html">独立 FLIP 页面</a> · <a href="flip/metrics.csv">数值 CSV</a> · '
        '<a href="flip/summary.json">输入路径、哈希及参数</a></p>']
    if kind == 'overview':
        parts.append('<p><a href="res256_spp256/flip_comparison.html">正式展示：曝光 0.25</a> · '
                     '<a href="single_realization/flip_comparison.html">固定随机场单独对比</a></p>')
    if unmatched:
        parts.append(f'<p>另有 {len(unmatched)} 组低分辨率或其他配置的试验没有匹配的 GPIS renewal+ 参考图，'
                     '未跨视角、分量或分辨率混配。具体清单见 JSON。</p>')
    if kind == 'single':
        parts.append('<p>此项仅为固定随机场与 renewal+ 展示的外观差异。二者的随机场共享和相关性处理不同，'
                     'FLIP 数值不能单独解释为算法偏差或采样收敛误差。</p>')
    parts.append('<table><tr><th>参数组</th><th>模式</th><th>mean FLIP</th><th>P95</th></tr>')
    for row in rows:
        parts.append(f'<tr><td>{row["group"]}</td><td>{LABELS[row["mode"]]}</td>'
                     f'<td>{row["mean"]:.6f}</td><td>{row["p95"]:.6f}</td></tr>')
    parts.append('</table>')
    for group in sorted({r['group'] for r in rows}):
        selected = [r for r in rows if r['group'] == group]
        first = selected[0]
        parts.append(f'<h3>{group} · σ={first["sigma"]:g} · ℓ={first["ell"]:g}</h3>'
                     f'<div class="flip-grid" style="--flip-columns:{len(selected)}">')
        for row in selected:
            label = LABELS[row['mode']]
            parts.append(f'<figure><figcaption>{label}<br>mean FLIP = {row["mean"]:.6f}</figcaption>'
                f'<a href="{row["test"]}"><img src="{row["test"]}" alt="{label}"></a>'
                f'<a href="{row["image"]}"><img src="{row["image"]}" alt="FLIP: {label}"></a>'
                f'<a href="{row["values"]}">原始误差 NPY</a></figure>')
        parts.append('</div>')
    return ''.join(parts) + '</section>'


def run(folder, kind, exposure, ppd, parameters):
    page, groups, unmatched = collect(folder, kind)
    rows = evaluate(folder, groups, exposure, ppd, parameters)
    output = folder / 'flip'
    summary = dict(metric='LDR-FLIP', version=version('flip-evaluator'), ppd=ppd,
        reference=LABELS[REFERENCE], exposure=exposure, color_scale=[0, 1], colormap='FLIP native magma',
        input='Existing displayed sRGB PNGs; no resizing, extra tone mapping, or denoising',
        pooling='All pixels including background; float64 mean of scalar map; evaluator_mean retains native float pooling', unmatched=unmatched,
        interpretation=('Appearance difference between a fixed field and renewal+; not an estimate of algorithm bias'
                        if kind == 'single' else 'Differences include single-seed rendering noise'), rows=rows)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (output / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    body = fragment(rows, exposure, ppd, kind, unmatched)
    standalone = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>U 槽 FLIP vs GPIS renewal+</title>'
        '<style>body{max-width:1500px;margin:28px auto;padding:0 20px;font:15px/1.6 system-ui;background:#f4f6f9;color:#233145}a{color:#225ca0}</style>'
        f'<h1>U 槽 · {folder.name} · FLIP</h1><p><a href="comparison.html">返回原对照页</a></p>' + body + '</html>')
    (folder / 'flip_comparison.html').write_text(standalone, encoding='utf-8')
    backup = output / 'comparison_before_flip.html'
    if not backup.exists():
        backup.write_text(page, encoding='utf-8')
    addition = BEGIN + body + END
    position = page.lower().rfind('</html>')
    updated = page[:position] + addition + page[position:] if position >= 0 else page + addition
    (folder / 'comparison.html').write_text(updated, encoding='utf-8')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--ppd', type=float, default=67.0)
    args = parser.parse_args()
    if not np.isfinite(args.ppd) or args.ppd <= 0:
        parser.error('--ppd must be positive and finite')
    summary = json.loads((args.root / 'res256_spp256/summary.json').read_text(encoding='utf-8'))
    if summary['state'] != 'complete' or summary['completed'] != 16 or summary['display_exposure'] != .25:
        raise ValueError('Expected completed formal U-groove experiment')
    parameters = {c['group']: c for c in summary['comparisons']}
    count = run(args.root / 'res256_spp256', 'formal', .25, args.ppd, parameters)
    single = args.root / 'single_realization'
    if (single / 'comparison.html').exists():
        status = json.loads((single / 'status.json').read_text(encoding='utf-8'))
        if status['state'] != 'complete':
            raise ValueError('Single realization is still rendering')
        count += run(single, 'single', .25, args.ppd, parameters)
    count += run(args.root, 'overview', 1.0, args.ppd, parameters)
    print(f'Generated {count} maps, including reference self checks.')


if __name__ == '__main__':
    main()
