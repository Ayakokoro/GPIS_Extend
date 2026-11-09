"""Evaluate the currently displayed shaderballs against matched GPIS global PNGs.

Uses the locally installed NVIDIA flip_evaluator, without rerendering. Run again
after rebuilding comparison.html to refresh its FLIP section and standalone page.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
from html.parser import HTMLParser
from importlib.metadata import version
import json
from pathlib import Path
import re

import flip_evaluator as flip
import numpy as np

from compare_shaderball_fixed_ratio import write_png

ROOT = Path(__file__).resolve().parents[1]
MODES = ('classic_global', 'classic_local', 'global_conditional', 'gpis_global', 'neural')
BEGIN, END = '<!-- shaderball-flip-begin -->', '<!-- shaderball-flip-end -->'


class Images(HTMLParser):
    def __init__(self):
        super().__init__()
        self.images = []

    def handle_starttag(self, tag, attrs):
        if tag == 'img':
            self.images.append(dict(attrs))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strip_report(page):
    return re.sub(re.escape(BEGIN) + r'.*?' + re.escape(END), '', page, flags=re.S)


def collect(folder):
    page = strip_report((folder / 'comparison.html').read_text(encoding='utf-8'))
    parser = Images()
    parser.feed(page)
    groups = {}
    for image in parser.images:
        raw = image.get('data-raw', image['src'])
        mode = Path(raw).stem
        if mode not in MODES:
            continue  # Existing signed differences and repeat-noise maps are diagnostics.
        match = re.search(r'ell(\d+(?:p\d+)?)', raw)
        if not match:
            raise ValueError(f'Cannot determine length scale: {raw}')
        ell = float(match[1].replace('p', '.'))
        paths = {'raw': raw, 'block2': image['data-block']}
        for path in paths.values():
            if not (folder / path).is_file():
                raise FileNotFoundError(folder / path)
        previous = groups.setdefault(ell, {}).get(mode)
        if previous and any(digest(folder / previous[v]) != digest(folder / paths[v]) for v in paths):
            raise ValueError(f'Different displayed images for {ell}, {mode}')
        groups[ell][mode] = paths
    for ell, modes in groups.items():
        if 'gpis_global' not in modes:
            raise ValueError(f'Missing GPIS global reference at {ell}')
    return page, groups


def evaluate(folder, groups, ppd):
    rows = []
    for ell, modes in sorted(groups.items()):
        for mode in MODES:
            if mode not in modes:
                continue
            for variant in ('raw', 'block2'):
                ref = folder / modes['gpis_global'][variant]
                test = folder / modes[mode][variant]
                reference, candidate = flip.load(str(ref)), flip.load(str(test))
                if reference.shape != candidate.shape or reference.shape[2] != 3:
                    raise ValueError(f'Image shape mismatch: {ref}, {test}')
                if not all(np.isfinite(a).all() and a.min() >= 0 and a.max() <= 1
                           for a in (reference, candidate)):
                    raise ValueError('Invalid LDR image')
                errors, mean, _ = flip.evaluate(reference, candidate, 'LDR',
                    applyMagma=False, parameters={'ppd': ppd})
                magma, magma_mean, _ = flip.evaluate(reference, candidate, 'LDR',
                    applyMagma=True, parameters={'ppd': ppd})
                errors = errors[..., 0]
                if (not np.isfinite(errors).all() or errors.min() < 0 or errors.max() > 1
                        or not np.isclose(mean, errors.mean(), atol=1e-6)
                        or not np.isclose(mean, magma_mean, atol=1e-7)):
                    raise ValueError('Invalid FLIP output')
                if mode == 'gpis_global' and np.any(errors):
                    raise ValueError('GPIS self comparison must be zero')
                suffix = '' if variant == 'raw' else '_block2'
                relative = Path('flip') / ('ell' + f'{ell:g}'.replace('.', 'p')) / (mode + suffix)
                write_png(folder / relative.with_suffix('.png'), magma)
                np.save(folder / relative.with_suffix('.npy'), errors)
                row = dict(ell=ell, mode=mode, variant=variant,
                    reference=modes['gpis_global'][variant], test=modes[mode][variant],
                    reference_sha256=digest(ref), test_sha256=digest(test),
                    width=errors.shape[1], height=errors.shape[0], ppd=ppd,
                    mean=float(mean), minimum=float(errors.min()), maximum=float(errors.max()),
                    p95=float(np.quantile(errors, .95)),
                    image=relative.with_suffix('.png').as_posix(),
                    values=relative.with_suffix('.npy').as_posix())
                rows.append(row)
                print(f'{folder.name} ell={ell:g} {mode} {variant}: mean={mean:.6f}', flush=True)
    return rows


STYLE = '''<style>
body{box-sizing:border-box;max-width:1280px;margin:20px auto;padding:0 clamp(10px,2vw,24px)}
table{width:100%;table-layout:fixed}th,td{overflow-wrap:anywhere;padding:clamp(4px,1vw,12px)}
img{max-width:100%;height:auto}
.images>figure{min-width:0;box-sizing:border-box}
.images img{max-width:256px;height:auto}
.flip-section{box-sizing:border-box;min-width:0;background:white;padding:clamp(10px,2vw,18px);margin:20px 0;border-radius:8px}
.flip-grid{display:grid;grid-template-columns:repeat(var(--flip-columns),minmax(0,1fr));gap:14px;
max-width:calc(var(--flip-columns)*220px + (var(--flip-columns) - 1)*14px)}
.flip-grid figure{margin:0;min-width:0;overflow-wrap:anywhere}.flip-grid img{width:100%;max-width:220px;height:auto;display:block}
.flip-grid figcaption{font:13px/1.6 system-ui;min-height:44px}
.flip-section table{border-collapse:collapse;width:100%}.flip-section td,.flip-section th{padding:6px;border-bottom:1px solid #ddd;text-align:right}
@media(max-width:850px){.flip-grid{grid-template-columns:repeat(2,minmax(0,1fr));max-width:454px}}
@media(max-width:480px){.flip-grid{gap:10px}.flip-grid figcaption{font-size:12px}}
</style>'''


SCRIPT = '''<script>
(() => {
const box=document.getElementById('block');
function updateFlip(){const v=box&&box.checked?'block':'raw';
document.querySelectorAll('.flip-image').forEach(img=>{img.src=img.dataset[v];img.parentElement.href=img.src});
document.querySelectorAll('.flip-mean').forEach(span=>span.textContent=span.dataset[v==='block'?'meanBlock':'meanRaw']);}
if(box)box.addEventListener('change',updateFlip);updateFlip();
})();
</script>'''


def fragment(rows, ppd):
    content = [STYLE, '<section class="flip-section" id="flip"><h2>FLIP vs GPIS global</h2>',
        f'<p>LDR-FLIP · NVIDIA FLIP {html.escape(version("flip-evaluator"))} · {ppd:g} pixels/degree。'
        '每张展示 PNG 与相同 ℓ、相同分辨率的 GPIS global 比较；输入为已有 filmic 映射后的 sRGB PNG。'
        '统一使用 FLIP 原生 magma 色标 [0, 1]，暗色表示误差小，亮色表示误差大；未逐图归一化。</p>',
        '<p>2×2 模式使用已有线性均值后的 PNG 重新计算 FLIP，PPD 保持相同。数值为全图平均（包括背景）；'
        '差异包含渲染采样噪声。GPIS global 自比较应为 0。</p>',
        '<p><a href="flip_comparison.html">独立 FLIP 页面</a> · <a href="flip/metrics.csv">数值 CSV</a> · '
        '<a href="flip/summary.json">输入路径、哈希与参数</a></p>']
    for ell in sorted({r['ell'] for r in rows}):
        raw = [r for r in rows if r['ell'] == ell and r['variant'] == 'raw']
        content.append(f'<h3>ℓ = σ = {ell:g}</h3><div class="flip-grid" style="--flip-columns:{len(raw)}">')
        for row in raw:
            block = next(r for r in rows if r['ell'] == ell and r['mode'] == row['mode'] and r['variant'] == 'block2')
            label = 'GPIS global (reference)' if row['mode'] == 'gpis_global' else row['mode']
            content.append(f'<figure><figcaption>{label}<br>mean FLIP = '
                f'<span class="flip-mean" data-mean-raw="{row["mean"]:.6f}" data-mean-block="{block["mean"]:.6f}">{row["mean"]:.6f}</span>'
                f'</figcaption><a href="{row["test"]}"><img class="flip-image" src="{row["test"]}" '
                f'data-raw="{row["test"]}" data-block="{block["test"]}" alt="{label}"></a>'
                f'<a href="{row["image"]}"><img class="flip-image" src="{row["image"]}" '
                f'data-raw="{row["image"]}" data-block="{block["image"]}" alt="FLIP: {label}"></a>'
                f'<a href="{row["values"]}">原始误差 NPY</a></figure>')
        content.append('</div>')
    content.append('</section>' + SCRIPT)
    return ''.join(content)


def run(folder, ppd):
    page, groups = collect(folder)
    if not groups:
        raise ValueError(f'No displayed shaderballs: {folder}')
    rows = evaluate(folder, groups, ppd)
    out = folder / 'flip'
    summary = dict(metric='LDR-FLIP', version=version('flip-evaluator'), ppd=ppd,
        reference='gpis_global at matched length scale and display resolution',
        input='Existing filmic-mapped sRGB PNGs; no further tone mapping or denoising',
        color_scale=[0, 1], colormap='FLIP native magma',
        pooling='All pixels, including background',
        block2='Reevaluated on existing 2x2 linear-averaged display PNGs at the same PPD',
        rows=rows)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    with (out / 'metrics.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    body = fragment(rows, ppd)
    standalone = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shaderball FLIP vs GPIS global</title>'
        '<style>body{max-width:1500px;margin:28px auto;padding:0 20px;font:15px/1.6 system-ui;background:#f4f6f9;color:#233145}a{color:#225ca0}</style>'
        f'<h1>{folder.name} · FLIP vs GPIS global</h1><p><a href="comparison.html">返回原对照页</a></p>'
        '<label><input id="block" type="checkbox">2×2 线性均值显示及对应 FLIP</label>' + body + '</html>')
    (folder / 'flip_comparison.html').write_text(standalone, encoding='utf-8')
    backup = out / 'comparison_before_flip.html'
    if not backup.exists():
        backup.write_text(page, encoding='utf-8')
    addition = BEGIN + body + END
    position = page.lower().rfind('</html>')
    updated = page[:position] + addition + page[position:] if position >= 0 else page + addition
    (folder / 'comparison.html').write_text(updated, encoding='utf-8')
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT / 'outputs/shaderball_fixed_ratio')
    parser.add_argument('--variants', nargs='+', default=['main', 'res256_spp256'])
    parser.add_argument('--ppd', type=float, default=67.0)
    args = parser.parse_args()
    if not np.isfinite(args.ppd) or args.ppd <= 0:
        parser.error('--ppd must be positive and finite')
    count = 0
    for variant in args.variants:
        count += run(args.root / variant, args.ppd)
    links = ''.join(f'<li><a href="{html.escape(v)}/flip_comparison.html">{html.escape(v)}</a></li>' for v in args.variants)
    (args.root / 'flip_comparison.html').write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball FLIP</title>'
        '<h1>Shaderball FLIP vs GPIS global</h1><ul>' + links + '</ul></html>', encoding='utf-8')
    print(f'Generated {count} maps (raw + block2, including reference self checks).')


if __name__ == '__main__':
    main()
