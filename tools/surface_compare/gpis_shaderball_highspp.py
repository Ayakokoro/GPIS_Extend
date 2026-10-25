"""Render GPIS global at higher spp and compare it with the original 128 spp image.

python -B tools/surface_compare/gpis_shaderball_highspp.py run
<plot-python> -B tools/surface_compare/gpis_shaderball_highspp.py figures
"""
from __future__ import annotations
import argparse
import copy
import os
import subprocess
from pathlib import Path
import numpy as np
from gpis_shaderball import ROOT, SOURCE, ROI, folder, output, render
from gpis_shaderball_sweep import configuration, PLOT_PYTHON
from run import save
from shaderball_length import digest

MODE = 'sparse_3d_global'


def figures(args):
    from figures import plt, Image, Rectangle
    from gpis_shaderball_figures import read, regions

    reference = copy.copy(args)
    reference.stage = 'final'
    data = [read(reference, MODE), read(args, MODE)]
    old, new = data
    assert old['source']['result']['spp'] == 128
    assert new['source']['result']['spp'] == args.spp > 128
    assert old['source']['result']['provenance'] == new['source']['result']['provenance']
    scenes = [copy.deepcopy(d['scene']) for d in data]
    for s in scenes:
        s['renderer'].pop('output_directory')
        s['renderer'].pop('spp')
    assert scenes[0] == scenes[1], 'Only spp and output paths may differ'
    assert old['rgb'].shape == new['rgb'].shape == (args.size, args.size, 3)

    report = output(args) / args.stage
    x0, y0, x1, y1 = [int(v * args.size) for v in ROI]
    fig, axes = plt.subplots(2, 2, figsize=(9, 9))
    statistics = []
    for col, d in enumerate(data):
        source = d['source']
        count = source['result']['spp']
        with Image.open(Path(source['folder']) / 'render.png') as im:
            rgb = np.asarray(im.convert('RGB'))
        axes[0, col].imshow(rgb, interpolation='nearest')
        axes[0, col].add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
            fill=False, lw=1, edgecolor='#ffcf52'))
        axes[0, col].set_title(f'GPIS global · {count} spp')
        axes[1, col].imshow(rgb[y0:y1, x0:x1], interpolation='nearest')
        axes[1, col].set_title('环形凹槽局部')
        for ax in axes[:, col]:
            ax.axis('off')
        for name, mask in regions(args.size).items():
            noise = .5 * (d['halves'][0] - d['halves'][1])
            statistics.append(dict(spp=count, region=name,
                mean_luminance=float(d['y'][mask].mean()),
                paired_half_noise_rms=float(np.sqrt(np.mean(noise[mask]**2)))))
    fig.suptitle(f'GPIS shaderball · 采样数对照\nℓ={args.ell:g}，σ={args.sigma:g} · {args.size}×{args.size}', fontsize=16)
    fig.text(.06, .025, '相同场景、随机种子、步长与 filmic 显示；无降噪。\n每条路径内保持同一随机场，不同路径使用不同实现。', fontsize=10)
    fig.subplots_adjust(left=.045, right=.98, top=.88, bottom=.09, wspace=.045, hspace=.10)
    fig.savefig(report / 'comparison.png', dpi=160)
    plt.close(fig)

    manifest = dict(complete=True, source=str(SOURCE), source_sha256=digest(SOURCE),
        ell=args.ell, sigma=args.sigma, resolution=[args.size, args.size],
        step=args.step, seed=args.seed, sources=[d['source'] for d in data],
        statistics=statistics, detail_roi=ROI, figure_sha256=digest(report / 'comparison.png'),
        notes=['Only spp and output paths differ from the original 128 spp GPIS render.',
            'Paired-half noise RMS is descriptive; both renders share a seed and are not independent replicates.',
            'Higher spp reduces sampling noise, not finite-step or sparse-convolution approximation error.'])
    save(report / 'manifest.json', manifest)
    def link(path):
        return Path(os.path.relpath(path, report)).as_posix()
    links = []
    for d in data:
        s = d['source']; p = Path(s['folder']); count = s['result']['spp']
        links.append(f'<li>{count} spp：<a href="{link(p / "render.png")}">PNG 原图</a> · '
                     f'<a href="{link(p / "render.pfm")}">线性 HDR</a> · '
                     f'<a href="{link(p / "scene.json")}">场景</a></li>')
    html = f'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPIS shaderball · {args.spp} spp</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;max-width:1100px;margin:32px auto;padding:0 20px;background:#f6f7f8;color:#222}}img{{width:100%;height:auto}}a{{color:#1763a5}}</style>
<h1>GPIS shaderball · 128 / {args.spp} spp</h1>
<p>ℓ={args.ell:g}，σ={args.sigma:g}；{args.size}×{args.size}；GPIS global 平均外观。
沿用原相机、灯光和绿色镜面材质，步长 {args.step:g}，统一 filmic 显示，无降噪。</p>
<p>提高 spp 用于降低采样噪声；有限步长和稀疏卷积近似保持原设置。</p>
<img src="comparison.png" alt="128 spp 与更高 spp 的全图和环形凹槽对照">
<ul>{''.join(links)}</ul>
<p><a href="../../ell0p2_neural_comparison/comparison.html">返回五模式对比</a> · <a href="manifest.json">结果记录</a></p>
</html>'''
    (report / 'comparison.html').write_text(html, encoding='utf-8')
    overview = output(args).parent / 'ell0p2_neural_comparison/comparison.html'
    if overview.exists():
        target = Path(os.path.relpath(report / 'comparison.html', overview.parent)).as_posix()
        page = overview.read_text(encoding='utf-8')
        if f'href="{target}"' not in page:
            end = page.index('</h1>') + len('</h1>')
            navigation = f'\n<p><a href="{target}">GPIS {args.spp} spp 原图与凹槽对照（ℓ={args.ell:g}，σ={args.sigma:g}）</a></p>'
            overview.write_text(page[:end] + navigation + page[end:], encoding='utf-8')
    print(report / 'comparison.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('run', 'figures'))
    parser.add_argument('--spp', type=int, default=512)
    parser.add_argument('--ell', type=float, default=.2)
    parser.add_argument('--sigma', type=float, default=.02)
    options = parser.parse_args()
    if options.spp <= 128 or options.spp % 2 or not all(np.isfinite(v) and v > 0 for v in (options.ell, options.sigma)):
        parser.error('Even spp > 128 and finite positive ell/sigma required')
    args = configuration(options.ell, options.sigma, stage=f'highspp_{options.spp}',
        spp=options.spp, modes=[MODE])
    if options.command == 'run':
        render(args)
        subprocess.run([str(PLOT_PYTHON), '-B', str(Path(__file__).resolve()), 'figures',
            '--spp', str(args.spp), '--ell', str(args.ell), '--sigma', str(args.sigma)], cwd=ROOT, check=True)
    else:
        figures(args)
