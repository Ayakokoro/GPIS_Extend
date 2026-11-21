"""Summarize fixed-ray first downcross samples from the two renderers and SE reference."""
import argparse
import csv
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
deps = ROOT / 'build/fp-gradient-validation-deps'
if deps.exists():
    sys.path.insert(0, str(deps))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/ray_curve_matplotlib'))
import matplotlib
matplotlib.use('Agg')
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np


CONFIGS = {'base': (.02, .2), 'large': (.08, .8)}
METHODS = {'reference': ('SE 参考', '#222c33', '-', 2.1),
           'lattice': ('固定格点', '#007d80', '--', 1.65),
           'bounded': ('稀疏卷积', '#d16b23', ':', 1.9)}
RAY_LABELS = {'center': '相机中心', 'tangent': '轮廓切线',
              'outside': '轮廓外侧 0.04', 'outgoing': '条件化出射'}


def load_distances(path, rays, n):
    samples = {r: [] for r in rays}
    with path.open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            name = row['ray']
            distance = float(row['distance']) if row['distance'] else np.inf
            if int(row['realization']) != len(samples[name]):
                raise ValueError(f'Unordered or missing realization in {path}, ray {name}')
            if row['distance'] and not (0 < distance <= rays[name]['length'] + 1e-6):
                raise ValueError(f'Invalid hit {distance} in {path}, ray {name}')
            samples[name].append(distance)
    if any(len(v) != n for v in samples.values()):
        raise ValueError(f'Incorrect sample count in {path}')
    return {name: np.sort(values) for name, values in samples.items()}


def cdf_gap(sample, reference):
    # For finite t, misses contribute to the denominator and never increment the CDF.
    knots = np.union1d(sample[np.isfinite(sample)], reference[np.isfinite(reference)])
    if not len(knots):
        return 0., None
    gap = (np.searchsorted(sample, knots, side='right') / len(sample)
           - np.searchsorted(reference, knots, side='right') / len(reference))
    idx = np.abs(gap).argmax()
    return float(abs(gap[idx])), float(knots[idx])


def wilson(k, n):
    z = 1.959963984540054
    p = k / n
    c = (p + z*z/(2*n)) / (1+z*z/n)
    w = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return max(0., c-w), min(1., c+w)


def summarize(out, spec):
    n = spec['samples']
    rays = {r['name']: r for r in spec['rays']}
    series = {}
    rows = []
    for config in CONFIGS:
        draws = {m: load_distances(out/config/(m+'.csv'), rays, n) for m in METHODS}
        series[config] = draws
        for name in rays:
            for method, by_ray in draws.items():
                samples = by_ray[name]
                hits = samples[np.isfinite(samples)]
                q = np.quantile(hits, [.1, .5, .9]) if len(hits) else [None] * 3
                lo, hi = wilson(len(hits), n)
                gap, where = cdf_gap(samples, draws['reference'][name])
                rows.append(dict(config=config, ray=name, method=method, samples=n,
                                 hits=len(hits), misses=n-len(hits), hit_probability=len(hits)/n,
                                 hit_wilson95_low=lo, hit_wilson95_high=hi,
                                 hit_p10=q[0], hit_median=q[1], hit_p90=q[2],
                                 max_cdf_gap_to_reference=gap, t_at_max_gap=where))
    with (out/'statistics.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    return series


def plot(out, spec, series):
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.unicode_minus': False, 'font.size': 10})
    rays = spec['rays']
    n = spec['samples']
    fig, axes = plt.subplots(len(rays), 2, figsize=(11, 11.5), sharey=True)
    fig.subplots_adjust(left=.10, right=.98, bottom=.078, top=.884, hspace=.53, wspace=.17)
    for col, (config, (sigma, ell)) in enumerate(CONFIGS.items()):
        for row, ray in enumerate(rays):
            ax = axes[row, col]
            ax.set_title(RAY_LABELS[ray['name']], loc='left', fontsize=11)
            if row == 0:
                fig.text(ax.get_position().x0, .912, rf'$\sigma={sigma},\quad\ell={ell}$',
                         fontsize=14)
            curves = {m: series[config][m][ray['name']] for m in METHODS}
            finite = np.concatenate([v[np.isfinite(v)] for v in curves.values()])
            xmax, xmin = (finite.max(), finite.min()) if len(finite) else (ray['length'], 0.)
            if ray['name'] == 'outgoing' and len(finite):
                xmin = np.quantile(finite, .001)
            pad = max(.09 * (xmax-xmin), .002)
            left, right = max(0., xmin-pad), min(ray['length'], xmax+pad)
            for method, (label, color, ls, width) in METHODS.items():
                hits = curves[method]
                hits = hits[np.isfinite(hits)]
                ax.step(np.r_[0., hits, ray['length']],
                        np.r_[0, np.arange(1, len(hits)+1)/n, len(hits)/n],
                        where='post', lw=width, color=color, ls=ls, label=label)
            ax.set(xlim=(left, right), ylim=(-.025, 1.025), yticks=[0,.25,.5,.75,1])
            ax.grid(alpha=.16)
            if col == 0:
                ax.set_ylabel('累计命中概率')
            if row == len(rays)-1:
                ax.set_xlabel('距射线起点的距离（场景单位）')
    fig.suptitle('Shaderball 固定光线的首碰距离分布', fontsize=19, x=.10, y=.987, ha='left')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.985,.986), ncol=3, frameon=False)
    fig.text(.10, .947, f'每条曲线 {n:,} 个 realization，右端平台为整段命中率。', fontsize=10)
    fig.text(.10, .018, '各面板分别放大横轴。出射光线放大主峰，所有样本均计入曲线，包括零点附近的命中。', fontsize=9)
    fig.savefig(out/'first_hit_cdf.png', dpi=165, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', nargs='?', type=Path, default=ROOT/'outputs/shaderball_first_hit_rays')
    directory = parser.parse_args().out
    spec = json.loads((directory/'rays.json').read_text(encoding='utf-8'))
    plot(directory, spec, summarize(directory, spec))
