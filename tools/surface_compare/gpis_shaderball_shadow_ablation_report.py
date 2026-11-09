"""Compare the surface-shadow conditioning experiment with the original renders.

Run with the project's plot Python after the experimental renders finish:
  python -B tools/surface_compare/gpis_shaderball_shadow_ablation_report.py

This only writes the standalone shadow_condition_ablation report. It does not
render, replace the six-mode comparison, or depend on the currently built exe.
"""
from __future__ import annotations

import argparse
import copy
import html
import os
import subprocess
from pathlib import Path

from figures import Image, Rectangle, plt
import numpy as np
from run import load, save, sha
from gpis_shaderball_flip import DEFAULT_FLIP, FIELDS, evaluate, verify_links
from gpis_shaderball_renewal_3d import (
    ROOT, REPORT, GROUPS, GLOBAL, SEED, destination, original, verified,
)

OUT = REPORT / 'shadow_condition_ablation'
STAGE = 'renewal_3d_shadow_conditioned'
BOUNDS = [44, 164, 213, 216]
PPD = 67.0
LABELS = {'global': 'GPIS global', 'old': 'Original 3D Renewal+',
          'new': 'Renewal+ 地面条件保留变体'}
PAIRS = (('old_vs_global', 'old', 'global'),
         ('new_vs_global', 'new', 'global'), ('new_vs_old', 'new', 'old'))


def crop(a):
    x0, y0, x1, y1 = BOUNDS
    return a[y0:y1, x0:x1]


def relative(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def image_array(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('RGB'))


def normalized_scene(scene):
    scene = copy.deepcopy(scene)
    scene['renderer'].pop('output_directory')
    return scene


def read_group(key, stage):
    data = {name: verified(folder) for name, folder in (
        ('global', original(key, GLOBAL)),
        ('old', destination(key, 'renewal_3d')),
        ('new', destination(key, stage)))}
    for name, d in data.items():
        result = d['source']['result']
        assert result['resolution'] == [256, 256] and result['spp'] == 128
        # Archived global records predate the explicit seed field; their run
        # directories carry the seed used by the original experiment runner.
        assert Path(d['source']['folder']).name == f'seed{SEED}'
        if 'seed' in result:
            assert result['seed'] == SEED
        assert np.isfinite(d['rgb']).all() and (d['rgb'] >= 0).all()
        medium = d['scene']['media'][0]
        assert medium['type'] == 'sparse_conv_noise'
        assert not medium['1D_sampling'] and not medium['single_realization']
        assert medium['1D_sampling_scheme'] == 'uni'
        assert not medium['1D_gradient_correlationXY']
        assert medium['step_size'] == .01
        assert medium['correlation_context'] == ('global' if name == 'global' else 'renewal+')
        d['png'] = Path(d['source']['folder']) / 'render.png'
        with Image.open(d['png']) as image:
            assert image.size == (256, 256)
    assert normalized_scene(data['old']['scene']) == normalized_scene(data['new']['scene']), 'Ablation scenes differ beyond output directory'
    s = normalized_scene(data['global']['scene'])
    s['media'][0]['correlation_context'] = 'renewal+'
    assert s == normalized_scene(data['old']['scene']), 'Global/Renewal scenes differ beyond memory model and output directory'
    return dict(key=key, label='A' if key.endswith('0p02') else 'B',
                sigma=.02 if key.endswith('0p02') else .04, data=data)


def linear_statistics(group):
    rows = []
    for pair, test, reference in PAIRS:
        a, b = group['data'][test], group['data'][reference]
        delta = a['y'] - b['y']
        halves = [x-y for x, y in zip(a['halves'], b['halves'])]
        assert np.allclose(delta, .5*(halves[0]+halves[1]), rtol=1e-4, atol=3e-7)
        noise = .5*(halves[0]-halves[1])
        for region in ('full', 'bottom'):
            select = crop if region == 'bottom' else lambda x: x
            d, n = select(delta), select(noise)
            test_mean = float(select(a['y']).mean())
            ref_mean = float(select(b['y']).mean())
            half_means = [float(select(x).mean()) for x in halves]
            se = float(np.sqrt(np.sum(n*n))/d.size)
            rows.append(dict(group=group['label'], pair=pair, region=region,
                pixels=d.size, test_mean_y=test_mean, reference_mean_y=ref_mean,
                mean_delta_y=float(d.mean()), relative_mean_delta=test_mean/ref_mean-1,
                luminance_rmse=float(np.sqrt(np.mean(d*d))),
                paired_half_delta_means=half_means,
                same_mean_sign_in_both_halves=bool(half_means[0]*half_means[1] > 0),
                approximate_paired_mean_se=se))
    return rows


def grayscale_flip(reference, test, folder, expected):
    """Use CLI grayscale output for ROI pooling, recording 8-bit quantization."""
    folder.mkdir(parents=True, exist_ok=True)
    command = [str(DEFAULT_FLIP), '-r', str(reference), '-t', str(test),
               '-ppd', str(PPD), '-d', str(folder), '-txt', '-nm']
    identity = dict(command=command, executable_sha256=sha(DEFAULT_FLIP),
                    reference_sha256=sha(reference), test_sha256=sha(test))
    record = folder/'result.json'
    if record.exists():
        result = load(record)
        assert result['identity'] == identity
        for name, digest in result['sha256'].items():
            assert sha(folder/name) == digest
    else:
        run = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                             encoding='utf-8', errors='replace')
        (folder/'flip.log').write_text(run.stdout+run.stderr, encoding='utf-8')
        assert run.returncode == 0 and 'Invoking LDR-FLIP' in run.stdout
        maps = list(folder.glob('flip.*.ldr.png'))
        pooled = list(folder.glob('pooled_values.flip.*.ldr.txt'))
        assert len(maps) == len(pooled) == 1
        statistics = {}
        for line in pooled[0].read_text(encoding='utf-8').splitlines():
            if ':' in line:
                key, value = line.split(':', 1)
                if key in FIELDS:
                    statistics[FIELDS[key]] = float(value)
        assert statistics == expected['statistics'], 'Magma/grayscale should only change display'
        result = dict(identity=identity, statistics=statistics, error_map=maps[0].name,
            pooled_values=pooled[0].name,
            sha256={p.name: sha(p) for p in (maps[0], pooled[0], folder/'flip.log')})
        save(record, result)
    rgb = image_array(folder/result['error_map'])
    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 1])
    assert np.array_equal(rgb[:, :, 0], rgb[:, :, 2])
    errors = rgb[:, :, 0].astype(np.float64)/255.0
    assert abs(errors.mean()-expected['statistics']['mean']) <= 1/255+1e-6
    return result, errors


def flip_statistics(group):
    label, data = group['label'], group['data']
    inputs = OUT/'inputs'/label
    inputs.mkdir(parents=True, exist_ok=True)
    for name, d in data.items():
        d['bottom_png'] = inputs/f'{name}_bottom.png'
        Image.fromarray(crop(image_array(d['png']))).save(d['bottom_png'])
    rows, records = [], []
    group['flip'] = {}
    for pair, test, reference in PAIRS:
        a, b = data[test], data[reference]
        folder = OUT/'flip'/label/pair
        full = evaluate(DEFAULT_FLIP, b['png'], a['png'], folder/'full', PPD)
        bottom = evaluate(DEFAULT_FLIP, b['bottom_png'], a['bottom_png'], folder/'bottom_input', PPD)
        gray, errors = grayscale_flip(b['png'], a['png'], folder/'full_gray', full)
        group['flip'][pair] = dict(full=full, bottom_input=bottom,
            error_map=folder/'full'/full['error_map'])
        rows.append(dict(group=label, pair=pair, full=full['statistics'],
            bottom_cropped_input=bottom['statistics'],
            bottom_full_context_quantized_mean=float(crop(errors).mean()),
            full_context_quantized_mean=float(errors.mean()),
            quantization_absolute_bound=1/255,
            bottom_crop_sha256=dict(reference=sha(b['bottom_png']), test=sha(a['bottom_png']))))
        for kind, result in (('full', full), ('bottom_input', bottom), ('full_gray', gray)):
            record = folder/kind/'result.json'
            records.append(dict(group=label, pair=pair, kind=kind, path=relative(record),
                                sha256=sha(record), result=result))
    print('FLIP complete', label, flush=True)
    return rows, records


def plots(groups):
    names = []
    for detail in (False, True):
        fig, axes = plt.subplots(len(groups), 5, figsize=(18, (3.0 if detail else 4.4)*len(groups)), squeeze=False)
        for row, g in enumerate(groups):
            panels = [(image_array(g['data'][name]['png']), LABELS[name]) for name in ('global', 'old', 'new')]
            for pair, title in (('old_vs_global', 'Original vs global'), ('new_vs_global', 'Modified vs global')):
                f = g['flip'][pair]
                panels.append((image_array(f['error_map']), f"{title}\nFull FLIP = {f['full']['statistics']['mean']:.6f}"))
            for col, (pixels, title) in enumerate(panels):
                ax = axes[row, col]
                ax.imshow(crop(pixels) if detail else pixels, interpolation='nearest')
                if not detail:
                    x0, y0, x1, y1 = BOUNDS
                    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='#ffd050', lw=.8))
                ax.set_title(title, fontsize=11)
                ax.set_xticks([]); ax.set_yticks([])
                if col == 0:
                    ax.set_ylabel(f"{g['label']}: sigma={g['sigma']:g}")
        fig.suptitle('Surface-shadow conditioning ablation: '+('fixed bottom crop' if detail else 'full images')+'\n256 x 256, 128 spp, identical scene and seed; native local FLIP, 67 PPD', fontsize=14)
        fig.tight_layout(rect=(0, .04, 1, .90))
        fig.text(.03, .012, 'Original PNGs; no exposure fitting. Bottom error panels are crops of full-image FLIP maps (magma 0-1).', fontsize=9)
        name = 'bottom_comparison.png' if detail else 'full_comparison.png'
        fig.savefig(OUT/name, dpi=150); plt.close(fig); names.append(name)
    fig, axes = plt.subplots(len(groups), 4, figsize=(16, 3.2*len(groups)), squeeze=False)
    for row, g in enumerate(groups):
        d = g['data']
        deltas = [d['old']['y']-d['global']['y'], d['new']['y']-d['global']['y'], d['new']['y']-d['old']['y']]
        for col, (delta, title) in enumerate(zip(deltas, ('Original - global', 'Modified - global', 'Modified - original'))):
            handle = axes[row, col].imshow(crop(delta), cmap='RdBu_r', vmin=-.12, vmax=.12, interpolation='nearest')
            axes[row, col].set_title(title+'\nLinear luminance, scale +/-0.12')
        axes[row, 3].imshow(crop(image_array(g['flip']['new_vs_old']['error_map'])), interpolation='nearest')
        axes[row, 3].set_title('Modified vs original\nFLIP, scale 0-1')
        for ax in axes[row]:
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{g['label']}: sigma={g['sigma']:g}")
    fig.suptitle('Fixed bottom region: x=[44,213), y=[164,216)', fontsize=14)
    fig.tight_layout(rect=(0, .04, .95, .90))
    fig.colorbar(handle, cax=fig.add_axes([.96, .19, .01, .55]), label='Linear delta Y')
    fig.text(.03, .012, 'Red: brighter; blue: darker. Signed differences include sampling noise. No causal attribution is imposed.', fontsize=9)
    fig.savefig(OUT/'bottom_changes.png', dpi=150); plt.close(fig)
    return names+['bottom_changes.png']


def outcome_note(experiment_info):
    """Render an explicitly recorded decision, without inferring one from metrics."""
    if not experiment_info:
        return ''
    decision = experiment_info.get('decision')
    conclusion = experiment_info.get('conclusion')
    if not (isinstance(decision, str) and decision.strip()
            and isinstance(conclusion, str) and conclusion.strip()):
        return ''
    return (f'<div class="note"><b>实验结论：</b>{html.escape(conclusion)}'
            f'<br><b>处理决定：</b>{html.escape(decision)}</div>')


def write_page(groups, linear, flip, figures, experiment, experiment_info=None):
    def link(path, label):
        return f'<a href="{html.escape(relative(path), quote=True)}">{html.escape(label)}</a>'
    rows, source_rows = [], []
    pair_labels = {'old_vs_global': '原 Renewal+ − Global', 'new_vs_global': '修改后 Renewal+ − Global', 'new_vs_old': '修改后 − 原 Renewal+'}
    for g in groups:
        for pair, test, ref in PAIRS:
            f = next(r for r in flip if r['group'] == g['label'] and r['pair'] == pair)
            for region, region_label in (('full', '全图'), ('bottom', '固定底部')):
                r = next(r for r in linear if r['group'] == g['label'] and r['pair'] == pair and r['region'] == region)
                metric = f['full']['mean'] if region == 'full' else f['bottom_cropped_input']['mean']
                gray = '—' if region == 'full' else f"{f['bottom_full_context_quantized_mean']:.6f}"
                rows.append(f'<tr><td>{g["label"]}</td><td>{pair_labels[pair]}</td><td>{region_label}</td><td>{r["mean_delta_y"]:+.7f}</td><td>{r["relative_mean_delta"]:+.3%}</td><td>{r["approximate_paired_mean_se"]:.7f}</td><td>{metric:.6f}</td><td>{gray}</td></tr>')
        for name, d in g['data'].items():
            folder = Path(d['source']['folder'])
            links = ' · '.join(link(folder/file, label) for file, label in (
                ('render.png', '原 PNG'), ('render.pfm', '线性 PFM'), ('scene.json', '场景'), ('result.json', '渲染记录')))
            source_rows.append(f'<tr><td>{g["label"]}</td><td>{LABELS[name]}</td><td>{links}</td><td><code>{d["source"]["result"]["provenance"]["renderer_sha256"]}</code></td></tr>')
    experiment_link = ' · '+link(experiment, '补丁与回退记录') if experiment else ''
    policies_page = REPORT/'conditional_surface_policies/comparison.html'
    policies_link = ' · '+link(policies_page, 'global_conditional 两种策略比较') if policies_page.is_file() else ''
    figure_html = ''.join(f'<a href="{name}"><img src="{name}" alt="{name}"></a>' for name in figures)
    content = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Renewal+ 地面阴影条件化消融</title><style>body{{max-width:1800px;margin:28px auto;padding:0 24px;color:#263448;background:#f5f6f8;font:16px/1.65 system-ui}}h1{{font-size:28px}}h2{{margin-top:30px}}img{{width:100%;background:white;margin-top:16px}}table{{width:100%;border-collapse:collapse;background:white}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}td:nth-child(-n+3),th:nth-child(-n+3){{text-align:left}}.scroll{{overflow:auto}}.note{{padding:16px;background:#e6edf5;border-radius:8px}}a{{color:#235e91}}code{{font-size:11px}}</style>
<h1>Renewal+：地面直接光查询保留端点条件的消融实验</h1>
{outcome_note(experiment_info)}
<p>A：ℓ=0.2、σ=0.02；B：ℓ=0.2、σ=0.04。相同场景、256×256、128 spp、seed={SEED}、步长 0.01。比较原 3D Renewal+、修改后的 3D Renewal+ 与原 GPIS Global。</p>
<p class="note">这份报告独立保留实验结果。原始对比页面和渲染结果未由本脚本替换。全图 FLIP 直接调用本地 flip.exe 1.7，输入原始 filmic PNG，固定 67 PPD。线性亮度使用原始 PFM。数值更接近 128 spp Global 并不等同于证明物理正确或已经收敛。</p>
<p>{link(REPORT/'comparison.html', '原六模式对比')} · {link(OUT/'metrics.json', '完整指标和来源')}{experiment_link}{policies_link}</p>
<h2>全图与固定底部</h2>{figure_html}
<h2>数值比较</h2><p>固定底部为 x=[44,213)、y=[164,216)，与此前诊断相同。ΔY 为测试图减参考图；相对差除以参考图区域均值。配对 SE 由两图奇偶样本半图之差估计，仅作单种子描述，假设像素采样流近似独立。</p>
<p><b>FLIP 两种底部统计不能混为一谈：</b>“原生 mean FLIP”的底部行是在固定 ROI 的原 PNG 裁图上重新调用 FLIP，数值未经 PNG 误差图量化，但 ROI 边缘滤波上下文与全图不同。“全图上下文底部”则从 CLI 全图灰度误差 PNG 裁取并求均值，保留全图滤波上下文，存在最多 1/255 的绝对量化误差。上方展示的底部误差图均裁自全图 magma 输出。</p>
<div class="scroll"><table><tr><th>组</th><th>测试 − 参考</th><th>区域</th><th>平均 ΔY</th><th>相对均值差</th><th>配对 SE 估计</th><th>原生 mean FLIP</th><th>全图上下文底部（量化）</th></tr>{''.join(rows)}</table></div>
<p>同一方向的奇偶半图变化、改前改后配对误差和所有 FLIP 原始输出记录均保存在 metrics.json。本报告不自动把改善解释为唯一原因；需结合补丁范围及剩余偏差判断。</p>
<h2>输入与渲染器来源</h2><div class="scroll"><table><tr><th>组</th><th>版本</th><th>原始文件</th><th>渲染器 SHA256</th></tr>{''.join(source_rows)}</table></div>
</html>'''
    (OUT/'comparison.html').write_text(content, encoding='utf-8')


def main(args):
    OUT.mkdir(parents=True, exist_ok=True)
    groups = [read_group(key, args.stage) for key in args.groups]
    tool_help = subprocess.run([str(DEFAULT_FLIP), '--help'], capture_output=True,
                              text=True, encoding='utf-8', errors='replace', check=True)
    help_text = tool_help.stdout+tool_help.stderr
    assert 'FLIP v1.7.' in help_text
    (OUT/'flip_tool_help.txt').write_text(help_text, encoding='utf-8')
    linear, flip, records = [], [], []
    for g in groups:
        linear.extend(linear_statistics(g))
        f, r = flip_statistics(g)
        flip.extend(f); records.extend(r)
    figures = plots(groups)
    experiment = OUT/'experiment.json'
    experiment = experiment if experiment.is_file() else None
    experiment_info = None
    if experiment:
        recorded = load(experiment)
        experiment_info = dict(path=relative(experiment), sha256=sha(experiment),
                               decision=recorded.get('decision'),
                               conclusion=recorded.get('conclusion'))
    sources = []
    for g in groups:
        for name, d in g['data'].items():
            folder = Path(d['source']['folder'])
            sources.append(dict(group=g['label'], mode=name, folder=str(folder),
                result_sha256=sha(folder/'result.json'), result=d['source']['result']))
    metrics = dict(complete=True, groups=[g['key'] for g in groups], stage=args.stage,
        bottom_bounds=BOUNDS, settings=dict(resolution=[256, 256], spp=128, seed=SEED, step=.01),
        linear_statistics=linear, flip_statistics=flip, flip_records=records, sources=sources,
        tool=dict(path=str(DEFAULT_FLIP), sha256=sha(DEFAULT_FLIP), version='1.7', ppd=PPD,
                  help_sha256=sha(OUT/'flip_tool_help.txt')),
        report_script_sha256=sha(Path(__file__)),
        experiment=experiment_info,
        figures={name: sha(OUT/name) for name in figures},
        definitions=dict(delta='test - reference', relative_delta='mean_delta / reference_mean',
            paired_mean_se='sqrt(sum(((half_delta_A-half_delta_B)/2)**2)) / pixel_count',
            bottom_cropped_input_flip='Local FLIP on lossless crops of original PNG inputs; changed boundary filtering context.',
            bottom_full_context_quantized_flip='Mean of bottom ROI in full-image local FLIP grayscale PNG; absolute quantization bound 1/255.',
            limitations='One seed, one odd/even split, 128-spp approximate reference. Sampling, discretization and field-approximation errors remain. No automatic causal or convergence claim.'))
    save(OUT/'metrics.json', metrics)
    write_page(groups, linear, flip, figures, experiment, experiment_info)
    links = verify_links(OUT/'comparison.html')
    save(OUT/'verification.json', dict(passed=True, render_records=len(sources),
        flip_records=len(records), local_links=links,
        generated_sha256={name: sha(OUT/name) for name in ('comparison.html', 'metrics.json', 'flip_tool_help.txt', *figures)}))
    for g in groups:
        for pair, _, _ in PAIRS:
            r = next(r for r in linear if r['group']==g['label'] and r['pair']==pair and r['region']=='bottom')
            f = next(r for r in flip if r['group']==g['label'] and r['pair']==pair)
            print(g['label'], pair, 'bottom delta', r['mean_delta_y'], 'relative', r['relative_mean_delta'],
                  'SE', r['approximate_paired_mean_se'], 'half means', r['paired_half_delta_means'],
                  'full FLIP', f['full']['mean'], 'bottom input FLIP', f['bottom_cropped_input']['mean'], flush=True)
    print(OUT/'comparison.html', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', default=STAGE)
    parser.add_argument('--groups', nargs='+', choices=GROUPS, default=list(GROUPS))
    main(parser.parse_args())
