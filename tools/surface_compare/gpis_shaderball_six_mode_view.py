"""Presentation for the seven-mode shaderball/native-FLIP comparison.

The data-producing report calls ``write_report(out, groups, sources)``.  This
module never renders a scene, changes an input image, or calculates FLIP.  It
arranges the original PNGs and the native CLI's error maps into figures and an
offline HTML report, returning the generated relative filenames.
"""
from __future__ import annotations

import html
import os
from pathlib import Path

import numpy as np
from figures import Image, Rectangle, plt


METHODS = ('classic_local', 'classic_global', 'start_conditioning', 'neural')
REFERENCES = ('gpis_global', 'gpis_renewal_plus',
              'gpis_renewal_plus_surface_conditioned')
MODES = METHODS + REFERENCES
LABELS = {
    'classic_local': 'classic_local',
    'classic_global': 'classic_global',
    'start_conditioning': 'start_conditioning',
    'neural': 'neural',
    'gpis_global': 'GPIS global',
    'gpis_renewal_plus': 'GPIS renewal+',
    'gpis_renewal_plus_surface_conditioned':
        'GPIS renewal+（保留非 GPIS 表面的实际 F 和梯度）',
}
PLOT_LABELS = dict(LABELS, gpis_renewal_plus_surface_conditioned=
                   'GPIS renewal+\n非 GPIS 表面保留 F 和梯度')
BOUNDS = (44, 164, 213, 216)
PAIRS = tuple((method, reference) for method in METHODS for reference in REFERENCES) + (
    ('gpis_renewal_plus', 'gpis_global'),
    ('gpis_renewal_plus_surface_conditioned', 'gpis_global'),
)


def pixels(path):
    with Image.open(path) as source:
        return np.asarray(source.convert('RGB'))


def crop(array):
    x0, y0, x1, y1 = BOUNDS
    return array[y0:y1, x0:x1]


def relative(path, out):
    return Path(os.path.relpath(path, out)).as_posix()


def esc(value):
    return html.escape(str(value), quote=True)


def link(path, label, out):
    return f'<a href="{esc(relative(path, out))}">{esc(label)}</a>'


def pair_lookup(group):
    lookup = {(item['test'], item['reference']): item
              for item in group['flip'].values()}
    assert len(lookup) == len(PAIRS) == len(group['flip'])
    assert set(lookup) == set(PAIRS)
    return lookup


def image_panel(ax, image, detail=False):
    ax.imshow(crop(image) if detail else image, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def finish_figure(fig, out, name, note):
    fig.text(.025, .012, note, fontsize=9, color='#455468')
    fig.savefig(out/name, dpi=150, facecolor='white')
    plt.close(fig)
    return name


def appearance(group, out, detail):
    fig, axes = plt.subplots(2, 4, figsize=(19, 5.8 if detail else 10.8))
    for ax, mode in zip(axes.flat, MODES):
        image_panel(ax, pixels(group['images'][mode]), detail)
        ax.set_title(PLOT_LABELS[mode], fontsize=12, pad=10)
        if not detail:
            x0, y0, x1, y1 = BOUNDS
            ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0,
                                   fill=False, edgecolor='#ffd050', lw=1))
    for ax in axes.flat[len(MODES):]:
        ax.set_axis_off()
    region = '固定底部' if detail else '七种渲染结果'
    fig.suptitle(f"{group['label']} · {region} · ℓ=0.2，σ={group['sigma']:g}",
                 fontsize=17, y=.98)
    fig.text(.5, .925 if detail else .935,
             '256 × 256 · 128 spp · seed 20261024 · 相同场景、相机、光照和材质',
             ha='center', fontsize=11, color='#455468')
    fig.subplots_adjust(left=.025, right=.985, bottom=.10 if detail else .055,
                        top=.79 if detail else .865,
                        wspace=.10, hspace=.50 if detail else .28)
    name = f"{group['label']}_appearance{'_bottom' if detail else ''}.png"
    return finish_figure(fig, out, name,
                         '原始 filmic PNG；未调曝光、未降噪。黄色框为固定底部区域。'
                         if not detail else
                         '原始 PNG 的固定裁图：x=[44,213)，y=[164,216)。')


def flip_matrix(group, out, detail):
    lookup = pair_lookup(group)
    fig, axes = plt.subplots(4, 3, figsize=(15, 10.0 if detail else 19.0))
    for row, method in enumerate(METHODS):
        for col, reference in enumerate(REFERENCES):
            item, ax = lookup[method, reference], axes[row, col]
            image_panel(ax, pixels(item['error_map']), detail)
            title = f"全图 mean FLIP = {item['full']['mean']:.6f}"
            if row == 0:
                title = PLOT_LABELS[reference]+'\n'+title
            ax.set_title(title, fontsize=11, pad=10)
            if col == 0:
                ax.set_ylabel(LABELS[method], rotation=0, ha='right', va='center',
                              labelpad=18, fontsize=12)
    region = '底部误差图' if detail else '全图误差图'
    fig.suptitle(f"{group['label']} · 四种方法 × 三种参考 · {region}",
                 fontsize=17, y=.985)
    fig.text(.5, .93 if detail else .945,
             f"ℓ=0.2，σ={group['sigma']:g} · 本地 FLIP 1.7 · 67 PPD · 固定 magma 色标 [0,1]",
             ha='center', fontsize=11, color='#455468')
    fig.subplots_adjust(left=.16, right=.985, bottom=.08 if detail else .045,
                        top=.82 if detail else .89,
                        wspace=.10, hspace=.50 if detail else .25)
    name = f"{group['label']}_flip_matrix{'_bottom' if detail else ''}.png"
    return finish_figure(fig, out, name,
                         '每行是测试方法，每列是参考。底部图裁自全图 FLIP；面板数值始终为全图 mean FLIP。'
                         if detail else
                         '每行是测试方法，每列是参考。直接展示本地 FLIP 输出，所有面板使用相同色标。')


def renewal_vs_global(group, out, detail):
    lookup = pair_lookup(group)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.1 if detail else 7.2))
    for ax, method in zip(axes, REFERENCES[1:]):
        item = lookup[method, 'gpis_global']
        image_panel(ax, pixels(item['error_map']), detail)
        title = PLOT_LABELS[method]+f"\n全图 mean FLIP = {item['full']['mean']:.6f}"
        ax.set_title(title, fontsize=12, pad=10)
    region = '固定底部' if detail else '全图'
    fig.suptitle(f"{group['label']} · 两种 GPIS renewal+ 对 GPIS global · {region}",
                 fontsize=16, y=.98)
    fig.subplots_adjust(left=.045, right=.985, bottom=.12 if detail else .065,
                        top=.64 if detail else .79, wspace=.12)
    name = f"{group['label']}_renewal_vs_global{'_bottom' if detail else ''}.png"
    return finish_figure(fig, out, name,
                         '本地 FLIP 1.7，67 PPD，magma [0,1]。底部图裁自全图误差图，面板数值为全图 mean FLIP。')


def figure_switch(group, stem, alt, out):
    label = group['label']
    figures = []
    for region in ('full', 'bottom'):
        name = f"{label}_{stem}{'_bottom' if region == 'bottom' else ''}.png"
        figures.append(f'<figure class="region-{region}">'
                       f'<a href="{esc(name)}" title="打开原尺寸图片">'
                       f'<img src="{esc(name)}" alt="{esc(alt)}" loading="lazy"></a>'
                       f'<figcaption><a href="{esc(name)}" download>下载此图 PNG</a>'
                       f'<span>{"全图" if region == "full" else "固定底部裁图"}</span>'
                       '</figcaption></figure>')
    return ''.join(figures)


def group_section(group, out):
    lookup = pair_lookup(group)
    rows = []
    for test, reference in PAIRS:
        item = lookup[test, reference]
        rows.append('<tr>'
                    f'<td>{esc(LABELS[test])}</td><td>{esc(LABELS[reference])}</td>'
                    f'<td>{item["full"]["mean"]:.6f}</td>'
                    f'<td>{item["bottom"]["mean"]:.6f}</td>'
                    f'<td>{link(item["error_map"], "原生 FLIP 图", out)}</td></tr>')
    label = esc(group['label'])
    renders = figure_switch(group, 'appearance', f'{label} 组七种渲染结果', out)
    matrix = figure_switch(group, 'flip_matrix', f'{label} 组四种方法对三种参考的 FLIP', out)
    renewal = figure_switch(group, 'renewal_vs_global', f'{label} 组两种 Renewal+ 对 Global 的 FLIP', out)
    return f'''<section class="group" data-group="{label}" id="group-{label}">
<div class="section-title"><h2>{label} 组 <span>ℓ = 0.2，σ = {group['sigma']:g}</span></h2>
<span class="badge">{len(MODES)} 种渲染 · {len(PAIRS)} 组误差配对</span></div>
<h3>渲染结果</h3>{renders}
<h3>classic_local、classic_global、start_conditioning、neural 对三种 GPIS 参考</h3>
<p class="caption">行对应测试方法，列对应参考方法。点击图片可查看原尺寸。</p>{matrix}
<h3>两种 GPIS renewal+ 对 GPIS global</h3>{renewal}
<h3>{len(PAIRS)} 组配对的 FLIP 指标</h3>
<p class="caption">全图数值对应上方误差图。底部数值是在无损裁切的输入图上重新计算 FLIP，边缘滤波上下文与全图不同。</p>
<div class="scroll"><table><thead><tr><th>测试方法</th><th>参考方法</th>
<th>全图 mean FLIP</th><th>底部输入 mean FLIP</th><th>单独查看</th></tr></thead>
<tbody>{''.join(rows)}</tbody></table></div></section>'''


def source_section(sources, out):
    rows = []
    for source in sources:
        folder = Path(source['folder'])
        links = ' · '.join(link(folder/name, label, out) for name, label in (
            ('render.png', 'PNG'), ('render.pfm', 'PFM'),
            ('scene.json', '场景配置'), ('result.json', '渲染记录')))
        rows.append(f'<tr><td>{esc(source["group"])}</td>'
                    f'<td>{esc(LABELS[source["mode"]])}</td><td>{links}</td></tr>')
    return '<details><summary>原始图像、配置与渲染记录</summary><div class="scroll">' \
        '<table><thead><tr><th>组别</th><th>方法</th><th>原始文件</th></tr></thead>' \
        f'<tbody>{"".join(rows)}</tbody></table></div></details>'


def write_report(out: Path, groups: list, sources: list) -> list[str]:
    """Write figures/HTML for prevalidated native-FLIP data; return filenames."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    names = []
    for group in groups:
        assert set(group['images']) == set(MODES)
        pair_lookup(group)
        for detail in (False, True):
            names.extend((appearance(group, out, detail),
                          flip_matrix(group, out, detail),
                          renewal_vs_global(group, out, detail)))
    sections = ''.join(group_section(group, out) for group in groups)
    records = source_section(sources, out)
    report = out.parent
    historical_links = ' · '.join(link(path, label, out) for path, label in (
        (report/'comparison.html', '此前模式比较'),
        (report/'shadow_condition_ablation/comparison.html', '非 GPIS 端点条件保留实验'),
        (report/'conditional_surface_policies/comparison.html', '起点保留 / 重启策略比较')))
    group_buttons = ''.join(f'<button type="button" data-select-group="{esc(g["label"])}" '
                            f'aria-pressed="false">{esc(g["label"])} 组 · σ={g["sigma"]:g}</button>'
                            for g in groups)
    content = '''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>七种方法比较 · 非 GPIS 表面端点条件与 FLIP</title>
<style>
:root{color-scheme:light;--ink:#24354a;--muted:#53677e;--line:#dbe3ec;--blue:#175e99;--paper:#fff}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1680px;margin:auto;padding:30px 26px 48px}h1{font-size:30px;line-height:1.35;margin:10px 0 14px}
h2{font-size:25px;margin:0}h2 span{font-size:17px;color:var(--muted);font-weight:500;margin-left:12px}
h3{font-size:20px;margin:28px 0 8px}p{margin:10px 0}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}
.eyebrow{color:var(--blue);font-size:13px;letter-spacing:.08em}.lead{max-width:1160px;color:var(--muted)}
.note{background:#e8f0f8;border:1px solid #cfdeec;border-radius:10px;padding:14px 18px;margin:18px 0}
.note p{margin:5px 0}.note strong{font-weight:650}.toolbar{position:sticky;top:0;z-index:2;background:rgba(243,246,250,.97);border-bottom:1px solid var(--line);padding:13px 0;display:flex;flex-wrap:wrap;gap:16px 28px;align-items:center}
.control{display:flex;gap:6px;align-items:center;flex-wrap:wrap}.control>span{font-size:14px;color:var(--muted);margin-right:4px}
button{font:inherit;font-size:14px;line-height:1.4;border:1px solid #c5d3e1;border-radius:7px;padding:8px 12px;background:#fff;color:var(--ink);cursor:pointer}
button[aria-pressed=true]{background:var(--blue);border-color:var(--blue);color:white}button:focus-visible,a:focus-visible{outline:3px solid #e6ad30;outline-offset:3px}
.group{margin-top:26px;padding:22px;background:#fff;border:1px solid var(--line);border-radius:12px}.section-title{display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap}
.badge{font-size:13px;color:var(--muted);background:#f0f4f8;padding:5px 10px;border-radius:20px}.caption{font-size:14px;color:var(--muted)}
figure{margin:10px 0 22px}figure img{display:block;width:100%;height:auto;background:white;border:1px solid #edf0f4}figcaption{display:flex;justify-content:space-between;padding:6px 2px;font-size:13px;color:var(--muted)}
.scroll{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:10px 12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}th{background:#f1f5f9;font-weight:650}td:nth-child(3),td:nth-child(4),th:nth-child(3),th:nth-child(4){text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}td:last-child{white-space:nowrap}
details{margin-top:26px;background:#fff;border:1px solid var(--line);border-radius:10px;padding:15px 18px}summary{font-weight:650;cursor:pointer}details table{margin-top:15px}details td:nth-child(3),details th:nth-child(3){text-align:left}
.footer{font-size:14px;color:var(--muted);margin-top:25px}.links{font-size:14px;line-height:2}code{background:#f0f4f8;padding:1px 4px;border-radius:3px;font-size:.9em}
body[data-region=full] .region-bottom,body[data-region=bottom] .region-full{display:none}
body[data-group=A] .group:not([data-group=A]),body[data-group=B] .group:not([data-group=B]){display:none}
@media(max-width:700px){main{padding:18px 12px}h1{font-size:25px}.group{padding:12px}h2 span{display:block;margin-left:0}.toolbar{gap:8px}.control{gap:4px}button{padding:7px 9px}.badge{display:none}}
@media print{body{background:#fff}.toolbar{display:none}.group{display:block!important;break-before:page;border:0}main{max-width:none;padding:0}details{display:none}.region-bottom{display:none!important}.region-full{display:block!important}}
</style></head><body data-region="full" data-group="all"><main>
<div class="eyebrow">GPIS SHADERBALL · MATCHED CONFIGURATION</div>
<h1>七种方法的渲染与 FLIP 比较</h1>
<p class="lead">classic_local、classic_global、start_conditioning、neural 分别对 GPIS global、GPIS renewal+ 以及保留非 GPIS 表面实际 F 和梯度的 GPIS renewal+ 做误差比较；另外比较两种 renewal+ 与 GPIS global。</p>
<div class="note"><p><strong>统一设置：</strong>ℓ=0.2；A 组 σ=0.02，B 组 σ=0.04。256×256、128 spp、seed=20261024，相同场景、相机、光照和材质。</p>
<p><strong>名称说明：</strong>start_conditioning 是 global_conditional 的展示名称，使用保留端点策略 retain_endpoint。</p>
<p><strong>变体说明：</strong>“GPIS renewal+（保留非 GPIS 表面的实际 F 和梯度）”是已保存的地面条件保留变体：普通表面直接光连接沿用端点观测，GPIS 碰撞规则保持原 Renewal+。GPIS renewal+ 列为回退后的原实现基线。</p></div>
<p class="links"><a href="metrics.json">完整指标与来源 JSON</a> · <a href="metrics.csv" download>指标 CSV</a> · <a href="verification.json">文件与配对核验</a> · HISTORICAL_LINKS</p>
<nav class="toolbar" aria-label="比较视图">
<div class="control"><span>参数组</span><button type="button" data-select-group="all" aria-pressed="true">A + B 全部</button>GROUP_BUTTONS</div>
<div class="control"><span>查看区域</span><button type="button" data-select-region="full" aria-pressed="true">全图</button><button type="button" data-select-region="bottom" aria-pressed="false">固定底部</button></div>
</nav>
SECTIONS
RECORDS
<div class="footer"><p>FLIP 直接调用本地 flip.exe 1.7，使用原始 filmic PNG，固定 67 PPD 和 magma [0,1] 色标；未进行曝光匹配、降噪或单图色标归一化。</p>
<p>固定底部区域为 x=[44,213)、y=[164,216)。底部误差图裁自全图 FLIP；表中底部指标则在裁切后的输入上独立计算，两者滤波上下文不同。参考图同为 128 spp，不代表已收敛真值。</p></div>
</main><script>
for(const kind of ['group','region']){
  const buttons=document.querySelectorAll('[data-select-'+kind+']');
  for(const button of buttons)button.addEventListener('click',()=>{
    const selected=button.getAttribute('data-select-'+kind);
    document.body.dataset[kind]=selected;
    for(const item of buttons)item.setAttribute('aria-pressed',String(item===button));
  });
}
</script></body></html>'''
    content = content.replace('HISTORICAL_LINKS', historical_links)
    content = content.replace('GROUP_BUTTONS', group_buttons)
    content = content.replace('SECTIONS', sections).replace('RECORDS', records)
    (out/'comparison.html').write_text(content, encoding='utf-8')
    return names+['comparison.html']
