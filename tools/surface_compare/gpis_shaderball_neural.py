"""Add the existing neural model to all three original shaderball parameter groups.

python -B tools/surface_compare/gpis_shaderball_neural.py run
<plot-python> -B tools/surface_compare/gpis_shaderball_neural.py figures

Rendering reuses completed, hash-verified images. The original four-mode report
is preserved and gains a link once a validated five-mode report is generated.
The optional `available` command builds a clearly marked interim report from
completed neural groups while the remaining renders are still running.
"""
from __future__ import annotations
import argparse
import copy
import csv
import subprocess
from pathlib import Path
import numpy as np
from gpis_shaderball import ROOT, SOURCE, NEURAL_MODEL, MODES, ROI, MODEL_ROI, render, field_path, folder
from gpis_shaderball_sweep import GROUPS, REPORT as PREVIOUS_REPORT, PLOT_PYTHON, configuration
from run import load, save
from shaderball_length import digest

REPORT=PREVIOUS_REPORT.parent/'ell0p2_neural_comparison'
DISPLAY_MODES=('classic_local','classic_global','global_conditional','neural','sparse_3d_global')
LABELS=('classic_local','classic_global','global_conditional','neural','GPIS global ensemble')
COLORS=('#1675b9','#df822d','#9a2b85','#b4513f','#168469')


def run():
    # The two requested groups precede the reference group.
    for ell,sigma in GROUPS[1:]+GROUPS[:1]:
        render(configuration(ell,sigma,modes=['neural']))
    subprocess.run([str(PLOT_PYTHON),'-B',str(Path(__file__).resolve()),'figures'],cwd=ROOT,check=True)


def plots(groups):
    from figures import plt, Image, Rectangle
    from plane_figures import blocks, filmic
    from gpis_shaderball_figures import regions
    from gpis_shaderball_sweep_figures import NAMES, group_label
    for kind in ('appearance','detail','block_means'):
        height=4*len(groups)+1
        fig,axes=plt.subplots(len(groups),5,figsize=(20,height),squeeze=False)
        for row,g in enumerate(groups):
            label_index=GROUPS.index((g['ell'],g['sigma']))
            for col,mode in enumerate(DISPLAY_MODES):
                ax=axes[row,col];d=g['data'][mode]
                if kind=='block_means': im=filmic(blocks(d['rgb']))
                else:
                    im=np.asarray(Image.open(Path(d['source']['folder'])/'render.png').convert('RGB'))
                    x0,y0,x1,y1=[int(v*256) for v in ROI]
                    if kind=='detail': im=im[y0:y1,x0:x1]
                    else: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=.8,edgecolor='#ffcf52'))
                ax.imshow(im,interpolation='nearest');ax.axis('off')
                if row==0: ax.set_title(LABELS[col],color=COLORS[col],fontsize=13)
                if col==0:
                    ax.text(-.07,.5,f'{NAMES[label_index]}\nℓ={g["ell"]:g}\nσ={g["sigma"]:g}\nσ/ℓ={g["gradient_stddev"]:g}',
                        transform=ax.transAxes,ha='right',va='center',fontsize=11)
        title={'appearance':'五模式平均外观','detail':'环形凹槽与球身细节','block_means':'16×16 分区平均外观'}[kind]
        fig.suptitle(f'原 GPIS shaderball · {title}\n相同相机、灯光、绿色镜面材质 · 256×256，128 spp',fontsize=18)
        fig.text(.13,.025,'统一 filmic 显示，无降噪或逐图曝光调整。neural 使用同一份已有模型权重，显式设置各组 ℓ、σ。',fontsize=11,color='#555b65')
        fig.subplots_adjust(left=.115,right=.99,top=1-1.2/height,bottom=.75/height,wspace=.035,hspace=.10)
        fig.savefig(REPORT/f'{kind}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(len(groups),4,figsize=(17,4*len(groups)+1),squeeze=False)
    for row,g in enumerate(groups):
        label_index=GROUPS.index((g['ell'],g['sigma']))
        ref=g['data']['sparse_3d_global']
        norm=float(g['data']['classic_global']['y'][regions(256)['shaderball_box']].mean())
        for col,mode in enumerate(DISPLAY_MODES[:-1]):
            ax=axes[row,col]
            delta=blocks((g['data'][mode]['y']-ref['y'])[:,:,None])[:,:,0]/norm
            im=ax.imshow(delta,cmap='RdBu_r',vmin=-.25,vmax=.25,interpolation='nearest');ax.axis('off')
            if row==0: ax.set_title(LABELS[col],color=COLORS[col],fontsize=13)
            if col==0: ax.text(-.08,.5,f'{NAMES[label_index]}\n{group_label(g)}',transform=ax.transAxes,ha='right',va='center',fontsize=10)
    fig.suptitle('各模式相对 GPIS global 平均外观的分区亮度差',fontsize=18)
    fig.colorbar(im,cax=fig.add_axes([.94,.22,.012,.56]),label='ΔY / 本组 classic_global 模型区域平均亮度')
    fig.text(.13,.025,'16×16 分区；统一色标 ±25%，超出范围截断显示。差值含采样噪声，GPIS 未视为已收敛真值。',fontsize=11,color='#555b65')
    fig.subplots_adjust(left=.14,right=.92,top=.92,bottom=.06,wspace=.04,hspace=.10)
    fig.savefig(REPORT/'difference.png',dpi=150);plt.close(fig)


def figures(available_only=False):
    from gpis_shaderball_figures import read
    from gpis_shaderball_sweep_figures import read_group, measure, PAIR_MODES, NAMES
    REPORT.mkdir(parents=True,exist_ok=True)
    previous=load(PREVIOUS_REPORT/'manifest.json');cache={}
    assert previous['source_sha256']==digest(SOURCE)
    model=load(NEURAL_MODEL);model_hash=digest(NEURAL_MODEL)
    assert model['kernel']=='squared_exponential' and model['distance_convention']=='q=distance_from_birth/ell'
    assert model['mean_convention']=='b=m/sigma'
    selected=[p for p in GROUPS if not available_only or (folder(configuration(*p),'neural')/'result.json').exists()]
    assert selected,'No completed neural render is available'
    groups=[read_group(*p,cache) for p in selected]
    statistics=[];pairs=[]
    pair_modes=PAIR_MODES+tuple(('neural',m) for m in MODES)
    current=previous['renderer_compatibility']['current_renderer_sha256']
    for i,g in enumerate(groups):
        reference_index=GROUPS.index((g['ell'],g['sigma']))
        # Reuse exactly the validated reference renders in the previous report.
        for d in g['data'].values():
            old=next(s for s in previous['groups'][reference_index]['sources'] if s['mode']==d['source']['mode'])
            assert d['source']==old
        args=configuration(g['ell'],g['sigma']);d=read(args,'neural')
        r=d['source']['result'];s=d['scene'];m=s['media'][0]
        assert r['resolution']==[256,256] and r['spp']==128
        assert r['provenance']['renderer_sha256']==current
        assert r['provenance']['model_sha256']==model_hash
        assert r['provenance']['field_sha256']==cache[field_path(args)]
        assert m['type']=='neural_causal_renewal' and m['model']==str(NEURAL_MODEL)
        assert m['covariance']==dict(type='squared_exponential',sigma=g['sigma'],lengthScale=g['ell'])
        ref=g['data']['classic_global']['scene']
        assert m['material']==ref['media'][0]['material']
        for key in ('primitives','bsdfs','camera','integrator','renderer'):
            a,b=copy.deepcopy(s[key]),copy.deepcopy(ref[key])
            if key=='renderer':
                a.pop('output_directory');b.pop('output_directory')
            assert a==b,key
        assert s['integrator']['enable_volume_light_sampling'] and s['integrator']['enable_light_sampling']
        log=(Path(d['source']['folder'])/'render.log').read_text(encoding='utf-8')
        assert 'Completed 128/128 spp' in log and 'Finished render.' in log
        g['data']['neural']=d
        a,b=measure(g,pair_modes);statistics.extend(a);pairs.extend(b)
    plots(groups)
    group_records=[]
    for i,g in enumerate(groups):
        group_records.append(dict(name=NAMES[GROUPS.index((g['ell'],g['sigma']))],ell=g['ell'],sigma=g['sigma'],gradient_stddev=g['gradient_stddev'],
            sources=[g['data'][m]['source'] for m in DISPLAY_MODES],field=g['field'],
            gpis_step_checks=g['step_checks']))
    manifest=dict(source=str(SOURCE),source_sha256=digest(SOURCE),resolution=[256,256],spp=128,
        complete=len(selected)==len(GROUPS),pending_groups=[dict(ell=p[0],sigma=p[1]) for p in GROUPS if p not in selected],
        reference_manifest=str(PREVIOUS_REPORT/'manifest.json'),reference_manifest_sha256=digest(PREVIOUS_REPORT/'manifest.json'),
        model=dict(path=str(NEURAL_MODEL),sha256=model_hash,checkpoint_sha256=model['checkpoint_sha256'],
            selected_epoch=model['selected_epoch'],config=model['model_config'],gradient_config=model['gradient_config'],
            distance_convention=model['distance_convention'],mean_convention=model['mean_convention'],retrained=False),
        groups=group_records,statistics=statistics,pair_metrics=pairs,model_roi=MODEL_ROI,detail_roi=ROI,
        block_roi_indices=[int(v*16) for v in MODEL_ROI],
        figures=[f'{name}.png' for name in ('appearance','detail','block_means','difference')],
        notes=['All modes use NEE/MIS and identical scene, camera, material, resolution, spp and display settings.',
            'Neural uses the full-domain trilinear SDF at dx=.005 and its unnormalized one-sided derivative. Classic_local normalizes the interpolated mean gradient; other modes use the procedural mean.',
            'The existing normalized model is evaluated at all three parameter groups without retraining; accuracy at those parameters is not guaranteed by normalization.',
            'Fixed rectangular image regions include some background. ROI endpoints are rounded down on the 16x16 block grid, matching the previous report. Block RMS includes sampling noise; paired-half noise estimates are descriptive, not confidence intervals.',
            'GPIS is an approximate reference; the previous report records step-size sensitivity and zero-length-ray diagnostics.'])
    save(REPORT/'manifest.json',manifest)
    for name,rows in [('metrics',statistics),('pair_metrics',pairs)]:
        with (REPORT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_page(groups,manifest)
    # Keep the currently open four-mode page useful without replacing its data.
    page=PREVIOUS_REPORT/'comparison.html';html=page.read_text(encoding='utf-8')
    if '../ell0p2_neural_comparison/comparison.html' not in html:
        end=html.index('</h1>')+len('</h1>')
        html=html[:end]+'\n<p><a href="../ell0p2_neural_comparison/comparison.html">查看加入 neural 的五模式对比</a></p>'+html[end:]
        page.write_text(html,encoding='utf-8')
    print(REPORT/'comparison.html',flush=True)


def write_page(groups,manifest):
    from gpis_shaderball_sweep_figures import NAMES, group_label
    pairs=manifest['pair_metrics'];stats=manifest['statistics']
    def link(path): return '../'+Path(path).relative_to(REPORT.parent).as_posix()
    distance_rows='';raw_rows='';mean_rows='';step_rows=''
    for i,g in enumerate(groups):
        label=f'{NAMES[GROUPS.index((g["ell"],g["sigma"]))]}<br>{group_label(g)}';cells=[]
        for mode,reference in [('classic_local','sparse_3d_global'),('classic_global','sparse_3d_global'),
            ('global_conditional','sparse_3d_global'),('neural','sparse_3d_global'),('neural','global_conditional')]:
            r=next(r for r in pairs if (r['ell'],r['sigma'],r['mode'],r['reference'])==(g['ell'],g['sigma'],mode,reference))
            cells.append(f'<td>{100*r["relative_rms"]:.2f}%<small>噪声 RMS 约 {100*r["relative_noise_rms"]:.2f}%</small></td>')
        distance_rows+=f'<tr><td>{label}</td>'+''.join(cells)+'</tr>'
        links=[]
        for mode in DISPLAY_MODES:
            d=Path(g['data'][mode]['source']['folder'])
            links.append(f'<td><a href="{link(d/"render.png")}">PNG</a> · <a href="{link(d/"scene.json")}">场景</a></td>')
            r=next(r for r in stats if (r['ell'],r['sigma'],r['mode'],r['region'])==(g['ell'],g['sigma'],mode,'shaderball_box'))
            mean_rows+=f'<tr><td>{label}</td><td>{mode}</td><td>{r["mean_luminance"]:.6f} ± {1.96*r["standard_error"]:.6f}</td><td>{100*r["relative_difference"]:+.2f}%</td></tr>'
        raw_rows+=f'<tr><td>{label}</td>'+''.join(links)+'</tr>'
        for c in g['step_checks']:
            if c['region'] in ('shaderball_box','detail'):
                step_rows+=f'<tr><td>{label}</td><td>{c["region"]}</td><td>{100*c["relative_difference"]:+.3f}%</td></tr>'
    headers=''.join(f'<th>{s}</th>' for s in LABELS)
    status=''
    if not manifest['complete']:
        pending='、'.join(f'ℓ={g["ell"]:g}，σ={g["sigma"]:g}' for g in manifest['pending_groups'])
        status=f'<p class="note">本页展示已完成的正式图。待补：{pending}；该组完成后会更新本页。</p>'
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>GPIS shaderball · neural 五模式对比</title>
<style>body{{max-width:1800px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#253145;background:#f5f6f8}}h1{{font-size:28px}}h2{{font-size:23px;margin-top:36px}}img{{width:100%;background:white}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}small{{display:block;font-size:12px;color:#657083}}.note{{background:#e7edf5;padding:16px;border-radius:10px}}details{{margin-top:25px}}summary{{cursor:pointer;font-weight:600}}.table{{overflow:auto}}</style>
<h1>原 GPIS shaderball：加入 neural 的五模式对比</h1>
{status}
<p>五模式均为 256×256、128 spp，使用原 shaderball-stationary-270 场景的相机、灯光、地面和绿色镜面材质。NEE/MIS 开启，统一 filmic 显示，无降噪。</p>
<div class="note"><b>A：ℓ=0.2，σ=0.02</b>，只增加相关长度，σ/ℓ=0.1。<br><b>B：ℓ=0.2，σ=0.04</b>，保持原基准的梯度扰动幅度 σ/ℓ=0.2。<br>基准为 ℓ=0.1、σ=0.02。neural 三组共用已有权重，分别输入对应参数；本轮未重新训练。</div>
<p><a href="../ell0p2_comparison/comparison.html">四模式原页、固定场图与步长核验</a> · <a href="manifest.json">参数与来源</a> · <a href="metrics.csv">区域亮度统计</a> · <a href="pair_metrics.csv">模式间差异</a></p>
<h2>五模式平均外观</h2><a href="appearance.png"><img src="appearance.png" alt="三组参数的五模式平均外观"></a>
<h2>环形凹槽与球身细节</h2><a href="detail.png"><img src="detail.png" alt="五模式的相同局部放大"></a>
<h2>与 GPIS 及 conditional 的差异</h2>
<p>统计固定模型区域在 16×16 分区网格上取整后的线性亮度 RMS 差，再除以本组 classic_global 的区域平均亮度；与前一页采用相同算法。数值越小表示本次图像越接近；噪声估计为描述性统计，未从差值中扣除，也不是置信区间。</p>
<div class="table"><table><tr><th>参数组</th><th>classic_local ↔ GPIS</th><th>classic_global ↔ GPIS</th><th>conditional ↔ GPIS</th><th>neural ↔ GPIS</th><th>neural ↔ conditional</th></tr>{distance_rows}</table></div>
<a href="difference.png"><img src="difference.png" alt="相对 GPIS 的分区亮度差"></a>
<details><summary>查看分区平均外观</summary><a href="block_means.png"><img src="block_means.png" alt="分区平均外观"></a></details>
<h2>原图与场景</h2><div class="table"><table><tr><th>参数组</th>{headers}</tr>{raw_rows}</table></div>
<details><summary>模型、几何与参考结果的核验</summary>
<p>neural 模型：renewal_shader_ball_sigma002_r05_joint/renewal_model.json，epoch {manifest['model']['selected_epoch']}。其输入采用 q=distance/ℓ、b=m/σ，支持本次参数设置；归一化并不保证所有参数下的预测精度。这些图展示现有权重的表现。</p>
<p>neural 读取本次原 knob 的完整 0.005 网格，仅使用 SDF 及其未归一化的单侧插值梯度；classic_local 使用同一几何网格，并将均值梯度单位化。其余三模式使用原程序化均值与梯度。因此模式差异仍包含表示和近似方法的影响。</p>
<p>GPIS 为 3D global 稀疏卷积平均外观，每格 10 个脉冲，主图步长 0.01。已有 96×96、32 spp 的步长减半检查显示以下区域平均亮度变化；未将其视为已收敛真值。原页保留了完整检查和边界诊断。</p>
<table><tr><th>参数组</th><th>区域</th><th>GPIS 步长 0.01 → 0.005</th></tr>{step_rows}</table>
<p>模型统计矩形为 x∈[{MODEL_ROI[0]},{MODEL_ROI[2]})、y∈[{MODEL_ROI[1]},{MODEL_ROI[3]})，包含部分背景。以下 ± 为近似 95% 采样误差，不包含模型或网格误差。</p>
<table><tr><th>参数组</th><th>模式</th><th>区域平均亮度</th><th>相对 classic_global</th></tr>{mean_rows}</table>
<p>复现：<code>python -B tools/surface_compare/gpis_shaderball_neural.py run</code></p>
</details></html>'''
    (REPORT/'comparison.html').write_text(html,encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('run','figures','available'))
    args=parser.parse_args()
    if args.command=='run': run()
    else: figures(available_only=args.command=='available')
