"""Merge independent shaderball runs in linear radiance and extend their report.

python -B tools/surface_compare/report_shaderball_small_light.py
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import gpis_shaderball_small_light as experiment
from gpis_shaderball_small_light import ROOT,OUT,MODES,Y,pfm,read,save,sha,table
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def write_pfm(path,rgb):
    h,w,_=rgb.shape
    path.write_bytes(f'PF\n{w} {h}\n-1.0\n'.encode()+rgb[::-1].astype('<f4').tobytes())

def filmic(rgb):
    x=np.maximum(rgb-.004,0)
    return x*(6.2*x+.5)/(x*(6.2*x+1.7)+.06)

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args();out=args.output.resolve()
    experiment.report(SimpleNamespace(output=out))
    data=read(out/'summary.json');manifest=read(out/'manifest.json')
    combined=[];captured={};all_sources={}
    averages=out/'averages';averages.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(15,5.6),constrained_layout=True)
    detail,detail_axes=plt.subplots(1,3,figsize=(15,4.8),constrained_layout=True)
    for ax,detail_ax,mode in zip(axes,detail_axes,MODES):
        sources=[r for r in data['rows'] if r['stage']=='convergence' and r['mode']==mode]
        assert len(sources)==2 and len({r['seed'] for r in sources})==2
        assert len({r['spp'] for r in sources})==1
        inputs=[out/r['folder'] for r in sources]
        rgbs=[pfm(p/'render.pfm') for p in inputs]
        half_noise=[.5*((pfm(p/'samplesA.pfm')-pfm(p/'samplesB.pfm'))@Y) for p in inputs]
        mean=np.mean(rgbs,axis=0);n=len(inputs);spp=sum(r['spp'] for r in sources)
        noise_rms=float(np.sqrt(np.mean([np.mean(h*h) for h in half_noise])/n))
        independent_noise=float(np.sqrt(np.mean(((rgbs[0]-rgbs[1])@Y)**2))/2)
        h,w,_=mean.shape
        write_pfm(averages/(mode+'.pfm'),mean)
        display=np.rint(np.clip(filmic(mean),0,1)*255).astype(np.uint8)
        Image.fromarray(display).save(averages/(mode+'.png'))
        entry=dict(mode=mode,resolution=[w,h],total_spp=spp,seeds=[r['seed'] for r in sources],
            mean_luminance=float((mean@Y).mean()),
            seed_mean_luminances=[r['mean_luminance'] for r in sources],
            noise_rms_from_half_buffers=noise_rms,noise_rms_from_independent_seeds=independent_noise,
            render_seconds=sum(r['render_seconds'] for r in sources),
            png=f'averages/{mode}.png',pfm=f'averages/{mode}.pfm')
        combined.append(entry);captured[mode]=(mean,half_noise)
        ax.imshow(display,interpolation='nearest');ax.axis('off')
        ax.set_title(mode+f'\n{spp} total spp | half-noise {noise_rms:.5g}',fontsize=11)
        detail_ax.imshow(display[int(.18*h):int(.68*h),int(.2*w):int(.8*w)],interpolation='nearest')
        detail_ax.axis('off');detail_ax.set_title(mode,fontsize=11)
        for p in inputs:
            for name in ['render.pfm','samplesA.pfm','samplesB.pfm','scene.json','render.log']:
                all_sources[str((p/name).relative_to(out))]=sha(p/name)
    assert len({r['total_spp'] for r in combined})==1
    total_spp=combined[0]['total_spp']
    fig.suptitle(f'Pure GPIS shaderball | Small sphere light | Two independent 512-spp seeds averaged\n'
                 f'64 x 64 | {total_spp} total spp per method | Same filmic mapping; linear averaging')
    fig.savefig(out/'figures/combined_1024.png',dpi=140);plt.close(fig)
    detail.suptitle('Same crop | Linear average before filmic mapping')
    detail.savefig(out/'figures/combined_1024_detail.png',dpi=140);plt.close(detail)
    curve,ax=plt.subplots(figsize=(9,5.3),constrained_layout=True)
    colors=['#b96528','#346db4','#329270']
    for mode,color in zip(MODES,colors):
        preview=next(r for r in data['rows'] if r['stage']=='preview' and r['mode']==mode)
        high=[r for r in data['rows'] if r['stage']=='convergence' and r['mode']==mode]
        pooled=float(np.sqrt(np.mean([r['noise_rms']**2 for r in high])))
        average=next(r for r in combined if r['mode']==mode)
        spp=np.array([preview['spp'],high[0]['spp'],average['total_spp']])
        values=np.array([preview['noise_rms'],pooled,average['noise_rms_from_half_buffers']])
        ax.loglog(spp,values,'o-',c=color,label=mode)
        ax.scatter([r['spp'] for r in high],[r['noise_rms'] for r in high],c=color,marker='x',s=60)
        ax.loglog(spp,preview['noise_rms']*np.sqrt(preview['spp']/spp),c=color,ls=':',alpha=.45)
    ax.set(xlabel='spp at 64 x 64',ylabel='Estimated linear luminance noise RMS',
           title='Observed half-buffer noise | dotted: reference N^(-1/2) from 32 spp\n1024 spp is the two-seed average, not an additional render')
    ax.grid(which='both',alpha=.2);ax.legend()
    curve.savefig(out/'figures/noise_vs_spp.png',dpi=150);plt.close(curve)
    a=combined[0];comparisons=[]
    for b in combined[1:]:
        comparisons.append(dict(test=b['mode'],reference=a['mode'],
            half_noise_ratio_renewal_over_test=a['noise_rms_from_half_buffers']/b['noise_rms_from_half_buffers'],
            independent_noise_ratio_renewal_over_test=a['noise_rms_from_independent_seeds']/b['noise_rms_from_independent_seeds'],
            relative_mean_luminance_difference=b['mean_luminance']/a['mean_luminance']-1))
    discarded=[]
    for r in data['rows']:
        text=(out/r['folder']/'render.log').read_text(encoding='utf-8',errors='replace')
        if 'discarded surface births:' in text:
            import re
            match=re.search(r'discarded surface births:\s*(\{[^\n]+\})',text)
            counters=json.loads(match.group(1));discarded.append(dict(stage=r['stage'],seed=r['seed'],**counters))
            assert counters['paths']==0 and counters['shadow_connections']==0
    save(out/'analysis.json',dict(combined=combined,noise_ratios=comparisons,discarded_surface_births=discarded,
        note='Two seeds and half buffers estimate noise around each model expectation; no common exact reference or established equal-error spp multiplier.'))
    rows=[]
    for r in combined:
        rows.append([r['mode'],f'{r["render_seconds"]:.2f}',f'{r["noise_rms_from_half_buffers"]:.6g}',
            f'{r["noise_rms_from_independent_seeds"]:.6g}',f'{r["mean_luminance"]:.6g}',
            f'<a href="{r["png"]}">PNG</a> · <a href="{r["pfm"]}">PFM</a>'])
    section='''<section id="combined"><h2>64×64：两个独立 512 spp 种子在线性空间合并</h2>
<p>每种方法总计 1024 spp。先平均原始线性辐射，再用同一个 filmic 映射显示；没有降噪或逐方法调曝光。
所有普通平面、地面和环境光均已移除，球形发光体是唯一光源。</p>
<a href="figures/combined_1024.png"><img src="figures/combined_1024.png" alt="两个独立种子的三方法平均图"></a>'''
    section+=table(['方法','两次渲染时间合计 / s','平均图的半样本噪声估计','平均图的独立种子噪声估计','平均亮度','文件'],rows)
    section+='''<details><summary>高光与凹槽放大</summary><img src="figures/combined_1024_detail.png" alt="相同区域放大"></details>
<img src="figures/noise_vs_spp.png" alt="相同分辨率下的 spp 与噪声关系">
<p>曲线仅比较同为 64×64 的数据；128×128 正式图单独展示。虚线为从 32 spp 锚定的 1/√N 参考，1024 点来自两个 512 spp 的平均。
稀有亮点会使有限样本的方差估计波动，不应据此给出确定的等质量 spp 倍数。
模型间平均亮度与细节差异也需要和采样噪声区分，classic_global 的平滑不能直接当作对同一 GPIS 真值的精确程度。</p>
<p><a href="analysis.json">合并数据与独立种子统计</a></p></section>'''
    page=(out/'comparison.html').read_text(encoding='utf-8')
    page=page.replace('来源 shaderball 场景','裁剪前的来源 shaderball 场景')
    marker='<h2>高采样与独立种子复核</h2>'
    page=page.replace(marker,section+marker,1)
    (out/'comparison.html').write_text(page,encoding='utf-8')
    links=0
    for attr in ['href','src']:
        for part in page.split(attr+'='+chr(34))[1:]:
            link=part.split(chr(34))[0]
            if '://' not in link:assert (out/link).exists(),link;links+=1
    for p in (out/'renderer').iterdir():
        if p.is_file():assert sha(p)==manifest['runtime_sha256'][p.name]
    assert sha(Path(experiment.__file__))==manifest['runner_sha256']
    for prior in manifest.get('runner_history',[]):assert sha(out/prior['snapshot'])==prior['sha256']
    verification=read(out/'verification.json')
    verification.update(html_sha256=sha(out/'comparison.html'),local_links=links,
        independent_seed_averages_verified=True,total_spp_per_average=total_spp,
        all_conditional_discarded_surface_counts_zero=True,all_nee_warnings_zero=all(not r['warnings'] for r in data['rows']))
    save(out/'verification.json',verification)
    save(out/'analysis_manifest.json',dict(analysis_runner_sha256=sha(Path(__file__)),inputs=all_sources,
        generated_sha256={str(p.relative_to(out)):sha(p) for folder in [averages,out/'figures'] for p in folder.glob('*') if p.is_file()},
        html_sha256=sha(out/'comparison.html'),analysis_sha256=sha(out/'analysis.json')))
    index=ROOT/'outputs/global_conditional_shaderball_profile_20261113/comparison.html'
    if index.exists():
        text=index.read_text(encoding='utf-8')
        if 'id="shaderball-small-light"' not in text:
            link='<p id="shaderball-small-light"><a href="../gpis_shaderball_small_light_20261114/comparison.html">新增：去掉普通平面的小光源 shaderball，Renewal+／conditional／classic 三方法与独立种子对照</a></p>'
            index.write_text(text.replace('</h1>','</h1>'+link,1),encoding='utf-8')
    print(json.dumps(dict(combined=combined,noise_ratios=comparisons),indent=2))
    print(out/'comparison.html')

if __name__=='__main__':main()
