"""Build the A/B shaderball report using the installed FLIP executable.

<plot-python> -B tools/surface_compare/gpis_shaderball_flip.py

Uses the exact, hash-verified 128 spp PNGs from ell0p2_neural_comparison.
FLIP computes the errors; this script arranges original PNGs and native heatmaps.
The main comparison page and its manifest contain only the A/B parameter groups.
"""
from __future__ import annotations

import argparse
import copy
import csv
import html
import re
import subprocess
import sys
from pathlib import Path

from figures import plt, Image, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle
import numpy as np
from run import ROOT, load, save, sha

REPORT=ROOT/'outputs/gpis_shaderball_compare/ell0p2_neural_comparison'
DEFAULT_FLIP=Path('E:/Research/Result/Test/flip.exe')
MODES=('classic_local','classic_global','global_conditional','neural')
REFERENCE='sparse_3d_global'
FIELDS={'Mean':'mean','Weighted median':'weighted_median',
    '1st weighted quartile':'weighted_q1','3rd weighted quartile':'weighted_q3',
    'Min':'minimum','Max':'maximum'}
BEGIN='<!-- BEGIN SHADERBALL FLIP -->'
END='<!-- END SHADERBALL FLIP -->'


def evaluate(exe, reference, test, dest, ppd):
    dest.mkdir(parents=True,exist_ok=True)
    command=[str(exe),'-r',str(reference),'-t',str(test),'-ppd',str(ppd),'-d',str(dest),'-txt']
    identity=dict(command=command,executable_sha256=sha(exe),
        reference_sha256=sha(reference),test_sha256=sha(test))
    if (dest/'result.json').exists():
        result=load(dest/'result.json')
        assert result['identity']==identity, 'Existing FLIP result uses different inputs or settings'
        for name,digest in result['sha256'].items():
            assert sha(dest/name)==digest
        return result
    completed=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace')
    (dest/'flip.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    assert completed.returncode==0 and 'Invoking LDR-FLIP' in completed.stdout, str(dest/'flip.log')
    # Avoid --basename: FLIP 1.7 also substitutes it in printed input names.
    maps=list(dest.glob('flip.*.ldr.png'))
    pooled=list(dest.glob('pooled_values.flip.*.ldr.txt'))
    assert len(maps)==len(pooled)==1
    statistics={}
    for line in pooled[0].read_text(encoding='utf-8').splitlines():
        if ':' in line:
            key,value=line.split(':',1)
            if key in FIELDS: statistics[FIELDS[key]]=float(value)
    assert set(statistics)==set(FIELDS.values())
    assert all(np.isfinite(v) and 0<=v<=1 for v in statistics.values())
    with Image.open(maps[0]) as im, Image.open(reference) as ref:
        assert im.size==ref.size
    result=dict(identity=identity,statistics=statistics,error_map=maps[0].name,
        pooled_values=pooled[0].name,
        sha256={p.name:sha(p) for p in (maps[0],pooled[0],dest/'flip.log')})
    save(dest/'result.json',result)
    return result


def rgb(path):
    with Image.open(path) as im:
        return np.asarray(im.convert('RGB'))


def plot(groups,dest,roi,ppd):
    figures=[]
    scalar=ScalarMappable(norm=Normalize(0,1),cmap='magma')
    for detail in (False,True):
        fig,axes=plt.subplots(2,4,figsize=(15.5,8.8),squeeze=False)
        for i,g in enumerate(groups):
            for j,mode in enumerate(MODES):
                row=g['results'][mode]
                error=rgb(dest/row['map'])
                if detail:
                    x0,y0,x1,y1=[int(v*error.shape[0]) for v in roi]
                    error=error[y0:y1,x0:x1]
                ax=axes[i,j]
                ax.imshow(error,interpolation='nearest'); ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{mode}\n全图 mean FLIP = {row["statistics"]["mean"]:.6f}',fontsize=11)
                if j==0: ax.set_ylabel(f'{g["key"]} 组\nℓ={g["ell"]:g} · σ={g["sigma"]:g}',fontsize=12)
        suffix='凹槽局部放大' if detail else '全图误差'
        fig.suptitle(f'Shaderball · LDR-FLIP · {suffix}\n参考：同参数组 GPIS global ensemble · FLIP v1.7 · {ppd:g} PPD',fontsize=16)
        fig.subplots_adjust(left=.07,right=.91,top=.84,bottom=.105,wspace=.08,hspace=.28)
        fig.colorbar(scalar,cax=fig.add_axes([.935,.20,.012,.56]),label='FLIP error · 0–1')
        note='局部直接裁自全图误差图；未在裁剪图上重新计算 FLIP。' if detail else '原始 256×256 PNG；各图使用相同的 0–1 magma 色标。'
        fig.text(.07,.035,note+'\n数值越小越接近参考；差异包含两张图的采样噪声，GPIS 128 spp 不是已收敛真值。',fontsize=10)
        name='error_detail.png' if detail else 'error_overview.png'
        fig.savefig(dest/name,dpi=150); plt.close(fig); figures.append(name)
    for g in groups:
        fig,axes=plt.subplots(2,5,figsize=(17,7.6),squeeze=False)
        for j,mode in enumerate((REFERENCE,)+MODES):
            axes[0,j].imshow(rgb(dest/g['inputs'][mode]['copy']),interpolation='nearest')
            name='GPIS global ensemble\nreference' if mode==REFERENCE else mode
            axes[0,j].set_title(name,fontsize=12)
            r=g['self_check'] if mode==REFERENCE else g['results'][mode]
            axes[1,j].imshow(rgb(dest/r['map']),interpolation='nearest')
            axes[1,j].set_title(f'mean FLIP = {r["statistics"]["mean"]:.6f}',fontsize=11)
            for ax in axes[:,j]: ax.set_xticks([]); ax.set_yticks([])
        axes[0,0].set_ylabel('原始渲染 PNG',fontsize=12)
        axes[1,0].set_ylabel('LDR-FLIP',fontsize=12)
        fig.suptitle(f'{g["key"]} 组 · ℓ={g["ell"]:g}，σ={g["sigma"]:g} · 原图与误差\n256×256 · 128 spp · {ppd:g} PPD · 统一色标 0–1',fontsize=16)
        fig.subplots_adjust(left=.04,right=.93,top=.83,bottom=.10,wspace=.06,hspace=.25)
        fig.colorbar(scalar,cax=fig.add_axes([.95,.17,.01,.29]),label='FLIP error')
        fig.text(.04,.025,'直接使用本地 FLIP v1.7 输出的 magma 误差图；原图不降噪、不调曝光、不缩放后再计算。',fontsize=10)
        name=f'{g["key"]}_comparison.png'
        fig.savefig(dest/name,dpi=150); plt.close(fig); figures.append(name)
    return figures


def plot_appearance(groups,dest,roi):
    names=[]
    for detail in (False,True):
        fig,axes=plt.subplots(2,5,figsize=(18,8.8),squeeze=False)
        for i,g in enumerate(groups):
            for j,mode in enumerate(MODES+(REFERENCE,)):
                image=rgb(dest/'flip'/g['inputs'][mode]['copy'])
                h,w=image.shape[:2]
                x0,y0,x1,y1=[int(v*n) for v,n in zip(roi,(w,h,w,h))]
                ax=axes[i,j]
                if detail:
                    image=image[y0:y1,x0:x1]
                ax.imshow(image,interpolation='nearest')
                if not detail:
                    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=.9,edgecolor='#ffcf52'))
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title('GPIS global ensemble\nreference' if mode==REFERENCE else mode,fontsize=11)
                if j==0: ax.set_ylabel(f'{g["key"]} 组\nℓ={g["ell"]:g} · σ={g["sigma"]:g}',fontsize=12)
        title='凹槽与球身局部放大' if detail else '五模式渲染结果'
        fig.suptitle(f'Shaderball · A/B 两组 · {title}\n同相机、灯光与材质 · 256×256 · 128 spp',fontsize=17)
        fig.subplots_adjust(left=.065,right=.99,top=.84,bottom=.10,wspace=.08,hspace=.22)
        fig.text(.065,.035,'统一 filmic 显示；原始 PNG，不降噪、不逐图调曝光。'+
            ('局部对应全图中的黄色矩形。' if detail else '黄色矩形标出局部放大范围。'),fontsize=10)
        name='detail.png' if detail else 'appearance.png'
        fig.savefig(dest/name,dpi=150); plt.close(fig); names.append(name)
    return names


def section(groups,ppd,prefix):
    rows=''
    for g in groups:
        for mode in MODES:
            r=g['results'][mode]; s=r['statistics']
            rows+=(f'<tr><td>{g["key"]} · ℓ={g["ell"]:g}, σ={g["sigma"]:g}</td><td>{mode}</td>'
                f'<td>{s["mean"]:.6f}</td><td>{s["weighted_median"]:.6f}</td><td>{s["maximum"]:.6f}</td>'
                f'<td><a href="{prefix}{r["map"]}">原始误差图</a> · <a href="{prefix}{r["pooled"]}">FLIP 原生统计</a></td></tr>')
    pairs=''.join(f'<details><summary>{g["key"]} 组：原图与 FLIP 并排查看</summary><a href="{prefix}{g["key"]}_comparison.png"><img loading="lazy" src="{prefix}{g["key"]}_comparison.png" alt="{g["key"]} 组原图与误差"></a></details>' for g in groups)
    return f'''<section id="flip"><h2>A/B 两组：相对 GPIS global ensemble 的 FLIP 误差</h2>
<p>使用本地 <b>FLIP v1.7</b>，在页面对应的原始 256×256、128 spp PNG 上计算 <b>LDR-FLIP</b>；固定 <b>{ppd:g} pixels per degree</b>。每组以自己的 GPIS global ensemble 为参考，比较 classic_local、classic_global、global_conditional 和 neural。</p>
<p>输入为原报告使用的 filmic 显示 PNG，结果衡量该显示条件下的感知差异。所有误差图统一使用 <b>0–1 magma 色标</b>，数值越小越接近参考。采样噪声也计入误差，128 spp 的 GPIS 参考未视为已收敛真值。</p>
<p><a href="{prefix}comparison.html">独立 FLIP 页面</a> · <a href="{prefix}metrics.csv">全图统计 CSV</a> · <a href="{prefix}manifest.json">工具版本、参数与输入哈希</a></p>
<a href="{prefix}error_overview.png"><img loading="lazy" src="{prefix}error_overview.png" alt="A B 两组的八张 FLIP 误差图"></a>
<h3 id="flip-detail">局部放大的 FLIP 误差</h3><p>放大范围与原图的黄色矩形一致。直接裁剪已计算的全图误差图，保留完整图像参与空间滤波时的上下文；标题中的 mean 为全图统计，色标保持 0–1。</p>
<a href="{prefix}error_detail.png"><img loading="lazy" src="{prefix}error_detail.png" alt="A B 两组凹槽局部 FLIP 放大"></a>
<h3 id="flip-statistics">全图 FLIP 统计</h3>
<div class="table"><table><tr><th>参数组</th><th>模型</th><th>Mean FLIP</th><th>Weighted median</th><th>Max</th><th>原始文件</th></tr>{rows}</table></div>
{pairs}</section>'''


def write_ab_manifest(report,original,appearance_names,flip_names):
    selected=copy.deepcopy(original)
    parameters={(.2,.02),(.2,.04)}
    if any((g['ell'],g['sigma']) not in parameters for g in original['groups']):
        archive=report/'history/manifest_with_baseline.json'
        if not archive.exists():
            archive.parent.mkdir(parents=True,exist_ok=True)
            archive.write_bytes((report/'manifest.json').read_bytes())
        selected['previous_manifest']=dict(path=archive.relative_to(report).as_posix(),sha256=sha(archive))
    selected['groups']=[g for g in selected['groups'] if (g['ell'],g['sigma']) in parameters]
    assert len(selected['groups'])==2
    for key,g in zip(('A','B'),selected['groups']):
        g['name']=f'{key}：ℓ={g["ell"]:g}，σ={g["sigma"]:g}'
    for key,filename in (('statistics','metrics.csv'),('pair_metrics','pair_metrics.csv')):
        selected[key]=[r for r in selected[key] if (r['ell'],r['sigma']) in parameters]
        with (report/filename).open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(selected[key][0])); writer.writeheader(); writer.writerows(selected[key])
    selected.pop('reference_manifest',None)
    selected.pop('reference_manifest_sha256',None)
    selected.update(complete=True,pending_groups=[],displayed_groups=['A','B'],
        figures=appearance_names+['flip/'+name for name in flip_names],
        error_metric='LDR-FLIP',error_reference='Same-group GPIS global ensemble',
        error_manifest='flip/manifest.json',error_metrics='flip/metrics.csv',
        generator='tools/surface_compare/gpis_shaderball_flip.py',
        notes=['A: ell=.2, sigma=.02; B: ell=.2, sigma=.04. Only these two groups are displayed.',
            'Original 256x256, 128 spp filmic PNGs are verified against their recorded hashes.',
            'Displayed errors are native FLIP maps against the GPIS global ensemble of the same group.',
            'Error-detail panels are crops of the full-image error maps; all use the same 0-1 scale.',
            'The GPIS reference has sampling noise and approximation error; FLIP is not a convergence proof.',
            'The original linear-light statistics for A/B remain in metrics.csv and pair_metrics.csv.'])
    selected['figure_sha256']={name:sha(report/name) for name in selected['figures']}
    save(report/'manifest.json',selected)


def write_main_page(report,groups,ppd):
    sources=''
    for g in groups:
        for mode in MODES+(REFERENCE,):
            source=Path(g['inputs'][mode]['source'])
            relative='../'+source.parent.relative_to(report.parent).as_posix()
            sources+=(f'<tr><td>{g["key"]}</td><td>{html.escape(mode)}</td>'
                f'<td><a href="{relative}/render.png">PNG</a> · <a href="{relative}/render.pfm">线性 PFM</a> · '
                f'<a href="{relative}/scene.json">场景参数</a></td></tr>')
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaderball · A/B 两组与 FLIP 误差</title>
<style>body{{max-width:1800px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#253145;background:#f5f6f8}}h1{{font-size:28px}}h2{{font-size:23px;margin-top:36px}}h3{{font-size:21px;margin-top:28px}}img{{width:100%;height:auto;background:white}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.note{{background:#e7edf5;padding:16px;border-radius:10px}}details{{margin-top:25px}}summary{{cursor:pointer;font-weight:600}}.table{{overflow:auto}}nav a{{display:inline-block;margin-right:18px}}</style>
<h1>Shaderball：A/B 两组结果与 FLIP 误差</h1>
<div class="note"><b>A：ℓ=0.2，σ=0.02，σ/ℓ=0.1</b><br><b>B：ℓ=0.2，σ=0.04，σ/ℓ=0.2</b><br>每组比较 classic_local、classic_global、global_conditional、neural 与 GPIS global ensemble。GPIS global ensemble 是各组 FLIP 的参考图。</div>
<p>同相机、光源、地面与绿色镜面材质；256×256、128 spp，NEE/MIS 开启。两组 neural 使用同一份已有模型权重。原图采用统一 filmic 显示。</p>
<nav><a href="#appearance">渲染结果</a><a href="#appearance-detail">原图放大</a><a href="#flip">FLIP 全图误差</a><a href="#flip-detail">FLIP 放大误差</a><a href="#flip-statistics">FLIP 统计</a><a href="manifest.json">参数与来源</a></nav>
<h2 id="appearance">A/B 两组渲染结果</h2><a href="appearance.png"><img src="appearance.png" alt="A B 两组的五模式渲染结果"></a>
<h2 id="appearance-detail">凹槽与球身局部放大</h2><a href="detail.png"><img loading="lazy" src="detail.png" alt="A B 两组原图局部放大"></a>
{BEGIN}
{section(groups,ppd,'flip/')}
{END}
<details><summary>原始图像、场景与线性统计</summary><div class="table"><table><tr><th>参数组</th><th>模型</th><th>文件</th></tr>{sources}</table></div>
<p><a href="metrics.csv">A/B 区域亮度统计</a> · <a href="pair_metrics.csv">A/B 线性亮度差统计</a> · <a href="flip/metrics.csv">A/B FLIP 误差统计</a></p></details>
</html>'''
    (report/'comparison.html').write_text(page,encoding='utf-8')


def verify_links(page):
    from html.parser import HTMLParser
    from urllib.parse import urlsplit,unquote
    links=[]
    class Parser(HTMLParser):
        def handle_starttag(self,tag,attrs):
            links.extend(v for k,v in attrs if k in ('href','src') and v)
    Parser().feed(page.read_text(encoding='utf-8'))
    count=0
    for value in links:
        link=urlsplit(value)
        if not link.scheme and link.path:
            assert (page.parent/unquote(link.path)).is_file(),value
            count+=1
    return count


def main(report,exe,ppd):
    assert exe.is_file(),exe
    assert np.isfinite(ppd) and ppd>0
    dest=report/'flip'; dest.mkdir(parents=True,exist_ok=True)
    help_result=subprocess.run([str(exe),'--help'],cwd=ROOT,capture_output=True,text=True,check=True)
    assert 'FLIP v1.7.' in help_result.stdout, 'Review usage before using another local FLIP version'
    (dest/'tool_help.txt').write_text(help_result.stdout,encoding='utf-8')
    original=load(report/'manifest.json')
    groups=[]; metrics=[]
    for key,sigma in (('A',.02),('B',.04)):
        g=next(g for g in original['groups'] if (g['ell'],g['sigma'])==(.2,sigma))
        group=dict(key=key,ell=g['ell'],sigma=g['sigma'],inputs={},results={})
        for mode in (REFERENCE,)+MODES:
            source=next(s for s in g['sources'] if s['mode']==mode)
            directory=Path(source['folder']); png=directory/'render.png'
            assert sha(png)==source['result']['sha256']['render.png']
            assert sha(directory/'scene.json')==source['result']['sha256']['scene.json']
            assert source['result']['resolution']==[256,256] and source['result']['spp']==128
            scene=load(directory/'scene.json')
            if mode==REFERENCE:
                medium=scene['media'][0]
                assert medium['correlation_context']=='global' and not medium['single_realization']
                assert not medium['1D_sampling']
            with Image.open(png) as im: assert im.size==(256,256)
            copy=dest/'inputs'/key/(mode+'.png'); copy.parent.mkdir(parents=True,exist_ok=True)
            if copy.exists(): assert sha(copy)==sha(png)
            else: copy.write_bytes(png.read_bytes())
            group['inputs'][mode]=dict(source=str(png),sha256=sha(png),
                scene_sha256=sha(directory/'scene.json'),copy=copy.relative_to(dest).as_posix())
        reference=dest/group['inputs'][REFERENCE]['copy']
        for mode in (REFERENCE,)+MODES:
            folder=dest/('reference_checks' if mode==REFERENCE else 'evaluations')/key/mode
            r=evaluate(exe,reference,dest/group['inputs'][mode]['copy'],folder,ppd)
            item=dict(statistics=r['statistics'],map=(folder/r['error_map']).relative_to(dest).as_posix(),
                pooled=(folder/r['pooled_values']).relative_to(dest).as_posix(),
                verification=r)
            if mode==REFERENCE:
                assert all(value==0 for value in r['statistics'].values())
                group['self_check']=item
            else:
                group['results'][mode]=item
                metrics.append(dict(group=key,ell=.2,sigma=sigma,mode=mode,reference=REFERENCE,
                    metric='LDR-FLIP',pixels_per_degree=ppd,resolution='256x256',spp=128,
                    **r['statistics'],error_map=item['map']))
                print(key,mode,'mean FLIP',r['statistics']['mean'],flush=True)
        groups.append(group)
    figure_names=plot(groups,dest,original['detail_roi'],ppd)
    appearance_names=plot_appearance(groups,report,original['detail_roi'])
    write_ab_manifest(report,original,appearance_names,figure_names)
    with (dest/'metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(metrics[0])); writer.writeheader(); writer.writerows(metrics)
    manifest=dict(complete=True,metric='LDR-FLIP',groups=groups,pixels_per_degree=ppd,
        tool=dict(path=str(exe),version='1.7',sha256=sha(exe),help_file='tool_help.txt'),
        source_manifest=str(report/'manifest.json'),source_manifest_sha256=sha(report/'manifest.json'),
        resolution=[256,256],spp=128,reference_mode=REFERENCE,test_modes=list(MODES),
        input='Exact original filmic PNG bytes; no resampling, denoising or exposure adjustment',
        detail_roi=original['detail_roi'],detail_method='Crop the full-image FLIP heatmap after evaluation',
        pooling='Native FLIP full-image pooled values (six decimal places); weighted median is not an ordinary median',
        color_scale=[0,1],colormap='Native FLIP magma',figures=figure_names,
        figure_sha256={name:sha(dest/name) for name in figure_names},metrics=metrics,
        script_sha256=sha(Path(__file__)),python_executable=sys.executable,
        reproduction=f'"{sys.executable}" -B tools/surface_compare/gpis_shaderball_flip.py --flip-exe "{exe}" --ppd {ppd:g}')
    save(dest/'manifest.json',manifest)
    standalone=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaderball A/B · FLIP 误差</title><style>body{{max-width:1700px;margin:30px auto;padding:0 24px;background:#f5f6f8;color:#253145;font:16px/1.65 system-ui}}img{{width:100%;height:auto}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.table{{overflow:auto}}details{{margin:20px 0}}summary{{cursor:pointer;font-weight:600}}</style>
<h1>Shaderball · A/B 两组 FLIP 误差</h1><p><a href="../comparison.html#flip">返回原比较页面</a></p>
{section(groups,ppd,'')}</html>'''
    (dest/'comparison.html').write_text(standalone,encoding='utf-8')
    write_main_page(report,groups,ppd)
    page=report/'comparison.html'
    checked=verify_links(dest/'comparison.html')
    main_checked=verify_links(page)
    assert '基准' not in page.read_text(encoding='utf-8')
    for name in figure_names:
        with Image.open(dest/name) as im: im.verify()
    assert len(metrics)==8 and len(groups)==2
    save(dest/'verification.json',dict(passed=True,input_images=10,comparisons=8,
        reference_self_checks=2,local_links_checked=checked,main_page_links_checked=main_checked,
        displayed_groups=['A','B'],source_png_and_scene_hashes_verified=True,
        source_manifest_hash_matches=sha(report/'manifest.json')==manifest['source_manifest_sha256']))
    print('Saved',dest/'comparison.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--flip-exe',type=Path,default=DEFAULT_FLIP)
    parser.add_argument('--report',type=Path,default=REPORT)
    parser.add_argument('--ppd',type=float,default=67.)
    args=parser.parse_args()
    main(args.report.resolve(),args.flip_exe.resolve(),args.ppd)
