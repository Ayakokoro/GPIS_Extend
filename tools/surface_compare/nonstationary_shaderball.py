"""Matched classic_global / GPIS global-ensemble nonstationary shaderball renders.

python -B tools/surface_compare/nonstationary_shaderball.py render --stage final
python -B tools/surface_compare/nonstationary_shaderball.py render --stage step_base --size 64 --spp 64 --modes sparse_3d_global
python -B tools/surface_compare/nonstationary_shaderball.py render --stage step_half --size 64 --spp 64 --step .005 --modes sparse_3d_global
<plot-python> -B tools/surface_compare/nonstationary_shaderball.py report
"""
from __future__ import annotations
import argparse
import copy
import html
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from run import ROOT, load, save, sha, pfm

SOURCE = ROOT/'data/example-scenes/classic-global/shaderball-nonstationary.json'
OUT = ROOT/'outputs/shaderball_nonstationary_compare'
MODES = ('classic_global', 'sparse_3d_global')
LABELS = ('classic_global', 'GPIS 3D global ensemble')
Y = np.array([.2126,.7152,.0722])
REGIONS = {'shaderball_box':(.14,.1,.86,.94), 'upper_groove':(.28,.14,.76,.56),
           'lower_body':(.2,.55,.8,.93)}
FAILURE = re.compile(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|'
                     r'Wrong normal direction|Failed to sample gradient|Gradient zero|'
                     r'Clamp one (?:inf|nan) sample', re.I)
FILES = ('scene.json','render.png','render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm','render.log')


def folder(stage,mode):
    return OUT/stage/mode


def scene(args,mode):
    s=load(SOURCE)
    a=s['media'][0]
    if mode=='sparse_3d_global':
        s['media']=[dict(name=a['name'],type='sparse_conv_noise',max_bounces=a['max_bounces'],
            gaussian_process=copy.deepcopy(a['gaussian_process']),
            phase_function=dict(type='brdf',bsdf=copy.deepcopy(a['material'])),
            sigma_a=0,sigma_s=1,density=1,step_size=args.step,min_step=8,seed=17429,
            impulse_density=10,single_realization=False,correlation_context='global',
            intersect_method='gp_discrete',normal_method='conditioned_gaussian',isotropic_3D_sampling=False,
            **{'1D_sampling':False,'1D_sampling_scheme':'uni','1D_gradient_correlationXY':False})]
    s['camera']['resolution']=[args.size,args.size]
    s['integrator'].update(independent_path_sampling=True,wavefront_gpis=False)
    s['renderer'].update(output_directory=str(folder(args.stage,mode)),output_file='render.png',
        hdr_output_file='render.pfm',spp=args.spp,spp_step=min(4,args.spp),
        output_buffers=[dict(type='color',hdr_output_file='samples.pfm',sample_variance=True,two_buffer_variance=True)])
    return s


def render(args):
    exe=ROOT/'build/Release/tungsten.exe'
    provenance=dict(renderer_sha256=sha(exe),source_sha256=sha(SOURCE),seed=args.seed)
    for mode in args.modes:
        d=folder(args.stage,mode); s=scene(args,mode)
        if (d/'result.json').exists():
            r=load(d/'result.json')
            assert r['provenance']==provenance and load(d/'scene.json')==s, 'Existing render differs; use a new stage'
            for name,digest in r['sha256'].items(): assert sha(d/name)==digest
            print('Reuse',d,flush=True)
            continue
        if (d/'render.pfm').exists(): raise RuntimeError('Unvalidated output exists: '+str(d))
        save(d/'scene.json',s)
        save(OUT/'status.json',dict(state='running',stage=args.stage,mode=mode,size=args.size,spp=args.spp,step=args.step))
        print('Render',args.stage,mode,args.size,args.spp,'step',args.step,flush=True)
        started=time.monotonic()
        with (d/'render.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(exe),'-t',str(args.threads),'-s',str(args.seed),'-d',str(d),str(d/'scene.json')],
                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        log=(d/'render.log').read_text(encoding='utf-8',errors='replace')
        if result.returncode or f'Completed {args.spp}/{args.spp} spp' not in log or 'Finished render.' not in log or FAILURE.search(log):
            save(OUT/'status.json',dict(state='failed',stage=args.stage,mode=mode,returncode=result.returncode))
            raise RuntimeError('Render failed: '+str(d/'render.log'))
        rgb=pfm(d/'render.pfm')
        assert rgb.shape==(args.size,args.size,3) and rgb.max()>0
        assert np.allclose(rgb,(pfm(d/'samplesA.pfm')+pfm(d/'samplesB.pfm'))*.5,rtol=3e-6,atol=3e-7)
        pfm(d/'samplesVariance.pfm')
        r=dict(seconds=time.monotonic()-started,resolution=[args.size,args.size],spp=args.spp,
            step=args.step if mode=='sparse_3d_global' else None,provenance=provenance,
            mean_luminance=float((rgb@Y).mean()),zero_length_ray_messages=log.count('Path tracing bounces ended with farT == 0'),
            segment_exit_messages=log.count('Reached end of ray, but not counting it as exited'),
            sha256={name:sha(d/name) for name in FILES})
        save(d/'result.json',r)
        print('Done',mode,round(r['seconds'],2),'seconds, mean Y',round(r['mean_luminance'],6),flush=True)
    save(OUT/'status.json',dict(state='render_complete',stage=args.stage,modes=args.modes))


def read(stage,mode):
    d=folder(stage,mode); r=load(d/'result.json')
    for name,digest in r['sha256'].items(): assert sha(d/name)==digest
    return dict(folder=d,result=r,scene=load(d/'scene.json'),rgb=pfm(d/'render.pfm'),
                halves=[pfm(d/f'samples{c}.pfm') for c in 'AB'])


def masks(n):
    result={'full':np.ones((n,n),dtype=bool)}
    for name,rect in REGIONS.items():
        x0,y0,x1,y1=[int(v*n) for v in rect]
        mask=np.zeros((n,n),dtype=bool);mask[y0:y1,x0:x1]=True;result[name]=mask
    return result


def compare(a,b):
    ya,yb=a['rgb']@Y,b['rgb']@Y
    halves=[(right-left)@Y for left,right in zip(a['halves'],b['halves'])]
    variance=.25*(halves[0]-halves[1])**2
    rows=[]
    for region,mask in masks(ya.shape[0]).items():
        ma,mb=float(ya[mask].mean()),float(yb[mask].mean())
        rows.append(dict(region=region,reference_mean=ma,other_mean=mb,difference=mb-ma,
            relative_difference=mb/ma-1,paired_half_standard_error=float(np.sqrt(variance[mask].sum())/mask.sum())))
    return rows


def report(args):
    from figures import plt, Image, Rectangle
    from plane_figures import blocks
    data=[read(args.stage,m) for m in MODES]
    a,b=data
    assert a['result']['provenance']==b['result']['provenance']
    assert a['result']['resolution']==b['result']['resolution'] and a['result']['spp']==b['result']['spp']
    for key in ('camera','integrator','bsdfs','primitives'): assert a['scene'][key]==b['scene'][key]
    ra,rb=[copy.deepcopy(x['scene']['renderer']) for x in data]
    ra.pop('output_directory');rb.pop('output_directory');assert ra==rb
    ma,mb=[x['scene']['media'][0] for x in data]
    assert ma['gaussian_process']==mb['gaussian_process'] and ma['material']==mb['phase_function']['bsdf']
    assert mb['correlation_context']=='global' and not mb['single_realization'] and not mb['1D_sampling']
    rows=compare(a,b)
    size=a['rgb'].shape[0]; spp=a['result']['spp']
    fig,axes=plt.subplots(2,3,figsize=(13.5,8),gridspec_kw={'height_ratios':[1,.72]})
    rect=REGIONS['upper_groove'];x0,y0,x1,y1=[int(v*size) for v in rect]
    for i,(d,label) in enumerate(zip(data,LABELS)):
        im=np.asarray(Image.open(d['folder']/'render.png').convert('RGB'))
        axes[0,i].imshow(im,interpolation='nearest');axes[0,i].set_title(label,fontsize=13)
        axes[0,i].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=1,edgecolor='#ffcf52'))
        axes[1,i].imshow(im[y0:y1,x0:x1],interpolation='nearest');axes[1,i].set_title('黄色框原图放大')
    delta=(b['rgb']-a['rgb'])@Y
    limit=max(float(np.quantile(np.abs(delta),.98)),1e-5)
    im=axes[0,2].imshow(delta,cmap='RdBu_r',vmin=-limit,vmax=limit,interpolation='nearest')
    axes[0,2].set_title('GPIS − classic_global：线性亮度')
    fig.colorbar(im,ax=axes[0,2],fraction=.046,pad=.03)
    block_delta=blocks(delta[...,None],16)[...,0]
    block_limit=max(float(np.max(np.abs(block_delta))),1e-5)
    im=axes[1,2].imshow(block_delta,cmap='RdBu_r',vmin=-block_limit,vmax=block_limit,interpolation='nearest')
    axes[1,2].set_title('16×16 分区平均亮度差')
    fig.colorbar(im,ax=axes[1,2],fraction=.046,pad=.03)
    for ax in axes.flat: ax.axis('off')
    fig.suptitle(f'非平稳 shaderball：相同 cov / var / ls · {size}×{size} · {spp} spp',fontsize=17)
    fig.text(.025,.02,f'同相机、灯光、绿色镜面材质与 filmic 显示；不降噪、不逐图调曝光。GPIS：3D global ensemble，步长 {mb["step_size"]}。\n'
             '右上色标按 |亮度差| 的 98% 分位截断显示；差值含采样噪声。右下先在线性亮度中求分区平均。',fontsize=9)
    fig.subplots_adjust(left=.025,right=.95,top=.89,bottom=.11,hspace=.16,wspace=.12)
    fig.savefig(OUT/'comparison.png',dpi=150,bbox_inches='tight',pad_inches=.15);plt.close(fig)
    if args.preview:
        print(OUT/'comparison.png',flush=True)
        return
    base,half=[read(stage,'sparse_3d_global') for stage in ('step_base','step_half')]
    assert base['result']['provenance']==half['result']['provenance']
    assert base['result']['resolution']==half['result']['resolution'] and base['result']['spp']==half['result']['spp']
    sa,sb=[copy.deepcopy(d['scene']) for d in (base,half)]
    for s in (sa,sb):
        s['renderer'].pop('output_directory');s['media'][0].pop('step_size')
    assert sa==sb
    steps=compare(base,half)
    step_ball=next(r for r in steps if r['region']=='shaderball_box')
    step_groove=next(r for r in steps if r['region']=='upper_groove')
    validation=dict(resolution=[size,size],spp=spp,settings_match=True,
        comparison=rows,step_check=dict(resolution=base['result']['resolution'],spp=base['result']['spp'],
            base_step=base['result']['step'],half_step=half['result']['step'],statistics=steps),
        sources=[dict(mode=m,folder=str(d['folder']),result=d['result']) for m,d in zip(MODES,data)],
        limitations=[f'{spp} spp is an appearance comparison, not a converged reference.',
            'Paired half-sample errors describe sampling noise, not discretization or model bias.',
            'GPIS uses finite-density sparse convolution and multiResolutionGrid; matching configuration does not make its field exactly the analytic kernel.',
            'Current GPIS gradient evaluation freezes amplitude and scale-blend derivatives; this comparison includes that implementation difference.',
            'Zero-length ray diagnostics, if present, are counted in each result.json.'])
    save(OUT/'validation.json',validation)
    def table(entries):
        return '<table><tr><th>区域</th><th>参考平均 Y</th><th>对照平均 Y</th><th>相对差</th><th>配对半样本 SE</th></tr>'+''.join(
            f'<tr><td>{html.escape(r["region"])}</td><td>{r["reference_mean"]:.6f}</td><td>{r["other_mean"]:.6f}</td><td>{100*r["relative_difference"]:+.2f}%</td><td>{r["paired_half_standard_error"]:.6f}</td></tr>' for r in entries)+'</table>'
    cards=''.join(f'<article><h3>{label}</h3><a href="{args.stage}/{mode}/render.png"><img src="{args.stage}/{mode}/render.png"></a>'
        f'<p>{d["result"]["seconds"]:.1f} 秒 · farT=0 诊断 {d["result"]["zero_length_ray_messages"]} 次</p>'
        f'<a href="{args.stage}/{mode}/scene.json">配置</a> · <a href="{args.stage}/{mode}/render.pfm">线性 PFM</a> · '
        f'<a href="{args.stage}/{mode}/render.log">日志</a></article>' for mode,label,d in zip(MODES,LABELS,data))
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>非平稳 shaderball：classic_global 与 GPIS</title><style>
body{{font:16px/1.65 system-ui;max-width:1250px;margin:32px auto;padding:0 24px;color:#253247}}a{{color:#176ba0}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:24px}}img{{max-width:100%;height:auto}}article img{{width:100%;image-rendering:auto}}
table{{border-collapse:collapse}}th,td{{padding:8px 16px;border-bottom:1px solid #dce2e8;text-align:right}}th:first-child,td:first-child{{text-align:left}}
@media(max-width:700px){{.pair{{grid-template-columns:1fr}}}}</style>
<h1>非平稳 shaderball：classic_global 与 GPIS</h1>
<p>{size}×{size}、{spp} spp；相同几何、材质、灯光、相机、cov/var/ls。GPIS 使用 3D global ensemble，跨路径平均，每条路径内保留同一随机场。
var: 0.05→0.0016，ls: 0.1→0.032，世界 y: 0→0.6，与 statue 第一组 GP 一致。统一 filmic 显示，无降噪和逐图曝光调整。</p>
<div class="pair">{cards}</div><h2>图像和亮度差</h2><a href="comparison.png"><img src="comparison.png"></a>
<p>下面的相对差为 GPIS / classic_global − 1；在线性 HDR 亮度中统计。</p>{table(rows)}
<h2>GPIS 步长检查</h2><p>{base['result']['resolution'][0]}×{base['result']['resolution'][1]}、{base['result']['spp']} spp；
参考步长 {base['result']['step']}，对照步长 {half['result']['step']}。配对半样本 SE 只描述采样噪声，不证明数值收敛。</p>{table(steps)}
<p>步长减半后，球体框选区域的平均亮度变化 {100*step_ball['relative_difference']:+.2f}%，上方凹槽区域变化 {100*step_groove['relative_difference']:+.2f}%。
这组检查只覆盖两个步长；主对比采用原步长，不能视为已经消除步长误差的 GPIS 真值。</p>
<p><a href="step_base/sparse_3d_global/render.png">原步长图</a> · <a href="step_half/sparse_3d_global/render.png">半步长图</a></p>
<h2>解释范围</h2><p>本次是有限 spp 的外观比较。GPIS 的有限脉冲密度、多分辨率混合与离散步进都有近似；其现有梯度求值冻结了振幅和尺度混合权重的导数。
classic_global 则使用解析核的局部条件统计。因此图像差异包含历史相关性和这些实现差异，不能全部归因于 global_pointwise 假设。</p>
<p><a href="validation.json">参数、统计及哈希记录</a> · <a href="status.json">状态</a></p></html>'''
    (OUT/'comparison.html').write_text(body,encoding='utf-8')
    save(OUT/'status.json',dict(state='complete',stage=args.stage,resolution=[size,size],spp=spp,report='comparison.html'))
    print(OUT/'comparison.html',flush=True)
    for r in rows: print(r,flush=True)
    print('Step sensitivity:',steps,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('render','report'));p.add_argument('--stage',default='final')
    p.add_argument('--preview',action='store_true',help='Write the main comparison figure while step checks are still running')
    p.add_argument('--size',type=int,default=256);p.add_argument('--spp',type=int,default=32)
    p.add_argument('--step',type=float,default=.01);p.add_argument('--threads',type=int,default=32)
    p.add_argument('--seed',type=int,default=20261025);p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=p.parse_args()
    if min(args.size,args.spp,args.threads)<=0 or args.size%16 or args.spp%2 or not np.isfinite(args.step) or args.step<=0:
        p.error('size must be a positive multiple of 16, spp positive and even, step finite and positive')
    if args.command=='render':render(args)
    else:report(args)
