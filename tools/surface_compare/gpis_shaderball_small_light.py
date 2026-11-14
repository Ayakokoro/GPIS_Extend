"""Three matched shaderball methods with one small sphere emitter and no planes.

python -B tools/surface_compare/gpis_shaderball_small_light.py preview
python -B tools/surface_compare/gpis_shaderball_small_light.py final
python -B tools/surface_compare/gpis_shaderball_small_light.py convergence
python -B tools/surface_compare/gpis_shaderball_small_light.py report
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
if (ROOT/'build/ray_curve_python').exists():
    sys.path.insert(0,str(ROOT/'build/ray_curve_python'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/ray_curve_matplotlib'))
import numpy as np
from PIL import Image

OUT=ROOT/'outputs/gpis_shaderball_small_light_20261114'
SOURCE=ROOT/'outputs/global_conditional_shaderball_profile_20261113/simple_majorant_20261114/sources/ell0p2.json'
MODES=('gpis_renewal_plus','global_conditional','classic_global')
QUALITY=dict(preview=(64,32),final=(128,128),convergence=(64,512))
Y=np.array([.2126,.7152,.0722])
LIGHT=dict(position=[-3.,3.,4.],radius=.06,radiance=2000.)
FATAL=re.compile(r'Caught an internal error|unrecoverable|Unable to load|Gradient invalid|'
    r'Wrong normal direction|Failed to sample gradient|Gradient zero|Conditioning .* error|'
    r'GPContext does not exist|Clamp one (?:inf|nan) sample|invalid FP|terminate called|nonfinite',re.I)

def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,data):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    tmp.replace(p)

def pfm(path):
    with path.open('rb') as f:
        assert f.readline().strip()==b'PF'
        w,h=map(int,f.readline().split());scale=float(f.readline())
        pixels=np.frombuffer(f.read(),dtype='<f4' if scale<0 else '>f4').reshape(h,w,3)
    pixels=np.flipud(pixels).astype(float)*abs(scale)
    assert np.isfinite(pixels).all() and pixels.min()>=0,path
    return pixels

def scene(out,stage,mode,seed,step=.0025):
    base=read(out/'sources/shaderball.json');source=base['media'][0]
    gp=copy.deepcopy(source['gaussian_process']);material=copy.deepcopy(source['material'])
    if mode=='gpis_renewal_plus':
        medium=dict(name='gp',type='sparse_conv_noise',gaussian_process=gp,
            phase_function=dict(type='brdf',bsdf=material),max_bounces=1024,
            sigma_a=0,sigma_s=1,density=1,step_size=step,min_step=8,seed=17429,
            impulse_density=10,single_realization=False,correlation_context='renewal+',
            intersect_method='gp_discrete',normal_method='conditioned_gaussian',
            isotropic_3D_sampling=False,surf_vol_phase_separate=False,
            **{'1D_sampling':True,'1D_sampling_scheme':'mis','1D_gradient_correlationXY':True})
    else:
        medium=dict(name='gp',type=mode,gaussian_process=gp,material=material,
            max_bounces=1024,sdf_lipschitz=1.1,gradient_bound=2.,tracking_step=0.)
        if mode=='global_conditional':
            medium.update(non_gp_surface_policy='retain_endpoint',fp_gradient=dict(enabled=False),
                tracking_fixed_step_limit=False,tracking_age_scale=.5,tracking_geometry_step=True,
                tracking_point_bounds=True,tracking_classic_start_scale=0.,
                tracking_simple_bound_start_scale=2.,tracking_max_expected_events=64.)
    box=copy.deepcopy(next(p for p in base['primitives'] if p.get('name')=='processBox'))
    box['bsdf']=dict(type='forward',albedo=1)
    light=dict(name='small_sphere_key',type='sphere',emission=LIGHT['radiance'],
        bsdf=dict(type='null',albedo=1),transform=dict(position=LIGHT['position'],scale=LIGHT['radius']))
    size,spp=QUALITY[stage]
    base['media']=[medium];base['bsdfs']=[];base['primitives']=[box,light]
    base['camera']['resolution']=[size,size]
    base['integrator'].update(max_bounces=32,use_cuda_gpis=False,independent_path_sampling=True,
        wavefront_gpis=False,enable_light_sampling=True,enable_volume_light_sampling=True)
    base['renderer']=dict(output_directory=str(out/stage/mode/f'seed{seed}'),
        output_file='render.png',hdr_output_file='render.pfm',spp=spp,spp_step=8,
        overwrite_output_files=True,adaptive_sampling=False,stratified_sampler=False,
        scene_bvh=True,enable_resume_render=stage!='preview',resume_render_file='resume.dat',
        checkpoint_interval='300' if stage!='preview' else '0',timeout='0',
        output_buffers=[dict(type='color',hdr_output_file='samples.pfm',
                             sample_variance=True,two_buffer_variance=True)])
    return base

def prepare(args):
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    manifest=out/'manifest.json'
    if manifest.exists():
        m=read(manifest)
        assert m['runner_sha256']==sha(Path(__file__)),'Runner changed; use a new output directory'
        assert m['threads']==args.threads
        for name,digest in m['runtime_sha256'].items():assert sha(out/'renderer'/name)==digest,name
        assert sha(out/'sources/shaderball.json')==m['source_sha256']
        return m
    (out/'sources').mkdir(exist_ok=True);shutil.copy2(SOURCE,out/'sources/shaderball.json')
    runtime=out/'renderer';runtime.mkdir(exist_ok=True)
    for path in [ROOT/'build/Release/tungsten.exe',*sorted((ROOT/'build/Release').glob('*.dll'))]:
        shutil.copy2(path,runtime/path.name)
    m=dict(date='2026-11-14',runner_sha256=sha(Path(__file__)),source=str(SOURCE),
        source_sha256=sha(out/'sources/shaderball.json'),threads=args.threads,
        runtime_sha256={p.name:sha(p) for p in runtime.iterdir() if p.is_file()},
        quality={k:list(v) for k,v in QUALITY.items()},modes=list(MODES),light=LIGHT,
        covariance=read(SOURCE)['media'][0]['gaussian_process']['covariance'],
        notes=['Only procedural GPIS shaderball, transparent process boundary and a small sphere emitter.',
               'No floor, ordinary reflecting geometry, quad emitter or environment light.',
               'Same GP, material, camera, light and integrator across methods.',
               'Conditional uses the tested point/geometric/simple-Q2 proposal settings; target law unchanged.',
               'Renewal+ uses 1D MIS with transverse gradient correlation and finite step .0025.',
               'Noise comparisons concern each model expectation; none is an exact common reference.'])
    save(manifest,m);return m

def validate_scenes(scenes):
    base=scenes[0]
    for s in scenes:
        for key in ['camera','primitives','bsdfs','integrator']:assert s[key]==base[key],key
        m=s['media'][0];b=base['media'][0]
        assert m['gaussian_process']==b['gaussian_process']
        material=m['material'] if 'material' in m else m['phase_function']['bsdf']
        expected=b['material'] if 'material' in b else b['phase_function']['bsdf']
        assert material==expected
        assert [p['type'] for p in s['primitives']]==['cube','sphere']
        assert s['primitives'][0]['bsdf']['type']=='forward'
        assert s['primitives'][1]['bsdf']['type']=='null'
        assert 'emission' in s['primitives'][1]
        a,c=copy.deepcopy(s['renderer']),copy.deepcopy(base['renderer'])
        a.pop('output_directory');c.pop('output_directory');assert a==c

def render(args):
    out=args.output.resolve();m=prepare(args)
    for seed in args.seeds:
        scenes=[scene(out,args.stage,mode,seed,args.step) for mode in MODES]
        validate_scenes(scenes)
        for mode in args.modes:
            value=scenes[MODES.index(mode)];folder=Path(value['renderer']['output_directory'])
            identity=dict(scene=value,seed=seed,renderer_sha256=m['runtime_sha256']['tungsten.exe'])
            if (folder/'result.json').exists():
                result=read(folder/'result.json');assert result['identity']==identity
                for name,digest in result['sha256'].items():assert sha(folder/name)==digest
                print('REUSE',args.stage,mode,seed,flush=True);continue
            save(folder/'scene.json',value)
            if (folder/'job.json').exists():assert read(folder/'job.json')['identity']==identity
            command=[str(out/'renderer/tungsten.exe'),'-t',str(args.threads),'-s',str(seed),
                     '-d',str(folder),str(folder/'scene.json')]
            print('START',args.stage,mode,seed,QUALITY[args.stage],flush=True)
            started=time.monotonic()
            with (folder/'render.log').open('ab') as log:
                process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
                save(folder/'job.json',dict(identity=identity,command=command,pid=process.pid))
                try:
                    while process.poll() is None:
                        save(out/'progress.json',dict(stage=args.stage,mode=mode,seed=seed,pid=process.pid,
                             elapsed_seconds=time.monotonic()-started,log=str(folder/'render.log')))
                        try:process.wait(timeout=5)
                        except subprocess.TimeoutExpired:pass
                except BaseException:
                    process.terminate();process.wait();raise
            text=(folder/'render.log').read_text(encoding='utf-8',errors='replace');size,spp=QUALITY[args.stage]
            if process.returncode or not re.search(rf'Completed\s+{spp}/{spp}\s+spp',text) or FATAL.search(text):
                raise RuntimeError('Render failed: '+str(folder/'render.log'))
            rgb,a,b,variance=[pfm(folder/name) for name in ['render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm']]
            assert rgb.shape==a.shape==b.shape==variance.shape==(size,size,3)
            assert rgb.max()>0,'Black render'
            assert np.allclose(rgb,(a+b)/2,rtol=2e-5,atol=2e-6),'Half-buffer normalization mismatch'
            seconds=float(re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/',text)[-1])
            warnings=[line for line in text.splitlines() if '1D NEE sanity check failed' in line]
            half_noise=.5*((a-b)@Y)
            result=dict(identity=identity,stage=args.stage,mode=mode,seed=seed,resolution=[size,size],spp=spp,
                threads=args.threads,render_seconds=seconds,wall_seconds=time.monotonic()-started,
                mean_luminance=float((rgb@Y).mean()),maximum=float(rgb.max()),
                noise_rms=float(np.sqrt(np.mean(half_noise**2))),warnings=warnings,
                sha256={name:sha(folder/name) for name in ['scene.json','render.png','render.pfm',
                    'samplesA.pfm','samplesB.pfm','samplesVariance.pfm','render.log']})
            save(folder/'result.json',result)
            print('DONE',args.stage,mode,seed,'seconds',seconds,'noise_rms',result['noise_rms'],flush=True)
    save(out/'progress.json',dict(stage=args.stage,state='complete',modes=args.modes,seeds=args.seeds))

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(v))+'</th>' for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+str(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</tbody></table>'

def report(args):
    out=args.output.resolve();m=read(out/'manifest.json');summaries=[];sections=[];linked=0
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for stage in ['final','convergence','preview']:
        for seed in sorted({int(p.name[4:]) for mode in MODES for p in (out/stage/mode).glob('seed*') if (p/'result.json').exists()}):
            records={mode:read(out/stage/mode/f'seed{seed}/result.json') for mode in MODES
                     if (out/stage/mode/f'seed{seed}/result.json').exists()}
            paths=[];rows=[]
            fig,axes=plt.subplots(1,len(records),figsize=(5*len(records),5.4),squeeze=False,constrained_layout=True)
            detail_fig,detail_axes=plt.subplots(1,len(records),figsize=(5*len(records),4.8),squeeze=False,constrained_layout=True)
            for ax,detail_ax,(mode,r) in zip(axes[0],detail_axes[0],records.items()):
                folder=out/stage/mode/f'seed{seed}'
                for name,digest in r['sha256'].items():assert sha(folder/name)==digest,(folder,name)
                rgb=pfm(folder/'render.pfm');a=pfm(folder/'samplesA.pfm');b=pfm(folder/'samplesB.pfm')
                luminance=rgb@Y;noise=.5*((a-b)@Y)
                mask=np.indices(luminance.shape)
                object_roi=((mask[1]/luminance.shape[1]-.5)/.43)**2+((mask[0]/luminance.shape[0]-.5)/.47)**2<=1
                s=dict(stage=stage,mode=mode,seed=seed,spp=r['spp'],resolution=r['resolution'],
                    render_seconds=r['render_seconds'],threads=r['threads'],mean_luminance=r['mean_luminance'],
                    noise_rms=r['noise_rms'],object_noise_rms=float(np.sqrt(np.mean(noise[object_roi]**2))),
                    relative_object_noise=float(np.sqrt(np.mean(noise[object_roi]**2))/np.sqrt(np.mean(luminance[object_roi]**2))),
                    warnings=r['warnings'],folder=str(folder.relative_to(out)))
                summaries.append(s);paths.append(str(folder/'render.png'))
                ax.imshow(Image.open(folder/'render.png'));ax.axis('off')
                ax.set_title(mode+f'\n{r["spp"]} spp | {r["render_seconds"]:.2f} s | noise {r["noise_rms"]:.5g}',fontsize=10)
                display=np.asarray(Image.open(folder/'render.png'))
                hh,ww=display.shape[:2]
                detail_ax.imshow(display[int(.18*hh):int(.68*hh),int(.20*ww):int(.80*ww)],interpolation='nearest')
                detail_ax.set_title(mode,fontsize=10);detail_ax.axis('off')
                link=folder.relative_to(out).as_posix()
                rows.append([mode,str(r['threads']),f'{r["render_seconds"]:.2f}',f'{r["noise_rms"]:.6g}',f'{s["relative_object_noise"]:.3%}',
                    str(len(r['warnings'])),f'<a href="{link}/render.png">PNG</a> · <a href="{link}/render.pfm">PFM</a> · <a href="{link}/scene.json">配置</a> · <a href="{link}/render.log">日志</a>'])
            fig.suptitle(f'Shaderball | Small sphere light | {stage} | seed {seed}')
            (out/'figures').mkdir(exist_ok=True);image_path=f'figures/{stage}_seed{seed}.png'
            fig.savefig(out/image_path,dpi=130);plt.close(fig)
            detail_path=f'figures/{stage}_seed{seed}_detail.png'
            detail_fig.suptitle(f'{stage} | Same crop: normalized x=[.20,.80], y=[.18,.68]')
            detail_fig.savefig(out/detail_path,dpi=130);plt.close(detail_fig)
            size,spp=QUALITY[stage]
            sections.append(f'<h2>{stage}：{size}×{size}，{spp} spp，seed {seed}</h2><a href="{image_path}"><img src="{image_path}" alt="三方法对照"></a>'+table(
                ['方法','线程','渲染时间 / s','亮度噪声 RMS','对象噪声 / 当前图像 RMS','NEE 告警数','文件'],rows)+
                f'<details><summary>高光与凹槽局部放大</summary><img src="{detail_path}" alt="一致区域局部放大"></details>')
    independent=[]
    for mode in MODES:
        records=[s for s in summaries if s['stage']=='convergence' and s['mode']==mode]
        if len(records)>=2:
            a,b=(pfm(out/s['folder']/'render.pfm')@Y for s in records[:2])
            independent.append(dict(mode=mode,seeds=[s['seed'] for s in records[:2]],spp=records[0]['spp'],
                independent_seed_noise_rms=float(np.sqrt(np.mean((a-b)**2)/2)),
                half_buffer_noise_rms=float(np.sqrt(np.mean([s['noise_rms']**2 for s in records]))),
                mean_luminance=float(np.mean([s['mean_luminance'] for s in records])),
                seed_mean_luminances=[s['mean_luminance'] for s in records]))
    noise_rows=[[s['mode'],str(s['seeds']),f'{s["independent_seed_noise_rms"]:.6g}',
                 f'{s["half_buffer_noise_rms"]:.6g}',f'{s["mean_luminance"]:.6g}'] for s in independent]
    noise_section='<h2>高采样与独立种子复核</h2>'+table(['方法','种子','独立种子差 / √2 的 RMS','半样本噪声合并 RMS','平均亮度'],noise_rows)
    noise_section+='<p>独立种子比较使用相同模型和相同 spp 的两张图；半样本统计合并各种子的噪声平方。样本仍有限，不能把比值当作已验证的等质量 spp 倍数。</p>'
    save(out/'summary.json',dict(rows=summaries,independent_seed_checks=independent,light=LIGHT,covariance=m['covariance'],
        noise_estimator='RMS of (halfA-halfB)/2 in linear luminance; frame/ROI diagnostics, not a common-reference error.',
        no_non_gpis_reflecting_geometry=True))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>纯 GPIS Shaderball · 小光源三方法对照</title>
<style>body{max-width:1550px;margin:30px auto;padding:0 22px;font:16px/1.65 system-ui;color:#203044;background:#f5f7fa}
img{max-width:100%;height:auto;background:#fff}table{width:100%;border-collapse:collapse;background:#fff}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}
th:first-child,td:first-child{text-align:left}a{color:#2166b6}.note{background:#e8eef7;padding:16px}</style>
<h1>纯 GPIS Shaderball：Renewal+ / global_conditional / classic_global</h1>
<div class="note">2026-11-14 · 同一个解析 knob SDF · σ=0.02，ℓ=0.2 · 原绿色镜面材质和相机。<br>
单个白色球形面积光源：位置 (-3,3,4)，半径 0.06，radiance=2000。移除地面、大平面光源和环境光，黑色背景。
普通 primitive 只有透明介质边界和发光球；边界没有反射／遮挡作用，所有物体反射来自 GPIS shaderball。</div>
<p>三方法共用 GP 参数、材质、相机、光源、路径上限、NEE/MIS 开关与随机种子；使用同一冻结程序和线程数。
global_conditional 开启此前已测试的几何／点上界与 simple-Q2 提议，精确 hazard 不变；Renewal+ 使用 1D MIS、横向梯度相关、step=0.0025、impulse_density=10。</p>
<p>所有 PNG 使用渲染器相同 filmic 显示，没有按方法归一化曝光。噪声取线性亮度 (halfA−halfB)/2 的 RMS；
对象区域使用固定的椭圆画面遮罩，其相对量以当前图像 RMS 归一化。不同模型的期望可能不同，因此噪声不是相对同一真值的误差，任何一张图都不是精确参考。
若有 1D NEE 一致性告警，表格保留计数并可查看日志。</p>'''+noise_section+''.join(sections)+'''
<p><a href="summary.json">完整数据</a> · <a href="manifest.json">参数与冻结程序哈希</a> · <a href="sources/shaderball.json">来源 shaderball 场景</a></p></html>'''
    (out/'comparison.html').write_text(page,encoding='utf-8')
    for attr in ['href','src']:
        for part in page.split(attr+'='+chr(34))[1:]:
            link=part.split(chr(34))[0]
            if '://' not in link:
                assert (out/link).exists(),link
                linked+=1
    save(out/'verification.json',dict(render_records=len(summaries),all_render_hashes_verified=True,
        local_links=linked,matched_scene_validation=True,no_ordinary_planes=True,
        html_sha256=sha(out/'comparison.html'),summary_sha256=sha(out/'summary.json')))
    print('REPORT',len(summaries),'renders:',out/'comparison.html',flush=True)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','preview','final','convergence','report'])
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--threads',type=int,default=16)
    parser.add_argument('--seeds',type=int,nargs='+',default=[20261024])
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    parser.add_argument('--step',type=float,default=.0025)
    args=parser.parse_args()
    if args.threads<=0 or args.step<=0:parser.error('threads and step must be positive')
    if args.stage=='prepare':prepare(args)
    elif args.stage=='report':report(args)
    else:render(args);report(args)

if __name__=='__main__':main()
