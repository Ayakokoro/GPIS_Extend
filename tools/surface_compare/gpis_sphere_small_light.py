"""Pure GPIS sphere under a small area light: Renewal+ versus start conditioning.

Use the local plot Python: prepare, preview, convergence, run, report, or status.
Convergence is 64x64/1024 spp; earlier preview and final renders are preserved.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback

import numpy as np
from figures import Image
from run import ROOT, load, sha, pfm, light
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links
from gpis_shaderball_parameter_comparison import grid, render_duration

OUT = ROOT/'outputs/gpis_sphere_small_light_compare_20261112'
MODES = ('global_conditional', 'gpis_renewal_plus')
LABELS = dict(global_conditional='start_conditioning', gpis_renewal_plus='GPIS renewal+ (1D MIS)')
QUALITY = dict(preview=(64,32), final=(256,128), convergence=(64,1024))
SEED, STEP, PPD = 20261024, .0025, 67.
SIGMA, ELL, RADIUS = .05, .1, 1.
LIGHT_POSITION, LIGHT_SIZE, LIGHT_RADIANCE = [-3.,1.5,3.], .12, 2000.
FILES = ('scene.json','render.png','render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm','render.log')
Y = np.array([.2126,.7152,.0722])
FATAL = re.compile(r'Caught an internal error|unrecoverable|Unable to load|Gradient invalid|'
    r'Wrong normal direction|Failed to sample gradient|Gradient zero|Conditioning .* error|'
    r'GPContext does not exist|Clamp one (?:inf|nan) sample|invalid FP|terminate called', re.I)


def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temporary.replace(path)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def folder(args,stage,mode):
    return args.output/stage/mode/f'seed{SEED}'


def scene(args,stage,mode):
    assert mode in MODES
    gp=dict(type='standard',mean=dict(type='spherical',center=[0,0,0],radius=RADIUS),
            covariance=dict(type='squared_exponential',sigma=SIGMA,lengthScale=ELL))
    material=dict(type='mirror',albedo=.9)
    if mode=='global_conditional':
        medium=dict(name='gp',type='global_conditional',gaussian_process=gp,material=material,
            max_bounces=128,sdf_lipschitz=1.001,gradient_bound=1.001,tracking_step=0.,
            non_gp_surface_policy='retain_endpoint',fp_gradient=dict(enabled=False),
            tracking_step_scale=.5,tracking_age_scale=.25,tracking_max_expected_events=64.)
    else:
        medium=dict(name='gp',type='sparse_conv_noise',gaussian_process=gp,
            phase_function=dict(type='brdf',bsdf=material),max_bounces=128,
            sigma_a=0,sigma_s=1,density=1,step_size=STEP,min_step=8,seed=17429,
            impulse_density=10,single_realization=False,correlation_context='renewal+',
            intersect_method='gp_discrete',normal_method='conditioned_gaussian',
            isotropic_3D_sampling=False,
            **{'1D_sampling':True,'1D_sampling_scheme':'mis','1D_gradient_correlationXY':True})
    size,spp=QUALITY[stage]
    return dict(media=[medium],camera=dict(type='pinhole',resolution=[size,size],fov=36,
        tonemap='filmic',reconstruction_filter='tent',
        transform=dict(position=[0,0,4],look_at=[0,0,0],up=[0,1,0])),
        integrator=dict(type='path_tracer',min_bounces=0,max_bounces=32,
            enable_light_sampling=True,enable_volume_light_sampling=True,
            enable_consistency_checks=False,enable_two_sided_shading=True,
            use_cuda_gpis=False,independent_path_sampling=True,wavefront_gpis=False),
        primitives=[dict(name='process_bounds',type='cube',int_medium='gp',
            bsdf=dict(type='forward',albedo=1),transform=dict(position=[0,0,0],scale=3)),
            light('small_area_key',LIGHT_POSITION,[LIGHT_SIZE,1,LIGHT_SIZE],LIGHT_RADIANCE)],
        renderer=dict(output_directory=str(folder(args,stage,mode)),output_file='render.png',
            hdr_output_file='render.pfm',spp=spp,spp_step=8,overwrite_output_files=True,
            adaptive_sampling=False,stratified_sampler=False,scene_bvh=True,
            enable_resume_render=stage!='preview',resume_render_file='resume.dat',
            checkpoint_interval='300' if stage!='preview' else '0',timeout='0',
            output_buffers=[dict(type='color',hdr_output_file='samples.pfm',
                sample_variance=True,two_buffer_variance=True)]))


def prepare(args):
    path=args.output/'manifest.json'
    if path.exists():
        manifest=load(path)
        assert manifest.get('active_runner_sha256',manifest['runner_sha256'])==sha(Path(__file__)),'Runner changed; use a new output'
        if 'original_runner' in manifest:
            assert sha(args.output/manifest['original_runner'])==manifest['runner_sha256']
        assert manifest['modes']==list(MODES) and manifest['seed']==SEED
        assert manifest['quality']=={k:list(v) for k,v in QUALITY.items()}
        assert manifest['flip']['sha256']==sha(args.flip)
        for name,digest in manifest['runtime_sha256'].items():
            assert sha(args.output/'renderer'/name)==digest
    else:
        runtime=args.output/'renderer'
        runtime.mkdir(parents=True,exist_ok=True)
        hashes={}
        for source in [ROOT/'build/Release/tungsten.exe',*sorted((ROOT/'build/Release').glob('*.dll'))]:
            shutil.copy2(source,runtime/source.name)
            hashes[source.name]=sha(runtime/source.name)
        manifest=dict(created_at=now(),modes=list(MODES),labels=LABELS,seed=SEED,
            quality={k:list(v) for k,v in QUALITY.items()},runtime_sha256=hashes,
            runner_sha256=sha(Path(__file__)),
            plot_helper_sha256=sha(ROOT/'tools/surface_compare/gpis_shaderball_parameter_comparison.py'),
            sphere=dict(center=[0,0,0],radius=RADIUS,sigma=SIGMA,lengthScale=ELL,
                covariance='squared_exponential',material=dict(type='mirror',albedo=.9)),
            light=dict(type='quad',position=LIGHT_POSITION,width=LIGHT_SIZE,height=LIGHT_SIZE,
                radiance=LIGHT_RADIANCE),
            renewal=dict(step_size=STEP,impulse_density=10,one_dimensional=True,
                sampling_scheme='mis',gradient_correlation_xy=True),
            flip=dict(executable=str(args.flip),sha256=sha(args.flip),ppd=PPD),
            notes=['One GPIS sphere, one small emissive quad, one forward medium bound; black background.',
                'No ordinary reflecting geometry and no other rendering modes.',
                'The two methods share mean, covariance, material, camera, light, integrator and seed.',
                'Known 1D NEE consistency diagnostics are retained and labeled if emitted.',
                'Finite-spp comparisons are exploratory; neither image is an exact reference.'])
        save(path,manifest)
    for stage in QUALITY:
        scenes=[scene(args,stage,m) for m in MODES]
        a,b=copy.deepcopy(scenes[0]),copy.deepcopy(scenes[1])
        assert a['media'][0]['gaussian_process']==b['media'][0]['gaussian_process']
        assert a['media'][0]['material']==b['media'][0]['phase_function']['bsdf']
        a.pop('media');b.pop('media')
        a['renderer'].pop('output_directory');b['renderer'].pop('output_directory')
        assert a==b,'Unexpected scene mismatch'
        for mode,value in zip(MODES,scenes):
            assert len(value['primitives'])==2
            assert [p['bsdf']['type'] for p in value['primitives']]==['forward','null']
            dest=folder(args,stage,mode)/'scene.json'
            if dest.exists():assert load(dest)==value
            else:save(dest,value)
    return manifest


def identity(args,stage,mode):
    return dict(scene=scene(args,stage,mode),seed=SEED,
        renderer_sha256=load(args.output/'manifest.json')['runtime_sha256']['tungsten.exe'])


def verified(args,stage,mode):
    d=folder(args,stage,mode)
    row=load(d/'result.json')
    assert row['identity']==identity(args,stage,mode)
    for name,digest in row['sha256'].items():assert sha(d/name)==digest,str(d/name)
    return row


def status(args,active=None,failures=None):
    rows=[]
    for stage in QUALITY:
        for mode in MODES:
            d=folder(args,stage,mode)
            text=(d/'render.log').read_text(encoding='utf-8',errors='replace') if (d/'render.log').exists() else ''
            progress=re.findall(r'Completed (\d+)/(\d+) spp[^\r\n]*',text)
            rows.append(dict(stage=stage,mode=mode,label=LABELS[mode],
                state='complete' if (d/'result.json').exists() else 'running' if active==str(d) else 'pending',
                spp=int(progress[-1][0]) if progress else 0,target_spp=QUALITY[stage][1]))
    final=[r for r in rows if r['stage']=='convergence']
    save(args.output/'progress.json',dict(updated_at=now(),active=active,failures=failures or [],
        completed=sum(r['state']=='complete' for r in final),total=2,
        complete=all(r['state']=='complete' for r in final),target_stage='convergence',
        target_resolution=[64,64],target_spp=1024,jobs=rows))


def render_one(args,stage,mode):
    d=folder(args,stage,mode)
    if (d/'result.json').exists():return verified(args,stage,mode)
    expected=identity(args,stage,mode)
    if (d/'job.json').exists():assert load(d/'job.json')['identity']==expected
    if (d/'render.log').exists():
        archive=d/'attempts';archive.mkdir(exist_ok=True)
        shutil.copy2(d/'render.log',archive/f'{time.time_ns()}.log')
    command=[str(args.output/'renderer/tungsten.exe'),'-t',str(args.threads),'-s',str(SEED),
             '-d',str(d),str(d/'scene.json')]
    print('START',stage,mode,QUALITY[stage],flush=True)
    started=time.monotonic()
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0)
    with (d/'render.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
        save(d/'job.json',dict(identity=expected,process_id=process.pid,command=command,started_at=now()))
        try:
            while process.poll() is None:
                status(args,active=str(d))
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
        except BaseException:
            process.terminate();process.wait();raise
    text=(d/'render.log').read_text(encoding='utf-8',errors='replace')
    size,spp=QUALITY[stage]
    assert process.returncode==0 and 'Finished render.' in text,str(d/'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp',text) and not FATAL.search(text),str(d/'render.log')
    warnings=[line for line in text.splitlines() if '1D NEE sanity check failed' in line]
    assert not any('sanity check failed' in line and '1D NEE sanity check failed' not in line for line in text.splitlines())
    rgb,a,b,variance=[pfm(d/name) for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')]
    assert rgb.shape==variance.shape==(size,size,3) and rgb.max()>0
    assert np.allclose((a+b)/2,rgb,rtol=4e-6,atol=4e-7)
    with Image.open(d/'render.png') as im:assert im.size==(size,size)
    progress=re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/[\d.]+\s+sec',text)
    assert sha(args.output/'renderer/tungsten.exe')==expected['renderer_sha256']
    discarded=re.search(r'discarded surface births: (\{[^\n]+\})',text)
    row=dict(identity=expected,resolution=[size,size],spp=spp,seed=SEED,threads=args.threads,
        seconds=time.monotonic()-started,render_seconds=float(progress[-1]),finished_at=now(),command=command,
        mean_luminance=float((rgb@Y).mean()),maximum=float(rgb.max()),warnings=warnings,
        discarded_surface_births=json.loads(discarded[1]) if discarded else None,
        sha256={name:sha(d/name) for name in FILES})
    save(d/'result.json',row)
    print('DONE',stage,mode,round(row['seconds'],1),'sec; mean Y',round(row['mean_luminance'],6),flush=True)
    return verified(args,stage,mode)


def noise_diagnostic(test,reference):
    def read(d):
        rgb,a,b=[pfm(d/name)@Y for name in ('render.pfm','samplesA.pfm','samplesB.pfm')]
        return rgb,a,b
    a,b=read(test),read(reference)
    difference=a[0]-b[0]
    noise=.5*((a[1]-b[1])-(a[2]-b[2]))
    result=dict(difference_rms=float(np.sqrt(np.mean(difference**2))),
        estimated_difference_noise_rms=float(np.sqrt(np.mean(noise**2))),
        signed_noise_subtracted_mse=float(np.mean(difference**2)-np.mean(noise**2)),
        mean_delta=float(difference.mean()),halfA_mean_delta=float((a[1]-b[1]).mean()),
        halfB_mean_delta=float((a[2]-b[2]).mean()),
        renewal_relative_mean=float(b[0].mean()/a[0].mean()-1))
    for mode,values in zip(MODES,(a,b)):
        result[mode]=dict(mean_luminance=float(values[0].mean()),
            noise_rms=float(np.sqrt(np.mean(((values[1]-values[2])/2)**2))),
            halfA_mean=float(values[1].mean()),halfB_mean=float(values[2].mean()))
    return result


def report(args):
    sections=[];records=[];metrics=[];diagnostics=[]
    for stage in ('convergence','final','preview'):
        available={}
        for mode in MODES:
            d=folder(args,stage,mode)
            if (d/'result.json').exists():
                row=verified(args,stage,mode);available[mode]=d
                records.append(dict(stage=stage,mode=mode,result=row,folder=d.relative_to(args.output).as_posix()))
        dest=args.output/'figures'/stage;dest.mkdir(parents=True,exist_ok=True)
        paths=[available[m]/'render.png' for m in MODES if m in available]
        labels=[LABELS[m]+'\nRender: '+render_duration(load(available[m]/'result.json')['render_seconds']) for m in MODES if m in available]
        detail=''
        if len(available)==2:
            ref,test=available['gpis_renewal_plus'],available['global_conditional']
            fd=args.output/'flip'/stage
            result=evaluate(args.flip,ref/'render.png',test/'render.png',fd,PPD)
            paths.append(fd/result['error_map']);labels.append(f'LDR-FLIP\nmean {result["statistics"]["mean"]:.6f}')
            a,b=pfm(test/'render.pfm'),pfm(ref/'render.pfm')
            metric=dict(stage=stage,reference='gpis_renewal_plus',test='global_conditional',
                flip=result['statistics'],linear_rgb_rmse=float(np.sqrt(np.mean((a-b)**2))),
                native_record=(fd/'result.json').relative_to(args.output).as_posix())
            metrics.append(metric)
            diagnostic=dict(stage=stage,resolution=list(QUALITY[stage][:1])*2,spp=QUALITY[stage][1],
                            **noise_diagnostic(test,ref))
            diagnostics.append(diagnostic)
            detail=f'<p>Mean FLIP：<b>{metric["flip"]["mean"]:.6f}</b>；线性 RGB RMSE：{metric["linear_rgb_rmse"]:.6f}。<a href="{metric["native_record"]}">原生 FLIP 统计</a></p>'
            detail+=(f'<p>Renewal+ 平均亮度相对 start_conditioning：{diagnostic["renewal_relative_mean"]:+.3%}；'
                f'线性亮度差 RMS：{diagnostic["difference_rms"]:.6f}，'
                f'半采样估计的噪声差 RMS：{diagnostic["estimated_difference_noise_rms"]:.6f}。'
                '这是噪声量级诊断，不是对 FLIP 的噪声分解。</p>')
        picture=''
        if paths:
            display_paths=[]
            for i,path in enumerate(paths):
                with Image.open(path) as im:
                    if im.width<256:
                        enlarged=dest/f'display_{i}.png'
                        im.convert('RGB').resize((256,256),Image.Resampling.NEAREST).save(enlarged)
                        display_paths.append(enlarged)
                    else:display_paths.append(path)
            grid(display_paths,labels,dest/'comparison.png',columns=3)
            url=(dest/'comparison.png').relative_to(args.output).as_posix()
            picture=f'<a href="{url}"><img src="{url}"></a>'
        source_rows=''
        for mode,d in available.items():
            row=load(d/'result.json');url=d.relative_to(args.output).as_posix()
            half_noise=.5*((pfm(d/'samplesA.pfm')-pfm(d/'samplesB.pfm'))@Y)
            rms=float(np.sqrt(np.mean(half_noise**2)))
            source_rows+=f'<tr><td>{LABELS[mode]}</td><td>{row["render_seconds"]:.2f} s · {row["threads"]} 线程</td><td>{rms:.6f}</td><td><a href="{url}/render.png">原图</a> · <a href="{url}/scene.json">配置</a> · <a href="{url}/render.log">日志</a></td></tr>'
            if row['warnings']:
                detail+='<p class="warn">该图含 NEE 一致性告警：'+html.escape('; '.join(row['warnings']))+'</p>'
        size,spp=QUALITY[stage]
        title=dict(convergence='高采样复核',final='先前正式结果',preview='先前预览')[stage]
        sections.append(f'<h2>{title}：{size}×{size} · {spp} spp · 已完成 {len(available)}/2</h2>'+picture+detail+
            '<table><tr><th>方法</th><th>渲染时间</th><th>估计噪声 RMS</th><th>文件</th></tr>'+source_rows+'</table>')
    complete=any(r['stage']=='convergence' for r in metrics)
    save(args.output/'metrics.json',dict(complete=complete,sources=records,comparisons=metrics))
    save(args.output/'convergence_diagnostics.json',dict(stages=diagnostics,
        method='Full-image linear luminance: noise=(halfA-halfB)/2; two half buffers do not establish convergence or an independent-seed confirmation.'))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPIS 球体 · 小面积光源</title><style>body{max-width:1300px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}.warn{background:#fff0d2;padding:12px}h2{margin-top:32px}</style>
<h1>GPIS 球体：小面积光源下的两种方法</h1>'''+f'''<div class="note">球半径 {RADIUS:g}，σ={SIGMA:g}，ℓ={ELL:g}，mirror albedo=0.9。<br>
单个侧前方小面积光源：位置 {LIGHT_POSITION}，边长 {LIGHT_SIZE:g}，radiance={LIGHT_RADIANCE:g}。背景黑色，反射只发生在 GPIS 球体上。<br>
GPIS Renewal+ 使用 1D sparse convolution、MIS、横向梯度相关；步长 {STEP:g}，impulse density=10。start_conditioning 关闭 fixed5。</div>
<p>两种方法共用均值场、协方差、材质、相机、光源、随机种子。统一 filmic 显示，本地 FLIP，{PPD:g} PPD，误差色标 0–1；参考图含采样噪声。</p>
<p>64×64 图使用最近邻放大展示；FLIP 始终在原始分辨率上计算。比较采样收敛时，应看同为 64×64 的 32 spp 与 1024 spp。</p>
<p><a href="manifest.json">参数与运行版本</a> · <a href="metrics.json">指标</a> · <a href="convergence_diagnostics.json">噪声诊断</a> · <a href="progress.json">进度</a></p>'''+''.join(sections)+'</html>'
    (args.output/'comparison.html').write_text(page,encoding='utf-8')
    links=verify_links(args.output/'comparison.html')
    save(args.output/'verification.json',dict(passed=True,complete=complete,
        render_records=len(records),native_flip_records=len(metrics),matched_scene_parameters=True,
        no_ordinary_reflecting_geometry=True,local_links=links,html_sha256=sha(args.output/'comparison.html'),
        metrics_sha256=sha(args.output/'metrics.json')))
    print('REPORT',len(records),'renders,',len(metrics),'FLIP pairs',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','preview','convergence','run','report','status'))
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--flip',type=Path,default=DEFAULT_FLIP)
    args=parser.parse_args();args.output=args.output.resolve()
    assert args.threads>0 and args.flip.is_file()
    if args.command=='status':print(json.dumps(load(args.output/'progress.json'),ensure_ascii=False,indent=2));return
    prepare(args)
    if args.command in ('prepare','report'):
        status(args);report(args);return
    lock=args.output/'runner.lock'
    descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:json.dump(dict(pid=os.getpid(),started_at=now()),stream)
    try:
        stages=(args.command,) if args.command in ('preview','convergence') else tuple(QUALITY)
        for stage in stages:
            for mode in MODES:
                try:render_one(args,stage,mode)
                except Exception as exc:
                    failure=dict(stage=stage,mode=mode,error=str(exc),traceback=traceback.format_exc())
                    save(args.output/'failure.json',failure);status(args,failures=[failure]);raise
                status(args);report(args)
        print('COMPLETE',args.command,flush=True)
    finally:lock.unlink()


if __name__=='__main__':main()
