"""Shared-field absorbing GPIS plates: first-passage visibility comparison.

Run prepare, smoke, run, report or status with the local plot Python.
Only start_conditioning and original 1D sparse-convolution Renewal+ are rendered.
"""
from __future__ import annotations

import argparse
import copy
import html
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time

import numpy as np
from figures import Image, plt
from run import ROOT, load, sha, pfm
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links
from gpis_shaderball_parameter_comparison import grid
from gpis_sphere_small_light import save, now, Y, FATAL, FILES, noise_diagnostic

OUT=ROOT/'outputs/gpis_two_slabs_compare_20261113'
MODES=('global_conditional','gpis_renewal_plus')
LABELS=dict(global_conditional='start_conditioning',gpis_renewal_plus='GPIS renewal+ (1D MIS)')
QUALITY=dict(smoke=(32,8),final=(64,1024))
SEEDS=(20261024,20261113)
SIGMA,ELL,STEP,PPD=.02,.2,.0025,67.
SOURCES=('src/core/math/SdfFunctions.hpp','src/core/math/SdfFunctions.cpp',
         'src/core/math/SdfFunctionsQueries.cpp','tests/SdfPointJetTest.cpp')


def folder(args,stage,mode,seed):
    return args.output/stage/mode/f'seed{seed}'


def scene(args,stage,mode,seed):
    gp=dict(type='standard',mean=dict(type='procedural',func='two_slabs'),
            covariance=dict(type='squared_exponential',sigma=SIGMA,lengthScale=ELL))
    material=dict(type='lambert',albedo=0)
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
    return dict(media=[medium],camera=dict(type='pinhole',resolution=[size,size],fov=44,
        tonemap='filmic',reconstruction_filter='box',
        transform=dict(position=[0,2,0],look_at=[0,0,0],up=[0,0,1])),
        integrator=dict(type='path_tracer',min_bounces=0,max_bounces=8,
            enable_light_sampling=False,enable_volume_light_sampling=False,
            enable_consistency_checks=False,enable_two_sided_shading=True,
            use_cuda_gpis=False,independent_path_sampling=True,wavefront_gpis=False),
        primitives=[dict(name='shared_process_bounds',type='cube',int_medium='gp',
            bsdf=dict(type='forward',albedo=1),transform=dict(position=[0,0,0],scale=[2.2,.6,2.2])),
            dict(name='backlight',type='quad',emission=1,bsdf=dict(type='null',albedo=1),
                 transform=dict(position=[0,-.35,0],y_axis=[0,1,0],scale=[2.4,1,2.4]))],
        renderer=dict(output_directory=str(folder(args,stage,mode,seed)),output_file='render.png',
            hdr_output_file='render.pfm',spp=spp,spp_step=8,overwrite_output_files=True,
            adaptive_sampling=False,stratified_sampler=False,scene_bvh=True,
            enable_resume_render=stage=='final',resume_render_file='resume.dat',
            checkpoint_interval='300' if stage=='final' else '0',timeout='0',
            output_buffers=[dict(type='color',hdr_output_file='samples.pfm',
                sample_variance=True,two_buffer_variance=True)]))


def jobs(stage):
    return [(stage,m,s) for s in (SEEDS if stage=='final' else SEEDS[:1]) for m in MODES]


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    dest=args.output/'manifest.json'
    if dest.exists():
        manifest=load(dest)
        assert manifest.get('active_runner_sha256',manifest['runner_sha256'])==sha(Path(__file__)),'Runner changed; use a new output'
        for name,digest in manifest['runtime_sha256'].items():
            assert sha(args.output/'renderer'/name)==digest
    else:
        runtime=args.output/'renderer';runtime.mkdir(exist_ok=True)
        for src in [ROOT/'build/Release/tungsten.exe',*sorted((ROOT/'build/Release').glob('*.dll'))]:
            shutil.copy2(src,runtime/src.name)
        source_dest=args.output/'source';source_dest.mkdir(exist_ok=True)
        for filename in SOURCES:
            target=source_dest/Path(filename).name
            shutil.copy2(ROOT/filename,target)
        shutil.copy2(Path(__file__),source_dest/Path(__file__).name)
        manifest=dict(created_at=now(),modes=MODES,labels=LABELS,quality=QUALITY,seeds=SEEDS,
            runner_sha256=sha(Path(__file__)),runtime_sha256={p.name:sha(p) for p in runtime.iterdir()},
            source_sha256={p.name:sha(p) for p in source_dest.iterdir()},
            geometry=dict(centers=[[.18,.02,0],[-.18,-.02,0]],size=[1.2,.02,1.2],
                center_separation=.04,face_gap=.02,horizontal_offset=.36,
                mean='min_i max_j(abs(p_j-center_ij)-half_extent_j)',
                note='1-Lipschitz box implicit mean; exact signed distance near planar faces.'),
            sigma=SIGMA,lengthScale=ELL,step_size=STEP,impulse_density=10,
            material=dict(type='lambert',albedo=0),flip=dict(path=str(args.flip),sha256=sha(args.flip),ppd=PPD),
            interpretation=['Both slabs share one GP and one continuous medium, with no internal boundary.',
                'Absorbing surfaces and disabled NEE isolate uncollided camera-to-emitter visibility.',
                'No ordinary reflecting geometry, GPIS Global or fixed5.',
                'No GP collision contributes to an escaped path; this tests within-segment survival, not memory after a reflection.',
                'Neither noisy render is automatically an exact GP reference.'])
        save(dest,manifest)
    for stage in QUALITY:
        for _,mode,seed in jobs(stage):
            dest=folder(args,stage,mode,seed)/'scene.json';value=scene(args,stage,mode,seed)
            if dest.exists():assert load(dest)==value
            else:save(dest,value)
        a,b=[copy.deepcopy(scene(args,stage,m,SEEDS[0])) for m in MODES]
        assert a['media'][0]['gaussian_process']==b['media'][0]['gaussian_process']
        assert a['media'][0]['material']==b['media'][0]['phase_function']['bsdf']
        for value in (a,b):value.pop('media');value['renderer'].pop('output_directory')
        assert a==b
    return manifest


def verified(args,stage,mode,seed):
    d=folder(args,stage,mode,seed);row=load(d/'result.json')
    assert row['scene']==scene(args,stage,mode,seed)
    assert row['renderer_sha256']==sha(args.output/'renderer/tungsten.exe')
    for name,digest in row['sha256'].items():assert sha(d/name)==digest
    return row


def status(args,active=None):
    rows=[]
    for stage in QUALITY:
        for _,mode,seed in jobs(stage):
            d=folder(args,stage,mode,seed)
            text=(d/'render.log').read_text(encoding='utf-8',errors='replace') if (d/'render.log').exists() else ''
            samples=re.findall(r'Completed (\d+)/(\d+) spp',text)
            rows.append(dict(stage=stage,mode=mode,seed=seed,
                state='complete' if (d/'result.json').exists() else 'running' if str(d)==active else 'pending',
                spp=int(samples[-1][0]) if samples else 0,target_spp=QUALITY[stage][1]))
    final=[r for r in rows if r['stage']=='final']
    save(args.output/'progress.json',dict(updated_at=now(),active=active,jobs=rows,
        completed=sum(r['state']=='complete' for r in final),total=len(final),
        complete=all(r['state']=='complete' for r in final)))


def render(args,stage,mode,seed):
    d=folder(args,stage,mode,seed)
    if (d/'result.json').exists():return verified(args,stage,mode,seed)
    assert not (d/'render.log').exists(),'Unverified existing attempt: '+str(d)
    command=[str(args.output/'renderer/tungsten.exe'),'-t',str(args.threads),'-s',str(seed),'-d',str(d),str(d/'scene.json')]
    started=time.monotonic();print('START',stage,mode,seed,flush=True)
    with (d/'render.log').open('wb') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        save(d/'job.json',dict(pid=p.pid,command=command,started_at=now()))
        try:
            while p.poll() is None:
                status(args,str(d))
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
        except BaseException:p.terminate();p.wait();raise
    text=(d/'render.log').read_text(encoding='utf-8',errors='replace')
    size,spp=QUALITY[stage]
    assert p.returncode==0 and 'Finished render.' in text and not FATAL.search(text),str(d/'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp',text)
    warnings=[line for line in text.splitlines() if 'sanity check failed' in line]
    assert not warnings,warnings
    rgb,a,b,var=[pfm(d/name) for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')]
    assert rgb.shape==var.shape==(size,size,3) and rgb.max()>0
    assert np.allclose((a+b)/2,rgb,rtol=4e-6,atol=4e-7)
    # OutputBuffer accumulates a float online mean, so N*mean has rounding
    # error even when all original samples are exactly zero or one.
    assert np.max(np.abs(rgb*spp-np.round(rgb*spp)))<2e-3,'Visibility must be a Bernoulli count'
    assert np.allclose(var*(spp-1),rgb*(1-rgb),rtol=2e-4,atol=2e-6),'Bernoulli variance mismatch'
    progress=re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/[\d.]+\s+sec',text)
    row=dict(scene=scene(args,stage,mode,seed),seed=seed,spp=spp,resolution=[size,size],threads=args.threads,
        renderer_sha256=sha(args.output/'renderer/tungsten.exe'),render_seconds=float(progress[-1]),
        wall_seconds=time.monotonic()-started,mean_transmittance=float(rgb.mean()),warnings=warnings,
        finished_at=now(),sha256={name:sha(d/name) for name in FILES})
    save(d/'result.json',row)
    print('DONE',stage,mode,seed,round(row['render_seconds'],2),'s; T',round(row['mean_transmittance'],6),flush=True)
    return verified(args,stage,mode,seed)


def roi_masks(size):
    # Project the y=0 plane. ROIs avoid mean edges and the single/overlap seam.
    v=(np.arange(size)+.5)/size*2-1
    x,z=np.meshgrid(v*2*np.tan(np.deg2rad(22)),v*2*np.tan(np.deg2rad(22)))
    return dict(overlap=(np.abs(x)<.30)&(np.abs(z)<.40),
                left_single=(x>-.69)&(x<-.51)&(np.abs(z)<.40),
                right_single=(x>.51)&(x<.69)&(np.abs(z)<.40))


def schematic(args):
    from matplotlib.patches import Rectangle
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    ax=axes[0]
    for cx,cy,color in ((.18,.02,'#2680ac'),(-.18,-.02,'#da8b35')):
        ax.add_patch(Rectangle((cx-.6,cy-.01),1.2,.02,color=color,alpha=.8))
    for x in (-.62,0,.62):ax.annotate('',xy=(x,-.14),xytext=(x,.15),arrowprops=dict(arrowstyle='->',color='#555'))
    ax.axhline(-.16,color='#dfb83c',lw=5)
    ax.text(0,.18,'Camera rays',ha='center');ax.text(0,-.20,'Area backlight',ha='center')
    ax.set(xlim=(-.9,.9),ylim=(-.23,.23),xlabel='x',ylabel='y (depth, enlarged)',title='Side view: one shared GP, no internal boundary')
    ax=axes[1]
    for cx,color in ((.18,'#2680ac'),(-.18,'#da8b35')):
        ax.add_patch(Rectangle((cx-.6,-.6),1.2,1.2,color=color,alpha=.45))
    ax.text(0,0,'Overlap',ha='center');ax.text(-.6,0,'Single',ha='center');ax.text(.6,0,'Single',ha='center')
    ax.set(xlim=(-.9,.9),ylim=(-.8,.8),xlabel='x',ylabel='z',title='Camera view: two plates offset in x',aspect='equal')
    fig.tight_layout();fig.savefig(args.output/'geometry.png',dpi=130);plt.close(fig)


def report(args):
    sections=[];pairs=[];sources=[]
    for stage in ('final','smoke'):
        size,spp=QUALITY[stage];masks=roi_masks(size)
        for seed in (SEEDS if stage=='final' else SEEDS[:1]):
            available={m:folder(args,stage,m,seed) for m in MODES if (folder(args,stage,m,seed)/'result.json').exists()}
            if not available:continue
            fd=args.output/'figures'/stage/f'seed{seed}';fd.mkdir(parents=True,exist_ok=True)
            paths=[];labels=[];rows=''
            for mode,d in available.items():
                row=verified(args,stage,mode,seed);rgb=pfm(d/'render.pfm')@Y
                roi={name:float(rgb[mask].mean()) for name,mask in masks.items()}
                sources.append(dict(stage=stage,mode=mode,seed=seed,roi=roi,result=row))
                paths.append(d/'render.png');labels.append(LABELS[mode]+f'\n{row["render_seconds"]:.2f} s')
                url=d.relative_to(args.output).as_posix()
                rows+=f'<tr><td>{LABELS[mode]}</td><td>{row["render_seconds"]:.2f} s</td><td>{roi["overlap"]:.6f}</td><td>{roi["left_single"]:.6f}</td><td>{roi["right_single"]:.6f}</td><td><a href="{url}/render.png">PNG</a> · <a href="{url}/render.pfm">PFM</a> · <a href="{url}/scene.json">配置</a> · <a href="{url}/render.log">日志</a></td></tr>'
            detail=''
            if len(available)==2:
                test,ref=[available[m] for m in MODES]
                flip_dest=args.output/'flip'/stage/f'seed{seed}'
                flip=evaluate(args.flip,ref/'render.png',test/'render.png',flip_dest,PPD)
                paths.append(flip_dest/flip['error_map']);labels.append(f'LDR-FLIP\nmean {flip["statistics"]["mean"]:.6f}')
                diagnostic=noise_diagnostic(test,ref)
                rois={name:{m:float((pfm(available[m]/'render.pfm')@Y)[mask].mean()) for m in MODES} for name,mask in masks.items()}
                pairs.append(dict(stage=stage,seed=seed,flip=flip['statistics'],diagnostic=diagnostic,roi=rois,
                                  native_flip=(flip_dest/'result.json').relative_to(args.output).as_posix()))
                detail=f'<p>Mean FLIP：<b>{flip["statistics"]["mean"]:.6f}</b>。线性亮度差 RMS：{diagnostic["difference_rms"]:.6f}；估计噪声差 RMS：{diagnostic["estimated_difference_noise_rms"]:.6f}。</p>'
                fig,ax=plt.subplots(figsize=(9,3.4))
                for m in MODES:
                    values=pfm(available[m]/'render.pfm')@Y
                    ax.plot(np.arange(size),values[size//2],label=LABELS[m])
                ax.set(xlabel='Pixel x',ylabel='Linear transmittance',ylim=(0,1.03),title='Center-row visibility; backlight radiance = 1')
                ax.legend();fig.tight_layout();fig.savefig(fd/'profile.png',dpi=130);plt.close(fig)
                profile=(fd/'profile.png').relative_to(args.output).as_posix()
                detail+=f'<a href="{profile}"><img src="{profile}"></a>'
            display=[]
            for i,path in enumerate(paths):
                with Image.open(path) as im:
                    target=fd/f'display_{i}.png';im.convert('RGB').resize((256,256),Image.Resampling.NEAREST).save(target)
                    display.append(target)
            grid(display,labels,fd/'comparison.png',columns=3)
            picture=(fd/'comparison.png').relative_to(args.output).as_posix()
            sections.append(f'<h2>{stage} · {size}×{size} · {spp} spp · seed {seed}</h2><a href="{picture}"><img src="{picture}"></a>'+detail+
                '<table><tr><th>方法</th><th>耗时</th><th>重叠区透射率</th><th>左侧单层区</th><th>右侧单层区</th><th>文件</th></tr>'+rows+'</table>')
    gaussian=''
    gaussian_path=args.output/'central_ray_gaussian_check.json'
    if gaussian_path.exists():
        oracle=load(gaussian_path)
        gaussian=(f'<h2>独立的中心射线 Gaussian 检查</h2><p>直接按 SE 协方差采样离散 GP，'
            f'中心轴线透射率：<b>{oracle["transmittance"]:.6f}</b>；'
            f'采样不确定性的 95% Wilson 区间 [{oracle["wilson_95"][0]:.6f}, {oracle["wilson_95"][1]:.6f}]。'
            f'{oracle["samples"]} 条随机场样本，最细步长 0.000625。'
            '这是单根轴向射线的有限网格检查，不是 GPIS Global 渲染，也不等于整幅透视图的真值。'
            '<a href="central_ray_gaussian_check.json">完整统计与网格复核</a></p>')
    complete=sum(r['stage']=='final' for r in pairs)==len(SEEDS)
    save(args.output/'metrics.json',dict(complete=complete,sources=sources,comparisons=pairs,
        roi_definition='y=0 projected plane: overlap |x|<.30, |z|<.40; single .51<|x|<.69, |z|<.40',
        noise_note='Half-buffer estimates characterize sampling noise, not FLIP decomposition or a proof of zero bias.'))
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GPIS 双薄板背光比较</title><style>body{max-width:1200px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}h2{margin-top:32px}</style>
<h1>共享 GP 场的双薄板：背光透射</h1><div class="note">每块板 1.2×1.2，厚度 0.02；中心距 0.04，净间隙 0.02；横向错开 0.36。<br>
σ=0.02，ℓ=0.2，SE 协方差。两层间的噪声相关系数 exp(−0.04²/(2×0.2²))≈0.9802。<br>
完全吸收材质（lambert albedo=0），背光 radiance=1。无地面、无普通反射表面、无间隔介质边界；关闭 NEE，只统计未碰撞光线。<br>
Renewal+：1D sparse convolution、MIS、横向梯度相关，步长 0.0025，impulse density=10。start_conditioning 关闭 fixed5。</div>
<p>这个实验检查同一条直线上“先穿过一层后，又穿过另一层”的联合生存概率。它不测试反射后的历史重置；透射路径没有 GP 碰撞，也没有 Renewal+ 更新事件。两种有限采样图均不自动视为精确真值。</p>
<p>三个区域的数字均来自线性 PFM，不从 filmic PNG 反推。64×64 图仅用最近邻放大显示；本地 FLIP 使用原始分辨率 PNG，67 PPD，同一 filmic 显示，不做逐图曝光或降噪。</p>
<p><a href="manifest.json">参数与版本</a> · <a href="metrics.json">统计</a> · <a href="progress.json">进度</a></p>
<h2>场景示意</h2><img src="geometry.png">'''+''.join(sections)+gaussian+'</html>'
    (args.output/'comparison.html').write_text(page,encoding='utf-8')
    links=verify_links(args.output/'comparison.html')
    save(args.output/'verification.json',dict(passed=True,complete=complete,render_records=len(sources),
        flip_records=len(pairs),matched_parameters=True,all_render_hashes_verified=True,
        bernoulli_visibility_verified=True,local_links=links,html_sha256=sha(args.output/'comparison.html'),
        metrics_sha256=sha(args.output/'metrics.json')))
    print('REPORT',len(sources),'renders;',len(pairs),'FLIP pairs',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','smoke','run','report','status'))
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--flip',type=Path,default=DEFAULT_FLIP)
    args=parser.parse_args();args.output=args.output.resolve()
    assert args.threads>0 and args.flip.is_file()
    if args.command=='status':print(json.dumps(load(args.output/'progress.json'),indent=2));return
    prepare(args);schematic(args)
    if args.command in ('prepare','report'):status(args);report(args);return
    lock=args.output/'runner.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:json.dump(dict(pid=os.getpid(),started_at=now()),stream)
    try:
        for stage in (('smoke',) if args.command=='smoke' else ('smoke','final')):
            for job in jobs(stage):render(args,*job);status(args);report(args)
        print('COMPLETE',args.command,flush=True)
    finally:lock.unlink()


if __name__=='__main__':main()
