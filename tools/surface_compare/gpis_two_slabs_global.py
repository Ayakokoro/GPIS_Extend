"""Add native 3D GPIS Global at step .0025 to the two-slab experiment.

Use prepare, smoke, run, report or status with the local plot Python. Existing
Renewal+ and start_conditioning renders are verified and reused unchanged.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import numpy as np
from figures import Image, plt
from run import ROOT, load, sha, pfm
from gpis_sphere_small_light import save, now, FATAL, FILES, Y
from gpis_shaderball_flip import DEFAULT_FLIP, verify_links
from gpis_two_slabs import roi_masks
from gpis_two_slabs_step_refinement import compare, picture_grid, url

BASE=ROOT/'outputs/gpis_two_slabs_compare_20261113'
OUT=BASE/'gpis_global_comparison'
SEEDS=(20261024,20261113)
QUALITY=dict(smoke=(32,8),final=(64,1024))
MODES=('global_conditional','gpis_renewal_plus','gpis_global')
LABELS=dict(global_conditional='start_conditioning',gpis_renewal_plus='GPIS renewal+ (1D MIS)',
            gpis_global='GPIS Global (3D)')
CHANGES={'correlation_context':'global','1D_sampling':False,
         '1D_sampling_scheme':'uni','1D_gradient_correlationXY':False}
BEGIN='<!-- BEGIN TWO SLABS GPIS GLOBAL -->'
END='<!-- END TWO SLABS GPIS GLOBAL -->'


def folder(args,stage,mode,seed):
    if mode=='gpis_global':return args.output/stage/mode/f'seed{seed}'
    return args.base/stage/mode/f'seed{seed}'


def scene(args,stage,seed):
    source=folder(args,stage,'gpis_renewal_plus',seed)
    old=load(source/'scene.json');value=copy.deepcopy(old)
    value['media'][0].update(CHANGES)
    value['renderer']['output_directory']=str(folder(args,stage,'gpis_global',seed))
    check=copy.deepcopy(value)
    for key in CHANGES:check['media'][0][key]=old['media'][0][key]
    check['renderer']['output_directory']=old['renderer']['output_directory']
    assert check==old
    assert value['media'][0]['step_size']==.0025 and value['media'][0]['impulse_density']==10
    assert not value['media'][0]['single_realization']
    assert value['camera']['resolution']==[QUALITY[stage][0]]*2
    assert value['renderer']['spp']==QUALITY[stage][1]
    return value


def jobs(stage):return [(stage,s) for s in (SEEDS if stage=='final' else SEEDS[:1])]


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    original_manifest=load(args.base/'manifest.json')
    for name,digest in original_manifest['runtime_sha256'].items():assert sha(args.base/'renderer'/name)==digest
    path=args.output/'manifest.json'
    if path.exists():
        manifest=load(path)
        assert manifest.get('active_runner_sha256',manifest['runner_sha256'])==sha(Path(__file__)),'Runner changed since preparation'
        assert manifest['threads']==args.threads and manifest.get('parallel_jobs',1)==args.parallel_jobs
        for name,digest in manifest['preserved_original_sha256'].items():assert sha(args.base/name)==digest,name
    else:
        history=args.output/'history';history.mkdir(exist_ok=True)
        for name in ('comparison.html','manifest.json','metrics.json','verification.json','progress.json'):
            shutil.copy2(args.base/name,history/name)
        preserved={}
        for result in list((args.base/'final').glob('*/seed*/result.json'))+list((args.base/'smoke').glob('*/seed*/result.json')):
            row=load(result)
            for name,digest in row['sha256'].items():
                assert sha(result.parent/name)==digest
                preserved[(result.parent/name).relative_to(args.base).as_posix()]=digest
            preserved[result.relative_to(args.base).as_posix()]=sha(result)
        for name in ('manifest.json','metrics.json','renewal_step_refinement/manifest.json','renewal_step_refinement/metrics.json'):
            preserved[name]=sha(args.base/name)
        sources=args.output/'source';sources.mkdir(exist_ok=True)
        for name in ('gpis_two_slabs.py','gpis_two_slabs_step_refinement.py',Path(__file__).name):
            shutil.copy2(Path(__file__).with_name(name),sources/name)
        manifest=dict(created_at=now(),runner_sha256=sha(Path(__file__)),seeds=SEEDS,quality=QUALITY,
            resolution=[64,64],spp=1024,threads=args.threads,parallel_jobs=args.parallel_jobs,step=.0025,global_settings=CHANGES,
            renderer=str(args.base/'renderer/tungsten.exe'),runtime_sha256=original_manifest['runtime_sha256'],
            source_sha256={p.name:sha(p) for p in sources.iterdir()},preserved_original_sha256=preserved,
            flip=dict(path=str(args.flip),sha256=sha(args.flip),ppd=67.),
            notes=['Native GPIS Global: a single 3D sparse-convolution realization per camera path.',
                'Renewal+ baseline uses 1D sparse convolution; this changes noise approximation as well as context.',
                'All paths reaching the backlight are uncollided; intersegment collision-memory differences are inactive.',
                'Same step .0025, density 10, GP mean and covariance, absorbing material, camera, lights, integrator and seeds.',
                'Neither finite-spp, finite-step sparse-convolution renderer is an exact Gaussian reference.'])
        save(path,manifest)
    assert sha(args.flip)==manifest['flip']['sha256']
    for stage in QUALITY:
        for _,seed in jobs(stage):
            dest=folder(args,stage,'gpis_global',seed)/'scene.json';value=scene(args,stage,seed)
            if dest.exists():assert load(dest)==value
            else:save(dest,value)
    return manifest


def verified(args,stage,mode,seed):
    d=folder(args,stage,mode,seed);row=load(d/'result.json')
    assert row['renderer_sha256']==load(args.output/'manifest.json')['runtime_sha256']['tungsten.exe']
    if mode=='gpis_global':
        assert row['scene']==scene(args,stage,seed)
        if stage=='final':assert row['threads']==args.threads
    assert row['seed']==seed and row['spp']==QUALITY[stage][1]
    for name,digest in row['sha256'].items():assert sha(d/name)==digest,str(d/name)
    assert not row['warnings']
    return row


def status(args,active=None,failure=None):
    active_paths={active} if isinstance(active,str) else set(active or [])
    rows=[]
    for stage in QUALITY:
        for _,seed in jobs(stage):
            d=folder(args,stage,'gpis_global',seed)
            text=(d/'render.log').read_text(encoding='utf-8',errors='replace') if (d/'render.log').exists() else ''
            samples=re.findall(r'Completed (\d+)/(\d+) spp',text)
            rows.append(dict(stage=stage,mode='gpis_global',seed=seed,
                state='complete' if (d/'result.json').exists() else 'running' if str(d) in active_paths else 'pending',
                spp=int(samples[-1][0]) if samples else 0,target_spp=QUALITY[stage][1]))
    final=[r for r in rows if r['stage']=='final']
    save(args.output/'progress.json',dict(updated_at=now(),active=active,jobs=rows,
        complete=all(r['state']=='complete' for r in final),
        completed=sum(r['state']=='complete' for r in final),total=len(final),failure=failure))


def render(args,stage,seed,update_status=True):
    d=folder(args,stage,'gpis_global',seed)
    if (d/'result.json').exists():return verified(args,stage,'gpis_global',seed)
    assert not (d/'render.log').exists(),'Unverified previous attempt: '+str(d)
    exe=args.base/'renderer/tungsten.exe'
    command=[str(exe),'-t',str(args.threads),'-s',str(seed),'-d',str(d),str(d/'scene.json')]
    started=time.monotonic();print('START',stage,'gpis_global',seed,flush=True)
    with (d/'render.log').open('wb') as log:
        p=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        save(d/'job.json',dict(pid=p.pid,command=command,started_at=now()))
        try:
            while p.poll() is None:
                if update_status:status(args,str(d))
                try:p.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
        except BaseException:p.terminate();p.wait();raise
    text=(d/'render.log').read_text(encoding='utf-8',errors='replace')
    size,spp=QUALITY[stage]
    assert p.returncode==0 and 'Finished render.' in text and not FATAL.search(text),str(d/'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp',text)
    warnings=[line for line in text.splitlines() if 'sanity check failed' in line];assert not warnings,warnings
    rgb,a,b,var=[pfm(d/name) for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')]
    assert rgb.shape==var.shape==(size,size,3) and rgb.max()>0
    assert np.allclose((a+b)/2,rgb,rtol=4e-6,atol=4e-7)
    assert np.max(np.abs(rgb*spp-np.round(rgb*spp)))<2e-3
    assert np.allclose(var*(spp-1),rgb*(1-rgb),rtol=2e-4,atol=2e-6)
    progress=re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/[\d.]+\s+sec',text)
    row=dict(scene=scene(args,stage,seed),seed=seed,spp=spp,resolution=[size,size],threads=args.threads,
        renderer_sha256=sha(exe),render_seconds=float(progress[-1]),wall_seconds=time.monotonic()-started,
        mean_transmittance=float(rgb.mean()),warnings=warnings,finished_at=now(),
        sha256={name:sha(d/name) for name in FILES})
    save(d/'result.json',row)
    print('DONE',stage,'gpis_global',seed,row['render_seconds'],'s; T',row['mean_transmittance'],flush=True)
    return verified(args,stage,'gpis_global',seed)


def link_parent(args,complete,aggregate):
    path=args.base/'comparison.html';text=path.read_text(encoding='utf-8')
    link=os.path.relpath(args.output/'comparison.html',args.base).replace('\\','/')
    rows=''.join(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td><td>{r["average_render_seconds"]:.2f} s</td></tr>' for r in aggregate)
    block=BEGIN+f'<section id="gpis-global-comparison"><h2>新增 GPIS Global：步长 0.0025</h2><div class="note">'+('已完成' if complete else '进行中')+f' · 64×64、1024 spp、两个种子。Global 使用三维稀疏卷积场。<br><a href="{link}">查看三种模式原图、FLIP 和透射率</a></div>'
    if rows:block+='<table><tr><th>方法</th><th>两个种子的重叠区透射率均值</th><th>平均时间</th></tr>'+rows+'</table>'
    for seed in SEEDS:
        image=args.output/'figures'/f'seed{seed}'/'appearance.png'
        if image.exists():
            relative=os.path.relpath(image,args.base).replace('\\','/')
            block+=f'<p>Seed {seed}</p><a href="{link}"><img src="{relative}"></a>'
    block+='</section>'+END
    if BEGIN in text:
        a=text.index(BEGIN);b=text.index(END,a)+len(END);text=text[:a]+block+text[b:]
    else:
        a=text.index('</h1>')+len('</h1>');text=text[:a]+block+text[a:]
    path.write_text(text,encoding='utf-8')
    verification=load(args.base/'verification.json')
    verification.update(html_sha256=sha(path),local_links=verify_links(path),
        gpis_global_comparison=dict(complete=complete,page=link,
            verification=link.replace('comparison.html','verification.json')))
    save(args.base/'verification.json',verification)


def report(args):
    sources=[];pairs=[];sections=[]
    for stage in ('final','smoke'):
        size,spp=QUALITY[stage];masks=roi_masks(size)
        for _,seed in jobs(stage):
            available=[m for m in MODES if (folder(args,stage,m,seed)/'result.json').exists()]
            fd=args.output/'figures'/(f'seed{seed}' if stage=='final' else f'smoke_seed{seed}')
            images=[];labels=[];rows=''
            for mode in available:
                d=folder(args,stage,mode,seed);row=verified(args,stage,mode,seed)
                rgb=pfm(d/'render.pfm')@Y
                roi={name:float(rgb[mask].mean()) for name,mask in masks.items()}
                sources.append(dict(stage=stage,mode=mode,seed=seed,roi=roi,render_seconds=row['render_seconds'],result=url(args,d/'result.json')))
                images.append(d/'render.png');labels.append(LABELS[mode]+f'\n{row["render_seconds"]:.2f} s')
                rows+=f'<tr><td>{LABELS[mode]}</td><td>{roi["overlap"]:.6f}</td><td>{roi["left_single"]:.6f}</td><td>{roi["right_single"]:.6f}</td><td>{row["render_seconds"]:.2f} s</td><td><a href="{url(args,d/"render.png")}">PNG</a> · <a href="{url(args,d/"scene.json")}">配置</a> · <a href="{url(args,d/"render.log")}">日志</a></td></tr>'
            section=f'<h2>{stage} · {size}×{size} · {spp} spp · seed {seed}</h2>'+picture_grid(args,images,labels,fd/'appearance.png',3)
            section+='<table><tr><th>方法</th><th>重叠区透射率</th><th>左侧单层区</th><th>右侧单层区</th><th>时间</th><th>文件</th></tr>'+rows+'</table>'
            maps=[];map_labels=[]
            if 'gpis_global' in available:
                for mode in MODES[:2]:
                    pair=compare(args,seed,folder(args,stage,'gpis_global',seed),folder(args,stage,mode,seed),stage+'_'+mode+'_vs_global')
                    pair.update(stage=stage,reference='gpis_global',test=mode);pairs.append(pair)
                    maps.append(args.output/pair['error_map']);map_labels.append(LABELS[mode]+' vs Global\n'+f'FLIP {pair["flip"]["mean"]:.6f}')
            native_path=args.base/'flip'/stage/f'seed{seed}'/'result.json'
            native=load(native_path)
            pair=dict(stage=stage,seed=seed,reference='gpis_renewal_plus',test='global_conditional',
                flip=native['statistics'],native_record=url(args,native_path),error_map=url(args,native_path.parent/native['error_map']))
            pairs.append(pair);maps.append(args.output/pair['error_map'])
            map_labels.append('start_conditioning vs Renewal+\n'+f'FLIP {pair["flip"]["mean"]:.6f}')
            section+='<h3>原始分辨率上的本地 FLIP</h3>'+picture_grid(args,maps,map_labels,fd/'flip.png',3)
            fig,ax=plt.subplots(figsize=(10,3.4))
            for mode in available:ax.plot((pfm(folder(args,stage,mode,seed)/'render.pfm')@Y)[size//2],label=LABELS[mode])
            ax.set(xlabel='Pixel x',ylabel='Linear transmittance',ylim=(0,1.03),title='Center-row profile')
            ax.legend();fig.tight_layout();fig.savefig(fd/'profile.png',dpi=130);plt.close(fig)
            section+=f'<img src="{url(args,fd/"profile.png")}">';sections.append(section)
    aggregate=[]
    for mode in MODES:
        rows=[r for r in sources if r['stage']=='final' and r['mode']==mode]
        if len(rows)==len(SEEDS):aggregate.append(dict(mode=mode,
            overlap_transmittance=float(np.mean([r['roi']['overlap'] for r in rows])),
            single_transmittance=float(np.mean([r['roi'][name] for r in rows for name in ('left_single','right_single')])),
            average_render_seconds=float(np.mean([r['render_seconds'] for r in rows]))))
    complete=sum(r['stage']=='final' and r['mode']=='gpis_global' for r in sources)==len(SEEDS)
    save(args.output/'metrics.json',dict(complete=complete,sources=sources,comparisons=pairs,aggregate=aggregate))
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['mode','overlap_transmittance','single_transmittance','average_render_seconds'])
        writer.writeheader();writer.writerows(aggregate)
    rows=''.join(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td><td>{r["single_transmittance"]:.6f}</td><td>{r["average_render_seconds"]:.2f} s</td></tr>' for r in aggregate)
    page=('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双薄板 GPIS Global 比较</title>
<style>body{max-width:1250px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}h2{margin-top:32px}</style>
<h1>双薄板：GPIS Global、Renewal+ 与 start_conditioning</h1>
<div class="note">64×64、1024 spp、两个原有种子。GPIS 两种模式均使用步长 0.0025、impulse density=10。<br>Global：3D sparse convolution、global context、UNI；Renewal+：原始 1D sparse convolution、MIS、横向梯度相关。<br>使用同一个冻结渲染程序，共享 GP 均值、SE 协方差（σ=0.02、ℓ=0.2）、完全吸收材质、相机和背光。start_conditioning 与 Renewal+ 原图保持不变。</div>
<p>能到达背光的路径都没有 GP 碰撞；该实验不触发反射后的记忆更新。Global 与原 Renewal+ 的比较同时涉及 3D / 1D 稀疏噪声近似，不能把它们之间的全部差异归因于记忆模型。有限步长和有限采样均有误差，Global 图也不是精确 GP 真值。</p>
<p>透射率来自线性 PFM；显示采用同一 filmic，不降噪、不逐图曝光。小图仅最近邻放大；本地 FLIP 使用原始分辨率、67 PPD、统一 0–1 原生色标。</p>
<table><tr><th>方法</th><th>重叠区透射率均值</th><th>两侧单层区均值</th><th>平均渲染时间</th></tr>'''+rows+'</table>'+
        f'<p>Global 每种子 {args.threads} 线程，最多 {args.parallel_jobs} 个任务并行；原基线各 32 线程单任务。上述耗时是实际运行记录，不作为受控的性能比较。</p>'+
        f'<p><a href="{url(args,args.base/"comparison.html")}">原比较页</a> · <a href="manifest.json">配置与版本</a> · <a href="metrics.json">指标</a> · <a href="summary.csv">CSV</a> · <a href="progress.json">进度</a></p>'+''.join(sections)+'</html>')
    (args.output/'comparison.html').write_text(page,encoding='utf-8')
    save(args.output/'verification.json',dict(passed=True,complete=complete,render_records=len(sources),
        native_flip_records=len(pairs),matched_scene_except_declared_sampling_settings=True,
        original_outputs_preserved=True,local_links=verify_links(args.output/'comparison.html'),
        html_sha256=sha(args.output/'comparison.html'),metrics_sha256=sha(args.output/'metrics.json')))
    link_parent(args,complete,aggregate)
    print('REPORT',len(sources),'images;',len(pairs),'FLIP records',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','smoke','run','report','status'))
    parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--threads',type=int,default=16);parser.add_argument('--parallel-jobs',type=int,choices=(1,2),default=2)
    parser.add_argument('--flip',type=Path,default=DEFAULT_FLIP)
    args=parser.parse_args();args.base=args.base.resolve();args.output=args.output.resolve()
    assert args.threads>0 and args.threads*args.parallel_jobs<=os.cpu_count() and args.flip.is_file()
    if args.command=='status':print(json.dumps(load(args.output/'progress.json'),indent=2));return
    prepare(args)
    if args.command in ('prepare','report'):status(args);report(args);return
    lock=args.output/'runner.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:json.dump(dict(pid=os.getpid(),started_at=now()),stream)
    try:
        for stage in (('smoke',) if args.command=='smoke' else ('smoke','final')):
            if len(jobs(stage))==1 or args.parallel_jobs==1:
                for job in jobs(stage):render(args,*job);status(args);report(args)
            else:
                with ThreadPoolExecutor(max_workers=args.parallel_jobs) as pool:
                    pending={pool.submit(render,args,*job,False):job for job in jobs(stage)}
                    while pending:
                        status(args,active=[str(folder(args,job[0],'gpis_global',job[1])) for job in pending.values()])
                        done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                        for future in done:
                            future.result();pending.pop(future);report(args)
                status(args)
        print('COMPLETE',args.command,flush=True)
    except Exception as exc:
        failure=dict(error=str(exc),traceback=traceback.format_exc());save(args.output/'failure.json',failure)
        status(args,failure=failure);raise
    finally:lock.unlink()


if __name__=='__main__':main()
