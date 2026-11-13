"""Render the native single_history_gpis model in the matched two-slab scene.

Reuse the frozen renderer and all existing references. Run prepare, smoke, run,
report or status with the local plotting Python. No transport code is changed.
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
from figures import plt
from run import ROOT, load, sha, pfm
from gpis_sphere_small_light import save, now, FATAL, FILES, Y
from gpis_shaderball_flip import DEFAULT_FLIP, verify_links
from gpis_two_slabs import roi_masks
from gpis_two_slabs_step_refinement import compare, picture_grid, url

BASE=ROOT/'outputs/gpis_two_slabs_compare_20261113'
OUT=BASE/'single_history_comparison'
MODE='single_history_gpis'
SEEDS=(20261024,20261113)
QUALITY=dict(smoke=(32,8),final=(64,1024))
MODES=('global_conditional',MODE,'gpis_renewal_plus','gpis_global')
LABELS=dict(global_conditional='start_conditioning',single_history_gpis='single_history',
            gpis_renewal_plus='GPIS renewal+ (1D MIS)',gpis_global='GPIS Global (3D)')
REMOVE=('tracking_step','non_gp_surface_policy','fp_gradient','tracking_step_scale',
        'tracking_age_scale','tracking_max_expected_events')
SETTINGS=dict(tracking_method='delta',selection_step_over_length=.5,history_groups=8,
    history_enabled=True,candidate_collection='fine',shadow_roulette_threshold=.1,
    normalized_bound_only=True,far_bound_start_scale=4.,tighten_expected_events=8.,
    probability_method='numeric',probability_absolute_tolerance=1e-9,
    probability_relative_tolerance=2e-4,probability_initial_samples=256,
    probability_max_samples=8192,max_work=1000000,profile=True,profile_detail=False)
BEGIN='<!-- BEGIN TWO SLABS SINGLE HISTORY -->'
END='<!-- END TWO SLABS SINGLE HISTORY -->'


def folder(args,stage,mode,seed):
    if mode==MODE:return args.output/stage/mode/f'seed{seed}'
    if mode=='gpis_global':return args.base/'gpis_global_comparison'/stage/mode/f'seed{seed}'
    return args.base/stage/mode/f'seed{seed}'


def jobs(stage):return [(stage,s) for s in (SEEDS if stage=='final' else SEEDS[:1])]


def scene(args,stage,seed):
    old=load(folder(args,stage,'global_conditional',seed)/'scene.json');value=copy.deepcopy(old)
    medium=value['media'][0];medium['type']=MODE
    for name in REMOVE:medium.pop(name)
    medium['single_history']=copy.deepcopy(SETTINGS)
    value['renderer']['output_directory']=str(folder(args,stage,MODE,seed))
    check=copy.deepcopy(value);check['media']=old['media']
    check['renderer']['output_directory']=old['renderer']['output_directory']
    assert check==old
    assert medium['gaussian_process']==old['media'][0]['gaussian_process']
    assert medium['material']==old['media'][0]['material']
    assert value['camera']['resolution']==[QUALITY[stage][0]]*2 and value['renderer']['spp']==QUALITY[stage][1]
    return value


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    original=load(args.base/'manifest.json')
    for name,digest in original['runtime_sha256'].items():assert sha(args.base/'renderer'/name)==digest
    path=args.output/'manifest.json'
    if path.exists():
        m=load(path)
        assert m['runner_sha256']==sha(Path(__file__)),'Runner changed since setup'
        assert m['settings']==SETTINGS and m['threads']==args.threads and m['parallel_jobs']==args.parallel_jobs
        for name,digest in m['preserved_original_sha256'].items():assert sha(args.base/name)==digest,name
    else:
        history=args.output/'history';history.mkdir(exist_ok=True)
        for name in ('comparison.html','manifest.json','metrics.json','verification.json','progress.json'):
            shutil.copy2(args.base/name,history/name)
        preserved={}
        for stage in QUALITY:
            for _,seed in jobs(stage):
                for mode in MODES:
                    if mode==MODE:continue
                    result=folder(args,stage,mode,seed)/'result.json';r=load(result)
                    for name,digest in r['sha256'].items():
                        assert sha(result.parent/name)==digest
                        preserved[(result.parent/name).relative_to(args.base).as_posix()]=digest
                    preserved[result.relative_to(args.base).as_posix()]=sha(result)
        for sub in ('','gpis_global_comparison/','renewal_step_refinement/'):
            for name in ('manifest.json','metrics.json'):preserved[sub+name]=sha(args.base/(sub+name))
        source=args.output/'source';source.mkdir(exist_ok=True)
        for name in (Path(__file__).name,'gpis_two_slabs_step_refinement.py','gpis_two_slabs.py'):
            shutil.copy2(Path(__file__).with_name(name),source/name)
        for name in ('SingleHistoryFlight.cpp','SingleHistoryFlight.hpp','SparseHistoryGaussian.cpp','SparseHistoryGaussian.hpp'):
            shutil.copy2(ROOT/'src/core/math'/name,source/name)
        for name in ('SingleHistoryGpisMedium.cpp','SingleHistoryGpisMedium.hpp'):
            shutil.copy2(ROOT/'src/core/media'/name,source/name)
        m=dict(created_at=now(),runner_sha256=sha(Path(__file__)),mode=MODE,settings=SETTINGS,
            quality=QUALITY,seeds=SEEDS,threads=args.threads,parallel_jobs=args.parallel_jobs,
            runtime_sha256=original['runtime_sha256'],source_sha256={p.name:sha(p) for p in source.iterdir()},
            preserved_original_sha256=preserved,flip=dict(path=str(args.flip),sha256=sha(args.flip),ppd=67.),
            notes=['Native single_history_gpis, with numerical probabilities and delta tracking.',
                'The example selection settings are retained: 0.5 ell groups and 8 retained candidates; at most one is active.',
                'Exterior birth retains F>0; GP collision births retain value and gradient. This differs from sampled start_conditioning births.',
                'History participates in both collision hazard and gradient law. No transport code is changed.',
                'This mode has no fixed GPIS ray-marching step. Existing GPIS references both use .0025.',
                'All physical scene parameters, image settings, random seeds and the frozen renderer are shared.'])
        save(path,m)
    assert sha(args.flip)==m['flip']['sha256']
    for stage in QUALITY:
        for _,seed in jobs(stage):
            dest=folder(args,stage,MODE,seed)/'scene.json';value=scene(args,stage,seed)
            if dest.exists():assert load(dest)==value
            else:save(dest,value)
    return m


def verified(args,stage,mode,seed):
    d=folder(args,stage,mode,seed);r=load(d/'result.json')
    assert r['renderer_sha256']==load(args.output/'manifest.json')['runtime_sha256']['tungsten.exe']
    if mode==MODE:assert r['scene']==scene(args,stage,seed)
    assert r['spp']==QUALITY[stage][1] and r['seed']==seed and not r['warnings']
    for name,digest in r['sha256'].items():assert sha(d/name)==digest,str(d/name)
    return r


def status(args,active=None,failure=None):
    active_paths={active} if isinstance(active,str) else set(active or [])
    rows=[]
    for stage in QUALITY:
        for _,seed in jobs(stage):
            d=folder(args,stage,MODE,seed)
            text=(d/'render.log').read_text(encoding='utf-8',errors='replace') if (d/'render.log').exists() else ''
            samples=re.findall(r'Completed (\d+)/(\d+) spp',text)
            rows.append(dict(stage=stage,seed=seed,mode=MODE,
                state='complete' if (d/'result.json').exists() else 'running' if str(d) in active_paths else 'pending',
                spp=int(samples[-1][0]) if samples else 0,target_spp=QUALITY[stage][1]))
    final=[r for r in rows if r['stage']=='final']
    save(args.output/'progress.json',dict(updated_at=now(),active=active,jobs=rows,
        complete=all(r['state']=='complete' for r in final),completed=sum(r['state']=='complete' for r in final),
        total=len(final),failure=failure))


def render(args,stage,seed,update_status=True):
    d=folder(args,stage,MODE,seed)
    if (d/'result.json').exists():return verified(args,stage,MODE,seed)
    assert not (d/'render.log').exists(),'Unverified existing attempt: '+str(d)
    exe=args.base/'renderer/tungsten.exe'
    command=[str(exe),'-t',str(args.threads),'-s',str(seed),'-d',str(d),str(d/'scene.json')]
    print('START',stage,MODE,seed,flush=True);started=time.monotonic()
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
    text=(d/'render.log').read_text(encoding='utf-8',errors='replace');size,spp=QUALITY[stage]
    assert p.returncode==0 and 'Finished render.' in text and not FATAL.search(text),str(d/'render.log')
    assert re.search(rf'Completed\s+{spp}/{spp}\s+spp',text)
    warnings=[line for line in text.splitlines() if 'sanity check failed' in line];assert not warnings,warnings
    discarded=re.search(r'GPIS single-history discarded surface births: (\{[^\n]+\})',text)
    assert discarded and json.loads(discarded[1])==dict(paths=0,shadow_connections=0)
    profile=re.search(r'GPIS single-history profile: (\{[^\n]+\})',text)
    assert profile,'No single-history execution counters'
    counters=json.loads(profile[1]);assert counters['exterior_path']['calls']>0 and counters['exterior_path']['selections']>0
    rgb,a,b,var=[pfm(d/name) for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')]
    assert rgb.shape==var.shape==(size,size,3) and rgb.max()>0
    assert np.allclose((a+b)/2,rgb,rtol=4e-6,atol=4e-7)
    assert np.max(np.abs(rgb*spp-np.round(rgb*spp)))<2e-3
    assert np.allclose(var*(spp-1),rgb*(1-rgb),rtol=2e-4,atol=2e-6)
    times=re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/[\d.]+\s+sec',text)
    r=dict(scene=scene(args,stage,seed),seed=seed,spp=spp,resolution=[size,size],threads=args.threads,
        renderer_sha256=sha(exe),render_seconds=float(times[-1]),wall_seconds=time.monotonic()-started,
        warnings=warnings,discarded_surface_births=json.loads(discarded[1]),profile=counters,
        mean_transmittance=float(rgb.mean()),finished_at=now(),sha256={name:sha(d/name) for name in FILES})
    save(d/'result.json',r)
    print('DONE',stage,MODE,seed,r['render_seconds'],'s; T',r['mean_transmittance'],flush=True)
    return verified(args,stage,MODE,seed)


def link_parent(args,complete,aggregate):
    path=args.base/'comparison.html';text=path.read_text(encoding='utf-8')
    link=os.path.relpath(args.output/'comparison.html',args.base).replace('\\','/')
    block=BEGIN+'<section id="single-history-comparison"><h2>新增 single history 双薄板比较</h2><div class="note">'+('已完成' if complete else '进行中')+f' · 64×64、1024 spp、两个种子。原生 single_history_gpis，同时修正消光和梯度。<br><a href="{link}">查看四种方法的原图、透射率和 FLIP</a></div>'
    if aggregate:
        block+='<table><tr><th>方法</th><th>重叠区透射率均值</th></tr>'+''.join(
            f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td></tr>' for r in aggregate)+'</table>'
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
    v=load(args.base/'verification.json')
    v.update(html_sha256=sha(path),local_links=verify_links(path),single_history_comparison=dict(
        complete=complete,page=link,verification=link.replace('comparison.html','verification.json')))
    save(args.base/'verification.json',v)


def report(args):
    sources=[];pairs=[];sections=[]
    for stage in ('final','smoke'):
        size,spp=QUALITY[stage];masks=roi_masks(size)
        for _,seed in jobs(stage):
            available=[m for m in MODES if (folder(args,stage,m,seed)/'result.json').exists()]
            fd=args.output/'figures'/(f'seed{seed}' if stage=='final' else f'smoke_seed{seed}')
            images=[];labels=[];rows=''
            for mode in available:
                d=folder(args,stage,mode,seed);r=verified(args,stage,mode,seed);rgb=pfm(d/'render.pfm')@Y
                roi={name:float(rgb[mask].mean()) for name,mask in masks.items()}
                sources.append(dict(stage=stage,mode=mode,seed=seed,roi=roi,render_seconds=r['render_seconds'],result=url(args,d/'result.json')))
                images.append(d/'render.png');labels.append(LABELS[mode]+f'\n{r["render_seconds"]:.2f} s')
                rows+=f'<tr><td>{LABELS[mode]}</td><td>{roi["overlap"]:.6f}</td><td>{roi["left_single"]:.6f}</td><td>{roi["right_single"]:.6f}</td><td>{r["render_seconds"]:.2f} s</td><td><a href="{url(args,d/"render.png")}">PNG</a> · <a href="{url(args,d/"scene.json")}">配置</a> · <a href="{url(args,d/"render.log")}">日志</a></td></tr>'
            section=f'<h2>{stage} · {size}×{size} · {spp} spp · seed {seed}</h2>'+picture_grid(args,images,labels,fd/'appearance.png',4)
            section+='<table><tr><th>方法</th><th>重叠区透射率</th><th>左侧单层区</th><th>右侧单层区</th><th>耗时</th><th>文件</th></tr>'+rows+'</table>'
            if MODE in available:
                maps=[];map_labels=[]
                for ref in ('gpis_global','gpis_renewal_plus','global_conditional'):
                    pair=compare(args,seed,folder(args,stage,ref,seed),folder(args,stage,MODE,seed),stage+'_single_history_vs_'+ref)
                    pair.update(stage=stage,reference=ref,test=MODE);pairs.append(pair)
                    maps.append(args.output/pair['error_map']);map_labels.append('single_history vs '+LABELS[ref]+f'\nFLIP {pair["flip"]["mean"]:.6f}')
                section+='<h3>single history 对三个原有方法的 FLIP</h3>'+picture_grid(args,maps,map_labels,fd/'flip.png',3)
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
    complete=sum(r['stage']=='final' and r['mode']==MODE for r in sources)==len(SEEDS)
    save(args.output/'metrics.json',dict(complete=complete,sources=sources,comparisons=pairs,aggregate=aggregate))
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['mode','overlap_transmittance','single_transmittance','average_render_seconds'])
        writer.writeheader();writer.writerows(aggregate)
    rows=''.join(f'<tr><td>{LABELS[r["mode"]]}</td><td>{r["overlap_transmittance"]:.6f}</td><td>{r["single_transmittance"]:.6f}</td><td>{r["average_render_seconds"]:.2f} s</td></tr>' for r in aggregate)
    page=('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双薄板 single history 比较</title>
<style>body{max-width:1320px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}h2{margin-top:32px}</style>
<h1>双薄板：single history 与三个原有参考</h1>
<div class="note">64×64、1024 spp、两个原有种子。沿用同一个冻结渲染程序、GP 场（σ=0.02，ℓ=0.2）、完全吸收材质、相机和背光。<br>single_history_gpis：delta tracking、history_enabled=true、selection_step_over_length=0.5、history_groups=8、fine 候选收集、numeric 概率计算；profile=true 记录实际执行。<br>选点组宽度为 0.5ℓ=0.1，保留 8 组候选，当前最多启用一个历史正值约束。它同时参与消光和梯度，不是 fixed5 仅梯度修正。</div>
<p>该模式使用原生历史相关风险率，不使用 GPIS 固定步进步长。Global 与 Renewal+ 引用原有步长 0.0025 的结果，均不重渲染。single history 的外部起点只条件化 F&gt;0；其状态模型也与 start_conditioning 的起点 F、梯度采样不同。</p>
<p>数字来自线性 PFM，显示统一 filmic，不降噪、不逐图曝光。图像仅最近邻放大；本地 FLIP 在原始分辨率上计算，67 PPD，原生 0–1 色标。</p>
<table><tr><th>方法</th><th>重叠区透射率均值</th><th>两侧单层区均值</th><th>平均渲染时间</th></tr>'''+rows+'</table>'+
        f'<p>正式 single history 每种子 {args.threads} 线程，最多 {args.parallel_jobs} 个任务并行；旧参考沿用各自的执行记录，耗时不作为受控性能比较。</p>'+
        f'<p><a href="{url(args,args.base/"comparison.html")}">原比较页</a> · <a href="manifest.json">配置与版本</a> · <a href="metrics.json">指标</a> · <a href="summary.csv">CSV</a> · <a href="progress.json">进度</a></p>'+''.join(sections)+'</html>')
    (args.output/'comparison.html').write_text(page,encoding='utf-8')
    save(args.output/'verification.json',dict(passed=True,complete=complete,render_records=len(sources),
        native_flip_records=len(pairs),shared_physical_scene=True,original_outputs_preserved=True,
        local_links=verify_links(args.output/'comparison.html'),html_sha256=sha(args.output/'comparison.html'),
        metrics_sha256=sha(args.output/'metrics.json')))
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
                        status(args,active=[str(folder(args,j[0],MODE,j[1])) for j in pending.values()])
                        done,_=wait(pending,timeout=15,return_when=FIRST_COMPLETED)
                        for future in done:future.result();pending.pop(future);report(args)
                status(args)
        print('COMPLETE',args.command,flush=True)
    except Exception as exc:
        failure=dict(error=str(exc),traceback=traceback.format_exc());save(args.output/'failure.json',failure)
        status(args,failure=failure);raise
    finally:lock.unlink()


if __name__=='__main__':main()
