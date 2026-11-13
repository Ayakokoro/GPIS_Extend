"""Render matched Renewal+ step refinements of the existing two-slab scene.

Only step_size and output_directory change. Reuse the original frozen renderer,
64x64/1024 spp baselines and both original seeds. Use the local plot Python.
"""
from __future__ import annotations

import argparse
import copy
import csv
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
from figures import Image, plt
from run import ROOT, load, sha, pfm
from gpis_sphere_small_light import save, now, FATAL, FILES, Y
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links
from gpis_shaderball_parameter_comparison import grid
from gpis_two_slabs import roi_masks

BASE=ROOT/'outputs/gpis_two_slabs_compare_20261113'
OUT=BASE/'renewal_step_refinement'
STEPS=(.0025,.00125,.000625)
SEEDS=(20261024,20261113)
PPD=67.
BEGIN='<!-- BEGIN TWO SLABS STEP REFINEMENT -->'
END='<!-- END TWO SLABS STEP REFINEMENT -->'


def key(step):return 'step'+str(step).replace('.','p')


def original(args,mode,seed):return args.base/'final'/mode/f'seed{seed}'


def folder(args,step,seed):
    if step==STEPS[0]:return original(args,'gpis_renewal_plus',seed)
    return args.output/key(step)/f'seed{seed}'


def scene(args,step,seed):
    source=original(args,'gpis_renewal_plus',seed)
    old=load(source/'scene.json');value=copy.deepcopy(old)
    value['media'][0]['step_size']=step
    value['renderer']['output_directory']=str(folder(args,step,seed))
    check=copy.deepcopy(value)
    check['media'][0]['step_size']=old['media'][0]['step_size']
    check['renderer']['output_directory']=old['renderer']['output_directory']
    assert check==old
    assert value['camera']['resolution']==[64,64] and value['renderer']['spp']==1024
    return value


def prepare(args):
    args.output.mkdir(parents=True,exist_ok=True)
    base_manifest=load(args.base/'manifest.json')
    for name,digest in base_manifest['runtime_sha256'].items():
        assert sha(args.base/'renderer'/name)==digest
    dest=args.output/'manifest.json'
    if dest.exists():
        manifest=load(dest)
        assert manifest['runner_sha256']==sha(Path(__file__)),'Runner changed since preparation'
        assert manifest['steps']==list(STEPS) and manifest['seeds']==list(SEEDS)
        for filename,digest in manifest['preserved_original_sha256'].items():
            assert sha(args.base/filename)==digest,filename
    else:
        history=args.output/'history';history.mkdir(exist_ok=True)
        for filename in ('manifest.json','metrics.json','progress.json','verification.json','comparison.html'):
            shutil.copy2(args.base/filename,history/filename)
        preserved={}
        for mode in ('global_conditional','gpis_renewal_plus'):
            for seed in SEEDS:
                d=original(args,mode,seed);row=load(d/'result.json')
                assert row['renderer_sha256']==base_manifest['runtime_sha256']['tungsten.exe']
                for filename,digest in row['sha256'].items():
                    assert sha(d/filename)==digest
                    preserved[(d/filename).relative_to(args.base).as_posix()]=digest
                preserved[(d/'result.json').relative_to(args.base).as_posix()]=sha(d/'result.json')
        preserved['metrics.json']=sha(args.base/'metrics.json')
        preserved['manifest.json']=sha(args.base/'manifest.json')
        for native in (args.base/'flip').glob('*/seed*/result.json'):
            row=load(native)
            for filename,digest in row['sha256'].items():
                assert sha(native.parent/filename)==digest
                preserved[(native.parent/filename).relative_to(args.base).as_posix()]=digest
            preserved[native.relative_to(args.base).as_posix()]=sha(native)
        manifest=dict(created_at=now(),base=str(args.base),steps=STEPS,seeds=SEEDS,
            resolution=[64,64],spp=1024,threads=args.threads,
            runner_sha256=sha(Path(__file__)),renderer=str(args.base/'renderer/tungsten.exe'),
            renderer_sha256=base_manifest['runtime_sha256']['tungsten.exe'],
            flip=dict(path=str(args.flip),sha256=sha(args.flip),ppd=PPD),
            preserved_original_sha256=preserved,
            notes=['Only Renewal+ step_size and output_directory change.',
                'Same 1D sparse-convolution MIS, impulse_density=10, material, camera, light and seeds.',
                'The 0.0025 and start_conditioning images are verified original results.',
                'Equal seeds do not guarantee identical random-number consumption or nested marching lattices.',
                'This isolates step sensitivity, not sparse-convolution density convergence.'])
        save(dest,manifest);shutil.copy2(Path(__file__),args.output/Path(__file__).name)
    assert sha(args.flip)==manifest['flip']['sha256']
    for step in STEPS[1:]:
        for seed in SEEDS:
            path=folder(args,step,seed)/'scene.json';value=scene(args,step,seed)
            if path.exists():assert load(path)==value
            else:save(path,value)
    return manifest


def verified(args,step,seed):
    d=folder(args,step,seed);row=load(d/'result.json')
    assert row['scene']==scene(args,step,seed)
    assert row['seed']==seed and row['spp']==1024 and row['resolution']==[64,64]
    assert row['renderer_sha256']==load(args.output/'manifest.json')['renderer_sha256']
    assert not row['warnings']
    for filename,digest in row['sha256'].items():assert sha(d/filename)==digest
    return row


def status(args,active=None,failure=None):
    jobs=[]
    for step in STEPS[1:]:
        for seed in SEEDS:
            d=folder(args,step,seed)
            text=(d/'render.log').read_text(encoding='utf-8',errors='replace') if (d/'render.log').exists() else ''
            samples=re.findall(r'Completed (\d+)/(\d+) spp',text)
            jobs.append(dict(step=step,seed=seed,state='complete' if (d/'result.json').exists()
                else 'running' if str(d)==active else 'pending',spp=int(samples[-1][0]) if samples else 0,target_spp=1024))
    save(args.output/'progress.json',dict(updated_at=now(),active=active,jobs=jobs,
        completed=sum(j['state']=='complete' for j in jobs),total=len(jobs),
        complete=all(j['state']=='complete' for j in jobs),failure=failure))


def render(args,step,seed):
    d=folder(args,step,seed)
    if (d/'result.json').exists():return verified(args,step,seed)
    assert not (d/'render.log').exists(),'Unverified previous attempt: '+str(d)
    manifest=load(args.output/'manifest.json');exe=args.base/'renderer/tungsten.exe'
    assert sha(exe)==manifest['renderer_sha256']
    command=[str(exe),'-t',str(args.threads),'-s',str(seed),'-d',str(d),str(d/'scene.json')]
    print('START',step,seed,flush=True);started=time.monotonic()
    with (d/'render.log').open('wb') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0)|getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
        save(d/'job.json',dict(pid=process.pid,command=command,started_at=now()))
        try:
            while process.poll() is None:
                status(args,str(d))
                try:process.wait(timeout=15)
                except subprocess.TimeoutExpired:pass
        except BaseException:process.terminate();process.wait();raise
    text=(d/'render.log').read_text(encoding='utf-8',errors='replace')
    assert process.returncode==0 and 'Finished render.' in text and not FATAL.search(text),str(d/'render.log')
    assert re.search(r'Completed\s+1024/1024\s+spp',text)
    warnings=[line for line in text.splitlines() if 'sanity check failed' in line]
    assert not warnings,warnings
    rgb,a,b,var=[pfm(d/name) for name in ('render.pfm','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')]
    assert rgb.shape==var.shape==(64,64,3) and rgb.max()>0
    assert np.allclose((a+b)/2,rgb,rtol=4e-6,atol=4e-7)
    assert np.max(np.abs(rgb*1024-np.round(rgb*1024)))<2e-3
    assert np.allclose(var*1023,rgb*(1-rgb),rtol=2e-4,atol=2e-6)
    progress=re.findall(r'Completed\s+\d+/\d+\s+spp,\s*([\d.]+)/[\d.]+\s+sec',text)
    row=dict(scene=scene(args,step,seed),step=step,seed=seed,spp=1024,resolution=[64,64],threads=args.threads,
        renderer_sha256=sha(exe),render_seconds=float(progress[-1]),wall_seconds=time.monotonic()-started,
        warnings=warnings,finished_at=now(),mean_transmittance=float(rgb.mean()),
        sha256={name:sha(d/name) for name in FILES})
    save(d/'result.json',row)
    print('DONE',step,seed,row['render_seconds'],'s; T',row['mean_transmittance'],flush=True)
    return verified(args,step,seed)


def url(args,path):return os.path.relpath(path,args.output).replace('\\','/')


def picture_grid(args,paths,labels,target,columns):
    target.parent.mkdir(parents=True,exist_ok=True);display=[]
    for i,path in enumerate(paths):
        dest=target.with_name(target.stem+f'_display{i}.png')
        with Image.open(path) as im:im.convert('RGB').resize((256,256),Image.Resampling.NEAREST).save(dest)
        display.append(dest)
    grid(display,labels,target,columns=columns)
    link=url(args,target)
    return f'<a href="{link}"><img src="{link}"></a>'


def compare(args,seed,reference,test,label):
    dest=args.output/'flip'/f'seed{seed}'/label
    result=evaluate(args.flip,reference/'render.png',test/'render.png',dest,PPD)
    difference=(pfm(test/'render.pfm')-pfm(reference/'render.pfm'))@Y
    half=.5*((pfm(test/'samplesA.pfm')-pfm(reference/'samplesA.pfm'))@Y-
             (pfm(test/'samplesB.pfm')-pfm(reference/'samplesB.pfm'))@Y)
    return dict(seed=seed,label=label,flip=result['statistics'],error_map=url(args,dest/result['error_map']),
        native_record=url(args,dest/'result.json'),linear_difference_rms=float(np.sqrt(np.mean(difference**2))),
        estimated_difference_noise_rms=float(np.sqrt(np.mean(half**2))),
        changed_ldr_pixels=int(np.count_nonzero(np.any(np.asarray(Image.open(test/'render.png').convert('RGB'))!=
            np.asarray(Image.open(reference/'render.png').convert('RGB')),axis=2))))


def link_parent(args,complete):
    path=args.base/'comparison.html';text=path.read_text(encoding='utf-8')
    relative=os.path.relpath(args.output/'comparison.html',args.base).replace('\\','/')
    block=BEGIN+f'<div class="note"><b>Renewal+ 步长复核：'+('已完成' if complete else '进行中')+f'</b> · 0.0025 / 0.00125 / 0.000625 · 64×64、1024 spp、两个种子。<br><a href="{relative}">查看原图、步长间 FLIP、透射率与渲染时间</a></div>'+END
    if BEGIN in text:
        a=text.index(BEGIN);b=text.index(END,a)+len(END);text=text[:a]+block+text[b:]
    else:
        index=text.index('</h1>')+len('</h1>');text=text[:index]+block+text[index:]
    path.write_text(text,encoding='utf-8')
    verification=load(args.base/'verification.json')
    verification.update(html_sha256=sha(path),local_links=verify_links(path),
        renewal_step_refinement=dict(complete=complete,page=relative,verification=relative.replace('comparison.html','verification.json')))
    save(args.base/'verification.json',verification)


def report(args):
    sources=[];comparisons=[];sections=[];masks=roi_masks(64)
    for seed in SEEDS:
        available=[s for s in STEPS if (folder(args,s,seed)/'result.json').exists()]
        paths=[];labels=[];rows=''
        gc=original(args,'global_conditional',seed)
        paths.append(gc/'render.png');labels.append('start_conditioning')
        gc_rgb=pfm(gc/'render.pfm')@Y
        for step in available:
            d=folder(args,step,seed);row=verified(args,step,seed)
            rgb=pfm(d/'render.pfm')@Y;base_rgb=pfm(folder(args,STEPS[0],seed)/'render.pfm')@Y
            roi={name:float(rgb[mask].mean()) for name,mask in masks.items()}
            source=dict(seed=seed,step=step,roi=roi,render_seconds=row['render_seconds'],
                mean_transmittance=float(rgb.mean()),overlap_delta_from_baseline=float((rgb-base_rgb)[masks['overlap']].mean()),
                overlap_gap_from_start=float((rgb-gc_rgb)[masks['overlap']].mean()),result=url(args,d/'result.json'))
            sources.append(source);paths.append(d/'render.png');labels.append(f'Renewal+ h={step:g}\n{row["render_seconds"]:.2f} s')
            rows+=f'<tr><td>{step:g}</td><td>{row["render_seconds"]:.2f} s</td><td>{roi["overlap"]:.6f}</td><td>{source["overlap_delta_from_baseline"]:+.6f}</td><td>{source["overlap_gap_from_start"]:+.6f}</td><td><a href="{url(args,d/"render.png")}">PNG</a> · <a href="{url(args,d/"scene.json")}">配置</a> · <a href="{url(args,d/"render.log")}">日志</a></td></tr>'
        fd=args.output/'figures'/f'seed{seed}'
        section=f'<h2>Seed {seed} · 64×64 · 1024 spp</h2>'+picture_grid(args,paths,labels,fd/'appearance.png',4)
        section+=f'<p>start_conditioning 原始重叠区透射率：{float(gc_rgb[masks["overlap"]].mean()):.6f}。</p>'
        section+='<table><tr><th>Renewal+ 步长</th><th>渲染时间</th><th>重叠区透射率</th><th>相对 0.0025 的变化</th><th>与 start_conditioning 的差</th><th>文件</th></tr>'+rows+'</table>'
        step_pairs=[(STEPS[0],s) for s in available[1:]]
        if all(s in available for s in STEPS[1:]):step_pairs.append((STEPS[1],STEPS[2]))
        maps=[];map_labels=[];flip_rows=''
        for reference,test in step_pairs:
            pair=compare(args,seed,folder(args,reference,seed),folder(args,test,seed),key(reference)+'_vs_'+key(test))
            comparisons.append(dict(reference_step=reference,test_step=test,kind='step',**pair))
            maps.append(args.output/pair['error_map']);map_labels.append(f'h={reference:g} vs {test:g}\nFLIP {pair["flip"]["mean"]:.6f}')
            flip_rows+=f'<tr><td>{reference:g} → {test:g}</td><td>{pair["flip"]["mean"]:.6f}</td><td>{pair["linear_difference_rms"]:.6f}</td><td>{pair["estimated_difference_noise_rms"]:.6f}</td><td>{pair["changed_ldr_pixels"]}/4096</td></tr>'
        if maps:
            section+='<h3>Renewal+ 不同步长间的原生 FLIP</h3>'+picture_grid(args,maps,map_labels,fd/'step_flip.png',3)
            section+='<table><tr><th>步长比较</th><th>Mean FLIP</th><th>线性差 RMS</th><th>半采样估计的差噪声 RMS</th><th>PNG 变化像素</th></tr>'+flip_rows+'</table>'
        maps=[];map_labels=[]
        for step in available:
            if step==STEPS[0]:
                dest=args.base/'flip/final'/f'seed{seed}';native=load(dest/'result.json')
                pair=dict(seed=seed,kind='start',step=step,flip=native['statistics'],
                          error_map=url(args,dest/native['error_map']),native_record=url(args,dest/'result.json'))
            else:
                pair=dict(kind='start',step=step,**compare(args,seed,folder(args,step,seed),gc,'start_vs_'+key(step)))
            comparisons.append(pair);maps.append(args.output/pair['error_map']);map_labels.append(f'start vs Renewal+ h={step:g}\nFLIP {pair["flip"]["mean"]:.6f}')
        section+='<h3>start_conditioning 与各步长 Renewal+ 的 FLIP</h3>'+picture_grid(args,maps,map_labels,fd/'start_flip.png',3)
        fig,ax=plt.subplots(figsize=(10,3.6))
        ax.plot(gc_rgb[32],label='start_conditioning',color='black',linestyle='--')
        for step in available:ax.plot((pfm(folder(args,step,seed)/'render.pfm')@Y)[32],label=f'Renewal+ h={step:g}')
        ax.set(xlabel='Pixel x',ylabel='Linear transmittance',ylim=(0,1.03),title='Center-row profile')
        ax.legend();fig.tight_layout();fig.savefig(fd/'profile.png',dpi=130);plt.close(fig)
        profile=url(args,fd/'profile.png');section+=f'<img src="{profile}">';sections.append(section)
    complete=len(sources)==len(STEPS)*len(SEEDS)
    aggregate=[]
    for step in STEPS:
        rows=[r for r in sources if r['step']==step]
        if len(rows)==len(SEEDS):
            aggregate.append(dict(step=step,overlap_transmittance=float(np.mean([r['roi']['overlap'] for r in rows])),
                average_render_seconds=float(np.mean([r['render_seconds'] for r in rows])),
                overlap_delta_from_baseline=float(np.mean([r['overlap_delta_from_baseline'] for r in rows])),
                overlap_gap_from_start=float(np.mean([r['overlap_gap_from_start'] for r in rows]))))
    save(args.output/'metrics.json',dict(complete=complete,sources=sources,comparisons=comparisons,aggregate=aggregate,
        note='Difference maps include Monte Carlo and marching-lattice effects; same seeds do not make samples nested.'))
    with (args.output/'summary.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['step','overlap_transmittance','average_render_seconds','overlap_delta_from_baseline','overlap_gap_from_start'])
        writer.writeheader();writer.writerows(aggregate)
    summary=''.join(f'<tr><td>{r["step"]:g}</td><td>{r["overlap_transmittance"]:.6f}</td><td>{r["overlap_delta_from_baseline"]:+.6f}</td><td>{r["overlap_gap_from_start"]:+.6f}</td><td>{r["average_render_seconds"]:.2f} s</td></tr>' for r in aggregate)
    page=('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>双薄板 Renewal+ 步长复核</title>
<style>body{max-width:1320px;margin:30px auto;padding:0 20px;font:16px/1.65 system-ui;color:#243346;background:#f5f6f8}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}td,th{padding:9px;border-bottom:1px solid #ddd;text-align:left}a{color:#235e91}.note{padding:16px;background:#e7edf5}h2{margin-top:34px}</style>
<h1>双薄板：Renewal+ 步长减半与减至四分之一</h1>
<div class="note">64×64、1024 spp、32 线程，原有两个种子。步长 0.0025 / 0.00125 / 0.000625。<br>仅修改 step_size 和输出目录；使用原实验冻结的同一个渲染程序。1D sparse convolution、MIS、impulse density=10、均值场、协方差、材质、灯光及相机均固定。</div>
<p>原始图不覆盖。原图使用相同 filmic 显示，不降噪、不逐图曝光；64×64 图仅以最近邻放大展示。本地 FLIP 在原始分辨率 PNG 上计算，67 PPD，原生 0–1 色标。</p>
<p>两个种子的 ROI 均值如下；步长改变可能改变步进网格和随机数消耗，不能把逐像素差全部视为确定性的离散误差。该实验不检查 sparse convolution 的密度近似误差。</p>
<table><tr><th>步长</th><th>重叠区透射率</th><th>相对原步长变化</th><th>与 start_conditioning 的差</th><th>平均渲染时间</th></tr>'''+summary+'</table>'+
        f'<p><a href="{url(args,args.base/"comparison.html")}">返回原始双薄板比较</a> · <a href="manifest.json">参数与版本</a> · <a href="metrics.json">指标</a> · <a href="summary.csv">CSV</a> · <a href="progress.json">进度</a></p>'+''.join(sections)+'</html>')
    (args.output/'comparison.html').write_text(page,encoding='utf-8')
    verification=dict(passed=True,complete=complete,render_records=len(sources),new_renders=sum(r['step']!=STEPS[0] for r in sources),
        flip_records=len(comparisons),local_links=verify_links(args.output/'comparison.html'),
        matched_except_step_and_output=True,original_outputs_preserved=True,
        html_sha256=sha(args.output/'comparison.html'),metrics_sha256=sha(args.output/'metrics.json'))
    save(args.output/'verification.json',verification);link_parent(args,complete)
    print('REPORT',len(sources),'images;',len(comparisons),'FLIP records',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('prepare','run','report','status'))
    parser.add_argument('--base',type=Path,default=BASE);parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--threads',type=int,default=32);parser.add_argument('--flip',type=Path,default=DEFAULT_FLIP)
    args=parser.parse_args();args.base=args.base.resolve();args.output=args.output.resolve()
    assert args.threads==32 and args.flip.is_file()
    if args.command=='status':print(json.dumps(load(args.output/'progress.json'),indent=2));return
    prepare(args)
    if args.command in ('prepare','report'):status(args);report(args);return
    lock=args.output/'runner.lock';descriptor=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY)
    with os.fdopen(descriptor,'w') as stream:json.dump(dict(pid=os.getpid(),started_at=now()),stream)
    try:
        report(args)
        for step in STEPS[1:]:
            for seed in SEEDS:
                render(args,step,seed);status(args);report(args)
        print('COMPLETE',flush=True)
    except Exception as exc:
        failure=dict(error=str(exc),traceback=traceback.format_exc());save(args.output/'failure.json',failure)
        status(args,failure=failure);raise
    finally:lock.unlink()


if __name__=='__main__':main()
