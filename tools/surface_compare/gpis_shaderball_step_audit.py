"""Paired path replay to isolate step-size sensitivity at the shaderball base.

python -B tools/surface_compare/gpis_shaderball_step_audit.py

Only step_size and output destination change. Keep the full 256x256 camera,
both original memory models, and identical sample IDs. This is a selected-pixel
diagnostic, not an image-wide convergence test or a correction to either model.
"""
from __future__ import annotations
import copy
import csv
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from run import ROOT, load, save, sha, pfm
from gpis_shaderball_renewal import REPORT, GROUPS, GLOBAL, MODE, SEED, WEIGHTS, original, destination

OUT = REPORT/'step_audit'
EXE = ROOT/'build/Release/surface_path_trace.exe'
STEPS = (.01,.005,.0025)
SAMPLES = 512
PIXELS = [('left_1',(64,174)),('left_2',(68,178)),('left_3',(72,182)),
          ('right_1',(180,174)),('right_2',(184,178)),('right_3',(188,182)),
          ('base_groove',(177,166)),('center_control',(126,82))]


def run(job):
    group,mode,step = job
    source = destination(group,'renewal_full_mis') if mode==MODE else original(group,mode)
    result = load(source/'result.json')
    assert sha(source/'scene.json')==result['sha256']['scene.json']
    old = load(source/'scene.json'); s=copy.deepcopy(old)
    d=OUT/group/mode/str(step).replace('.','p')
    s['media'][0]['step_size']=step
    s['renderer']['output_directory']=str(d)
    check=copy.deepcopy(s)
    check['media'][0]['step_size']=old['media'][0]['step_size']
    check['renderer']=old['renderer']
    assert check==old
    spec=dict(scene=str(d/'scene.json'),seed=SEED,samples=SAMPLES,verify_samples=16,
        pixels=[dict(label=label,xy=list(xy)) for label,xy in PIXELS],export_mean_mesh=False)
    save(d/'scene.json',s);save(d/'spec.json',spec)
    identity=dict(executable_sha256=sha(EXE),spec_sha256=sha(d/'spec.json'),
        scene_sha256=sha(d/'scene.json'),original_result_sha256=sha(source/'result.json'))
    if (d/'validation.json').exists():
        record=load(d/'validation.json')
        assert record['identity']==identity and sha(d/'paths.json')==record['paths_sha256']
    else:
        print('Replay',group,mode,step,flush=True)
        with (d/'replay.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(EXE),str(d/'spec.json'),str(d/'paths.json')],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True,
                creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    replay=load(d/'paths.json')
    assert replay['spec']==spec and replay['observer_mismatches']==0
    assert replay['verified_samples']==len(PIXELS)*16
    assert len(replay['paths'])==len(PIXELS)*SAMPLES
    rgb=np.zeros((len(PIXELS),SAMPLES,3));rejected=np.zeros((len(PIXELS),SAMPLES))
    clamped=0
    for i,(label,xy) in enumerate(PIXELS):
        paths=sorted((p for p in replay['paths'] if p['pixel_label']==label),key=lambda p:p['sample'])
        assert [p['sample'] for p in paths]==list(range(SAMPLES))
        for j,p in enumerate(paths):
            assert p['pixel']==list(xy)
            assert not p['termination'].startswith(('nonfinite','runtime_error'))
            rgb[i,j]=p['radiance']
            rejected[i,j]=p['termination']=='outward_crossing_rejected'
            clamped+=bool(p['renderer_clamped'])
    assert np.isfinite(rgb).all() and rgb.min()>=0 and clamped==0
    error=None
    if step==.01:
        assert sha(source/'render.pfm')==result['sha256']['render.pfm']
        archived=pfm(source/'render.pfm')
        expected=np.array([archived[y,x] for _,(x,y) in PIXELS])
        error=float(np.abs(rgb[:,:128].mean(1)-expected).max())
        assert np.allclose(rgb[:,:128].mean(1),expected,rtol=3e-5,atol=2e-7)
    record=dict(identity=identity,passed=True,paths_sha256=sha(d/'paths.json'),
        archived_128spp_max_absolute_error=error,clamped_samples=clamped,
        observer_verified_samples=replay['verified_samples'])
    save(d/'validation.json',record)
    print('Done',group,mode,step,'mean bottom Y',float((rgb[:7]@WEIGHTS).mean()),flush=True)
    return job,dict(y=rgb@WEIGHTS,rejected=rejected,validation=record)


def estimate(values):
    # Each pixel is a stratum; retain sample pairing before estimating differences.
    return dict(mean=float(values.mean()),se=float(np.sqrt(values.var(1,ddof=1).sum()/values.shape[1])/values.shape[0]))


def main():
    jobs=[(g,m,s) for g in GROUPS for m in (MODE,GLOBAL) for s in STEPS]
    save(OUT/'plan.json',dict(groups=GROUPS,modes=[MODE,GLOBAL],steps=STEPS,samples=SAMPLES,
        seed=SEED,pixels=[dict(label=l,xy=xy) for l,xy in PIXELS],
        note='Locations fixed from the existing bottom discrepancy before any refined-step results.'))
    with ThreadPoolExecutor(max_workers=2) as pool: data=dict(pool.map(run,jobs))
    regions={'bottom_selected':list(range(7)),'left_selected':[0,1,2],
        'right_selected':[3,4,5],'base_groove':[6],'center_control':[7]}
    rows=[];pairs=[]
    for g in GROUPS:
        for name,ids in regions.items():
            for mode in (MODE,GLOBAL):
                base=data[g,mode,.01]['y'][ids]
                for step in STEPS:
                    current=data[g,mode,step]['y'][ids]
                    delta=estimate(current-base)
                    rows.append(dict(group=g,region=name,mode=mode,step=step,
                        **estimate(current),difference_from_0p01=delta['mean'],paired_change_se=delta['se'],
                        relative_change=delta['mean']/float(base.mean()),
                        rejection_fraction=float(data[g,mode,step]['rejected'][ids].mean())))
            initial=data[g,MODE,.01]['y'][ids]-data[g,GLOBAL,.01]['y'][ids]
            for step in STEPS:
                delta=data[g,MODE,step]['y'][ids]-data[g,GLOBAL,step]['y'][ids]
                change=estimate(delta-initial)
                pairs.append(dict(group=g,region=name,step=step,**estimate(delta),
                    gap_change_from_0p01=change['mean'],paired_gap_change_se=change['se']))
    summary=dict(complete=True,scope='Eight selected pixels per group, same 512 sample IDs; seven bottom pixels and one control.',
        settings=load(OUT/'plan.json'),statistics=rows,renewal_minus_global=pairs,
        validations=[dict(group=g,mode=m,step=s,**d['validation']) for (g,m,s),d in data.items()],
        interpretation='SE describes within-pixel Monte Carlo uncertainty. Selected pixels do not establish whole-image convergence. Step refinement also changes subsequent path geometry and RNG consumption.')
    save(OUT/'summary.json',summary)
    for name,records in [('statistics',rows),('renewal_minus_global',pairs)]:
        with (OUT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    for r in rows:
        if r['region']=='bottom_selected': print(r,flush=True)
    for r in pairs:
        if r['region']=='bottom_selected': print('GAP',r,flush=True)


if __name__=='__main__': main()
