"""Restore finite plane-path estimates discarded by the film's 65536 threshold.

Replay every sample in affected pixels through the same production tracer,
including an unaffected control pixel. Preserve all original PFM files and
verify that applying the old clamp reconstructs their means and variances
before writing the unclamped estimates. No transport or sampling is changed.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from run import ROOT, load, save, sha, pfm

LAYERS = ('order00', 'order01', 'order02', 'order03', 'order04', 'order05plus', 'multiple', 'beauty')


def write_pfm(path, values):
    h,w,_=values.shape
    with path.open('wb') as stream:
        stream.write(f'PF\n{w} {h}\n-1\n'.encode('ascii'))
        stream.write(np.flipud(values).astype('<f4').tobytes())


def restore_finite_clamps(folder):
    folder=Path(folder)
    record=folder/'finite_clamp_restoration.json'
    if record.exists():
        restored=load(record)
        assert restored['passed']
        for name,digest in restored['output_sha256'].items():
            assert sha(folder/name)==digest
        return restored
    ends=pfm(folder/'termination_fractions.pfm')
    affected=[(int(x),int(y)) for y,x in np.argwhere(ends[:,:,2]>0)]
    if not affected:
        return None
    scene=load(folder/'scene.json')
    assert scene['primitives'][1]['type']=='point'
    assert len(scene['primitives'])==2 and scene['primitives'][0]['bsdf']['type']=='forward'
    result=load(folder/'result.json')
    spp=result['samples']; h,w,_=ends.shape
    diagnostic=folder/'finite_clamp_replay'
    original=diagnostic/'original'
    names=[f'{part}{suffix}.pfm' for part in LAYERS for suffix in ('','_variance')]+['termination_fractions.pfm']
    original.mkdir(parents=True,exist_ok=True)
    if not (diagnostic/'original.json').exists():
        for name in names:
            assert not (original/name).exists(), 'Partial backup requires review'
            (original/name).write_bytes((folder/name).read_bytes())
        save(diagnostic/'original.json',dict(files={name:sha(original/name) for name in names}))
    input_hashes=load(diagnostic/'original.json')['files']
    for name,digest in input_hashes.items():
        assert sha(original/name)==digest
    raw={name:pfm(original/name) for name in names}
    # A nearby unaffected pixel independently checks event-to-AOV reconstruction.
    control=next((x,y) for y in range(h) for x in range(w) if (x,y) not in affected)
    pixels=affected+[control]
    spec=dict(scene=str(folder/'scene.json'),seed=result['spec']['seed'],samples=spp,
        pixels=[dict(xy=[x,y],label='restore' if (x,y) in affected else 'control') for x,y in pixels])
    save(diagnostic/'spec.json',spec)
    exe=ROOT/'build/Release/surface_path_trace.exe'
    with (diagnostic/'replay.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(exe),str(diagnostic/'spec.json'),str(diagnostic/'replay.json')],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    replay=load(diagnostic/'replay.json')
    assert replay['observer_mismatches']==0 and replay['verified_samples']==spp*len(pixels)
    checks=[]; restored_paths=[]
    for x,y in pixels:
        paths=sorted((p for p in replay['paths'] if p['pixel']==[x,y]),key=lambda p:p['sample'])
        assert [p['sample'] for p in paths]==list(range(spp))
        values=np.zeros((spp,len(LAYERS),3))
        clamped=np.array([p['renderer_clamped'] for p in paths],dtype=bool)
        assert int(clamped.sum())==int(round(raw['termination_fractions.pfm'][y,x,2]*spp))
        for index,p in enumerate(paths):
            assert not p['termination'].startswith(('nonfinite','runtime_error'))
            radiance=np.asarray(p['unclamped_radiance'])
            assert np.isfinite(radiance).all() and np.all(radiance>=0)
            assert bool(clamped[index])==bool(radiance.max()>65536)
            for event in p['events']:
                if event['type']=='direct_estimate':
                    assert event['surface_reflections_before']==0
                    order=event['medium_scatter_order']
                    values[index,min(order,5)]+=event['emission_added']
                elif 'emission_added' in event:
                    assert np.all(np.asarray(event['emission_added'])==0), 'Only nonemissive plane transport is supported'
            values[index,6]=values[index,2:6].sum(axis=0)
            values[index,7]=radiance
            assert np.allclose(values[index,:6].sum(axis=0),radiance,rtol=2e-6,atol=2e-7)
            if clamped[index]:
                restored_paths.append(dict(pixel=[x,y],sample=p['sample'],radiance=radiance.tolist(),
                    orders=values[index,:6].tolist(),termination=p['termination']))
        old_values=values.copy(); old_values[clamped]=0
        old_mean=old_values.mean(axis=0); old_variance=old_values.var(axis=0,ddof=1)
        new_mean=values.mean(axis=0); new_variance=values.var(axis=0,ddof=1)
        for k,part in enumerate(LAYERS):
            for suffix,old,new in (('',old_mean,new_mean),('_variance',old_variance,new_variance)):
                name=part+suffix+'.pfm'
                stored=raw[name][y,x].copy()
                assert np.allclose(old[k],stored,rtol=3e-6,atol=2e-7), (x,y,name,old[k],stored)
                checks.append(dict(pixel=[x,y],file=name,max_error=float(np.max(np.abs(old[k]-stored)))))
                if (x,y) in affected:
                    raw[name][y,x]=new[k]
        if (x,y) in affected:
            raw['termination_fractions.pfm'][y,x,2]=0
    for name in names:
        write_pfm(folder/name,raw[name])
    restored=dict(passed=True,method='Full production-path replay of affected pixels; all finite estimates retained',
        reason='The ordinary film discards a complete path when radiance exceeds 65536',
        restored_paths=restored_paths,affected_pixels=[list(p) for p in affected],control_pixel=list(control),
        verified_samples=replay['verified_samples'],old_clamped_moments_verified=checks,
        source_sha256=sha(Path(__file__)),replay_executable_sha256=sha(exe),
        core_sha256=sha(ROOT/'build/Release/core.lib'),replay_sha256=sha(diagnostic/'replay.json'),
        original_sha256=input_hashes,output_sha256={name:sha(folder/name) for name in names})
    save(record,restored)
    print('Restored',len(restored_paths),'finite high-radiance paths in',len(affected),'pixels',flush=True)
    return restored
