"""Collect reproducible numerical checks for the completed plane experiment.

Prerequisites beyond plane.py's pilot and final runs:
  plane.py render --stage validation --size 128 --spp 32 --cases same_80 --components total single multiple_check
  plane.py render --stage density_half --size 256 --spp 128 --dy .00125 --modes classic_local --components total
  plane.py render --stage validation_step_half --size 128 --spp 32 --step .00125 --cases same_80 --modes sparse_3d_global --components total
  plane.py render --stage validation_domain --size 128 --spp 32 --extent 3 --cases same_80 --modes global_conditional sparse_3d_global --components total
  plane.py render --stage validation_bright --size 128 --spp 32 --cases opposite_80 --modes sparse_3d_global --components total
  plane.py render --stage validation_bright_step_half --size 128 --spp 32 --step .00125 --cases opposite_80 --modes sparse_3d_global --components total

These checks measure sensitivity, not convergence to an exact Gaussian field.
"""
from __future__ import annotations
import copy
import math
from pathlib import Path
import numpy as np
from plane import ROOT,OUT,CASES,RENDER_MODES,folder,load,save,pfm,sha,check_decomposition


def compare(a,b):
    paths=[a,b]
    image=[pfm(p/'render.pfm')[:,:,0] for p in paths]
    if image[0].shape!=image[1].shape: raise RuntimeError('Unmatched resolutions')
    delta=image[1]-image[0]
    halves=[[pfm(p/f'samples{c}.pfm')[:,:,0] for c in ('A','B')] for p in paths]
    diff_a=halves[1][0]-halves[0][0];diff_b=halves[1][1]-halves[0][1]
    se=float(np.sqrt(np.sum(.25*(diff_a-diff_b)**2))/delta.size)
    mean=float(delta.mean());base=float(image[0].mean())
    return dict(baseline=str(a),refined=str(b),baseline_mean=base,refined_mean=float(image[1].mean()),
        difference=mean,relative_change=mean/base if base else None,
        paired_standard_error=se,within_approximate_95_percent_sampling_interval=abs(mean)<=1.96*se,
        difference_standard_errors=mean/se if se else None,
        maximum_pixel_difference=float(np.abs(delta).max()),
        source_sha256=[sha(p/'render.pfm') for p in paths])


def main():
    check_decomposition('pilot')
    check_decomposition('validation',cases=('same_80',))
    matched=[]
    for case in CASES:
        scenes=[load(folder('final',case,m,'total')/'scene.json') for m in RENDER_MODES]
        for key in ('camera','primitives','integrator','renderer'):
            values=[copy.deepcopy(s[key]) for s in scenes]
            if key=='renderer':
                for v in values: v.pop('output_directory')
            assert all(v==values[0] for v in values[1:]),(case,key)
        for i,s in enumerate(scenes):
            m=s['media'][0]
            cov=m['covariance'] if i==0 else m['gaussian_process']['covariance']
            assert cov==dict(type='squared_exponential',sigma=.01,lengthScale=.025)
            material=m['phase_function']['bsdf'] if i==3 else m['material']
            assert material==dict(type='mirror',albedo=.9)
            if i>0: assert m['gaussian_process']['mean']==dict(type='linear',reference_point=[0,0,0],direction=[0,1,0],scale=1)
        theta=math.radians(int(case.split('_')[1]));v=np.array([math.sin(theta),math.cos(theta),0])
        source=np.array(scenes[0]['primitives'][1]['transform']['position'])
        assert source[1]>0 and np.isclose(np.linalg.norm(source),3)
        assert np.isclose(source[1]/3,v[1])
        assert np.sign(source[0])==(1 if case.startswith('same') else -1)
        # All corner rays intersect the mean plane well inside the domain.
        forward=-v;up=np.array([-v[1],v[0],0]);right=np.array([0,0,1.])
        points=[]
        for x in (-1,1):
            for y in (-1,1):
                w=forward+math.tan(math.radians(4))*(x*right+y*up)
                p=v-v[1]/w[1]*w;points.append(p.tolist())
                assert max(abs(p[0]),abs(p[2]))<1.5
        matched.append(dict(case=case,scene_sha256=[sha(folder('final',case,m,'total')/'scene.json') for m in RENDER_MODES],
                            camera_mean_plane_corner_positions=points))
    density=[dict(case=c,**compare(folder('final',c,'classic_local','total'),
                                   folder('density_half',c,'classic_local','total'))) for c in CASES]
    step=[dict(case='same_80',**compare(folder('validation','same_80','sparse_3d_global','total'),
                 folder('validation_step_half','same_80','sparse_3d_global','total'))),
          dict(case='opposite_80',**compare(folder('validation_bright','opposite_80','sparse_3d_global','total'),
                 folder('validation_bright_step_half','opposite_80','sparse_3d_global','total')))]
    domain=[dict(mode=m,**compare(folder('validation','same_80',m,'total'),
                                  folder('validation_domain','same_80',m,'total')))
            for m in ('global_conditional','sparse_3d_global')]
    report=dict(matched_scenes=matched,density_spacing_half=density,gpis_step_half=step,horizontal_extent_doubled=domain,
        decomposition='24 pilot cases and four higher-sample grazing cases passed',
        limitations='Sampling intervals only. Finite impulse density/kernel support remain approximations; a step-size pair and a domain-size pair are sensitivity checks, not a proof of convergence.')
    save(OUT/'validation.json',report)
    # Refresh per-render timings/provenance after any independently revalidated
    # component; the outer batch may have retained an earlier result in memory.
    batch=OUT/'final/run.json'
    if batch.exists():
        run=load(batch)
        for source in run['sources']:
            source['result']=load(Path(source['folder'])/'result.json')
        save(batch,run)
    print('Matched scene groups:',len(matched))
    print('Density spacing changes:',[round(100*r['relative_change'],4) for r in density],'%')
    print('GPIS step changes:',[(r['case'],100*r['relative_change'],r['paired_standard_error']) for r in step])
    print('Domain changes:',[(r['mode'],100*r['relative_change']) for r in domain],'%')


if __name__=='__main__': main()
