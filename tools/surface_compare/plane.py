"""Plane, camera-side/opposite-side illumination, using the production renderer.

  python tools/surface_compare/plane.py setup
  python tools/surface_compare/plane.py render --stage pilot --size 32 --spp 8
  python tools/surface_compare/plane.py render --stage final --size 256 --spp 128
  python tools/surface_compare/plane.py probe --cases same_30 same_80 --path-samples 200000
  <plot-python> tools/surface_compare/plane_figures.py --stage final

All four modes use mu(x,y,z)=y, sigma=.01, ell=.025 and mirror albedo=.9.
The local mode interpolates the baked density; the other three evaluate the
analytic mean. Both lights lie above the plane and are rendered separately.
Total and single use identical per-path seeds. Multiple is their difference,
validated against a separate min_bounces=2 render before final rendering.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from run import ROOT, RENDER_MODES, light, load, medium, pfm, save, sha

OUT = ROOT / 'outputs/plane_conditional_compare'
ANGLES = (30, 60, 80)
SIDES = ('same', 'opposite')
CASES = tuple(f'{side}_{angle}' for side in SIDES for angle in ANGLES)
SEED = 20261024


def field_path(dy=.0025, extent=1.5):
    return OUT / f'fields_dy{dy:g}_extent{extent:g}' / 'plane.nvdb'


def setup(dy=.0025, extent=1.5):
    field = field_path(dy, extent)
    spec = dict(output_directory=str(field.parent), voxel_size=dy,
                voxel_spacing=[.04, dy, .04], sigma=.01, length_scale=.025,
                padding=.08, cases=[dict(name='plane', shape='plane', half_extent=extent)])
    request = field.parent / 'bake.json'
    if request.exists():
        if load(request) != spec:
            raise RuntimeError('Existing incompatible field request: ' + str(request))
        if field.exists() and Path(str(field)+'.json').exists():
            return field
    save(request, spec)
    with (field.parent/'bake.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(ROOT/'build/Release/bake_contact_fields.exe'), str(request)],
                       cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return field


def folder(stage, case, mode, component, seed=SEED):
    return OUT / stage / case / mode / f'seed{seed}' / component


def scene(case, mode, stage, size, spp, component='total', seed=SEED,
          dy=.0025, extent=1.5, step=.0025):
    side, angle = case.split('_')
    angle = int(angle)
    theta = math.radians(angle)
    view = np.array([math.sin(theta), math.cos(theta), 0.])
    illumination = view.copy()
    if side == 'opposite':
        illumination[0] *= -1
    field = field_path(dy, extent)
    meta = load(Path(str(field)+'.json'))
    lo = np.array(meta['process_box']['min'])
    hi = np.array(meta['process_box']['max'])
    m = medium(mode, field, .025, step)
    if mode != 'classic_local':
        m['gaussian_process']['mean'] = dict(type='linear', reference_point=[0, 0, 0],
                                            direction=[0, 1, 0], scale=1)
    output = folder(stage, case, mode, component, seed)
    return dict(media=[m], camera=dict(type='pinhole', resolution=[size, size], fov=8,
        tonemap='filmic', reconstruction_filter='box', transform=dict(position=view.tolist(),
        look_at=[0, 0, 0], up=[0, 1, 0], scale=[-1, 1, 1])),
        integrator=dict(type='path_tracer', min_bounces=2 if component=='multiple_check' else 1,
            max_bounces=2 if component=='single' else 128, enable_light_sampling=True,
            enable_volume_light_sampling=True, enable_consistency_checks=False,
            enable_two_sided_shading=True, use_cuda_gpis=False,
            independent_path_sampling=True, wavefront_gpis=False),
        primitives=[dict(name='processBox', type='cube', int_medium='surface',
            bsdf=dict(type='forward', albedo=1), transform=dict(position=((lo+hi)/2).tolist(),
            scale=(hi-lo).tolist())), light('key', (3*illumination).tolist(), [.45, 1, .45], 30)],
        renderer=dict(spp=spp, spp_step=min(16, spp), adaptive_sampling=False,
            stratified_sampler=False, enable_resume_render=False, overwrite_output_files=True,
            output_buffers=[dict(type='color',hdr_output_file='samples.pfm',
                                 sample_variance=True,two_buffer_variance=True)],
            output_file='render.png', hdr_output_file='render.pfm', output_directory=str(output)))


def render_one(s, threads, seed, provenance):
    output = Path(s['renderer']['output_directory'])
    result = output/'result.json'
    if result.exists():
        old = load(result)
        if old['provenance']==provenance and load(output/'scene.json')==s:
            print('Reuse', output.relative_to(OUT), flush=True)
            return old
        raise RuntimeError('Existing incompatible render: '+str(output))
    if (output/'render.pfm').exists():
        raise RuntimeError('Existing unvalidated render: '+str(output))
    save(output/'scene.json', s)
    start = time.monotonic()
    print('Render', output.relative_to(OUT), flush=True)
    with (output/'render.log').open('w', encoding='utf-8') as log:
        p = subprocess.run([str(ROOT/'build/Release/tungsten.exe'), '-t', str(threads),
                            '-s', str(seed), '-d', str(output), str(output/'scene.json')],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                           creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    log = (output/'render.log').read_text(encoding='utf-8', errors='replace')
    spp = s['renderer']['spp']
    if (p.returncode or 'Finished render.' not in log or
        not re.search(rf'Completed {spp}/{spp} spp', log) or
        re.search(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|'
                  r'Wrong normal direction|Failed to sample gradient|Gradient zero|Conditioning .* error',log,re.I)):
        raise RuntimeError('Failed render: '+str(output))
    rgb = pfm(output/'render.pfm')
    w,h = s['camera']['resolution']
    assert rgb.shape==(h,w,3)
    discarded = re.search(r'discarded surface births: (\{.*\})',log)
    result_data = dict(seconds=time.monotonic()-start, provenance=provenance,
        scene_sha256=sha(output/'scene.json'), pfm_sha256=sha(output/'render.pfm'),
        png_sha256=sha(output/'render.png'), completed_spp=spp, size=[w,h],
        mean_radiance=float(rgb.mean()), maximum=float(rgb.max()),
        discarded_surface_births=json.loads(discarded[1]) if discarded else None)
    save(result, result_data)
    print('Done', output.relative_to(OUT), round(result_data['seconds'],2),
          'mean',round(result_data['mean_radiance'],7),flush=True)
    return result_data


def render(args):
    field = setup(args.dy,args.extent)
    provenance = dict(renderer_sha256=sha(ROOT/'build/Release/tungsten.exe'),
                      nvdb_sha256=sha(field), baker_sha256=sha(ROOT/'build/Release/bake_contact_fields.exe'))
    sources = []
    for case in args.cases:
        for mode in args.modes:
            for part in args.components:
                s=scene(case,mode,args.stage,args.size,args.spp,part,args.seed,args.dy,args.extent,args.step)
                row=render_one(s,args.threads,args.seed,provenance)
                sources.append(dict(case=case,mode=mode,component=part,result=row,
                                    folder=str(folder(args.stage,case,mode,part,args.seed))))
    save(OUT/args.stage/'run.json',dict(parameters=vars(args),sources=sources))


def check_decomposition(stage, cases=CASES, modes=RENDER_MODES, seed=SEED):
    report=[]
    for case in cases:
        for mode in modes:
            paths=[folder(stage,case,mode,c,seed) for c in ('total','single','multiple_check')]
            a,b,c=[pfm(p/'render.pfm') for p in paths]
            err=np.abs(a-b-c)
            scale=max(float(a.max()),1.)
            if float(err.max())>2e-5*scale or float((a-b).min()) < -2e-5*scale:
                raise RuntimeError(f'Scattering decomposition failed: {case} {mode}')
            report.append(dict(case=case,mode=mode,max_absolute_error=float(err.max()),
                               total=float(a.mean()),single=float(b.mean()),multiple=float(c.mean())))
    save(OUT/stage/'decomposition_checks.json',report)
    print('Verified scattering decomposition:',len(report),'renders',flush=True)


def probe(args):
    field=setup(args.dy,args.extent)
    meta=load(Path(str(field)+'.json'))
    exe=ROOT/'build/Release/surface_probe.exe'
    for angle in sorted({int(case.split('_')[1]) for case in args.cases}):
        theta=math.radians(angle)
        view=np.array([math.sin(theta),math.cos(theta),0.])
        # The ensemble tool only needs this reference frame and central ray.
        # This is an exact central direction, not a sampled collision record.
        reference=dict(vertices=[view.tolist()],wi=(-view).tolist(),pixel=[64,64],
            diagnostic_resolution=[128,128],laws=[{},dict(mean=[0,1,0])],
            description='Exact central camera ray, no pixel jitter. No fixed collision position is imposed.')
        spec=dict(modes=list(args.modes),media=[scene(f'same_{angle}',m,'probe',128,1,
            dy=args.dy,extent=args.extent,step=args.step)['media'][0] for m in args.modes],
            reference_ray=reference,box_min=meta['process_box']['min'],box_max=meta['process_box']['max'],
            seed=args.seed,samples=args.path_samples,threads=args.threads,nz=48,nphi=96,angle_bins=90)
        output=OUT/'probes'/f'angle_{angle}.json'
        provenance=dict(probe_sha256=sha(exe),field_sha256=sha(field))
        if output.exists():
            old=load(output)
            if old['spec']==spec and old.get('provenance')==provenance:
                print('Reuse',output,flush=True);continue
            raise RuntimeError('Existing incompatible probe: '+str(output))
        request=output.with_name(output.stem+'_input.json');save(request,spec)
        with output.with_suffix('.log').open('w',encoding='utf-8') as log:
            subprocess.run([str(exe),'ensemble',str(request),str(output)],cwd=ROOT,
                stdout=log,stderr=subprocess.STDOUT,check=True)
        data=load(output)
        for m in data['results']:
            for k,c in enumerate(m['collisions']):
                assert c['attempted']==c['hits']+c['escaped']+c['distance_failed']+c['phase_failed']
                assert c['attempted']==(args.path_samples if k==0 else m['collisions'][0]['hits'])
                assert sum(c['normal_counts'])==sum(c['theta_counts'])==sum(c['incidence_counts'])==c['hits']
        data['provenance']=provenance;save(output,data)
        print('Saved',output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['setup','render','check','probe'])
    parser.add_argument('--stage',default='pilot')
    parser.add_argument('--size',type=int,default=32)
    parser.add_argument('--spp',type=int,default=8)
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--seed',type=int,default=SEED)
    parser.add_argument('--dy',type=float,default=.0025)
    parser.add_argument('--extent',type=float,default=1.5)
    parser.add_argument('--step',type=float,default=.0025)
    parser.add_argument('--path-samples',type=int,default=200000)
    parser.add_argument('--cases',nargs='+',choices=CASES,default=list(CASES))
    parser.add_argument('--modes',nargs='+',choices=RENDER_MODES,default=list(RENDER_MODES))
    parser.add_argument('--components',nargs='+',choices=['total','single','multiple_check'],default=['total','single'])
    args=parser.parse_args()
    if min(args.size,args.spp,args.threads,args.path_samples)<=0 or any(not math.isfinite(x) or x<=0 for x in (args.dy,args.extent,args.step)):
        parser.error('Sizes, sample counts, spacings and extent must be positive and finite')
    if args.command=='setup': print(setup(args.dy,args.extent))
    elif args.command=='render': render(args)
    elif args.command=='probe': probe(args)
    else: check_decomposition(args.stage,args.cases,args.modes,args.seed)
