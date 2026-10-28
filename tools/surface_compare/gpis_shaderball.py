"""Render the actual gpis-scenes/shaderball stationary scene (default ell=.1, sigma=.02).

cmake --build build --config Release --target gpis_shaderball_bake --parallel 4
python -B tools/surface_compare/gpis_shaderball.py bake
python -B tools/surface_compare/gpis_shaderball.py render --stage pilot --size 64 --spp 16
python -B tools/surface_compare/gpis_shaderball.py render
python -B tools/surface_compare/gpis_shaderball.py render --spp 32 --modes gpis_fixed
python -B tools/surface_compare/gpis_shaderball.py render --stage step_base --size 96 --spp 32 --modes sparse_3d_global
python -B tools/surface_compare/gpis_shaderball.py render --stage step_half --size 96 --spp 32 --step .005 --modes sparse_3d_global
<plot-python> -B tools/surface_compare/gpis_shaderball.py figures

Additional parameters: --ell .2 --sigma .04 (bake before rendering a new sigma).
Use gpis_shaderball_sweep.py for the two ell=.2 groups and combined figures.
Use --modes neural for the existing learned model, or gpis_shaderball_neural.py
to render all three neural parameter groups and produce five-mode figures.
"""
from __future__ import annotations
import argparse
import copy
import json
import re
import subprocess
import time
from pathlib import Path
import numpy as np
from run import ROOT, save, load, pfm
from shaderball_length import digest

SOURCE = ROOT/'gpis-scenes/shaderball/shaderball-stationary-270.json'
OUT = ROOT/'outputs/gpis_shaderball_compare/ell0p1'
FIELD = OUT/'fields/knob_dx0p005.nvdb'
NEURAL_MODEL = ROOT/'models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json'
MODES = ('classic_local','classic_global','global_conditional','sparse_3d_global')
ALL_MODES = MODES+('gpis_fixed','neural')
LABELS = ('classic_local','classic_global','global_conditional','GPIS global ensemble','GPIS fixed realization')
WEIGHTS = np.array([.2126,.7152,.0722])
ROI = (.25,.2,.76,.72)
MODEL_ROI = (.14,.1,.86,.94)


def source():
    # Preserve quoted strings while removing JSONC comments.
    text = SOURCE.read_text(encoding='utf-8-sig')
    pattern = r'"(?:\\.|[^"\\])*"|//[^\r\n]*|/\*[\s\S]*?\*/'
    text = re.sub(pattern, lambda m: m[0] if m[0].startswith('"') else ' ', text)
    return json.loads(text)


def folder(args,mode):
    return output(args)/args.stage/mode/f'seed{args.seed}'


def parameters(args):
    return getattr(args,'ell',.1),getattr(args,'sigma',.02)


def output(args):
    ell,sigma=parameters(args)
    if (ell,sigma)==(.1,.02): return OUT
    key=lambda v: format(v,'.8g').replace('.','p')
    return OUT.parent/f'ell{key(ell)}_sigma{key(sigma)}'


def field_path(args):
    # Density depends on sigma and mean geometry, not on correlation length.
    return FIELD if parameters(args)[1]==.02 else output(args)/'fields/knob_dx0p005.nvdb'


def scene(args,mode):
    s=source(); original=s['media'][0]
    gp=copy.deepcopy(original['gaussian_process'])
    assert gp['covariance']['sigma']==.02 and gp['mean']['func']=='knob'
    gp['covariance']['lengthScale'],gp['covariance']['sigma']=parameters(args)
    if mode in ('sparse_3d_global','gpis_fixed'):
        m=original
        m['gaussian_process']=gp
        m.update(single_realization=mode=='gpis_fixed',correlation_context='global',step_size=args.step)
    else:
        m=dict(name=original['name'],type=mode,max_bounces=original['max_bounces'],
               material=copy.deepcopy(original['phase_function']['bsdf']))
        if mode=='classic_local':
            m.update(field=dict(file=str(field_path(args)),sdf_grid='sdf',density_grid='density',macrocell_size=8),
                     covariance=gp['covariance'])
        elif mode=='neural':
            m.update(type='neural_causal_renewal',backend='scalar',model=str(NEURAL_MODEL),
                     field=dict(file=str(field_path(args))),covariance=gp['covariance'],
                     profile_maximum_step=.25,distance_tolerance=1e-9)
        else:
            m.update(gaussian_process=gp,sdf_lipschitz=1.1,gradient_bound=2.,tracking_step=0.)
            if mode=='global_conditional':
                m.update(tracking_step_scale=.5,tracking_age_scale=.25,tracking_max_expected_events=64.)
    s['media']=[m]
    s['camera']['resolution']=[args.size,args.size]
    s['integrator'].update(use_cuda_gpis=False,independent_path_sampling=True,wavefront_gpis=False)
    s['renderer'].update(output_directory=str(folder(args,mode)),output_file='render.png',
        hdr_output_file='render.pfm',overwrite_output_files=True,spp=args.spp,spp_step=min(16,args.spp),
        output_buffers=[dict(type='color',hdr_output_file='samples.pfm',sample_variance=True,two_buffer_variance=True)])
    return s


def bake(args):
    field=field_path(args);ell,sigma=parameters(args)
    snapshot=output(args)/'source.json'
    save(snapshot,source())
    if field.exists():
        meta=load(Path(str(field)+'.json'))
        assert meta['dx']==.005 and meta['sigma']==sigma
        assert meta['source_original_sha256']==digest(SOURCE)
        assert meta['source_mean']==source()['media'][0]['gaussian_process']['mean']
        assert meta['source_scene_sha256']==digest(Path(meta['source_scene']))
        assert meta['field_sha256']==digest(field)
        print('Reuse verified field',field,flush=True)
        return
    bake_scene=snapshot
    if sigma!=.02:
        bake_scene=output(args)/'bake_scene.json';s=source()
        s['media'][0]['gaussian_process']['covariance'].update(lengthScale=ell,sigma=sigma)
        save(bake_scene,s)
    field.parent.mkdir(parents=True,exist_ok=True)
    with (field.parent/'bake.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(ROOT/'build/Release/gpis_shaderball_bake.exe'),str(bake_scene),str(field),'.005'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,
            creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
    meta=load(Path(str(field)+'.json'))
    meta.update(source_scene_sha256=digest(bake_scene),source_original_sha256=digest(SOURCE),field_sha256=digest(field),
                baker_sha256=digest(ROOT/'build/Release/gpis_shaderball_bake.exe'))
    save(Path(str(field)+'.json'),meta)
    print('Baked',field,flush=True)


def render(args):
    provenance=dict(renderer_sha256=digest(ROOT/'build/Release/tungsten.exe'),source_sha256=digest(SOURCE))
    if any(mode in args.modes for mode in ('classic_local','neural')):
        field=field_path(args);meta=load(Path(str(field)+'.json'))
        assert meta['source_original_sha256']==provenance['source_sha256']
        assert meta['sigma']==parameters(args)[1] and meta['dx']==.005
        assert meta['field_sha256']==digest(field)
        provenance['field_sha256']=meta['field_sha256']
    results=[]
    for mode in args.modes:
        s=scene(args,mode);d=folder(args,mode)
        p=copy.deepcopy(provenance)
        if mode not in ('classic_local','neural'): p.pop('field_sha256',None)
        if mode=='neural': p['model_sha256']=digest(NEURAL_MODEL)
        if (d/'result.json').exists():
            r=load(d/'result.json')
            assert r['provenance']==p and load(d/'scene.json')==s
            for name,h in r['sha256'].items(): assert digest(d/name)==h
            print('Reuse',args.stage,mode,flush=True)
        else:
            if (d/'render.pfm').exists(): raise RuntimeError('Unvalidated render: '+str(d))
            save(d/'scene.json',s)
            print('Render',args.stage,mode,flush=True);start=time.monotonic()
            with (d/'render.log').open('w',encoding='utf-8') as log:
                result=subprocess.run([str(ROOT/'build/Release/tungsten.exe'),'-t',str(args.threads),'-s',str(args.seed),
                    '-d',str(d),str(d/'scene.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                    creationflags=getattr(subprocess,'BELOW_NORMAL_PRIORITY_CLASS',0))
            log=(d/'render.log').read_text(encoding='utf-8',errors='replace')
            if result.returncode or 'Finished render.' not in log or not re.search(rf'Completed {args.spp}/{args.spp} spp',log):
                save(d/'failure.json',dict(returncode=result.returncode,renderer_sha256=p['renderer_sha256'],
                    seconds=time.monotonic()-start,log_sha256=digest(d/'render.log')))
                raise RuntimeError(f'Render failed (exit {result.returncode}): '+str(d))
            if re.search(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|Wrong normal direction|Failed to sample gradient|Gradient zero',log,re.I):
                raise RuntimeError('Render error: '+str(d))
            rgb=pfm(d/'render.pfm'); assert rgb.shape==(args.size,args.size,3)
            assert np.allclose((pfm(d/'samplesA.pfm')+pfm(d/'samplesB.pfm'))/2,rgb,rtol=3e-6,atol=3e-7)
            pfm(d/'samplesVariance.pfm')
            discarded=re.search(r'discarded surface births: (\{.*\})',log)
            r=dict(seconds=time.monotonic()-start,resolution=[args.size,args.size],spp=args.spp,provenance=p,
                mean_rgb=rgb.mean((0,1)).tolist(),mean_luminance=float((rgb@WEIGHTS).mean()),
                discarded_surface_births=json.loads(discarded[1]) if discarded else None,
                sha256={name:digest(d/name) for name in ('scene.json','render.pfm','render.png','samplesA.pfm','samplesB.pfm','samplesVariance.pfm')})
            save(d/'result.json',r)
            print('Done',args.stage,mode,round(r['seconds'],2),'mean Y',round(r['mean_luminance'],6),flush=True)
        results.append(dict(mode=mode,folder=str(d),result=r))
    save(output(args)/args.stage/('run_'+'_'.join(args.modes)+'.json'),dict(parameters=vars(args),sources=results))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=('bake','render','figures'))
    p.add_argument('--stage',default='final');p.add_argument('--size',type=int,default=256)
    p.add_argument('--spp',type=int,default=128);p.add_argument('--seed',type=int,default=20261024)
    p.add_argument('--threads',type=int,default=32);p.add_argument('--step',type=float,default=.01)
    p.add_argument('--ell',type=float,default=.1);p.add_argument('--sigma',type=float,default=.02)
    p.add_argument('--modes',nargs='+',choices=ALL_MODES,default=list(MODES))
    args=p.parse_args()
    if min(args.size,args.spp,args.threads)<=0 or args.spp%2 or args.size%16 or not np.isfinite(args.step) or args.step<=0:
        p.error('Positive sizes, even spp, image size divisible by 16, and positive finite step required')
    if not all(np.isfinite(v) and v>0 for v in parameters(args)):
        p.error('ell and sigma must be finite and positive')
    if args.command=='figures' and parameters(args)!=(.1,.02):
        p.error('Use gpis_shaderball_sweep.py figures for the ell=.2 comparison')
    if args.command=='bake': bake(args)
    elif args.command=='render': render(args)
    else:
        from gpis_shaderball_figures import figures
        figures(args)
