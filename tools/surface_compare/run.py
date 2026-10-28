"""Matched surface appearance renders; outputs stay under outputs/surface_like_compare.

Reproduce from the repository root after building bake_contact_fields and surface_probe:
  python tools/surface_compare/run.py setup
  build/Release/bake_contact_fields.exe outputs/surface_like_compare/bake.json
  python tools/surface_compare/run.py render --stage final --size 512 --spp 128 --cases studio_ell0p025 grazing_ell0p025
  python tools/surface_compare/run.py probe --cases studio_ell0p025
  python tools/surface_compare/run.py select --cases studio_ell0p025
  python tools/surface_compare/figures.py final

Add the matched world-space GPIS approximation and empirical normal histograms:
  python tools/surface_compare/run.py render --stage final --size 512 --spp 128 --cases studio_ell0p025 grazing_ell0p025 --modes sparse_3d_global
  python tools/surface_compare/run.py ensemble --stage final --path-samples 200000 --threads 24
  python tools/surface_compare/figures.py gpis

Rendering/probing need NumPy. Figures additionally need requirements-plot.txt.
The default six-case preview sweep uses `render` without extra arguments.
The selected examples deliberately keep geometry, sigma and ell identical.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, math, re, subprocess, time
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'outputs/surface_like_compare'
MODES=('classic_local','classic_global','global_conditional')
RENDER_MODES=MODES+('sparse_3d_global',)
CASES={f'{light}_ell{str(ell).replace(".","p")}':(light,ell)
       for light in ('studio','grazing') for ell in (.05,.025,.0167)}
def load(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,j):
    p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(j,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pfm(path):
    with path.open('rb') as f:
        assert f.readline().strip()==b'PF'
        w,h=map(int,f.readline().split());scale=float(f.readline())
        data=np.frombuffer(f.read(),dtype='<f4' if scale<0 else '>f4').reshape(h,w,3)
    data=np.flipud(data).astype(float)*abs(scale)
    assert np.isfinite(data).all() and data.min()>=0
    return data
def light(name,pos,scale,emission):
    normal=(-np.array(pos,dtype=float));normal/=np.linalg.norm(normal)
    # Quad's local +Y is its emission normal.
    return dict(name=name,type='quad',emission=emission,bsdf=dict(type='null',albedo=1),
        transform=dict(position=pos,y_axis=normal.tolist(),scale=scale))
def medium(mode,field,ell,sparse_step=None,impulse_density=10):
    cov=dict(type='squared_exponential',sigma=.01,lengthScale=ell)
    result=dict(name='surface',type=mode,max_bounces=128,material=dict(type='mirror',albedo=.9))
    if mode=='classic_local':
        result.update(field=dict(file=str(field),sdf_grid='sdf',density_grid='density',macrocell_size=8),covariance=cov)
    else:
        result['gaussian_process']=dict(type='standard',covariance=cov,mean=dict(type='tabulated',grid=dict(
            type='vdb',file=str(field.with_name(field.stem+'_sdf.vdb')),density_name='sdf',density_scale=1,
            emission_name='',normalize_size=False,request_sdf=False,interpolate='linear',integration_method='raymarching')))
    if mode=='sparse_3d_global':
        # Global context hashes (pixel, sample) but ignores the segment index:
        # independent fields across camera paths, one fixed field along each path.
        # With 1D_sampling=False, the production sampler uses the realized 3D
        # field gradient; inherited normal_method does not resample Gaussian normals.
        material=result.pop('material')
        result.update(type='sparse_conv_noise',phase_function=dict(type='brdf',bsdf=material),
            sigma_a=0,sigma_s=1,density=1,step_size=min(.0025,ell/10) if sparse_step is None else sparse_step,
            min_step=8,seed=17429,impulse_density=impulse_density,single_realization=False,
            correlation_context='global',intersect_method='gp_discrete',normal_method='conditioned_gaussian',
            isotropic_3D_sampling=False)
        result.update({'1D_sampling':False,'1D_sampling_scheme':'uni','1D_gradient_correlationXY':False})
    return result
def scene(case,mode,stage,size,spp,seed,field,sparse_step=None,impulse_density=10):
    lighting,ell=CASES[case];meta=load(Path(str(field)+'.json'))
    lo=np.array(meta['process_box']['min']);hi=np.array(meta['process_box']['max'])
    key=light('key',[-.85,1.35,-1.5],[.55,1,1.6],9) if lighting=='studio' else light('key',[-1.65,.5,.2],[.4,1,1.4],13)
    folder=OUT/stage/case/mode/f'seed{seed}'
    return dict(media=[medium(mode,field,ell,sparse_step,impulse_density)],camera=dict(type='pinhole',resolution=[size,size],fov=27,
        tonemap='filmic',reconstruction_filter='tent',transform=dict(position=[1.0,.9,-2.5],look_at=[0,0,0],up=[0,1,0],scale=[-1,1,1])),
        integrator=dict(type='path_tracer',min_bounces=0,max_bounces=128,enable_light_sampling=True,
            enable_volume_light_sampling=True,enable_consistency_checks=False,enable_two_sided_shading=True,
            use_cuda_gpis=False,independent_path_sampling=True,wavefront_gpis=False),
        primitives=[dict(name='processBox',type='cube',int_medium='surface',bsdf=dict(type='forward',albedo=1),
            transform=dict(position=((lo+hi)/2).tolist(),scale=(hi-lo).tolist())),key,
            dict(type='infinite_sphere',emission=.025,bsdf=dict(type='null',albedo=1))],
        renderer=dict(spp=spp,spp_step=min(16,spp),adaptive_sampling=False,stratified_sampler=False,
            enable_resume_render=False,overwrite_output_files=True,output_file='render.png',hdr_output_file='render.pfm',
            output_directory=str(folder)))
def render(args):
    field=OUT/('fields_fine' if args.fine else 'fields')/'rounded_shaderball.nvdb'
    exe=ROOT/'build/Release/tungsten.exe';version=sha(exe)
    for case in args.cases:
        for mode in args.modes or MODES:
            s=scene(case,mode,args.stage,args.size,args.spp,args.seed,field,args.sparse_step,args.impulse_density)
            folder=Path(s['renderer']['output_directory']);result=folder/'result.json'
            if result.exists() and load(result).get('renderer_sha256')==version and load(folder/'scene.json')==s:
                print('Reuse',case,mode,flush=True);continue
            if (folder/'render.pfm').exists(): raise RuntimeError('Existing unmatched render: '+str(folder))
            save(folder/'scene.json',s)
            print('Render',args.stage,case,mode,args.size,args.spp,flush=True);t=time.monotonic()
            with (folder/'render.log').open('w',encoding='utf-8') as log:
                p=subprocess.run([str(exe),'-t',str(args.threads),'-s',str(args.seed),'-d',str(folder),str(folder/'scene.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            log=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
            if p.returncode or 'Finished render.' not in log or not re.search(rf'Completed {args.spp}/{args.spp} spp',log) or re.search(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|Wrong normal direction|Failed to sample gradient|Gradient zero|Conditioning .* error',log,re.I):
                raise RuntimeError('Failed render: '+str(folder))
            rgb=pfm(folder/'render.pfm');assert rgb.shape==(args.size,args.size,3)
            discarded=re.search(r'discarded surface births: (\{.*\})',log)
            row=dict(seconds=time.monotonic()-t,renderer_sha256=version,field=str(field),
                mean_rgb=float(rgb.mean()),maximum=float(rgb.max()),discarded_surface_births=json.loads(discarded[1]) if discarded else None)
            save(result,row);print('Done',case,mode,round(row['seconds'],2),flush=True)
def setup(fine):
    request=dict(output_directory=str(OUT/('fields_fine' if fine else 'fields')),voxel_size=.0025 if fine else .005,
        sigma=.01,length_scale=.025,padding=.06,cases=[dict(name='rounded_shaderball',shape='rounded_shaderball')])
    path=OUT/('bake_fine.json' if fine else 'bake.json');save(path,request)
    print(path)
def probe(args):
    field=OUT/('fields_fine' if args.fine else 'fields')/'rounded_shaderball.nvdb'
    s=scene(args.cases[0],'global_conditional','probe',128,128,args.seed,field)
    meta=load(Path(str(field)+'.json'));ell=CASES[args.cases[0]][1]
    spec=dict(media=[medium(m,field,ell) for m in MODES],camera=s['camera'],length_scale=ell,
        box_min=meta['process_box']['min'],box_max=meta['process_box']['max'],seed=args.seed,stride=6,samples=3)
    path=OUT/'probes'/f'{args.cases[0]}_input.json';save(path,spec)
    out=path.with_name(f'{args.cases[0]}_records.json')
    with out.with_suffix('.log').open('w',encoding='utf-8') as log:
        p=subprocess.run([str(ROOT/'build/Release/surface_probe.exe'),'scan',str(path),str(out)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    if p.returncode: raise RuntimeError('Probe failed: '+str(out.with_suffix('.log')))
    records=load(out)
    records['provenance']=dict(probe_sha256=sha(ROOT/'build/Release/surface_probe.exe'),
        nvdb_sha256=sha(field),vdb_sha256=sha(field.with_name(field.stem+'_sdf.vdb')))
    save(out,records)
    print(out)

def select(args):
    # Both examples are successive collisions along one reproducible diagnostic
    # camera path. The exact production gradient laws are retained in the JSON.
    source=OUT/'probes'/f'{args.cases[0]}_records.json';j=load(source)
    ell=j['spec']['length_scale']
    for label,bounce in (('small',0),('large',1)):
        matches=[r for r in j['records'] if r['pixel']==args.pixel and r['sample']==args.sample and r['bounce']==bounce]
        if len(matches)!=1: raise RuntimeError(f'Expected one collision {bounce} for selected ray')
        r=copy.deepcopy(matches[0]);r.update(sigma=.01,length_scale=ell,
            diagnostic_resolution=j['spec']['camera']['resolution'],diagnostic_seed=j['spec']['seed'],
            field=j['spec']['media'][0]['field']['file'],sampling=j['sampling'],label=label,
            provenance=j.get('provenance',{}))
        selected=OUT/'probes'/f'selected_{label}.json';save(selected,r)
        output=selected.with_name(f'distribution_{label}.json')
        subprocess.run([str(ROOT/'build/Release/surface_probe.exe'),'grid',str(selected),str(output)],cwd=ROOT,check=True)
        print(output,flush=True)

def ensemble(args):
    field=OUT/('fields_fine' if args.fine else 'fields')/'rounded_shaderball.nvdb'
    record=load(OUT/'probes/selected_small.json')
    meta=load(Path(str(field)+'.json'));ell=record['length_scale']
    modes=list(args.modes or RENDER_MODES)
    spec=dict(modes=modes,media=[medium(m,field,ell,args.sparse_step,args.impulse_density) for m in modes],
        reference_ray=record,box_min=meta['process_box']['min'],box_max=meta['process_box']['max'],
        seed=args.seed,samples=args.path_samples,threads=args.threads,nz=48,nphi=96,angle_bins=90)
    path=OUT/'probes'/f'ensemble_{args.stage}_input.json';output=path.with_name(f'ensemble_{args.stage}.json')
    exe=ROOT/'build/Release/surface_probe.exe'
    provenance=dict(probe_sha256=sha(exe),nvdb_sha256=sha(field),
        vdb_sha256=sha(field.with_name(field.stem+'_sdf.vdb')))
    if output.exists():
        old=load(output)
        if old['spec']==spec and old.get('provenance')==provenance:
            print('Reuse',output,flush=True);return
        raise RuntimeError('Existing unmatched ensemble: '+str(output))
    save(path,spec)
    print('Ensemble',args.stage,modes,args.path_samples,flush=True)
    with output.with_suffix('.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(exe),'ensemble',str(path),str(output)],cwd=ROOT,
            stdout=log,stderr=subprocess.STDOUT,check=True)
    data=load(output)
    for mode in data['results']:
        for k,c in enumerate(mode['collisions']):
            assert c['attempted']==c['hits']+c['escaped']+c['distance_failed']+c['phase_failed']
            assert c['attempted']==(args.path_samples if k==0 else mode['collisions'][0]['hits'])
            assert sum(c['normal_counts'])==sum(c['theta_counts'])==sum(c['incidence_counts'])==c['hits']
            assert np.array_equal(np.array(c['split_counts']).sum(0),c['normal_counts'])
            assert c['hit_probability']==c['hits']/args.path_samples
    data['provenance']=provenance;save(output,data)
    print(output,flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('command',choices=['setup','render','probe','select','ensemble'])
    parser.add_argument('--fine',action='store_true');parser.add_argument('--stage',default='preview')
    parser.add_argument('--size',type=int,default=128);parser.add_argument('--spp',type=int,default=128)
    parser.add_argument('--seed',type=int,default=20261024);parser.add_argument('--threads',type=int,default=24)
    parser.add_argument('--cases',nargs='+',choices=list(CASES),default=list(CASES))
    parser.add_argument('--modes',nargs='+',choices=RENDER_MODES,default=None)
    parser.add_argument('--path-samples',type=int,default=200000)
    parser.add_argument('--sparse-step',type=float,default=None)
    parser.add_argument('--impulse-density',type=int,default=10)
    parser.add_argument('--pixel',nargs=2,type=int,default=[33,63]);parser.add_argument('--sample',type=int,default=1)
    args=parser.parse_args()
    if min(args.size,args.spp,args.threads,args.path_samples)<=0: parser.error('size, spp, threads and path samples must be positive')
    if args.impulse_density<=0 or (args.sparse_step is not None and not (math.isfinite(args.sparse_step) and args.sparse_step>0)):
        parser.error('sparse step and integer impulse density must be positive')
    if args.command in ('probe','select') and len(args.cases)!=1: parser.error('probe/select requires exactly one --cases value')
    if args.command=='setup':setup(args.fine)
    elif args.command=='render':render(args)
    elif args.command=='probe':probe(args)
    elif args.command=='ensemble':ensemble(args)
    else:select(args)
