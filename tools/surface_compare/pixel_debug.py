"""Production pixel paths and controlled conditional-flight comparisons."""
from pathlib import Path
import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/single-pixel-groove-debug-20261109'
sys.path.insert(0,str(ROOT/'build/fp-gradient-validation-deps'))
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/hybrid-report-matplotlib-cache'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np

MODES = ['renewal_plus','global_conditional','conditional_fixed5']
LABELS = dict(zip(MODES,['GPIS renewal+','global_conditional','global_conditional + fixed5']))
SOURCES = dict(zip(MODES,[
    'outputs/renewal-plus-shaderball-l0p2-s0p02-256-128spp-20261108/renewal_plus',
    'outputs/conditional-fp-gradient-shaderball-l0p2-s0p02-256-128spp-20261108/baseline',
    'outputs/fixed5-anchor-shaderball-l0p2-s0p02-256-128spp-20261109/conditional']))
WEIGHTS = np.array([.2126,.7152,.0722])

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def write(path,data):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def prepare(samples):
    selected=read(OUT/'selection.json')['pixel_xy']
    for mode in MODES:
        folder=OUT/'paths'/mode; folder.mkdir(parents=True,exist_ok=True)
        scene=read(ROOT/SOURCES[mode]/'scene.json')
        scene['integrator'].update(independent_path_sampling=True,wavefront_gpis=False,use_cuda_gpis=False)
        write(folder/'scene.json',scene)
        write(folder/'spec.json',dict(scene=str(folder/'scene.json'),seed=41,samples=samples,
            verify_samples=min(samples,128),pixels=[dict(label='selected_groove',xy=selected)],
            export_mean_mesh=mode=='renewal_plus',mesh_grid=40))
    write(OUT/'path-provenance.json',dict(pixel=selected,samples=samples,seed=41,
        source_scenes={m:dict(path=SOURCES[m]+'/scene.json',sha256=digest(ROOT/SOURCES[m]/'scene.json')) for m in MODES},
        changes=['independent_path_sampling=true','wavefront_gpis=false','use_cuda_gpis=false'],
        note='New reproducible pixel samples. These are not exact replay of the old tile-RNG images.'))
    print('Prepared pixel',selected,'samples per mode',samples,flush=True)

def run_paths():
    exe=ROOT/'build/Release/surface_path_trace.exe'
    def run(mode):
        folder=OUT/'paths'/mode; start=time.perf_counter()
        with (folder/'trace.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(exe),str(folder/'spec.json'),str(folder/'paths.json')],cwd=ROOT,
                                  stdout=log,stderr=subprocess.STDOUT)
        write(folder/'run.json',dict(exit_code=result.returncode,wall_seconds=time.perf_counter()-start,
                                    executable_sha256=digest(exe),scene_sha256=digest(folder/'scene.json')))
        if result.returncode: raise RuntimeError('Path replay failed: '+mode)
        data=read(folder/'paths.json')
        print(mode,len(data['paths']),'paths; verified',data['verified_samples'],flush=True)
        return data
    with ThreadPoolExecutor(max_workers=3) as pool:
        data=dict(zip(MODES,pool.map(run,MODES)))
    cameras=[]; checks={}; stats={}
    for mode,d in data.items():
        assert d['observer_mismatches']==0
        camera=[next(e for e in p['events'] if e['type']=='camera') for p in d['paths']]
        cameras.append([(e['position'],e['outgoing']) for e in camera])
        gradients=0; flights=0
        for p in d['paths']:
            events=p['events']; scatters=[e for e in events if e['type']=='medium_scatter']
            assert len(scatters)==p['medium_scatters']
            for e in scatters:
                g=np.array(e['gradient']); wi=np.array(e['incoming']); wo=np.array(e['outgoing'])
                assert np.isfinite(g).all() and np.dot(g,wi)<=1e-7 and np.dot(g,wo)>0
                predicted=wi-2*np.dot(wi,g)/np.dot(g,g)*g
                assert np.max(abs(predicted-wo))<1e-5
                assert np.allclose(np.array(e['throughput_before'])*e['scatter_weight'],e['throughput_after'],rtol=3e-6,atol=1e-10)
                gradients+=1
            for e in events:
                if isinstance(e.get('flight'),dict):
                    f=e['flight']; assert np.isfinite(f['birth_gradient']).all() and np.isfinite(f['birth_value'])
                    flights+=1
        y=np.array([p['radiance'] for p in d['paths']])@WEIGHTS
        from collections import Counter
        counts=np.array([p['medium_scatters'] for p in d['paths']])
        stats[mode]=dict(samples=len(y),mean_luminance=float(y.mean()),luminance_standard_error=float(y.std(ddof=1)/len(y)**.5),
            mean_scatters=float(counts.mean()),multiple_fraction=float(np.mean(counts>=2)),
            scatter_histogram=dict(Counter(map(int,counts))),terminations=dict(Counter(p['termination'] for p in d['paths'])))
        checks[mode]=dict(observer_verified_samples=d['verified_samples'],observer_mismatches=0,checked_scatter_gradients=gradients,recorded_flights=flights)
    assert all(c==cameras[0] for c in cameras)
    write(OUT/'path-validation.json',dict(all_pass=True,common_camera_rays_identical=True,modes=checks))
    write(OUT/'path-statistics.json',stats)
    print(json.dumps(stats),flush=True)

def groove_score(p):
    p=np.asarray(p)-[0,.8,0]
    return float(abs(np.linalg.norm(p)-.6)+abs(np.linalg.norm(p-[0,.4,.4])-.56))

def select_flights():
    data={m:read(OUT/'paths'/m/'paths.json') for m in MODES}
    queries=[]
    for mode,d in data.items():
        for path in d['paths']:
            order=0
            for event in path['events']:
                f=event.get('flight')
                if isinstance(f,dict) and f and f['max_distance']>0:
                    w=np.array(f['direction']); w/=np.linalg.norm(w)
                    value=f['birth_value']; derivative=float(w@f['birth_gradient'])
                    if value<0 or (value==0 and derivative<=0) or abs(f['age_at_origin'])>1e-6: continue
                    queries.append(dict(mode=mode,sample=path['sample'],event_index=event['index'],
                        type=event['type'],order=order,flight=f,
                        groove_score=groove_score(f['origin']),event=event))
                if event['type']=='medium_scatter':order+=1
    chosen=[]
    def choose(key,mode,kind):
        eligible=[q for q in queries if q['mode']==mode]
        if kind=='primary':
            eligible=[q for q in eligible if q['type']=='medium_collision' and q['order']==0 and q['flight'].get('sampled_distance',0)>.02]
        elif kind=='secondary':
            eligible=[q for q in eligible if q['type']=='medium_collision' and q['order']>0 and q['groove_score']<.12 and q['flight'].get('sampled_distance',0)>.02]
        else:
            eligible=[q for q in eligible if q['type']=='shadow_segment' and q['flight']['birth_value']==0 and q['groove_score']<.12 and q['flight']['max_distance']>.05]
        if not eligible:raise RuntimeError('No eligible flight '+key)
        # Deterministic selection uses geometry and path order, never a measured model discrepancy.
        q=min(eligible,key=lambda q:(q['sample'],q['event_index']))
        f=q['flight']; r=dict(id=len(chosen)+1,key=key,source=mode,pixel=[76,117],sample=q['sample'],
            event_index=q['event_index'],bounce=q['order'],kind=kind,
            birth_position=f['birth_position'],birth_gradient=f['birth_gradient'],wi=f['direction'],
            f0=f['birth_value'],g0=float(np.dot(f['birth_gradient'],f['direction'])),length=f['max_distance'],
            selection='Earliest (sample, event) satisfying the stated geometric and distance criteria.',
            eligible_count=len(eligible),groove_score=q['groove_score'],source_event=q['event'])
        if kind!='shadow':r['focus_distance']=f['sampled_distance']
        chosen.append(r)
    choose('P','renewal_plus','primary')
    choose('N','renewal_plus','secondary')
    choose('C','global_conditional','secondary')
    choose('F','conditional_fixed5','secondary')
    choose('SN','renewal_plus','shadow')
    choose('SC','global_conditional','shadow')
    for r in chosen:
        write(OUT/'flights'/r['key']/'spec.json',dict(record=r,
            native_scene=str(OUT/'paths/renewal_plus/scene.json'),conditional_scene=str(OUT/'paths/global_conditional/scene.json'),
            seed=20261109,samples=8192,shadow_samples=4096,denominator=512,gradient_denominator=1664))
    write(OUT/'flight-selection.json',dict(records=chosen,
        rules=dict(secondary='Start groove_score < 0.12; observed t > 0.02 world units.',
                   shadow='Surface birth, groove_score < 0.12; medium segment length > 0.05.',
                   primary='First collision, observed t > 0.02.'),
        note='Recorded double birth jets are quantized to float for controlled replay, matching native conditioning. Actual production path records remain unmodified.'))
    print(json.dumps([{k:r.get(k) for k in ['key','source','sample','event_index','bounce','length','focus_distance','groove_score']} for r in chosen]),flush=True)

def run_flights():
    records=read(OUT/'flight-selection.json')['records']; exe=ROOT/'build/Release/pixel_flights.exe'
    def run(r):
        folder=OUT/'flights'/r['key']; start=time.perf_counter()
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            result=subprocess.run([str(exe),str(folder/'spec.json'),str(folder/'result.json')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        write(folder/'run.json',dict(exit_code=result.returncode,wall_seconds=time.perf_counter()-start,executable_sha256=digest(exe)))
        if result.returncode:raise RuntimeError('Controlled flight failed: '+r['key'])
        print('Finished controlled flight',r['key'],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,records))
    rows=[read(OUT/'flights'/r['key']/'result.json')['record'] for r in records]
    write(OUT/'gradients/rays.json',dict(scene=str(OUT/'paths/global_conditional/scene.json'),records=rows))

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['prepare','paths','select','flights'])
    parser.add_argument('--samples',type=int,default=512)
    args=parser.parse_args()
    if args.command=='prepare': prepare(args.samples)
    if args.command=='paths': run_paths()
    if args.command=='select': select_flights()
    if args.command=='flights': run_flights()

if __name__=='__main__': main()
