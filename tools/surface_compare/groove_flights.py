"""Fixed-birth free-flight diagnostics for the shaderball's nested-sphere groove.

The SE reference conditions only on the birth value/gradient and explicitly
tests all intermediate signs. It is distinct from sparse convolution renewal+.
Production hazards and SDF samples are exported by fp_gradient_rays flights.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import gradient_rays as g
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.linalg import svd
from scipy.special import ndtri,ndtr
from scipy.stats import qmc

ROOT=g.ROOT
DEFAULT=ROOT/'outputs/shaderball-groove-flight-diagnostic-20261108'
SOURCE=ROOT/'outputs/shaderball-gradient-ray-diagnostic-20261108/expanded/rays.json'

def groove_score(p):
    q=np.asarray(p)-[0,.8,0]
    return float(abs(np.linalg.norm(q)-.6)+abs(np.linalg.norm(q-[0,.4,.4])-.56))

def prepare(out):
    data=g.read(SOURCE); groups=[[],[]]; seen=set()
    for original in data['records']:
        secondary=original['bounce']>0
        if groove_score(original['birth_position' if secondary else 'position'])>.08: continue
        if original['position'][1]<.6: continue
        signature=tuple(original['birth_position']+original['wi'])
        if signature in seen: continue
        seen.add(signature)
        r=copy.deepcopy(original)
        for k in ['mean_profile','production','conditional_projection']: r.pop(k,None)
        r['groove_score']=groove_score(r['birth_position' if secondary else 'position'])
        p,w=np.asarray(r['birth_position']),np.asarray(r['wi'])
        far=np.min(np.where(w>0,(np.array([1.5,2.501,1.5])-p)/w,
                                  (np.array([-1.5,-.499,-1.5])-p)/w))
        if w[1]<0 and p[1]>0: far=min(far,-p[1]/w[1])
        r['length']=float(min(far*.999999,.8 if secondary else r['t']+.2))
        groups[int(secondary)].append(r)
    # Spatially stratified subset, with both source modes. Selection precedes
    # any free-flight result; subsequent illustrative cases disclose ranking.
    chosen=[]
    for rs,n in zip(groups,[24,120]):
        rs.sort(key=lambda r:(r['pixel'][1],r['pixel'][0],r['bounce'],r['id']))
        chosen.extend(rs[i] for i in np.linspace(0,len(rs)-1,min(n,len(rs)),dtype=int))
    out.mkdir(parents=True,exist_ok=True)
    spec=dict(scene=data['spec']['scene'],records=chosen,denominator=1024,samples=0,integrate_hazard=True,
              selection=dict(first_candidates=len(groups[0]),secondary_candidates=len(groups[1]),
                  selected=len(chosen),criterion='abs(inner sphere distance)+abs(cutout sphere distance)<.08'))
    g.write(out/'spec.json',spec)
    run_export(out/'spec.json',out/'profiles.json')

def run_export(spec,target):
    with target.with_suffix('.log').open('w',encoding='utf-8') as log:
        subprocess.run([str(ROOT/'build/Release/fp_gradient_rays.exe'),'flights',str(spec),str(target)],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)

class BornRay:
    features=g.Ray.features
    def __init__(self,r):
        self.r=r; self.t=r['length']
        z=r['gamma']*self.t/2
        self.count=max(96,int(math.ceil(z*z+14*z+64)))
        f,d=self.features([0])
        observed=np.array([f[0],d[0]])
        qr,upper=np.linalg.qr(observed.T,mode='complete')
        delta=np.array([r['f0']-r['flight_mean'][0],r['g0']-r['flight_mean_derivative'][0]])
        self.coefficient_mean=qr[:,:2]@np.linalg.solve(upper[:2,:2].T,delta)
        self.free=qr[:,2:]
        ages=np.asarray(r['ages']); f,d=self.features(ages)
        self.mean=np.asarray(r['flight_mean'])+f@self.coefficient_mean
        self.residual=f@self.free
        u,s,self.vh=svd(self.residual,full_matrices=False,check_finite=False)
        self.rank=int(np.sum(s>s[0]*1e-12))
        self.factor=u[:,:self.rank]*s[:self.rank]
        stats=np.asarray(r['scalar_statistics'])
        check=np.linspace(0,len(ages)-1,min(65,len(ages)),dtype=int)
        self.checks=dict(mean_max_error=float(np.max(abs(self.mean[1:]-stats[1:,0]))),
            variance_relative_error=float(np.max(abs(np.sum(self.residual[1:]**2,axis=1)-stats[1:,1]))/r['sigma']**2),
            feature_count=self.count,rank=self.rank,
            svd_max_residual=float(np.max(abs(self.factor[check]@self.factor[check].T-self.residual[check]@self.residual[check].T))))

def analyze_job(job):
    r,power,replicates,out,save=job
    ray=BornRay(r); ages=np.asarray(r['ages']); n=len(ages)-1
    cdfs=[]; coarse=[[],[]]; crossings=[]
    for rep in range(replicates):
        rng=qmc.Sobol(ray.rank,scramble=True,seed=78117+rep*100003+r['id'],bits=52)
        counts=np.zeros(n+1); coarsecounts=[np.zeros(n//2+1),np.zeros(n//4+1)]
        ts=[]
        for normals in np.array_split(ndtri(rng.random_base2(power)),max(1,2**power//1024)):
            values=ray.mean+normals@ray.factor.T
            negative=values<0; negative[:,0]=False
            prefix=np.maximum.accumulate(negative,axis=1)
            counts+=prefix.sum(axis=0)
            for j,factor in enumerate([2,4]):
                coarsecounts[j]+=np.maximum.accumulate(negative[:,::factor],axis=1).sum(axis=0)
            if save:
                first=np.argmax(negative,axis=1); hit=first>0
                a=np.arange(len(first))[hit]; k=first[hit]
                t=np.full(len(first),r['length'])
                t[hit]=ages[k-1]+(ages[k]-ages[k-1])*values[a,k-1]/(values[a,k-1]-values[a,k])
                ts.extend(t.tolist())
        cdfs.append(counts/(2**power))
        for j in range(2): coarse[j].append(coarsecounts[j]/(2**power))
        if save: crossings.append(ts)
    cdfs=np.array(cdfs); ref=cdfs.mean(axis=0)
    hazard=np.asarray(r['hazard']); integral=cumulative_trapezoid(hazard,ages,initial=0)
    conditional=np.asarray(r['conditional_cdf']) if 'conditional_cdf' in r else -np.expm1(-integral)
    endpoint=ndtr(-np.divide(ray.mean,np.sqrt(np.maximum(0,np.sum(ray.residual**2,axis=1))),
        out=np.full_like(ray.mean,np.inf),where=np.sum(ray.residual**2,axis=1)>1e-28))
    endpoint[0]=0
    delta=conditional-ref; k=int(np.argmax(abs(delta)))
    se=float(cdfs[:,k].std(ddof=1)/math.sqrt(replicates)) if replicates>1 else None
    result=dict(id=r['id'],pixel=r['pixel'],bounce=r['bounce'],source=r['source'],length=r['length'],
        maximum_cdf_difference=float(abs(delta[k])),signed_difference=float(delta[k]),distance=float(ages[k]),
        conditional_collision_cdf=float(conditional[k]),reference_collision_cdf=float(ref[k]),
        endpoint_positive_probability=float(1-endpoint[k]),reference_survival=float(1-ref[k]),
        reference_standard_error_at_max=se,reference_samples=2**power*replicates,
        reference_escape=float(1-ref[-1]),conditional_escape=float(1-conditional[-1]),
        grid_cdf_change=[float(np.max(abs(np.mean(coarse[j],axis=0)-ref[::factor]))) for j,factor in enumerate([2,4])],
        quadrature_cdf_change=float(np.max(abs(-np.expm1(-cumulative_trapezoid(hazard[::2],ages[::2],initial=0))-conditional[::2]))),
        adaptive_quadrature_error_estimate=r.get('quadrature_error_estimate'),
        checks=ray.checks)
    if save:
        np.savez_compressed(Path(out)/f'flight-{r["id"]}.npz',ages=ages,mean=r['flight_mean'],posterior_mean=ray.mean,
            hazard=hazard,conditional_cdf=conditional,reference_cdf=ref,reference_replicates=cdfs,
            endpoint_negative=endpoint,reference_times=np.asarray(crossings))
        g.write(Path(out)/f'flight-{r["id"]}.json',result)
    print(f"Flight {r['id']}: max CDF delta {delta[k]:.5f}, t={ages[k]:.5f}, grid={result['grid_cdf_change'][0]:.3g}",flush=True)
    return result

def analyze(out,args):
    data=g.read(out/args.profiles)
    rows=data['records']
    if args.ids: rows=[r for r in rows if r['id'] in args.ids]
    jobs=[(r,args.power,args.replicates,str(out),True) for r in rows]
    with ProcessPoolExecutor(max_workers=args.workers) as pool: result=list(pool.map(analyze_job,jobs))
    result.sort(key=lambda r:r['maximum_cdf_difference'],reverse=True)
    g.write(out/args.summary,dict(profiles=args.profiles,power=args.power,replicates=args.replicates,results=result))

def sparse_job(job):
    out,spec,r,step,density,samples,radius=job
    tag=f'id{r["id"]}-step{step:g}-density{density:g}-n{samples}'
    if radius!=3: tag+=f'-radius{radius:g}'
    folder=Path(out)/'sparse';folder.mkdir(exist_ok=True)
    spec=copy.deepcopy(spec)
    spec.update(records=[r],samples=samples,step=step,impulse_density=density,export_profile=False)
    if radius!=3:
        scene=g.read(ROOT/spec['scene'])
        scene['media'][0]['gaussian_process']['covariance']['localScale']=radius
        scene_path=folder/f'{tag}-scene.json';g.write(scene_path,scene)
        spec['scene']=str(scene_path.resolve())
    source=folder/f'{tag}-spec.json';target=folder/f'{tag}.json'
    g.write(source,spec);run_export(source,target)
    r=g.read(target)['records'][0]
    print(tag,'failed',r['sparse']['failed'],'escaped',r['sparse']['escaped'],flush=True)
    return dict(id=r['id'],step=step,density=density,radius=radius,samples=samples,path=str(target.relative_to(out)),
                failures=r['sparse']['failures'],failed=r['sparse']['failed'],escaped=r['sparse']['escaped'])

def sparse(out,args):
    spec=g.read(out/'spec.json');rs=[r for r in spec['records'] if r['id'] in args.ids]
    jobs=[(str(out),spec,r,step,args.density,args.samples,args.radius) for r in rs for step in args.steps]
    with ProcessPoolExecutor(max_workers=args.workers) as pool: result=list(pool.map(sparse_job,jobs))
    g.write(out/args.summary,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['prepare','analyze','sparse']); p.add_argument('--out',type=Path,default=DEFAULT)
    p.add_argument('--profiles',default='profiles.json');p.add_argument('--summary',default='screen.json')
    p.add_argument('--power',type=int,default=11);p.add_argument('--replicates',type=int,default=1)
    p.add_argument('--workers',type=int,default=6);p.add_argument('--ids',type=int,nargs='+')
    p.add_argument('--samples',type=int,default=4096);p.add_argument('--density',type=float,default=10)
    p.add_argument('--radius',type=float,default=3)
    p.add_argument('--steps',type=float,nargs='+',default=[.01,.005,.0025,.00125])
    args=p.parse_args()
    if args.command=='prepare':prepare(args.out)
    elif args.command=='analyze':analyze(args.out,args)
    else:sparse(args.out,args)
