"""White-environment regression for Lambert micro-scattering with/without NEE."""
from __future__ import annotations
from argparse import Namespace
import re
import subprocess
from rough_geometry import ROOT, OUT, SEED, scene, sha, save, load, pfm
import numpy as np


def main():
    exe=ROOT/'build/Release/tungsten.exe'
    version=sha(exe)
    rows=[]
    args=Namespace(stage='material_check',size=32,spp=1024,threads=8,seed=SEED,
                   step=.004,domain_scale=1.,material='lambert')
    for mode in ('classic_global','global_conditional'):
        for nee in (True,False):
            d=OUT/'material_check'/mode/('nee_on' if nee else 'nee_off')
            s=scene(args,'plane_same_75',mode)
            s['media'][0]['material']['albedo']=1
            s['integrator'].update(min_bounces=0,max_bounces=128,enable_light_sampling=nee,
                                   enable_volume_light_sampling=nee)
            s['primitives']=s['primitives'][:1]+[dict(type='infinite_sphere',emission=1,
                                                       bsdf=dict(type='null',albedo=1))]
            s['renderer']['output_directory']=str(d)
            if (d/'result.json').exists():
                r=load(d/'result.json')
                assert r['renderer_sha256']==version and load(d/'scene.json')==s
                assert r['scene_sha256']==sha(d/'scene.json') and r['pfm_sha256']==sha(d/'render.pfm')
            else:
                save(d/'scene.json',s)
                print('Furnace',mode,'NEE',nee,flush=True)
                with (d/'render.log').open('w',encoding='utf-8') as f:
                    subprocess.run([str(exe),'-t',str(args.threads),'-s',str(SEED),'-d',str(d),str(d/'scene.json')],
                                   cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
                log=(d/'render.log').read_text(encoding='utf-8',errors='replace')
                assert 'Finished render.' in log and 'Completed 1024/1024 spp' in log
                assert not re.search(r'Caught an internal error|unrecoverable|Gradient invalid',log,re.I)
                y=pfm(d/'render.pfm')[...,0]
                v=pfm(d/'samplesVariance.pfm')[...,0]
                se=float(np.sqrt(v.sum())/y.size)
                mean=float(y.mean())
                assert abs(mean-1)<=max(.002,5*se), (mode,nee,mean,se)
                r=dict(mode=mode,nee=nee,mean=mean,standard_error=se,expected=1,
                       renderer_sha256=version,scene_sha256=sha(d/'scene.json'),pfm_sha256=sha(d/'render.pfm'))
                save(d/'result.json',r)
            rows.append(r)
            print(mode,nee,r['mean'],r['standard_error'],flush=True)
    save(OUT/'material_check'/'validation.json',dict(passed=True,checks=rows))


if __name__=='__main__': main()
