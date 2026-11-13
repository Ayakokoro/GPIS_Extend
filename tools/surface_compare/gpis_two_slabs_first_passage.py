"""Independent, discretized Gaussian first-passage check for the central ray.

This is a Gaussian covariance sampler, not a GPIS Global renderer. It checks a
single axial ray, not the full pinhole image. Nested grids expose missed hits.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('outputs/gpis_two_slabs_compare_20261113/central_ray_gaussian_check.json'))
    parser.add_argument('--samples',type=int,default=131072)
    args=parser.parse_args();started=time.monotonic()
    sigma,ell=.02,.2
    y=np.linspace(.3,-.3,961)
    mean=np.minimum(np.abs(y-.02),np.abs(y+.02))-.01
    covariance=sigma**2*np.exp(-.5*((y[:,None]-y[None,:])/ell)**2)
    eigenvalues,eigenvectors=np.linalg.eigh(covariance)
    keep=eigenvalues>sigma**2*1e-12
    factor=eigenvectors[:,keep]*np.sqrt(eigenvalues[keep])
    covariance_error=float(np.max(np.abs(factor@factor.T-covariance)))
    rng=np.random.default_rng(2026111301)
    counts={stride:0 for stride in (1,2,4)}
    disagreements={stride:0 for stride in (2,4)}
    for begin in range(0,args.samples,4096):
        noise=rng.standard_normal((min(4096,args.samples-begin),factor.shape[1]))@factor.T
        values=noise+mean
        survival={stride:np.all(values[:,::stride]>0,axis=1) for stride in counts}
        for stride in counts:counts[stride]+=int(np.count_nonzero(survival[stride]))
        for stride in disagreements:disagreements[stride]+=int(np.count_nonzero(survival[stride]!=survival[1]))
    p=counts[1]/args.samples
    # Wilson interval: sampling uncertainty only, not discretization error.
    n=args.samples;z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    result=dict(ray=dict(start=[0,.3,0],end=[0,-.3,0]),sigma=sigma,lengthScale=ell,
        samples=n,seed=2026111301,transmittance=p,wilson_95=[center-half,center+half],
        factor_rank=int(keep.sum()),maximum_covariance_error=covariance_error,
        nested_grids=[dict(step=.000625*stride,transmittance=counts[stride]/n,
                           disagreements_with_finest=disagreements.get(stride,0)) for stride in (4,2,1)],
        seconds=time.monotonic()-started,
        limitations=['Single central axial ray, not an image or a Global rendering.',
            'A finite-grid GP check; nested agreement is not a mathematical continuous-crossing certificate.',
            'Covariance eigentruncation and Monte Carlo uncertainty are reported explicitly.'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
