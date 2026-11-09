"""Audit the stationary isotropic SE transverse-gradient formulas without rendering.

python -B tools/surface_compare/gpis_transverse_gradient_audit.py
<plot-python> -B tools/surface_compare/gpis_transverse_gradient_audit.py --plot

This reproduces the current C++ formulas in Python, compares them with independent
mixed finite differences of the SE covariance, and checks their conditional
moments by Monte Carlo. It does not call or modify the C++ sampler and does not
measure the image impact of changing it. Results are in normalized ray coordinates.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
import numpy as np
from run import ROOT, save, sha

OUT=ROOT/'outputs/gpis_shaderball_compare/ell0p2_renewal_comparison/gradient_audit'
SAMPLES=1_000_000
SEED=20261104


def kernel(a,b):
    return np.exp(-np.dot(a-b,a-b)/4.)


def mixed_difference(u,axis,h=1e-4):
    a=np.zeros(3);b=np.array([0.,0.,u]);v=np.eye(3)[axis]*h
    return float((kernel(a+v,b+v)-kernel(a+v,b-v)-kernel(a-v,b+v)+kernel(a-v,b-v))/(4*h*h))


def moments(ell,d):
    u=np.sqrt(2.)*d/ell
    rho=np.exp(-u*u/4.)
    x=.5*d*d
    c=.5*np.exp(-.5*x)*(1.-x)
    qvar=.5*(-np.expm1(-x)+np.exp(-x)*x*(2.-x)) if x<1 else .5-2*c*c
    r=2*c
    target_var=-.5*np.expm1(-u*u/2.)
    # Average over the regenerated latent starting gradient Z~N(0,1/2).
    current_var=qvar+.5*(r-rho)**2
    return dict(ell=ell,distance=d,normalized_distance=float(u),
        rho_target=float(rho),rho_current=float(r),
        pdf_variance_given_latent_start=float(qvar),
        target_variance_given_observed_start=float(target_var),
        current_variance_given_observed_start=float(current_var),
        variance_ratio=float(current_var/target_var) if target_var else None)


def plot(rows):
    from figures import plt
    ell=.2;ds=np.linspace(0,.6,601)
    ms=[moments(ell,float(d)) for d in ds]
    fig,axes=plt.subplots(1,2,figsize=(12,4.8))
    axes[0].plot(ds,[m['rho_current'] for m in ms],label='Current sampler / PDF',color='#ba4b40')
    axes[0].plot(ds,[m['rho_target'] for m in ms],label='SE transverse conditioning kernel',color='#1675b9')
    axes[0].set_ylabel('Prior endpoint correlation coefficient')
    axes[1].plot(ds,[m['current_variance_given_observed_start'] for m in ms],
        label='Current formula after pathwise update',color='#ba4b40')
    axes[1].plot(ds,[m['target_variance_given_observed_start'] for m in ms],
        label='Ideal SE Renewal+ conditional variance',color='#1675b9')
    axes[1].set_ylabel('Conditional variance (normalized gradient)')
    for ax in axes:
        ax.set_xlabel('World-space segment distance');ax.legend(fontsize=9);ax.grid(alpha=.2)
    fig.suptitle('Transverse gradient audit: stationary isotropic SE, length scale = 0.2')
    fig.text(.08,.02,'Formula analysis; not empirical 3D-global measurements or a corrected render.',fontsize=10)
    fig.tight_layout(rect=(0,.06,1,.94));fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)


def main(make_plot):
    # Mixed derivatives independently distinguish transverse and longitudinal terms.
    derivatives=[]
    for u in (0.,.25,.5,1.,np.sqrt(2.),2.,3.):
        r=np.exp(-u*u/4.)
        transverse=.5*r;longitudinal=(.5-.25*u*u)*r
        tx=mixed_difference(u,0);tz=mixed_difference(u,2)
        assert abs(tx-transverse)<2e-8 and abs(tz-longitudinal)<2e-8
        derivatives.append(dict(normalized_distance=float(u),
            transverse_covariance=float(transverse),transverse_finite_difference=tx,
            longitudinal_covariance=float(longitudinal),longitudinal_finite_difference=tz))
    rng=np.random.default_rng(SEED)
    z=rng.standard_normal(SAMPLES)/np.sqrt(2.)
    epsilon=rng.standard_normal(SAMPLES)
    observed_start=.4
    rows=[]
    for ell,d in ((.2,.01),(.2,.1),(.2,.2),(.2,.4),(.1,.1),(.4,.4)):
        m=moments(ell,d)
        prior=m['rho_current']*z+np.sqrt(m['pdf_variance_given_latent_start'])*epsilon
        current=prior+m['rho_target']*(observed_start-z)
        target=m['rho_target']*observed_start+np.sqrt(m['target_variance_given_observed_start'])*epsilon
        mean=m['rho_target']*observed_start
        assert abs(current.mean()-mean)<6*np.sqrt(m['current_variance_given_observed_start']/SAMPLES)
        assert abs(current.var(ddof=1)/m['current_variance_given_observed_start']-1)<.015
        assert abs(prior.var(ddof=1)-.5)<.0075
        assert abs(target.var(ddof=1)/m['target_variance_given_observed_start']-1)<.015
        # The PDF mean used in C++ matches this sampler conditional on latent Z.
        pdf_mean=m['rho_target']*(observed_start-z)+m['rho_current']*z
        residual=(current-pdf_mean)/np.sqrt(m['pdf_variance_given_latent_start'])
        assert np.allclose(residual,epsilon,rtol=1e-10,atol=1e-12)
        m.update(observed_start=observed_start,conditional_mean=float(mean),
            mc_current_mean=float(current.mean()),mc_current_variance=float(current.var(ddof=1)),
            mc_target_variance=float(target.var(ddof=1)),mc_unconditioned_variance=float(prior.var(ddof=1)))
        rows.append(m)
    source_files=['src/core/math/SparseConvolutionNoise.cpp','src/core/math/GPFunctions.cpp',
        'src/core/math/GPFunctions.hpp','src/core/media/SparseConvolutionNoiseMedium.cpp',
        'src/core/bsdfs/MirrorBsdf.cpp']
    report=dict(scope='Stationary isotropic squared-exponential kernel; one straight segment and observed starting gradient.',
        method='Python formula reproduction, independent mixed finite differences, and Monte Carlo; not a C++ invocation or renderer experiment.',
        normalization='q=sqrt(2)*p/ell, unit-variance field, transverse prior gradient variance=1/2.',
        source_sha256={name:sha(ROOT/name) for name in source_files},
        runner_sha256=sha(Path(__file__)),samples=SAMPLES,seed=SEED,
        mixed_derivative_checks_passed=True,conditional_moment_checks_passed=True,
        sampler_pdf_latent_conditional_consistency_passed=True,
        target_conditional_distribution_matches=False,
        formulas=dict(rho_target='exp(-d*d/(2*ell*ell))',rho_current='exp(-d*d/4)*(1-d*d/2)',
            current_gradient='rho_target*g0 + (rho_current-rho_target)*Z + sqrt((1-rho_current**2)/2)*epsilon',
            target_gradient='rho_target*g0 + sqrt((1-rho_target**2)/2)*epsilon'),
        mixed_derivatives=derivatives,statistics=rows,
        limitations=['Finite sparse-convolution global statistics are not measured by this audit.',
            'No direct attribution of bottom-image differences; corrected-render and MIS/UNI checks are still needed.',
            'The limiting observation at zero distance is distinct from an unconditioned camera-ray origin.'])
    save(OUT/'summary.json',report)
    with (OUT/'moments.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    if make_plot: plot(rows)
    for m in rows:
        print(f"ell={m['ell']:g} d={m['distance']:g}: rho {m['rho_current']:.6f} vs {m['rho_target']:.6f}; conditional variance {m['current_variance_given_observed_start']:.6f} vs {m['target_variance_given_observed_start']:.6f}")
    print('Finite-difference, Monte Carlo and latent-conditional sampler/PDF checks passed.')
    print(OUT/'summary.json')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--plot',action='store_true')
    main(parser.parse_args().plot)
