"""Validate the production gradient sampler against the supplied PDF's 18 cases.

Run the C++ test's `document OUTPUT_DIR 100000` command first. This independent
oracle uses an SE feature expansion and QR conditioning, not the renderer's
Schur complements. A nested finite-prefix reference integrates the terminal
negative-flux Gaussian analytically and samples only the residual GP bridge.
The PDF's rounded numbers are observations to compare, not pass/fail fixtures.
"""
from __future__ import annotations

import argparse
from datetime import date
import hashlib
import html
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
for directory in ("plot-packages-py310", "fp-gradient-validation-deps"):
    packages = ROOT / "build" / directory
    if packages.exists():
        sys.path.insert(0, str(packages))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build" / "matplotlib-cache"))

import numpy as np
import scipy
from scipy.integrate import cumulative_simpson, quad, simpson
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri, log_ndtr
from scipy.stats import qmc, t as student_t


# PDF pp. 12-13, in table order. The linear intercept is taken as zero;
# nonlinear sigma=gamma=1, as also checked by reproducing the ablation means.
DOC = [
    (-.3721,-.3721,-.3721,-1.1323,0,[-.537,-.370,-.212]),
    (-.3413,-.3413,-.3413,-1.1710,0,[-.570,-.327,-.136]),
    (-.5545,-.5465,-.3249,-1.0599,.0209,[-.825,-.541,-.306]),
    (-.6042,-.6027,-.5964,-1.0599,.0016,[-1.076,-.555,-.209]),
    (-1.4885,-1.4885,-1.4885,-1.4428,0,[-2.320,-1.460,-.705]),
    (-.8663,-.8663,-.8408,-1.0599,.0012,[-1.521,-.802,-.310]),
    (-1.3653,-1.3657,-1.0928,-1.3902,.0023,[-2.131,-1.336,-.656]),
    (-1.0444,-1.0427,-1.0304,-1.0599,.0011,[-1.828,-.973,-.373]),
    (-1.5790,-1.5657,-1.4586,-1.4396,.0086,[-2.533,-1.525,-.708]),
    (-1.6726,-1.6674,-1.5045,-1.4909,.0078,[-2.646,-1.624,-.780]),
    (-1.5397,-1.5664,-1.3904,-1.3902,.0189,[-2.490,-1.483,-.678]),
    (-1.3319,-1.3441,-1.2533,-1.2533,.0096,[-2.234,-1.269,-.531]),
    (-.8902,-.8883,-.8171,-.9741,.0020,[-1.538,-.832,-.329]),
    (-.9400,-.9376,-.8939,-.9575,.0018,[-1.641,-.874,-.338]),
    (-1.5819,-1.5761,-1.5515,-1.6454,.0037,[-2.552,-1.529,-.698]),
    (-1.1903,-1.1854,-1.1166,-1.1879,.0034,[-1.985,-1.131,-.485]),
    (-1.7530,-1.7389,-1.6783,-1.6682,.0082,[-2.766,-1.704,-.817]),
    (-1.5265,-1.5360,-1.3848,-1.4036,.0113,[-2.498,-1.473,-.647]),
]
LOG2PI = math.log(2*math.pi)


def mean_functions(case):
    p = case["parameters"]
    kind = case["mean_kind"]
    if kind == "linear":
        return lambda s: p[0]*np.asarray(s), lambda s: np.zeros_like(s)+p[0]
    if kind == "sqrt":
        b, sign, center, radius = p
        return (lambda s: b+sign*np.hypot(np.asarray(s)-center,radius),
                lambda s: sign*(np.asarray(s)-center)/np.hypot(np.asarray(s)-center,radius))
    if kind == "quadratic":
        return (lambda s: p[0]+p[1]*np.asarray(s)+p[2]*np.asarray(s)**2,
                lambda s: p[1]+2*p[2]*np.asarray(s))
    wave, derivative = (np.sin,np.cos) if kind == "sin" else (np.cos,lambda x:-np.sin(x))
    return (lambda s: p[0]+p[1]*np.asarray(s)+p[2]*wave(p[3]*np.asarray(s)),
            lambda s: p[1]+p[2]*p[3]*derivative(p[3]*np.asarray(s)))


class GaussianRay:
    """Independent GP conditioning in an orthonormal coefficient space.

    SE features: sigma exp(-x^2/2) x^n/sqrt(n!), centered on t/2.
    Ninety-six terms make the kernel truncation negligible for these cases.
    Conditioning projects out observed linear functionals using complete QR;
    no diagonal jitter or clamping of a Schur-complement variance is needed.
    """
    def __init__(self, case):
        self.case = case
        self.mean, self.derivative = mean_functions(case)
        self.t = case["t"]
        f, g = self.features([0,self.t])
        observed = np.array([f[0],g[0],f[1]])
        delta = np.array([case["f0"]-self.mean(0),case["g0"]-self.derivative(0),-self.mean(self.t)])
        qr, r = np.linalg.qr(observed.T, mode="complete")
        self.coefficient_mean = qr[:,:3] @ np.linalg.solve(r[:3,:3].T,delta)
        self.free3 = qr[:,3:]
        free_gradient = g[1] @ self.free3
        self.mu = float(self.derivative(self.t)+g[1]@self.coefficient_mean)
        self.sd = float(np.linalg.norm(free_gradient))
        self.unit_gradient = free_gradient/self.sd
        self.free4 = np.linalg.qr(np.vstack([observed,g[1]]).T,mode="complete")[0][:,4:]

    def features(self, positions):
        x = (np.atleast_1d(positions)-self.t/2)*self.case["gamma"]
        f = np.empty((len(x),96))
        f[:,0] = self.case["sigma"]*np.exp(-x*x/2)
        for n in range(1,96):
            f[:,n] = f[:,n-1]*x/math.sqrt(n)
        g = -x[:,None]*f
        g[:,1:] += np.sqrt(np.arange(1,96))*f[:,:-1]
        return f,g*self.case["gamma"]

    def bridge(self, positions):
        f,_ = self.features(positions)
        mu = self.mean(positions)+f@self.coefficient_mean
        beta = (f@self.free3)@self.unit_gradient
        residual = f@self.free4
        return mu,beta,residual

    def anchors(self, positions):
        mu,beta,residual = self.bridge(np.asarray(positions))
        sd = np.linalg.norm(residual,axis=1)
        return mu/sd,beta/sd,(residual@residual.T)/np.outer(sd,sd)


class Density:
    def __init__(self, mu, sd, p=0., q=0., bridge=True):
        self.mu,self.sd,self.p,self.q,self.bridge = mu,sd,p,q,bridge
        self.lo = min(0.,mu)-12*sd
        self.grid = np.linspace(self.lo,0,16385)
        self.mass = quad(self.raw,self.lo,0,epsabs=1e-12,epsrel=1e-11)[0]
        assert self.mass>0
        pdf = self.pdf(self.grid)
        cumulative = cumulative_simpson(pdf,x=self.grid,initial=0)
        assert abs(cumulative[-1]-1)<2e-8
        self.cdf_interpolator = PchipInterpolator(self.grid,cumulative)
        self.mean = quad(lambda v:v*self.raw(v),self.lo,0,epsabs=1e-12)[0]/self.mass
        self.second = quad(lambda v:v*v*self.raw(v),self.lo,0,epsabs=1e-12)[0]/self.mass

    def raw(self, v):
        v = np.asarray(v)
        u = (v-self.mu)/self.sd
        weight = np.exp(-u*u/2-LOG2PI/2)/self.sd
        if self.bridge:
            weight = weight*ndtr(self.p+self.q*u)
        return np.maximum(0,-v)*weight

    def pdf(self, v):
        return self.raw(v)/self.mass

    def cdf(self, v):
        return np.clip(self.cdf_interpolator(np.clip(v,self.lo,0)),0,1)

    def quantiles(self):
        return [brentq(lambda v:self.cdf(v)-q,self.lo,0,xtol=1e-12) for q in [.1,.5,.9]]

    def summary(self):
        return {"mean":self.mean,"sd":math.sqrt(max(0,self.second-self.mean**2)),"quantiles":self.quantiles()}


def bivariate(a,b,rho):
    a,b = sorted((a,b))
    residual = math.sqrt(1-rho*rho)
    # A semi-infinite quadrature ending far to the right can miss the entire
    # Gaussian peak. Keep the peak in a finite interval; omitted mass < 4e-33.
    upper = min(a,12.)
    return quad(lambda x:math.exp(-x*x/2-LOG2PI/2)*ndtr((b-rho*x)/residual),
                min(-12.,upper-12),upper,epsabs=2e-13,epsrel=2e-11)[0]


def single_anchor(ray):
    case = ray.case
    m = ray.mu/ray.sd
    speed = ray.sd*(2/(math.hypot(m,2)+m) if m>=0 else (math.hypot(m,2)-m)/2)
    P = (4*case["f0"]+ray.t*case["g0"])/(ray.t*speed)
    Q = 2*case["f0"]/(ray.t*speed)
    upper = 1.
    while upper**3-P*upper-Q<=0:
        upper *= 2
    x = math.sqrt(P) if case["f0"]==0 else brentq(lambda x:x**3-P*x-Q,0,upper)
    s = max(ray.t*x/(1+x),ray.t-1.5/case["gamma"])
    p,q,_ = ray.anchors([s])
    return Density(ray.mu,ray.sd,float(p[0]),float(q[0])),s


def fixed_five_anchor(ray):
    begin = max(0.,ray.t-1.5/ray.case['gamma'])
    positions = begin+(np.arange(5)+.5)*(ray.t-begin)/5
    p,q,_ = ray.anchors(positions)
    c = -ray.mu/ray.sd
    u0 = -2/(c+math.hypot(c,2)) if c>=0 else .5*(c-math.hypot(c,2))
    selected = int(np.argmin(p+q*u0))
    return Density(ray.mu,ray.sd,float(p[selected]),float(q[selected])),float(positions[selected])


def five_node_document_moments(law):
    """Reproduce the PDF's five-node bivariate CDF and closed moments.

    This approximate normalizer is diagnostic only. Neither the C++ law nor
    our normalized comparison curves use it.
    """
    p,q,mu,sd = law.p,law.q,law.mu,law.sd
    rho = q/math.sqrt(1+q*q)
    alpha = p/math.sqrt(1+q*q)
    c = -mu/sd
    correlation = -rho
    angle = math.asin(correlation)
    nodes,weights = np.polynomial.legendre.leggauss(5)
    if correlation<=.8:
        theta = angle*(1+nodes)/2
        sign,base,width = 1,ndtr(c)*ndtr(alpha),angle
    else:
        theta = (math.pi/2+angle)/2+(math.pi/2-angle)*nodes/2
        sign,base,width = -1,ndtr(min(c,alpha)),math.pi/2-angle
    integrand = np.exp(-(c*c+alpha*alpha-2*c*alpha*np.sin(theta))/(2*np.cos(theta)**2))
    I0 = base+sign*width/(4*math.pi)*np.dot(weights,integrand)
    phi = lambda x:math.exp(-x*x/2-LOG2PI/2)
    residual = math.sqrt(1-rho*rho)
    w = (c+rho*alpha)/residual
    I1 = -phi(c)*ndtr(p+q*c)+rho*phi(alpha)*ndtr(w)
    I2 = I0-c*phi(c)*ndtr(p+q*c)-rho*phi(alpha)*(residual*phi(w)+rho*alpha*ndtr(w))
    Z = -mu*I0-sd*I1
    mean = mu+sd*(-mu*I1-sd*I2)/Z
    exact_I0 = quad(lambda u:phi(u)*ndtr(p+q*u),min(-12,c-12),min(c,12),epsabs=2e-13)[0]
    return {"mean":float(mean),"bivariate_cdf_error":float(I0-exact_I0),
            "mean_error_vs_adaptive_integral":float(mean-law.mean),
            "density_mass_using_approximate_normalizer":float(law.mass/Z)}


def document_reproduction(results, production):
    by_id = {r["id"]:r for r in results}
    for i,case in enumerate(production["cases"]):
        if case["id"] not in by_id:
            continue
        ray = GaussianRay(case)
        law,_ = single_anchor(ray)
        r = by_id[case["id"]]
        five_node = five_node_document_moments(law)
        errors = {"single_anchor_mean":five_node["mean"]-DOC[i][1],
                  "start_only_mean":r["start_only"]["mean"]-DOC[i][2],
                  "naive_mean":r["naive"]["mean"]-DOC[i][3]}
        r["document_reproduction"] = {"five_node":five_node,"rounded_table_errors":errors,
            "analytic_means_match_table_rounding":all(abs(e)<=.0000501 for e in errors.values())}


def reference_self_checks(production):
    results = []
    for i in [2,11,17]:
        ray = GaussianRay(production["cases"][i])
        law,s = single_anchor(ray)
        mu,beta,residual = ray.bridge([s])
        uniforms = qmc.Sobol(1,scramble=True,seed=910+i,bits=52).random_base2(18).ravel()
        normals = ndtri((uniforms*2.**52+.5)/2.**52)
        reference = PrefixLaw((mu[0]+np.linalg.norm(residual)*normals)/(-beta[0]),ray.mu,ray.sd)
        grid = np.linspace(law.lo,0,5001)
        positions = np.linspace(0,ray.t,31)
        features,_ = ray.features(positions)
        exact = ray.case["sigma"]**2*np.exp(-.5*(ray.case["gamma"]*(positions[:,None]-positions[None,:]))**2)
        kernel_error = float(np.max(abs(features@features.T-exact)))
        ks = float(np.max(abs(reference.cdf(grid)-law.cdf(grid))))
        mean_error = reference.mean-law.mean
        results.append(dict(id=ray.case["id"],kernel_max_error=kernel_error,
            single_point_control_ks=ks,single_point_control_mean_error=mean_error,
            pass_=bool(kernel_error<1e-13 and ks<1e-6 and abs(mean_error)<1e-6)))
    return results


def flux_moments(b,mu,sd):
    phi = np.exp(-b*b/2-LOG2PI/2)
    Phi = ndtr(b)
    mass = -mu*Phi+sd*phi
    first = -(mu*mu+sd*sd)*Phi+(2*mu*sd+sd*sd*b)*phi
    second = -(mu**3+3*mu*sd*sd)*Phi+(3*mu*mu*sd+3*mu*sd*sd*b+sd**3*(b*b+2))*phi
    return mass,first,second


class PrefixLaw:
    def __init__(self, bounds, mu, sd):
        self.mu,self.sd = mu,sd
        self.bounds = np.sort(np.minimum(bounds,-mu/sd))
        mass,first,second = flux_moments(self.bounds,mu,sd)
        assert np.all(mass>=0) and all(np.all(np.isfinite(v)) for v in [mass,first,second])
        self.prefix_mass = np.r_[0,np.cumsum(mass)]
        self.total = self.prefix_mass[-1]
        self.mean = float(np.sum(first)/self.total)
        self.second = float(np.sum(second)/self.total)
        self.lo = min(0,mu)-12*sd

    def cdf(self, v):
        u = np.minimum((np.asarray(v)-self.mu)/self.sd,-self.mu/self.sd)
        k = np.searchsorted(self.bounds,u,side="right")
        mass = flux_moments(u,self.mu,self.sd)[0]
        return (self.prefix_mass[k]+(len(self.bounds)-k)*mass)/self.total

    def pdf(self, v):
        v = np.asarray(v)
        u = (v-self.mu)/self.sd
        k = np.searchsorted(self.bounds,u,side="right")
        return np.maximum(0,-v)*np.exp(-u*u/2-LOG2PI/2)/self.sd*(len(self.bounds)-k)/self.total

    def summary(self):
        return {"mean":self.mean,"sd":math.sqrt(max(0,self.second-self.mean**2)),
                "quantiles":[brentq(lambda v:self.cdf(v)-q,self.lo,0,xtol=1e-11) for q in [.1,.5,.9]]}


def prefix_references(ray, power, replicates, seed):
    # 25, 51 and 103 strictly interior points, nested on the same GP draws.
    points = np.arange(1,104)*ray.t/104
    mu,beta,residual = ray.bridge(points)
    assert np.all(beta<0), "This reference requires negative conditional slope correlations"
    u,s,_ = np.linalg.svd(residual,full_matrices=False)
    keep = s>s[0]*1e-12
    factor = u[:,keep]*s[keep]
    subsets = {25:np.arange(3,103,4),51:np.arange(1,103,2),103:np.arange(103)}
    all_bounds = {n:[] for n in subsets}
    replicate_means = {n:[] for n in subsets}
    replicate_cdfs = {n:[] for n in subsets}
    probes = np.linspace(min(0,ray.mu)-8*ray.sd,0,1001)
    for repeat in range(replicates):
        uniforms = qmc.Sobol(int(sum(keep)),scramble=True,seed=seed+repeat,bits=52).random_base2(power)
        # Open-interval midpoints of the 52-bit cells; inverse Phi never sees 0.
        uniforms = (uniforms*2.**52+.5)/2.**52
        normal = ndtri(uniforms)
        assert np.all(np.isfinite(normal))
        bounds = (normal@factor.T+mu)/(-beta)
        for n,index in subsets.items():
            b = np.min(bounds[:,index],axis=1)
            law = PrefixLaw(b,ray.mu,ray.sd)
            all_bounds[n].append(b)
            replicate_means[n].append(law.mean)
            replicate_cdfs[n].append(law.cdf(probes))
    laws = {n:PrefixLaw(np.concatenate(all_bounds[n]),ray.mu,ray.sd) for n in subsets}
    stats = {"points":{}}
    for n,law in laws.items():
        error = np.std(replicate_means[n],ddof=1)/math.sqrt(replicates)
        cdf_se = np.std(replicate_cdfs[n],axis=0,ddof=1)/math.sqrt(replicates)
        stats["points"][str(n)] = dict(law.summary(),mean_standard_error=float(error),
            mean_approx_ci95_half_width=float(student_t.ppf(.975,replicates-1)*error),
            max_pointwise_cdf_standard_error=float(np.max(cdf_se)))
    stats.update(rank=int(sum(keep)),largest_discarded_singular_value=float(max(s[~keep],default=0)),
                 feature_count=96,qmc_power=power,replicates=replicates,seed=seed,
                 ks_25_vs_103=float(np.max(np.abs(laws[25].cdf(probes)-laws[103].cdf(probes)))),
                 ks_51_vs_103=float(np.max(np.abs(laws[51].cdf(probes)-laws[103].cdf(probes)))))
    return laws,stats


def differences(model,reference,grid):
    return {"mean_error":float(model.mean-reference.mean),
            "ks":float(np.max(np.abs(model.cdf(grid)-reference.cdf(grid)))),
            "overlap":float(1-.5*simpson(np.abs(model.pdf(grid)-reference.pdf(grid)),x=grid))}


def empirical_checks(samples,law):
    values = np.sort(samples)
    n = len(values)
    cdf = law.cdf(values)
    ks = max(np.max(np.arange(1,n+1)/n-cdf),np.max(cdf-np.arange(n)/n))
    # Family-wise DKW bound over 18 cases and two RNG adapters, alpha=1e-6.
    threshold = math.sqrt(math.log(2*36/1e-6)/(2*n))
    return {"count":n,"mean":float(np.mean(values)),"quantiles":np.quantile(values,[.1,.5,.9]).tolist(),
            "ks_to_implemented_law":float(ks),"familywise_dkw_threshold":threshold,
            "pass":bool(np.all(np.isfinite(values)) and np.all(values<0) and ks<threshold)}


def validate_case(case, index, root, args):
    ray = GaussianRay(case)
    current = Density(case["gaussian_mean"],case["gaussian_sd"],case["p"],case["q"])
    new,sstar = single_anchor(ray)
    start = Density(ray.mu,ray.sd,bridge=False)
    naive = Density(float(ray.derivative(ray.t)),case["sigma"]*case["gamma"],bridge=False)
    fixed = case.get("approximation")=="single_anchor_fixed5_snr"
    single = fixed or case.get("approximation")=="single_anchor_v2"
    selected,position = fixed_five_anchor(ray) if fixed else (new,sstar)
    positions = [position] if single else [ray.t/3,max(2*ray.t/3,ray.t-1/case["gamma"])]
    p,q,rho = ray.anchors(positions)
    endpoint_error = max(abs(ray.mu-case["gaussian_mean"]),abs(ray.sd-case["gaussian_sd"]))
    anchor_error = max(np.max(abs(p-case["anchor_p"])/(1+abs(p))),
                       np.max(abs(q-case["anchor_q"])/(1+abs(q))),
                       abs(position-case["anchor_distance"])/(1+ray.t) if single else abs(rho[0,1]-case["anchor_correlation"]))
    if single:
        tangent_error = max(abs(selected.p-case["p"])/(1+abs(selected.p)),abs(selected.q-case["q"])/(1+abs(selected.q)))
    else:
        u0 = (-ray.mu/ray.sd-math.hypot(ray.mu/ray.sd,2))/2
        a,b = p+q*u0
        r = float(rho[0,1]); residual = math.sqrt(1-r*r)
        probability = bivariate(float(a),float(b),r)
        derivative = (q[0]*math.exp(-a*a/2-LOG2PI/2)*ndtr((b-r*a)/residual)
                      +q[1]*math.exp(-b*b/2-LOG2PI/2)*ndtr((a-r*b)/residual))
        z = case["p"]+case["q"]*u0
        tangent_error = max(abs(ndtr(z)-probability),abs(math.exp(-z*z/2-LOG2PI/2)*case["q"]-derivative))
    exported = np.asarray(case["grid"])
    pdf_error = float(np.max(np.abs(exported[:,1]-current.pdf(exported[:,0]))))
    cdf_error = float(np.max(np.abs(exported[:,2]-current.cdf(exported[:,0]))))
    sampling = []
    for sample in case["samples"]:
        values = np.fromfile(root/sample["file"],dtype="<f8")
        assert len(values)==sample["count"]
        sampling.append(dict(empirical_checks(values,current),rng=sample["rng"],fallbacks=sample["fallbacks"]))
    implementation = dict(endpoint_max_error=endpoint_error,anchor_max_relative_error=float(anchor_error),
        correction_parameter_max_error=float(tangent_error),pdf_max_error=pdf_error,cdf_max_error=cdf_error,
        pass_=bool(endpoint_error<1e-9 and anchor_error<1e-5 and tangent_error<1e-7
                   and pdf_error<1e-7 and cdf_error<1e-7 and all(s["pass"] for s in sampling)))
    if args.reference_directory:
        reference_path = args.reference_directory/(case["id"]+".validation.json")
        previous = json.loads(reference_path.read_text(encoding="utf-8"))
        # Reuse only the unchanged GP reference and already evaluated single-
        # anchor/model differences; all production/RNG checks above run anew.
        assert single and abs(new.mean-previous["single_anchor"]["mean"])<1e-10
        assert previous["input"]=={k:case[k] for k in ["mean_kind","parameters","t","f0","g0","sigma","gamma"]}
        result = dict(previous)
        result.update(implementation=implementation,sampling=sampling,current=current.summary(),
            current_vs_25=previous["single_vs_25"],current_vs_103=previous["single_vs_103"])
        curves = result["curves"]; v = np.asarray(curves["v"])
        curves["current"] = dict(pdf=current.pdf(v).tolist(),cdf=current.cdf(v).tolist())
        if fixed:
            # The new selector changes the law: never reuse Hermite error
            # metrics. Compare anew on the archived reference curve grid.
            for n in (25,103):
                ref = curves['reference'+str(n)]
                result['current_vs_'+str(n)] = dict(
                    mean_error=current.mean-result['reference']['points'][str(n)]['mean'],
                    ks=float(np.max(abs(current.cdf(v)-ref['cdf']))),
                    overlap=float(1-.5*simpson(abs(current.pdf(v)-ref['pdf']),x=v)),
                    evaluation_grid_points=len(v))
            result['current_anchor_selection'] = 'fixed5_snr'
            result['current_anchor_fraction'] = position/ray.t
        result["reference_reused_from"] = str(reference_path)
        result["reference_reused_sha256"] = hashlib.sha256(reference_path.read_bytes()).hexdigest()
        return result
    laws,reference = prefix_references(ray,args.power,args.replicates,args.seed+100*index)
    grid = np.linspace(min(current.lo,new.lo),0,16385)
    doc = DOC[index]
    result = dict(id=case["id"],input={k:case[k] for k in ["mean_kind","parameters","t","f0","g0","sigma","gamma"]},
        implementation=implementation,sampling=sampling,current=current.summary(),single_anchor=new.summary(),
        single_anchor_fraction=sstar/ray.t,start_only=start.summary(),naive=naive.summary(),reference=reference,
        document=dict(reference_mean=doc[0],single_anchor_mean=doc[1],start_only_mean=doc[2],naive_mean=doc[3],
                      single_anchor_ks=doc[4],reference_quantiles=doc[5]),
        current_vs_25=differences(current,laws[25],grid),current_vs_103=differences(current,laws[103],grid),
        single_vs_25=differences(new,laws[25],grid),single_vs_103=differences(new,laws[103],grid))
    curves = np.linspace(min(current.quantiles()[0],new.quantiles()[0],laws[103].summary()["quantiles"][0])-2*ray.sd,0,801)
    result["curves"] = {"v":curves.tolist()}
    for label,law in [("current",current),("single_anchor",new),("reference25",laws[25]),("reference103",laws[103]),("start_only",start),("naive",naive)]:
        result["curves"][label] = {"pdf":law.pdf(curves).tolist(),"cdf":law.cdf(curves).tolist()}
    return result


def plots(results, root):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    labels = {"reference103":"103-point GP reference","current":"Current C++ law", "single_anchor":"PDF single anchor","start_only":"Start only","naive":"Memoryless Rice"}
    colors = {"reference103":"#1e293b","current":"#2563eb","single_anchor":"#d97706","start_only":"#16a34a","naive":"#94a3b8"}
    saved = []
    for metric in ["pdf","cdf"]:
        fig,axes = plt.subplots(6,3,figsize=(14,20),layout="constrained")
        for ax,result in zip(axes.flat,results):
            curves = result["curves"]
            for label in labels:
                ax.plot(curves["v"],curves[label][metric],label=labels[label],color=colors[label],
                        linewidth=1.6 if label in ["reference103","current"] else 1.1,
                        linestyle="--" if label in ["single_anchor","naive"] else "-")
            ax.set_title(result["id"]+f" | current KS={result['current_vs_103']['ks']:.4f}")
            ax.set_xlabel("Longitudinal gradient v"); ax.set_ylabel(metric.upper()); ax.grid(alpha=.2)
            if metric=="cdf": ax.set_ylim(0,1)
        axes.flat[0].legend(fontsize=7)
        fig.savefig(root/f"{metric}-comparison.png",dpi=150)
        fig.savefig(root/f"{metric}-comparison.svg")
        plt.close(fig)
        saved.append(f"{metric}-comparison.png")
    return saved


def report(payload, root):
    rows = []
    for r in payload["cases"]:
        ci = r["reference"]["points"]["103"]["mean_approx_ci95_half_width"]
        ref = r["reference"]["points"]["103"]["mean"]
        rows.append(f'<tr><td>{r["id"]}</td><td>{r["current"]["mean"]:.6f}</td><td>{ref:.6f} ± {ci:.2g}</td>'
                    f'<td>{r["document"]["reference_mean"]:.4f}</td><td>{r["single_anchor"]["mean"]:.6f}</td>'
                    f'<td>{r["current_vs_103"]["ks"]:.5f}</td><td>{100*r["current_vs_103"]["overlap"]:.2f}%</td>'
                    f'<td>{r["reference"]["ks_25_vs_103"]:.5f}</td><td>{max(x["ks_to_implemented_law"] for x in r["sampling"]):.5f}</td></tr>')
    figures = ''.join(f'<a href="{p}"><img src="{p}" alt="distribution comparison"></a>' for p in payload["plots"])
    details = []
    for r in payload["cases"]:
        details.append('<details><summary>'+r["id"]+'：参数、分位数与数值检查</summary><pre>'+
                       html.escape(json.dumps({k:v for k,v in r.items() if k!="curves"},ensure_ascii=False,indent=2))+'</pre></details>')
    content = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>FP gradient 文档实验验证</title>
<style>body{max-width:1400px;margin:36px auto;padding:0 24px;font:16px/1.6 system-ui,"Microsoft YaHei",sans-serif;color:#1e293b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}th{background:#f1f5f9}img{max-width:100%;margin:24px 0}pre{white-space:pre-wrap;font-size:13px;background:#f8fafc;padding:16px}details{margin:12px 0}a{color:#2563eb}</style>
<h1>当前 FP 梯度实现：文档 18 组实验验证</h1>
<p>当前生产模型：MODEL_DESCRIPTION。此报告验证固定碰撞位置的纵向梯度，不验证首次碰撞距离或整幅图像的物理正确性。</p>
<p>每组从实际 C++ Law::sample 抽取两套各 100,000 个样本，分别使用独立 64 位 RNG 和渲染器 RNG。高斯参考用 96 项 SE 特征展开和 QR 条件化独立构造。
使用同一批桥样本检查 25 / 51 / 103 个嵌套内部点的正性，终点负通量高斯解析积分。103 点仍是有限离散参考，并非严格连续路径真值。</p>
<p>PDF 未提供实验脚本、随机种子或 25 点具体位置。本次使用等距内部点；线性均值截距设为 0，非线性场设 σ=γ=1，并通过复算仅起点及无记忆均值核对这些约定。
表中 ± 为多个独立扰乱 Sobol 重复之间估计的约 95% 置信半宽，不包含路径离散误差。KS 为最大 CDF 差；重合度为 1−TV。</p>
<p><a href="validation.json">完整 JSON 数据</a> · <a href="production.json">实际 C++ 导出</a> · <a href="metrics.csv">指标 CSV</a></p>'''
    descriptions = {'single_anchor_v2':'Hermite 单 anchor，直接使用单点正性概率',
                    'single_anchor_fixed5_snr':'固定 5 个候选点，按真实 SDF 的 bridge SNR 选择 1 个 anchor'}
    content = content.replace('MODEL_DESCRIPTION',descriptions.get(payload.get('approximation'),'旧版双 anchor + probit 切线'))
    content += '<p><strong>采样器与数值实现检查：'+('全部通过' if payload["implementation_pass"] else '存在未通过项')+'</strong></p>'
    if payload.get('reference_directory'):
        content += '<p>GP 参考和其网格收敛检查沿用上一份文档验证数据；本次重新导出生产参数、PDF/CDF，并重新检查两套 RNG 的各 100,000 个样本。</p>'
    content += '<p>文档的 18 组解析均值可以在其四位小数精度下复现。部分单锚点均值依赖文档给出的五点二元 CDF 近似：例如 L06 的表格值为 −0.8663，五点复算为 −0.8663185，而自适应积分为 −0.8646770。图中使用自适应积分归一化；这部分差异来自文档的数值积分近似。</p>'
    content += '<table><tr><th>算例</th><th>当前均值</th><th>103 点参考均值</th><th>文档参考均值</th><th>复算新版均值</th><th>当前 vs 参考 KS</th><th>重合度</th><th>25→103 KS</th><th>采样器 KS</th></tr>'+''.join(rows)+'</table>'
    content += figures+''.join(details)+'</html>'
    (root/"comparison.html").write_text(content,encoding="utf-8")
    import csv
    with (root/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case","current_mean","reference103_mean","document_reference_mean","single_anchor_mean","current_ks","current_overlap","single_anchor_ks","prefix25_to_103_ks"])
        for r in payload["cases"]:
            writer.writerow([r["id"],r["current"]["mean"],r["reference"]["points"]["103"]["mean"],r["document"]["reference_mean"],r["single_anchor"]["mean"],r["current_vs_103"]["ks"],r["current_vs_103"]["overlap"],r["single_vs_103"]["ks"],r["reference"]["ks_25_vs_103"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory",type=Path)
    parser.add_argument("--power",type=int,default=15)
    parser.add_argument("--replicates",type=int,default=8)
    parser.add_argument("--seed",type=int,default=20261108)
    parser.add_argument("--case",help="Restrict to one case for diagnostics")
    parser.add_argument("--report-only",action="store_true",help="Reuse completed numerical references; rebuild report and PDF table comparisons")
    parser.add_argument("--document",type=Path,help="Record the supplied PDF's filename and SHA-256")
    parser.add_argument("--reference-directory",type=Path,help="Reuse frozen GP references; fixed5 errors use their stored curve grid")
    args = parser.parse_args()
    assert args.replicates>=2
    root = args.directory
    production = json.loads((root/"production.json").read_text())
    results = json.loads((root/"validation.json").read_text(encoding="utf-8"))["cases"] if args.report_only else []
    for i,case in enumerate([] if args.report_only else production["cases"]):
        if args.case and case["id"]!=args.case:
            continue
        begin = time.perf_counter()
        result = validate_case(case,i,root,args)
        results.append(result)
        print(f'{case["id"]}: implementation={result["implementation"]["pass_"]} '
              f'mean={result["current"]["mean"]:.6f} ref={result["reference"]["points"]["103"]["mean"]:.6f} '
              f'KS={result["current_vs_103"]["ks"]:.6f} ({time.perf_counter()-begin:.1f}s)',flush=True)
        (root/(case["id"]+".validation.json")).write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    document_reproduction(results,production)
    oracle_checks = reference_self_checks(production)
    payload = {"method":"SE feature QR + negative-flux integrated randomized Sobol bridge",
               "approximation":production["cases"][0].get("approximation","two_anchor_v1"),
               "reference_directory":str(args.reference_directory) if args.reference_directory else None,
               "date":date.today().isoformat(),"numpy_version":np.__version__,"scipy_version":scipy.__version__,
               "implementation_pass":all(r["implementation"]["pass_"] for r in results),
               "reference_self_checks":oracle_checks,
               "reference_self_checks_pass":all(r["pass_"] for r in oracle_checks),
               "document_analytic_means_reproduced":all(r["document_reproduction"]["analytic_means_match_table_rounding"] for r in results),
               "cases":results,"plots":plots(results,root),
               "source_sha256":{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                   [ROOT/"src/core/math/FirstPassageGradient.cpp",ROOT/"tests/FpGradientDocumentExport.hpp",Path(__file__)]}}
    payload["production_sha256"] = hashlib.sha256((root/"production.json").read_bytes()).hexdigest()
    if args.document:
        payload["document"] = {"name":args.document.name,"sha256":hashlib.sha256(args.document.read_bytes()).hexdigest()}
    (root/"validation.json").write_text(json.dumps(payload,indent=2,allow_nan=False),encoding="utf-8")
    report(payload,root)
    return 0 if all(payload[k] for k in ["implementation_pass","reference_self_checks_pass","document_analytic_means_reproduced"]) else 1


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
