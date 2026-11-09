"""Fixed-collision shaderball gradient diagnosis (not an image-error estimate).

Build fp_gradient_rays, run its scan command with spec.json, then:
  python -B tools/surface_compare/gradient_rays.py screen OUTPUT_DIR
  python -B tools/surface_compare/gradient_rays.py refine OUTPUT_DIR --ids ...

The reference conditions the actual SDF ray on F(0), F'(0), F(t)=0 and
positivity on nested interior grids. SE features and QR are independent of
production Schur complements. Terminal crossing flux is integrated exactly;
scrambled Sobol draws integrate the residual bridge. No path resampling,
free-flight correction or renderer change is performed by this script.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "build/fp-gradient-validation-deps"))
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "build/matplotlib-cache"))
import numpy as np
from scipy.integrate import cumulative_simpson, simpson
from scipy.interpolate import PchipInterpolator
from scipy.optimize import brentq
from scipy.special import erfcx, log_ndtr, ndtri
from scipy.stats import qmc, t as student_t

LOG2PI = math.log(2*math.pi)

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

def mills(x):
    x = np.asarray(x)
    with np.errstate(over="ignore", invalid="ignore"):
        return np.where(x < -5, math.sqrt(2/math.pi)/erfcx(-x/math.sqrt(2)),
                        np.exp(-x*x/2-LOG2PI/2-log_ndtr(x)))

class Density:
    """Scaled quadrature, including very unlikely negative Gaussian tails."""
    def __init__(self, mu, sd, p=0., q=0., bridge=False):
        self.mu, self.sd, self.p, self.q = map(float, (mu,sd,p,q))
        self.bridge = bridge
        m = mu/sd
        def derivative(x):
            u = -x-m
            return 1/x+u-(q*float(mills(p+q*u)) if bridge else 0.)
        hi = max(1.,1-m)
        while derivative(hi)>0:
            hi *= 2
        mode = brentq(derivative, np.finfo(float).tiny, hi, xtol=5e-15)
        def relative(x):
            x = np.asarray(x)
            with np.errstate(divide="ignore", invalid="ignore"):
                y = np.log(x/mode)-.5*(x-mode)*(x+mode+2*m)
            if bridge:
                y = y + log_ndtr(p+q*(-x-m))-log_ndtr(p+q*(-mode-m))
            return y
        high = mode+max(1.,mode/8)
        while relative(high)>-42:
            high = mode+2*(high-mode)
        high = brentq(lambda x:float(relative(x))+42,mode,high)
        low = brentq(lambda x:float(relative(x))+42,np.finfo(float).tiny,mode,xtol=5e-16)
        x = np.linspace(low,high,4097)
        values = np.exp(relative(x))
        mass = simpson(values,x=x)
        self.grid = -sd*x[::-1]
        self.values = values[::-1]/(mass*sd)
        self.lo,self.hi = float(self.grid[0]),float(self.grid[-1])
        c = cumulative_simpson(self.values,x=self.grid,initial=0)
        c = np.maximum.accumulate(np.clip(c/c[-1],0,1))
        self.interpolator = PchipInterpolator(self.grid,c)
        self.mode = -sd*mode
        self.mean = float(simpson(self.grid*self.values,x=self.grid))
        self.second = float(simpson(self.grid**2*self.values,x=self.grid))
        self.relative, self.mass = relative, mass

    def cdf(self,v):
        return self.interpolator(np.clip(v,self.lo,self.hi))

    def pdf(self,v):
        v = np.asarray(v)
        return np.where(v<0,np.exp(self.relative(np.maximum(-v/self.sd,np.finfo(float).tiny)))/(self.mass*self.sd),0.)

    def summary(self):
        return dict(mean=self.mean,sd=math.sqrt(max(0,self.second-self.mean**2)),
                    quantiles=[brentq(lambda v:float(self.cdf(v))-p,self.lo,self.hi) for p in [.1,.5,.9]])

class Ray:
    def __init__(self,r):
        self.r,self.t = r,r["t"]
        # The largest normalized distance from the feature expansion center.
        # The feature-square tail is a Poisson tail with mean z^2.
        z = r["gamma"]*self.t/2
        self.count = max(96,int(math.ceil(z*z+14*z+64)))
        f,g = self.features([0,self.t])
        observed = np.array([f[0],g[0],f[1]])
        delta = np.array([r["f0"]-r["mean_profile"][0],
                          r["g0"]-r["birth_mean_derivative"],-r["mean_profile"][-1]])
        qr,upper = np.linalg.qr(observed.T,mode="complete")
        self.coefficient_mean = qr[:,:3]@np.linalg.solve(upper[:3,:3].T,delta)
        self.free3 = qr[:,3:]
        free_gradient = g[1]@self.free3
        self.mu = float(r["end_mean_derivative"]+g[1]@self.coefficient_mean)
        self.sd = float(np.linalg.norm(free_gradient))
        self.unit_gradient = free_gradient/self.sd
        self.free4 = np.linalg.qr(np.vstack([observed,g[1]]).T,mode="complete")[0][:,4:]

    def features(self,positions):
        r = self.r
        x = (np.atleast_1d(positions)-self.t/2)*r["gamma"]
        f = np.empty((len(x),self.count))
        f[:,0] = r["sigma"]*np.exp(-x*x/2)
        for i in range(1,self.count):
            f[:,i] = f[:,i-1]*x/math.sqrt(i)
        g = -x[:,None]*f
        g[:,1:] += np.sqrt(np.arange(1,self.count))*f[:,:-1]
        return f,g*r["gamma"]

    def bridge(self,denominator):
        original = self.r["profile_denominator"]
        assert original%denominator==0, "Reference uses exact exported SDF nodes, no interpolation"
        means = np.asarray(self.r["mean_profile"])[::original//denominator][1:-1]
        f,_ = self.features(np.arange(1,denominator)*self.t/denominator)
        return means+f@self.coefficient_mean,(f@self.free3)@self.unit_gradient,f@self.free4

    def checks(self):
        p = self.r["production"]
        x = np.linspace(0,self.t,31)
        f,g = self.features(x)
        k = self.r["sigma"]**2*np.exp(-.5*(self.r["gamma"]*(x[:,None]-x[None,:]))**2)
        return dict(endpoint_error=max(abs(self.mu-p["mu"]),abs(self.sd-p["sd"])),
                    kernel_relative_error=float(np.max(abs(f@f.T-k)))/self.r["sigma"]**2,
                    derivative_variance_relative_error=float(np.max(abs(np.sum(g*g,axis=1)-(self.r["sigma"]*self.r["gamma"])**2)))/(self.r["sigma"]*self.r["gamma"])**2,
                    feature_count=self.count)

class Prefix:
    def __init__(self,bounds,mu,sd):
        self.mu,self.sd,self.m = mu,sd,mu/sd
        self.bounds = np.sort(np.minimum(bounds,-mu/sd))
        logs = self.log_mass(self.bounds)
        self.scale = float(np.max(logs))
        weights = np.exp(logs-self.scale)
        self.prefix = np.r_[0,np.cumsum(weights)]
        self.total = self.prefix[-1]
        self.ess = float(self.total**2/np.sum(weights**2))
        self.max_weight = float(np.max(weights)/self.total)
        self.log_survival_ratio = float(self.scale+math.log(self.total/len(weights))-self.log_mass(-mu/sd))

    def log_mass(self,b):
        b = np.minimum(np.asarray(b),-self.m)
        difference = mills(b)-self.m
        if np.any(difference<=0):
            raise ArithmeticError("Unresolved truncated flux cancellation")
        return math.log(self.sd)+log_ndtr(b)+np.log(difference)

    def cdf(self,v):
        u = np.minimum((np.asarray(v)-self.mu)/self.sd,-self.m)
        k = np.searchsorted(self.bounds,u,side="right")
        # Where k=N the tail term is zero; clipping prevents an irrelevant overflow.
        mass = np.exp(np.minimum(0,self.log_mass(u)-self.scale))
        return np.clip((self.prefix[k]+(len(self.bounds)-k)*mass)/self.total,0,1)

    def pdf(self,v):
        v = np.asarray(v)
        u = (v-self.mu)/self.sd
        k = np.searchsorted(self.bounds,u,side="right")
        with np.errstate(divide="ignore",over="ignore",invalid="ignore"):
            result = np.exp(np.log(np.maximum(0,-v))-u*u/2-LOG2PI/2-math.log(self.sd)
                            +np.log(len(self.bounds)-k)-self.scale-math.log(self.total))
        return np.nan_to_num(result,nan=0.)

    def summarize(self,grid):
        F = self.cdf(grid)
        mean = -simpson(F,x=grid)
        second = -2*simpson(grid*F,x=grid)
        return dict(mean=float(mean),sd=math.sqrt(max(0,float(second-mean*mean))),
                    quantiles=[brentq(lambda v:float(self.cdf(v))-q,grid[0],grid[-1]) for q in [.1,.5,.9]],
                    residual_weight_ess=self.ess,max_residual_weight_fraction=self.max_weight,
                    log_prefix_mass_over_endpoint_flux=self.log_survival_ratio)

def models(r):
    p = r["production"]
    return dict(classic=Density(r["end_mean_derivative"],r["sigma"]*r["gamma"]),
                conditional=Density(p["mu"],p["sd"]),
                bridge=Density(p["mu"],p["sd"],p["p"],p["q"],True))

def grid_for(laws,reference=None,size=8193):
    lo = min(m.lo for m in laws.values())
    if reference is not None:
        while reference.cdf(lo)>1e-12:
            lo *= 2
    pieces = [np.linspace(lo,0,size)]
    pieces += [m.grid[::2] for m in laws.values()]
    return np.unique(np.concatenate(pieces))

def ks(a,b,grid):
    return float(np.max(abs(a.cdf(grid)-b.cdf(grid))))

def reference(ray,power,replicates,denominators,seed):
    maximum = max(denominators)
    mean,beta,residual = ray.bridge(maximum)
    negligible = beta>=0
    marginal_sd = np.sqrt(beta**2+np.sum(residual**2,axis=1))
    if np.any(negligible):
        # At very long distances the true negative covariance can be below QR
        # roundoff. Omit only constraints whose marginal failure probability
        # has a separately recorded union bound < 2e-33 per node.
        if np.any(abs(beta[negligible])>ray.r["sigma"]*1e-12) or np.any(mean[negligible]/marginal_sd[negligible]<12):
            raise ArithmeticError("Unresolved nonnegative slope correlation at a relevant bridge node")
    u,s,_ = np.linalg.svd(residual,full_matrices=False)
    keep = s>s[0]*1e-12
    factor = u[:,keep]*s[keep]
    bounds = {n:[] for n in denominators}
    repeats = {n:[] for n in denominators}
    for rep in range(replicates):
        uniforms = qmc.Sobol(int(sum(keep)),scramble=True,seed=seed+rep,bits=52).random_base2(power)
        normal = ndtri((uniforms*2.**52+.5)/2.**52)
        # Bound temporary matrices even when resolving sharp SDF features with
        # several thousand prefix nodes.
        collected = {n:[] for n in denominators}
        chunk=max(64,min(4096,2**int(math.floor(math.log2(2**21/maximum)))))
        for start in range(0,len(normal),chunk):
            b = (mean+normal[start:start+chunk]@factor.T)/(-beta)
            b[:,negligible] = np.inf
            for n in denominators:
                index = np.arange(maximum//n-1,maximum-1,maximum//n)
                collected[n].append(np.min(b[:,index],axis=1))
        for n in denominators:
            b = np.concatenate(collected[n])
            bounds[n].append(b)
            repeats[n].append(Prefix(b,ray.mu,ray.sd))
    laws = {n:Prefix(np.concatenate(v),ray.mu,ray.sd) for n,v in bounds.items()}
    return laws,repeats,dict(power=power,replicates=replicates,seed=seed,
        rank=int(sum(keep)),points=[n-1 for n in denominators],
        negligible_roundoff_nodes=int(sum(negligible)),
        omitted_node_failure_union_bound=float(np.sum(np.exp(log_ndtr(-mean[negligible]/marginal_sd[negligible])))),
        largest_discarded_singular_value=float(max(s[~keep],default=0)))

def screen_one(r,threshold,power):
    try:
        laws = models(r)
        grid = grid_for(laws,size=2049)
        change = ks(laws["conditional"],laws["bridge"],grid)
        row = dict(id=r["id"],source=r["source"],pixel=r["pixel"],sample=r["sample"],
            bounce=r["bounce"],normalized_age=r["t"]*r["gamma"],bridge_change_ks=change)
        if change>=threshold:
            ray = Ray(r)
            ref,reps,stats = reference(ray,power,2,[52,104],331000+r["id"]*2)
            grid = grid_for(laws,ref[104],size=4097)
            row["errors"] = {name:ks(law,ref[104],grid) for name,law in laws.items()}
            row["gain_vs_conditional"] = row["errors"]["conditional"]-row["errors"]["bridge"]
            row["gain_vs_classic"] = row["errors"]["classic"]-row["errors"]["bridge"]
            row["grid_ks"] = ks(ref[52],ref[104],grid)
            row["replicate_ks"] = ks(reps[104][0],reps[104][1],grid)
            row["prefix"] = ref[104].summarize(grid)
            row["checks"] = ray.checks()
        return row,None
    except Exception as e:
        return None,dict(id=r["id"],error=str(e))

def screen(folder,args):
    data = read(folder/"rays.json")
    rows,failed = [],[]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        work = partial(screen_one,threshold=args.threshold,power=args.power)
        for i,(row,failure) in enumerate(pool.map(work,data["records"],chunksize=8)):
            if failure: failed.append(failure)
            else: rows.append(row)
            if (i+1)%100==0:
                print(f"screen {i+1}/{len(data['records'])}, referenced {sum('errors' in r for r in rows)}, failures {len(failed)}",flush=True)
    tested = [r for r in rows if "errors" in r]
    result = dict(counts=data["counts"],records=rows,failures=failed,threshold=args.threshold,
        qmc_power=args.power,seconds=time.monotonic()-started,
        improved=sorted(tested,key=lambda r:-r["gain_vs_conditional"])[:30],
        degraded=sorted(tested,key=lambda r:r["gain_vs_conditional"])[:30],
        degraded_both=sorted(tested,key=lambda r:max(r["gain_vs_conditional"],r["gain_vs_classic"]))[:30])
    write(folder/(args.output or "screen.json"),result)
    print(json.dumps({"referenced":len(tested),"failures":failed[:5],
        "top_improved":result["improved"][:3],"top_degraded":result["degraded"][:3]},indent=2),flush=True)

def refine(folder,args):
    data = read(folder/"rays.json")
    by_id = {r["id"]:r for r in data["records"]}
    result = []
    for identity in args.ids:
        r = by_id[identity]
        laws = models(r)
        ray = Ray(r)
        maximum=args.denominator
        coarse,middle=maximum//4,maximum//2
        refs,reps,stats = reference(ray,args.power,args.replicates,[coarse,middle,maximum],910000+identity*20)
        grid = grid_for(laws,refs[maximum])
        stats["checks"] = ray.checks()
        stats["grid_ks_coarse_vs_fine"] = ks(refs[coarse],refs[maximum],grid)
        stats["grid_ks_medium_vs_fine"] = ks(refs[middle],refs[maximum],grid)
        cdfs = np.array([p.cdf(grid) for p in reps[maximum]])
        stats["max_pointwise_cdf_se"] = float(np.max(np.std(cdfs,axis=0,ddof=1)/math.sqrt(args.replicates)))
        stats["summary"] = refs[maximum].summarize(grid)
        means = np.array([p.summarize(grid)["mean"] for p in reps[maximum]])
        stats["mean_ci95_halfwidth"] = float(student_t.ppf(.975,args.replicates-1)*np.std(means,ddof=1)/math.sqrt(args.replicates))
        errors = {name:dict(ks=ks(law,refs[maximum],grid),
            mean_error=law.mean-stats["summary"]["mean"]) for name,law in laws.items()}
        gain_samples = {name:np.array([ks(laws[name],rep,grid)-ks(laws["bridge"],rep,grid) for rep in reps[maximum]])
                        for name in ["classic","conditional"]}
        gains = {name:dict(value=errors[name]["ks"]-errors["bridge"]["ks"],
            ci95_halfwidth=float(student_t.ppf(.975,args.replicates-1)*np.std(v,ddof=1)/math.sqrt(args.replicates)))
                 for name,v in gain_samples.items()}
        row = dict(id=identity,record=r,reference=stats,errors=errors,gains=gains,
                   models={k:v.summary() for k,v in laws.items()})
        plot_grid = np.unique(np.concatenate([np.linspace(grid[0],0,4097)]+[law.grid for law in laws.values()]))
        curves = dict(v=plot_grid,reference_pdf=refs[maximum].pdf(plot_grid),reference_cdf=refs[maximum].cdf(plot_grid))
        for name,law in laws.items():
            curves[name+"_pdf"]=law.pdf(plot_grid)
            curves[name+"_cdf"]=law.cdf(plot_grid)
        mean,beta,residual = ray.bridge(maximum)
        curves.update(s=np.arange(1,maximum)*ray.t/maximum,bridge_mean=mean,
                      bridge_sd=np.sqrt(beta**2+np.sum(residual**2,axis=1)),beta=beta)
        np.savez_compressed(folder/f"ray-{identity}-curves.npz",**curves)
        write(folder/f"ray-{identity}.json",row)
        result.append(row)
        print(f"refined {identity}, errors { {k:v['ks'] for k,v in errors.items()} }, gains {gains}",flush=True)
    write(folder/(args.output or "refined.json"),result)

def main():
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["screen","refine"])
    parser.add_argument("folder",type=Path)
    parser.add_argument("--power",type=int,default=11)
    parser.add_argument("--replicates",type=int,default=8)
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--denominator",type=int,default=416)
    parser.add_argument("--threshold",type=float,default=.001)
    parser.add_argument("--ids",type=int,nargs="+",default=[])
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command=="screen": screen(args.folder,args)
    else: refine(args.folder,args)

if __name__=="__main__":
    main()
