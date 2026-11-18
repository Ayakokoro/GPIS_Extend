"""Monte Carlo survival curves on the three exported slab rays.

python -B tools/surface_compare/slabs_transmittance_mc.py --sample-gp
python -B tools/surface_compare/slabs_transmittance_mc.py --report
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'build/fp-gradient-validation-deps'))
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
os.environ.setdefault('MKL_NUM_THREADS','4')
os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'build/ray_curve_matplotlib'))
import numpy as np

RAYS=ROOT/'outputs/gpis_three_slabs_compare_20261116/sdf_ray_profiles_20261117'
OUT=RAYS/'transmittance_mc'
IDS=['three-overlap','two-overlap','two-overlap-dark']
LABELS=['三重叠','正常二重叠','暗弧二重叠']
COLORS=['#0879b8','#17835d','#c83e4d']
GP_SEED=2026111702


def load(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def wilson(count,n):
    p=np.asarray(count,dtype=float)/n
    z=1.959963984540054
    center=(p+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return np.maximum(0,center-half),np.minimum(1,center+half)


def covariance_factor(t,sigma,gamma):
    # Pivoted Cholesky of the finite SE correlation matrix. Generate columns
    # analytically instead of allocating the dense ~6,700 by ~6,700 matrix.
    basis=np.zeros((len(t),64));diagonal=np.ones(len(t));pivots=[]
    for k in range(basis.shape[1]):
        pivot=int(np.argmax(diagonal));remaining=float(diagonal[pivot])
        if remaining<=5e-14:break
        col=np.exp(-.5*(gamma*(t-t[pivot]))**2)
        if k: col-=basis[:,:k]@basis[pivot,:k]
        col/=np.sqrt(remaining)
        basis[:,k]=col
        diagonal-=col*col
        assert diagonal.min()>-5e-14
        diagonal=np.maximum(0,diagonal)
        pivots.append(pivot)
    else:
        raise RuntimeError('Covariance factor did not converge')
    basis=basis[:,:len(pivots)]
    # Check EVERY covariance entry in bounded-memory blocks.
    max_error=0.
    for begin in range(0,len(t),128):
        end=min(begin+128,len(t))
        residual=basis[begin:end]@basis.T-np.exp(-.5*(gamma*(t[begin:end,None]-t[None,:]))**2)
        max_error=max(max_error,float(np.max(np.abs(residual))))
    assert max_error<5e-12
    return sigma*basis,dict(rank=len(pivots),pivots=pivots,
                           max_residual_diagonal_over_variance=float(diagonal.max()),
                           all_entries_max_covariance_error_over_variance=max_error,
                           all_entries_max_covariance_error=max_error*sigma*sigma)


def sample_gp(name,index,n):
    started=time.monotonic()
    record=load(OUT/'inputs'/f'{name}-ray-input.json')
    original=np.asarray(record['sdf_curve']['samples'])
    knots=record['geometry']['piecewise_linear_knots']
    essential=np.array([p['s'] for p in knots]+[v for h in record['geometry']['slab_intersections'] for v in [h['begin'],h['end']]])
    t=np.unique(np.r_[original[:,0],essential])
    mean=np.interp(t,[p['s'] for p in knots],[p['mean'] for p in knots])
    # This interpolation is exact piecewise-linear evaluation; check it against
    # the independent scene formula and all production-exported samples.
    origin=np.array(record['ray']['origin']);direction=np.array(record['ray']['direction'])
    xyz=origin+t[:,None]*direction
    exact=np.min(np.max(np.abs(xyz[:,None,:]-np.array(record['geometry']['centers']))-np.array(record['geometry']['half_size']),axis=2),axis=1)
    assert np.max(np.abs(exact-mean))<1e-12
    assert np.max(np.abs(mean[np.searchsorted(t,original[:,0])]-original[:,2]))<1e-12
    kernel=record['kernel']['effective_along_ray'];sigma=kernel['sigma'];gamma=kernel['gamma']
    factor,audit=covariance_factor(t,sigma,gamma)
    print(f'{name}: covariance rank {factor.shape[1]}, {len(t)} points, error {audit["all_entries_max_covariance_error_over_variance"]:.3g}',flush=True)
    grid_indices={stride:np.searchsorted(t,np.unique(np.r_[original[::stride,0],essential])) for stride in [1,2,4]}
    hist={stride:np.zeros(len(t)+1,dtype=np.int64) for stride in grid_indices}
    marginal=np.zeros(len(t),dtype=np.int64)
    first_indices=np.empty(n,dtype=np.int32)
    rng=np.random.default_rng(np.random.SeedSequence([GP_SEED,index]))
    accepted=0;rejected=0;next_progress=32768
    probe_indices=np.linspace(0,len(t)-1,12,dtype=int)
    sums=np.zeros(12);cross=np.zeros((12,12))
    while accepted<n:
        batch=min(512,n-accepted)
        noise=rng.standard_normal((batch,factor.shape[1]))@factor.T
        values=noise+mean
        positive_start=values[:,0]>0
        rejected+=int(np.count_nonzero(~positive_start))
        noise=noise[positive_start];values=values[positive_start]
        batch=len(values)
        if not batch:continue
        probes=noise[:,probe_indices];sums+=probes.sum(axis=0);cross+=probes.T@probes
        negative=values<=0
        marginal+=np.count_nonzero(~negative,axis=0)
        for stride,indices in grid_indices.items():
            tested=negative[:,indices]
            any_hit=np.any(tested,axis=1)
            local=np.argmax(tested,axis=1)
            first=np.where(any_hit,indices[local],len(t))
            hist[stride]+=np.bincount(first,minlength=len(t)+1)
            if stride==1:first_indices[accepted:accepted+batch]=first
        accepted+=batch
        if accepted>=next_progress:
            print(f'{name}: {accepted}/{n} GP draws',flush=True);next_progress+=32768
    survival={stride:n-np.cumsum(h[:-1]) for stride,h in hist.items()}
    assert survival[1][0]==n and np.all(np.diff(survival[1])<=0)
    assert np.all(survival[1]<=marginal)
    # Sanity-check the sampled noise against its actual mean and covariance.
    empirical_mean=sums/n
    empirical_covariance=cross/n-np.outer(empirical_mean,empirical_mean)
    expected=sigma*sigma*np.exp(-.5*(gamma*(t[probe_indices,None]-t[None,probe_indices]))**2)
    audit.update(empirical_max_abs_noise_mean_over_sigma=float(np.max(np.abs(empirical_mean))/sigma),
                 empirical_max_covariance_error_over_variance=float(np.max(np.abs(empirical_covariance-expected))/(sigma*sigma)))
    assert audit['empirical_max_abs_noise_mean_over_sigma']<.04
    assert audit['empirical_max_covariance_error_over_variance']<.04
    nested=[]
    for stride in [4,2,1]:
        idx=grid_indices[stride]
        delta=survival[stride][idx]-survival[1][idx]
        assert delta.min()>=0
        nested.append(dict(stride=stride,base_step=record['sdf_curve']['step']*stride,
                           point_count=len(idx),max_survivor_difference_at_common_points=int(delta.max()),
                           max_transmittance_difference_at_common_points=float(delta.max()/n),
                           extra_survivors_at_exit=int(survival[stride][-1]-survival[1][-1])))
    distances=np.full(n,-1.);hit=first_indices<len(t);distances[hit]=t[first_indices[hit]]
    np.save(OUT/f'{name}-gp-first-hits.npy',distances)
    np.savez_compressed(OUT/f'{name}-gp-counts.npz',t=t,mean=mean,survivors=survival[1],marginal_positive=marginal,
                        coarse_survivors=survival[4],middle_survivors=survival[2])
    lo,hi=wilson(survival[1][-1],n)
    result=dict(id=name,ray_input=str((RAYS/f'{name}-ray-input.json').relative_to(ROOT)),input_sha256=sha(OUT/'inputs'/f'{name}-ray-input.json'),
                ray=record['ray'],kernel=record['kernel'],samples=n,rejected_nonpositive_starts=rejected,
                seed_entropy=[GP_SEED,index],grid_points=len(t),covariance=audit,nested_grids=nested,
                endpoint=dict(survivors=int(survival[1][-1]),transmittance=float(survival[1][-1]/n),wilson_95=[float(lo),float(hi)]),
                seconds=time.monotonic()-started,
                definition='T(t)=P(F(s)>0 for all 0<=s<=t | F(0)>0); sampled first passage on nested finite grids.',
                estimator='IID Gaussian coefficient draws from a checked pivoted-Cholesky factor of the actual SE covariance; cumulative first-hit counts. No hazard integration.',
                limitations=['Wilson intervals are pointwise Monte Carlo intervals, not simultaneous bands.',
                             'Finite-grid first passage can miss between-node negative excursions; nested-grid differences are measured, not a continuous-path proof.',
                             'Covariance-factor residual is explicitly checked against every kernel entry; it is a small numerical approximation.'])
    save(OUT/f'{name}-gp-summary.json',result)
    print(f'{name}: T(exit)={result["endpoint"]["transmittance"]:.6f}; {result["seconds"]:.2f}s',flush=True)


def report():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    font=Path('C:/Windows/Fonts/msyh.ttc')
    font_manager.fontManager.addfont(str(font))
    plt.rcParams.update({'font.family':[font_manager.FontProperties(fname=str(font)).get_name(),'DejaVu Sans'],
                         'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
    renderer=load(OUT/'renderer_samples.json')
    gp_count=load(OUT/f'{IDS[0]}-gp-summary.json')['samples']
    renderer_count=renderer['spec']['samples']
    by_key={(r['id'],r['method']):r for r in renderer['records']}
    summaries=[];plot_records=[]
    methods=['gp','numeric','gauss_legendre5']
    for name,label in zip(IDS,LABELS):
        source=load(OUT/'inputs'/f'{name}-ray-input.json')
        gp=load(OUT/f'{name}-gp-summary.json')
        assert gp['samples']==gp_count
        assert gp['input_sha256']==sha(OUT/'inputs'/f'{name}-ray-input.json')
        data=np.load(OUT/f'{name}-gp-counts.npz')
        t,mean=data['t'],data['mean']
        counts={'gp':data['survivors']};sizes={'gp':gp['samples']};provenance={'gp':gp}
        for method in methods[1:]:
            r=by_key[name,method];assert r['ray']==source['ray']
            d=np.array(r['first_hit_distances']);hit=np.sort(d[d>=0]);n=len(d)
            assert n==r['samples'] and np.all(hit<=source['ray']['t'])
            counts[method]=n-np.searchsorted(hit,t,side='right');sizes[method]=n
            provenance[method]={k:v for k,v in r.items() if k!='first_hit_distances'}
        estimates={};intervals={}
        for method in methods:
            c=counts[method];n=sizes[method]
            assert c[0]==n and np.all(np.diff(c)<=0) and np.all(c>=0)
            estimates[method]=c/n
            lo,hi=wilson(c,n);lo[0]=hi[0]=1
            intervals[method]=(lo,hi)
        columns=['distance_t','distance_over_ell','sdf_mean']
        values=[t,t/source['kernel']['configured']['length_scale_ell'],mean]
        for method in methods:
            columns.extend([f'{method}_transmittance',f'{method}_wilson95_lower',f'{method}_wilson95_upper',f'{method}_survivors'])
            values.extend([estimates[method],*intervals[method],counts[method]])
        columns.append('gp_marginal_positive_probability')
        values.append(data['marginal_positive']/sizes['gp'])
        matrix=np.column_stack(values)
        np.savetxt(OUT/f'{name}-transmittance.csv',matrix,delimiter=',',fmt='%.17g',header=','.join(columns),comments='')
        output=dict(schema_version=1,case=source['provenance']['case'],ray=source['ray'],kernel=source['kernel'],
                    start_condition='F(0)>0; no observed f0 or birth gradient',
                    transmittance_curve=dict(definition='T(t)=P(no first zero crossing up to t | F(0)>0)',
                                             columns=columns,samples=matrix.tolist(),sample_count=len(t),
                                             note='All T columns are sample survival counts. Marginal positivity is not transmittance.'),
                    monte_carlo=provenance)
        save(OUT/f'{name}-transmittance.json',output)
        endpoint={m:dict(transmittance=float(estimates[m][-1]),wilson_95=[float(a[-1]) for a in intervals[m]],
                         survivors=int(counts[m][-1]),samples=sizes[m]) for m in methods}
        queries=[]
        for q in [.25,.28,.30,.32,.34,.36,.38,.40,.42]:
            j=int(np.searchsorted(t,q))
            queries.append(dict(requested_t=q,grid_t=float(t[j]),**{m:float(estimates[m][j]) for m in methods}))
        summaries.append(dict(id=name,label=label,pixel=source['provenance']['case']['pixel'],endpoint=endpoint,
                              nested_grids=gp['nested_grids'],covariance=gp['covariance'],queries=queries,
                              checks=dict(monotone=True,in_unit_interval=True,starts_at_one=True,same_ray_as_sdf=True)))
        plot_records.append((source,t,mean,estimates,intervals))
    def write_figure(fig,name):
        for ext in ['png','svg']:
            fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight',facecolor='white')
        plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.9),sharey=True)
    for i,((source,t,mean,est,ci),label,color) in enumerate(zip(plot_records,LABELS,COLORS)):
        for ax in axes:
            ax.plot(t,est['gp'],color=color,lw=2,label=f'R{i+1} {label}')
            ax.fill_between(t,*ci['gp'],color=color,alpha=.18)
    for j,ax in enumerate(axes):
        ax.set(xlim=(.21,.44) if j else (0,.66),ylim=(-.025,1.025),xlabel='距介质入口的距离 t',ylabel='透射率 T(t)',
               title='发生吸收的区间' if j else '完整介质段')
        ax.grid(alpha=.2);ax.legend(fontsize=9,loc='upper right')
    fig.suptitle('直接 GP 轨迹采样的透射率 · 每条光线 262,144 个样本',fontsize=15)
    fig.text(.5,.015,'阴影为逐点 95% Wilson 区间。T(t) 统计整段存活，首次碰撞后不再恢复。',ha='center')
    fig.tight_layout(rect=(0,.07,1,.94))
    write_figure(fig,'gp_transmittance')
    fig,axes=plt.subplots(2,3,figsize=(15,8.5),sharex='col',sharey='row')
    for i,(source,t,mean,est,ci) in enumerate(plot_records):
        top,bottom=axes[:,i]
        top.plot(t,mean,color=COLORS[i],lw=2)
        top.axhline(0,color='#444',lw=.8)
        top.set(title=f'R{i+1} {LABELS[i]} · {source["provenance"]["case"]["pixel"]}',ylabel='SDF 均值 m(t)',ylim=(-.018,.09))
        for h in source['geometry']['slab_intersections']:
            for ax in [top,bottom]:ax.axvspan(h['begin'],h['end'],color=COLORS[i],alpha=.09)
        for ax in [top,bottom]:
            for b in source['selection_groups']['boundaries']:ax.axvline(b,color='#7d7298',ls=':',lw=.8,alpha=.7)
            ax.set_xlim(.21,.44);ax.grid(alpha=.15)
        for method,color,ls,lw,label in [('gp','#176fa9','-',2.4,'直接 GP 采样'),
                                        ('numeric','#c75d22','-',1.6,'Single history / numeric'),
                                        ('gauss_legendre5','#a5258e','--',1.5,'Single history / CDF5')]:
            bottom.plot(t,est[method],color=color,ls=ls,lw=lw,label=label)
            bottom.fill_between(t,*ci[method],color=color,alpha=.1)
        bottom.set(xlabel='距介质入口的距离 t',ylabel='透射率 T(t)',ylim=(-.025,1.025))
        bottom.legend(fontsize=9,loc='upper right')
    fig.suptitle('同一固定光线：实际 GP 的首次穿越采样 vs 原 single_history 距离采样',fontsize=15)
    fig.text(.5,.015,'GP：262,144 条轨迹 / 光线；single_history：每种模式 32,768 次 / 光线。\n阴影：均值 slab 内部；竖虚线：原选择组边界；置信带仅表示采样误差。',ha='center')
    fig.tight_layout(rect=(0,.08,1,.94),h_pad=1.6)
    write_figure(fig,'transmittance_comparison')
    rows='';links=''
    for i,r in enumerate(summaries):
        rows+=f'<tr><td>R{i+1} {r["label"]}</td><td>{r["pixel"]}</td>'
        for m in methods:
            v=r['endpoint'][m];lo,hi=v['wilson_95']
            rows+=f'<td><strong>{v["transmittance"]:.6f}</strong><br><small>[{lo:.6f}, {hi:.6f}]</small></td>'
        rows+='</tr>'
        name=r['id']
        links+=f'<li>R{i+1} {r["label"]}：<a href="{name}-transmittance.json">完整 JSON</a> · <a href="{name}-transmittance.csv">CSV</a> · <a href="{name}-gp-summary.json">GP 采样与网格检查</a> · <a href="{name}-gp-first-hits.npy">GP 首次碰撞距离 NPY</a></li>'
    max_grid=max(q['max_transmittance_difference_at_common_points'] for r in summaries for q in r['nested_grids'])
    max_cov=max(r['covariance']['all_entries_max_covariance_error_over_variance'] for r in summaries)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>三条光线的采样透射率</title>
<style>body{max-width:1360px;margin:30px auto;padding:0 24px;font:16px/1.7 system-ui;color:#24313c;background:#fbfcfe}h1{font-size:28px}img{width:100%;height:auto;background:white}table{border-collapse:collapse;width:100%;background:white}th,td{padding:12px;text-align:left;border-bottom:1px solid #ddd}a{color:#096baf}.note{padding:16px 22px;background:#edf3f8;border-radius:9px}small{color:#56616b}code{background:#edf1f5;padding:2px 4px}</style>
<h1>三重叠、正常二重叠、暗弧二重叠：采样得到的 T(t)</h1>
<p>沿用上一份 SDF 导出的三条固定光线，原点、方向、长度逐项一致；ℓ=0.2、σ=0.02。这里的 t 从介质入口起算。没有重新抖动像素，也没有把图像像素均值当作单条光线。</p>
<div class="note"><strong>T(t) = P(F(s)&gt;0 对所有 0≤s≤t 成立 | F(0)&gt;0)</strong>。直接抽样完整 SE-GP 轨迹，统计至 t 仍未发生首次零交叉的比例。每条光线 262,144 个独立样本；一旦碰撞，该轨迹在所有后续 t 的贡献均为零。不是只检查 F(t)&gt;0，也不是对近似消光系数积分。</div>
<h2>直接 GP 采样</h2><img src="gp_transmittance.png" alt="三条光线的直接 GP 采样透射率"><p><a href="gp_transmittance.svg">矢量 SVG</a></p>
<h2>介质出口的透射率</h2><p>方括号为逐点 95% Wilson 置信区间。Single history 每条光线、每种模式另做 32,768 次原距离采样器调用，参数保持原场景设置。</p>
<table><tr><th>案例</th><th>像素</th><th>直接 GP</th><th>Single history / numeric</th><th>Single history / CDF5</th></tr>'''+rows+'''</table>
<h2>与原 single_history 采样对照</h2><img src="transmittance_comparison.png" alt="SDF 与三种采样透射率逐光线对照"><p><a href="transmittance_comparison.svg">矢量 SVG</a> · <a href="../comparison.html">原 SDF 曲线与像素定位</a></p>
<p>两种 single_history 曲线使用相同随机种子，差异很小。蓝线来自完整 GP 联合分布的首次穿越采样；其与 single_history 的差异超出这些样本的置信区间时，不能用有限采样噪声解释。暗弧位置的 single_history 透射率偏低，而直接 GP 采样不会出现同样幅度的下降。</p>
<h2>数据与复现</h2><ul>'''+links+'''</ul><p><a href="summary.json">汇总</a> · <a href="manifest.json">来源、哈希与命令</a> · <a href="renderer_samples.json">原 single_history 首次碰撞样本</a> · <a href="renderer_spec.json">采样配置</a></p>
<h2>数值检查</h2><p>直接使用导出的有效 SE 核参数。在原 6,657 个距离样本上加入全部 SDF 分段拐点和 slab 交点；每条轨迹同时检查原网格及 2 倍、4 倍间距的嵌套网格。所有协方差矩阵元素均与解析核逐项核对。</p>
<p>嵌套网格共同节点上的最大 T 差异：<code>'''+f'{max_grid:.6g}'+'''</code>；协方差因子最大逐项误差 / σ²：<code>'''+f'{max_cov:.3g}'+'''</code>。每条曲线均满足 T(0)=1、0≤T≤1、随 t 单调不增。</p>
<p>这些是有限网格 Monte Carlo 估计。网格一致性不是连续路径零交叉的数学证明；置信区间仅涵盖采样误差，不包含时间离散化或协方差数值近似误差。所有 NPY 首次碰撞距离中 -1 表示到介质出口仍未碰撞。</p></html>'''
    page=page.replace('262,144',f'{gp_count:,}').replace('32,768',f'{renderer_count:,}')
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    summary=dict(complete=True,definition='GP first-passage survival conditional on positive entrance',rays=summaries,
                 gp_samples_per_ray=gp_count,renderer_samples_per_method_per_ray=renderer_count,
                 largest_nested_grid_difference=max_grid,largest_covariance_error_over_variance=max_cov)
    save(OUT/'summary.json',summary)
    sources=[ROOT/'tools/surface_compare/slabs_transmittance_mc.py',ROOT/'tools/surface_compare/slabs_transmittance_samples.cpp',
             ROOT/'build/Release/slabs_transmittance_samples.exe',OUT/'renderer_spec.json',OUT/'renderer_samples.json']
    sources += [OUT/'inputs'/f'{name}-ray-input.json' for name in IDS]
    sources += [OUT/f'{name}-{suffix}' for name in IDS for suffix in ['gp-summary.json','gp-counts.npz','gp-first-hits.npy']]
    save(OUT/'manifest.json',dict(source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},
         gp_seed=GP_SEED,renderer_seed=renderer['spec']['seed'],
         notes=['Input JSON snapshots preserve the fixed rays even if the parent files are subsequently edited.',
                'GP uses the effective kernel reported by the production exporter, not rounded decimal constants.',
                'Single-history diagnostics disable profiling counters only; the transport law is unchanged.'],
         commands=['cmake --build build --config Release --target slabs_transmittance_samples',
                   'build/Release/slabs_transmittance_samples.exe '+str((OUT/'renderer_spec.json').relative_to(ROOT))+' '+str((OUT/'renderer_samples.json').relative_to(ROOT)),
                   'python -B tools/surface_compare/slabs_transmittance_mc.py --sample-gp --report']))
    local_links=re.findall(r'(?:href|src)="([^"]+)"',page)
    assert all((OUT/p).exists() for p in local_links)
    print(json.dumps(summary,ensure_ascii=True),flush=True)
    print(OUT/'comparison.html',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--sample-gp',action='store_true')
    parser.add_argument('--report',action='store_true')
    parser.add_argument('--samples',type=int,default=262144)
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.sample_gp:
        (OUT/'inputs').mkdir(exist_ok=True)
        for name in IDS:
            snapshot=OUT/'inputs'/f'{name}-ray-input.json'
            if not snapshot.exists():
                save(snapshot,load(RAYS/f'{name}-ray-input.json'))
        for i,name in enumerate(IDS):sample_gp(name,i,args.samples)
    if args.report:report()


if __name__=='__main__':
    main()
