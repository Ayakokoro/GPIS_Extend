"""Move only the F1 anchor on frozen shaderball data; do not change rendering.

Candidate means are exact exported SDF nodes. The SNR choices do not consult
the GP reference; the KS-minimizing position is an explicitly oracle diagnostic.
"""
from pathlib import Path
import hashlib
import math
import sys
import gradient_rays as g
import numpy as np
from scipy.special import ndtr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=g.ROOT
SOURCE=ROOT/'outputs/single-anchor-groove-search-20261108'
OUT=ROOT/'outputs/f1-anchor-position-ablation-20261109'


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    OUT.mkdir(parents=True,exist_ok=True)
    case=next(row for row in g.read(SOURCE/'report-data.json')['cases'] if row['key']=='F1')
    r=case['record'];ray=g.Ray(r);base=g.models(r);z=np.load(SOURCE/'F1-curves.npz')
    n=r['profile_denominator'];s=np.arange(1,n)*r['t']/n
    mean,beta,residual=ray.bridge(n)
    residual_sd=np.linalg.norm(residual,axis=1)
    marginal_sd=np.hypot(beta,residual_sd)
    pp=mean/residual_sd;qq=beta/residual_sd
    u_mode=(base['conditional'].mode-ray.mu)/ray.sd
    snr=mean/marginal_sd;mode_snr=pp+qq*u_mode
    j_snr=int(np.argmin(snr));j_mode=int(np.argmin(mode_snr))
    active=np.flatnonzero(s*r['gamma']>=r['t']*r['gamma']-1.5)
    coarse=np.unique(active[np.linspace(0,len(active)-1,161,dtype=int)])
    cache={}
    def evaluate(j):
        j=int(j)
        if j not in cache:
            law=g.Density(ray.mu,ray.sd,pp[j],qq[j],True)
            cache[j]=(float(np.max(abs(law.cdf(z['v'])-z['reference_cdf']))),law)
        return cache[j]
    for j in coarse:evaluate(j)
    top=sorted(coarse,key=lambda j:evaluate(j)[0])[:4]
    for j in top:
        for k in range(max(active[0],int(j)-16),min(active[-1],int(j)+16)+1):evaluate(k)
    j_best=min(cache,key=lambda j:cache[j][0])
    chosen={'marginal_snr':j_snr,'mode_snr':j_mode,'reference_selected':j_best}
    models={'conditional':base['conditional'],'current':base['bridge']}
    old=r['old_production'];models['old']=g.Density(old['mu'],old['sd'],old['p'],old['q'],True)
    results={}
    for name,j in chosen.items():
        error,law=evaluate(j);models[name]=law
        results[name]=dict(distance=float(s[j]),distance_in_ell=float(s[j]*r['gamma']),
            fraction_of_flight=float(s[j]/r['t']),distance_to_hit_in_ell=float((r['t']-s[j])*r['gamma']),
            mean_sigma=float(mean[j]/r['sigma']),marginal_sd_sigma=float(marginal_sd[j]/r['sigma']),
            marginal_snr=float(snr[j]),p=float(pp[j]),q=float(qq[j]),
            positive_probability_at_base_mode=float(ndtr(mode_snr[j])),ks=error,
            mean_gradient=law.mean,world_position=(np.asarray(r['birth_position'])+s[j]*np.asarray(r['wi'])).tolist())
    errors={name:float(np.max(abs(law.cdf(z['v'])-z['reference_cdf']))) for name,law in models.items()}
    result=dict(case='F1',source_ray_id=r['id'],pixel=r['pixel'],sample=r['sample'],hit=r['bounce']+1,
        flight_length_in_ell=r['t']*r['gamma'],current_anchor_in_ell=r['production']['anchor_distance']*r['gamma'],
        current_positive_probability_at_base_mode=float(ndtr(r['production']['p']+r['production']['q']*u_mode)),
        current_mean_gradient=base['bridge'].mean,reference_mean_gradient=case['reference']['summary']['mean'],
        errors=errors,candidates=results,sweep=[dict(distance_in_ell=float(s[j]*r['gamma']),ks=cache[j][0]) for j in sorted(cache)],
        source_curves_sha256=hashlib.sha256((SOURCE/'F1-curves.npz').read_bytes()).hexdigest(),
        scope='One frozen ray; only anchor position changes. SNR candidates use actual SDF and exact conditional SE covariance. '
              'reference_selected minimizes stored GP-reference KS on a finite coarse/refined node search, not a deployable selector or a global optimum proof.',
        reference=case['reference'],production_unchanged=True)
    curves={k:z[k] for k in ['v','reference_pdf','reference_cdf']}
    for name,law in models.items():
        for suffix in ['pdf','cdf']:curves[name+'_'+suffix]=getattr(law,suffix)(z['v'])
    np.savez_compressed(OUT/'curves.npz',**curves)
    g.write(OUT/'results.json',result)
    colors=dict(reference='#172131',conditional='#d78816',old='#9a5de2',current='#a7574a',marginal_snr='#007b77')
    labels=dict(reference='GP reference',conditional='global_conditional',old='old 2-anchor',current='current anchor: 3.654 ell',marginal_snr='moved anchor: 4.423 ell')
    styles=dict(reference='-',conditional='--',old='-.',current=':',marginal_snr='-')
    fig,axes=plt.subplots(2,2,figsize=(13,8.7),layout='constrained')
    ax=axes[0,0];xx=s*r['gamma']
    ax.plot(xx,snr,color='#337c99',label='conditional mean / std. dev.')
    ax.axhline(0,color='#777',lw=.8)
    for name,pos in [('current',result['current_anchor_in_ell']),('marginal_snr',results['marginal_snr']['distance_in_ell'])]:
        ax.axvline(pos,color=colors[name],ls='--',label=labels[name])
    ax.set(xlim=(3.3,r['t']*r['gamma']),ylim=(-1.3,7),xlabel='distance / ell',ylabel='mean / std. dev.',title='F1: actual conditional profile')
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axes[0,1];sweep=result['sweep']
    ax.plot([r['distance_in_ell'] for r in sweep],[r['ks'] for r in sweep],color='#337c99')
    ax.scatter([result['current_anchor_in_ell']],[errors['current']],color=colors['current'],label='current anchor')
    ax.scatter([results['marginal_snr']['distance_in_ell']],[errors['marginal_snr']],color=colors['marginal_snr'],label='minimum mean / std. dev.')
    ax.set(xlabel='anchor distance / ell',ylabel='KS against GP reference',title='Only anchor position changes')
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    scale=r['sigma']*r['gamma']
    for col,suffix in enumerate(['pdf','cdf']):
        ax=axes[1,col]
        for name in colors:
            ax.plot(z['v']/scale,curves[name+'_'+suffix]*(scale if suffix=='pdf' else 1),color=colors[name],
                    ls=styles[name],lw=2.3 if name=='reference' else 1.7,label=labels[name])
        ax.set(xlim=(-13,-6),xlabel='axial gradient / (sigma gamma)',ylabel=suffix.upper(),title='Collision gradient '+suffix.upper())
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(OUT/'comparison.png',dpi=160);fig.savefig(OUT/'comparison.svg');plt.close(fig)
    table=''.join(f'<tr><td>{name}</td><td>{errors[name]:.8f}</td></tr>' for name in models)
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>F1 锚点位置诊断</title>
<style>body{max-width:1350px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#172131}img{width:100%}td,th{padding:8px 18px;border-bottom:1px solid #ddd}a{color:#007b77}</style>
<h1>F1：把唯一锚点移到实际条件剖面的低谷</h1>
<p>像素 (138,130)，sample 2，第 1 次碰撞。固定出生状态、方向、碰撞距离和 GP reference，仅改变用于梯度修正的一个锚点位置。生产渲染器与旧报告均未修改。</p>
<p>当前锚点位于 3.654ℓ，在未修正梯度分布峰值处的正值概率约为 0.99999999855，因此几乎不改变梯度权重。
实际条件均值/标准差的最低点在 4.423ℓ；移到这里后，KS 从约 0.53900 降到 0.00386。</p>
<p>marginal_snr：最小化 E[F(s)|出生状态,F(t)=0] / std[F(s)|出生状态,F(t)=0]。
mode_snr：额外固定终点梯度为未修正 Rice 分布众数，再最小化条件均值/标准差。
reference_selected：使用 GP reference 挑选 KS 最小的扫描位置，仅用于事后诊断；不等于可部署的选点规则，也不证明连续区间全局最优。</p>
<p>所有候选的真实 SDF 均来自生产导出的精确网格点，没有插值。图中的扫描使用最后 1.5ℓ 的 161 个初始位置，再在最好的四个邻域逐点加密。
仍然只对当前光线做了验证，不能直接推广为所有光线的统一偏移量。</p>
<img src="comparison.png" alt="F1 anchor comparison"><table><tr><th>分布</th><th>KS</th></tr>'''+table+'''</table>
<p><a href="results.json">数值与位置扫描</a> · <a href="curves.npz">分布曲线</a> · <a href="comparison.svg">SVG</a> · <a href="../single-anchor-groove-search-20261108/comparison.html#F1">原报告 F1</a></p></html>'''
    (OUT/'comparison.html').write_text(document,encoding='utf-8')
    print(g.read(OUT/'results.json')['candidates'],flush=True)
    print('errors',errors,flush=True)


if __name__=='__main__':main()
