"""Validate fixed5_snr on the seven frozen groove rays, including F1.

prepare writes detail requests for both selectors. report checks the C++
exports with independent SE-feature QR and reuses the frozen GP references.
No new ray selection, dense anchor search or reference-based placement occurs.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import html
import sys
import gradient_rays as g
import single_anchor_update as single
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradient_rays_report import inline_svg

SOURCE = g.ROOT/'outputs/single-anchor-groove-search-20261108'
DEFAULT = g.ROOT/'outputs/fixed5-anchor-validation-20261109/rays'
COLORS = dict(reference='#172131',classic='#8a929d',conditional='#d28713',hermite='#9c517b',fixed5='#007f7c')
LABELS = dict(reference='GP reference',classic='classic_global',conditional='global_conditional',
              hermite='single anchor: Hermite',fixed5='single anchor: fixed 5 SNR')
STYLES = dict(reference='-',classic=':',conditional='--',hermite='-.',fixed5='-')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=True)
    cases = g.read(SOURCE/'report-data.json')['cases']
    scene = g.read(SOURCE/'refined/rays.json')['scene']
    records = []
    for case in cases:
        r = copy.deepcopy(case['record'])
        r['comparison_key'] = case['key']
        z = np.load(SOURCE/(case['key']+'-curves.npz'))
        r['query_derivatives'] = z['v'][np.linspace(0,len(z['v'])-1,129,dtype=int)].tolist()
        records.append(r)
    for selector in ('fixed5_snr','hermite'):
        g.write(folder/(selector+'-input.json'),dict(scene=scene,anchor_selection=selector,records=records))
    files = [SOURCE/'report-data.json',g.ROOT/scene]+[SOURCE/(c['key']+'-curves.npz') for c in cases]
    g.write(folder/'source-provenance.json',dict(files={str(p.relative_to(g.ROOT)):digest(p) for p in files},
        description='Same birth jets, direction, collision distance and SDF profiles; frozen GP references.'))
    print('Prepared',len(records),'unchanged collision states.',flush=True)


def plot(folder,case,r,curves,laws):
    key=case['key'];gamma=r['gamma'];scale=r['sigma']*gamma
    candidates=r['production']['anchor_candidates'];chosen=r['production']['selected_anchor_index']
    old=case['record']['production']['anchor_distance']*gamma
    fig,axes=plt.subplots(2,2,figsize=(13,8.7),layout='constrained')
    ax=axes[0,0]
    x=np.linspace(0,r['t'],len(r['mean_profile']))*gamma
    ax.plot(x,np.asarray(r['mean_profile'])/r['sigma'],c='#858b96',label='actual SDF / sigma (stored profile)')
    xs=np.asarray([a['distance'] for a in candidates])*gamma
    ys=np.asarray([a['mean'] for a in candidates])/r['sigma']
    ax.scatter(xs,ys,c=COLORS['fixed5'],s=32,label='5 fixed candidates',zorder=3)
    ax.scatter([xs[chosen]],[ys[chosen]],c=COLORS['fixed5'],marker='*',s=150,zorder=4,label='selected anchor')
    ax.axvline(old,c=COLORS['hermite'],ls='-.',label='Hermite anchor')
    ax.axhline(0,c='#555',lw=.8)
    left=max(0,r['t']*gamma-1.8)
    visible=np.asarray(r['mean_profile'])[x>=left]/r['sigma']
    ymin,ymax=min(0,float(np.min(visible))),max(0,float(np.max(visible)))
    padding=max(.1,.12*(ymax-ymin))
    ax.set(xlim=(left,r['t']*gamma),ylim=(ymin-padding,ymax+padding),xlabel='distance / ell',ylabel='SDF / sigma',
           title=f"{key}: pixel {r['pixel']}, sample {r['sample']}, hit #{r['bounce']+1}")
    ax.legend(fontsize=8);ax.grid(alpha=.15)
    ax=axes[0,1];scores=[a['score'] for a in candidates]
    ax.scatter(xs,scores,color=COLORS['fixed5'],s=36)
    ax.scatter([xs[chosen]],[scores[chosen]],color=COLORS['fixed5'],marker='*',s=170)
    for i,(x,y) in enumerate(zip(xs,scores)):ax.annotate(str(i+1),(x,y),xytext=(5,6),textcoords='offset points')
    ax.set(xlabel='candidate distance / ell',ylabel='p(s) + q(s) u0',title='Exactly five scores; lowest wins')
    ax.grid(alpha=.15)
    summaries=[law.summary() for law in laws.values()]+[case['reference']['summary']]
    lo=min(s['quantiles'][0]-2.5*s['sd'] for s in summaries)/scale
    hi=min(0,max(s['quantiles'][2]+2.5*s['sd'] for s in summaries)/scale)
    for col,suffix in enumerate(('pdf','cdf')):
        ax=axes[1,col]
        for j,name in enumerate(COLORS):
            ax.plot(curves['v']/scale,curves[name+'_'+suffix]*(scale if suffix=='pdf' else 1),
                c=COLORS[name],ls=STYLES[name],lw=2.4 if name=='reference' else 1.7,label=LABELS[name],
                marker=[None,'x','o','s',None][j],ms=4,mfc='white',markevery=(.03+.05*j,.25),
                gid='curve-'+suffix+'-'+name)
        ax.set(xlim=(lo,hi),xlabel='axial gradient / (sigma gamma)',ylabel=suffix.upper(),title='Collision gradient '+suffix.upper())
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(folder/(key+'.png'),dpi=160)
    fig.savefig(folder/(key+'.svg'));plt.close(fig)


def report(folder):
    provenance=g.read(folder/'source-provenance.json')
    assert all(digest(g.ROOT/path)==sha for path,sha in provenance['files'].items())
    fixed={r['comparison_key']:r for r in g.read(folder/'fixed5_snr.json')['records']}
    legacy={r['comparison_key']:r for r in g.read(folder/'hermite.json')['records']}
    cases=g.read(SOURCE/'report-data.json')['cases']
    rows=[];checks=[];cards=[];table=[]
    for case in cases:
        key=case['key'];r=fixed[key];before=case['record'];h=legacy[key]
        for field in ('birth_position','birth_gradient','f0','g0','wi','t','sigma','gamma','mean_profile'):
            assert r[field]==before[field] and h[field]==before[field]
        law,check=single.check(r);_,oldcheck=single.check(h)
        check['hermite_check']=oldcheck
        check['hermite_reproduction_error']=max(abs(h['production'][f]-before['production'][f])/(1+abs(before['production'][f]))
                                               for f in ('mu','sd','p','q','anchor_distance'))
        check['pass']=bool(check['pass'] and oldcheck['pass'] and check['hermite_reproduction_error']<2e-8)
        checks.append(check)
        base=g.models(r);p=h['production']
        laws=dict(classic=base['classic'],conditional=base['conditional'],fixed5=law,
                  hermite=g.Density(p['mu'],p['sd'],p['p'],p['q'],True))
        z=np.load(SOURCE/(key+'-curves.npz'))
        curves={name:z[name] for name in ('v','reference_pdf','reference_cdf')}
        for name,model in laws.items():
            for suffix in ('pdf','cdf'):curves[name+'_'+suffix]=getattr(model,suffix)(curves['v'])
        errors={name:float(np.max(abs(curves[name+'_cdf']-curves['reference_cdf']))) for name in laws}
        check['comparison_grid_tail_mass']=max(float(model.cdf(curves['v'][0])) for model in laws.values())
        check['pass']=bool(check['pass'] and check['comparison_grid_tail_mass']<1e-6)
        row=dict(key=key,id=r['id'],pixel=r['pixel'],sample=r['sample'],hit=r['bounce']+1,
            anchor_in_ell=r['production']['anchor_distance']*r['gamma'],
            hermite_anchor_in_ell=h['production']['anchor_distance']*r['gamma'],
            errors=errors,gain_vs_hermite=errors['hermite']-errors['fixed5'],
            gain_vs_conditional=errors['conditional']-errors['fixed5'],
            production=r['production'],reference=case['reference'],check=check)
        rows.append(row);np.savez_compressed(folder/(key+'-curves.npz'),**curves)
        plot(folder,case,r,curves,laws)
        table.append(f'<tr><td><a href="#{key}">{key}</a></td>'+''.join(f'<td>{errors[name]:.6f}</td>' for name in ('conditional','hermite','fixed5'))+
                     f'<td>{row["gain_vs_hermite"]:+.6f}</td><td>{row["anchor_in_ell"]:.6f}ℓ</td></tr>')
        controls=''.join(f'<label><input type="checkbox" checked data-case="{key}" data-curve="{name}">{html.escape(LABELS[name])}</label>' for name in COLORS)
        candidates=''.join(f'<tr><td>{i+1}{" ✓" if i==r["production"]["selected_anchor_index"] else ""}</td>'+
            ''.join(f'<td>{value:.8g}</td>' for value in (a['distance']*r['gamma'],a['mean'],a['p'],a['q'],a['score']))+'</tr>'
            for i,a in enumerate(r['production']['anchor_candidates']))
        cards.append(f'<article id="{key}"><h2>{key} · 像素 {r["pixel"]}，sample {r["sample"]}，第 {r["bounce"]+1} 次碰撞</h2>'
            f'<p>Hermite 锚点 {row["hermite_anchor_in_ell"]:.6f}ℓ → 固定点选出的锚点 {row["anchor_in_ell"]:.6f}ℓ。'
            f'相对 Hermite 的 KS 改善 {row["gain_vs_hermite"]:+.6f}；相对 global_conditional 的改善 {row["gain_vs_conditional"]:+.6f}。正数表示改善。</p>'
            f'<div class="toggles">{controls}</div>{inline_svg(folder/(key+".svg"),key)}'
            '<details><summary>全部 5 个候选点</summary><table><tr><th>候选</th><th>s/ℓ</th><th>实际 SDF</th><th>p</th><th>q</th><th>p+qu₀</th></tr>'+
            candidates+'</table></details>'+f'<p><a href="{key}.png">PNG</a> · <a href="{key}.svg">SVG</a> · <a href="{key}-curves.npz">曲线数据</a></p></article>')
        print(key,errors,'anchor / ell',row['anchor_in_ell'],'pass',check['pass'],flush=True)
    validation=dict(all_pass=all(c['pass'] for c in checks),checks=checks,source_provenance=provenance,
        executable_sha256=digest(g.ROOT/'build/Release/fp_gradient_rays.exe'),candidate_count=5,
        refinement=False,reference_used_for_selection=False)
    g.write(folder/'validation.json',validation);g.write(folder/'report-data.json',dict(cases=rows,validation=validation))
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>固定 5 点的单锚点修正</title>
<style>body{max-width:1380px;margin:30px auto;padding:0 22px;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#172131;background:#f6f8fa}article{padding:24px;background:white;border:1px solid #dde3e9;border-radius:10px;margin:24px 0}svg,img{max-width:100%;height:auto}a{color:#007f7c}table{border-collapse:collapse}th,td{padding:9px 16px;border-bottom:1px solid #ddd;text-align:right}.toggles{display:flex;gap:18px;flex-wrap:wrap}label{cursor:pointer}</style>
<h1>真实 SDF：固定 5 个候选点，最终只用 1 个锚点</h1>
<p>ℓ=0.2、σ=0.02。沿用原来的 G1–G6 和 F1，固定出生值、完整梯度、方向、碰撞距离和 GP reference；没有重新筛选有利案例。</p>
<p>窗口起点 a=max(0,t−1.5/γ)，候选 sᵢ=a+(i+0.5)(t−a)/5，i=0,…,4。用真实 SDF 和 SE 条件协方差计算 p、q，在未修正 Rice 分布的标准化众数 u₀ 处，选 p+qu₀ 最小的候选。每次碰撞只选一次，不加密，不用参考误差参与选择。</p>
<p>上方 SDF 曲线来自已保存的剖面；选锚实际只查询五个固定位置。下方曲线可单独开关。KS=max|CDF−CDF_reference|，越小越好。GP reference 沿用原来的加密有限前缀参考；这是固定碰撞点的梯度比较，不能直接推断整幅渲染误差。五个候选仍可能漏掉狭窄瓶颈。</p>
<p><a href="validation.json">独立数值验证</a> · <a href="report-data.json">完整结果</a> · <a href="fixed5_snr.json">C++ 固定点导出</a> · <a href="hermite.json">C++ Hermite 导出</a></p>
<table><tr><th>光线</th><th>conditional KS</th><th>Hermite KS</th><th>固定 5 点 KS</th><th>相对 Hermite 改善</th><th>新锚点</th></tr>'''
    doc+=''.join(table)+'</table>'
    worse=[r['key'] for r in rows if r['gain_vs_hermite']<0]
    f1=next(r for r in rows if r['key']=='F1')
    doc+=f'<p>F1 的 KS 从 {f1["errors"]["hermite"]:.6f} 降至 {f1["errors"]["fixed5"]:.6f}。'
    doc+=f'本组中 {"、".join(worse)} 比旧 Hermite 更差；固定点方法并不保证每条光线都改善。</p>'
    doc+=''.join(cards)+'''<script>document.querySelectorAll('input[data-curve]').forEach(box=>box.addEventListener('change',()=>{for(const suffix of ['pdf','cdf']){const line=document.getElementById(box.dataset.case+'_curve-'+suffix+'-'+box.dataset.curve);if(line)line.style.display=box.checked?'':'none';}}));</script></html>'''
    (folder/'comparison.html').write_text(doc,encoding='utf-8')
    if not validation['all_pass']:raise RuntimeError('Fixed-ray checks failed; inspect validation.json')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','report'))
    parser.add_argument('--output',type=Path,default=DEFAULT)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else report)(args.output.resolve())
