"""Plots and an offline interactive report for the selected groove pixel."""
import base64
import gzip
import hashlib
import html
import json
from html.parser import HTMLParser
from pathlib import Path
import re
import pixel_debug as p
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

OUT=p.OUT
COLORS={'renewal':'#1c577b','conditional':'#d58516','fixed5':'#bf326b'}
LABELS={'renewal':'Native renewal+','conditional':'global_conditional','fixed5':'conditional + fixed5'}

def ecdf(a,x):
    return np.searchsorted(np.sort(a),x,side='right')/len(a)

def wilson(successes,n):
    z=1.959963984540054;q=np.asarray(successes)/n
    center=(q+z*z/(2*n))/(1+z*z/n)
    half=z*np.sqrt(q*(1-q)/n+z*z/(4*n*n))/(1+z*z/n)
    return center-half,center+half

def gradient_variables(rows,w):
    w=np.array(w);w/=np.linalg.norm(w)
    basis=np.eye(3)[np.argmin(abs(w))];u=np.cross(w,basis);u/=np.linalg.norm(u);v=np.cross(w,u)
    result={}
    for key in COLORS:
        g=np.asarray([r[key] for r in rows]);normal=g/np.linalg.norm(g,axis=1)[:,None]
        result[key]=dict(axial=g@w,tangent_u=g@u,tangent_v=g@v,
            incidence_angle=np.degrees(np.arccos(np.clip(-normal@w,-1,1))))
    return result

def analyze():
    cases=[]
    sample_checks={};integral_checks=[]
    old_samples=p.read(OUT/'native-sample-check-before-refinement.json')
    for selected in p.read(OUT/'flight-selection.json')['records']:
        key=selected['key'];d=p.read(OUT/'flights'/key/'result.json');r=d['record']
        n=d['samples'];times=np.array(d['times']);ages=np.array(d['ages']);fail=times<0
        hit=np.array(d['statuses'])=='hit';accepted_times=times[hit]
        upper=1-np.searchsorted(np.sort(accepted_times),ages,side='right')/n
        lower=np.maximum(0,upper-fail.mean());conditional=1-np.array(d['conditional_cdf'])
        center=(lower+upper)/2;k=int(np.argmax(abs(center-conditional)))
        rows=d['gradient_samples'];enough_gradients=len(rows)>=128
        variables=gradient_variables(rows,r['wi']) if enough_gradients else {}
        narrow=[q for q in rows if abs(q['t']-r['focus_distance'])<=d['gradient_half_width']/2]
        narrow_variables=gradient_variables(narrow,r['wi']) if len(narrow)>=128 else {}
        errors={};narrow_errors={}
        for mode in ['conditional','fixed5']:
            errors[mode]={v:float(ks_2samp(variables['renewal'][v],variables[mode][v]).statistic) for v in variables.get('renewal',{})}
            narrow_errors[mode]={v:float(ks_2samp(narrow_variables['renewal'][v],narrow_variables[mode][v]).statistic) for v in narrow_variables.get('renewal',{})}
        shadow=[]
        for q in d['shadow_queries']:
            lo,hi=wilson(q['renewal_mean']*q['samples'],q['samples'])
            value=dict(q,renewal_ci95=[float(lo),float(hi)])
            value['conditional_ci95']=[q['conditional_mean']-1.96*q['conditional_standard_error'],q['conditional_mean']+1.96*q['conditional_standard_error']]
            # The cumulative hazard is integrated adaptively on its grid.
            value['conditional_integral_interpolated']=float(np.interp(q['distance'],ages,conditional))
            error=abs(value['conditional_integral_interpolated']-q['conditional_mean'])
            integral_checks.append(dict(key=key,distance=q['distance'],absolute_difference=error,
                standard_errors=error/max(q['conditional_standard_error'],1e-10),
                pass_3p5se=bool(error<3.5*q['conditional_standard_error']+1e-8)))
            shadow.append(value)
        case=dict(key=key,record=r,samples=n,accepted=int(hit.sum()),escaped=d['escaped'],failures=d['failures'],
            cdf_difference_at_selected_distance=dict(distance=float(ages[k]),conditional_T=float(conditional[k]),
                renewal_T_bounds=[float(lower[k]),float(upper[k])],absolute_midpoint_difference=float(abs(center[k]-conditional[k]))),
            matched_gradient_count=len(rows),narrow_gradient_count=len(narrow),gradient_half_width=d['gradient_half_width'],
            gradient_samples_sufficient=enough_gradients,
            gradient_KS=errors,narrow_gradient_KS=narrow_errors,shadow=shadow,
            quadrature_error_estimate=d['quadrature_error_estimate'])
        case['gradient_moments']={m:{v:dict(mean=float(a.mean()),sd=float(a.std(ddof=1))) for v,a in vs.items()} for m,vs in variables.items()}
        payload={k:d[k] for k in ['times','statuses','native_gradients','gradient_samples']}
        sample_checks[key]=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==old_samples[key]
        refpath=OUT/'gradients'/f'ray-{r["id"]}.json'
        if refpath.exists():
            ref=p.read(refpath);case['fixed_t_SE_GP_reference']=dict(errors=ref['errors'],reference=ref['reference'])
        cases.append(case)
        fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
        ax=axes[0,0];ax.plot(ages,center,c=COLORS['renewal'],label='Native collision survival')
        ci_low,_=wilson(lower*n,n);_,ci_high=wilson(upper*n,n)
        ax.fill_between(ages,ci_low,ci_high,color=COLORS['renewal'],alpha=.14,label='95% pointwise + failed samples')
        ax.plot(ages,conditional,c=COLORS['conditional'],label='Conditional (both gradient modes)')
        for i,q in enumerate(shadow):
            lo,hi=q['renewal_ci95'];ax.errorbar(q['distance'],q['renewal_mean'],yerr=[[q['renewal_mean']-lo],[hi-q['renewal_mean']]],
                fmt='o',c=COLORS['renewal'],ms=4,capsize=3,label='Native shadow samples' if i==0 else None)
            ax.errorbar(q['distance'],q['conditional_mean'],yerr=1.96*q['conditional_standard_error'],fmt='x',
                c=COLORS['conditional'],ms=5,capsize=3,label='Conditional shadow samples' if i==0 else None)
        ax.axvline(r['focus_distance'],c='#777',ls=':',lw=.8)
        if len(accepted_times)>0:
            lo,hi=np.quantile(accepted_times,[.005,.995]);width=max(hi-lo,.002)
            lo=max(0,min(lo,r['focus_distance'])-.12*width);hi=min(r['length'],max(hi,r['focus_distance'])+.12*width)
            ax.set_xlim(lo,hi)
            visible=(ages>=lo)&(ages<=hi);ymin=min(lower[visible].min(),conditional[visible].min())
            ax.set_ylim(max(-.03,ymin-.03*max(1-ymin,.01)),1+.03*max(1-ymin,.01))
        ax.set(title='Survival / shadow: crossing region',xlabel='Distance from frozen birth (zoom)',ylabel='T');ax.legend(fontsize=7)
        ax=axes[0,1];ax.plot(ages,np.array(d['flight_mean'])/r['sigma'],c='#52616d')
        ax.axhline(0,c='#777',lw=.7);ax.axvline(r['focus_distance'],c='#bd3b69',ls='--')
        ax.axvspan(max(0,r['focus_distance']-d['gradient_half_width']),r['focus_distance']+d['gradient_half_width'],alpha=.12,color='#bd3b69')
        ax.set(title='SDF and matched-distance gradient window',xlabel='Distance',ylabel='SDF / sigma')
        for ax,name,title in zip([axes[0,2],axes[1,0],axes[1,1],axes[1,2]],
                ['axial','incidence_angle','tangent_u','tangent_v'],['Axial gradient','Incidence angle (degrees)','Transverse gradient u','Transverse gradient v']):
            if enough_gradients:
                for mode in COLORS:
                    a=np.sort(variables[mode][name]);ax.step(a,np.arange(1,len(a)+1)/len(a),where='post',c=COLORS[mode],label=LABELS[mode],lw=1.3,ls={'renewal':'-','conditional':'--','fixed5':':'}[mode])
                ax.legend(fontsize=7)
            else:
                ax.text(.5,.5,f'Only {len(rows)} collision samples\nInsufficient for a gradient comparison',ha='center',va='center',transform=ax.transAxes)
            ax.set(title=title,xlabel=title,ylabel='Empirical CDF',ylim=(-.02,1.02))
        for ax in axes.flat:ax.grid(alpha=.15)
        fig.suptitle(f'{key}: {r["source"]}, sample {r["sample"]}, event {r["event_index"]} | '
                     f'{n} native flights; {len(rows)} matched gradients',fontsize=12)
        fig.savefig(OUT/'flights'/key/'comparison.png',dpi=140);fig.savefig(OUT/'flights'/key/'comparison.svg');plt.close(fig)
        if refpath.exists():
            curves=np.load(OUT/'gradients'/f'ray-{r["id"]}-curves.npz')
            fig,axes=plt.subplots(1,2,figsize=(10,3.8),layout='constrained')
            for ax,suffix in zip(axes,['pdf','cdf']):
                for label,name,color,width,style in [('SE-GP prefix reference','reference','#253849',3,'-'),('global_conditional','conditional','#d58516',2.2,'--'),('conditional + fixed5','bridge','#bf326b',1.7,':')]:
                    ax.plot(curves['v'],curves[name+'_'+suffix],label=label,color=color,lw=width,ls=style)
                ax.set(title=f'Exact fixed t = {r["focus_distance"]:.6g}: {suffix.upper()}',xlabel='Axial gradient',ylabel=suffix.upper());ax.grid(alpha=.15);ax.legend(fontsize=8)
            fig.savefig(OUT/'flights'/key/'fixed-t-gradient.png',dpi=140);plt.close(fig)
    p.write(OUT/'controlled-statistics.json',cases)
    assert all(sample_checks.values()),sample_checks
    assert all(q['pass_3p5se'] for q in integral_checks),integral_checks
    p.write(OUT/'controlled-validation.json',dict(all_pass=True,native_samples_unchanged_after_plot_grid_refinement=sample_checks,
        conditional_shadow_vs_integrated_hazard=integral_checks))
    print(json.dumps([dict(key=c['key'],delta=c['cdf_difference_at_selected_distance'],ks=c['gradient_KS']) for c in cases]),flush=True)

def table(headers,rows):
    return '<div class="scroll"><table><thead><tr>'+''.join('<th>'+str(h)+'</th>' for h in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+str(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</tbody></table></div>'

def numerical_diagnostics():
    stats=p.read(OUT/'path-statistics.json');base=stats['renewal_plus']
    steps=[dict(mode='renewal_plus',step=.01,samples=base['samples'],mean_luminance=base['mean_luminance'],
        standard_error=base['luminance_standard_error'],failed=base['terminations'].get('outward_crossing_rejected',0),verified=128)]
    steps+=p.read(OUT/'step-ablation.json')
    modes=[s['mode'] for s in steps]+['global_conditional','conditional_fixed5']
    ys={m:np.array([q['radiance'] for q in p.read(OUT/'paths'/m/'paths.json')['paths']])@p.WEIGHTS for m in modes}
    paired=[]
    for a,b in [('global_conditional','renewal_plus'),('global_conditional','renewal_step00125'),
                ('conditional_fixed5','global_conditional'),('renewal_step00125','renewal_plus')]:
        delta=ys[a]-ys[b]
        paired.append(dict(a=a,b=b,mean=float(delta.mean()),ci95_halfwidth=float(1.96*delta.std(ddof=1)/len(delta)**.5)))
    fig,ax=plt.subplots(figsize=(10,4.3),layout='constrained')
    for i,m in enumerate(modes):
        y=ys[m];ax.errorbar(y.mean(),i,xerr=1.96*y.std(ddof=1)/len(y)**.5,fmt='o',capsize=4,
            c=COLORS['renewal' if i<4 else 'conditional' if i==4 else 'fixed5'])
    ax.set_yticks(range(len(modes)),[f'Native renewal+, step {s["step"]:g}' for s in steps]+['global_conditional','conditional + fixed5'])
    ax.invert_yaxis();ax.set(xlabel='Mean linear luminance, 512 paths (approx. 95% interval)',title='Selected pixel: native stepping sensitivity');ax.grid(axis='x',alpha=.2)
    fig.savefig(OUT/'step-ablation.png',dpi=150);fig.savefig(OUT/'step-ablation.svg');plt.close(fig)

    field=p.read(OUT/'failure-field/result.json');candidate=field['at_candidate'];points=field['points']
    times=np.array([q['t'] for q in points]);values=np.array([q['value'] for q in points]);grad=np.array([q['axial_gradient'] for q in points])
    indices=np.flatnonzero((values[:-1]>0)&(values[1:]<0));assert len(indices)>0
    i=int(indices[0]);root_bracket=[float(times[i]),float(times[i+1])]
    replay=field['spec']['replay_realization'];paths=p.read(OUT/'paths/renewal_plus/paths.json')['paths']
    original=next(e for q in paths if q['sample']==replay['sample'] for e in q['events'] if e['type']=='distance_failed')
    replay_error=abs(float(np.dot(original['gradient'],field['record']['wi']))-candidate['axial_gradient'])
    assert replay_error<1e-7
    fig,axes=plt.subplots(1,2,figsize=(11,3.8),layout='constrained')
    axes[0].plot(times,values/.02,c='#1c577b',label='Same realized field F / sigma')
    axes[0].scatter([candidate['t']],[candidate['value']/.02],c='#bf326b',zorder=5)
    axes[1].plot(times,grad,c='#1c577b',label='Same realized axial gradient')
    for ax in axes:
        ax.axhline(0,c='#888',lw=.7);ax.axvline(candidate['t'],c='#bf326b',ls='--',label='Rejected candidate')
        ax.axvline(sum(root_bracket)/2,c='#13805d',ls=':',label='First downward zero (dense bracket)')
        ax.grid(alpha=.15);ax.set_xlabel('Distance from birth');ax.legend(fontsize=8)
    axes[0].set_ylabel('F / sigma');axes[1].set_ylabel('Gradient dot ray direction')
    fig.suptitle('Native sample 1, bounce 1: rejected point is before the actual downward crossing')
    fig.savefig(OUT/'failure-field/comparison.png',dpi=150);plt.close(fig)

    controlled_steps=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,key in zip(axes,['N','C']):
        for suffix,step,color,style in [('',.01,'#1c577b','-'),('-step00125',.00125,'#36a2b4','--')]:
            d=p.read(OUT/f'flights/{key}{suffix}/result.json');ages=np.array(d['ages']);t=np.array(d['times'])
            hit=np.array(d['statuses'])=='hit';fail=(t<0).mean();upper=1-np.searchsorted(np.sort(t[hit]),ages,side='right')/len(t)
            ax.plot(ages,upper-fail/2,c=color,ls=style,label=f'Native step {step:g}')
            ax.fill_between(ages,upper-fail,upper,color=color,alpha=.15)
            q=d['shadow_queries'][0]
            ax.scatter(q['distance'],q['renewal_mean'],color=color,marker='o',s=30)
            controlled_steps.append(dict(key=key,step=step,failures=d['failures'],focus=q,
                collision_survival_at_focus_bounds=[float(np.interp(q['distance'],ages,upper-fail)),float(np.interp(q['distance'],ages,upper))]))
            if not suffix:
                ax.plot(ages,1-np.array(d['conditional_cdf']),c=COLORS['conditional'],ls=':',label='Conditional survival')
                lo,hi=np.quantile(t[hit],[.001,.999]);width=max(hi-lo,.002);ax.set_xlim(lo-.1*width,hi+.1*width)
        ax.set(title=f'{key}: curves = distance survival, dots = shadow',xlabel='Distance',ylabel='T',ylim=(-.02,1.02));ax.grid(alpha=.15);ax.legend(fontsize=8)
    fig.savefig(OUT/'controlled-step-ablation.png',dpi=150);plt.close(fig)
    result=dict(path_steps=steps,paired_differences=paired,controlled_steps=controlled_steps,
        failure_replay=dict(sample=replay['sample'],bounce=replay['bounce'],candidate=candidate,
            first_downward_root_bracket=root_bracket,original_gradient_replay_error=replay_error))
    p.write(OUT/'numerical-diagnostics.json',result)
    step_table=table(['模式 / 原生最大步长','平均线性亮度 ± 约95%区间','朝外梯度拒绝 / 512'],[
        [f'renewal+ / {s["step"]:g}',f'{s["mean_luminance"]:.4f} ± {1.96*s["standard_error"]:.4f}',s['failed']] for s in steps]+[
        [p.LABELS[m],f'{stats[m]["mean_luminance"]:.4f} ± {1.96*stats[m]["luminance_standard_error"]:.4f}',0] for m in ['global_conditional','conditional_fixed5']])
    shadow_table=table(['段','原生最大步长','原生阴影 T','终点 F≥0 的比例','返回无遮挡但终点 F&lt;0 / 4096','conditional 阴影 T'],[
        [s['key'],s['step'],f'{s["focus"]["renewal_mean"]:.5f}',f'{s["focus"]["native_positive_endpoint_fraction"]:.5f}',
         s['focus']['native_escaped_negative_endpoint'],f'{s["focus"]["conditional_mean"]:.5f}'] for s in controlled_steps])
    return f'''<section id="numerics"><h2>先核查原生 renewal+ 的数值基准</h2>
<p>原生最大步长由 0.01 缩小到 0.00125 后，该像素平均亮度由 <b>0.6532 升到 0.9565</b>，拒绝路径由 174 降到 30。相同样本编号的亮度差为 +0.3034 ± 0.0652（约95%区间）。这是当前结果中最明确的变化；原来较暗的 renewal+ 像素不能直接当作已收敛的真值。</p>
{step_table}<img src="step-ablation.png">
<p>步长 0.00125 的 renewal+ 与 conditional 之差已落在本次采样的不确定性内：conditional − renewal+ = 0.0322 ± 0.1155。fixed5 − conditional = −0.0911 ± 0.1096，也不能仅凭这 512 条路径确认 fixed5 改善了像素。以上差值区间用相同编号样本的差计算；相同编号不表示两模型共享相同随机表面。</p>
<h3>一个实际拒绝事件：求交返回点尚未穿零</h3>
<p>重建 sample 1、bounce 1 的同一随机场，候选距离为 {candidate['t']:.8f}，场值 F={candidate['value']:.7f}，沿光线梯度为 +{candidate['axial_gradient']:.6f}。密集检查的首次向下穿零位于 [{root_bracket[0]:.8f}, {root_bracket[1]:.8f}]。候选点梯度与原路径记录一致（误差 {replay_error:.1g}）。这条路径的拒绝由求交返回点过早触发，不能解释成正确碰撞点上的梯度分布差异。</p>
<img src="failure-field/comparison.png"><p class="note">这里只重建并检查原有实现，没有修改正式求交算法；单个例子也不能代表全部拒绝事件。</p>
<h3>短段阴影还有终点漏检</h3>
<p>N、C 的起点 F=0 且出射导数为正。如果终点 F&lt;0，连续场中必然已经穿零。然而现有步进循环只检查 t&lt;farT 的采样点，结束时计算了终点值却没有检查最后一个区间。表中出现了大量“终点为负但返回无遮挡”的样本。圆点与曲线不一致因此不能直接归咎于 conditional 的透射率模型。终点正值比例只是上界检查，不能替代整段存活概率。</p>
{shadow_table}<img src="controlled-step-ablation.png">
<p class="note">这些阴影测试在原段的碰撞附近截短后重新调用阴影函数；属于固定条件诊断，不是原路径中真实光源连接的能量贡献。原生步长还受段长 / 8 限制。缩短最大步长减少了漏检，但仍未消除，也不保证所有固定起点的拒绝数量单调下降。</p>
<p><a href="numerical-diagnostics.json">数值诊断数据</a> · <a href="failure-field/result.json">同一随机场的密集采样</a> · <a href="step-ablation.svg">步长图 SVG</a></p></section>'''

def build():
    data={m:p.read(OUT/'paths'/m/'paths.json') for m in p.MODES};stats=p.read(OUT/'path-statistics.json');validation=p.read(OUT/'path-validation.json')
    cases=p.read(OUT/'controlled-statistics.json')
    mesh=data['renewal_plus'].pop('geometry')
    bundle=dict(data=data,modes=p.MODES,labels=p.LABELS,mesh=mesh,cases=[{k:c['record'][k] for k in ['key','source','sample','event_index']} for c in cases])
    payload=json.dumps(base64.b64encode(gzip.compress(json.dumps(bundle,separators=(',',':'),allow_nan=False).encode(),mtime=0)).decode())
    rows=[]
    for m in p.MODES:
        s=stats[m];rows.append([p.LABELS[m],s['samples'],f"{s['mean_luminance']:.5f} ± {1.96*s['luminance_standard_error']:.5f}",
            f"{s['mean_scatters']:.3f}",f"{s['multiple_fraction']:.1%}",html.escape(str(s['terminations']))])
    overview=table(['模式','路径数','平均线性亮度 ± 约95%区间','平均反射次数','至少2次','终止原因'],rows)
    overview2=table(['段','来源 / sample / event','最大存活差（含失败时取区间中点）','原生接受 / 逃逸 / 失败','梯度样本','轴向 KS: conditional / fixed5'],[
        [f'<a href="#{c["key"]}">{c["key"]}</a>',f'{c["record"]["source"]} / {c["record"]["sample"]} / {c["record"]["event_index"]}',
         f'{c["cdf_difference_at_selected_distance"]["absolute_midpoint_difference"]:.4f}',f'{c["accepted"]} / {c["escaped"]} / {sum(c["failures"].values())}',c['matched_gradient_count'],
         ' / '.join(f'{c["gradient_KS"][m]["axial"]:.4f}' for m in ['conditional','fixed5']) if c['gradient_samples_sufficient'] else '样本不足'] for c in cases])
    sections=[]
    for c in cases:
        key=c['key'];r=c['record'];srows=[]
        for q in c['shadow']:
            srows.append([f'{q["distance"]:.6f}',f'{q["renewal_mean"]:.5f} [{q["renewal_ci95"][0]:.5f}, {q["renewal_ci95"][1]:.5f}]',
                f'{q["conditional_mean"]:.5f} ± {1.96*q["conditional_standard_error"]:.5f}',q['samples']])
        ref=c.get('fixed_t_SE_GP_reference');refhtml=''
        if ref:
            refhtml=f'<h3>固定同一个 t 的 SE-GP 梯度参考</h3><p>这里参考是带正值前缀约束的 SE-GP 数值积分，与上方原生稀疏卷积样本分开。conditional / fixed5 的轴向 KS 为 {ref["errors"]["conditional"]["ks"]:.6g} / {ref["errors"]["bridge"]["ks"]:.6g}。参考末两级网格 CDF 差 {ref["reference"]["grid_ks_medium_vs_fine"]:.2g}，重复采样最大逐点标准误 {ref["reference"]["max_pointwise_cdf_se"]:.2g}。</p><img src="flights/{key}/fixed-t-gradient.png">'
            if ref['reference']['summary']['log_prefix_mass_over_endpoint_flux']==0:
                refhtml+='<p class="note">本次参考的前缀检查未观察到额外拒绝，三条曲线基本重合。这说明此固定 t 下，起始观测与终点穿零已使前缀正值约束几乎不再改变梯度；不代表所有光线都精确，或参考没有任何数值误差。</p>'
        narrow_text=' / '.join(f"{c['narrow_gradient_KS'][m]['axial']:.4f}" for m in ['conditional','fixed5']) if c['narrow_gradient_count']>=128 else '样本不足，不计算'
        shortage='' if c['gradient_samples_sufficient'] else '<p><b>本段只有 5 个接受碰撞，不能据此判断梯度分布好坏；这里只比较存活与阴影。</b></p>'
        sections.append(f'''<section id="{key}"><h2>{key}：{p.LABELS[r['source']]} · sample {r['sample']} · event {r['event_index']}</h2>
<p>起始值 {r['f0']:.6g}，沿射线的起始导数 {r['g0']:.6g}，长度 {r['length']:.6g}。梯度中心距离 t={r['focus_distance']:.6g}（{r['focus_distance']*r['gamma']:.4g} ℓ），半窗口 {c['gradient_half_width']:.6g}，共 {c['matched_gradient_count']} 个原生接受碰撞。</p>
<p>conditional 和 fixed5 都在每个原生样本的同一 t 处采样完整梯度，因此三条经验梯度曲线共享碰撞距离分布。此窗口结果不是严格固定一个 t 的条件分布。原生失败事件：{html.escape(str(c['failures']))}。</p>{shortage}
<img src="flights/{key}/comparison.png">{table(['阴影距离','renewal+ T（Wilson 95%区间）','conditional T ± 约95%区间','每种重复数'],srows)}
<p>窗口减半后保留 {c['narrow_gradient_count']} 个样本；轴向 KS（conditional / fixed5）：{narrow_text}。经验 KS 包含有限采样噪声及原生数值误差，不能直接当作 SE-GP 近似误差。</p>
{refhtml}<p><a href="flights/{key}/comparison.svg">SVG</a> · <a href="flights/{key}/result.json">原始重复采样数据</a> · <a href="flights/{key}/spec.json">固定条件</a></p></section>''')
    template=(Path(__file__).with_name('pixel_debug_viewer.html')).read_text(encoding='utf-8')
    page=template.replace('__OVERVIEW__',overview).replace('__CONTROLLED_OVERVIEW__',overview2).replace('__SECTIONS__',''.join(sections)).replace('__PAYLOAD__',payload)
    page=page.replace('__NUMERICS__',numerical_diagnostics())
    page=page.replace('__VERIFIED__',str(sum(v['observer_verified_samples'] for v in validation['modes'].values())))
    if not (OUT/'location.html').exists():
        (OUT/'location.html').write_text((OUT/'comparison.html').read_text(encoding='utf-8'),encoding='utf-8')
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    class Audit(HTMLParser):
        def __init__(self):super().__init__();self.missing=[];self.count=0
        def handle_starttag(self,tag,attrs):
            a=dict(attrs);target=a.get('href') or a.get('src')
            if target and not target.startswith(('#','http:','https:','data:')):
                self.count+=1
                if not (OUT/target).exists():self.missing.append(target)
    audit=Audit();audit.feed(page);assert not audit.missing,audit.missing
    p.write(OUT/'artifact-validation.json',dict(all_pass=True,links_checked=audit.count,report_sha256=p.digest(OUT/'comparison.html'),missing=[]))
    print('Built',OUT/'comparison.html',flush=True)

if __name__=='__main__':
    analyze();build()
