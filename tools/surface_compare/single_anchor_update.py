"""Compare the updated production law on the exact previously diagnosed rays.

prepare freezes the old records and GP curves and writes a C++ detail request.
report consumes that C++ export, independently checks it with SE-feature QR,
and writes interactive HTML plus standalone PNG/SVG distribution figures.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import html
import math
from pathlib import Path
import shutil
import sys

import gradient_rays as g
import numpy as np
from scipy.optimize import brentq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gradient_rays_report import projected, inline_svg

ROOT = g.ROOT
GENERAL = ROOT/'outputs/shaderball-gradient-ray-diagnostic-20261108'
GROOVE = ROOT/'outputs/shaderball-groove-flight-diagnostic-20261108'
COLORS = dict(classic='#737d8c', conditional='#d78816', old='#9a5de2', single='#007b77', reference='#172131')
LABELS = dict(classic='classic_global', conditional='global_conditional', old='old 2-anchor', single='new 1-anchor', reference='GP reference')
MARKERS = dict(classic='o',conditional='s',old='^',single='d',reference='x')
STYLES = dict(classic=':',conditional='--',old='-.',single='-',reference='-')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(root):
    folder = root/'rays'; folder.mkdir(parents=True,exist_ok=True)
    cases = []
    for row in g.read(GENERAL/'report-data.json')['cases']:
        curves = GENERAL/row['artifact_folder']/f"ray-{row['id']}-curves.npz"
        cases.append(dict(key=row['key'],group='general',old=row,source_curves=str(curves.relative_to(ROOT))))
    for row in g.read(GROOVE/'report-data.json')['cases']:
        if row['gradient'] is None:
            continue
        curves = GROOVE/'gradients'/f"ray-{row['id']}-curves.npz"
        cases.append(dict(key='G-'+row['key'],group='groove',old=row['gradient'],
                          old_title=row['title'],source_curves=str(curves.relative_to(ROOT))))
    records = []
    for case in cases:
        source = ROOT/case['source_curves']
        case['source_sha256'] = digest(source)
        shutil.copyfile(source,folder/(case['key']+'-old-curves.npz'))
        z = np.load(source)
        record = copy.deepcopy(case['old']['record'])
        record['comparison_key'] = case['key']
        v = z['v']; record['query_derivatives'] = v[np.linspace(0,len(v)-1,129).astype(int)].tolist()
        records.append(record)
    scene = g.read(GENERAL/'detail.json')['scene']
    g.write(folder/'detail-input.json',dict(scene=scene,records=records,anchor_selection='hermite'))
    g.write(folder/'source-cases.json',dict(cases=cases,scene=scene,
        source_reports={str(p.relative_to(ROOT)):digest(p) for p in [GENERAL/'report-data.json',GROOVE/'report-data.json']},
        note='Frozen birth jet, direction, distance, actual SDF profiles and previous GP reference. No reselection.'))
    shutil.copyfile(ROOT/Path(scene).parent.parent/'baseline/shaderball.png',folder/'shaderball.png')
    shutil.copyfile(GROOVE/'E.png',folder/'G-E-flight.png')
    print(f'Prepared {len(cases)} identical fixed-event rays',flush=True)


def check(record):
    ray = g.Ray(record); p = record['production']
    m = ray.mu/ray.sd
    speed = ray.sd*(2/(math.hypot(m,2)+m) if m>=0 else (math.hypot(m,2)-m)/2)
    f,t,initial = record['f0'],record['t'],record['g0']
    # Solve the derivative of the Hermite SNR in theta directly, independent
    # of the production Cardano solver in x=theta/(1-theta).
    def derivative_numerator(theta):
        return -2*f*(1-theta)**2-(2*f+t*initial)*theta*(1-theta)**2+t*speed*theta**3
    theta = brentq(derivative_numerator,1e-15,1-1e-15,xtol=5e-15)
    expected = max(t*theta,t-1.5/record['gamma'])
    positions = record['anchor_positions']
    assert len(positions)==1
    candidate_error = 0.
    if p['approximation']=='single_anchor_fixed5_snr':
        begin = max(0.,t-1.5/record['gamma'])
        candidates = begin+(np.arange(5)+.5)*(t-begin)/5
        exported = p['anchor_candidates']
        assert len(exported)==5
        assert np.max(abs(candidates-[a['distance'] for a in exported]))<1e-12
        f,_ = ray.features(candidates)
        residual_sd = np.linalg.norm(f@ray.free4,axis=1)
        pp = (np.asarray([a['mean'] for a in exported])+f@ray.coefficient_mean)/residual_sd
        qq = ((f@ray.free3)@ray.unit_gradient)/residual_sd
        c = -ray.mu/ray.sd
        u0 = -2/(c+math.hypot(c,2)) if c>=0 else .5*(c-math.hypot(c,2))
        scores = pp+qq*u0
        selected = int(np.argmin(scores))
        assert selected==p['selected_anchor_index']
        expected = candidates[selected]
        candidate_error = max(float(np.max(abs(pp-[a['p'] for a in exported])/(1+abs(pp)))),
                              float(np.max(abs(qq-[a['q'] for a in exported])/(1+abs(qq)))),
                              float(np.max(abs(scores-[a['score'] for a in exported])/(1+abs(scores)))))
    else:
        assert p['approximation']=='single_anchor_v2'
    features,_ = ray.features(positions)
    mean = np.asarray(record['anchor_means'])+features@ray.coefficient_mean
    beta = (features@ray.free3)@ray.unit_gradient
    residual_sd = np.linalg.norm(features@ray.free4,axis=1)
    pp,qq = float(mean[0]/residual_sd[0]),float(beta[0]/residual_sd[0])
    law = g.models(record)['bridge']; v = np.asarray(record['query_derivatives'])
    result = dict(key=record['comparison_key'],**ray.checks(),
        candidate_parameter_relative_error=candidate_error,
        anchor_position_error=abs(expected-positions[0]),
        anchor_parameter_relative_error=max(abs(pp-p['p'])/(1+abs(pp)),abs(qq-p['q'])/(1+abs(qq))),
        production_cdf_max_error=float(np.max(abs(law.cdf(v)-record['production_cdf']))),
        production_pdf_relative_error=float(np.max(abs(law.pdf(v)-record['production_pdf'])))/(1+float(np.max(record['production_pdf']))))
    result['pass'] = bool(result['endpoint_error']<1e-8 and result['anchor_position_error']<1e-9 and
        result['anchor_parameter_relative_error']<2e-5 and candidate_error<2e-5 and result['production_cdf_max_error']<2e-7 and
        result['production_pdf_relative_error']<2e-7)
    return law,result


def plot_case(folder,case,record,curves,law,scene,background):
    key = case['key']; r = record; scale = r['sigma']*r['gamma']; v = curves['v']
    fig,axes = plt.subplots(2,2,figsize=(13,9),layout='constrained')
    ax = axes[0,0]; ax.imshow(background,extent=(-.5,255.5,255.5,-.5))
    path = projected(r['vertices'][1:],scene)
    ax.plot(*path.T,color='white',lw=1.1,marker='.',ms=4)
    segment = projected([r['birth_position'],r['position']],scene)
    ax.plot(*segment.T,color='#ee5265',lw=2.2,marker='o',ms=4)
    ax.plot(*r['pixel'],marker='o',ms=11,mfc='none',mec='#ffeb63',mew=1.7)
    ax.set(xlim=(-.5,255.5),ylim=(255.5,-.5),xlabel='camera pixel x',ylabel='camera pixel y',
           title=f"{key}: same pixel {tuple(r['pixel'])}, sample {r['sample']}, hit #{r['bounce']+1}")
    ax = axes[0,1]
    x = np.linspace(0,r['t'],len(r['mean_profile']))*r['gamma']
    ax.plot(x,np.asarray(r['mean_profile'])/r['sigma'],c='#858b96',label='actual shaderball SDF / sigma')
    x = curves['s']*r['gamma']; mean = curves['bridge_mean']/r['sigma']; sd = curves['bridge_sd']/r['sigma']
    ax.plot(x,mean,c='#337c99',label='GP mean | birth jet, F(t)=0')
    ax.fill_between(x,mean-2*sd,mean+2*sd,color='#337c99',alpha=.18,label='+/- 2 conditional std. dev.')
    for j,s in enumerate([r['t']/3,max(2*r['t']/3,r['t']-1/r['gamma'])]):
        ax.axvline(s*r['gamma'],c=COLORS['old'],ls=':',lw=1.3,label='old anchors' if j==0 else None)
    ax.axvline(r['production']['anchor_distance']*r['gamma'],c=COLORS['single'],ls='--',lw=2,label='new anchor')
    ax.axhline(0,c='#172131',lw=.8)
    ax.set(xlabel='distance / ell',ylabel='field value / sigma',ylim=(-8,12),title='Prefix profile (before positivity test)')
    ax.legend(fontsize=8)
    summaries = [case['old']['models'][n] for n in ['classic','conditional','bridge']]+[case['old']['reference']['summary'],law.summary()]
    low = min(s['quantiles'][0]-2.5*s['sd'] for s in summaries)/scale
    high = min(0,max(s['quantiles'][2]+2.5*s['sd'] for s in summaries))/scale
    for col,suffix in enumerate(['pdf','cdf']):
        ax = axes[1,col]
        for j,name in enumerate(['reference','old','single','conditional','classic']):
            ax.plot(v/scale,curves[name+'_'+suffix]*(scale if suffix=='pdf' else 1),
                color=COLORS[name],ls=STYLES[name],lw=2.5 if name=='reference' else 1.7,label=LABELS[name],
                marker=MARKERS[name],markevery=(.035+.05*j,.27),ms=4,mfc='white',gid=f'curve-{suffix}-{name}')
        ax.set(xlim=(low,high),xlabel='axial gradient v / (sigma gamma)',ylabel='density' if suffix=='pdf' else 'CDF',
            title='Normalized collision gradient '+suffix.upper())
        if suffix=='cdf': ax.set_ylim(-.02,1.02)
        ax.legend(fontsize=8); ax.grid(alpha=.15)
    fig.savefig(folder/(key+'.png'),dpi=160); fig.savefig(folder/(key+'.svg')); plt.close(fig)
    fig,axes = plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
    for ax,names,title in [(axes[0],['classic','conditional'],'Base laws minus GP reference'),
                           (axes[1],['old','single'],'Anchor corrections minus GP reference')]:
        for name in names:
            ax.plot(v/scale,curves[name+'_cdf']-curves['reference_cdf'],c=COLORS[name],ls=STYLES[name],label=LABELS[name])
        ax.axhline(0,c='#777',lw=.7);ax.set(xlim=(low,high),xlabel='axial gradient v / (sigma gamma)',ylabel='CDF difference',title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(folder/(key+'-difference.png'),dpi=160);plt.close(fig)


def report(root):
    folder=root/'rays'; source=g.read(folder/'source-cases.json')
    new={r['comparison_key']:r for r in g.read(folder/'production-detail.json')['records']}
    scene=g.read(ROOT/source['scene']);background=plt.imread(folder/'shaderball.png')
    checks=[];results=[];cards=[];table=[]
    for case in source['cases']:
        key=case['key'];r=new[key];old=case['old']['record']
        for field in ['birth_position','birth_gradient','wi','position','t','f0','g0','sigma','gamma','mean_profile']:
            assert r[field]==old[field],(key,field)
        assert digest(folder/(key+'-old-curves.npz'))==case['source_sha256']
        law,checked=check(r);checks.append(checked)
        curves=dict(np.load(folder/(key+'-old-curves.npz')))
        for suffix in ['pdf','cdf']:curves['old_'+suffix]=curves['bridge_'+suffix]
        curves['single_cdf']=law.cdf(curves['v']);curves['single_pdf']=law.pdf(curves['v'])
        assert curves['single_cdf'][0]<1e-7 and curves['single_cdf'][-1]>1-1e-7,key
        errors={name:float(np.max(abs(curves[name+'_cdf']-curves['reference_cdf']))) for name in ['classic','conditional','old','single']}
        delta=errors['old']-errors['single']
        np.savez_compressed(folder/(key+'-curves.npz'),**curves)
        plot_case(folder,case,r,curves,law,scene,background)
        ref=case['old']['reference']
        row=dict(key=key,id=r['id'],group=case['group'],record=r,errors=errors,
            improvement_over_old=delta,single_summary=law.summary(),reference=ref,
            old_reference_curves_sha256=case['source_sha256'],validation=checked)
        results.append(row)
        table.append('<tr><td><a href="#'+key+'">'+key+'</a></td><td>'+str(r['pixel'])+'</td>'+''.join(
            f'<td>{errors[n]:.6f}</td>' for n in ['classic','conditional','old','single'])+f'<td>{delta:+.6f}</td></tr>')
        controls=''.join(f'<label><input type="checkbox" checked data-case="{key}" data-curve="{name}">{LABELS[name]}</label>' for name in COLORS)
        cards.append(f'<article id="{key}"><h2>{key} · 像素 {r["pixel"]} · 第 {r["bounce"]+1} 次碰撞</h2>'
            f'<p>固定距离 t={r["t"]:.8g}（{r["t"]*r["gamma"]:.4f} ℓ）；新版 anchor 位于 {r["production"]["anchor_distance"]*r["gamma"]:.4f} ℓ。'
            f'相对 GP reference，旧版 KS={errors["old"]:.6f}，新版 KS={errors["single"]:.6f}，误差减少 {delta:+.6f}。</p>'
            f'<div class="toggles">{controls}</div>{inline_svg(folder/(key+".svg"),key)}'
            f'<details><summary>CDF 差值图：查看重叠曲线与细小变化</summary><img src="{key}-difference.png" alt="CDF differences"></details>'
            f'<p><a href="{key}.png">PNG</a> · <a href="{key}.svg">SVG</a> · <a href="{key}-curves.npz">曲线数据</a></p></article>')
        print(key,errors,flush=True)
    validation=dict(all_pass=all(x['pass'] for x in checks),checks=checks,fixed_event_count=len(results),
        unchanged_birth_direction_distance_sdf=True,unchanged_gp_reference=True,
        source_reports=source['source_reports'],production_executable_sha256=digest(ROOT/'build/Release/fp_gradient_rays.exe'),
        gp_reference_note='Reused exact stored curves, with original grid and QMC convergence metadata. No new selection or reference resampling.')
    g.write(folder/'validation.json',validation);g.write(folder/'report-data.json',dict(cases=results,validation=validation))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball · 单 anchor 更新 · 同光线比较</title>
<style>body{max-width:1440px;margin:30px auto;padding:0 22px;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#172131;background:#f6f8fa}article,.intro{background:white;padding:24px;margin:24px 0;border:1px solid #dde3e9;border-radius:12px}a{color:#006b70}svg,img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #dde3e9}th:first-child,td:first-child{text-align:left}.toggles{display:flex;gap:18px;flex-wrap:wrap}label{cursor:pointer}h1{font-size:28px}details{margin:12px 0}.scroll{overflow-x:auto}</style>
<h1>新版单 anchor：同一批 Shaderball 光线</h1><p><a href="../comparison.html">256×256 / 128 spp 渲染对比</a> · <a href="validation.json">验证记录</a> · <a href="report-data.json">完整指标</a></p>
<div class="intro"><p>沿用原梯度报告的 8 条光线，以及凹槽报告中有固定碰撞梯度参考的 5 条光线。出生值与完整梯度、方向、碰撞距离、真实 SDF 和 GP reference 全部固定；只重新评估新版梯度 Law，没有重新挑选对新版有利的光线。</p>
<p>五条曲线依次为 classic_global、global_conditional、旧双 anchor、新单 anchor、GP reference。固定相同输入时，classic FP 与 conditional FP 共用同一个新版梯度分布。GP reference 检查当前段的正值前缀，不包含更早路径历史。</p>
<p>新版取 s*=max(θ*t,t−1.5/γ)，直接计算单点条件正性 Φ(p+qu)。图中的紫色虚线保留旧 anchor，绿色虚线表示新版唯一 anchor。蓝色区间是施加正性前的 GP 条件均值 ±2 标准差。</p>
<p>KS=max|CDF−CDF_reference|，越小越好；“误差减少”=旧版 KS−新版 KS，负数表示新版更差。可取消勾选以查看重叠曲线；差值图保留实际数值，不人为错开曲线。参考的有限网格及 QMC 误差沿用原报告，见每条记录的 reference 元数据。</p>
<div class="scroll"><table><tr><th>光线</th><th>像素</th><th>classic KS</th><th>conditional KS</th><th>旧双 anchor KS</th><th>新单 anchor KS</th><th>误差减少</th></tr>'''
    document+=''.join(table)+'</table></div></div>'+''.join(cards)
    document+='''<article><h2>凹槽 E：沿用首次碰撞对照</h2><p>E 原本只有自由程对比，没有固定碰撞处的梯度参考。当前更换梯度近似不会改变给定出生状态下的首次碰撞分布，此图沿用原结果。</p><img src="G-E-flight.png" alt="Groove E unchanged flight comparison"></article>
<script>document.querySelectorAll('input[data-curve]').forEach(box=>box.addEventListener('change',()=>{for(const suffix of ['pdf','cdf']){const line=document.getElementById(box.dataset.case+'_curve-'+suffix+'-'+box.dataset.curve);if(line)line.style.display=box.checked?'':'none';}}));</script></html>'''
    (folder/'comparison.html').write_text(document,encoding='utf-8')
    if not validation['all_pass']:raise RuntimeError('Fixed-ray production validation failed')


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','report']);parser.add_argument('directory',type=Path)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else report)(args.directory.resolve())
