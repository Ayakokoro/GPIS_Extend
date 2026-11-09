"""Compare conditional flights and fixed-event gradients with the GP reference."""
from __future__ import annotations
import argparse
import hashlib
import html
import re
from pathlib import Path
import shutil
import groove_flights as f
import gradient_rays as g
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CASES=[('A',4876,'入口：先掠过外壁，再碰内球'),('B',5032,'入口：bridge 明显改善梯度'),
       ('C',11368,'内壁短光线：conditional 已接近 GP reference'),('D',4999,'贴壁短光线：首次碰撞分布对照'),
       ('N',329,'对照：入口分布已经接近'),('E',2507,'对照：大部分光线通过当前区间')]
COLORS=dict(conditional='#d47b00',reference='#18202d',bridge='#2674dc')
LABELS=dict(conditional='global_conditional',bridge='conditional + bridge',reference='GP reference')

def inline_svg(path,key):
    text=path.read_text(encoding='utf-8');text=text[text.index('<svg'):]
    text=re.sub(r'\bid="([^"]+)"',lambda m:f'id="{key}_{m[1]}"',text)
    text=re.sub(r'((?:xlink:)?href)="#([^"]+)"',lambda m:f'{m[1]}="#{key}_{m[2]}"',text)
    return re.sub(r'url\(#([^)]+)\)',lambda m:f'url(#{key}_{m[1]})',text)

def project(points,scene):
    camera=scene['camera'];t=camera['transform'];o=np.array(t['position'])
    w=np.array(t['look_at'])-o;w=w/np.linalg.norm(w)
    right=np.cross(w,t['up']);right/=np.linalg.norm(right);up=np.cross(right,w)
    p=np.asarray(points)-o;depth=p@w;scale=np.tan(np.deg2rad(camera['fov'])/2)
    return np.column_stack([(p@right/depth/scale+1)*128,(1-p@up/depth/scale)*128])

def table(headers,rows):
    return '<table><thead><tr>'+''.join('<th>'+str(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+str(x)+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'

def check_gradients(folder):
    results=[]
    for r in g.read(folder/'gradients/production-validation.json')['records']:
        law=g.models(r)['bridge'];q=np.array(r['query_derivatives'])
        results.append(dict(id=r['id'],cdf_error=float(np.max(abs(law.cdf(q)-r['production_cdf']))),
            pdf_error=float(np.max(abs(law.pdf(q)-r['production_pdf'])))))
    return results

def main(folder):
    folder=folder.resolve();spec=g.read(folder/'spec.json');scene=g.read(g.ROOT/spec['scene'])
    records={r['id']:r for r in g.read(folder/'refined-profiles.json')['records']}
    records.update({r['id']:r for r in g.read(folder/'short-profiles.json')['records']})
    stats={r['id']:r for r in g.read(folder/'refined.json')['results']}
    stats.update({r['id']:r for r in g.read(folder/'short-refined.json')['results']})
    gradients={r['id']:r for r in g.read(folder/'gradients/refined.json')}
    background=g.ROOT/'outputs/conditional-fp-gradient-shaderball-l0p2-s0p02-256-128spp-20261108/baseline/shaderball.png'
    shutil.copyfile(background,folder/'location-background.png');background=plt.imread(background)
    fig,ax=plt.subplots(figsize=(6.2,6.2),layout='constrained');ax.imshow(background)
    for key,i,_ in CASES:
        x,y=records[i]['pixel'];ax.scatter([x],[y],s=85,facecolors='none',edgecolors='white',linewidth=1.5)
        ax.annotate(key,(x,y),xytext=(7,-10),textcoords='offset points',color='white',weight='bold',fontsize=12,
                    bbox=dict(boxstyle='round',facecolor='#243244',alpha=.85,edgecolor='none'))
    ax.set(xlim=(0,255),ylim=(255,0),title='Nested-sphere groove: source camera pixels',xlabel='pixel x',ylabel='pixel y')
    fig.savefig(folder/'locations.png',dpi=170);plt.close(fig)
    cases=[];cards=[];rows=[]
    for key,i,title in CASES:
        r=records[i];s=stats[i];z=dict(np.load(folder/f'flight-{i}.npz'))
        entry=dict(key=key,id=i,title=title,record=r,flight=s,gradient=gradients.get(i))
        fig,axes=plt.subplots(2,2,figsize=(12.5,8.2),layout='constrained')
        ax=axes[0,0];ax.imshow(background);path=project(r['vertices'][1:],scene)
        ax.plot(path[:,0],path[:,1],color='white',lw=1,marker='.',ms=3)
        seg=project([r['birth_position'],r['position']],scene)
        ax.plot(seg[:,0],seg[:,1],color='#ed42a9',lw=2,marker='o',ms=3)
        ax.scatter([r['pixel'][0]],[r['pixel'][1]],marker='+',c='yellow',s=75)
        ax.set(xlim=(45,205),ylim=(155,25),title=f'{key}: source pixel {r["pixel"]}, hit #{r["bounce"]+1}')
        times=z['reference_times'].ravel();lo,hi=np.quantile(times,[.001,.999]);padding=max((hi-lo)*.12,.0001)
        lo=max(0,lo-padding);hi=min(r['length'],hi+padding)
        ages=z['ages'];inside=(ages>=lo)&(ages<=hi)
        ax=axes[0,1];sigma=np.sqrt(np.maximum(0,np.array(r['scalar_statistics'])[:,1]))
        ax.plot(ages,z['mean'],label='SDF mean m(t)',color='#526579',lw=1.3)
        ax.plot(ages,z['posterior_mean'],label='mean given birth',color='#047b8b',ls='--',lw=1.3)
        ax.fill_between(ages,z['posterior_mean']-2*sigma,z['posterior_mean']+2*sigma,color='#4fa9a9',alpha=.15,label='birth-conditioned +/-2 sd')
        ax.axhline(0,color='black',lw=.7);ax.axvline(r['t'],color='#2674dc',ls=':',lw=1,label='fixed gradient event')
        y=z['posterior_mean'][inside];sy=sigma[inside]
        ax.set(xlim=(lo,hi),ylim=(float(np.min(y-2*sy))-.002,float(np.max(y+2*sy))+.002),
               title='Actual mean geometry along this flight',xlabel='distance from birth',ylabel='signed field value')
        ax.legend(fontsize=7);ax.grid(alpha=.16)
        ax=axes[1,0]
        lines=[('reference',z['reference_cdf'],LABELS['reference'],'-',None),
               ('conditional',z['conditional_cdf'],'global_conditional (+bridge: same flight)','--','s')]
        for j,(name,cdf,label,style,marker) in enumerate(lines):
            ax.plot(ages,cdf,label=label,color=COLORS.get(name,'#b34c9a'),ls=style,lw=1.8,
                    marker=marker,markevery=(.03+j*.035,.22),ms=3.5,mfc='white',gid='flight-'+name)
        ax.set(xlim=(lo,hi),ylim=(-.025,1.025),xlabel='distance from birth',ylabel='first-hit cumulative probability',title='Free-flight first-passage CDF')
        ax.legend(fontsize=7,loc='lower right');ax.grid(alpha=.16)
        ax=axes[1,1]
        if i in gradients:
            gr=gradients[i];curves=dict(np.load(folder/f'gradients/ray-{i}-curves.npz'))
            # CDF is easier to compare for nearly delta-valued short flights.
            distributions=[gr['models'][n] for n in ['conditional','bridge']]+[gr['reference']['summary']]
            a=min(d['quantiles'][0]-d['sd'] for d in distributions);b=max(d['quantiles'][2]+d['sd'] for d in distributions)
            for j,(name,marker,style) in enumerate([('reference','x','-'),('bridge','^','-'),('conditional','s','--')]):
                ax.plot(curves['v'],curves[name+'_cdf'],label=LABELS[name],color=COLORS[name],ls=style,lw=1.7,
                        marker=marker,markevery=(.04+j*.05,.25),ms=4,mfc='white',gid='gradient-'+name)
            ax.set(xlim=(a,min(0,b)),ylim=(-.025,1.025),xlabel="axial gradient v = grad(F) dot direction",ylabel='CDF',
                   title=f'Gradient at fixed t={r["t"]:.6g} (separate experiment)')
            ax.ticklabel_format(axis='x',style='plain',useOffset=False)
            ax.legend(fontsize=8);ax.grid(alpha=.16)
        else:
            ax.plot(ages,z['endpoint_negative'],label='P(F(t)<0 | birth)',color='#8d8d8d',ls=':')
            ax.plot(ages,z['reference_cdf'],label='GP reference: P(first hit <= t | birth)',color=COLORS['reference'])
            ax.set(xlim=(lo,hi),ylim=(0,1),xlabel='distance from birth',ylabel='probability',title='Endpoint sign vs complete prefix')
            ax.legend(fontsize=8);ax.grid(alpha=.16)
        fig.savefig(folder/f'{key}.png',dpi=160);fig.savefig(folder/f'{key}.svg');plt.close(fig)
        gradtext=''
        if i in gradients:
            e=gradients[i]['errors'];gradtext=f'固定原始碰撞距离的梯度 KS：conditional {e["conditional"]["ks"]:.5f}，bridge {e["bridge"]["ks"]:.5f}。'
        controls=''.join(f'<label><input type="checkbox" checked data-prefix="{key}" data-curve="flight-{name}"> {html.escape(label)}</label>' for name,_,label,_,_ in lines)
        if i in gradients:controls+=''.join(f'<label><input type="checkbox" checked data-prefix="{key}" data-curve="gradient-{name}"> 梯度 {LABELS[name]}</label>' for name in ['conditional','bridge','reference'])
        cards.append(f'<article id="{key}"><h2>{key} · {title}</h2><p>记录 {i}，来源 {r["source"]}，像素 {r["pixel"]}，第 {r["bounce"]+1} 次碰撞。'
            f'同一起点和方向下，conditional 与 GP 首次碰撞参考的最大 CDF 差约 {100*s["maximum_cdf_difference"]:.3f} 个百分点。{gradtext}</p>'
            f'<div class="toggles">{controls}</div>{inline_svg(folder/f"{key}.svg",key)}</article>')
        rows.append([f'<a href="#{key}">{key}</a>',str(r['pixel']),r['bounce']+1,f'{s["distance"]:.6f}',
            f'{100*s["conditional_collision_cdf"]:.2f}%',f'{100*s["reference_collision_cdf"]:.2f}%'])
        # Avoid duplicating all profile arrays in the compact report metadata.
        entry['record']={k:v for k,v in r.items() if k not in ['ages','flight_mean','flight_mean_derivative','hazard','scalar_statistics','conditional_cdf']}
        cases.append(entry)
    checks=check_gradients(folder)
    coarse={r['id']:r for r in g.read(folder/'adaptive-profiles.json')['records']}
    integral_grid_error=max(float(np.max(abs(np.asarray(r['conditional_cdf'])[::4]-coarse[r['id']]['conditional_cdf'])))
        for r in g.read(folder/'refined-profiles.json')['records'])
    validation=dict(birth_moment_mean_max_error=max(v['checks']['mean_max_error'] for v in stats.values()),
        birth_moment_variance_relative_max_error=max(v['checks']['variance_relative_error'] for v in stats.values()),
        maximum_nested_reference_grid_cdf_change=max(max(v['grid_cdf_change']) for v in stats.values()),
        maximum_reference_standard_error_at_selected_distance=max(v['reference_standard_error_at_max'] for v in stats.values()),
        maximum_adaptive_hazard_integral_error_estimate=max(v['adaptive_quadrature_error_estimate'] for v in stats.values()),
        adaptive_hazard_cdf_grid_refinement_max_difference=integral_grid_error,
        gradient_production_checks=checks,
        maximum_gradient_endpoint_moment_error=max(v['reference']['checks']['endpoint_error'] for v in gradients.values()),
        maximum_gradient_reference_grid_cdf_change=max(v['reference']['grid_ks_medium_vs_fine'] for v in gradients.values()),
        note='Quadrature and grid checks are numerical convergence evidence, not a proof of continuum exactness.')
    validation['pass']=bool(validation['birth_moment_mean_max_error']<1e-12 and
        validation['birth_moment_variance_relative_max_error']<1e-11 and
        validation['maximum_nested_reference_grid_cdf_change']<1e-4 and
        integral_grid_error<1e-7 and
        validation['maximum_gradient_reference_grid_cdf_change']<1e-4 and
        validation['maximum_gradient_endpoint_moment_error']<1e-8 and
        max(c['cdf_error'] for c in checks)<1e-7)
    if not validation['pass']:raise AssertionError(validation)
    provenance={}
    for path in [Path(spec['scene']),Path('tools/surface_compare/gradient_rays.cpp'),
        Path('tools/surface_compare/groove_flights.py'),Path('tools/surface_compare/groove_flights_report.py'),Path('build/Release/fp_gradient_rays.exe')]:
        provenance[str(path)]=hashlib.sha256((g.ROOT/path).read_bytes()).hexdigest()
    g.write(folder/'validation.json',validation)
    g.write(folder/'report-data.json',dict(cases=cases,selection=spec['selection'],validation=validation,provenance=provenance))
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball 环形凹槽 · conditional / bridge / GP reference 光线诊断</title>
<style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#eef1f6;color:#202a38;margin:0}main{max-width:1250px;margin:auto;padding:28px}article,.intro{background:white;border-radius:12px;padding:24px;margin:24px 0}h1{font-size:28px}h2{font-size:22px}p{max-width:1100px}a{color:#236cbb}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border-bottom:1px solid #dce1e8;text-align:left}svg{width:100%;height:auto}.toggles{display:flex;flex-wrap:wrap;gap:8px 20px;font-size:13px}summary{cursor:pointer}code{background:#edf0f4;padding:2px 5px}.map{width:min(100%,530px)}.note{color:#576477;font-size:14px}.callout{border-left:4px solid #217db0;padding-left:18px}</style><main>
<h1>Shaderball 内嵌球环形凹槽：global_conditional / bridge / GP reference</h1>
<p>同一 procedural knob 几何、l=0.2、s=0.02、256×256 相机。这里比较实际光线的局部分布；背景图仅用于定位，来自已有 conditional 渲染。</p>
<div class="intro"><p class="callout"><b>以 GP reference 为参考：</b>比较同一起点与方向下的首次碰撞分布，以及固定碰撞位置下的 conditional / bridge 梯度分布。入口光线的条件碰撞率缺少完整“此前从未碰撞”的约束；部分反射后短光线中，conditional 已接近 GP reference。</p>
<p><b>GP reference 的含义：</b>基于 SE GP，条件化于当前段的起点值与梯度，并检查当前段此前从未碰撞；不保留更早反弹的整条路径历史。参考仍含有限特征展开、网格与数值采样误差，其收敛检查见下方验证数据。</p>
<p><b>入口 A/B：</b>global_conditional 的首次碰撞 CDF 偏高约 11 / 8.7 个百分点。该光线先经过一处接近零的外壁，再到内球，场值在这两个位置仍相关。<code>P(F(t)&gt;0 | C0)</code> 包含“以前穿过表面、后来又回到正值”的场，不等于整段存活概率。</p>
<p><b>梯度：</b>固定碰撞点后，B 的梯度 KS 从约 0.169 降到 0.0183；A 从 0.539 降到 0.364，仍有较大剩余误差。bridge 可以改善法线，但它不改同一出生条件下的自由程，不能消除 A/B 的碰撞位置偏差。</p>
<p><b>短光线 C/D：</b>conditional 的首次碰撞 CDF 与 GP reference 相差不足 0.03 个百分点，作为入口 A/B 的对照。C/D 的加密参考绘制当前段前 0.06，参考样本在该范围内均已碰撞。</p>
<img class="map" src="locations.png" alt="环形凹槽的相机像素位置"><p class="note">像素采用左上角为 (0,0)。次级光线的位置由实际记录的路径投影给出，不把整个像素的亮度归因于单条光线。</p>'''
    doc+=table(['光线','相机像素','碰撞序号','比较距离','conditional CDF','GP reference CDF'],rows)
    doc+='''<p class="note">每行距离取 conditional 与 GP reference 的 CDF 差最大的采样位置。GP reference 的采样误差与网格收敛检查见验证数据；CDF 差不是图像亮度误差。</p></div>'''
    doc+=''.join(cards)
    doc+='''<article><h2>conditional 与 GP reference 的区别</h2>
<p>目前 <code>ConditionalFlightKernel::evaluate()</code> 使用 <code>h(t)=p(F(t)=0|C0) E[(-F′(t))₊|C0,F(t)=0] / P(F(t)&gt;0|C0)</code>，再由 <code>exp(-∫h)</code> 得透射率。GP reference 以 <code>F(u)&gt;0, 0&lt;u&lt;t</code> 定义当前段存活，并寻找同一个条件场的首次过零。这会同时改变碰撞位置和碰撞梯度的分布。</p>
<h2>实验范围与复现</h2><p>从已有生产相机、conditional / bridge 延续路径中，按内球与挖空球交界的空间条件找出 425 条入口候选和 1674 条次级候选，按空间顺序抽取 24+120 条筛查。代表图含差异较大的例子及对照，不估计整幅图的平均偏差。</p>
<p>每条自由程参考使用 4 组独立 scramble、每组 32768 个 SE 特征样本；直接检查中间节点的正值前缀。固定梯度实验使用 4×16384 个 GP bridge 样本并解析积分终点通量；这里的 bridge 样本是参考积分所用的条件 GP 样本，与待评估的双 anchor 梯度修正不同。出生条件、方向与几何保持相同；不额外条件化更早的整条路径。</p>
<p>首次相机段也固定其记录的正值和梯度，以保持对照一致；起点在物体外足够远。次级光线来自诊断路径，不能视为原始图像随机序列的逐路径回放。没有进行完整多次反射、NEE 和最终辐射亮度的因果分解。本次没有修改生产渲染模式。</p>
<p><a href="validation.json">数值验证</a> · <a href="report-data.json">指标、出生状态与文件哈希</a> · <a href="spec.json">144 条光线规格</a> · <a href="refined-spec.json">精细参考规格</a> · <a href="short-spec.json">短光线加密规格</a></p>
<pre>cmake --build build --config Release --target fp_gradient_rays --parallel 4
python -B tools/surface_compare/groove_flights.py prepare
build/Release/fp_gradient_rays.exe flights OUTPUT/refined-spec.json OUTPUT/refined-profiles.json
python -B tools/surface_compare/groove_flights.py analyze --profiles refined-profiles.json --summary refined.json --power 15 --replicates 4
python -B tools/surface_compare/groove_flights_report.py</pre>
<p class="note">默认 OUTPUT 为 outputs/shaderball-groove-flight-diagnostic-20261108。参考的网格、积分及 QR / 生产公式一致性检查均记录在 validation.json。</p></article></main>
<script>document.querySelectorAll('input[data-curve]').forEach(el=>el.addEventListener('change',()=>{const group=document.getElementById(el.dataset.prefix+'_'+el.dataset.curve);if(group)group.style.display=el.checked?'':'none';}));</script></html>'''
    (folder/'comparison.html').write_text(doc,encoding='utf-8')
    print('Report',folder/'comparison.html');print('Validation',validation)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=f.DEFAULT)
    main(p.parse_args().out)
