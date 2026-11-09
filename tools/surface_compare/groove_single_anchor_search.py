"""Search the frozen shaderball path pool for single-anchor groove improvements.

Selection is geometric and precedes evaluation. This is a fixed-event gradient
diagnostic, not a measurement of rendered image error or path frequency.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

import gradient_rays as g
import numpy as np

ROOT = g.ROOT
SOURCE = ROOT / 'outputs/shaderball-gradient-ray-diagnostic-20261108/expanded/rays.json'
DEFAULT = ROOT / 'outputs/single-anchor-groove-search-20261108'
CASES = [
    ('G1',4567,'凹槽碰撞：接近参考分布'),
    ('G2',2850,'凹槽碰撞：另一侧的改善'),
    ('G3',4785,'凹槽碰撞：较好的近似'),
    ('G4',10667,'二次碰撞：明显改善但仍有偏差'),
    ('G5',2482,'从凹槽出发：第三次碰撞'),
    ('G6',2647,'凹槽碰撞：低误差案例'),
    ('F1',4876,'原报告 G-A：未获明显改善的对照'),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def groove_score(position):
    q = np.asarray(position) - [0., .8, 0.]
    return float(abs(np.linalg.norm(q) - .6) + abs(np.linalg.norm(q - [0., .4, .4]) - .56))


def export(spec, target):
    with target.with_suffix('.log').open('w', encoding='utf-8') as log:
        subprocess.run([str(ROOT / 'build/Release/fp_gradient_rays.exe'), 'detail',
                        str(spec), str(target)], cwd=ROOT, stdout=log,
                       stderr=subprocess.STDOUT, check=True)


def prepare(folder):
    data = g.read(SOURCE)
    chosen, seen = [], set()
    counts = dict(pool=len(data['records']), duplicate=0, hit_in_groove=0,
                  birth_in_groove=0, both=0)
    for original in data['records']:
        hit_score = groove_score(original['position'])
        birth_score = groove_score(original['birth_position'])
        hit = hit_score <= .08 and original['position'][1] >= .6
        birth = (original['bounce'] > 0 and birth_score <= .08
                 and original['birth_position'][1] >= .6)
        if not (hit or birth):
            continue
        signature = tuple(original['birth_position'] + original['wi'] +
                          original['birth_gradient'] + [original['f0'], original['t']])
        if signature in seen:
            counts['duplicate'] += 1
            continue
        seen.add(signature)
        r = copy.deepcopy(original)
        r['old_production'] = r.pop('production')
        r['groove'] = dict(hit=hit, birth=birth, hit_score=hit_score,
                           birth_score=birth_score)
        r['query_derivatives'] = []
        chosen.append(r)
        counts['hit_in_groove'] += int(hit)
        counts['birth_in_groove'] += int(birth)
        counts['both'] += int(hit and birth)
    counts['selected'] = len(chosen)
    folder.mkdir(parents=True, exist_ok=True)
    selection = dict(counts=counts, source=str(SOURCE.relative_to(ROOT)),
        source_sha256=digest(SOURCE), original_scan_spec=data['spec'],
        criterion='sum of absolute distances to inner sphere (center [0,.8,0], radius .6) '
                  'and cutout sphere (center [0,1.2,.4], radius .56) <= .08, y >= .6; '
                  'either the current collision or a secondary flight birth qualifies',
        scope='Frozen old conditional/2-anchor paths, new law evaluated at identical events; '
              'results describe selected cases, not unbiased image-wide frequencies.')
    g.write(folder/'selection.json', selection)
    g.write(folder/'detail-input.json', dict(scene=data['spec']['scene'],
            records=chosen, counts=counts, anchor_selection='hermite'))
    print('Geometric selection:', counts, flush=True)
    export(folder/'detail-input.json', folder/'rays.json')
    print('New production parameters exported', flush=True)


def summarize(folder):
    screen = g.read(folder/'screen.json')
    records = {r['id']:r for r in g.read(folder/'rays.json')['records']}
    rows = []
    for row in screen['records']:
        if 'errors' in row:
            rows.append(dict(row, groove=records[row['id']]['groove']))
    for title, subset in [('hit', [r for r in rows if r['groove']['hit']]),
                           ('depart', [r for r in rows if r['groove']['birth']]),
                           ('all', rows)]:
        ranked = sorted(subset, key=lambda r:-r['gain_vs_conditional'])
        print(title, 'referenced', len(ranked), 'positive gain',
              sum(r['gain_vs_conditional'] > .005 for r in ranked), flush=True)
        for row in ranked[:12]:
            print(row['id'], row['pixel'], 'hit',row['bounce']+1,
                  'age',round(row['normalized_age'],3), 'errors',row['errors'],
                  'grid',row['grid_ks'], 'rep',row['replicate_ks'], row['groove'], flush=True)
    g.write(folder/'ranked.json', sorted(rows,key=lambda r:-r['gain_vs_conditional']))


def prepare_refine(folder):
    data = g.read(folder/'rays.json')
    by_id = {r['id']:r for r in data['records']}
    records = []
    for key, identity, title in CASES:
        r = copy.deepcopy(by_id[identity])
        r['comparison_key'] = key
        laws = g.models(r)
        old = r['old_production']
        laws['old'] = g.Density(old['mu'],old['sd'],old['p'],old['q'],True)
        grid = g.grid_for(laws, size=2049)
        r['query_derivatives'] = grid[np.linspace(0,len(grid)-1,129,dtype=int)].tolist()
        records.append(r)
    detailed = folder/'refined'
    detailed.mkdir(exist_ok=True)
    g.write(detailed/'detail-input.json', dict(scene=data['scene'],records=records,
                                              profile_denominator=6656,anchor_selection='hermite'))
    export(detailed/'detail-input.json', detailed/'rays.json')
    print('Prepared detailed candidates:', [r['id'] for r in records], flush=True)


def plot_case(folder, key, row, curves, laws, background, scene):
    import matplotlib.pyplot as plt
    from gradient_rays_report import projected
    from single_anchor_update import COLORS, LABELS, MARKERS, STYLES
    r = row['record']; scale = r['sigma']*r['gamma']; v = curves['v']
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    ax = axes[0,0]
    ax.imshow(background, extent=(-.5,255.5,255.5,-.5))
    path = projected(r['vertices'][1:],scene)
    ax.plot(*path.T,color='white',lw=1.2,marker='.',ms=5)
    segment = projected([r['birth_position'],r['position']],scene)
    ax.plot(*segment.T,color='#ee5265',lw=2.2,marker='o',ms=5)
    if np.linalg.norm(segment[1]-segment[0]) > 5:
        ax.annotate('',xy=segment[1],xytext=segment[0],
                    arrowprops=dict(arrowstyle='->',color='#ee5265',lw=2))
        ax.annotate('birth / hit #'+str(r['bounce']),segment[0],xytext=(7,-18),
                    textcoords='offset points',fontsize=8,color='#9c1530',
                    bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax.plot(*r['pixel'],marker='o',ms=11,mfc='none',mec='#ffeb63',mew=1.8)
    ax.annotate('selected hit #'+str(r['bounce']+1),segment[1],xytext=(7,12),
                textcoords='offset points',fontsize=8,color='#9c1530',
                bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax.text(.02,.02,'Yellow ring: source camera pixel\nRed arrow: current flight',
            transform=ax.transAxes,fontsize=8,bbox=dict(facecolor='white',alpha=.9,edgecolor='none'))
    ax.set(xlim=(-.5,255.5),ylim=(255.5,-.5),xlabel='camera pixel x',ylabel='camera pixel y',
           title=f"{key}: pixel {tuple(r['pixel'])}, sample {r['sample']}, hit #{r['bounce']+1}")
    ax = axes[0,1]
    x = np.linspace(0,r['t'],len(r['mean_profile']))*r['gamma']
    ax.plot(x,np.asarray(r['mean_profile'])/r['sigma'],c='#858b96',label='actual shaderball SDF / sigma')
    x=curves['s']*r['gamma']; mean=curves['bridge_mean']/r['sigma']; sd=curves['bridge_sd']/r['sigma']
    ax.plot(x,mean,c='#337c99',label='GP mean | birth jet, F(t)=0')
    ax.fill_between(x,mean-2*sd,mean+2*sd,color='#337c99',alpha=.18,label='+/- 2 conditional std. dev.')
    for j,s in enumerate([r['t']/3,max(2*r['t']/3,r['t']-1/r['gamma'])]):
        ax.axvline(s*r['gamma'],c=COLORS['old'],ls=':',lw=1.3,label='old anchors' if j==0 else None)
    ax.axvline(r['production']['anchor_distance']*r['gamma'],c=COLORS['single'],ls='--',lw=2,label='new anchor')
    ax.axhline(0,c='#172131',lw=.8)
    ax.set(xlabel='distance / ell',ylabel='field value / sigma',ylim=(-8,12),title='Prefix profile (before positivity test)')
    ax.legend(fontsize=8)
    summaries = [law.summary() for law in laws.values()]+[row['reference']['summary']]
    low=min(s['quantiles'][0]-2.5*s['sd'] for s in summaries)/scale
    high=min(0,max(s['quantiles'][2]+2.5*s['sd'] for s in summaries))/scale
    for col,suffix in enumerate(['pdf','cdf']):
        ax=axes[1,col]
        for j,name in enumerate(['reference','old','single','conditional','classic']):
            ax.plot(v/scale,curves[name+'_'+suffix]*(scale if suffix=='pdf' else 1),
                color=COLORS[name],ls=STYLES[name],lw=2.5 if name=='reference' else 1.7,
                label=LABELS[name],marker=MARKERS[name],markevery=(.035+.05*j,.27),
                ms=4,mfc='white',gid=f'curve-{suffix}-{name}')
        ax.set(xlim=(low,high),xlabel='axial gradient v / (sigma gamma)',
               ylabel='density' if suffix=='pdf' else 'CDF',title='Collision gradient '+suffix.upper())
        if suffix=='cdf':ax.set_ylim(-.02,1.02)
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(folder/(key+'.png'),dpi=160);fig.savefig(folder/(key+'.svg'));plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,3.6),layout='constrained')
    for ax,names,title in [(axes[0],['classic','conditional'],'Base laws minus GP reference'),
                           (axes[1],['old','single'],'Anchor corrections minus GP reference')]:
        for name in names:
            ax.plot(v/scale,curves[name+'_cdf']-curves['reference_cdf'],
                    c=COLORS[name],ls=STYLES[name],label=LABELS[name])
        ax.axhline(0,c='#777',lw=.7)
        ax.set(xlim=(low,high),xlabel='axial gradient v / (sigma gamma)',ylabel='CDF difference',title=title)
        ax.legend(fontsize=8);ax.grid(alpha=.15)
    fig.savefig(folder/(key+'-difference.png'),dpi=160);plt.close(fig)


def report(folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from gradient_rays_report import inline_svg
    from single_anchor_update import check, COLORS, LABELS
    selection=g.read(folder/'selection.json');screen=g.read(folder/'screen.json')
    frozen={r['id']:r for r in g.read(folder/'detail-input.json')['records']}
    current=g.read(folder/'rays.json')['records']
    assert len(current)==len(frozen)
    for r in current:
        for field in ['birth_position','position','birth_gradient','wi','f0','g0','t',
                      'sigma','gamma','mean_profile','old_production']:
            assert r[field]==frozen[r['id']][field],(r['id'],field)
        assert r['production']['approximation']=='single_anchor_v2'
    scene_path=ROOT/selection['original_scan_spec']['scene'];scene=g.read(scene_path)
    background_path=scene_path.parent.parent/'baseline/shaderball.png'
    shutil.copyfile(background_path,folder/'shaderball.png')
    background=plt.imread(background_path)
    results=[];cards=[];table=[];checks=[]
    for key,identity,title in CASES:
        row=g.read(folder/'refined'/f'ray-{identity}.json');r=row['record']
        law,validation=check(r)
        curves=dict(np.load(folder/'refined'/f'ray-{identity}-curves.npz'))
        laws=g.models(r);laws['single']=laws.pop('bridge')
        p=r['old_production'];laws['old']=g.Density(p['mu'],p['sd'],p['p'],p['q'],True)
        for name in ['single','old']:
            for suffix in ['pdf','cdf']:curves[name+'_'+suffix]=getattr(laws[name],suffix)(curves['v'])
        for name in laws:
            assert laws[name].cdf(curves['v'][0])<1e-7 and laws[name].cdf(curves['v'][-1])>1-1e-7,(key,name)
        errors={name:float(np.max(abs(curves[name+'_cdf']-curves['reference_cdf']))) for name in laws}
        delta=errors['conditional']-errors['single']
        old_gain=errors['old']-errors['single']
        ref=row['reference'];grid_error=ref['grid_ks_medium_vs_fine'];se=ref['max_pointwise_cdf_se']
        # These bounds are empirical convergence checks, not continuum proofs.
        convergence=bool(grid_error<1e-4 and se<1e-4)
        confirmed=bool(delta>max(.005,10*grid_error,10*se) and
                       row['gains']['conditional']['value']>row['gains']['conditional']['ci95_halfwidth'])
        validation.update(reference_grid_ks=grid_error,reference_cdf_se=se,
                          reference_converged=convergence,improvement_confirmed=confirmed)
        validation['pass']=bool(validation['pass'] and convergence)
        checks.append(validation)
        snr=curves['bridge_mean']/curves['bridge_sd'];minimum=int(np.argmin(snr))
        bottleneck=dict(distance_in_ell=float(curves['s'][minimum]*r['gamma']),
                        marginal_mean_over_sd=float(snr[minimum]),
                        anchor_distance_in_ell=r['production']['anchor_distance']*r['gamma'])
        record=dict(key=key,title=title,id=identity,pixel=r['pixel'],sample=r['sample'],hit=r['bounce']+1,
            groove=r['groove'],errors=errors,gain_vs_conditional=delta,gain_vs_old=old_gain,
            relative_reduction=delta/errors['conditional'] if errors['conditional'] else 0,
            production_check=validation,reference=ref,record=r,anchor_diagnostic=bottleneck,
            gain_ci95_halfwidth=row['gains']['conditional']['ci95_halfwidth'])
        results.append(record)
        np.savez_compressed(folder/(key+'-curves.npz'),**curves)
        plot_case(folder,key,row,curves,laws,background,scene)
        location='碰撞点在凹槽区域' if r['groove']['hit'] else '起点在凹槽，碰撞点已离开该区域'
        table.append(f'<tr><td><a href="#{key}">{key}</a></td><td>{r["pixel"]}</td><td>{r["sample"]}</td>'
                     f'<td>{r["bounce"]+1}</td>'+''.join(f'<td>{errors[n]:.6f}</td>' for n in ['classic','conditional','old','single'])+
                     f'<td>{delta:+.6f}</td><td>{old_gain:+.6f}</td></tr>')
        controls=''.join(f'<label><input type="checkbox" checked data-case="{key}" data-curve="{name}">{LABELS[name]}</label>' for name in COLORS)
        cards.append(f'<article id="{key}"><h2>{key} · {title}</h2><p>像素 {r["pixel"]}，sample {r["sample"]}，'
            f'第 {r["bounce"]+1} 次碰撞；{location}。固定距离 {r["t"]*r["gamma"]:.4f} ℓ，'
            f'新锚点 {r["production"]["anchor_distance"]*r["gamma"]:.4f} ℓ。</p>'
            f'<p>相对 global_conditional，KS 减少 {delta:.6f}（{100*record["relative_reduction"]:.1f}%）；'
            f'相对原双锚点，KS 减少 {old_gain:+.6f}。正数表示改善，负数表示变差。</p>'
            f'<p>正值检验前，条件均值/标准差的最低点位于 {bottleneck["distance_in_ell"]:.4f} ℓ；'
            f'新锚点位于 {bottleneck["anchor_distance_in_ell"]:.4f} ℓ。该位置比较帮助解释约束是否被捕捉，不代替完整 GP 前缀检验。</p>'
            f'<p>参考网格加密的最大 CDF 变化 {grid_error:.2g}；8 次随机重复的最大逐点标准误 {se:.2g}；'
            f'KS 改善量的 95% 重复采样区间半宽 {record["gain_ci95_halfwidth"]:.2g}。</p>'
            f'<div class="toggles">{controls}</div>{inline_svg(folder/(key+".svg"),key)}'
            f'<details><summary>CDF 差值图</summary><img src="{key}-difference.png" alt="CDF difference"></details>'
            f'<p><a href="{key}.png">PNG</a> · <a href="{key}.svg">SVG</a> · '
            f'<a href="{key}-curves.npz">曲线数据</a> · <a href="refined/ray-{identity}.json">完整光线与参考记录</a></p></article>')
        print(key,'errors',errors,'confirmed',confirmed,'grid',grid_error,'se',se,flush=True)
    counts=selection['counts'];tested=sum('errors' in r for r in screen['records'])
    refined_improvements=sum(r['production_check']['improvement_confirmed'] for r in results)
    validation=dict(all_pass=all(c['pass'] for c in checks),checks=checks,
        geometric_counts=counts,screen_reference_count=tested,screen_failures=screen['failures'],
        frozen_inputs_checked=len(current),unchanged_birth_direction_distance_sdf=True,
        refined_improvements=refined_improvements,production_executable_sha256=digest(ROOT/'build/Release/fp_gradient_rays.exe'),
        scene_sha256=digest(scene_path),source_sha256=selection['source_sha256'],
        reference_note='SE feature QR with nested prefix grids and 8 scrambled Sobol replicates of 65536 draws; '
                       'each case records the actual grid sizes; convergence is empirical, not a proof of continuum exactness.')
    g.write(folder/'validation.json',validation)
    g.write(folder/'report-data.json',dict(cases=results,selection=selection,validation=validation))
    document='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>凹槽光线：单锚点有效案例</title>
<style>body{max-width:1440px;margin:30px auto;padding:0 22px;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#172131;background:#f6f8fa}article,.intro{background:white;padding:24px;margin:24px 0;border:1px solid #dde3e9;border-radius:12px}a{color:#006b70}svg,img{max-width:100%;height:auto}table{width:100%;border-collapse:collapse;font-size:14px}th,td{padding:9px;text-align:right;border-bottom:1px solid #dde3e9}th:first-child,td:first-child{text-align:left}.toggles{display:flex;gap:18px;flex-wrap:wrap}label{cursor:pointer}h1{font-size:28px}details{margin:12px 0}.scroll{overflow-x:auto}</style>
<h1>Shaderball 凹槽：单锚点能够改善的光线</h1>
<p><a href="../single-anchor-shaderball-l0p2-s0p02-256-128spp-20261108/rays/comparison.html">原 13 条固定光线报告</a> · <a href="validation.json">验证记录</a> · <a href="report-data.json">完整结果</a> · <a href="ranked.json">全部初筛排名</a></p><div class="intro">'''
    document+=f'<p><strong>检测到 {refined_improvements} 条经加密验证的改善案例。</strong>同一配置 ℓ=0.2、σ=0.02。G1/G2/G3/G6 '
    document+='在凹槽处的梯度分布明显更接近 GP reference；G4 有明显改善但仍有偏差；G5 是从凹槽出发后的碰撞。F1 保留原报告 G-A 作为未改善对照。</p>'
    document+=f'<p>从原来的 {counts["pool"]:,} 个碰撞记录中，按三维几何筛出并去重得到 {counts["selected"]:,} 段：'
    document+=f'碰撞点位于凹槽 {counts["hit_in_groove"]:,} 段，起点位于凹槽 {counts["birth_in_groove"]:,} 段，二者重叠 {counts["both"]:,} 段。'
    document+='几何条件沿用旧报告：到半径 0.6 的内球与半径 0.56 的切除球表面的距离绝对值之和 ≤0.08，且 y≥0.6。</p>'
    document+=f'<p>先计算单锚点与 global_conditional 的 CDF 差；有 {tested} 条达到 0.001，进一步计算粗 GP reference。'
    document+=f'其余 {counts["selected"]-tested:,} 条修正幅度低于此阈值。候选按改善幅度和最终拟合误差选择，之后独立加密验证。'
    document+='这是有意挑选的有效案例，不代表凹槽区域的平均改善比例。</p>'
    document+='<p>所有曲线固定同一出生值/完整梯度、方向、碰撞距离及真实 SDF。GP reference 检查当前段的正值前缀。'
    document+='这里衡量的是碰撞处的轴向梯度分布；不能由此直接推断自由程、透射率或整张渲染图的误差。旧双锚点使用同一记录中保存的生产参数。</p>'
    document+='<p>黄圈是来源相机像素；红线是当前段，箭头指向被比较的碰撞；标题 hit #n 给出碰撞序号。第一段投影重合时红线会缩为一个点。'
    document+='曲线可单独开关，重叠时也可展开 CDF 差值图。KS=max|CDF−CDF_reference|，越小越好。</p>'
    document+='<div class="scroll"><table><tr><th>案例</th><th>像素</th><th>sample</th><th>碰撞</th><th>classic KS</th><th>conditional KS</th><th>双锚点 KS</th><th>单锚点 KS</th><th>对 conditional 改善</th><th>对双锚点改善</th></tr>'
    document+=''.join(table)+'</table></div></div>'+''.join(cards)
    document+='''<script>document.querySelectorAll('input[data-curve]').forEach(box=>box.addEventListener('change',()=>{for(const suffix of ['pdf','cdf']){const line=document.getElementById(box.dataset.case+'_curve-'+suffix+'-'+box.dataset.curve);if(line)line.style.display=box.checked?'':'none';}}));</script></html>'''
    (folder/'comparison.html').write_text(document,encoding='utf-8')
    if not validation['all_pass']:
        raise RuntimeError('Candidate validation or reference convergence failed; inspect validation.json')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare','summarize','prepare-refine','report'])
    parser.add_argument('--directory', type=Path, default=DEFAULT)
    args = parser.parse_args()
    {'prepare':prepare, 'summarize':summarize,
     'prepare-refine':prepare_refine, 'report':report}[args.action](args.directory.resolve())


if __name__ == '__main__':
    main()
