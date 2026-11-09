"""Audit the current fixed-five gradient law on all previously refined rays.

Reuse GP references and frozen collision states. This is not a new scene-wide
search, a new rendering mode, or a refinement of the five production anchors.
"""
from pathlib import Path
import argparse
import copy
import hashlib
import html
import sys
import gradient_rays as g
import single_anchor_update as single
import fixed_anchor_update as fixed
import numpy as np
from gradient_rays_report import inline_svg

ROOT=g.ROOT
OLD=ROOT/'outputs/single-anchor-shaderball-l0p2-s0p02-256-128spp-20261108/rays'
RECENT=ROOT/'outputs/fixed5-anchor-validation-20261109/rays'
GROOVE=ROOT/'outputs/single-anchor-groove-search-20261108'
FLIGHTS=ROOT/'outputs/shaderball-groove-flight-diagnostic-20261108'
OUT=ROOT/'outputs/fixed5-anchor-residual-diagnostic-20261109'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder):
    folder.mkdir(parents=True,exist_ok=True)
    spec=g.read(OLD/'production-detail.json')
    spec['anchor_selection']='fixed5_snr'
    g.write(folder/'detail-input.json',spec)


def report(folder):
    prior={c['key']:c for c in g.read(OLD/'source-cases.json')['cases']}
    hermite={r['comparison_key']:r for r in g.read(OLD/'production-detail.json')['records']}
    current={r['comparison_key']:r for r in g.read(folder/'production-detail.json')['records']}
    cases=[]
    for key,r in current.items():
        if key=='G-A':
            continue # Same ray as the recent groove F1; use its finer reference.
        name='F-general' if key=='F1' else key
        r=copy.deepcopy(r);r['comparison_key']=name
        cases.append(dict(key=name,source_key=key,record=r,hermite=hermite[key],
            reference=prior[key]['old']['reference'],curves=OLD/(key+'-old-curves.npz'),
            source='original 13-ray report'))
    recent={r['comparison_key']:r for r in g.read(RECENT/'fixed5_snr.json')['records']}
    previous={c['key']:c for c in g.read(GROOVE/'report-data.json')['cases']}
    for row in g.read(RECENT/'report-data.json')['cases']:
        key=row['key']
        cases.append(dict(key=key,source_key=key,record=recent[key],hermite=previous[key]['record'],
            reference=row['reference'],curves=GROOVE/(key+'-curves.npz'),source='recent 7-ray groove report'))
    assert len(cases)==19
    identities=[tuple(c['record']['birth_position'])+tuple(c['record']['wi'])+
                (c['record']['t'],c['record']['f0'],c['record']['g0']) for c in cases]
    assert len(set(identities))==19
    rows=[];checks=[];plot_inputs={}
    for case in cases:
        key=case['key'];r=case['record'];h=case['hermite'];reference=case['reference']
        for field in ('birth_position','birth_gradient','f0','g0','wi','t','sigma','gamma','mean_profile'):
            assert r[field]==h[field],(key,field)
        law,check=single.check(r)
        base=g.models(r);p=h['production']
        models=dict(fixed5=law,conditional=base['conditional'],classic=base['classic'],
                    hermite=g.Density(p['mu'],p['sd'],p['p'],p['q'],True))
        z=np.load(case['curves'])
        curves={name:z[name] for name in ('v','reference_pdf','reference_cdf')}
        for name,model in models.items():
            for suffix in ('pdf','cdf'):curves[name+'_'+suffix]=getattr(model,suffix)(curves['v'])
        errors={name:float(np.max(abs(curves[name+'_cdf']-curves['reference_cdf']))) for name in models}
        grid_error=reference.get('grid_ks_medium_vs_fine',reference.get('grid_ks_207_vs_415'))
        se=reference['max_pointwise_cdf_se']
        row=dict(key=key,source_key=case['source_key'],source=case['source'],id=r['id'],pixel=r['pixel'],
            sample=r['sample'],hit=r['bounce']+1,errors=errors,mean_gradient=law.mean,
            reference_mean_gradient=reference['summary']['mean'],
            reference_grid_change=grid_error,reference_cdf_standard_error=se,
            residual_exceeds_10x_reference_checks=errors['fixed5']>10*max(grid_error,se),
            age_in_ell=r['t']*r['gamma'],anchor_in_ell=r['production']['anchor_distance']*r['gamma'],
            production=r['production'],reference=reference,record=r,check=check,
            source_curves=str(case['curves'].relative_to(ROOT)),source_curves_sha256=digest(case['curves']))
        rows.append(row);checks.append(check)
        np.savez_compressed(folder/(key+'-curves.npz'),**curves)
        plot_inputs[key]=(dict(key=key,record=h,reference=reference),r,curves,models)
    rows.sort(key=lambda c:-c['errors']['fixed5'])
    assert all(c['pass'] for c in checks)
    flights=[]
    for case in g.read(FLIGHTS/'report-data.json')['cases']:
        f=case['flight']
        flights.append(dict(key=case['key'],id=case['id'],pixel=f['pixel'],
            distance_at_max_cdf_difference=f['distance'],max_collision_cdf_difference=f['maximum_cdf_difference'],
            conditional_collision_cdf=f['conditional_collision_cdf'],reference_collision_cdf=f['reference_collision_cdf'],
            conditional_survival=1-f['conditional_collision_cdf'],reference_survival=1-f['reference_collision_cdf'],
            reference_standard_error=f['reference_standard_error_at_max'],
            reference_grid_changes=f['grid_cdf_change']))
    plotted=[r['key'] for r in rows[:6]]+['G4','F1']
    cards=[]
    for key in plotted:
        case,r,curves,models=plot_inputs[key]
        fixed.plot(folder,case,r,curves,models)
        row=next(c for c in rows if c['key']==key)
        controls=''.join(f'<label><input type="checkbox" checked data-case="{key}" data-curve="{name}">{html.escape(label)}</label>'
                         for name,label in fixed.LABELS.items())
        ref=row['reference']
        cards.append(f'<article id="{key}"><h2>{key} · id {r["id"]} · 像素 {r["pixel"]} · sample {r["sample"]} · hit #{r["bounce"]+1}</h2>'
            f'<p>固定 5 点 KS：<strong>{row["errors"]["fixed5"]:.6f}</strong>；conditional：{row["errors"]["conditional"]:.6f}；'
            f'Hermite：{row["errors"]["hermite"]:.6f}。当前梯度均值 {row["mean_gradient"]:.6f}；GP reference {row["reference_mean_gradient"]:.6f}。</p>'
            f'<p>参考约束点数 {ref["points"]}；末两级网格 CDF 变化 {row["reference_grid_change"]:.3g}；'
            f'重复采样最大逐点 CDF 标准误 {row["reference_cdf_standard_error"]:.3g}。</p>'
            f'<div class="toggles">{controls}</div>{inline_svg(folder/(key+".svg"),key)}'
            f'<p><a href="{key}.png">PNG</a> · <a href="{key}.svg">SVG</a> · <a href="{key}-curves.npz">曲线数据</a></p></article>')
    table=[]
    for row in rows:
        key=row['key'];label=f'<a href="#{key}">{key}</a>' if key in plotted else key
        table.append(f'<tr><td>{label}</td><td>{row["id"]}</td><td>{row["pixel"]}</td><td>{row["hit"]}</td>'+
            ''.join(f'<td>{row["errors"][name]:.6f}</td>' for name in ('classic','conditional','hermite','fixed5'))+
            f'<td>{row["reference_grid_change"]:.2g}</td><td>{row["reference_cdf_standard_error"]:.2g}</td></tr>')
    flight_table=''.join(f'<tr><td>{f["key"]}</td><td>{f["id"]}</td><td>{f["distance_at_max_cdf_difference"]:.6f}</td>'
        f'<td>{f["conditional_survival"]:.6f}</td><td>{f["reference_survival"]:.6f}</td><td>{f["max_collision_cdf_difference"]:.6f}</td></tr>'
        for f in flights)
    validation=dict(all_pass=all(c['pass'] for c in checks),unique_fixed_rays=19,checks=checks,
        candidate_count=5,production_unchanged=True,reference_reused=True,new_scene_wide_search=False,
        deduplication='Old G-A and recent groove F1 are id 4876; retain the recent finer reference. Old general F1 is id 914 and is named F-general here.',
        executable_sha256=digest(ROOT/'build/Release/fp_gradient_rays.exe'))
    g.write(folder/'validation.json',validation)
    g.write(folder/'report-data.json',dict(cases=rows,flights=flights,validation=validation))
    doc='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>固定 5 点：剩余梯度误差与自由程误差</title>
<style>body{max-width:1450px;margin:30px auto;padding:0 22px;font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;color:#172131;background:#f6f8fa}article,.intro{background:white;padding:24px;margin:24px 0;border:1px solid #dde3e9;border-radius:10px}a{color:#007f7c}svg,img{max-width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px 11px;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.scroll{overflow:auto}.toggles{display:flex;gap:16px;flex-wrap:wrap}label{cursor:pointer}</style>
<h1>固定 5 点之后，还剩哪些误差？</h1><p>Shaderball · ℓ=0.2 · σ=0.02 · 19 条已保存的不同光线</p>
<p><a href="report-data.json">完整结果</a> · <a href="validation.json">独立数值检查</a> · <a href="production-detail.json">新增 13 条 C++ 导出</a> · <a href="../fixed5-anchor-shaderball-l0p2-s0p02-256-128spp-20261109/comparison.html">256×256 / 128 spp 渲染图</a></p>
<div class="intro"><p><strong>自由程／透射率和碰撞梯度都仍可能贡献差异。</strong>同一出生状态、方向下，当前距离采样和阴影透射率仍采用原 classic 或 global_conditional 算法。fixed5_snr 只在已接受的碰撞位置改变归一化梯度分布。</p>
<p>global_conditional 已对出生值和梯度做条件化，但其 hazard 使用单点边缘统计；它还没有使用“整段此前始终为正”的首次穿零存活概率。梯度修正后的反射又会改变后续方向和出生状态，因此实际整条路径的差异还会累积。</p>
<p>下表的梯度比较固定了同一个出生状态和碰撞位置：这里剩余的 KS 就是梯度近似误差，不能归因于透射率。当前仍只选 1 个锚点，且只在末端窗口检查 5 个位置、在一个代表斜率 u₀ 处选择；这不能完整替代整个正值前缀事件。</p>
<p>本页重新评估旧 13 条光线，并合并最近 7 条凹槽光线，去重后为 19 条。不是全图随机统计，也不是新的大范围搜索。原普通光线报告的 <strong>F1（id 914，像素 [116,172]）</strong>在此改名 <strong>F-general</strong>；最近讨论的凹槽 <strong>F1（id 4876，像素 [138,130]）</strong>保留原名。</p>
<p>全部使用真实 SDF 的 C++ 固定 5 点导出和独立 QR 条件化检查。GP reference 沿用已保存的有限前缀参考；下表同时给出网格变化及随机标准误。它不是连续路径严格真值。</p></div>
<h2>梯度分布：按当前 KS 从大到小排列</h2><p>KS=max|CDF−CDF_reference|，越小越好。例如 0.159 表示某个梯度阈值处累积概率相差约 15.9 个百分点，不是图像误差百分比。</p>
<div class="scroll"><table><tr><th>光线</th><th>id</th><th>像素</th><th>hit</th><th>classic</th><th>conditional</th><th>Hermite</th><th>固定 5 点</th><th>参考网格变化</th><th>参考 CDF SE</th></tr>TABLE</table></div>
<h2>自由程／透射率：沿用同一出生状态的已有诊断</h2><p>在首次碰撞 CDF 差异最大的距离处，T=1−CDF。A 就是凹槽 F1（id 4876）；B 是 G-B（id 5032）。梯度选锚不会改变这些固定出生状态下的自由程曲线。</p>
<div class="scroll"><table><tr><th>光线</th><th>id</th><th>距离 s</th><th>conditional T(s)</th><th>GP reference T(s)</th><th>最大 CDF 差</th></tr>FLIGHTS</table></div>
<p><a href="../shaderball-groove-flight-diagnostic-20261108/comparison.html">原自由程报告</a>。这些证据说明两种误差都存在，不能据此量化整幅渲染中各自所占比例；128 spp 噪声以及原生 renewal+ 的场表示和步进近似也需要单独控制。</p>CARDS
<script>document.querySelectorAll('input[data-curve]').forEach(box=>box.addEventListener('change',()=>{for(const suffix of ['pdf','cdf']){const line=document.getElementById(box.dataset.case+'_curve-'+suffix+'-'+box.dataset.curve);if(line)line.style.display=box.checked?'':'none';}}));</script></html>'''
    doc=doc.replace('TABLE',''.join(table)).replace('FLIGHTS',flight_table).replace('CARDS',''.join(cards))
    (folder/'comparison.html').write_text(doc,encoding='utf-8')
    for row in rows:
        print(row['key'],row['id'],row['errors']['fixed5'],'reference checks',row['reference_grid_change'],row['reference_cdf_standard_error'],flush=True)


if __name__=='__main__':
    sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','report'))
    parser.add_argument('--output',type=Path,default=OUT)
    args=parser.parse_args()
    (prepare if args.action=='prepare' else report)(args.output.resolve())
