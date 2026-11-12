"""Build measured three-seed error/time curves; safe to rerun during rendering."""
from __future__ import annotations

import csv
from datetime import datetime, timezone
import html
import io
import json
import math
import os
from pathlib import Path
import sys

from gpis_shaderball_equal_quality import ROOT, MODES, load, save

OUT = Path(sys.argv[1]).resolve()
os.environ['MPLCONFIGDIR'] = str(OUT/'.matplotlib-cache')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

LABELS = dict(global_conditional='global_conditional', gpis_renewal_plus='GPIS renewal+ (1D MIS)')
COLORS = dict(global_conditional='#087e8b', gpis_renewal_plus='#d47722')
REGIONS = dict(full='全图', bottom='底部 ROI')


def rel(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(value, encoding='utf-8')
    temporary.replace(path)


def write_csv(path, fields, rows):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(path, buffer.getvalue())


def pct(value):
    return f'{value*100:.2f}%'


def seconds(value):
    return f'{value:.1f} s' if value<120 else f'{value/60:.2f} min'


def summarize(manifest, plan):
    output = dict(updated_utc=datetime.now(timezone.utc).isoformat(),
                  metric=manifest['metric'], timing=manifest['timing'],
                  aggregation='sqrt(mean of three independent per-seed MSEs / reference mean square); '
                              'time is mean cumulative duration of one run, not total cost of three runs; no image averaging',
                  crossing='First observed checkpoint with all three seeds present and aggregate error <= target; '
                           'no interpolation or extrapolation; may not remain below target at later checkpoints',
                  renderer_sha256=manifest['renderer_sha256'], plan=plan,
                  expected_records=len(plan['cases'])*len(plan['seeds'])*len(plan['modes'])*len(plan['points']),
                  observed_records=0, cases={})
    output['excluded_timing_attempts']=[dict(folder=rel(p.parent),reason=load(p))
        for p in sorted(OUT.rglob('reason.json')) if 'excluded_timing' in p.parts]
    observations, crossing_csv = [], []
    for case in plan['cases']:
        source = manifest['reference_sources'][case]
        data = dict(reference=source, modes={}, comparisons={})
        for mode in plan['modes']:
            runs = []
            for seed in plan['seeds']:
                for spp in plan['points']:
                    folder = OUT/'renders'/case/mode/f'seed{seed}'/f'spp{spp:04d}'
                    path = folder/'result.json'
                    if not path.exists():
                        continue
                    row = load(path)
                    assert (row['case'],row['mode'],row['seed'],row['spp'])==(case,mode,seed,spp)
                    assert row['renderer_sha256']==manifest['renderer_sha256']
                    assert row['reference_pfm_sha256']==source['pfm_sha256']
                    assert row['threads']==manifest['threads']
                    assert row['cumulative_process_seconds']>0
                    row.update(folder=rel(folder), image=rel(folder/'render.png'))
                    log=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
                    row['path_diagnostics']=dict(far_t_zero_lines=log.count('farT == 0'),
                        internal_error_lines=log.count('Caught an internal error'))
                    assert not row['path_diagnostics']['internal_error_lines']
                    for region in REGIONS:
                        metric=row['errors'][region]
                        assert all(math.isfinite(metric[k]) and metric[k]>=0 for k in ('mse','relative_rmse'))
                        observations.append(dict(case=case,mode=mode,seed=seed,spp=spp,region=region,
                            cumulative_process_seconds=row['cumulative_process_seconds'],
                            cumulative_render_seconds=row['cumulative_render_seconds'],
                            segment_process_seconds=row['process_seconds'], mse=metric['mse'],
                            relative_rmse=metric['relative_rmse'], folder=row['folder']))
                    runs.append(row)
            output['observed_records']+=len(runs)
            aggregate, crossings, seed_crossings = {}, {}, {}
            for region in REGIONS:
                agg=[]
                for spp in plan['points']:
                    same=[r for r in runs if r['spp']==spp]
                    if len(same)!=len(plan['seeds']):
                        continue
                    errors=[r['errors'][region] for r in same]
                    reference_ms=errors[0]['reference_ms']
                    assert all(e['reference_ms']==reference_ms for e in errors)
                    mse=float(np.mean([e['mse'] for e in errors]))
                    times=[r['cumulative_process_seconds'] for r in same]
                    agg.append(dict(spp=spp,seeds=len(same),mse=mse,relative_rmse=(mse/reference_ms)**.5,
                        mean_process_seconds=float(np.mean(times)),
                        mean_render_seconds=float(np.mean([r['cumulative_render_seconds'] for r in same])),
                        min_process_seconds=min(times),max_process_seconds=max(times),
                        min_relative_rmse=min(e['relative_rmse'] for e in errors),
                        max_relative_rmse=max(e['relative_rmse'] for e in errors)))
                aggregate[region]=agg
                hits=[]
                for threshold in manifest['thresholds']:
                    found=next((point for point in agg if point['relative_rmse']<=threshold),None)
                    reached=[]
                    per_seed=[]
                    for seed in plan['seeds']:
                        samples=[r for r in runs if r['seed']==seed]
                        first=next((r for r in samples if r['errors'][region]['relative_rmse']<=threshold),None)
                        if first:
                            reached.append(seed)
                        per_seed.append(dict(seed=seed,first_hit_spp=first['spp'] if first else None,
                            first_hit_process_seconds=first['cumulative_process_seconds'] if first else None))
                    hit=dict(threshold=threshold,first_hit=found,reached_seeds=reached,per_seed=per_seed)
                    hits.append(hit)
                    crossing_csv.append(dict(case=case,mode=mode,region=region,threshold=threshold,
                        first_hit_spp=found['spp'] if found else '',
                        mean_process_seconds=found['mean_process_seconds'] if found else '',
                        measured_relative_rmse=found['relative_rmse'] if found else '',
                        seeds_reached=len(reached),requested_seeds=len(plan['seeds']),
                        largest_complete_spp=agg[-1]['spp'] if agg else ''))
                crossings[region]=hits
                seed_crossings[region]=[dict(seed=seed,points=[dict(spp=r['spp'],
                    relative_rmse=r['errors'][region]['relative_rmse'],
                    process_seconds=r['cumulative_process_seconds']) for r in runs if r['seed']==seed])
                    for seed in plan['seeds']]
            data['modes'][mode]=dict(runs=runs,aggregate=aggregate,crossings=crossings,seed_curves=seed_crossings)
        for region in REGIONS:
            comparisons=[]
            a=data['modes']['global_conditional']['crossings'][region]
            b=data['modes']['gpis_renewal_plus']['crossings'][region]
            for left,right in zip(a,b):
                one,two=left['first_hit'],right['first_hit']
                comparisons.append(dict(threshold=left['threshold'],
                    global_conditional=one,gpis_renewal_plus=two,
                    renewal_time_over_conditional_time=two['mean_process_seconds']/one['mean_process_seconds']
                        if one and two else None))
            data['comparisons'][region]=comparisons
        output['cases'][case]=data
    output['complete']=output['observed_records']==output['expected_records']
    save(OUT/'results.json',output)
    write_csv(OUT/'observations.csv',list(observations[0]) if observations else
              ['case','mode','seed','spp','region','cumulative_process_seconds'],observations)
    write_csv(OUT/'time_to_error.csv',list(crossing_csv[0]),crossing_csv)
    return output


def plot_case(case, data, plan):
    figures=OUT/'figures'
    figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for index,region in enumerate(REGIONS):
        for col in (0,1):
            ax=axes[index,col]
            for mode,entry in data['modes'].items():
                for seed in plan['seeds']:
                    runs=[r for r in entry['runs'] if r['seed']==seed]
                    x=[r['cumulative_process_seconds'] if col==0 else r['spp'] for r in runs]
                    y=[r['errors'][region]['relative_rmse'] for r in runs]
                    ax.plot(x,y,color=COLORS[mode],lw=.8,alpha=.35,ls='--',marker='.',ms=3)
                agg=entry['aggregate'][region]
                if agg:
                    x=[p['mean_process_seconds'] if col==0 else p['spp'] for p in agg]
                    y=[p['relative_rmse'] for p in agg]
                    ax.plot(x,y,color=COLORS[mode],marker='o',ms=4,lw=2,label=LABELS[mode])
                    ax.fill_between(x,[p['min_relative_rmse'] for p in agg],
                        [p['max_relative_rmse'] for p in agg],color=COLORS[mode],alpha=.12)
                else:
                    ax.plot([],[],color=COLORS[mode],label=LABELS[mode]+' (individual seeds only)')
            noise=data['reference']['estimated_reference_noise_relative_rms'][region]
            ax.axhline(noise,color='#666666',ls=':',lw=1,label=f'Reference noise estimate ({noise:.1%})')
            ax.axhline(.10,color='#999999',ls='-.',lw=.7)
            ax.set(xscale='log',yscale='log',xlabel='Measured cumulative process time per run (s)' if col==0 else 'Samples per pixel',
                   ylabel='Linear RGB relative RMSE',title=('Full image' if region=='full' else 'Bottom ROI')+
                   (' | error vs time' if col==0 else ' | error vs spp'))
            ax.grid(True,which='both',alpha=.18)
            if col==1:
                ax.set_xticks(plan['points'],labels=[str(s) for s in plan['points']])
            low,high=ax.get_ylim()
            ticks=[v for v in (.05,.075,.10,.125,.15,.20,.30,.50,.75,1.,1.5,2.) if low<=v<=high]
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(ticks))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda value, pos:f'{value*100:g}%'))
            ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax.legend(fontsize=7,loc='best')
    fig.suptitle('Shaderball | lengthScale=0.2, sigma=0.02 | 256 x 256 | seeds 41, 137, 313\n'
                 'Solid: sqrt(mean seed MSE), shaded: seed range; dashed: individual seeds',fontsize=11)
    for ext in ('png','svg'):
        fig.savefig(figures/f'{case}_curves.{ext}',dpi=180)
    plt.close(fig)


def crossing_cell(point, max_spp, complete):
    if point:
        return f'<b>{seconds(point["mean_process_seconds"])}</b><br><small>{point["spp"]} spp · {pct(point["relative_rmse"])}</small>'
    return f'{max_spp} spp 内未达标' if complete else '三种子数据尚未齐全'


def build_html(result,manifest):
    parts=[]
    for case,data in result['cases'].items():
        source=data['reference']
        noise=source['estimated_reference_noise_relative_rms']
        parts.append(f'<section><h2>标准组 · lengthScale=0.2，sigma=0.02</h2>'
            f'<p>固定参考：归档 GPIS global，256×256，128 spp，seed 20261024。'
            f'参考噪声估计：全图 {pct(noise["full"])}，底部 {pct(noise["bottom"])}。</p>'
            f'<a href="figures/{case}_curves.svg"><img class="plot" src="figures/{case}_curves.png" alt="误差随时间及采样数变化的曲线"></a>'
            '<p class="muted">实线：三个种子的 MSE 取均值后开方；阴影：三个种子的误差范围；虚线：单个种子。'
            '横轴时间为单次渲染的平均累计时间。仅三个种子齐全的检查点参与粗实线和达标时间比较。</p>')
        for region,label in REGIONS.items():
            parts.append(f'<h3>{label}：达到误差阈值的实测时间</h3><div class="scroll"><table>'
                '<thead><tr><th>目标 NRMSE</th><th>global_conditional</th><th>GPIS renewal+ (1D MIS)</th>'
                '<th>renewal+ 耗时 / conditional 耗时</th></tr></thead><tbody>')
            for comparison in data['comparisons'][region]:
                factor=comparison['renewal_time_over_conditional_time']
                cells=''.join('<td>'+crossing_cell(comparison[mode],max(result['plan']['points']),result['complete'])+'</td>'
                              for mode in MODES)
                ratio=f'{factor:.2f}×' if factor is not None else '—'
                parts.append(f'<tr><td>≤ {pct(comparison["threshold"])}</td>{cells}<td>{ratio}</td></tr>')
            parts.append('</tbody></table></div>')
        parts.append('<p>达标时间取首次满足阈值的离散检查点；真实的首次穿越时间可能更早。'
            '没有在检查点之间插值，也没有对更高 spp 外推。误差可能随采样波动，首次达标不代表后续一定持续达标。</p>')
        parts.append('<h3>各采样检查点的三种子结果（全图）</h3><div class="scroll"><table>'
            '<thead><tr><th>spp</th><th>模式</th><th>聚合 NRMSE</th><th>种子误差范围</th>'
            '<th>平均累计进程耗时</th><th>平均累计渲染循环耗时</th></tr></thead><tbody>')
        for spp in result['plan']['points']:
            for mode in MODES:
                point=next((p for p in data['modes'][mode]['aggregate']['full'] if p['spp']==spp),None)
                if point:
                    parts.append(f'<tr><td>{spp}</td><td>{LABELS[mode]}</td><td>{pct(point["relative_rmse"])}</td>'
                        f'<td>{pct(point["min_relative_rmse"])}–{pct(point["max_relative_rmse"])}</td>'
                        f'<td>{seconds(point["mean_process_seconds"])}</td><td>{seconds(point["mean_render_seconds"])}</td></tr>')
        parts.append('</tbody></table></div>')
        parts.append('<h3>各种子的独立达标情况（全图）</h3><div class="scroll"><table><thead><tr>'
            '<th>目标</th><th>模式</th><th>达标种子数</th><th>seed 41</th><th>seed 137</th><th>seed 313</th></tr></thead><tbody>')
        for index,threshold in enumerate(manifest['thresholds']):
            for mode in MODES:
                hit=data['modes'][mode]['crossings']['full'][index]
                parts.append(f'<tr><td>{pct(threshold)}</td><td>{LABELS[mode]}</td><td>{len(hit["reached_seeds"])}/3</td>')
                for seed in hit['per_seed']:
                    parts.append('<td>'+ (f'{seconds(seed["first_hit_process_seconds"])} / {seed["first_hit_spp"]} spp'
                        if seed['first_hit_spp'] else '未达标' if result['complete'] else '等待数据')+'</td>')
                parts.append('</tr>')
        parts.append('</tbody></table></div><h3>原始渲染图</h3><p>PNG 仅用于观察；数值误差始终由线性 HDR PFM 计算。'
                     '显示已有色调映射结果，不对两种方法分别调整曝光。</p>')
        parts.append('<div class="controls"><label>种子 <select id="seed">'+
            ''.join(f'<option>{s}</option>' for s in result['plan']['seeds'])+'</select></label>'
            '<label>spp <select id="spp">'+''.join(f'<option>{s}</option>' for s in result['plan']['points'])+
            '</select></label></div><div class="gallery">')
        reference=ROOT/Path(source['folder'])/'render.png'
        parts.append(f'<figure><figcaption>GPIS global · 固定参考 · 128 spp</figcaption>'
                     f'<a href="{rel(reference)}"><img src="{rel(reference)}" alt="固定 GPIS global 参考"></a></figure>')
        for mode in MODES:
            parts.append(f'<figure><figcaption>{LABELS[mode]}</figcaption><a id="link_{mode}">'
                f'<img id="image_{mode}" alt="{LABELS[mode]}"></a><p id="metric_{mode}" class="muted"></p></figure>')
        parts.append('</div></section>')
    complete=result['complete']
    status=f'{"完成" if complete else "进行中"} · {result["observed_records"]}/{result["expected_records"]} 个检查点'
    validation=load(OUT/'validation/resume.json')
    assert len(validation)==2 and all(v['equal'] for v in validation)
    timing_notes=''
    if result['excluded_timing_attempts']:
        timing_notes='<p>本轮有 '+str(len(result['excluded_timing_attempts']))+' 次计时受到执行环境停顿影响，已保留原始日志并从前一检查点重测。被排除的尝试未进入曲线：'+ ' · '.join(
            f'<a href="{html.escape(item["folder"])}/reason.json">尝试 {i+1}</a>'
            for i,item in enumerate(result['excluded_timing_attempts']))+'。</p>'
    diagnostics='；'.join(f'{LABELS[mode]}：'+str(sum(row['path_diagnostics']['far_t_zero_lines']
        for case in result['cases'].values() for row in case['modes'][mode]['runs']))+' 条'
        for mode in MODES)
    timing_notes+=(
        '<p>原实现的 <code>farT == 0</code> 路径诊断计数（各区段相加）：'+diagnostics+
        '。完整日志随每个检查点保存，本次测试未修改该实现。</p>')
    html_text='''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaderball：三种子等误差耗时比较</title>
<style>
:root{font:15px/1.65 system-ui,sans-serif;color:#243440;background:#f5f7fa}body{max-width:1160px;margin:0 auto;padding:30px 22px}h1{font-size:27px;line-height:1.35}h2{font-size:22px}h3{font-size:18px;margin-top:28px}a{color:#087e8b}section,.protocol{background:white;padding:25px;border:1px solid #e0e6ed;border-radius:12px;margin:24px 0}small,.muted{color:#61717e}table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}th,td{text-align:left;border-bottom:1px solid #e1e7ee;padding:11px 12px;vertical-align:top}th{background:#f2f6f8}.scroll{overflow:auto}.plot{width:100%;height:auto}.gallery{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.gallery figure{margin:0}.gallery img{width:100%;background:#eaeef2}.gallery figcaption{min-height:52px}.gallery p{font-size:13px}select{font:inherit;padding:4px 10px;margin-right:25px}.controls{margin:18px 0}.status{display:inline-block;padding:5px 12px;background:#e4f2f3;border-radius:7px}code{overflow-wrap:anywhere}details{margin-top:20px}@media(max-width:700px){.gallery{grid-template-columns:1fr}section,.protocol{padding:16px}body{padding:20px 12px}}
</style><body><h1>global_conditional 与 GPIS renewal+：达到固定参考误差的时间</h1>
<p class="status">STATUS</p><p>仅测试标准组。两个模式都使用当前同一个渲染器、32 线程、256×256 分辨率、独立种子 41 / 137 / 313。</p>
BODY
<div class="protocol"><h2>测量定义与限制</h2>
<p><code>NRMSE = sqrt(mean((RGB_test − RGB_ref)²) / mean(RGB_ref²))</code>，所有像素、RGB 通道在原始线性 HDR 中等权计算。底部 ROI 为 x∈[44,213)、y∈[164,216)，使用该区域自己的参考均方值归一化。</p>
<p>三个种子的误差先计算各自 MSE，再取均值并开方；没有将三张图平均成一张再计算误差。时间是单次渲染的三种子平均值，没有将三次运行成本相加。三个种子只提供有限的重复性观察，不作为高置信度统计结论。</p>
<p>累计时间是从 1 spp 逐级续渲到目标 spp 的所有渲染器进程耗时之和，包含场景加载、状态读取和保存，排除误差计算与报告生成。两个模式均设置 spp_step=1，关闭自适应采样，输出检查点为 1、2、4、8、16、32、64 spp。CSV 同时保留不含加载与最终保存的渲染循环计时。</p>
<p>32×32 的一致性测试已确认：两个模式的 1→2 spp 续渲与连续 2 spp 结果逐像素相同。场景、相机、材质和光照沿用原报告，只修改输出路径、采样控制和统计输出。</p>
TIMING_NOTES
<p>GPIS renewal+ 使用原报告的 <b>1D MIS</b> 配置（step=0.0025、启用横向梯度相关）。global_conditional 使用原报告的 <b>retain_endpoint</b> 配置，fp_gradient 关闭。</p>
<p>参考是用户指定的 128 spp 有噪声图，不是无噪声真值。参考噪声由 0.5×(samplesA−samplesB) 估计；该值用于提示比较精度，不从测得误差中扣除，也不作为精确的误差下限。</p>
<details><summary>可复现性与归档差异</summary><p>当前两个被测模式使用的渲染器 SHA256：<code>CURRENT_SHA</code>。归档参考渲染器 SHA256：<code>OLD_SHA</code>。原报告中的时间来自另一台机器，本次曲线完全使用当前机器的实测时间。归档方法图与参考使用同一个旧种子，未用于本次独立种子统计。</p></details>
<p><a href="results.json">聚合结果 JSON</a> · <a href="observations.csv">逐种子检查点 CSV</a> · <a href="time_to_error.csv">达标时间 CSV</a> · <a href="manifest.json">配置与参考哈希</a> · <a href="plan.json">本轮计划</a> · <a href="validation/resume.json">续渲验证</a> · <a href="environment.json">本机环境</a>AUDIT_LINK</p>
<p class="muted">更新时间：UPDATED。进行中的页面可刷新以读取最新结果。</p></div>
<script>
const data=DATA;
const seed=document.getElementById('seed'), spp=document.getElementById('spp');
const modes=['global_conditional','gpis_renewal_plus'];
function update(){for(const mode of modes){const row=data.modes[mode].runs.find(r=>r.seed===Number(seed.value)&&r.spp===Number(spp.value));const img=document.getElementById('image_'+mode),a=document.getElementById('link_'+mode),p=document.getElementById('metric_'+mode);if(row){img.src=row.image;img.hidden=false;a.href=row.image;p.textContent='全图 NRMSE '+(row.errors.full.relative_rmse*100).toFixed(2)+'% · 累计 '+row.cumulative_process_seconds.toFixed(1)+' s';}else{img.hidden=true;a.removeAttribute('href');p.textContent='此检查点尚未完成';}}}
const available=modes.map(m=>data.modes[m].runs.filter(r=>r.seed===Number(seed.value)).map(r=>r.spp));
const common=available[0].filter(n=>available[1].includes(n));
spp.value=String(Math.max(...(common.length?common:available[0]),1));seed.onchange=update;spp.onchange=update;update();
</script></body></html>'''
    case=next(iter(result['cases']))
    html_text=html_text.replace('STATUS',status).replace('BODY','\n'.join(parts))
    html_text=html_text.replace('TIMING_NOTES',timing_notes)
    html_text=html_text.replace('AUDIT_LINK',' · <a href="audit.json">最终数据复核</a>' if (OUT/'audit.json').exists() else '')
    html_text=html_text.replace('CURRENT_SHA',manifest['renderer_sha256']).replace('OLD_SHA',result['cases'][case]['reference']['renderer_sha256'])
    html_text=html_text.replace('UPDATED',html.escape(result['updated_utc']))
    html_text=html_text.replace('DATA',json.dumps(result['cases'][case],ensure_ascii=False).replace('</','<\\/'))
    atomic_text(OUT/'comparison.html',html_text)


def main():
    manifest=load(OUT/'manifest.json')
    plan=load(OUT/'plan.json')
    # This report is intentionally scoped to the user's one approved standard case.
    assert plan['cases']==['ell0p2_sigma0p02'] and plan['seeds']==[41,137,313]
    result=summarize(manifest,plan)
    for case,data in result['cases'].items():
        plot_case(case,data,plan)
    build_html(result,manifest)
    print('REPORT',result['observed_records'],'/',result['expected_records'],OUT/'comparison.html',flush=True)


if __name__=='__main__':
    main()
