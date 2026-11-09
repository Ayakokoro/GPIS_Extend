"""Report the matched strong-light, full-Renewal+ plane experiment."""
from __future__ import annotations

import argparse
import copy
import csv
from pathlib import Path

from figures import plt, Image, COLORS
import numpy as np
from plane_figures import filmic
from plane_strong_light import (OUT, ROOT, CASES, SEED, STAGE, PILOT, MODES, RENEWAL,
    CHANGES, EMISSION, LAYERS, destination, reference, aov_scene, validate_aov, light_parameters)
from run import load, save, sha, pfm

LABELS = ('classic_local', 'classic_global', 'global_conditional', 'GPIS 1D renewal+ MIS\ncorrelationXY=true')
PARTS = ('beauty', 'order01', 'multiple', 'order02', 'order03', 'order04', 'order05plus')
PART_LABELS = {'beauty':'总结果', 'order01':'单次散射', 'multiple':'两次及以上散射',
               'order02':'2 次散射', 'order03':'3 次散射', 'order04':'4 次散射', 'order05plus':'5 次及以上'}


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def point_light_comparisons(dest, run, data, stats):
    """Keep changes of light type separate from changes of sample count."""
    area_stage='light60_full_renewal_mis_spp512'
    area_manifest=load(OUT/area_stage/'manifest.json')
    assert area_manifest['run']['aov_sha256']==run['aov_sha256']
    assert area_manifest['parameters']['emission']==run['emission']==60
    area_stats={(r['case'],r['mode'],r['component']):r for r in csv.DictReader(
        (OUT/area_stage/'metrics.csv').open(encoding='utf-8-sig',newline=''))}
    rows=[]
    for case in CASES:
        for mode in MODES:
            area_dir=destination(area_stage,case,mode,'bounces')
            area_scene=load(area_dir/'scene.json')
            point_scene=load(destination(run['stage'],case,mode,'bounces')/'scene.json')
            assert point_scene['primitives'][1]['transform']['position']==area_scene['primitives'][1]['transform']['position']
            comparable=copy.deepcopy(point_scene)
            comparable['renderer']=area_scene['renderer']
            comparable['primitives'][1]=area_scene['primitives'][1]
            assert comparable==area_scene
            verified=load(area_dir/'verified.json')
            for part in PARTS[:3]:
                assert sha(area_dir/(part+'.pfm'))==verified['files'][part+'.pfm']
                old=area_stats[case,mode,part]; current=stats[case,mode,part]
                rows.append(dict(case=case,mode=mode,component=part,
                    area_mean=float(old['mean']),point_mean=current['mean'],
                    point_over_area=current['mean']/float(old['mean']),
                    area_spp=int(old['spp']),point_spp=run['spp'],
                    area_standard_error=float(old['standard_error']),point_standard_error=current['standard_error'],
                    normalization='equal total power; not equal irradiance'))
    write_csv(dest/'point_vs_area.csv',rows)
    section=(f'<h2>点光源与面光源</h2><p><a href="../{area_stage}/comparison.html">原面光源 60 / 512 spp</a> · '
        '<a href="point_vs_area.csv">逐模型、逐分量的均值比较</a>。两种光源匹配总功率，发射方向分布不同；亮度差不是纯粹的采样噪声差。</p>')
    figures=[]
    preview_stage=run['stage']+'_preview32'
    preview_dir=OUT/preview_stage
    if run['spp']<=32 or not (preview_dir/'manifest.json').exists():
        return rows,section,figures
    preview=load(preview_dir/'manifest.json')
    assert preview['run']['spp']==32 and preview['run']['light']==run['light']
    assert preview['run']['aov_sha256']==run['aov_sha256']
    old_stats={(r['case'],r['mode'],r['component']):r for r in csv.DictReader(
        (preview_dir/'metrics.csv').open(encoding='utf-8-sig',newline=''))}
    noise_rows=[]; old_data={}
    for case in CASES:
        for mode in MODES:
            old_dir=destination(preview_stage,case,mode,'bounces')
            old_scene=load(old_dir/'scene.json')
            current_scene=load(destination(run['stage'],case,mode,'bounces')/'scene.json')
            current_scene['renderer']=old_scene['renderer']
            assert current_scene==old_scene
            verified=load(old_dir/'verified.json')
            for part in PARTS[:3]:
                assert sha(old_dir/(part+'.pfm'))==verified['files'][part+'.pfm']
                old_data[case,mode,part]=pfm(old_dir/(part+'.pfm'))
                old=old_stats[case,mode,part]; current=stats[case,mode,part]
                noise_rows.append(dict(case=case,mode=mode,component=part,
                    old_spp=32,new_spp=run['spp'],old_mean=float(old['mean']),new_mean=current['mean'],
                    old_rms_pixel_standard_error=float(old['rms_pixel_standard_error']),
                    new_rms_pixel_standard_error=current['rms_pixel_standard_error'],
                    noise_ratio=current['rms_pixel_standard_error']/float(old['rms_pixel_standard_error'])))
    write_csv(dest/'noise_vs_32.csv',noise_rows)
    for side,label in (('same','同侧'),('opposite','对侧')):
        fig,axes=plt.subplots(3,4,figsize=(14.5,11.4))
        for i,angle in enumerate((30,60,80)):
            case=f'{side}_{angle}'
            gain={30:1,60:8,80:32}[angle] if side=='same' else 1
            for j,(part,old) in enumerate((('beauty',True),('beauty',False),('multiple',True),('multiple',False))):
                rgb=old_data[case,RENEWAL,part] if old else data[case,RENEWAL][part]
                ax=axes[i,j]; ax.imshow(filmic(rgb,gain),interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f'{32 if old else run["spp"]} spp · {PART_LABELS[part]}\nL={rgb.mean():.6f}',fontsize=10)
                if j==0: ax.set_ylabel(f'{angle}° · 显示增益 ×{gain}',fontsize=11)
        fig.suptitle(f'点光源 I={run["light"]["intensity"]:g} · Full Renewal+ MIS · {label}\n32 → {run["spp"]} spp · correlationXY=true',fontsize=15)
        fig.text(.035,.025,'两批均为实际渲染；相同 seed，样本存在关联。每行使用同一显示增益，无降噪。\n多次散射中的稀有高贡献样本可能使低 spp 方差估计不稳定。',fontsize=9)
        fig.subplots_adjust(top=.88,bottom=.1,left=.07,right=.99,hspace=.3,wspace=.06)
        filename=f'spp_comparison_{side}.png'
        fig.savefig(dest/filename,dpi=150); plt.close(fig); figures.append(filename)
    section+=(f'<h2>32 → {run["spp"]} spp</h2><p><a href="../{preview_stage}/comparison.html">32 spp 完整预览</a> · '
        '<a href="noise_vs_32.csv">逐模型、逐分量的采样误差</a>。相同 seed，两批样本有关联；稀有高贡献路径会使低 spp 方差估计不稳定。</p>'
        '<img loading="lazy" src="spp_comparison_same.png"><img loading="lazy" src="spp_comparison_opposite.png">')
    return rows,section,figures


def verify_report(dest, manifest):
    from html.parser import HTMLParser
    from urllib.parse import unquote, urlsplit
    links=[]
    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            links.extend(v for k,v in attrs if k in ('src','href') and v)
    Links().feed((dest/'comparison.html').read_text(encoding='utf-8'))
    checked=0
    for link in links:
        parsed=urlsplit(link)
        if not parsed.scheme and parsed.path:
            assert (dest/unquote(parsed.path)).is_file(), link
            checked+=1
    for name in manifest['figures']:
        with Image.open(dest/name) as img:
            img.verify()
    sources=manifest['run']['sources']
    save(dest/'verification.json',dict(passed=True,renders=len(sources),
        figures=len(manifest['figures']),local_links=checked,
        verified_samples=sum(s['verification']['result']['verified_samples'] for s in sources),
        finite_clamp_restored_paths=sum(len(s['record']['restored_paths']) for s in manifest.get('finite_clamp_restorations',[])),
        finite_clamp_replay_samples=sum(s['record']['verified_samples'] for s in manifest.get('finite_clamp_restorations',[])),
        max_reconstruction_error=max(s['verification']['max_reconstruction_error'] for s in sources),
        scene_and_output_hashes_checked=True))


def main(stage=STAGE):
    dest = OUT/stage
    run = load(dest/'aov_run.json')
    spp, size, emission = run['spp'], run['size'], run['emission']
    light = run.get('light', light_parameters(emission))
    point = light['type'] == 'point'
    restorations=[dict(case=s['case'],mode=s['mode'],folder=s['folder'],
        record=s['verification']['finite_clamp_restoration']) for s in run['sources']
        if 'finite_clamp_restoration' in s['verification']]
    restoration_note=''
    if restorations:
        restored_count=sum(len(s['record']['restored_paths']) for s in restorations)
        links=' · '.join(f'<a href="{(Path(s["folder"])/"finite_clamp_restoration.json").relative_to(dest).as_posix()}">{s["case"]} 恢复记录</a>' for s in restorations)
        restoration_note=(f'<div class="note"><b>保留有限的大贡献：</b>普通 film 会把路径辐亮度超过 65536 的样本置零。'
            f'本轮发现并恢复了 {restored_count} 条这样的有限路径：回放受影响像素的全部样本，重算均值与方差，'
            '并用未受影响像素作对照。重放施加原阈值时须重现原始统计，才采用未截断结果。'
            '原始截断文件和回放数据均保留。采样及输运核心未改变；大贡献会增加方差，512 spp 仍不能视为已收敛。'
            f'{links}</div>')
    light_title = f'点光源 I={light["intensity"]:g}' if point else f'光源辐亮度 {emission:g}'
    geometry_note = (f'相机位置、方向、FOV、平面参数以及光源中心位置保持原配置；面光源改为各向同性点光源。'
        f'按总功率匹配原辐亮度 {emission:g}、面积 {light["reference_area"]:g} 的面光源：'
        f'4πI=πAL，故点光源辐射强度 I=<b>{light["intensity"]:g}</b>，总功率 {light["total_power"]:.6f}。'
        '等总功率不等于平面上等照度；原面光源朝向平面，而点光源向整个球面发光。'
        if point else f'相机位置、方向、FOV、光源位置与尺寸、平面参数均保持原配置；面光源辐亮度 <b>{emission:g}</b>。')
    assert size == 256
    pilot_stage=run.get('pilot_stage',PILOT)
    pilot = load(OUT/pilot_stage/'aov_run.json')
    decomposition = load(OUT/pilot_stage/'decomposition_checks.json')
    assert len(run['sources']) == len(pilot['sources']) == 24 and decomposition['passed']
    assert run['aov_sha256'] == pilot['aov_sha256']
    assert run['renderer_sha256'] == pilot['renderer_sha256']
    assert pilot.get('light',light_parameters(emission)) == light
    regressions=load(dest/'numerical_regressions.json')
    assert regressions['passed'] and regressions['core_sha256']==run['core_sha256']
    for source in pilot['sources']:
        assert 'native_max_absolute_error' in source['verification']
        for name, digest in source['verification']['files'].items():
            assert sha(Path(source['folder'])/name) == digest
    data, rows, stats, sources, summaries = {}, [], {}, [], []
    for source in run['sources']:
        case, mode, d = source['case'], source['mode'], Path(source['folder'])
        assert load(d/'scene.json') == aov_scene(stage, case, mode, size, spp, emission, light['type'])
        for name, digest in source['verification']['files'].items():
            assert sha(d/name) == digest
        result, arrays = validate_aov(d)
        assert result['samples'] == spp and result['resolution'] == [size,size]
        data[case,mode] = arrays
        for part in PARTS:
            a = arrays[part][:,:,0]
            assert np.max(np.ptp(arrays[part],axis=2)) == 0
            # The AOV tool stores per-sample Welford variance, not variance of the mean.
            v = pfm(d/(part+'_variance.pfm'))[:,:,0]/result['samples']
            row = dict(case=case,mode=mode,component=part,mean=float(a.mean()),
                standard_error=float(np.sqrt(v.sum())/a.size),
                rms_pixel_standard_error=float(np.sqrt(v.mean())),zero_fraction=float((a==0).mean()),
                maximum=float(a.max()),spp=result['samples'],
                variance_method='per_path_sample_variance; covariance retained in multiple aggregate',
                raw_pfm=(d/(part+'.pfm')).relative_to(dest).as_posix())
            rows.append(row); stats[case,mode,part] = row
            Image.fromarray(np.rint(255*filmic(arrays[part])).astype('uint8')).save(d/(part+'.png'))
        total = stats[case,mode,'beauty']['mean']
        summaries.append(dict(case=case,mode=mode,total=total,single=stats[case,mode,'order01']['mean'],
            multiple=stats[case,mode,'multiple']['mean'],
            multiple_fraction=stats[case,mode,'multiple']['mean']/total,
            seconds=result['seconds']))
        sources.append(source)
    scaled_reference_checks=[]
    for case in CASES:
        scenes=[load(destination(stage,case,mode,'bounces')/'scene.json') for mode in MODES]
        for key in ('camera','primitives','integrator'):
            assert all(s[key]==scenes[0][key] for s in scenes)
        for mode in MODES[:3]:
            if spp != 128 or point:
                continue  # A different sample count cannot match the original image pixel by pixel.
            original=reference(case,mode)
            assert sha(original/'render.pfm')==load(original/'result.json')['pfm_sha256']
            old_scaled=pfm(original/'render.pfm')*(emission/30)
            current=data[case,mode]['beauty']
            assert np.allclose(current,old_scaled,rtol=3e-5,atol=2e-7), (case,mode)
            scaled_reference_checks.append(dict(case=case,mode=mode,
                max_absolute_error=float(np.abs(current-old_scaled).max()),
                original_pfm_sha256=sha(original/'render.pfm')))

    figure_files = []

    def grid(cells, row_labels, title, filename, gains, block_average=False):
        fig, axes = plt.subplots(len(cells),4,figsize=(14.5,3.4*len(cells)+1.2),squeeze=False)
        for i, ((case,part),label,gain) in enumerate(zip(cells,row_labels,gains)):
            for j,mode in enumerate(MODES):
                ax = axes[i,j]
                rgb=data[case,mode][part]
                if block_average:
                    # Explicit spatial summary, applied before the display curve.
                    rgb=rgb.reshape(16,16,16,16,3).mean(axis=(1,3))
                ax.imshow(filmic(rgb,gain),interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                r = stats[case,mode,part]
                ax.set_title(f'{LABELS[j]}\nL={r["mean"]:.6f} ± {1.96*r["standard_error"]:.6f}',fontsize=9,color=COLORS[j])
                if j==0: ax.set_ylabel(f'{label}\n显示增益 ×{gain:g}',fontsize=11)
        fig.suptitle(title+f'\n{light_title} · {size}×{size} · {spp} spp',fontsize=15)
        note='每个格子为 16×16 原始像素的线性平均（不是原始分辨率，不增加 spp）。' if block_average else '原始像素，无降噪。'
        fig.text(.035,.025,'同一行四种模型使用相同曝光；'+note+'\nL 为线性平均辐亮度，± 为近似 95% 采样误差；总结果及各阶贡献来自同一批完整路径。',fontsize=9,color='#4b535e')
        fig.subplots_adjust(top=1-.12/len(cells)*3,bottom=.1,left=.07,right=.99,hspace=.35,wspace=.06)
        fig.savefig(dest/filename,dpi=150); plt.close(fig)
        figure_files.append(filename)

    for side,label in (('same','同侧照明'),('opposite','对侧照明')):
        cells = [(f'{side}_{angle}','beauty') for angle in (30,60,80)]
        grid(cells,[f'{a}°' for a in (30,60,80)],f'平面 · {label} · 四种模型',f'appearance_{side}.png',[1,1,1])
    grid([(f'same_{a}','beauty') for a in (30,60,80)],[f'{a}°' for a in (30,60,80)],
         '平面 · 同侧照明 · 提亮查看掠射角', 'appearance_same_lifted.png',[1,8,32])
    for case in CASES:
        label = ('同侧' if case.startswith('same') else '对侧')+' '+case.split('_')[1]+'°'
        grid([(case,p) for p in PARTS[:3]],[PART_LABELS[p] for p in PARTS[:3]],
             f'{label} · 总结果 / 单次 / 多次',f'scattering_{case}.png',[1]*3)
        if case in ('same_60','same_80'):
            gain=8 if case=='same_60' else 32
            grid([(case,p) for p in PARTS[:3]],[PART_LABELS[p] for p in PARTS[:3]],
                 f'{label} · 提亮查看（所有模型与分量统一增益 ×{gain}）',
                 f'scattering_{case}_lifted.png',[gain]*3)
        grid([(case,p) for p in PARTS[:3]],[PART_LABELS[p] for p in PARTS[:3]],
             f'{label} · 16×16 分块均值（空间汇总）',f'scattering_{case}_blocks.png',[1]*3,block_average=True)
        grid([(case,p) for p in ('order01','order02','order03','order04','order05plus')],
             [PART_LABELS[p] for p in ('order01','order02','order03','order04','order05plus')],
             f'{label} · 按散射次数分解',f'bounce_orders_{case}.png',[1]*5)

    comparisons=[]
    for case in CASES:
        for part in PARTS[:3]:
            ref=stats[case,RENEWAL,part]['mean']
            local=stats[case,'classic_local',part]['mean']
            a=stats[case,'classic_global',part]['mean']
            b=stats[case,'global_conditional',part]['mean']
            comparisons.append(dict(case=case,component=part,renewal_mean=ref,classic_mean=a,
                conditional_mean=b,classic_relative_difference=(a-ref)/ref,
                conditional_relative_difference=(b-ref)/ref,
                closer_in_image_mean='global_conditional' if abs(b-ref)<abs(a-ref) else 'classic_global',
                local_mean=local,local_relative_difference=(local-ref)/ref,
                conditional_vs_local_relative_difference=(b-local)/local))
    write_csv(dest/'metrics.csv',rows)
    write_csv(dest/'scattering_summary.csv',summaries)
    write_csv(dest/'model_differences.csv',comparisons)
    noise_rows, noise_section = [], ''
    if not point and stage != STAGE and spp != 128:
        previous=load(OUT/STAGE/'manifest.json')
        assert previous['run']['aov_sha256']==run['aov_sha256']
        old_stats={(r['case'],r['mode'],r['component']):r for r in csv.DictReader(
            (OUT/STAGE/'metrics.csv').open(encoding='utf-8-sig',newline=''))}
        old_emission=previous['parameters']['emission']
        factor=emission/old_emission
        old_data={}
        for case in CASES:
            for mode in MODES:
                old_dir=destination(STAGE,case,mode,'bounces')
                old_scene=load(old_dir/'scene.json')
                new_scene=load(destination(stage,case,mode,'bounces')/'scene.json')
                comparable=copy.deepcopy(new_scene)
                comparable['renderer']=old_scene['renderer']
                comparable['primitives'][1]['emission']=old_emission
                assert comparable==old_scene
                verified=load(old_dir/'verified.json')
                for part in PARTS[:3]:
                    assert sha(old_dir/(part+'.pfm'))==verified['files'][part+'.pfm']
                    old_data[case,mode,part]=pfm(old_dir/(part+'.pfm'))*factor
                    old=old_stats[case,mode,part]; current=stats[case,mode,part]
                    old_rms=float(old['rms_pixel_standard_error'])*factor
                    noise_rows.append(dict(case=case,mode=mode,component=part,
                        old_spp=128,new_spp=spp,emission=emission,
                        old_scaled_mean=float(old['mean'])*factor,new_mean=current['mean'],
                        old_scaled_rms_pixel_standard_error=old_rms,
                        new_rms_pixel_standard_error=current['rms_pixel_standard_error'],
                        noise_ratio=current['rms_pixel_standard_error']/old_rms,
                        old_scaled_standard_error=float(old['standard_error'])*factor,
                        new_standard_error=current['standard_error']))
        write_csv(dest/'noise_vs_128.csv',noise_rows)
        for side,label in (('same','同侧'),('opposite','对侧')):
            fig,axes=plt.subplots(3,4,figsize=(14.5,11.4))
            for i,angle in enumerate((30,60,80)):
                case=f'{side}_{angle}'
                gain={30:1,60:8,80:32}[angle] if side=='same' else 1
                for j,(part,old) in enumerate((('beauty',True),('beauty',False),('multiple',True),('multiple',False))):
                    rgb=old_data[case,RENEWAL,part] if old else data[case,RENEWAL][part]
                    ax=axes[i,j]; ax.imshow(filmic(rgb,gain),interpolation='nearest')
                    ax.set_xticks([]);ax.set_yticks([])
                    ax.set_title(f'{128 if old else spp} spp · {PART_LABELS[part]}\nL={rgb.mean():.6f}',fontsize=10)
                    if j==0:ax.set_ylabel(f'{angle}° · 显示增益 ×{gain}',fontsize=11)
            fig.suptitle(f'Full Renewal+ MIS · correlationXY=true · {label}\n128 → {spp} spp，统一到光源辐亮度 {emission:g}',fontsize=15)
            fig.text(.035,.025,f'128 spp 图从光强 {old_emission:g} 线性缩放至 {emission:g} 后显示；{spp} spp 为本轮实际渲染。\n无降噪；同一行曝光一致；相同 seed 的两批样本存在关联。',fontsize=9)
            fig.subplots_adjust(top=.88,bottom=.1,left=.07,right=.99,hspace=.3,wspace=.06)
            filename=f'spp_comparison_{side}.png'
            fig.savefig(dest/filename,dpi=150);plt.close(fig);figure_files.append(filename)
        noise_section=f'<h2>128 → {spp} spp：Renewal+ 噪声对照</h2><p>旧 128 spp 图的线性辐亮度按 {old_emission:g} → {emission:g} 缩放，用于同光强比较；{spp} spp 为本轮实际渲染。两者相同 seed，样本有关联。<a href="noise_vs_128.csv">逐模型、逐分量的噪声统计</a>。</p><img loading="lazy" src="spp_comparison_same.png"><img loading="lazy" src="spp_comparison_opposite.png">'
    point_comparisons=[]
    if point:
        point_comparisons,point_section,extra_figures=point_light_comparisons(dest,run,data,stats)
        noise_section+=point_section
        figure_files.extend(extra_figures)
    manifest=dict(parameters=dict(emission=emission,light=light,resolution=[size,size],spp=spp,seed=SEED,
        cases=CASES,modes=MODES,renewal=CHANGES,max_bounces=128,
        camera_plane_light_geometry=('camera, plane and light center copied; quad replaced with point' if point
                                     else 'exactly copied from original plane scenes')),
        run=run,pilot_validation=str(OUT/pilot_stage/'aov_run.json'),
        native_decomposition_validation=str(OUT/pilot_stage/'decomposition_checks.json'),
        numerical_regressions=regressions,
        baseline_scaled_checks=scaled_reference_checks,noise_vs_128=noise_rows,point_comparisons=point_comparisons,
        finite_clamp_restorations=restorations,
        definition='AOV on the same full path: total=1+2+3+4+5plus; multiple=2+3+4+5plus.',
        display_gain=1,figures=figure_files,summaries=summaries,mean_comparisons=comparisons,
        numerical_fix='At zero segment distance the Renewal+ NEE PDF uses the existing start-point gradient distribution. For short segments the same analytic covariance is evaluated without subtractive cancellation. Both fixes prevent nonfinite black paths.',
        baseline_renderer_sha256=load(OUT/'final/same_30/classic_global/seed20261024/total/result.json')['provenance']['renderer_sha256'],
        report_script_sha256=sha(Path(__file__)))
    save(dest/'manifest.json',manifest)
    table=''.join(f'<tr><td>{r["case"]}</td><td>{r["mode"]}</td><td>{r["total"]:.6f}</td><td>{r["single"]:.6f}</td><td>{r["multiple"]:.6f}</td><td>{r["multiple_fraction"]:.2%}</td></tr>' for r in summaries)
    diff_table=''.join(f'<tr><td>{r["case"]}</td><td>{PART_LABELS[r["component"]]}</td><td>{r["local_relative_difference"]:+.2%}</td><td>{r["classic_relative_difference"]:+.2%}</td><td>{r["conditional_relative_difference"]:+.2%}</td><td>{r["conditional_vs_local_relative_difference"]:+.2%}</td></tr>' for r in comparisons)
    sections=''
    for case in CASES:
        lifted=(f'<details><summary>提亮查看：四种模型及三个分量统一增益 ×{8 if case=="same_60" else 32}</summary><img loading="lazy" src="scattering_{case}_lifted.png"></details>'
                if case in ('same_60','same_80') else '')
        sections+=f'<section id="{case}"><h2>{case}：散射分解</h2><a href="scattering_{case}.png"><img loading="lazy" src="scattering_{case}.png"></a>{lifted}<details><summary>查看分块均值，辅助比较亮度（每格汇总 16×16 原始像素）</summary><img loading="lazy" src="scattering_{case}_blocks.png"></details><details><summary>展开 1 / 2 / 3 / 4 / 5+ 次散射</summary><a href="bounce_orders_{case}.png"><img loading="lazy" src="bounce_orders_{case}.png"></a></details></section>'
    raw_links=''.join(f'<tr><td>{r["case"]}</td><td>{r["mode"]}</td><td>{PART_LABELS[r["component"]]}</td><td><a href="{r["raw_pfm"]}">线性 PFM</a> · <a href="{r["raw_pfm"].replace(".pfm",".png")}">PNG</a></td></tr>' for r in rows)
    baseline_note=(f'前三种模型另与原图的 {emission/30:g} 倍逐像素核对，在浮点误差范围内一致。'
                   if scaled_reference_checks else '本轮仅改变光源强度与 spp；渲染核心与上一轮 128 spp 一致。不同 spp 的图像不要求逐像素相等。')
    if point:
        baseline_note='本轮沿用面光源实验的渲染核心。已知失败样本在原面光源配置回放；点光源另通过下述独立预检。点光源直接光由 NEE 连接，delta 光源权重为 1；仍保留 1D_sampling_scheme=mis，继续路径用于多次散射。'
    page=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>平面：{light_title} · {spp} spp · Full Renewal+ MIS</title>
<style>body{{max-width:1600px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#253145;background:#f5f6f8}}img{{width:100%;background:white}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.note{{background:#e7edf5;padding:18px;border-radius:10px}}.scroll{{overflow-x:auto}}details{{margin:18px 0}}summary{{cursor:pointer}}nav a{{margin-right:16px}}</style>
<h1>平面 · {light_title} · {spp} spp · 四种模型</h1>
<nav><a href="../comparison.html">原实验</a><a href="metrics.csv">线性统计 CSV</a><a href="model_differences.csv">模型差异 CSV</a><a href="manifest.json">参数与验证</a><a href="#differences">定量差异</a></nav>
<p>原有六组相机与布光几何：同侧／对侧，各 30°、60°、80°。{geometry_note} {size}×{size}，<b>{spp} spp</b>，seed={SEED}。</p>
<p>均值 μ=y；σ=0.01，ℓ=0.025；镜面反射率 0.9；无环境光。最后一列使用 1D_sampling=true、correlation_context=renewal+、1D_sampling_scheme=mis、<b>1D_gradient_correlationXY=true</b>、single_realization=false。</p>
<div class="note">四列均为本轮实际重渲染。总结果、单次、2+ 次，以及 2／3／4／5+ 次贡献由<b>同一批完整路径逐阶记录</b>；max_bounces=128，透明边界与光源端点不计散射。默认图统一显示增益 ×1；无降噪或逐模型自动曝光。误差来自逐路径 Welford 方差，多次散射的汇总方差保留各阶间协方差。</div>
{restoration_note}
<details><summary>数值修复与验证</summary><p>沿用已修复的完整 Renewal+：零长度 NEE PDF 与起点梯度采样分布一致，短段方差使用稳定的等价计算，避免非有限贡献将整条路径置黑。两个已知失败样本通过本轮回放验证。四种模型统一使用同一核心，哈希记录在 manifest 中。{baseline_note}</p><p>预检覆盖六场景×四模型：总图与独立 bounce 限制分解一致；AOV 总图与普通渲染一致。正式渲染检查逐路径贡献重建、记录器开关不改变辐亮度和随机序列、没有非有限路径或亮度截断。<a href="numerical_regressions.json">回归验证记录</a></p></details>
<h2>总体结果</h2><a href="appearance_same.png"><img src="appearance_same.png"></a><details><summary>同侧掠射角提亮版（30° ×1，60° ×8，80° ×32）</summary><img loading="lazy" src="appearance_same_lifted.png"></details><a href="appearance_opposite.png"><img src="appearance_opposite.png"></a>
{noise_section}<h2>多次散射分解</h2><nav>{''.join(f'<a href="#{c}">{c}</a>' for c in CASES)}</nav>{sections}
<h2>线性平均辐亮度与多次散射比例</h2><div class="scroll"><table><tr><th>场景</th><th>模型</th><th>总结果</th><th>单次</th><th>2+ 次</th><th>2+ 占比</th></tr>{table}</table></div>
<h2 id="differences">相对 Full Renewal+ 的图像均值差异</h2><p>前三列差异=(模型均值−Renewal+ 均值)/Renewal+ 均值；最后一列直接比较 conditional 与 local。只比较这批样本的全图平均值，不表示图像逐像素更接近，也不作为无偏参考或收敛证明。提高光强会缩放线性能量，本身不会改变相对模型差异。</p><div class="scroll"><table><tr><th>场景</th><th>分量</th><th>classic_local</th><th>classic_global</th><th>global_conditional</th><th>conditional 相对 local</th></tr>{diff_table}</table></div>
<details><summary>所有单图与线性原始数据</summary><div class="scroll"><table><tr><th>场景</th><th>模型</th><th>分量</th><th>文件</th></tr>{raw_links}</table></div></details></html>'''
    (dest/'comparison.html').write_text(page,encoding='utf-8')
    old=OUT/'comparison.html'
    text=old.read_text(encoding='utf-8')
    link_id='strong-light-renewal-link' if stage==STAGE else f'plane-{stage}-link'
    if f'id="{link_id}"' not in text:
        link=f'<p id="{link_id}"><a href="{stage}/comparison.html">{light_title} · {spp} spp：四种模型、correlationXY=true、逐 bounce 分解</a></p>'
        if '</h1>' in text: text=text.replace('</h1>','</h1>'+link,1)
        else: text+=link
        old.write_text(text,encoding='utf-8')
    if stage != STAGE:
        old=OUT/('light60_full_renewal_mis_spp512' if point else STAGE)/'comparison.html'
        text=old.read_text(encoding='utf-8')
        if f'id="{link_id}"' not in text:
            link=f'<p id="{link_id}"><a href="../{stage}/comparison.html">新结果：{light_title} · {spp} spp，总结果与逐 bounce 分解</a></p>'
            text=text.replace('</h1>','</h1>'+link,1)
            old.write_text(text,encoding='utf-8')
    verify_report(dest,manifest)
    print('Saved',dest/'comparison.html',len(rows),'statistics and',len(figure_files),'figures',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default=STAGE)
    args=parser.parse_args()
    main(args.stage)
