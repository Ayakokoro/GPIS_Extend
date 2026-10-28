"""Validate and plot both ell=.2 shaderball groups beside the existing baseline."""
import copy
import csv
from pathlib import Path
import numpy as np
from gpis_shaderball import SOURCE, MODES, LABELS, ROI, MODEL_ROI, output, field_path, source
from gpis_shaderball_sweep import REPORT, GROUPS, BASELINE_REFRESH_STAGE, configuration
from gpis_shaderball_figures import read, regions
from run import load, save
from shaderball_length import digest
from figures import plt, Image, Rectangle, COLORS
from plane_figures import blocks, filmic

NAMES=('基准','A：只加长相关长度','B：保持梯度扰动幅度')
PAIR_MODES=(('classic_local','classic_global'),('global_conditional','classic_global'),
    ('sparse_3d_global','classic_global'),('classic_local','sparse_3d_global'),
    ('classic_global','sparse_3d_global'),('global_conditional','sparse_3d_global'))


def group_label(group):
    return f'ℓ={group["ell"]:g}，σ={group["sigma"]:g}'


def paired_variance(a,b):
    return .25*((a['halves'][0]-b['halves'][0])-(a['halves'][1]-b['halves'][1]))**2


def read_group(ell,sigma,field_cache):
    args=configuration(ell,sigma)
    data={m:read(args,m,BASELINE_REFRESH_STAGE if (ell,sigma)==GROUPS[0] and m=='global_conditional' else None) for m in MODES}
    fixed=read(args,'gpis_fixed');original=source()
    expected_cov=dict(type='squared_exponential',lengthScale=ell,sigma=sigma)
    for mode,d in list(data.items())+[('gpis_fixed',fixed)]:
        s=d['scene'];m=s['media'][0];r=d['source']['result']
        assert r['resolution']==[256,256] and r['spp']==(32 if mode=='gpis_fixed' else 128)
        assert s['primitives']==original['primitives'] and s['bsdfs']==original['bsdfs']
        camera=copy.deepcopy(s['camera']);camera['resolution']=original['camera']['resolution']
        assert camera==original['camera']
        cov=m['covariance'] if mode=='classic_local' else m['gaussian_process']['covariance']
        assert cov==expected_cov
        if mode!='classic_local': assert m['gaussian_process']['mean']==original['media'][0]['gaussian_process']['mean']
        material=m['phase_function']['bsdf'] if mode in ('sparse_3d_global','gpis_fixed') else m['material']
        assert material==original['media'][0]['phase_function']['bsdf']
        if mode in ('sparse_3d_global','gpis_fixed'):
            assert m['step_size']==.01 and m['correlation_context']=='global'
            assert m['single_realization']==(mode=='gpis_fixed') and m['seed']==1
            assert not m['1D_sampling'] and not m['isotropic_3D_sampling'] and m['impulse_density']==10
    field=field_path(args);metadata=load(Path(str(field)+'.json'))
    if field not in field_cache: field_cache[field]=digest(field)
    assert field_cache[field]==metadata['field_sha256']==data['classic_local']['source']['result']['provenance']['field_sha256']
    assert metadata['sigma']==sigma and metadata['dx']==.005
    assert metadata['source_original_sha256']==digest(SOURCE)
    assert metadata['source_mean']==original['media'][0]['gaussian_process']['mean']
    assert metadata['source_scene_sha256']==digest(Path(metadata['source_scene']))
    base,half=[read(args,'sparse_3d_global',stage) for stage in ('step_base','step_half')]
    a,b=copy.deepcopy(base['scene']),copy.deepcopy(half['scene'])
    steps=[s['media'][0].pop('step_size') for s in (a,b)]
    for s in (a,b): s['renderer'].pop('output_directory')
    assert a==b and steps==[.01,.005]
    assert base['source']['result']['provenance']==half['source']['result']['provenance']
    assert base['source']['result']['resolution']==[96,96] and base['source']['result']['spp']==32
    checks=[];variance=paired_variance(half,base)
    for region,mask in regions(96).items():
        delta=float((half['y']-base['y'])[mask].mean());ref=float(base['y'][mask].mean())
        se=float(np.sqrt(variance[mask].sum())/mask.sum())
        checks.append(dict(region=region,steps=steps,resolution=[96,96],spp=32,
            relative_difference=delta/ref,difference=delta,paired_standard_error=se,
            exceeds_approximate_95_percent_sampling_interval=abs(delta)>1.96*se))
    return dict(ell=ell,sigma=sigma,gradient_stddev=sigma/ell,output=str(output(args)),
        data=data,fixed=fixed,field=metadata,step_checks=checks,
        step_sources=[base['source'],half['source']])


def measure(group,pair_modes=PAIR_MODES):
    data=group['data'];ref=data['classic_global'];stats=[];pairs=[]
    for mode,d in data.items():
        variance=.25*(d['halves'][0]-d['halves'][1])**2
        diff_var=paired_variance(d,ref)
        for region,mask in regions(256).items():
            n=int(mask.sum());mean=float(d['y'][mask].mean());baseline=float(ref['y'][mask].mean())
            stats.append(dict(ell=group['ell'],sigma=group['sigma'],mode=mode,region=region,
                pixels=n,mean_luminance=mean,standard_error=float(np.sqrt(variance[mask].sum())/n),
                difference=mean-baseline,relative_difference=mean/baseline-1,
                paired_standard_error=float(np.sqrt(diff_var[mask].sum())/n)))
    reference_mean=float(ref['y'][regions(256)['shaderball_box']].mean())
    mask=regions(16)['shaderball_box']
    for mode,reference in pair_modes:
        d,r=data[mode],data[reference]
        delta=blocks((d['y']-r['y'])[:,:,None])[:,:,0]
        halves=[blocks((a-b)[:,:,None])[:,:,0] for a,b in zip(d['halves'],r['halves'])]
        variance=.25*(halves[0]-halves[1])**2
        rms=float(np.sqrt(np.mean(delta[mask]**2)));noise=float(np.sqrt(np.mean(variance[mask])))
        pairs.append(dict(ell=group['ell'],sigma=group['sigma'],mode=mode,reference=reference,
            region='shaderball_box',block_grid=[16,16],observed_luminance_rms=rms,
            paired_half_noise_rms_estimate=noise,normalization_mean=reference_mean,
            relative_rms=rms/reference_mean,relative_noise_rms=noise/reference_mean))
    return stats,pairs


def render_main_grids(groups):
    for kind in ('appearance','detail','block_means'):
        fig,axes=plt.subplots(3,4,figsize=(16,13))
        for row,group in enumerate(groups):
            for col,mode in enumerate(MODES):
                d=group['data'][mode];ax=axes[row,col]
                if kind=='block_means': im=filmic(blocks(d['rgb']))
                else:
                    im=np.asarray(Image.open(Path(d['source']['folder'])/'render.png').convert('RGB'))
                    x0,y0,x1,y1=[int(v*256) for v in ROI]
                    if kind=='detail': im=im[y0:y1,x0:x1]
                    else: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=.8,edgecolor='#ffcf52'))
                ax.imshow(im,interpolation='nearest');ax.axis('off')
                if row==0: ax.set_title(LABELS[col],color=COLORS[col],fontsize=13)
                if col==0:
                    ax.text(-.07,.5,f'{NAMES[row]}\nℓ={group["ell"]:g}\nσ={group["sigma"]:g}\nσ/ℓ={group["gradient_stddev"]:g}',
                        transform=ax.transAxes,ha='right',va='center',fontsize=11)
        title={'appearance':'四模式平均外观','detail':'环形凹槽与球身细节','block_means':'16×16 分区平均外观'}[kind]
        fig.suptitle(f'原 GPIS shaderball · {title}\n相同相机、灯光、绿色镜面材质 · 256×256，128 spp',fontsize=17)
        note='统一 filmic 显示，无降噪或逐图曝光调整。基准 conditional 已重跑；A、B 为本次新渲染。'
        if kind=='block_means': note+=' 分区平均先在线性 RGB 中计算。'
        fig.text(.13,.025,note,fontsize=10,color='#555b65')
        fig.subplots_adjust(left=.14,right=.99,top=.91,bottom=.06,wspace=.035,hspace=.10)
        fig.savefig(REPORT/f'{kind}.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(3,3,figsize=(14,13))
    for row,group in enumerate(groups):
        ref=group['data']['classic_global']
        norm=float(ref['y'][regions(256)['shaderball_box']].mean())
        for col,mode in enumerate(('classic_local','global_conditional','sparse_3d_global')):
            ax=axes[row,col]
            delta=blocks((group['data'][mode]['y']-ref['y'])[:,:,None])[:,:,0]/norm
            im=ax.imshow(delta,cmap='RdBu_r',vmin=-.25,vmax=.25,interpolation='nearest');ax.axis('off')
            if row==0: ax.set_title('GPIS global ensemble' if mode=='sparse_3d_global' else mode,fontsize=12)
            if col==0: ax.text(-.08,.5,f'{NAMES[row]}\n{group_label(group)}',transform=ax.transAxes,ha='right',va='center',fontsize=10)
    fig.suptitle('各组相对 classic_global 的分区亮度差',fontsize=18)
    fig.colorbar(im,cax=fig.add_axes([.92,.22,.014,.56]),label='ΔY / 本组 classic_global 模型区域平均亮度')
    fig.text(.14,.025,'统一色标 ±25%，超出范围截断显示；16×16 分区，包含采样噪声。矩形统计区域与基准保持一致。',fontsize=10,color='#555b65')
    fig.subplots_adjust(left=.17,right=.89,top=.92,bottom=.06,wspace=.045,hspace=.10)
    fig.savefig(REPORT/'difference.png',dpi=150);plt.close(fig)


def render_realizations(groups):
    fig,axes=plt.subplots(2,3,figsize=(14,10))
    for col,group in enumerate(groups):
        for row,d in enumerate((group['data']['sparse_3d_global'],group['fixed'])):
            ax=axes[row,col]
            ax.imshow(Image.open(Path(d['source']['folder'])/'render.png'),interpolation='nearest');ax.axis('off')
            if row==0: ax.set_title(f'{NAMES[col]}\n{group_label(group)}',fontsize=12)
            if col==0: ax.text(-.06,.5,'平均外观\n128 spp' if row==0 else '固定随机场\n种子 1\n32 spp',transform=ax.transAxes,ha='right',va='center',fontsize=12)
    fig.suptitle('GPIS global：平均外观与固定场实现',fontsize=18)
    fig.text(.12,.025,'固定场图展示一次实际表面实现，与平均外观的统计目标不同；各参数组统一显示。',fontsize=11,color='#555b65')
    fig.subplots_adjust(left=.12,right=.99,top=.88,bottom=.07,wspace=.04,hspace=.06)
    fig.savefig(REPORT/'realizations.png',dpi=150);plt.close(fig)


def figures():
    REPORT.mkdir(parents=True,exist_ok=True)
    field_cache={};groups=[read_group(*p,field_cache) for p in GROUPS]
    all_sources=[];reference=groups[0]['data']['classic_global']['scene']
    for group in groups:
        for d in list(group['data'].values())+[group['fixed']]:
            all_sources.append(d['source'])
            for key in ('camera','integrator','renderer'):
                a,b=copy.deepcopy(d['scene'][key]),copy.deepcopy(reference[key])
                if key=='renderer':
                    for v in (a,b):
                        for setting in ('output_directory','spp','spp_step'): v.pop(setting)
                assert a==b
        all_sources.extend(group['step_sources'])
    compatibility=load(REPORT/'renderer_compatibility.json')
    current=compatibility['current_renderer_sha256'];previous=compatibility['previous_renderer_sha256']
    assert {s['result']['provenance']['renderer_sha256'] for s in all_sources}=={previous,current}
    for group in groups:
        for mode,d in list(group['data'].items())+[('gpis_fixed',group['fixed'])]:
            expected=current if group['ell']==.2 or mode=='global_conditional' else previous
            assert d['source']['result']['provenance']['renderer_sha256']==expected
        for s in group['step_sources']:
            assert s['result']['provenance']['renderer_sha256']==(current if group['ell']==.2 else previous)
    assert {c['mode'] for c in compatibility['nonconditional_checks']}=={'classic_local','classic_global','sparse_3d_global','gpis_fixed'}
    for c in compatibility['nonconditional_checks']:
        for name,h in c['equal_images'].items():
            assert digest(Path(c['before'])/name)==digest(Path(c['after'])/name)==h
    statistics=[];pairs=[]
    for group in groups:
        a,b=measure(group);statistics.extend(a);pairs.extend(b)
    render_main_grids(groups)
    render_realizations(groups)
    manifest_groups=[]
    for i,g in enumerate(groups):
        manifest_groups.append(dict(name=NAMES[i],ell=g['ell'],sigma=g['sigma'],gradient_stddev=g['gradient_stddev'],
            output=g['output'],sources=[d['source'] for d in g['data'].values()],
            fixed_realization_source=g['fixed']['source'],field=g['field'],
            step_checks=g['step_checks'],step_sources=g['step_sources']))
    figure_names=[f'{n}.png' for n in ('appearance','detail','block_means','difference','realizations')]
    manifest=dict(source=str(SOURCE),source_sha256=digest(SOURCE),groups=manifest_groups,
        resolution=[256,256],spp=128,fixed_spp=32,statistics=statistics,pair_metrics=pairs,
        renderer_compatibility=compatibility,
        model_roi=MODEL_ROI,detail_roi=ROI,figures=figure_names,
        notes=['Fixed image rectangles, not segmented object masks. Linear luminance uses Rec.709 weights.',
            'Block RMS differences contain noise. Paired-half noise estimates are descriptive, not confidence intervals.',
            'Classic_local interpolates density and normalizes the mean gradient; other modes use the native procedural mean.',
            'Two GPIS steps do not establish convergence. Zero-length ray diagnostics are retained, not treated as unique discarded paths.'])
    save(REPORT/'manifest.json',manifest)
    for name,rows in [('metrics',statistics),('pair_metrics',pairs)]:
        with (REPORT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    write_page(groups,statistics,pairs)
    print(REPORT/'comparison.html',flush=True)


def write_page(groups,statistics,pairs):
    def link(path):
        return '../'+Path(path).relative_to(REPORT.parent).as_posix()
    def metric(group,mode,reference):
        return next(r for r in pairs if (r['ell'],r['sigma'],r['mode'],r['reference'])==
            (group['ell'],group['sigma'],mode,reference))
    distance_rows='';mean_rows='';raw_rows='';step_rows='';diagnostic_rows=''
    for i,g in enumerate(groups):
        label=f'{NAMES[i]}<br>{group_label(g)}'
        cells=[]
        for mode,reference in [('global_conditional','classic_global'),('sparse_3d_global','classic_global'),('global_conditional','sparse_3d_global'),('classic_local','sparse_3d_global')]:
            r=metric(g,mode,reference)
            cells.append(f'<td>{100*r["relative_rms"]:.2f}%<small>噪声 RMS 约 {100*r["relative_noise_rms"]:.2f}%</small></td>')
        distance_rows+=f'<tr><td>{label}</td>'+''.join(cells)+'</tr>'
        links=[]
        for mode in MODES:
            r=next(r for r in statistics if (r['ell'],r['sigma'],r['mode'],r['region'])==(g['ell'],g['sigma'],mode,'shaderball_box'))
            mean_rows+=f'<tr><td>{label}</td><td>{mode}</td><td>{r["mean_luminance"]:.6f} ± {1.96*r["standard_error"]:.6f}</td><td>{100*r["relative_difference"]:+.2f}%</td></tr>'
            d=Path(g['data'][mode]['source']['folder'])
            links.append(f'<td><a href="{link(d/"render.png")}">PNG</a> · <a href="{link(d/"scene.json")}">场景</a></td>')
        fixed=Path(g['fixed']['source']['folder'])
        raw_rows+=f'<tr><td>{label}</td>'+''.join(links)+f'<td><a href="{link(fixed/"render.png")}">固定场 PNG</a></td></tr>'
        for check in g['step_checks']:
            step_rows+=f'<tr><td>{label}</td><td>{check["region"]}</td><td>{100*check["relative_difference"]:+.3f}%</td><td>{check["difference"]:+.6f} ± {1.96*check["paired_standard_error"]:.6f}</td></tr>'
        discards=g['data']['global_conditional']['source']['result']['discarded_surface_births']
        zero=g['data']['sparse_3d_global']['source']['log_diagnostics']['zero_length_ray_messages']
        fixed_zero=g['fixed']['source']['log_diagnostics']['zero_length_ray_messages']
        diagnostic_rows+=f'<tr><td>{label}</td><td>{discards["paths"]} / {discards["shadow_connections"]}</td><td>{zero} / {fixed_zero}</td><td>{g["field"]["near_surface_interpolation_error"]["value_rmse"]:.7f}</td></tr>'
    base=metric(groups[0],'global_conditional','classic_global')['relative_rms']
    ratios=[metric(g,'global_conditional','classic_global')['relative_rms']/base for g in groups[1:]]
    neural_link=('<p><a href="../ell0p2_neural_comparison/comparison.html">查看加入 neural 的五模式对比</a></p>'
        if (REPORT.parent/'ell0p2_neural_comparison/comparison.html').exists() else '')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GPIS shaderball · 两组长相关长度对比</title>
<style>body{{max-width:1500px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;color:#253145;background:#f5f6f8}}h1{{font-size:28px}}h2{{font-size:23px;margin-top:36px}}img{{width:100%;background:white}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:10px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,th:first-child{{text-align:left}}small{{display:block;font-size:12px;color:#657083}}.note{{background:#e7edf5;padding:16px;border-radius:10px}}details{{margin-top:25px}}summary{{cursor:pointer;font-weight:600}}code{{font-size:14px}}</style>
<h1>原 GPIS shaderball：ℓ=0.2 的两组对比</h1>
{neural_link}
<p>使用 gpis-scenes/shaderball/shaderball-stationary-270.json 的原程序化 knob、底座、地面、相机、灯光和绿色镜面材质。四模式主图均为 256×256、128 spp；相同 filmic 显示，无降噪。GPIS 使用沿路径保持同一随机场的 3D global 稀疏卷积。</p>
<div class="note"><b>A：ℓ=0.2、σ=0.02</b>，只加长相关长度，梯度标准差 σ/ℓ 从 0.2 降到 0.1。<br><b>B：ℓ=0.2、σ=0.04</b>，保持梯度扰动幅度 σ/ℓ=0.2，同时增大表面位移幅度。<br>基准为已有 ℓ=0.1、σ=0.02 结果。B 的变化不能单独归因于相关长度；对于本模型非单位均值梯度区域，σ/ℓ 也不等同于最终法线分布的宽度。</div>
<p><a href="../ell0p1/comparison.html">原基准对比页</a> · <a href="manifest.json">参数、来源与核验数据</a> · <a href="metrics.csv">区域亮度统计</a> · <a href="pair_metrics.csv">模式间分区差异</a></p>
<h2>四模式平均外观</h2><a href="appearance.png"><img src="appearance.png"></a>
<h2>环形凹槽与球身细节</h2><a href="detail.png"><img src="detail.png"></a>
<h2>模式差异是否变大</h2>
<p>下面比较固定模型矩形内 16×16 分区的亮度 RMS 差，除以本组 classic_global 的区域平均亮度。conditional 与 classic_global 的相对 RMS 差，A 为基准的 <b>{ratios[0]:.2f} 倍</b>，B 为基准的 <b>{ratios[1]:.2f} 倍</b>。这是本次观察值，仍含采样噪声。</p>
<table><tr><th>参数组</th><th>conditional ↔ classic_global</th><th>GPIS ↔ classic_global</th><th>conditional ↔ GPIS</th><th>classic_local ↔ GPIS</th></tr>{distance_rows}</table>
<p>噪声 RMS 使用配对奇偶两半样本估计，属于描述性统计，不是置信区间，也没有从观察差值中扣除。GPIS 在这里是当前实现的参考结果，未视为已收敛真值。</p>
<a href="difference.png"><img src="difference.png"></a>
<details><summary>查看分区平均外观</summary><a href="block_means.png"><img src="block_means.png"></a></details>
<h2>固定随机场补图</h2><a href="realizations.png"><img src="realizations.png"></a>
<p>固定场图为 256×256、32 spp，所有像素和路径共享场种子 1。它展示单次表面实现，与上方平均外观的统计目标不同。</p>
<h2>原始图与场景配置</h2>
<table><tr><th>参数组</th><th>classic_local</th><th>classic_global</th><th>global_conditional</th><th>GPIS global</th><th>补图</th></tr>{raw_rows}</table>
<details><summary>区域亮度、步长与实现核验</summary>
<p>区域为 x∈[{MODEL_ROI[0]},{MODEL_ROI[2]})、y∈[{MODEL_ROI[1]},{MODEL_ROI[3]}) 的固定图像矩形，包含模型及部分背景，不是物体分割。亮度使用线性 RGB 的 Rec.709 权重；± 为近似 95% 采样误差。</p>
<table><tr><th>参数组</th><th>模式</th><th>区域平均亮度</th><th>相对本组 classic_global</th></tr>{mean_rows}</table>
<p>GPIS 保留每格 10 个脉冲、主图步长 0.01。下表为 96×96、32 spp 下将步长减半到 0.005 的变化；两档检查不能证明收敛。</p>
<table><tr><th>参数组</th><th>区域</th><th>相对变化</th><th>亮度变化 ± 近似95%采样误差</th></tr>{step_rows}</table>
<p>classic_local 使用原几何的 0.005 网格，插值密度并将均值梯度单位化；其余三种模式使用原程序化几何与梯度。σ=0.02 复用已有网格，σ=0.04 重新烘焙对应密度，两份网格均验证了 4096 个持久化节点及 100000 个插值查询。</p>
<p>本轮修复了 conditional 几何端点的浮点位置不一致问题，避免将已更新观测的转向误判为非法状态。A、B 使用同一修复版；基准 conditional 也已重跑。其余基准图经新旧版同参数逐像素核验后复用，详见 <a href="renderer_compatibility.json">渲染器版本与核验记录</a>。</p>
<table><tr><th>参数组</th><th>conditional 丢弃路径 / 阴影连接</th><th>GPIS farT==0 消息：平均 / 固定</th><th>近表面均值网格 RMS 误差</th></tr>{diagnostic_rows}</table>
<p>farT==0 日志语句本身不终止路径；消息次数不能直接换算成丢弃路径比例。边界或自交影响未单独排除。采样误差不包括网格插值、有限脉冲或模型近似误差。</p>
<p>复现：<code>python -B tools/surface_compare/gpis_shaderball_sweep.py run</code></p>
</details></html>'''
    (REPORT/'comparison.html').write_text(html,encoding='utf-8')
