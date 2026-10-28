"""Figures and checks for gpis_shaderball.py; uses the plotting Python environment."""
import copy
import csv
from pathlib import Path
import numpy as np
from gpis_shaderball import ROOT, OUT, SOURCE, FIELD, MODES, LABELS, ROI, MODEL_ROI, WEIGHTS, folder, source, scene
from run import load, save, pfm
from shaderball_length import digest
from figures import plt, Image, Rectangle, COLORS
from plane_figures import filmic, blocks


def regions(n):
    result={'full':np.ones((n,n),bool)}
    for name,rect in [('shaderball_box',MODEL_ROI),('detail',ROI)]:
        x0,y0,x1,y1=[int(v*n) for v in rect]
        mask=np.zeros((n,n),bool);mask[y0:y1,x0:x1]=True;result[name]=mask
    return result


def read(args,mode,stage=None):
    a=copy.copy(args)
    if stage: a.stage=stage
    d=folder(a,mode);r=load(d/'result.json');s=load(d/'scene.json')
    for name,h in r['sha256'].items(): assert digest(d/name)==h,str(d/name)
    assert r['provenance']['source_sha256']==digest(SOURCE)
    a.size=r['resolution'][0];a.spp=r['spp']
    a.step=s['media'][0].get('step_size',.01)
    assert s==scene(a,mode),str(d)
    rgb=pfm(d/'render.pfm')
    assert rgb.shape==(*r['resolution'],3)
    halves=[pfm(d/f'samples{c}.pfm')@WEIGHTS for c in ('A','B')]
    log=(d/'render.log').read_text(encoding='utf-8',errors='replace')
    diagnostics=dict(zero_length_ray_messages=log.count('Path tracing bounces ended with farT == 0'),
        segment_exit_messages=log.count('Reached end of ray, but not counting it as exited'),
        log_sha256=digest(d/'render.log'))
    return dict(rgb=rgb,y=rgb@WEIGHTS,halves=halves,scene=s,
        source=dict(mode=mode,folder=str(d),result=r,log_diagnostics=diagnostics))


def figures(args):
    data={m:read(args,m) for m in MODES}
    fixed=read(args,'gpis_fixed')
    original=source();ref=data['classic_global'];n=ref['y'].shape[0]
    sources=[d['source'] for d in data.values()]
    renderer_hashes={s['result']['provenance']['renderer_sha256'] for s in sources+[fixed['source']]}
    assert len(renderer_hashes)==1
    assert {(tuple(s['result']['resolution']),s['result']['spp']) for s in sources}=={((256,256),128)}
    assert fixed['y'].shape==ref['y'].shape
    for mode,d in data.items():
        s=d['scene'];m=s['media'][0]
        for key in ('primitives','bsdfs'): assert s[key]==original[key]
        camera=copy.deepcopy(s['camera']);camera['resolution']=original['camera']['resolution']
        assert camera==original['camera']
        for key in ('camera','integrator','renderer'):
            a,b=copy.deepcopy(s[key]),copy.deepcopy(ref['scene'][key])
            if key=='renderer': a.pop('output_directory');b.pop('output_directory')
            assert a==b
        cov=m['covariance'] if mode=='classic_local' else m['gaussian_process']['covariance']
        assert cov==dict(type='squared_exponential',lengthScale=.1,sigma=.02)
        material=m['phase_function']['bsdf'] if mode=='sparse_3d_global' else m['material']
        assert material==original['media'][0]['phase_function']['bsdf']
        if mode!='classic_local': assert m['gaussian_process']==ref['scene']['media'][0]['gaussian_process']
    for mode,d in [('sparse_3d_global',data['sparse_3d_global']),('gpis_fixed',fixed)]:
        m=d['scene']['media'][0]
        assert m['correlation_context']=='global' and m['single_realization']==(mode=='gpis_fixed')
        assert not m['1D_sampling'] and not m['isotropic_3D_sampling'] and m['impulse_density']==10
    metadata=load(Path(str(FIELD)+'.json'))
    assert digest(FIELD)==metadata['field_sha256']==data['classic_local']['source']['result']['provenance']['field_sha256']
    assert metadata['source_original_sha256']==digest(SOURCE)
    masks=regions(n);stats=[];spatial=[]
    for mode,d in data.items():
        delta=d['y']-ref['y']
        variance=.25*(d['halves'][0]-d['halves'][1])**2
        pairs=[a-b for a,b in zip(d['halves'],ref['halves'])]
        delta_variance=.25*(pairs[0]-pairs[1])**2
        for name,mask in masks.items():
            count=int(mask.sum());mean=float(d['y'][mask].mean());baseline=float(ref['y'][mask].mean())
            stats.append(dict(mode=mode,region=name,pixels=count,mean_luminance=mean,
                standard_error=float(np.sqrt(variance[mask].sum())/count),
                difference=float(delta[mask].mean()),relative_difference=mean/baseline-1,
                paired_standard_error=float(np.sqrt(delta_variance[mask].sum())/count)))
    def stat(mode,region='shaderball_box'):
        return next(r for r in stats if (r['mode'],r['region'])==(mode,region))
    fig,axes=plt.subplots(2,4,figsize=(18,9),gridspec_kw={'height_ratios':[1,.67]})
    for i,mode in enumerate(MODES):
        im=np.asarray(Image.open(Path(data[mode]['source']['folder'])/'render.png').convert('RGB'))
        x0,y0,x1,y1=[int(v*n) for v in ROI]
        axes[0,i].imshow(im,interpolation='nearest')
        axes[0,i].add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,lw=1,edgecolor='#ffcf52'))
        axes[0,i].set_title(LABELS[i],color=COLORS[i],fontsize=13)
        axes[1,i].imshow(im[y0:y1,x0:x1],interpolation='nearest')
        for ax in axes[:,i]: ax.axis('off')
    fig.suptitle('原 GPIS shaderball 场景 · ℓ=0.1，σ=0.02\n原相机、灯光、地面与绿色镜面材质 · 四模式平均外观 · 256×256，128 spp',fontsize=18)
    fig.text(.035,.035,'下排为黄色框内原图放大；统一 filmic 显示，无降噪或逐图调整曝光。GPIS 在每条路径内保留同一随机场，不同路径采用不同实现。\nclassic_local 使用原几何的 0.005 网格；其余三种模式直接使用原程序化 knob。',fontsize=10,color='#555b65')
    fig.subplots_adjust(left=.025,right=.99,top=.83,bottom=.11,wspace=.045,hspace=.05)
    fig.savefig(OUT/'appearance.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8))
    reference_mean=stat('classic_global')['mean_luminance']
    for ax,mode in zip(axes,('classic_local','global_conditional','sparse_3d_global')):
        d=data[mode]
        delta=blocks((d['y']-ref['y'])[:,:,None])[:,:,0]/reference_mean
        halves=[blocks((a-b)[:,:,None])[:,:,0]/reference_mean for a,b in zip(d['halves'],ref['halves'])]
        mask=regions(16)['shaderball_box'];noise=.25*(halves[0]-halves[1])**2
        spatial.append(dict(mode=mode,observed_rms=float(np.sqrt(np.mean(delta[mask]**2))),
            noise_rms_estimate=float(np.sqrt(np.mean(noise[mask])))))
        im=ax.imshow(delta,cmap='RdBu_r',vmin=-.25,vmax=.25,interpolation='nearest');ax.axis('off')
        ax.set_title('GPIS global ensemble' if mode=='sparse_3d_global' else mode)
    fig.suptitle('相对 classic_global 的 16×16 分区亮度差',fontsize=17)
    fig.colorbar(im,cax=fig.add_axes([.935,.23,.012,.53]),label='ΔY / 参考模型区域平均亮度')
    fig.text(.035,.04,'在线性亮度中求分区平均与差值；统一色标 ±25%，超出范围截断显示。图中仍含采样噪声。',fontsize=10,color='#555b65')
    fig.subplots_adjust(left=.02,right=.91,top=.82,bottom=.16,wspace=.04)
    fig.savefig(OUT/'difference.png',dpi=160);plt.close(fig)
    gpis_comparison=[]
    gpis=data['sparse_3d_global']
    for mode in MODES[:-1]:
        d=data[mode]
        delta=blocks((d['y']-gpis['y'])[:,:,None])[:,:,0]
        halves=[blocks((a-b)[:,:,None])[:,:,0] for a,b in zip(d['halves'],gpis['halves'])]
        mask=regions(16)['shaderball_box'];noise=.25*(halves[0]-halves[1])**2
        gpis_comparison.append(dict(mode=mode,region='shaderball_box',block_grid=[16,16],
            observed_luminance_rms=float(np.sqrt(np.mean(delta[mask]**2))),
            paired_half_noise_rms_estimate=float(np.sqrt(np.mean(noise[mask])))))
    fig,axes=plt.subplots(1,4,figsize=(16,5))
    for i,mode in enumerate(MODES):
        axes[i].imshow(filmic(blocks(data[mode]['rgb'])),interpolation='nearest');axes[i].axis('off')
        axes[i].set_title(LABELS[i],color=COLORS[i],fontsize=12)
    fig.suptitle('16×16 分区平均外观',fontsize=18)
    fig.text(.025,.025,'先在线性 RGB 中平均，再应用相同 filmic 显示；用于观察大范围明暗变化。',fontsize=10,color='#555b65')
    fig.subplots_adjust(left=.015,right=.99,top=.84,bottom=.1,wspace=.03)
    fig.savefig(OUT/'block_means.png',dpi=150);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,6.5))
    for ax,d,title in zip(axes,[data['sparse_3d_global'],fixed],['平均外观：逐路径采样不同场 · 128 spp',f'固定随机场：场种子 1 · {fixed["source"]["result"]["spp"]} spp']):
        ax.imshow(Image.open(Path(d['source']['folder'])/'render.png'),interpolation='nearest');ax.axis('off');ax.set_title(title,fontsize=12)
    fig.suptitle('原场景的 GPIS：ensemble 与 single_realization',fontsize=17)
    fig.text(.035,.035,'两图均为 global 3D 稀疏卷积，ℓ=0.1、σ=0.02、每格 10 个脉冲；统一显示。\n右图的所有像素与路径共享一个随机场，展示一次实际表面实现；它不是左侧统计平均的真值。',fontsize=10,color='#555b65')
    fig.subplots_adjust(left=.025,right=.99,top=.84,bottom=.14,wspace=.04)
    fig.savefig(OUT/'realizations.png',dpi=150);plt.close(fig)
    base,half=[read(args,'sparse_3d_global',stage) for stage in ('step_base','step_half')]
    a,b=copy.deepcopy(base['scene']),copy.deepcopy(half['scene'])
    steps=[s['media'][0].pop('step_size') for s in (a,b)]
    for s in (a,b): s['renderer'].pop('output_directory')
    assert a==b and steps==[.01,.005]
    assert base['source']['result']['provenance']==half['source']['result']['provenance']
    assert base['source']['result']['provenance']['renderer_sha256'] in renderer_hashes
    delta=half['y']-base['y'];pairs=[a-b for a,b in zip(half['halves'],base['halves'])]
    variance=.25*(pairs[0]-pairs[1])**2;checks=[]
    for name,mask in regions(delta.shape[0]).items():
        d=float(delta[mask].mean());se=float(np.sqrt(variance[mask].sum())/mask.sum());baseline=float(base['y'][mask].mean())
        checks.append(dict(region=name,steps=steps,resolution=base['source']['result']['resolution'],spp=base['source']['result']['spp'],
            difference=d,relative_difference=d/baseline,paired_standard_error=se,baseline_mean=baseline,
            exceeds_approximate_95_percent_sampling_interval=abs(d)>1.96*se))
    validation=dict(source_preserved=dict(camera_except_resolution=True,primitives_and_lights=True,bsdfs=True,
        mean_for_three_native_modes=True,sigma_and_mirror=True),field=metadata,
        discarded_surface_births=data['global_conditional']['source']['result']['discarded_surface_births'],
        log_diagnostics={d['source']['mode']:d['source']['log_diagnostics'] for d in list(data.values())+[fixed]},
        gpis_step_checks=checks,step_sources=[base['source'],half['source']],
        gradient_methods=dict(classic_local='Normalized derivative of interpolated native knob samples',
            classic_global='Native shared procedural knob derivative',global_conditional='Native shared procedural knob derivative',
            sparse_3d_global='Native shared procedural knob derivative plus realized noise gradient'),
        uncertainty='Approximate sampling errors from paired even/odd halves of linear luminance. Excludes field interpolation, model approximations and finite-step bias. Two steps do not establish convergence.')
    save(OUT/'validation.json',validation)
    save(OUT/'manifest.json',dict(source=str(SOURCE),source_sha256=digest(SOURCE),length_scale=.1,sigma=.02,
        resolution=[n,n],spp=128,sources=sources,fixed_realization_source=fixed['source'],
        statistics=stats,spatial_block_statistics=spatial,detail_roi=ROI,model_roi=MODEL_ROI,
        gpis_comparison=gpis_comparison,
        region_note='Fixed image rectangles, not segmented object masks. Linear luminance uses Rec.709 RGB weights.',
        spatial_note='RMS block differences divided by reference model-region mean; paired-half noise estimate is descriptive, not a significance test.',
        figures=['appearance.png','difference.png','block_means.png','realizations.png']))
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    rows=''
    for mode in MODES:
        r=stat(mode);err=196*r['paired_standard_error']/reference_mean
        rows+=f'<tr><td>{mode}</td><td>{r["mean_luminance"]:.6f} ± {1.96*r["standard_error"]:.6f}</td><td>{100*r["relative_difference"]:+.2f}% ± {err:.2f}%</td></tr>'
    step_rows=''.join(f'<tr><td>{r["region"]}</td><td>{100*r["relative_difference"]:+.3f}%</td><td>{r["difference"]:+.6f} ± {1.96*r["paired_standard_error"]:.6f}</td></tr>' for r in checks)
    gpis_rows=''.join(f'<tr><td>{r["mode"]}</td><td>{r["observed_luminance_rms"]:.6f}</td><td>{r["paired_half_noise_rms_estimate"]:.6f}</td></tr>' for r in gpis_comparison)
    discard=validation['discarded_surface_births']
    gpis_zero=data['sparse_3d_global']['source']['log_diagnostics']['zero_length_ray_messages']
    fixed_zero=fixed['source']['log_diagnostics']['zero_length_ray_messages']
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>原 GPIS shaderball · ℓ=0.1</title>
<style>body{{max-width:1500px;margin:35px auto;padding:0 24px;font:16px/1.65 system-ui;color:#253145;background:#f5f6f8}}h1{{font-size:28px}}img{{width:100%;background:white}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child{{text-align:left}}.note{{background:#e7edf5;padding:16px;border-radius:10px}}</style>
<h1>gpis-scenes/shaderball：ℓ=0.1 四模式对照</h1>
<p>来源为 shaderball-stationary-270.json；使用原程序化 knob、底座、地面、相机、灯光及绿色镜面反射率 [0.15, 0.8, 0.5]。保留原 σ=0.02，将相关长度改为 0.1。主对比统一 256×256、128 spp。</p>
<div class="note">四模式主图比较平均外观。GPIS 为 3D global 稀疏卷积，每格 10 个脉冲、步长 0.01，同一路径保持同一随机场，不同路径采样不同实现。原文件的 single_realization=true 另作补图，场种子仍为 1。<br>GPIS、classic_global 和 conditional 共用原程序化几何与梯度；classic_local 使用这份几何烘焙的 0.005 网格，插值密度并将均值梯度单位化。原场景的平均几何并非处处严格单位梯度，因此这项 local/global 区别也会影响外观。</div>
<p><a href="source.json">原场景配置快照</a> · <a href="manifest.json">参数与源文件</a> · <a href="metrics.csv">亮度统计</a> · <a href="validation.json">网格、参数及步长验证</a></p>
<a href="appearance.png"><img src="appearance.png"></a>
<h2>大范围变化与分区差值</h2><a href="block_means.png"><img src="block_means.png"></a><a href="difference.png"><img src="difference.png"></a>
<h2>GPIS 的固定随机场补图</h2><a href="realizations.png"><img src="realizations.png"></a>
<h2>线性亮度统计</h2><p>统计区域为图像归一化坐标 x∈[{MODEL_ROI[0]},{MODEL_ROI[2]})、y∈[{MODEL_ROI[1]},{MODEL_ROI[3]}) 的固定矩形，包含模型及部分背景。以 classic_global 为基准，± 为近似 95% 采样误差；相对差的误差单位为百分点。不同算法噪声水平不同，GPIS 的镜面事件不使用普通光源 NEE。</p>
<table><tr><th>模式</th><th>区域平均亮度</th><th>相对差 ± 采样误差</th></tr>{rows}</table>
<h2>与本次 GPIS 平均外观的距离</h2><p>在线性亮度中求 16×16 分区均值，再比较模型矩形内各分区与 GPIS 的 RMS 差。数值越小表示本次结果越接近；采样噪声 RMS 为配对两半样本的描述性估计，不是置信区间，也没有从观察差值中扣除。这里不将 GPIS 当作已收敛真值。</p>
<table><tr><th>模式</th><th>与 GPIS 的亮度 RMS 差</th><th>配对采样噪声 RMS 估计</th></tr>{gpis_rows}</table>
<h2>核验</h2><p>classic_local 的网格通过 4096 个持久化节点检查和 100000 个插值查询；表面附近均值 RMS 误差 {metadata['near_surface_interpolation_error']['value_rmse']:.7f}，σ=0.02。conditional 丢弃路径 {discard['paths']} 条、阴影连接 {discard['shadow_connections']} 条，未重试这些样本。</p>
<p>GPIS 日志中的 farT == 0 消息：平均图 {gpis_zero} 次，固定场图 {fixed_zero} 次。该日志语句本身不终止路径；这些是消息次数，不能直接换算成丢弃路径比例。边界或自交的影响未单独排除，结果用于比较当前实现。</p>
<p>GPIS 步长从 0.01 减到 0.005：{checks[0]['resolution'][0]}×{checks[0]['resolution'][1]}、{checks[0]['spp']} spp，同一随机种子。两档检查只说明步长敏感性，不能证明收敛，也未消除有限脉冲与模型近似误差。</p>
<table><tr><th>区域</th><th>相对变化</th><th>亮度变化 ± 近似95%采样误差</th></tr>{step_rows}</table></html>'''
    (OUT/'comparison.html').write_text(html,encoding='utf-8')
    print(OUT/'comparison.html',flush=True)
