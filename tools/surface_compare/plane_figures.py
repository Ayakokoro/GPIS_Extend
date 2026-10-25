"""Plots and a local comparison page for plane.py, using linear render outputs.

Main panels preserve the renderer's filmic curve. The same-side contact sheet
uses explicitly labelled, predetermined gains (1/8/32 by angle), shared by
all modes. A separate uniform-exposure sheet uses gain 1 everywhere. Spatial
block averages, when shown, are labelled and do not replace raw renders.
"""
from __future__ import annotations

import argparse
import csv
import numpy as np
from figures import plt, COLORS, STYLES, Image, disk_image, Normalize, Circle
from plane import OUT, ROOT, CASES, ANGLES, SIDES, RENDER_MODES, SEED, folder, load, save, sha, pfm

LABELS = ('classic_local','classic_global','global_conditional','GPIS 3D')
PARTS = ('total','single','multiple')
PART_LABELS = ('完整结果','单次散射','两次及以上散射')
GAINS = {'same_30':1.,'same_60':8.,'same_80':32.,
         'opposite_30':1.,'opposite_60':1.,'opposite_80':1.}


def filmic(rgb, gain=1.):
    x=np.maximum(0,np.asarray(rgb)*gain-.004)
    return np.clip(x*(6.2*x+.5)/(x*(6.2*x+1.7)+.06),0,1)


def write_pfm(path, rgb):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('wb') as f:
        f.write(f'PF\n{rgb.shape[1]} {rgb.shape[0]}\n-1\n'.encode('ascii'))
        f.write(np.flipud(rgb).astype('<f4').tobytes())


def blocks(rgb, n=16):
    h,w,c=rgb.shape
    assert h%n==0 and w%n==0
    return rgb.reshape(n,h//n,n,w//n,c).mean((1,3))


def collect(stage,seed):
    data={}; stats=[]; sources=[]; renderer_hashes=set(); resolution_spp=set()
    for case in CASES:
        data[case]={}
        for mode in RENDER_MODES:
            raw={}; halves={}; variance={}
            scenes=[]
            for part in ('total','single'):
                d=folder(stage,case,mode,part,seed)
                result=load(d/'result.json'); scene=load(d/'scene.json'); scenes.append(scene)
                assert sha(d/'render.pfm')==result['pfm_sha256']
                assert sha(d/'scene.json')==result['scene_sha256']
                assert sha(d/'render.png')==result['png_sha256']
                renderer_hashes.add(result['provenance']['renderer_sha256'])
                resolution_spp.add((tuple(result['size']),result['completed_spp']))
                raw[part]=pfm(d/'render.pfm')
                halves[part]=[pfm(d/f'samples{letter}.pfm') for letter in ('A','B')]
                variance[part]=pfm(d/'samplesVariance.pfm')
                assert np.allclose((halves[part][0]+halves[part][1])/2,raw[part],rtol=2e-6,atol=2e-7)
                # These scenes are achromatic; RGB channels are identical and
                # the scalar variance must not be divided by three.
                assert np.max(np.ptp(raw[part],axis=2))<1e-7
                sources.append(dict(case=case,mode=mode,component=part,folder=str(d),result=result,
                    variance_sha256=sha(d/'samplesVariance.pfm'),
                    split_sha256=[sha(d/f'samples{c}.pfm') for c in ('A','B')]))
            for key in ('media','camera','primitives'):
                assert scenes[0][key]==scenes[1][key]
            assert scenes[0]['integrator']['max_bounces']==128
            assert scenes[1]['integrator']['max_bounces']==2
            delta=raw['total']-raw['single']
            tolerance=2e-5*max(1.,float(raw['total'].max()))
            if float(delta.min()) < -tolerance:
                raise RuntimeError('Unpaired scattering renders: '+case+' '+mode)
            raw['multiple']=np.maximum(delta,0)
            halves['multiple']=[a-b for a,b in zip(halves['total'],halves['single'])]
            # Paired even/odd sample halves include the total/single covariance.
            # (A-B)^2/4 estimates the variance of the full sample mean.
            variance['multiple']=.25*(halves['multiple'][0]-halves['multiple'][1])**2
            target=OUT/stage/'derived'/case/mode
            write_pfm(target/'multiple.pfm',raw['multiple'])
            Image.fromarray(np.rint(255*filmic(raw['multiple'])).astype('uint8')).save(target/'multiple.png')
            data[case][mode]=raw
            for part in PARTS:
                a=raw[part][:,:,0]; v=variance[part][:,:,0]; h,w=a.shape
                for region, sl in (('full',(slice(None),slice(None))),
                                   ('center_half',(slice(h//4,3*h//4),slice(w//4,3*w//4)))):
                    pixels=a[sl].size
                    stats.append(dict(case=case,mode=mode,component=part,region=region,
                        mean=float(a[sl].mean()),standard_error=float(np.sqrt(v[sl].sum())/pixels),
                        pixels=pixels,nonzero_pixels=int(np.count_nonzero(a[sl])),
                        variance_method='paired_even_odd_halves' if part=='multiple' else 'per_sample_Welford',
                        gain=GAINS[case],minimum_paired_difference=float(delta.min())))
    assert len(renderer_hashes)==len(resolution_spp)==1
    size,spp=next(iter(resolution_spp))
    return data,stats,sources,dict(resolution=list(size),spp=spp,seed=seed,
        renderer_sha256=next(iter(renderer_hashes)))


def stat(stats,case,mode,part):
    return next(r for r in stats if (r['case'],r['mode'],r['component'],r['region'])==(case,mode,part,'full'))


def footer(fig,text):
    fig.text(.03,.025,text,fontsize=10,color='#4b535e')


def setup_figure(dest):
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    for ax,side in zip(axes,SIDES):
        ax.axhline(0,color='#475364',lw=2);ax.axhspan(-.08,.08,color='#b8cadc',alpha=.3)
        for i,angle in enumerate(ANGLES):
            theta=np.deg2rad(angle);v=np.array([np.sin(theta),np.cos(theta)])
            l=3*v*np.array([1 if side=='same' else -1,1])
            ax.plot([0,v[0]],[0,v[1]],color=COLORS[i],lw=1.8,label=f'{angle}°')
            ax.plot([0,l[0]],[0,l[1]],'--',color=COLORS[i],lw=1)
            ax.plot(*v,'o',color=COLORS[i],ms=5);ax.plot(*l,'*',color=COLORS[i],ms=13)
        ax.plot(0,0,'ko',ms=4);ax.text(-3.2,-.32,'均值平面 y=0',fontsize=10)
        ax.set(xlim=(-3.4,3.4),ylim=(-.45,3),aspect='equal',xlabel='x',ylabel='y',
               title='同侧光源' if side=='same' else '对侧光源')
        ax.grid(alpha=.15)
    axes[0].legend(loc='upper left',fontsize=10)
    fig.suptitle('布光侧视示意 · 圆点为相机，星号为面光源中心',fontsize=15)
    footer(fig,'每次只启用一个角度和一盏灯。相机距原点 1，光源距原点 3；两者均在平面上方，指向原点。蓝灰带表示计算域高度 ±0.08。')
    fig.subplots_adjust(top=.84,bottom=.15,left=.06,right=.99,wspace=.15)
    fig.savefig(dest/'setup.png',dpi=160);plt.close(fig)


def appearance(data,stats,meta,side,dest):
    fig,axes=plt.subplots(3,4,figsize=(14.5,11))
    for row,angle in enumerate(ANGLES):
        case=f'{side}_{angle}'
        for col,mode in enumerate(RENDER_MODES):
            ax=axes[row,col]; ax.imshow(filmic(data[case][mode]['total'],GAINS[case]),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            r=stat(stats,case,mode,'total')
            ax.set_title(f'{LABELS[col]}\nL = {r["mean"]:.6f} ± {1.96*r["standard_error"]:.6f}',fontsize=10,color=COLORS[col])
            if col==0: ax.set_ylabel(f'{angle}° · 显示增益 ×{GAINS[case]:g}',fontsize=11)
    side_label='同侧照明' if side=='same' else '对侧照明'
    fig.suptitle(f'平面 · {side_label} · 四种模型\nσ = 0.01，ℓ = 0.025，镜面反射率 0.9；{meta["resolution"][0]}²，{meta["spp"]} spp',fontsize=16)
    footer(fig,'各行四种模式使用相同显示增益；亮度 L 来自线性图，不受显示增益影响。± 为近似 95% 采样误差，仅包含蒙特卡洛噪声。\nGPIS 为沿整条路径固定的三维稀疏卷积场，有限 spp 下亮点与黑点是采样噪声。')
    fig.subplots_adjust(top=.88,bottom=.1,left=.06,right=.99,hspace=.28,wspace=.04)
    fig.savefig(dest/f'appearance_{side}.png',dpi=150);plt.close(fig)


def decomposition(data,stats,meta,case,dest):
    fig,axes=plt.subplots(3,4,figsize=(14.5,11))
    for row,part in enumerate(PARTS):
        for col,mode in enumerate(RENDER_MODES):
            ax=axes[row,col];ax.imshow(filmic(data[case][mode][part],GAINS[case]),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            r=stat(stats,case,mode,part)
            ax.set_title(f'{LABELS[col]} · L = {r["mean"]:.6f}',fontsize=10,color=COLORS[col])
            if col==0: ax.set_ylabel(PART_LABELS[row],fontsize=11)
    side,angle=case.split('_')
    fig.suptitle(f'{"同侧" if side=="same" else "对侧"}照明 · {angle}° · 按散射次数分解\n全部面板使用显示增益 ×{GAINS[case]:g}',fontsize=16)
    footer(fig,'上排 = 中排 + 下排（在线性辐亮度中）。透明计算域边界不计入散射次数。\n多次散射由同一随机种子的完整图减去单次图得到；已用独立的深度筛选渲染验证分解。显示曲线为统一 filmic。')
    fig.subplots_adjust(top=.89,bottom=.1,left=.06,right=.99,hspace=.25,wspace=.04)
    fig.savefig(dest/f'scattering_{case}.png',dpi=150);plt.close(fig)


def overview(data,dest):
    fig,axes=plt.subplots(6,4,figsize=(12.5,18))
    for row,case in enumerate(CASES):
        for col,mode in enumerate(RENDER_MODES):
            ax=axes[row,col];ax.imshow(filmic(data[case][mode]['total']),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0: ax.set_title(LABELS[col],color=COLORS[col])
            if col==0: ax.set_ylabel(case,fontsize=11)
    fig.suptitle('完整结果 · 所有角度、光源与模型均使用显示增益 ×1',fontsize=16)
    footer(fig,'相机同侧的掠射照明本来就很暗；原始线性 PFM 及独立 PNG 保留在各渲染目录中。')
    fig.subplots_adjust(left=.08,right=.99,top=.94,bottom=.04,wspace=.03,hspace=.08)
    fig.savefig(dest/'appearance_uniform.png',dpi=130);plt.close(fig)


def response(stats,dest):
    fig,axes=plt.subplots(3,2,figsize=(13,11))
    for row,part in enumerate(PARTS):
        for col,side in enumerate(SIDES):
            ax=axes[row,col]
            for i,mode in enumerate(RENDER_MODES):
                rows=[stat(stats,f'{side}_{angle}',mode,part) for angle in ANGLES]
                ax.errorbar(np.array(ANGLES)+(i-1.5)*.45,[r['mean'] for r in rows],
                    yerr=[1.96*r['standard_error'] for r in rows],color=COLORS[i],
                    linestyle=STYLES[i],marker='o',capsize=3,label=LABELS[i],linewidth=1.6)
            ax.set_title(f'{"同侧" if side=="same" else "对侧"}照明 · {PART_LABELS[row]}')
            ax.set_xticks(ANGLES);ax.set_xlabel('中心观察角（相对平面法线，度）');ax.set_ylabel('图像区域平均辐亮度 L')
            ax.grid(alpha=.2)
            if side=='same': ax.set_yscale('log')
    axes[0,0].legend(fontsize=9)
    fig.suptitle('平面方向性响应与散射分解（线性辐亮度统计）',fontsize=17)
    footer(fig,'统计区域为整幅图像；中心角周围含 8° 视场。误差棒为近似 95% 采样误差；不包含步长、有限域、密度网格或有限脉冲的误差。\n同侧列采用对数纵轴。角度上的轻微横向偏移仅用于区分误差棒。多次散射误差通过配对样本的奇偶两半估计。')
    fig.subplots_adjust(top=.91,bottom=.12,left=.09,right=.97,hspace=.44,wspace=.25)
    fig.savefig(dest/'response.png',dpi=160);plt.close(fig)


def averaged(data,dest):
    fig,axes=plt.subplots(6,4,figsize=(12.5,18))
    for row,case in enumerate(CASES):
        for col,mode in enumerate(RENDER_MODES):
            ax=axes[row,col];ax.imshow(filmic(blocks(data[case][mode]['total']),GAINS[case]),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0: ax.set_title(LABELS[col],color=COLORS[col])
            if col==0: ax.set_ylabel(f'{case}\n增益 ×{GAINS[case]:g}',fontsize=10)
    fig.suptitle('16 × 16 分区平均 · 降低噪声后观察大尺度亮度',fontsize=16)
    footer(fig,'每个方块是原始线性图的区域平均，再应用显示曲线；不是额外渲染或逐模式调亮。原始图另行保留。')
    fig.subplots_adjust(left=.1,right=.99,top=.94,bottom=.04,wspace=.03,hspace=.08)
    fig.savefig(dest/'appearance_block_means.png',dpi=130);plt.close(fig)


def normals(dest):
    reports=[]
    fig_curves,axes_curves=plt.subplots(2,2,figsize=(13,9))
    for row,angle in enumerate((30,80)):
        source=OUT/'probes'/f'angle_{angle}.json'
        d=load(source);paths=d['spec']['samples'];nz,np_=d['nz'],d['nphi']
        assert tuple(d['spec']['modes'])==RENDER_MODES
        densities=[]
        for bounce in (0,1):
            counts=np.array([m['collisions'][bounce]['normal_counts'] for m in d['results']]).reshape(4,nz,np_)
            hits=np.array([m['collisions'][bounce]['hits'] for m in d['results']])
            assert (hits>0).all() and np.array_equal(counts.sum((1,2)),hits)
            densities.append(counts/hits[:,None,None]/(4*np.pi/(nz*np_)))
        norm=Normalize(0,max(float(a.max()) for a in densities))
        fig,axes=plt.subplots(2,4,figsize=(15,8))
        for bounce in (0,1):
            ax_curve=axes_curves[row,bounce]
            for i,mode in enumerate(RENDER_MODES):
                c=d['results'][i]['collisions'][bounce]
                ax=axes[bounce,i]
                im=ax.imshow(disk_image(densities[bounce][i]),extent=(-1,1,-1,1),origin='lower',cmap='magma',norm=norm)
                ax.add_patch(Circle((0,0),1,fc='none',ec='#aeb4bf',lw=.6))
                ax.add_patch(Circle((0,0),1/np.sqrt(2),fc='none',ec='#aeb4bf',lw=.6,ls='--'))
                ax.axis('off');ax.set_title(f'{LABELS[i]}\n第 {bounce+1} 次：{c["hits"]:,} / {paths:,}',fontsize=10,color=COLORS[i])
                width=180/d['angle_bins'];p=np.array(c['theta_counts'])/c['hits']
                center=(np.arange(len(p))+.5)*width
                ax_curve.plot(center,p/width,color=COLORS[i],ls=STYLES[i],label=LABELS[i],lw=1.6)
                error=1.96*np.sqrt(p*(1-p)/c['hits'])/width
                ax_curve.fill_between(center,np.maximum(0,p/width-error),p/width+error,color=COLORS[i],alpha=.1)
                reports.append(dict(angle=angle,mode=mode,collision=bounce+1,hits=c['hits'],
                    hit_probability=c['hit_probability'],distance_failed=c['distance_failed'],phase_failed=c['phase_failed'],
                    source_sha256=sha(source)))
            ax_curve.set(xlim=(0,180),ylim=(0,None),xlabel='法线与 +Y 的夹角（度）',ylabel='每度的概率',
                         title=f'{angle}° 中心光线 · 第 {bounce+1} 次碰撞')
            ax_curve.grid(alpha=.2)
        fig.suptitle(f'{angle}° 中心光线 · 实际碰撞法线分布\n每种模式 {paths:,} 条独立路径',fontsize=16)
        fig.colorbar(im,cax=fig.add_axes([.93,.25,.012,.48]),label='命中条件下的概率密度 / sr⁻¹')
        failures=[[m['collisions'][b]['distance_failed']+m['collisions'][b]['phase_failed'] for m in d['results']] for b in (0,1)]
        footer(fig,f'各列使用相同世界坐标与色标。圆心为 +Y，虚线为 90°，外圈为 180°。上排第一次碰撞，下排第二次碰撞。\n采样失败数（列顺序）：第一次 {failures[0]}；第二次 {failures[1]}。失败路径不重试。\n这是命中条件下的法线统计；交点和第二次入射方向会变化，不是固定点解析 NDF。同侧与对侧布光共用此诊断。')
        fig.subplots_adjust(left=.02,right=.89,top=.85,bottom=.14,hspace=.28,wspace=.05)
        fig.savefig(dest/f'normals_{angle}.png',dpi=150);plt.close(fig)
    axes_curves[0,0].legend(fontsize=9)
    fig_curves.suptitle('实际碰撞法线的倾角分布',fontsize=17)
    footer(fig_curves,'按各模式的有效碰撞数归一化；阴影为逐分箱近似 95% 采样区间。采样失败数与命中率另存于 manifest.json，失败路径不重试。')
    fig_curves.subplots_adjust(top=.89,bottom=.12,left=.08,right=.97,hspace=.36,wspace=.2)
    fig_curves.savefig(dest/'normal_response.png',dpi=160);plt.close(fig_curves)
    return reports


def page(stats,meta,dest,stage):
    def conditional_change(case,part):
        baseline=stat(stats,case,'classic_global',part)['mean']
        value=stat(stats,case,'global_conditional',part)['mean']
        return 100*(value/baseline-1)
    findings=(f'本轮区域平均结果：以 classic_global 为基准，global_conditional 在 80° 同侧照明下，'
        f'单次贡献变化 {conditional_change("same_80","single"):+.2f}%，'
        f'多次贡献变化 {conditional_change("same_80","multiple"):+.2f}%，'
        f'总亮度变化为 {conditional_change("same_80","total"):+.3f}%。'
        f'80° 对侧照明下，单次贡献变化 {conditional_change("opposite_80","single"):+.2f}%，'
        f'多次贡献变化 {conditional_change("opposite_80","multiple"):+.2f}%，'
        f'总亮度变化为 {conditional_change("opposite_80","total"):+.2f}%。'
        '各项采样误差见曲线和下方统计表。单次散射图限制的是碰撞次数；'
        '首次碰撞后通向光源的可见性仍使用条件化传播，所以单次贡献也会变化。')
    cards=''.join(f'<a href="scattering_{case}.png"><img loading="lazy" src="scattering_{case}.png"><span>{case}：完整／单次／多次</span></a>' for case in CASES)
    table=''
    for case in CASES:
        for mode in RENDER_MODES:
            t,s,m=[stat(stats,case,mode,p) for p in PARTS]
            table+=f'<tr><td>{case}</td><td>{mode}</td><td>{t["mean"]:.7f}</td><td>±{1.96*t["standard_error"]:.7f}</td><td>{s["mean"]:.7f}</td><td>{m["mean"]:.7f}</td><td>{100*m["mean"]/t["mean"]:.2f}%</td></tr>'
    validation_section=''
    if (OUT/'validation.json').exists():
        validation=load(OUT/'validation.json')
        step_rows=''.join(f'<tr><td>{r["case"]}</td><td>{r["baseline_mean"]:.7f}</td><td>{r["refined_mean"]:.7f}</td><td>{100*r["relative_change"]:+.2f}%</td><td>{r["difference"]:+.7f} ± {1.96*r["paired_standard_error"]:.7f}</td></tr>' for r in validation['gpis_step_half'])
        density_change=max(abs(r['relative_change']) for r in validation['density_spacing_half'])
        validation_section=f'''<h2>数值敏感性检查</h2><p>六组 classic_local 的法线方向网格间距从 0.0025 减至 0.00125，平均亮度变化的最大绝对值为 {100*density_change:.3f}%。计算域与散射次数分解的详细检查见 <a href="validation.json">验证数据</a>。</p>
<p>以下为 80°、128²、32 spp 的配对步长检查。对侧光的变化超出了采样误差，说明当前 GPIS 参考仍有可见的步长偏差；它不能被当成精确 GP 的真值。法线图另外列出了实际采样失败数，这些路径不重试。</p>
<table><tr><th>照明</th><th>步长 0.0025</th><th>步长 0.00125</th><th>相对变化</th><th>亮度变化 ± 近似95%采样误差</th></tr>{step_rows}</table>'''
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>平面 conditional 对照</title>
<style>body{{max-width:1500px;margin:40px auto;padding:0 24px;color:#253145;background:#f5f6f8;font:16px/1.65 system-ui}}h1{{font-size:30px}}img{{width:100%;background:white}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}a{{color:#235e91;text-decoration:none}}.grid a{{background:white;padding:12px;border-radius:10px}}span{{display:block}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{padding:8px 12px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.note{{padding:18px;background:#e7edf5;border-radius:10px}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>平面：classic_local / classic_global / global_conditional / GPIS</h1>
<p>均值 μ(x,y,z)=y；各向同性 SE，σ=0.01，ℓ=0.025；理想镜面反射率 0.9。{meta['resolution'][0]}×{meta['resolution'][1]}，{meta['spp']} spp。</p>
<p>相机距原点 1，视场 8°；光源距原点 3，为 0.45×0.45 的方形面光，辐亮度 30。相机与光源都在平面上方，同侧或方位相差 180°，分别渲染，无环境光。</p>
<div class="note">同侧图的显示增益为 30° ×1、60° ×8、80° ×32；同一角度的四种模型共用增益。对侧图增益均为 ×1。跨角度比较绝对亮度请看响应曲线或<a href="appearance_uniform.png">统一曝光总览</a>。<br>GPIS 为 3D 稀疏卷积近似：每格 10 个脉冲，步长 0.0025，同一路径保持同一随机场。它的镜面事件不使用普通光源 NEE，噪声显著高于前三种模式。当前 conditional 只保留最近一次场值与梯度，不等于完整历史 GPIS。</div>
<a href="setup.png"><img src="setup.png"></a>
<h2>外观</h2><div class="grid"><a href="appearance_same.png"><img src="appearance_same.png"><span>同侧照明</span></a><a href="appearance_opposite.png"><img src="appearance_opposite.png"><span>对侧照明</span></a></div>
<p><a href="appearance_block_means.png">16×16 分区平均图</a> · <a href="appearance_uniform.png">所有条件统一曝光</a> · <a href="metrics.csv">线性统计 CSV</a> · <a href="manifest.json">参数、源文件哈希与方法说明</a> · <a href="validation.json">网格、步长、计算域与散射分解检查</a></p>
<h2>平均亮度与散射次数</h2><p>{findings}</p><a href="response.png"><img src="response.png"></a>
<h2>逐条件散射分解</h2><div class="grid">{cards}</div>
<h2>固定中心光线的实际碰撞法线</h2><p>30° 与 80°，各模式使用相同数量的独立路径，样本数见图。观察第一次与第二次碰撞；交点位置不固定，因此这是命中条件下的法线分布，不能与固定点解析 NDF 混用。布光不改变这项几何诊断。</p><div class="grid"><a href="normals_30.png"><img loading="lazy" src="normals_30.png"></a><a href="normals_80.png"><img loading="lazy" src="normals_80.png"></a></div><a href="normal_response.png">法线倾角曲线</a>
{validation_section}
<h2>线性统计</h2><p>平均区域为整幅图；± 表示近似 95% 蒙特卡洛采样误差，不包含模型近似和离散化误差。</p><table><thead><tr><th>条件</th><th>模式</th><th>完整</th><th>采样误差</th><th>单次</th><th>多次</th><th>多次占比</th></tr></thead><tbody>{table}</tbody></table>
</html>'''
    (dest/'comparison.html').write_text(body,encoding='utf-8')


def main(stage,seed):
    data,stats,sources,meta=collect(stage,seed)
    dest=OUT if stage=='final' else OUT/stage/'figures';dest.mkdir(parents=True,exist_ok=True)
    for side in SIDES: appearance(data,stats,meta,side,dest)
    for case in CASES: decomposition(data,stats,meta,case,dest)
    setup_figure(dest);overview(data,dest);averaged(data,dest);response(stats,dest)
    normal_report=normals(dest);page(stats,meta,dest,stage)
    with (dest/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(stats[0]));writer.writeheader();writer.writerows(stats)
    save(dest/'manifest.json',dict(render=meta,stage=stage,statistics=stats,sources=sources,
        display_gains=GAINS,normal_statistics=normal_report,mean='y',sigma=.01,length_scale=.025,mirror_albedo=.9,
        mean_evaluation='analytic for global/conditional/GPIS; trilinear unit-gradient plane for local',
        density_evaluation='classic_local interpolates baked density with normal spacing .0025; classic_global evaluates it online',
        memory='conditional: latest value and gradient; GPIS: same realization throughout each path',
        scattering_decomposition='paired total(max_bounces=128) minus single(max_bounces=2), both min_bounces=1; transparent boundaries do not count',
        uncertainty='sampling only: exact sample-mean variance for total/single; paired even/odd halves for multiple; approximate normal 95% intervals',
        gpis_approximation=dict(type='3D sparse convolution',step=.0025,impulses_per_cell=10,
             finite_kernel=True,independent_realization_per_path=True),
        figures=[p.name for p in dest.glob('*.png')]))
    print(dest/'comparison.html',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',default='final');parser.add_argument('--seed',type=int,default=SEED)
    args=parser.parse_args();main(args.stage,args.seed)
