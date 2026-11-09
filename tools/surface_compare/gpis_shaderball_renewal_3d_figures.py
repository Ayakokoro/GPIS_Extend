"""Validated figures, linear-light statistics and LDR-FLIP against 3D Renewal+."""
from __future__ import annotations

import copy
import csv
import html
import os
import subprocess
from pathlib import Path

from figures import plt, Image, Rectangle
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from run import load, save, sha
from plane_figures import blocks
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links
from gpis_shaderball_renewal_3d import (ROOT, REPORT, ORIGINAL, MODE, GLOBAL, MODES, CHANGES,
    GROUPS, EXE, SEED, ROI, MODEL_ROI, original, destination, verified, scene)

LABELS = (*MODES[:-2], 'GPIS global', 'GPIS 3D Renewal+')
LABEL_BY_MODE = dict(zip(MODES,LABELS))
PPD = 67.


def regions(n):
    masks = {'full': np.ones((n,n), bool)}
    for name, rect in [('shaderball_box', MODEL_ROI), ('detail', ROI)]:
        x0,y0,x1,y1 = [int(v*n) for v in rect]
        mask = np.zeros((n,n), bool)
        mask[y0:y1,x0:x1] = True
        masks[name] = mask
    return masks


def relative(path):
    return Path(os.path.relpath(path, REPORT)).as_posix()


def read_groups(args):
    assert CHANGES == {'correlation_context': 'renewal+'}
    archived = load(ORIGINAL)
    compatibility = load(REPORT/'renderer_compatibility.json')
    assert compatibility['passed'] and compatibility['renderer_sha256'] == sha(EXE)
    groups = []
    for key in args.groups:
        sigma = .02 if key.endswith('0p02') else .04
        previous = next(g for g in archived['groups'] if (g['ell'],g['sigma']) == (.2,sigma))
        data = {}
        for mode in MODES:
            d = destination(key,args.stage) if mode == MODE else original(key,mode)
            data[mode] = verified(d)
            data[mode]['source']['mode'] = mode
            result = data[mode]['source']['result']
            assert result['resolution'] == [256,256] and result['spp'] == 128
            if mode != MODE:
                src = next(s for s in previous['sources'] if s['mode'] == mode)
                assert result == src['result'] and Path(src['folder']) == d
                checks = [r for r in compatibility['checks'] if r['group']==key and r['mode']==mode]
                assert len(checks) == 3 and all(c['passed'] for c in checks)
                assert all(c['original_result_sha256']==sha(d/'result.json') for c in checks)
            else:
                expected, _ = scene(key,args.stage,256,128,.01)
                assert data[mode]['scene'] == expected
                assert result['provenance']['renderer_sha256'] == sha(EXE)
        ref = data[MODE]['scene']
        renewal_medium = ref['media'][0]
        assert renewal_medium['correlation_context'] == 'renewal+'
        assert not renewal_medium['1D_sampling']
        assert renewal_medium['1D_sampling_scheme'] == 'uni'
        assert not renewal_medium['1D_gradient_correlationXY']
        assert not renewal_medium['single_realization']
        for mode, d in data.items():
            s = d['scene']
            for section in ('primitives','bsdfs','camera','integrator','renderer'):
                a,b = copy.deepcopy(s[section]), copy.deepcopy(ref[section])
                if section == 'renderer':
                    a.pop('output_directory'); b.pop('output_directory')
                assert a == b, (mode,section)
            medium = s['media'][0]
            cov = medium.get('covariance', medium.get('gaussian_process',{}).get('covariance'))
            assert cov == dict(type='squared_exponential', sigma=sigma, lengthScale=.2)
            material = medium['phase_function']['bsdf'] if mode in (MODE,GLOBAL) else medium['material']
            assert material == ref['media'][0]['phase_function']['bsdf']
            if mode == GLOBAL:
                assert medium['correlation_context']=='global' and not medium['1D_sampling']
                assert not medium['single_realization'] and medium['step_size']==.01
                changed=copy.deepcopy(s)
                changed['media'][0].update(CHANGES)
                changed['renderer']['output_directory']=ref['renderer']['output_directory']
                assert changed==ref, 'Global and 3D Renewal+ differ beyond correlation_context and output directory'
            if 'field' in medium:
                assert sha(Path(medium['field']['file'])) == d['source']['result']['provenance']['field_sha256']
            if mode == 'neural':
                assert sha(Path(medium['model'])) == d['source']['result']['provenance']['model_sha256']
        groups.append(dict(key=key, label='A' if sigma==.02 else 'B',
                           ell=.2, sigma=sigma, data=data))
    return groups, compatibility


def measure(groups):
    rows = []
    for g in groups:
        ref = g['data'][MODE]
        block_mask = regions(16)['shaderball_box']
        norm = float(ref['y'][regions(256)['shaderball_box']].mean())
        for mode,d in g['data'].items():
            delta = d['y']-ref['y']
            half_delta = [a-b for a,b in zip(d['halves'],ref['halves'])]
            block_delta = blocks(delta[:,:,None])[:,:,0]
            block_half = [blocks(h[:,:,None])[:,:,0] for h in half_delta]
            rms = float(np.sqrt(np.mean(block_delta[block_mask]**2)))
            noise = float(np.sqrt(np.mean((.5*(block_half[0]-block_half[1]))[block_mask]**2)))
            d['block_relative_rms'] = rms/norm
            d['block_relative_noise_rms'] = noise/norm
            for region,mask in regions(256).items():
                count = int(mask.sum())
                mean = float(d['y'][mask].mean())
                ref_mean = float(ref['y'][mask].mean())
                se = float(np.sqrt(np.sum((.5*(d['halves'][0]-d['halves'][1]))[mask]**2))/count)
                delta_se = float(np.sqrt(np.sum((.5*(half_delta[0]-half_delta[1]))[mask]**2))/count)
                rows.append(dict(group=g['label'],ell=g['ell'],sigma=g['sigma'],mode=mode,
                    reference=MODE, region=region, pixels=count, mean_luminance=mean,
                    reference_mean_luminance=ref_mean, relative_mean_difference=mean/ref_mean-1,
                    half_split_mean_standard_error_estimate=se,
                    paired_half_difference_standard_error_estimate=delta_se,
                    luminance_rmse=float(np.sqrt(np.mean(delta[mask]**2))),
                    block_roi_relative_rms=rms/norm, block_roi_relative_noise_rms=noise/norm))
    return rows


def flip(groups):
    assert DEFAULT_FLIP.is_file(), str(DEFAULT_FLIP)
    help_result = subprocess.run([str(DEFAULT_FLIP), '--help'], capture_output=True,
        text=True, encoding='utf-8', errors='replace', check=True)
    help_text = help_result.stdout+help_result.stderr
    assert 'FLIP v1.7.' in help_text
    (REPORT/'flip_tool_help.txt').write_text(help_text, encoding='utf-8')
    rows = []
    for g in groups:
        ref = Path(g['data'][MODE]['source']['folder'])/'render.png'
        for mode,d in g['data'].items():
            src = Path(d['source']['folder'])/'render.png'
            folder = REPORT/'flip_3d'/g['label']/mode
            result = evaluate(DEFAULT_FLIP, ref, src, folder, PPD)
            if mode == MODE:
                assert all(v==0 for v in result['statistics'].values())
            d['flip'] = dict(result=result, error_map=str(folder/result['error_map']),
                             record=str(folder/'result.json'))
            rows.append(dict(group=g['label'],ell=g['ell'],sigma=g['sigma'],mode=mode,
                reference=MODE, ppd=PPD, **result['statistics']))
        print('FLIP complete',g['label'],flush=True)
    return rows


def plots(groups):
    for detail in (False,True):
        fig,axes = plt.subplots(len(groups),len(MODES),figsize=(22.8,4.2*len(groups)),squeeze=False)
        for row,g in enumerate(groups):
            for col,mode in enumerate(MODES):
                ax = axes[row,col]
                im = np.asarray(Image.open(Path(g['data'][mode]['source']['folder'])/'render.png').convert('RGB'))
                x0,y0,x1,y1 = [int(v*256) for v in ROI]
                if detail: im = im[y0:y1,x0:x1]
                else: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffcf52',lw=1))
                ax.imshow(im,interpolation='nearest')
                ax.set_xticks([]); ax.set_yticks([])
                if row==0: ax.set_title(LABELS[col],fontsize=13)
                if col==0: ax.set_ylabel(f"{g['label']} 组\nℓ={g['ell']:g}，σ={g['sigma']:g}",fontsize=12)
        fig.suptitle('Shaderball · GPIS global / 3D Renewal+ 六模式比较'+(' · 凹槽局部' if detail else '')+'\n256×256 · 128 spp · 相同 filmic 显示与曝光',fontsize=17)
        fig.subplots_adjust(left=.055,right=.99,top=.86,bottom=.035,wspace=.03,hspace=.07)
        fig.savefig(REPORT/('detail.png' if detail else 'appearance.png'),dpi=150)
        plt.close(fig)
    for kind in ('difference','flip','flip_detail'):
        fig,axes = plt.subplots(len(groups),len(MODES)-1,figsize=(19,4.4*len(groups)),squeeze=False)
        for row,g in enumerate(groups):
            ref = g['data'][MODE]
            norm = float(ref['y'][regions(256)['shaderball_box']].mean())
            for col,mode in enumerate(MODES[:-1]):
                ax = axes[row,col]; d = g['data'][mode]
                if kind=='difference':
                    im = blocks((d['y']-ref['y'])[:,:,None])[:,:,0]/norm
                    ax.imshow(im,cmap='RdBu_r',vmin=-.25,vmax=.25,interpolation='nearest')
                    title = f"{LABEL_BY_MODE[mode]}\n分区 RMS = {d['block_relative_rms']:.2%}"
                else:
                    im = np.asarray(Image.open(d['flip']['error_map']).convert('RGB'))
                    if kind=='flip_detail':
                        x0,y0,x1,y1 = [int(v*256) for v in ROI]
                        im = im[y0:y1,x0:x1]
                    ax.imshow(im,interpolation='nearest')
                    title = f"{LABEL_BY_MODE[mode]}\n全图 FLIP = {d['flip']['result']['statistics']['mean']:.5f}"
                ax.set_title(title,fontsize=11); ax.set_xticks([]); ax.set_yticks([])
                if col==0: ax.set_ylabel(f"{g['label']} 组\nℓ={g['ell']:g}，σ={g['sigma']:g}",fontsize=12)
        if kind=='difference':
            title='相对 GPIS 3D Renewal+ 的 16×16 分区亮度差'
            scalar=ScalarMappable(norm=Normalize(-.25,.25),cmap='RdBu_r')
            legend='ΔY / 3D Renewal+ 模型区域均值'
        else:
            title='相对 GPIS 3D Renewal+ 的 LDR-FLIP'+(' · 凹槽局部' if kind=='flip_detail' else '')
            scalar=ScalarMappable(norm=Normalize(0,1),cmap='magma'); legend='FLIP error'
        fig.suptitle(title+'\n同组 128 spp 参考；差异包含采样噪声',fontsize=17)
        fig.subplots_adjust(left=.06,right=.91,top=.85,bottom=.04,wspace=.09,hspace=.25)
        fig.colorbar(scalar,cax=fig.add_axes([.935,.15,.012,.59]),label=legend)
        fig.savefig(REPORT/f'{kind}.png',dpi=150); plt.close(fig)


def renewal_vs_global(groups):
    """Run the requested pair explicitly with GPIS global as FLIP reference."""
    rows=[]
    fig,axes=plt.subplots(len(groups),4,figsize=(16,4.6*len(groups)),squeeze=False)
    for row,g in enumerate(groups):
        ref=Path(g['data'][GLOBAL]['source']['folder'])/'render.png'
        test=Path(g['data'][MODE]['source']['folder'])/'render.png'
        dest=REPORT/'flip_3d'/g['label']/'renewal_vs_global'
        result=evaluate(DEFAULT_FLIP,ref,test,dest,PPD)
        g['renewal_vs_global']=dict(record=str(dest/'result.json'),
            error_map=str(dest/result['error_map']),pooled_values=str(dest/result['pooled_values']),
            result=result)
        rows.append(dict(group=g['label'],ell=g['ell'],sigma=g['sigma'],mode=MODE,
            reference=GLOBAL,ppd=PPD,**result['statistics']))
        error=np.asarray(Image.open(dest/result['error_map']).convert('RGB'))
        x0,y0,x1,y1=[int(v*256) for v in ROI]
        panels=[np.asarray(Image.open(p).convert('RGB')) for p in (ref,test)]+[error,error[y0:y1,x0:x1]]
        titles=['GPIS global（参考）','GPIS 3D Renewal+',
            f"全图 LDR-FLIP = {result['statistics']['mean']:.6f}",'凹槽局部 FLIP']
        for col,(panel,title) in enumerate(zip(panels,titles)):
            ax=axes[row,col];ax.imshow(panel,interpolation='nearest')
            ax.set_title(title,fontsize=12);ax.set_xticks([]);ax.set_yticks([])
            if col<3: ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffcf52',lw=.8))
            if col==0: ax.set_ylabel(f"{g['label']} 组\nℓ={g['ell']:g}，σ={g['sigma']:g}",fontsize=12)
    fig.suptitle('GPIS 3D Renewal+ 与 GPIS global 的直接比较\n本地 FLIP 1.7 · 67 PPD · 256×256 · 128 spp',fontsize=17)
    fig.subplots_adjust(left=.06,right=.91,top=.85,bottom=.045,wspace=.08,hspace=.20)
    fig.colorbar(ScalarMappable(norm=Normalize(0,1),cmap='magma'),
        cax=fig.add_axes([.935,.16,.012,.57]),label='FLIP error · 0–1')
    fig.savefig(REPORT/'renewal_vs_global.png',dpi=150);plt.close(fig)
    return rows


def page(groups, rows, flip_rows, manifest):
    def link(path,label): return f'<a href="{html.escape(relative(path),quote=True)}">{html.escape(label)}</a>'
    table = ''; sources = ''; diagnostics = ''; pair_table = ''
    for g in groups:
        pair=g['renewal_vs_global'];stats=pair['result']['statistics']
        pair_table+=f'<tr><td>{g["label"]}：ℓ={g["ell"]:g}，σ={g["sigma"]:g}</td><td>{stats["mean"]:.6f}</td><td>{stats["weighted_median"]:.6f}</td><td>{stats["maximum"]:.6f}</td><td>'+ ' · '.join(link(Path(pair[k]),label) for k,label in [('error_map','原始误差图'),('pooled_values','原始统计'),('record','FLIP 运行记录')])+'</td></tr>'
        for mode,d in g['data'].items():
            r = next(r for r in rows if r['group']==g['label'] and r['mode']==mode and r['region']=='shaderball_box')
            f = next(r for r in flip_rows if r['group']==g['label'] and r['mode']==mode)
            label = LABEL_BY_MODE[mode]
            table += f'<tr><td>{g["label"]}</td><td>{label}</td><td>{r["mean_luminance"]:.6f}</td><td>{r["relative_mean_difference"]:+.2%}</td><td>{r["block_roi_relative_rms"]:.2%}</td><td>{r["block_roi_relative_noise_rms"]:.2%}</td><td>{f["mean"]:.6f}</td></tr>'
            folder=Path(d['source']['folder']); result=d['source']['result']
            sources += f'<tr><td>{g["label"]}</td><td>{label}</td><td>'+ ' · '.join(link(folder/name,label) for name,label in [('render.png','PNG'),('render.pfm','PFM'),('scene.json','场景'),('result.json','运行记录')])+f'</td><td>{result["seconds"]:.1f} s</td><td><code>{result["provenance"]["renderer_sha256"][:12]}</code></td></tr>'
        log=manifest['diagnostics'][g['key']]
        diagnostics += f'<li>{g["label"]} 组 3D Renewal+：零长度射线诊断 {log["zero_length_ray_messages"]} 次，段末退出诊断 {log["segment_exit_messages"]} 次。</li>'
    best = []
    for g in groups:
        candidates=[r for r in rows if r['group']==g['label'] and r['mode']!=MODE and r['region']=='shaderball_box']
        br=min(candidates,key=lambda r:r['block_roi_relative_rms'])
        bf=min((r for r in flip_rows if r['group']==g['label'] and r['mode']!=MODE),key=lambda r:r['mean'])
        best.append(f'<li>{g["label"]} 组：模型区域分区 RMS 最小的是 <b>{LABEL_BY_MODE[br["mode"]]}</b>（{br["block_roi_relative_rms"]:.2%}）；全图 FLIP 最小的是 <b>{LABEL_BY_MODE[bf["mode"]]}</b>（{bf["mean"]:.6f}）。</li>')
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaderball · GPIS global / 3D Renewal+ 六模式比较</title>
<style>body{{max-width:1700px;margin:30px auto;padding:0 24px;background:#f5f6f8;color:#253145;font:16px/1.65 system-ui}}h1{{font-size:28px}}h2{{margin-top:32px}}img{{width:100%;background:white}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2),th:first-child,th:nth-child(2){{text-align:left}}.scroll{{overflow:auto}}.note{{background:#e6edf5;padding:16px;border-radius:8px}}a{{color:#235e91}}code{{font-size:13px}}</style>
<h1>Shaderball：classic_local / classic_global / global_conditional / neural / GPIS global / GPIS 3D Renewal+</h1>
<p>A：ℓ=0.2，σ=0.02；B：ℓ=0.2，σ=0.04。沿用原实验相机、光源、地面、绿色镜面材质、256×256、128 spp、seed={SEED}、最大反弹数 20 和 step_size=0.01。统一 filmic，无降噪或逐图曝光调整。</p>
<p class="note">六模式汇总以 <b>GPIS 3D Renewal+</b> 为参考：从 GPIS global 场景仅将 correlation_context 改为 renewal+；1D_sampling=false、1D_sampling_scheme=uni、1D_gradient_correlationXY=false、single_realization=false 以及其余渲染参数均保持不变（输出目录另设）。两者均使用三维稀疏卷积噪声与梯度；GPIS 镜面事件保持 UNI 采样。五个比较 mode 复用原报告通过哈希校验的结果；neural 权重未重训。GPIS global 使用全路径随机场记忆，3D Renewal+ 使用交点值和完整三维梯度的逐段条件化，二者的统计模型不同。下方另设两者直接比较，明确以 GPIS global 为 FLIP 参考。</p>
<p><a href="../ell0p2_neural_comparison/comparison.html">原 GPIS global 比较</a> · <a href="../ell0p2_renewal_1d_mis_comparison/comparison.html">旧 1D+MIS Renewal+ 归档</a> · <a href="manifest.json">参数与来源</a> · <a href="metrics.csv">线性亮度指标</a> · <a href="flip_metrics.csv">FLIP 指标</a> · <a href="renderer_compatibility.json">版本抽查记录</a></p>
<h2>本批结果</h2><ul>{''.join(best)}</ul>
<h2>六模式原图</h2><a href="appearance.png"><img src="appearance.png" alt="两组参数的六模式图像"></a>
<h2>凹槽局部放大</h2><a href="detail.png"><img src="detail.png" alt="固定凹槽区域"></a>
<h2 id="renewal-vs-global">3D Renewal+ 与 GPIS global 的 FLIP 误差</h2>
<p>直接调用本地 flip.exe：参考图为同组 <b>GPIS global</b>，测试图为 <b>GPIS 3D Renewal+</b>。输入原始 filmic PNG，固定 67 PPD，色标 0–1。局部图直接裁自全图误差结果；两幅图均为 128 spp，误差包含双方的采样噪声。</p>
<p><a href="renewal_vs_global_metrics.csv">两者直接比较 CSV</a></p>
<div class="scroll"><table><tr><th>参数组</th><th>Mean FLIP</th><th>Weighted median</th><th>Max FLIP</th><th>本地 FLIP 输出</th></tr>{pair_table}</table></div>
<a href="renewal_vs_global.png"><img src="renewal_vs_global.png" alt="GPIS global、3D Renewal+、全图及局部 FLIP 误差"></a>
<h2>相对 3D Renewal+ 的数值差异</h2>
<p>区域均值与均值差使用原报告的模型矩形区域；分区 RMS 在 16×16 网格的同一模型区域计算，并除以该组 3D Renewal+ 模型区域均值。噪声 RMS 由偶奇样本半图差估计，是描述性数值。FLIP 使用原始 filmic PNG，全图 LDR-FLIP，67 pixels/degree；数值越低越接近本批 3D Renewal+ 图像。</p>
<div class="scroll"><table><tr><th>组</th><th>Mode</th><th>区域平均 Y</th><th>相对均值差</th><th>分区 RMS</th><th>噪声 RMS 估计</th><th>全图 mean FLIP</th></tr>{table}</table></div>
<a href="difference.png"><img src="difference.png" alt="分区线性亮度差"></a>
<h2>FLIP 误差图</h2><p>色标统一为 0–1；局部图裁自完整误差图，保留全图滤波上下文。3D Renewal+ 自比较 FLIP 已核验为零。</p>
<a href="flip.png"><img src="flip.png" alt="全图 FLIP 误差"></a>
<a href="flip_detail.png"><img src="flip_detail.png" alt="凹槽局部 FLIP 误差"></a>
<h2>运行记录与适用范围</h2>
<p>3D Renewal+ 使用当前渲染器的三维条件化路径。其余五模式使用原报告渲染器；已用当前路径追踪诊断程序重放每组每模式三个固定 8×8 区域的全部 128 个样本，均与原图在浮点容差内一致。这是局部版本核验，不等于全图等价证明。</p>
<p>128 spp 的 3D Renewal+ 为本次比较参考，未视为收敛真值；图像差异同时含模型差异、离散步长和采样噪声。原实现各 mode 的均值梯度计算方式仍有区别。下列历史耗时来自各次运行，机器负载不同，不用于严格性能排名。</p>
<ul>{diagnostics}</ul>
<div class="scroll"><table><tr><th>组</th><th>Mode</th><th>原始文件</th><th>记录耗时</th><th>Renderer SHA256 前缀</th></tr>{sources}</table></div>
</html>'''
    (REPORT/'comparison.html').write_text(content,encoding='utf-8')
    return verify_links(REPORT/'comparison.html')


def report(args):
    assert args.stage=='renewal_3d', 'Report uses the matched final 3D Renewal+ runs'
    groups,compatibility = read_groups(args)
    rows = measure(groups)
    flip_rows = flip(groups)
    pair_rows = renewal_vs_global(groups)
    plots(groups)
    diagnostics={}
    for g in groups:
        folder=Path(g['data'][MODE]['source']['folder'])
        log=(folder/'render.log').read_text(encoding='utf-8',errors='replace')
        diagnostics[g['key']]=dict(zero_length_ray_messages=log.count('Path tracing bounces ended with farT == 0'),
            segment_exit_messages=log.count('Reached end of ray, but not counting it as exited'))
    manifest=dict(complete=True, reference=MODE, changes=CHANGES,
        resolution=[256,256],spp=128,seed=SEED,step_size=.01,
        original_manifest=str(ORIGINAL),original_manifest_sha256=sha(ORIGINAL),
        report_script_sha256=sha(Path(__file__)),
        runner_sha256=sha(ROOT/'tools/surface_compare/gpis_shaderball_renewal_3d.py'),
        run_manifest_sha256=sha(REPORT/f'{args.stage}_run.json'),
        renderer_sha256=sha(EXE), compatibility_sha256=sha(REPORT/'renderer_compatibility.json'),
        flip_tool=dict(executable=str(DEFAULT_FLIP), version='1.7',
            executable_sha256=sha(DEFAULT_FLIP), metric='LDR-FLIP',pixels_per_degree=PPD,
            input='Original renderer filmic PNG, no resampling or exposure adjustment',
            help_sha256=sha(REPORT/'flip_tool_help.txt')),
        compatibility_scope=compatibility['scope'], detail_roi=ROI,model_roi=MODEL_ROI,
        block_normalization='3D Renewal+ mean luminance within full-resolution shaderball_box',
        diagnostics=diagnostics,statistics=rows,flip_statistics=flip_rows,
        renewal_vs_global_statistics=pair_rows,
        groups=[dict(key=g['key'],ell=g['ell'],sigma=g['sigma'],
            sources=[d['source'] for d in g['data'].values()],
            renewal_vs_global_record=dict(path=g['renewal_vs_global']['record'],
                sha256=sha(Path(g['renewal_vs_global']['record']))),
            flip_records=[dict(mode=m,path=d['flip']['record'],sha256=sha(Path(d['flip']['record']))) for m,d in g['data'].items()]) for g in groups],
        reproduction=['python -B tools/surface_compare/gpis_shaderball_renewal_3d.py render',
            'python -B tools/surface_compare/gpis_shaderball_renewal_3d.py compatibility',
            '<plot-python> -B tools/surface_compare/gpis_shaderball_renewal_3d.py report'])
    for name,values in [('metrics',rows),('flip_metrics',flip_rows),('renewal_vs_global_metrics',pair_rows)]:
        with (REPORT/f'{name}.csv').open('w',encoding='utf-8-sig',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(values[0]));writer.writeheader();writer.writerows(values)
    save(REPORT/'manifest.json',manifest)
    links=page(groups,rows,flip_rows,manifest)
    save(REPORT/'verification.json',dict(passed=True,local_links=links,
        compatibility_checks=len(compatibility['checks']),flip_self_checks=len(groups),
        generated_sha256={p.name:sha(p) for p in [REPORT/'comparison.html',REPORT/'manifest.json',
            REPORT/'metrics.csv',REPORT/'flip_metrics.csv',REPORT/'renewal_vs_global_metrics.csv',
            *[REPORT/f'{name}.png' for name in ('appearance','detail','difference','flip','flip_detail','renewal_vs_global')]]}))
    print(REPORT/'comparison.html',flush=True)
