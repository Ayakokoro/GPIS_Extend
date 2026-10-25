"""Matched long-correlation shaderball renders and a local comparison page.

  python tools/surface_compare/shaderball_length.py render
  python tools/surface_compare/shaderball_length.py render --stage step_base --size 96 --spp 64 --modes sparse_3d_global
  python tools/surface_compare/shaderball_length.py render --stage step_half --size 96 --spp 64 --step .00125 --modes sparse_3d_global
  <plot-python> tools/surface_compare/shaderball_length.py figures

Default: ell=.1, sigma=.01, 256 squared, 128 spp, studio and grazing lights.
The existing fine SDF and density are reusable at fixed sigma: classic_local
uses the scene covariance, not the baked alpha grid. No old outputs are edited.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import subprocess
import time
from pathlib import Path

import numpy as np
from run import ROOT, OUT as SURFACE_OUT, RENDER_MODES, load, save, pfm, scene as base_scene

LIGHTINGS = ('studio', 'grazing')
LABELS = ('classic_local', 'classic_global', 'global_conditional', 'GPIS global 3D')
ROI = (.14, .34, .50, .62)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def root(args):
    return SURFACE_OUT / ('long_correlation_ell' + f'{args.length:g}'.replace('.', 'p'))


def folder(args, lighting, mode, stage=None):
    return root(args) / (stage or args.stage) / lighting / mode / f'seed{args.seed}'


def field_path(args):
    return SURFACE_OUT / ('fields' if args.coarse else 'fields_fine') / 'rounded_shaderball.nvdb'


def scene(args, lighting, mode):
    s = base_scene(f'{lighting}_ell0p025', mode, args.stage, args.size,
                   args.spp, args.seed, field_path(args), args.step)
    m = s['media'][0]
    covariance = m['covariance'] if mode == 'classic_local' else m['gaussian_process']['covariance']
    covariance['lengthScale'] = args.length
    s['renderer']['output_directory'] = str(folder(args, lighting, mode))
    s['renderer']['output_buffers'] = [dict(type='color', hdr_output_file='samples.pfm',
        sample_variance=True, two_buffer_variance=True)]
    return s


def render_one(args, lighting, mode, provenance):
    s = scene(args, lighting, mode)
    d = folder(args, lighting, mode)
    if (d/'result.json').exists():
        old = load(d/'result.json')
        if old['provenance'] != provenance or load(d/'scene.json') != s:
            raise RuntimeError('Incompatible existing render: ' + str(d))
        for name in ('scene.json', 'render.pfm', 'render.png', 'samplesVariance.pfm', 'samplesA.pfm', 'samplesB.pfm'):
            if digest(d/name) != old['sha256'][name]:
                raise RuntimeError('Changed render source: ' + str(d/name))
        print('Reuse', d.relative_to(root(args)), flush=True)
        return old
    if (d/'render.pfm').exists():
        raise RuntimeError('Unvalidated existing render: ' + str(d))
    save(d/'scene.json', s)
    print('Render', d.relative_to(root(args)), flush=True)
    start = time.monotonic()
    with (d/'render.log').open('w', encoding='utf-8') as log:
        process = subprocess.run([str(ROOT/'build/Release/tungsten.exe'), '-t', str(args.threads),
            '-s', str(args.seed), '-d', str(d), str(d/'scene.json')], cwd=ROOT,
            stdout=log, stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, 'BELOW_NORMAL_PRIORITY_CLASS', 0))
    log = (d/'render.log').read_text(encoding='utf-8', errors='replace')
    if (process.returncode or 'Finished render.' not in log or
        not re.search(rf'Completed {args.spp}/{args.spp} spp', log) or
        re.search(r'Caught an internal error|unrecoverable|Gradient invalid|Unable to load|'
                  r'Wrong normal direction|Failed to sample gradient|Gradient zero|Conditioning .* error', log, re.I)):
        raise RuntimeError('Render failed: ' + str(d))
    a = pfm(d/'render.pfm')
    assert a.shape == (args.size, args.size, 3)
    assert np.allclose((pfm(d/'samplesA.pfm')+pfm(d/'samplesB.pfm'))/2, a, rtol=2e-6, atol=2e-7)
    pfm(d/'samplesVariance.pfm')
    discarded = re.search(r'discarded surface births: (\{.*\})', log)
    result = dict(seconds=time.monotonic()-start, resolution=[args.size, args.size],
        spp=args.spp, mean=float(a.mean()), maximum=float(a.max()), provenance=provenance,
        discarded_surface_births=json.loads(discarded[1]) if discarded else None,
        sha256={name: digest(d/name) for name in
                ('scene.json', 'render.pfm', 'render.png', 'samplesVariance.pfm', 'samplesA.pfm', 'samplesB.pfm')})
    save(d/'result.json', result)
    print('Done', d.relative_to(root(args)), round(result['seconds'], 2),
          'mean', round(result['mean'], 7), flush=True)
    return result


def render(args):
    field = field_path(args)
    meta = load(Path(str(field)+'.json'))
    assert meta['sigma'] == .01 and meta['verification']['sdf_formats_identical']
    provenance = dict(renderer_sha256=digest(ROOT/'build/Release/tungsten.exe'),
        nvdb_sha256=digest(field), vdb_sha256=digest(field.with_name('rounded_shaderball_sdf.vdb')),
        field_metadata_sha256=digest(Path(str(field)+'.json')))
    sources = []
    for lighting in args.lightings:
        for mode in args.modes:
            result = render_one(args, lighting, mode, provenance)
            sources.append(dict(lighting=lighting, mode=mode, folder=str(folder(args, lighting, mode)), result=result))
    save(root(args)/args.stage/'run.json', dict(parameters=vars(args), sources=sources))


def masks(n):
    y, x = np.mgrid[:n, :n]
    central = ((x+.5)/n-.5)**2 + ((y+.5)/n-.5)**2 < (5.5/16)**2
    groove = np.zeros((n,n), dtype=bool)
    x0, y0, x1, y1 = [int(v*n) for v in ROI]
    groove[y0:y1, x0:x1] = True
    return dict(full=np.ones_like(central), central_circle=central, groove=groove)


def verified_source(d):
    result = load(d/'result.json')
    for name, expected in result['sha256'].items():
        assert digest(d/name) == expected, str(d/name)
    a = pfm(d/'render.pfm')
    assert np.max(np.ptp(a, axis=2)) < 1e-7, 'Scene must be achromatic'
    return dict(result=result, scene=load(d/'scene.json'), image=a,
                variance=pfm(d/'samplesVariance.pfm')[:,:,0],
                halves=[pfm(d/f'samples{c}.pfm')[:,:,0] for c in ('A','B')])


def collect(args):
    data = {}; statistics = []; sources = []; consistency = []
    for lighting in LIGHTINGS:
        data[lighting] = {mode: verified_source(folder(args, lighting, mode)) for mode in RENDER_MODES}
        ref = data[lighting]['classic_global']
        regions = masks(ref['image'].shape[0])
        for mode, item in data[lighting].items():
            s = item['scene']; m = s['media'][0]
            for key in ('camera', 'primitives', 'integrator', 'renderer'):
                a, b = copy.deepcopy(s[key]), copy.deepcopy(ref['scene'][key])
                if key == 'renderer':
                    a.pop('output_directory'); b.pop('output_directory')
                assert a == b, (lighting, mode, key)
            cov = m['covariance'] if mode == 'classic_local' else m['gaussian_process']['covariance']
            assert cov == dict(type='squared_exponential', sigma=.01, lengthScale=args.length)
            material = m['phase_function']['bsdf'] if mode == 'sparse_3d_global' else m['material']
            assert material == dict(type='mirror', albedo=.9)
            assert item['result']['provenance'] == ref['result']['provenance']
            if mode != 'classic_local':
                assert m['gaussian_process'] == ref['scene']['media'][0]['gaussian_process']
            if mode == 'sparse_3d_global':
                assert m['correlation_context'] == 'global' and not m['1D_sampling'] and not m['single_realization']
            image = item['image'][:,:,0]
            delta = image-ref['image'][:,:,0]
            halves = [a-b for a,b in zip(item['halves'],ref['halves'])]
            delta_variance = .25*(halves[0]-halves[1])**2
            for region, mask in regions.items():
                n = int(mask.sum()); mean = float(image[mask].mean())
                reference = float(ref['image'][:,:,0][mask].mean())
                statistics.append(dict(lighting=lighting, mode=mode, region=region, pixels=n,
                    mean=mean, standard_error=float(np.sqrt(item['variance'][mask].sum())/n),
                    difference=float(delta[mask].mean()), relative_difference=(mean/reference-1),
                    paired_difference_standard_error=float(np.sqrt(delta_variance[mask].sum())/n)))
            sources.append(dict(lighting=lighting, mode=mode, folder=str(folder(args,lighting,mode)), result=item['result']))
        consistency.append(dict(lighting=lighting, camera_lights_material_kernel_integrator_matched=True))
    resolutions = {(tuple(x['result']['resolution']), x['result']['spp']) for x in sources}
    assert len(resolutions) == 1
    return data, statistics, sources, consistency


def step_checks(args):
    results = []
    for lighting in LIGHTINGS:
        paths = [folder(args,lighting,'sparse_3d_global',stage) for stage in ('step_base','step_half')]
        if not all((p/'result.json').exists() for p in paths):
            raise RuntimeError('Missing GPIS step-size check: '+lighting)
        base, half = [verified_source(p) for p in paths]
        assert base['result']['provenance'] == half['result']['provenance']
        a, b = copy.deepcopy(base['scene']), copy.deepcopy(half['scene'])
        steps = [s['media'][0].pop('step_size') for s in (a,b)]
        for s in (a,b): s['renderer'].pop('output_directory')
        assert a == b and steps[1] == steps[0]/2
        delta = half['image'][:,:,0]-base['image'][:,:,0]
        halves = [a-b for a,b in zip(half['halves'],base['halves'])]
        variance = .25*(halves[0]-halves[1])**2
        for name, mask in masks(delta.shape[0]).items():
            mean = float(delta[mask].mean()); se = float(np.sqrt(variance[mask].sum())/mask.sum())
            results.append(dict(lighting=lighting, region=name, steps=steps,
                resolution=base['result']['resolution'], spp=base['result']['spp'],
                baseline_mean=float(base['image'][:,:,0][mask].mean()),
                refined_mean=float(half['image'][:,:,0][mask].mean()),
                difference=mean, relative_difference=mean/float(base['image'][:,:,0][mask].mean()),
                paired_standard_error=se, exceeds_approximate_95_percent_sampling_interval=abs(mean)>1.96*se,
                sources=[dict(folder=str(p),result=v['result']) for p,v in zip(paths,(base,half))]))
    return results


def figures(args):
    from figures import plt, Image, COLORS, Rectangle
    from plane_figures import filmic, blocks
    data, statistics, sources, consistency = collect(args)
    checks = step_checks(args)
    for check in checks:
        for source in check['sources']:
            assert source['result']['provenance'] == sources[0]['result']['provenance']
    dest = root(args)
    settings = sources[0]['result']; n = settings['resolution'][0]
    actual_field = Path(data[LIGHTINGS[0]]['classic_local']['scene']['media'][0]['field']['file'])
    spacing = load(Path(str(actual_field)+'.json'))['dx']
    gpis = data[LIGHTINGS[0]]['sparse_3d_global']['scene']['media'][0]
    step, impulses = gpis['step_size'], gpis['impulse_density']
    def stat(lighting,mode,region='central_circle'):
        return next(r for r in statistics if (r['lighting'],r['mode'],r['region']) == (lighting,mode,region))
    spatial_statistics = []
    for lighting in LIGHTINGS:
        fig, axes = plt.subplots(2,4,figsize=(18,9),gridspec_kw={'height_ratios':[1,.65]})
        for i,mode in enumerate(RENDER_MODES):
            im = np.asarray(Image.open(folder(args,lighting,mode)/'render.png').convert('RGB'))
            r = stat(lighting,mode)
            x0,y0,x1,y1 = [int(v*n) for v in ROI]
            ax = axes[0,i]; ax.imshow(im,interpolation='nearest')
            ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffcf52',lw=1))
            ax.set_title(f'{LABELS[i]}\n中心区域 L={r["mean"]:.5f} ± {1.96*r["standard_error"]:.5f}',color=COLORS[i],fontsize=11)
            axes[1,i].imshow(im[y0:y1,x0:x1],interpolation='nearest')
            ax.axis('off');axes[1,i].axis('off')
        label = '宽条光源' if lighting == 'studio' else '掠射光照'
        fig.suptitle(f'Shaderball · {label} · ℓ={args.length:g}，σ=0.01\n镜面反射率 0.9；{n}×{n}，{settings["spp"]} spp；SDF 网格 {spacing:g}',fontsize=18)
        fig.text(.035,.035,'下排为黄色框的原图放大；统一 filmic 显示，无单独调亮或降噪。± 为中心圆形区域均值的近似 95% 采样误差。\nGPIS 为沿路径保持不变的 3D 稀疏卷积场；其真实镜面事件跳过普通光源 NEE，噪声与前三种模式不同。',fontsize=10,color='#555b65')
        fig.subplots_adjust(left=.025,right=.99,top=.83,bottom=.11,wspace=.05,hspace=.06)
        fig.savefig(dest/f'appearance_{lighting}.png',dpi=150);plt.close(fig)
        fig, axes = plt.subplots(1,3,figsize=(13,4.8))
        ref = data[lighting]['classic_global']['image']
        ref_mean = stat(lighting,'classic_global')['mean']
        for i,mode in enumerate(('classic_local','global_conditional','sparse_3d_global')):
            delta = (blocks(data[lighting][mode]['image'])-blocks(ref)).mean(2)/ref_mean
            half_differences = [blocks((a-b)[:,:,None])[:,:,0]/ref_mean
                for a,b in zip(data[lighting][mode]['halves'],data[lighting]['classic_global']['halves'])]
            noise = .25*(half_differences[0]-half_differences[1])**2
            central_blocks = masks(16)['central_circle']
            spatial_statistics.append(dict(lighting=lighting,mode=mode,blocks=int(central_blocks.sum()),
                observed_rms_relative_to_reference_mean=float(np.sqrt(np.mean(delta[central_blocks]**2))),
                estimated_noise_rms_relative_to_reference_mean=float(np.sqrt(np.mean(noise[central_blocks])))))
            im = axes[i].imshow(delta,cmap='RdBu_r',vmin=-.2,vmax=.2,interpolation='nearest')
            axes[i].set_title(mode if i<2 else 'GPIS global 3D');axes[i].axis('off')
        fig.suptitle(f'{label} · 相对 classic_global 的 16×16 分区亮度差',fontsize=16)
        fig.colorbar(im,cax=fig.add_axes([.94,.23,.012,.53]),label='ΔL / 参考中心区域平均亮度')
        fig.text(.035,.04,'先在线性辐亮度中做区域平均与相减；统一色标 ±20%，超出范围截断显示。图中仍包含采样噪声。',fontsize=10,color='#555b65')
        fig.subplots_adjust(left=.02,right=.91,top=.82,bottom=.16,wspace=.04)
        fig.savefig(dest/f'difference_{lighting}.png',dpi=160);plt.close(fig)
    fig, axes = plt.subplots(2,4,figsize=(15,8))
    for row,lighting in enumerate(LIGHTINGS):
        for col,mode in enumerate(RENDER_MODES):
            ax=axes[row,col];ax.imshow(filmic(blocks(data[lighting][mode]['image'])),interpolation='nearest')
            ax.set_xticks([]);ax.set_yticks([])
            if row==0: ax.set_title(LABELS[col],color=COLORS[col],fontsize=11)
            if col==0: ax.set_ylabel('宽条光源' if lighting=='studio' else '掠射光照')
    fig.suptitle(f'ℓ={args.length:g} · 16×16 分区平均外观',fontsize=18)
    fig.text(.035,.025,'在线性图中求分区平均，再应用相同 filmic 显示；该图用于观察大尺度差别，原始图另行保留。',fontsize=10,color='#555b65')
    fig.subplots_adjust(left=.04,right=.99,top=.88,bottom=.08,wspace=.03,hspace=.08)
    fig.savefig(dest/'block_means.png',dpi=150);plt.close(fig)
    validation = dict(matched_scene_groups=consistency,gpis_step_half=checks,
        field_note='SDF and density reused at fixed sigma=.01. Baked alpha and length_scale metadata are unused by classic_local; its gradient covariance comes from the scene kernel.',
        mean_gradient_evaluation=dict(classic_local='normalized trilinear SDF gradient',
            classic_global='trilinear SDF gradient',global_conditional='SdfQueries trilinear SDF gradient',
            sparse_3d_global='TabulatedMean forward difference with epsilon=.001; plus realized sparse-noise gradient'),
        sdf_spacing=spacing,
        limits='Sampling errors exclude model and discretization bias. Step halving is a sensitivity check, not proof of convergence; no finite-impulse or SDF refinement claim is made.')
    save(dest/'validation.json',validation)
    save(dest/'manifest.json',dict(length_scale=args.length,sigma=.01,mirror_albedo=.9,
        resolution=settings['resolution'],spp=settings['spp'],seed=args.seed,
        statistics=statistics,sources=sources,roi_normalized=ROI,
        spatial_block_statistics=spatial_statistics,
        spatial_block_note='RMS of 16x16 block differences inside the central circle, divided by reference central mean. Noise RMS estimated from paired even/odd halves; descriptive diagnostic, not a significance test.',
        central_region='Circle centered at (.5,.5), radius 5.5/16 of image width; excludes most background but is not a segmented object mask.',
        uncertainty='Per-sample variance for region means; paired even/odd halves for differences. Approximate 95% sampling intervals only.',
        gpis=dict(type='3D sparse convolution',independent_realization_per_path=True,
                  fixed_realization_within_path=True,impulse_density=impulses,step_size=step),
        conditional='Latest field value and gradient conditioning, not a full fixed realization.',
        field_reuse=validation['field_note'],mean_gradient_evaluation=validation['mean_gradient_evaluation'],
        figures=[p.name for p in dest.glob('*.png')]))
    with (dest/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(statistics[0]));writer.writeheader();writer.writerows(statistics)
    rows=''
    for lighting in LIGHTINGS:
        for mode in RENDER_MODES:
            r,g=stat(lighting,mode),stat(lighting,mode,'groove')
            reference=stat(lighting,'classic_global')['mean']
            relative_error=1.96*r['paired_difference_standard_error']/reference
            rows+=f'<tr><td>{lighting}</td><td>{mode}</td><td>{r["mean"]:.6f} ± {1.96*r["standard_error"]:.6f}</td><td>{100*r["relative_difference"]:+.2f}% ± {100*relative_error:.2f}%</td><td>{g["mean"]:.6f}</td><td>{100*g["relative_difference"]:+.2f}%</td></tr>'
    check_rows=''.join(f'<tr><td>{r["lighting"]}</td><td>{r["region"]}</td><td>{100*r["relative_difference"]:+.3f}%</td><td>{r["difference"]:+.6f} ± {1.96*r["paired_standard_error"]:.6f}</td><td>{"是" if r["exceeds_approximate_95_percent_sampling_interval"] else "否"}</td></tr>' for r in checks if r['region']!='full')
    cards=''.join(f'<h2>{"宽条光源" if l=="studio" else "掠射光照"}</h2><a href="appearance_{l}.png"><img src="appearance_{l}.png"></a><a href="difference_{l}.png"><img src="difference_{l}.png"></a>' for l in LIGHTINGS)
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball ℓ={args.length:g} 四模式对照</title>
<style>body{{max-width:1500px;margin:35px auto;padding:0 24px;font:16px/1.65 system-ui;color:#253145;background:#f5f6f8}}h1{{font-size:28px}}img{{width:100%;background:white}}a{{color:#235e91}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:right}}td:first-child,td:nth-child(2){{text-align:left}}.note{{background:#e7edf5;padding:16px;border-radius:10px}}</style>
<h1>Shaderball：长相关长度 ℓ={args.length:g}</h1><p>classic_local / classic_global / global_conditional / GPIS global 3D。σ=0.01，镜面反射率 0.9；{n}×{n}，{settings['spp']} spp；相同相机、光源和 SDF 样本，网格间距 {spacing:g}。</p>
<div class="note">GPIS 在同一条路径上保留同一个三维稀疏卷积场，不同路径使用独立实现；每格 {impulses} 个脉冲，步长 {step:g}。conditional 保留最近一次场值和梯度。GPIS 的真实镜面事件跳过普通光源 NEE，采样噪声更大。<br>本轮固定 σ 增大 ℓ，同时减小了梯度方差；与旧 ℓ=0.025 图比较时，高光变化也包含这一粗糙度变化。旧图的分辨率和 SDF 网格也不同，跨轮比较只能定性观察外观。本页四模式之间的参数保持一致。</div>
<p><a href="metrics.csv">线性统计 CSV</a> · <a href="manifest.json">参数与来源</a> · <a href="validation.json">验证记录</a></p>{cards}
<h2>分区平均外观</h2><a href="block_means.png"><img src="block_means.png"></a>
<h2>线性亮度统计</h2><p>中心区域为图像中心、半径为图像宽度 5.5/16 的圆；凹槽区域为黄色矩形。相对差均以 classic_global 为基准。± 为近似 95% 采样误差；中心相对差的误差由配对差值估计，并除以测得的参考亮度，单位为百分点。更多区域与误差见 CSV。</p>
<table><tr><th>布光</th><th>模式</th><th>中心平均亮度</th><th>中心相对差 ± 采样误差</th><th>凹槽平均亮度</th><th>凹槽相对差</th></tr>{rows}</table>
<h2>GPIS 步长敏感性</h2><p>以下为 {checks[0]['resolution'][0]}²、{checks[0]['spp']} spp，步长由 {checks[0]['steps'][0]} 减至 {checks[0]['steps'][1]} 的同种子对照。误差只包含采样误差；两档步长不能证明收敛，也未消除有限脉冲与 SDF 离散化误差。</p>
<table><tr><th>布光</th><th>区域</th><th>相对变化</th><th>亮度变化 ± 近似95%采样误差</th><th>超出采样区间</th></tr>{check_rows}</table>
<p>四种模式共享 SDF 样本，但保留各自现有的梯度规则：classic_local 将三线性梯度单位化；classic_global 与 conditional 使用三线性 SDF 导数；GPIS 的均值梯度使用步长 0.001 的前向差分，再加上实际稀疏卷积场的梯度。因此图中差异也可能包含梯度计算方法的影响。密度网格在固定 σ 下复用，烘焙文件中的 alpha 不参与本轮计算，四模式的相关长度均为 ℓ={args.length:g}。</p></html>'''
    (dest/'comparison.html').write_text(html,encoding='utf-8')
    print(dest/'comparison.html',flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['render','figures'])
    parser.add_argument('--length',type=float,default=.1)
    parser.add_argument('--stage',default='final')
    parser.add_argument('--size',type=int,default=256)
    parser.add_argument('--spp',type=int,default=128)
    parser.add_argument('--threads',type=int,default=32)
    parser.add_argument('--seed',type=int,default=20261024)
    parser.add_argument('--step',type=float,default=.0025)
    parser.add_argument('--coarse',action='store_true')
    parser.add_argument('--lightings',nargs='+',choices=LIGHTINGS,default=list(LIGHTINGS))
    parser.add_argument('--modes',nargs='+',choices=RENDER_MODES,default=list(RENDER_MODES))
    args=parser.parse_args()
    if min(args.size,args.spp,args.threads)<=0 or args.spp%2 or args.size%16:
        parser.error('Positive sizes, even spp, and size divisible by 16 are required')
    if not all(math.isfinite(v) and v>0 for v in (args.length,args.step)):
        parser.error('Length and step must be positive and finite')
    if args.command=='render': render(args)
    else: figures(args)
