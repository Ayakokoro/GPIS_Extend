"""Validated native-FLIP report for the two non-GP surface policies."""
from __future__ import annotations

import copy
import csv
import html
import os
import subprocess
from pathlib import Path

from figures import plt, Image, Rectangle
import numpy as np
from run import load, save, sha
from gpis_shaderball_flip import DEFAULT_FLIP, evaluate, verify_links
from gpis_shaderball_shadow_ablation_report import grayscale_flip, image_array
from gpis_shaderball_conditional_surface_policies import (
    ROOT, REPORT, OUT, GROUPS, GLOBAL, SEED, POLICIES, KEY, destination,
    original, renewal_destination, verified, normalized_scene,
)

BOUNDS = [44,164,213,216]
PPD = 67.0
MODES = ('global', 'renewal', *POLICIES)
LABELS = {'global':'GPIS Global', 'renewal':'GPIS Renewal+ original',
          'retain_endpoint':'global_conditional\nretain_endpoint',
          'restart_exterior':'global_conditional\nrestart_exterior'}
PAIRS = tuple((f'{p}_vs_{ref}',p,ref) for p in POLICIES for ref in ('renewal','global')) + (
    ('restart_vs_retain','restart_exterior','retain_endpoint'),)


def crop(image):
    x0,y0,x1,y1 = BOUNDS
    return image[y0:y1,x0:x1]


def relative(path):
    return Path(os.path.relpath(path, OUT)).as_posix()


def read_group(key, controls):
    folders = {'global':original(key, GLOBAL), 'renewal':renewal_destination(key,'renewal_3d'),
               **{p:destination(key,p) for p in POLICIES}}
    data = {mode:verified(folder) for mode,folder in folders.items()}
    archived_retain = verified(original(key,'global_conditional'))
    reference = data['retain_endpoint']['scene']
    for mode,d in data.items():
        r = d['source']['result']
        assert r['resolution']==[256,256] and r['spp']==128
        assert folders[mode].name == f'seed{SEED}'
        if 'seed' in r:
            assert r['seed']==SEED
        if mode in POLICIES:
            assert d['scene']['media'][0]['type']=='global_conditional'
            assert d['scene']['media'][0][KEY]==mode
            assert r['provenance']['renderer_sha256']==controls['current_renderer_sha256']
            assert r.get('discarded_surface_births') is not None, 'Conditional birth diagnostics required'
            assert normalized_scene(d['scene'])==normalized_scene(archived_retain['scene'])
        else:
            medium = d['scene']['media'][0]
            assert medium['type']=='sparse_conv_noise'
            assert medium['correlation_context']==('global' if mode=='global' else 'renewal+')
            assert not medium['1D_sampling'] and not medium['single_realization']
            assert medium['1D_sampling_scheme']=='uni' and not medium['1D_gradient_correlationXY']
            assert medium['step_size']==.01
            if mode=='renewal':
                assert r['provenance']['renderer_sha256']==controls['baseline_renderer_sha256']
        for section in ('primitives','bsdfs','camera','integrator','renderer'):
            a,b = copy.deepcopy(d['scene'][section]),copy.deepcopy(reference[section])
            if section=='renderer':
                a.pop('output_directory'); b.pop('output_directory')
            assert a==b, (key,mode,section)
        medium = d['scene']['media'][0]
        cov = medium.get('covariance',medium.get('gaussian_process',{}).get('covariance'))
        assert cov == reference['media'][0]['gaussian_process']['covariance']
        material = medium['material'] if mode in POLICIES else medium['phase_function']['bsdf']
        assert material==reference['media'][0]['material']
        d['png'] = folders[mode]/'render.png'
    delta = data['retain_endpoint']['rgb']-archived_retain['rgb']
    old_comparison = dict(archived_folder=archived_retain['source']['folder'],
        archived_result_sha256=sha(original(key,'global_conditional')/'result.json'),
        maximum_absolute_rgb_difference=float(np.abs(delta).max()),
        rgb_rmse=float(np.sqrt(np.mean(delta*delta))),
        bit_identical=bool(np.array_equal(data['retain_endpoint']['rgb'],archived_retain['rgb'])),
        within_prior_compatibility_tolerance=bool(np.allclose(data['retain_endpoint']['rgb'],
            archived_retain['rgb'],rtol=3e-5,atol=3e-7)))
    return dict(key=key,label='A' if key.endswith('0p02') else 'B',
                sigma=.02 if key.endswith('0p02') else .04,data=data,
                fresh_retain_vs_archived=old_comparison)


def statistics(group):
    rows = []
    for pair,test,reference in PAIRS:
        a,b = group['data'][test],group['data'][reference]
        delta = a['y']-b['y']
        halves = [x-y for x,y in zip(a['halves'],b['halves'])]
        assert np.allclose(delta,.5*(halves[0]+halves[1]),rtol=1e-4,atol=3e-7)
        noise = .5*(halves[0]-halves[1])
        for region in ('full','bottom'):
            select = crop if region=='bottom' else lambda x:x
            d,n = select(delta),select(noise)
            mean_a,mean_b = float(select(a['y']).mean()),float(select(b['y']).mean())
            half_means = [float(select(x).mean()) for x in halves]
            rows.append(dict(group=group['label'],pair=pair,test=test,reference=reference,region=region,
                pixels=d.size,test_mean_y=mean_a,reference_mean_y=mean_b,
                mean_delta_y=float(d.mean()),relative_mean_delta=mean_a/mean_b-1,
                luminance_rmse=float(np.sqrt(np.mean(d*d))),
                approximate_paired_mean_se=float(np.sqrt(np.sum(n*n))/d.size),
                paired_half_delta_means=half_means,
                same_mean_sign_in_both_halves=bool(half_means[0]*half_means[1]>0)))
    return rows


def flip(group):
    label,data = group['label'],group['data']
    inputs = OUT/'inputs'/label
    inputs.mkdir(parents=True,exist_ok=True)
    for mode,d in data.items():
        d['bottom_png'] = inputs/(mode+'_bottom.png')
        Image.fromarray(crop(image_array(d['png']))).save(d['bottom_png'])
    rows,records = [],[]
    group['flip'] = {}
    for pair,test,reference in PAIRS:
        a,b = data[test],data[reference]
        folder = OUT/'flip'/label/pair
        full = evaluate(DEFAULT_FLIP,b['png'],a['png'],folder/'full',PPD)
        bottom = evaluate(DEFAULT_FLIP,b['bottom_png'],a['bottom_png'],folder/'bottom_input',PPD)
        gray,errors = grayscale_flip(b['png'],a['png'],folder/'full_gray',full)
        group['flip'][pair] = dict(error_map=folder/'full'/full['error_map'],full=full,bottom=bottom)
        rows.append(dict(group=label,pair=pair,test=test,reference=reference,
            full=full['statistics'],bottom_cropped_input=bottom['statistics'],
            bottom_full_context_quantized_mean=float(crop(errors).mean()),
            quantization_absolute_bound=1/255))
        for kind,result in (('full',full),('bottom_input',bottom),('full_gray',gray)):
            path = folder/kind/'result.json'
            records.append(dict(group=label,pair=pair,kind=kind,path=relative(path),sha256=sha(path),result=result))
    print('FLIP complete',label,flush=True)
    return rows,records


def plots(groups):
    names = []
    for detail in (False,True):
        fig,axes = plt.subplots(len(groups),4,figsize=(16,(2.9 if detail else 4.3)*len(groups)),squeeze=False)
        for row,g in enumerate(groups):
            for col,mode in enumerate(MODES):
                ax=axes[row,col]; pixels=image_array(g['data'][mode]['png'])
                ax.imshow(crop(pixels) if detail else pixels,interpolation='nearest')
                if not detail:
                    x0,y0,x1,y1=BOUNDS
                    ax.add_patch(Rectangle((x0,y0),x1-x0,y1-y0,fill=False,edgecolor='#ffd050',lw=.8))
                ax.set_title(LABELS[mode]); ax.set_xticks([]); ax.set_yticks([])
                if col==0: ax.set_ylabel(f"{g['label']}: sigma={g['sigma']:g}")
        fig.suptitle('global_conditional: non-GP surface policy comparison\nMatched scene, 256 x 256, 128 spp, seed 20261024',fontsize=14)
        if detail:
            fig.tight_layout(rect=(0,.04,1,.86))
        else:
            fig.subplots_adjust(left=.04,right=.99,bottom=.07,top=.80,wspace=.08,hspace=.24)
        fig.text(.025,.012,'Original filmic PNGs; no exposure adjustment. Renewal+ is the original implementation baseline.',fontsize=10)
        name='appearance_bottom.png' if detail else 'appearance.png'
        fig.savefig(OUT/name,dpi=150); plt.close(fig); names.append(name)
    for detail in (False,True):
        fig,axes=plt.subplots(len(groups),4,figsize=(16,(3.0 if detail else 5.0)*len(groups)),squeeze=False)
        for row,g in enumerate(groups):
            for col,(pair,test,reference) in enumerate(PAIRS[:4]):
                f=g['flip'][pair]; ax=axes[row,col]; pixels=image_array(f['error_map'])
                ax.imshow(crop(pixels) if detail else pixels,interpolation='nearest')
                ax.set_title(f"{test}\nvs {LABELS[reference]}\nFull FLIP = {f['full']['statistics']['mean']:.6f}",fontsize=10)
                ax.set_xticks([]); ax.set_yticks([])
                if col==0: ax.set_ylabel(g['label'])
        fig.suptitle('Native local FLIP: each policy against both reference models\n'+('Fixed bottom crop of full-image maps' if detail else 'Full images')+'; 67 PPD, magma scale 0-1',fontsize=14)
        if detail:
            fig.tight_layout(rect=(0,.03,1,.85),h_pad=4.0)
        else:
            fig.subplots_adjust(left=.04,right=.99,bottom=.04,top=.82,wspace=.08,hspace=.32)
        name='flip_bottom.png' if detail else 'flip.png'
        fig.savefig(OUT/name,dpi=150); plt.close(fig); names.append(name)
    fig,axes=plt.subplots(len(groups),2,figsize=(12,2.8*len(groups)),squeeze=False)
    for row,g in enumerate(groups):
        delta=g['data']['restart_exterior']['y']-g['data']['retain_endpoint']['y']
        handle=axes[row,0].imshow(crop(delta),cmap='RdBu_r',vmin=-.12,vmax=.12,interpolation='nearest')
        axes[row,0].set_title('restart_exterior - retain_endpoint\nLinear luminance, scale +/-0.12')
        axes[row,1].imshow(crop(image_array(g['flip']['restart_vs_retain']['error_map'])),interpolation='nearest')
        axes[row,1].set_title('Policy change: FLIP\nCrop of full-image error map')
        axes[row,0].set_ylabel(g['label'])
        for ax in axes[row]: ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle('Effect of changing only the non-GP surface policy',fontsize=14)
    fig.tight_layout(rect=(0,.03,.93,.87)); fig.colorbar(handle,cax=fig.add_axes([.95,.15,.012,.65]))
    fig.savefig(OUT/'policy_change.png',dpi=150); plt.close(fig); names.append('policy_change.png')
    return names


def implementation_record(controls):
    path=OUT/'implementation.json'
    if not path.is_file():
        return None
    doc=load(path)
    assert doc['complete'] and doc['policy_key']==KEY
    assert doc['default']=='retain_endpoint' and tuple(doc['choices'])==POLICIES
    assert doc['renderer_sha256']==controls['current_renderer_sha256']
    assert doc['renewal_baseline_restored']
    for source in doc.get('source_files',[]):
        assert sha(Path(source['snapshot']))==source['sha256']
    for validation in doc.get('validations',[]):
        assert sha(Path(validation['path']))==validation['sha256']
    return dict(path=relative(path),sha256=sha(path),
        renderer_sha256=doc['renderer_sha256'],renewal_baseline_restored=True,
        source_snapshots_verified=len(doc.get('source_files',[])),
        validation_snapshots_verified=len(doc.get('validations',[])),
        test_summary=doc.get('test_summary'),scope=doc.get('scope'))


def write_page(groups,linear,flips,names,implementation=None):
    def link(path,label):
        return f'<a href="{html.escape(relative(path),quote=True)}">{html.escape(label)}</a>'
    rows=[]
    for g in groups:
        for pair,test,reference in PAIRS:
            f=next(r for r in flips if r['group']==g['label'] and r['pair']==pair)
            for region,region_label in (('full','全图'),('bottom','固定底部')):
                r=next(r for r in linear if r['group']==g['label'] and r['pair']==pair and r['region']==region)
                value=f['full']['mean'] if region=='full' else f['bottom_cropped_input']['mean']
                rows.append(f'<tr><td>{g["label"]}</td><td>{test}</td><td>{LABELS[reference]}</td><td>{region_label}</td><td>{value:.6f}</td><td>{r["relative_mean_delta"]:+.3%}</td><td>{r["luminance_rmse"]:.6f}</td><td>{r["approximate_paired_mean_se"]:.7f}</td></tr>')
    sources=[]
    for g in groups:
        for mode,d in g['data'].items():
            folder=Path(d['source']['folder'])
            links=' · '.join(link(folder/name,label) for name,label in (('render.png','PNG'),('render.pfm','线性 PFM'),('scene.json','场景'),('result.json','渲染记录')))
            sources.append(f'<tr><td>{g["label"]}</td><td>{LABELS[mode]}</td><td>{links}</td></tr>')
    images=''.join(f'<a href="{name}"><img src="{name}" alt="{name}"></a>' for name in names)
    diagnostics='；'.join(f'{g["label"]} {p}: paths={g["data"][p]["source"]["result"]["discarded_surface_births"]["paths"]}, shadow={g["data"][p]["source"]["result"]["discarded_surface_births"]["shadow_connections"]}' for g in groups for p in POLICIES)
    implementation_link=(' · '+link(OUT/'implementation.json','实现、回退与测试记录')) if implementation else ''
    content=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>global_conditional：非 GP 表面两种策略</title>
<style>body{{max-width:1700px;margin:28px auto;padding:0 24px;background:#f5f6f8;color:#253145;font:16px/1.65 system-ui}}h1{{font-size:28px}}img{{width:100%;background:white;margin:16px 0}}table{{width:100%;border-collapse:collapse;background:white}}td,th{{padding:9px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}}td:nth-child(-n+4),th:nth-child(-n+4){{text-align:left}}.scroll{{overflow:auto}}.note{{background:#e6edf5;padding:16px;border-radius:8px}}a{{color:#235e91}}</style>
<h1>global_conditional：保留端点 / 非 GP 表面重启</h1>
<p>A：ℓ=0.2，σ=0.02；B：ℓ=0.2，σ=0.04。相同相机、光照、材质、256×256、128 spp、seed={SEED}。两种策略均重新渲染，使用同一版程序，其他场景参数保持原值。</p>
<table><tr><th>配置值</th><th>普通非 GP 表面后的规则</th></tr><tr><td>retain_endpoint</td><td>保留根据旧观测采样的端点 F&gt;0 和完整梯度。</td></tr><tr><td>restart_exterior</td><td>丢弃旧观测，按该点先验重新采样 F&gt;0 和梯度。</td></tr></table>
<p class="note">策略统一用于表面发出的主路径与直接光分支。参照图为原实现 GPIS Renewal+ 和 GPIS Global；此前修改阴影条件化的 Renewal+ 另存为“地面条件保留变体”，不混入本页的原实现基线。这里比较的是两种有限记忆近似，均不等同于保留完整路径历史。128 spp 的 Global 参考也并非已收敛真值。</p>
<p>{link(REPORT/'comparison.html','原六模式基线')} · {link(REPORT/'shadow_condition_ablation/comparison.html','Renewal+ 地面条件保留变体')} · {link(OUT/'metrics.json','完整指标与来源')} · {link(OUT/'metrics.csv','指标 CSV')} · {link(OUT/'compatibility.json','兼容性检查')}{implementation_link}</p>
{images}
<h2>数值比较</h2><p>底部固定为 x=[44,213)、y=[164,216)。亮度使用原始线性 PFM；FLIP 直接调用本地 flip.exe 1.7，输入原始 filmic PNG，67 PPD。表中底部 FLIP 在无损 ROI 裁图上重新计算，边缘滤波上下文与全图不同；上方底部误差图则裁自全图 FLIP。保留全图上下文的量化 ROI 均值也记录在 metrics.json。</p>
<div class="scroll"><table><tr><th>组</th><th>测试</th><th>参考</th><th>区域</th><th>Mean FLIP</th><th>相对亮度差</th><th>亮度 RMSE</th><th>配对 SE</th></tr>{''.join(rows)}</table></div>
<p>配对 SE 使用奇偶半图估计，假设像素采样流近似独立，仅作单种子描述。具体的两半变化方向、全部 FLIP 输出与哈希见完整记录。数值更接近某个参考，不自动证明其模型与参考等价。</p>
<p>不支持的表面 birth 丢弃计数：{diagnostics}。</p>
<h2>原始输入</h2><div class="scroll"><table><tr><th>组</th><th>模式</th><th>文件</th></tr>{''.join(sources)}</table></div>
</html>'''
    (OUT/'comparison.html').write_text(content,encoding='utf-8')


def report(args):
    OUT.mkdir(parents=True,exist_ok=True)
    controls=load(OUT/'compatibility.json')
    assert controls['passed'] and set(args.groups)<=set(controls['groups'])
    for source in controls['sources']:
        folder=Path(source['folder'])
        assert sha(folder/'result.json')==source['result_sha256']
        verified(folder)
    groups=[read_group(key,controls) for key in args.groups]
    implementation=implementation_record(controls)
    help_result=subprocess.run([str(DEFAULT_FLIP),'--help'],capture_output=True,text=True,encoding='utf-8',errors='replace',check=True)
    help_text=help_result.stdout+help_result.stderr
    assert 'FLIP v1.7.' in help_text
    (OUT/'flip_tool_help.txt').write_text(help_text,encoding='utf-8')
    linear,flips,records=[],[],[]
    for group in groups:
        linear.extend(statistics(group)); f,r=flip(group); flips.extend(f); records.extend(r)
    names=plots(groups)
    sources=[dict(group=g['label'],mode=mode,folder=d['source']['folder'],
        result_sha256=sha(Path(d['source']['folder'])/'result.json'),result=d['source']['result'])
        for g in groups for mode,d in g['data'].items()]
    metrics=dict(complete=True,groups=args.groups,policy_key=KEY,policies=POLICIES,
        settings=dict(resolution=[256,256],spp=128,seed=SEED),bottom_bounds=BOUNDS,
        linear_statistics=linear,flip_statistics=flips,flip_records=records,sources=sources,
        fresh_retain_vs_archived=[dict(group=g['label'],**g['fresh_retain_vs_archived']) for g in groups],
        compatibility_sha256=sha(OUT/'compatibility.json'),
        implementation=implementation,
        tool=dict(path=str(DEFAULT_FLIP),sha256=sha(DEFAULT_FLIP),version='1.7',ppd=PPD),
        report_script_sha256=sha(Path(__file__)),figures={n:sha(OUT/n) for n in names},
        definitions=dict(relative_difference='(test_mean-reference_mean)/reference_mean',
            paired_se='sqrt(sum(((half_delta_A-half_delta_B)/2)^2))/pixel_count',
            bottom_flip='Native FLIP on lossless ROI input crops; boundary context differs from full-image filtering.',
            quantized_bottom_flip='ROI mean of full-image grayscale FLIP PNG; absolute quantization bound 1/255.',
            scope='Original Renewal+ implementation baseline, not a claim of exact paper equivalence. Both conditional policies are latest-observation approximations, not exact globally fixed GP transport.'))
    save(OUT/'metrics.json',metrics)
    fields=('group','test','reference','region','mean_flip','mean_delta_y','relative_mean_delta','luminance_rmse','approximate_paired_mean_se')
    with (OUT/'metrics.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for r in linear:
            f=next(f for f in flips if f['group']==r['group'] and f['pair']==r['pair'])
            row={k:r[k] for k in fields if k!='mean_flip'}
            row['mean_flip']=f['full' if r['region']=='full' else 'bottom_cropped_input']['mean']
            writer.writerow(row)
    write_page(groups,linear,flips,names,implementation)
    save(OUT/'verification.json',dict(passed=True,render_records=len(sources),flip_records=len(records),
        local_links=verify_links(OUT/'comparison.html'),
        generated_sha256={n:sha(OUT/n) for n in ('comparison.html','metrics.json','metrics.csv','flip_tool_help.txt',*names)}))
    for r in linear:
        if r['region']=='bottom':
            f=next(f for f in flips if f['group']==r['group'] and f['pair']==r['pair'])
            print(r['group'],r['pair'],'bottom relative',r['relative_mean_delta'],'RMSE',r['luminance_rmse'],
                  'FLIP full',f['full']['mean'],'bottom',f['bottom_cropped_input']['mean'],flush=True)
    print(OUT/'comparison.html',flush=True)
