"""Audit completed groove renders and publish seed-level experimental findings."""
from __future__ import annotations
import argparse
import html
from pathlib import Path
import struct

import numpy as np

import test_conditional_groove as g


def common_wall_mask(scene):
    cam=scene['camera'];n=cam['resolution'][0]
    p=np.array(cam['transform']['position']);f=np.array(cam['transform']['look_at'])-p
    f/=np.linalg.norm(f)
    r=np.cross(f,np.array(cam['transform']['up']));r/=np.linalg.norm(r)
    u=np.cross(r,f)
    pixel=(np.arange(n)+.5)*2/n-1
    xx,yy=np.meshgrid(pixel,-pixel)
    ray=xx[...,None]*r+yy[...,None]*u+f/np.tan(np.deg2rad(cam['fov']/2))
    hit=p+((.5*g.W-p[0])/ray[...,0])[...,None]*ray
    return g.roi_mask(cam)&(hit[...,1]>=-.45*g.W)


def independent_difference(a,b):
    from scipy.stats import t
    a,b=np.asarray(a),np.asarray(b)
    delta=float(b.mean()-a.mean());interval=None
    if min(len(a),len(b))>=2:
        sa,sb=a.var(ddof=1)/len(a),b.var(ddof=1)/len(b)
        half=0.0
        if sa+sb>0:
            df=(sa+sb)**2/(sa**2/(len(a)-1)+sb**2/(len(b)-1))
            half=float(t.ppf(.975,df)*np.sqrt(sa+sb))
        interval=[delta-half,delta+half]
    return dict(classic_mean=float(a.mean()),conditional_mean=float(b.mean()),
                delta=delta,delta_ci95=interval,relative_delta=delta/float(a.mean()),
                n_classic=len(a),n_conditional=len(b))


def run(out):
    summary = g.report(out)
    records = [g.load(p) for p in sorted(out.glob('*/**/result.json'))]
    complete = [r for r in records if r['status']=='complete']
    validations = []
    for record in complete:
        folder = (out/record['scene']).parent
        scene = g.load(folder/'scene.json')
        image = g.validate_output(folder, record)
        if record['scene_sha256'] != g.digest(folder/'scene.json'):
            raise RuntimeError('Scene changed after rendering')
        png = (folder/'render.png').read_bytes()
        if png[:8]!=b'\x89PNG\r\n\x1a\n' or struct.unpack('>II',png[16:24])!=(record['size'],record['size']):
            raise RuntimeError('Native PNG dimensions differ')
        mask = g.roi_mask(scene['camera'])
        if not np.isclose(float(image[mask].mean()),record['roi_mean'],rtol=0,atol=1e-12):
            raise RuntimeError('ROI statistic differs')
        stage = record['scene'].split('/')[0]
        manifest = g.load(out/stage/'manifest.json')
        if manifest['executable_sha256'] != record['executable_sha256']:
            raise RuntimeError('Executable provenance differs')
        validations.append(record['scene'])
    paired = 0
    for path in out.glob('*/**/global_conditional/*/scene.json'):
        baseline = path.parents[1].parent/'classic_global'/path.parent.name/'scene.json'
        if not (path.parent/'result.json').exists() or not (baseline.parent/'result.json').exists():
            continue
        if any(g.load(p)['status']!='complete' for p in (path.parent/'result.json',baseline.parent/'result.json')):
            continue
        a,b = g.load(baseline),g.load(path)
        for scene in (a,b):
            scene['renderer'].pop('output_directory',None)
            scene['media'][0].pop('type')
            for key in list(scene['media'][0]):
                if key.startswith('tracking_'):
                    scene['media'][0].pop(key)
        if a!=b:
            raise RuntimeError(f'Unmatched comparison: {path}')
        paired += 1
    assets={}
    executables={}
    for path in out.glob('*/manifest.json'):
        manifest=g.load(path)
        exe=Path(manifest['executable'])
        if exe not in executables:
            executables[exe]=g.digest(exe)
        if executables[exe]!=manifest['executable_sha256']:
            raise RuntimeError('Renderer executable changed after stage preparation')
        for name,files in manifest.get('assets',{}).items():
            for file,expected in files.items():
                asset=out/'fields'/name/file
                if asset not in assets:
                    assets[asset]=g.digest(asset)
                if assets[asset]!=expected:
                    raise RuntimeError('Field changed after stage preparation')
    audit=dict(validated_renders=len(validations),matched_classic_conditional_scene_pairs=paired,
               field_assets_verified=len(assets),executable_files_verified=len(executables),failed_renders=len(summary['failed']),
               finite_nonnegative_pfm=True,native_png_dimensions_verified=True,
               scene_and_executable_provenance_verified=True,roi_values_recomputed=True)
    g.save(out/'validation.json',audit)

    by_key={(r['scene'].split('/')[0],r['group'],r['angle'],r['variant'],r['mode'],r['seed_id']):r for r in complete}
    fractions={}
    for mode in g.MODES[:2]:
        rows=[]
        for seed in g.SEEDS:
            total=by_key.get(('matrix64_spp64','A_strong',45,'total',mode,seed))
            single=by_key.get(('components64_spp64','A_strong',45,'single',mode,seed))
            multiple=by_key.get(('components64_spp64','A_strong',45,'multiple',mode,seed))
            if total and single and multiple:
                rows.append(dict(seed=seed,total=total['roi_mean'],single=single['roi_mean'],
                                 multiple=multiple['roi_mean'],
                                 fraction=multiple['roi_mean']/total['roi_mean'],
                                 residual=total['roi_mean']-single['roi_mean']-multiple['roi_mean']))
        if rows:
            fractions[mode]=dict(rows=rows,multiple_fraction=g.mean_interval([r['fraction'] for r in rows]),
                                 decomposition_residual=g.mean_interval([r['residual'] for r in rows]))

    controls={}
    variants=('fine','expanded','no_nee','bounces','tracking')
    for mode in g.MODES[:2]:
        controls[mode]={}
        for variant in variants:
            deltas=[]
            for seed in g.SEEDS:
                base=by_key.get(('controls32_spp64','A_strong',45,'total',mode,seed))
                item=by_key.get(('controls32_spp64','A_strong',45,variant,mode,seed))
                if base and item:
                    deltas.append(item['roi_mean']-base['roi_mean'])
            if deltas:
                controls[mode][variant]=g.mean_interval(deltas)
    angles={}
    for angle in (30,45,60):
        values={}
        pixels=0
        for mode in g.MODES[:2]:
            values[mode]=[]
            for seed in g.SEEDS[:3]:
                record=by_key.get(('angles32_spp64','A_strong',angle,'total',mode,seed))
                if record:
                    folder=(out/record['scene']).parent
                    mask=common_wall_mask(g.load(folder/'scene.json'))
                    pixels=int(mask.sum())
                    if not pixels:
                        raise RuntimeError('Empty common-wall angle ROI')
                    values[mode].append(float(g.read_image(folder/'render.pfm')[mask].mean()))
        if all(values.values()):
            angles[str(angle)]=dict(independent_difference(values['classic_global'],values['global_conditional']),roi_pixels=pixels)
    findings=dict(audit=audit,components=fractions,controls_paired_delta=controls,common_wall_angles=angles,
                  caveat='Exploratory seed-level intervals; controls with wide intervals do not prove equivalence.')
    g.save(out/'findings.json',findings)
    matrix=[c for c in summary['comparisons'] if c['stage']=='matrix64_spp64' and 'delta' in c]
    images=out/'figures'
    images.mkdir(exist_ok=True)

    rows=[]
    for c in matrix:
        ci=c['delta_ci95']
        ci_text='—' if ci is None else f'[{ci[0]:+.4f}, {ci[1]:+.4f}]'
        ratio=c['relative_delta']
        rows.append(f"<tr><td>{c['group']}</td><td>{c['roi']['classic_global']['mean']:.4f}</td>"
                    f"<td>{c['roi']['global_conditional']['mean']:.4f}</td><td>{100*ratio:+.1f}%</td><td>{ci_text}</td></tr>")
    gallery=[]
    for c in matrix:
        for mode in g.MODES[:2]:
            source=out/c['stage']/'averages'/c['group']/f"view{c['angle']}"/c['variant']/f'{mode}.pfm'
            image=g.read_image(source)
            target=images/f"{c['group']}_{mode}_exposure025.png"
            g.write_png(target,g.filmic(image*.25))
            gallery.append(f'<figure><img width="256" src="{target.relative_to(out).as_posix()}"><figcaption>{c["group"]} · {mode}</figcaption></figure>')
    component_rows=[]
    for mode,data in fractions.items():
        frac=data['multiple_fraction'];res=data['decomposition_residual']
        component_rows.append(f"<tr><td>{mode}</td><td>{100*frac['mean']:.2f}%</td><td>{res['mean']:.6g}</td><td>{frac['n']}</td></tr>")
    control_rows=[]
    for mode,items in controls.items():
        for variant,data in items.items():
            ci=data['ci95']
            interval='—' if ci is None else f'[{ci[0]:+.4g}, {ci[1]:+.4g}]'
            control_rows.append(f"<tr><td>{mode}</td><td>{variant}</td><td>{data['mean']:+.5g}</td><td>{interval}</td><td>{data['n']}</td></tr>")
    angle_rows=[]
    for angle,data in angles.items():
        ci=data['delta_ci95']
        interval='—' if ci is None else f'[{ci[0]:+.4g}, {ci[1]:+.4g}]'
        angle_rows.append(f"<tr><td>{angle}°</td><td>{data['classic_mean']:.4f}</td><td>{data['conditional_mean']:.4f}</td>"
                          f"<td>{100*data['relative_delta']:+.1f}%</td><td>{interval}</td><td>{data['roi_pixels']}</td></tr>")
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U 槽预试分析</title>
<style>body{font:16px system-ui;color:#233146;background:#fafbfc;margin:32px;line-height:1.8}p{max-width:1100px}table{border-collapse:collapse}td,th{border:1px solid #ccd;padding:9px 16px}a{color:#175ab0}.gallery{display:grid;grid-template-columns:repeat(2,300px);gap:14px}figure{margin:0}figure img{image-rendering:pixelated}</style>
<h1>U 槽预试：已完成结果与限制</h1><p><a href="plan.html">完整方案</a> · <a href="mechanism_plan.html">void-chord 机制探针方案（未执行）</a> · <a href="comparison.html">全部原图、差分、场景与 ROI</a> · <a href="findings.json">分量/数值对照数据</a> · <a href="validation.json">完整性校验</a> · <a href="run_prescreen.ps1">本轮实际复现命令</a></p>
<p>本轮是低分辨率预试，主矩阵为 64×64、每 seed 64 spp、每模式 4 个独立 seeds；平均图合计 256 spp。
256×256 正式队列尚未运行。主视角 45°、ROI 和四组参数在看结果前固定。当前使用镜面微表面模型，不是漫反射墙。
结果回答“本项目最近观测条件化是否改变窄槽成像”；它不能单凭亮度证明精确 void-chord 机制。</p>
<h2>固定右内壁 ROI</h2><table><tr><th>组</th><th>classic_global</th><th>conditional</th><th>相对变化</th><th>线性差值 95% CI</th></tr>'''+''.join(rows)+'''</table>
<p>数值为线性辐亮度。相对变化 = 两组均值之差 / classic 均值；区间针对绝对差值，由独立 seed 间方差计算。
这些探索性区间未做多重比较校正，不包括 SDF 插值、边界条件、有限反射次数等系统误差。</p>
<h2>统一曝光的平均图</h2><p>下图先平均线性 PFM，再统一乘 0.25 并用 filmic 显示，便于看清亮部；所有图使用同一映射。
原始分辨率 64×64，放大仅供观察，无降噪、无 AI 放大。原始 renderer PNG、PFM 和固定 ±0.5 差分在完整对比页。</p>
<div class="gallery">'''+''.join(gallery)+'''</div>
<h2>散射次数分量</h2><table><tr><th>模式</th><th>多次散射 / 总亮度</th><th>total−single−multiple</th><th>seeds</th></tr>'''+''.join(component_rows)+'''</table>
<p>比例是每个 seed 的 ROI 能量比例再平均，不是跨壁光路数量比例。multiple 包括同壁、两壁与槽底事件；
single 仍包含第一次碰撞后的阴影/逃逸。分量和总图共享 seed 编号，因此校验残差按 seed 配对计算。
少量零次散射背景和蒙特卡洛路径差异可造成非零残差；不能把分量图称为精确的跨壁 AOV。</p>
<h2>补充视角比较：共同可见的右壁</h2><table><tr><th>角度</th><th>classic_global</th><th>conditional</th><th>相对变化</th><th>差值 95% CI</th><th>ROI 像素</th></tr>'''+''.join(angle_rows)+'''</table>
<p>补充角度组使用 32×32、64 spp、每模式 3 个 seeds。这里另用深度 0.15w–0.45w、|z|&lt;w 的共同可见区域，
避免 60° 遮挡较深内壁造成区域变化。该区域在本组第一张 conditional 图完成前登记，依据几何而非亮度选择，属于探索性分析。
主参数实验的 45° 和原 ROI 保持不变。不同视角的像素离散化仍有影响；不能把三角度比较称为全局最优搜索。</p>
<h2>数值与布置敏感性</h2><table><tr><th>模式</th><th>改动</th><th>相对本模式 total 的线性变化</th><th>配对 95% CI</th><th>seeds</th></tr>'''+''.join(control_rows)+'''</table>
<p>对照使用 32×32、64 spp，逐项改变。fine=网格减半，expanded=介质 padding 翻倍，no_nee=关闭显式光源采样，
bounces=上限 64，tracking=更小 tracking 区间。区间宽时只能说“当前采样无法分辨”，不能据此宣称等价或收敛。</p>
<p>仍需正式高分辨率重复、足够样本的 NEE/网格等价检查，以及固定起点和方向的自由程探针，才能把效果进一步归因到跨壁 void-chord。
三角度 pilot 保留在完整对比页；单 seed 的角度排名不视为最优视角证据。</p></html>'''
    (out/'findings.html').write_text(page,encoding='utf-8')
    print(audit)
    for c in matrix:
        print(c['group'],c.get('relative_delta'),c.get('delta_ci95'))
    return findings


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=g.OUT)
    run(parser.parse_args().output.resolve())
