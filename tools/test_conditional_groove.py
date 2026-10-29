"""Reproducible constant-width groove experiment, using existing transport models.

Run --stage pilot first. Each stage has an immutable manifest and independent
seed streams across models. Images are averaged in linear space, without denoising.
The renderer and random-field evaluation are not modified by this tool.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time

import numpy as np
from scipy.stats import t as student_t

from compare_shaderball_fixed_ratio import save, load, write_png, write_pfm, filmic, signed_color

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/conditional_ugroove'
W = .1
GROUPS = {
    'A_strong': (.25, 4),
    'B_medium': (.25, 2),
    'C_short': (.25, .25),
    'D_small_sigma': (.025, 4),
}
MODES = ('classic_global', 'global_conditional', 'classic_local')
SEEDS = (202610271, 202610373, 202610477, 202610581)
ERROR = re.compile(r'Caught an internal error|unrecoverable error|Unable to load|'
                   r'Wrong normal direction|Gradient invalid|Conditioning .* error|'
                   r'Clamp one (?:inf|nan) sample|Failed to sample gradient|'
                   r'terminate called|Reached end of ray, but not counting', re.I)


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else hashlib.sha256(stream.read()).hexdigest()


def read_image(path):
    # Zero-valued single-scattering components are valid experimental outcomes.
    with path.open('rb') as stream:
        if stream.readline().strip() != b'PF':
            raise ValueError('Expected RGB PFM')
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        data = np.frombuffer(stream.read(), dtype='<f4' if scale < 0 else '>f4')
    if not math.isfinite(scale) or scale == 0:
        raise ValueError('Invalid PFM scale')
    image = data.reshape(height, width, 3)[::-1].astype(float)*abs(scale)
    if not np.isfinite(image).all() or image.min() < -1e-6:
        raise ValueError('Invalid radiance')
    return image


def fixture(out, fine=False, expanded=False):
    name = 'fine' if fine else 'expanded' if expanded else 'main'
    folder = out / 'fields' / name
    request = dict(output_directory=folder.as_posix(), voxel_size=W/(64 if fine else 32),
                   sigma=.25*W, length_scale=4*W, padding=(3 if expanded else 1.5)*W,
                   cases=[dict(name='u_groove', shape='u_groove', width=W, depth=W,
                               wall_thickness=W, base_thickness=.5*W, length=4*W)])
    request_path = folder / 'request.json'
    if request_path.exists() and load(request_path) != request:
        raise RuntimeError('Existing fixture request differs; choose a new output directory')
    save(request_path, request)
    meta_path = folder / 'u_groove.nvdb.json'
    if not meta_path.exists():
        print(f'Baking {name} fixture', flush=True)
        with (folder/'bake.log').open('wb') as log:
            subprocess.run([str(ROOT/'build/Release/bake_contact_fields.exe'), str(request_path)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    meta = load(meta_path)
    if meta['geometry'] != request['cases'][0] or meta['dx'] != request['voxel_size']:
        raise RuntimeError('Fixture metadata mismatch')
    if not meta['verification']['sdf_formats_identical'] or not meta['verification']['density_verified']:
        raise RuntimeError('Fixture verification failed')
    return folder, meta


def camera(angle, size):
    theta = math.radians(angle)
    target = np.array([.5*W, -.35*W, 0])
    position = target + 8*W*np.array([-math.sin(theta), math.cos(theta), 0])
    return dict(type='pinhole', resolution=[size, size], fov=20, tonemap='filmic',
                reconstruction_filter='box', transform=dict(position=position.tolist(),
                look_at=target.tolist(), up=[0, 0, -1], scale=[-1, 1, 1]))


def roi_mask(cam):
    """Mean-geometry first-hit mask, fixed independently of rendered intensity.

    Right inner wall, depth 0.15w..0.65w; central |z|<w; no floor/rim/endcaps.
    This is a registered image-space ROI, not an identification of random hits.
    """
    size = cam['resolution'][0]
    p = np.array(cam['transform']['position'])
    f = np.array(cam['transform']['look_at'])-p
    f /= np.linalg.norm(f)
    up = np.array(cam['transform']['up'])
    # JsonPtr gramSchmidt supplies x=up cross f, then scale.x=-1.
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    pixel = (np.arange(size)+.5)*2/size-1
    xx, yy = np.meshgrid(pixel, -pixel)
    rays = xx[..., None]*r + yy[..., None]*u + f/math.tan(math.radians(cam['fov']/2))
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    best = np.full((size, size), np.inf)
    points = np.zeros((size, size, 3))
    for lo, hi in [([-1.5*W,-W,-2*W],[-.5*W,0,2*W]),
                   ([.5*W,-W,-2*W],[1.5*W,0,2*W]),
                   ([-1.5*W,-1.5*W,-2*W],[1.5*W,-W,2*W])]:
        with np.errstate(divide='ignore'):
            t0 = (np.array(lo)-p)/rays
            t1 = (np.array(hi)-p)/rays
        near = np.minimum(t0, t1).max(axis=-1)
        far = np.maximum(t0, t1).min(axis=-1)
        valid = (near >= 0) & (near <= far) & (near < best)
        best[valid] = near[valid]
        points[valid] = (p+near[..., None]*rays)[valid]
    return (np.isfinite(best) & (np.abs(points[..., 0]-.5*W)<1e-8)
            & (points[..., 1] >= -.65*W) & (points[..., 1] <= -.15*W)
            & (np.abs(points[..., 2]) < W))


def make_scene(out, group, angle, mode, size, spp, variant='total'):
    field, meta = fixture(out, fine=variant=='fine', expanded=variant=='expanded')
    sigma_w, ell_w = GROUPS[group]
    cov = dict(type='squared_exponential', sigma=sigma_w*W, lengthScale=ell_w*W)
    medium = dict(name='groove', type=mode, max_bounces=64,
                  material=dict(type='mirror', albedo=.85))
    if mode == 'classic_local':
        if sigma_w != .25:
            raise ValueError('classic_local density was baked for sigma/w=0.25 only')
        medium.update(field=dict(file=(field/'u_groove.nvdb').as_posix(),
                      sdf_grid='sdf', density_grid='density', macrocell_size=8), covariance=cov)
    else:
        medium['gaussian_process'] = dict(type='standard', covariance=cov, mean=dict(
            type='tabulated', grid=dict(type='vdb', file=(field/'u_groove_sdf.vdb').as_posix(),
            density_name='sdf', density_scale=1, emission_name='', normalize_size=False,
            request_sdf=False, interpolate='linear', integration_method='raymarching')))
    if mode == 'global_conditional':
        medium.update(tracking_majorant='adaptive', tracking_step_scale=.5,
                      tracking_age_scale=.25, tracking_max_expected_events=64)
        if variant == 'tracking':
            medium.update(tracking_step_scale=.25, tracking_age_scale=.125,
                          tracking_max_expected_events=32)
    bounds = meta['process_box']
    lo, hi = np.array(bounds['min']), np.array(bounds['max'])
    scene = dict(media=[medium], camera=camera(angle, size), integrator=dict(
        type='path_tracer', min_bounces=0, max_bounces=32, enable_light_sampling=True,
        enable_volume_light_sampling=True, enable_consistency_checks=False,
        enable_two_sided_shading=True, use_cuda_gpis=False, independent_path_sampling=True,
        wavefront_gpis=False), primitives=[
            dict(name='processBox', type='cube', int_medium='groove', bsdf=dict(type='forward', albedo=1),
                 transform=dict(position=((lo+hi)/2).tolist(), scale=(hi-lo).tolist())),
            dict(name='right_area', type='quad', emission=8, bsdf=dict(type='null', albedo=1),
                 transform=dict(position=[3.15*W,3*W,0], scale=[3*W,1,4*W], rotation=[180,0,0])),
            dict(type='infinite_sphere', emission=.002, bsdf=dict(type='null', albedo=1))],
        renderer=dict(spp=spp, spp_step=max(1,spp//8), adaptive_sampling=False,
            stratified_sampler=False, enable_resume_render=False, overwrite_output_files=True,
            output_file='render.png', hdr_output_file='render.pfm', timeout='0', checkpoint_interval='0'))
    if variant == 'single':
        scene['integrator'].update(min_bounces=1, max_bounces=2)
    elif variant == 'multiple':
        scene['integrator']['min_bounces'] = 2
    elif variant == 'no_nee':
        scene['integrator'].update(enable_light_sampling=False, enable_volume_light_sampling=False)
    elif variant == 'bounces':
        scene['integrator']['max_bounces'] = 64
    return scene


def jobs_for(stage, size=None, spp=None, seeds=None):
    defaults = {'pilot': (48,32,1), 'matrix': (64,128,4), 'angles': (64,128,4),
                'components': (64,128,4), 'controls': (48,128,3),
                'presentation': (256,256,1), 'confirm': (256,256,8)}
    n, s, repetitions = defaults[stage]
    n, s = size or n, spp or s
    seed_list = seeds or list(SEEDS[:repetitions])
    if stage == 'confirm' and seeds is None:
        seed_list = [202620001+104729*i for i in range(8)]
    if stage in ('matrix', 'confirm'):
        specs = [(g,45,m,'total') for g in GROUPS for m in MODES[:2]]
        specs += [('A_strong',45,'classic_local','total')]
    elif stage in ('pilot', 'angles'):
        specs = [('A_strong',a,m,'total') for a in (30,45,60) for m in MODES[:2]]
    elif stage == 'components':
        specs = [('A_strong',45,m,v) for v in ('single','multiple') for m in MODES[:2]]
    elif stage == 'controls':
        specs = [('A_strong',45,m,v) for v in ('total','fine','expanded','no_nee','bounces','tracking')
                 for m in MODES[:2] if v != 'tracking' or m=='global_conditional']
    else:
        specs = [('A_strong',45,m,'total') for m in MODES]
    return [dict(group=g, angle=a, mode=m, variant=v, size=n, spp=s, seed_id=seed)
            for g,a,m,v in specs for seed in seed_list]


def job_key(job):
    return f"{job['group']}/view{job['angle']}/{job['variant']}/{job['mode']}/seed{job['seed_id']}"


def render(out, stage, job, exe, exe_hash, threads, timeout):
    folder = out / stage / job_key(job)
    scene = make_scene(out, **{k:job[k] for k in ('group','angle','mode','size','spp','variant')})
    scene['renderer']['output_directory'] = folder.as_posix()
    seed = job['seed_id']+10000000*MODES.index(job['mode'])
    expected = dict(job, render_seed=seed, executable_sha256=exe_hash)
    record = folder/'result.json'
    if record.exists():
        old = load(record)
        if any(old.get(k) != v for k,v in expected.items()) or load(folder/'scene.json') != scene:
            raise RuntimeError(f'Existing run differs: {folder}')
        if old['status']=='complete':
            validate_output(folder, job)
            return old
        raise RuntimeError(f'Previous failed attempt retained at {folder}; use a new --stage-label')
    save(folder/'scene.json', scene)
    mask = roi_mask(scene['camera'])
    if mask.sum() < 8:
        raise RuntimeError('ROI too small')
    command = [str(exe),'-t',str(threads),'-s',str(seed),'-d',str(folder),str(folder/'scene.json')]
    print(f"START {stage}/{job_key(job)} {job['size']}px {job['spp']}spp", flush=True)
    start = time.monotonic()
    status, error, returncode = 'complete', None, None
    environment = os.environ.copy()
    for k in ('TUNGSTEN_GPIS_CONDITIONAL_PROFILE','TUNGSTEN_GPIS_PROFILE','TUNGSTEN_GPIS_FS_PROFILE'):
        environment.pop(k, None)
    try:
        with (folder/'render.log').open('wb') as log:
            process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=timeout, env=environment)
        returncode = process.returncode
        if returncode:
            raise RuntimeError(f'Exit code {returncode}')
        image = validate_output(folder, job)
    except (subprocess.TimeoutExpired, ValueError, RuntimeError, OSError) as exc:
        status, error = 'failed', str(exc)
    result = dict(expected, status=status, error=error, returncode=returncode,
                  seconds=time.monotonic()-start, command=command,
                  scene=(folder/'scene.json').relative_to(out).as_posix(),
                  scene_sha256=digest(folder/'scene.json'))
    if status=='complete':
        result.update(roi_pixels=int(mask.sum()), roi_mean=float(image[mask].mean()),
                      image_mean=float(image.mean()), image_maximum=float(image.max()))
        display = filmic(image)
        boundary = mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
        display[boundary] = [1,.15,.1]
        write_png(folder/'roi.png', display)
    save(record, result)
    print(f"{status.upper()} {job_key(job)} {result['seconds']:.1f}s ROI={result.get('roi_mean')}", flush=True)
    return result


def validate_output(folder, job):
    log = (folder/'render.log').read_text(encoding='utf-8', errors='replace')
    match = ERROR.search(log)
    if match or 'Finished render.' not in log or not re.search(rf"Completed\s+{job['spp']}/{job['spp']}\s+spp", log):
        raise RuntimeError(f"Render log failed: {match.group() if match else 'incomplete'}")
    image = read_image(folder/'render.pfm')
    if image.shape != (job['size'],job['size'],3):
        raise ValueError('Unexpected resolution')
    return image


def mean_interval(values):
    values = np.asarray(values, dtype=float)
    mean = float(values.mean())
    if len(values) < 2:
        return dict(mean=mean, n=len(values), ci95=None)
    half = float(student_t.ppf(.975,len(values)-1)*values.std(ddof=1)/math.sqrt(len(values)))
    return dict(mean=mean, n=len(values), ci95=[mean-half,mean+half])


def report(out):
    rows = [load(p) for p in sorted(out.glob('*/**/result.json'))]
    grouped = {}
    for row in rows:
        if row['status'] != 'complete':
            continue
        stage = row['scene'].split('/')[0]
        key = (stage, row['group'],row['angle'],row['variant'],row['size'],row['spp'])
        grouped.setdefault(key,{}).setdefault(row['mode'],[]).append(row)
    comparisons, table = [], []
    for key, modes in grouped.items():
        stage, group, angle, variant, size, spp = key
        derived = out / stage / 'averages' / group / f'view{angle}' / variant
        derived.mkdir(parents=True, exist_ok=True)
        cells, arrays, metrics = [], {}, {}
        for mode in MODES:
            if mode not in modes:
                cells.append('<td>—</td>')
                continue
            images = [read_image((out/r['scene']).parent/'render.pfm') for r in modes[mode]]
            average = np.mean(images,axis=0)
            arrays[mode] = average
            write_pfm(derived/f'{mode}.pfm', average)
            write_png(derived/f'{mode}.png', filmic(average))
            interval = mean_interval([r['roi_mean'] for r in modes[mode]])
            metrics[mode] = interval
            ci = interval['ci95']
            ci_text = '单 seed，无置信区间' if ci is None else f"95% CI [{ci[0]:.5g}, {ci[1]:.5g}]"
            img = (derived/f'{mode}.png').relative_to(out).as_posix()
            raw = (Path(modes[mode][0]['scene']).parent/'roi.png').as_posix()
            cells.append(f'<td><a href="{img}"><img src="{img}" width="240"></a><br>'
                         f"ROI {interval['mean']:.5g}<br>{ci_text}<br>"
                         f"{len(images)} seeds × {spp} spp<br><a href='{raw}'>ROI 位置</a> · "
                         f"<a href='{modes[mode][0]['scene']}'>场景</a></td>")
        comparison = dict(stage=stage,group=group,angle=angle,variant=variant,size=size,spp=spp,roi=metrics)
        if all(m in arrays for m in MODES[:2]):
            a, b = metrics['classic_global'], metrics['global_conditional']
            va = np.array([r['roi_mean'] for r in modes['classic_global']])
            vb = np.array([r['roi_mean'] for r in modes['global_conditional']])
            difference = b['mean']-a['mean']
            half = None
            if min(len(va),len(vb)) >= 2:
                sa, sb = va.var(ddof=1)/len(va), vb.var(ddof=1)/len(vb)
                if sa+sb>0:
                    df = (sa+sb)**2/(sa*sa/(len(va)-1)+sb*sb/(len(vb)-1))
                    half = float(student_t.ppf(.975,df)*math.sqrt(sa+sb))
                else:
                    half = 0.0
            comparison.update(delta=difference, relative_delta=difference/a['mean'] if a['mean']>0 else None,
                              delta_ci95=None if half is None else [difference-half,difference+half],
                              inference='exploratory, independent-seed Welch interval; no multiple-testing adjustment')
            delta = (arrays['global_conditional']-arrays['classic_global']).mean(axis=-1)
            write_png(derived/'difference.png', signed_color(delta, .5))
            difference_path = (derived/'difference.png').relative_to(out).as_posix()
            pct = 'n/a' if comparison['relative_delta'] is None else f"{100*comparison['relative_delta']:+.1f}%"
            uncertainty = '单 seed' if half is None else f'Δ 95% CI [{difference-half:.5g}, {difference+half:.5g}]'
            cells.append(f'<td><a href="{difference_path}"><img src="{difference_path}" width="240"></a>'
                         f'<br>ROI Δ={difference:+.5g} ({pct})<br>{uncertainty}</td>')
        else:
            cells.append('<td>等待对应结果</td>')
        comparisons.append(comparison)
        table.append(f'<tr><th>{stage}<br>{group}<br>{angle}° · {variant}<br>{size}×{size}</th>'+''.join(cells)+'</tr>')
    summary = dict(completed=len([r for r in rows if r['status']=='complete']),
                   failed=[r for r in rows if r['status']!='complete'],comparisons=comparisons)
    save(out/'summary.json',summary)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U 槽条件化实验</title>
<style>body{font:15px system-ui;margin:28px;color:#213040;background:#f6f7fa}p{max-width:1100px;line-height:1.7}
table{border-collapse:collapse}td,th{padding:12px;border:1px solid #cbd2dd;text-align:center}img{image-rendering:pixelated}a{color:#175ab0}</style>
<h1>U 槽：classic 与 global_conditional</h1><p><a href="plan.html">完整实验方案、限制与复现命令</a> · <a href="experiment.json">固定参数</a> · <a href="summary.json">线性统计</a></p>
<p>灰色 mirror albedo=0.85；同一 SDF、相机和光源。classic_global 是主对照；classic_local 是烘焙密度辅助对照。
45° 在出图前固定为主视角。ROI 按确定性均值几何的右内壁投影生成，不按亮度或差异挑选。
各模式使用独立随机 seed；置信区间来自独立渲染的 ROI 均值，不把像素当作独立重复。</p>
<p>PNG 采用相同曝光和 filmic；先平均线性 PFM，再映射显示。没有降噪。差分为 conditional − classic_global，
统一色标：蓝 −0.5、白 0、红 +0.5（线性辐亮度，超出范围饱和）；数值统计不截断。
single = 恰好一次 GP 散射；multiple = 至少两次 GP 散射，包含同壁与槽底事件，不能等同“恰好跨壁”。
pilot 仅检查构图和数值稳定性；所有多重对比均属探索性，不能凭单张噪点或一个区间宣称机制已证明。</p>'''
    page += f"<p>已完成 {summary['completed']} 次渲染；失败 {len(summary['failed'])} 次（失败结果不进入均值）。</p>"
    page += '<table><tr><th>试验</th>'+''.join(f'<th>{m}</th>' for m in MODES)+'<th>差分</th></tr>'+''.join(table)+'</table></html>'
    (out/'comparison.html').write_text(page, encoding='utf-8')
    return summary


def write_plan(out):
    out.mkdir(parents=True,exist_ok=True)
    groups = []
    for name,(s,l) in GROUPS.items():
        u = 1/l
        groups.append(dict(name=name,sigma=s*W,ell=l*W,sigma_w=s,ell_w=l,sigma_ell=s/l,
                           rho_at_w=math.exp(-u*u/2),
                           fixed_value_gradient_variance_fraction=1-(1+u*u)*math.exp(-u*u)))
    plan = dict(width=W,world_axes=dict(width='x',mouth='+y',length='z'),
                geometry=dict(depth=W,wall_thickness=W,base_thickness=.5*W,length=4*W),
                groups=groups,primary_angle=45,pilot_angles=[30,45,60],
                camera_target=[.5*W,-.35*W,0],camera_distance=8*W,fov=20,
                light=dict(position=[3.15*W,3*W,0],size=[3*W,4*W],radiance=8,environment=.002),
                material=dict(type='mirror',albedo=.85),grid_dx=W/32,grid_fine_dx=W/64,
                boundary_padding=1.5*W,expanded_padding=3*W,
                roi=dict(surface='mean right inner wall',depth_w=[.15,.65],abs_z_w_less_than=1),
                primary_contrast='global_conditional minus classic_global, A_strong, 45 degrees, total',
                renderer_evaluation_changed=False,
                independent_mode_seed_offset=10000000,seed_ids=list(SEEDS),
                stages={stage:jobs_for(stage) for stage in ('pilot','matrix','angles','components','controls','presentation','confirm')},
                sources=['https://gaussianprocess.org/gpml/chapters/RW2.pdf',
                         'https://gaussianprocess.org/gpml/chapters/RW4.pdf'])
    path=out/'experiment.json'
    if path.exists() and load(path)!=plan:
        raise RuntimeError('Frozen experiment differs; use a new output directory')
    save(path,plan)
    rows=''.join(f"<tr><td>{g['name']}</td><td>{g['sigma']:.5g}</td><td>{g['ell']:.5g}</td>"
                 f"<td>{g['sigma_w']}</td><td>{g['ell_w']}</td><td>{g['sigma_ell']:.5g}</td>"
                 f"<td>{g['rho_at_w']:.6g}</td></tr>" for g in groups)
    page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>U 槽条件化完整方案</title>
<style>body{font:16px system-ui;max-width:1100px;margin:35px auto;padding:0 20px;color:#233146;line-height:1.8}table{border-collapse:collapse}td,th{padding:9px 16px;border:1px solid #ccd}code,pre{background:#eef1f6}pre{padding:15px;overflow:auto}a{color:#175ab0}</style>
<h1>窄槽 global_conditional：可复现实验方案</h1><p><a href="comparison.html">查看实际结果</a> · <a href="experiment.json">机器可读参数与全部队列</a></p>
<h2>问题与实现边界</h2><p>检验在固定几何和材料下，保留最近一次场值及梯度观测是否改变窄槽的出射辐亮度与多次散射贡献。
主比较为 classic_global 对 global_conditional；classic_local 仅作辅助，因为它插值烘焙密度、归一化均值梯度，不能当作完全相同的未条件化实现。</p>
<p>当前 conditional 在介质入口就采样正场值和梯度，并在 GP 碰撞或几何端点更新观测；它既不是只在首次碰撞后启动，
也不是保存完整光路所有观测的 GP 后验。主光线阶段不能先验断言相同。边界位置也可能影响最近观测模型，必须做扩边对照。
依据：src/core/media/GlobalConditionalMedium.cpp 的 context、sampleDistance、transmittance；不改动这些实现。</p>
<p>现有材质为随机微表面镜面或导体，没有漫反射墙模式。本实验统一 mirror albedo=0.85。强相关组 σ/ℓ=1/16 很平滑，
竖直两壁无法像漫反射那样任意双跳转向，因此采用有底 U 槽，允许“右壁→底→左壁→光源”等路径。
这是对原漫反射方案的明确适配；多次散射的差异仍不能自动归因于两侧壁。</p>
<h2>固定场景</h2><p>项目采用 Y-up：x 为槽宽，z 为槽长，+y 为槽口方向（对应原方案的 +z）。
w=0.1；深度 w；壁厚 w；底厚 0.5w；长度 4w。使用闭合凹多边形的精确节点 SDF，再作原有三线性插值，
左右墙和槽底属于同一个场。只在数据烘焙器增加 U 槽形状，随机场求值与传输公式不变。</p>
<p>相机看向 (0.5w,−0.35w,0)，距离 8w，位置 = 目标 + 8w(−sin θ, cos θ,0)，θ=30°/45°/60°，FOV=20°。
45° 在看图前固定为主视角，其余是敏感性检查。面积灯在 (3.15w,3w,0)，水平向下，尺寸 3w×4w，辐亮度 8；
环境 0.002。灯和相机位于相反侧。因为几何和镜面方向限制，这不保证主 ROI 的照明一定由跨壁路径主导，必须用分量检查。</p>
<svg viewBox="0 0 760 330" width="760" role="img" aria-label="槽剖面与相机光源位置"><rect width="760" height="330" fill="#f4f6fa"/>
<path d="M270 170V295H480V170H410V250H340V170Z" fill="#b1bfd0" stroke="#344960" stroke-width="2"/>
<path d="M130 65L410 195L364 250L340 225L550 45" fill="none" stroke="#d68a25" stroke-width="2" stroke-dasharray="6 5"/>
<rect x="105" y="50" width="50" height="30" rx="5" fill="#3972ae"/><rect x="505" y="35" width="120" height="12" fill="#e3ac37"/>
<path d="M340 155H410" stroke="#344960"/><text x="370" y="145">w</text><text x="90" y="40">Camera</text><text x="530" y="25">Area light</text>
<text x="430" y="215">ROI</text><text x="245" y="320">x →   +y ↑   z: along groove</text></svg>
<h2>四组参数</h2><table><tr><th>组</th><th>σ</th><th>ℓ</th><th>σ/w</th><th>ℓ/w</th><th>σ/ℓ</th><th>ρ(w)</th></tr>'''+rows+'''</table>
<p>A 为主验证组；B 减弱跨壁相关；C 为短相关长度对照；D 为小位移对照。σ/ℓ 控制随机梯度统计，σ/w 和 ℓ/w
分别控制位移相对尺度与跨壁相关。只有两个独立比值。C 的法线同时变粗，组间差异不能单独解释为相关性变化；
每组应比较 conditional−classic 的差值，再讨论组间趋势，且仍保留混杂限制。</p>
<h2>理论预期与可证伪性</h2><p>SE 核 K(r)=σ²exp[−r²/(2ℓ²)]，Cov(∇g)=σ²/ℓ² I。
跨壁方向 ω 的直线距离 L=w/|ωx|，相关系数 ρ(L)=exp[−L²/(2ℓ²)]；实际 ω 是散射后方向，不能用相机角度直接代替。
按标准高斯条件化公式推导，在给定起点完整场值和梯度的情况下，目标点的条件方差为
σ²{1−[1+(L/ℓ)²] exp[−(L/ℓ)²]}。A 组 L=w 时约为原方差的 0.001874。
这只是固定观测的场值方差，不是槽宽、飞行长度或最终像素的方差，也不确定亮暗符号。
参考 <a href="https://gaussianprocess.org/gpml/chapters/RW2.pdf">GPML 第 2 章：条件高斯</a>、
<a href="https://gaussianprocess.org/gpml/chapters/RW4.pdf">第 4 章：SE 核与导数协方差</a>；上述方差式为本实验推导。</p>
<p>void chord 对应从零交叉出发的正场值连通区间；相机从外部进入的首段属于另一种起点分布。
此处只保留最近观测的实现并未条件化于整条路径的持续存活，所以渲染差异不能称为“精确 void-chord 分布已验证”。
若需证明机制，应另外固定碰撞状态与出射方向，采样下一次碰撞距离/逃逸概率，并与保存完整场的 GPIS 参考比较。
当前图像实验提供效果和稳定性证据，不替代该机制实验。</p>
<h2>执行阶段</h2><ol>
<li>pilot：A 组，三角度×两模式，48²、32 spp、1 seed。只检查构图、ROI、光照及数值稳定性，不用于显著性结论。</li>
<li>matrix：固定 45°，四组×主两模式，另加 A 的 classic_local；64²、128 spp、4 独立 seeds。主统计量为右内壁 ROI 的线性辐亮度差。</li>
<li>components：A/45°，一次散射与至少两次散射分别渲染，各两模式；64²、128 spp、4 seeds。
利用现有 min_bounces/max_bounces：single=(1,2)、multiple=(2,32)，总图=(0,32)。前向边界不计反射次数；本场景无普通反射物体。
single 包含第一次碰撞后的阴影/逃逸；它不是“仅首段相机射线”。检查 total≈single+multiple（ROI 内零次直接发光应可忽略）。</li>
<li>angles：用同样固定几何 ROI 比较 30°/45°/60°；64²、128 spp、4 seeds，全部保留，不只发布差异最大角度。</li>
<li>controls：48²、128 spp、3 seeds；同一 A/45° 分别检验 dx 从 w/32 减半、边界 padding 从 1.5w 扩为 3w、
关闭 NEE、反射上限从 32 增至 64、tracking 区间约束减半。逐项改变，不能混合更改后宣称单因素验证。</li>
<li>presentation：A/45° 三模式，256²、256 spp、1 seed，供视觉查看，不替代重复试验。</li>
<li>confirm：若预试与数值对照支持继续，四组及 A 的 classic_local，256²、256 spp、8 个新 seeds；
预先固定主比较 A/45°/total；其余比较报告为探索性或使用多重比较校正。此队列只在实际执行后才标记完成。</li></ol>
<h2>统计、判定和噪声控制</h2><p>ROI 是均值几何右内壁上深度 0.15w–0.65w、|z|&lt;w 的可见投影，在所有配对模式中相同。
不使用图像阈值选亮点，不把 stochastic hit 误称为确定墙面。像素映射随视角变化，遮挡后可见部分可能不同，需报告 ROI 像素数。
本批不降噪、不单图自动曝光，PFM 线性平均后统一 filmic。固定差分色标 ±0.5，不用不同色标放大弱差异。</p>
<p>各模式 seed 加不同偏移，作为独立蒙特卡洛重复；同一 seed 编号不代表相同光路。
对每个 seed 先计算 ROI 均值，再用 seed 间方差计算 Welch 95% 差值区间。预试的区间不校正多重比较，只作探索。
正式主比较需至少 8 seeds；建议判定为区间不跨零，且差异相对 baseline 超过事先约定的实际关注阈值（本方案 5%）。
若区间宽度不足以判断，结论为采样不足；不得称为“无区别”。像素 RMSE 不作为纯模型误差，因为包含渲染噪声。</p>
<p>置信区间仅反映蒙特卡洛不确定性，不含网格、截断、入口观测和模型偏差。要声称跨壁机制，还需多次散射差异、
网格/NEE/扩边对照，以及后续定向飞行探针。参数对照不能保证趋近为零，尤其短 ℓ 组仍有起点附近条件化。</p>
<h2>复现</h2><pre>cmake --build build --config Release --target bake_contact_fields --parallel 4
python tools/test_conditional_groove.py --stage pilot
python tools/test_conditional_groove.py --stage matrix
python tools/test_conditional_groove.py --stage components
python tools/test_conditional_groove.py --stage angles
python tools/test_conditional_groove.py --stage controls
python tools/test_conditional_groove.py --stage presentation
python tools/test_conditional_groove.py --stage confirm
python tools/test_conditional_groove.py --report-only</pre>
<p>每次渲染保存 scene、完整日志、命令、退出状态、耗时、实际 seed、可执行文件 SHA256；阶段 manifest 冻结参数。
检查完整 spp 与 Finished render、错误日志、PFM 尺寸和有限非负值；失败尝试不加入均值。可用 --size/--spp/--seeds 调整规模，
同时指定新的 --stage-label 防止覆盖历史。summary.json 与对比页反映实际完成状态，而非上述计划队列数量。</p></html>'''
    (out/'plan.html').write_text(page,encoding='utf-8')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--stage',choices=('pilot','matrix','angles','components','controls','presentation','confirm'),default='pilot')
    p.add_argument('--stage-label')
    p.add_argument('--size',type=int);p.add_argument('--spp',type=int)
    p.add_argument('--seeds',nargs='+',type=int)
    p.add_argument('--groups',nargs='+',choices=tuple(GROUPS))
    p.add_argument('--modes',nargs='+',choices=MODES)
    p.add_argument('--variants',nargs='+',choices=('total','single','multiple','fine','expanded','no_nee','bounces','tracking'))
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--timeout',type=float,default=1800)
    p.add_argument('--executable',type=Path,default=ROOT/'build/Release/tungsten-endpoint-sync.exe')
    p.add_argument('--report-only',action='store_true')
    p.add_argument('--prepare-only',action='store_true')
    args=p.parse_args()
    for value in (args.size,args.spp,args.threads,args.timeout):
        if value is not None and value<=0:
            p.error('size/spp/threads/timeout must be positive')
    if args.stage_label and (Path(args.stage_label).name!=args.stage_label or args.stage_label in ('.','..')):
        p.error('stage-label must be a directory name')
    out=args.output.resolve()
    write_plan(out)
    if args.report_only:
        report(out);return
    jobs=jobs_for(args.stage,args.size,args.spp,args.seeds)
    for key,allowed in [('group',args.groups),('mode',args.modes),('variant',args.variants)]:
        if allowed:
            jobs=[j for j in jobs if j[key] in allowed]
    stage=args.stage_label or args.stage
    exe=args.executable.resolve()
    exe_hash=digest(exe)
    assets={}
    for variant in {j['variant'] for j in jobs}:
        field,meta=fixture(out,variant=='fine',variant=='expanded')
        assets[field.name]={name:digest(field/name) for name in ('u_groove_sdf.vdb','u_groove.nvdb')}
    manifest=dict(stage=stage,jobs=jobs,executable=exe.as_posix(),executable_sha256=exe_hash,
                  assets=assets,threads=args.threads,timeout=args.timeout,
                  tool_sha256=digest(Path(__file__)),baker_sha256=digest(ROOT/'build/Release/bake_contact_fields.exe'))
    manifest_path=out/stage/'manifest.json'
    if manifest_path.exists() and load(manifest_path)!=manifest:
        raise RuntimeError('Stage manifest differs; use a new --stage-label')
    save(manifest_path,manifest)
    if args.prepare_only:
        for job in jobs:
            scene=make_scene(out,**{k:job[k] for k in ('group','angle','mode','size','spp','variant')})
            save(out/stage/'prepared'/job_key(job)/'scene.json',scene)
        report(out);return
    failed=0
    for i,job in enumerate(jobs):
        save(out/stage/'status.json',dict(state='running',index=i,total=len(jobs),job=job))
        row=render(out,stage,job,exe,exe_hash,args.threads,args.timeout)
        failed += row['status']!='complete'
        report(out)
    save(out/stage/'status.json',dict(state='complete' if not failed else 'finished_with_failures',
                                   total=len(jobs),failed=failed))
    if failed:
        raise SystemExit(1)


if __name__=='__main__':
    main()
