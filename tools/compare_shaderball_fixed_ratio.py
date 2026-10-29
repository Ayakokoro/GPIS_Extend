"""Matched shaderball scale sweeps at fixed sigma / lengthScale.

GPIS uses one 3D sparse-convolution realization per complete camera path.
Requires NumPy and SciPy; renders and all derived artifacts stay under --output.
"""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import struct
import time
import zlib

import numpy as np
from scipy.stats import t as student_t

ROOT = Path(__file__).resolve().parents[1]
MODES = ('classic_global', 'gpis_global')
ERRORS = re.compile(r'Caught an internal error|unrecoverable error|Unable to load|'
                    r'Wrong normal direction|Gradient invalid|Conditioning .* error|'
                    r'Clamp one (?:inf|nan) sample|Failed to sample gradient', re.I)


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def render_rows(output):
    """Merge completed queue and optional ahead-of-queue work without duplicates."""
    completed = load(output/'render_rows.json') if (output/'render_rows.json').exists() else []
    if (output/'prefetch_rows.json').exists():
        cached = load(output/'prefetch_rows.json')
        if cached['manifest'] == load(output/'manifest.json'):
            completed += cached['rows']
    rows = {(row['ell'], row['mode'], row['seed_id']): row for row in completed}
    return [rows[key] for key in sorted(rows)]


def read_pfm(path):
    with path.open('rb') as f:
        assert f.readline().strip() == b'PF'
        width, height = map(int, f.readline().split())
        scale = float(f.readline())
        rgb = np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4')
    rgb = rgb.reshape(height, width, 3)[::-1].astype(np.float64) * abs(scale)
    if not np.isfinite(rgb).all() or rgb.min() < -1e-6 or rgb.max() <= 0:
        raise ValueError(f'Invalid image: {path}')
    return rgb


def tag(value):
    return f'{value:.9g}'.replace('.', 'p')


def make_scene(base, mode, ell, args):
    scene = copy.deepcopy(base)
    source = scene['media'][0]
    gp = copy.deepcopy(source['gaussian_process'])
    gp['covariance'].update(sigma=args.ratio * ell, lengthScale=ell)
    if mode == 'classic_global':
        source['gaussian_process'] = gp
    else:
        scene['media'] = [dict(
            name=source['name'], type='sparse_conv_noise', max_bounces=source['max_bounces'],
            gaussian_process=gp, phase_function={'type': 'brdf', 'bsdf': source['material']},
            sigma_a=0, sigma_s=1, density=1, step_size=ell * args.step_fraction,
            min_step=8, seed=17429, impulse_density=args.impulse_density,
            single_realization=False, correlation_context='global',
            intersect_method='gp_discrete', normal_method='conditioned_gaussian',
            **{'1D_sampling': False, 'isotropic_3D_sampling': False,
               '1D_sampling_scheme': 'uni', '1D_gradient_correlationXY': False})]
    scene['camera']['resolution'] = [args.size, args.size]
    scene['integrator'].update(
        use_cuda_gpis=False, independent_path_sampling=True, wavefront_gpis=False,
        enable_light_sampling=not args.no_nee, enable_volume_light_sampling=not args.no_nee)
    scene['renderer'] = dict(
        spp=args.spp, spp_step=max(1, args.spp // 16), adaptive_sampling=False,
        stratified_sampler=False, enable_resume_render=False, overwrite_output_files=True,
        scene_bvh=True, output_file='render.png', hdr_output_file='render.pfm',
        checkpoint_interval='0', timeout='0')
    return scene


def verified(folder, scene, seed, executable_hash):
    try:
        record = load(folder / 'result.json')
        if (load(folder / 'scene.json') != scene or record['render_seed'] != seed
                or record['executable_sha256'] != executable_hash):
            return False
        log = (folder / 'render.log').read_text(encoding='utf-8', errors='replace')
        if (ERRORS.search(log) or 'Finished render.' not in log
                or not re.search(rf"Completed\s+{scene['renderer']['spp']}/{scene['renderer']['spp']}\s+spp", log)):
            return False
        return read_pfm(folder / 'render.pfm').shape[:2] == tuple(scene['camera']['resolution'][::-1])
    except (OSError, ValueError, KeyError, AssertionError):
        return False


def write_pfm(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w, _ = rgb.shape
    path.write_bytes(f'PF\n{w} {h}\n-1.0\n'.encode() + rgb[::-1].astype('<f4').tobytes())


def write_png(path, rgb):
    """Small RGB PNG writer; no image-processing dependency is required."""
    pixels = np.rint(np.clip(rgb, 0, 1)*255).astype(np.uint8)
    h, w, _ = pixels.shape
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind+data))
    scanlines = b''.join(b'\0' + row.tobytes() for row in pixels)
    data = (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', w, h, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(scanlines)) + chunk(b'IEND', b''))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def filmic(rgb):
    # Same mapping as src/core/cameras/Tonemap.hpp; applied AFTER linear averaging.
    x = np.maximum(rgb - .004, 0)
    return x*(6.2*x+.5)/(x*(6.2*x+1.7)+.06)


def block_mean(a, size=2):
    h, w = a.shape[-3:-1]
    if h % size or w % size:
        raise ValueError('Resolution must be divisible by the display block size.')
    shape = (*a.shape[:-3], h//size, size, w//size, size, a.shape[-1])
    return a.reshape(shape).mean(axis=(-4, -2))


def signed_color(values, limit=.15):
    """Fixed linear-radiance scale: blue negative, white zero, orange positive."""
    v = np.clip(values/limit, -1, 1)[..., None]
    blue, orange = np.array([.12, .35, .8]), np.array([.9, .25, .08])
    return np.where(v < 0, 1+v*(1-blue), 1-v*(1-orange))


def interval(samples):
    n = len(samples)
    if n < 2:
        return None
    se = float(np.std(samples, ddof=1)/np.sqrt(n))
    return dict(standard_error=se, half_width_95=float(student_t.ppf(.975, n-1)*se))


def roi_stats(classic, gpis, roi):
    h, w = classic.shape[1:3]
    x0, y0, x1, y1 = [int(round(v*s)) for v, s in zip(roi, [w, h, w, h])]
    c = classic[:, y0:y1, x0:x1] @ np.array([.2126, .7152, .0722])
    g = gpis[:, y0:y1, x0:x1] @ np.array([.2126, .7152, .0722])
    cr, gr = c.mean(axis=(1, 2)), g.mean(axis=(1, 2))
    delta = float(gr.mean()-cr.mean())
    result = dict(pixel_bounds=[x0, y0, x1, y1], classic_mean=float(cr.mean()),
                  gpis_mean=float(gr.mean()), relative_difference=float(gr.mean()/cr.mean()-1),
                  difference=delta, classic_replicate_means=cr.tolist(), gpis_replicate_means=gr.tolist(),
                  classic_mean_uncertainty=interval(cr), gpis_mean_uncertainty=interval(gr))
    cm, gm = c.mean(axis=0), g.mean(axis=0)
    denominator = float(np.mean(cm**2))
    mse = float(np.mean((gm-cm)**2))
    result['raw_relative_rmse'] = math.sqrt(mse/denominator)
    if len(c) >= 2 and len(g) >= 2:
        vc, vg = float(cr.var(ddof=1)/len(c)), float(gr.var(ddof=1)/len(g))
        variance = vc+vg
        divisor = vc*vc/(len(c)-1)+vg*vg/(len(g)-1)
        df = variance*variance/divisor if divisor > 0 else float(len(c)+len(g)-2)
        half = float(student_t.ppf(.975, df)*math.sqrt(variance))
        result.update(difference_ci95=[delta-half, delta+half], difference_standard_error=math.sqrt(variance),
                      welch_degrees_of_freedom=df)
        noise_mse = float(np.mean(c.var(axis=0, ddof=1)/len(c) + g.var(axis=0, ddof=1)/len(g)))
        result.update(noise_relative_rms=math.sqrt(noise_mse/denominator),
                      noise_subtracted_relative_rmse=math.sqrt(max(0, mse-noise_mse)/denominator))
    return result


def summarize_checks(output):
    """Paired-seed sensitivity checks; these are separate from the main sweep."""
    experiment = load(output/'manifest.json')
    check_root = output.parent/'checks'
    base_path = check_root/'step_baseline'
    baseline = load(base_path/'manifest.json')
    base_rows = load(base_path/'render_rows.json')
    summary = dict(resolution=baseline['resolution'], spp_per_seed=baseline['spp_per_seed'],
                   seed_ids=baseline['seed_ids'], cases=[],
                   scope='Paired-seed sensitivity tests at a separate resolution; not a proof of exact-GP convergence.')
    for name in ['half_step', 'density30']:
        path = check_root/name
        config = load(path/'manifest.json')
        rows = load(path/'render_rows.json')
        for key in ['resolution', 'spp_per_seed', 'seed_ids', 'ratio', 'lengths']:
            if config[key] != baseline[key]:
                raise ValueError(f'Unmatched check {key}: {name}')
        if config['ratio'] != experiment['ratio']:
            raise ValueError('Check roughness differs from main sweep.')
        for ell in baseline['lengths']:
            if ell not in experiment['lengths']:
                raise ValueError('Check length scale is absent from main sweep.')
            b = sorted([r for r in base_rows if r['ell']==ell], key=lambda r:r['seed_id'])
            v = sorted([r for r in rows if r['ell']==ell], key=lambda r:r['seed_id'])
            if len(b)!=len(baseline['seed_ids']) or len(v)!=len(b):
                raise ValueError(f'Check is incomplete: {name}')
            for br, vr in zip(b, v):
                assert br['seed_id']==vr['seed_id'] and br['render_seed']==vr['render_seed']
                bs, vs = load(Path(br['folder'])/'scene.json'), load(Path(vr['folder'])/'scene.json')
                for scene in [bs, vs]:
                    for key in ['step_size', 'impulse_density']:
                        scene['media'][0].pop(key)
                assert bs==vs, 'Unmatched check scene'
            bs = np.stack([read_pfm(Path(r['folder'])/'render.pfm') for r in b])
            vs = np.stack([read_pfm(Path(r['folder'])/'render.pfm') for r in v])
            case = dict(name=name, ell=ell, sigma=ell*baseline['ratio'],
                        baseline_step_fraction=baseline['step_fraction'], step_fraction=config['step_fraction'],
                        baseline_impulse_density=baseline['impulse_density'], impulse_density=config['impulse_density'], rois={})
            for roi_name, roi in experiment['rois'].items():
                s = roi_stats(bs, vs, roi)
                differences = np.asarray(s['gpis_replicate_means']) - np.asarray(s['classic_replicate_means'])
                uncertainty = interval(differences)
                difference = float(differences.mean())
                half = uncertainty['half_width_95']
                case['rois'][roi_name] = dict(
                    baseline_mean=s['classic_mean'], variant_mean=s['gpis_mean'], relative_difference=s['relative_difference'],
                    paired_differences=differences.tolist(), difference=difference,
                    difference_ci95=[difference-half, difference+half], **uncertainty)
            summary['cases'].append(case)
    save(output/'parameter_checks.json', summary)
    return summary


def build_report(output):
    manifest = load(output/'manifest.json')
    rows = render_rows(output)
    result = dict(manifest=manifest, completed_renders=len(rows), cases=[],
                  statistics='Linear Rec.709 luminance. Disjoint seeds between models. Welch t intervals over independent replicate ROI means.',
                  display='Average linear radiance before filmic mapping; block images average 2x2 pixels in linear radiance. No denoising.',
                  difference_scale=.15)
    cards, table, csv_rows = [], [], []
    for ell in manifest['lengths']:
        case = dict(ell=ell, sigma=ell*manifest['ratio'], modes={}, rois={})
        stacks = {}
        for mode in manifest['modes']:
            selected = [r for r in rows if r['ell']==ell and r['mode']==mode]
            if not selected:
                continue
            arrays = [read_pfm(Path(r['folder'])/'render.pfm') for r in selected]
            stacks[mode] = np.stack(arrays)
            mean = stacks[mode].mean(axis=0)
            relative = Path('averages')/f'ell{tag(ell)}'/mode
            write_pfm(output/relative.with_suffix('.pfm'), mean)
            write_png(output/relative.with_suffix('.png'), filmic(mean))
            write_png(output/relative.with_name(mode+'_block2.png'), filmic(block_mean(mean)))
            case['modes'][mode] = dict(seeds=[r['seed_id'] for r in selected],
                                      spp_total=sum(r['spp'] for r in selected),
                                      seconds=sum(r['seconds'] for r in selected),
                                      image=relative.with_suffix('.png').as_posix(),
                                      block_image=relative.with_name(mode+'_block2.png').as_posix(),
                                      pfm=relative.with_suffix('.pfm').as_posix())
        images = []
        for mode in manifest['modes']:
            if mode in case['modes']:
                entry = case['modes'][mode]
                images.append(f'<figure><figcaption>{mode} · {entry["spp_total"]} spp</figcaption>'
                              f'<a href="{entry["image"]}"><img class="switchable" src="{entry["image"]}" '
                              f'data-raw="{entry["image"]}" data-block="{entry["block_image"]}"></a>'
                              f'<a href="{entry["pfm"]}">线性 PFM</a></figure>')
            else:
                images.append(f'<figure><figcaption>{mode}</figcaption><p>等待渲染</p></figure>')
        if len(stacks) == 2:
            c, g = stacks['classic_global'], stacks['gpis_global']
            for name, roi in {'full': [0, 0, 1, 1], **manifest['rois']}.items():
                case['rois'][name] = roi_stats(c, g, roi)
            delta = (g.mean(axis=0)-c.mean(axis=0)) @ np.array([.2126, .7152, .0722])
            relative = Path('averages')/f'ell{tag(ell)}'
            for suffix, data in [('difference', delta), ('difference_block2', block_mean(delta[..., None])[..., 0])]:
                write_png(output/relative/(suffix+'.png'), signed_color(data))
            images.append(f'<figure><figcaption>GPIS − classic · 线性亮度 ±0.15</figcaption><img class="switchable" '
                          f'src="{relative.as_posix()}/difference.png" data-raw="{relative.as_posix()}/difference.png" '
                          f'data-block="{relative.as_posix()}/difference_block2.png"></figure>')
            if len(c)>=2 and len(g)>=2:
                # Equal-size halves: half their difference has the noise variance of the full mean.
                n = 2*(min(len(c), len(g))//2)
                half = n//2
                noise = .5*((g[:half].mean(axis=0)-g[half:n].mean(axis=0))
                            -(c[:half].mean(axis=0)-c[half:n].mean(axis=0)))
                noise = noise @ np.array([.2126, .7152, .0722])
                write_png(output/relative/'repeat_noise.png', signed_color(noise))
                write_png(output/relative/'repeat_noise_block2.png', signed_color(block_mean(noise[..., None])[..., 0]))
                images.append(f'<figure><figcaption>重复渲染噪声 · 同一色标</figcaption><img class="switchable" '
                              f'src="{relative.as_posix()}/repeat_noise.png" data-raw="{relative.as_posix()}/repeat_noise.png" '
                              f'data-block="{relative.as_posix()}/repeat_noise_block2.png"></figure>')
            for name in ['body', 'cavity', 'base']:
                s = case['rois'][name]
                ci = s.get('difference_ci95')
                missing_ci = '单个 seed；未估计' if len(manifest['seed_ids']) == 1 else '需要至少 2 个 seed'
                confidence = f'[{ci[0]:+.5f}, {ci[1]:+.5f}]' if ci else missing_ci
                table.append(f'<tr><td>{ell:g}</td><td>{name}</td><td>{s["classic_mean"]:.5f}</td>'
                             f'<td>{s["gpis_mean"]:.5f}</td><td>{s["relative_difference"]:+.2%}</td>'
                             f'<td>{confidence}</td></tr>')
                csv_rows.append([ell, case['sigma'], name, s['classic_mean'], s['gpis_mean'],
                                 s['relative_difference'], s['difference'], *(ci or ['', ''])])
        cards.append(f'<section><h2>ℓ = {ell:g} · σ = {ell*manifest["ratio"]:g}</h2><div class="images">'+''.join(images)+'</div></section>')
        result['cases'].append(case)
    expected = len(manifest['lengths'])*len(manifest['modes'])*len(manifest['seed_ids'])
    with (output/'roi_statistics.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['ell', 'sigma', 'roi', 'classic_mean', 'gpis_mean', 'relative_difference',
                         'difference', 'difference_ci95_low', 'difference_ci95_high'])
        writer.writerows(csv_rows)
    if (output/'neural_ell0p1/summary.json').exists():
        result['neural_comparison'] = load(output/'neural_ell0p1/summary.json')
    if (output/'global_conditional/summary.json').exists():
        conditional = load(output/'global_conditional/summary.json')
        assert conditional['manifest']['reference_manifest'] == manifest
        result['global_conditional_comparison'] = conditional
    if (output/'classic_local/summary.json').exists():
        local = load(output/'classic_local/summary.json')
        assert local['manifest']['reference_manifest'] == manifest
        result['classic_local_comparison'] = local
    total_completed, total_expected = len(rows), expected
    if 'neural_comparison' in result:
        neural = result['neural_comparison']
        total_completed += neural['completed_renders']
        total_expected += len(neural['manifest']['seed_ids'])
    if 'global_conditional_comparison' in result:
        total_completed += conditional['completed_renders']
        total_expected += conditional['expected_renders']
    if 'classic_local_comparison' in result:
        total_completed += local['completed_renders']
        total_expected += local['expected_renders']
    result.update(total_completed_renders=total_completed, total_expected_renders=total_expected)
    save(output/'summary.json', result)
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>Shaderball · fixed sigma/ell</title>
<style>body{max-width:1500px;margin:28px auto;padding:0 20px;font:15px/1.6 system-ui;color:#233145;background:#f4f6f9}
h1{font-size:28px}h2{font-size:20px}section{background:white;padding:18px;margin:20px 0;border-radius:8px}
.images{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:16px}figure{margin:0}figcaption{min-height:30px;font-size:13px}
img{width:100%;image-rendering:auto}table{border-collapse:collapse;width:100%}th,td{border-bottom:1px solid #d9e0e8;padding:8px;text-align:right}
th:first-child,td:first-child{text-align:left}a{color:#225ca0}code{background:#e8edf3;padding:2px 5px}.controls{position:sticky;top:0;background:#e7edf5;padding:12px;z-index:1}
.muted{color:#56677b}@media(max-width:850px){.images{grid-template-columns:repeat(2,minmax(0,1fr))}}</style>
<h1>Shaderball：固定 σ / ℓ 的模式对照</h1>'''
    alternatives = []
    for name in ['main', 'res256_spp256']:
        other = output.parent/name
        if other != output and (other/'manifest.json').exists() and (other/'comparison.html').exists():
            settings = load(other/'manifest.json')
            width, height = settings['resolution']
            total = settings['spp_per_seed']*len(settings['seed_ids'])
            alternatives.append(f'<a href="../{name}/comparison.html">{width}×{height} · {total} spp 版本</a>')
    if alternatives:
        page += '<p>其他分辨率：'+' · '.join(alternatives)+'</p>'
    page += (f'<p>σ / ℓ = <b>{manifest["ratio"]:g}</b>，α = {manifest["alpha"]:.4f}。'
             f'{manifest["resolution"][0]} × {manifest["resolution"][1]}；每个 seed {manifest["spp_per_seed"]} spp；'
             f'计划每模式每尺度 {len(manifest["seed_ids"])} 个 seed。含追加模式共完成 <b>{total_completed} / {total_expected}</b> 份渲染。</p>'
             '<p>classic / GPIS 沿用项目 shaderball 的 procedural knob、相机、灯光、绿色镜面材质及 20 次反射上限。'
             'GPIS 使用当前渲染器的三维 world-space global 模式，同一路径保留同一份随机场。'
             '所有图像先在线性空间合并，再使用相同 filmic 映射；未做降噪。</p>'
             f'<p class="muted">impulse density = {manifest["impulse_density"]:g}；步长 = {manifest["step_fraction"]:g} × ℓ。'
             '这是有限密度 sparse-convolution 参考。固定 σ / ℓ 保持梯度协方差，也会改变表面扰动厚度。'
             '<a href="manifest.json">配置清单</a> · <a href="summary.json">完整统计</a> · <a href="roi_statistics.csv">区域统计 CSV</a> · <a href="render_rows.json">逐次渲染记录</a></p>'
             '<div class="controls"><label><input id="block" type="checkbox"> 以 2×2 像素均值显示（降低空间分辨率，在线性空间平均）</label>'
             '　差值色标：<span style="color:#2059cc">蓝 = 差值为负</span>，白 = 0，<span style="color:#e64014">橙 = 差值为正</span>。</div>')
    if (output/'classic_local/section.html').exists():
        page += (output/'classic_local/section.html').read_text(encoding='utf-8')
    elif (output/'global_conditional/section.html').exists():
        page += (output/'global_conditional/section.html').read_text(encoding='utf-8')
    else:
        if (output/'neural_ell0p1/section.html').exists():
            page += (output/'neural_ell0p1/section.html').read_text(encoding='utf-8')
        page += ''.join(cards)
    page += ('<section><h2>区域亮度与重复渲染不确定性</h2><p>百分比为 (GPIS / classic − 1)。'
             '95% 区间针对 GPIS − classic 的绝对线性亮度差，使用独立 seed 的区域均值和 Welch t 区间。'
             '区域在正式渲染前固定：body 为球体中部，cavity 为中央凹面，base 为底座。'
             '完整像素边界、每个 seed 的均值与噪声 RMS 见 summary.json。区间仅反映采样不确定性，'
             '不涵盖有限脉冲密度、射线步长与反射次数上限的系统误差。</p>'
             '<table><thead><tr><th>ℓ</th><th>区域</th><th>classic</th><th>GPIS</th><th>相对差</th><th>差值 95% 区间</th></tr></thead>'
             '<tbody>'+''.join(table)+'</tbody></table></section>')
    if (output/'parameter_checks.json').exists():
        checks = load(output/'parameter_checks.json')
        entries = []
        for case in checks['cases']:
            for name, s in case['rois'].items():
                lo, hi = s['difference_ci95']
                entries.append(f'<tr><td>{case["name"]}</td><td>{case["ell"]:g}</td><td>{name}</td>'
                               f'<td>{s["relative_difference"]:+.2%}</td><td>[{lo:+.5f}, {hi:+.5f}]</td></tr>')
        page += (f'<section><h2>参考参数敏感性检查</h2><p>单独使用 {checks["resolution"][0]}×{checks["resolution"][1]}，'
                 f'{len(checks["seed_ids"])} 个 seed、每 seed {checks["spp_per_seed"]} spp。基线为步长 ℓ/4、脉冲密度 10；'
                 'half_step 将步长减半，density30 将脉冲密度提高至 30。区间基于相同 seed 的配对差值，'
                 '这项检查不构成精确 GP 收敛证明。<a href="parameter_checks.json">完整检查数据</a></p>'
                 '<table><tr><th>检查</th><th>ℓ</th><th>区域</th><th>相对基线</th><th>差值 95% 区间</th></tr>'
                 +''.join(entries)+'</table></section>')
    page += '<script>document.getElementById("block").addEventListener("change", e => {document.querySelectorAll(".switchable").forEach(img => img.src=e.target.checked?img.dataset.block:img.dataset.raw)});</script></html>'
    (output/'comparison.html').write_text(page, encoding='utf-8')
    print(f'Report: {output / "comparison.html"} ({total_completed}/{total_expected} renders)', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'outputs/shaderball_fixed_ratio')
    parser.add_argument('--scene', type=Path, default=ROOT / 'gpis-scenes/shaderball/shaderball-stationary-27-classic-global.json')
    parser.add_argument('--executable', type=Path, default=ROOT / 'build/Release/tungsten.exe')
    parser.add_argument('--lengths', type=float, nargs='+', default=[.02, .05, .1])
    parser.add_argument('--ratio', type=float, default=1)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--spp', type=int, default=256)
    parser.add_argument('--seeds', type=int, nargs='+', default=[20261026, 20261027, 20261028, 20261029])
    parser.add_argument('--threads', type=int, default=min(28, os.cpu_count() or 1))
    parser.add_argument('--modes', nargs='+', choices=MODES, default=list(MODES))
    parser.add_argument('--step-fraction', type=float, default=.125)
    parser.add_argument('--impulse-density', type=float, default=10)
    parser.add_argument('--no-nee', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--watch', action='store_true', help='Update the report as the render queue completes; implies --report-only.')
    parser.add_argument('--summarize-checks', action='store_true', help='Validate and summarize the separate paired-seed checks.')
    args = parser.parse_args()
    if args.summarize_checks:
        summarize_checks(args.output.resolve())
        build_report(args.output.resolve())
        return
    if args.report_only or args.watch:
        output = args.output.resolve()
        last_count = -1
        while True:
            records = render_rows(output)
            if len(records) != last_count:
                report = build_report(output)
                last_count = len(records)
                m = report['manifest']
                if last_count == len(m['lengths'])*len(m['modes'])*len(m['seed_ids']):
                    break
            if not args.watch:
                break
            time.sleep(30)
        return
    if (args.size < 8 or args.size % 2 or min(args.spp, args.threads) < 1 or
            any(not math.isfinite(x) or x <= 0 for x in [args.ratio, args.step_fraction, args.impulse_density, *args.lengths])):
        parser.error('Resolution must be even and at least 8; counts and scale parameters must be positive and finite.')
    if not float(args.impulse_density).is_integer():
        parser.error('Use an integer impulse density so the impulse count matches its normalization.')
    if any(not 0 <= seed < 2**32-1000000 for seed in args.seeds):
        parser.error('Seeds must fit uint32 after the independent-mode seed offset.')
    args.output = args.output.resolve()
    args.executable = args.executable.resolve()
    args.seeds = list(dict.fromkeys(args.seeds))
    args.modes = list(dict.fromkeys(args.modes))
    args.lengths = list(dict.fromkeys(args.lengths))
    base = load(args.scene)
    executable_hash = hashlib.sha256(args.executable.read_bytes()).hexdigest()
    manifest = dict(
        source_scene=str(args.scene.resolve()), ratio=args.ratio, alpha=math.sqrt(2)*args.ratio,
        lengths=args.lengths, resolution=[args.size, args.size], spp_per_seed=args.spp,
        seed_ids=args.seeds, modes=args.modes, step_fraction=args.step_fraction,
        impulse_density=args.impulse_density, executable_sha256=executable_hash,
        reference='Finite-density 3D sparse convolution; world-space global memory; one independent realization per camera path.',
        note='Changing sigma with ell preserves gradient covariance, but changes spatial thickness and geometry correlations.',
        rois={'body': [.2, .14, .8, .72], 'cavity': [.36, .22, .66, .56], 'base': [.22, .60, .78, .84]})
    save(args.output / 'manifest.json', manifest)
    rows = []
    for ell in args.lengths:
        scenes = {mode: make_scene(base, mode, ell, args) for mode in args.modes}
        first = next(iter(scenes.values()))
        for scene in scenes.values():
            for key in ['camera', 'primitives', 'bsdfs', 'integrator', 'renderer']:
                assert scene[key] == first[key], f'Unmatched {key}'
            assert scene['media'][0]['gaussian_process'] == first['media'][0]['gaussian_process']
        for mode, scene in scenes.items():
            for seed in args.seeds:
                # Disjoint random streams allow noise variances to be added across models.
                render_seed = seed + (1000000 if mode == 'gpis_global' else 0)
                folder = args.output / f'ell{tag(ell)}' / mode / f'seed{seed}'
                if verified(folder, scene, render_seed, executable_hash):
                    rows.append(load(folder / 'result.json'))
                    print(f'Reuse ell={ell:g} {mode} seed={seed}', flush=True)
                elif args.render:
                    save(folder / 'scene.json', scene)
                    command = [str(args.executable), '-t', str(args.threads), '-s', str(render_seed),
                               '-d', str(folder), str(folder / 'scene.json')]
                    print(f'Render ell={ell:g} sigma={args.ratio*ell:g} {mode} seed={seed}', flush=True)
                    start = time.monotonic()
                    with (folder / 'render.log').open('w', encoding='utf-8') as log:
                        process = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                    elapsed = time.monotonic() - start
                    record = dict(ell=ell, sigma=args.ratio*ell, mode=mode, seed_id=seed,
                                  render_seed=render_seed, spp=args.spp, seconds=elapsed,
                                  executable_sha256=executable_hash, command=command, folder=str(folder))
                    if process.returncode:
                        raise RuntimeError(f'Render failed: {folder / "render.log"}')
                    save(folder / 'result.json', record)
                    if not verified(folder, scene, render_seed, executable_hash):
                        raise RuntimeError(f'Invalid render: {folder / "render.log"}')
                    rows.append(record)
                    print(f'Finished {elapsed:.2f}s', flush=True)
                else:
                    save(args.output / 'configs' / f'ell{tag(ell)}_{mode}.json', scene)
                save(args.output / 'render_rows.json', rows)
    print(f'Completed {len(rows)} renders. Output: {args.output}', flush=True)
    build_report(args.output)


if __name__ == '__main__':
    main()
