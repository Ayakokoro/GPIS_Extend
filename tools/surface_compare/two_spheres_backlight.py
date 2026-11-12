"""Build and render a shared-SDF, fully occluded two-sphere backlight scene.

python -B tools/surface_compare/two_spheres_backlight.py prepare
python -B tools/surface_compare/two_spheres_backlight.py render --stage preview
python -B tools/surface_compare/two_spheres_backlight.py report --stage preview

The native global/renewal+ pair differs only in correlation_context. Smooth
geometry controls check the lighting arrangement, not stochastic visibility.
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'build/fp-gradient-validation-deps'))
import numpy as np
from PIL import Image, ImageDraw

OUT = ROOT / 'outputs/two-spheres-backlight-20261112'
EXE = ROOT / 'build/Release/tungsten.exe'
MODES = ('global', 'renewal_plus')
CONTROLS = ('smooth_front', 'smooth_rear_only', 'smooth_side')
LABELS = dict(global_='3D global', renewal_plus='3D renewal+',
              smooth_front='光滑双球 / 正视', smooth_rear_only='移除前球 / 检查后球照明',
              smooth_side='光滑双球 / 侧视')
LABELS['global'] = LABELS.pop('global_')
QUALITY = dict(smoke=(32, 2), preview=(64, 8), full=(256, 256))
RADIUS, SDF_SCALE = .95, .1
SIGMA, ELL, STEP = .08, .2, .0025
LIGHT_POSITION, LIGHT_RADIUS, RADIANCE = [.65, 0, 0], .01, 2500.
CAMERA = [0, 6, 0]
SEED = 41


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scene(mode, stage):
    size, spp = QUALITY[stage]
    result = dict(
        media=[], primitives=[],
        camera=dict(type='pinhole', resolution=[size, size], fov=28,
                    tonemap='filmic', reconstruction_filter='box',
                    transform=dict(position=CAMERA, look_at=[0, 1, 0], up=[0, 0, 1])),
        integrator=dict(type='path_tracer', min_bounces=0, max_bounces=20,
                        enable_light_sampling=True, enable_volume_light_sampling=True,
                        enable_consistency_checks=False, enable_two_sided_shading=True,
                        use_cuda_gpis=False, independent_path_sampling=True, wavefront_gpis=False),
        renderer=dict(output_directory='', output_file='render.png', hdr_output_file='render.exr',
                      overwrite_output_files=True, adaptive_sampling=False, enable_resume_render=False,
                      stratified_sampler=False, scene_bvh=True, spp=spp, spp_step=min(8, spp),
                      checkpoint_interval='0', timeout='0', output_buffers=[]))
    material = dict(type='lambert', albedo=.8)
    if mode in MODES:
        result['media'] = [dict(name='gp', type='sparse_conv_noise', max_bounces=128,
            gaussian_process=dict(type='standard', mean=dict(type='procedural', func='two_spheres',
                transform=dict(scale=SDF_SCALE), scale=SDF_SCALE),
                covariance=dict(type='squared_exponential', sigma=SIGMA, lengthScale=ELL)),
            phase_function=dict(type='brdf', bsdf=material),
            sigma_a=0, sigma_s=1, density=1, step_size=STEP, min_step=8,
            seed=17429, impulse_density=10, single_realization=False,
            correlation_context='global' if mode == 'global' else 'renewal+',
            intersect_method='gp_discrete', normal_method='conditioned_gaussian',
            isotropic_3D_sampling=False,
            **{'1D_sampling': False, '1D_sampling_scheme': 'uni', '1D_gradient_correlationXY': False})]
        result['primitives'].append(dict(name='shared_process_bounds', type='cube', int_medium='gp',
            bsdf=dict(type='forward', albedo=1), transform=dict(position=[0, 0, 0], scale=[3, 5, 3])))
    else:
        for name, center in [('front_sphere', [0, 1, 0]), ('rear_sphere', [0, -1, 0])]:
            if mode == 'smooth_rear_only' and name == 'front_sphere':
                continue
            result['primitives'].append(dict(name=name, type='sphere', bsdf=material,
                transform=dict(position=center, scale=RADIUS)))
        if mode == 'smooth_side':
            result['camera']['transform'].update(position=[4, 0, 2.5], look_at=[0, 0, 0])
            result['camera']['fov'] = 52
    result['primitives'].append(dict(name='rear_only_disk', type='disk', emission=RADIANCE,
        cone_angle=90, bsdf=dict(type='null', albedo=1),
        transform=dict(position=LIGHT_POSITION, scale=LIGHT_RADIUS, rotation=[180, 0, 0])))
    return result


def geometry_validation():
    front_angle = math.asin(RADIUS / 5)
    rear_angle = math.asin(RADIUS / 7)
    # Every camera-to-disk segment crosses the front sphere's center plane
    # at t=5/6. Its radial distance there is bounded for the entire disk.
    # A point strictly inside that sphere proves a preceding intersection.
    center_plane_t = 5/6
    disk_radial_bound = center_plane_t * (math.hypot(LIGHT_POSITION[0], LIGHT_POSITION[2]) + LIGHT_RADIUS)
    a = scene('global', 'preview')
    b = scene('renewal_plus', 'preview')
    b['media'][0]['correlation_context'] = 'global'
    checks = dict(rear_mean_sphere_completely_occluded=rear_angle < front_angle,
                  whole_emitter_hidden_from_camera=0 < center_plane_t < 1 and disk_radial_bound < RADIUS,
                  front_mean_sphere_entirely_behind_emitter=1-RADIUS > LIGHT_POSITION[1],
                  rear_mean_sphere_entirely_in_emitting_halfspace=-1+RADIUS < LIGHT_POSITION[1],
                  matched_pair_except_correlation_context=a == b,
                  one_shared_sdf_and_medium=len(a['media']) == 1 and len(a['primitives']) == 2)
    assert all(checks.values()), checks
    return dict(checks=checks, all_pass=True,
        front_angular_radius_degrees=math.degrees(front_angle),
        rear_angular_radius_degrees=math.degrees(rear_angle),
        mean_surface_gap=.1, disk_radius=LIGHT_RADIUS, emitter_ray_radial_bound_at_front_center_plane=disk_radial_bound,
        note='Occlusion and halfspace certificates refer to the mean geometry. Random GP surfaces can differ.')


def prepare():
    for stage in QUALITY:
        for mode in MODES + CONTROLS:
            save(OUT/stage/mode/'scene.json', scene(mode, stage))
    save(OUT/'geometry-validation.json', geometry_validation())
    save(OUT/'manifest.json', dict(seed=SEED, quality=QUALITY, radius=RADIUS,
        sphere_centers=[[0, 1, 0], [0, -1, 0]], mean_surface_gap=.1,
        mean=dict(type='procedural', func='two_spheres', transform_scale=SDF_SCALE, value_scale=SDF_SCALE),
        sigma=SIGMA, lengthScale=ELL, step_size=STEP, step_convergence_established=False,
        material=dict(type='lambert', albedo=.8), disk=dict(position=LIGHT_POSITION,
        normal=[0, -1, 0], radius=LIGHT_RADIUS, radiance=RADIANCE, cone_angle=90),
        renderer_sha256=sha(EXE), comparison='Only correlation_context differs: global versus renewal+.',
        interpretation='Exploratory scene. Image differences alone are not proof of history bias.',
        global_conditional='Not included: procedural two_spheres currently lacks certified differential queries.'))
    # Keep a directly runnable, asset-free example in the repository as well.
    example = ROOT/'data/example-scenes/two-spheres-backlight'
    save(example/'scene.json', scene('global', 'preview'))
    save(example/'renewal-plus.json', scene('renewal_plus', 'preview'))
    (OUT/'render-full.ps1').write_text('''param([string]$Renderer, [int]$Threads = 16)
$ErrorActionPreference = 'Stop'
if (-not $Renderer) { $Renderer = Join-Path $PSScriptRoot '../../build/Release/tungsten.exe' }
$Renderer = (Resolve-Path -LiteralPath $Renderer).Path
foreach ($mode in @('global', 'renewal_plus')) {
    $folder = Join-Path $PSScriptRoot ('full/' + $mode)
    & $Renderer -t $Threads -s 41 -d $folder (Join-Path $folder 'scene.json')
    if ($LASTEXITCODE -ne 0) { throw ('Render failed: ' + $mode) }
}
''', encoding='utf-8')
    print('Prepared', OUT, flush=True)


def pfm(path):
    with path.open('rb') as stream:
        assert stream.readline().strip() == b'PF'
        w, h = map(int, stream.readline().split())
        scale = float(stream.readline())
        values = np.frombuffer(stream.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)[::-1]*abs(scale)
    assert np.isfinite(values).all() and values.min() >= 0
    return values


def render(args):
    if not (OUT/'manifest.json').exists():
        prepare()
    manifest = read(OUT/'manifest.json')
    assert sha(EXE) == manifest['renderer_sha256'], 'Renderer changed since preparation'
    for mode in args.modes:
        dest = OUT/args.stage/mode
        if (dest/'run.json').exists():
            previous = read(dest/'run.json')
            if (previous['scene_sha256'] == sha(dest/'scene.json') and previous['exit_code'] == 0
                    and previous.get('finite') and (dest/'render.pfm').exists()):
                print('Reuse', mode, flush=True)
                continue
        cmd = [str(EXE), '-t', str(args.threads), '-s', str(SEED), '-d', str(dest), str(dest/'scene.json')]
        print('START', args.stage, mode, flush=True)
        start = time.monotonic()
        with (dest/'render.log').open('wb') as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            while proc.poll() is None:
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print('RUNNING', mode, round(time.monotonic()-start), 's', flush=True)
        record = dict(command=cmd, exit_code=proc.returncode, seconds=time.monotonic()-start,
            scene_sha256=sha(dest/'scene.json'), renderer_sha256=sha(EXE))
        save(dest/'run.json', record)
        if proc.returncode:
            raise RuntimeError('Render failed: ' + str(dest/'render.log'))
        log = (dest/'render.log').read_text(encoding='utf-8', errors='replace')
        assert f'Completed {QUALITY[args.stage][1]}/{QUALITY[args.stage][1]} spp' in log
        assert 'Finished render.' in log and 'Caught an internal error' not in log
        subprocess.run([str(ROOT/'build/Release/hdrmanip.exe'), '-o', str(dest/'render.pfm'),
            str(dest/'render.exr')], check=True, capture_output=True)
        rgb = pfm(dest/'render.pfm')
        assert rgb.shape == (QUALITY[args.stage][0], QUALITY[args.stage][0], 3)
        y = rgb @ np.array([.2126, .7152, .0722])
        record.update(mean_luminance=float(y.mean()), max_luminance=float(y.max()),
            nonzero_pixels=int((y > 0).sum()), finite=True, image_sha256=sha(dest/'render.exr'))
        save(dest/'run.json', record)
        print('DONE', mode, round(record['seconds'], 1), 's; mean Y', record['mean_luminance'], flush=True)
        report(args.stage)


def report(stage):
    cards, images, records = [], [], {}
    for mode in MODES + CONTROLS:
        dest = OUT/stage/mode
        if not (dest/'run.json').exists() or not (dest/'render.pfm').exists():
            continue
        record = read(dest/'run.json')
        assert record['exit_code'] == 0 and record['scene_sha256'] == sha(dest/'scene.json')
        rgb = pfm(dest/'render.pfm')
        y = rgb @ np.array([.2126, .7152, .0722])
        records[mode] = record
        # Shared gain for all experimental images, with a separate explicit
        # diagnostic view to make very small contributions visible.
        lifted = np.clip(rgb * 100, 0, 1)**(1/2.2)
        Image.fromarray(np.round(lifted * 255).astype(np.uint8)).save(dest/'gain100.png')
        cards.append(f'<article><h2>{LABELS[mode]}</h2><img src="{mode}/render.png">'
            f'<p>{record["seconds"]:.1f} s · mean Y={float(y.mean()):.6g} · '
            f'非零像素 {int((y > 0).sum())}</p><details><summary>线性亮度 ×100 / gamma 2.2</summary>'
            f'<img src="{mode}/gain100.png"></details><p><a href="{mode}/scene.json">JSON</a> · '
            f'<a href="{mode}/render.exr">EXR</a> · <a href="{mode}/render.log">log</a></p></article>')
        images.append((mode, dest/'render.png'))
    checks = dict(geometry=read(OUT/'geometry-validation.json'))
    if all(mode in records for mode in CONTROLS):
        checks['smooth_controls'] = dict(
            front_completely_dark=records['smooth_front']['max_luminance'] == 0,
            rear_illuminated=records['smooth_rear_only']['mean_luminance'] > 0,
            side_illuminated=records['smooth_side']['mean_luminance'] > 0)
        assert all(checks['smooth_controls'].values()), checks
    save(OUT/stage/'validation.json', dict(records=records, checks=checks,
        complete=all(mode in records for mode in MODES + CONTROLS),
        interpretation='Single-seed exploratory preview; convergence and statistical significance not established.'))
    if images:
        canvas = Image.new('RGB', (256*len(images), 286), 'white')
        draw = ImageDraw.Draw(canvas)
        for i, (mode, path) in enumerate(images):
            draw.text((256*i+6, 6), mode, fill='black')
            canvas.paste(Image.open(path).convert('RGB').resize((256, 256), Image.Resampling.NEAREST), (256*i, 30))
        canvas.save(OUT/stage/'overview.png')
    size, spp = QUALITY[stage]
    sweep_names = dict(coarse='参数初扫', refined='参数复核', methods_smoke='四方法检查图', methods='四方法对比',
                       classic_high='128×128 / 128 spp', classic_high_stats='128×128 / 三种子复核',
                       renewal_high='GPIS renewal+ / 128×128 / 128 spp')
    sweep_links = ''.join(f'<a href="../sweep/{p.parent.name}/comparison.html">{sweep_names.get(p.parent.name,p.parent.name)}</a> · '
                          for p in sorted((OUT/'sweep').glob('*/comparison.html')))
    methods_link = ('<p><a href="../sweep/methods/comparison.html">查看新增的 classic_global / global_conditional 四方法对比</a></p>'
                    if (OUT/'sweep/methods/comparison.html').exists() else '')
    for latest_stage in ('classic_high', 'classic_high_stats'):
        if (OUT/'sweep'/latest_stage/'comparison.html').exists():
            methods_link = f'<p><a href="../sweep/{latest_stage}/comparison.html">查看 classic_global / global_conditional 高分辨率、高采样对比</a> · <a href="../sweep/methods/comparison.html">保留的四方法参考</a></p>'
    if (OUT/'sweep/renewal_high/comparison.html').exists():
        methods_link = ('<p><a href="../sweep/renewal_high/comparison.html">查看新增 GPIS renewal+ 高采样对比</a> · '
            '<a href="../sweep/classic_high_stats/comparison.html">classic_global / global_conditional 三种子复核</a></p>')
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>双球 / 后球面光源</title>
<style>body{{max-width:1350px;margin:30px auto;padding:0 20px;background:#f4f5f7;color:#20303d;font:16px/1.6 system-ui}}h1{{font-size:27px}}h2{{font-size:18px}}.intro,article{{padding:18px;background:white;border:1px solid #d9dfe4;border-radius:8px}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin:20px 0}}img{{display:block;width:100%;image-rendering:pixelated}}a{{color:#17659a}}svg{{width:100%;max-height:300px}}@media(max-width:800px){{.cards{{grid-template-columns:1fr}}}}</style>
<h1>双球 SDF：前球遮挡，面光源照后球</h1><div class="intro">
{methods_link}
<p>两球共用一个 <code>two_spheres</code> SDF 和一个随机场；半径 0.95，球心 (0,±1,0)，平均表面间隙 0.1。
相机 (0,6,0) 沿 −Y 正视前球，后球平均轮廓完全落在前球内。没有环境光和地面。</p>
<p>圆盘面光源位于 (0.65,0,0)，半径 0.01，法线 −Y，辐亮度 2500。前球在光源背面，后球在发光半球。
这是普通单面面光源，<code>cone_angle=90</code>。圆盘尺寸小不等于聚光灯的窄光束。</p>
<svg viewBox="0 0 1000 210" role="img" aria-label="双球和光源的侧面示意，非等比">
<defs><marker id="a" markerWidth="8" markerHeight="8" refX="6" refY="3" orient="auto"><path d="M0,0 L6,3 L0,6" fill="none" stroke="currentColor"/></marker></defs>
<circle cx="270" cy="100" r="75" fill="#e4ebf0" stroke="#607787"/><circle cx="470" cy="100" r="75" fill="#c9d9e3" stroke="#607787"/>
<text x="220" y="195">后球 y=−1</text><text x="420" y="195">前球 y=+1</text>
<path d="M370 28v16" stroke="#eb9c15" stroke-width="8"/><path d="M360 36 L300 55 M360 36 L315 90" stroke="#eb9c15" stroke-width="2" marker-end="url(#a)"/>
<text x="340" y="18">缝隙中的小面光源</text><rect x="775" y="75" width="48" height="44" fill="#607787"/>
<path d="M760 100H565" stroke="#607787" stroke-width="2" marker-end="url(#a)"/><text x="765" y="150">相机 y=6</text></svg>
<p>先用白色 Lambert 微材质（albedo=0.8）检查光路。σ={SIGMA}，ℓ={ELL}，原生 3D sparse convolution，步长 {STEP}。
global 与 renewal+ 只改变 <code>correlation_context</code>。{size}×{size} / {spp} spp / seed={SEED}。</p>
<p>光滑正视图应全黑；移除前球和侧视控制图应显示后球被照亮。几何遮挡保证针对平均球面；随机表面可能变化。
这轮预览用于检查布置，不能据此宣称已证明忽略历史的偏差，尚未完成步长收敛或多种子统计。
本页保留原始的 3D global/renewal+ 对照；新增认证查询支持后的 classic_global / global_conditional 对照见下方扫描链接。</p>
<p><a href="../manifest.json">参数</a> · <a href="../geometry-validation.json">几何验证</a> · <a href="validation.json">渲染验证</a> · <a href="../render-full.ps1">正式渲染脚本</a></p>
<p>σ / lengthScale 扫描：{sweep_links}</p></div>
<div class="cards">{''.join(cards)}</div></html>'''
    (OUT/stage/'comparison.html').write_text(page, encoding='utf-8')
    (OUT/'comparison.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url='+html.escape(stage)+'/comparison.html">', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'render', 'report'))
    parser.add_argument('--stage', choices=QUALITY, default='preview')
    parser.add_argument('--modes', nargs='+', choices=MODES+CONTROLS, default=CONTROLS+MODES)
    parser.add_argument('--threads', type=int, default=16)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare()
    elif args.action == 'render':
        render(args)
    else:
        report(args.stage)


if __name__ == '__main__':
    main()
