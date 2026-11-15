"""Render the current double Adaptive GPIS with a small light and mirror material."""
import argparse
import array
import copy
import html
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from compare_adaptive_shaderball_original import ROOT, FATAL, differences, read, save, sha

ORIGINAL = ROOT/'outputs/adaptive_gpis_double_20261114/final_256_128spp/adaptive'
SMALL = ROOT/'outputs/gpis_shaderball_small_light_20261114/final/gpis_renewal_plus/seed20261024'
OUTPUT = ROOT/'outputs/adaptive_gpis_mis_20261115/small_light_256_128spp'
PREVIOUS = ROOT/'outputs/adaptive_gpis_double_20261114/small_light_256_128spp/adaptive'


def configure(original, small, setup, size, spp, sampling='mis'):
    scene = copy.deepcopy(small if setup == 'isolated-sphere' else original)
    medium = copy.deepcopy(original['media'][0])
    if sampling != 'observed':
        medium.update({'1D_sampling': True, '1D_sampling_scheme': sampling})
    physical = ('gaussian_process', 'phase_function', 'max_bounces',
                'sigma_a', 'sigma_s', 'density', 'correlation_context')
    for key in physical:
        assert medium[key] == scene['media'][0][key], key
    scene['media'] = [medium]
    assert medium['phase_function']['bsdf']['type'] == 'mirror'
    assert medium['type'] == 'adaptive_function_space_gaussian_process'
    if setup == 'isolated-sphere':
        assert [p['type'] for p in scene['primitives']] == ['cube', 'sphere']
        for key in ('primitives', 'bsdfs', 'integrator'):
            assert scene[key] == small[key], key
    else:
        light = next(p for p in scene['primitives'] if p.get('name') == 'light')
        assert light['type'] == 'quad' and light['transform']['scale'] == 5
        light['transform']['scale'] = .25  # Keep total power, position and orientation.
    scene['camera']['resolution'] = [size, size]
    camera = copy.deepcopy(scene['camera'])
    source = small if setup == 'isolated-sphere' else original
    camera['resolution'] = source['camera']['resolution']
    assert camera == source['camera']
    scene['renderer'] = copy.deepcopy(original['renderer'])
    scene['renderer'].update(spp=spp, spp_step=2)
    return scene


def pixels(path, size):
    with path.open('rb') as stream:
        assert stream.readline().strip() == b'PF', path
        assert tuple(map(int, stream.readline().split())) == (size, size), path
        scale = float(stream.readline())
        values = array.array('f')
        values.frombytes(stream.read())
    if (scale < 0) != (sys.byteorder == 'little'):
        values.byteswap()
    assert len(values) == size*size*3 and math.isfinite(scale) and scale != 0, path
    assert all(math.isfinite(x) and x >= 0 for x in values), path
    return [x*abs(scale) for x in values]


def metrics(folder, size):
    rgb = pixels(folder/'render.pfm', size)
    a, b = (pixels(folder/name, size) for name in ('samplesA.pfm', 'samplesB.pfm'))
    assert all(math.isclose(x, (y+z)*.5, rel_tol=2e-5, abs_tol=2e-6)
               for x, y, z in zip(rgb, a, b)), 'Half-buffer normalization'
    weights = (.2126, .7152, .0722)
    luminance = [sum(rgb[i+k]*weights[k] for k in range(3)) for i in range(0, len(rgb), 3)]
    noise = [sum((a[i+k]-b[i+k])*.5*weights[k] for k in range(3))
             for i in range(0, len(rgb), 3)]
    return dict(mean_luminance=sum(luminance)/len(luminance), maximum_rgb=max(rgb),
                noise_rms=math.sqrt(sum(x*x for x in noise)/len(noise)),
                nonzero_pixels=sum(x > 0 for x in luminance), total_pixels=size*size)


def report(output, manifest, row):
    data=dict(manifest=manifest, run=row)
    conditional_card=''
    conditional_path=output/'global_conditional/result.json'
    if conditional_path.exists():
        conditional=read(conditional_path)
        assert conditional['identity']['manifest']==manifest
        for name,digest in conditional['sha256'].items():
            assert sha(conditional_path.parent/name)==digest,name
        data['global_conditional']=conditional
        conditional_card=('<article><h2>global_conditional · 同场景计时</h2>'
            '<a href="global_conditional/render.png"><img src="global_conditional/render.png" alt="global_conditional 小光源"></a>'
            f'<p>渲染耗时：{conditional["render_seconds"]:.2f} s；平均亮度：{conditional["mean_luminance"]:.6g}；'
            f'半样本亮度噪声 RMS：{conditional["noise_rms"]:.6g}。<br>'
            f'Adaptive 当前计时 / global_conditional：{row["render_seconds"]/conditional["render_seconds"]:.2f}。'
            '相同程序、分辨率、spp、线程、种子、相机、灯光和材质；模型及数值近似不同，非等误差加速结论。</p>'
            '<p><a href="global_conditional/scene.json">配置</a> · <a href="global_conditional/render.log">日志</a> · '
            '<a href="global_conditional/render.pfm">线性 PFM</a> · <a href="global_conditional/result.json">完整记录</a></p></article>')
    save(output/'report.json',data)
    stats = row['statistics']
    isolated = manifest['setup'] == 'isolated-sphere'
    description = ('沿用已有小光源场景：球灯半径 0.06，位置 [-3, 3, 4]，发射辐亮度 2000；'
                   '无地面和环境光，路径反射上限 32。' if isolated else
                   '沿用完整场景：矩形灯 scale 从 5 缩为 0.25，总功率、位置和朝向保持不变；'
                   '保留地面、环境光和 20 次路径反射上限。')
    archive_link = Path(os.path.relpath(SMALL/'render.png', output)).as_posix()
    original_link = Path(os.path.relpath(ORIGINAL/'render.png', output)).as_posix()
    previous_link = Path(os.path.relpath(PREVIOUS/'render.png', output)).as_posix()
    sampling = manifest['sampling']
    sampling_note = ('已接入条件横向梯度的 1D '+sampling.upper()+'；光源连接和前向采样使用同一方向 PDF，'
                     '阴影分支以连接方向对应的完整梯度作 Renewal+ 条件。' if sampling in ('mis', 'nee') else
                     '仅按采样法线作镜面反射，未使用光源连接估计；小灯下可能出现稀疏亮点。')
    before = ''
    if isolated and manifest['size'] == 256 and manifest['spp'] == 128:
        previous=read(PREVIOUS/'result.json')
        before=(f'<article><h2>此前：按已采样法线反射</h2><a href="{previous_link}">'
                f'<img src="{previous_link}" alt="接入 MIS 前的小光源图像"></a>'
                f'<p>{previous["render_seconds"]:.2f} s；平均亮度 {previous["mean_luminance"]:.6g}；'
                f'噪声 RMS {previous["noise_rms"]:.6g}；非零像素 {previous["nonzero_pixels"]}。'
                '同场景、256×256、128 spp、32 线程；旧版程序。有限样本均值不能作为无偏性证明。</p></article>')
    archive = (f'<p><a href="{archive_link}">已有小光源 Renewal+ / 1D MIS 图像</a>：'
               '128×128、128 spp、16 线程、旧版程序。它使用不同的采样器与光源采样策略，'
               '不是本次同设置计时基线。</p>' if isolated else '')
    document = ('<!doctype html><html lang="zh-CN"><meta charset="utf-8">'
        '<title>Adaptive GPIS · 镜面 Shaderball 小光源</title>'
        '<style>body{max-width:1000px;margin:32px auto;padding:0 24px;font:16px/1.65 system-ui;'
        'background:#f5f7fa;color:#203044}article{padding:24px;background:white;border-radius:8px}'
        'img{width:512px;max-width:100%;image-rendering:pixelated}a{color:#2166b6}</style>'
        '<h1>镜面 Shaderball · 小光源</h1>'
        f'<p>{description}</p><p>绿色镜面材质、原相机、ℓ=0.2、σ=0.02。'
        '使用当前纯 double Adaptive GPIS 与 Renewal+ 条件；minimum_step=0.01，'
        'root_tolerance=1e-5，rank_tolerance=1e-10，probability_tolerance=0.001。</p>'
        f'<p>{manifest["size"]}×{manifest["size"]}，{manifest["spp"]} spp，'
        f'{manifest["threads"]} 线程，seed {manifest["seed"]}。</p>'
        f'<p>{sampling_note} 数值秩截断、最小步长和根定位容差仍是近似。'
        '本图不作为无偏性或等质量加速证明。</p>'
        f'<article><h2>当前：{html.escape(sampling.upper())}</h2><a href="adaptive/render.png"><img src="adaptive/render.png" alt="小光源镜面 Shaderball"></a>'
        f'<p>渲染耗时：{row["render_seconds"]:.2f} s；平均亮度：{row["mean_luminance"]:.6g}；'
        f'半样本亮度噪声 RMS：{row["noise_rms"]:.6g}。<br>'
        f'非零像素：{row["nonzero_pixels"]}/{row["total_pixels"]}；异常梯度：{stats["invalid_gradients"]}；'
        f'不透明起点：{stats["invalid_births"]}；计算精度：{html.escape(stats["arithmetic"])}。<br>'
        f'MIS/NEE 事件：{stats.get("mis_events",0)}；方向 PDF 查询：{stats.get("nee_pdf_queries",0)}；'
        f'连接梯度查询：{stats.get("nee_gradient_queries",0)}；未启用连续梯度 MIS 的事件：{stats.get("degenerate_gradient_events",0)}。'
        '该计数包含退化横向分布及无效纵向导数；异常梯度另计并终止对应路径。</p>'
        '<p><a href="adaptive/render.pfm">线性 PFM</a> · <a href="adaptive/scene.json">场景配置</a> · '
        '<a href="adaptive/render.log">日志</a> · <a href="adaptive/changes.json">相对来源的全部配置差异</a></p></article>'
        f'{conditional_card}{before}{archive}<p><a href="{original_link}">此前完整场景的大光源 Adaptive 图像</a></p>'
        '<p><a href="report.json">完整结果</a> · <a href="manifest.json">参数及程序哈希</a> · '
        '<a href="source.json">来源场景快照</a></p></html>')
    (output/'comparison.html').write_text(document, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--setup', choices=['isolated-sphere', 'full-small-quad'], default='isolated-sphere')
    parser.add_argument('--sampling', choices=['observed', 'uni', 'nee', 'mis'], default='mis')
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--spp', type=int, default=128)
    parser.add_argument('--threads', type=int, default=32)
    parser.add_argument('--seed', type=int, default=20261024)
    parser.add_argument('--timeout', type=int, default=7200)
    args = parser.parse_args()
    if min(args.size, args.spp, args.threads, args.timeout) <= 0 or args.spp % 2:
        parser.error('Positive parameters and even spp are required')
    original, small = read(ORIGINAL/'scene.json'), read(SMALL/'scene.json')
    for folder in (ORIGINAL, SMALL):
        record = read(folder/'result.json')
        assert sha(folder/'scene.json') == record['sha256']['scene.json'], folder
    source = small if args.setup == 'isolated-sphere' else original
    scene = configure(original, small, args.setup, args.size, args.spp, args.sampling)
    exe, output = ROOT/'build/Release/tungsten.exe', args.output.resolve()
    runtime = {p.name:sha(p) for p in [exe, *sorted(exe.parent.glob('*.dll'))]}
    manifest = dict(setup=args.setup, sampling=args.sampling, size=args.size, spp=args.spp, threads=args.threads, seed=args.seed,
        executable=str(exe), runtime_sha256=runtime,
        sources={str(p/'scene.json'):sha(p/'scene.json') for p in (ORIGINAL, SMALL)})
    if (output/'manifest.json').exists():
        assert read(output/'manifest.json') == manifest, 'Inputs changed; select a new output directory'
    save(output/'manifest.json', manifest)
    save(output/'source.json', source)
    folder = output/'adaptive'
    identity = dict(scene=scene, manifest=manifest)
    if (folder/'result.json').exists():
        row = read(folder/'result.json')
        assert row['identity'] == identity
    else:
        save(folder/'scene.json', scene)
        save(folder/'changes.json', differences(source, scene))
        command = [str(exe), '-t', str(args.threads), '-s', str(args.seed), '-d', str(folder), str(folder/'scene.json')]
        save(folder/'command.json', command)
        print('RENDER', args.setup, args.size, args.spp, flush=True)
        begin = time.perf_counter()
        with (folder/'render.log').open('wb') as log:
            completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                timeout=args.timeout, creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        wall = time.perf_counter()-begin
        assert sha(exe) == runtime[exe.name], 'Renderer changed during run'
        log = (folder/'render.log').read_text(encoding='utf-8', errors='replace')
        if completed.returncode or FATAL.search(log) or not re.search(rf'Completed {args.spp}/{args.spp} spp', log):
            raise RuntimeError(f'Render failed: {folder / "render.log"}')
        stats = json.loads(re.findall(r'^Adaptive GPIS statistics: (.+)$', log, re.M)[-1])
        assert stats['arithmetic'] == 'double' and stats['hits'] > 0
        if args.sampling in ('mis', 'nee'):
            assert stats['mis_events'] > 0 and stats['nee_pdf_queries'] > 0 and stats['nee_gradient_queries'] > 0
        row = dict(identity=identity, wall_seconds=wall,
            render_seconds=float(re.findall(r'Completed \d+/\d+ spp, ([\d.]+)/', log)[-1]),
            statistics=stats, **metrics(folder, args.size),
            warnings=dict(zero_length_path_terminations=log.count('Path tracing bounces ended with farT == 0'),
                          nee_sanity_checks=log.count('1D NEE sanity check failed')),
            sha256={name:sha(folder/name) for name in ('scene.json', 'changes.json', 'command.json',
                'render.png', 'render.pfm', 'samplesA.pfm', 'samplesB.pfm', 'samplesVariance.pfm', 'render.log')})
        save(folder/'result.json', row)
        print('DONE', row['render_seconds'], 's', flush=True)
    for name, digest in row['sha256'].items():
        assert sha(folder/name) == digest, name
    report(output, manifest, row)
    print(output/'comparison.html', flush=True)


if __name__ == '__main__':
    main()
