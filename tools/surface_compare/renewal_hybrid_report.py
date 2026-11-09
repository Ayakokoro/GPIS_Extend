"""Validate and compare native renewal+, its fixed5 hybrid, and conditional fixed5.

Image differences include Monte Carlo noise. This report does not treat an
archived 128-spp render as a noise-free reference or infer a causal percentage.
"""
from pathlib import Path
import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
import os
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
os.environ.setdefault('MPLCONFIGDIR', str(REPO / 'build/hybrid-report-matplotlib-cache'))
sys.path.insert(0, str(REPO / 'build/fp-gradient-validation-deps'))
sys.path.insert(0, str(REPO / 'tools'))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from compare_shaderball_fixed_ratio import read_pfm

DEFAULT = REPO / 'outputs/renewal-plus-fixed5-hybrid-shaderball-20261109'
REFERENCE = REPO / 'outputs/renewal-plus-shaderball-l0p2-s0p02-256-128spp-20261108/renewal_plus'
CONDITIONAL = REPO / 'outputs/fixed5-anchor-shaderball-l0p2-s0p02-256-128spp-20261109/conditional'
ROI = (54, 25, 188, 146)  # Chosen on the native image before the hybrid finished.
COLORS = {'native': '#176b96', 'hybrid': '#bf3569', 'conditional': '#ae6c0d'}
LABELS = {'native': '原生 renewal+', 'hybrid': 'renewal+ 自由程 / 阴影 + fixed5 梯度',
          'conditional': 'global_conditional + fixed5'}
ENGLISH = {'native': 'Native renewal+', 'hybrid': 'Renewal+ transport / fixed5 gradient',
           'conditional': 'Global conditional / fixed5 gradient'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def link(root, path):
    return Path(os.path.relpath(path, root)).as_posix()


def read_log(path):
    raw = Path(path).read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8', errors='replace')


def luminance(rgb):
    return rgb @ np.array([.2126, .7152, .0722])


def build(root):
    root = Path(root).resolve()
    provenance = read(root / 'render-provenance.json')
    for name, expected in provenance['sha256'].items():
        assert digest(REPO / name) == expected, ('changed since render start', name)
    checks = read(root / 'implementation-validation.json')
    assert checks['all_pass'] and checks['disabled_native_pixels_bit_identical']
    assert checks['renderer_sha256'] == digest(REPO / 'build/Release/tungsten.exe')
    folders = {'native': REFERENCE, 'hybrid': root / 'hybrid', 'conditional': CONDITIONAL}
    native_scene = read(REFERENCE / 'scene.json')
    hybrid_scene = read(root / 'hybrid/scene.json')
    expected = json.loads(json.dumps(native_scene))
    expected['media'][0]['type'] = 'renewal_plus_fp_gradient'
    expected['media'][0]['fp_gradient'] = dict(enabled=True, bridge_correction=True, anchor_selection='fixed5_snr')
    assert hybrid_scene == expected, 'native/hybrid configuration differs beyond the gradient mode'
    images, records, profiles = {}, {}, {}
    cache = root / 'image-data'
    cache.mkdir(exist_ok=True)
    x0, y0, x1, y1 = ROI
    for key, folder in folders.items():
        scene = read(folder / 'scene.json')
        assert all(scene[k] == native_scene[k] for k in native_scene if k != 'media'), key
        covariance = scene['media'][0]['gaussian_process']['covariance']
        assert covariance['lengthScale'] == .2 and covariance['sigma'] == .02
        assert scene['camera']['resolution'] == [256, 256] and scene['renderer']['spp'] == 128
        run = read(folder / 'run.json')
        assert run['exit_code'] == 0 and run['command'][run['command'].index('-s') + 1] == '41'
        log = read_log(folder / 'render.log')
        assert 'Completed 128/128 spp' in log and 'Finished render.' in log
        errors = re.findall(r'^.*(?:Caught an internal error|unrecoverable error|Failed to sample gradient|Gradient invalid|Gradient zero|'
                            r'Wrong normal direction|invalid renewal\+|invalid FP|nonfinite|terminate called).*$',
                            log, re.I | re.M)
        assert not errors, (key, errors)
        checked = subprocess.run([str(REPO / 'build/Release/classic_global_fp_gradient_test.exe'),
                                  'image', str(folder / 'shaderball.exr')], capture_output=True, text=True, check=True)
        (root / (key + '-image-validation.log')).write_text(checked.stdout, encoding='utf-8')
        pfm = cache / (key + '.pfm')
        subprocess.run([str(REPO / 'build/Release/hdrmanip.exe'), '-o', str(pfm), str(folder / 'shaderball.exr')],
                       capture_output=True, check=True)
        rgb = read_pfm(pfm)
        assert rgb.shape == (256, 256, 3) and np.isfinite(rgb).all() and np.min(rgb) >= 0
        images[key] = rgb
        yy = luminance(rgb)
        # Average thin bands to expose shape while displaying the unsmoothed images too.
        profiles[key] = {'vertical': yy[24:159, 110:146].mean(axis=1).tolist(),
                         'horizontal': yy[122:135, 50:191].mean(axis=0).tolist()}
        records[key] = dict(source=link(root, folder), reused=key != 'hybrid',
            elapsed_seconds=run['wall_seconds'], mean_rgb=float(rgb.mean()),
            mean_luminance=float(yy.mean()), groove_rectangle_mean_luminance=float(yy[y0:y1, x0:x1].mean()),
            minimum=float(rgb.min()), maximum=float(rgb.max()),
            native_far_t_zero_messages=log.count('bounces ended with farT == 0'),
            sha256={name: digest(folder / name) for name in ['scene.json', 'shaderball.png', 'shaderball.exr', 'render.log']})
    log = read_log(root / 'hybrid/render.log')
    match = re.search(r'GPIS renewal\+ FP-gradient statistics: (\{[^\n]+\})', log)
    assert match, 'missing hybrid correction counters'
    counters = json.loads(match[1])
    assert counters['corrected_collisions'] > 0
    assert counters['corrected_collisions'] + counters['retained_native_unsupported_births'] + counters['retained_native_zero_length'] == counters['collisions']
    metrics = {}
    for key in ['hybrid', 'conditional']:
        delta = images[key] - images['native']
        metrics[key] = dict(rgb_rmse=float(np.sqrt(np.mean(delta**2))),
            groove_rectangle_rgb_rmse=float(np.sqrt(np.mean(delta[y0:y1, x0:x1]**2))),
            vertical_profile_rmse=float(np.sqrt(np.mean((np.array(profiles[key]['vertical']) - profiles['native']['vertical'])**2))),
            horizontal_profile_rmse=float(np.sqrt(np.mean((np.array(profiles[key]['horizontal']) - profiles['native']['horizontal'])**2))))
    data = dict(configuration=provenance['configuration'], renders=records, correction_counters=counters,
                groove_rectangle_xyxy=list(ROI), profiles=profiles, differences_to_native=metrics,
                interpretation='Differences include Monte Carlo noise; one seed, no image-level confidence interval. '
                'Native finite-density sparse convolution and finite marching step are retained.',
                implementation_checks=checks, all_pass=True)
    write(root / 'validation.json', data)
    write(root / 'report-data.json', data)
    np.savez_compressed(root / 'linear-images.npz', **images)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for key in folders:
        axes[0].plot(np.arange(24, 159), profiles[key]['vertical'], color=COLORS[key], label=ENGLISH[key], lw=1.7)
        axes[1].plot(np.arange(50, 191), profiles[key]['horizontal'], color=COLORS[key], label=ENGLISH[key], lw=1.7)
    axes[0].set(title='Vertical profile: mean over x = 110..145', xlabel='Image y', ylabel='Linear luminance')
    axes[1].set(title='Lower groove: mean over y = 122..134', xlabel='Image x', ylabel='Linear luminance')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    fig.savefig(root / 'groove-profiles.svg')
    fig.savefig(root / 'groove-profiles.png', dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 5), constrained_layout=True)
    ax.imshow(plt.imread(REFERENCE / 'shaderball.png'))
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, color='#ffe66a', lw=1.5, label='Groove viewing rectangle'))
    ax.add_patch(Rectangle((110, 24), 36, 135, fill=False, color='#cf4a84', lw=1.3, label='Vertical profile band'))
    ax.add_patch(Rectangle((50, 122), 141, 13, fill=False, color='#4d9ac9', lw=1.3, label='Horizontal profile band'))
    ax.set(xlim=(0, 256), ylim=(256, 0), xlabel='Image x', ylabel='Image y')
    ax.legend(fontsize=7, loc='lower left')
    fig.savefig(root / 'profile-regions.svg')
    plt.close(fig)

    cards = []
    for key, folder in folders.items():
        src = html.escape(link(root, folder / 'shaderball.png'), quote=True)
        links = ' · '.join(f'<a href="{html.escape(link(root, folder / name), quote=True)}">{label}</a>'
                           for name, label in [('shaderball.png', 'PNG'), ('shaderball.exr', 'EXR'), ('scene.json', '场景'), ('render.log', '日志')])
        cards.append(f'<article><h2>{LABELS[key]}</h2><p>{"本次新渲染" if key == "hybrid" else "复用已完成的同配置渲染"}</p>'
                     f'<a href="{src}"><img class="full" src="{src}" alt="{LABELS[key]}"></a>'
                     f'<h3>凹槽局部</h3><svg class="crop" viewBox="{x0} {y0} {x1-x0} {y1-y0}">'
                     f'<image href="{src}" width="256" height="256"/></svg><p>{links}</p></article>')

    def slider(left, right, name):
        a = html.escape(link(root, folders[left] / 'shaderball.png'), quote=True)
        b = html.escape(link(root, folders[right] / 'shaderball.png'), quote=True)
        return (f'<section><h2>{name}</h2><div class="slider"><img src="{b}" alt="{LABELS[right]}">'
                f'<img class="overlay" src="{a}" alt="{LABELS[left]}"><div class="divider"></div></div>'
                f'<input class="wipe" type="range" min="0" max="100" value="50" aria-label="{name}">'
                f'<p>左：{LABELS[left]}　｜　右：{LABELS[right]}</p></section>')

    rows = ''.join(f'<tr><td>{LABELS[k]}</td><td>{records[k]["groove_rectangle_mean_luminance"]:.6f}</td>'
                   f'<td>{metrics[k]["groove_rectangle_rgb_rmse"]:.6f}</td>'
                   f'<td>{metrics[k]["vertical_profile_rmse"]:.6f}</td><td>{metrics[k]["horizontal_profile_rmse"]:.6f}</td></tr>'
                   for k in ['hybrid', 'conditional'])
    retained = counters['retained_native_zero_length'] + counters['retained_native_unsupported_births']
    correction_percent = 100 * counters['corrected_collisions'] / counters['collisions']
    profile_comparison = ('两条预先选定的亮度剖面差异都更小' if all(
        metrics['hybrid'][k] < metrics['conditional'][k]
        for k in ['vertical_profile_rmse', 'horizontal_profile_rmse']) else
        '两条预先选定的亮度剖面未同时改善')
    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>renewal+ 输运 × fixed5 梯度：shaderball 混合测试</title>
<style>
body{margin:0 auto;max-width:1460px;padding:28px;font:16px/1.6 system-ui,sans-serif;color:#172936;background:#f4f6f8}
h1{font-size:28px}h2{font-size:19px}h3{font-size:16px}.intro,article,section{background:white;padding:20px;border:1px solid #dae0e6;border-radius:10px;margin-bottom:20px}
.cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.cards article{padding:16px}.full,.crop{display:block;width:100%;image-rendering:pixelated}.crop{margin-top:8px;background:#e8ebee}
.sliders{display:grid;grid-template-columns:1fr 1fr;gap:20px}.slider{position:relative;width:100%;aspect-ratio:1}.slider img{display:block;width:100%;image-rendering:pixelated}.slider .overlay{position:absolute;inset:0;clip-path:inset(0 50% 0 0)}.divider{position:absolute;left:50%;top:0;bottom:0;border-left:2px solid white}.wipe{width:100%}
table{border-collapse:collapse;width:100%}th,td{padding:9px;text-align:left;border-bottom:1px solid #dde2e6}th{background:#eef3f7}.scroll{overflow:auto}a{color:#175f91}.figure{width:100%;height:auto}.regions{max-width:430px}.note{color:#53606c}code{font-family:ui-monospace,monospace}.metric{font-variant-numeric:tabular-nums}
@media(max-width:900px){body{padding:14px}.cards,.sliders{grid-template-columns:1fr}}
</style>'''
    page += f'''<h1>renewal+ 输运 × fixed5 梯度：shaderball 混合测试</h1>
<div class="intro"><p><strong>控制变量：保留原生 renewal+ 的求交、阴影可见性、材质和镜面反射估计方式，只在接受碰撞后替换完整梯度。</strong></p>
<p>ℓ = 0.2，σ = 0.02，256 × 256，128 spp，seed = 41；原生步长 0.01，impulse density = 10，3D sampling，renewal+ memory。</p>
<p>固定起始状态的距离 / 阴影一致性、完整梯度传播及波前接口等 {checks['focused_ctest_count']} 项检查通过。关闭修正时，32 × 32 / 4 spp 试渲染与原生结果逐像素一致。</p>
<p>本次 {counters['collisions']:,} 次接受碰撞中，{counters['corrected_collisions']:,} 次使用 fixed5（{correction_percent:.3f}%）；保留原生梯度 {retained:,} 次，其中距离不满足 t &gt; 0（统计键 zero_length）{counters['retained_native_zero_length']:,} 次，不满足正侧起始条件 {counters['retained_native_unsupported_births']:,} 次。采样器备用精确包络被调用 {counters['sampling_fallbacks']:,} 次，它仍采样同一个 fixed5 分布。</p>
<p class="note">原生参考与 conditional 图复用此前的同参数渲染。替换梯度后，反射方向和下一段起始梯度会变化，因此后续路径会分叉。这里检验的是替换梯度的整体影响，不是逐路径保持相同。</p></div>
<section><p><strong>本次观测：</strong>相对原生 renewal+ 图，混合模式的凹槽矩形 RGB RMSE 为 {metrics['hybrid']['groove_rectangle_rgb_rmse']:.6f}，conditional + fixed5 为 {metrics['conditional']['groove_rectangle_rgb_rmse']:.6f}；{profile_comparison}。这是单个随机种子的图像差异，仍包含采样噪声，不能据此认定全部残差都来自透射率。</p></section>
<div class="cards">{''.join(cards)}</div>
<div class="sliders">{slider('hybrid','native','混合模式与原生 renewal+')}{slider('hybrid','conditional','混合模式与 conditional + fixed5')}</div>
<section><h2>凹槽亮度剖面：线性 EXR</h2><p>区域在混合渲染完成前按原生图选定。黄色矩形用于局部放大和区域统计，它不是精确凹槽分割；两条带状区域用于取平均亮度剖面。</p>
<img class="regions" src="profile-regions.svg" alt="区域与亮度剖面位置"><img class="figure" src="groove-profiles.svg" alt="两条线性亮度剖面">
<p><a href="groove-profiles.svg">剖面 SVG</a> · <a href="groove-profiles.png">剖面 PNG</a> · <a href="linear-images.npz">线性图像数据</a></p>
<div class="scroll"><table><tr><th>模式</th><th>矩形内平均亮度</th><th>矩形 RGB RMSE</th><th>纵向剖面 RMSE</th><th>横向剖面 RMSE</th></tr>{rows}</table></div>
<p class="note">RMSE 均相对这张 128 spp 原生图，包含两张图的蒙特卡洛噪声，不是对连续 GP 真值的误差。只有一个随机种子，没有图像层面的置信区间。</p></section>
<section><h2>如何解释这次对照</h2><p>重点比较混合模式与原生 renewal+：若二者的凹槽外观接近，说明在原生输运下，fixed5 梯度足以保留主要外观；conditional 与它们的差异应继续检查自由程、阴影与起始状态处理。若混合模式本身明显改变凹槽，则梯度近似仍有可见影响。</p>
<p>原生 renewal+ 使用有限密度稀疏卷积和离散求交；本测试保留这些数值近似。因此它不等于理想连续 SE-GP 的精确参考。也不能把图像 RMSE 直接解释为透射率或梯度各占多少误差。</p>
<p><a href="validation.json">完整验证与指标</a> · <a href="implementation-validation.json">实现检查</a> · <a href="test-details.log">测试详情</a> · <a href="render-provenance.json">渲染来源与哈希</a> · <a href="report-data.json">报告数据</a> · <a href="{link(root, REPO/'docs/classic-global-fp-gradient.md')}">模式说明</a></p></section>
<script>document.querySelectorAll('.wipe').forEach(r=>r.addEventListener('input',()=>{{const s=r.previousElementSibling;s.querySelector('.overlay').style.clipPath=`inset(0 ${{100-r.value}}% 0 0)`;s.querySelector('.divider').style.left=r.value+'%';}}));</script></html>'''
    (root / 'comparison.html').write_text(page, encoding='utf-8')
    parser = LinkAudit(root)
    parser.feed(page)
    assert parser.ranges == 2 and not parser.missing, parser.missing
    write(root / 'artifact-validation.json', dict(all_pass=True, links_checked=parser.count,
        sliders=parser.ranges, missing=parser.missing, report_sha256=digest(root/'comparison.html')))
    print(json.dumps(dict(all_pass=True, correction_counters=counters, differences_to_native=metrics,
        report=str(root/'comparison.html')), ensure_ascii=False))


class LinkAudit(HTMLParser):
    def __init__(self, root):
        super().__init__()
        self.root, self.missing, self.count, self.ranges = root, [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag == 'input' and a.get('type') == 'range':
            self.ranges += 1
        target = a.get('href') or a.get('src')
        if target and not target.startswith(('data:', '#', 'http:','https:')):
            self.count += 1
            if not (self.root / target.split('#')[0]).exists():
                self.missing.append(target)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    build(parser.parse_args().root)
