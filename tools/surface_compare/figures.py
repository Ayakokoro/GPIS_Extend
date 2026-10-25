"""Create the four comparison figures from renderer and production-probe outputs.

Run with a Python environment containing NumPy, Pillow and Matplotlib. Optional
project-local packages in build/plot-packages-py310 take precedence on Python 3.10.
No image denoising, exposure fitting or per-panel normalization is performed.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
packages = ROOT / 'build/plot-packages-py310'
if sys.version_info[:2] == (3, 10) and packages.exists():
    sys.path.insert(0, str(packages))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'build/matplotlib-cache'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import Normalize
from matplotlib.patches import Circle, Rectangle
from PIL import Image
from run import OUT, MODES, RENDER_MODES, CASES, load, save, pfm, sha

font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'savefig.facecolor': 'white', 'figure.facecolor': 'white'})
COLORS = ('#1675b9', '#df822d', '#9a2b85', '#168469')
STYLES = ('-', '--', ':', '-.')
CASES_SELECTED = {'small': 'studio_ell0p025', 'large': 'grazing_ell0p025'}
SEED = 20261024
# Fixed before viewing the final render: the left groove and adjacent highlights.
ROI = (.14, .34, .50, .62)  # x0, y0, x1, y1 in image coordinates


def folder(stage, case, mode):
    return OUT / stage / case / mode / f'seed{SEED}'


def png(path):
    # Tungsten writes a single-channel PNG for this achromatic scene.
    # Converting to RGB prevents Matplotlib from applying a false color map.
    return np.asarray(Image.open(path).convert('RGB'))


def blocked_metrics(stage, case, modes=MODES):
    images = np.array([pfm(folder(stage, case, m) / 'render.pfm').mean(2) for m in modes])
    n = images.shape[1]
    assert images.shape[2] == n and n % 16 == 0
    b = images.reshape(len(modes), 16, n // 16, 16, n // 16).mean((2, 4))
    y, x = np.mgrid[:16, :16]
    mask = (x - 7.5) ** 2 + (y - 7.5) ** 2 < 5.5 ** 2
    reference = b[1][mask]
    result = {'block_grid': [16, 16], 'mask': 'circle, center=(7.5,7.5), radius=5.5 blocks',
              'normalization': 'mean classic_global radiance within mask',
              'object_mean_radiance': b[:, mask].mean(1).tolist()}
    pairs = [('local_global', 0), ('conditional_global', 2)]
    if len(modes) == 4:
        pairs.append(('gpis_global', 3))
    for key, i in pairs:
        result[key + '_nrmse'] = float(np.sqrt(np.mean((b[i][mask] - reference) ** 2)) / reference.mean())
    x0, y0, x1, y1 = [int(t * n) for t in ROI]
    result['groove_roi_normalized'] = ROI
    result['groove_mean_radiance'] = images[:, y0:y1, x0:x1].mean((1, 2)).tolist()
    return result


def preview():
    fig, axes = plt.subplots(6, 3, figsize=(10, 18))
    ranking = {}
    for row, case in enumerate(CASES):
        ranking[case] = blocked_metrics('preview', case)
        for col, mode in enumerate(MODES):
            ax = axes[row, col]
            ax.imshow(png(folder('preview', case, mode) / 'render.png'))
            ax.axis('off')
            if row == 0:
                ax.set_title(mode)
            if col == 0:
                ax.text(-.04, .5, case, rotation=90, transform=ax.transAxes,
                        va='center', ha='right', fontsize=10)
    fig.subplots_adjust(wspace=.02, hspace=.06, left=.09, right=.99, top=.97, bottom=.01)
    fig.savefig(OUT / 'preview_grid.png', dpi=140)
    plt.close(fig)
    save(OUT / 'preview_metrics.json', {
        'cases': ranking, 'selected': CASES_SELECTED,
        'selection': 'Use ell=0.025 in both figures to hold roughness fixed. Studio lighting gives '
                     'the lowest preview conditional/global block error; grazing lighting increases it. '
                     'These noisy preview errors are screening measurements, not significance tests.'})


def appearance(label, include_gpis=False, stage='final', gpis_stage=None):
    modes = RENDER_MODES if include_gpis else MODES
    case = CASES_SELECTED[label]
    ray = load(OUT / 'probes' / f'distribution_{label}.json')
    source = [folder(gpis_stage if m == 'sparse_3d_global' and gpis_stage else stage, case, m) for m in modes]
    for d in source:
        assert (d / 'result.json').exists(), f'Render not complete: {d}'
    scenes = [load(d / 'scene.json') for d in source]
    assert Path(ray['field']).resolve() == Path(scenes[0]['media'][0]['field']['file']).resolve()
    size = scenes[0]['camera']['resolution'][0]
    spp = scenes[0]['renderer']['spp']
    for s in scenes[1:]:
        assert s['camera'] == scenes[0]['camera']
        assert s['integrator'] == scenes[0]['integrator']
        assert s['primitives'] == scenes[0]['primitives']
        assert s['renderer']['spp'] == spp
    if include_gpis:
        gpis = scenes[3]['media'][0]
        assert gpis['gaussian_process'] == scenes[1]['media'][0]['gaussian_process'] == scenes[2]['media'][0]['gaussian_process']
        assert gpis['phase_function']['bsdf'] == scenes[0]['media'][0]['material'] == scenes[1]['media'][0]['material'] == scenes[2]['media'][0]['material']
        assert gpis['gaussian_process']['covariance'] == scenes[0]['media'][0]['covariance']
        assert gpis['correlation_context'] == 'global' and not gpis['1D_sampling'] and not gpis['single_realization']
    title = '外观对照 A · 宽条光源' if label == 'small' else '外观对照 B · 掠射光照'
    fig, axes = plt.subplots(2, len(modes), figsize=(5.2*len(modes), 9.4), gridspec_kw={'height_ratios': [1, .60]})
    for i, (mode, d) in enumerate(zip(modes, source)):
        im = png(d / 'render.png')
        assert im.shape == (size, size, 3)
        ax = axes[0, i]
        ax.imshow(im, interpolation='nearest')
        ax.set_title('GPIS · sparse_3d_global' if mode == 'sparse_3d_global' else mode, fontsize=14, color=COLORS[i], pad=12)
        x0, y0, x1, y1 = [int(v * size) for v in ROI]
        ax.add_patch(Rectangle((x0, y0), x1-x0, y1-y0, fill=False, lw=1.0, edgecolor='#ffcf52'))
        px, py = (np.array(ray['pixel']) + .5) * size / ray['diagnostic_resolution'][0] - .5
        ax.plot(px, py, '+', color='#66e7ff', ms=10, mew=1.3)
        ax.axis('off')
        axes[1, i].imshow(im[y0:y1, x0:x1], interpolation='nearest')
        axes[1, i].axis('off')
    fig.suptitle(title, x=.04, ha='left', y=.985, fontsize=20, weight='bold')
    sampling = '镜面反射率 0.9；GPIS 为逐路径独立的 3D 稀疏卷积场' if include_gpis else 'NEE + MIS'
    fig.text(.04, .935, f'同一圆滑凹槽模型  |  σ = 0.01, ℓ = 0.025  |  {size} × {size}, {spp} spp  |  {sampling}', color='#555b65', fontsize=11)
    if include_gpis:
        fig.text(.04, .078, f'GPIS：每个噪声网格单元 {gpis["impulse_density"]} 个随机脉冲；步长 {gpis["step_size"]:g}；沿整条路径保持同一随机场。', fontsize=10)
        fig.text(.04, .053, '下排为原图放大；统一 filmic 映射，无降噪、无单独调亮。前三种模式使用 NEE + MIS；GPIS 的真实镜面事件跳过普通光源 NEE，噪声更大。', fontsize=10)
    else:
        fig.text(.04, .065, '下排为黄色框的原图放大；统一 filmic 映射，无降噪、无单独调亮。', fontsize=10)
    pixel_text = f'({ray["pixel"][0]}, {ray["pixel"][1]}) / {ray["diagnostic_resolution"][0]}²'
    fig.text(.04, .028 if include_gpis else .035, f'青色十字：诊断相机像素 {pixel_text}。法线统计固定该相机光线；诊断路径不消耗直接光采样随机数。', fontsize=10, color='#555b65')
    fig.subplots_adjust(left=.035, right=.99, top=.87, bottom=.12 if include_gpis else .105, wspace=.035, hspace=.06)
    path = OUT / f'appearance_{label}{"_with_gpis" if include_gpis else ""}{"" if stage == "final" else "_"+stage}.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {'case': case, 'figure': str(path), 'size': size, 'spp': spp,
            'metrics': blocked_metrics(stage, case, modes) if not gpis_stage else None,
            'sources': [{'mode': m, 'folder': str(d), 'png_sha256': sha(d / 'render.png'),
                         'result': load(d / 'result.json')} for m, d in zip(modes, source)]}


def disk_image(grid, side=501):
    """Bilinear reconstruction on a full-sphere Lambert equal-area disk.

    The center is the common mean normal. Disk radius r obeys cos(theta)=1-2r²;
    d(omega)=4 dx dy. No probability or peak normalization is performed here.
    """
    nz, nphi = grid.shape
    v = np.linspace(-1, 1, side)
    x, y = np.meshgrid(v, v)
    radius2 = x*x + y*y
    z = (1 - 2 * radius2 + 1) * nz / 2 - .5
    phi = (np.arctan2(y, x) + np.pi) * nphi / (2*np.pi) - .5
    z = np.clip(z, 0, nz-1)
    zi = np.floor(z).astype(int)
    pi = np.floor(phi).astype(int)
    dz, dp = z-zi, phi-pi
    result = ((1-dz)*((1-dp)*grid[zi, pi % nphi] + dp*grid[zi, (pi+1) % nphi]) +
              dz*((1-dp)*grid[np.minimum(zi+1, nz-1), pi % nphi] + dp*grid[np.minimum(zi+1, nz-1), (pi+1) % nphi]))
    return np.ma.array(result, mask=radius2 > 1)


def project(n, frame):
    x, y, z = frame.T @ n
    r = np.sqrt(max(0., (1-z)/2))
    p = np.arctan2(y, x)
    return r*np.cos(p), r*np.sin(p)


def normals(label):
    record_path = OUT / 'probes' / f'distribution_{label}.json'
    r = load(record_path)
    d = r['distribution']
    nz, nphi = d['nz'], d['nphi']
    fig = plt.figure(figsize=(14.6, 12))
    layout = fig.add_gridspec(3, 3, left=.045, right=.93, bottom=.19, top=.85,
                             height_ratios=[1, 1, .70], hspace=.22, wspace=.14)
    title = '法线分布 A · 差异很小' if label == 'small' else '法线分布 B · 差异明显'
    fig.suptitle(title, x=.04, y=.983, ha='left', fontsize=21, weight='bold')
    pixel_text = f'({r["pixel"][0]}, {r["pixel"][1]}) / {r["diagnostic_resolution"][0]}²'
    fig.text(.04, .939, f'同一相机路径，碰撞 #{r["bounce"]+1}  |  起始观测距离 / ℓ = {r["age"]/r["length_scale"]:.3f}  |  相机像素 {pixel_text}', fontsize=11)
    tv = d['tv_conditional_global']
    tv_text = f'≈ {tv:.4f}' if tv > 1e-6 else '< 10⁻⁶'
    fig.text(.04, .907, f'VNDF 总变差距离：conditional ↔ global {tv_text}；local ↔ global ≈ {d["tv_local_global"]:.4f}  （0 表示相同，1 表示完全分离）', fontsize=10, color='#555b65')
    frame = np.array(d['frame'])
    incoming = -np.array(r['wi'])
    wi_disk = project(incoming, frame)
    quantities = []
    for row, key in enumerate(('ndf', 'vndf')):
        grids = np.array(d[key]).reshape(3, nz, nphi)
        quantities.append(grids)
        norm = Normalize(0, float(grids.max()))
        for col, mode in enumerate(MODES):
            ax = fig.add_subplot(layout[row, col])
            im = ax.imshow(disk_image(grids[col]), extent=(-1, 1, -1, 1), origin='lower',
                           cmap='magma', norm=norm, interpolation='nearest')
            ax.add_patch(Circle((0, 0), 1, fc='none', ec='#b1a7b4', lw=.6))
            ax.add_patch(Circle((0, 0), 1/np.sqrt(2), fc='none', ec='#b1a7b4', lw=.7, ls='--'))
            ax.plot(*wi_disk, 'x', color='#72edff', ms=6, mew=1)
            ax.plot(0, 0, '+', color='white', ms=5, mew=.7)
            ax.set_xlim(-1.03, 1.03); ax.set_ylim(-1.03, 1.03)
            ax.axis('off')
            if row == 0:
                ax.set_title(mode, color=COLORS[col], fontsize=13, pad=11)
            if col == 0:
                ax.text(-.11, .5, 'NDF · D(n)' if row == 0 else 'VNDF · p(n | w)', transform=ax.transAxes,
                        rotation=90, va='center', ha='center', fontsize=12)
        cax = fig.add_axes([.947, .65 if row == 0 else .405, .012, .16])
        bar = fig.colorbar(im, cax=cax)
        bar.ax.tick_params(labelsize=8)
        bar.set_label('D(n)' if row == 0 else 'sr⁻¹', fontsize=9)
    cos_theta = -1 + 2*(np.arange(nz)+.5)/nz
    theta = np.degrees(np.arccos(cos_theta))
    for col, grids in enumerate(quantities):
        ax = fig.add_subplot(layout[2, col])
        for i, mode in enumerate(MODES):
            # Integrate azimuth; transform d cos(theta) to degrees of theta.
            marginal = grids[i].mean(1) * 2*np.pi * np.sqrt(1-cos_theta**2) * np.pi/180
            ax.plot(np.r_[0, theta[::-1], 180], np.r_[0, marginal[::-1], 0], color=COLORS[i], ls=STYLES[i], lw=2,
                    label=mode)
        ax.set(xlim=(0, 180), xlabel='θ / degrees', ylabel='每度的质量',
               title='NDF 的 θ 边缘分布' if col == 0 else 'VNDF 的 θ 边缘分布')
        ax.grid(alpha=.18)
        ax.tick_params(labelsize=8)
        if col == 0:
            ax.legend(fontsize=7.5, frameon=False)
    ax = fig.add_subplot(layout[2, 2])
    final_folder = folder('final', CASES_SELECTED[label], 'global_conditional')
    background = (final_folder if (final_folder / 'result.json').exists() else
                  folder('preview', CASES_SELECTED[label], 'global_conditional')) / 'render.png'
    ax.imshow(png(background))
    image_size = Image.open(background).width
    px, py = (np.array(r['pixel']) + .5) * image_size / r['diagnostic_resolution'][0] - .5
    ax.plot(px, py, '+', color='#66e7ff', ms=12, mew=1.5)
    ax.set_title('对应的相机光线', fontsize=10)
    position_text = ', '.join(f'{v:.3f}' for v in r['position'])
    ax.text(.5, -.07, f'碰撞位置 p = ({position_text})', transform=ax.transAxes,
            ha='center', va='top', fontsize=8, color='#555b65')
    ax.axis('off')
    masses = ', '.join(f'{x:.6f}' for x in d['vndf_integral'])
    fig.text(.04, .13, '每一行共用色标；三列在同一碰撞点、同一入射方向计算。NDF 保留原始面积权重，VNDF 积分为 1。', fontsize=10)
    fig.text(.04, .103, '等面积圆盘覆盖完整球面：中心为 global 平均法线；虚线为 θ=90°；外圈为 θ=180°；青色 × 为 −w。', fontsize=10, color='#555b65')
    fig.text(.04, .076, 'VNDF = max(−w·n, 0) D(n) / A(w)。conditional 使用该路径最近一次起始观测及当前 F=0 条件。', fontsize=10, color='#555b65')
    fig.text(.04, .049, f'VNDF 数值积分（local / global / conditional）：{masses}。网格：{nz} × {nphi}。', fontsize=9, color='#555b65')
    fig.text(.04, .023, '诊断路径调用生产相机、自由程及散射采样器，省略直接光采样的随机数消耗；它不与渲染器某个样本逐随机数对应。', fontsize=9, color='#555b65')
    path = OUT / f'ndf_{label}.png'
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return {'figure': str(path), 'record': str(record_path), 'record_sha256': sha(record_path),
            'pixel': r['pixel'], 'sample': r['sample'], 'collision_index': r['bounce']+1,
            'age_over_ell': r['age']/r['length_scale'], 'vndf_integrals': d['vndf_integral'],
            'tv_local_global': d['tv_local_global'], 'tv_conditional_global': tv}


def empirical_normals(label):
    source = OUT / 'probes/ensemble_final.json'
    data = load(source)
    modes = tuple(data['spec']['modes'])
    assert modes == RENDER_MODES
    bounce = 0 if label == 'small' else 1
    rows = [m['collisions'][bounce] for m in data['results']]
    paths = data['spec']['samples']
    reference = data['spec']['reference_ray']
    nz, nphi, na = data['nz'], data['nphi'], data['angle_bins']
    counts = np.array([r['normal_counts'] for r in rows]).reshape(4, nz, nphi)
    hits = np.array([r['hits'] for r in rows])
    assert (hits > 0).all() and np.array_equal(counts.sum((1, 2)), hits)
    density = counts / hits[:, None, None] / (4*np.pi/(nz*nphi))
    assert np.allclose(density.sum((1, 2))*4*np.pi/(nz*nphi), 1)
    fig = plt.figure(figsize=(19.8, 10.2))
    layout = fig.add_gridspec(2, 4, left=.045, right=.935, bottom=.23, top=.81,
                             height_ratios=[1, .70], hspace=.31, wspace=.23)
    fig.suptitle(f'实际碰撞法线分布 {"A" if bounce == 0 else "B"} · 第 {bounce+1} 次碰撞',
                 x=.04, y=.98, ha='left', fontsize=22, weight='bold')
    pixel_text = f'({reference["pixel"][0]}, {reference["pixel"][1]}) / {reference["diagnostic_resolution"][0]}²'
    fig.text(.04, .929, f'固定诊断光线：像素 {pixel_text}，原相机样本 {reference["sample"]}  |  每种 {paths:,} 条独立路径  |  σ = 0.01, ℓ = 0.025', fontsize=12)
    fig.text(.04, .89, '分布包含各方法的交点位置变化；第二次碰撞还包含各自的反射方向变化。这里比较命中后的法线统计，不是固定点解析 NDF。', fontsize=11, color='#555b65')
    frame = np.array(data['frame'])
    wi_disk = project(-np.array(reference['wi']), frame)
    norm = Normalize(0, float(density.max()))
    for i, mode in enumerate(modes):
        ax = fig.add_subplot(layout[0, i])
        im = ax.imshow(disk_image(density[i]), extent=(-1, 1, -1, 1), origin='lower',
                       cmap='magma', norm=norm, interpolation='nearest')
        ax.add_patch(Circle((0, 0), 1, fc='none', ec='#b1a7b4', lw=.6))
        ax.add_patch(Circle((0, 0), 1/np.sqrt(2), fc='none', ec='#b1a7b4', lw=.7, ls='--'))
        ax.plot(0, 0, '+', color='white', ms=5, mew=.7)
        ax.plot(*wi_disk, 'x', color='#72edff', ms=6, mew=1)
        ax.axis('off')
        ax.set_title('GPIS · sparse_3d_global' if i == 3 else mode, fontsize=13, color=COLORS[i], pad=12)
        ax.text(.5, -.06, f'有效命中 {hits[i]:,} / {paths:,}  ({hits[i]/paths:.2%})',
                ha='center', va='top', transform=ax.transAxes, fontsize=10)
    bar = fig.colorbar(im, cax=fig.add_axes([.95, .55, .012, .21]))
    bar.set_label('命中条件下的密度 / sr⁻¹', fontsize=10)
    theta_ax = fig.add_subplot(layout[1, :2])
    incidence_ax = fig.add_subplot(layout[1, 2])
    for key, xmax, ax, xlabel in [('theta_counts', 180, theta_ax, 'θ：与公共参考法线的夹角 / degrees'),
                                  ('incidence_counts', 90, incidence_ax, 'α：与实际 −w 的夹角 / degrees')]:
        width = xmax/na
        centers = (np.arange(na)+.5)*width
        for i, mode in enumerate(modes):
            p = np.array(rows[i][key])/hits[i]
            error = 1.96*np.sqrt(p*(1-p)/hits[i])/width
            ax.plot(centers, p/width, color=COLORS[i], ls=STYLES[i], lw=1.7, label=mode)
            ax.fill_between(centers, np.maximum(0, p/width-error), p/width+error, color=COLORS[i], alpha=.10)
        ax.set(xlim=(0, xmax), ylim=(0, None), xlabel=xlabel, ylabel='每度的概率')
        ax.grid(alpha=.18)
        ax.tick_params(labelsize=8)
    theta_ax.legend(fontsize=9, frameon=False)
    ax = fig.add_subplot(layout[1, 3])
    p = hits/paths
    # Wilson intervals remain meaningful when all primary paths hit.
    denom = 1+1.96**2/paths
    center = (p+1.96**2/(2*paths))/denom
    radius = 1.96*np.sqrt(p*(1-p)/paths+1.96**2/(4*paths**2))/denom
    ax.barh(np.arange(4), p, color=COLORS, height=.55, alpha=.85)
    ax.errorbar(center, np.arange(4), xerr=radius, fmt='none', ecolor='#30343b', capsize=3, lw=1)
    ax.set(yticks=np.arange(4), yticklabels=['local', 'global', 'conditional', 'GPIS'],
           xlim=(0, min(1.12, max(.1, float(p.max())*1.25))), xlabel='有效命中数 / 所有起始路径')
    ax.invert_yaxis()
    for i, probability in enumerate(p):
        ax.text(probability+.012, i, f'{probability:.2%}', va='center', fontsize=8)
    ax.grid(axis='x', alpha=.18)
    discarded = [r['distance_failed']+r['phase_failed'] for r in rows]
    fig.text(.04, .159, f'四列共用色标与世界坐标参考系；各列按有效命中数归一化，球面积分为 1。网格 {nz} × {nphi}；曲线阴影为逐角度分箱的近似 95% 采样区间。', fontsize=10)
    fig.text(.04, .126, '等面积圆盘：中心为原诊断路径第一次碰撞的 global 平均法线；虚线 θ=90°，外圈 θ=180°；青色 × 始终标记初始相机 −w。', fontsize=10, color='#555b65')
    fig.text(.04, .093, f'本次尝试中采样失败的路径数（local / global / conditional / GPIS）：{discarded}。这些路径不重试；命中率以全部 {paths:,} 条起始路径为分母。', fontsize=10, color='#555b65')
    fig.text(.04, .06, 'GPIS 每条路径使用独立的 3D 随机场，反射后保持该场不变。四种方法均调用生产自由程与镜面散射采样器；前两次碰撞不使用直接光采样或俄罗斯轮盘。', fontsize=10, color='#555b65')
    fig.text(.04, .027, '原 ndf_small.png / ndf_large.png 保留为三种解析模型在同一点、同一入射方向的 NDF / VNDF 对照；与本图的统计口径不同。', fontsize=10, color='#555b65')
    path = OUT / f'normal_samples_{label}_with_gpis.png'
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {'figure': str(path), 'source': str(source), 'source_sha256': sha(source),
            'pixel': reference['pixel'], 'reference_camera_sample': reference['sample'],
            'diagnostic_resolution': reference['diagnostic_resolution'],
            'collision_index': bounce+1, 'paths_per_mode': paths, 'hits': hits.tolist(),
            'hit_probabilities': p.tolist(), 'sampling_failures': discarded,
            'quantity': 'Empirical normal distribution conditioned on a recorded collision; not a fixed-point NDF.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['preview', 'normals', 'final', 'empirical', 'gpis'])
    args = parser.parse_args()
    if args.command == 'preview':
        preview()
    elif args.command == 'normals':
        save(OUT / 'ndf_figures.json', {label: normals(label) for label in CASES_SELECTED})
    elif args.command == 'empirical':
        save(OUT / 'normal_sample_figures.json', {label: empirical_normals(label) for label in CASES_SELECTED})
    elif args.command == 'gpis':
        save(OUT / 'figures_with_gpis.json', {
            'appearance': {label: appearance(label, include_gpis=True) for label in CASES_SELECTED},
            'empirical_normal_distributions': {label: empirical_normals(label) for label in CASES_SELECTED},
            'analytic_ndf': {label: str(OUT / f'ndf_{label}.png') for label in CASES_SELECTED},
            'validation': {key: load(OUT / path) for key, path in (
                ('matched_parameters', 'matched_parameters.json'),
                ('render_step_halving', 'gpis_step_convergence.json'),
                ('normal_step_halving', 'probes/ensemble_step_convergence.json'),
                ('ensemble_checks', 'probes/ensemble_checks.json')) if (OUT / path).exists()},
            'gpis_approximation': 'World-space 3D sparse convolution, 10 random signed impulses per cell; '
                'finite kernel support and production intersection/gradient rules. Independent fields '
                'across camera paths, one field retained along each path. Target SE covariance uses '
                'sigma=0.01 and ell=0.025; this is not an exact Gaussian-field reference.',
            'gradient_rules': {'classic_local': 'Normalized gradient of the trilinear SDF.',
                               'classic_global_and_global_conditional': 'Gradient of the trilinear SDF.',
                               'gpis': 'Realized sparse-convolution noise gradient plus the existing '
                                   'TabulatedMean forward difference of the SDF, epsilon=0.001.'}})
    else:
        save(OUT / 'figures.json', {
            'appearance': {label: appearance(label) for label in CASES_SELECTED},
            'normal_distributions': {label: normals(label) for label in CASES_SELECTED}})
