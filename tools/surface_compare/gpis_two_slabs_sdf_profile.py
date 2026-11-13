"""Plot the central ray's exact mean and illustrative ideal-GP prior samples."""
from __future__ import annotations

import json
from pathlib import Path

from figures import np, plt
from run import ROOT


def main():
    base = ROOT / 'outputs/gpis_two_slabs_compare_20261113'
    scene_path = base / 'final/global_conditional/seed20261024/scene.json'
    scene = json.loads(scene_path.read_text(encoding='utf-8'))
    process = scene['media'][0]['gaussian_process']
    assert process['mean'] == {'type': 'procedural', 'func': 'two_slabs'}
    sigma = float(process['covariance']['sigma'])
    ell = float(process['covariance']['lengthScale'])
    out = base / 'sdf_ray_profile'
    out.mkdir(parents=True, exist_ok=True)

    # Exact represented mean along the camera optical axis, measured from y=.3.
    s = np.linspace(0, .6, 601)
    y = .3 - s
    mu = np.abs(np.abs(y) - .02) - .01
    points = np.array([.27, .28, .29, .30, .31, .32, .33])
    values = np.abs(np.abs(.3 - points) - .02) - .01
    assert np.allclose(values, [0, -.01, 0, .01, 0, -.01, 0], atol=1e-14)
    # Independently evaluate the full procedural min(max(affine faces)) formula.
    xyz = np.stack([np.zeros_like(s), y, np.zeros_like(s)], axis=1)
    centers = np.array([[.18, .02, 0], [-.18, -.02, 0]])
    full = np.min(np.max(np.abs(xyz[:, None, :] - centers) - [.6, .01, .6], axis=2), axis=1)
    assert np.allclose(mu, full, atol=1e-14)

    # Illustrative unconditioned ideal-GP draws on this grid, not renderer traces.
    # Select examples by outcome to explain first passage; selection is not an MC estimate.
    seed = 20261113
    rng = np.random.default_rng(seed)
    covariance = sigma**2 * np.exp(-.5 * ((s[:, None] - s[None, :]) / ell)**2)
    jitter = sigma**2 * 1e-12
    factor = np.linalg.cholesky(covariance + jitter * np.eye(len(s)))
    selected = {}
    draw_index = 0
    center_index = lambda t: int(round(t / .001))
    while len(selected) < 3 and draw_index < 4096:
        draws = mu[:, None] + factor @ rng.standard_normal((len(s), 64))
        for k, f in enumerate(draws.T):
            negative = np.flatnonzero(f <= 0)
            hit = None if not len(negative) else int(negative[0])
            kind = None
            if hit is None:
                kind = 'pass_both'
            elif .29 < s[hit] < .33:
                kind = 'pass_first_hit_second'
            elif .24 < s[hit] <= .29 and f[center_index(.30)] > 0 and f[center_index(.32)] < 0:
                kind = 'hit_first_reenter_second'
            if kind is not None and kind not in selected:
                selected[kind] = {'field': f.copy(), 'draw_index': draw_index + k, 'hit_index': hit}
        draw_index += draws.shape[1]
    assert len(selected) == 3, 'Could not find all illustrative outcomes'
    order = ['pass_both', 'hit_first_reenter_second', 'pass_first_hit_second']
    colors = ['#17805f', '#b85d21', '#6754ad']
    names = ['示例 A：整段保持正值', '示例 B：第一层已发生碰撞', '示例 C：通过第一层、在第二层碰撞']

    fig, axes = plt.subplots(2, 1, figsize=(11.5, 9), sharex=True)
    fig.suptitle('双平面：沿光轴的 SDF / 随机场 profile', fontsize=16, y=.98)
    for ax in axes:
        ax.axvspan(.27, .29, color='#719dc6', alpha=.12)
        ax.axvspan(.31, .33, color='#719dc6', alpha=.12)
        ax.axhline(0, color='#30333a', lw=1)
        ax.set_xlim(.24, .36)
        ax.set_ylabel('场值')
        ax.grid(alpha=.18)
        ax.set_xticks(np.arange(.24, .361, .01))
    ax = axes[0]
    ax.fill_between(s, mu-sigma, mu+sigma, color='#c1d9ec', alpha=.35,
                    label='先验逐点 ±σ（非整条路径置信带）')
    ax.plot(s, mu, color='#1764a0', lw=2.5, label='均值 SDF  μ(s)')
    ax.scatter(points, values, s=25, color='#1764a0', zorder=5)
    for x, value, label in [(.28, -.01, '第一层内部\n−0.01'), (.30, .01, '间隙中心\n+0.01'), (.32, -.01, '第二层内部\n−0.01')]:
        ax.annotate(label, (x, value), xytext=(0, -30 if value < 0 else 13),
                    textcoords='offset points', ha='center', va='top' if value < 0 else 'bottom', fontsize=10)
    ax.set_ylim(-.033, .065)
    ax.set_title('确定的均值几何：两个负值低谷，中间恢复为正', loc='left', fontsize=12)
    ax.legend(loc='upper center', fontsize=9, ncol=2)

    ax = axes[1]
    ax.plot(s, mu, color='#8b939b', lw=1.3, ls=':', label='均值 μ(s)')
    for kind, color, name in zip(order, colors, names):
        record = selected[kind]
        f = record['field']
        hit = record['hit_index']
        if hit is None:
            ax.plot(s, f, color=color, lw=2, label=name)
        else:
            before = hit - 1
            crossing = s[before] + (s[hit] - s[before]) * f[before] / (f[before] - f[hit])
            ax.plot(np.r_[s[:hit], crossing], np.r_[f[:hit], 0], color=color, lw=2, label=name)
            ax.plot(np.r_[crossing, s[hit:]], np.r_[0, f[hit:]], color=color, lw=1.7, ls='--', alpha=.8)
            ax.scatter([crossing], [0], color=color, marker='x', s=65, zorder=6)
            record['first_zero_linear_interpolation'] = float(crossing)
    ax.set_ylim(-.04, .09)
    ax.set_title('相同 σ=0.02、ℓ=0.2 的理想 GP 先验示例：F(s)=μ(s)+g(s)', loc='left', fontsize=12)
    ax.legend(loc='upper center', fontsize=9, ncol=2)
    ax.set_xlabel('从介质顶边界起，沿光线前进的距离 s；方向为 y 从正到负')
    fig.text(.5, .025, '阴影：均值几何内部。×：首次零交叉；其后虚线仅展示场的延续，光线已终止。\n'
             '示例按通过/碰撞结果挑选，非渲染日志中的路径，也不是透射率估计；图中放大了 s∈[0.24,0.36]。',
             ha='center', fontsize=9, color='#485363')
    fig.tight_layout(rect=(0, .07, 1, .95))
    for ext in ['png', 'svg']:
        fig.savefig(out / f'profile.{ext}', dpi=180)
    plt.close(fig)

    columns = [s, y, mu] + [selected[k]['field'] for k in order]
    np.savetxt(out / 'profile.csv', np.stack(columns, axis=1), delimiter=',',
               header='s,y,mean_sdf,' + ','.join(order), comments='')
    metadata = {
        'scene': str(scene_path.relative_to(ROOT)),
        'ray': {'origin': [0, .3, 0], 'direction': [0, -1, 0], 'interval': [0, .6]},
        'mean_formula': 'abs(abs(0.3-s)-0.02)-0.01',
        'sigma': sigma, 'lengthscale': ell,
        'key_points': [{'s': float(x), 'mean': float(v)} for x, v in zip(points, values)],
        'examples': {'description': 'Outcome-selected, unconditioned ideal-GP prior samples on a 0.001 grid; not renderer path data or a probability estimate.',
                     'seed': seed, 'covariance_diagonal_jitter': jitter,
                     'samples': {kind: {k: v for k, v in selected[kind].items() if k != 'field'} for kind in order}},
    }
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'output': str(out), 'mean_checks': 'passed', 'samples': metadata['examples']['samples']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
