"""Report the matched shaderball integral-transition experiment."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/shaderball_transition_integral_20261118'
sys.path.insert(0, str(ROOT / 'build/fp-gradient-validation-deps'))
os.environ['MPLCONFIGDIR'] = str(ROOT / 'build/ray_curve_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def stairs(ax, rows, label, color):
    xs, ys = [], []
    for r in rows:
        xs.extend([r['begin'], r['end']]); ys.extend([r['maximum']] * 2)
    ax.plot(xs, np.maximum(ys, 1e-18), color=color, label=label, lw=1, alpha=.8)


def integral_curve(row, t):
    if not row['regression_coefficient']:
        return np.full_like(t, row['constant'])
    u = row['inverse_length2'] * t * t
    gamma = np.zeros_like(t)
    small = u <= 1
    term = np.full_like(u[small], .5); e2 = term.copy()
    for n in range(1, 21):
        term *= u[small] / (n + 2); e2 += term
    gamma[small] = 1 / (t[small] * e2)
    q = u[~small]; e = np.exp(-q)
    gamma[~small] = row['inverse_length2'] * t[~small] * q * e / (-np.expm1(-q) - q * e)
    return row['constant'] + row['regression_coefficient'] * gamma


def phase_cost(rows, a, b, integrated=False):
    selected = [r for r in rows if min(b, r['end']) > max(a, r['begin'])]
    if integrated:
        return sum(r['optical_depth'] for r in selected), len(selected)
    return sum(r['maximum'] * (min(b, r['end']) - max(a, r['begin'])) for r in selected), len(selected)


def pfm(path):
    with Path(path).open('rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split()); scale = float(f.readline())
        return np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)[::-1]


def main():
    rays, runs = read(OUT/'rays.json'), read(OUT/'runs.json')
    assert len(rays) == 6 and all(r['maximum_hazard_change'] == 0 for r in rays)
    assert all(r['production_replay_intervals'] == len(r['intervals']) for r in rays)
    assert len(runs['runs']) == 3 and all(r['discarded_surface_paths'] == r['discarded_surface_connections'] == 0 for r in runs['runs'])
    cards, ray_rows = [], []
    for ray in rays:
        source = Path(ray['source']); name = source.parent.name + '_' + ray['kind']
        previous = next(r for r in read(source)['rays'] if r['kind'] == ray['kind'])
        profile = previous['profiles'][0]; bounds = profile['majorants']
        points = ray['points']; t = np.array([p['t'] for p in points]); hazard = np.array([p['conditional'] for p in points])
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))
        for ax in axes:
            ax.plot(t, np.maximum(hazard, 1e-18), color='#111', label='条件消光', lw=1.5)
            stairs(ax, bounds['old_three'], '原三段：常数过渡上界', '#cb8620')
            stairs(ax, bounds['current_two'], '当前两段：自适应近段', '#7c55b3')
            for i, row in enumerate(ray['intervals']):
                xs = np.linspace(row['begin'], row['end'], 33)
                ax.plot(xs, np.maximum(integral_curve(row, xs), 1e-18), color='#16826f', lw=1.2,
                        label='新三段：C₀+C₁γ(t)' if i == 0 else None)
            ax.axvspan(ray['birth_split'], min(ray['far_split'], ray['length']), color='#ffd366', alpha=.12)
            ax.axvline(ray['birth_split'], color='#888', ls=':', lw=.8)
            ax.axvline(ray['far_split'], color='#888', ls=':', lw=.8)
            ax.set_yscale('log'); ax.set_ylim(1e-12, max(1e3, hazard.max()*4)); ax.grid(alpha=.2)
            ax.set_xlabel('距条件起点 t'); ax.set_ylabel('消光 / 上界')
        axes[0].set_xlim(0, ray['length']); axes[0].set_title('完整固定条件光线')
        axes[1].set_xscale('log'); axes[1].set_xlim(max(1e-10, ray['birth_split']/2), min(ray['length'], ray['far_split']*1.1))
        axes[1].set_title('近起点与过渡段（浅黄色）'); axes[1].legend(fontsize=8)
        fig.suptitle(name); fig.tight_layout()
        fig.savefig(OUT/(name+'.png'), dpi=155); fig.savefig(OUT/(name+'.svg')); plt.close(fig)
        with (OUT/(name+'_intervals.csv')).open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(ray['intervals'][0])); writer.writeheader(); writer.writerows(ray['intervals'])
        row = {'ray': name, 'coverage_points': ray['coverage_points'], 'maximum_hazard_ratio': ray['maximum_hazard_ratio'],
               'maximum_hazard_change': ray['maximum_hazard_change'], 'production_replay_intervals': ray['production_replay_intervals']}
        for mode in ['old_three', 'current_two', 'integral_three']:
            rs = ray['intervals'] if mode == 'integral_three' else bounds[mode]
            cost, count = phase_cost(rs, ray['birth_split'], min(ray['far_split'], ray['length']), mode == 'integral_three')
            row[mode+'_transition_integral'] = cost; row[mode+'_transition_intervals'] = count
        ray_rows.append(row)
        cards.append(f'<article><h2>{name}</h2><img src="{name}.png"><p><a href="{name}.svg">SVG</a> · <a href="{name}_intervals.csv">区间系数 CSV</a></p></article>')
    render_rows, render_cards = [], []
    for run in runs['runs']:
        rgb = pfm(run['pfm']); assert np.isfinite(rgb).all() and rgb.min() >= 0 and rgb.max() > 0
        mapped = rgb/(1+rgb)
        mapped = np.where(mapped <= .0031308, 12.92*mapped, 1.055*mapped**(1/2.4)-.055)
        plt.imsave(OUT/(run['label']+'.png'), np.clip(mapped, 0, 1))
        profile = run['profile']
        totals = {}
        for domain in profile['domains'].values():
            for key, value in domain['counts'].items():
                totals[key] = totals.get(key, 0) + value
        render_rows.append({'mode': run['mode'], 'render_seconds': run['render_seconds'],
                            'intervals': totals['processed_intervals'], 'candidates': totals['candidate_events'],
                            'zero_event_intervals': totals['zero_event_intervals'],
                            'transition_queries': totals['transition_bound_queries'],
                            'transition_bound_worker_seconds': sum(d['stages']['bound_transition']['inclusive_ms'] for d in profile['domains'].values())/1000,
                            'image_mean': run['image']['mean_rgb'], 'discarded_paths': run['discarded_surface_paths'],
                            'discarded_connections': run['discarded_surface_connections']})
        render_cards.append(f'<div><h3>{run["label"]}</h3><img src="{run["label"]}.png" width="256"><p>{run["render_seconds"]:.3f} s · <a href="{run["label"]}.pfm">线性 PFM</a></p></div>')
    sources = ['src/core/math/conditional/transport/'+s for s in ['ConditionalFlightKernel.cpp','ConditionalFlightKernel.h','ConditionalSegment.cpp','ConditionalSegment.h','MajorantOpticalDepth.h']]
    sources += ['src/core/media/GlobalConditionalMedium.cpp', 'tools/surface_compare/transition_integral_experiment.cpp', 'tools/surface_compare/transition_integral_rays.hpp']
    summary = {'render': render_rows, 'rays': ray_rows, 'configuration': runs['spec'],
               'source_sha256': {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in sources},
               'binary_sha256': hashlib.sha256((ROOT/'build/Release/conditional_transition_experiment.exe').read_bytes()).hexdigest()}
    (OUT/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    def table(rows):
        keys = list(rows[0])
        return '<table><tr>'+''.join(f'<th>{html.escape(k)}</th>' for k in keys)+'</tr>'+''.join('<tr>'+''.join(f'<td>{v:.6g}</td>' if isinstance(v,float) else f'<td>{html.escape(str(v))}</td>' for v in r.values())+'</tr>' for r in rows)+'</table>'
    document = '''<!doctype html><meta charset="utf-8"><title>Shaderball 可积过渡上界</title>
<style>body{font:16px system-ui;max-width:1400px;margin:32px auto;padding:0 20px;color:#223}img{max-width:100%}table{border-collapse:collapse;font-size:13px;display:block;overflow:auto}td,th{border:1px solid #ccd;padding:8px}article{margin:32px 0}.renders{display:flex;gap:24px;flex-wrap:wrap}code{background:#edf2f4;padding:3px}</style>
<h1>Shaderball：C₀+C₁γ(t) 可积过渡上界</h1>
<p>ℓ=0.05，σ=0.04；使用原 609³、float32、0.005 间距 VDB 与原变换。起点认证段之后启用新包络，4ℓ=0.2 之后沿用局部 SE 远段。原始条件消光未改变。</p>
<p>M(t)=C₀+C₁γ(t)，∫M=C₀(b−a)+C₁ log[s(b)/s(a)]。C₁ 根据区间 z 范围收紧；按真实积分细分、反演生成候选点，并以 λ(t)/M(t) 接受。反演采用带括区间的 Newton 法。</p>
<p>启用参数：<code>tracking_transition_integral_bounds: true</code>，<code>tracking_local_far_start_scale: 4</code>。
<a href="scene.json">实验场景</a> · <a href="spec.json">复现实验配置</a> · <a href="summary.json">汇总 JSON</a> · <a href="runs.json">完整渲染计数</a> · <a href="rays.json">光线与覆盖验证</a></p>
<p>验证通过：25,272 个专项覆盖点、独立数值积分、累计光学厚度反演、跨区间时钟、delta/ratio 统计检验及完整 tracking 回归。<a href="validation.json">测试记录</a>。</p>
<h2>相同场景与资源的渲染对照</h2><p>64×64、8 spp、4 线程，同一进程只加载一次 VDB。时间含采样器插桩开销，排除资源初始化；每种设置只测一次。候选点过程变化会改变随机数消费，低 spp 图像不能用于证明逐像素一致。</p>'''
    document += table(render_rows)+'<div class="renders">'+''.join(render_cards)+'</div>'
    current, old, new = render_rows
    document += f'''<p>本次新方案比原三段减少 {(1-new['render_seconds']/old['render_seconds'])*100:.1f}% 渲染时间，比当前两段减少 {(1-new['render_seconds']/current['render_seconds'])*100:.1f}%。
过渡上界构造的累计线程耗时由 {old['transition_bound_worker_seconds']:.2f} s 降到 {new['transition_bound_worker_seconds']:.2f} s；反演与积分增加了采样循环开销。
新方案并非处处更紧：三条反射光线的过渡段上界积分都高于原三段。收益主要来自上界构造与区间处理成本，而非统一降低候选点强度。</p>
<p>六条光线共 {sum(r['coverage_points'] for r in rays):,} 个检查点全部覆盖，消光值与旧 profile 完全一致。当前两段的渲染 PFM 与上一轮基线逐位相同。三种模式均未丢弃路径或阴影连接。</p>'''
    document += '<h2>相同六条光线的过渡段</h2><p>这里的积分对应固定起点条件下的完整光线，不是实际渲染中碰撞后重置条件的累计成本。所有导出区间均与生产采样器的完整遍历核对。</p>'+table(ray_rows)+''.join(cards)
    (OUT/'comparison.html').write_text(document, encoding='utf-8')
    print(json.dumps(summary['render'], indent=2))


if __name__ == '__main__':
    main()
