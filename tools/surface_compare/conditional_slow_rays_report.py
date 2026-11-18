"""Visualize actual expensive flights under the integral transition envelope."""
import csv
import hashlib
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'outputs/shaderball_integral_slow_rays_20261118'
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
plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none', 'axes.unicode_minus': False})
PHASES = ['birth', 'transition', 'far']
NAMES = ['起点段', '过渡段', '远段']
COLORS = ['#8794b4', '#e7a336', '#4ca796']
KINDS = {'null': ('null collision', '#c44b43', 'x'), 'accepted': ('真实碰撞', '#19814d', '*'),
         'ratio': ('ratio 候选点', '#8054b2', '|')}
REASONS = {'replay_time_1': '候选池复测耗时第 1', 'replay_time_2': '候选池复测耗时第 2',
           'most_bound_calls': '上界查询次数最多', 'most_null_collisions': 'null collision 最多',
           'most_ratio_events': 'ratio 候选点最多'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def table(headers, rows):
    return '<div class="scroll"><table><thead><tr>' + ''.join(f'<th>{v}</th>' for v in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join(f'<td>{v}</td>' for v in row) + '</tr>' for row in rows) + '</tbody></table></div>'


def csv_write(path, rows):
    if not rows:
        return
    with Path(path).open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def shade(ax, r, xmax=None):
    xmax = r['visited'] if xmax is None else xmax
    boundaries = [0, r['birth_split'], r['far_split'], r['visited']]
    for i, name in enumerate(NAMES):
        a, b = max(0, boundaries[i]), min(xmax, boundaries[i+1])
        if b > a:
            ax.axvspan(a, b, color=COLORS[i], alpha=.12)
    for x, label, color in [(r['birth_split'], '起点段结束', COLORS[0]), (r['far_split'], '4ℓ', COLORS[2])]:
        if 0 < x < xmax:
            ax.axvline(x, color=color, lw=.9, ls='--')
            ax.text(x, 1.01, label, transform=ax.get_xaxis_transform(), fontsize=8, color=color)
    ax.set_xlim(0, xmax)
    ax.set_xlabel('t（从本段光线起点起算，世界单位）')
    ax.grid(alpha=.18)


def phase_bars(ax, phases, counts=False):
    x = np.arange(3)
    if counts:
        keys = ['bound_calls', 'intervals', 'nulls', 'ratio_events']
        labels = ['上界查询', '处理区间', 'null collision', 'ratio 候选点']
        for i, (key, label) in enumerate(zip(keys, labels)):
            values = [phases[p][key] for p in PHASES]
            bars = ax.bar(x+(i-1.5)*.2, values, width=.19, label=label)
            for bar, value in zip(bars, values):
                if value:
                    ax.text(bar.get_x()+bar.get_width()/2, value, str(value), ha='center', va='bottom', fontsize=7)
        ax.set_yscale('symlog', linthresh=1)
        ax.set_ylabel('次数（对数尺度；保留 0）')
    else:
        bottom = np.zeros(3)
        for key, label, color in [('bound_us', '上界计算', '#3a6b98'), ('hazard_us', '候选点消光计算', '#cb6048'),
                                  ('clock_us', '光学深度推进 / 反解', '#9263a4'), ('other', '其余推进与记录开销', '#c0c8cf')]:
            values = np.array([max(0, phases[p]['total_us']-sum(phases[p][k] for k in ['bound_us', 'hazard_us', 'clock_us']))
                               if key == 'other' else phases[p][key] for p in PHASES])
            ax.bar(x, values, bottom=bottom, color=color, label=label)
            bottom += values
        ax.set_ylabel('累计 worker 耗时 / μs')
    ax.set_xticks(x, NAMES)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=.18)


def ray_figure(r):
    points = r['curve']
    t = np.array([p['t'] for p in points])
    hazard = np.array([p['hazard'] for p in points])
    bound = np.array([p['majorant'] for p in points])
    fig, axes = plt.subplots(4, 2, figsize=(15, 15), constrained_layout=True)
    title = f"{r['id']} · {'路径光线' if r['domain']=='path' else '阴影光线'} · " + ' / '.join(REASONS[x] for x in r['selection'])
    fig.suptitle(title + f"\n实际距离 {r['visited']:.6g}；复测中位数 {r['replay_median_us']:.1f} μs；bounce={r['bounce']}", fontsize=14)
    for ax, zoom in [(axes[0, 0], False), (axes[0, 1], True)]:
        ax.plot(t, bound, color='#398754', lw=1.1, label='实际使用的上界 M(t)')
        ax.plot(t, hazard, color='#191919', lw=1.2, label='条件消光 λ(t)')
        for kind, (label, color, marker) in KINDS.items():
            selected = [e for e in r['events'] if e['kind'] == kind]
            if selected:
                ax.scatter([e['t'] for e in selected], [e['hazard'] for e in selected],
                           marker=marker, c=color, s=38 if kind=='accepted' else 12, label=label, zorder=5)
        ax.set_yscale('symlog', linthresh=.01)
        ax.set_ylabel('消光 / 世界单位⁻¹')
        end = min(r['visited'], r['far_split']) if zoom else r['visited']
        shade(ax, r, end)
        if zoom:
            positives = t[t > 0]
            ax.set_xscale('log')
            ax.set_xlim(max(positives.min()*.8, 1e-16), max(end, positives.min()*2))
        ax.set_title('近段放大（横轴 log t）' if zoom else '实际经过范围：上界、消光与候选点')
        ax.legend(fontsize=8)
    ax = axes[1, 0]
    ax.plot(t, [p['mean'] for p in points], color='#245c8c', label='原 SDF 均值 m(t)')
    ax.plot(t, [p['conditional_mean'] for p in points], color='#875da8', label='条件均值 μ(t)')
    ax.axhline(0, color='black', lw=.8)
    ax.axhspan(-r['sigma'], r['sigma'], color='#777', alpha=.08, label='±σ')
    ax.set_ylabel('场值'); ax.legend(fontsize=8); shade(ax, r)
    ax = axes[1, 1]
    for phase, color, name in zip(PHASES, COLORS, NAMES):
        queries = sorted([q for q in r['bound_queries'] if q['phase']==phase], key=lambda q:q['begin'])
        if queries:
            x = [q['begin'] for q in queries]; y = np.cumsum([q['us'] for q in queries])
            ax.step(x+[r['visited']], list(y)+[y[-1]], where='post', label=name, color=color)
    ax.set_title('上界计算累计耗时（父区间、细分重算均计入）')
    ax.set_ylabel('累计 μs'); shade(ax, r); ax.legend(fontsize=8)
    log_min = max(min(p['t'] for p in points if p['t'] > 0)*.8, 1e-16)
    ax.set_xscale('log'); ax.set_xlim(log_min, r['visited'])
    ax.set_xlabel('t（log 尺度；t=0 的预计算已计入累计值）')
    ax = axes[2, 0]
    bins = np.geomspace(log_min, r['visited'], 65)
    for kind, (label, color, _) in KINDS.items():
        values = [e['t'] for e in r['events'] if e['kind']==kind]
        if values:
            ax.hist(values, bins=bins, histtype='step', lw=1.5, label=label, color=color)
    ax.set_ylabel('每个 log t bin 的事件数'); ax.set_title('候选事件沿 t 的分布'); shade(ax, r); ax.legend(fontsize=8)
    ax.set_xscale('log'); ax.set_xlim(log_min, r['visited'])
    ax.set_xlabel('t（log 尺度；等宽 log bin）')
    ax = axes[2, 1]
    events = r['events']
    if events:
        ax.scatter([e['t'] for e in events], [e['hazard']/e['majorant'] for e in events], s=9,
                   color='#b75f43', label='λ(t) / M(t)')
    ax.set_ylim(-.02, 1.02); ax.set_ylabel('接受概率 / ratio 单点衰减比例'); shade(ax, r); ax.legend(fontsize=8)
    ax.set_title('低 λ/M 表示候选点多数只产生小作用')
    ax.set_xscale('log'); ax.set_xlim(log_min, r['visited'])
    phase_bars(axes[3, 0], r['phases']); phase_bars(axes[3, 1], r['phases'], True)
    axes[3, 0].set_title('分段耗时：起点段包含构造时预计算')
    axes[3, 1].set_title('分段工作量：上界调用包括父区间与重算')
    fig.savefig(OUT/f"{r['id']}.png", dpi=150)
    fig.savefig(OUT/f"{r['id']}.svg")
    plt.close(fig)


def main():
    data = read(OUT/'slow_rays.json'); runs = read(OUT/'runs.json'); run = runs['runs'][0]
    rays = data['rays']
    reference = ROOT/'outputs/shaderball_transition_integral_20261118/integral_three.pfm'
    equal_image = (OUT/'integral_three.pfm').read_bytes() == reference.read_bytes()
    assert equal_image, 'Diagnostic recording changed the rendered image'
    assert run['discarded_surface_paths'] == run['discarded_surface_connections'] == 0
    assert all(r['validation']['replay_exact'] for r in rays)
    summary_rows, aggregate_rows, cards = [], [], []
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for i, domain in enumerate(['path', 'shadow']):
        a = data['aggregate'][domain]
        phase_bars(axes[i, 0], a['phases']); phase_bars(axes[i, 1], a['phases'], True)
        axes[i, 0].set_title(('路径' if i==0 else '阴影')+f"：{a['flights']:,} 条光线的分段耗时")
        axes[i, 1].set_title('整次渲染的分段工作量')
        total_time = sum(p['total_us'] for p in a['phases'].values())
        for phase, name in zip(PHASES, NAMES):
            p = a['phases'][phase]
            aggregate_rows.append([domain, name, f"{p['total_us']/1000:.1f} ({p['total_us']/total_time:.1%})",
                f"{p['bound_us']/1000:.1f}", f"{p['clock_us']/1000:.1f}", f"{p['bound_calls']:,}", f"{p['intervals']:,}",
                f"{p['nulls']:,}", f"{p['ratio_events']:,}", f"{p['empty_intervals']/max(1,p['intervals']):.1%}"])
    fig.savefig(OUT/'aggregate.png', dpi=150); fig.savefig(OUT/'aggregate.svg'); plt.close(fig)
    for r in rays:
        ray_figure(r)
        for key in ['curve', 'events', 'leaves', 'bound_queries']:
            csv_write(OUT/f"{r['id']}-{key}.csv", r[key])
        p = r['phases']; bounds = sum(x['bound_calls'] for x in p.values()); nulls = sum(x['nulls'] for x in p.values())
        ratio = sum(x['ratio_events'] for x in p.values())
        slow_phase = max(PHASES, key=lambda k:p[k]['bound_us']); many_phase = max(PHASES, key=lambda k:p[k]['nulls']+p[k]['ratio_events'])
        summary_rows.append([f"<a href=\"#{r['id']}\">{r['id']}</a>", '路径' if r['domain']=='path' else '阴影',
            '、'.join(REASONS[k] for k in r['selection']), f"{r['replay_median_us']:.1f}", f'{bounds:,}', nulls, ratio,
            NAMES[PHASES.index(slow_phase)], NAMES[PHASES.index(many_phase)]])
        rows = []
        for phase, name in zip(PHASES, NAMES):
            q = p[phase]
            rows.append([name, f"{q['total_us']:.1f}", f"{q['bound_us']:.1f}", f"{q['clock_us']:.1f}",
                q['bound_calls'], q['splits'], q['intervals'], q['nulls'], q['accepted'], q['ratio_events']])
        ratio_text = ('null collision' if r['domain']=='path' else 'ratio 候选点')
        description = f"上界耗时主要在<b>{NAMES[PHASES.index(slow_phase)]}</b>；{ratio_text}主要在<b>{NAMES[PHASES.index(many_phase)]}</b>。"
        events = r['events']
        acceptance_median = float(np.median([e['hazard']/e['majorant'] for e in events]))
        transition = [v for v in r['leaves'] if v['phase']=='transition']
        peak = max(transition, key=lambda v:v['maximum'])
        c0_fraction = peak['C0']/peak['maximum'] if peak['maximum'] else 0
        description += (f"候选点 λ/M 的中位数为 <b>{acceptance_median:.3%}</b>。过渡段最大的区间起始上界在 t={peak['begin']:.6g}，"
                        f"M={peak['maximum']:.5g}，其中 C₀={peak['C0']:.5g}（占 {c0_fraction:.2%}）。")
        events_link = f'<a href="{r["id"]}-events.csv">事件 CSV</a>' if r['events'] else ''
        cards.append(f'''<section id="{r['id']}"><h2>{r['id']} · {'路径' if r['domain']=='path' else '阴影'}光线</h2>
          <p>{' / '.join(REASONS[k] for k in r['selection'])}。{description}</p>
          <p>复测中位数 <b>{r['replay_median_us']:.1f} μs</b>；原渲染记录 {r['setup_us']+r['track_us']:.1f} μs。
          实际走到 t={r['visited']:.8g}，原最大长度 {r['length']:.8g}；起点段结束 {r['birth_split']:.6g}；远段从 {r['far_split']:.6g} 开始。
          {'真实碰撞终止' if r['domain']=='path' and r['result']==0 else '到达端点' if r['visited']==r['length'] else 'ratio 权重归零或 roulette 终止'}。</p>
          {table(['段','总耗时 μs','上界 μs','推进 / 反解 μs','上界调用','细分','区间','null','真实碰撞','ratio 点'], rows)}
          <a href="{r['id']}.svg"><img loading="lazy" src="{r['id']}.png" alt="{r['id']} 分段曲线与成本"></a>
          <p><a href="{r['id']}.json">完整光线、随机数与轨迹 JSON</a> · <a href="{r['id']}-curve.csv">曲线 CSV</a> ·
          <a href="{r['id']}-leaves.csv">实际区间 CSV</a> · <a href="{r['id']}-bound_queries.csv">上界调用 CSV</a> · {events_link}</p></section>''')
    fig = plt.figure(figsize=(10, 8)); ax = fig.add_subplot(111, projection='3d')
    for r in rays:
        origin, direction = np.array(r['origin']), np.array(r['effective_direction'])
        xyz = origin+np.linspace(0,r['visited'],200)[:,None]*direction
        ax.plot(*xyz.T, label=f"{r['id']} ({r['domain']})")
        ax.scatter(*origin, s=22)
    ax.set(xlabel='x',ylabel='y',zlabel='z',title='所选光线在世界坐标中的实际轨迹（不含碰撞后的延长线）')
    ax.legend();fig.savefig(OUT/'rays_world.png',dpi=150);plt.close(fig)
    combined = {p:{key:sum(data['aggregate'][d]['phases'][p][key] for d in ['path','shadow'])
                    for key in data['aggregate']['path']['phases'][p]} for p in PHASES}
    total_bound = sum(p['bound_us'] for p in combined.values())
    validation = {'image_bitwise_equal_to_previous_integral_render':equal_image,
        'discarded_paths':run['discarded_surface_paths'],'discarded_connections':run['discarded_surface_connections'],
        'selected_rays':len(rays),'candidate_pool':len(data['candidate_pool']),
        'replay_exact_all':True,'coverage_samples':sum(r['validation']['coverage_samples'] for r in rays),
        'maximum_hazard_bound_ratio':max(r['validation']['maximum_hazard_bound_ratio'] for r in rays),
        'combined_phases':combined,
        'tracking_regression_passed': 'tracking: carried optical depth' in (OUT/'tracking-regression.log').read_bytes().decode(
            'utf-16' if (OUT/'tracking-regression.log').read_bytes().startswith(b'\xff\xfe') else 'utf-8', errors='replace'),
        'sources_sha256':{str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in
            [ROOT/'src/core/math/ConditionalRayProfile.hpp',ROOT/'tools/surface_compare/conditional_slow_rays.hpp',
             Path(__file__),ROOT/'build/Release/conditional_slow_rays.exe']}}
    (OUT/'validation.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2),encoding='utf-8')
    conclusion = '；'.join(f"{name}占上界计算耗时 {combined[p]['bound_us']/total_bound:.1%}" for p,name in zip(PHASES,NAMES))
    null_total = sum(p['nulls'] for p in combined.values())
    ratio_total = sum(p['ratio_events'] for p in combined.values())
    bottleneck = (f"过渡段产生 {combined['transition']['nulls']:,} 次路径 null collision（占 {combined['transition']['nulls']/null_total:.1%}），"
                  f"以及 {combined['transition']['ratio_events']:,} 个阴影 ratio 候选点（占 {combined['transition']['ratio_events']/ratio_total:.1%}）。"
                  f"远段的 {combined['far']['intervals']:,} 个区间中 {combined['far']['empty_intervals']/combined['far']['intervals']:.1%} 没有候选事件；"
                  "它仍有大量空区间的上界计算成本。所选最慢光线的热点更集中在刚离开起点段的过渡区间，"
                  "那里较大的 C₀ 同时增加了细分重算和候选点数量。")
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
    <title>Shaderball · 积分过渡上界的慢光线</title><style>
    body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;background:#f3f5f7;color:#23303e;margin:0 auto;max-width:1480px;padding:26px}}
    h1{{font-size:29px}}h2{{font-size:23px}}section,.intro{{background:white;padding:24px;margin:22px 0;border-radius:12px}}
    a{{color:#215f97}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border-bottom:1px solid #dbe0e5;padding:8px;text-align:right;white-space:nowrap}}th:first-child,td:first-child{{text-align:left}}th{{background:#edf2f6}}.scroll{{overflow-x:auto}}.note{{color:#586574}}code{{background:#edf1f5;padding:2px 5px}}
    </style><h1>Shaderball：积分过渡上界的慢光线</h1>
    <div class="intro"><p>ℓ=0.05，σ=0.04；原大光源与 VDB；64×64、8 spp、4 线程。过渡段 M(t)=C₀+C₁γ(t)，远段从 4ℓ 开始。</p>
    <p><b>{conclusion}。</b></p>
    <p>{bottleneck}</p>
    <p>从真实渲染的路径和阴影光线中，分别保留原始耗时、上界调用次数、null / ratio 事件数排名前 8 的光线；合并得到 {len(data['candidate_pool'])} 条候选。
    每条用同一随机数序列独立重放 7 次，从候选池中选择复测耗时前 2，以及上界调用和事件数量最多的代表，共 {len(rays)} 条。
    “复测较慢”只针对该候选池，不代表对所有光线做了重复计时排序。</p>
    <p class="note">计时覆盖 ConditionalSegment 的构造与 tracking。分段表是原渲染中的累计 worker 时间，包含诊断开销及可能的线程调度等待；不能直接等同墙钟渲染耗时。
    复测时间用于甄别单次尖峰。起点段时间包含构造时的 birth certificate；各段上界次数包含实际查询、父区间和细分重算。
    绘图止于真实碰撞、roulette / 零权重终止或端点。阴影使用 ratio tracking，候选点不计作路径 null collision。</p>
    <p><b>验证：</b>图像与之前积分方案逐字节一致；计数与原 profiler 对齐；候选池全部重放 7 次，端点、权重、随机数用量、区间及候选点数完全一致；
    {validation['coverage_samples']:,} 个曲线采样点无上界违例，max λ/M={validation['maximum_hazard_bound_ratio']:.6g}。
    tracking 回归检查通过。采样覆盖检查是数值验证，不替代上界证明。</p>
    <p><a href="../shaderball_transition_integral_20261118/comparison.html">积分方案与旧方案对照</a> ·
    <a href="slow_rays.json">全部原始记录</a> · <a href="runs.json">渲染 profiler</a> · <a href="validation.json">验证</a> · <a href="spec.json">重现参数</a></p></div>
    <section><h2>整次渲染：哪一段更贵？</h2>
    {table(['域','段','总时间 ms (占域内分段时间)','上界 ms','推进 / 反解 ms','上界调用','区间','null collision','ratio 点','零事件区间'],aggregate_rows)}
    <a href="aggregate.svg"><img src="aggregate.png" alt="整体分段耗时与工作量"></a></section>
    <section><h2>所选慢光线</h2>{table(['光线','域','入选原因','复测 μs','上界调用','null','ratio','上界最贵段','事件最多段'],summary_rows)}
    <details><summary>世界坐标中的实际光线轨迹</summary><img src="rays_world.png" alt="世界坐标光线"></details></section>
    {''.join(cards)}</html>'''
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps({'report':str(OUT/'comparison.html'),'validation':validation},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
