"""Reproducible figures/report for the shaderball conditional tracking sweep."""
import csv
import html
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/shaderball_tracking_optimization_20261118'
sys.path.insert(0, str(ROOT/'build/fp-gradient-validation-deps'))
os.environ['MPLCONFIGDIR'] = str(ROOT/'build/ray_curve_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = [font_manager.FontProperties(fname=str(font)).get_name(), 'DejaVu Sans']
plt.rcParams.update({'svg.fonttype': 'none', 'axes.unicode_minus': False, 'font.size': 10})


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(headers, rows):
    return '<div class="scroll"><table><tr>'+''.join(f'<th>{h}</th>' for h in headers)+'</tr>'+''.join(
        '<tr>'+''.join(f'<td>{v}</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'


def pfm(path):
    with Path(path).open('rb') as f:
        assert f.readline().strip() == b'PF'
        w, h = map(int, f.readline().split()); scale = float(f.readline())
        return np.frombuffer(f.read(), dtype='<f4' if scale < 0 else '>f4').reshape(h, w, 3)[::-1]


def image_png(source, output):
    a = pfm(source)
    assert a.shape == (64, 64, 3) and np.isfinite(a).all() and a.min() >= 0 and a.max() > 0
    mapped = a/(1+a)
    mapped = np.where(mapped <= .0031308, 12.92*mapped, 1.055*mapped**(1/2.4)-.055)
    plt.imsave(output, np.clip(mapped, 0, 1))


def main():
    selected = read(OUT/'selection.json')
    final_folder = selected.get('final_directory', 'final')
    final = read(OUT/final_folder/'runs.json')
    validation = read(OUT/'validation.json')
    diagnostic_folder = selected.get('diagnostic_directory', 'diagnostic')
    fixed = read(OUT/diagnostic_folder/'fixed_rays.json')
    fresh = read(OUT/diagnostic_folder/'slow_rays.json')
    optimized = [r for r in final['runs'] if r['mode'] == 'coupled']
    controls = [r for r in final['runs'] if r['mode'] == 'integral_three']
    times = [r['render_seconds'] for r in optimized]
    first, control = optimized[0], controls[0]
    image_png(first['pfm'], OUT/'optimized.png')
    image_png(control['pfm'], OUT/'baseline.png')
    all_rows = []
    for folder in ['baseline', 'round1', 'round2', 'round3-render', 'round4', 'round5', 'final', 'no-split-optimized', 'final-no-split', 'final-prefix']:
        file = OUT/folder/'runs.json'
        if not file.exists():
            continue
        batch = read(file)
        affinity = (OUT/folder/'host.json').exists() and read(OUT/folder/'host.json')['performance_core_affinity']
        for r in batch['runs']:
            all_rows.append(dict(batch=folder, label=r['label'], mode=r['mode'], threads=batch['spec']['threads'],
                performance_cores=affinity, seconds=r['render_seconds'], local_scale=r.get('local_scale', '-'),
                age_scale=r.get('age_scale', '-'), far_scale=r.get('far_scale', 4), far_integral=r.get('far_integral', False)))
    with (OUT/'iterations.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0])); writer.writeheader(); writer.writerows(all_rows)
    cards = []
    phase_names = ['近起点段', '过渡段', '远段']
    phase_keys = ['birth', 'transition', 'far']
    for ident in ['P1', 'P2', 'S3', 'S4']:
        rows = [r for r in fixed if r['id'] == ident]
        fig, axes = plt.subplots(2, 2, figsize=(13, 8))
        for ax in axes[0]:
            edges = [0, rows[0]['birth_split'], rows[0]['far_split'], rows[0]['length']]
            for j, color in enumerate(['#98a3b8', '#edbf57', '#77b9a9']):
                ax.axvspan(edges[j], edges[j+1], color=color, alpha=.10)
            for x, label in [(edges[1], '近段结束'), (edges[2], '4ℓ')]:
                ax.axvline(x, color='#8896a5', ls=':', lw=.8)
                ax.text(x, .98, label, transform=ax.get_xaxis_transform(), va='top', fontsize=8, rotation=90)
        for row, color, label in zip(rows, ['#b78233', '#087d76'], ['原积分上界', '最终方案']):
            curve = row['curve']; t = np.array([p['t'] for p in curve])
            hazard = np.array([p['hazard'] for p in curve]); bound = np.array([p['majorant'] for p in curve])
            for ax in axes[0]:
                ax.plot(t, np.maximum(bound, 1e-15), color=color, label=label, lw=1)
            axes[0, 0].plot(t, np.maximum(hazard, 1e-15), color='#30343b', lw=.8, alpha=.65)
            axes[0, 1].plot(t, np.maximum(hazard, 1e-15), color='#30343b', lw=.8, alpha=.65)
            for phase, name in enumerate(phase_names):
                key = phase_keys[phase]; x = phase + (-.19 if not row['optimized'] else .19)
                for ax, metric in [(axes[1, 0], 'bound_calls'), (axes[1, 1], 'nulls' if row['domain'] == 'path' else 'ratio_events')]:
                    value = row['phases'][key][metric]
                    ax.bar(x, value, width=.36, color=color, label=label if phase == 0 else None)
                    ax.text(x, value, str(value), ha='center', va='bottom', fontsize=8)
        for ax in axes[0]:
            ax.set_yscale('log'); ax.set_ylim(1e-7, max(1e4, max(p['majorant'] for r in rows for p in r['curve'])*2))
            ax.set_xlabel('t（世界距离）'); ax.set_ylabel('消光系数 / 上界'); ax.grid(alpha=.2); ax.legend()
        axes[0, 0].set_xlim(0, rows[0]['length']); axes[0, 0].set_title('各次实际行程：黑线为条件消光')
        axes[0, 1].set_xscale('log'); axes[0, 1].set_xlim(min(r['birth_split'] for r in rows)/2, .22)
        axes[0, 1].set_title('近起点及过渡段放大')
        event_title = 'null collision 次数' if rows[0]['domain'] == 'path' else 'ratio tracking 候选点数'
        for ax, title in zip(axes[1], ['上界计算次数', event_title]):
            ax.set_xticks(range(3), phase_names); ax.set_title(title); ax.grid(axis='y', alpha=.2)
            ax.set_yscale('symlog', linthresh=1)
            ax.set_ylim(0, max(2, max(bar.get_height() for bar in ax.patches)*2))
            ax.legend()
        fig.suptitle(f'{ident} · 固定起点和初始条件，原随机序列重放'); fig.tight_layout()
        fig.savefig(OUT/f'{ident}.png', dpi=150); fig.savefig(OUT/f'{ident}.svg'); plt.close(fig)
        count_rows = [[('最终方案' if r['optimized'] else '原积分方案'), r['candidates'], r['intervals'],
                       f"{r['visited']:.6g}", r['coverage_samples'], f"{r['maximum_hazard_bound_ratio']:.6g}"] for r in rows]
        cards.append(f'<article><h2>{ident}</h2><img src="{ident}.png">'+table(
            ['方案', '候选点', '区间', '实际行程', '额外覆盖点', '最大 λ/M'], count_rows)+f'<p><a href="{ident}.svg">SVG</a></p></article>')
    mc_rows = []
    for run in final['runs']:
        for r in run.get('slow_rays', []):
            mc_rows.append([run['label'], r['id'], r['samples'], f"{r['median_us']:.2f}",
                f"{r['mean_candidates']:.2f}", f"{r['mean_intervals']:.2f}", f"{r['mean_transmittance']:.6g}",
                f"{r['standard_error']:.3g}", f"{r['reference']['transmittance']:.6g}", '通过' if r['within_six_sigma'] else '未通过'])
    render_rows = [[r['label'], r['mode'], r.get('seed', final['spec']['seed']), f"{r['render_seconds']:.4f}",
                    f"{r['image']['mean_rgb']:.6f}", r['discarded_surface_paths']+r['discarded_surface_connections']] for r in final['runs']]
    iteration_rows = [[r['batch'], r['label'], r['threads'], '性能核' if r['performance_cores'] else '系统调度',
                       f"{r['seconds']:.4f}"] for r in all_rows]
    aggregate_rows = []
    for phase, name in zip(phase_keys, phase_names):
        data = [domain['phases'][phase] for domain in fresh['aggregate'].values()]
        aggregate_rows.append([name, sum(p['bound_calls'] for p in data), sum(p['candidates'] for p in data),
            sum(p['nulls'] for p in data), sum(p['ratio_events'] for p in data),
            f"{sum(p['bound_us'] for p in data)/1000:.2f}", f"{sum(p['clock_us'] for p in data)/1000:.2f}"])
    fresh_rows = [[r['id'], r['domain'], f"{r['replay_median_us']:.2f}", len(r['events']), len(r['leaves']),
                   f"{r['validation']['maximum_hazard_bound_ratio']:.6f}"] for r in fresh['rays']]
    passed = max(times) <= 1
    page = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>Shaderball 条件消光上界优化</title>
<style>body{{font:16px/1.65 system-ui,sans-serif;background:#f3f5f8;color:#203040;max-width:1280px;margin:30px auto;padding:0 20px}}
article,.box{{background:white;border-radius:12px;padding:22px;margin:20px 0}}h1,h2{{line-height:1.3}}img{{max-width:100%}}
.renders{{display:flex;gap:30px;flex-wrap:wrap}}.render img{{width:320px;image-rendering:pixelated}}table{{border-collapse:collapse;font-size:14px;width:100%}}
td,th{{padding:8px;border-bottom:1px solid #dce2e9;text-align:left;white-space:nowrap}}.scroll{{overflow:auto}}.equation,pre{{background:#edf3f6;padding:14px;overflow:auto}}
.good{{color:#087d76}}code{{font-family:Consolas,monospace}}a{{color:#1769aa}}</style>
<h1>Shaderball：条件消光上界与分段优化</h1>
<div class="box"><h2 class="good">{'达到' if passed else '尚未达到'} 1 秒目标</h2>
<p>64 × 64、8 spp、{final['spec']['threads']} 个 CPU 渲染线程，固定性能核。最终 {len(times)} 次完整渲染：
中位数 <b>{np.median(times):.4f} s</b>，范围 <b>{min(times):.4f}–{max(times):.4f} s</b>。
同批原积分方案中位数 {np.median([r['render_seconds'] for r in controls]):.4f} s。
计时包含所有 8 spp 的路径追踪；不包含场景加载、写图和详细 profiling。加载耗时 {final['initialization_seconds']:.2f} s 单列。</p>
<p>正式 <code>tungsten.exe</code> 独立渲染：<b>{validation['standalone_renderer_seconds']:.6f} s</b>，输出 PFM 与验收运行逐位一致。</p>
<p>ℓ = 0.05、σ = 0.04；使用原 609³ float32 VDB、0.005 体素间距及原坐标变换。
条件消光公式、完整场景与样本数不变。新分段会改变随机数的消费顺序，因此与旧方案的小图噪声不逐像素对应。</p>
<div class="renders"><div class="render"><h3>最终方案</h3><img src="optimized.png"></div><div class="render"><h3>原积分方案</h3><img src="baseline.png"></div></div>
<p><a href="scene.json">最终场景</a> · <a href="{selected.get('final_spec','final-spec.json')}">复现实验配置</a> · <a href="{final_folder}/runs.json">完整计时数据</a> · <a href="validation.json">验证结果</a></p>
<p>从仓库根目录复现单张正式渲染（8 线程，性能核 affinity 记录在输出目录）：</p>
<pre>python -B tools/surface_compare/run_conditional_benchmark.py outputs/shaderball_tracking_optimization_20261118/standalone-spec.json outputs/shaderball_tracking_optimization_20261118/recheck_render --exe tungsten --performance-cores</pre></div>
<article><h2>条件消光的特点</h2>
<div class="equation">λ(t) = φ(z(t))/Φ(z(t)) × [κ(t) φ(z′(t)/κ(t)) − z′(t) Φ(−z′(t)/κ(t))]</div>
<p>其中 z(t) = μ(t)/s(t)，μ、s 分别为条件场值的均值和标准差；κ 为给定场值等于零后的方向导数标准差除以 s。
近起点 s(t) ≈ σt²/(√2ℓ²)。对于 F(0)=0、正向起点斜率 b&gt;0，z(t) ≈ √2ℓ²b/(σt)，
故消光呈 t⁻² exp(−c/t²) 型衰减；掠射 b 很小时峰会移向起点，并变得很窄。
分别使用宽泛的全局均值和导数界会丢失这一抵消关系。</p>
<p>旧积分界 M(t)=C₀+C₁s′/s 的慢光线开销主要来自 C₀：全局导数界在小方差下被放大。
取消积分阈值二分会造成大量 null collision；本次通过改善证书和初始区间宽度解决，而不是截断采样。</p></article>
<article><h2>最终分段与候选表示</h2>
<p>最终参数：<code>{html.escape(json.dumps(selected['settings'],ensure_ascii=False))}</code>。</p>
<p><b>最终保留过渡段事件阈值 4。</b>新上界关闭该阈值后，第一个种子为 1.075 秒；第二个种子报错
<code>conditional event cannot advance</code>，没有完成渲染。因此“完全关闭二分”未通过本次验收。
<a href="no-split-optimized/execution.log">失败运行日志</a>与成功结果分开保存。</p>
<p>近起点段：表面起点沿用局部曲率证书；F(0)&gt;0 时改用 Lipschitz 证书，可跨体素边界。
令 Δv=F(0)−m(0)，Δk=b−m′(0)，a=ℓ⁻²，则</p>
<div class="equation">|μ(t)−F(0)| ≤ (L+|Δk|)t + ½at²(|Δv|+t|Δk|).</div>
<p>选取右端 h，使上式不超过 F(0)/2；在整个 [0,h] 使用解析指数包络的最大值。
过渡段：步长按起点距离增长，早期使用局部导数证书，之后使用中点 Lipschitz 证书；两者均形成条件均值和方向导数的区间，再计算分段常数上界。
远段：从配置的倍数 ℓ 开始使用局部 SE 尾界，尾部残差被保守保留，并复用有效证书。</p>
<p>具体地，若区间内 μ∈[f₋,f₊]、μ′≥k₋、s∈[s₋,s₊]、γ=s′/s∈[γ₋,γ₊]、κ≤κ₊，则采用保守的区间乘积界：</p>
<div class="equation">z₋=min(f₋/s₋, f₋/s₊)，u₋=k₋−max(γ₋f₊,γ₊f₊)，d₋=min(u₋/s₋,u₋/s₊)<br>
基础界 = φ(z₋)/Φ(z₋) × [κ₊φ(d₋/κ₊) − d₋Φ(−d₋/κ₊)]。</div>
<p>安全性来自逆 Mills 比随 z 递减，以及负部高斯期望随 d 递减、随 κ 递增。
局部导数证书和条件中点包络保留了更多起点抵消关系；紧度改善主要来自这些输入范围。
实现中，z≤−8 时以 1−z 代替逆 Mills 比；d&gt;4κ 时以 κφ(d/κ) 代替负部期望，均是向上放宽。
最后加入浮点保护裕量，并与仍有效的父区间上界取较小值。
有证书覆盖的近起点段整段推进，过渡段和远段按各自上界分区。</p>
<p>还测试了可直接积分的远段函数包络：</p>
<div class="equation">M(t)=C E(z₀−β(t−a))； E(z)=√(2/π)e⁻ᶻ²⁄²（z≥0），E(z)=√(2/π)−z（z&lt;0）。</div>
<p>正侧来自 Φ(z)≥1/2；负侧来自逆 Mills 比的导数大于 −1。
其积分分别为 erfc 差和二次多项式，可直接反演；每个候选点后更新 Lipschitz 锚点。
测试中它没有胜过最终常数方案，保留为实验选项。下表列出所有实际运行结果。</p></article>
<article><details><summary>远段函数包络的系数与积分</summary>
<p>令 Q≥2 为尾界的起始距离除以方向相关长度 ℓ，Δv=F(0)−m(0)，Δk=b−m′(0)。
以下残差界在此后的整条尾段有效：</p>
<pre>ε₀ = (|Δv| + Qℓ|Δk|) exp(−Q²/2)
ε₁ = (Q|Δv|/ℓ + (Q²−1)|Δk|) exp(−Q²/2)
s₋ = σ sqrt(1−(1+Q²)exp(−Q²))，s₊ = σ
γ₊ = Q³ exp(−Q²) / [ℓ(1−(1+Q²)exp(−Q²))]</pre>
<p>在 [a,b]，以 m(a) 为 Lipschitz 锚点，f₊=m(a)+L(b−a)+ε₀，取</p>
<pre>z₀ = min((m(a)−ε₀)/s₋, (m(a)−ε₀)/s₊)，β=L/s₋
u₋ = −G−ε₁−γ₊ max(f₊,0)
C = [(σ/ℓ)φ(u₋ℓ/σ) − u₋Φ(−u₋ℓ/σ)]/s₋</pre>
<p>L、G 分别是均值的 Lipschitz 界和梯度范数界；实现包含浮点保护。
z(t)=z₀−β(t−a) 非负的区间，包络积分为 C/β × [erfc(z(b)/√2)−erfc(z(a)/√2)]；
负侧为 C(b−a)[√(2/π)−z(a)+β(b−a)/2]。跨过零点时相加，β=0 时退化为常数界。</p>
</details></article>
<article><h2>最终重复渲染</h2>{table(['运行','方案','种子','秒','平均 RGB','丢弃路径/连接'],render_rows)}
<p>时间阈值检查：{passed}。完整验证信息见 <a href="validation.json">validation.json</a>。</p></article>
<article><h2>原四条慢光线：采样与独立积分</h2>
<p>每次样本包含 segment 构造与跟踪；时间为无 profiling 单线程中位数。
参考透射率由条件消光在体素边界及对数距离网格上独立积分得到，并提高积分精度再次核对。
路径样本为 0/1 存活，阴影样本为 ratio tracking 权重；检验容差为 6 个标准误加 10⁻⁴。</p>
{table(['运行','光线','样本数','中位 μs','平均候选点','平均区间','采样 T','标准误','积分 T','检验'],mc_rows)}</article>
{''.join(cards)}
<article><h2>最终方案整图热点</h2><p>下表来自单独的详细 profiling 运行。耗时是各 worker 的累计毫秒，包含诊断开销，不能与验收用的无 profiling 墙钟秒数直接比较。</p>
{table(['分段','上界调用','候选点','null collision','ratio 候选','上界累计 ms','时钟累计 ms'],aggregate_rows)}
<p>从最终方案自身的慢光线池重新筛选，并逐条重放核对：</p>
{table(['光线','类型','重放中位 μs','候选点','区间','最大 λ/M'],fresh_rows)}</article>
<article><h2>迭代记录</h2><p>不同批次可能受到核调度和系统负载影响；最终性能结论采用同批固定性能核测量。
<a href="iterations.csv">所有参数与时间 CSV</a> · <a href="{diagnostic_folder}/fixed_rays.json">固定光线详细记录</a> · <a href="{diagnostic_folder}/slow_rays.json">新方案整图慢光线</a></p>
{table(['批次','配置','线程','核调度','秒'],iteration_rows)}</article></html>'''
    (OUT/'comparison.html').write_text(page, encoding='utf-8')
    print(f'Report: {OUT / "comparison.html"}; worst final render {max(times):.4f} s')


if __name__ == '__main__':
    main()
