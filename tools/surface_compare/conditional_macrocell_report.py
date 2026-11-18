"""Reproduce the macrocell timing, slow-ray figures and HTML report."""
from conditional_optimization_report import ROOT, read, table, pfm, image_png, plt, np
import csv
import html
import json
import statistics

OUT = ROOT/'outputs/shaderball_macrocell_hybrid_20261118'


def main():
    all_rows, groups, batches = [], [], {}
    for batch in ['sweep', 'selective', 'final']:
        path = OUT/batch/'runs.json'
        if not path.exists():
            continue
        data = read(path)
        runs = batches[batch] = data['runs']
        for r in runs:
            a = pfm(r['pfm'])
            assert np.isfinite(a).all() and a.min() >= 0
            assert r['discarded_surface_paths'] == r['discarded_surface_connections'] == 0
            all_rows.append(dict(batch=batch, label=r['label'], size=r.get('macro_size', 0),
                start=r.get('macro_start', 0), gate=r.get('macro_min_events', 0),
                seed=r.get('seed', data['spec']['seed']), seconds=r['render_seconds'],
                macro_build_seconds=r.get('macro_build_seconds', 0)))
        keys = sorted({(r['macro_size'], r['macro_start'], r.get('macro_min_events', 0))
                       for r in runs if r.get('macro_bounds')})
        for size, start, gate in keys:
            selected = [r for r in runs if r.get('macro_bounds') and
                (r['macro_size'], r['macro_start'], r.get('macro_min_events', 0)) == (size, start, gate)]
            controls = [r for r in runs if not r.get('macro_bounds') and
                (batch == 'final' or r['label'].startswith(f'control_b{size}_'))]
            old = statistics.median(r['render_seconds'] for r in controls)
            new = statistics.median(r['render_seconds'] for r in selected)
            groups.append(dict(batch=batch, size=size, start=start, gate=gate,
                               control=old, hybrid=new, ratio=new/old))
    with (OUT/'iterations.csv').open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    selected = read(OUT/'selection.json')
    validation = read(OUT/'validation.json')
    final = batches['final']
    control = next(r for r in final if not r.get('macro_bounds') and r.get('slow_rays'))
    hybrid = next(r for r in final if r.get('macro_bounds') and r.get('slow_rays') and r['macro_size'] == selected['macro_size'])
    image_png(control['pfm'], OUT/'baseline.png')
    image_png(hybrid['pfm'], OUT/'hybrid.png')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, batch, title in zip(axes, ['sweep', 'selective'], ['始终使用宏块', '按预期候选数选择宏块']):
        rows = [g for g in groups if g['batch'] == batch]
        ax.bar(range(len(rows)), [g['ratio'] for g in rows],
               color=['#bc824b' if g['ratio'] > 1 else '#128779' for g in rows])
        ax.axhline(1, color='#333', ls='--')
        labels = [f"{g['size']}³\n" + (f"起点 {g['start']}ℓ" if batch == 'sweep' else f"阈值 {g['gate']}") for g in rows]
        ax.set_xticks(range(len(rows)), labels, rotation=35, ha='right')
        ax.set_title(title)
        ax.set_ylabel('耗时 / 同组当前方案耗时（越低越好）')
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'sweep.png', dpi=160)
    fig.savefig(OUT/'sweep.svg')
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric, title in zip(axes, ['median_us', 'mean_candidates', 'mean_intervals'],
                                ['单次耗时中位数 / μs', '平均候选点数', '平均区间数']):
        for j, (r, label, color) in enumerate([(control, '当前方案', '#bc824b'), (hybrid, '选择性宏块', '#128779')]):
            ax.bar(np.arange(4)+(j-.5)*.36, [v[metric] for v in r['slow_rays']], width=.36, label=label, color=color)
        ax.set_xticks(range(4), [v['id'] for v in control['slow_rays']])
        ax.set_title(title)
        ax.legend()
        ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(OUT/'slow_rays.png', dpi=160)
    plt.close(fig)
    phase_html = ''
    diagnostic = OUT/selected.get('diagnostic_directory', 'diagnostic')
    if (diagnostic/'fixed_rays.json').exists():
        base = [r for r in read(ROOT/'outputs/shaderball_tracking_optimization_20261118/diagnostic-prefix/fixed_rays.json') if r['optimized']]
        macro = [r for r in read(diagnostic/'fixed_rays.json') if r['optimized']]
        for old, new in zip(base, macro):
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            for r, label, color in [(old, '当前方案', '#bc824b'), (new, '宏块混合', '#128779')]:
                axes[0].plot([p['t'] for p in r['curve']], [max(p['majorant'], 1e-12) for p in r['curve']], label=label, color=color, lw=.9)
            axes[0].plot([p['t'] for p in new['curve']], [max(p['hazard'], 1e-12) for p in new['curve']], color='#333', lw=.8, label='条件消光')
            axes[0].set_yscale('log')
            axes[0].set_ylim(1e-6, 1e7)
            axes[0].set_title(old['id']+' 上界与实际行程')
            axes[0].set_xlabel('t')
            axes[0].legend()
            for ax, metric, title in [(axes[1], 'bound_calls', '上界调用'),
                    (axes[2], 'nulls' if old['domain'] == 'path' else 'ratio_events', 'null collision / ratio 候选')]:
                for j, (r, label, color) in enumerate([(old, '当前方案', '#bc824b'), (new, '宏块混合', '#128779')]):
                    ax.bar(np.arange(3)+(j-.5)*.36, [r['phases'][k][metric] for k in ['birth', 'transition', 'far']], width=.36, color=color, label=label)
                ax.set_xticks(range(3), ['近起点', '过渡', '远段'])
                ax.set_title(title)
                ax.legend()
            fig.tight_layout()
            fig.savefig(OUT/(old['id']+'.png'), dpi=150)
            plt.close(fig)
            phase_html += f'<img src="{old["id"]}.png">'
    rows = [[g['batch'], g['size'], g['start'], g['gate'], f"{g['control']:.4f}", f"{g['hybrid']:.4f}", f"{g['ratio']:.3f}×"] for g in groups]
    slow = []
    for r, label in [(control, '当前方案'), (hybrid, '选择性宏块')]:
        for v in r['slow_rays']:
            slow.append([label, v['id'], v['samples'], f"{v['median_us']:.2f}", f"{v['mean_candidates']:.2f}",
                f"{v['mean_intervals']:.2f}", f"{v['mean_transmittance']:.6g}", f"{v['reference']['transmittance']:.6g}",
                '通过' if v['within_six_sigma'] else '失败'])
    body = f'''<!doctype html><meta charset="utf-8"><title>Shaderball 宏块混合上界测试</title>
    <style>body{{font:16px system-ui;max-width:1250px;margin:35px auto;background:#f5f7f9;color:#243141;line-height:1.65}}article{{background:white;padding:22px;margin:18px 0;border-radius:12px}}img{{max-width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:7px;border-bottom:1px solid #ddd;text-align:right}}.scroll{{overflow:auto}}pre{{white-space:pre-wrap}}.pair{{display:flex;gap:20px}}.pair img{{width:256px;image-rendering:pixelated}}a{{color:#087d76}}</style>
    <h1>Shaderball：宏块范围 + 自适应收紧</h1>
    <article><p>{html.escape(selected['conclusion'])}</p><p>ℓ=0.05，σ=0.04；原 609³ float32 VDB、坐标变换、相机、灯光、材质与积分器不变。小图 64×64 / 8 spp，8 个物理性能核；宏块预处理与渲染分别计时。</p>
    <p><a href="selection.json">选型记录</a> · <a href="validation.json">验证</a> · <a href="iterations.csv">全部计时 CSV</a> · <a href="scene.json">实验场景</a></p></article>
    <article><h2>实现</h2><p>缓存宏块中的先验均值和方向导数范围，与局部中点范围相交，再使用原条件修正和消光上界公式。上界仍按每条光线的起点条件计算；候选点使用真实条件消光。</p>
    <p>始终使用宏块会迫使长光线逐块推进；选择性方案先计算当前上界，仅当预期候选数超过阈值时才截断至宏块边界并收紧。过渡段事件阈值 4、远段 64 保留。默认不开启宏块混合选项。</p>
    <p>首次覆盖检查发现扩展域 10³⁰ 导致宏块查询回退；已改为有限前向缓存窗口，以下性能结果均在修复后测量。</p></article>
    <article><h2>小图同批对照</h2><img src="sweep.png">{table(['批次','块边长/体素','启用距离/ℓ','宏块触发阈值','当前中位秒','混合中位秒','耗时比'], rows)}</article>
    <article><h2>相同慢光线</h2><img src="slow_rays.png">{table(['方案','光线','样本','中位μs','候选点','区间','采样T','独立积分T','检验'], slow)}
    <p>计时包含 segment 构造和跟踪；透射率按 6σ + 10⁻⁴ 检验。下方曲线比较相同输入及随机数序列，各方案实际行程可能不同。</p>{phase_html}</article>
    <article><h2>渲染图</h2><div class="pair"><div>当前方案<br><img src="baseline.png"></div><div>宏块混合<br><img src="hybrid.png"></div></div><p>共同显示映射：Reinhard + sRGB；原线性 PFM 保存在各批次目录。</p></article>
    <article><h2>验证记录</h2><pre>{html.escape(json.dumps(validation, indent=2, ensure_ascii=False))}</pre></article>'''
    high = OUT/'high-resolution/runs.json'
    if high.exists():
        runs = read(high)['runs']
        for r in runs:
            a = pfm(r['pfm'])
            x = np.maximum(0, a-.004)
            mapped = (x*(6.2*x+.5))/(x*(6.2*x+1.7)+.06)
            plt.imsave(OUT/'high-resolution'/(r['label']+'.png'), np.clip(mapped, 0, 1))
        body += '<article><h2>256×256 / 128 spp</h2>'+table(['方案','纯渲染秒','宏块构建秒'],
            [[r['label'], f"{r['render_seconds']:.3f}", f"{r['macro_build_seconds']:.3f}"] for r in runs])
        body += '<p>以上两组各完整渲染一次，未开启 profiling。16³ 方案触发 <code>conditional event cannot advance</code>，未完成 128 spp，不计入成功结果。<a href="high-resolution/execution.log">完整日志</a> · <a href="failed-macro16-spec.json">失败配置</a></p>'
        body += '<div class="pair"><div>当前方案<br><a href="high-resolution/control.png"><img src="high-resolution/control.png"></a></div><div>选择性 4³ 宏块<br><a href="high-resolution/macro4.png"><img src="high-resolution/macro4.png"></a></div></div><p>显示使用场景的 filmic 映射；原始 <a href="high-resolution/control.pfm">基线 PFM</a> 与 <a href="high-resolution/macro4.pfm">宏块 PFM</a> 均保留。</p>'
        body += '<p>独立渲染复现：</p><pre>python -B tools/surface_compare/run_conditional_benchmark.py outputs/shaderball_macrocell_hybrid_20261118/standalone-spec.json outputs/shaderball_macrocell_hybrid_20261118/recheck --performance-cores --exe tungsten</pre></article>'
    (OUT/'comparison.html').write_text(body, encoding='utf-8')
    (OUT/'groups.json').write_text(json.dumps(groups, indent=2))


if __name__ == '__main__':
    main()
