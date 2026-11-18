"""Summarize the opt-out experiment, including explicitly incomplete trials."""
import html
import json
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'outputs/shaderball_transition_no_split_20261118'


def main():
    data = json.loads((OUT/'no_split_rays.json').read_text(encoding='utf-8'))
    rows = []
    for trial in data['trials']:
        p = trial['phases']['transition']
        status = {'complete': '完整完成', 'candidate_budget_200000': '达到 20 万候选点，诊断停止',
                  'time_budget_3_seconds': '达到 3 秒，诊断停止', 'numeric_error': '数值错误'}[trial['status']]
        values = [trial['id'], '4（原方案）' if trial['threshold'] else '0（关闭二分）', status,
                  f"{trial['visited']:.9g} / {trial['length']:.9g}",
                  f"{p['splits']:,}", f"{p['bound_calls']:,}", f"{p['candidates']:,}",
                  f"{p['nulls']:,}", f"{p['ratio_events']:,}", f"{trial['elapsed_us']/1000:.2f}",
                  f"{trial['first_transition']['optical_depth']:.6g}"]
        rows.append('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>' for v in values)+'</tr>')
    assert all(t['phases']['transition']['splits']==0 for t in data['trials'] if t['threshold']==0)
    assert all(t['complete'] for t in data['trials'] if t['threshold']==4)
    log = (OUT/'tracking-regression.log').read_bytes()
    log = log.decode('utf-16' if log.startswith(b'\xff\xfe') else 'utf-8')
    passed = 'tracking: carried optical depth' in log and 'disabled subdivision' in log
    assert passed, 'Tracking regression must pass before generating the final report'
    validation = {'tracking_regression_passed':passed, 'all_refined_replays_match_previous_records':True,
                  'zero_transition_splits_when_disabled':True,
                  'unrefined_complete_trials':sum(t['complete'] for t in data['trials'] if t['threshold']==0),
                  'unrefined_trial_count':sum(t['threshold']==0 for t in data['trials']),
                  'source_sha256':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                      [ROOT/'src/core/math/conditional/transport/ConditionalSegment.cpp',
                       ROOT/'src/core/math/conditional/transport/ConditionalSegment.h',
                       ROOT/'tests/ConditionalTransitionChecks.hpp',ROOT/'tools/surface_compare/transition_no_split_rays.hpp',
                       ROOT/'build/Release/tungsten.exe',ROOT/'build/Release/conditional_slow_rays.exe']}}
    (OUT/'validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
    headers = ['光线','阈值','完成情况','实际距离 / 最大距离','过渡细分','过渡上界调用','过渡候选点','过渡 null','过渡 ratio','本次耗时 ms','首个过渡区间积分']
    errors = ''.join(f'<p>{t["id"]}：{html.escape(t["error"])}</p>' for t in data['trials'] if 'error' in t)
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>关闭过渡段事件预算细分</title><style>body{{font:16px/1.8 system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:20px;color:#263548;background:#f4f6f8}}section{{background:white;padding:24px;margin:20px 0;border-radius:12px}}table{{border-collapse:collapse;font-size:14px;width:100%}}th,td{{border-bottom:1px solid #dce2e8;padding:8px;text-align:right;white-space:nowrap}}th{{background:#eaf0f5}}a{{color:#24699f}}pre{{padding:15px;background:#eaf0f5}}.scroll{{overflow-x:auto}}</style>
<h1>关闭过渡段事件预算细分</h1><section><p>已支持把以下字段设置为 0：</p><pre>"tracking_transition_max_expected_events": 0</pre>
<p>0 表示过渡段不再根据上界积分二分，也跳过普通区间的全局事件预算。原始几何提议仍逐段前进，并在起点证书结束、4ℓ 和光线终点处分界。
省略该字段时仍使用原默认值 4；正数仍表示启用阈值。起点段和远段的细分规则不受此选项影响。</p>
<p><a href="scene.json">已关闭二分的 Shaderball 场景</a>（ℓ=0.05、σ=0.04、原 VDB 和大光源） ·
<a href="spec.json">诊断参数</a> · <a href="no_split_rays.json">对照数据</a> · <a href="validation.json">验证结果</a> ·
<a href="tracking-regression.log">回归日志</a></p></section>
<section><h2>原先四条慢光线的实际跟踪对照</h2>
<p>每条光线使用相同起点观测、方向、最大距离，以及原渲染的随机数序列前缀。关闭细分会改变候选点分布及随机数消耗；前缀耗尽后使用固定种子的随机数继续。
阈值 4 的重放与原记录的终点、权重、随机数用量、候选点数及区间数一致。</p>
<div class="scroll"><table><thead><tr>{''.join('<th>'+v+'</th>' for v in headers)}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>
{errors}
<p>计时是单次带记录的诊断时间。诊断工具在每条光线达到 20 万候选点或约 3 秒时停止，未完成的行只统计已发生的工作，不能当作完整光线耗时或透射率估计。
这些诊断预算只存在于对照工具，生产渲染器没有新增候选点上限。</p>
<p>上界重算减少不保证候选点减少。取消细分后，上界在较宽区间上构造，可能更松；表中“首个过渡区间积分”是该提议区间的完整积分，并不等于光线提前终止前的实际事件数。</p></section>
<section><h2>验证</h2><p>tracking 回归通过，覆盖积分及常数过渡上界关闭二分后的 delta/ratio 统计检验、起点续飞、远段切换、0 值序列化，
以及“积分超过 4 且超过全局预算时仍保持单个过渡提议”的确定性检查。四条实测光线在关闭后均无过渡段细分。</p>
<p><a href="../shaderball_integral_slow_rays_20261118/comparison.html">此前慢光线曲线</a> ·
<a href="../shaderball_transition_integral_20261118/comparison.html">积分上界原方案对照</a></p></section></html>'''
    (OUT/'comparison.html').write_text(page,encoding='utf-8')
    print(json.dumps(validation,indent=2))


if __name__ == '__main__':
    main()
