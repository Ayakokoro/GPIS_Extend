"""Validate and summarize the causes limiting conditional tracking intervals."""
import argparse
import html
import json
import math
from pathlib import Path


LIMITS = ("step_cap", "age_cap", "fine_boundary", "macro_boundary", "birth",
          "segment_end", "event_threshold", "two_constants")
PHASES = ("initial", "proposed", "processed")
AGES = ("q_lt1", "q_1_4", "q_4_16", "q_ge16")
FIELDS = ("intervals", "planned_length_ell", "traversed_length_ell",
          "expected_candidates", "candidates", "zero_candidate_intervals")


def total(values):
    return {key: sum(v.get(key, 0) for v in values) for key in FIELDS}


def group(partitions, phase, ages):
    limits = {limit: total([p[phase][age].get(limit, {})
                           for p in partitions for age in ages]) for limit in LIMITS}
    return {"limits": limits, "total": total(list(limits.values()))}


def validate_profile(profile):
    for name, domain in profile["domains"].items():
        partition = domain["interval_partition"]
        counts = domain["counts"]
        for phase in PHASES:
            values = group([partition], phase, AGES)
            expected = counts["processed_intervals" if phase == "processed" else "proposed_intervals"]
            if values["total"]["intervals"] != expected:
                raise ValueError(f"{name}/{phase}: interval count does not reconcile")
            for row in values["limits"].values():
                if any(not math.isfinite(v) or v < 0 for v in row.values()):
                    raise ValueError(f"{name}/{phase}: invalid aggregate")
                if row["traversed_length_ell"] > row["planned_length_ell"] * (1 + 1e-7):
                    raise ValueError(f"{name}/{phase}: traversed length exceeds plan")
        processed = group([partition], "processed", AGES)["total"]
        if processed["candidates"] != counts["candidate_events"]:
            raise ValueError(f"{name}: candidate count does not reconcile")
        if processed["zero_candidate_intervals"] != counts["zero_event_intervals"]:
            raise ValueError(f"{name}: zero-event count does not reconcile")
        if not counts["proposed_intervals"] <= processed["intervals"] <= counts["proposed_intervals"] + counts["splits"]:
            raise ValueError(f"{name}: invalid subdivision accounting")
        if not counts["splits"]:
            for age in AGES:
                for limit in LIMITS:
                    a = partition["proposed"][age].get(limit, {}).get("intervals", 0)
                    b = partition["processed"][age].get(limit, {}).get("intervals", 0)
                    if a != b:
                        raise ValueError(f"{name}: unsplit proposals changed classification")


def summarize_run(run):
    profile = run["profile"]
    validate_profile(profile)
    groups = {}
    for name in ("all", *profile["domains"]):
        domains = list(profile["domains"].values()) if name == "all" else [profile["domains"][name]]
        partitions = [domain["interval_partition"] for domain in domains]
        groups[name] = {
            "counts": {key: sum(d["counts"][key] for d in domains) for key in domains[0]["counts"]},
            "phases": {phase: {"all": group(partitions, phase, AGES),
                               **{age: group(partitions, phase, [age]) for age in AGES}}
                       for phase in PHASES},
        }
    return {"label": run["label"], "resolution": run["resolution"], "spp": run["spp"],
            "settings": run["settings"], "instrumented_render_seconds": run["render_seconds"],
            "groups": groups}


DOCUMENT = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>条件化追踪：区间长度受限原因</title>
<style>
body{font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif;background:#f5f7fb;color:#172337;margin:0}
main{max-width:1180px;margin:auto;padding:30px}h1{font-size:26px;margin:0 0 8px}h2{font-size:19px}
p{max-width:1050px}.muted{color:#54657a}.box{background:white;border:1px solid #dce3ec;border-radius:9px;padding:20px;margin:18px 0}
select{padding:8px;border:1px solid #b9c7d9;border-radius:5px;background:white;margin:0 20px 12px 5px}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}td,th{padding:9px 12px;border-bottom:1px solid #e4e9f1;text-align:right}th:first-child,td:first-child{text-align:left}
thead{background:#edf2f8}a{color:#2265bb}.bar{height:28px;display:flex;background:#edf2f8;border-radius:4px;overflow:hidden;margin:6px 0 14px}
.legend{display:flex;gap:16px;flex-wrap:wrap;font-size:13px}.swatch{display:inline-block;width:10px;height:10px;margin-right:5px}
.cards{display:flex;gap:20px;flex-wrap:wrap}.card{flex:1;min-width:300px}.number{font-size:24px;font-weight:650}
code{background:#edf2f8;padding:2px 5px;border-radius:3px}.scroll{overflow:auto}details{margin:16px 0}summary{cursor:pointer}
</style><main>
<h1>条件化追踪：区间长度受限原因</h1>
<p class="muted">@@SUBTITLE@@ · @@CONDITIONS@@ · @@PROGRESS@@ · 仅增加诊断统计</p>
<p class="muted">@@SETTINGS@@</p>
<p>统计实际执行的区间划分，区分固定步长、距起点距离限制、细体素或宏块边界，以及事件阈值细分。
下方占比是<strong>区间数量占比</strong>；可同时查看覆盖距离与候选点，不能把数量占比直接当成耗时占比。</p>
<div class="box"><label>射线<select id="domain"><option value="all">全部</option><option value="path">主路径</option><option value="shadow">阴影</option></select></label>
<label>阶段<select id="phase"><option value="processed">最终已处理区间</option><option value="proposed">边界截断后、事件细分前</option><option value="initial">初始长度（边界截断前）</option></select></label>
<label>起点距离 q=d/ℓ<select id="age"><option value="all">全部距离</option><option value="q_lt1">0 ≤ q &lt; 1</option><option value="q_1_4">1 ≤ q &lt; 4</option><option value="q_4_16">4 ≤ q &lt; 16</option><option value="q_ge16">q ≥ 16</option></select></label>
<div id="bars"></div><div id="legend" class="legend"></div></div>
<div class="box scroll"><table id="counts"></table></div>
<div class="box scroll"><h2>覆盖距离与候选点</h2><p class="muted">长度以当前射线的相关长度 ℓ 为单位。实际覆盖距离计入碰撞导致的提前停止。</p><table id="work"></table></div>
<div class="box"><h2>整体工作量</h2><div id="cards" class="cards"></div></div>
<div class="box"><h2>口径与验证</h2>
<p>每个阶段中，一个区间只属于一个类别。初始阶段先执行起点特殊区间规则或步长 / 距离限制；随后实际截短它的几何边界覆盖此前类别；任何事件阈值细分后的子区间都归入“事件阈值”，包括右子区间。未走到的待处理区间不计入最终统计。</p>
<p>距离分桶按区间起点的 q=d/ℓ，不额外拆分跨桶区间。“细体素边界”也包含网格夹取边界。
宏块拒绝由宏块接受阈值控制，它与 <code>maxExpectedEvents</code> 导致的对半细分单独统计；宏块拒绝后可能出现多个细体素区间。</p>
<p>脚本核对初始 / 提议数量与 <code>proposed_intervals</code> 一致、最终数量与 <code>processed_intervals</code> 一致、候选点与零候选区间数量一致，并核对与已有普通构建结果的像素一致性。带统计的渲染时间不用于普通构建的速度比较。</p>
<p id="validation"></p><p>数据：<a href="@@JSON@@">汇总 JSON</a> · <a href="@@SOURCE@@">原始报告</a></p></div>
</main><script id="data" type="application/json">@@DATA@@</script><script>
const data=JSON.parse(document.getElementById('data').textContent),keys=Object.keys(data.cases);
const names={step_cap:'固定步长上限',age_cap:'距起点距离限制',fine_boundary:'细体素边界',macro_boundary:'宏块边界',birth:'起点特殊区间',segment_end:'射线段终点',event_threshold:'事件阈值细分',two_constants:'双常数模式'};
const colors=['#2563eb','#22a2b9','#df8c32','#8b5ac7','#728092','#adb6c4','#db4056','#4d9671'];
const labels={step_default:'基线：8 体素 / 阈值 4',block32_far64:'32 体素 / 远处阈值 64'};
const n=x=>x.toLocaleString('zh-CN',{maximumFractionDigits:0}),pct=(x,y)=>y?(100*x/y).toFixed(3)+'%':'—',f=x=>x.toLocaleString('zh-CN',{maximumFractionDigits:5});
function render(){
const domain=document.getElementById('domain').value,phase=document.getElementById('phase').value,age=document.getElementById('age').value;
const rows=keys.map(k=>data.cases[k].groups[domain].phases[phase][age]);
document.getElementById('bars').innerHTML=keys.map((k,i)=>`<strong>${labels[k]||k}</strong> · ${n(rows[i].total.intervals)} 个区间<div class="bar">${Object.keys(names).map((l,j)=>`<div title="${names[l]} ${pct(rows[i].limits[l].intervals,rows[i].total.intervals)}" style="width:${rows[i].total.intervals?100*rows[i].limits[l].intervals/rows[i].total.intervals:0}%;background:${colors[j]}"></div>`).join('')}</div>`).join('');
document.getElementById('legend').innerHTML=Object.keys(names).map((k,i)=>`<span><i class="swatch" style="background:${colors[i]}"></i>${names[k]}</span>`).join('');
document.getElementById('counts').innerHTML=`<thead><tr><th>限制来源</th>${keys.map(k=>`<th>${labels[k]||k}：数量</th><th>占比</th>`).join('')}</tr></thead><tbody>`+Object.keys(names).map(l=>`<tr><td>${names[l]}</td>${rows.map(r=>`<td>${n(r.limits[l].intervals)}</td><td>${pct(r.limits[l].intervals,r.total.intervals)}</td>`).join('')}</tr>`).join('')+`<tr><th>合计</th>${rows.map(r=>`<th>${n(r.total.intervals)}</th><th>${r.total.intervals?'100%':'—'}</th>`).join('')}</tr></tbody>`;
document.getElementById('work').innerHTML=`<thead><tr><th>来源 / 配置</th><th>平均提议长度 / ℓ</th><th>实际覆盖距离占比</th><th>候选点数</th><th>零候选区间占比</th></tr></thead><tbody>`+Object.keys(names).flatMap(l=>keys.map((k,i)=>{const v=rows[i].limits[l];return `<tr><td>${names[l]} · ${labels[k]||k}</td><td>${v.intervals?f(v.planned_length_ell/v.intervals):'—'}</td><td>${phase==='processed'?pct(v.traversed_length_ell,rows[i].total.traversed_length_ell):'—'}</td><td>${phase==='processed'?n(v.candidates):'—'}</td><td>${phase==='processed'?pct(v.zero_candidate_intervals,v.intervals):'—'}</td></tr>`})).join('')+'</tbody>';
document.getElementById('cards').innerHTML=keys.map(k=>{const c=data.cases[k].groups[domain].counts;return `<div class="card"><strong>${labels[k]||k}</strong><div class="number">${n(c.processed_intervals)} 个区间</div><div>候选点 ${n(c.candidate_events)} · 对半细分 ${n(c.splits)} 次</div><div>宏块接受 ${n(c.macro_accepted)} · 拒绝 ${n(c.macro_rejected)}</div><div>上界收紧 ${n(c.tightened_intervals)} 次</div></div>`}).join('');
document.getElementById('validation').textContent=keys.map(k=>`${labels[k]||k}：计数核对通过，像素完全一致 ${data.cases[k].parity_exact===true?'通过':'尚未验证'}`).join('；');
}
document.querySelectorAll('select').forEach(e=>e.addEventListener('change',render));render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summary = {"source_report": str(args.report.resolve()), "configuration": report["configuration"],
               "binary_sha256": report["binary_sha256"], "reference_report": report.get("reference_report"),
               "source_scene": report["source"],
               "base_medium": next(m for m in report["scene_snapshot"]["media"] if m["type"] == "global_conditional"),
               "age_bins": "Interval start distance from conditioning birth, divided by directional ell; [0,1), [1,4), [4,16), [16,infinity)",
               "attribution": "One cause per visited interval; any refined descendant is event_threshold. Counts are not timing percentages.",
               "cases": {}}
    for run in report["runs"]:
        if not run["profile_enabled"]:
            continue
        case = run["case"]
        if case in summary["cases"]:
            raise ValueError("Select a report containing one profile per case")
        result = summarize_run(run)
        parity = report.get("parity", {}).get(case + "_plain_1_vs_" + run["label"])
        result["parity_exact"] = parity["exact"] if parity else None
        if result["parity_exact"] is False:
            raise ValueError("Instrumentation changed the image")
        summary["cases"][case] = result
    if not summary["cases"]:
        raise ValueError("No completed instrumented run")
    summary["expected_cases"] = len(report.get("cases", summary["cases"]))
    summary["complete"] = len(summary["cases"]) == summary["expected_cases"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    json_file = args.output.with_suffix(".json")
    json_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    import os
    config = summary["configuration"]
    document = DOCUMENT.replace("@@SUBTITLE@@", f"{config['size']} × {config['size']} · {config['spp']} spp")
    document = document.replace("@@PROGRESS@@", f"已完成 {len(summary['cases'])}/{summary['expected_cases']} 组")
    affinity = config.get("affinity_mask")
    conditions = (f"固定种子 {config['seed']} · {config['threads']} 线程，"
                  f"亲和性 {hex(affinity) if affinity else '未固定'}")
    document = document.replace("@@CONDITIONS@@", html.escape(conditions))
    medium = summary["base_medium"]
    settings = (f"共同设置：tracking_step={medium.get('tracking_step', 0)}，"
                f"ageScale={medium.get('tracking_age_scale', 0.25)}，"
                f"maxExpectedEvents={medium.get('tracking_max_expected_events', 64)}，"
                f"tightenExpectedEvents={medium.get('tracking_tighten_expected_events', 4)}；"
                "各配置的宏块大小与远处阈值见图例。")
    document = document.replace("@@SETTINGS@@", html.escape(settings))
    document = document.replace("@@JSON@@", html.escape(json_file.name))
    document = document.replace("@@SOURCE@@", html.escape(os.path.relpath(args.report.resolve(), args.output.parent.resolve()).replace('\\', '/')))
    document = document.replace("@@DATA@@", json.dumps(summary, ensure_ascii=False).replace("</", "<\\/"))
    args.output.with_suffix(".html").write_text(document, encoding="utf-8")
    for case, result in summary["cases"].items():
        values = result["groups"]["all"]["phases"]["processed"]["all"]
        print(json.dumps({"case": case, "intervals": values["total"]["intervals"],
                          "percent": {k: 100 * v["intervals"] / values["total"]["intervals"]
                                      for k, v in values["limits"].items()},
                          "parity_exact": result["parity_exact"]}))


if __name__ == "__main__":
    main()
