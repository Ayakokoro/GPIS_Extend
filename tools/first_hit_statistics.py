"""Validate first-hit CSV/JSON counters and make a self-contained heatmap report.

Uses only Python's standard library. No package installation or web assets.
"""
import argparse
import collections
import csv
import heapq
import json
import math
from pathlib import Path


PRECHECK_COUNTERS = (
    "precheck_calls", "precheck_survived", "precheck_hit",
    "precheck_marginal_hits", "precheck_with_extra_nodes", "precheck_nanoseconds",
)


def precheck_work(counters):
    calls = counters.get("precheck_calls", 0)
    survived = counters.get("precheck_survived", 0)
    hit = counters.get("precheck_hit", 0)
    decisions = survived + hit
    elapsed_ms = counters.get("precheck_nanoseconds", 0) / 1e6
    return {
        "calls": calls,
        "survived": survived,
        "hit": hit,
        "marginal_hits": counters.get("precheck_marginal_hits", 0),
        "decisions": decisions,
        "undecided": calls - decisions,
        "decision_fraction": decisions / calls if calls else 0,
        "decisions_with_extra_nodes": counters.get("precheck_with_extra_nodes", 0),
        "total_ms": elapsed_ms,
        "mean_ms_per_call": elapsed_ms / calls if calls else 0,
    }


def qmc_work(counters):
    new = counters.get("qmc_points", 0)
    reused = counters.get("qmc_reused_points", 0)
    requested = new + reused
    return {
        "new_points": new,
        "reused_points": reused,
        "requested_points": requested,
        "preparations": counters.get("qmc_preparations", 0),
        "saved_fraction": reused / requested if requested else 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--csv", type=Path, help="Override the CSV path stored in the summary")
    parser.add_argument("--scene", type=Path, help="Scene JSON: preserves unsampled pixels in the heatmap")
    parser.add_argument("--threads", type=int, help="Worker count used by this captured render (metadata)")
    parser.add_argument("--wall-seconds", type=float, help="Renderer-reported wall time (metadata)")
    parser.add_argument("--output", type=Path, help="Output HTML (defaults beside the summary)")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    precheck_available = all(key in summary["sum_per_flight_statistics"] for key in PRECHECK_COUNTERS)
    csv_path = args.csv or Path(summary["csv_path"])
    totals = collections.Counter()
    histograms = {name: collections.Counter() for name in summary["histograms"]}
    pixels = {}
    counts = collections.Counter()
    groups = collections.defaultdict(collections.Counter)
    expensive = []
    seeds = set()
    with csv_path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            counts["flights"] += 1
            outcome = row["outcome"]
            counts[outcome] += 1
            seeds.add(int(row["scene_seed"]))
            for key in summary["sum_per_flight_statistics"]:
                value = row.get(key, 0) if key in ("qmc_reused_points", "qmc_preparations", *PRECHECK_COUNTERS) else row[key]
                totals[key] += int(value)
            for key, histogram in histograms.items():
                histogram[int(row[key])] += 1
            x, y = int(row["pixel_x"]), int(row["pixel_y"])
            pixel = pixels.setdefault((x, y), collections.Counter())
            if outcome == "failed":
                pixel["failed"] += 1
                continue
            extra = int(row["unique_added_nodes"]) > 0
            without_orthant = int(row["orthant_calls"]) == 0
            precheck_decisions = int(row.get("precheck_survived", 0)) + int(row.get("precheck_hit", 0))
            counts["extra" if extra else "no_extra"] += 1
            counts["without_orthant"] += without_orthant
            counts["with_precheck_decision"] += precheck_decisions > 0
            pixel["flights"] += 1
            pixel["extra"] += extra
            pixel["hits"] += outcome == "hit"
            pixel["without_orthant"] += without_orthant
            pixel["precheck_nanoseconds"] += int(row.get("precheck_nanoseconds", 0))
            pixel["max_active"] = max(pixel["max_active"], int(row["max_active_nodes"]))
            pixel["max_added"] = max(pixel["max_added"], int(row["unique_added_nodes"]))
            pixel["orthant_ms"] += float(row["orthant_ms"])
            pixel["multivariate_ms"] += float(row["multivariate_ms"])
            pixel["qmc_ms"] += float(row["qmc_ms"])
            group = groups["with_extra_nodes" if extra else "without_extra_nodes"]
            group["flights"] += 1
            group["hits"] += outcome == "hit"
            group["without_orthant"] += without_orthant
            group["with_precheck_decision"] += precheck_decisions > 0
            group["orthant_ms"] += float(row["orthant_ms"])
            group["multivariate_ms"] += float(row["multivariate_ms"])
            group["qmc_ms"] += float(row["qmc_ms"])
            group["flight_ms"] += float(row["flight_ms"])
            for key in ("qmc_points", "qmc_reused_points", "qmc_preparations", *PRECHECK_COUNTERS):
                group[key] += int(row.get(key, 0))
            entry = (float(row["orthant_ms"]), int(row["flight_id"]), row)
            if len(expensive) < 10:
                heapq.heappush(expensive, entry)
            elif entry[:2] > expensive[0][:2]:
                heapq.heapreplace(expensive, entry)

    def check(condition, message):
        if not condition:
            raise ValueError("CSV/summary mismatch: " + message)

    check(not summary["csv_write_failed"], "renderer reported a CSV write failure")
    check(counts["flights"] == summary["counts"]["flights"], "flight count")
    for outcome, key in (("hit", "hits"), ("miss", "misses"), ("failed", "failed"),
                         ("extra", "with_extra_nodes"), ("no_extra", "without_extra_nodes")):
        check(counts[outcome] == summary["counts"][key], key)
    for key, total in totals.items():
        check(total == summary["sum_per_flight_statistics"][key], key)
    for key, histogram in histograms.items():
        check(dict(histogram) == {int(k): v for k, v in summary["histograms"][key].items()}, key)
    precheck = precheck_work(totals)
    if precheck_available:
        check(precheck["undecided"] >= 0, "precheck decisions exceed calls")
        check(precheck["marginal_hits"] <= precheck["hit"], "precheck marginal hits exceed hit decisions")
        check(precheck["decisions_with_extra_nodes"] <= precheck["decisions"],
              "precheck extra-node decisions exceed all decisions")
    for key, group in groups.items():
        group["mean_orthant_ms"] = group["orthant_ms"] / group["flights"]
        group["mean_multivariate_ms"] = group["multivariate_ms"] / group["flights"]
        group["mean_flight_ms"] = group["flight_ms"] / group["flights"]

    width = max((p[0] for p in pixels), default=0) + 1
    height = max((p[1] for p in pixels), default=0) + 1
    spp = None
    if args.scene:
        scene = json.loads(args.scene.read_text(encoding="utf-8-sig"))
        scene_width, scene_height = scene["camera"]["resolution"]
        spp = scene.get("renderer", {}).get("spp")
        check(width <= scene_width and height <= scene_height, "pixel outside scene image")
        width, height = scene_width, scene_height
    image = [None] * (width * height)
    for (x, y), p in pixels.items():
        if not p["flights"]:
            continue
        image[y * width + x] = {
            "flights": p["flights"], "failed": p["failed"],
            "extra_fraction": p["extra"] / p["flights"],
            "hit_fraction": p["hits"] / p["flights"],
            "without_orthant_fraction": p["without_orthant"] / p["flights"],
            "mean_precheck_ms": p["precheck_nanoseconds"] / 1e6 / p["flights"],
            "max_active": p["max_active"], "max_added": p["max_added"],
            "mean_orthant_ms": p["orthant_ms"] / p["flights"],
            "mean_multivariate_ms": p["multivariate_ms"] / p["flights"],
            "mean_qmc_ms": p["qmc_ms"] / p["flights"],
        }

    def quantile(histogram, fraction):
        target = max(1, math.ceil(sum(histogram.values()) * fraction))
        cumulative = 0
        for value, count in sorted(histogram.items()):
            cumulative += count
            if cumulative >= target:
                return value
        return 0

    analysis = {
        "csv_validated": True, "resolution": [width, height], "scene_seeds": sorted(seeds),
        "spp": spp, "worker_threads": args.threads, "renderer_wall_seconds": args.wall_seconds,
        "groups": groups,
        "survival_precheck": {
            **precheck,
            "counters_available": precheck_available,
            "enabled": summary.get("settings", {}).get("survival_precheck"),
            "completed_flights_with_decision": counts["with_precheck_decision"],
            "completed_flights_without_orthant": counts["without_orthant"],
            "completed_flights_without_orthant_fraction": counts["without_orthant"] / max(1, summary["counts"]["completed"]),
            "groups": {key: precheck_work(group) for key, group in groups.items()},
        },
        "qmc_incremental": {
            **qmc_work(totals),
            "reuse_counters_available": "qmc_reused_points" in summary["sum_per_flight_statistics"],
            "groups": {key: qmc_work(group) for key, group in groups.items()},
        },
        "node_quantiles_all_flights": {
            name: {f"p{p}": quantile(histogram, p / 100) for p in (50, 90, 95, 99, 100)}
            for name, histogram in histograms.items()
        },
        "top_orthant_time_rays": [row for _, _, row in sorted(expensive, reverse=True)],
    }
    output = args.output or args.summary.with_name("first-hit-report.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    classification = summary["classification"]
    total = summary["counts"]["completed"]
    table = "".join(
        f"<tr><td>{label}</td><td>{classification[key]['completed']:,}</td>"
        f"<td>{classification[key]['completed'] / max(1, total):.2%}</td>"
        f"<td>{classification[key]['hits']:,}</td><td>{classification[key]['misses']:,}</td></tr>"
        for key, label in (("without_extra_nodes", "无需额外内部节点"), ("with_extra_nodes", "需要额外内部节点"))
    )
    timing = "".join(
        f"<tr><td>{label}</td><td>{summary['timing'][key]['calls']:,}</td>"
        f"<td>{summary['timing'][key]['total_ms'] / 1000:.4f} s</td>"
        f"<td>{summary['timing'][key]['mean_ms_per_call']:.5f} ms</td></tr>"
        for key, label in (("orthant", "全部全正概率调用（含一、二维）"),
                           ("multivariate", "实际维数 ≥ 3 的全正概率调用"),
                           ("qmc", "其中：QMC 积分循环"), ("pair", "二维符号筛选"))
    )
    if precheck_available:
        timing += (
            f"<tr><td>生存上下界预判（部分包含二维符号计算）</td><td>{precheck['calls']:,}</td>"
            f"<td>{precheck['total_ms'] / 1000:.4f} s</td>"
            f"<td>{precheck['mean_ms_per_call']:.5f} ms</td></tr>"
        )
    top = "".join(
        f"<tr><td>({row['pixel_x']}, {row['pixel_y']})</td><td>{row['sample_index']}</td>"
        f"<td>{row['outcome']}</td><td>{row['unique_added_nodes']}</td>"
        f"<td>{row['max_active_nodes']}</td><td>{float(row['orthant_ms']):.3f}</td></tr>"
        for _, _, row in sorted(expensive, reverse=True)
    )
    document = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>First-hit GPIS 光线统计</title><style>
body{font:16px/1.6 system-ui,sans-serif;color:#142638;background:#f4f6fa;margin:0;padding:28px;max-width:1250px;margin:auto}
h1{margin:0 0 10px}h2{font-size:21px;margin:28px 0 10px}p{max-width:1000px}
table{border-collapse:collapse;background:white;width:100%;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:9px 15px;border-bottom:1px solid #dce3ec}th{background:#e4ebf4}
.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:20px}.panel{background:white;padding:16px;border-radius:10px}
canvas{width:100%;image-rendering:pixelated;aspect-ratio:1;display:block;background:#ddd}.legend{height:12px;background:linear-gradient(to right,rgb(22,33,76),rgb(45,155,160),rgb(250,220,90));margin-top:12px}
.range{display:flex;justify-content:space-between;font-size:13px}#hover{position:sticky;bottom:0;background:#142638;color:white;padding:12px;border-radius:8px;white-space:pre-wrap;font-variant-numeric:tabular-nums}
pre{overflow:auto;background:#fff;padding:16px}small{color:#526174}@media(max-width:700px){.grid{grid-template-columns:1fr}}
</style><h1>First-hit GPIS 光线统计</h1>
<p>只统计进入实验介质的首段光线。两个端点不计入“额外内部节点”。所有距离二分前缀共用一条 flight 的统计。
鼠标移到图中可读取像素数据；灰色表示没有完成的介质光线，坐标从 0 开始。</p>
<p id="facts"></p><table><tr><th>类别</th><th>光线数</th><th>占已完成光线</th><th>命中</th><th>未命中</th></tr>__CLASSES__</table>
<h2>生存上下界预判</h2><p id="precheck-work"></p>
<p>一次预判决定表示当前前缀跳过一次节点全正概率求值，不能直接计作省下一次 QMC：原本的调用可能只有一、二维，或被空域约束筛选降维。改变节点选择历史也会改变后续调用次数，净节省量应通过同场景关闭预判的对照渲染测量。</p>
<h2>光线分布</h2><div class="grid" id="maps"></div>
<h2>概率计算耗时</h2><p>时间为各工作线程的区间耗时之和，不能当作渲染墙钟时间。
QMC 时间包含在全正概率时间内；实际维数 ≥ 3 的调用也属于全部调用。预判计时可能包含二维符号计算，以上时间不能直接相加。</p>
<table><tr><th>范围</th><th>调用次数</th><th>累计耗时</th><th>每次平均</th></tr>__TIMING__</table>
<p id="qmc-work"></p>
<h2>全正概率累计耗时最高的光线</h2><table><tr><th>像素</th><th>采样编号</th><th>结果</th><th>累计额外点</th><th>峰值节点数（含端点）</th><th>全正概率 ms</th></tr>__TOP__</table>
<h2>节点数量分布</h2><pre id="quantiles"></pre><p><small>统计仅代表本次场景、精度、种子和硬件。渲染器的候选池不等于实际加入 CDF 的节点，均值非常高的约束还可能在积分后端被剔除并计入尾概率误差。整条光线未调用全正概率后端时，峰值 CDF 节点数为 0；这不代表没有计算 SDF、Rice 积分或二维预判概率。</small></p>
<div id="hover">在任意热图上移动鼠标查看像素。</div><script>
const data=__DATA__;
const {width,height,pixels,summary,analysis}=data;
const panels=[['extra_fraction','需要额外节点的光线比例',false,1],['max_active','单次 CDF 的最大节点数（含端点）',false,null],['mean_multivariate_ms','实际维数 ≥ 3 的全正概率 ms / 光线',true,null],['hit_fraction','首碰命中比例',false,1]];
if(analysis.survival_precheck.counters_available)panels.push(['without_orthant_fraction','整条光线未调用全正概率后端的比例',false,1],['mean_precheck_ms','生存上下界预判 ms / 光线',true,null]);
function color(t){const colors=[[22,33,76],[45,155,160],[250,220,90]],i=t<.5?0:1,u=t<.5?t*2:(t-.5)*2;return colors[i].map((v,j)=>Math.round(v*(1-u)+colors[i+1][j]*u));}
for(const [key,title,log,fixed] of panels){
 const max=fixed??pixels.reduce((m,p)=>p?Math.max(m,p[key]):m,0);
 const panel=document.createElement('section');panel.className='panel';
 const heading=document.createElement('h3');heading.textContent=title;panel.appendChild(heading);
 const canvas=document.createElement('canvas');canvas.width=width;canvas.height=height;canvas.style.aspectRatio=`${width}/${height}`;panel.appendChild(canvas);
 const ctx=canvas.getContext('2d'),img=ctx.createImageData(width,height);
 pixels.forEach((p,i)=>{const value=p?p[key]:0;const t=max?Math.min(1,log?Math.log1p(value*1000)/Math.log1p(max*1000):value/max):0;img.data.set([...(p?color(t):[210,213,218]),255],i*4);});ctx.putImageData(img,0,0);
 const legend=document.createElement('div');legend.className='legend';panel.appendChild(legend);
 const range=document.createElement('div');range.className='range';range.innerHTML='<span>0</span><span>'+(log?'对数颜色刻度 · ':'')+max.toPrecision(4)+'</span>';panel.appendChild(range);
 canvas.addEventListener('mousemove',e=>{const r=canvas.getBoundingClientRect(),x=Math.min(width-1,Math.floor((e.clientX-r.left)*width/r.width)),y=Math.min(height-1,Math.floor((e.clientY-r.top)*height/r.height)),p=pixels[y*width+x];const precheckDetails=p&&analysis.survival_precheck.counters_available?`\n整条光线无全正概率调用 ${(100*p.without_orthant_fraction).toFixed(1)}% · 生存上下界预判 ${p.mean_precheck_ms.toFixed(5)} ms/光线`:'';document.querySelector('#hover').textContent=p?`像素 (${x}, ${y}) · 完成光线 ${p.flights} · 失败 ${p.failed}\n需额外点 ${(100*p.extra_fraction).toFixed(1)}% · 命中 ${(100*p.hit_fraction).toFixed(1)}% · 峰值节点 ${p.max_active} · 最多累计额外点 ${p.max_added}\n全正概率 ${p.mean_orthant_ms.toFixed(5)} ms/光线 · 其中实际维数≥3 ${p.mean_multivariate_ms.toFixed(5)} ms/光线 · QMC循环 ${p.mean_qmc_ms.toFixed(5)} ms/光线${precheckDetails}`:`像素 (${x}, ${y})：无完成光线`;});
 document.querySelector('#maps').appendChild(panel);
}
document.querySelector('#facts').textContent=`${width} × ${height} · ${analysis.spp??'?'} spp · ${analysis.worker_threads??'?'} 线程 · 渲染墙钟 ${analysis.renderer_wall_seconds??'?'} s · scene seed ${analysis.scene_seeds.join(', ')} · 首段光线 ${summary.counts.flights} · 数值失败 ${summary.counts.failed} · 单条光线最多累计额外点 ${summary.max_per_flight_statistics.unique_added_nodes} · 单次 CDF 最多 ${summary.max_per_flight_statistics.max_active_nodes} 个节点 · 概率容差 ${summary.settings.probability_tolerance} · 距离容差 ${summary.settings.distance_tolerance}`;
document.querySelector('#quantiles').textContent=JSON.stringify(analysis.node_quantiles_all_flights,null,2);
const precheck=analysis.survival_precheck;
document.querySelector('#precheck-work').textContent=precheck.counters_available?`预判 ${precheck.calls.toLocaleString()} 次 · 直接决定 ${precheck.decisions.toLocaleString()} 次（${(100*precheck.decision_fraction).toFixed(2)}%）：畅通 ${precheck.survived.toLocaleString()} 次、此前已碰撞 ${precheck.hit.toLocaleString()} 次，其中一维上界决定 ${precheck.marginal_hits.toLocaleString()} 次。考虑额外内部节点时决定 ${precheck.decisions_with_extra_nodes.toLocaleString()} 次。${precheck.completed_flights_with_decision.toLocaleString()} 条已完成光线至少一次由预判决定；${precheck.completed_flights_without_orthant.toLocaleString()} 条全程未调用全正概率后端，占已完成光线 ${(100*precheck.completed_flights_without_orthant_fraction).toFixed(2)}%。${precheck.enabled===false?' 本次场景已关闭预判。':''}`:'本次输入为旧统计，未记录生存上下界预判计数；不能据此判断节省量。';
const work=analysis.qmc_incremental;
document.querySelector('#qmc-work').textContent=`QMC 实际新计算 ${work.new_points.toLocaleString()} 点 · 复用 ${work.reused_points.toLocaleString()} 点 · 矩阵与积分准备 ${work.preparations.toLocaleString()} 次 · 相对逐次从头计算减少 ${(100*work.saved_fraction).toFixed(2)}% 积分点评估。点数已计入 8 组 Sobol。QMC 调用数包含只读取已有结果的请求；节省比例不代表耗时降幅。${work.reuse_counters_available?'':' 本次输入为旧统计，未记录复用和准备次数，按 0 显示。'}`;
</script></html>'''
    data = {"width": width, "height": height, "pixels": image, "summary": summary, "analysis": analysis}
    document = document.replace("__CLASSES__", table).replace("__TIMING__", timing).replace("__TOP__", top)
    document = document.replace("__DATA__", json.dumps(data, ensure_ascii=False).replace("</", "<\\/"))
    output.write_text(document, encoding="utf-8")
    print(json.dumps({"validated_flights": counts["flights"], "report": str(output),
                      "groups": groups, "qmc_incremental": analysis["qmc_incremental"],
                      "survival_precheck": analysis["survival_precheck"],
                      "node_quantiles": analysis["node_quantiles_all_flights"]}, indent=2))


if __name__ == "__main__":
    main()
