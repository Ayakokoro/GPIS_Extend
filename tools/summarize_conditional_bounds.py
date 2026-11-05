"""Reconcile detailed adaptive majorant timings with an earlier instrumented build."""
import argparse
import html
import json
import math
from pathlib import Path
import re
import statistics

from summarize_conditional_intervals import validate_profile


ENTRY_STAGES = ("birth_bound", "interval_bound", "taylor_bound")
BUCKETS = {
    "sdf_coarse": ("宏块查询、边界与梯度范围", ("bound_sdf_coarse",)),
    "sdf_fine": ("细体素 SDF 范围", ("bound_sdf_fine",)),
    "sdf_points": ("宏块中点及额外 SDF 点/差值", ("bound_sdf_point", "bound_sdf_difference")),
    "algebra": ("条件均值、导数界与合并", ("bound_algebra",)),
    "prepare": ("条件统计量准备", ("bound_moment_prepare",)),
    "far_tail": ("Cached SE tail envelope", ("bound_far_tail",)),
    "variance": ("SE 方差保守界", ("se_variance",)),
    "cdf": ("一维 CDF / logCDF", ("bound_normal_cdf",)),
    "flux": ("穿越通量其余计算", ("bound_flux",)),
    "mills": ("Mills 比其余计算", ("bound_mills",)),
    "apply": ("标准化及消光上界组合", ("bound_moment_apply",)),
    "extrema": ("精细极值搜索", ("exp_polynomial",)),
    "other": ("其余上界控制及未分项开销", ("birth_bound", "interval_bound", "taylor_bound", "moment_bound")),
}
GENERIC_SDF = ("sdf_jet", "sdf_difference", "sdf_ray_bounds", "sdf_bounds")


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def profiles(report, case):
    return [r for r in report["runs"] if r["profile_enabled"] and r["case"] == case]


def same_work(actual, expected, location="", differences=None):
    """Integer counts are exact; worker reduction order can change floating sums."""
    if isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise ValueError(f"{location}: changed fields")
        for key in expected:
            same_work(actual[key], expected[key], f"{location}/{key}", differences)
    elif isinstance(expected, list):
        if len(actual) != len(expected):
            raise ValueError(f"{location}: changed length")
        for i, (a, b) in enumerate(zip(actual, expected)):
            same_work(a, b, f"{location}/{i}", differences)
    elif isinstance(expected, float):
        if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8):
            raise ValueError(f"{location}: aggregate changed: {actual} != {expected}")
        if actual != expected and differences is not None:
            differences.append(abs(actual - expected) / max(1.0, abs(expected)))
    elif actual != expected:
        raise ValueError(f"{location}: changed: {actual} != {expected}")


def validate_run(run, report, reference):
    validate_profile(run["profile"])
    parity_key = run["case"] + "_plain_1_vs_" + run["label"]
    if not report.get("parity", {}).get(parity_key, {}).get("exact"):
        raise ValueError(f"{run['label']}: missing exact image parity")
    differences = []
    for name, domain in run["profile"]["domains"].items():
        original = reference["profile"]["domains"][name]
        for field in ("counts", "histograms", "interval_partition"):
            same_work(domain[field], original[field], name + "/" + field, differences)
        for stage, value in original["stages"].items():
            if domain["stages"][stage]["calls"] != value["calls"]:
                raise ValueError(f"{run['label']}/{name}/{stage}: original call count changed")
        if run["settings"].get("tracking_fixed_step_limit") is False:
            for phase in domain["interval_partition"].values():
                if any(age.get("step_cap", {}).get("intervals", 0) for age in phase.values()):
                    raise ValueError("Uncapped case retained a fixed step cap")
    return {"label": run["label"], "image_exact": True, "counts_exact": True,
            "original_stage_calls_exact": True, "floating_aggregates_close": True,
            "max_relative_floating_aggregate_difference": max(differences, default=0.0)}


def group_run(run, domain_name, detailed):
    if run["profile"]["unit"] != "sum_worker_ms":
        raise ValueError("Expected summed elapsed worker milliseconds")
    domains = run["profile"]["domains"]
    selected = list(domains.values()) if domain_name == "all" else [domains[domain_name]]
    stages = {
        name: {key: sum(d["stages"][name][key] for d in selected)
               for key in ("calls", "inclusive_ms", "exclusive_ms")}
        for name in selected[0]["stages"]
    }
    total = sum(s["exclusive_ms"] for s in stages.values())
    bounds = sum(stages[s]["inclusive_ms"] for s in ENTRY_STAGES)
    result = {"total_ms": total, "bounds_ms": bounds, "stages": stages,
              "counts": {k: sum(d["counts"][k] for d in selected) for k in selected[0]["counts"]}}
    if detailed:
        if any(stages[s]["calls"] for s in GENERIC_SDF):
            raise ValueError("Generic SDF scopes need context attribution for this grouping")
        times = {key: sum(stages.get(s, {}).get("exclusive_ms", 0) for s in names)
                 for key, (_, names) in BUCKETS.items()}
        error = sum(times.values()) - bounds
        if not math.isclose(sum(times.values()), bounds, rel_tol=1e-7, abs_tol=0.02):
            raise ValueError(f"{run['label']}/{domain_name}: timing partition does not reconcile: {error}")
        if stages["bound_moment_prepare"]["calls"] + stages["bound_moment_apply"]["calls"] != stages["moment_bound"]["calls"]:
            raise ValueError("Moment preparation/application call count mismatch")
        result.update(bucket_ms=times, partition_error_ms=error)
    else:
        result["old_percent_of_bounds"] = {
            "ordinary_own": 100 * stages["interval_bound"]["exclusive_ms"] / bounds if bounds else 0,
            "moment_own": 100 * stages["moment_bound"]["exclusive_ms"] / bounds if bounds else 0,
            "variance": 100 * stages["se_variance"]["exclusive_ms"] / bounds if bounds else 0,
            "extrema": 100 * stages["exp_polynomial"]["exclusive_ms"] / bounds if bounds else 0,
            "birth_taylor_own": 100 * sum(stages[s]["exclusive_ms"] for s in ("birth_bound", "taylor_bound")) / bounds if bounds else 0,
        }
    return result


def pool(samples):
    bound_sum = sum(s["bounds_ms"] for s in samples)
    total_sum = sum(s["total_ms"] for s in samples)
    result = {"mean_bounds_worker_ms": bound_sum / len(samples),
              "bounds_percent_of_conditional": 100 * bound_sum / total_sum if total_sum else 0,
              "counts_per_render": samples[0]["counts"], "buckets": {}}
    for key in BUCKETS:
        values = [s["bucket_ms"][key] for s in samples]
        percents = [100 * v / s["bounds_ms"] if s["bounds_ms"] else 0 for v, s in zip(values, samples)]
        result["buckets"][key] = {
            "mean_worker_ms": statistics.mean(values),
            "percent_of_bounds": 100 * sum(values) / bound_sum if bound_sum else 0,
            "percent_range": [min(percents), max(percents)]}
    result["stages"] = {
        name: {"calls_per_render": samples[0]["stages"][name]["calls"],
               "mean_inclusive_ms": statistics.mean(s["stages"][name]["inclusive_ms"] for s in samples),
               "mean_exclusive_ms": statistics.mean(s["stages"][name]["exclusive_ms"] for s in samples)}
        for name in samples[0]["stages"]
    }
    result["max_partition_error_ms"] = max(abs(s["partition_error_ms"]) for s in samples)
    return result


PAGE = r'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>上界构造：细分计时</title><style>
body{font:15px/1.7 system-ui,"Microsoft YaHei",sans-serif;margin:0;background:#f4f6fa;color:#1b2b42}
main{max-width:1160px;margin:auto;padding:32px}h1{font-size:27px;margin:0}h2{font-size:20px}
.muted{color:#53647b}.box{background:white;border:1px solid #dae2ed;border-radius:8px;padding:20px;margin:18px 0}
select{padding:8px;margin:0 22px 10px 6px;border:1px solid #bac8da;background:white;border-radius:4px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{padding:8px 12px;text-align:right;border-bottom:1px solid #e5eaf1}th:first-child,td:first-child{text-align:left}thead{background:#edf2f8}
.scroll{overflow:auto}.bar{display:flex;height:30px;overflow:hidden;border-radius:4px}.legend{display:flex;gap:10px 20px;flex-wrap:wrap;font-size:13px;margin-top:12px}.swatch{display:inline-block;width:10px;height:10px;margin-right:5px}
code{background:#edf2f8;padding:2px 5px}a{color:#2166b6}summary{cursor:pointer}
</style><main><h1>上界构造：细分计时</h1>
<p class="muted">@@CONFIG@@ · global_conditional / adaptive · shaderball stationary VDB</p>
<p>本次只增加计时，不改变采样公式。比较 32 体素宏块、远处接受阈值 64 下，保留或去掉 0.08 固定步长上限的两组设置。重复次数见下方计时表。</p>
<p><strong>下表百分比的分母是上界构造的工作线程耗时之和。</strong>使用各子项的 exclusive 时间，避免父子计时重复相加。它不是整帧墙钟时间。</p>
<div class="box"><label>配置<select id="case"></select></label><label>用途<select id="domain"><option value="all">全部</option><option value="path">主路径</option><option value="shadow">阴影</option></select></label>
<p id="description"></p><div class="bar" id="bar"></div><div class="legend" id="legend"></div></div>
<div class="box scroll"><table id="breakdown"></table></div>
<div class="box"><h2>上界查询工作量</h2><p id="work"></p><p class="muted">调用数是每次渲染的计数，不是不同子项之和；同一次上界计算会调用多个子项。</p></div>
<div class="box scroll"><h2>计时开销与一致性</h2><table id="overhead"></table>
<p>细分计时会增加大量时钟读取与统计写入，特别会扰动短小的 CDF 调用及其父项；因此这些占比用于定位，不应直接推算普通构建的可加速比例。旧版与细分版的时间差也包含运行顺序及 CPU 频率波动。</p>
<p>所有计时图像与既有普通版逐像素一致；计数、直方图整数项和原有 stage 调用数完全相同。长度与光学厚度的线程归并浮点和以相对 10⁻⁹ / 绝对 10⁻⁸ 容差核对。所有时间分项与上界总时间核对通过。@@TESTS@@ 普通版图像从旧实验复用，未重新测量普通版速度。</p></div>
<div class="box"><h2>分项口径</h2>
<p>宏块查询包括宏块证书读取、边界截断及梯度投影；其内部的中点插值单列。细体素项包含范围计算及其内部中点计算；它不包含所有发生在遍历或起点准备阶段的细体素工作。</p>
<p>条件均值项包括起点修正的指数/多项式区间包围、中点半径包围和 cheap/tight 分支合并。统计量准备包括端点 SE 因子、标准差与回归系数界；认证 SE 方差界另列。Mills 与穿越通量列均已扣除 CDF/logCDF 子调用。</p>
<p>其余项包含 birth/Taylor 未继续细分的工作、调用包装与部分插桩开销。计时范围没有覆盖的开销不能解释为不存在。</p>
<details><summary>原始 stage 时间及调用数</summary><div class="scroll"><table id="raw"></table></div></details>
<p><a href="comparison.json">汇总 JSON</a> · <a href="validation.json">核对结果</a> · <a href="control/report.json">旧计时报告</a> · <a href="detail/report.json">细分计时报告</a></p></div>
</main><script id="data" type="application/json">@@DATA@@</script><script>
const data=JSON.parse(document.getElementById('data').textContent),cases=Object.keys(data.cases),keys=Object.keys(data.bucket_definitions);
const labels={block32_far64:'保留 0.08 上限',block32_no_cap:'去掉固定步长上限'};
const colors=['#2672ca','#4aaac5','#8cc6b8','#9364c6','#de9b2b','#db633c','#c33d69','#976b4a','#b1a750','#5970a7','#6c9466','#9ca6b6'];
const num=x=>x.toLocaleString('zh-CN',{maximumFractionDigits:2}),pct=x=>x.toFixed(2)+'%';
document.getElementById('case').innerHTML=cases.map(k=>`<option value="${k}" ${k==='block32_no_cap'?'selected':''}>${labels[k]||k}</option>`).join('');
function render(){const c=data.cases[document.getElementById('case').value],domain=document.getElementById('domain').value,g=c.groups[domain],b=g.buckets;
document.getElementById('description').textContent=`每次渲染上界计时平均 ${num(g.mean_bounds_worker_ms/1000)} 工作线程秒，占该用途条件传输计时的 ${pct(g.bounds_percent_of_conditional)}。`;
document.getElementById('bar').innerHTML=keys.map((k,i)=>`<div style="width:${b[k].percent_of_bounds}%;background:${colors[i]}" title="${data.bucket_definitions[k].label}: ${pct(b[k].percent_of_bounds)}"></div>`).join('');
document.getElementById('legend').innerHTML=keys.map((k,i)=>`<span><i class="swatch" style="background:${colors[i]}"></i>${data.bucket_definitions[k].label}</span>`).join('');
document.getElementById('breakdown').innerHTML='<thead><tr><th>上界子项</th><th>占上界时间</th><th>两次占比范围</th><th>平均工作线程秒</th></tr></thead><tbody>'+keys.map(k=>`<tr><td>${data.bucket_definitions[k].label}</td><td>${pct(b[k].percent_of_bounds)}</td><td>${b[k].percent_range.map(pct).join(' – ')}</td><td>${num(b[k].mean_worker_ms/1000)}</td></tr>`).join('')+'</tbody>';
const w=g.counts_per_render,s=g.stages;
document.getElementById('work').textContent=`普通区间上界调用 ${num(s.interval_bound.calls_per_render)}；宏块查询 ${num(w.grid_macro_queries)}；细体素构建 ${num(w.grid_cell_builds)}；处理区间 ${num(w.processed_intervals)}；无候选点区间 ${num(w.zero_event_intervals)}（${pct(100*w.zero_event_intervals/w.processed_intervals)}）。`;
document.getElementById('raw').innerHTML='<thead><tr><th>Stage</th><th>每次调用数</th><th>Inclusive 平均 ms</th><th>Exclusive 平均 ms</th></tr></thead><tbody>'+Object.entries(s).map(([k,v])=>`<tr><td>${k}</td><td>${num(v.calls_per_render)}</td><td>${num(v.mean_inclusive_ms)}</td><td>${num(v.mean_exclusive_ms)}</td></tr>`).join('')+'</tbody>';
}
document.getElementById('overhead').innerHTML='<thead><tr><th>配置</th><th>旧计时版渲染秒</th><th>细分版渲染秒</th><th>细分中位数 / 旧版</th></tr></thead><tbody>'+cases.map(k=>{const t=data.cases[k].timing;return `<tr><td>${labels[k]||k}</td><td>${t.control_seconds.map(num).join(', ')}</td><td>${t.detail_seconds.map(num).join(', ')}</td><td>${t.ratio.toFixed(2)}×</td></tr>`}).join('')+'</tbody>';
document.querySelectorAll('select').forEach(e=>e.addEventListener('change',render));render();
</script></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("detail", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ctest-log", type=Path)
    args = parser.parse_args()
    control, detail, reference = (read(p) for p in (args.control, args.detail, args.reference))
    for report in (control, detail):
        if report["source"] != reference["source"] or report["scene_snapshot"] != reference["scene_snapshot"]:
            raise ValueError("Source scene changed")
        for key in ("size", "spp", "threads", "seed", "affinity_mask"):
            if report["configuration"][key] != reference["configuration"][key]:
                raise ValueError(f"Benchmark configuration changed: {key}")
        plain_binary = reference["executable"]
        if report["binary_sha256"].get(plain_binary) != reference["binary_sha256"][plain_binary]:
            raise ValueError("Plain reference binary changed")
    summary = {"configuration": detail["configuration"], "source": detail["source"],
               "reports": {"control": str(args.control.resolve()), "detail": str(args.detail.resolve()),
                           "reference": str(args.reference.resolve())},
               "binary_sha256": {"control": control["binary_sha256"], "detail": detail["binary_sha256"]},
               "denominator": "exclusive worker elapsed time inside disjoint birth/interval/Taylor bounds",
               "bucket_definitions": {k: {"label": label, "exclusive_stages": names} for k, (label, names) in BUCKETS.items()},
               "cases": {}, "validation": []}
    for case in detail["cases"]:
        original = profiles(reference, case)[0]
        controls, details = profiles(control, case), profiles(detail, case)
        if not controls or len(details) < 2:
            raise ValueError(f"{case}: need control and at least two detailed profiles")
        for report, runs in ((control, controls), (detail, details)):
            for run in runs:
                if run["settings"] != original["settings"]:
                    raise ValueError("Case settings changed")
                summary["validation"].append({"report": "control" if report is control else "detail",
                                               "case": case, **validate_run(run, report, original)})
        groups = {}
        controls_summary = {}
        for name in ("all", *original["profile"]["domains"]):
            groups[name] = pool([group_run(run, name, True) for run in details])
            controls_summary[name] = [group_run(run, name, False) for run in controls]
        old_times = [r["render_seconds"] for r in controls]
        new_times = [r["render_seconds"] for r in details]
        summary["cases"][case] = {"settings": original["settings"], "groups": groups,
            "control": controls_summary,
            "timing": {"control_seconds": old_times, "detail_seconds": new_times,
                       "ratio": statistics.median(new_times) / statistics.median(old_times)}}
    validation = {"all_passed": True, "renders": summary["validation"],
                  "floating_sum_tolerance": {"relative": 1e-9, "absolute": 1e-8},
                  "bound_partition_verified": True, "moment_stage_calls_verified": True}
    tests_caption = ""
    if args.ctest_log:
        raw = args.ctest_log.read_bytes()
        log = raw.decode("utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig")
        match = re.search(r"100% tests passed, 0 tests failed out of (\d+)", log)
        if not match:
            raise ValueError("CTest log does not report all tests passed")
        total = int(match[1])
        validation["ctest"] = {"passed": total, "total": total, "log": str(args.ctest_log.resolve())}
        tests_caption = f"{total}/{total} CTest 通过。"
    summary["checks"] = validation
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    cfg = summary["configuration"]
    caption = f"{cfg['size']} × {cfg['size']} · {cfg['spp']} spp · {cfg['threads']} threads · seed {cfg['seed']} · affinity {hex(cfg['affinity_mask'])}"
    document = PAGE.replace("@@CONFIG@@", html.escape(caption))
    document = document.replace("@@TESTS@@", tests_caption)
    document = document.replace("@@DATA@@", json.dumps(summary, ensure_ascii=False).replace("</", "<\\/"))
    (args.output / "comparison.html").write_text(document, encoding="utf-8")
    for case, value in summary["cases"].items():
        print(json.dumps({"case": case, "percent_of_bounds": {k: round(v["percent_of_bounds"], 3)
              for k, v in value["groups"]["all"]["buckets"].items()}, "timing": value["timing"]}))


if __name__ == "__main__":
    main()
