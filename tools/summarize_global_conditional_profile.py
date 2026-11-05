"""Split adaptive conditional transport cost without double-counting shadow work."""
import argparse
import html
import json
from pathlib import Path
import statistics


BOUND_STAGES = ("birth_bound", "interval_bound", "taylor_bound")


def summarize_run(run):
    profile = run["profile"]
    if profile["unit"] != "sum_worker_ms":
        raise ValueError("Expected summed worker elapsed time")
    domains = {}
    for name, domain in profile["domains"].items():
        stages = domain["stages"]
        total = sum(stage["exclusive_ms"] for stage in stages.values())
        # These three entry-point scopes are disjoint in adaptive mode. Their
        # inclusive times include nested SDF queries and moment/bound algebra.
        bounds = sum(stages[stage]["inclusive_ms"] for stage in BOUND_STAGES)
        hazard = stages["hazard"]["inclusive_ms"]
        other = total - bounds - hazard
        if other < -1e-3:
            raise ValueError("Overlapping bound/hazard scopes or inconsistent timers")
        counts = domain["counts"]
        if stages["hazard"]["calls"] != counts["candidate_events"]:
            raise ValueError("Hazard scopes do not exclusively cover tracking candidates")
        domains[name] = {
            "total_ms": total, "bounds_ms": bounds, "hazard_ms": hazard,
            "other_ms": max(0.0, other), "track_inclusive_ms": stages["track"]["inclusive_ms"],
            "bound_stages_ms": {stage: stages[stage]["inclusive_ms"] for stage in BOUND_STAGES},
            "counts": counts, "histograms": domain["histograms"],
        }
    total = sum(domain["total_ms"] for domain in domains.values())
    if total <= 0:
        raise ValueError("Empty profile")
    for domain in domains.values():
        domain["percent"] = {
            key: 100 * domain[key + "_ms"] / total
            for key in ("total", "bounds", "hazard", "other")
        }
    return {"label": run["label"], "total_ms": total, "domains": domains}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--case", default="step_default")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    media = [medium for medium in report["scene_snapshot"]["media"]
             if medium["type"] == "global_conditional"]
    if len(media) != 1 or media[0].get("tracking_majorant", "adaptive") != "adaptive":
        parser.error("This grouping requires a single adaptive global_conditional medium")
    selected = [run for run in report["runs"] if run["case"] == args.case]
    profiles = [summarize_run(run) for run in selected if run["profile_enabled"]]
    plain = [run["render_seconds"] for run in selected
             if run["label"].startswith(args.case + "_plain_")]
    profiled_seconds = [run["render_seconds"] for run in selected if run["profile_enabled"]]
    if not profiles or not plain:
        parser.error("Both plain and instrumented renders are required")
    pooled_total = sum(run["total_ms"] for run in profiles)
    domains = {}
    for name in profiles[0]["domains"]:
        samples = [run["domains"][name] for run in profiles]
        if any(sample["counts"] != samples[0]["counts"] for sample in samples):
            raise ValueError("Repeated profiles changed event counts")
        domains[name] = {
            "percent": {key: 100 * sum(s[key + "_ms"] for s in samples) / pooled_total
                        for key in ("total", "bounds", "hazard", "other")},
            "percent_range": {key: [min(s["percent"][key] for s in samples),
                                    max(s["percent"][key] for s in samples)]
                              for key in ("total", "bounds", "hazard", "other")},
            "mean_worker_ms": {key: statistics.mean(s[key + "_ms"] for s in samples)
                               for key in ("total", "bounds", "hazard", "other")},
            "counts_per_render": samples[0]["counts"],
            "histograms_per_render": samples[0]["histograms"],
        }
    summary = {
        "source_report": str(args.report.resolve()), "case": args.case,
        "source_scene": report["source"], "configuration": report["configuration"],
        "denominator": "sum of exclusive elapsed worker times inside instrumented conditional transport scopes",
        "bounds_definition": "inclusive birth_bound + interval_bound + taylor_bound (disjoint in adaptive mode)",
        "hazard_definition": "inclusive hazard, including scalar statistics and SDF evaluation",
        "shadow_definition": "entire shadow domain, including its own bound and hazard costs; overlaps column totals",
        "profile_repeats": len(profiles), "mean_conditional_worker_ms": pooled_total / len(profiles),
        "plain_render_seconds": {"values": plain, "median": statistics.median(plain)},
        "instrumented_render_seconds": profiled_seconds,
        "instrumentation_runtime_ratio": statistics.median(profiled_seconds) / statistics.median(plain),
        "domains": domains,
        "totals_percent": {key: sum(domain["percent"][key] for domain in domains.values())
                           for key in ("total", "bounds", "hazard", "other")},
        "parity": report["parity"], "profiles": profiles,
    }
    output = args.output or args.report.parent
    output.mkdir(parents=True, exist_ok=True)
    (output / "cost-summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    names = {"path": "路径距离采样", "shadow": "阴影透射率", "phase": "相位函数", "connection": "连接处理"}
    rows = []
    for name, domain in domains.items():
        p = domain["percent"]
        rows.append("<tr><th>" + names.get(name, name) + "</th>" +
                    "".join(f"<td>{p[key]:.2f}%</td>" for key in ("bounds", "hazard", "other", "total")) + "</tr>")
    p = summary["totals_percent"]
    rows.append("<tr><th>合计</th>" + "".join(f"<td>{p[key]:.2f}%</td>" for key in ("bounds", "hazard", "other", "total")) + "</tr>")
    event_rows = []
    for name in ("path", "shadow"):
        d = domains[name]
        c = d["counts_per_render"]
        flights = d["histograms_per_render"]["candidates_per_flight"]["count"]
        values = [flights] + [c[key] for key in ("processed_intervals", "candidate_events", "accepted_collisions", "rejected_candidates", "ratio_events")]
        event_rows.append("<tr><th>" + names[name] + "</th>" + "".join(f"<td>{v:,}</td>" for v in values) + "</tr>")
    cfg = report["configuration"]
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>起点条件化：成本测量</title><style>
body{{font:16px/1.7 system-ui,sans-serif;max-width:1060px;margin:40px auto;padding:0 24px;color:#203047;background:#f7f9fc}}
table{{border-collapse:collapse;width:100%;background:white;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #dce2eb;text-align:right}}th:first-child{{text-align:left}}code{{background:#e8edf5;padding:2px 5px}}small{{color:#596a80}}
</style><h1>起点条件化：成本测量</h1>
<p><code>global_conditional / adaptive</code> · {cfg['size']} × {cfg['size']} · {cfg['spp']} spp · {cfg['threads']} threads · seed {cfg['seed']}</p>
<p>场景：{html.escape(report['source'])}</p>
<p>普通版耗时：{', '.join(f'{t:.3f}' for t in plain)} 秒；中位数 <b>{statistics.median(plain):.3f} 秒</b>。
计时版：{', '.join(f'{t:.3f}' for t in profiled_seconds)} 秒；耗时倍率 {summary['instrumentation_runtime_ratio']:.2f}×。</p>
<p>分母是条件输运计时范围内的工作线程耗时之和；不是整帧墙钟时间，也不是无计时开销的 CPU 采样结果。
上界和消光使用包含内部调用的耗时；表格各格不重叠。阴影是整行，包含其自身的上界和消光，不能再与这两列相加。</p>
<table><thead><tr><th>调用用途</th><th>上界构造</th><th>候选点消光</th><th>其余</th><th>总占比</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>多次计时按累计工作线程耗时汇总。逐次比例、原始计时和直方图保存在 JSON 中。</p>
<table><thead><tr><th>用途</th><th>flight 数</th><th>处理区间</th><th>候选点</th><th>真实碰撞</th><th>拒绝候选</th><th>ratio 事件</th></tr></thead><tbody>{''.join(event_rows)}</tbody></table>
<p>阴影使用 ratio tracking，ratio 事件不等同于路径采样的拒绝空碰撞。运行脚本检查完整 spp、数值异常及普通版／计时版 PFM 逐像素一致性。</p>
<p><a href="cost-summary.json">汇总 JSON</a> · <a href="report.json">原始报告</a></p></html>"""
    (output / "cost-summary.html").write_text(page, encoding="utf-8")
    print(json.dumps({key: summary[key] for key in
                      ("plain_render_seconds", "instrumented_render_seconds", "totals_percent")}, indent=2))
    print("Saved", output / "cost-summary.html")


if __name__ == "__main__":
    main()
