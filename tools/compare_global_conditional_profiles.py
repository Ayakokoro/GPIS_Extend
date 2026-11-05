"""Compare runtime and work counts across conditional-tracking benchmark cases."""
import argparse
import html
import json
from pathlib import Path
import statistics

from summarize_global_conditional_profile import summarize_run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("--baseline", default="step_default")
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    summaries = {}
    for case, timings in report["plain_summary"].items():
        runs = [run for run in report["runs"] if run["case"] == case]
        profiles = [summarize_run(run) for run in runs if run["profile_enabled"]]
        plain = [run for run in runs if run["label"].startswith(case + "_plain_")]
        if not profiles:
            raise ValueError(f"No profile for {case}")
        domains = {}
        for name in profiles[0]["domains"]:
            samples = [run["domains"][name] for run in profiles]
            domains[name] = {
                "mean_ms": {key: statistics.mean(sample[key + "_ms"] for sample in samples)
                            for key in ("total", "bounds", "hazard", "other")},
                "counts": samples[0]["counts"],
                "histograms": samples[0]["histograms"],
            }
            if any(sample["counts"] != samples[0]["counts"] for sample in samples):
                raise ValueError("Repeated profiles changed event counts")
        summaries[case] = {
            "settings": plain[0]["settings"], "timing": timings,
            "render_seconds": [run["render_seconds"] for run in plain],
            "cpu_seconds": [run["process_cpu_seconds"] for run in plain],
            "mean_rgb": plain[0]["mean_rgb"], "domains": domains,
            "intervals": sum(d["counts"]["processed_intervals"] for d in domains.values()),
            "candidates": sum(d["counts"]["candidate_events"] for d in domains.values()),
            "macro_rejected": sum(d["counts"]["macro_rejected"] for d in domains.values()),
            "fine_cell_builds": sum(d["counts"]["grid_cell_builds"] for d in domains.values()),
            "bounds_worker_seconds": sum(d["mean_ms"]["bounds"] for d in domains.values()) / 1000,
            "hazard_worker_seconds": sum(d["mean_ms"]["hazard"] for d in domains.values()) / 1000,
        }
    baseline = summaries[args.baseline]
    for case in summaries.values():
        case["speedup"] = baseline["timing"]["median_seconds"] / case["timing"]["median_seconds"]
        case["time_saved_percent"] = 100 * (1 - 1 / case["speedup"])
        for key in ("intervals", "candidates", "fine_cell_builds"):
            case[key + "_ratio"] = case[key] / baseline[key]
    summary = {
        "source_report": str(args.report.resolve()), "configuration": report["configuration"],
        "baseline": args.baseline, "cases": summaries,
        "all_repetition_and_instrumentation_parity_exact": all(p["exact"] for p in report["parity"].values()),
        "notes": ["Render times exclude scene loading and image saving; profiles include timer overhead.",
                  "Bounds/hazard time is summed worker elapsed time, not render wall time.",
                  "Changed proposals consume different random streams; cross-case pixels need not match.",
                  "Uniform macroblock sizes were measured. No octree performance is claimed."],
    }
    output = args.report.parent
    (output / "comparison.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for name, case in summaries.items():
        t = case["timing"]
        rows.append(f"<tr><th>{html.escape(name)}</th><td>{t['median_seconds']:.2f}</td>"
                    f"<td>{t['min_seconds']:.2f}–{t['max_seconds']:.2f}</td>"
                    f"<td>{case['speedup']:.2f}×</td><td>{case['intervals']/1e6:.2f}</td>"
                    f"<td>{case['candidates']/1e6:.2f}</td><td>{case['bounds_worker_seconds']:.2f}</td>"
                    f"<td>{case['hazard_worker_seconds']:.2f}</td></tr>")
    cfg = report["configuration"]
    affinity = (f"0x{cfg['affinity_mask']:x}" if "affinity_mask" in cfg else "OS default")
    repetition_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>"
        + ", ".join(f"{seconds:.2f}" for seconds in case["render_seconds"])
        + "</td></tr>" for name, case in summaries.items())
    work_rows = "".join(
        f"<tr><th>{html.escape(name)}</th><td>{case['macro_rejected']:,}</td>"
        f"<td>{case['fine_cell_builds']:,}</td>"
        f"<td>{case['domains']['path']['counts']['candidate_events']:,}</td>"
        f"<td>{case['domains']['shadow']['counts']['candidate_events']:,}</td></tr>"
        for name, case in summaries.items())
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>远处宏块阈值实验</title><style>
body{{font:15px/1.7 system-ui,sans-serif;max-width:1200px;margin:36px auto;padding:0 24px;color:#203047;background:#f7f9fc}}
table{{border-collapse:collapse;background:white;width:100%;margin:20px 0}}th,td{{padding:10px;border-bottom:1px solid #dce2eb;text-align:right}}th:first-child{{text-align:left}}pre{{white-space:pre-wrap;background:#e8edf5;padding:16px}}
</style><h1>远处宏块阈值实验</h1>
<p>global_conditional / adaptive · {cfg['size']} × {cfg['size']} · {cfg['spp']} spp · {cfg['threads']} threads · seed {cfg['seed']}</p>
<p>CPU affinity：{affinity}。运行器持续检查进程掩码，恢复初始化期间被运行库重设的亲和性；记录见 report.json。</p>
<p>起点距离达到 4 个方向相关长度后，允许更大的宏块预期事件数。消光、起点条件化和保守上界公式不变；全局预期事件上限仍是 64。</p>
<p>主耗时取未插桩版本多次渲染的中位数；上界与消光耗时取计时版各工作线程之和，不能与墙钟时间直接相加。</p>
<table><thead><tr><th>方案</th><th>中位秒</th><th>范围</th><th>加速</th><th>区间/百万</th><th>候选/百万</th><th>上界 worker-s</th><th>消光 worker-s</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
<p>重复计时反映了机器运行状态的波动；中位数不是置信区间。普通渲染与性能计数分别运行，计时版自身有开销。</p>
<details><summary>逐次普通版计时（秒，按完成顺序）</summary><table>{repetition_rows}</table></details>
<table><thead><tr><th>方案</th><th>宏块拒绝</th><th>细体素构造</th><th>路径候选</th><th>阴影候选</th></tr></thead><tbody>{work_rows}</tbody></table>
<p>同一方案的重复渲染和计时版输出逐像素一致：{summary['all_repetition_and_instrumentation_parity_exact']}。
不同方案改变候选位置及随机数消耗，像素差异不能直接解释为偏差。采样正确性由另行运行的积分参考测试检查。</p>
<p>本表测试的是单层固定宏块，没有实现或计时八叉树。更大块的收益只能为多层结构提供动机，不能直接视为八叉树的加速比。</p>
<details><summary>各方案参数</summary><pre>{html.escape(json.dumps({k:v['settings'] for k,v in summaries.items()}, indent=2))}</pre></details>
<p><a href="comparison.json">比较 JSON</a> · <a href="report.json">完整运行记录</a></p></html>"""
    (output / "comparison.html").write_text(page, encoding="utf-8")
    for name, case in summaries.items():
        print(name, f"{case['timing']['median_seconds']:.2f}s", f"{case['speedup']:.3f}x",
              f"intervals={case['intervals']:,}", f"candidates={case['candidates']:,}",
              f"bounds={case['bounds_worker_seconds']:.2f}s", f"hazard={case['hazard_worker_seconds']:.2f}s")
    print("Saved", output / "comparison.html")


if __name__ == "__main__":
    main()
