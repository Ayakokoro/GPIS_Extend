"""Held-out accuracy, fresh label audits, inference timing and offline HTML report."""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import ndtr
import torch

from .gaussian import orthant_qmc, se_covariance, standardized_statistics, surface_conditioned_statistics
from .infer import DomainError, OrthantPredictor, prepare_gaussian


ROOT = Path(__file__).resolve().parents[2]
MODES = {0: "unconditioned_se", 1: "surface_conditioned_se", 2: "generic_spd"}


def error_metrics(reference, estimate):
    if len(reference) == 0:
        return {"count": 0}
    delta = np.asarray(estimate, dtype=float) - reference
    absolute = np.abs(np.exp(estimate) - np.exp(reference))
    relative = np.abs(np.expm1(delta))
    return {
        "count": len(delta), "probability_mae": float(absolute.mean()),
        "probability_max_error": float(absolute.max()),
        "log_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "log_error_p50": float(np.quantile(np.abs(delta), .5)),
        "log_error_p95": float(np.quantile(np.abs(delta), .95)),
        "relative_error_median": float(np.median(relative)),
        "relative_error_p95": float(np.quantile(relative, .95)),
        "relative_error_max": float(relative.max()),
    }


def forward(model, z, corr, dim, device, batch=8192):
    values = []
    with torch.inference_mode():
        for start in range(0, len(dim), batch):
            values.append(model(torch.as_tensor(z[start:start + batch], dtype=torch.float32, device=device),
                                torch.as_tensor(corr[start:start + batch], dtype=torch.float32, device=device),
                                torch.as_tensor(dim[start:start + batch], dtype=torch.long, device=device))
                          .cpu().numpy())
    return np.concatenate(values).astype(float)


def examples(predictor):
    positions = np.array([.7, 1.4, 2.2, 3.2])
    profile = lambda q: .2 + .3 * q - .14 * q * q + .25 * np.sin(2 * q)
    raw_mean = profile(positions)
    raw_cov = se_covariance(positions)
    conditioned = surface_conditioned_statistics(positions, raw_mean, .2, .8, .4)
    camera_positions = np.r_[0, positions]
    camera_mean, camera_cov = profile(camera_positions), se_covariance(camera_positions)
    result = []
    for name, mean, cov, positive in (
            ("unconditioned", raw_mean, raw_cov, False),
            ("F0=0_G0_known", *conditioned, False),
            ("camera_F0_positive", camera_mean, camera_cov, True)):
        z, corr = standardized_statistics(mean, cov)
        ref = orthant_qmc(z, corr, seed=718391, relative_tolerance=1e-4,
                          initial_power=11, max_power=17)
        reference_logp = ref.log_probability
        if positive:
            from scipy.special import log_ndtr
            reference_logp -= float(log_ndtr(z[0]))
        record = {"name": name, "total_joint_variables": len(mean),
                  "mean": mean.tolist(), "covariance": cov.tolist(),
                  "reference": float(np.exp(reference_logp)), "reference_status": ref.status,
                  "reference_relative_error_estimate": ref.relative_error}
        try:
            prediction = (predictor.predict_given_positive(mean, cov) if positive
                          else predictor.predict(mean, cov))
            record.update(prediction=prediction,
                          relative_error=abs(prediction / np.exp(reference_logp) - 1))
        except DomainError as exc:
            record["unsupported"] = str(exc)
        result.append(record)
    return result


def benchmark(checkpoint, z, corr, dim, numerical_count):
    predictor = OrthantPredictor(checkpoint, "cpu")
    selected = np.flatnonzero(dim >= 2)
    selected = selected[np.linspace(0, len(selected) - 1, min(128, len(selected)), dtype=int)]
    supported, unsupported, analytic = [], 0, 0
    for index in selected:
        n = int(dim[index])
        try:
            prepared = prepare_gaussian(z[index, :n], corr[index, :n, :n])
            predictor._validate_domain(prepared)
            if prepared.method != "neural":
                analytic += 1
                continue
            predictor.predict(z[index, :n], corr[index, :n, :n])
            supported.append(index)
        except DomainError:
            unsupported += 1
    timings = []
    for _ in range(3):
        start = time.perf_counter()
        for index in supported:
            n = int(dim[index])
            predictor.predict(z[index, :n], corr[index, :n, :n])
        timings.append((time.perf_counter() - start) / max(1, len(supported)))
    result = {"cpu_threads": torch.get_num_threads(), "cpu_scalar_supported_cases": len(supported),
              "cpu_scalar_rejected_cases": unsupported,
              "analytic_cases_excluded_from_neural_timing": analytic,
              "cpu_scalar_full_api_microseconds": 1e6 * float(np.median(timings)),
              "cpu_scalar_includes": "PSD/domain checks, standardization, features, network, scalar result",
              "numerical_reference": "Python/SciPy Sobol Genz, 8 scrambles, estimated relative error <= 5e-4; not the C++ renderer"}
    numeric_seconds, numeric_results = [], []
    neural_same_seconds = []
    for index in supported[:numerical_count]:
        n = int(dim[index])
        start = time.perf_counter()
        predictor.predict(z[index, :n], corr[index, :n, :n])
        neural_same_seconds.append(time.perf_counter() - start)
        start = time.perf_counter()
        label = orthant_qmc(z[index, :n], corr[index, :n, :n], seed=92371,
                            initial_power=9, max_power=15, relative_tolerance=5e-4)
        numeric_seconds.append(time.perf_counter() - start)
        numeric_results.append(label.status)
    successful = np.asarray(numeric_results) == "ok"
    if np.any(successful):
        reference_times = np.asarray(numeric_seconds)[successful]
        neural_times = np.asarray(neural_same_seconds)[successful]
        result.update(numerical_cases=len(numeric_seconds),
                      numerical_successful_cases=numeric_results.count("ok"),
                      numerical_mean_microseconds=1e6 * float(np.mean(reference_times)),
                      same_cases_neural_mean_microseconds=1e6 * float(np.mean(neural_times)),
                      python_reference_speed_ratio=float(np.sum(reference_times) / np.sum(neural_times)))
    for device in (["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]):
        net = OrthantPredictor(checkpoint, device).model
        count = min(len(z), 4096)
        inputs = (torch.as_tensor(z[:count], dtype=torch.float32, device=device),
                  torch.as_tensor(corr[:count], dtype=torch.float32, device=device),
                  torch.as_tensor(dim[:count], dtype=torch.long, device=device))
        samples = []
        with torch.inference_mode():
            for _ in range(5):
                net(*inputs)
            for _ in range(20):
                if device == "cuda":
                    torch.cuda.synchronize()
                start = time.perf_counter()
                net(*inputs)
                if device == "cuda":
                    torch.cuda.synchronize()
                samples.append(time.perf_counter() - start)
        result[f"{device}_batch"] = {"batch_size": count,
                                      "median_microseconds_per_gaussian": 1e6 * float(np.median(samples)) / count,
                                      "scope": "validated device-resident tensors; features and forward only; excludes transfer/API validation"}
    if torch.cuda.is_available():
        result["gpu"] = torch.cuda.get_device_name(0)
    return result


def label_audit(z, corr, factors, dim, mode, original, seed, count):
    rng = np.random.default_rng(seed)
    selected = []
    for birth in MODES:
        for n in range(2, 6):
            rows = np.flatnonzero((mode == birth) & (dim == n))
            if len(rows):
                selected.extend((int(rows[np.argmin(original[rows])]), int(rng.choice(rows))))
    selected = list(dict.fromkeys(selected))[:count]
    remaining = np.setdiff1d(np.flatnonzero(dim > 1), selected)
    selected.extend(rng.choice(remaining, min(max(0, count - len(selected)), len(remaining)), replace=False).tolist())
    records = []
    for index in selected:
        n = int(dim[index])
        label = orthant_qmc(z[index, :n], corr[index, :n, :n],
                            factor=None if factors is None else factors[index, :n, :n], seed=seed + int(index),
                            initial_power=11, max_power=17, relative_tolerance=1e-4)
        records.append({"test_index": int(index), "dimension": n, "mode": MODES[int(mode[index])], "status": label.status,
                        "reference_logp": label.log_probability if np.isfinite(label.log_probability) else None,
                        "stored_logp": float(original[index]),
                        "log_disagreement": float(abs(label.log_probability - original[index])) if np.isfinite(label.log_probability) else None,
                        "fresh_relative_error_estimate": label.relative_error if np.isfinite(label.relative_error) else None})
    usable = [row["log_disagreement"] for row in records if row["status"] == "ok"]
    return {"count": len(records), "successful": len(usable), "independent_seed": seed,
            "selection": "rarest and random member of each mode/dimension, followed by random multivariate rows",
            "maximum_log_disagreement": max(usable) if usable else None,
            "records": records}


def write_report(path, report, data):
    rows = []
    for key, values in report["by_mode_and_dimension"].items():
        if not values["count"]:
            continue
        rows.append(f"<tr><td>{html.escape(key)}</td><td>{values['count']}</td>"
                    f"<td>{values['probability_mae']:.4g}</td>"
                    f"<td>{values['relative_error_median']:.2%}</td>"
                    f"<td>{values['relative_error_p95']:.2%}</td>"
                    f"<td>{values['log_rmse']:.4g}</td></tr>")
    content = """<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>Gaussian 正值域概率网络实验</title>
<style>body{font:16px/1.65 system-ui,sans-serif;background:#f4f6fa;color:#172638;max-width:1160px;margin:30px auto;padding:0 24px}
h1{line-height:1.25}section{background:white;padding:22px;margin:22px 0;border:1px solid #dbe3ed;border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:right;padding:8px;border-bottom:1px solid #e2e7ed}
td:first-child,th:first-child{text-align:left}pre{font-size:12px;overflow:auto;background:#eef2f7;padding:15px}
select{padding:6px;margin:8px}canvas{width:100%;height:auto;background:white}.note{color:#526375}</style>
<h1>Gaussian 正值域概率网络实验</h1>
<p>同一个模型：原始 SE、起点值与梯度条件化 SE、一般正定高斯。目标是联合正值域概率。
该实验未接入渲染器；普通正域概率不能代替消光中带下穿速度权重的分子。</p>
<section><h2>独立测试集</h2><p id="summary"></p>
<p class="note">n=1 使用精确公式，不参与多点网络汇总。误差相对于数值标签，标签自身有统计误差。
相对误差 0.1 = 10%；log 误差使用自然对数。按完整均值剖面划分数据，原始/条件化配对不跨分区。</p>
<table><tr><th>模式 / n</th><th>数量</th><th>概率 MAE</th><th>相对误差中位数</th><th>相对误差 P95</th><th>log RMSE</th></tr>ROWS</table></section>
<section><h2>预测与参考</h2><label>模式<select id="mode"><option value="all">全部</option><option value="0">原始 SE</option><option value="1">起点等式条件 SE</option><option value="2">一般 SPD</option></select></label>
<label>变量数<select id="dim"><option value="all">全部多点</option><option>2</option><option>3</option><option>4</option><option>5</option></select></label>
<canvas id="plot" width="1080" height="480"></canvas><p class="note">横轴：参考 log10 P；纵轴：预测 log10 P。虚线为准确预测。颜色表示数据模式。</p></section>
<section><h2>三种起点用法</h2><pre id="examples"></pre></section>
<section><h2>速度与测量范围</h2><p>CPU 完整单次 API、CPU/GPU 批量前向分开测量。
数值参考是本实验的 Python/SciPy 实现，速度比不等于 C++ 渲染加速比。</p><pre id="timing"></pre></section>
<section><h2>尾部、近奇异与边界检查</h2><pre id="details"></pre>
<p class="note">按均值排序减少排列影响；完全相同均值的 ties 不能保证严格排列不变。
训练域只是一组覆盖范围检查，不构成准确度保证。概率很小和相关矩阵接近奇异时需要特别检查。</p></section>
<section><h2>数值标签复核</h2><pre id="audit"></pre></section>
<section><h2>数据生成与训练</h2><pre id="provenance"></pre></section>
<p><a href="evaluation.json">完整评估 JSON</a> · <a href="generation.json">数据生成记录</a> · <a href="examples.json">示例参数</a> · <a href="predictions.npz">测试集预测</a></p>
<script>const report=REPORT;const points=POINTS;
document.getElementById('summary').textContent=`测试集 ${report.test_count} 个样本，多点 ${report.multivariate.count} 个；多点概率 MAE ${report.multivariate.probability_mae.toPrecision(4)}，相对误差中位数 ${(100*report.multivariate.relative_error_median).toFixed(2)}%，P95 ${(100*report.multivariate.relative_error_p95).toFixed(2)}%。`;
for(const key of ['examples','timing'])document.getElementById(key).textContent=JSON.stringify(report[key],null,2);
document.getElementById('details').textContent=JSON.stringify({tails:report.by_probability,conditioning:report.by_smallest_eigenvalue,domain:report.domain_coverage,bounds:report.bounds,permutation:report.permutation},null,2);
document.getElementById('audit').textContent=JSON.stringify({...report.label_audit,records:undefined},null,2);
document.getElementById('provenance').textContent=JSON.stringify({data:report.data_generation,training:report.training},null,2);
function draw(){const chosen=points.filter(p=>p[3]>1&&(mode.value==='all'||p[2]==mode.value)&&(dim.value==='all'||p[3]==dim.value));
const c=document.getElementById('plot'),ctx=c.getContext('2d'),left=76,right=1040,top=24,bottom=425;
ctx.clearRect(0,0,c.width,c.height);if(!chosen.length)return;
const low=Math.min(-1,...chosen.flatMap(p=>[p[0],p[1]]));const x=v=>left+(v-low)/(-low)*(right-left),y=v=>bottom-(v-low)/(-low)*(bottom-top);
ctx.font='14px system-ui';ctx.fillStyle='#526375';ctx.strokeStyle='#dce4ee';ctx.lineWidth=1;
for(let i=0;i<=5;i++){const v=low+(-low)*i/5;ctx.beginPath();ctx.moveTo(x(v),top);ctx.lineTo(x(v),bottom);ctx.moveTo(left,y(v));ctx.lineTo(right,y(v));ctx.stroke();ctx.fillText(v.toFixed(1),x(v)-16,bottom+24);ctx.fillText(v.toFixed(1),18,y(v)+4)}
ctx.strokeStyle='#78889e';ctx.setLineDash([5,5]);ctx.beginPath();ctx.moveTo(left,bottom);ctx.lineTo(right,top);ctx.stroke();ctx.setLineDash([]);
const colors=['#2279b9','#d97d17','#6e52af'];ctx.globalAlpha=.45;for(const p of chosen){ctx.fillStyle=colors[p[2]];ctx.beginPath();ctx.arc(x(p[0]),y(p[1]),2.7,0,2*Math.PI);ctx.fill()}ctx.globalAlpha=1;ctx.fillStyle='#172638';ctx.fillText('参考 log10 P',480,470);ctx.fillText('预测 log10 P',left+10,top+17)}
const mode=document.getElementById('mode'),dim=document.getElementById('dim');mode.onchange=dim.onchange=draw;draw();</script></html>"""
    # Data is numeric only and the generated JSON contains no user HTML.
    content = content.replace("ROWS", "\n".join(rows)).replace("REPORT", json.dumps(report, ensure_ascii=False))
    content = content.replace("POINTS", json.dumps(data, separators=(",", ":")))
    path.write_text(content, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "outputs/gaussian-orthant-nn/dataset.npz")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "models/gaussian-orthant-nn/best.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/gaussian-orthant-nn")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--audit-count", type=int, default=48)
    parser.add_argument("--numerical-benchmark-count", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    with np.load(args.dataset, allow_pickle=False) as archive:
        full = {key: archive[key] for key in archive.files}
    family_splits = {}
    for group, split in zip(full["group_id"], full["split"]):
        if int(group) in family_splits and family_splits[int(group)] != int(split):
            raise ValueError("family leakage between dataset splits")
        family_splits[int(group)] = int(split)
    selected = full["split"] == 2
    z, corr, dim, mode = [full[key][selected] for key in ("z", "corr", "dim", "mode")]
    reference = full["logp"][selected].astype(float)
    if not len(dim) or not np.any(dim > 1):
        raise ValueError("nonempty multivariate held-out test split required")
    predictor = OrthantPredictor(args.checkpoint, args.device)
    prediction = forward(predictor.model, z, corr, dim, args.device)
    eigenvalues = np.array([np.linalg.eigvalsh(c[:int(n), :int(n)])[0] for c, n in zip(corr, dim)])
    report = {"dataset": str(args.dataset), "checkpoint": str(args.checkpoint),
              "test_count": len(dim), "test_families": len(np.unique(full["group_id"][selected])),
              "family_leakage": False, "all": error_metrics(reference, prediction),
              "multivariate": error_metrics(reference[dim > 1], prediction[dim > 1]),
              "model_configuration": predictor.model.configuration(),
              "parameters": sum(p.numel() for p in predictor.model.parameters()),
              "by_mode_and_dimension": {}, "by_mode": {}, "by_probability": {},
              "by_smallest_eigenvalue": {}}
    manifest_path = args.dataset.parent / "generation.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        report["data_generation"] = {key: manifest[key] for key in
                                     ("seed", "requested_groups", "candidate_rows", "accepted_rows", "split_counts",
                                      "mode_counts", "rejected", "label_options", "label_log_probability_range",
                                      "total_seconds") if key in manifest}
    training_path = args.output / "metrics.json"
    if training_path.exists():
        training = json.loads(training_path.read_text(encoding="utf-8"))
        report["training"] = {key: training[key] for key in ("best_epoch", "epochs_completed", "training_seconds")}
    for key, name in MODES.items():
        rows = (mode == key) & (dim > 1)
        report["by_mode"][name] = error_metrics(reference[rows], prediction[rows])
        for n in range(1, predictor.model.max_dim + 1):
            rows = (mode == key) & (dim == n)
            report["by_mode_and_dimension"][f"{name} / {n}"] = error_metrics(reference[rows], prediction[rows])
    for low, high in ((0, 1e-8), (1e-8, 1e-5), (1e-5, 1e-3), (1e-3, .1), (.1, 1.0000001)):
        rows = (dim > 1) & (np.exp(reference) >= low) & (np.exp(reference) < high)
        report["by_probability"][f"[{low:g}, {min(high, 1):g})"] = error_metrics(reference[rows], prediction[rows])
    for low, high in ((0, 1e-8), (1e-8, 1e-4), (1e-4, .1), (.1, 6)):
        rows = (dim > 1) & (eigenvalues >= low) & (eigenvalues < high)
        report["by_smallest_eigenvalue"][f"[{low:g}, {high:g})"] = error_metrics(reference[rows], prediction[rows])
    active = np.arange(z.shape[1])[None] < dim[:, None]
    marginal = np.where(active, ndtr(z), 1)
    lower = np.maximum(0, marginal.sum(axis=1) - z.shape[1] + 1)
    report["bounds"] = {"marginal_upper_violations_over_1e_6": int(np.sum(np.exp(prediction) > marginal.min(axis=1) + 1e-6)),
                        "frechet_lower_violations_over_1e_6": int(np.sum(np.exp(prediction) < lower - 1e-6)),
                        "frechet_max_deficit": float(np.maximum(0, lower - np.exp(prediction)).max())}
    domain = predictor.domain
    outside_mean = np.any(active & ((z < domain.get("minimum_z", -np.inf) - 1e-8) |
                                   (z > domain.get("maximum_z", np.inf) + 1e-8)), axis=1)
    outside_eigen = eigenvalues < max(1e-12, .9 * domain.get("minimum_correlation_eigenvalue", 0))
    report["domain_coverage"] = {"training_envelope": domain,
                                 "outside_mean_count": int(np.sum(outside_mean & (dim > 1))),
                                 "outside_eigenvalue_count": int(np.sum(outside_eigen & (dim > 1))),
                                 "evaluation_scope": "all accepted held-out labels are evaluated, including extrapolation; public API rejects out-of-domain inputs"}
    api_methods, api_rejections = {}, {}
    for i, n in enumerate(dim):
        try:
            prepared = prepare_gaussian(z[i, :int(n)], corr[i, :int(n), :int(n)])
            predictor._validate_domain(prepared)
            api_methods[prepared.method] = api_methods.get(prepared.method, 0) + 1
        except DomainError as exc:
            reason = str(exc)
            api_rejections[reason] = api_rejections.get(reason, 0) + 1
    report["domain_coverage"].update(public_api_methods=api_methods, public_api_rejections=api_rejections,
                                     public_api_supported_count=sum(api_methods.values()))
    rng = np.random.default_rng(8139)
    permuted_z, permuted_corr = z.copy(), corr.copy()
    for i, n in enumerate(dim):
        order = np.r_[rng.permutation(int(n)), np.arange(n, z.shape[1])]
        permuted_z[i], permuted_corr[i] = z[i, order], corr[i][np.ix_(order, order)]
    permuted = forward(predictor.model, permuted_z, permuted_corr, dim, args.device)
    report["permutation"] = {"maximum_log_difference": float(np.abs(permuted - prediction).max()),
                              "equal_mean_ties": "not strictly invariant; stable sorting preserves original tie order"}
    ties_z = np.zeros((32, 5))
    ties_corr = np.broadcast_to(np.eye(5), (32, 5, 5)).copy()
    loadings = rng.normal(size=(4, 7))
    loadings /= np.linalg.norm(loadings, axis=1)[:, None]
    tied_matrix = loadings @ loadings.T
    for i in range(32):
        order = rng.permutation(4)
        ties_corr[i, :4, :4] = tied_matrix[np.ix_(order, order)]
    ties_prediction = forward(predictor.model, ties_z, ties_corr, np.full(32, 4), args.device)
    report["permutation"]["equal_mean_probe_log_range"] = float(np.ptp(ties_prediction))
    report["permutation"]["equal_mean_probe_probability_range"] = [float(np.exp(ties_prediction.min())), float(np.exp(ties_prediction.max()))]
    worst = np.argsort(np.abs(prediction - reference))[-20:][::-1]
    report["worst_cases"] = [{"test_index": int(i), "mode": MODES[int(mode[i])], "dim": int(dim[i]),
                               "z": z[i, :int(dim[i])].tolist(),
                               "correlation": corr[i, :int(dim[i]), :int(dim[i])].tolist(),
                               "reference_logp": float(reference[i]), "prediction_logp": float(prediction[i])}
                              for i in worst]
    print(json.dumps({"accuracy": report["by_mode"]}), flush=True)
    factors = full["factor"][selected] if "factor" in full else None
    report["label_audit"] = label_audit(z, corr, factors, dim, mode, reference, 193745, args.audit_count)
    print("Independent label audit complete", flush=True)
    report["timing"] = benchmark(args.checkpoint, z, corr, dim, args.numerical_benchmark_count)
    report["examples"] = examples(OrthantPredictor(args.checkpoint, "cpu"))
    example_directory = args.output / "examples"
    example_directory.mkdir(exist_ok=True)
    for example in report["examples"]:
        (example_directory / (example["name"] + ".json")).write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(args.output / "predictions.npz", reference_logp=reference, predicted_logp=prediction,
                        mode=mode, dim=dim, group_id=full["group_id"][selected])
    (args.output / "examples.json").write_text(json.dumps(report["examples"], indent=2) + "\n", encoding="utf-8")
    (args.output / "evaluation.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    plotting = np.linspace(0, len(dim) - 1, min(len(dim), 6000), dtype=int)
    points = [[float(reference[i] / np.log(10)), float(prediction[i] / np.log(10)), int(mode[i]), int(dim[i])]
              for i in plotting]
    write_report(args.output / "report.html", report, points)
    print(json.dumps({"report": str(args.output / "report.html"), "timing": report["timing"]}), flush=True)


if __name__ == "__main__":
    main()
