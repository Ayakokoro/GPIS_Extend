"""Make a standalone Chinese report for the sparse-history reference experiment.

Usage:
    python tools/report_single_history_reference.py [path/to/results.json]

Only the standard library is needed. Each chart is both embedded in the HTML
and saved as an independent SVG beside it. This script does not run sampling.
"""

from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path
import re


COLORS = {"reference": "#20242a", "baseline": "#c64742", "selected": "#2269c3", "fine_selected": "#25845b",
          "coarse": "#89929e", "mean": "#657344"}
NAMES = {"reference": "密历史参考", "baseline": "无内部历史", "selected": "所选单点历史",
         "fine_selected": "细体素选点", "coarse": "较粗参考网格", "mean": "均值剖面"}
MODEL_ORDER = ("baseline", "selected", "fine_selected")
FAMILY_NAMES = {"fixed_depth": "固定深度：三线性均值压力测试",
                "slope_limited": "限制斜率：沿射线满足 SDF 斜率约束"}


def esc(value):
    return html.escape(str(value), quote=True)


def number(value, digits=5):
    if value is None:
        return "—"
    value = float(value)
    if not math.isfinite(value):
        return esc(value)
    if value and abs(value) < 0.0001:
        return f"{value:.2e}"
    return f"{value:.{digits}g}"


def percent(value):
    return "—" if value is None else f"{100 * float(value):.3f} 个百分点"


def checked(values, count, name):
    result = [float(value) for value in values]
    if len(result) != count or not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name}: expected {count} finite values")
    return result


def chart(x, series, *, title, xlabel, ylabel, bounds=None, band=None):
    """Return a complete SVG; probabilities are never independently renormalized."""
    x = [float(value) for value in x]
    if not x or not all(math.isfinite(value) for value in x):
        raise ValueError(f"{title}: finite, nonempty x required")
    data = [(key, checked(values, len(x), title + ": " + key)) for key, values in series]
    width, height = 800, 390 if len(data) > 4 else 370
    left, right, top, bottom = 80, 28, 51, 92 if len(data) > 4 else 72
    inner_width, inner_height = width - left - right, height - top - bottom
    xmin, xmax = min(x), max(x)
    if xmax == xmin:
        xmax = xmin + 1
    if bounds is None:
        all_y = [value for _, values in data for value in values]
        ymin, ymax = min(all_y), max(all_y)
        pad = 0.08 * max(ymax - ymin, 0.1)
        ymin, ymax = ymin - pad, ymax + pad
    else:
        ymin, ymax = bounds
    px = lambda value: left + (value - xmin) / (xmax - xmin) * inner_width
    py = lambda value: top + (ymax - value) / (ymax - ymin) * inner_height
    points = lambda pairs: " ".join(f"{px(a):.3f},{py(b):.3f}" for a, b in pairs)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" role="img" aria-label="{esc(title)}">',
           '<style>text{font-family:system-ui,"Microsoft YaHei",sans-serif;fill:#293445}</style>',
           f'<rect width="{width}" height="{height}" fill="white"/>',
           f'<text x="{left}" y="27" font-size="17" font-weight="600">{esc(title)}</text>']
    for index in range(6):
        fraction = index / 5
        xv, yv = xmin + fraction * (xmax - xmin), ymin + fraction * (ymax - ymin)
        xx, yy = px(xv), py(yv)
        out.extend([
            f'<path d="M{xx:.3f},{top}V{top + inner_height}" stroke="#edf0f4"/>',
            f'<path d="M{left},{yy:.3f}H{left + inner_width}" stroke="#edf0f4"/>',
            f'<text x="{xx:.3f}" y="{top + inner_height + 21}" text-anchor="middle" font-size="12">{number(xv, 3)}</text>',
            f'<text x="{left - 11}" y="{yy + 4:.3f}" text-anchor="end" font-size="12">{number(yv, 3)}</text>',
        ])
    if band is not None:
        center, half_width = (checked(values, len(x), title + ": interval") for values in band)
        lower = [max(ymin, value - error) for value, error in zip(center, half_width)]
        upper = [min(ymax, value + error) for value, error in zip(center, half_width)]
        polygon = points(list(zip(x, upper)) + list(zip(reversed(x), reversed(lower))))
        out.append(f'<polygon points="{polygon}" fill="#77818f" fill-opacity="0.22"/>')
    for key, values in data:
        dash = ' stroke-dasharray="5 4"' if key == "coarse" else ""
        weight = 1.6 if key == "coarse" else 2.2
        out.append(f'<polyline points="{points(zip(x, values))}" fill="none" '
                   f'stroke="{COLORS[key]}" stroke-width="{weight}"{dash}/>')
    out.extend([
        f'<path d="M{left},{top}V{top + inner_height}H{left + inner_width}" fill="none" stroke="#7d8896"/>',
        f'<text x="{left + inner_width / 2}" y="{top + inner_height + 43}" text-anchor="middle" font-size="13">{esc(xlabel)}</text>',
        f'<text transform="translate(19,{top + inner_height / 2}) rotate(-90)" text-anchor="middle" font-size="13">{esc(ylabel)}</text>',
    ])
    for index, (key, _) in enumerate(data):
        xx = left + 169 * (index % 4)
        yy = top + inner_height + 60 + 20 * (index // 4)
        out.append(f'<path d="M{xx},{yy}h23" stroke="{COLORS[key]}" stroke-width="2.3"/>'
                   f'<text x="{xx + 29}" y="{yy + 4}" font-size="12">{esc(NAMES[key])}</text>')
    out.append('</svg>')
    return "".join(out)


def table(headers, rows):
    return ('<div class="table-scroll"><table><thead><tr>'
            + "".join(f"<th>{esc(header)}</th>" for header in headers)
            + '</tr></thead><tbody>'
            + "".join('<tr>' + "".join(f"<td>{value}</td>" for value in row) + '</tr>' for row in rows)
            + '</tbody></table></div>')


def history_text(value):
    if value is None or value == []:
        return "无内部历史点"
    if isinstance(value, list):
        return ", ".join(number(item) for item in value)
    if isinstance(value, (int, float)):
        return number(value)
    return esc(json.dumps(value, ensure_ascii=False))


def case_width(case):
    return case.get("width_over_length", case.get("metadata", {}).get("width_over_length"))


def case_metadata(case, key, default=None):
    return case.get(key, case.get("metadata", {}).get(key, default))


def case_family(case):
    return case_metadata(case, "family", "fixed_depth" if case_width(case) is not None else "default")


def family_name(family):
    return FAMILY_NAMES.get(family, str(family))


def mean_at(knots, q):
    for (a, ma), (b, mb) in zip(knots, knots[1:]):
        if a <= q <= b:
            return float(ma) if b == a else ma + (mb - ma) * (q - a) / (b - a)
    return None


def selection_description(probe, model, knots):
    details = probe.get("selection_details", {}).get(model)
    if details is not None:
        q = details.get("q")
        mean = details.get("mean")
    else:
        value = probe.get("selected" if model == "selected" else model)
        if model != "selected" and model not in probe:
            return "未提供所选坐标"
        if isinstance(value, list):
            if len(value) > 1:
                return "u/ℓ = " + history_text(value)
            value = value[0] if value else None
        q, mean = value, None
    if q is None:
        return "无内部历史点"
    if mean is None and isinstance(q, (int, float)):
        mean = mean_at(knots, q)
    return f"u/ℓ = {number(q)}，M(u) = {number(mean)}"


def width_observations(cases):
    """Describe measured cases only; these are not statistical significance tests."""
    lookup = {case["id"]: case for case in cases}
    first = lookup.get("first_narrow_w0p05")
    if first is None or "fine_selected" not in first["transmittance"]["models"]:
        return ""
    tr = first["transmittance"]
    first_probe = next((probe for probe in first["probes"] if abs(probe["q"] - 2.25) < 1e-9), None)
    out = ['<div class="intro"><h2>本次数据观察</h2>']
    physical = lookup.get("surface_narrow_w0p1_slope_limited")
    if physical is not None:
        tmodels = physical["transmittance"]["models"]
        physical_probe = next((probe for probe in physical["probes"] if abs(probe["q"] - 2.25) < 1e-9), None)
        out.append('<p><b>沿线斜率相容的案例也有改善：</b>在限制斜率、宽度 0.1ℓ 的表面起射中，透射率最大差依次为 '
                   + '、'.join(f'{NAMES[name]} {percent(tmodels[name]["max_abs_error"])}' for name in MODEL_ORDER) + '。')
        if physical_probe is not None:
            macro, fine = (physical_probe["models"][name] for name in ("selected", "fine_selected"))
            out.append(f'q = 2.25 的速度 CDF 最大差由宏块 {number(macro["ks"])} 降为细体素 {number(fine["ks"])}，'
                       f'法线 CDF 差由 {number(macro["normal_ks"])} 降为 {number(fine["normal_ks"])}。'
                       f'参考速度 CDF 最大点态半宽为 {number(max(physical_probe["reference_cdf_ci95"]))}，'
                       '细体素剩余差异需结合 Monte Carlo 不确定性看待。')
        out.append('</p>')
    errors = "、".join(f'{NAMES[name]} {percent(tr["models"][name]["max_abs_error"])}'
                       for name in MODEL_ORDER if name in tr["models"])
    out.append(f'<p>在固定深度、宽度 0.05ℓ 的空域起射压力测试中，透射率最大差分别为：{errors}。')
    if first_probe is not None:
        out.append(f'在 q = 2.25，宏块选点为 {selection_description(first_probe, "selected", first["mean_knots"])}；'
                   f'细体素选点为 {selection_description(first_probe, "fine_selected", first["mean_knots"])}。')
    out.append('这个配置暴露了不利宏块对齐下的槽底遗漏。</p>')
    if first_probe is not None:
        macro = first_probe["models"]["selected"]
        fine = first_probe["models"]["fine_selected"]
        out.append(f'<p>同一位置的速度 CDF 最大差从宏块的 {number(macro["ks"])} 降至细体素的 {number(fine["ks"])}；'
                   f'法线 CDF 最大差从 {number(macro["normal_ks"])} 降至 {number(fine["normal_ks"])}。'
                   f'细体素速度差与参考最大点态 95% 半宽 {number(max(first_probe["reference_cdf_ci95"]))} 同量级，'
                   '不能把这部分观测差直接当作确定的模型误差，也不能称两者精确一致。')
        later = next((probe for probe in first["probes"] if abs(probe["q"] - 2.75) < 1e-9), None)
        if later is not None:
            out.append(f'在 q = 2.75，细体素速度 CDF 最大差仍为 {number(later["models"]["fine_selected"]["ks"])}，'
                       f'参考最大点态半宽为 {number(max(later["reference_cdf_ci95"]))}；单点历史并未恢复完整前缀模型。')
        out.append('</p>')
    q2 = next((index for index, q in enumerate(tr["q"]) if abs(q - 2) < 1e-9), None)
    if q2 is not None:
        difference = tr["models"]["fine_selected"]["values"][q2] - tr["reference"][q2]
        same_prefix = all(a == b for q, a, b in zip(tr["q"], tr["models"]["selected"]["values"],
                                                  tr["models"]["fine_selected"]["values"]) if q <= 2)
        if same_prefix:
            out.append(f'<p>此例在 q ≤ 2 的采样网格上，宏块和细体素 T 完全相同；q = 2 时已累计 '
                       f'{percent(difference)} 的有符号参考差，接近细体素方案的全程最大差 '
                       f'{percent(tr["models"]["fine_selected"]["max_abs_error"])}。'
                       '后续选到槽底不会撤销已累计的光学厚度。该早期差异也包含起点基线和单点历史近似，不能全部归因于更新延迟。</p>')
    shallow = [lookup[key] for key in ("first_narrow_w0p1_slope_limited", "first_narrow_w0p05_slope_limited") if key in lookup]
    if shallow:
        descriptions = []
        for case in shallow:
            details = []
            for probe in case["probes"]:
                selection = probe.get("selection_details", {}).get("fine_selected", {})
                details.append(f'q = {number(probe["q"])} 时 u/ℓ = {number(selection.get("q"))}')
            descriptions.append(f'宽度 {number(case_width(case))}ℓ：' + '，'.join(details))
        out.append('<p>在沿线斜率相容的空域起射案例中，细体素所选坐标为：' + '；'.join(descriptions)
                   + '。找到候选槽底，并不意味着每个更新时刻都会选中它；固定更新节奏与评分仍会影响结果。'
                   '因此候选覆盖只是误差来源之一，各剖面必须分别比较。</p>')
    narrow_physical = lookup.get("surface_narrow_w0p05_slope_limited")
    if narrow_physical is not None:
        descriptions = []
        for probe in narrow_physical["probes"]:
            fine = probe["models"]["fine_selected"]
            descriptions.append(f'q = {number(probe["q"])}：{selection_description(probe, "fine_selected", narrow_physical["mean_knots"])}，'
                                f'速度 CDF 最大差 {number(fine["ks"])}')
        out.append('<p>限制斜率、宽度 0.05ℓ 的表面起射进一步显示选点时机的影响。细体素方案在 '
                   + '；'.join(descriptions) + '。收集到候选与查询时实际使用它，是两个不同环节。</p>')
    out.append('</div>')
    return "".join(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="?", default=Path(
        "outputs/single-history-reference-20261106/results.json"))
    args = parser.parse_args()
    data = json.loads(args.results.read_text(encoding="utf-8"))
    root = args.results.resolve().parent
    settings = data["settings"]
    speed_unit = settings.get("speed_unit", "无量纲速度 w = ℓV/σ")
    models = tuple(name for name in MODEL_ORDER if any(
        name in case["transmittance"]["models"] for case in data["cases"]))
    width_study = any(case_width(case) is not None for case in data["cases"])
    families = list(dict.fromkeys(case_family(case) for case in data["cases"]))
    if "fine_selected" in models:
        NAMES["selected"] = "宏块选点"
    title = ("窄槽宽度实验：宏块选点、细体素选点与无内部历史" if width_study
             else "单点历史与无内部历史：透射率和梯度参考比较")
    sections, navigation, summaries, probe_summaries = [], [], [], []

    def figure(svg, filename):
        (root / filename).write_text(svg, encoding="utf-8")
        return f'<figure>{svg}<figcaption><a href="{esc(filename)}">下载 SVG</a></figcaption></figure>'

    for index, case in enumerate(data["cases"]):
        slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", case["id"]).strip("-") or f"case-{index + 1}"
        label = str(case.get("label", case["id"]))
        birth = str(case["start_kind"])
        surface = "surface" in birth.lower() or "gradient" in birth.lower()
        start_label = "表面起射：F₀ = 0，G₀ 已知" if surface else "空域起射：保留 F₀ > 0"
        navigation.append(f'<a href="#{slug}">{esc(label)}</a>')
        content = [f'<section id="{slug}"><h2>{esc(label)}</h2><p>{start_label}</p>']
        if case_width(case) is not None:
            content.append(f'<p>完整槽宽/ℓ = <b>{number(case_width(case))}</b>（两侧槽沿之间的距离）。</p>')
            content.append(f'<p>剖面族：{esc(family_name(case_family(case)))}；'
                           f'最大物理斜率 |dm/dt| = <b>{number(case_metadata(case, "max_physical_slope"))}</b>。</p>')
        knots = case["mean_knots"]
        content.append(figure(chart([item[0] for item in knots], [("mean", [item[1] for item in knots])],
                                    title="沿射线均值剖面", xlabel="q = t/ℓ", ylabel="M = m/σ"),
                              f"{slug}-mean.svg"))
        if case_width(case) is not None:
            center = float(case_metadata(case, "center_q", 1.7))
            half_width = max(0.08, 0.6 * float(case_width(case)))
            lo, hi = max(float(knots[0][0]), center - half_width), min(float(knots[-1][0]), center + half_width)
            zoom = [[lo, mean_at(knots, lo)]]
            zoom.extend(item for item in knots if lo < item[0] < hi)
            zoom.append([hi, mean_at(knots, hi)])
            content.append(figure(chart([item[0] for item in zoom], [("mean", [item[1] for item in zoom])],
                                        title="槽底附近的均值剖面放大", xlabel="q = t/ℓ", ylabel="M = m/σ"),
                                  f"{slug}-mean-zoom.svg"))
        tr = case["transmittance"]
        case_models = tuple(name for name in models if name in tr["models"])
        series = [("coarse", tr["reference_coarse"])]
        series.extend((name, tr["models"][name]["values"]) for name in case_models)
        series.append(("reference", tr["reference"]))
        content.append(figure(chart(tr["q"], series, title="生存透射率 T(q)", xlabel="q = t/ℓ",
                                    ylabel="T", bounds=(0, 1), band=(tr["reference"], tr["ci95"])),
                              f"{slug}-transmittance.svg"))
        content.append('<p class="note">灰带采用 95% 点态 Wilson 区间的对称包络，不是整条曲线的同时置信带。'
                       '灰虚线显示较粗参考网格，检查未检出穿越带来的离散误差。</p>')
        rows = []
        show_divisions = any("integration_fine_divisions" in tr["models"][name] for name in case_models)
        for model in case_models:
            item = tr["models"][model]
            row = [NAMES[model], percent(item["max_abs_error"]), percent(item["rmse"]),
                   percent(item["endpoint_error"]), percent(item["integration_max_delta"])]
            if show_divisions:
                row.append(number(item.get("integration_coarse_divisions")) + " → " + number(item.get("integration_fine_divisions")))
            rows.append(row)
        content.append(table(["模型", "最大 |T − T参考|", "T 曲线 RMSE", "终点有符号差值", "积分加密最大变化"]
                             + (["积分细分数"] if show_divisions else []), rows))
        if show_divisions:
            content.append('<p class="note">积分细分数控制消光数值积分精度，与每 0.5ℓ 的历史更新间隔不同；'
                           '加密积分不会新增历史约束，也不会加快历史点更新。</p>')
        content.append(f'<p>参考网格加密最大变化：<b>{percent(tr["reference_grid_max_delta"])}</b>；'
                       f'最大点态 95% 半宽：<b>{percent(max(tr["ci95"]))}</b>。</p>')
        summary = [esc(label), esc(start_label)]
        if width_study:
            summary.append(number(case_width(case)))
        summary.extend(percent(tr["models"].get(name, {}).get("max_abs_error")) for name in models)
        summary.append(percent(tr["reference_grid_max_delta"]))
        summaries.append((case_family(case), summary))
        for probe_index, probe in enumerate(case["probes"]):
            prefix = f"{slug}-probe-{probe_index + 1}"
            probe_models = tuple(name for name in models if name in probe["models"])
            selections = "；".join(f'{NAMES[name]}：<b>{selection_description(probe, name, knots)}</b>'
                                    for name in probe_models if name != "baseline")
            content.append(f'<h3>固定位置 q = {number(probe["q"])} 的碰撞梯度</h3>'
                           f'<p>{selections}。</p><p>'
                           f'参考样本数 {int(probe["samples"]):,}；速度加权有效样本量 ESS ≈ {number(probe["ess"], 7)}。</p>')
            speed_series = [(name, probe["models"][name]["cdf"]) for name in probe_models]
            speed_series.append(("reference", probe["reference_cdf"]))
            normal_series = [(name, probe["models"][name]["normal_cdf"]) for name in probe_models]
            normal_series.append(("reference", probe["reference_normal_cdf"]))
            content.append('<div class="charts">')
            content.append(figure(chart(probe["speed_grid"], speed_series, title="下穿速度累计分布",
                                        xlabel=speed_unit, ylabel="P(W ≤ w | t 处碰撞)", bounds=(0, 1),
                                        band=(probe["reference_cdf"], probe["reference_cdf_ci95"])),
                                  f"{prefix}-speed.svg"))
            normal_band = ((probe["reference_normal_cdf"], probe["reference_normal_cdf_ci95"])
                           if "reference_normal_cdf_ci95" in probe else None)
            content.append(figure(chart(probe["normal_grid"], normal_series, title="法线入射余弦累计分布",
                                        xlabel="c = −ω·n = V/‖G‖", ylabel="P(cos θ ≤ c)", bounds=(0, 1),
                                        band=normal_band),
                                  f"{prefix}-normal.svg"))
            content.append('</div>')
            reference_q = probe["reference_speed_quantiles"]
            rows = [[NAMES["reference"], "—", "—", "—", number(probe["reference_mean_speed"]),
                     *[number(reference_q[key]) for key in ("p10", "p50", "p90")]]]
            for model in probe_models:
                item = probe["models"][model]
                validation = probe.get("model_mc_validation", {}).get(model, {})
                rows.append([NAMES[model], number(item["ks"]), number(item["normal_ks"]),
                             number(validation.get("max_cdf_difference")),
                             number(item["mean_speed"]),
                             *[number(item["quantiles"][key]) for key in ("p10", "p50", "p90")]])
            content.append(table(["模型", "速度 CDF 最大差", "法线 CDF 最大差", "同模型 MC 核验最大差",
                                  "速度均值", "P10", "P50", "P90"], rows))
            content.append(f'<p class="note">速度单位：{esc(speed_unit)}。CDF 最大差在所绘网格上计算；'
                           f'参考前缀网格加密后的速度 CDF 最大变化为 {number(probe["reference_grid_cdf_max_delta"])}。'
                           '这里比较同一个位置的梯度条件分布，不是整条 flight 中不同碰撞位置的混合分布。</p>')
            content.append('<p class="note">CDF 灰带以 8 组样本的组间波动和自由度 7 的 t 分布近似 95% 点态误差；'
                           '它不是 CDF 最大差（KS 距离）的同时置信区间。同模型 MC 核验使用同批 Fₜ = 0 条件样本，'
                           '分别只保留 0 或 1 个内部历史正号并乘 Rice 正速度权重，再与 C++ 计算的速度 CDF 比较。'
                           '该核验检查概率实现，并非密历史模型误差。</p>')
            normal_deltas = [probe["models"][name]["normal_quadrature_max_delta"] for name in probe_models
                             if "normal_quadrature_max_delta" in probe["models"][name]]
            if normal_deltas:
                content.append(f'<p class="note">各模型法线 CDF 的额外数值求积加密最大变化：{number(max(normal_deltas))}。</p>')
            if width_study:
                for metric, metric_name in (("ks", "速度 CDF"), ("normal_ks", "法线 CDF")):
                    probe_summaries.append((case_family(case),
                                            [esc(label), number(case_width(case)), number(probe["q"]), metric_name,
                                             *[number(probe["models"].get(name, {}).get(metric)) for name in models]]))
        content.append('</section>')
        sections.append("".join(content))

    page = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__REPORT_TITLE__</title>
<style>
body{margin:0;background:#f5f7fb;color:#253145;font:15px/1.75 system-ui,"Microsoft YaHei",sans-serif}
main{max-width:1280px;padding:30px 24px 60px;margin:auto}h1{font-size:27px}h2{font-size:23px;margin-top:0}
h3{font-size:19px;margin-top:32px}a{color:#2269c3}nav{display:flex;gap:20px;flex-wrap:wrap;margin:22px 0}
section,.intro{background:white;border:1px solid #dce3ed;border-radius:9px;padding:24px;margin:24px 0}
p{margin:12px 0}.note{color:#5c6879;font-size:14px}.table-scroll{overflow:auto}
table{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}td,th{text-align:right;padding:10px 12px;border-bottom:1px solid #dce3ed;white-space:nowrap}
td:first-child,th:first-child{text-align:left}th{background:#f4f6fa}figure{margin:20px 0;min-width:0}
svg{display:block;max-width:100%;height:auto;margin:auto}figcaption{text-align:right;font-size:12px}
.charts{display:grid;grid-template-columns:1fr 1fr;gap:16px}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}
@media(max-width:900px){.charts{grid-template-columns:1fr}main{padding:16px 12px}section,.intro{padding:16px}}
</style><main><h1>__REPORT_TITLE__</h1>'''
    page = page.replace("__REPORT_TITLE__", esc(title))
    model_description = ('红线关闭内部历史，蓝线从宏块候选中选一个历史点，绿线从细体素候选中选一个历史点；'
                         if "fine_selected" in models else
                         '红线关闭内部历史，蓝线使用生产实现选出的单个历史点；')
    page += (f'<p>ℓ = {number(settings["ell"])}，σ = {number(settings["sigma"])}；'
             f'配置样本数 {int(settings["samples"]):,}。<a href="{esc(args.results.name)}">完整数据 JSON</a></p>'
             '<div class="intro"><p>本实验比较可控三线性网格剖面，不是 shaderball 全图统计。'
             + model_description + '黑线使用密集前缀的 SE 高斯特征 Monte Carlo 参考。</p>'
             '<p>空域起射始终保留 F₀ &gt; 0；表面起射始终固定 F₀ = 0 和已知未归一化梯度 G₀。'
             '“无内部历史”仍包含各自的起点条件。两种起射方式分别比较。</p>'
             '<p>参考保留有限密度节点上的全部正号，并检查网格加密。它不是已证明的连续路径真值；'
             'Monte Carlo 误差、前缀离散误差和模型透射率积分误差分别报告。</p>'
             '<p>固定位置的梯度参考条件于 Fₜ = 0，使用 V₊ 乘以过去前缀全部为正的指示函数加权。'
             '速度偏置不能省略。在当前固定各向同性 SE 模型中，给定位置和轴向速度后的两个横向分量服从相同 Gaussian；'
             '历史通过轴向速度分布影响完整梯度和法线。法线图显示入射余弦的边缘分布。</p>'
             f'<p>图中 w = ℓV/σ，实际轴向梯度 ω·G = −(σ/ℓ)w = −{number(settings["sigma"] / settings["ell"])}w。'
             '这些可控剖面的横向均值为零；以 σ/ℓ 为单位，空域起射横向标准差为 1，'
             '表面起射为 √(1 − exp(−q²))。</p></div>')
    if width_study:
        page += ('<div class="intro"><p>窄槽中心位于 q = 1.7，平台 M = 0.5。槽宽指两侧槽沿间的完整宽度，'
                 '两种剖面族均分别测试空域和表面起射，图表按剖面族分开。</p>'
                 '<p><b>固定深度族：</b>槽底 M = −0.8，深度为 1.3σ，宽度/ℓ 为 1、0.5、0.2、0.1、0.05。'
                 '槽侧物理斜率为 0.26/(宽度/ℓ)；后三种宽度的槽侧斜率分别为 1.3、2.6、5.2，'
                 '超过欧氏 SDF 的 1-Lipschitz 约束。这些窄而深的剖面属于任意三线性均值压力测试，不能当作真实 SDF 案例。</p>'
                 '<p><b>限制斜率族：</b>宽度/ℓ 为 0.2、0.1、0.05，归一化深度取 5×(宽度/ℓ)，'
                 '槽底 M = 0.5 − 5×(宽度/ℓ)，最大物理斜率为 1。它们与沿射线的 SDF 斜率约束相容，'
                 '但这不证明相应三维三线性场就是精确欧氏 SDF。固定深度族的宽度 1ℓ 和 0.5ℓ 也满足沿线斜率限制。</p>'
                 '<p>细网格间距为 0.0125ℓ；宏块宽为 0.2ℓ（16 个细体素）。'
                 '宏块和细体素方案均每隔 0.5ℓ 重新选点，更新之间固定一个历史点。'
                 '细体素方案扩大候选来源，并没有改成更频繁地更新或同时保留多个历史点。</p>'
                 '<p>窄特征可能落在宏块采样位置之间，导致宏块候选遗漏槽底。'
                 '即使细体素找到了槽底，也可能受固定选点时间、候选评分和单点历史容量限制。'
                 '下表和各位置的所选坐标可区分这些因素；不能只凭槽越窄就预判哪种模型更准。</p>'
                 '<p>本实验固定槽中心 q = 1.7，位于相邻宏块候选 q = 1.65 和 1.75 之间，'
                 '刻意使用不利的采样对齐来暴露遗漏。因此结果不是随机槽位置或宏块相位下的平均性能。'
                 '细体素对照使用相同几何和相同选点时间，用于检查候选覆盖范围造成的差异。</p></div>')
        page += width_observations(data["cases"])
    page += '<h2>透射率差异概览</h2>'
    summary_headers = ["剖面", "起射条件"] + (["槽宽/ℓ"] if width_study else [])
    summary_headers += [NAMES[name] + "最大误差" for name in models] + ["参考网格加密变化"]
    if width_study:
        for family in families:
            page += f'<h3>{esc(family_name(family))}</h3>'
            page += table(summary_headers, [row for row_family, row in summaries if row_family == family])
    else:
        page += table(summary_headers, [row for _, row in summaries])
    if width_study:
        page += '<h2>槽宽与误差</h2><p class="note">T 图为整条测试剖面的最大绝对误差；速度和法线图为所列固定探针位置中最大的 CDF 差，不能代表所有碰撞位置。</p>'
        for family in families:
            page += f'<h3>{esc(family_name(family))}</h3>'
            family_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(family)).strip("-") or "family"
            family_cases = [case for case in data["cases"] if case_family(case) == family]
            for birth in dict.fromkeys(case["start_kind"] for case in family_cases):
                group = sorted((case for case in family_cases if case["start_kind"] == birth and case_width(case) is not None),
                               key=case_width)
                if not group:
                    continue
                widths = [case_width(case) for case in group]
                if len(set(widths)) != len(widths):
                    raise ValueError(f"Duplicate widths within family {family}, birth {birth}")
                kind_name = "表面起射" if "surface" in str(birth).lower() or "gradient" in str(birth).lower() else "空域起射"
                birth_slug = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(birth)).strip("-") or "birth"
                for metric, metric_name in (("transmittance", "透射率最大绝对误差"),
                                            ("ks", "固定探针的最大速度 CDF 差"),
                                            ("normal_ks", "固定探针的最大法线 CDF 差")):
                    curves = []
                    for name in models:
                        if any(name not in case["transmittance"]["models"] or any(name not in probe["models"] for probe in case["probes"])
                               for case in group):
                            continue
                        values = [(case["transmittance"]["models"][name]["max_abs_error"] if metric == "transmittance"
                                   else max(probe["models"][name][metric] for probe in case["probes"])) for case in group]
                        curves.append((name, values))
                    if curves:
                        upper = max(0.001, max(value for _, values in curves for value in values) * 1.1)
                        chart_title = ("限制斜率" if family == "slope_limited" else "固定深度") + " / " + kind_name + "：" + metric_name
                        page += figure(chart(widths, curves, title=chart_title, xlabel="完整槽宽/ℓ", ylabel="绝对差",
                                             bounds=(0, upper)), f"width-{family_slug}-{birth_slug}-{metric}.svg")
        page += '<h2>各固定位置的梯度分布差异</h2>'
        for family in families:
            page += f'<h3>{esc(family_name(family))}</h3>'
            page += table(["剖面", "槽宽/ℓ", "探针 q", "分布"] + [NAMES[name] + "最大差" for name in models],
                          [row for row_family, row in probe_summaries if row_family == family])
    page += '<nav>' + "".join(navigation) + '</nav>' + "".join(sections)
    page += ('<details><summary>实验参数</summary><pre>'
             + esc(json.dumps(settings, ensure_ascii=False, indent=2)) + '</pre></details></main></html>')
    destination = root / "comparison.html"
    destination.write_text(page, encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
