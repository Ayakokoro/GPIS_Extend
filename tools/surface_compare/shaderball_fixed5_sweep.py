"""Render matched shaderball parameter sweeps and screen fixed5 differences.

python -B tools/surface_compare/shaderball_fixed5_sweep.py run
python -B tools/surface_compare/shaderball_fixed5_sweep.py report

All primary images use the requested resolution and spp. Independent seed
confirmation precedes optional native 3D GPIS renewal+ renders. No renderer
source or previous experiment is changed.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import datetime as dt
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
# Reuse the existing local plotting environment without installing packages.
sys.path.insert(0, str(ROOT / "build/fp-gradient-validation-deps"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

SOURCE = ROOT / "outputs/conditional-fp-gradient-shaderball-l0p2-s0p02-256-128spp-20261108/baseline/scene.json"
RENEWAL_SOURCE = ROOT / "outputs/renewal-plus-step00125-shaderball-l0p2-s0p02-256-128spp-20261109/shaderball-renewal-plus-step00125.json"
DEFAULT = ROOT / "outputs/shaderball-fixed5-sweep-128-64spp-20261110"
GROUPS = ((.2, .02), (.2, .04), (.2, .08), (.4, .04), (.4, .08), (.1, .04), (.1, .02), (.05, .01))
MODES = ("classic_global", "global_conditional", "conditional_fixed5")
LABELS = dict(classic_global="classic_global", global_conditional="global_conditional",
              conditional_fixed5="global_conditional + fixed5", renewal_plus="GPIS renewal+ (3D)")
Y = np.array([.2126, .7152, .0722])
# Bounds refer to the existing 256px camera, fixed before this experiment.
REGIONS = dict(groove=(54, 25, 188, 146), base=(49, 146, 205, 238), object=(36, 24, 220, 241))
ERRORS = re.compile(r"Caught an internal error|unrecoverable error|Unable to load|Wrong normal direction|"
                    r"Gradient invalid|Failed to sample gradient|terminate called|invalid FP|"
                    r"Clamp one (?:inf|nan) sample", re.I)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def key(ell, sigma):
    tag = lambda x: format(x, ".7g").replace(".", "p")
    return f"ell{tag(ell)}_sigma{tag(sigma)}"


def primary_seed(output, default_seed, ell, sigma, mode):
    plan = Path(output)/"seed-overrides.json"
    overrides = read(plan).get("seeds", {}) if plan.exists() else {}
    return int(overrides.get(f"{key(ell,sigma)}/{mode}", default_seed))


def pfm(path):
    with Path(path).open("rb") as stream:
        if stream.readline().strip() != b"PF":
            raise ValueError(f"Expected RGB PFM: {path}")
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        values = np.frombuffer(stream.read(), dtype="<f4" if scale < 0 else ">f4")
    rgb = values.reshape(height, width, 3)[::-1].astype(float) * abs(scale)
    if not np.isfinite(rgb).all() or rgb.min() < -1e-6:
        raise ValueError(f"Invalid pixels: {path}")
    return rgb


def filmic(rgb):
    x = np.maximum(rgb - .004, 0)
    return np.clip(x * (6.2*x + .5) / (x * (6.2*x + 1.7) + .06), 0, 1)


def png(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.rint(np.clip(rgb, 0, 1)*255).astype("uint8")).save(path)


def bounds(name, size):
    return tuple(round(x * size / 256) for x in REGIONS[name])


def blocks(rgb, region):
    """Fixed 4x4 pixel spatial blocks, retaining covariance within each block."""
    x0, y0, x1, y1 = bounds(region, rgb.shape[0])
    a = (rgb @ Y)[y0:y1, x0:x1]
    h, w = (n//4*4 for n in a.shape)
    return a[:h, :w].reshape(h//4, 4, w//4, 4).mean(axis=(1, 3))


def compare(first, second):
    """Positive difference means second is brighter. A/B are half samples."""
    result = {}
    for name in REGIONS:
        a, b = blocks(first[0], name), blocks(second[0], name)
        da = blocks(second[1], name) - blocks(first[1], name)
        db = blocks(second[2], name) - blocks(first[2], name)
        difference = b-a
        reference = max(float((a.mean()+b.mean())/2), 1e-12)
        mse = float(np.mean(difference**2))
        noise = float(np.mean(((da-db)/2)**2))
        result[name] = dict(mean_first=float(a.mean()), mean_second=float(b.mean()),
            relative_mean_change=float(difference.mean()/reference),
            raw_relative_block_rms=math.sqrt(mse)/reference,
            estimated_noise_relative_block_rms=math.sqrt(noise)/reference,
            noise_subtracted_relative_block_rms=math.sqrt(max(0., mse-noise))/reference,
            signed_half_cross_product=float(np.mean(da*db)),
            pixel_bounds=list(bounds(name, first[0].shape[0])))
    return result


def scene_for(args, ell, sigma, mode):
    scene = copy.deepcopy(read(SOURCE))
    medium = scene["media"][0]
    medium["gaussian_process"]["covariance"].update(lengthScale=ell, sigma=sigma)
    medium.pop("fp_gradient", None)
    if mode == "renewal_plus":
        renewal = copy.deepcopy(read(RENEWAL_SOURCE)["media"][0])
        renewal["gaussian_process"] = copy.deepcopy(medium["gaussian_process"])
        renewal["phase_function"]["bsdf"] = copy.deepcopy(medium["material"])
        # A pilot numerical setting, NOT a statement of continuous-GP accuracy.
        renewal["step_size"] = min(.0025, ell/40, sigma/8)
        scene["media"] = [renewal]
    elif mode == "classic_global":
        medium["type"] = "classic_global"
    else:
        medium["type"] = "global_conditional"
        medium["fp_gradient"] = dict(enabled=mode == "conditional_fixed5",
            bridge_correction=True, anchor_selection="fixed5_snr")
    scene["camera"]["resolution"] = [args.size, args.size]
    scene["integrator"].update(use_cuda_gpis=False, wavefront_gpis=False, independent_path_sampling=True)
    scene["renderer"].update(output_directory="", output_file="render.png", hdr_output_file="render.pfm",
        resume_render_file="", enable_resume_render=False, overwrite_output_files=True,
        adaptive_sampling=False, stratified_sampler=False, spp=args.spp, spp_step=8,
        checkpoint_interval="0", timeout="0", output_buffers=[dict(type="color",
            hdr_output_file="samples.pfm", sample_variance=True, two_buffer_variance=True)])
    interval = getattr(args, "checkpoint_interval", "0")
    if interval != "0":
        scene["renderer"].update(resume_render_file="resume.dat", enable_resume_render=True,
                                 checkpoint_interval=interval)
    return scene


def snapshot(args):
    folder = args.output / "renderer"
    source = ROOT / "build/Release"
    paths = [source/"tungsten.exe", *sorted(source.glob("*.dll"))]
    hashes = {p.name: digest(p) for p in paths}
    if (args.output/"manifest.json").exists():
        existing = read(args.output/"manifest.json")
        settings = dict(size=args.size, spp=args.spp, seed=args.seed, confirmation_seed=args.confirmation_seed,
                        threshold=args.threshold, groups=[list(g) for g in GROUPS])
        if getattr(args, "checkpoint_interval", "0") != "0":
            settings["checkpoint_interval"] = args.checkpoint_interval
        if existing["settings"] != settings:
            raise ValueError("Existing experiment has different settings; use a new --output.")
        for name, expected in existing["runtime_sha256"].items():
            if digest(folder/name) != expected:
                raise ValueError(f"Changed frozen runtime: {name}")
        if existing["source_sha256"] != digest(SOURCE) or existing["renewal_source_sha256"] != digest(RENEWAL_SOURCE):
            raise ValueError("Source scene changed since experiment creation")
        if "seed_overrides" in existing:
            plan = existing["seed_overrides"]
            if digest(args.output/plan["file"]) != plan["sha256"]:
                raise ValueError("Seed retry plan changed without a provenance update")
        return existing
    folder.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, folder/path.name)
    manifest = dict(created_at=now(), settings=dict(size=args.size, spp=args.spp, seed=args.seed,
        confirmation_seed=args.confirmation_seed, threshold=args.threshold, groups=[list(g) for g in GROUPS]),
        source=str(SOURCE.relative_to(ROOT)), source_sha256=digest(SOURCE),
        renewal_source=str(RENEWAL_SOURCE.relative_to(ROOT)), renewal_source_sha256=digest(RENEWAL_SOURCE),
        runtime_sha256=hashes, modes=list(MODES), regions_256px=REGIONS,
        sampling=f"Independent per-path sampling; shared render seeds; {args.spp//2}+{args.spp//2} parity half buffers at {args.spp} spp",
        screening="4x4 block luminance RMS after subtracting half-buffer noise power, normalized by pair mean; any preset region >= threshold",
        confirmation="Same preset region must have cross-seed signal RMS >= threshold; positive cross product is required",
        renewal="Native 3D sparse convolution, renewal+, impulse density 10; step=min(.0025,ell/40,sigma/8). Finite-step comparison, not exact GP truth.")
    if getattr(args, "checkpoint_interval", "0") != "0":
        manifest["settings"]["checkpoint_interval"] = args.checkpoint_interval
    save(args.output/"manifest.json", manifest)
    return manifest


def render_one(args, runtime_hash, ell, sigma, mode, seed):
    folder = args.output/key(ell, sigma)/mode/f"seed{seed}"
    scene = scene_for(args, ell, sigma, mode)
    identity = dict(scene=scene, seed=seed, renderer_sha256=runtime_hash)
    if (folder/"result.json").exists():
        record = read(folder/"result.json")
        if record["identity"] != identity:
            raise ValueError(f"Incompatible completed result: {folder}")
        for name, expected in record["sha256"].items():
            if digest(folder/name) != expected:
                raise ValueError(f"Changed output: {folder/name}")
        return record
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/"scene.json").exists() and read(folder/"scene.json") != scene:
        raise ValueError(f"Incompatible partial scene: {folder}")
    if (folder/"job.json").exists():
        previous = read(folder/"job.json")
        if previous.get("renderer_sha256", runtime_hash) != runtime_hash:
            raise ValueError(f"Partial scene used a different renderer: {folder}")
    if (folder/"render.log").exists():
        archive = folder/"attempts"
        archive.mkdir(exist_ok=True)
        shutil.copy2(folder/"render.log", archive/f"{time.time_ns()}.log")
    save(folder/"scene.json", scene)
    command = [str(args.output/"renderer/tungsten.exe"), "-t", str(args.threads), "-s", str(seed),
               "-d", str(folder), str(folder/"scene.json")]
    print(f"START {key(ell,sigma)} {mode} seed={seed}", flush=True)
    started = time.monotonic()
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) | getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
    with (folder/"render.log").open("wb") as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)
        save(folder/"job.json", dict(command=command, process_id=process.pid, started_at=now(),
                                   renderer_sha256=runtime_hash))
        code = process.wait()
    elapsed = time.monotonic()-started
    text = (folder/"render.log").read_text(encoding="utf-8", errors="replace")
    if code or ERRORS.search(text) or "Finished render." not in text or not re.search(rf"Completed\s+{args.spp}/{args.spp}\s+spp", text):
        save(folder/"failure.json", dict(exit_code=code, seconds=elapsed, last_log=text[-4000:], identity=identity))
        raise RuntimeError(f"Render failed: {folder}; exit={code}")
    rgb, a, b, variance = (pfm(folder/name) for name in ("render.pfm", "samplesA.pfm", "samplesB.pfm", "samplesVariance.pfm"))
    if rgb.shape != (args.size, args.size, 3) or not np.allclose((a+b)/2, rgb, rtol=4e-6, atol=4e-7):
        raise ValueError(f"Invalid dimensions or half-buffer average: {folder}")
    if rgb.max() <= 0 or variance.shape != rgb.shape:
        raise ValueError(f"Empty image or invalid variance: {folder}")
    counters = {}
    for label, pattern in [("gradient", r"FP-gradient statistics: (\{[^\n]+\})"),
                           ("discarded_surface_births", r"discarded surface births: (\{[^\n]+\})")]:
        match = re.search(pattern, text)
        if match:
            counters[label] = json.loads(match[1])
    files = ("scene.json", "render.png", "render.pfm", "samplesA.pfm", "samplesB.pfm", "samplesVariance.pfm", "render.log")
    record = dict(ell=ell, sigma=sigma, mode=mode, seed=seed, folder=str(folder.relative_to(args.output)),
        identity=identity, command=command, seconds=elapsed, finished_at=now(), counters=counters,
        mean_luminance=float((rgb@Y).mean()), sha256={name:digest(folder/name) for name in files})
    save(folder/"result.json", record)
    print(f"DONE {key(ell,sigma)} {mode} seed={seed} {elapsed:.1f}s", flush=True)
    return record


def run_jobs(args, jobs, stage):
    runtime_hash = read(args.output/"manifest.json")["runtime_sha256"]["tungsten.exe"]
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = {pool.submit(render_one, args, runtime_hash, *job): job for job in jobs}
        done = 0
        while futures:
            ready, _ = concurrent.futures.wait(futures, timeout=15, return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                job = futures.pop(future)
                done += 1
                try:
                    future.result()
                except Exception as exc:
                    failures.append(dict(job=list(job), error=str(exc)))
                    print(f"FAILED {job}: {exc}", flush=True)
            if ready and getattr(args, "flip_exe", None):
                report(args)
            save(args.output/"progress.json", dict(stage=stage, completed_jobs=done, total_jobs=len(jobs),
                failures=failures, updated_at=now()))
    if failures:
        save(args.output/f"{stage}-failures.json", failures)
        print(f"{stage}: {len(failures)} failed jobs", flush=True)
    return failures


def image_set(args, ell, sigma, mode, seed):
    folder = args.output/key(ell, sigma)/mode/f"seed{seed}"
    return tuple(pfm(folder/name) for name in ("render.pfm", "samplesA.pfm", "samplesB.pfm"))


def analyze(args):
    groups = []
    for ell, sigma in GROUPS:
        row = dict(ell=ell, sigma=sigma, key=key(ell, sigma), comparisons={}, screened_regions=[],
                   confirmed_regions=[], renewal_selected=False, completed={}, render_seeds={})
        images = {}
        for mode in (*MODES, "renewal_plus"):
            seed = primary_seed(args.output,args.seed,ell,sigma,mode)
            row["render_seeds"][mode] = seed
            folder = args.output/key(ell, sigma)/mode/f"seed{seed}"
            row["completed"][mode] = (folder/"result.json").exists()
            if row["completed"][mode]:
                images[mode] = image_set(args, ell, sigma, mode, seed)
        if "global_conditional" in images and "conditional_fixed5" in images:
            row["comparisons"]["fixed5_vs_conditional"] = compare(images["global_conditional"], images["conditional_fixed5"])
            row["screened_regions"] = [name for name, stats in row["comparisons"]["fixed5_vs_conditional"].items()
                if stats["noise_subtracted_relative_block_rms"] >= args.threshold]
            if all((args.output/key(ell, sigma)/mode/f"seed{args.confirmation_seed}/result.json").exists()
                   for mode in ("global_conditional", "conditional_fixed5")):
                c2 = image_set(args, ell, sigma, "global_conditional", args.confirmation_seed)
                f2 = image_set(args, ell, sigma, "conditional_fixed5", args.confirmation_seed)
                row["confirmation"] = compare(c2, f2)
                row["cross_seed"] = {}
                for name in REGIONS:
                    d1 = blocks(images["conditional_fixed5"][0], name)-blocks(images["global_conditional"][0], name)
                    d2 = blocks(f2[0], name)-blocks(c2[0], name)
                    cross = float(np.mean(d1*d2))
                    reference = np.mean([blocks(im, name).mean() for im in
                        (images["conditional_fixed5"][0], images["global_conditional"][0], c2[0], f2[0])])
                    rms = math.sqrt(max(0., cross))/max(float(reference), 1e-12)
                    row["cross_seed"][name] = dict(relative_signal_rms=rms, signed_cross_power=cross,
                        relative_mean_change_seed1=float(d1.mean()/reference), relative_mean_change_seed2=float(d2.mean()/reference))
                    if name in row["screened_regions"] and rms >= args.threshold:
                        row["confirmed_regions"].append(name)
                row["renewal_selected"] = bool(row["confirmed_regions"])
        if "classic_global" in images and "global_conditional" in images:
            row["comparisons"]["conditional_vs_classic"] = compare(images["classic_global"], images["global_conditional"])
        if "classic_global" in images and "conditional_fixed5" in images:
            row["comparisons"]["fixed5_vs_classic"] = compare(images["classic_global"], images["conditional_fixed5"])
        if "renewal_plus" in images:
            for mode in MODES:
                if mode in images:
                    row["comparisons"][f"{mode}_vs_renewal"] = compare(images["renewal_plus"], images[mode])
        groups.append(row)
    save(args.output/"summary.json", dict(updated_at=now(), groups=groups, threshold=args.threshold,
        note="Pilot screening estimates include sampling uncertainty. Noise subtraction and cross-seed power are diagnostic estimates, not confidence bounds or exact-GP bias measurements."))
    return groups


def report(args):
    groups = analyze(args)
    status_link = "live_status.json" if (args.output/"live_status.json").exists() else "progress.json"
    retry_note = ""
    if (args.output/"seed-overrides.json").exists():
        plan = read(args.output/"seed-overrides.json")
        details = "；".join(f"{name} → seed={seed}" for name,seed in plan["seeds"].items())
        failures = plan.get("failed_attempts", [plan.get("failed_attempt", "")])
        links = ''.join(f' · <a href="{html.escape(name)}">原失败记录 {i+1}</a>'
                        for i,name in enumerate(failures) if name)
        retry_note = (f'<p>数值异常后的独立种子复跑：{html.escape(details)}。场景、物理参数、分辨率、spp 和渲染器保持一致。'
                      f'<a href="seed-overrides.json">复跑记录</a>' +
                      links + '</p>')
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 16)
    has_renewal = any(row["completed"]["renewal_plus"] for row in groups)
    contact = Image.new("RGB", ((5 if has_renewal else 4)*276, len(groups)*306+48), "#141b24")
    draw = ImageDraw.Draw(contact)
    headers = [*map(LABELS.get, MODES), "fixed5 - conditional"]
    if has_renewal:
        headers.append(LABELS["renewal_plus"])
    for i, title in enumerate(headers):
        draw.text((i*276+8, 10), title, font=font, fill="white")
    rows, sections = [], []
    for index, row in enumerate(groups):
        ell, sigma = row["ell"], row["sigma"]
        images, cards = {}, []
        caption = f"ell={ell:g}, sigma={sigma:g}, slope={sigma/ell:g}"
        for col, mode in enumerate((*MODES, "renewal_plus")):
            if not row["completed"][mode]:
                continue
            seed = row["render_seeds"][mode]
            rgb = image_set(args, ell, sigma, mode, seed)[0]
            images[mode] = rgb
            folder = args.output/row["key"]/mode/f"seed{seed}"
            display = folder/"display.png"
            png(display, filmic(rgb))
            slot = 4 if mode == "renewal_plus" else col
            with Image.open(display) as image:
                contact.paste(image.resize((256, 256), Image.Resampling.NEAREST), (slot*276+8, 48+index*306))
            draw.text((slot*276+8, 307+index*306), caption, font=font, fill="white")
            link = folder.relative_to(args.output).as_posix()
            region = bounds("groove", args.size)
            crop = folder/"groove.png"
            with Image.open(display) as image:
                image.crop(region).resize((268, 242), Image.Resampling.NEAREST).save(crop)
            cards.append(f'<article><b>{html.escape(LABELS[mode])}</b><p>seed={seed}</p><a href="{link}/display.png"><img src="{link}/display.png"></a>'
                         f'<img class="crop" src="{link}/groove.png"><p><a href="{link}/scene.json">配置</a> · '
                         f'<a href="{link}/render.pfm">线性 PFM</a> · <a href="{link}/render.log">日志</a></p></article>')
        if "global_conditional" in images and "conditional_fixed5" in images:
            difference = (images["conditional_fixed5"]-images["global_conditional"])@Y
            limit = .1
            v = np.clip(difference/limit, -1, 1)[..., None]
            colored = np.where(v<0, 1+v*(1-np.array([.12,.35,.8])), 1-v*(1-np.array([.9,.25,.08])))
            path = args.output/row["key"]/"fixed5-minus-conditional.png"
            png(path, colored)
            with Image.open(path) as image:
                contact.paste(image.resize((256,256), Image.Resampling.NEAREST), (3*276+8,48+index*306))
            draw.text((3*276+8,307+index*306), caption, font=font, fill="white")
            cards.append(f'<article><b>fixed5 − conditional</b><img src="{row["key"]}/fixed5-minus-conditional.png"><p>线性亮度差：蓝 −0.1，白 0，橙 +0.1；包含采样噪声。</p></article>')
        stats = row["comparisons"].get("fixed5_vs_conditional", {})
        scores = [stats[n]["noise_subtracted_relative_block_rms"] for n in REGIONS] if stats else [0]*3
        state = "已补 renewal+" if row["completed"]["renewal_plus"] else "复核通过" if row["renewal_selected"] else "已复核，未达阈值" if "cross_seed" in row else "待独立种子复核" if row["screened_regions"] else "未达预筛阈值" if stats else "渲染中／失败，见日志"
        rows.append(f'<tr><td><a href="#{row["key"]}">{ell:g} / {sigma:g}</a></td>' + ''.join(f'<td>{100*s:.2f}%</td>' for s in scores) + f'<td>{state}</td></tr>')
        details = html.escape(json.dumps(dict(comparisons=row["comparisons"], cross_seed=row.get("cross_seed")), ensure_ascii=False, indent=2))
        sections.append(f'<section id="{row["key"]}"><h2>ℓ={ell:g} · σ={sigma:g} · σ/ℓ={sigma/ell:g}</h2><p>{state}</p>'
                        f'<div class="cards">{"".join(cards)}</div><details><summary>线性统计与噪声估计</summary><pre>{details}</pre></details></section>')
    contact.save(args.output/"comparison.png")
    document = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Shaderball 参数预览 · {args.size}px / {args.spp}spp</title>
<style>body{{max-width:1600px;margin:24px auto;padding:0 20px;background:#141b24;color:#e3e9f1;font:15px/1.65 system-ui,"Microsoft YaHei",sans-serif}}a{{color:#8dd9e0}}section,.intro{{background:#202b38;padding:20px;margin:24px 0;border-radius:10px}}.cards{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px}}article img{{display:block;width:100%;image-rendering:pixelated;margin:12px 0}}.crop{{max-height:242px;object-fit:contain}}table{{border-collapse:collapse;width:100%}}td,th{{padding:9px;border-bottom:1px solid #425063;text-align:left}}pre{{overflow:auto;font-size:12px}}summary{{cursor:pointer}}</style>
<h1>Shaderball：classic_global / global_conditional / fixed5</h1>
<p>{args.size}×{args.size} · 每张 {args.spp} spp · 默认主 seed={args.seed}，实际 seed 见各图 · 复核用 seed={args.confirmation_seed}（仅预筛通过者） · 相同相机、灯光、材质与路径深度。</p>
{retry_note}
<p><a href="comparison.png">整组并排 PNG</a> · <a href="summary.json">统计 JSON</a> · <a href="manifest.json">实验设置与程序哈希</a> · <a href="{status_link}">实时进度</a> · <a href="validation.json">输出校验</a></p>
<div class="intro"><p>下表为 fixed5 相对 conditional 的 4×4 分块亮度差，按两者区域平均亮度归一化，并用两份半样本图估计和扣除噪声功率。预筛阈值 {100*args.threshold:g}%；通过者以独立 seed 复核相同区域的跨种子信号功率，再决定是否补跑原生 3D GPIS renewal+。</p>
<p>凹槽矩形沿用已有实验的预设区域。所有展示图统一使用 filmic 映射，无单图曝光调整。表中 0 表示信号功率估计不高于噪声、截为 0，不表示两个模式完全一致。小图用于筛选，统计含不确定性，不能视为收敛结论。renewal+ 使用有限密度稀疏卷积和离散步长，并非连续 GP 的精确真值。</p>
<table><tr><th>ℓ / σ</th><th>凹槽估计 RMS</th><th>底座估计 RMS</th><th>物体区域估计 RMS</th><th>状态</th></tr>{''.join(rows)}</table></div>{''.join(sections)}</html>'''
    (args.output/"comparison.html").write_text(document, encoding="utf-8")
    if getattr(args, "flip_exe", None):
        from shaderball_fixed5_flip import build as build_flip
        build_flip(args.output, args.flip_exe.resolve(), args.ppd)
    return groups


def validate(args):
    """Audit the actual outputs and control variables after the queue finishes."""
    manifest = read(args.output/"manifest.json")
    runtime_hash = manifest["runtime_sha256"]["tungsten.exe"]
    for name, expected in manifest["runtime_sha256"].items():
        assert digest(args.output/"renderer"/name) == expected, name
    if "seed_overrides" in manifest:
        plan=manifest["seed_overrides"]
        assert digest(args.output/plan["file"])==plan["sha256"]
    records, missing = [], []
    common_scene = common_medium = None
    for ell, sigma in GROUPS:
        for mode in MODES:
            seed = primary_seed(args.output,args.seed,ell,sigma,mode)
            folder = args.output/key(ell, sigma)/mode/f"seed{seed}"
            if not (folder/"result.json").exists():
                missing.append(dict(ell=ell,sigma=sigma,mode=mode,seed=seed,
                    folder=str(folder.relative_to(args.output)),
                    state="failed" if (folder/"failure.json").exists() else "unfinished"))
                if getattr(args,"allow_incomplete",False):
                    continue
                raise RuntimeError(f"Missing validated primary render: {folder}")
            result = read(folder/"result.json")
            scene = read(folder/"scene.json")
            assert result["identity"] == dict(scene=scene, seed=seed, renderer_sha256=runtime_hash)
            assert scene["camera"]["resolution"] == [args.size, args.size]
            assert scene["renderer"]["spp"] == args.spp
            assert scene["integrator"]["independent_path_sampling"]
            nonmedium = {k:v for k,v in scene.items() if k != "media"}
            if common_scene is None:
                common_scene = nonmedium
            assert nonmedium == common_scene, (ell, sigma, mode, "non-medium settings")
            medium = copy.deepcopy(scene["media"][0])
            cov = medium["gaussian_process"]["covariance"]
            assert cov.pop("lengthScale") == ell and cov.pop("sigma") == sigma
            medium_type = medium.pop("type")
            fp = medium.pop("fp_gradient", None)
            assert medium_type == ("classic_global" if mode == "classic_global" else "global_conditional")
            if mode == "classic_global":
                assert fp is None
            else:
                assert fp == dict(enabled=mode == "conditional_fixed5", bridge_correction=True, anchor_selection="fixed5_snr")
            if common_medium is None:
                common_medium = medium
            assert medium == common_medium, (ell, sigma, mode, "other medium settings")
            for name, expected in result["sha256"].items():
                assert digest(folder/name) == expected, str(folder/name)
            records.append(dict(folder=result["folder"], seconds=result["seconds"], counters=result["counters"]))
    # Also audit any completed confirmation/reference renders.
    all_results = sorted(args.output.glob("*/*/seed*/result.json"))
    for path in all_results:
        result = read(path)
        assert result["identity"]["renderer_sha256"] == runtime_hash
        for name, expected in result["sha256"].items():
            assert digest(path.parent/name) == expected, str(path.parent/name)
    groups = report(args)
    focused = args.output/"focused-tests.log"
    focused_passes = focused.read_text(encoding="utf-8", errors="replace").count("Test Passed.") if focused.exists() else None
    if focused_passes is not None:
        assert focused_passes == 11, focused_passes
    validation = dict(validated_at=now(), all_pass=not missing, complete=not missing,
        available_outputs_pass=bool(records), missing_primary=missing, primary_renders=len(records),
        total_renders=len(all_results), resolution=[args.size, args.size], spp_per_image=args.spp,
        focused_tests_passed=focused_passes, renderer_sha256=runtime_hash,
        script_sha256=digest(Path(__file__)), control_variables_identical=True, output_hashes_verified=True,
        primary_records=records, selected_renewal_groups=[g["key"] for g in groups if g["renewal_selected"]],
        report_sha256={name:digest(args.output/name) for name in ("summary.json", "comparison.html", "comparison.png")})
    if (args.output/"seed-overrides.json").exists():
        validation["seed_overrides"] = read(args.output/"seed-overrides.json")
    save(args.output/"validation.json", validation)
    print(json.dumps({k:v for k,v in validation.items() if k not in ("primary_records", "report_sha256")}, ensure_ascii=False), flush=True)


def finish(args, failures=None):
    """Finish screening after external/queued primary render attempts end."""
    queue_finished=failures is not None
    failures=list(failures or [])
    known={tuple(f["job"]) for f in failures if "job" in f}
    for ell,sigma in GROUPS:
        for mode in MODES:
            seed=primary_seed(args.output,args.seed,ell,sigma,mode)
            folder=args.output/key(ell,sigma)/mode/f"seed{seed}"
            if (folder/"result.json").exists():
                continue
            if not queue_finished and not (folder/"failure.json").exists():
                raise RuntimeError(f"Cannot finalize an unfinished primary attempt: {folder}")
            job=(ell,sigma,mode,seed)
            if job not in known:
                failures.append(dict(job=list(job),error="Primary render has no validated final image",
                                     folder=str(folder.relative_to(args.output))))
    groups=report(args)
    candidates=[g for g in groups if g["screened_regions"]]
    save(args.output/"screening.json",dict(selected=[dict(ell=g["ell"],sigma=g["sigma"],
        regions=g["screened_regions"]) for g in candidates],threshold=args.threshold))
    jobs=[(g["ell"],g["sigma"],mode,args.confirmation_seed) for g in candidates
          for mode in ("global_conditional","conditional_fixed5")]
    if jobs:
        failures+=run_jobs(args,jobs,"confirmation")
    groups=report(args)
    selected=[g for g in groups if g["renewal_selected"]]
    save(args.output/"renewal-selection.json",dict(selected=[dict(ell=g["ell"],sigma=g["sigma"],
        regions=g["confirmed_regions"]) for g in selected],skipped=args.skip_renewal))
    if selected and not args.skip_renewal:
        jobs=[(g["ell"],g["sigma"],"renewal_plus",args.seed) for g in selected]
        failures+=run_jobs(args,jobs,"renewal")
    groups=report(args)
    save(args.output/"progress.json",dict(stage="complete" if not failures else "complete_with_failures",
        primary_completed=sum(sum(g["completed"][m] for m in MODES) for g in groups),
        primary_total=len(GROUPS)*len(MODES),confirmation_groups=len(candidates),
        renewal_groups=len(selected),failures=failures,updated_at=now()))
    print(json.dumps(dict(report=str(args.output/"comparison.html"),failed=len(failures),
        confirmation_groups=len(candidates),renewal_groups=len(selected)),ensure_ascii=False),flush=True)
    if failures:
        raise SystemExit(1)


def status(args):
    """Read live spp without changing the renderer queue's progress file."""
    progress = read(args.output/"progress.json") if (args.output/"progress.json").exists() else {}
    active, failed = [], []
    for path in sorted(args.output.glob("ell*/*/seed*/job.json")):
        if (path.parent/"result.json").exists():
            continue
        if (path.parent/"failure.json").exists():
            failed.append(str(path.parent.relative_to(args.output)))
            continue
        job = read(path)
        log = (path.parent/"render.log").read_text(encoding="utf-8", errors="replace")
        matches = re.findall(r"Completed (\d+)/(\d+) spp, ([\d.]+)/([\d.]+) sec, remaining ([\d.]+) sec", log)
        entry = dict(folder=str(path.parent.relative_to(args.output)), process_id=job["process_id"],
                     completed_spp=0, started_at=job["started_at"])
        if matches:
            done, total, elapsed, estimate, remaining = matches[-1]
            entry.update(completed_spp=int(done), total_spp=int(total), render_seconds=float(elapsed),
                         estimated_remaining_seconds=float(remaining))
        entry["checkpoint_saved"] = (path.parent/"resume.dat").exists()
        active.append(entry)
    flip_summary = args.output/"flip/summary.json"
    pairs = read(flip_summary).get("primary_pairs_completed", 0) if flip_summary.exists() else 0
    settings = read(args.output/"manifest.json")["settings"]
    completed = sum((args.output/key(ell,sigma)/mode/f"seed{primary_seed(args.output,settings['seed'],ell,sigma,mode)}"/"result.json").exists()
                    for ell,sigma in GROUPS for mode in MODES)
    if progress.get("stage")=="primary":
        # The live snapshot counts validated files, including external retries.
        progress["completed_jobs"]=completed
        progress["updated_at"]=now()
    result = dict(updated_at=now(), queue=progress, active=active, flip_pairs=pairs,
                  primary_completed=completed, failed_attempts=failed)
    save(args.output/"live_status.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "report", "validate", "status", "finalize"))
    parser.add_argument("--output", type=Path, default=DEFAULT)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--spp", type=int, default=64)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--confirmation-seed", type=int, default=1041)
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=.05)
    parser.add_argument("--skip-renewal", action="store_true")
    parser.add_argument("--checkpoint-interval", default="0")
    parser.add_argument("--flip-exe", type=Path)
    parser.add_argument("--ppd", type=float, default=67.)
    parser.add_argument("--allow-incomplete",action="store_true")
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.size < 32 or args.size % 16 or args.spp < 8 or args.spp % 8 or min(args.jobs, args.threads) < 1:
        parser.error("size must be a positive multiple of 16 >=32; spp a multiple of 8; jobs/threads positive")
    if args.seed == args.confirmation_seed or not 0 < args.threshold < 1:
        parser.error("distinct seeds and threshold between 0 and 1 required")
    if args.command == "report":
        report(args)
        return
    if args.command == "status":
        status(args)
        return
    if args.command == "validate":
        validate(args)
        return
    snapshot(args)
    if args.command=="finalize":
        finish(args)
        return
    report(args)
    jobs = [(ell, sigma, mode, primary_seed(args.output,args.seed,ell,sigma,mode))
            for ell, sigma in GROUPS for mode in MODES]
    failures = run_jobs(args, jobs, "primary")
    finish(args,failures)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
