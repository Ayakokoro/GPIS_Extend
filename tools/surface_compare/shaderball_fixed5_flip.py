"""Native local FLIP maps for the three-mode shaderball parameter sweep.

python -B tools/surface_compare/shaderball_fixed5_flip.py --report <experiment>
The executable produces the heatmaps; its matching local Python API exports
scalar maps for full-image and fixed-ROI statistics. No image is resized before
FLIP evaluation. Partial reports can be updated while the renderer queue runs.
"""
from __future__ import annotations

import argparse
import csv
import html
from importlib.metadata import version
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"build/fp-gradient-validation-deps"))
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import flip_evaluator as flip
import shaderball_fixed5_sweep as sweep

DEFAULT_FLIP = Path("B:/Python/Python310/Scripts/flip.exe")
PAIRS = (("conditional_vs_classic", "classic_global", "global_conditional"),
         ("fixed5_vs_classic", "classic_global", "conditional_fixed5"),
         ("fixed5_vs_conditional", "global_conditional", "conditional_fixed5"))
PAIR_LABELS = dict(conditional_vs_classic="conditional / classic",
                   fixed5_vs_classic="fixed5 / classic", fixed5_vs_conditional="fixed5 / conditional")
FIELDS = {"Mean":"mean", "Weighted median":"weighted_median", "1st weighted quartile":"weighted_q1",
          "3rd weighted quartile":"weighted_q3", "Min":"minimum", "Max":"maximum"}
BEGIN, END = "<!-- FIXED5 FLIP BEGIN -->", "<!-- FIXED5 FLIP END -->"


def tool_identity(executable):
    folder = Path(flip.__file__).parent
    files = [executable, folder/"flip_python_api.py", *folder.glob("nbflip*.pyd")]
    if len(files) != 3:
        raise RuntimeError("Expected one matching local FLIP native extension")
    return dict(version=version("flip-evaluator"), executable=str(executable),
                files_sha256={str(p):sweep.digest(p) for p in files})


def evaluate(executable, tool, reference, candidate, dest, ppd):
    dest.mkdir(parents=True, exist_ok=True)
    command = [str(executable), "-r", str(reference), "-t", str(candidate), "-ppd", str(ppd),
               "-d", str(dest), "-txt"]
    identity = dict(tool=tool, command=command, reference_sha256=sweep.digest(reference),
                    candidate_sha256=sweep.digest(candidate), ppd=ppd)
    result_path = dest/"result.json"
    if result_path.exists():
        result = sweep.read(result_path)
        if result["identity"] != identity:
            raise ValueError(f"Changed FLIP input or tool: {dest}")
        for name, expected in result["sha256"].items():
            if sweep.digest(dest/name) != expected:
                raise ValueError(f"Changed FLIP artifact: {dest/name}")
        return result
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
        encoding="utf-8", errors="replace", creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    (dest/"flip.log").write_text(completed.stdout+completed.stderr, encoding="utf-8")
    if completed.returncode or "Invoking LDR-FLIP" not in completed.stdout:
        raise RuntimeError(f"Local FLIP failed: {dest/'flip.log'}")
    maps, pooled = list(dest.glob("flip.*.ldr.png")), list(dest.glob("pooled_values.flip.*.ldr.txt"))
    if len(maps) != 1 or len(pooled) != 1:
        raise ValueError(f"Unexpected native FLIP outputs: {dest}")
    native = {}
    for line in pooled[0].read_text(encoding="utf-8").splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            if key in FIELDS:
                native[FIELDS[key]] = float(value)
    if set(native) != set(FIELDS.values()):
        raise ValueError(f"Missing native FLIP statistics: {dest}")
    ref, test = flip.load(str(reference)), flip.load(str(candidate))
    if ref.shape != test.shape or ref.shape[2] != 3:
        raise ValueError("Unmatched FLIP dimensions")
    errors, mean, _ = flip.evaluate(ref, test, "LDR", applyMagma=False, parameters={"ppd":ppd})
    errors = errors[..., 0]
    if not np.isfinite(errors).all() or errors.min() < 0 or errors.max() > 1:
        raise ValueError("Invalid scalar FLIP output")
    measured_mean = float(errors.mean(dtype=np.float64))
    if not np.isclose(measured_mean, native["mean"], atol=2e-6, rtol=3e-5):
        raise ValueError(("CLI/API FLIP disagreement", measured_mean, native["mean"]))
    if sweep.digest(reference) == sweep.digest(candidate) and np.any(errors):
        raise ValueError("FLIP self comparison must be exactly zero")
    with Image.open(maps[0]) as image:
        if image.size != (errors.shape[1], errors.shape[0]):
            raise ValueError("Incorrect native heatmap dimensions")
    np.save(dest/"values.npy", errors)
    regions = {}
    for name in sweep.REGIONS:
        x0, y0, x1, y1 = sweep.bounds(name, errors.shape[0])
        roi = errors[y0:y1, x0:x1]
        regions[name] = dict(mean=float(roi.mean(dtype=np.float64)), p95=float(np.quantile(roi,.95)),
                            maximum=float(roi.max()), pixel_bounds=[x0,y0,x1,y1])
    files = [maps[0], pooled[0], dest/"flip.log", dest/"values.npy"]
    result = dict(identity=identity, error_map=maps[0].name, pooled_values=pooled[0].name,
        width=errors.shape[1], height=errors.shape[0], native_statistics=native,
        mean=measured_mean, p95=float(np.quantile(errors,.95)), minimum=float(errors.min()),
        maximum=float(errors.max()), regions=regions,
        sha256={p.name:sweep.digest(p) for p in files})
    sweep.save(result_path, result)
    return result


def overview(report, rows, name, columns, *, groove=False, ppd=67.):
    font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
    image = Image.new("RGB", (len(columns)*292, len(sweep.GROUPS)*310+116), "#141b24")
    draw = ImageDraw.Draw(image)
    settings=sweep.read(report/"manifest.json")["settings"]
    spp=settings["spp"]//2 if name.startswith("noise") else settings["spp"]
    draw.text((12,8),f"NVIDIA FLIP 1.7 | LDR | {settings['size']}x{settings['size']} / {spp} spp | {ppd:g} PPD",font=font,fill="white")
    scope="Groove means; crops of full-image maps" if groove else "Full-image means"
    draw.text((12,30),"Native magma, fixed range 0 to 1 | "+scope,font=font,fill="white")
    for column, (key, label) in enumerate(columns):
        draw.text((column*292+12,64), label, font=font, fill="white")
    by_key = {(r["ell"],r["sigma"],r["pair"]):r for r in rows}
    for index, (ell, sigma) in enumerate(sweep.GROUPS):
        for column, (key, _) in enumerate(columns):
            row = by_key.get((ell,sigma,key))
            if not row:
                draw.rectangle((column*292+12,index*310+110,column*292+268,index*310+366),outline="#394756")
                draw.text((column*292+30,index*310+220),"Pending / unavailable",font=font,fill="#8b99a9")
                draw.text((column*292+12,index*310+369),f"ell={ell:g}, sigma={sigma:g}",font=font,fill="white")
                continue
            with Image.open(report/row["image"]) as tile:
                if groove:
                    tile = tile.crop(sweep.bounds("groove",tile.height))
                height=round(256*tile.height/tile.width)
                tile = tile.resize((256,height),Image.Resampling.NEAREST)
                image.paste(tile,(column*292+12,index*310+110+(256-height)//2))
            mean = row["groove_mean"] if groove else row["mean"]
            draw.text((column*292+12,index*310+369),
                      f"ell={ell:g}, sigma={sigma:g}, mean={mean:.5f}",font=font,fill="white")
            draw.text((column*292+12,index*310+389),
                      f"ref seed={row['reference_seed']} / test seed={row['test_seed']}",font=font,fill="#b7c2d1")
    image.save(report/"flip"/name)


def build(report, executable=DEFAULT_FLIP, ppd=67.):
    report, executable = Path(report).resolve(), Path(executable).resolve()
    manifest = sweep.read(report/"manifest.json")
    settings = manifest["settings"]
    dest = report/"flip"
    dest.mkdir(parents=True, exist_ok=True)
    tool = tool_identity(executable)
    (dest/"tool-help.txt").write_text(subprocess.run([str(executable),"--help"],capture_output=True,
        text=True,encoding="utf-8",errors="replace",creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0)).stdout,
        encoding="utf-8")
    rows, noise_rows, group_cards = [], [], []
    self_check = None
    for ell, sigma in sweep.GROUPS:
        group = sweep.key(ell,sigma)
        sources = {}
        for mode in sweep.MODES:
            seed=sweep.primary_seed(report,settings["seed"],ell,sigma,mode)
            folder = report/group/mode/f"seed{seed}"
            if not (folder/"result.json").exists():
                continue
            record = sweep.read(folder/"result.json")
            for name, expected in record["sha256"].items():
                if sweep.digest(folder/name) != expected:
                    raise ValueError(f"Changed render input: {folder/name}")
            source = folder/"display.png"
            sweep.png(source,sweep.filmic(sweep.pfm(folder/"render.pfm")))
            with Image.open(source) as image:
                if image.size != (settings["size"],settings["size"]):
                    raise ValueError("Incorrect FLIP input resolution")
            sources[mode] = source
            if self_check is None:
                # The earliest completed group can change between live updates.
                # Keep its self-comparison cache separate from other inputs.
                self_check = evaluate(executable,tool,source,source,dest/"self-check"/group/mode,ppd)
            a,b = folder/"halfA.png",folder/"halfB.png"
            sweep.png(a,sweep.filmic(sweep.pfm(folder/"samplesA.pfm")))
            sweep.png(b,sweep.filmic(sweep.pfm(folder/"samplesB.pfm")))
            result = evaluate(executable,tool,a,b,dest/group/f"noise_{mode}",ppd)
            noise_rows.append(row_for(report,ell,sigma,mode,mode,mode,a,b,result,
                dest/group/f"noise_{mode}",settings,ppd,kind="half_sample_noise"))
        cards = []
        for pair, reference, test in PAIRS:
            if reference not in sources or test not in sources:
                continue
            folder = dest/group/pair
            result = evaluate(executable,tool,sources[reference],sources[test],folder,ppd)
            row = row_for(report,ell,sigma,pair,reference,test,sources[reference],sources[test],
                          result,folder,settings,ppd,kind="mode_comparison")
            rows.append(row)
            crop = folder/"groove.png"
            with Image.open(report/row["image"]) as image:
                image.crop(sweep.bounds("groove",image.height)).save(crop)
            crop_link = crop.relative_to(report).as_posix()
            cards.append(f'<article><b>{html.escape(PAIR_LABELS[pair])}</b><p>reference seed={row["reference_seed"]} · test seed={row["test_seed"]}</p><p>全图 FLIP {row["mean"]:.6f} · 凹槽 {row["groove_mean"]:.6f}</p>'
                f'<a href="{row["image"]}"><img src="{row["image"]}"></a><img class="detail" src="{crop_link}">'
                f'<p><a href="{row["values"]}">原始误差 NPY</a> · <a href="{row["native_pooled"]}">本地工具统计</a></p></article>')
        if cards:
            group_cards.append(f'<section><h2>ℓ={ell:g} · σ={sigma:g}</h2><div class="flip-cards">{"".join(cards)}</div></section>')
    for filename, selected in (("metrics.csv",rows),("half_sample_noise.csv",noise_rows)):
        available=rows+noise_rows
        fields=list(selected[0]) if selected else list(available[0]) if available else ["ell","sigma","pair","mean","groove_mean"]
        with (dest/filename).open("w",newline="",encoding="utf-8-sig") as stream:
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(selected)
    overview(report,rows,"overview.png",[(k,PAIR_LABELS[k]) for k,_,_ in PAIRS],ppd=ppd)
    overview(report,rows,"groove-overview.png",[(k,PAIR_LABELS[k]) for k,_,_ in PAIRS],groove=True,ppd=ppd)
    overview(report,noise_rows,"noise-overview.png",[(k,k+" half A/B") for k in sweep.MODES],ppd=ppd)
    summary = dict(updated_at=sweep.now(),tool=tool,metric="LDR-FLIP",ppd=ppd,
        resolution=[settings["size"]]*2,spp_per_image=settings["spp"],seed=settings["seed"],
        color_scale=[0,1],colormap="Native FLIP magma",input="Report filmic display.png, original resolution; no resizing, denoising or exposure fitting",
        pairwise_reference="Each pair uses its named first mode as reference; this measures model differences, not error against ground truth",
        roi_pooling="Crop scalar maps computed on the complete image, retaining full spatial-filter context",
        half_sample_noise=f"Compare disjoint {settings['spp']//2} spp halves. Nonlinear FLIP noise cannot be directly subtracted from model FLIP.",
        primary_pairs_completed=len(rows),primary_pairs_expected=len(sweep.GROUPS)*len(PAIRS),
        seed_overrides=sweep.read(report/"seed-overrides.json") if (report/"seed-overrides.json").exists() else None,
        self_comparison_mean=self_check["mean"] if self_check else None,rows=rows,half_sample_noise_rows=noise_rows)
    sweep.save(dest/"summary.json",summary)
    table = ''.join(f'<tr><td>{r["ell"]:g} / {r["sigma"]:g}</td><td>{html.escape(PAIR_LABELS[r["pair"]])}</td>'
        f'<td>{r["mean"]:.6f}</td><td>{r["groove_mean"]:.6f}</td><td>{r["p95"]:.6f}</td></tr>' for r in rows)
    seed_note = ('<p>数值异常后的复跑使用独立 seed，实际种子标在各图及 CSV 中。不同 seed 会增加未共享的采样噪声，'
                 '请结合凹槽区域和半样本噪声参照解读 FLIP。</p>') if summary["seed_overrides"] else ''
    fragment = f'''<section id="flip"><h2>本地 FLIP 1.7：模式间两两比较</h2>
<p>{settings['size']}×{settings['size']} · {settings['spp']} spp · {ppd:g} PPD · 已完成 {len(rows)}/{len(sweep.GROUPS)*len(PAIRS)} 对。直接输入统一 filmic 的原尺寸 PNG；统一原生 magma 色标 0–1。暗色差异小，亮色差异大，未逐图归一化。</p>
<p>这里衡量模式间的感知差异，包含采样噪声。凹槽图和统计均裁自完整图像计算的误差，保留滤波上下文。</p>
{seed_note}
<p><a href="flip_comparison.html">独立 FLIP 页面</a> · <a href="flip/metrics.csv">统计 CSV</a> · <a href="flip/summary.json">输入、工具哈希和区域数据</a></p>
<a href="flip/overview.png"><img class="flip-overview" src="flip/overview.png" alt="FLIP 全图总览"></a>
<details><summary>凹槽 FLIP 放大总览</summary><img class="flip-overview" src="flip/groove-overview.png"></details>
<table><tr><th>ℓ / σ</th><th>比较</th><th>全图 mean FLIP</th><th>凹槽 mean FLIP</th><th>P95</th></tr>{table}</table>
<details><summary>半样本噪声参照</summary><p>每个模式的两个 {settings['spp']//2} spp 半样本互比，仅供观察噪声；不能直接从模式间 FLIP 中相减。</p><img class="flip-overview" src="flip/noise-overview.png"><a href="flip/half_sample_noise.csv">噪声参照 CSV</a></details></section>{''.join(group_cards)}'''
    style = '<style>.flip-cards{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.flip-cards img{width:100%;height:auto;image-rendering:pixelated}.flip-cards article{min-width:0}.flip-overview{display:block;width:min(100%,876px);height:auto;margin:18px auto}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #425063}.detail{max-height:220px;object-fit:contain}@media(max-width:700px){.flip-cards{grid-template-columns:1fr}}</style>'
    standalone = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Shaderball FLIP 对比</title><style>body{max-width:1500px;margin:24px auto;padding:0 20px;background:#141b24;color:#e3e9f1;font:15px/1.65 system-ui}a{color:#8dd9e0}section{padding:18px;background:#202b38;margin:24px 0;border-radius:10px}</style>'
    (report/"flip_comparison.html").write_text(standalone+style+'<p><a href="comparison.html">返回渲染结果</a></p>'+fragment+'</html>',encoding="utf-8")
    page = report/"comparison.html"
    if page.exists():
        body = re.sub(re.escape(BEGIN)+r".*?"+re.escape(END),"",page.read_text(encoding="utf-8"),flags=re.S)
        position=body.lower().rfind("</html>")
        addition=BEGIN+style+fragment+END
        body=body[:position]+addition+body[position:] if position>=0 else body+addition
        page.write_text(body,encoding="utf-8")
    print(f"FLIP {len(rows)}/24 pairs, {len(noise_rows)} half-sample checks",flush=True)
    return summary


def row_for(report,ell,sigma,pair,reference,test,ref_path,test_path,result,folder,settings,ppd,kind):
    return dict(ell=ell,sigma=sigma,pair=pair,kind=kind,reference_mode=reference,test_mode=test,
        reference_seed=int(ref_path.parent.name.removeprefix("seed")),test_seed=int(test_path.parent.name.removeprefix("seed")),
        reference=ref_path.relative_to(report).as_posix(),test=test_path.relative_to(report).as_posix(),
        reference_sha256=sweep.digest(ref_path),test_sha256=sweep.digest(test_path),ppd=ppd,
        width=settings["size"],height=settings["size"],spp_per_input=settings["spp"] if kind=="mode_comparison" else settings["spp"]//2,
        mean=result["mean"],p95=result["p95"],maximum=result["maximum"],
        groove_mean=result["regions"]["groove"]["mean"],base_mean=result["regions"]["base"]["mean"],
        object_mean=result["regions"]["object"]["mean"],
        image=(folder/result["error_map"]).relative_to(report).as_posix(),
        values=(folder/"values.npy").relative_to(report).as_posix(),
        native_pooled=(folder/result["pooled_values"]).relative_to(report).as_posix())


def validate(report,executable=DEFAULT_FLIP,ppd=67.,allow_incomplete=False):
    from html.parser import HTMLParser
    from urllib.parse import unquote,urlsplit
    report=Path(report).resolve()
    summary=build(report,executable,ppd)
    expected=len(sweep.GROUPS)*len(PAIRS)
    complete=(summary["primary_pairs_completed"]==expected and
              len(summary["half_sample_noise_rows"])==len(sweep.GROUPS)*len(sweep.MODES))
    if not allow_incomplete:
        assert complete
    assert summary["self_comparison_mean"]==0
    for row in summary["rows"]+summary["half_sample_noise_rows"]:
        assert sweep.digest(report/row["reference"])==row["reference_sha256"]
        assert sweep.digest(report/row["test"])==row["test_sha256"]
        values=np.load(report/row["values"])
        assert values.shape==(row["height"],row["width"])
        assert np.isfinite(values).all() and values.min()>=0 and values.max()<=1
        assert np.isclose(values.mean(dtype=np.float64),row["mean"],atol=1e-10)
        folder=(report/row["image"]).parent
        cached=sweep.read(folder/"result.json")
        for name,expected_hash in cached["sha256"].items():
            assert sweep.digest(folder/name)==expected_hash
        with Image.open(report/row["image"]) as image:
            assert image.size==(row["width"],row["height"])
    links=[]
    class Parser(HTMLParser):
        def handle_starttag(self,tag,attrs):
            links.extend(v for k,v in attrs if k in ("href","src") and v)
    count=0
    for name in ("comparison.html","flip_comparison.html"):
        page=report/name
        links.clear()
        Parser().feed(page.read_text(encoding="utf-8"))
        for value in links:
            target=urlsplit(value)
            if not target.scheme and target.path:
                assert (page.parent/unquote(target.path)).is_file(),value
                count+=1
    result=dict(validated_at=sweep.now(),all_pass=complete,complete=complete,available_outputs_pass=True,
        primary_pairs=len(summary["rows"]),expected_primary_pairs=expected,
        noise_pairs=len(summary["half_sample_noise_rows"]),self_comparison=0,
        verified_local_links=count,tool=summary["tool"],ppd=ppd,
        input_and_output_hashes_verified=True,scalar_maps_verified=True,
        summary_sha256=sweep.digest(report/"flip/summary.json"),script_sha256=sweep.digest(Path(__file__)))
    sweep.save(report/"flip/validation.json",result)
    print(json.dumps(result,ensure_ascii=False),flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report",type=Path,required=True)
    parser.add_argument("--flip-exe",type=Path,default=DEFAULT_FLIP)
    parser.add_argument("--ppd",type=float,default=67.)
    parser.add_argument("--validate",action="store_true")
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args()
    if args.validate:
        validate(args.report,args.flip_exe,args.ppd,args.allow_incomplete)
    else:
        build(args.report,args.flip_exe,args.ppd)


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
