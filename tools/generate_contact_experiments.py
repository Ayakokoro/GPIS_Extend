#!/usr/bin/env python3
"""Bake unit-scale contact fixtures and matched scenes for the Tungsten adapter.

Run from this checkout; inputs default to build/ and models/, outputs to outputs/.
Only Python's standard library is required. Build bake_contact_fields first.
"""
from __future__ import annotations

import argparse
import array
import copy
import html
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODES = ("classic_local", "classic_global", "global_conditional", "neural")


def save(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def path_arg(text: str) -> Path:
    path = Path(text).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def default_executable(name: str) -> Path:
    filename = name + (".exe" if os.name == "nt" else "")
    candidates = [ROOT / directory / filename for directory in
                  ("build/Release", "build", "build/RelWithDebInfo", "build/Debug")]
    return next((path for path in candidates if path.is_file()), candidates[0])


def number(value: float) -> str:
    return f"{value:.9g}".replace(".", "p")


def scene_for(mode: str, meta: dict, field: Path, model: Path, args) -> dict:
    covariance = {"type": "squared_exponential", "sigma": args.sigma,
                  "lengthScale": args.length_scale}
    medium = {"name": "contact", "type": mode, "max_bounces": 128,
              "material": {"type": "mirror", "albedo": 0.85}}
    if mode in ("neural", "classic_local"):
        medium.update(field={"file": field.as_posix()}, covariance=covariance)
        if mode == "neural":
            medium.update(type="neural_causal_renewal", backend="scalar",
                          model=model.as_posix(), profile_maximum_step=0.25)
        else:
            medium["field"].update(sdf_grid="sdf", density_grid="density", macrocell_size=8)
    else:
        medium["gaussian_process"] = {
            "type": "standard", "covariance": covariance,
            "mean": {"type": "tabulated", "grid": {
                "type": "vdb", "file": field.with_name(field.stem + "_sdf.vdb").as_posix(),
                "density_name": "sdf", "density_scale": 1.0, "emission_name": "",
                "normalize_size": False, "request_sdf": False,
                "interpolate": "linear", "integration_method": "raymarching"}}}
    lower, upper = meta["process_box"]["min"], meta["process_box"]["max"]
    box = {"position": [(a+b)/2 for a, b in zip(lower, upper)],
           "scale": [b-a for a, b in zip(lower, upper)]}
    spheres = meta["geometry"]["shape"] == "two_spheres"
    return {
        "media": [medium],
        "camera": {"type": "pinhole", "resolution": [args.width, args.height],
                   "fov": 28 if spheres else 38, "tonemap": "filmic",
                   "reconstruction_filter": "tent", "transform": {
                       "position": [0, 0.75, -2.7] if spheres else [0.8, 1.1, -1.8],
                       "look_at": [0, 0, 0], "up": [0, 1, 0], "scale": [-1, 1, 1]}},
        "integrator": {"type": "path_tracer", "min_bounces": 0, "max_bounces": 128,
                       "enable_light_sampling": False, "enable_volume_light_sampling": False,
                       "enable_consistency_checks": False, "enable_two_sided_shading": True,
                       "use_cuda_gpis": False, "independent_path_sampling": True,
                       "wavefront_gpis": False},
        "primitives": [
            {"name": "processBox", "type": "cube", "int_medium": "contact",
             "bsdf": {"type": "forward", "albedo": 1}, "transform": box},
            {"name": "key", "type": "quad", "emission": 8.0,
             "bsdf": {"type": "null", "albedo": 1}, "transform": {
                 "position": [-0.6, 1.5, -0.5], "scale": [1.5, 1, 1.5], "rotation": [180, 0, 0]}},
            {"type": "infinite_sphere", "emission": 0.04,
             "bsdf": {"type": "null", "albedo": 1}}],
        "renderer": {"spp": args.spp, "spp_step": 1, "adaptive_sampling": False,
                     "stratified_sampler": False, "enable_resume_render": False,
                     "overwrite_output_files": True, "output_file": "render.png",
                     "hdr_output_file": "render.pfm"}}


def pfm_stats(path: Path) -> dict:
    with path.open("rb") as stream:
        def line():
            value = stream.readline()
            while value.startswith(b"#"):
                value = stream.readline()
            return value.strip()
        if line() != b"PF":
            raise RuntimeError(f"Not an RGB PFM: {path}")
        width, height = map(int, line().split())
        scale = float(line())
        data = array.array("f")
        data.frombytes(stream.read())
    if len(data) != width * height * 3 or not math.isfinite(scale) or scale == 0:
        raise RuntimeError(f"Invalid PFM payload: {path}")
    if (scale < 0) != (sys.byteorder == "little"):
        data.byteswap()
    if any(not math.isfinite(v) or v < -1e-6 for v in data):
        raise RuntimeError(f"Nonfinite or negative radiance: {path}")
    if max(data, default=0) <= 0:
        raise RuntimeError(f"Black image: {path}")
    return {"width": width, "height": height, "max_rgb": max(data)*abs(scale),
            "mean_rgb": sum(data)/len(data)*abs(scale), "finite_nonnegative": True}


def cross_section(meta: dict) -> str:
    geometry = meta["geometry"]
    if geometry["shape"] == "two_spheres":
        c = .25 + geometry["gap"] / 2
        elements = "".join(f'<circle cx="{x}" cy="0" r=".25"/>' for x in (-c, c))
    else:
        points = " ".join(f"{x},{y}" for x, y in meta["polygon_xy"])
        elements = f'<polygon points="{points}"/>'
    return ('<svg viewBox="-.7 -.38 1.4 .76" width="280" height="152">'
            '<g transform="scale(1,-1)" fill="#ccd5e0" stroke="#37475c" stroke-width=".004">'
            + elements + '</g></svg>')


def write_report(out: Path, manifest: dict) -> None:
    rows = []
    for case in manifest["cases"]:
        name = case["name"]
        cells = [f'<td>{html.escape(name)}<br>{cross_section(case["metadata"])}</td>']
        for mode in MODES:
            folder = Path("renders") / name / mode
            image = folder / "render.png"
            text = f'<a href="configs/{name}/{mode}.json">scene</a>'
            if (out / image).is_file():
                text += f'<br><a href="{image.as_posix()}"><img src="{image.as_posix()}"></a>'
            if (out / folder / "result.json").is_file():
                info = load(out / folder / "result.json")
                text += f'<br>{html.escape(str(info.get("status", "unknown")))}'
                if "seconds" in info:
                    text += f', {info["seconds"]:.1f}s'
                if "stats" in info:
                    text += f', {info["stats"]["width"]}x{info["stats"]["height"]}, {info["spp"]} spp'
            cells.append(f"<td>{text}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    content = ('<!doctype html><meta charset="utf-8"><title>Contact experiments</title>'
               '<style>body{font:15px system-ui;background:#f7f8fb;color:#263344;margin:24px}'
               'table{border-collapse:collapse}th,td{border:1px solid #ccd;padding:12px;text-align:center}'
               'img{width:220px;image-rendering:auto}a{color:#185fa1}</style>'
               '<h1>Prebaked contact experiments</h1>'
               f'<p>sigma={manifest["sigma"]}, ell={manifest["length_scale"]}, '
               f'dx={manifest["voxel_size"]}. Geometry in world units; Y up.</p>'
               '<p>Matched process box, camera, light and material across all four modes. '
               'Path sampling only (NEE disabled). Low-spp renders check loading and appearance; '
               'they do not establish model agreement. PFM is linear radiance; PNG uses filmic tonemapping.</p>'
               '<table><tr><th>XY geometry</th>' + ''.join(f'<th>{m}</th>' for m in MODES)
               + '</tr>' + ''.join(rows) + '</table>')
    (out / "comparison.html").write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=path_arg, default=ROOT / "outputs/contact_experiments")
    parser.add_argument("--baker", type=path_arg, default=default_executable("bake_contact_fields"))
    parser.add_argument("--tungsten", type=path_arg, default=default_executable("tungsten"))
    parser.add_argument("--model", type=path_arg, default=ROOT /
                        "models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json")
    parser.add_argument("--gaps", type=float, nargs="*", default=[0, .02, .06])
    parser.add_argument("--angles", type=float, nargs="*", default=[45, 90, 120])
    parser.add_argument("--voxel-size", type=float, default=.005)
    parser.add_argument("--sigma", type=float, default=.02)
    parser.add_argument("--length-scale", type=float, default=.02)
    parser.add_argument("--padding-sigmas", type=float, default=6)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--spp", type=int, default=128)
    parser.add_argument("--skip-bake", action="store_true", help="Reuse only matching verified fields")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render-cases", nargs="+", help="Subset of case names to render")
    parser.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    parser.add_argument("--render-size", type=int, help="Square resolution override for rendered copies only")
    parser.add_argument("--render-spp", type=int, help="SPP override for rendered copies only")
    parser.add_argument("--threads", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=17429)
    args = parser.parse_args()
    for key in ("sigma", "length_scale", "voxel_size", "padding_sigmas"):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive and finite")
    for key in ("width", "height", "spp", "threads", "render_size", "render_spp"):
        value = getattr(args, key)
        if value is not None and value < 1:
            parser.error(f"--{key.replace('_', '-')} must be positive")
    if args.padding_sigmas < 4 or any(not math.isfinite(g) or not 0 <= g <= .5 for g in args.gaps):
        parser.error("padding must be at least 4 sigma; gaps must lie in [0, 0.5]")
    if any(not math.isfinite(a) or not 15 <= a <= 120 for a in args.angles):
        parser.error("angles must lie in [15, 120] degrees")
    if not args.model.is_file():
        parser.error(f"Joint model not found: {args.model}; supply --model")
    specs = ([{"name": "two_spheres_g" + number(g), "shape": "two_spheres", "gap": g} for g in args.gaps]
             + [{"name": "v_groove_a" + number(a), "shape": "v_groove", "opening_degrees": a} for a in args.angles])
    if not specs or len({s["name"] for s in specs}) != len(specs):
        parser.error("choose at least one case, with distinct gaps and angles")
    if args.render_cases and not set(args.render_cases) <= {s["name"] for s in specs}:
        parser.error("--render-cases must name cases in the requested gap/angle set")
    out = args.output.resolve()
    fields = out / "fields"
    request = {"output_directory": fields.as_posix(), "voxel_size": args.voxel_size,
               "sigma": args.sigma, "length_scale": args.length_scale,
               "padding": args.padding_sigmas * args.sigma, "cases": specs}
    save(out / "bake_request.json", request)
    if not args.skip_bake:
        if not args.baker.is_file():
            parser.error("Build bake_contact_fields first, or supply --baker")
        subprocess.run([str(args.baker), str(out / "bake_request.json")], check=True, cwd=ROOT)
    manifest = {"generator": str(Path(__file__).resolve()), **request,
                "model": args.model.as_posix(), "cases": []}
    for spec in specs:
        field = fields / (spec["name"] + ".nvdb")
        meta = load(Path(str(field) + ".json"))
        if (meta["geometry"] != spec or meta["dx"] != args.voxel_size or meta["sigma"] != args.sigma
                or meta["length_scale"] != args.length_scale
                or not math.isclose(meta["band"], args.padding_sigmas, rel_tol=1e-12)
                or not meta["verification"]["sdf_formats_identical"] or not field.is_file()
                or not field.with_name(field.stem + "_sdf.vdb").is_file()):
            raise RuntimeError(f"Existing bake does not match request: {field}; bake again")
        item = {"name": spec["name"], "metadata": meta, "scenes": {}}
        for mode in MODES:
            scene = scene_for(mode, meta, field, args.model, args)
            scene["renderer"]["output_directory"] = (out / "renders" / spec["name"] / mode).as_posix()
            path = out / "configs" / spec["name"] / (mode + ".json")
            save(path, scene)
            item["scenes"][mode] = str(path)
        manifest["cases"].append(item)
    save(out / "manifest.json", manifest)
    write_report(out, manifest)
    failed = False
    if args.render:
        if not args.tungsten.is_file():
            parser.error(f"Tungsten not found: {args.tungsten}")
        for case in manifest["cases"]:
            if args.render_cases and case["name"] not in args.render_cases:
                continue
            for mode in args.modes:
                scene = copy.deepcopy(load(Path(case["scenes"][mode])))
                folder = out / "renders" / case["name"] / mode
                if args.render_size:
                    scene["camera"]["resolution"] = [args.render_size] * 2
                if args.render_spp:
                    scene["renderer"]["spp"] = args.render_spp
                path = folder / "scene.json"
                save(path, scene)
                # Remove only our previous image files, so a failed rerun cannot
                # accidentally pass the checks using a stale image.
                for filename in ("render.png", "render.pfm"):
                    (folder / filename).unlink(missing_ok=True)
                command = [str(args.tungsten), "-t", str(args.threads), "-s", str(args.seed),
                           "-d", str(folder), str(path)]
                print(shlex.join(command), flush=True)
                start = time.monotonic()
                with (folder / "render.log").open("w", encoding="utf-8") as log:
                    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
                record = {"command": command, "seconds": time.monotonic()-start,
                          "returncode": result.returncode, "spp": scene["renderer"]["spp"]}
                try:
                    log = (folder / "render.log").read_text(encoding="utf-8", errors="replace")
                    spp = record["spp"]
                    if result.returncode or not re.search(rf"Completed\s+{spp}/{spp}\s+spp", log):
                        raise RuntimeError("renderer did not complete the requested spp")
                    if re.search(r"Caught an internal error|unrecoverable error|Unable to load|not implemented|Wrong normal direction|terminate called", log, re.I):
                        raise RuntimeError("renderer reported an internal error")
                    record["stats"] = pfm_stats(folder / "render.pfm")
                    if [record["stats"][k] for k in ("width", "height")] != scene["camera"]["resolution"]:
                        raise RuntimeError("rendered dimensions do not match the scene")
                    if not (folder / "render.png").is_file():
                        raise RuntimeError("PNG output missing")
                    record["status"] = "passed"
                except (OSError, ValueError, RuntimeError) as error:
                    record.update(status="failed", error=str(error))
                    failed = True
                save(folder / "result.json", record)
                write_report(out, manifest)
                print(f'{case["name"]}/{mode}: {record["status"]}, {record["seconds"]:.1f}s', flush=True)
    print(f"Scenes and comparison: {out / 'comparison.html'}", flush=True)
    return int(failed)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(f"Contact experiments failed: {error}", file=sys.stderr)
        sys.exit(1)
