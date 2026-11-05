"""Compare numerical and table probabilities with identical history and render settings."""
import argparse
import copy
import html
import json
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", type=Path, default=ROOT / "build/Release/tungsten.exe")
    parser.add_argument("--checker", type=Path, default=ROOT / "build/Release/sparse_history_gpis_test.exe")
    parser.add_argument("--table", type=Path, default=ROOT / "data/gpis/bivariate-normal-cdf-v1.bin")
    parser.add_argument("--scene", type=Path, default=ROOT / "data/example-scenes/sparse-history-gpis/scene.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/sparse-history-lut")
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--timeout", type=float, default=300)
    args = parser.parse_args()
    for path in (args.renderer, args.checker, args.table, args.scene):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    base = json.loads(args.scene.read_text(encoding="utf-8"))
    base["camera"]["resolution"] = [args.resolution] * 2
    base["renderer"].update(spp=args.spp, spp_step=args.spp, adaptive_sampling=False,
                            output_file="image.png", hdr_output_file="image.exr")
    base["integrator"].update(use_cuda_gpis=False, wavefront_gpis=False)
    results = {}
    for points in (1, 2):
        for method in ("numeric", "lut"):
            name = f"k{points}-{method}"
            directory = output / name
            directory.mkdir(parents=True, exist_ok=True)
            scene = copy.deepcopy(base)
            settings = scene["media"][0]["sparse_history"]
            settings.update(points=points, probability_method=method)
            settings.pop("cdf_table", None)
            if method == "lut":
                settings["cdf_table"] = args.table.resolve().as_posix()
            scene_path = directory / "scene.json"
            scene_path.write_text(json.dumps(scene, indent=2) + "\n", encoding="utf-8")
            command = [str(args.renderer.resolve()), "-t", str(args.threads), "-s", str(args.seed),
                       "-d", str(directory), str(scene_path)]
            start = time.perf_counter()
            run = subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=args.timeout)
            wall = time.perf_counter() - start
            log = run.stdout + "\n" + run.stderr
            (directory / "render.log").write_text(log, encoding="utf-8")
            if (run.returncode or f"Completed {args.spp}/{args.spp} spp" not in log or
                    re.search(r"Caught an internal error|unrecoverable error|Unable to load|nonfinite|budget exhausted", log)):
                raise RuntimeError(f"{name} failed; see {directory / 'render.log'}")
            checked = subprocess.run([str(args.checker.resolve()), "image", str(directory / "image.exr")],
                                     capture_output=True, text=True, check=True)
            (directory / "validation.log").write_text(checked.stdout, encoding="utf-8")
            timing = re.search(r"Completed \d+/\d+ spp, ([\d.]+)/[\d.]+ sec", log)
            result = {"points": points, "method": method, "render_seconds": float(timing[1]),
                      "wall_seconds": wall}
            table_stats = re.search(r"CDF table: (\{[^\n]+\})", log)
            if table_stats:
                result["table"] = json.loads(table_stats[1])
            results[name] = result
            print(name, json.dumps(result), flush=True)
        numeric, lut = results[f"k{points}-numeric"], results[f"k{points}-lut"]
        compared = subprocess.run([str(args.checker.resolve()), "compare-images",
                                   str(output / f"k{points}-numeric/image.exr"),
                                   str(output / f"k{points}-lut/image.exr")],
                                  capture_output=True, text=True, check=True)
        lut["image_difference"] = json.loads(compared.stdout)
        lut["speedup"] = numeric["render_seconds"] / lut["render_seconds"]
    report = {"resolution": args.resolution, "spp": args.spp, "threads": args.threads,
              "seed": args.seed, "results": results}
    (output / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    rows, images = [], []
    for name, result in results.items():
        stats = result.get("table", {})
        accepted, fallback = stats.get("accepted", 0), stats.get("numerical_fallback", 0)
        hit_rate = f"{accepted / (accepted + fallback):.1%}" if accepted + fallback else "—"
        error = result.get("image_difference", {}).get("relative_rmse")
        rows.append(f"<tr><td>{name}</td><td>{result['render_seconds']:.2f} s</td>"
                    f"<td>{hit_rate}</td><td>{f'{error:.3g}' if error is not None else '—'}</td></tr>")
        images.append(f'<figure><img src="{name}/image.png"><figcaption>{html.escape(name)}</figcaption></figure>')
    page = """<!doctype html><meta charset="utf-8"><title>Sparse history CDF table comparison</title>
<style>body{font:16px system-ui;max-width:1000px;margin:40px auto;background:#fafafa;color:#222}
table{border-collapse:collapse}td,th{padding:10px;border-bottom:1px solid #ccc;text-align:left}
.images{display:flex;flex-wrap:wrap}figure{margin:16px}img{width:192px;image-rendering:pixelated}</style>
<h1>Sparse history: numerical / CDF table</h1>
"""
    page += (f"<p>{args.resolution}×{args.resolution}, {args.spp} spp, seed {args.seed}, "
             f"{args.threads} threads. Compare methods within the same K. "
             "Image differences are same-seed checks, not a converged bias estimate.</p>")
    page += "<table><tr><th>Method</th><th>Render time</th><th>Table accepted</th><th>Relative image RMSE</th></tr>"
    page += "".join(rows) + '</table><div class="images">' + "".join(images) + "</div>"
    (output / "comparison.html").write_text(page, encoding="utf-8")
    print(output / "comparison.html")


if __name__ == "__main__":
    main()
