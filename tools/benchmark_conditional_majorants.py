"""Compare adaptive and two-constant tracking on the same baked SDF scene.

Timeouts and renderer errors are recorded as unfinished runs, never timings of
successful renders. The optional flight probe has a separate random-draw budget.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
sys.dont_write_bytecode = True
from verify_wavefront_gpis import read_pfm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("build/Release/tungsten.exe"))
    parser.add_argument("--test-executable", type=Path, default=Path("build/Release/global_conditional_test.exe"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--spp", type=int, default=1)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--birth-lengths", default="0", help="Comma-separated two_constants prefix lengths; 0 uses analytic birth")
    args = parser.parse_args()
    if min(args.size, args.spp, args.threads, args.repeats, args.timeout) <= 0:
        parser.error("resolution, samples, threads, repeats and timeout must be positive")
    try:
        birth_lengths = list(dict.fromkeys(float(x) for x in args.birth_lengths.split(",")))
    except ValueError:
        parser.error("birth lengths must be numbers separated by commas")
    if not birth_lengths or any(not math.isfinite(x) or x < 0 for x in birth_lengths):
        parser.error("birth lengths must be finite and nonnegative")
    args.scene = args.scene.resolve()
    args.output = args.output.resolve()
    args.executable = args.executable.resolve()
    args.test_executable = args.test_executable.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    if report_path.exists():
        parser.error("output already contains report.json; choose a new directory")
    original = json.loads(args.scene.read_text(encoding="utf-8-sig"))
    selected = [i for i, medium in enumerate(original["media"]) if medium["type"] == "global_conditional"]
    if len(selected) != 1:
        parser.error("scene must contain exactly one global_conditional medium")
    report = {
        "source": str(args.scene), "executable": str(args.executable),
        "executable_sha256": hashlib.sha256(args.executable.read_bytes()).hexdigest(),
        "source_sha256": hashlib.sha256(args.scene.read_bytes()).hexdigest(),
        "size": args.size, "spp": args.spp, "threads": args.threads,
        "seed": 12345, "timeout_seconds": args.timeout, "birth_lengths": birth_lengths, "runs": [],
    }
    grid = original["media"][selected[0]]["gaussian_process"]["mean"].get("grid", {})
    if "file" in grid:
        asset = (args.scene.parent / grid["file"]).resolve()
        report["mean_asset"] = str(asset)
        report["mean_asset_sha256"] = hashlib.sha256(asset.read_bytes()).hexdigest()

    def save():
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def execute(command, log_path):
        start = time.perf_counter()
        environment = os.environ.copy()
        for key in ("TUNGSTEN_GPIS_CONDITIONAL_PROFILE", "TUNGSTEN_GPIS_PROFILE", "TUNGSTEN_GPIS_FS_PROFILE"):
            environment.pop(key, None)
        with log_path.open("wb") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout, env=environment)
                status = "exited"
                code = result.returncode
            except subprocess.TimeoutExpired:
                status, code = "timeout", None
        return {"status": status, "exit_code": code, "wall_seconds": time.perf_counter() - start}

    for mode, birth_length in [("adaptive",0)] + [("two_constants",h) for h in birth_lengths]:
        for index in range(args.repeats + (mode == "adaptive")):
            warmup = mode == "adaptive" and index == 0
            case = mode + (f"-h{birth_length:g}" if birth_length else "")
            name = f"{case}-{'warmup' if warmup else index}"
            folder = args.output / name
            folder.mkdir()
            scene = copy.deepcopy(original)
            scene["media"][selected[0]]["tracking_majorant"] = mode
            scene["media"][selected[0]]["tracking_constant_birth_length"] = birth_length
            scene["camera"]["resolution"] = [args.size, args.size]
            settings = scene["renderer"]
            settings.update({"spp": args.spp, "spp_step": args.spp, "output_directory": str(folder),
                             "output_file": "render.png", "hdr_output_file": "render.pfm",
                             "overwrite_output_files": True, "enable_resume_render": False,
                             "adaptive_sampling": False})
            path = folder / "scene.json"
            path.write_text(json.dumps(scene, indent=2), encoding="utf-8")
            command = [str(args.executable), "-i", str(args.scene.parent), "-d", str(folder),
                       "-t", str(args.threads), "-s", "12345", str(path)]
            row = {"mode": mode, "birth_length": birth_length, "warmup": warmup, "command": command,
                   **execute(command, folder / "render.log")}
            log = (folder / "render.log").read_text(encoding="utf-8", errors="replace")
            matches = re.findall(r"Completed (\d+)/(\d+) spp, ([\d.]+)/", log)
            row["completed"] = bool(row["exit_code"] == 0 and matches and int(matches[-1][0]) == args.spp)
            if re.search(r"Caught an internal error|unrecoverable error|nonfinite|PROGRAM FAILURE", log):
                row["completed"] = False
            if row["completed"]:
                row["render_seconds"] = float(matches[-1][2])
                _, pixels = read_pfm(folder / "render.pfm")
                row["finite_nonnegative"] = all(math.isfinite(x) and x >= 0 for x in pixels)
                row["mean_rgb"] = sum(pixels) / len(pixels)
                row["completed"] = row["finite_nonnegative"] and row["mean_rgb"] > 0
            if not row["completed"] and row["status"] != "timeout":
                row["status"] = "render_failed"
            report["runs"].append(row)
            save()
            print(json.dumps(row), flush=True)
            if not row["completed"]:
                break
    if args.test_executable.exists():
        stage = "tracking_split_probe" if any(birth_lengths) else "tracking_probe"
        command = [str(args.test_executable), stage, str(args.scene)]
        report["probe"] = {"command": command, **execute(command, args.output / "probe.log")}
        report["probe"]["rows"] = [json.loads(line) for line in
            (args.output / "probe.log").read_text(encoding="utf-8", errors="replace").splitlines()
            if line.startswith('{"')]
        save()
        print(json.dumps(report["probe"]), flush=True)


if __name__ == "__main__":
    main()
