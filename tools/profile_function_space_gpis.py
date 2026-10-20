"""Measure function-space CPU stages, profiling overhead, and image parity."""
import argparse
import ctypes
import json
import os
from pathlib import Path
import re
import subprocess
import time

from verify_wavefront_gpis import compare, read_pfm


def process_cpu_seconds(process):
    if os.name != "nt":
        return None
    from ctypes import wintypes
    get_times = ctypes.windll.kernel32.GetProcessTimes
    get_times.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4
    get_times.restype = wintypes.BOOL
    values = [wintypes.FILETIME() for _ in range(4)]
    if not get_times(int(process._handle), *(ctypes.byref(value) for value in values)):
        return None
    return sum((value.dwHighDateTime << 32) + value.dwLowDateTime for value in values[2:]) * 1e-7


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("build/Release/tungsten.exe"))
    parser.add_argument("--baseline-executable", type=Path)
    parser.add_argument("--output", type=Path, default=Path("build/function_space_profile"))
    parser.add_argument("--cases", default="128x4,512x2,n8,n16,n32,one_thread")
    parser.add_argument("--resume", action="store_true", help="Reuse completed runs and recover completed logs")
    args = parser.parse_args()
    source = args.scene.resolve()
    original = json.loads(source.read_text(encoding="utf-8"))
    assert original["media"][0]["type"] == "function_space_gaussian_process"
    executable = args.executable.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = {"source": str(source), "scene_snapshot": original,
              "executable": str(executable), "seed": 12345, "runs": [], "parity": {}}
    if args.resume:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        assert report["scene_snapshot"] == original, "Scene changed since the saved run"
    cases = {
        "128x4": (128, 4, 32, 8, 2),
        "512x2": (512, 2, 32, 8, 1),
        "n8": (128, 2, 8, 8, 1),
        "n16": (128, 2, 16, 8, 1),
        "n32": (128, 2, 32, 8, 1),
        "one_thread": (128, 2, 32, 1, 1),
    }

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    def run(label, size, spp, points, threads, profile, binary, record=True):
        if args.resume and record:
            existing = next((run for run in report["runs"] if run["label"] == label), None)
            if existing:
                text = (output / (label + ".log")).read_text(encoding="utf-8", errors="replace")
                existing["render_seconds"] = float(re.findall(r"Completed \d+/\d+ spp, ([\d.]+)/", text)[-1])
                save()
                print("Reusing", label, flush=True)
                return existing
        scene = json.loads(json.dumps(original))
        scene["camera"]["resolution"] = [size, size]
        scene["media"][0]["sample_points"] = points
        scene["integrator"].update(use_cuda_gpis=False, wavefront_gpis=False,
                                   independent_path_sampling=True)
        scene["renderer"].update(spp=spp, spp_step=1, hdr_output_file="scene.pfm",
                                  output_file="scene.exr", adaptive_sampling=False,
                                  overwrite_output_files=True, enable_resume_render=False)
        scene_file = output / (label + ".json")
        scene_file.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        run_output = output / label
        environment = os.environ.copy()
        environment.pop("TUNGSTEN_GPIS_PROFILE", None)
        environment.pop("TUNGSTEN_GPIS_FS_PROFILE", None)
        if profile:
            environment["TUNGSTEN_GPIS_FS_PROFILE"] = "1"
        log_file = output / (label + ".log")
        print("Rendering", label, flush=True)
        command = [str(binary), str(scene_file), "-i", str(source.parent),
                   "-d", str(run_output), "-t", str(threads), "-s", "12345"]
        recovered = False
        if args.resume and log_file.exists() and (run_output / "scene.pfm").exists():
            text = log_file.read_text(encoding="utf-8", errors="replace")
            recovered = bool(re.search(rf"Completed {spp}/{spp} spp", text))
        if recovered:
            code, cpu, wall = 0, None, None
        else:
            start = time.perf_counter()
            with log_file.open("wb") as stream:
                process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
                code = process.wait()
                cpu = process_cpu_seconds(process)
            wall = time.perf_counter() - start
            text = log_file.read_text(encoding="utf-8", errors="replace")
        if code or not re.search(rf"Completed {spp}/{spp} spp", text):
            raise RuntimeError(f"{label}: render failed; inspect {log_file}")
        if re.search(r"Matrix square root failed|Gradient invalid|Conditioning .* failed", text):
            raise RuntimeError(f"{label}: numerical failure; inspect {log_file}")
        read_pfm(run_output / "scene.pfm")
        result = {"label": label, "resolution": [size, size], "spp": spp,
                  "sample_points": points, "threads": threads, "profile_enabled": profile,
                  "executable": str(binary), "wall_seconds": wall, "process_cpu_seconds": cpu,
                  "recovered_from_log": recovered,
                  "render_seconds": float(re.findall(r"Completed \d+/\d+ spp, ([\d.]+)/", text)[-1]),
                  "pfm": str(run_output / "scene.pfm")}
        prefix = "GPIS function-space profile: "
        lines = [line[len(prefix):] for line in text.splitlines() if line.startswith(prefix)]
        if profile:
            assert len(lines) == 1, f"Missing/duplicate profile: {label}"
            result["profile"] = json.loads(lines[0])
            stages = {}
            for domain in result["profile"]["domains"].values():
                for stage, values in domain["stages"].items():
                    stages[stage] = stages.get(stage, 0) + values["exclusive_ms"]
            total = sum(stages.values())
            result["exclusive_worker_ms"] = stages
            result["gp_worker_ms"] = total
            result["exclusive_percent"] = {stage: 100 * ms / total for stage, ms in stages.items()}
        if record:
            report["runs"].append(result)
            save()
        print(json.dumps({key: result[key] for key in
              ("label", "render_seconds", "wall_seconds", "process_cpu_seconds")}), flush=True)
        if profile:
            print(json.dumps(result["exclusive_percent"]), flush=True)
        return result

    run("warmup", 32, 1, 32, 8, False, executable, record=False)
    baseline = None
    if args.baseline_executable:
        baseline = run("before_instrumentation", 128, 4, 32, 8, False,
                       args.baseline_executable.resolve())
    for case in args.cases.split(","):
        size, spp, points, threads, repeats = cases[case]
        reference = None
        for repeat in range(repeats):
            for profile in (False, True):
                label = f"{case}_{'profile' if profile else 'plain'}_{repeat+1}"
                result = run(label, size, spp, points, threads, profile, executable)
                if reference is None:
                    reference = result
                else:
                    metrics = compare(Path(reference["pfm"]), Path(result["pfm"]))
                    report["parity"][reference["label"] + "_vs_" + label] = metrics
                    save()
                    assert metrics["exact"], f"Profiling/repetition changed pixels: {label}"
                if baseline and case == "128x4" and repeat == 0 and not profile:
                    metrics = compare(Path(baseline["pfm"]), Path(result["pfm"]))
                    report["parity"]["before_instrumentation_vs_plain"] = metrics
                    save()
                    assert metrics["exact"], "Instrumentation changed the baseline pixels"
    by_label = {run["label"]: run for run in report["runs"]}
    if "n32_plain_1" in by_label and "one_thread_plain_1" in by_label:
        metrics = compare(Path(by_label["n32_plain_1"]["pfm"]),
                          Path(by_label["one_thread_plain_1"]["pfm"]))
        report["parity"]["one_vs_eight_threads"] = metrics
        assert metrics["exact"], "Thread count changed pixels"
    save()
    print("Saved", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
