"""Benchmark conditional tracking, retain stage profiles, and verify image parity."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import time

from profile_function_space_gpis import process_cpu_seconds
from verify_wavefront_gpis import compare, read_pfm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--medium-type", choices=("global_conditional", "classic_global"), default="global_conditional",
                        help="Classic is supported for plain timing comparisons only")
    parser.add_argument("--executable", type=Path, default=Path("build/Release/tungsten.exe"))
    parser.add_argument("--profile-executable", type=Path,
                        help="Separate instrumented build; plain timings use --executable")
    parser.add_argument("--baseline-executable", type=Path)
    parser.add_argument("--reference-report", type=Path,
                        help="Reuse one completed plain render per case from a matching report; only profiles are rerun")
    parser.add_argument("--output", type=Path, default=Path("build/conditional-profile"))
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--spp", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--affinity-mask", type=lambda value: int(value, 0),
                        help="Windows process CPU affinity, e.g. 0x5555 for one thread per selected core")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--profile-repeats", type=int, default=1)
    parser.add_argument("--steps", default="default,0.02,0.04")
    parser.add_argument("--age-scale", type=float)
    parser.add_argument("--max-expected-events", type=float)
    parser.add_argument("--tighten-expected-events", type=float)
    parser.add_argument("--macrocell-size", type=int)
    parser.add_argument("--macrocell-max-expected-events", type=float)
    parser.add_argument("--cases-json", type=Path,
                        help="JSON mapping case names to tracking_* medium overrides; replaces --steps cases")
    parser.add_argument("--no-profiles", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.medium_type == "classic_global" and not args.no_profiles:
        parser.error("classic_global timing requires --no-profiles")
    if min(args.size, args.spp, args.threads, args.repeats, args.profile_repeats) <= 0:
        parser.error("size, spp, threads and repeat counts must be positive")
    if args.affinity_mask is not None and (os.name != "nt" or args.affinity_mask <= 0):
        parser.error("affinity-mask requires Windows and a positive processor mask")
    if args.reference_report and (args.repeats != 1 or args.baseline_executable or args.no_profiles):
        parser.error("reference-report requires repeats=1, profiles enabled, and no baseline-executable")
    source = args.scene.resolve()
    original = json.loads(source.read_text(encoding="utf-8"))
    indices = [i for i, medium in enumerate(original["media"])
               if medium["type"] == args.medium_type]
    if len(indices) != 1:
        parser.error(f"scene must contain exactly one {args.medium_type} medium")
    medium_index = indices[0]
    executable = args.executable.resolve()
    profile_executable = (args.profile_executable or args.executable).resolve()
    binaries = {executable}
    if not args.no_profiles:
        binaries.add(profile_executable)
    if args.baseline_executable:
        binaries.add(args.baseline_executable.resolve())
    binary_hashes = {str(binary): hashlib.sha256(binary.read_bytes()).hexdigest()
                     for binary in binaries}
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    configuration = {key: getattr(args, key) for key in
                     ("size", "spp", "threads", "seed", "steps", "age_scale",
                      "max_expected_events", "tighten_expected_events", "macrocell_size",
                      "macrocell_max_expected_events", "no_profiles")}
    if args.affinity_mask is not None:
        configuration["affinity_mask"] = args.affinity_mask
    if args.medium_type != "global_conditional":
        configuration["medium_type"] = args.medium_type
    report = {"source": str(source), "scene_snapshot": original,
              "executable": str(executable), "configuration": configuration,
              "profile_executable": str(profile_executable),
              "binary_sha256": binary_hashes,
              "machine": {"platform": platform.platform(), "processor": platform.processor(),
                          "logical_processors": os.cpu_count()}, "runs": [], "parity": {}}
    if args.resume:
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        if report["scene_snapshot"] != original or report["configuration"] != configuration:
            parser.error("saved scene or benchmark configuration differs")
        if report["executable"] != str(executable):
            parser.error("saved executable path differs")
        if report.get("profile_executable", str(executable)) != str(profile_executable):
            parser.error("saved profiling executable path differs")
        if report.get("binary_sha256", binary_hashes) != binary_hashes:
            parser.error("benchmark executable changed since the saved runs")
        report["binary_sha256"] = binary_hashes
    elif (output / "report.json").exists():
        parser.error("output contains a report; use --resume or a new output directory")

    def save():
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    cases = {}
    for step in args.steps.split(","):
        settings = {}
        if step != "default":
            settings["tracking_step"] = float(step)
        if args.age_scale is not None:
            settings["tracking_age_scale"] = args.age_scale
        if args.max_expected_events is not None:
            settings["tracking_max_expected_events"] = args.max_expected_events
        if args.tighten_expected_events is not None:
            settings["tracking_tighten_expected_events"] = args.tighten_expected_events
        if args.macrocell_size is not None:
            settings["tracking_macrocell_size"] = args.macrocell_size
        if args.macrocell_max_expected_events is not None:
            settings["tracking_macrocell_max_expected_events"] = args.macrocell_max_expected_events
        cases["step_" + step.replace(".", "_")] = settings

    if args.cases_json:
        cases = json.loads(args.cases_json.read_text(encoding="utf-8-sig"))
        if not isinstance(cases, dict) or not cases or any(
                not re.fullmatch(r"[A-Za-z0-9_]+", name) or not isinstance(settings, dict)
                or any(not key.startswith("tracking_") for key in settings)
                for name, settings in cases.items()):
            parser.error("cases must map safe names to dictionaries of tracking_* overrides")
        if args.resume:
            if report.get("cases") != cases:
                parser.error("saved benchmark cases differ")
        else:
            report["cases"] = cases
    elif args.resume and "cases" in report:
        parser.error("saved custom cases require --cases-json when resuming")

    if args.reference_report:
        reference_file = args.reference_report.resolve()
        reference = json.loads(reference_file.read_text(encoding="utf-8"))
        if reference["scene_snapshot"] != original:
            parser.error("reference scene differs")
        for key in ("size", "spp", "threads", "seed", "affinity_mask"):
            if reference["configuration"].get(key) != configuration.get(key):
                parser.error(f"reference configuration differs: {key}")
        if reference["binary_sha256"].get(str(executable)) != binary_hashes[str(executable)]:
            parser.error("plain executable differs from reference")
        provenance = {"path": str(reference_file),
                      "sha256": hashlib.sha256(reference_file.read_bytes()).hexdigest()}
        if args.resume and report.get("reference_report") != provenance:
            parser.error("saved reference report differs")
        report["reference_report"] = provenance
        for case, settings in cases.items():
            label = case + "_plain_1"
            candidates = [r for r in reference["runs"] if r["label"] == label
                          and r["case"] == case and not r["profile_enabled"]]
            if len(candidates) != 1:
                parser.error(f"missing unique plain reference for {case}")
            item = candidates[0]
            if (item["settings"] != settings or item["executable"] != str(executable)
                    or item["resolution"] != [args.size, args.size] or item["spp"] != args.spp
                    or item["threads"] != args.threads or not Path(item["pfm"]).is_file()):
                parser.error(f"reference run differs or image missing: {case}")
            if not any(r["label"] == label for r in report["runs"]):
                reused = copy.deepcopy(item)
                reused["reused_from_report"] = str(reference_file)
                report["runs"].append(reused)

    def run(label, case, settings, profile=False, binary=executable, warmup=False):
        existing = next((item for item in report["runs"] if item["label"] == label), None)
        if existing:
            print("Reusing", label, flush=True)
            return existing
        scene = copy.deepcopy(original)
        scene["camera"]["resolution"] = [16, 16] if warmup else [args.size, args.size]
        scene["media"][medium_index].update(settings)
        spp = 1 if warmup else args.spp
        scene["renderer"].update(spp=spp, spp_step=1, hdr_output_file="scene.pfm",
                                 output_file="scene.exr", adaptive_sampling=False,
                                 overwrite_output_files=True, enable_resume_render=False)
        scene_file = output / (label + ".json")
        scene_file.write_text(json.dumps(scene, indent=2), encoding="utf-8")
        run_output = output / label
        environment = os.environ.copy()
        for key in ("TUNGSTEN_GPIS_PROFILE", "TUNGSTEN_GPIS_FS_PROFILE",
                    "TUNGSTEN_GPIS_CONDITIONAL_PROFILE"):
            environment.pop(key, None)
        if profile:
            environment["TUNGSTEN_GPIS_CONDITIONAL_PROFILE"] = "1"
        command = [str(binary), str(scene_file), "-i", str(source.parent),
                   "-d", str(run_output), "-t", str(args.threads), "-s", str(args.seed)]
        log_file = output / (label + ".log")
        print("Rendering", label, flush=True)
        start = time.perf_counter()
        affinity_resets = 0
        affinity_reset_seconds = []
        affinity_observations = {}
        with log_file.open("wb") as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       env=environment)
            if args.affinity_mask is not None:
                import ctypes
                from ctypes import wintypes
                set_affinity = ctypes.windll.kernel32.SetProcessAffinityMask
                set_affinity.argtypes = [wintypes.HANDLE, ctypes.c_size_t]
                set_affinity.restype = wintypes.BOOL
                get_affinity = ctypes.windll.kernel32.GetProcessAffinityMask
                get_affinity.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_size_t),
                                        ctypes.POINTER(ctypes.c_size_t)]
                get_affinity.restype = wintypes.BOOL
                if not set_affinity(int(process._handle), args.affinity_mask):
                    error = ctypes.WinError()
                    process.terminate()
                    process.wait()
                    raise error
                # Runtime libraries can restore process affinity during scene
                # initialization. Observe and enforce the requested benchmark
                # placement instead of merely recording the launch-time mask.
                while True:
                    try:
                        code = process.wait(timeout=0.25)
                        break
                    except subprocess.TimeoutExpired:
                        current, system = ctypes.c_size_t(), ctypes.c_size_t()
                        if not get_affinity(int(process._handle), ctypes.byref(current), ctypes.byref(system)):
                            if process.poll() is not None:
                                code = process.returncode
                                break
                            error = ctypes.WinError()
                            process.terminate()
                            process.wait()
                            raise error
                        key = hex(current.value)
                        affinity_observations[key] = affinity_observations.get(key, 0) + 1
                        if current.value != args.affinity_mask:
                            if not set_affinity(int(process._handle), args.affinity_mask):
                                error = ctypes.WinError()
                                # A short warmup may exit after GetProcessAffinityMask.
                                if process.poll() is not None:
                                    code = process.returncode
                                    break
                                process.terminate()
                                process.wait()
                                raise error
                            affinity_resets += 1
                            affinity_reset_seconds.append(time.perf_counter() - start)
            else:
                code = process.wait()
            cpu = process_cpu_seconds(process)
        wall = time.perf_counter() - start
        log = log_file.read_text(encoding="utf-8", errors="replace")
        if code or not re.search(rf"Completed {spp}/{spp} spp", log):
            raise RuntimeError(f"{label}: incomplete render; inspect {log_file}")
        if re.search(r"Caught an internal error|unrecoverable error|nonfinite|"
                     r"Matrix square root failed|Gradient invalid|Conditioning .* failed", log):
            raise RuntimeError(f"{label}: numerical failure; inspect {log_file}")
        _, pixels = read_pfm(run_output / "scene.pfm")
        result = {"label": label, "case": case, "settings": settings,
                  "profile_enabled": profile, "executable": str(binary),
                  "resolution": scene["camera"]["resolution"], "spp": spp,
                  "threads": args.threads, "wall_seconds": wall, "process_cpu_seconds": cpu,
                  "render_seconds": float(re.findall(r"Completed \d+/\d+ spp, ([\d.]+)/", log)[-1]),
                  "pfm": str(run_output / "scene.pfm"), "mean_rgb": sum(pixels) / len(pixels)}
        if args.affinity_mask is not None:
            result["affinity"] = {"requested": hex(args.affinity_mask),
                                  "restored_count": affinity_resets, "reset_seconds": affinity_reset_seconds,
                                  "observed_masks": affinity_observations}
        prefix = "GPIS conditional profile: "
        lines = [line[len(prefix):] for line in log.splitlines() if line.startswith(prefix)]
        if profile:
            if len(lines) != 1:
                raise RuntimeError(f"{label}: missing or duplicate profile; the profiling "
                                   "executable requires TUNGSTEN_ENABLE_CONDITIONAL_PROFILE=ON")
            result["profile"] = json.loads(lines[0])
            stages = {}
            for domain in result["profile"]["domains"].values():
                for stage, values in domain["stages"].items():
                    stages[stage] = stages.get(stage, 0) + values["exclusive_ms"]
            total = sum(stages.values())
            if total <= 0:
                raise RuntimeError(f"{label}: empty profile")
            result["exclusive_worker_ms"] = stages
            result["conditional_worker_ms"] = total
            result["exclusive_percent"] = {stage: 100 * ms / total for stage, ms in stages.items()}
        elif lines:
            raise RuntimeError(f"{label}: profile unexpectedly enabled")
        if not warmup:
            report["runs"].append(result)
            save()
        print(json.dumps({key: result[key] for key in
                          ("label", "render_seconds", "wall_seconds", "process_cpu_seconds")}),
              flush=True)
        if profile:
            print(json.dumps(result["exclusive_percent"]), flush=True)
        return result

    def check_parity(first, second):
        label = first["label"] + "_vs_" + second["label"]
        metrics = compare(Path(first["pfm"]), Path(second["pfm"]))
        report["parity"][label] = metrics
        save()
        if not metrics["exact"]:
            raise RuntimeError(f"profiling, instrumentation or repetition changed pixels: {label}")

    save()
    run("warmup", "warmup", {}, warmup=True)
    baseline = None
    references = {}
    for repeat in range(args.repeats):
        ordered = list(cases.items())
        if repeat % 2:
            ordered.reverse()
        if args.baseline_executable and repeat == 0 and "step_default" in cases:
            baseline = run("before_instrumentation", "step_default", cases["step_default"],
                           binary=args.baseline_executable.resolve())
        for case, settings in ordered:
            result = run(f"{case}_plain_{repeat+1}", case, settings)
            if case in references:
                check_parity(references[case], result)
            else:
                references[case] = result
                if baseline and case == "step_default":
                    check_parity(baseline, result)
    if not args.no_profiles:
        for repeat in range(args.profile_repeats):
            for case, settings in cases.items():
                suffix = "" if repeat == 0 else f"_{repeat+1}"
                result = run(f"{case}_profile{suffix}", case, settings,
                             profile=True, binary=profile_executable)
                check_parity(references[case], result)
    report["plain_summary"] = {}
    for case in cases:
        runs = [r for r in report["runs"] if r["case"] == case
                and r["label"].startswith(case + "_plain_")
                and r["executable"] == str(executable)]
        times = [r["render_seconds"] for r in runs]
        report["plain_summary"][case] = {"repeats": len(times), "median_seconds": statistics.median(times),
                                        "min_seconds": min(times), "max_seconds": max(times)}
    save()
    print("Saved", output / "report.json", flush=True)


if __name__ == "__main__":
    main()
