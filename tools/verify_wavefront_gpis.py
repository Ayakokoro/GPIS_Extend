"""Render fixed-seed GPIS scheduler comparisons and retain scenes/logs/PFM metrics."""
import argparse
import array
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def read_pfm(path):
    with path.open("rb") as stream:
        magic = stream.readline().strip()
        width, height = map(int, stream.readline().split())
        scale = float(stream.readline())
        data = array.array("f")
        data.frombytes(stream.read())
    assert magic == b"PF" and len(data) == width * height * 3
    if (scale < 0) != (sys.byteorder == "little"):
        data.byteswap()
    assert all(math.isfinite(x) for x in data), "nonfinite image output"
    return (width, height), data


def compare(first, second):
    shape_a, a = read_pfm(first)
    shape_b, b = read_pfm(second)
    assert shape_a == shape_b
    differences = [abs(x-y) for x, y in zip(a, b)]
    return {
        "exact": a == b,
        "max_abs": max(differences),
        "mean_abs": sum(differences) / len(differences),
        "rmse": math.sqrt(sum(x*x for x in differences) / len(differences)),
        "pixels_above_1e-3": sum(max(differences[i:i+3]) > 1e-3
                                for i in range(0, len(differences), 3)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scene", type=Path)
    parser.add_argument("--executable", type=Path, default=Path("build/Release/tungsten.exe"))
    parser.add_argument("--output", type=Path, default=Path("build/gp_wavefront_validation"))
    parser.add_argument("--resolution", default="128x128")
    parser.add_argument("--spp", type=int, default=2)
    parser.add_argument("--spp-step", type=int, default=1)
    parser.add_argument("--bounces", type=int, default=2)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--no-direct-lighting", action="store_true")
    parser.add_argument("--profile", action="store_true", help="Report GP CPU work and GPU fallback reasons")
    parser.add_argument("--legacy-tile-sampling", action="store_true",
                        help="Use the historical tile scheduler and continuous tile RNG")
    parser.add_argument("--tile-scheduler", action="store_true",
                        help="Use one path per tile, retaining independent sampling for comparison")
    parser.add_argument("--active-paths", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--inflight-batches", type=int, default=4)
    parser.add_argument("--modes", default="cpu_legacy,cpu_wavefront,cuda_legacy,cuda_wavefront")
    args = parser.parse_args()
    resolution = list(map(int, args.resolution.lower().split("x")))
    assert len(resolution) == 2 and min(resolution) > 0 and args.spp > 0
    scene_path = args.scene.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads(scene_path.read_text())
    environment = os.environ.copy()
    if args.profile:
        environment["TUNGSTEN_GPIS_PROFILE"] = "1"
    report = {"scene": str(scene_path), "resolution": resolution, "spp": args.spp,
              "spp_step": args.spp_step, "bounces": args.bounces, "threads": args.threads,
              "seed": args.seed, "direct_lighting": not args.no_direct_lighting,
              "path_pool": not (args.legacy_tile_sampling or args.tile_scheduler),
              "independent_sampling": not args.legacy_tile_sampling,
              "active_paths": args.active_paths, "batch_size": args.batch_size,
              "inflight_batches": args.inflight_batches,
              "runs": {}, "comparisons": {}}
    for mode in args.modes.split(","):
        assert mode in {"cpu_legacy", "cpu_wavefront", "cuda_legacy", "cuda_wavefront"}
        scene = json.loads(json.dumps(original))
        scene["camera"]["resolution"] = resolution
        integrator = scene["integrator"]
        integrator.update(use_cuda_gpis=mode.startswith("cuda"),
                          wavefront_gpis=mode.endswith("wavefront"), max_bounces=args.bounces,
                          gpis_path_pool=report["path_pool"],
                          independent_path_sampling=report["independent_sampling"],
                          gpis_max_active_paths=args.active_paths, gpis_batch_size=args.batch_size,
                          gpis_inflight_batches=args.inflight_batches)
        if args.no_direct_lighting:
            integrator.update(enable_light_sampling=False, enable_volume_light_sampling=False)
        renderer = scene["renderer"]
        renderer.update(spp=args.spp, spp_step=args.spp_step, adaptive_sampling=False,
                        enable_resume_render=False, overwrite_output_files=True,
                        hdr_output_file="scene.pfm", output_file="scene.exr")
        scene_file = output / (mode + ".json")
        scene_file.write_text(json.dumps(scene, indent=2))
        run_output = output / mode
        command = [str(args.executable.resolve()), str(scene_file), "-i", str(scene_path.parent),
                   "-d", str(run_output), "-t", str(args.threads), "-s", str(args.seed)]
        print("Rendering", mode, flush=True)
        start = time.monotonic()
        log_path = output / (mode + ".log")
        with log_path.open("wb") as stream:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=environment)
        wall = time.monotonic() - start
        log = log_path.read_text(encoding="utf-8", errors="replace")
        process.check_returncode()
        if not re.search(rf"Completed {args.spp}/{args.spp} spp", log):
            raise RuntimeError(f"{mode}: render did not complete the requested spp; inspect {log_path}")
        read_pfm(run_output / "scene.pfm")
        run = {"wall_seconds": wall, "pfm": str(run_output / "scene.pfm")}
        completed = re.findall(r"Completed \d+/\d+ spp, ([\d.]+)/", log)
        if completed:
            run["render_seconds"] = float(completed[-1])
        batches = re.search(r"GPIS CUDA batches: (\d+), rays: (\d+), CPU fallbacks: (\d+)(?:, largest batch: (\d+))?", log)
        if batches:
            run.update(batches=int(batches[1]), rays=int(batches[2]), fallbacks=int(batches[3]),
                       largest_batch=int(batches[4] or 0))
            run["rays_per_batch"] = run["rays"] / max(1, run["batches"])
        elif mode.startswith("cuda"):
            raise RuntimeError(f"{mode}: CUDA backend was not enabled; inspect {log_path}")
        waves = re.findall(r"GPIS wavefront waves: (\d+), rays: (\d+), largest wave: (\d+)", log)
        if waves:
            run.update(waves=sum(int(x[0]) for x in waves),
                       wave_rays=sum(int(x[1]) for x in waves),
                       largest_wave=max(int(x[2]) for x in waves))
        segments = re.findall(r"GPIS wavefront segments: path=(\d+), shadow=(\d+)", log)
        if segments:
            run.update(path_segments=sum(int(x[0]) for x in segments),
                       shadow_segments=sum(int(x[1]) for x in segments))
        pools = re.findall(r"GPIS path pool: capacity=(\d+), peak active=(\d+), "
                           r"peak submissions=(\d+), peak buffered results=(\d+), committed=(\d+)", log)
        if pools:
            run["path_pool"] = dict(capacity=int(pools[0][0]),
                peak_active=max(int(x[1]) for x in pools),
                peak_submissions=max(int(x[2]) for x in pools),
                peak_buffered_results=max(int(x[3]) for x in pools),
                committed=sum(int(x[4]) for x in pools))
        execution = re.search(r"GPIS CUDA execution: slots=(\d+), peak in flight=(\d+)", log)
        if execution:
            run["cuda_execution"] = dict(slots=int(execution[1]), peak_inflight=int(execution[2]))
        gpu_profile = re.search(r"GPIS CUDA profile: step_limit=(\d+), invalid_input=(\d+), "
                                r"device_callback_failure=(\d+), total_steps=(\d+)", log)
        if gpu_profile:
            run["gpu_profile"] = dict(zip(
                ("step_limit", "invalid_input", "device_callback_failure", "total_steps"),
                map(int, gpu_profile.groups())))
        cpu_profile = re.search(r"GPIS CPU profile \(sum of worker times\): conditioning_ms=([\d.eE+-]+), "
                                r"gradient_ms=([\d.eE+-]+), marching_ms=([\d.eE+-]+), "
                                r"marching_calls=(\d+), steps=(\d+)", log)
        if cpu_profile:
            run["cpu_profile"] = dict(zip(
                ("conditioning_worker_ms", "gradient_worker_ms", "marching_worker_ms"),
                map(float, cpu_profile.groups()[:3])))
            run["cpu_profile"].update(marching_calls=int(cpu_profile[4]), steps=int(cpu_profile[5]))
        report["runs"][mode] = run
        print(json.dumps(run), flush=True)
    for first, second in [("cpu_legacy", "cpu_wavefront"), ("cuda_legacy", "cuda_wavefront"),
                          ("cpu_wavefront", "cuda_wavefront")]:
        if first in report["runs"] and second in report["runs"]:
            report["comparisons"][first + "_vs_" + second] = compare(
                Path(report["runs"][first]["pfm"]), Path(report["runs"][second]["pfm"]))
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["comparisons"], indent=2))
    cpu = report["comparisons"].get("cpu_legacy_vs_cpu_wavefront")
    if cpu and not cpu["exact"]:
        raise RuntimeError("CPU scheduler parity failed")


if __name__ == "__main__":
    main()
