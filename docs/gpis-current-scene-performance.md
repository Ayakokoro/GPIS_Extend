# Current shaderball view: performance diagnosis and boundary repair

This records the diagnosis before the path-pool implementation. The boundary
repair remains in place; see [path-pool validation](gpis-path-pool-validation.md)
for the subsequent scheduling changes and updated CPU/CUDA timings.

The current `gpis-scenes/shaderball_ply/scene.json` snapshot uses the external
camera `(0.134, -0.4, 0.2)`, 256x256, 1 spp, maximum bounces 20, and both direct
light options. Its integrator omits `use_cuda_gpis`, which defaults to CPU.
Benchmarks enable CUDA in generated copies. The camera, GP parameters, and
lighting are preserved; runs use eight workers and seed 12345.

The earlier 0.96-second result used a camera **inside** the GP box, maximum
bounces 2, and 2 spp. It does not represent this external view. Comparing
blocking CUDA with asynchronous CUDA also does not measure CPU-to-CUDA speedup.

## Confirmed long-ray bottleneck

Before the repair, a 64x64 diagnostic with this same camera and maximum bounces
20 recorded:

| Mode | Process wall time | Render time | GPU step-limit fallbacks |
| --- | ---: | ---: | ---: |
| CPU legacy | 14.97 s | 14.55 s | - |
| CUDA resumable | 22.78 s | 22.23 s | 17 |

All 17 rejected GPU segments were step limits, not invalid input or callback
errors. Their `farT` was 2000. CPU fallback marched 77,232,059 steps, with
81,567 ms summed across worker threads. This sum is not frame wall time.
A short CTest invocation overlapped the beginning of this diagnostic run, so
these timings are descriptive, not an isolated statistical benchmark.

For example, one rejected main-path segment started at
`(-0.0337273, 0.0473607, 0.0845245)`, pointing
`(-0.341237, 0.910145, -0.234932)`, with `nearT=0.0005`.
The box's upper Y face is approximately 0.0475450; the exit is approximately
0.000203 away, before `nearT`. Both scene/BVH intersection and the primitive's
normal intersection can skip that exit. The current medium then remains GP,
so an infinite ray receives the existing 2000-distance GP fallback. Shadow
origins rounded just outside a face exhibit a related miss with `nearT=0`.

Artifacts: `build/gp_current_scene_profile_64x64/report.json` and its logs.

## Repair

`TraceBase::intersectScene()` queries exits directly for transparent cubes
whose interior GP medium matches the current ray medium. `Cube::intersectMediumExit()`
recovers exits at/before `nearT`, allowing only floating-point tolerance around
the face. Ordinary exterior-ray near clipping is preserved. Both legacy and
resumable PathTracer main paths and direct-light shadow paths use this query.
Empty GP intervals finish without marching. The GPU step limit and march step
size are unchanged.

The four CTest checks pass. Added boundary tests cover both directions on all
three axes, exits at/before nearT, and shadow origins one ULP outside a face.
CPU legacy/resumable outputs remain exactly equal. CUDA legacy/resumable
outputs also match exactly in the 256x256 repaired-scene comparison.

The repair changes boundary behavior, so matching the old incorrect image is
not expected: in the 64x64 comparison 34 CPU pixels and 35 CUDA pixels changed
by more than 1e-3. CPU and CUDA outputs themselves still differ; their GP
crossing arithmetic and subsequent paths need separate investigation.

After the repair, the 64x64 diagnostic recorded CPU render time 0.06 seconds
and CUDA render time 0.62 seconds, with zero GPU fallbacks and zero CPU marches
in CUDA mode. Artifacts: `build/gp_current_scene_boundary_fixed_64x64/`.

## Full current resolution after repair

Two final runs, with optional profiling disabled:

| Mode | Process wall time | Render stage |
| --- | ---: | ---: |
| CPU legacy | 1.42-1.52 s | 1.02-1.05 s |
| CUDA resumable | 2.19-2.25 s | 1.66-1.69 s |

CUDA is still slower than CPU for this configuration. It submits 155,449 GP
segments, including 51,056 shadow segments, in 4,598-4,601 batches. Average batch
size is 33.8, maximum 232, with zero CPU fallbacks. The profile-enabled run
recorded 8,173,245 GPU march steps, around 53 steps per submitted segment.
The original scene offers substantially smaller average batches than the
inside-box benchmark, whose average was 247.5 rays.

Each tile retains one live path to preserve its continuous uniform RNG stream.
The remaining concurrency, per-segment scheduling/transfer costs, CPU
conditioning/gradient/shading, and process/device setup limit the benefit of
moving the march alone to CUDA. Increasing simultaneously active paths and
reducing submission/resumption traffic are the next architectural priorities;
preserving sampler semantics requires care.

A 32-thread block experiment did not show a meaningful gain (1.51 seconds
rendering versus 1.53 seconds in the preceding 128-thread profile run). The
original 128-thread launch was restored. Images matched exactly across launch
sizes. Final images also match the corresponding repaired 128-thread reference.

Final artifacts:
- `build/gp_current_scene_final_256x256_run1/report.json`
- `build/gp_current_scene_final_256x256_run2/report.json`
- `build/gp_current_scene_boundary_fixed_256x256/report.json` (profile/parity)

Reproduce:

```powershell
cmake --build build --config Release --target tungsten tungsten_gp_wavefront_test
ctest --test-dir build -C Release --output-on-failure -R tungsten_gp_
python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --resolution 256x256 --spp 1 --bounces 20 --modes cpu_legacy,cuda_wavefront --profile --output build/gp_current_scene_profile
```
