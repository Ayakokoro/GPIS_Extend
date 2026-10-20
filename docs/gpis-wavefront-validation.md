# GPIS resumable path scheduler validation

The sections below record the earlier tile scheduler. The current default CUDA
scheduler uses an independent path pool and multiple in-flight submissions; see
[path-pool validation](gpis-path-pool-validation.md) for configuration and results.

Tested on Windows, Release, CUDA 13.1, RTX 4060, eight rendering workers,
seed 12345. Timings below are individual runs, excluding scene setup and output
writing; they are not a statistical benchmark.

The scheduler keeps one path per 16x16 tile, preserving that tile's live sampler
and pixel/sample order. Rendering workers yield at GPIS intersection requests.
The coordinator consumes asynchronous notifications and schedules ready tiles;
CPU work can overlap a GPU submission. Long CPU paths do not hold other tiles
at a wave barrier. Shadow GPIS queries remain on CPU.

The measurements in the bounded/original-view sections below precede asynchronous
shadow support. In that implementation only main-path GPIS segments yielded;
shadow segments ran on CPU. See the asynchronous-shadow section for current results.

## Correctness checks

`ctest --test-dir build -C Release --output-on-failure -R tungsten_gp_`
passes all four tests. The new wavefront test covers:

- Exact legacy/resumable CPU image equality for uniform and Sobol sampling.
- Uniform 2 spp within a single render pass and Sobol 2 spp across two passes.
- Partial 17x13 tiles and a 32x32 image with more suspended paths than workers.
- An injected rejecting backend: every segment falls back to its prepared CPU
  realization with identical output and RNG order.
- Cancellation while an asynchronous reply is withheld: cancellation drains the
  reply, releases paths safely, and does not advance spp or call completion.
- Two outstanding submissions with 900 requests each, two supported media, and
  unsupported requests. Reply order is verified, request arrays can be released,
  and the submissions span GPU chunks of 256 rays.

The original 32x32/1 spp CPU output from the executable saved before this
refactor was also identical to the resumable CPU output.

## Bounded shaderball view, 256x256, 2 spp

This uses the existing shaderball VDB, SE kernel, materials, and lights. The
camera is inside its GP bounding box, at `(0, -0.04, 0.04)`, looking at
`(0, 0, 0.04)`, with camera medium `gp`. This changed view avoids the original
view's expensive long segments and isolates the scheduling/submission costs.
Both direct-light sampling options are enabled; maximum bounces is 2 and both
samples are rendered in one pass.

| Mode | Render time | CUDA batches | GPU segments | Mean batch | Maximum batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU legacy | 1.65 s | — | — | — | — |
| CPU resumable | 1.74 s | — | — | — | — |
| CUDA legacy | 21.05 s | 86,057 | 392,732 | 4.56 | 8 |
| CUDA resumable | 1.41 s | 1,058 | 261,902 | 247.54 | 256 |

The CPU scheduler outputs are exactly equal. The CUDA scheduler outputs are
exactly equal. GPU segment counts decrease because shadow queries now run on
CPU; the larger batches account for the much lower number of GPU submissions.
Neither CUDA run required CPU fallback for submitted segments.

CPU and CUDA outputs differ at 11 of 65,536 pixels by more than 1e-3. Maximum
absolute channel difference is 0.47598; mean absolute difference is 4.306e-5.
This discrepancy is identical for the old and new CUDA schedulers, so this
change preserves the CUDA result in this case. General CPU/CUDA image parity
still requires investigation of crossing arithmetic and subsequent paths.

Scenes, logs, PFM outputs and numeric comparisons are retained under
`build/gp_wavefront_ready_256x256_2spp_bounded/`, including `report.json`.
The view fixture is `build/gp_wavefront_bounded_scene.json`.

```powershell
python tools/verify_wavefront_gpis.py build/gp_wavefront_bounded_scene.json --resolution 256x256 --spp 2 --spp-step 2 --output build/gp_wavefront_ready_256x256_2spp_bounded
```

These timings describe this bounded view. The original view retains expensive
CPU work and GPU fallbacks. The aggregate fallback counter does not distinguish
step limits from invalid input/device errors, and no separate shadow timing was
recorded, so it cannot establish the reason or cost of those fallbacks.

## Original shaderball view, 64x64, 1 spp

Same seed, eight workers, maximum bounces 2, direct-light sampling enabled:

| Mode | Render time | CUDA batches | GPU segments | Mean batch | Maximum batch | GPU segment fallbacks |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CUDA legacy | 35.85 s | 5,163 | 6,624 | 1.28 | 7 | 30 |
| CUDA resumable | 33.93 s | 1,786 | 4,984 | 2.79 | 16 | 20 |

The CUDA images are exactly equal. The maximum batch now reaches the number of
tiles (16), exceeding the worker count (8). Overall time improves only slightly:
expensive CPU work and GPU fallback persist. Individual fallback reasons were not recorded. The fallback counter counts only segments submitted to GPU; shadow
queries moved to CPU are not included in that counter.

The retained results are in `build/gp_wavefront_ready_64x64_1spp_original/`.

```powershell
python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --resolution 64x64 --spp 1 --modes cuda_legacy,cuda_wavefront --output build/gp_wavefront_ready_64x64_1spp_original
```

An initial implementation used complete-wave barriers. A 128x128/2 spp CPU
comparison exposed the long-path waiting problem, so that scheduler was replaced
with the ready queue before the final measurements above. The incomplete
barrier experiment and a long original-view 256x256 CPU experiment were stopped;
they are not used as performance results.

## Asynchronous shadow continuations

GPIS visibility in direct lighting now yields to the same `submitBatch()` queue
as main-path intersections. The continuation retains light intersection data,
MIS inputs, medium state, the shadow ray, and the current forward-boundary
segment. Light sampling and phase/BSDF sampling remain sequential within a path,
preserving RNG order. CPU conditioning and gradient evaluation remain on CPU.
Unsupported media trace on CPU; rejected GPU segments reuse the prepared CPU
realization. The legacy blocking mode is retained for comparison.

The current four CTest checks pass (3.35 seconds in this run). The wavefront test
now explicitly requires submitted shadow requests and exact CPU/rejecting-backend
parity. Added fixtures cover isotropic phase direct lighting and a diffuse cube
inside GP media. Abort is tested with a withheld shadow reply and a separately
withheld completion notification, as well as the existing main-path case.

### Bounded shaderball, 256x256, 2 spp, maximum bounces 2

Eight workers, seed 12345, both direct-light options enabled, Release/RTX 4060.
These are single-run timings, excluding setup/output; they are not statistical
speedup measurements.

| Mode | Render time | Submitted GP segments | Shadow segments | Mean CUDA batch |
| --- | ---: | ---: | ---: | ---: |
| CPU legacy | 1.55 s | - | - | - |
| CPU resumable | 1.73 s | 392,734 | 130,831 | - |
| CUDA legacy | 22.12 s | 392,732 | not separately counted | 4.58 |
| CUDA resumable with asynchronous shadows | 0.96 s | 392,732 | 130,830 | 247.47 |

CUDA resumable used 1,587 batches, maximum 256, with zero GPU-segment fallbacks.
CPU legacy/resumable outputs are exactly equal; CUDA legacy/resumable outputs
are exactly equal. Both outputs also match the corresponding saved outputs from
before this change. The prior resumable implementation with CPU shadows took
1.41 seconds in the historical run. Existing CPU/CUDA crossing differences
remain: 11 pixels above 1e-3, maximum channel difference 0.47598.

Artifacts: `build/gp_shadow_async_256x256_2spp_bounded/report.json`.

### Non-specular volume and diffuse surface lighting

An isotropic-phase fixture with a diffuse cube inside the GP box was rendered
at 32x32, 2 spp, maximum bounces 4. CPU legacy/resumable outputs and CUDA
legacy/resumable outputs are respectively exactly equal. CUDA resumable submitted
3,948 shadow segments and 4,978 main-path segments, with zero fallbacks.
Only four tiles were available, so batches averaged 1.99 rays; CUDA resumable
took 0.74 seconds versus 0.59 seconds for blocking CUDA. Small images can remain
limited by submission overhead. CPU/CUDA differences also persist in this fixture;
backend-to-backend equality is not a general numerical equivalence guarantee.

Artifacts: `build/gp_shadow_async_surface_volume/report.json`.

### Maximum bounces 20

The bounded original-material view was also checked at 32x32, 2 spp, maximum
bounces 20. CPU legacy/resumable and CUDA legacy/resumable outputs remain
respectively exactly equal. CUDA resumable submitted 2,335 shadow segments and
4,383 main-path segments, with zero fallbacks. The small-image timings were
0.51 seconds resumable versus 0.44 seconds blocking CUDA.

Artifacts: `build/gp_shadow_async_20bounces/report.json`.

## Current external shaderball view and boundary repair

See [current-scene performance diagnosis](gpis-current-scene-performance.md) for
confirmed epsilon-skipped GP exits, the repair, and CPU/CUDA measurements using
the original external camera at 256x256, 1 spp, maximum bounces 20. Historical
measurements above precede this boundary repair.
