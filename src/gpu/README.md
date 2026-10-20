# GPIS CUDA backend

For the fix to a 512x512 / 32 spp render stalled at 15/32, see
[cube edge exit diagnosis](../../docs/gpis-32spp-stall.md). Cube intersections now
retain the actual slab face for normals and medium transitions.

This is an optional, standalone GPU library. The CPU renderer does not depend on it.
Configure with `-DTUNGSTEN_ENABLE_CUDA=ON`; set `CMAKE_CUDA_ARCHITECTURES`
for the target GPU when needed. Build the `tungsten_gp_cuda` target, or build
`tungsten_gp_cuda_test` and run CTest.

`cuda/Kernel1D.hpp` is the host dispatch interface. The caller owns all device
buffers, passes arrays of query and center coordinates, and chooses a CUDA stream.
`launchKernel1D` queues work and returns a launch status; the caller must
synchronize before using the results. `Kernel1DKind` is the extension point for
later 1D kernel families. CUDA implementation files stay under `cuda/` and are
not included in Tungsten's recursive `src/core/*.cpp` target.

The first implementation is the 1D squared exponential **splat** used by
`SparseConvolutionNoiseRealization::cell1D()`. It returns

`value = exp(-(query - center)^2 / (2 * localScale^2))`

and its derivative with respect to `query`. The GP amplitude (`sigma`), grid
transform, normalization, impulse weights, and ray marching are handled by the
ray segment batcher described below.

## Ray segment batches

`cuda/RaySegmentBatch.hpp` has two layers:

- `launchSe1DRaySegments()` accepts device arrays and a device mean grid. It
  enqueues one complete segment per CUDA thread and does not synchronize.
- `Se1DRayBatcher` accepts host arrays, uploads the mean grid once, reuses ray
  buffers, and synchronizes once per `trace()` call. For asynchronous use,
  `reserve()` device buffers first, then call `enqueue()` with pinned host
  arrays and a completion callback. The batcher is not thread safe; use one
  instance per execution slot/stream. `enqueueWithGrid()` lets these slots share
  a borrowed immutable medium grid while owning separate ray buffers.

Each ray supplies its origin, unit direction, near/far bounds, one path-sampler
jitter, its existing sparse-noise seed, and its normal seed. For a secondary
segment it also supplies the previous GPIS value and gradient; the CUDA kernel
computes the stationary SE 1D Renewal+ value and longitudinal-gradient
conditioning before marching. The result contains a first-hit distance and
gradient, or the field value at the end of a miss. Check `RaySegmentStatus`
before using either result. `StepLimit` means the configured maximum number of
march steps was reached and is not a valid miss.

`makeHostMeanGrid()` in `VdbMeanGrid.hpp` exports an already loaded `VdbGrid`
with linear interpolation and no detail or emission grid. Pass that result to
`setMeanGrid()`; it preserves Tungsten's world-to-index transform and boundary
clamp. The optional `tungsten_gp_cuda_vdb` target supplies this adapter.

```cpp
Tungsten::Gpu::HostMeanGrid mean;
if (!Tungsten::Gpu::makeHostMeanGrid(vdbGrid, meanOffset, meanScale, mean))
    throw std::runtime_error("unsupported VDB mean");
Tungsten::Gpu::Se1DRayBatcher batcher;
checkCuda(batcher.setMeanGrid(mean));
checkCuda(batcher.trace(rays.data(), hits.data(), rays.size(), parameters));
```

Fill `noiseSeed` using the existing `computeSeed()` rule and `normalSeed` using
`xxhash32(pixelSampleSegment) + xxhash32(sceneSeed) + 1`. Supply the same jitter
that the CPU path sampler would pass to `intersectGP()`. `lengthScale`,
`amplitude`, `kernelRadius`, `impulseDensity`, and `maxStepSize` correspond to
the scene's SE covariance and sparse-convolution medium settings.

The current implementation supports stationary, isotropic, 1D SE sparse noise
with a single tabulated mean. It does not handle 3D sampling, nonstationary
covariances, extra means, or the optional XY gradient correlation.

## PathTracer integration

Build the `tungsten` target with `-DTUNGSTEN_ENABLE_CUDA=ON`. It connects
`PathTracer` to a shared CUDA dispatcher. Enable it in a scene's path tracer
integrator:

```json
"integrator": {
    "type": "path_tracer",
    "use_cuda_gpis": true
}
```

The default is `false`, including in CUDA builds. `TUNGSTEN_GPIS_CUDA=0`
temporarily disables the backend even when the scene option is true. A render
that uses the backend prints the number of CUDA batches, submitted rays, and
CPU fallbacks at completion.

`use_cuda_gpis: true` selects the resumable **path pool** by default. Explicit
`wavefront_gpis: false` selects blocking `traceSample()` for comparisons.
`wavefront_gpis: true` with `use_cuda_gpis: false` exercises the pool with CPU
ray marching. The pool defaults to independent path sampling. To compare with
blocking CPU fairly, also set `independent_path_sampling: true` on that CPU scene.

```json
"integrator": {
    "type": "path_tracer",
    "use_cuda_gpis": true,
    "wavefront_gpis": true,
    "gpis_path_pool": true,
    "independent_path_sampling": true,
    "gpis_max_active_paths": 4096,
    "gpis_batch_size": 1024,
    "gpis_inflight_batches": 4
}
```

The capacities are configurable: active paths 1..1048576, batch size 1..65536,
in-flight batches/slots 1..64. Pool mode requires independent sampling.
`gpis_path_pool: false` retains the one-path-per-tile scheduler. For the historical
continuous tile RNG as well, explicitly set `independent_path_sampling: false`.
The input shaderball scene is not automatically switched to CUDA.

### Resumable paths

`PathTracer::PathState` stores the live ray, throughput, medium state, intersection
data, and medium-distance scratch at a stable address. `advanceSample()` runs CPU
work until the path finishes or returns `NeedGpis`; `resumeGpis()` publishes the
reply, and the next CPU wave resumes that path. Surface-event pointers keep
referring to that path's intersection data and its own sampler.

`GaussianProcessMedium::beginDistance()` and `resumeDistance()` split distance
sampling around intersection, including the case that another segment is
required. The synchronous `sampleDistance()` wrapper uses these same functions.
`beginTransmittance()` and `resumeTransmittance()` do the same for GP visibility:
intersection replies resume gradient/state updates before advancing the shadow
ray. Unsupported media continue through their synchronous CPU transmittance.
`SparseConvolutionNoiseMedium::prepareGpisSegment()` performs CPU conditioning,
draws jitter once, and saves the realization and its auxiliary sampler.
`finishGpisSegment()` applies a valid GPU reply or traces that exact prepared
segment on CPU. Failed submissions and `StepLimit` do not recondition the noise
or consume another path-sampler sample.

The pool owns stable slots, each containing a path and an
`IndependentPathSampler`. Tiles only enumerate pixel/sample jobs. Free slots
are refilled as paths finish; concurrent paths can belong to the same tile or
pixel. Handles include a generation to reject stale replies after slot reuse.
The live pool is bounded by `gpis_max_active_paths`; a separate ordered retirement
window is bounded by twice that capacity. A long early path can temporarily
limit refill when that window is full, without growing an image-sized path array.

Independent sampling keys all RNG state by scene seed, pixel index and absolute
sample index. In Sobol mode the scramble is fixed per pixel and the supplemental
PCG is reset per sample, including boolean/discrete draws and dimensions beyond
1024. This changes historical same-seed samples. GP noise/normal seed formulas
are unchanged. Independent mode uses the absolute sample index for the GP path
state too, including adaptive allocations.

Rendering workers advance paths until completion or a GP suspension; they never
wait on supported main-path or shadow queries. The coordinator consumes worker
events, maintains per-medium request queues, and keeps multiple submissions in
flight. Full chunks are submitted immediately. When CPU producers are drained,
remaining tail requests are flushed without a timed batching delay. Completed
chunks resume independently, without a global wave barrier.

Color, depth, normal, albedo, visibility and adaptive statistics are committed by
the coordinator in tile/pixel/sample order. Workers retain deferred AOV values in
the path, so multiple samples of the same pixel cannot race on output buffers.
Adaptive job counts and sample indices are fixed for each render pass. Direct
lighting retains light and phase/BSDF visibility continuations, forward surfaces
and medium transitions; `path` and `shadow` segment counts are logged separately.

### Submissions and lifetime

`GpRayBatchBackend::submitBatch()` returns immediately with a future for the complete reply vector.
Replies retain request order, including unsupported and failed requests. One
dispatcher thread groups requests by medium and enqueues chunks up to
`gpis_batch_size` without a timed batching delay. It can take new requests while
other batches are in flight. Every execution slot owns a nonblocking CUDA stream,
pinned input/output arrays and separate device ray buffers; medium grids are
uploaded once and shared read-only. CUDA callbacks publish replies and release
slots, and never resume shading or call CUDA APIs.

Submissions own copied requests until completion. Pool paths stay at stable
addresses until their replies are consumed. Cancellation stops new scheduling,
drains every outstanding future **and notification**, waits for CPU tasks, and
joins the coordinator before freeing paths. The backend joins its dispatcher and
drains each stream before releasing slot buffers or shared grids. An aborted
pass does not advance completed spp or invoke its completion callback.

The GPU computes SE conditioning, ray marching and a hit gradient. CPU preparation
still constructs a conditioned realization for fallback and CPU consumers; the
core reply currently does not carry the GPU gradient, so gradient sampling,
phase functions, direct lighting and path continuation still run on CPU.
Those changes are a separate next stage. Each batch still copies rays both ways.
See [path-pool validation](../../docs/gpis-path-pool-validation.md) for measured
batch sizes, CPU/CUDA timings, correctness checks and reproduction commands.
CPU and CUDA field arithmetic can yield different crossings and later paths;
exact CPU scheduler parity does not establish general CPU/CUDA image parity.

### Verification

```powershell
cmake --build build --config Release --target tungsten tungsten_gp_wavefront_test
ctest --test-dir build -C Release --output-on-failure -R tungsten_gp_
python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --resolution 128x128 --spp 2
```

The wavefront test checks exact legacy/resumable CPU image equality, prepared
fallback equality, continuous uniform RNG across multiple spp, Sobol sampling,
partial tiles, non-specular volume and diffuse surface direct lighting,
cancellation while either a path or shadow submission is pending, and ordered replies
from multiple asynchronous submissions and media. The benchmark script retains
scene files, logs, PFM outputs, and `report.json`. Use `--no-direct-lighting` to
measure path batching without direct-light shadow queries, and `--spp-step 2` to
exercise several samples per pixel in one render pass.

### GP boundary exits and profiling

Transparent cubes that bound the current GP medium receive an additional exit
query when an exit falls at or before `nearT`. This handles BVH/epsilon-skipped
exits and floating-point roundoff at a face. The query is restricted to that
current medium and allows only a small floating-point tolerance around the cube.
Empty GP intervals exit immediately; they do not submit a CUDA ray march.
Both synchronous and resumable PathTracer paths use this boundary handling.

Enable optional diagnostic counters with `TUNGSTEN_GPIS_PROFILE=1`, or:

```powershell
python tools/verify_wavefront_gpis.py gpis-scenes/shaderball_ply/scene.json --resolution 256x256 --spp 1 --bounces 20 --modes cpu_legacy,cuda_wavefront --profile --output build/gp_current_scene_profile
```

The profile reports GPU step-limit/invalid-input/callback failures, rejected ray
bounds, GPU march steps, and CPU conditioning/gradient/marching time. CPU times
are sums across workers, not frame wall time. Profiling is disabled by default.
