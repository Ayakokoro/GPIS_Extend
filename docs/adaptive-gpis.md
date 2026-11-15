# Adaptive function-space GPIS reference

`adaptive_function_space_gaussian_process` implements an initial CPU reference
of Zhou, Seyb and Zhao, *Adaptive Ray Marching for Rendering Gaussian Process
Implicit Surfaces* (SIGGRAPH 2026):

- [Paper, Sections 4.1–4.2](https://111116.github.io/fast-gpis-site/static/pdfs/fast-gpis-compressed.pdf)
- [Supplement, Sections 1.3–1.5](https://111116.github.io/fast-gpis-site/static/pdfs/fast-gpis-supplemental.pdf)

This is a working stationary-SE implementation and validation harness, not a
claim to reproduce the paper's complete scene set or equal-time speedups.

## Implementation and supported domain

`src/core/math/AdaptiveGpis.{hpp,cpp}` contains the independent online sampler,
range-bound coefficients and marcher. The sampler draws correlated values and
directional derivatives by extending a Cholesky factor one observation at a
time. Field values are normalized by sigma, gradients by sigma/ell; the
normalized value/derivative covariance is evaluated analytically for isotropic
SE and checked against the existing `CovarianceFunction` derivative interface.

`src/core/media/AdaptiveFunctionSpaceGaussianProcessMedium.{hpp,cpp}` adapts it
to Tungsten's GPIS medium and observed-normal BRDF phase. The type is registered
in `MediumFactory.cpp`. Existing media keep their implementations and settings.

Supported inputs:

- `gaussian_process.type: standard`, stationary isotropic `squared_exponential`
  covariance with positive sigma and lengthScale.
- Spherical, linear and homogeneous means. Procedural means are supported when
  their `SdfQueries::rayBounds` provides certified finite directional-derivative
  bounds, including the supplied knob shaderball fixture.
- `correlation_context: renewal+`, `intersect_method: gp_discrete`, and
  `normal_method: conditioned_gaussian`.
- A `brdf` phase, finite process boundaries, CPU path tracing. The checked
  examples use `use_cuda_gpis: false` and `wavefront_gpis: false`.

Nonstationary/anisotropic kernels, tabulated means without the required bound
provider, CSG, embedded covariance, external conditioning and multiple phase
functions are rejected. State-free distance PDFs / bidirectional transport are
unsupported. No CUDA kernel is supplied.

## Sampling, history and bounds

One flight keeps its retained online history, including all newly sampled
root-refinement points. There is no batch boundary or sliding history window.
A `max_samples` limit raises an error instead of deleting old points or
returning a miss. A straight transparent split within the same medium retains
the sampler. Direction changes on ordinary geometry retain the existing field
observations until the next GP collision. Leaving the medium resets history.

After a GP collision, the next flight uses Renewal+: its value is zero and its
full unnormalized gradient is the observed collision gradient. Each shadow or
continuation constructs a separate context; shared state is never modified.
This is not full-history conditioning across all bounces.

For an exterior start, the original GP value is sampled without inventing a
known value or gradient. A sampled nonpositive origin is classified as already
inside solid and absorbed (opaque for shadows), rather than renormalizing the
initial distribution to a positive truncated Gaussian.

For a surface birth, the conditioned mean is

```
mu_birth(x) = mu(x) + sigma exp(-|r|²/2) (a + r·b)
r = (x-origin)/ell
a = -mu(origin)/sigma
b = ell (gradient-origin_mean_gradient)/sigma
```

Its mean Lipschitz bound includes
`sigma/ell * (abs(a)/sqrt(e) + |b|)` in addition to the base mean bound.
The posterior variance and mixed-derivative bounds are conservatively bounded
by their original SE values, `sigma²` and `sigma²/ell²`. The long-range bound
uses this birth-conditioned mean, not the original unconditioned mean.

The two inequalities use half of `probability_tolerance` each, as required by
the supplement's union-bound discussion. The long-range coefficient minimizes
the stated expression analytically: with `A=1+C1*sqrt(M)/Lambda`, its minimum
satisfies `z=sqrt(log(A/risk))` and
`kappa=erfc(z)/(risk*sqrt(2/pi))`. Zero mean Lipschitz uses only the local bound.
The mean is oriented by the observed field sign; taking `abs(mean)` when the
mean and sampled value disagree would incorrectly permit large steps.

The supplement explicitly says its local bound is not conditioned on the
observed value. This implementation follows the paper's marching rule, but
does not interpret eta as a certified conditional or whole-path failure budget.
The minimum-step clamp, approximate root location, birth escape and numerical
rank threshold introduce additional errors. Tightening eta alone does not
remove them. A sign-change bracket is assumed to contain the relevant first
root; multiple roots within a step remain a discretization concern.

## Numerical handling

All sampler arithmetic is now **double**. The previous 100-digit multiprecision
path has been removed. The normal path incrementally extends a Cholesky factor,
reuses the prediction in the update, and uses fused multiply-adds. Exact repeated
observations return the previously observed value.

When the Cholesky conditional variance falls below `sqrt(machine_epsilon)`
(about 1.49e-8 in normalized units), the sampler switches to a double-precision
rank-revealing eigendecomposition of its retained covariance matrix. It caches
the whitening transform until another observation is appended. If subtracting
the explained variance from one still gives a significant negative result, it
factors the augmented covariance and uses a double SVD to compute variance as
a squared null-space projection. Invalid covariance matrices still raise errors.
Previously drawn values are never resampled and retained observations are not
evicted. No diagonal jitter or observation noise is added.

`rank_tolerance` is an explicit cutoff in unit-variance normalized coordinates,
used for conditional variance and for eigenvalues in the rank-revealing solve.
A redundant query returns its conditional mean without appending an observation.
Truncating tiny eigenvalues is a numerical rank approximation, not exact GP
sampling at arbitrary precision. The stabilization path can refactor the matrix
after each retained observation, so it still differs from the paper's purely
incremental performance path. Statistics report `arithmetic: double`,
`double_rank_decompositions` and redundant-query counts.

Root refinement evaluates the same online realization using bisection, then
interpolates the final bracket. Its gradient is sampled from the joint
conditional derivative distribution, not replaced by the bracket's secant
slope. The next bounce sets the approximate root value to zero. The outgoing
origin uses the renderer's float position and excludes an interval no larger
than `root_tolerance` to avoid immediately rediscovering the birth root.
Inward births are opaque; zero/nonfinite or outward hit gradients are rejected
and counted. Inspect these counters when changing numerical settings.

## Optional 1D MIS for mirror and conductor materials

Enable the conditional-normal lighting estimator on the medium with
`"1D_sampling": true, "1D_sampling_scheme": "mis"`. The default remains
observed-normal reflection. `uni` uses the same new gradient sampler without
light connections, and `nee` uses light connections without the competing
direct forward estimator. All modes keep path continuation and Renewal+.

At a hit, the online sampler computes the joint conditional gradient moments
in an orthonormal ray frame, using all retained field and birth observations.
It first samples the longitudinal derivative, then samples the two transverse
components from their conditional Gaussian. The frozen longitudinal derivative
defines a plane of possible gradients. For a candidate normal n and unit
incoming travel direction w, the full gradient is
`g = n * g_parallel / dot(n,w)`. With `c = -dot(n,w) > 0`, its reflected-direction
density is `p_xy(g_x,g_y) * g_parallel^2 / (4*c^4)`. Both plane-area to normal-solid-angle
and normal to reflected-direction Jacobians are included.

`SpecularGpisEvent` carries this immutable distribution into `BRDFPhaseFunction`;
it does not cast an adaptive context to the old sparse-convolution context.
Mirror albedo and conductor Fresnel are evaluated with the candidate normal.
The renderer's existing power-heuristic MIS combines light and forward samples.
For shadow rays, the next Renewal+ flight uses the connection's full gradient
from `MediumState::lastAniso`, independently of the main continuation. A
numerically degenerate transverse distribution falls back to ordinary delta
reflection and is counted in `degenerate_gradient_events`.
That counter also includes invalid longitudinal derivatives, whose paths are
rejected and separately counted in `invalid_gradients`.

This is still a double, stationary isotropic SE implementation with the
documented rank, marching and root approximations. The sparse-convolution
transverse-correlation approximation is not imported: the online posterior
already includes the retained observations. MIS does not remove marching bias.

Use the portable [mirror small-light fixture](../data/example-scenes/adaptive-gpis/shaderball-small-light-mis.json)
or the archive-based runner:

```powershell
python -B tools/render_adaptive_shaderball_small_light.py --sampling mis --size 256 --spp 128 --threads 32
```

The runner preserves the existing mirror, camera, GP, radius-0.06 sphere emitter
at [-3,3,4] with emission 2000, no floor/environment and 32 path bounces. It keeps
the previous adaptive settings (minimum step 0.01, root tolerance 1e-5,
rank tolerance 1e-10). Use a distinct `--output` for different sampling modes.
The report links the earlier observed-normal result and records binary hashes,
source differences, finite linear buffers, noise diagnostics and MIS counters.

Tests verify joint posterior moments against dense conditioning, numerical
normalization of the reflected-direction PDF, forward/NEE/MIS estimates against
quadrature, mirror/conductor evaluation and branch-gradient isolation. The
existing sparse-convolution MIS 32x32/16-spp regression has a bit-identical linear
image before and after the integration; its records and the previous renderer
are retained under `outputs/adaptive_gpis_mis_20261115`.

The [256x256 / 128-spp small-light result](../outputs/adaptive_gpis_mis_20261115/small_light_256_128spp/comparison.html)
uses 32 threads and seed 20261024. Geometry, material, GP, integrator and all
adaptive tolerances match the prior observed-normal small-light render.

| Estimator | Time | Mean luminance | Half-buffer noise RMS | Nonzero pixels |
| --- | ---: | ---: | ---: | ---: |
| Prior observed-normal reflection | 405.97 s | 0.018785 | 0.392432 | 140 |
| Conditional-gradient 1D MIS | 574.37 s | 0.016549 | 0.030449 | 27,993 |

The MIS run recorded 5,347,620 MIS events, 17,842,047 direction-PDF queries and
8,245,369 connection-gradient queries. It completed without internal errors,
NaN/Inf output or NEE sanity warnings; one invalid gradient was rejected.
There were 210 events without a continuous valid MIS gradient law, including
that invalid gradient. The measured noise RMS ratio is 12.89; this is a
finite-sample noise diagnostic, not a proven equal-quality speedup or bias test.

## Configuration and commands

The complete sphere and plane scenes have no external assets.

The procedural shaderball also has no external assets. It reuses the current
small-light experiment's geometry, camera and light, with a Lambert micro-material
for low-spp NEE checks. The initial observed-normal mirror implementation lacked
1D MIS and produced a black 16x16, 2-spp pilot. The optional estimator above now
supports the actual mirror small-light fixture. The Lambert fixture remains a
separate smoke test and should not be compared directly against the mirror report.

For the **original full Renewal+ shaderball scene**, use
[`shaderball-original.json`](../data/example-scenes/adaptive-gpis/shaderball-original.json).
It is adapted from the hash-verified `ell0p2_sigma0p02/renewal_3d_shadow_conditioned`
archive, with the original white floor, large quad emitter, environment light,
green mirror, camera and 20-bounce integrator. Its GP has ell=0.2 and sigma=0.02.
The initial adaptive minimum step is 0.01, matching the archived fixed step.
The archived second phase is unreferenced by the standard process with id=0;
only that unused phase and old sampler-specific settings are removed.

```powershell
python -B tools/compare_adaptive_shaderball_original.py --size 64 --spp 16 --threads 16
```

This runner verifies the archived scene hash and checks equality of geometry,
lights, material, GP parameters, camera transform and integrator. It rerenders
3D Renewal+ and adaptive GPIS with the same executable, seed, threads,
resolution and spp. Add `--modes renewal_3d adaptive refined` to also check
refined settings. Per-run `changes.json` lists every change
from the archived scene. The report includes half-buffer noise measurements,
linear PFM outputs, full counters and a link to the original archived image.
The original 3D Renewal+ baseline uses ordinary observed-normal mirror
reflection, so this comparison does not introduce the 1D MIS estimator.

```json
"type": "adaptive_function_space_gaussian_process",
"correlation_context": "renewal+",
"intersect_method": "gp_discrete",
"normal_method": "conditioned_gaussian",
"adaptive": {
  "probability_tolerance": 0.001,
  "minimum_step": 0.005,
  "root_tolerance": 0.00001,
  "rank_tolerance": 1e-10,
  "max_samples": 512,
  "max_steps": 100000,
  "enabled": true,
  "statistics": true
}
```

Distances are in world units, including for nonunit ray directions. Set
`enabled: false` for uniform stepping with the **same** sampler, history and
gradient law. This isolates the effect of adaptive step sizes. It is distinct
from `function_space_gaussian_process`, whose batch history and secant-gradient
approximations differ. `step_size_cov` in that older implementation remains
disabled; this new type uses its own explicit minimum step.

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_ADAPTIVE_GPIS_TESTS=ON
cmake --build build --config Release --target tungsten adaptive_gpis_test --parallel 1
ctest --test-dir build -C Release -R '^adaptive_gpis_' --output-on-failure
.\build\Release\tungsten.exe -t 2 -s 41 -d outputs/adaptive-sphere data/example-scenes/adaptive-gpis/sphere.json
```

The comparison runner uses only the Python standard library:

```powershell
python -B tools/compare_adaptive_gpis.py --scenes sphere plane shaderball --modes adaptive uniform legacy refined --size 16 --spp 2 --output outputs/adaptive-gpis-compare
```

It saves scene snapshots, binary hashes, logs, PNG/PFM images, a JSON report and
`comparison.html`. Existing results are reused only when their input identity
matches. Numerical-error log messages fail validation even when Tungsten exits
with code zero. `refined` halves minimum step and root tolerance and tightens
the rank tolerance by 100. Low-spp RGB RMSE is a diagnostic, not a bias estimate
or an equal-time reproduction metric.

Tests compare value/derivative covariances to the existing GP interface,
conditional moments to dense LDLT, sampled moments to the target covariance,
and first-hit CDFs to independent dense eigensampling. They also check close
queries, explicit rank-limit failures, probability-bound coefficients,
nonunit rays, JSON round trips, inward births, transparent continuation and
shadow/reset isolation.

## Pure-double results (2026-11-14)

[Current original-shaderball comparison](../outputs/adaptive_gpis_double_20261114/experiment/comparison.html):
64x64 pixels, 16 spp, seed 20261024, 16 threads. The scene JSON is identical to
the previous multiprecision adaptive run. Geometry, lights, material, camera,
GP parameters, minimum step, root tolerance and rank tolerance were unchanged.

| Method | Render time | Mean luminance |
| --- | ---: | ---: |
| Current sparse-convolution 3D Renewal+ | 18.50 s | 0.521878 |
| Pure-double adaptive | 12.50 s | 0.521575 |
| Pure-double adaptive, refined tolerances | 15.95 s | 0.521376 |
| Archived multiprecision adaptive | 541.94 s | 0.521546 |

Both double adaptive variants completed with zero invalid gradients and no
logged internal numerical errors. All four CTest cases passed, including
near-birth analytic moments, joint covariance tests exercising the double
rank-revealing path, first-hit distributions and medium/branch integration.
The original executable and DLLs were archived under
`outputs/adaptive_gpis_double_20261114/previous_renderer`, with matching hashes.

The speed comparison includes the revised double rank treatment, not merely a
precision typedef substitution. Low-spp agreement of image means is not proof
of unbiasedness or equal quality; numerical rank truncation remains an explicit
approximation and differs from exact arithmetic and the paper's strictly
incremental Cholesky path.

## Archived multiprecision results (2026-11-14)

The results below predate the pure-double change. Their saved inputs, binary
hashes and images are retained for comparison; their timings do not describe
the current implementation.

The Release renderer and test executable built successfully. All four
`adaptive_gpis_*` CTest cases passed. The independent first-hit comparison used
2,500 trials: the adaptive sampler averaged 25.59 queries, versus 161 points
in the dense reference; all five tested CDF values met the 0.045 tolerance.
This is a statistical check, not proof of unbiasedness.

- [Sphere, plane and shaderball smoke comparisons](../outputs/adaptive_gpis_20261114/verified/comparison.html)
  use 16x16 pixels, 2 spp and seed 41. Sphere and plane include adaptive,
  uniform-online, legacy batch-32 and refined settings; shaderball includes
  adaptive and refined settings.
- [Shaderball preview](../outputs/adaptive_gpis_20261114/shaderball_preview/comparison.html)
  uses 32x32 pixels and 4 spp with adaptive stepping and the Lambert fixture.

For the sphere smoke run, adaptive used 17.87 queries/ray and 5.42 seconds;
uniform-online used 292.68 queries/ray and 169.45 seconds. Legacy batch-32
took 0.74 seconds, so this initial reference does **not** establish a speedup
over the existing renderer. The methods consume different random sequences
and have different numerical/history approximations. The reports retain the
exact binary hash, input scenes, counters and logs.

`invalid_births` includes sampled exterior origins already inside solid, which
are intentionally opaque; it is not exclusively a numerical-error counter.
This occurs in the finite half-space plane fixture. Inspect it separately
from `invalid_gradients` and internal-error messages.

### Original full shaderball follow-up

[The matched original-scene report](../outputs/adaptive_gpis_original_shaderball_20261114/experiment/comparison.html)
uses 64x64 pixels, 16 spp, seed 20261024 and 16 threads. Both methods use the
same current executable and the archived full scene (floor, quad emitter,
environment, mirror, ell=0.2, sigma=0.02 and 20 path bounces).
The archived scene hash and unchanged physical-scene parameters were verified.

The original sparse-convolution 3D Renewal+ took 18.20 seconds. Adaptive GPIS
took 541.94 seconds, about 29.8 times as long in this run. It recorded 66,632
GP hits, zero invalid hit gradients, and 145,778 high-precision flights out of
202,591 flights. There were no logged internal numerical errors. This result
confirms the full-scene integration, while showing the cost of the initial
numerically conservative implementation; it does not establish equal quality
or a paper speedup. The baseline's 25 zero-length path termination notices
are retained in the report alongside the images and half-buffer statistics.

The earlier [32x32 / 4 spp pilot](../outputs/adaptive_gpis_original_shaderball_20261114/pilot/comparison.html)
also uses this full original scene. The portable `shaderball-original.json`
matches the checked 64x64 / 16 spp adaptive configuration exactly.
