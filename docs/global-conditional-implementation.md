# SDF-only global_conditional

The scene type `global_conditional` uses latest-observation conditional
transport, renewing the observation at every GP collision and geometry
endpoint, including ordinary and transparent surfaces. Numerical tracking
subdivisions do not renew it. Geometry endpoint renewal uses macrofacet's
positive-endpoint sampler; this changes the earlier implementation that kept
the old observation through ordinary geometry. It is neither a GP conditioned
on every previous path observation nor one conditioned on survival along the
complete path.

Surface births with a finite nonpositive forward conditional slope are discarded
by the renderer. `sampleDistance` returns false, preserving radiance already
accumulated on the path; `transmittance` returns zero for the corresponding light
connection. Neither case resamples or skips a ray interval. The kernel reports a
dedicated `NonPositiveSurfaceBirth` status so other numerical failures still
propagate. Render teardown logs separate path and shadow-connection counts.
This is an absorbing approximation for unsupported contacts (including sharp
creases), not an unbiased solution of sharp-edge scattering.

The input is a directly evaluated procedural/analytic SDF or an existing SDF
grid. Extinction, conditional covariance and tracking bounds are evaluated
online. Supported means are homogeneous, linear (including its clamp),
spherical, procedural `func: "knob"`, and linearly interpolated VDB SDFs. Covariance must be a
stationary squared exponential; its full positive definite world-space metric is
obtained from the covariance derivatives. Procedural means without certified
SDF queries and nonstationary kernels are rejected. Materials reuse
classic_global's tinted mirror and conductor Fresnel implementations.

The shaderball preset is
`gpis-scenes/shaderball/shaderball-stationary-27-global-conditional.json`.
It retains the classic preset's knob transform, sigma/length scale (both 0.02),
green mirror, lights, camera, 512x512 resolution and 256 spp. Only the medium
type and image output names change. `SdfFunctions` evaluates the original knob
CSG formula in double precision, supplies analytic gradients and propagates
local Lipschitz/curvature bounds through its operations. At a hard CSG branch,
the birth tracker shrinks its first interval until a smooth branch can be
certified; it does not omit the remaining flight or bake a field. This is the
same shape as the legacy float formula. Both global modes now use this value
and analytic gradient through `ProceduralMean`; neither uses a separate knob
formula. Existing classic images may change slightly because its gradient
previously used a fixed 0.001 finite difference.

## Baking the knob mean

`bake_sdf_mean` samples the same `ProceduralMean::evaluateJet` used by the
renderer into a VDB FloatGrid named `sdf`. It includes the configured world
transform, scale, offset and minimum clamp. It does not reinitialize distances
or truncate the field to a narrow band. Thus it preserves the sampled GP mean,
including smoothed CSG, up to float storage and trilinear interpolation error.

Build and run from the repository root (OpenVDB is required):

```powershell
cmake --build build --config Release --target bake_sdf_mean --parallel 8
.\build\Release\bake_sdf_mean.exe gpis-scenes/shaderball/shaderball-stationary-27-global-conditional.json gpis-scenes/shaderball/knob-mean-0.01.vdb 0.01
```

The optional last argument selects the medium name, default `gp`. The tool
derives a world-space bounding box from all cube boundaries containing that
medium and adds three voxels of padding on each side. This keeps the entire
medium strictly inside `VdbGrid`'s clamped interpolation domain. Non-cube
boundaries and means without SDF queries are rejected. Existing output files
are not overwritten; use a new filename to bake another resolution.

The generated `knob-mean-0.01.vdb.json` contains the mean configuration, grid
dimensions, domain, automatic gradient bound and sampled interpolation errors.
Validation reloads the saved file through `VdbGrid` and `GpisSdfGrid`, checks
domain coverage and voxel-node values, and compares 100,000 random world-space
queries with the source mean. Error statistics are measurements, not certified
bounds on the approximation to the procedural field.

The ready-to-render scene is
`gpis-scenes/shaderball/shaderball-stationary-27-global-conditional-baked.json`.
Its tabulated mean references `knob-mean-0.01.vdb`, uses linear interpolation,
`normalize_size: false`, `request_sdf: false`, and no emission grid. The world
transform is already baked: do not apply the knob's `[0, 0.8, 0]` translation
again. Omit `sdf_lipschitz` and `gradient_bound` so they are computed from the
grid; the original procedural preset's explicit bounds need not enclose the
grid's conservative estimate.

The 2026-10-20 bake covers approximately `[-1.5, 1.5] x [-0.499, 2.501] x
[-1.5, 1.5]`, with 0.01 voxel spacing and 307 cubed samples including padding.
The VDB is 81.72 MiB; the renderer's dense float copy is 110.38 MiB, in addition
to the loaded VDB and renderer memory. The measured value RMSE is 0.000145 over
the whole domain and 0.000336 for the 6,788 samples with `abs(mean) <= 0.06`.
The maximum measured absolute error is 0.00445; reloaded voxel-node error is
below 7.15e-8. Smaller voxel spacing reduces approximation error at greater
memory cost and with more cell crossings.

Both the baked and procedural presets completed 64x64, 4 spp previews with
8 threads. One sequential comparison measured 19.79 s for the baked grid and
4.26 s for the procedural mean (render time, excluding loading). At that
measurement the grid tracker rebuilt interval queries at voxel boundaries;
cheaper point queries alone did not imply faster rendering.
The detailed bake and render reports are under `build/knob-mean-bake/`.

Grid interval queries now reuse a per-flight traversal cursor and the current
cell's cubic ray polynomial for values, derivatives, curvature and midpoints.
Midpoint rounding errors enter the conditional certificate; cell-face endpoint
checks preserve ordinary point-query semantics. Adaptive tracking keeps the parent
bound on deferred right halves and tightens them only when reached, avoiding
queries past a collision. These changes do not alter the tracking settings or
renew the birth observation at numerical subdivisions.

## Experimental two-constant tracking

Set `"tracking_majorant": "two_constants"` on the `global_conditional` medium
to compute one birth majorant and one tail majorant at the beginning of a
free flight. The default is `"adaptive"`. Both collision and shadow tracking
use the selected mode, with one exponential clock carried across the switch.
The two-constant sampling loop does not refine or query interval bounds, and
the medium does not build macrocell bounds. The ordinary step, tightening and
macrocell controls have no effect in this mode.
Candidate points still evaluate the exact conditional hazard using the grid's
ray cursor. There is no automatic fallback or event-count cutoff in the renderer.

This mode currently requires a clamped VDB SDF or a constant mean. Unbounded
analytic/procedural means do not expose finite global magnitude certificates
and are rejected during scene preparation. VDB initialization caches
`S0 = max |(node + offset) * scale|` over the interpolation domain and a
world-space gradient bound `S1` from maximum absolute grid-edge differences,
the absolute world-to-index matrix, and mean scale. Clamping cannot increase
either bound. Numerical guards are included; `S1` is not assumed to be one.

For each normalized ray, write `a = w^T P w`,
`deltaF = birthValue - mean(birthPosition)` and
`deltaG = w dot (birthGradient - gradientMean(birthPosition))`. The code uses

```text
B0 = S0 + abs(deltaF) + abs(deltaG) / sqrt(a * e)
B1 = S1 + abs(deltaG) + sqrt(a / e) * abs(deltaF)
s_h = sigma * sqrt(1 - (1 + a*h*h) * exp(-a*h*h))
M_tail = (sqrt(2/pi) + B0/s_h)
       * (sqrt(a/(2*pi)) + (B1 + 2*B0/h)/s_h)
```

By default, `h` and `M_birth` come from the existing certified birth envelope, limited by
the first smooth cell and positive-side separation. The SE variance uses its
stable small-distance series, not a subtraction of nearly equal numbers.
Only `M_birth` is used before `h`, and only `M_tail` afterwards. This is the
constant-majorant null-scattering construction described in
[PBRT's transmittance section](https://pbr-book.org/4ed/Volume_Scattering/Transmittance),
applied separately to the two parts of the conditioned flight.

Measured on the baked knob, sigma = ell = 0.02, dx = 0.01, Release CPU,
32x32, 1 spp, 8 threads, seed 12345 (2026-10-21):

| Mode | Render time | Entire process wall time |
| --- | --- | --- |
| Adaptive, 3 runs after warmup | 0.22 / 0.23 / 0.22 s | median 1.774 s |
| Two constants | no completed frame | stopped at the 45 s timeout |

The two modes used the same executable, asset, camera, lights, path settings
and sample count. Timeout is not a completed-render timing. The missing input
`knob-mean-0.01.vdb` was rebuilt with `bake_sdf_mean` before both runs; it has
307 cubed nodes. Its renderer bounds are `S0 = 1.7423921823502084` and
`S1 = 1.7320560383183847`.

A separate fixed-flight probe explains the result:

| Flight | h | M_tail | Adaptive delta candidates | Two-constant delta candidates |
| --- | --- | --- | --- | --- |
| Front entry, length 3 | 0.00999999987 | 5.90673e7 | 3, hit at 0.946497 | 250,000, only reached 0.0142325 |
| Reflected collision, length 0.5 | 0.00179358 | 2.85140e11 | 3, hit at 0.389481 | 250,000, only reached 0.00179446 |

The probe stops at 500,000 random draws and reports partial work explicitly;
it does not treat these interrupted flights as completed samples. Ratio
tracking also hit this diagnostic limit, with 500,000 candidates in each
two-constant flight. These are controlled flight examples, not aggregate
render counters. For small h and nonzero B0, `M_tail` grows approximately as
`4 * B0^2 / (sigma^2 * a^2 * h^5)`. The saved interval queries are outweighed by
null events; keep adaptive tracking for this knob scene.

Reproduce with:

```powershell
python -B tools/benchmark_conditional_majorants.py gpis-scenes/shaderball/shaderball-stationary-27-global-conditional-baked.json --output build/two-constants-new-run --size 32 --spp 1 --threads 8 --timeout 45
.\build\Release\global_conditional_test.exe tracking_probe gpis-scenes/shaderball/shaderball-stationary-27-global-conditional-baked.json
```

Evidence is in `build/two-constants-review/results.json` and `final/`: the benchmark records scene and
binary hashes, logs and unfinished runs. Regression coverage includes global
grid bounds with negative scale and nontrivial transforms, anisotropic hazard
coverage, small-distance variance, zero/positive births, resumed flights,
delta/ratio survival against numerical quadrature, JSON mode selection,
medium/shadow routing and absence of tail interval queries. All 14 conditional
CTest checks passed, including NEE/MIS, wavefront and furnace rendering, as did
both classic_global checks (16 checks total for the initial implementation).

### Extending the constant prefix

The medium now accepts an absolute world-space switch distance:

```json
"tracking_majorant": "two_constants",
"tracking_constant_birth_length": 3.0
```

The default length is zero, which retains the original analytic birth split.
A positive value selects `H = min(requested length, total flight age at exit)`;
the distance is measured from the observation's birth position. The setting
does not change adaptive tracking. If the prefix covers the entire segment,
sampling uses one constant and there is no tail.

Increasing H does **not** extrapolate the small Taylor birth envelope across
grid cells. `twoSegmentMajorants(H)` first certifies its original analytic
birth interval `(0,h0]`, then covers `[h0,H]` with ordinary certified interval
majorants. At initialization their maximum becomes the first constant:

```text
M_prefix(H) = max(M_birth(h0), maximum of all interval bounds covering [h0,H])
M_tail(H)   = the global tail formula above, evaluated at H
```

The interval list is discarded after constructing the prefix bound; tracking
still switches at most once. No new field observation is sampled at H. This
adds upfront interval queries, including regions that an early collision may
never reach. Increasing H lowers the tail bound and shortens its support,
while the larger prefix can require a larger constant and more initialization
work. `H*M_prefix(H) + (L-H)*M_tail(H)` is the proposal optical depth of a full
flight of length L; early termination can reduce the actual candidate count.

On the same 0.5-long reflected-flight diagnostic, extending the switch from
0.00179358 to 0.1 lowers `M_tail` from 2.8514e11 to 1.8987e5. Extending it to
0.5 removes the tail and gives `M_prefix = 119.18`: delta tracking completes
with 41 candidates, versus reaching the 250,000-candidate diagnostic limit
with the original split. Adaptive tracking completes with 3 candidates. This
is one controlled flight, not an aggregate render counter.

Final-build knob renders (32x32, 1 spp, 8 threads, seed 12345, same VDB):

| Mode / requested H | Render seconds, three runs | Median process wall seconds |
| --- | --- | --- |
| Adaptive | 0.23 / 0.24 / 0.25 | 1.882 |
| Two constants, H = 3 | 1.67 / 1.73 / 1.71 | 3.122 |
| Two constants, H = 6 | 2.49 / 2.66 / 2.76 | 4.188 |

Exploratory H = 0.1, 0.5 and 1 runs each timed out after 20 seconds of process
wall time without a completed image. H = 3 makes this experiment usable, but
the current adaptive tracker remains faster. Increasing H further to 6 was
slower in this test; H is not a monotonic performance control. Completed images
are finite and nonnegative; the low-spp renders are not a convergence test.

Run the same comparison with `--birth-lengths 3,6` added to the benchmark
command, or use `global_conditional_test tracking_split_probe <scene.json>`
for the split sweep. Results and hashes are under
`build/two-constants-split-review/{explore,final}/report.json`. Regression
coverage additionally checks extended prefixes across grid cells and clamps,
empty and resumed flights, full-prefix endpoint rounding, full-domain
validation, medium/shadow routing, and no interval queries during sampling.

## Shared procedural SDF interface

`math/SdfQueries.hpp` defines optional geometry queries without depending on a
medium or GP transport model:

- `evaluateJet(p)`: value and unnormalized analytic gradient.
- `valueDifference(p, d)`: stable `f(p+d)-f(p)` for short conditional flights.
- `bounds(minimum, maximum)`: certified value and gradient-norm enclosures.
- `rayBounds(origin, direction, begin, end)`: certified values and first/second
  directional derivatives over a closed interval. Directions need not be unit
  length; the derivatives are with respect to the supplied ray parameter.

`ProceduralSdf` delegates these queries to `SdfFunctions`. Its knob, knob-inner
and knob-outer scalar geometry share the single templated CSG definition in
`SdfFunctionsQueries.cpp`. Differential capabilities currently cover `knob`.
`ProceduralMean` converts world positions and displacements to local coordinates,
transforms gradients by the transpose inverse matrix, and applies the scene's
scale, offset and minimum clamp to values, derivatives and bounds. Negative
scale, zero scale, rotation and nonuniform spatial scale are supported.

`MeanFunction::sdfQueries()` and `ProceduralScalar::sdfQueries()` default to
`nullptr`. `conditional/gpss/SdfMeanField` adapts any mean exposing this capability
and retains shared ownership of it. The conditional medium does not inspect a
knob enum or implement geometry. `classic_global` uses the existing
`MeanFunction` value/gradient API, which calls the same query implementation.

To add another procedural SDF, implement the four queries in its geometry
provider and advertise the capability. Value-only or finite-difference-only
providers keep their existing rendering path; the conditional adapter rejects
them explicitly. At a CSG or clamp seam, an infinite second-derivative bound
honestly indicates that no smooth interval has yet been certified. The tracker
refines that interval instead of assuming a finite bound. The scene's existing
`sdf_lipschitz` and `gradient_bound` settings remain required by the base medium.

## Stage 1: straight conditional flights

The port in `src/core/math/conditional` retains macrofacet's stable SE series,
birth envelope, truncated exterior-value sampling and flux-weighted *raw*
gradient sampling. At a true medium entry, draw `F0 > 0` and the independent
prior gradient. At a GP collision, draw the full gradient and replace the birth
observation with `(position, 0, gradient)`. Only reflection normalizes it.

Let the latest observation be `(x0, f0, g0)`, `P` the SE precision metric,
`d = x-x0`, `u = dᵀPd`, `r = exp(-u/2)`,
`δf = f0-m(x0)`, and `δg = g0-∇m(x0)`.

```
μF = m(x) + r (δf + d·δg)
μG = ∇m(x) + r δg - r P d (δf + d·δg)
vF = σ² [1 - exp(-u)(1+u)]
cov(G,F | H) = σ² exp(-u) u P d
```

Condition these moments again on `F(x)=0`. The hazard along travel direction
`w` is `phi(μF/sF)/(sF Phi(μF/sF)) E[(-w·G)+ | F=0,H]`.
The small-distance implementation avoids subtracting nearly equal covariance
matrices. It does not skip a fixed epsilon-sized portion of the medium.

### Normalized interval bounds and tighter birth envelopes

Ordinary intervals now use both the original moment-form bound and a
standardized bound, taking their minimum. With `s=sqrt(vF)`,
`z=μF/s`, `z'=μG|F=0·w/s` and `κ=sqrt(vG|F=0 along w)/s`, the same hazard is

```
λ = phi(z)/Phi(z) [κ phi(z'/κ) - z' Phi(-z'/κ)].
```

The inverse Mills ratio decreases with `z`; the bracket decreases with `z'`
and increases with `κ`. Both Taylor mean bounds and SDF interval mean bounds
feed this common reducer. Dividing a positive mean lower bound uses the
largest `s`, while dividing a negative one uses the smallest `s`.
Pointwise hazard evaluation and the gradient/endpoint samplers are unchanged.

For `a=wᵀPw`, `q=a*t²`, define `V=1-(1+q)exp(-q)` and
`N=1-(2+q²)exp(-q)+exp(-2q)`. The helper
`ConditionalMajorantBounds.h` encloses `N/V`, `N/V²` and `V/q²`.
At small `q` it evaluates positive series for
`E2=(exp(q)-1-q)/q²` and
`E4=(exp(q)+exp(-q)-2-q²)/q⁴`, then uses

```
V/q² = exp(-q) E2
N/V  = q² E4/E2
N/V² = exp(q) E4/E2².
```

This cancels the common vanishing powers before computing `κ²=a*N/V²`.
The positive-series tails are bounded by the next omitted term and a geometric
series, with an allowance for floating-point arithmetic. Interval arithmetic
rounds outward using `nextafter`; exponentials additionally receive a libm
roundoff allowance. The helpers are checked against independent
100-decimal-digit fixtures. They use double precision, not an arbitrary
precision or formally verified transcendental library.

Both `N/V` and `N/V²` increase with `q`, so their right endpoints bound the
conditional derivative variance and `κ²`. This is an analytic monotonicity
result, not an assumption based on sampled values. For `p=1,2`, the sign of
`(N/V^p)'` is the sign of `N'V-pNV'`. Multiplying that numerator by `exp(3q)`
gives power series with coefficients (in units of `q^n/n!`)

```
p=1: 2^(n-2)(n²-7n+8) + n²+n-4,              n>=5
p=2: (n-1)[2^(n-2)(n-8) + n²-n+4],           n>=6.
```

All earlier coefficients vanish and all displayed coefficients are positive.
The global bounds `N/V<=1` and `N/V²<=1` also apply; far-tail evaluations use
these directly. `V/q² = integral_0^1 x exp(-q*x) dx` decreases from `1/2`.

The birth interval still certifies positive separation and a smooth mean
branch. Its envelope combines `γ μF-μF'` before bounding, with `γ=s'/s<=2/t`.
For `g=w·g0`, `δf=f0-m(x0)`, `δg=w·(g0-∇m(x0))` and the mean curvature
bound `H`, the common flux coefficient becomes

```
K = max(g,0) + (2H + 2a|δf| + 2.5 a h |δg|) h
    + sqrt(σ² a N(a h²)/V(a h²)) / sqrt(2π).
```

The variance bounds for `s(t)²/t⁴` are now
`σ² a² V(a h²)/(a h²)²` and `σ² a²/2`.
Zero-valued births retain the `t^-2 exp(-A/t²)` envelope; positive geometry
births retain `t^-3 exp(-A/t⁴)`. Their analytic maximizer and the existing
smooth-cell refinement remain in use.

Away from birth, the SE correction and its derivative are enclosed as whole
exponential-polynomial products. The derivative sign is a polynomial of
degree at most three. Interval sign exclusion, interval Newton contraction
and subdivision enclose possible critical points, including repeated roots,
endpoint roots and degenerate coefficients. The extrema enclosure includes
every partition endpoint. Newton only contracts a cell when the polynomial
derivative's interval excludes zero, so the contraction retains all its roots.
Unresolved cells retain their full interval range if the refinement work
limit is reached; no approximate root is silently dropped. The result is
intersected with the existing separate-factor enclosure.

## Stage 2: geometry and persistent state

`ConditionalSegment` covers a forward portion of a straight flight from its
latest observation. It uses the birth envelope, birth-adjacent Taylor bounds
and straight-ray SDF interval bounds. Expensive valid bounds are subdivided;
neither the rate nor the event count is capped. There is no off-axis evaluator
or majorant. Changing the line without first updating the observation is an
explicit input error.

Each `track()` call owns a `MajorantOpticalDepth` clock. It samples
`tau = -log(U)` on first reaching a positive majorant. A candidate-free interval
of length `L` and rate `M` consumes `M*L` from that threshold; its residual
continues across step boundaries, grid faces and deferred adaptive subdivisions.
When `tau < M*L`, the next candidate is `tau/M` from the interval's beginning.
After handling that candidate, another threshold is drawn only if tracking
continues. This applies to both collision and ratio tracking.

Zero-rate intervals consume no threshold and draw no random numbers. A
threshold exactly exhausted at a boundary belongs to the next positive-rate
interval; a candidate that rounds onto the right face follows the same rule.
The clock is local to one `track()` call, so a new flight starts a fresh clock.
This samples the same piecewise-majorant Poisson process while avoiding a
random draw and logarithm for every empty interval. Random-number consumption,
and consequently individual rendered pixels, can differ from earlier builds.

The medium exposes seven tracking settings directly in its JSON object:

```json
"tracking_step": 0.0,
"tracking_step_scale": 0.5,
"tracking_age_scale": 0.25,
"tracking_max_expected_events": 64.0,
"tracking_tighten_expected_events": 4.0,
"tracking_macrocell_size": 0,
"tracking_macrocell_max_expected_events": 1.0
```

These are the defaults; individual scene presets may override them.
`tracking_step > 0` sets the ordinary interval length cap in world units.
At `tracking_step == 0`, the cap is
`tracking_step_scale / sqrt(w^T P w)`, using the normalized travel direction.
An explicit positive step overrides the scale setting. For the isotropic
shaderball with length scale `0.02`, scale values `0.5`, `1` and `2` give
ordinary caps `0.01`, `0.02` and `0.04`; alternatively set `tracking_step`
directly to one of these world distances.

Outside the separately certified birth interval, the proposed length is
`min(cap, tracking_age_scale * distance_from_birth)`. Geometry endpoints and
SDF interpolation knots can shorten it. If `majorant * interval_length`
exceeds `tracking_max_expected_events`, tracking bisects and recomputes the
bound; the same threshold triggers the tighter local bound near birth.
This is an expected-candidate-count threshold, not a cap on sampled events.
Lowering it trades more bound evaluations for potentially fewer null events.
Changing these settings does not change the birth-envelope certificate,
the conditioned moments, or observation renewal. Collision sampling and
shadow transmittance share the settings.

Tabulated means optionally build a coarse grid with
`tracking_macrocell_size` fine interpolation cells per axis. Zero disables it;
supported widths are powers of two from 2 through 64. This acceleration
precomputes extrema of original SDF nodes and index-space edge differences,
including a one-cell halo for rounded boundary queries. It does not resample
the mean field. The block certificates are immutable and shared by all rays.

After the smooth birth region, tracking first clips a proposed interval to a
coarse-block or clamp boundary. It projects the precomputed gradient ranges
onto the ray, applies scale and clamp masks, and combines the resulting mean
certificate with the current birth observation using the cheap conditional
bound. A coarse interval is accepted only if its expected candidate count is
at most both `tracking_macrocell_max_expected_events` and
`tracking_max_expected_events`. A loose block descends to the existing fine
cell traversal; its valid coarse bound remains available as a parent bound.
That rejected window is not retried for each fine cell. The carried optical
depth is preserved across both levels.

Birth envelopes continue to use fine-cell curvature. Coarse certificates have
infinite curvature and are never used to claim smoothness across grid faces.
Candidate and collision queries still evaluate the original trilinear field.
Providers without coarse-grid support retain the fine path. The ordinary
step/age caps still apply, so increasing the coarse width beyond the step cap
need not improve throughput. Profiling adds `grid_macro_queries`,
`macro_accepted` and `macro_rejected` counters.

The baked shaderball preset enables width 8 and a coarse expected-event
threshold of 4. With its 0.01 voxel spacing, the nominal block width is 0.08
world units. Set `tracking_macrocell_size` to 0 to restore fine-cell traversal;
the default for scenes without this setting remains 0.

Ordinary intervals first build a cheap conservative majorant: bound the SE
exponential and its linear/quadratic correction separately, multiply their
outward-rounded intervals, and combine with the SDF ray certificate. A quadratic
uses endpoints and an enclosed vertex; coefficient uncertainty spanning zero
falls back to interval Horner evaluation. Midpoint/derivative tightening and
the conditional variance bounds are retained. This stage performs no
exponential-polynomial extremum search.

Let `M = min(cheap_majorant, inherited_parent_majorant)`. Only if
`M * interval_length > tracking_tighten_expected_events` does the ordinary
bound run the two exponential-polynomial extremum searches. The fine stage
reuses the SDF certificate, midpoint and conditional moment bounds, then
intersects the cheap and fine enclosures. An overflowing effective rate also
triggers tightening; failure to obtain a finite final rate remains an error.
The separate `tracking_max_expected_events` test then decides whether to
subdivide. Deferred right siblings keep the parent certificate until visited.
The birth envelope and birth-adjacent Taylor certificates are unchanged.

`tracking_tighten_expected_events: 0` always runs the fine ordinary bound;
larger values favor cheaper bounds and can increase null collisions. The
default `4` is a tuning starting point, not a guarantee of optimal speed.
Small positive candidate counts still perform normal delta/ratio tracking:
the screening threshold never declares an interval transparent. The hazard
and target distribution are unchanged, but random streams and image noise can
change. Profiling counters `cheap_bound_accepted`, `tightened_intervals` and
`cheap_bound_fallbacks` expose the choices. The profiling script accepts
`--tighten-expected-events` for comparisons.

The absolute step and tightening threshold must be finite and nonnegative;
the other three settings must be finite and strictly positive. Invalid values
fail at scene loading. Omitting the settings selects the defaults above,
including cheap-first screening. The inherited `tracking_step`, formerly
unused by the conditional medium, now takes effect.

For a tabulated SDF, each `ConditionalFlightKernel` owns a `MeanRayCursor`.
The grid and its voxel data remain immutable. The cursor caches the transformed
ray and the previous/next crossing on each axis, including clamp boundaries.
It advances on a cell crossing and can seek backwards when interval refinement
or a diagnostic query revisits an earlier distance.

Entering a cell loads its eight nodes once and constructs a cubic mean and
three quadratic world-gradient components along the ray. The parameter is
scaled by the largest active index-space ray component, keeping its magnitude
small within the cell. Candidate hazards use the scalar cubic and its
derivative; collision and endpoint gradients use the quadratic components.
Birth differences use a factored polynomial difference, with an error bound
proportional to displacement rather than subtraction of two nearly equal
values. The ordinary midpoint certificate includes that error.

Interval queries reuse the same coefficients. Quadratic Bernstein derivative bounds
certify monotonic intervals; otherwise cubic Bernstein control values enclose
the value range. Coefficient construction, coordinate transforms and query
arithmetic receive conservative error allowances. The coefficient errors and
absolute Horner norms bound the error in the scalar control-value calculations;
only the parameter endpoints and final limits need outward rounding.
Multi-cell intervals retain
an infinite Taylor-curvature bound. Queries near a rounded cell face or clamp
boundary preserve the original point-value/gradient convention; two endpoint
jets are retained for reuse. Adjacent-double intervals with no representable
interior, and transforms with excessive coordinate uncertainty, use the
original grid interval query.

This cache is automatic for `global_conditional` tabulated means and needs no
scene option. Providers without a cursor retain their existing queries.
Subdivision and null events preserve the cursor; a new conditional flight
creates its own state. Polynomial evaluation changes arithmetic and the
tighter bounds can change random streams, so cross-version pixel equality is
not required. Tests compare values, gradients, certified ranges and transport
survival against the uncached interpolant. Profiling counters
`grid_cell_builds`, `grid_point_fallbacks` and `grid_bound_fallbacks` expose
cache construction and numerical fallbacks.

When `sampleDistance` reaches a geometry endpoint, it samples a complete new
observation before surface direct lighting and BSDF continuation branch:

```
f ~ N(μF, vF) conditioned on f > 0
g | f ~ N(μG + cov(G,F)/vF (f-μF),
          cov(G,G) - cov(G,F) cov(F,G)/vF)
new birth = (endpoint, f, g)
```

All moments above are conditioned on the previous birth observation. The
ordinary surface's normal is not used as the GP gradient, and its field value
is not set to zero. `ConditionalFlightKernel::samplePositiveEndpoint` uses
stable SE formulas for this joint sample. Shadow `transmittance` applies the
same renewal at its endpoint when its weight is nonzero. Both blocking and
resumable tracing call these medium methods.

Subsequent directions start exactly at the new observation, so any outgoing
direction is on a straight conditional flight. Reversing after an ordinary
surface no longer produces a deterministic collision at an old GP zero:
the old observation has been replaced. A zero-length flight at the birth
preserves its observation. Inserting another transparent geometry surface
adds a renewal event; subdividing an internal tracking interval does not.

`ConditionalPathContext` is immutable and shared between branches. It stores the
owner medium, birth observation, and a double precision position corresponding
to the renderer's float ray origin. Geometry renews the birth value and raw
gradient at that precise position. Surface light connections and continuation
share this observation until they branch. A true medium transition clears the
context.

The CPU blocking and resumable trace paths both pass branch-specific state.
Medium/phase records now own context pointers; BDPT's former union therefore
uses `std::variant` for correct lifetime management. This does not add support
for conditional BDPT: its state-free distance-PDF interface remains unsupported.

## Stage 3: conditional visibility and direct lighting

A connection direction fixes the reflection normal
`h = normalize(wo-wi)`, oriented so `wi·h < 0`. It does not fix the gradient
length. The light-sampled branch draws

```
p(r | h, collision, H) ∝ r³ N(r h; μG|F=0,H, ΣG|F=0,H),  r > 0.
```

The cubic power contains the spherical-coordinate Jacobian and crossing flux.
The sampled gradient `r h` becomes the shadow branch's new birth observation.
The phase-sampled branch carries its sampled raw gradient in the same way.
Their conditional radial density cancels in the MIS ratio, so the direction
PDFs can use Tungsten's existing power heuristic.

Shadow transmittance uses ratio tracking of the same conditional hazard.
Shadow state is copied before traversal; it cannot mutate the continuation.
Numerical failures propagate through the path tracer and task group instead of
being converted to black samples or invoking a successful completion callback.

## Lightweight SDF point queries

`SdfFunctions::evaluateJet()` uses `queryPoint()` and a first-order `PointJet`
to evaluate the existing `knobGeometry<T>()` definition. It propagates only
the value, analytic gradient and two structural flags (`constant`, `affine`).
It does not propagate numerical Lipschitz, curvature or directional interval
bounds, and avoids the affine Jacobian matrix product used by the full `Jet`.
Spatial/ray certificates still use the full `Jet`; stable small differences
still use `Difference`.

The flags preserve the original distinction between a locally constant
expression and a stationary point with zero gradient. Guarded CSG choices,
equal-value branches, smooth-union clamp decisions, single-component absolute
values and gradient accumulation order retain the full Jet's conventions.

The standalone `sdf_point_jet_test` compiles the private implementation without
linking `core`, allowing a direct comparison to the existing radius-zero full
Jet without adding a production reference-query API. It checks primitive
degeneracies, smooth-clamp endpoints and their neighbors, and 32,181 knob
points on axes, caps, a Cartesian grid and seeded random scales. Values and
gradient components match bit for bit, including signed zeros; both structural
flags also agree with the reference.
All 16 conditional/classic CTest checks passed after the change (52.03 s
total), including the new point-jet comparison, transformed SDF queries,
majorant coverage, conditional tracking and render/furnace checks.

Validation on 2026-10-20 used the default 0.01 step cap, 32x32, 4 spp, two
CPU threads and seed 12345. Both executables were built in Release with
diagnostics compiled out. Runs used both before/after orders, and all three
image comparisons were exact:

| Implementation | Render times | Median |
| --- | --- | ---: |
| Full Jet point queries | 42.13, 40.82 s | 41.475 s |
| PointJet point queries | 29.62, 26.80, 30.27 s | 29.620 s |

This preview's median render time decreased by 28.6% (1.40x throughput).
Absolute timing differed from the earlier profiling session; these paired
runs are the implementation comparison. The source scene remained unchanged.
Inputs, executable hashes, images, build logs and results are in
`build/point-jet-review/`, with combined timings in `summary.json` and the
two run reports in `render32/` and `render32-reverse/`.

## Reusing SDF interval queries

Ordinary `ConditionalFlightKernel::intervalMajorant()` requests ray bounds
without `beginDerivative`. `SdfFunctions` skips its beginning-point query;
`ProceduralMean` also skips the world-space beginning-point query. Callers
that need the derivative, including the birth certificate, retain the default
request. The procedural wrapper never computes the local derivative that it
would then overwrite with the world-space result.

`SdfRayBounds` optionally returns `midpointValue` and `midpointError`. The knob
provider reuses the value already produced by the full interval `Jet`, and
the conditional majorant uses it instead of making another `PointJet` query.
This reduces the usual procedural interval from one full Jet plus three
PointJets to one full Jet. The near-birth stable-difference branch and
candidate collision/endpoint queries are separate and remain unchanged.

The midpoint is constructed with `0.5*begin + 0.5*end`, matching the transport
calculation. `SdfRayQueryOptions` can carry that position through coordinate
transforms, so the local provider evaluates the same transformed midpoint
as the point API. Its bounding ball also includes the roundoff displacement
from the reconstructed local ray midpoint. A full finite-radius Jet can take
different CSG arithmetic shortcuts from a PointJet; the returned midpoint
error covers this evaluation roundoff and is propagated through mean scaling,
offset and minimum clamping before entering the transport enclosure.

The default analytic mean provider also exposes its already evaluated
midpoint. Providers without a reusable midpoint, including the trilinear
grid, leave it as NaN; transport keeps its point-query fallback. Grid birth
derivatives retain their forward-cell semantics.

Validation on 2026-10-20 passed all 16 conditional/classic CTest checks
(9.27 s with four test jobs). Added checks cover 80 finite-radius/seam/large
coordinate ray intervals, supplied midpoint displacement, transformed cached
values and clamps, optional beginning derivatives, the uncached-provider
fallback, and propagation of a deliberately perturbed midpoint's error.
A counting mean also verifies that ordinary majorants do not request a
beginning derivative or make an extra point query when a midpoint is supplied.

Paired Release renders used the shaderball snapshot with `tracking_step=0.08`,
medium `max_bounces=1024`, integrator `max_bounces=20`, 512x512, 1 spp,
31 CPU threads and seed 12345. The order was before/after/after/before:

| Implementation | Render times | Median |
| --- | --- | ---: |
| PointJet, before interval reuse | 41.74, 45.47 s | 43.605 s |
| Reused interval midpoint, optional beginning queries | 30.58, 30.61 s | 30.595 s |

This run reduced median render time by 29.8%. These are 1 spp measurements;
a full 16 spp render was not timed. Scene parameters were not edited.

Each executable's repeated image is identical, but old/new images are not
bit-identical. Bounds now include the cached value's rounding error and
preserve a canonical transformed midpoint, so candidate placement and random
number consumption can change. With the preset's continuous tile sampler,
that also changes the samples of later paths in a tile. The 512x512 image's
mean RGB intensity changed from 0.38872115 to 0.38849313.

A separate comparison used independent per-path sampling, 128x128, 4 spp,
eight threads and the same seed. RGB mean differences were
`9.6293e-6`, `5.1129e-4`, `7.9147e-5`, respectively 1.421, 1.406 and 1.426
paired-image standard errors. This check found no mean shift outside sampling
variation; it does not assert pixel identity. Inputs, executable hashes, PFM
images, build/test logs, timings and comparisons are retained in
`build/sdf-query-reuse-review/`, especially `benchmark-summary.json` and
`independent-summary.json`.

## Performance diagnostics

`TUNGSTEN_ENABLE_CONDITIONAL_PROFILE=ON` compiles optional transport diagnostics.
It is **OFF by default**, so ordinary builds eliminate the diagnostic scopes
and counters. In a diagnostic build, set
`TUNGSTEN_GPIS_CONDITIONAL_PROFILE=1` to print one JSON record at process exit.
The record separates path transport, shadow transport, phase operations and
light connections. Nested stages have inclusive and exclusive times; only
exclusive times can be summed without double counting. These are elapsed
times summed over workers, not frame wall time or sampled CPU stacks.

`tools/profile_global_conditional.py` benchmarks ordinary builds separately
from a diagnostic executable (`--profile-executable`). It retains the scene
snapshot, fixed seed, per-run scenes/logs/PFM images, process CPU time, stage
profiles, event counts and image parity results. Repetitions alternate the
order of step sizes. Profiled/unprofiled images must match exactly for each
configuration; images across different tracking settings are not expected to
match because random-number consumption changes.

The following measurements precede the PointJet optimization. Measured on
2026-10-20, Windows, Intel Core i9-13900HX, Release CPU, two
rendering threads, seed 12345. The current shaderball preset was reduced to
32x32, 4 spp; lights, NEE/MIS, bounce limit, covariance and SDF were preserved.
Only the ordinary step cap changed; age scale stayed 0.25 and expected-event
threshold stayed 64. The source scene was not edited.

| Ordinary step cap | Uninstrumented render time, median of two runs | Range | Processed intervals | Candidate events, path + shadow |
| --- | ---: | ---: | ---: | ---: |
| 0.01 (default) | 17.585 s | 17.50–17.67 s | 2,924,892 | 295,456 |
| 0.02 | 9.780 s | 9.76–9.80 s | 1,555,472 | 305,955 |
| 0.04 | 5.880 s | 5.86–5.90 s | 897,443 | 360,393 |

A separate 64x64, 4 spp confirmation with the same two threads took
67.15 s at the default step and 24.82 s at step 0.04 (one run each, about
2.71x throughput). Its scenes/logs/results are under
`build/conditional-profile-review/confirm64/`. These are preview measurements,
not extrapolations to the full 512x512 preset.

At the default step, ordinary `intervalMajorant` calls, including their
children, account for about **95.5%** of instrumented conditional work.
The non-overlapping breakdown is:

| Stage | Exclusive worker time share |
| --- | ---: |
| SDF point value/gradient queries | 54.63% |
| SDF ray bounds, excluding nested point queries | 18.09% |
| Exponential-polynomial range bounds | 15.29% |
| Moment/hazard majorant reduction, excluding SE variance bounds | 5.97% |
| All other measured work | 6.02% |

There were 9,115,736 SDF point queries, 2,925,642 SDF ray-bound queries and
5,791,208 exponential-polynomial range calls. Of the processed intervals,
2,793,583 (95.51%) generated no candidate event. Only 48 tracking splits were
needed, and no polynomial range calculation reached its refinement/depth
limit. Birth/CSG smoothness refinement is a separate counter and is not part
of those 48 splits.

Path transport used 40.80% of measured worker time, shadow transport 59.08%,
and phase/connection operations together 0.13%. Path tracking generated
18,283 candidates: 5,449 accepted collisions and 12,834 rejections. Shadow
ratio tracking generated 277,173 candidates; these are weighted ratio events,
not accepted/rejected collisions. Pointwise hazard evaluation including its
SDF work took 2.14% of measured time. Thus the dominant cost in this scene is
constructing bounds for many mostly empty intervals.

The measurements identified these optimization candidates:

- Add a lightweight value/gradient SDF evaluator. At measurement time,
  `evaluateJet()` ran the full `Jet` propagation, including
  Lipschitz/curvature/interval bookkeeping, even at radius zero.
  The PointJet implementation above addresses this cost.
- Reuse SDF results within a bound. At measurement time,
  `SdfFunctions::rayBounds()` evaluated the begin gradient, then
  `ProceduralMean::rayBounds()` evaluated it again. The interval midpoint
  was also evaluated inside the SDF bound and again in
  `ConditionalFlightKernel::intervalMajorant()`. The interval-query reuse
  implementation above now addresses these repeated queries.
- Grow certified intervals in low-hazard regions. Step 0.04 reduced interval
  count by about 69% while increasing candidate count by about 22%, giving
  approximately 3x throughput in this preview. Every longer interval must
  retain its valid majorant; this does not authorize discarding small hazards.

Changing the 64-event split threshold is a lower priority in this measurement.
The previews do not establish an optimal step for every scene or equal
variance between tracking configurations.

Raw evidence is in `build/conditional-profile-review/steps32/report.json`.
The diagnostic runs took 17.54 / 9.94 / 6.03 s and produced exactly the same
pixels as their uninstrumented counterparts. The ordinary build also matched
the preserved pre-instrumentation executable exactly. An earlier run in
`default32/` had much slower timings for the same diagnostic executable;
the cause was not established, and those times are excluded from the table.
All 15 conditional/classic CTest checks passed after building the default
configuration (35.17 s total). Seven exact-image comparisons passed in the
32x32 benchmark, covering the previous executable, repetitions and profiles.
The renderer at `build/Release/tungsten.exe` has diagnostics compiled out.

To repeat the pre-PointJet comparison with the retained executables
(`tungsten-conditional-profile.exe` also predates PointJet):

```powershell
python tools/profile_global_conditional.py gpis-scenes/shaderball/shaderball-stationary-27-global-conditional.json --executable build/Release/tungsten-before-point-jet.exe --profile-executable build/Release/tungsten-conditional-profile.exe --output build/conditional-profile-repeat
```

## Validation

Configure with `TUNGSTEN_BUILD_GLOBAL_CONDITIONAL_TESTS=ON` and
`TUNGSTEN_BUILD_CLASSIC_GLOBAL_TESTS=ON`. Build `global_conditional_test`,
`classic_global_test`, `sdf_point_jet_test` and `tungsten` in Release.

The tests compare moments against an independent joint-Gaussian block
conditioner; compare tracking survival against numerical hazard integration;
check near-birth behavior, raw-gradient moments, positive endpoint joint moments,
changed directions after renewal, rejection of stale observations, numerical
split segments and branch isolation. Render examples live in
`data/example-scenes/global-conditional/`.

Validation after adding coarse-grid certificates (2026-10-20, Release, CPU):

- All 16 conditional/classic CTest checks passed in 14.41 s. New grid checks
  cover 158 intervals and 5,214 reference points, including negative scale,
  transformed rays, negative/shifted grid origins, clamps and rounded faces.
  Uncertain large transforms decline the coarse query and retain fine fallback.
- Routing fixtures verify that accepted coarse intervals skip fine bounds and
  fine DDA, while rejected windows are queried only once before descending.
  Independent hazard integration checks delta and ratio survival across
  multiple block widths and acceptance thresholds.
- At 32x32, 2 spp and eight threads, render times were 2.31 s for the retained
  baseline, 2.23 s with the new feature disabled, 0.67 s for width 4 / threshold
  1, 0.60 s for width 8 / threshold 1, and 0.56 s for width 8 / threshold 4.
  Disabled-mode pixels were identical to the retained baseline.
- A serial ABBA comparison at 64x64, 4 spp and eight threads used width 8 /
  threshold 4. Median render time decreased from 18.30 to 3.84 s (79.02%,
  about 4.77x). The new runs were 3.40 and 4.28 s. Median wall time, including
  loading and coarse precomputation, decreased from 19.94 to 5.83 s (70.74%).
  Median process CPU time decreased from 117.30 to 25.35 s. These measurements
  are for the small preview, not a measured 512x512 render.
- Repeated runs within each version produced identical pixels. Coarse bounds
  change the random stream, so cross-mode pixel identity is not expected;
  correctness is checked by conservative enclosures and survival statistics.
  The SDF nodes, trilinear interpolation, conditioned field and observation
  renewal are unchanged. The baked scene now uses the measured 8 / 4 setting.
- Logs, scene snapshots, retained baseline executable, task-only patch,
  binary hashes and render reports are under `build/conditional-macrogrid/`.

Validation after carrying exponential optical depth across intervals (2026-10-20,
Release, CPU):

- All 16 conditional/classic CTest checks passed in 15.99 s. Existing delta and
  ratio survival tests still match independently integrated hazards.
- Scripted random draws verify candidate positions against integrated-intensity
  inversion, changes in rate, zero-rate gaps, exact and rounded boundary events,
  and the same candidate stream under one versus 400 constant-rate intervals.
- Both real tracking modes traverse at least 400 candidate-free intervals with
  one exponential draw. A separate fixture forces adaptive subdivision and
  checks that deferred siblings retain the residual threshold.
- A serial ABBA comparison of the baked shaderball at 64x64, 4 spp and eight
  threads measured median render times of 18.625 s before and 18.155 s after
  (2.52% lower). The two new runs were 18.83 and 17.48 s, so the small timing
  difference is within the observed run variation. Repetitions within each
  version produced identical pixels; cross-version random streams differ.
- Build/test logs, the preserved baseline, task-only diff and benchmark evidence
  are under `build/conditional-optical-depth/`.

Validation after adding the ray cursor and cell polynomial cache (2026-10-20,
Release, CPU):

- All 16 conditional/classic CTest checks passed in 17.62 s. Grid checks compare
  63 intervals and 1,575 points with the uncached trilinear implementation,
  including transformed rays, negative directions, clamps and grid faces.
  Independent cubic fixtures verify derivatives, curvature, stable birth
  differences and single-cell reuse; transport checks compare delta and ratio
  survival with an independently integrated uncached hazard.
- A serial ABBA comparison of the baked shaderball at 64x64, 4 spp and eight
  threads measured 23.89 / 22.44 s before and 22.23 / 21.78 s after. Median
  render time fell from 23.165 to 22.005 s (5.01%). Repeated images within each
  version matched exactly. This small preview does not establish the speedup
  at 512x512, nor an equal-variance comparison between versions.
- Build logs, task-only diff, binary/source hashes and render evidence are in
  `build/conditional-grid-cursor/`; the final benchmark is
  `benchmark-fast-bounds-summary.json`. The earlier `benchmark-summary.json`
  records an intermediate implementation with expensive per-operation interval
  rounding, which was replaced before final validation.

Validation after exposing tracking settings (2026-10-20, Release, CPU):

- All 15 selected conditional/classic CTest checks passed. The added
  `global_conditional_tracking` check covers loading, round-trip serialization,
  invalid settings, actual SDF interval queries and absolute-step precedence.
  Coupled random streams verify that collision and shadow entry points both
  forward the settings; independent hazard integration checks both tracking
  estimators with longer intervals and a lower refinement threshold.
- On an empty `0.117`-unit flight with `ell=0.02`, the three scale settings
  `0.5`, `1` and `2` produced 12, 6 and 3 ordinary intervals. An explicit
  `tracking_step=0.03` produced 4 despite a smaller scale setting.
- The default shaderball smoke render passed with mean RGB `0.394794`.
  An additional 32x32, 4 spp preview using `tracking_step=0.02` passed with
  finite, nonnegative, nonblack output (mean RGB `0.395446`).

The original `build/Release/tungsten.exe` was running during validation, so
the updated renderer was linked as `build/Release/tungsten-tracking-config.exe`.
The selected CTest definitions were copied with renderer commands redirected
to that executable; the original process was left running. Build/test logs,
the copied test definitions and the explicit-step preview are under
`build/conditional-tracking-config-review/`. Rebuilding the regular `tungsten`
target after the running process exits produces the usual executable name.

Validation after the normalized-majorant change (2026-10-20, Release, CPU):

- All 14 selected conditional/classic CTest checks passed in a serial run.
  This includes the new `global_conditional_bounds` check, NEE/MIS, CPU
  wavefront tracing, both furnace modes and both media's render checks.
- The new check covers independent high-precision SE fixtures, interior,
  repeated and endpoint critical points, nearly degenerate polynomials,
  and hazard coverage on 90 flights / 360 intervals across three mean
  shapes, three metric scales, two amplitudes and zero/positive births.
- The homogeneous zero-birth example's majorant is `0.381476`; its regression
  check requires it to stay below `0.5`, guarding against the former loose
  `3|g|` coefficient.
- Resumed-flight survival was `0.510583`, or `0.507917` with a numerical
  subdivision, against integrated hazard `0.510523`. Conditional ratio
  tracking gave `0.162395` against `0.162533`. Furnace means were `1.000000`
  without NEE and `0.999504` with NEE.
- A shaderball smoke render at 32x32, 4 spp, two CPU threads took `16.932 s`,
  versus `30.833 s` with the pre-change executable and the same scene.
  The new image was finite, nonnegative and nonblack (mean RGB `0.394794`).
  These are single small renders, not a general speedup claim or an image
  convergence comparison. Tighter bounds change random-number consumption;
  simple scenes can still cost more because constructing the bounds takes
  more work. Interval Newton contraction reduces that extra work.

Build, regression and preserved render logs are under
`build/conditional-bounds-review/`. The majorant implementation still uses delta/ratio tracking and
latest-observation renewal; it adds no off-axis branch.

Validation after geometry endpoint renewal (2026-10-20, Release, CPU):

- `global_conditional_test`, `tungsten` and `classic_global_test` built.
- All 11 conditional CTest checks passed, including endpoint joint moments,
  geometry direction changes, NEE/MIS, CPU wavefront, the two furnace modes
  and the shaderball render.
- Resumed straight-flight survival was 0.511167, or 0.508250 with a numerical
  split, against an integrated-hazard reference of 0.510523.
- White furnace means were 1.000000 without NEE and 0.999508 with NEE.
  The shaderball smoke image was finite, nonnegative and nonblack
  (mean RGB 0.387099).
- The classic render passed. The classic unit test exited with `0xc0000409`
  once in the parallel suite, then passed four consecutive isolated runs.
  The cause of that one process failure has not been established.

Build and test logs are under `build/conditional-renewal-review/`.

Historical validation before geometry endpoint renewal (2026-10-20, Release,
CPU): all 13 conditional/classic CTest checks passed. These fixed-seed estimates
refer to the previous implementation:

| Check | Sampled | Reference |
| --- | ---: | ---: |
| Straight-flight survival | 0.284750 | 0.286860 |
| Off-axis survival | 0.163417 | 0.159755 |
| Same off-axis segment split in two | 0.158333 | 0.159755 |
| Conditional ratio tracking | 0.163883 | 0.163929 |
| Phase / light / MIS contribution | 0.665319 / 0.668032 / 0.665777 | Equal expectation |
| White furnace, NEE disabled / enabled | 1.000000 / 0.993038 | 1.000000 |

The furnace uses 16×16 pixels, 128 spp, a white medium, an internal mirror
triangle and an additional forward triangle. Tests also cover CPU wavefront
tracing and transformed nonlinear SDF cells with clamps. Two axis crossings
can round to adjacent doubles; ray bounds include the actual endpoint
derivatives. The collinearity tolerance includes world-coordinate subtraction
roundoff when resuming a numerically subdivided straight flight.

The knob checks compare frozen outputs from the original float implementation
against the shared CSG formula (26 positions and all three knob variants).
They require identical values and gradients from the actual classic and
conditional media, including transformed, negatively scaled and clamped means;
check serialization of scale/offset/clamp and rejection of missing capabilities;
and check analytic gradients, small mean differences, and spatial and
directional value/derivative/curvature/hazard enclosures. The render test uses
the shaderball at 32×32, 4 spp. Two real cylinder-cap flights guard
against overly loose birth envelopes. A one-component norm is evaluated as an
absolute value, so a planar cap does not inherit a spurious inverse-distance
curvature bound. Tracking can also choose the tighter of the birth Taylor
certificate and a local interval certificate.
After the shared SDF refactor, all 13 tests passed again. Both shaderball presets
also passed 64×64, 8 spp CPU previews (resolution/sample count/output settings
overridden for validation). Their EXRs were finite, nonnegative and nonblack:
mean RGB 0.386182 for classic and 0.387897 for conditional. These small previews
check integration and numerical behavior, not statistical equivalence of the
two transport models. Outputs and render logs are under
`build/sdf-interface-review/{classic_global,global_conditional}/`.

Additional smoke checks passed for the shaderball VDB scene at 32×32, 4 spp,
and for an ordinary homogeneous-medium BDPT scene after the record-lifetime
change. The shaderball check validates loading and transport through the actual
SDF; its low sample count is for implementation verification.

Run from the repository root:

```powershell
cmake --build build --config Release --target global_conditional_test sdf_point_jet_test tungsten classic_global_test --parallel 4
ctest --test-dir build -C Release -R "^(global_conditional_.*|classic_global.*)$" --output-on-failure
.\build\Release\tungsten.exe -t 2 --spp 32 -d build/global-conditional-preview data/example-scenes/global-conditional/scene.json
```

Use `"type": "global_conditional"` on the medium. The example `scene.json`
enables NEE/MIS; `stage2.json` disables direct-light sampling. The mathematical
transport runs on CPU, including the CPU resumable path. There is no dedicated
CUDA implementation of this medium.
