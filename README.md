# Sparse Convolution GPIS (C++ Code)

### [Project Page](https://cs.dartmouth.edu/~wjarosz/publications/xu25practical.html) | [Paper](https://cs.dartmouth.edu/~wjarosz/publications/xu25practical.pdf) | [Shadertoy Code](https://github.com/dartmouth-vcl/sparse-conv-gpis-shadertoy)

![](teaser.jpg)

This repository contains the source code for the SIGGRAPH Asia 2025 paper:

> **Practical Gaussian Process Implicit Surfaces with Sparse Convolutions**<br>
> Kehan Xu (Dartmouth College), Benedikt Bitterli (NVIDIA), Eugene d'Eon (NVIDIA), Wojciech Jarosz (Dartmouth College)

The implementation is built upon the CPU-based [Tungsten renderer](https://github.com/tunabrain/tungsten) and extends the [codebase](https://github.com/daseyb/gpis-light-transport) from the paper ["From microfacets to participating media: A unified theory of light transport with stochastic geometry"](https://cs.dartmouth.edu/~wjarosz/publications/seyb24from.html). For general usage and setup instructions, please refer to Tungsten’s official documentation.

# Overview

Different Gaussian process implicit surface (GPIS) representations are implemented as new medium types in `src/core/media/SparseConvolutionNoiseMedium/`. This includes the function space (`FunctionSpaceGaussianProcessMedium`) and weight space (`WeightSpaceGaussianProcessMedium`) representations from prior work, as well as our new method that approximates GPISes as procedural noise (`SparseConvolutionNoiseMedium`).

For core volumetric functionalities, `sampleDistance` computes the closest intersection with a randomly sampled GPIS realization, while `transmittance` returns `0` or `1` depending on whether the ray intersects a random realization.

All GPIS representations are rendered using Tungsten’s standard path tracer located at `src/core/integrators/path_tracer/PathTracer.cpp`.

The core logic of sparse convolution noise is implemented in `src/core/math/sparseConvolutionNoise.cpp`, which includes:
- Evaluation of values and gradients for both 3D and 1D noise.
- Conditioning for the Renewal and Renewal+ memory models.
- Next-event estimation for GPISes with mirror or conductor materials (supported for 1D noise).
- Multi-resolution sparse convolution noise.

Finally, `src/core/math/GPFunctions.cpp` contains the implementation of various stationary and non-stationary convolution kernels and their corresponding covariance functions.

# Building without macrofacet

`neural_causal_renewal` is a native Tungsten medium, enabled by default. Its
float32 MLP/GRU inference and flight sampler live in `src/core/math/neural/`,
and `src/core/media/NeuralSdfGrid.cpp` reads the SDF. These compile directly
into `core`, using the same exported model JSON and network equations as
macrofacet. No macrofacet source package or neighboring checkout is needed.
Initialize this repository's existing submodules and install the normal
Tungsten dependencies (including Boost, OpenEXR, OpenVDB, and TBB), plus
NanoVDB and nlohmann_json for neural rendering.

For Windows with an existing vcpkg installation, adjust the toolchain path:

```powershell
git submodule update --init --recursive
cmake -S . -B build "-DCMAKE_TOOLCHAIN_FILE=D:/vcpkg/scripts/buildsystems/vcpkg.cmake" -DBUILD_EDITOR=OFF -DTUNGSTEN_ENABLE_NEURAL_RENEWAL=ON -DTUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS=OFF
cmake --build build --config Release --target tungsten --parallel 4
```

Model weights and SDF files are runtime inputs, not compile-time dependencies.
The gitignored `models/` and `gpis-scenes/` directories must be supplied
separately for scenes that use them. See [neural renewal](docs/neural-renewal.md)
for asset paths and optional tests. Set `TUNGSTEN_ENABLE_NEURAL_RENEWAL=OFF`
to build only the other medium implementations.

# Scene Files

## Empty-origin first-hit GPIS experiment

`first_hit_gpis` samples a first intersection conditioned on a positive initial
field value, then reuses the unchanged `classic_global` phase for direct-light
shading. It stops continuation after that hit; light connections retain geometry
shadows with unit GP transmittance. This is a first-hit preview, with numerical
probability tolerances, rather than full GP path transport.

See the [implementation and configuration notes](docs/first-hit-gpis.md) and
[self-contained sphere scene](data/example-scenes/first-hit-gpis/scene.json).

## Classic transport with corrected gradients

`classic_global_fp_gradient` keeps classic distance sampling and shadow
transmittance, and changes only the accepted collision's gradient law using a
single-anchor first-passage approximation. By default, `fixed5_snr` selects
one of five fixed candidates using the actual SDF's bridge score; use
`fp_gradient.anchor_selection="hermite"` to reproduce the document's original
placement.
It supports stationary anisotropic SE
kernels and mirror, conductor or Lambert micro-materials. See the
[model, birth policy and run instructions](docs/classic-global-fp-gradient.md)
and the [sphere scene](data/example-scenes/classic-global-fp-gradient/scene.json).

The same gradient experiment can use `global_conditional` transport: add
`"fp_gradient": {"enabled": true, "bridge_correction": true}` to that medium.
Its conditional collision/shadow law is retained, and the corrected gradient
becomes the next flight's observation. See the
[conditional integration notes](docs/global-conditional-implementation.md#experimental-first-passage-gradient-correction)
and [conditional sphere scene](data/example-scenes/global-conditional-fp-gradient/scene.json).

## Fixed-history GPIS transport

`sparse_history_gpis` integrates a sparse-history crossing hazard and supports
multiple collisions with mirror, conductor, or Lambert micro-reflection. It
uses no learned model. Start from the
[self-contained sphere scene](data/example-scenes/sparse-history-gpis/scene.json),
or retain a classic scene's `gaussian_process` and `material`, change the medium
type, and remove `tracking_step`.
The example starts at 32 x 32 and 1 spp because this mode evaluates numerical
Gaussian probabilities during CPU transport.

```json
"type": "sparse_history_gpis",
"sparse_history": {
  "points": 2,
  "max_step_over_length": 0.25,
  "max_mean_range_over_sigma": 0.25,
  "optical_depth_tolerance": 0.0001,
  "distance_tolerance": 0.00001,
  "max_subdivisions": 4096,
  "probability_absolute_tolerance": 1e-9,
  "probability_relative_tolerance": 0.0002,
  "probability_initial_samples": 256,
  "probability_max_samples": 8192
}
```

`points` accepts 0, 1, or 2 **internal** history signs. An exterior flight uses
the original Gaussian process and an additional `F(origin)>0` sign; it does
not draw a fictitious known origin value or gradient. A reflected GP flight
conditions on `F(origin)=0` and the full, unnormalized collision gradient.
There is no positive-origin inequality on such a surface birth. Only the
latest collision observation is carried across bounces.

Geometry candidates are independent of integration nodes. Fixed physical
correlation-length buckets contain grid cell boundaries and all roots of the
cell's quadratic directional derivative. A sphere adds its closest approach;
other procedural means use declared breakpoints and fixed derivative probes
as a candidate heuristic. Selection uses only positions strictly before the
query. With two points, the first is `t-min(ell,t/2)` and the second is the
lowest standardized conditional mean in a different bucket, from the recent
pool or its older minimum. With one point, selection takes the lowest score
from this pool and the recent point. Coincident constraints are excluded.

The local hazard is `p(F=0)*E[V+ 1_history | F=0]/P(F>0,history)`.
Numerical Gaussian probabilities and crossing moments determine both distance
and the collision speed distribution. Speed sampling uses a flux-normal
proposal with historical acceptance, falling back to CDF inversion after 64
unsuccessful proposals. Given the sampled speed, transverse
gradient components are Gaussian. Mirror/conductor NEE and continuation share
that speed and use the matching directional PDF; Lambert branches share one
sampled full gradient. Shadow transmittance uses the same integrated hazard.

One exponential threshold is carried through straight transparent boundaries,
together with the original flight and physical age. Geometry `nearT` never
cuts away part of the GP flight. A direction change at an ordinary surface
starts an exterior-positive renewal approximation; it is not a GP zero.
Each shadow branch has a separate context. Invalid outward surface births
are counted and absorbed; other numerical failures are reported, not misses.

This CPU reference supports isotropic stationary SE covariance and analytic,
procedural `SdfQueries`, or trilinear VDB means. It approximates continuous
first passage using finitely many history signs. Probability and optical-depth tolerances are numerical error
estimates, not certified bounds. Tightening quadrature does not add history
points. Near-singular probabilities can be expensive; validate low-resolution
renders before increasing image size or sample count. Use `path_tracer` with
`use_cuda_gpis: false` and `wavefront_gpis: false`;
state-free distance PDFs for bidirectional transport are unsupported.
The sign count is bounded, while the geometry cache retained for backwards
queries grows with traversed cells. Probabilities too small to represent cause
an explicit numerical failure instead of silently treating the flight as empty.

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_SPARSE_HISTORY_GPIS_TESTS=ON
cmake --build build --config Release --target tungsten sparse_history_gpis_test --parallel 4
ctest --test-dir build -C Release -R "^sparse_history_gpis" --output-on-failure
.\build\Release\tungsten.exe -t 4 -s 41 -d outputs/sparse-history-gpis data/example-scenes/sparse-history-gpis/scene.json
```

The optional CDF table backend replaces eligible bivariate probability
integrals with cubic interpolation. Enable it inside `sparse_history`:

```json
"probability_method": "lut",
"cdf_table": "../../gpis/bivariate-normal-cdf-v1.bin"
```

The file path is relative to the scene. The shared table in
`data/gpis/bivariate-normal-cdf-v1.bin` is approximately 5.1 MB; all media loading
the same path share its immutable storage. It tabulates the scene-independent
function `Phi_2(a,b;rho)`, with correlation parameterized by `asin(rho)`.
Each cell carries an interpolation-error estimate. A lookup is accepted only
when its error is at most one quarter of the configured relative probability
budget. Tails, nearly singular correlations, and insufficiently accurate cells
use the numerical backend. Table errors propagate into the crossing moment;
cancellation first retries the moment with numerical CDFs, then uses numerical
speed integration if needed. These are numerical
error estimates, not certified interval bounds.

`probability_method: "numeric"` remains the default reference. Both backends
use the same history selection and start conditions: the exterior origin sign
is retained unless the existing probability-budget check permits pruning it.
The table supports both one- and two-point models, but it only replaces
bivariate subproblems. Higher-dimensional probabilities still require numerical
work. The [LUT example](data/example-scenes/sparse-history-gpis/scene-lut.json)
starts with one internal point. Collision speed rejection uses the same
history acceptance function; its inverse-CDF fallback computes its own
integral normalization rather than reusing an interpolated crossing moment.

Regenerate the table with Python, NumPy and SciPy, or compare both backends
at each of K=1 and K=2 with the supplied benchmark:

```powershell
python tools/generate_bivariate_normal_cdf_table.py
.\build\Release\tungsten.exe -t 4 -s 41 -d outputs/sparse-history-gpis-lut data/example-scenes/sparse-history-gpis/scene-lut.json
python tools/compare_sparse_history_lut.py
```

The comparison writes images, render timings, lookup acceptance counts and
same-seed image differences to `outputs/sparse-history-lut/comparison.html`
and `comparison.json`. Compare each K with itself; changing K also changes
the finite-history model. Render teardown reports accepted table requests
and numerical fallbacks.

## Pointwise local Poisson ellipsoids

The independent `poisson_ellipsoid_local` medium evaluates the SDF value **and
its full gradient at every tracking candidate**. Its local center-density law is
`lambda_i0 * exp(-(phi(x) - gradient(phi(x)).dot(q_i(m))) / decay_length)`.
The gradient magnitude is preserved for trilinear fields; only the particle
orientation frame uses its unit direction. Each new bounce uses the same
pointwise rule, without particle identities or empty-path history.

See [the runnable example](data/example-scenes/poisson-ellipsoid-local/scene.json).
The [shaderball preset](data/example-scenes/poisson-ellipsoid-local/shaderball.json)
uses the original procedural `knob` SDF, camera, lighting, ground and green
mirror reflectance. It queries the SDF directly without external grid assets.
It starts with `decay_length=0.001`, semiaxes `[0.007, 0.007, 0.004]`,
center density `200000`, and both light-sampling switches enabled. The preset
renders at 512 x 512 and 256 spp; `--spp` overrides only the sample count.

```powershell
.\build\Release\tungsten.exe -t 4 -s 41 data/example-scenes/poisson-ellipsoid-local/shaderball.json
```

`sdf` takes an existing scene mean: `linear`, `spherical`, `homogeneous`, a
procedural mean with certified SDF queries, or a `tabulated` trilinear VDB.
For the contact experiments, copy `gaussian_process.mean` from a `classic_global`
scene into this medium's `sdf` field; retain its process box. No GP covariance,
baked density channel, or learned model is used. Transport outside the process
box is vacuum; its boundary is a finite-domain truncation of the density law.

Each entry in `particles` specifies positive `semiaxes` in world length units,
nonnegative center `density` in inverse world-volume units at `phi=0`, optional
`rotation_degrees`, and a smooth `mirror` or `conductor` material. Rotations are
applied X, then Y, then Z. `frame="sdf"` (default) uses local Z along the current
SDF gradient; local X is projected world X, switching to projected world Y
when `abs(N.x) >= 0.9`. A zero gradient uses the world frame. `frame="world"`
fixes particle orientation. `decay_length` is positive; `tracking_step` controls
the maximum length of a majorant interval and defaults to `decay_length`.
Changing the step changes efficiency, not the specified collision law.
In rapidly changing density, tracking automatically shortens costly intervals
to at most one decay length of SDF variation. For SDF-aligned particles the
height bound uses their support along local Z, including `rotation_degrees`,
rather than their largest semiaxis. Grid rays reuse the current cell's
trilinear polynomial while evaluating its changing value and full gradient
at each candidate; this does not freeze the local plane.
Sparse grid regions use conservative bounds over blocks of 8 cells per axis,
within `tracking_step`. They retain their nonzero collision probability and
fall back to cell bounds when a block would generate too many candidates.

Distance, particle type and geometric normal are sampled together using a
marked Poisson majorant. The visible-normal proposal samples the projected
full ellipsoid, equivalent to the [SGGX visible-normal distribution](https://research.nvidia.com/labs/rtr/publication/heitz2015sggx/).
Acceptance includes the finite-particle height-density correction. Every
candidate uses a fresh local plane; the algorithm never evaluates the SDF at
a reconstructed particle center. The same process estimates shadow
transmittance by ratio tracking. `extinction(position, direction, order)` is a
deterministic angular quadrature for inspection; transport does not use this
approximate integral or a table of interpolated normals.

### Gaussian height density

To place a layer of particles around an SDF level, replace `decay_length` with:

```json
"density_profile": {
    "type": "gaussian",
    "center_height": 0.0,
    "stddev": 0.02
}
```

The center-density profile is
`lambda_i(h) = particles[i].density * exp(-0.5 * ((h - center_height) / stddev)^2)`.
Here `h = phi(x) - gradient(phi(x)).dot(q)` is the particle center height in
the plane queried at the current candidate, including the full SDF gradient
and finite ellipsoid offset `q`. `density` is the **peak** number per volume;
there is no Gaussian PDF normalization factor. `center_height` defaults to
zero, `stddev` must be finite and positive, and `tracking_step` defaults to
`stddev`. Gaussian configurations must omit `decay_length`. Existing scenes
without `density_profile` retain the exponential law (an explicit
`{"type":"exponential"}` also uses the original top-level `decay_length`).

This is a bell-shaped layer on both sides of the selected SDF level: density
decreases both outward and inward. It is not a filled solid with a smooth
boundary. At fixed particle sizes and peak density, shrinking `stddev` also
reduces the number of particles per unit planar area, proportional to
`sqrt(2*pi) * density * stddev`. The finite particle sizes also contribute to
the thickness of the collision layer.

Gaussian density is used consistently in extinction, type/normal sampling,
scattering and direct lighting. The tracker bounds both ends of the possible
center-height interval and includes an interior Gaussian peak. Sparse-block
skipping and pointwise SDF/gradient queries remain available. Gaussian direct
lighting uses a visible-hemisphere density bound; no angular quadrature is
added to rendering. Without a density multiplier, this is ordinary Poisson
transport, with transmittance `exp(-integral(extinction ds))`.

The [Gaussian shaderball example](data/example-scenes/poisson-ellipsoid-local/shaderball-gaussian.json)
uses two particle types (green mirror and red conductor), `stddev=0.02`,
`center_height=0`, 512 x 512 pixels and 16 spp, without Gamma renewal:

```powershell
.\build\Release\tungsten.exe -t 8 -s 41 data/example-scenes/poisson-ellipsoid-local/shaderball-gaussian.json
```

### Gamma renewal

An optional medium-level `density_multiplier` adds non-exponential free flights:

```json
"density_multiplier": {"type": "gamma_renewal", "alpha": 4.0}
```

All particle types share one multiplier `Z` within a free flight. External
origins draw `Gamma(shape=alpha, rate=alpha)`; particle-collision origins draw
`Gamma(shape=alpha+1, rate=alpha)`. For the original optical depth `tau`, their
survival probabilities are respectively `(1+tau/alpha)^(-alpha)` and
`(1+tau/alpha)^(-alpha-1)`. This is a renewal transport model, not exact
conditioning of a persistent finite-particle realization. `alpha` must be
finite and greater than one. Larger values approach the exponential model;
omitting the option or using `{"type":"none"}` preserves the original draws.

The multiplier stays fixed across null collisions, adaptive intervals and
same-medium forward boundaries. A real particle collision (including a virtual
collision for direct lighting) starts a fresh particle-origin flight. An
ordinary surface reflection/refraction starts an external-origin flight.
Entering a different medium starts a new external flight. Branch state is
immutable and shared only until a branch starts or samples its own flight.

Tracking scales both target intensity and its majorant, so the accepted
conditional particle/normal distribution and material reflection weights stay
unchanged. Shadows use the same conditional ratio tracker; candidate-light
intensity ratios cancel the common multiplier. No angular quadrature or
explicit optical-depth accumulation is added to rendering. This mode continues
to support `path_tracer`; marginalized distance PDFs for BDPT are unavailable.
The original `density` and `decay_length` parameters keep their meaning.
`extinction(...)` continues to report the unscaled local coefficient, not the
renewal hazard conditioned on a flight's age or sampled multiplier.
The [Gamma shaderball example](data/example-scenes/poisson-ellipsoid-local/shaderball-gamma-renewal.json)
uses two particle types (green mirror and red conductor), `decay_length=0.02`,
and `alpha=4`, at 512 x 512 and 16 spp.

The sampled normal is kept in the current collision context. Smooth particle
reflection is a delta event conditional on that normal, using Tungsten's
existing specular phase path. `sampleDistance()` returns the collision position
in `MediumSample::p` and its unit outward normal in `aniso`; calling
`sample.phase->sample(sampler, ray.dir(), sample, bounce)` produces the outgoing
direction and reflection weight for that collision.

Enable both `integrator.enable_light_sampling` and
`integrator.enable_volume_light_sampling` for light importance sampling with
MIS. At a tracking candidate, the light sampler proposes an outgoing direction;
its reflection half-vector determines a possible ellipsoid normal. The exact
unnormalized angular rate `K_i = normalIntensity_i / (4 * abs(travel.dot(normal)))`
includes the local height-density weight. Light evaluation sums `F_i * K_i`
over particle types, including each type's mirror color or conductor Fresnel.
Visibility uses the same changing-SDF ratio tracking as other shadow rays.

This estimate is accumulated **before collision acceptance, including null
candidates**. Independent thinning uses a capped moment estimate of the local
rate to limit shadow work when the tracking majorant is loose. This estimate
only chooses how often to query lighting; it is not used as the extinction or
to accept collisions. If the selected-candidate
intensity is `r`, the light estimator divides by `r * light_pdf`; its power-MIS
competitor is the intensity ratio `sum(K_i) / r`. At real collisions, the original
normal-sampled direct-light estimator receives the complementary MIS weight.
Continuation directions and indirect throughput retain their original law.
The intensity ratio is not a normalized phase PDF: no approximate extinction
quadrature enters either estimator. Simply enabling NEE on the conditional
delta phase would not implement this marginalization. Both lighting switches
are required so directly hit emitters are not also counted by continuation.

This implementation supports the CPU path tracer and its wavefront fallback,
not BDPT or a CUDA ellipsoid kernel. More shadow rays increase work per sample;
the benefit depends on light size, scattering and visibility. It does not
eliminate noise from multiple scattering or stochastic shadow transmission.
Finite bounce limits and the process-box truncation remain separate rendering
approximations.

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_LOCAL_ELLIPSOID_TESTS=ON
cmake --build build --config Release --target local_ellipsoid_test tungsten --parallel 4
ctest --test-dir build -C Release -R "^local_ellipsoid" --output-on-failure
```

Tests compare distances with analytic planar and varying-gradient references,
normal/type histograms with independent solid-angle integration, and VDB
transport with integrated local coefficients. Thin-layer tests cover both
orientation frames and tilted particles, and track candidate counts to guard
against a loose height bound. Grid tests check both sparse-block and cell
tracking against the same reference. They check process-box exits
within the geometry epsilon and at zero distance, and render a white furnace
with direct lighting enabled, disabled, through the wavefront path, and with
only one lighting switch enabled. Angular-kernel tests check the reflection
Jacobian and its integral; candidate-lighting tests compare NEE and MIS energy
with independently integrated references for different tracking steps.

We provide the JSON scene files on [Google Drive](https://drive.google.com/file/d/1nzmRUxxPsP5bIgX6SrOjjc0_664KtQgQ/view?usp=sharing) that reproduce the results presented in the paper. Some scene files include multiple rendering configurations, where only one is active while the others are commented out and annotated for clarity.

Additionally, the rendered results are available through an interactive viewer on the project webpage.

## Parameters
Here we list the key parameters for our sparse convolution noise implementation, as well as those for other methods used for comparison.

### Shared Parameters
- **step\_size**: `float`; fixed step size for ray marching.
- **correlation\_context**: `string`, "none", "renewal", "renewal+", or "global"; specifies the memory model.
- **single\_realization**: `bool`; whether to render a single realization (not applicable in function space) or the ensemble average of a GPIS.
- **seed**: `int`; random seed for the GPIS; changing it produces a different single realization.

### Sparse Convolution Noise
- **impulse\_density**: `float`; controls the density (number) of convolution kernels placed per cell.
- **1D\_sampling**: `bool`; chooses between 1D or 3D sparse convolution noise sampling.
- **isotropic\_3D\_sampling**: `bool`; when using 3D sampling, determines whether sampling in world space or isotropic space.
- **1D\_sampling\_scheme**: `string`, "uni", "nee", or "mis"; specifies the importance sampling scheme when next-event estimation is enabled.
- **1D\_gradient\_correlationXY**: `bool`; selects between Renewal+ (`true`) and Renewal Half+ (`false`) memory models for next-event estimation.

### Weight Space
- **basis\_functions**: `int`; number of random Fourier features used.

### Function Space
- **sample\_points**: `int`; number of points sampled along the ray per batch, conditioned on the previous batch.

# Prebaked contact experiments

Run from this project root to generate unit-scale two-sphere and V-groove fields,
matched scene configs, and four-branch render comparisons:

```powershell
cmake --build build --config Release --target bake_contact_fields --parallel 4
python tools/generate_contact_experiments.py
python tools/generate_contact_experiments.py --skip-bake --render --render-cases two_spheres_g0p02 v_groove_a90 --render-size 64 --render-spp 16
```

All generated files stay in this project: `outputs/contact_experiments/fields/`
contains NVDB/VDB models, `outputs/contact_experiments/configs/<case>/` contains
four matched scene configs, `outputs/contact_experiments/renders/` contains
images, logs and the configs actually rendered, and
`outputs/contact_experiments/comparison.html` shows the comparisons.
The defaults find `tungsten` and `bake_contact_fields` under this project's
`build/` (including `build/Release/`), and the joint checkpoint at
`models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json`.
Override `--tungsten`, `--model`, or `--baker` for other layouts. Relative CLI paths
are resolved from this repository root. Generated resource paths are absolute. The default
configs use 256 x 256 at 128 spp; `--render-size` / `--render-spp` override only
the copies saved with each render. Omit those overrides for the full run.
`--render-cases` selects geometry names and `--modes` selects rendering branches.
All outputs stay under `--output`; rerunning replaces the selected outputs.

The two spheres have radius 0.25 and centers at `x=+/- (0.25 + gap/2)`, so the
overall extent is `(1+gap) x 0.5 x 0.5`. Default `--gaps 0 0.02 0.06` includes
tangency and two separated cases. Both spheres belong to **one field and one
medium**. The finite V groove is a closed extruded polygon with outer dimensions
`1 x 0.4 x 1`, notch depth 0.25, and default `--angles 45 90 120` (opening angles
in degrees). Y is up; all geometry is baked in native world coordinates, with
no `field.transform`. Each process box encloses the geometry plus 6 sigma;
another four grid nodes keep interpolation and VDB boundary clamps outside it.

The baker writes the same analytic SDF float samples into NVDB and VDB and
checks their saved values and coordinate maps. NVDB also stores density for
`classic_local`, constant alpha, and sigma. Every grid node in the bake box is
sampled, including the interior; these are not truncated narrow-band fields.
Default `sigma=0.02`, `length_scale=0.02`, and `dx=0.005` give four voxels per
correlation length. Refine selected cases before interpreting model differences:

```powershell
python tools/generate_contact_experiments.py --output outputs/contact_experiments_dx0025 --voxel-size 0.0025 --gaps 0.02 --angles 90
```

The four branches share camera, process box, area light, dim environment,
mirror albedo 0.85, and bounce limits. NEE is disabled in this initial comparison
so all use path sampling; increasing spp is necessary for quantitative contact
brightness comparisons. Tungsten neural uses native scalar CPU inference.
The runner checks completion, internal-error logs, and finite linear PFM output;
PNG previews use filmic tonemapping. Small renders only verify the scene setup,
not statistical agreement. Changing length scale at fixed sigma also changes
roughness (`alpha = sqrt(2)*sigma/length_scale`), and is not an isolated correlation
test. These four branches do not provide a full-history GPIS reference.


## Sparse convolution reference for the contact close-up

After generating the matched close-up configs in `outputs/contact_contrast/configs/`,
compare them against sparse convolution GPIS in the same scene:

```powershell
python tools/compare_contact_gpis.py --render
python tools/compare_contact_gpis.py --render --name validation --size 32 --spp 128 --seeds 314159 271828 161803 --modes sparse_3d_global global_conditional neural sparse_1d_renewal
```

Outputs stay under `outputs/contact_contrast/sparse_convolution/<name>/`:
`configs/` contains runnable scenes, each mode/seed directory contains its actual
scene, images and log, and `comparison.html`, `summary.json`, and
`horizontal_profile.csv` compare linear radiance. Completed, matching prior
renders are reused. Without `--render`, the script generates configs and reports
already completed renders. Use `--base` for another directory with the four
matched template configs. This preset expects sigma=0.02, lengthScale=0.1 and
mirror albedo=0.85, and verifies the shared camera, lighting and covariance.

`sparse_3d_global` uses world-space 3D noise, `correlation_context="global"`, and
`single_realization=false`: a camera path retains the same realization across
all bounces, while different camera samples draw independent realizations.
`sparse_1d_renewal` uses 1D `renewal+` with `1D_gradient_correlationXY=false`;
it retains the last contact rather than the whole path. The default sparse
settings are `impulse_density=10` and `step_size=0.005`; both are configurable.
Finite impulse density and marching resolution make these numerical references,
not exact GP ground truth. Multi-seed statistics average linear images;
displayed PNGs show the first seed with the same filmic mapping. Pixel/profile
errors still include Monte Carlo noise.

# Citation

If you find this code useful in your academic research, please cite the original paper:
```
@article{xu25practical,
    author  = {Xu, Kehan and Bitterli, Benedikt and d'Eon, Eugene and Jarosz, Wojciech},
    title   = {Practical {G}aussian Process Implicit Surfaces with Sparse Convolutions},
    journal = {ACM Transactions on Graphics (Proceedings of SIGGRAPH Asia)},
    year    = {2025},
    month   = dec,
    volume  = {44},
    number  = {6},
    doi     = {10.1145/3763329}
}
