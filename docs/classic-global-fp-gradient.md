# Classic transport with first-passage gradient correction

`classic_global_fp_gradient` is an experimental CPU `path_tracer` medium. It
uses the **same distance sampling, extinction, majorants and shadow
transmittance as `classic_global`**, with a different gradient distribution at
accepted collisions. Mirror, conductor and Lambert micro-materials are supported.

The gradient model uses the single-point positivity factor from the updated
`GPIS_Gradient_Distribution(1).pdf`, sections 4 and 5. The default anchor
selector now compares five fixed candidates using the actual SDF. The original
document's Hermite selector remains available explicitly. Both classic and
conditional correction use this shared implementation.
It is an approximation to the first-passage gradient, combined deliberately
with classic transport; it is not a realization of a consistently conditioned
GP, nor an exact continuous first-passage renderer.

## Run

From the repository root in PowerShell:

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_FP_GRADIENT_TESTS=ON
cmake --build build --config Release --target tungsten classic_global_fp_gradient_test --parallel 4
ctest --test-dir build -C Release -R "^classic_global_fp_gradient_" --output-on-failure
.\build\Release\tungsten.exe -t 4 -s 41 -d outputs/fp-gradient-sphere data/example-scenes/classic-global-fp-gradient/scene.json
```

For an existing classic scene, retain its GP mean, covariance, material,
tracking settings, bounding geometry, camera and lighting. Change the medium:

```json
"type": "classic_global_fp_gradient",
"fp_gradient": {
  "enabled": true,
  "bridge_correction": true,
  "anchor_selection": "fixed5_snr"
}
```

Both booleans default to `true`; `anchor_selection` defaults to `fixed5_snr`.
Existing corrected scenes without that field also use the new default. Set it
to `hermite` to reproduce the old single-anchor renders. Other selector strings
are rejected. Set `enabled=false` to delegate distance and
phase sampling entirely to classic, including its random-number consumption.
Set only `bridge_correction=false` to retain the explicit start-conditioned
Gaussian and Rice slope weighting, while omitting bridge positivity. The latter
is a separate ablation and does **not** recover the memoryless classic law.

With correction enabled, the covariance must be a stationary
`squared_exponential` kernel. Its anisotropic metric is supported. Procedural
nonstationary covariance and other kernel families fail during preparation.
The mean can use any of the SDF sources supported by classic, with the same
required bounds. Existing SDF grids are queried, not rebaked.

## Probability model

For a straight flight, retain its birth position, observed field value `f0`,
full unnormalized gradient `g0`, direction `w`, and physical age `t`.

1. Sample the event distance through the shared classic implementation.
2. At a collision, condition the SE process on `F(0)=f0`, `w.G(0)=w.g0`, and
   `F(t)=0`. This gives the endpoint longitudinal Gaussian `N(mu, sd^2)`.
3. Set `begin=max(0,t-1.5/gamma)`, with `gamma=sqrt(w^T A w)`. Evaluate the
   actual mean SDF at `s_i=begin+(i+0.5)*(t-begin)/5`, `i=0,...,4`.
   These five fixed midpoints avoid the degenerate endpoints; no dense search,
   interpolation, local refinement or reference-driven selection is used.
4. At each candidate use exact SE conditioning to obtain its mean `a`, covariance
   `c` with the endpoint derivative, and conditional variance `b^2` after
   fixing that derivative. Then `p=a/b`, `q=c/(sd*b)`. Its positivity
   probability is exactly `Phi(p+q*u)`, where `u=(v-mu)/sd`; no bivariate
   probability, inverse normal or probit tangent is involved. Let `c0=-mu/sd`
   and `u0=(c0-sqrt(c0*c0+4))/2` be the uncorrected negative-flux mode in
   standardized units. Evaluate this without cancellation when `c0>=0`.
   Select the smallest `p+q*u0` (earliest candidate breaks a tie), and keep
   only that candidate's `p,q` throughout sampling. This remains one anchor,
   not a five-point positivity product.
5. Draw the negative derivative from
   `p(v) proportional to (-v) N(v;mu,sd^2) Phi(p+q*(v-mu)/sd)`, `v<0`.
6. Use `FixedSpeedGradientDistribution` for the transverse Gaussian and the
   matched directional density, conditional on this already sampled `v`.

The crossing slope factor occurs once, in step 5. Bridge probability and its
normalizer do not enter extinction, throughput, shadow weights or collision
acceptance. Rejected slope proposals resample the slope at the **same event**.
After 32 failed Rice proposals, a log-concave exponential-envelope sampler
samples the same target; it neither drops the path nor falls back to classic.

For `anchor_selection="hermite"`, step 3 instead solves `x^3-P*x-Q=0`,
where `P=(4*f0+t*g0)/(t*speed)` and `Q=2*f0/(t*speed)`, and
`speed` is the uncorrected negative-flux mode magnitude. It chooses
`s=max(t*x/(1+x),t-1.5/gamma)`. Surface starts use `x=sqrt(P)`.

NEE and continuation share the sampled axial derivative. Mirror and conductor
sample the remaining transverse gradient and preserve the full gradient in
the successor path context. Lambert samples one full gradient with the event,
shared by both strategies. Scattering evaluation, PDF and sampling all use the
same conditional measure. The shadow medium itself still uses classic tracking
and does not update this context.

The existing normalized sampler and deterministic diagnostic integrals are
retained. Small normalized distances (`gamma*t<0.5`) use 100-decimal-digit
SE conditioning to resolve near-singular endpoint and
bridge covariance. This adds cost for close repeated collisions. There is no
minimum-flight-distance clamp or rejection of close classic collisions.
Candidates with a tiny bridge variance also retry at high precision
on otherwise long flights, at the same positions. The legacy cubic root is evaluated with a scaled
discriminant and a cancellation-free negative-P branch.
Tail derivatives use CDF hazard ratios, and the rejection envelope uses
centered log-density ratios, avoiding subtraction of enormous log probabilities.
For very grazing reflections, the sampled gradient supplies the directional
PDF directly. The successor retains the double-precision reflected direction
and axial derivative; float ray rounding must not erase the positive outgoing
derivative or turn a valid sample PDF into zero. Directional PDF comparisons
therefore use the precision of the stored ray, rather than bitwise equality.
Sampled phase PDFs are retained as doubles through the continuation MIS
calculation. The power heuristic uses bounded density ratios so concentrated
lobes neither overflow their stored PDF nor overflow when squared.

## Birth policy and interpretation

On external entry (including a camera inside the medium), sample
`(F0,G0) | F0>0`: a truncated scalar Gaussian and the independent stationary
SE gradient Gaussian. Distance sampling remains independent of this sampled
jet. This is an explicit **sampled-positive-jet hybrid**, not the exact gradient
conditioned only on an unknown positive start and a subsequent first hit.
In particular, distance information does not reweight the distribution of
that latent starting jet.

After a GP reflection, use `F0=0` and the sampled full gradient. The outgoing
direction must point to the positive side. This extends the document's
strictly positive start to the outgoing surface boundary condition.
Straight geometry subdivisions retain the start and accumulated age. A
direction change at an ordinary mesh surface starts a new sampled-positive
jet; a mesh intersection is not a GP zero observation. Changing media resets
the renderer's branch-local context as usual.

The single-anchor survival approximation can miss features of a complex mean
profile. Five fixed candidates can miss a narrow bottleneck, and minimizing
the SNR at one representative slope does not minimize error against the full
GP reference. The Hermite selector also need not identify the bottleneck of the
actual shaderball mean. Exact single-point Gaussian conditioning does not
make the all-prefix positivity event exact. For narrow grooves, compare gradients at
fixed collision positions as well as rendered images; use refined finite
prefix references before interpreting agreement as continuous-path accuracy.

## Checks

`classic_global_fp_gradient_test` includes independent numerical distribution
fixtures, negative support, PDF normalization, sample moments/CDFs, short
surface flights, long flights, and rare-bridge rejection fallback. It also
checks exact classic event distances and shadow draws on fixed rays, exact
disabled-mode behavior for all three materials, JSON round trips, and
anisotropic phase normalization / matching MIS densities.

The render smoke test runs a small bounded sphere with NEE, checks the renderer
log for internal errors, and validates finite nonnegative EXR pixels with
visible image variation. Teardown reports corrected collisions, sampler
fallbacks, and exterior initializations. These counters are diagnostics only.

## Updated-document benchmark and historical two-anchor results (2026-11-08)

The 12 linear and 6 nonlinear cases in section 8 of the supplied updated
`GPIS_Gradient_Distribution(1).pdf` can be reproduced with:

```powershell
cmake --build build --config Release --target classic_global_fp_gradient_test --parallel 4
.\build\Release\classic_global_fp_gradient_test.exe document outputs/single-anchor-document-validation 100000
python tools/validate_fp_gradient_document.py outputs/single-anchor-document-validation --power 16 --replicates 8
```

The Python oracle requires NumPy and SciPy; Matplotlib adds standalone PNG/SVG
PDF and CDF figures. Project-local packages in `build/fp-gradient-validation-deps`
are supported. Outputs include `comparison.html`, `validation.json`, `metrics.csv`,
the production PDF/CDF export, and raw samples from both RNG adapters.

Each case exports 100,000 actual `Law::sample` draws with the 64-bit RNG and
100,000 with the renderer's `IndependentPathSampler` adapter. Independent SE
feature expansion and QR conditioning check the Gaussian and anchor parameters.
Eight scrambled Sobol replicates of 65,536 residual bridges compare 25, 51 and
103 nested interior positivity checks, analytically integrating the negative-flux
terminal Gaussian. These finite-prefix references are not continuous-path truth.
The PDF does not specify its exact 25-point placement or seed; this benchmark
uses equally spaced interior points, a zero linear mean intercept, and sigma =
gamma = 1 for the nonlinear cases, consistent with its ablation tables.

In the archived two-anchor run, all 18 implementation checks and both RNG
adapters passed. Maximum empirical
CDF error against the implemented distribution was 0.004761. Against the
103-point reference, that two-anchor approximation had maximum KS
distance 0.023539 (linear case 3), overlap 97.646% to 100%, and maximum absolute
mean error 0.026036 (nonlinear case 5). The largest 51-to-103-point CDF difference
was 0.000260. Thus these cases support the implementation of the stated
approximation, while measuring its remaining model error.

The updated document's single-anchor analytic means and both ablation means
reproduce to its four printed decimal places when using its five-node bivariate
CDF approximation. That approximation introduces visible moment error in some
cases: linear case 6 gives -0.8663185 with five nodes versus -0.8646770 with
adaptive integration. The validation plots use the adaptively normalized law.
The first benchmark evaluated the single-anchor scheme only as a comparison
model. Current exports use `single_anchor_fixed5_snr`; the validator also
recognizes the historical `single_anchor_v2` (Hermite) and two-anchor versions.
`--reference-directory` can reuse prior converged GP references while rerunning
production parameter/PDF/CDF and RNG checks. For the new selector it recomputes
GP-reference errors on the archived curve grid and records its point count;
it does not reuse the old Hermite error metrics. Existing distance and
transmittance algorithms remain unchanged.

## Shaderball ray diagnostics

`fp_gradient_rays` (built under `TUNGSTEN_BUILD_FP_GRADIENT_TESTS`) extracts
camera/GP collision states using production `global_conditional` distance and
mirror phase samplers, both with and without the bridge option. The diagnostic
omits light sampling, roulette and ordinary surface reflections, and stops at
the shaderball scene's floor or process box. Its paths are not exact replays of
rendered pixel samples. It does not modify production transport.

Each comparison fixes the birth jet, incoming direction and collision age.
`gradient_rays.py` compares the classic axial negative-flux Gaussian, the
start-conditioned law, and the exported production law against nested
positive-prefix GP references. Actual SDF values are exported at the reference
nodes; no fitted Hermite mean or SDF interpolation is used by the reference.
It reports grid changes and independent scrambled-Sobol replicates separately.

```powershell
cmake --build build --config Release --target fp_gradient_rays --parallel 2
.\build\Release\fp_gradient_rays.exe scan <spec.json> <output>/rays.json
python -B tools/surface_compare/gradient_rays.py screen <output> --threshold .00001 --power 11 --workers 6
python -B tools/surface_compare/gradient_rays.py refine <output> --ids <record-ids> --power 16 --replicates 8
```

The scan spec provides `scene`, `box_min`, `box_max`, `stride`, `samples`,
`max_bounces`, `seed` and `profile_denominator` (416 initially). The `detail`
C++ command accepts selected `records` and `query_derivatives`, exports actual
production PDF/CDF queries, and optionally increases `profile_denominator`.
Both commands accept an `anchor_selection` override; otherwise they follow
the scene setting and its default. Production metadata exports the candidates'
exact SDF values, positions, `p,q`, mode scores and selected index.
Use `refine --denominator N` with an exported profile divisible by N to compare
N/4-1, N/2-1 and N-1 strictly interior constraints.

The archived two-anchor experiment is in
`outputs/shaderball-gradient-ray-diagnostic-20261108/comparison.html`, generated
by `tools/surface_compare/gradient_rays_report.py`. It includes improving cases,
cases where classic is better than the combined start/bridge correction, an
ineffective correction, and a near-neutral control. No validated case in this
scan showed the bridge alone worsening the conditional axial CDF error.
These are fixed-event, finite-prefix distribution comparisons, not claims about
continuous first passage, free-flight accuracy, or final image error.

The single-anchor update uses the same 8 general rays and 5 fixed-collision
groove rays, retaining their exact birth observations, directions, distances,
SDF profiles and stored GP reference curves. The report is
`outputs/single-anchor-shaderball-l0p2-s0p02-256-128spp-20261108/rays/comparison.html`.
It overlays classic, conditional, old two-anchor, new single-anchor and GP
reference distributions. `tools/surface_compare/single_anchor_update.py`
prepares the fixed records and produces interactive plots after the C++
`detail` command exports the new law. Historical reports remain unchanged.

The updated 18-case production export passes both 100,000-sample RNG checks
per case (maximum empirical KS 0.004921). Its maximum KS to the reused
103-point GP reference is 0.022666. On the fixed shaderball selection, the
single-anchor update improves C2/C3, but is substantially worse on P1/P2 and
groove A/B. The document benchmark does not establish improvement on arbitrary
SDF profiles. See the report for all cases, including neutral short flights.

## Fixed five-candidate update (2026-11-09)

`tools/surface_compare/fixed_anchor_update.py` compares the new default with
explicit `hermite` on the seven previously frozen groove rays. Reproduce it with:

```powershell
python -B tools/surface_compare/fixed_anchor_update.py prepare
.\build\Release\fp_gradient_rays.exe detail outputs/fixed5-anchor-validation-20261109/rays/fixed5_snr-input.json outputs/fixed5-anchor-validation-20261109/rays/fixed5_snr.json
.\build\Release\fp_gradient_rays.exe detail outputs/fixed5-anchor-validation-20261109/rays/hermite-input.json outputs/fixed5-anchor-validation-20261109/rays/hermite.json
python -B tools/surface_compare/fixed_anchor_update.py report
```

The report is `outputs/fixed5-anchor-validation-20261109/rays/comparison.html`.
It checks all five candidates with independent SE-feature QR, the selected
index, actual C++ PDF/CDF, and reproduction of the archived Hermite law. GP
references and ray inputs are frozen; no new favorable rays are selected.
F1 chooses the third candidate at `4.403977764 ell`, moving from the Hermite
anchor at `3.653977764 ell`. Its KS to the stored GP reference falls from
`0.539003` to `0.011941`. G1, G3 and G6 are worse than Hermite; all seven
remain better than the uncorrected conditional gradient in this fixed set.
These results do not establish image-wide improvement.

All 11 focused CTest checks pass. The 18 document cases pass independent
parameter/PDF/CDF checks and 100,000 actual samples per case from each of the
two RNG adapters (maximum empirical CDF error `0.0048973`). Their benchmark is
`outputs/fixed5-anchor-validation-20261109/document/comparison.html`.

## Native renewal+ transport with fixed5 gradients

`renewal_plus_fp_gradient` isolates the effect of replacing the gradient in
the existing 3D sparse-convolution renewal+ renderer. Copy a native
`sparse_conv_noise` scene, change its medium type, and add:

```json
"fp_gradient": {
  "enabled": true,
  "bridge_correction": true,
  "anchor_selection": "fixed5_snr"
}
```

The supported configuration uses `correlation_context="renewal+"`,
`intersect_method="gp_discrete"`, `normal_method="conditioned_gaussian"`,
`single_realization=false`, `1D_sampling=false`, and
`1D_sampling_scheme="uni"`. The GP must be a single stationary SE process
without embedded covariance or additional observations. The existing BRDF
phase and its material settings are retained. In particular, the shaderball's
mirror phase samples the complete gradient as a latent observation, then uses
the original conditional delta-reflection PDF and weight. It does not invoke
the native 1D normal-distribution NEE/MIS code.

The hook executes after native distance sampling, inward-crossing validation,
and event weighting. It cannot change that query's accepted position, escape
decision, or collision weight. Shadow queries never invoke the hook: they
still trace a native conditional realization and return binary visibility.
The hook is also reached by the CPU wavefront distance interface.

For a conditioned segment, the birth value and full gradient are saved before
native intersection overwrites the state. The gradient is saved with the same
float conversion used by native conditioning. For an unconditioned exterior
segment, the birth value and gradient are evaluated from the same realization
that found the collision; they are not independently drawn. The fixed5 law
uses the unrounded native hit distance and the actual mean SDF. Its sampled
axial derivative and conditional transverse Gaussian produce a full gradient,
which replaces both `sample.aniso` and `state.lastAniso`. Thus reflection and
the next renewal+ segment consume the same replacement observation.

Native accepted events without a positive flight distance or births outside
the positive-start domain of the fixed5 law retain their native gradient. Separate counters
report every such exception (`retained_native_zero_length` counts distances
that do not satisfy `t > 0`); no collision is discarded or resampled. Setting
`fp_gradient.enabled=false` preserves native samples and RNG consumption.

```sh
cmake --build build --config Release --target tungsten renewal_plus_fp_gradient_test --parallel 4
ctest --test-dir build -C Release -R "^renewal_plus_fp_gradient_" --output-on-failure
```

Tests cover identical fixed-birth distance events and shadow queries,
independent reconstruction of the sampled fixed5 gradient, primary birth
observations, reflected directions and weights, propagation of the full jet,
disabled-mode identity, and the synchronous/wavefront interfaces.

This is an ablation of the native sparse-convolution renderer. Its finite
impulse density and ray-marching step remain unchanged, so it is not an exact
continuous SE-GP reference. After the first changed gradient, later paths
diverge; identical transport algorithms do not imply identical full paths.
