# First-hit GPIS experiment

`first_hit_gpis` renders the first GP intersection from an empty origin and
shades it with the existing `classic_global` phase function. It is a CPU
`path_tracer` experiment for checking the distance law before implementing
collision-origin conditioning or path history.

## What the image means

The initial condition is **the field is positive at the ray origin**. Its value
is unknown and is integrated out. The first zero of the field is sampled along
the entire finite ray, including the geometry `nearT` interval. A missed ray
reaches its geometry endpoint or the environment. A hit receives direct light
through the unchanged `classic_global` mirror, conductor, or Lambert phase.
Lambert's stored visible normal is also sampled by the existing implementation.
There is no correction of the gradient distribution for the preceding empty
path or for the sampled first-hit distance.

After that hit the continuation ends. Light connections have unit **GP**
transmittance; the renderer still traces geometry shadows. Thus the image is
first-hit shading with geometry visibility, not full GP light transport. Enable
both light-sampling switches: with direct lighting disabled a hit has no light
contribution. `max_bounces=1` in the *integrator* also suppresses direct light;
use `integrator.max_bounces=2`. The *medium* accepts only `max_bounces=1`.

Use one finite forward boundary around the field, with the camera outside it
or assigned this medium inside it. There should be no internal boundaries or
ordinary surface reflections before the GP query. The finite process domain
begins at entry into that boundary. The medium rejects continuation once either
the local collision count or the path tracer's non-forward bounce count is
positive. It does not carry survival history across boundary exits and reentry.
This version produces a shaded image; existing surface depth/normal AOVs are
not converted into GP first-hit AOVs.

## Run

The self-contained sphere example needs no VDB or model assets:

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_FIRST_HIT_GPIS_TESTS=ON
cmake --build build --config Release --target tungsten first_hit_gpis_test --parallel 4
ctest --test-dir build -C Release -R "^first_hit_gpis" --output-on-failure
.\build\Release\tungsten.exe -t 4 --spp 8 -d outputs/first-hit-gpis data/example-scenes/first-hit-gpis/scene.json
```

The example uses an analytic spherical SDF. To use an existing `classic_global`
scene, retain its `gaussian_process` and `material`, set `type` to
`first_hit_gpis`, remove `tracking_step`, and set the medium's `max_bounces` to
1. Restrict the scene to the first-hit setup described above. Only a stationary,
isotropic `squared_exponential` covariance with positive `sigma` and
`lengthScale` is supported. Supported means are homogeneous, spherical, linear,
trilinear VDB, and procedural means implementing `SdfQueries`. Finite-difference-only
mean derivatives are rejected because the Rice rate needs the derivative of the
same field being queried. The inherited SDF loading and gradient checks remain
in force, including explicit bounds for procedural means that need them.

```json
"first_hit": {
  "probability_tolerance": 0.002,
  "distance_tolerance": 0.001,
  "max_nodes": 48,
  "qmc_samples": 256,
  "max_qmc_samples": 8192,
  "max_refinements": 256,
  "survival_precheck": true
}
```

`distance_tolerance` is an absolute world distance. `probability_tolerance`
controls the width of the estimated conditional-survival envelope at an
ambiguous threshold. QMC sample counts are **per scramble**, with eight
scrambles. Increasing these budgets can be expensive. The active node limit
does not count SDF traversal or quadrature points. Flights with initial positive
probability below `1e-10` are explicitly unsupported. Exhausted quadrature,
refinement, node, or QMC budgets raise `ConditionalGpis::NumericError`; they do
not redraw the distance random number or report a miss.

`survival_precheck` defaults to `true`. It checks inexpensive survival bounds
before asking for an all-positive probability at the active nodes. Set it to
`false` for an otherwise identical comparison render. It does not alter the
standalone `Flight::survivalBounds()` query, which still refines the full
probability envelope rather than testing one random threshold.

## Per-ray statistics

Statistics are opt-in. Add this inside the medium's `first_hit` object:

```json
"statistics": {
  "enabled": true,
  "output_prefix": "first-hit-stats"
}
```

On render teardown the medium writes `first-hit-stats-summary.json` and
`first-hit-stats-rays.csv`. A relative prefix uses the final renderer output
directory, including the CLI `-d` override; an absolute prefix is also accepted.
Use distinct prefixes for multiple profiled media. Existing files at that prefix
are replaced when a new render is prepared. With statistics disabled no clocks,
output streams, or aggregate locks are used.

The JSON includes the following counts and their corresponding worst rays.
CSV rows contain zero-based pixel coordinates, sample index, scene seed, ray
origin/direction, distance uniform, outcome, and all counters/timings.

| Statistic | Meaning |
| --- | --- |
| `without_extra_nodes` | No internal node was selected anywhere in the entire distance inversion. A prefix can finish through precheck without any orthant call. |
| `unique_added_nodes` | Distinct internal points selected across all prefix queries for one flight. Endpoints are excluded. |
| `max_active_nodes` | Largest set passed to one orthant call, including both endpoints; zero when the entire flight avoids the orthant backend. |
| `max_effective_dimension` | Largest dimension actually evaluated after backend air pruning. This can be lower than the active node count. |
| `candidate_locations` | Cheap locations prepared for selection, including endpoints; these do not all enter the CDF. |
| `candidate_tests` | Evaluations of the two-dimensional insertion score, including repeated locations. |
| `orthant` time | All all-positive backend calls, including one/two-dimensional cases, covariance factorization, and QMC. |
| `multivariate` time | The subset of full orthant calls whose effective dimension is at least three. |
| `qmc` time | The actual Sobol integration loops, excluding factorization. Analytic low-dimensional cases are not counted. |
| `qmc_points` | Newly evaluated Sobol integrand points, summed over all eight replicates; cached points do not count again. |
| `qmc_reused_points` | Previously evaluated points used again by a probability estimate request, summed over all eight replicates. |
| `qmc_preparations` | New QMC matrix and integrand preparation sessions. Raising the sample budget for an unchanged problem does not prepare it again. |
| `precheck_calls` | Threshold-bound attempts before an orthant call, initially and after inserting a node. Raising only the QMC budget does not repeat this check. |
| `precheck_survived` | Attempts whose lower bound decides that the current prefix survives. |
| `precheck_hit` | Attempts whose upper bound decides that the first hit is before the current prefix endpoint. These are prefix decisions, not counts of hit rays. |
| `precheck_marginal_hits` | Subset of `precheck_hit` decided by the one-dimensional marginal upper bound. |
| `precheck_with_extra_nodes` | Successful decisions with more than two considered nodes, before backend pruning. This does not establish that a QMC call was saved. |
| `precheck_nanoseconds` | Total precheck elapsed nanoseconds, including any mean, Rice, and pair work performed inside the attempt. Preparation outside the attempt is excluded. |
| `pair` time | Separate two-dimensional sign probabilities used for bounds and candidate selection, excluding cache hits. |

Times are sums of `steady_clock` intervals across worker threads, **not render
wall time**. `qmc` and `multivariate` are contained in `orthant`; do not add them.
Precheck timing may contain pair-probability work, so it must not be added to
the pair timing as if they were disjoint categories.
`flight` time excludes phase sampling and writing the statistics. Per-call
maxima are provided for the orthant backend and QMC loops; per-flight maxima
and averages are provided for all categories. Failed numerical flights are
reported separately rather than counted as misses. Continuation and shadow
rays are excluded; camera rays that never enter this medium have no CSV row.
QMC call counts refer to estimate requests, including requests answered entirely
from cached results. Relative to recomputing every requested budget from scratch,
the integrand-evaluation saving is `qmc_reused_points / (qmc_points +
qmc_reused_points)`. This is a work-count saving, not a measured time reduction.
Each successful precheck avoids the orthant request immediately following that
attempt. It does not necessarily avoid QMC: the active problem may be analytic
or become low-dimensional after pruning. The changed refinement history can
also affect later calls. Measure the net orthant/QMC reduction by comparing
renders with precheck enabled and disabled, not by equating the decision count
to that reduction. Avoiding orthant throughout a flight does not imply avoiding
SDF evaluations, Rice preparation, or separate two-dimensional probabilities.

Generate a self-contained interactive heatmap report, using only standard Python:

```powershell
python tools/first_hit_statistics.py outputs/first-hit-statistics/first-hit-stats-summary.json --scene outputs/first-hit-statistics/scene.json
```

The script checks every CSV counter sum and node histogram against the JSON,
then writes `first-hit-report.html` and `first-hit-report.json` beside the input.
The HTML shows the fraction of rays requiring extra nodes, peak active node
count, mean high-dimensional probability time, and hit rate at each pixel.
Hover over a pixel for its values. The JSON adds group averages, node-count
percentiles, and the ten rays with the highest accumulated orthant time. Both
reports show new and reused QMC points, preparation counts, and the saved-point
fraction. Old captures without the reuse counters remain supported; the missing
counters display as zero and the report marks them as unavailable.
The reports also show precheck attempts, successful survival/hit decisions,
one-dimensional decisions, and completed flights with no orthant calls. New
heatmaps show that flight fraction and average precheck time. Captures without
precheck counters are marked unavailable and omit these heatmaps.
`--scene` preserves image dimensions even when edge pixels miss the medium;
`--csv` and `--output` can override paths when moving a captured report.
Use `--threads` and `--wall-seconds` to label a report with the worker count and
the renderer's own elapsed-time measurement; they do not change any measured
counter or timer.

## Code flow

1. `FirstHitGpisMedium` converts the finite renderer ray to world distance and
   constructs callbacks for the mean value, its directional derivative, a
   conservative interval minimum, and interpolation boundaries. VDB rays reuse
   `GpisSdfGrid`'s polynomial cursor. Spherical and linear means use analytic
   callbacks. One open uniform random number drives the entire distance query.
2. `FirstHitGpis::Flight` constructs and caches a piecewise Rice integral.
   Integration steps respect both the correlation length and the possible SDF
   change relative to the noise standard deviation. Far from zero, a Lipschitz
   bound permits longer steps. Interpolation boundaries, midpoints, and detected
   SDF minima are candidate locations; they are not automatically CDF nodes.
3. A prefix initially uses its two endpoints and relevant previously selected
   constraints. For a distance-threshold query, precheck first tries a marginal
   upper bound, a Rice lower bound, and an origin-to-node pair upper bound. A
   decisive bound skips the node orthant call. Otherwise `FirstHitGaussian`
   evaluates their joint all-positive
   probability. The Rice count minus adjacent endpoint downcrossing
   probabilities bounds paths missed between those active nodes. Whole-gap air
   bounds can further tighten that error.
4. If the uniform lies outside the envelope, the survival decision is ready.
   Otherwise compare candidate insertion gains using only two-dimensional
   probabilities and add the best candidate. Increase QMC samples when its
   estimated error dominates; for unchanged nodes, means, and seed, retain the
   prepared transform and append only the new Sobol points. The same per-flight
   SDF, pair-probability, and integral caches serve later prefix queries.
   Retry precheck after a node is inserted; a larger QMC sample budget alone
   cannot strengthen these inexpensive bounds, so it does not repeat precheck.
5. Query the full ray first. If it survives, return a miss. Otherwise bisect
   the distance bracket using the same uniform and the original-origin prefix
   law until the distance tolerance is met. A hit gets unit distance throughput
   and the inherited phase. The medium has no collision-origin flight sampler.

## Incremental QMC integration

Incremental integration is enabled automatically; there is no extra scene
setting. Each fixed node set keeps its pruning result, node order, factorization,
Sobol digital shifts, and per-replicate scalar prefix sums. Increasing the budget
from 256 to 512 points per replicate evaluates only indices 256 through 511.
The already accumulated first 256 weights are reused. Prefix sums also provide
the half-budget estimate for the existing numerical error test, including when
the requested budget is not a power of two. No per-point latent Gaussian vectors
are cached.

For a fixed problem stepped through 256, 512, 1024, 2048, 4096, and 8192 points
per replicate, a cold evaluation of every budget uses 129,024 integrand points
across eight replicates. The incremental path evaluates 65,536 new points and
reuses 63,488, a 49.2% reduction in integrand evaluations. This does not predict
the full render speedup: many queries finish at the initial budget, and different
node sets require separate integration. The scalar prefix sums use about 0.5 MiB
per active workspace at 8192 points per replicate, plus factorization storage.

Adding a node or moving the query endpoint rebuilds the workspace. This version
does not reuse transformed samples across different node sets or across distance
bisection queries. The existing stateless `positiveOrthant()` entry point remains
available as a cold-evaluation wrapper. The probability transform, digital
shifts, error estimator, and finite-accuracy distance decision are unchanged;
incremental integration does not make their estimated error bounds certified.

## Probability equations

Let `distance` be world distance along the unit ray. The field is its actual
SDF mean plus zero-mean noise. The noise covariance is

\[
\operatorname{Cov}(Y(a),Y(b))=
\sigma^2\exp\!\left(-\frac{(a-b)^2}{2\ell^2}\right).
\]

Here `sigma` is field-value standard deviation and `ell` is spatial correlation
length. The starting positive probability is
\(p_{\mathrm{start}}=\Phi(\operatorname{mean}(0)/\sigma)\).
The desired survival probability is the probability that **every** field value
on the prefix is positive, divided by this starting probability. Evaluating
the Gaussian probability at a finite set of nodes gives an upper bound because
it imposes fewer constraints.

For this stationary kernel, the noise value and its directional derivative at
the same point are independent. The derivative standard deviation is
\(\sigma/\ell\). Therefore the expected downcrossing rate is

\[
\operatorname{rate}_{\mathrm{down}}(s)=
\frac{\phi(\operatorname{mean}(s)/\sigma)}{\sigma}
\left[
\frac{\sigma}{\ell}\phi\!\left(\frac{\operatorname{mean}'(s)}{\sigma/\ell}\right)
-\operatorname{mean}'(s)\Phi\!\left(-\frac{\operatorname{mean}'(s)}{\sigma/\ell}\right)
\right].
\]

The first factor is the density of the field being zero. The bracket is the
expected negative part of its derivative, which counts downward crossings
with their crossing speed. Integrating this rate gives an expected crossing
**count**, not a Poisson hazard. The sampler does not exponentiate this integral.

For neighboring active nodes, define
\(q(a,b)=\Pr(Y(a)>0,Y(b)\leq0)\), evaluated as a bivariate Gaussian probability.

Before computing the joint all-positive probability, threshold queries use
three inexpensive bounds. Here `activeNodes` contains the current endpoint
and any retained internal nodes, and all probabilities are conditioned on
positivity at the original ray origin by dividing by `p_start`.

First, survival implies that every individual active node is positive:

\[
\operatorname{survival}(t) \leq
\operatorname{marginalUpper}(t)=
\min\!\left(1,\min_{s\in\operatorname{activeNodes}}
\frac{\Phi(\operatorname{mean}(s)/\sigma)}{p_{\mathrm{start}}}\right).
\]

This costs only one-dimensional Gaussian CDFs. If the distance uniform is at
least this upper bound, survival is already excluded; there is no need to
evaluate the active-node orthant probability.

Second, a continuous path that starts positive and fails to survive must make
at least one downward crossing. The probability of that event cannot exceed
the unconditional expected number of downward crossings divided by the starting
positive probability. Therefore

\[
\operatorname{survival}(t) \geq
\operatorname{riceLower}(t)=
\max\!\left(0,1-
\frac{\int_0^t\operatorname{rate}_{\mathrm{down}}(s)\,ds}
{p_{\mathrm{start}}}\right).
\]

The integral reuses the flight's existing Rice cache. If the uniform is below
this lower bound, the prefix survives. This is especially useful when the
expected crossing count is small; a large count merely makes the bound zero.
The computed lower bound is retained through node insertions for that prefix,
since extra nodes do not change its Rice integral.

Third, survival implies that the origin and each considered node are both
positive. Their joint positive probability is `p_start - q(0,s)`, giving

\[
\operatorname{survival}(t) \leq
\operatorname{pairUpper}(t)=
1-\frac{\max_{s\in\operatorname{activeNodes}}q(0,s)}
{p_{\mathrm{start}}}.
\]

These are only two-dimensional probabilities and reuse the pair cache. They
can strengthen the marginal upper bound because they retain the origin
condition. There is no sum over node events here: those events overlap, so
only the largest single-node exclusion probability is used. If the uniform
is at least the upper bound, the first hit lies before the queried endpoint.
If none of these checks decides the threshold, the existing orthant and
continuous-gap refinement proceeds.

The implementation includes numerical error allowances in these checks:
`Rice estimate + Rice error` is used in the lower bound, and
`max(0, pair estimate - pair error)` in the pair upper bound. The resulting
probability bounds include a small roundoff allowance and are clipped to
`[0, 1]`. These allowances use the existing
numerical error estimates; they do not turn the bounds into certified
interval calculations.

The continuous missed-event allowance for that gap is

\[
\operatorname{gapError}(a,b)=
\int_a^b\operatorname{rate}_{\mathrm{down}}(s)\,ds-q(a,b).
\]

An endpoint sign change already detects at least one downcrossing; subtracting
its probability removes that detected part. The remainder bounds paths that
go negative and return between positive endpoints. If `nodeProbability` is the
joint all-positive probability at all active nodes, the exact-arithmetic bounds
are

\[
\max\!\left(0,\frac{\operatorname{nodeProbability}-
\sum\operatorname{gapError}}{p_{\mathrm{start}}}\right)
\leq \operatorname{survival}(t)
\leq \frac{\operatorname{nodeProbability}}{p_{\mathrm{start}}}.
\]

Inserting an interior candidate `c` reduces the raw missed-crossing allowance by

\[
\operatorname{gain}(c)=q(a,c)+q(c,b)-q(a,b).
\]

This equals the probability of either alternating sign pattern `+ - +` or
`- + -`. It cheaply ranks useful constraints. It is **not** the actual reduction
of the all-positive node probability. The latter is recomputed after insertion.

If the mean is at least `k * sigma` over an entire active gap, a separate bound is

\[
\Pr(\text{some field value in the gap is nonpositive})
\leq \Phi(-k)+\frac{b-a}{2\pi\ell}\exp(-k^2/2).
\]

The terms cover an initially low noise value and subsequent crossings of the
constant noise threshold. The code may take the minimum of this bound and the
whole-gap missed-event allowance. During Rice integration it instead uses an
upper bound on the expected crossing count; these two quantities are not
interchanged. Gaussian tails are budgeted rather than removed at a fixed SDF
threshold.

## Numerical status and current limits

The probability inequalities above are mathematical bounds. The implementation
uses **estimated numerical envelopes**, not certified interval integration:

- One- and two-dimensional probabilities use analytic formulas or adaptive
  Gauss–Kronrod integration. Zero-mean three-dimensional probability is analytic.
  Other dimensions use the Genz sequential truncated-normal transform with
  eight digitally shifted Sobol estimates. Replicate variation and a half-budget
  comparison estimate integration error.
- The SE factorization prioritizes restrictive conditional probabilities while
  excluding numerically unstable pivots. Constraints at least eight standard
  deviations into air may be omitted from the integral, with their Gaussian
  tail probabilities added to its error. Numerically dependent residuals are
  truncated with an additional error allowance; no diagonal jitter is added.
  The factorization error allowance is also a numerical estimate.
- Rice quadrature's error estimate is not a proof. Interpolation knots and the
  noise-scale step restriction protect against narrow missed crossing peaks.
- If the uniform remains inside an envelope whose width is already below
  `probability_tolerance`, the code compares it to the envelope midpoint. This
  explicitly makes the experiment a finite-accuracy sampler. It must not be
  described as an exact continuous GP first-passage sampler, nor is the requested
  tolerance a certified final image-error guarantee.
- Precheck can avoid refinement that would otherwise add persistent nodes.
  Later prefix queries can consequently use a different node history. Enabling
  it preserves the same survival law and tolerance policy, but does not promise
  bitwise identical approximate sample distances or images. Compare hit/miss
  classifications and distances, as well as workload and wall time, when
  measuring the change.

This implementation does not yet use rank-updated persistent factorizations,
lazy CDF updates after insertion, or macrocell traversal to avoid all distant
voxel boundaries. Those are performance improvements; candidate count and
retained CDF dimension are already separate. More importantly, collision-origin
conditioning, first-hit-conditioned gradients, complete path history, and
multiple-scattering transport remain outside this experiment.
