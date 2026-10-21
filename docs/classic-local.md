# Stationary classic_local

`classic_local` renders an existing NanoVDB `sdf` and `density` field using a
local Gaussian microfacet distribution. It supports the CPU path tracer,
including NEE/MIS and its resumable CPU wavefront path. No field is baked or
modified by this medium. The `obj2json_macrofacet` converter and its
`--emit-baked-json` option are unchanged.

## Scene configuration

```json
{
  "name": "gp",
  "type": "classic_local",
  "field": {
    "file": "scene.nvdb",
    "sdf_grid": "sdf",
    "density_grid": "density",
    "macrocell_size": 8
  },
  "covariance": {
    "type": "squared_exponential",
    "sigma": 0.001,
    "lengthScale": 0.0028284271
  },
  "material": {
    "type": "conductor",
    "eta": [0.2, 0.9, 1.1],
    "k": [3.9, 2.5, 2.2]
  },
  "max_bounces": 64
}
```

The covariance must be stationary and have positive finite field variance and
a positive-definite gradient covariance. Its mixed derivatives at zero are
evaluated once during render preparation. For an isotropic SE kernel,
`Cov(grad F) = sigma^2 / lengthScale^2 I`. With macrofacet's roughness convention,
`alpha = sqrt(2) * sigma / lengthScale`. The example corresponds to alpha 0.5.
The covariance is specified in scene world coordinates.

The baked density is authoritative: changing `sigma` in this JSON does not
rebake or recompute it. Use the sigma with which the file was generated and
adjust lengthScale to select the fixed roughness. This version does not read
sigma sidecars or alpha grids. An alpha channel already inside the file is
ignored; requesting `alpha_grid`/`use_alpha_grid`, or a non-stationary kernel,
is rejected. A stationary input passed to `--emit-baked-json` can produce a
constant variance grid inside its non-stationary wrapper, but this medium
does not use that GPIS scene representation.

`material.type` can also be `mirror`, with RGB or scalar `albedo`. Conductor
materials accept `force_unit_fresnel: true` for energy tests. The phase
function is fixed to the exact Gaussian target VNDF so that `sample`, `eval`
and `pdf` describe the same distribution.

## Field coordinates and domain

Both named grids must be float grids. Density must be finite and nonnegative,
with zero background. Values use trilinear interpolation; the SDF gradient is
the analytic gradient of that same interpolant, transformed into world space.
The active domain follows macrofacet's SDF grid bounds, from the minimum
index node to one node past the maximum. Transport outside it is vacuum.
Use a `forward` boundary primitive with `int_medium` set to the medium name;
the geometry supplies finite ray segments and can bound the domain further.
Forward cube exits inside the ray epsilon use the same boundary recovery as
the existing GP/classic modes, so the current medium is still updated.

The optional `field.transform` is an affine transform from the file's world
coordinates to scene world coordinates, using Tungsten's ordinary transform
JSON. For NVDB files exported with the converter's Z reflection, a typical
mapping back to the original scene is:

```json
"transform": {"scale": [1, 1, -1]}
```

Apply this only to data that has that reflection. Native macrofacet fields do
not need it. The loader honors the stored grid transform and does not repair
old baking sample-center offsets. Transforms place the field; stored density
values retain their world-distance units. This medium does not infer the
converter's band cutoff: it uses the zeros and values actually stored.

## Transport

At each query, the SDF mean gradient is normalized. An exactly zero gradient
stays zero, matching macrofacet's isotropic constant-mean case. The random
gradient retains all three covariance components. With travel direction w:

```text
meanG(x)   = normalize(grad sdf(x)), or zero
G(x)       ~ Normal(meanG(x), stationaryGradientCovariance)
area(x,w)  = E[max(-dot(w,G(x)), 0)]
sigmaT(x,w)= bakedDensity(x) * area(x,w)
```

Each bounce uses this local point distribution; there is no conditional birth
observation. Reflections use the flux-weighted Gaussian VNDF already used by
classic_global, with the normalized local mean instead of the raw SDF gradient.

Loading builds immutable density maxima over macrocells (default width 8 fine
cells, supported widths 1 through 64). Bounds include interpolation nodes and
a halo for rounded face queries. During a flight the maximum projected area
uses `negativePartMean(-1, sqrt(w^T C w))`, since the local mean norm is at most
one. This is a conservative, direction-dependent bound, tighter than the
orientation-independent `sqrt(1 + trace(C))` bound in macrofacet.

DDA supplies piecewise-constant extinction majorants. Delta tracking samples
collisions; ratio tracking estimates shadow transmittance with unbiased
roulette. Both reuse the exponential optical-depth threshold across intervals.
Candidates with zero density skip the SDF query. Every nonzero candidate is
checked against its majorant. Field handles and precomputed bounds are shared
read-only; traversal state and NanoVDB accessors are local to each query/ray.

This version requires positive-definite Gaussian gradient distributions. It
does not implement GGX, spatial roughness, BDPT distance PDFs or a CUDA kernel.

## Build and run

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_CLASSIC_LOCAL_TESTS=ON
cmake --build build --config Release --target tungsten classic_local_test --parallel 8
ctest --test-dir build -C Release -R '^classic_local' --output-on-failure
.\build\Release\tungsten.exe -t 8 -d build/classic-local-shaderball data/example-scenes/classic-local/shaderball.json
```

The shaderball example references the existing sibling asset
`macrofacet/outputs/fields/shader_ball_full.nvdb` (sigma 0.001) and retains the
camera/environment of `build/global-conditional-shaderball/shaderball.json`.
Change `field.file` to use another existing bake. The renderer has no code or
build dependency on the sibling macrofacet project.

The tests create tiny synthetic NVDB fixtures solely for validation. They
check grid interpolation and derivatives, DDA bounds, transformed fields,
fixed covariance despite a varying alpha channel, analytic free-flight and
transmittance probabilities, Gaussian VNDF distributions, JSON round trips,
unsupported inputs, and mixed-geometry white furnaces with/without NEE and
with CPU wavefront tracing. A separate furnace starts inside a forward cube
less than one ray epsilon from its exit to exercise boundary recovery.

## Validation

On 2026-10-21 the Release build passed all 21 `classic_local`, `classic_global`
and `global_conditional` CTest checks. The local furnace means were 1.00089
with NEE, 1.0 without NEE, 0.997425 with CPU wavefront, and 1.0 for the
sub-epsilon boundary exit. The existing shaderball NVDB rendered at 64x64,
16 spp and 8 CPU threads in 1.160 seconds of reported render time. Its PFM
contained finite, nonnegative values and the renderer reported no internal
errors. This is one smoke render, not a performance comparison with macrofacet.
Logs and measurements are in `build/classic-local-review/`.
