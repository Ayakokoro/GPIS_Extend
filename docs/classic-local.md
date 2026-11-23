# classic_local

`classic_local` renders a baked NanoVDB field using a local Gaussian
microfacet distribution. A stationary covariance supplies fixed roughness.
For a nonstationary source such as Dragon, the renderer reads the spatial
`alpha` channel used by the GP branch of `primitive_macrofacet`.
Both configurations support the CPU path tracer, NEE/MIS and CPU wavefront tracing.

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

The stationary configuration above requires positive finite field variance and
a positive-definite gradient covariance. Its mixed derivatives at zero are
evaluated once during render preparation. For an isotropic SE kernel,
`Cov(grad F) = sigma^2 / lengthScale^2 I`. With macrofacet's roughness convention,
`alpha = sqrt(2) * sigma / lengthScale`. The example corresponds to alpha 0.5.
The covariance is specified in scene world coordinates.

For spatial roughness, replace the medium's `covariance` entry with a named
float grid inside `field`, as in the Dragon example.

```json
"field": {
  "file": "dragon-classic-local.nvdb",
  "sdf_grid": "sdf",
  "density_grid": "density",
  "alpha_grid": "alpha",
  "transform": {"scale": [1, 1, -1]}
}
```

Exactly one roughness source is required. The renderer uses the density from
the bake in both configurations. On retained nodes the exporter writes
`alpha = max(1e-6, sqrt(2 * Var(dF/dx)))` from the source kernel at that
point. The scalar alpha describes an isotropic distribution at each position.
To change the nonstationary source kernel or
the density cutoff, update the source scene or bake options and regenerate the field.

`material.type` can also be `mirror`, with RGB or scalar `albedo`. Conductor
materials accept `force_unit_fresnel: true` for energy tests. The phase
function is fixed to the exact Gaussian target VNDF so that `sample`, `eval`
and `pdf` describe the same distribution.

## Field coordinates and domain

All named grids must be float grids. Density must be finite and nonnegative,
with zero background. Spatial alpha must be finite and nonnegative on density
support and use the same index transform as density. Values use trilinear
interpolation. The SDF gradient is
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

At each query the SDF gradient is normalized, with a zero gradient kept at
zero. For a unit travel direction `w` the local law is

```text
meanG(x)   = normalize(grad sdf(x)), or zero
G(x)       ~ Normal(meanG(x), C(x))
area(x,w)  = E[max(-dot(w,G(x)), 0)]
sigmaT(x,w)= bakedDensity(x) * area(x,w)
```

In stationary scenes `C` is the gradient covariance of the kernel. For
positive spatial alpha, first interpolate `alpha(x)`, then set
`s = alpha(x)/sqrt(2)` and `C(x) = s^2 I`. With `c = -dot(w, meanG)`, the
GP projected area in `primitive_macrofacet` is

```text
area = 0.5 * c * erfc(-c/alpha) + alpha/(2*sqrt(pi)) * exp(-(c/alpha)^2)
```

At tangency the area is `alpha/(2 sqrt(pi))`. At exactly zero alpha the GP implementation
assigns zero projected area and the point produces no collision.

Reflections sample the flux-weighted Gaussian visible-normal distribution.
For spatial alpha, the scaled gradient `G/s` has mean `meanG/s` and unit
covariance. Scaling preserves the normal and cancels in the normalized flux
weight, so visible-normal sampling and the phase PDF can both use the same
unit-covariance distribution at the collision point. The reference renderer
uses a Beckmann and uniform-hemisphere mixture to sample the same GP target.

Loading builds immutable density maxima over macrocells (default width 8 fine
cells, supported widths 1 through 64). Bounds include interpolation nodes and
a halo for rounded face queries. For stationary covariance the maximum area
along a ray is `negativePartMean(-1, sqrt(w^T C w))`.
For spatial alpha, loading precomputes each block's extinction bound
`rhoMax * negativePartMean(-1, alphaMax/sqrt(2))`, with zero for an empty
or all-zero-alpha block. The projected mean is at least -1, and the negative
part expectation increases with standard deviation. Independent corner maxima
therefore bound the product throughout the block even when their locations
differ. The `1.2 * rhoMax` multiplier in `primitive_macrofacet` can be below
the true extinction when the baked alpha is large.

DDA supplies piecewise-constant extinction majorants. Delta tracking samples
collisions; ratio tracking estimates shadow transmittance with unbiased
roulette. Both reuse the exponential optical-depth threshold across intervals.
Candidates with zero density skip the SDF query. Every nonzero candidate is
checked against its majorant. Field handles and precomputed bounds are shared
read-only; traversal state and NanoVDB accessors are local to each query/ray.

## Build and run

```powershell
cmake -S . -B build -DTUNGSTEN_BUILD_CLASSIC_LOCAL_TESTS=ON
cmake --build build --config Release --target tungsten classic_local_test obj2json_macrofacet --parallel 8
ctest --test-dir build -C Release -R '^classic_local' --output-on-failure
.\build\Release\tungsten.exe -t 8 -d build/classic-local-shaderball data/example-scenes/classic-local/shaderball.json
```

From the repository root, bake the original nonstationary Dragon and render
its 128×64, 4 spp preview. The bake uses 128 voxels along the longest axis.
The exporter writes density and alpha inside the source 3σ band. Runtime
transport follows the interpolated baked field, including its edge voxels.

```powershell
.\build\Release\obj2json_macrofacet.exe --format nvdb --resolution 128 --padding-sigma 3 gpis-scenes/dragon/dragon-function-space.json gpis-scenes/dragon/dragon-classic-local
.\build\Release\tungsten.exe -t 8 -d build/classic-local-dragon data/example-scenes/classic-local/dragon.json
```

The shaderball example references the existing sibling asset
`macrofacet/outputs/fields/shader_ball_full.nvdb` (sigma 0.001) and retains the
camera/environment of `build/global-conditional-shaderball/shaderball.json`.
Change `field.file` to use another existing bake. The renderer has no code or
build dependency on the sibling macrofacet project.

The tests create tiny synthetic NVDB fixtures solely for validation. They
check grid interpolation and derivatives, DDA bounds, transformed fields,
fixed covariance despite a varying alpha channel, spatial-alpha extinction
and visible-normal sampling, analytic free-flight and
transmittance probabilities, Gaussian VNDF distributions, JSON round trips,
unsupported inputs, and mixed-geometry white furnaces with/without NEE and
with CPU wavefront tracing. A separate furnace starts inside a forward cube
less than one ray epsilon from its exit to exercise boundary recovery.

## Validation

On 2026-11-24 the Dragon example rendered at 128x64 on 8 CPU threads with
seed 41. The renderer reported 0.216 s at 4 spp and 8.897 s at 128 spp.
All 8,192 RGB pixels were finite in both EXRs. At grid node (44, 12, 41)
the baked density was 1.891386 and alpha was 77.650703. The tangential
extinction was 41.430542 in both the renderer and the reference GP formula.
The traversed block's precomputed bound was 63.767235. A 16x16 white furnace
with spatial alpha from 0 to 16 at 128 spp produced mean radiance 0.996765
with NEE and 0.995106 with CPU wavefront tracing (reference 1).

On 2026-10-21 the Release build passed all 21 `classic_local`, `classic_global`
and `global_conditional` CTest checks. The local furnace means were 1.00089
with NEE, 1.0 without NEE, 0.997425 with CPU wavefront, and 1.0 for the
sub-epsilon boundary exit. The existing shaderball NVDB rendered at 64x64,
16 spp and 8 CPU threads in 1.160 seconds of reported render time. Its PFM
contained finite, nonnegative values and the renderer reported no internal
errors. This is one smoke render, not a performance comparison with macrofacet.
Logs and measurements are in `build/classic-local-review/`.
