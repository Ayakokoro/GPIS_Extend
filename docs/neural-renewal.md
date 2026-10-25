# Neural causal Renewal medium

`neural_causal_renewal` is a native Tungsten medium that directly loads the same
`export-joint` model JSON and full-domain NanoVDB SDF used by macrofacet. It
implements the same float32 MLP/GRU inference, cubic ray profiles, and
full-gradient reconstruction inside `core`. No retraining, LibTorch, or
converted OpenVDB field is required. Neural rendering is enabled by default.

Build entirely from this repository, with the normal Tungsten dependencies
installed (see [the build instructions](../README.md#building-without-macrofacet)):

```powershell
cmake -S . -B build -DTUNGSTEN_ENABLE_NEURAL_RENEWAL=ON -DTUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS=OFF
cmake --build build --config Release --target tungsten --parallel 4
```

The native implementation consists of:

- `src/core/math/neural/CausalRenewalModel.{hpp,cpp}`: validates and loads the
  exported JSON, evaluates the MLPs and PyTorch-compatible GRU, integrates the
  corrected hazard, and samples a positive crossing speed.
- `src/core/math/neural/RayMeanProfile.{hpp,cpp}`: recovers the exact ray cubic
  in each trilinear SDF cell.
- `src/core/math/neural/RenewalSampler.{hpp,cpp}`: constructs A/B flights and
  reconstructs full collision gradients using the existing covariance and
  Gaussian utilities in `math/conditional`.
- `src/core/media/NeuralSdfGrid.{hpp,cpp}`: reads only the full-domain SDF and
  evaluates its interpolant and one-sided gradient.

These files are built directly into `core`, with no separate macrofacet
library, include paths, backend declarations, field factories, or baking code.
The `macrofacet.causal_renewal` format string in the JSON reader is retained
for model compatibility. Network weights are shared and immutable; each ray
query owns its recurrent state. The arithmetic follows the macrofacet working
tree based on commit `f8b6514cbcc49772760be579a60bd221b982caf8`, including its
then-uncommitted bounded-flight sampling and SDF-only loading changes, ported
on 2026-10-23. GPU inference is not part of this CPU implementation.

Neural builds require NanoVDB and nlohmann_json in addition to the normal
Tungsten dependencies. Compilation does not require model weights, SDF assets,
LibTorch, or a CUDA toolkit. `TUNGSTEN_MACROFACET_SOURCE_DIR` is not used.

Rendering still needs an exported joint JSON model and a full-domain NanoVDB
SDF. Scene resource paths are relative to the scene file. The medium is
unavailable when `TUNGSTEN_ENABLE_NEURAL_RENEWAL` is off.

The optional tests default to this repository's local assets:

- `models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json`
- `gpis-scenes/bunny/bunny-64-sigma005.nvdb`

These folders are gitignored, so a fresh clone does not include the test data.
After placing the files there, enable and run the tests:

```powershell
cmake -S . -B build -DTUNGSTEN_ENABLE_NEURAL_RENEWAL=ON -DTUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS=ON
cmake --build build --config Release --target tungsten neural_renewal_test --parallel 4
ctest --test-dir build -C Release -R neural_renewal --output-on-failure
```

For another asset location, set `TUNGSTEN_NEURAL_RENEWAL_TEST_MODEL` and
`TUNGSTEN_NEURAL_RENEWAL_TEST_FIELD` to the respective file paths during CMake
configuration. The field must be the bunny fixture in its original coordinates
because the test rays and camera use that geometry. Missing files produce a
configuration error only when these optional tests are enabled.

The CTest cases cover medium/sampler consistency, finite geometry cutoffs, B
gradient state, forward-surface conditional survival, shadow-branch isolation,
manual sigma/lengthScale, JSON round trips, direction-PDF normalization and
NEE/MIS agreement, and white-furnace energy with direct lighting on/off and the
CPU path pool. Render tests use their own scene template.

Configure the stationary covariance just like `classic_local`:

```json
"field": {"file": "knob-l0.02-s0.02-dx0.01.nvdb"},
"covariance": {
  "type": "squared_exponential",
  "sigma": 0.02,
  "lengthScale": 0.02
},
"material": {"type": "mirror", "albedo": [0.15, 0.8, 0.5]}
```

Both parameters are required positive finite scalars. The existing neural model
supports the stationary isotropic SE kernel. Its covariance is
`sigma^2 exp(-|x-y|^2 / (2 lengthScale^2))`, so the gradient covariance is
`sigma^2 / lengthScale^2 I`. `material` controls reflection and color only;
`material.roughness` and `field.sigma` are rejected with migration instructions.

Only the `sdf` grid is loaded. Sigma, alpha and density grids and `.nvdb.json`
sidecars are ignored. Changing the covariance does not require rebaking the SDF
or adding a sigma grid. The loader verifies full active-node coverage directly,
and ray queries still check that interpolation nodes are present and finite.

The example contains the neural bunny, a two-triangle UV checker ground and an
environment light. A camera medium is sufficient: the medium clips path and
shadow queries to the SDF active-node domain, treating outside space as vacuum.
Forward cubes with `int_medium` also work. Medium transforms are not supported;
use the baked field's world coordinates.

With the local bunny scene and its assets available:

```powershell
New-Item -ItemType Directory -Force outputs/neural-bunny | Out-Null
.\build\Release\tungsten.exe gpis-scenes/bunny/bunny.json --spp 4 -t 8 -d outputs/neural-bunny
```

Materials are `conductor` (RGB `eta`, `k`) or `mirror` (`albedo`). The full GP
gradient supplies the microscopic normal. Each sampled normal reflects
specularly and applies Fresnel once, without a second rough-conductor normal
distribution. The phase integrates over the remaining uncertain normal.

Distance queries stop at ordinary geometry. A is a positive-exterior start;
B retains the exact collision position and full gradient. Geometry `nearT`
suppresses geometric self-intersections, without removing the start of a GP
flight. Forward surfaces preserve the flight and condition on survival. A
direction change at an ordinary surface starts a new A flight: this is an
explicit exterior-renewal approximation, not a sampled positive GP endpoint
observation. Ordinary surfaces are never treated as GP zeros.

Surface and GP-collision next-event estimation use model transmittance
`exp(-(H(end)-H(begin)))`. Shadow branches replace immutable context pointers,
leaving the parent intact. Use `path_tracer`; bidirectional distance PDFs remain
unsupported. Set `enable_volume_light_sampling: true` for GP NEE and MIS;
`enable_light_sampling` controls ordinary surface direct lighting. Older scene
files that explicitly disable volume light sampling must enable it to use GP NEE.

At a learned collision, the network samples distance and the inward directional
derivative `v = dot(wi,g) < 0`. Both direct-light strategies condition on this
same `v`. The remaining two transverse gradient components have the existing
Gaussian reconstruction law, including the previous full gradient for B starts.
Phase sampling draws those components when a branch is created. A light
direction instead determines the full gradient:

```
h = normalize(wo - wi)
g = (v / dot(wi,h)) * h
p(wo | v, collision, history) = p_perp(g_perp) * v^2 / (4 * abs(dot(wi,h))^4)
```

The Jacobian combines the fixed-derivative plane-to-normal mapping with the
[reflection-direction Jacobian](https://pbr-book.org/4ed/Reflection_Models/Roughness_Using_Microfacet_Theory).
There is no additional Rice/flux factor: the network already samples the
collision-conditioned speed. Its density is common to both strategies and
cancels in their MIS ratio. This integrates the transverse Gaussian, without
replacing the learned speed mixture with a Gaussian or retraining the model.
Each successor retains its complete sampled/reconstructed gradient for B
transport. Exactly zero transverse variance keeps a delta phase; all positive
variances use the continuous direction PDF and Tungsten's power heuristic.
Sampled PDFs are evaluated from the double-precision gradient before storing
the float ray direction; very narrow distributions would otherwise lose their
own samples' density to rounding. MIS squares finite float PDFs in double.

The neural medium can run in Tungsten's resumable CPU path pool. Selecting
`backend: "cuda_wavefront"` is rejected: integration of the neural CUDA sampler
with `GpRayBatchBackend` is separate work. Image noise differs from macrofacet
because Tungsten owns the camera and path random streams.

## Existing shaderball knob bake

The local scene `gpis-scenes/shaderball/knob-neural-l0.02-s0.02.json` uses
`knob-l0.02-s0.02-dx0.01.nvdb` and the exported model under
`models/renewal_shader_ball_sigma002_r05_joint/renewal_model.json`.
Run from the Tungsten repository root:

```powershell
New-Item -ItemType Directory -Force results/knob-neural-l0.02-s0.02 | Out-Null
.\build\Release\tungsten.exe gpis-scenes/shaderball/knob-neural-l0.02-s0.02.json -t 8 -s 17429 -d results/knob-neural-l0.02-s0.02
```

Defaults are 128 x 128, 8 spp, CPU inference with the CPU path pool. Edit
`camera.resolution` and `renderer.spp` for a larger render. The green mirror
albedo, floor and lighting come from the classic-local knob scene.

The bake contains no sigma grid, which is now sufficient: the scene explicitly
sets `covariance.sigma: 0.02` and `covariance.lengthScale: 0.02`. The previously
created `.nvdb.json` is no longer needed by neural rendering. All 300 x 300 x 300
SDF nodes are active; no grid samples were changed. The bunny scene likewise
uses explicit covariance (sigma 0.05, lengthScale 0.14142135623730953) to preserve
its previous statistical scale.

The original bake's placement includes a Z reflection and half-voxel shift.
Because the neural medium reads the native NVDB coordinates, this scene
transforms the camera, floor and light into those coordinates. The camera's
negative X scale preserves image handedness; the light's rotation preserves
its emitting side. The medium uses the camera-medium mode and its own active
domain instead of the original forward cube.

These local scene/grid files remain under the ignored `/gpis-scenes/` folder;
the model remains under ignored `/models/`.
