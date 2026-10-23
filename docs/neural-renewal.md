# Neural causal Renewal medium

`neural_causal_renewal` loads the existing macrofacet `export-joint` JSON bundle
and original full-domain NanoVDB SDF. This adapter uses scalar CPU inference,
shared cubic ray profiles and full-gradient reconstruction. No LibTorch,
training, or converted OpenVDB field is required.

Build from this repository with the neighboring macrofacet checkout:

```powershell
cmake -S . -B build -DTUNGSTEN_ENABLE_NEURAL_RENEWAL=ON -DTUNGSTEN_BUILD_NEURAL_RENEWAL_TESTS=ON
cmake --build build --config Release --target tungsten neural_renewal_test --parallel 4
ctest --test-dir build -C Release -R neural_renewal --output-on-failure
New-Item -ItemType Directory -Force outputs/neural-bunny | Out-Null
.\build\Release\tungsten.exe gpis-scenes/bunny/bunny.json --spp 4 -t 8 -d outputs/neural-bunny
```

Set `TUNGSTEN_MACROFACET_SOURCE_DIR` when macrofacet is not at `../macrofacet`.
That checkout must include `CausalRenewalModel::sampleInterval` and
`NanoVdbMean::openSdf`. These provide bounded flights preserving GRU history and
an SDF-only import independent of baked statistical parameters. Keep the changes
in both repositories together.
The optional tests use that checkout's existing joint checkpoint and bunny SDF;
both must exist. Scene resource paths are relative to the scene file. The
medium is unavailable when the build option is off. Existing media and the
sparse-convolution CUDA backend retain their previous behavior.

The four CTest cases cover shared-sampler parity, finite geometry cutoffs, B
gradient state, forward-surface conditional survival, shadow-branch isolation,
manual sigma/lengthScale, JSON round trips, and white-furnace energy with direct
lighting on/off and the CPU path pool. Render tests use their own scene template.

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
environment light. A camera medium is sufficient: the adapter clips path and
shadow queries to the SDF active-node domain, treating outside space as vacuum.
Forward cubes with `int_medium` also work. Medium transforms are not supported;
use the baked field's world coordinates.

Materials are `conductor` (RGB `eta`, `k`) or `mirror` (`albedo`). The sampled
full GP gradient supplies the microscopic normal. Delta reflection applies
Fresnel once, without a second rough-conductor normal distribution.

Distance queries stop at ordinary geometry. A is a positive-exterior start;
B retains the exact collision position and full gradient. Geometry `nearT`
suppresses geometric self-intersections, without removing the start of a GP
flight. Forward surfaces preserve the flight and condition on survival. A
direction change at an ordinary surface starts a new A flight: this is an
explicit exterior-renewal approximation, not a sampled positive GP endpoint
observation. Ordinary surfaces are never treated as GP zeros.

Surface next-event estimation uses model transmittance
`exp(-(H(end)-H(begin)))`. Shadow branches replace immutable context pointers,
leaving the parent intact. GP collisions use a sampled-normal delta phase;
marginalized normal NEE/MIS and bidirectional distance PDFs are not implemented.
Use `path_tracer`. The example disables volume light sampling and enables
surface light sampling for ground shadows. If volume light sampling is enabled,
Tungsten's UNI estimator traces the phase-sampled delta direction to a light and
suppresses the corresponding continuation emission to avoid counting it twice.

The CPU adapter can run in Tungsten's resumable CPU path pool. Selecting
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
Because the neural adapter reads the native NVDB coordinates, this scene
transforms the camera, floor and light into those coordinates. The camera's
negative X scale preserves image handedness; the light's rotation preserves
its emitting side. The medium uses the camera-medium mode and its own active
domain instead of the original forward cube.

These local scene/grid files remain under the ignored `/gpis-scenes/` folder;
the model remains under ignored `/models/`.
