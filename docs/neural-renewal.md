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
.\build\Release\tungsten.exe data/example-scenes/neural-renewal/bunny.json --spp 4 -t 8 -d outputs/neural-bunny
```

Set `TUNGSTEN_MACROFACET_SOURCE_DIR` when macrofacet is not at `../macrofacet`.
That checkout must include `CausalRenewalModel::sampleInterval`, added alongside
this adapter to stop flights at ordinary geometry while preserving their GRU
history. Keep the changes in both repositories together.
The optional tests use that checkout's existing joint checkpoint and bunny SDF;
both must exist. Scene resource paths are relative to the scene file. The
medium is unavailable when the build option is off. Existing media and the
sparse-convolution CUDA backend retain their previous behavior.

The four CTest cases cover shared-sampler parity, finite geometry cutoffs, B
gradient state, forward-surface conditional survival, shadow-branch isolation,
and white-furnace energy with direct lighting on/off and the CPU path pool.

The example contains the neural bunny, a two-triangle UV checker ground and an
environment light. A camera medium is sufficient: the adapter clips path and
shadow queries to the SDF active-node domain, treating outside space as vacuum.
Forward cubes with `int_medium` also work. Medium transforms are not supported;
use the baked field's world coordinates. Optional `field.sigma` must match the
bake. `material.roughness` sets SE gradient covariance to `roughness^2 / 2 I`;
alpha grids are not used.

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
