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
