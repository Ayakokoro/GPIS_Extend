#ifndef TUNGSTEN_GPU_CUDA_SE_SPLAT_1D_HPP
#define TUNGSTEN_GPU_CUDA_SE_SPLAT_1D_HPP

#include <cmath>

namespace Tungsten::Gpu {

struct SeSplat1DResult {
    float value;
    float derivative;
};

// Matches SquaredExponentialCovariance::splattingKernel1DVal/1stGrad.
// sigma is applied by the GP realization, outside the splatting kernel.
#if defined(__CUDACC__)
__host__ __device__
#endif
inline SeSplat1DResult evaluateSeSplat1D(float query, float center, float localScale)
{
    const float delta = query - center;
    const float inverseScaleSquared = 1.0f / (localScale * localScale);
    const float value = ::expf(-0.5f * delta * delta * inverseScaleSquared);
    return {value, -delta * inverseScaleSquared * value};
}

} // namespace Tungsten::Gpu

#endif
