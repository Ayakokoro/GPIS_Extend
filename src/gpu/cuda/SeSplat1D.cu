#include "cuda/Kernel1D.hpp"
#include "cuda/SeSplat1D.hpp"

#include <cmath>
#include <limits>

namespace Tungsten::Gpu {
namespace {

constexpr unsigned int threadsPerBlock = 256;

__global__ void seSplat1D(const float *queries, const float *centers,
                          float *values, float *derivatives,
                          std::size_t count, float localScale)
{
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= count)
        return;

    const SeSplat1DResult result = evaluateSeSplat1D(queries[index], centers[index], localScale);
    values[index] = result.value;
    derivatives[index] = result.derivative;
}

} // namespace

cudaError_t launchKernel1D(const Kernel1DBatch &batch, cudaStream_t stream)
{
    if (batch.kind != Kernel1DKind::SquaredExponentialSplat ||
        !std::isfinite(batch.localScale) || batch.localScale <= 0.0f)
        return cudaErrorInvalidValue;

    if (batch.count == 0)
        return cudaSuccess;

    if (!batch.queries || !batch.centers || !batch.values || !batch.derivatives ||
        batch.count > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) * threadsPerBlock)
        return cudaErrorInvalidValue;

    const unsigned int blocks = static_cast<unsigned int>((batch.count - 1) / threadsPerBlock + 1);
    seSplat1D<<<blocks, threadsPerBlock, 0, stream>>>(
        batch.queries, batch.centers, batch.values, batch.derivatives, batch.count, batch.localScale);
    return cudaGetLastError();
}

} // namespace Tungsten::Gpu
