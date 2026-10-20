#ifndef TUNGSTEN_GPU_CUDA_KERNEL1D_HPP
#define TUNGSTEN_GPU_CUDA_KERNEL1D_HPP

#include <cstddef>

#include <cuda_runtime_api.h>

namespace Tungsten::Gpu {

// Add future 1D kernel families here; dispatch stays on the host.
enum class Kernel1DKind {
    SquaredExponentialSplat
};

// All pointers refer to device memory. The caller owns buffers and synchronization.
struct Kernel1DBatch {
    Kernel1DKind kind = Kernel1DKind::SquaredExponentialSplat;
    const float *queries = nullptr;
    const float *centers = nullptr;
    float *values = nullptr;
    float *derivatives = nullptr;
    std::size_t count = 0;
    float localScale = 1.0f;
};

// Asynchronous launch on stream. The derivative is with respect to query position.
// Both output arrays must be supplied. A zero-length batch is a successful no-op.
cudaError_t launchKernel1D(const Kernel1DBatch &batch, cudaStream_t stream = nullptr);

} // namespace Tungsten::Gpu

#endif
