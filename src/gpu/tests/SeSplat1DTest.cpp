#include "cuda/Kernel1D.hpp"

#include <cmath>
#include <cstddef>
#include <iostream>
#include <vector>

namespace {

bool check(cudaError_t status, const char *operation)
{
    if (status == cudaSuccess)
        return true;
    std::cerr << operation << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

} // namespace

int main()
{
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus == cudaErrorNoDevice || (deviceStatus == cudaSuccess && deviceCount == 0)) {
        std::cout << "No CUDA device; skipping test\n";
        return 77;
    }
    if (!check(deviceStatus, "query CUDA devices"))
        return 1;

    constexpr std::size_t count = 1025; // Crosses several CUDA blocks.
    constexpr float scale = 0.7f;
    std::vector<float> queries(count), centers(count), values(count), derivatives(count);
    for (std::size_t i = 0; i < count; ++i) {
        queries[i] = static_cast<float>(static_cast<int>(i % 31) - 15) * 0.1f;
        centers[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.2f;
    }
    queries[0] = centers[0];

    float *deviceQueries = nullptr, *deviceCenters = nullptr;
    float *deviceValues = nullptr, *deviceDerivatives = nullptr;
    const std::size_t bytes = count * sizeof(float);
    bool ok = check(cudaMalloc(reinterpret_cast<void **>(&deviceQueries), bytes), "allocate queries") &&
              check(cudaMalloc(reinterpret_cast<void **>(&deviceCenters), bytes), "allocate centers") &&
              check(cudaMalloc(reinterpret_cast<void **>(&deviceValues), bytes), "allocate values") &&
              check(cudaMalloc(reinterpret_cast<void **>(&deviceDerivatives), bytes), "allocate derivatives");

    if (ok) {
        ok = check(cudaMemcpy(deviceQueries, queries.data(), bytes, cudaMemcpyHostToDevice), "copy queries") &&
             check(cudaMemcpy(deviceCenters, centers.data(), bytes, cudaMemcpyHostToDevice), "copy centers");
    }

    Tungsten::Gpu::Kernel1DBatch batch;
    batch.queries = deviceQueries;
    batch.centers = deviceCenters;
    batch.values = deviceValues;
    batch.derivatives = deviceDerivatives;
    batch.count = count;
    batch.localScale = scale;

    cudaStream_t stream = nullptr;
    if (ok)
        ok = check(cudaStreamCreate(&stream), "create stream");

    if (ok) {
        ok = check(Tungsten::Gpu::launchKernel1D(batch, stream), "launch SE kernel") &&
             check(cudaStreamSynchronize(stream), "synchronize SE kernel") &&
             check(cudaMemcpy(values.data(), deviceValues, bytes, cudaMemcpyDeviceToHost), "copy values") &&
             check(cudaMemcpy(derivatives.data(), deviceDerivatives, bytes, cudaMemcpyDeviceToHost), "copy derivatives");
    }

    if (ok) {
        for (std::size_t i = 0; i < count; ++i) {
            const double delta = static_cast<double>(queries[i]) - centers[i];
            const double expectedValue = std::exp(-delta * delta / (2.0 * scale * scale));
            const double expectedDerivative = -delta / (scale * scale) * expectedValue;
            if (std::abs(values[i] - expectedValue) > 2e-6 ||
                std::abs(derivatives[i] - expectedDerivative) > 2e-6) {
                std::cerr << "SE mismatch at index " << i << '\n';
                ok = false;
                break;
            }
        }
    }

    batch.localScale = 0.0f;
    if (Tungsten::Gpu::launchKernel1D(batch) != cudaErrorInvalidValue) {
        std::cerr << "Invalid scale was accepted\n";
        ok = false;
    }
    batch.localScale = scale;
    batch.count = 0;
    batch.queries = nullptr;
    if (Tungsten::Gpu::launchKernel1D(batch) != cudaSuccess) {
        std::cerr << "Empty batch was rejected\n";
        ok = false;
    }

    if (stream)
        cudaStreamDestroy(stream);
    cudaFree(deviceDerivatives);
    cudaFree(deviceValues);
    cudaFree(deviceCenters);
    cudaFree(deviceQueries);
    return ok ? 0 : 1;
}
