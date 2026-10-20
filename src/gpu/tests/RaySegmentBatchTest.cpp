#include "cuda/RaySegmentBatch.hpp"
#include "math/MathUtil.hpp"
#include "sampling/UniformSampler.hpp"

#include <cmath>
#include <chrono>
#include <future>
#include <iostream>
#include <vector>

namespace {

bool check(cudaError_t status, const char *label)
{
    if (status == cudaSuccess)
        return true;
    std::cerr << label << ": " << cudaGetErrorString(status) << '\n';
    return false;
}

void CUDART_CB completeAsyncBatch(cudaStream_t, cudaError_t status, void *userData)
{
    static_cast<std::promise<cudaError_t> *>(userData)->set_value(status);
}

Tungsten::Gpu::HostMeanGrid makePlaneGrid()
{
    Tungsten::Gpu::HostMeanGrid grid;
    grid.origin[0] = grid.origin[1] = grid.origin[2] = 0;
    grid.dimensions[0] = grid.dimensions[1] = grid.dimensions[2] = 8;
    grid.worldToIndex = {{1, 0, 0, 0,
                          0, 1, 0, 0,
                          0, 0, 1, 0}};
    grid.clampMin = {0, 0, 0};
    grid.clampMax = {7, 7, 7};
    grid.voxels.resize(8 * 8 * 8);
    for (int z = 0; z < 8; ++z)
        for (int y = 0; y < 8; ++y)
            for (int x = 0; x < 8; ++x)
                grid.voxels[(z * 8 + y) * 8 + x] = static_cast<float>(x) - 2.5f;
    return grid;
}

float cpuSeNoise(float coordinate, std::uint32_t seed, float radius, unsigned int impulses)
{
    const float gridCoordinate = coordinate / radius;
    const float cellFloor = std::floor(gridCoordinate);
    const int cell = static_cast<int>(cellFloor);
    const float fraction = gridCoordinate - cellFloor;
    float sum = 0;
    Tungsten::UniformSampler sampler;
    for (int offset = -1; offset <= 1; ++offset) {
        sampler.set_state(Tungsten::MathUtil::xxhash32(
            Tungsten::Vec2u(static_cast<unsigned int>(cell + offset), seed)) + 1U);
        for (unsigned int k = 0; k < impulses; ++k) {
            const float impulse = sampler.next1D();
            const float weight = sampler.next1D() < 0.5f ? -1.0f : 1.0f;
            const float delta = (fraction - offset - impulse) * radius;
            if ((fraction - offset - impulse) * (fraction - offset - impulse) < 1.0f)
                sum += weight * std::exp(-0.5f * delta * delta);
        }
    }
    constexpr float pi = 3.14159265358979323846f;
    return sum / std::sqrt(static_cast<float>(impulses) / radius * std::sqrt(pi));
}

} // namespace

int main()
{
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus == cudaErrorNoDevice || (deviceStatus == cudaSuccess && deviceCount == 0))
        return 77;
    if (!check(deviceStatus, "query devices"))
        return 1;

    Tungsten::Gpu::Se1DRayBatcher batcher;
    if (!check(batcher.setMeanGrid(makePlaneGrid()), "upload mean grid"))
        return 1;

    constexpr std::size_t count = 513; // Exercises more than one CUDA block.
    std::vector<Tungsten::Gpu::RaySegmentInput> inputs(count);
    std::vector<Tungsten::Gpu::RaySegmentOutput> outputs(count);
    for (std::size_t i = 0; i < count; ++i) {
        auto &ray = inputs[i];
        ray.origin = {0.0f, 3.0f, 3.0f};
        ray.direction = i % 2 == 0 ? Tungsten::Gpu::Float3{1, 0, 0}
                                     : Tungsten::Gpu::Float3{-1, 0, 0};
        ray.nearT = 0.0f;
        ray.farT = 5.0f;
        ray.jitter = 0.2f;
        ray.noiseSeed = static_cast<std::uint32_t>(123 + i);
        ray.normalSeed = static_cast<std::uint32_t>(456 + i);
        ray.firstScatter = 1;
    }

    Tungsten::Gpu::Se1DRayParameters parameters;
    parameters.amplitude = 0.0f;
    parameters.maxStepSize = 0.05f;
    cudaStream_t stream = nullptr;
    if (!check(cudaStreamCreate(&stream), "create stream"))
        return 1;
    bool ok = check(batcher.trace(inputs.data(), outputs.data(), count, parameters, stream),
                    "trace batch");
    if (ok) {
        for (std::size_t i = 0; i < count; ++i) {
            const auto &result = outputs[i];
            if (result.status != Tungsten::Gpu::RaySegmentStatus::Ok ||
                result.hit != (i % 2 == 0 ? 1U : 0U) ||
                (result.hit && (std::abs(result.t - 2.5f) > 0.01f ||
                                std::abs(result.gradient.x - 1.0f) > 0.002f)) ||
                (!result.hit && (result.t != 5.0f || std::abs(result.value + 2.5f) > 1e-5f))) {
                std::cerr << "Ray " << i << " has incorrect intersection\n";
                ok = false;
                break;
            }
        }
    }

    // Renewal+ must make the field continuous at the start of a new segment.
    inputs.resize(1);
    outputs.resize(1);
    inputs[0].origin = {3.0f, 3.0f, 3.0f};
    inputs[0].direction = {1, 0, 0};
    inputs[0].nearT = 0.0f;
    inputs[0].farT = 0.001f;
    inputs[0].jitter = 0.2f;
    inputs[0].noiseSeed = 123;
    inputs[0].normalSeed = 456;
    inputs[0].firstScatter = 0;
    inputs[0].previousValue = 0.9f;
    inputs[0].previousGradient = {1.2f, 0, 0};
    parameters.amplitude = 0.1f;
    if (ok)
        ok = check(batcher.trace(inputs.data(), outputs.data(), 1, parameters, stream),
                   "trace conditioned segment");
    if (ok && (outputs[0].status != Tungsten::Gpu::RaySegmentStatus::Ok ||
               outputs[0].hit || std::abs(outputs[0].value - 0.9f) > 0.005f)) {
        std::cerr << "Renewal+ value continuity failed\n";
        ok = false;
    }

    // Compare seeded sparse convolution on GPU with the CPU PCG/hash sequence.
    inputs.resize(73);
    outputs.resize(inputs.size());
    parameters.amplitude = 0.2f;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto &ray = inputs[i];
        ray.origin = {5.0f, 3.0f, 3.0f};
        ray.direction = {1, 0, 0};
        ray.nearT = 0;
        ray.farT = 0.001f;
        ray.jitter = 0.2f;
        ray.noiseSeed = static_cast<std::uint32_t>(123 + i);
        ray.normalSeed = 456;
        ray.firstScatter = 1;
    }
    if (ok)
        ok = check(batcher.trace(inputs.data(), outputs.data(), inputs.size(), parameters, stream),
                   "trace seeded sparse noise");
    if (ok) {
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const float coordinate = 5.001f / (parameters.lengthScale * 0.7071067811865475f);
            const float expected = 2.501f + parameters.amplitude *
                cpuSeNoise(coordinate, inputs[i].noiseSeed, parameters.kernelRadius, 3);
            if (outputs[i].status != Tungsten::Gpu::RaySegmentStatus::Ok || outputs[i].hit ||
                std::abs(outputs[i].value - expected) > 3e-5f) {
                std::cerr << "Seeded noise mismatch at ray " << i << ": CPU " << expected
                          << ", CUDA " << outputs[i].value << '\n';
                ok = false;
                break;
            }
        }
    }

    // Two batches may share device buffers on one stream without a host sync
    // between submissions; each pinned host result has its own completion.
    Tungsten::Gpu::RaySegmentInput *asyncInputs[2]{};
    Tungsten::Gpu::RaySegmentOutput *asyncOutputs[2]{};
    std::promise<cudaError_t> callbacks[2];
    auto completed0 = callbacks[0].get_future();
    auto completed1 = callbacks[1].get_future();
    if (ok)
        ok = check(batcher.reserve(2), "reserve async device buffers");
    for (int i = 0; i < 2 && ok; ++i) {
        ok = check(cudaMallocHost(reinterpret_cast<void **>(&asyncInputs[i]),
                                  sizeof(*asyncInputs[i])), "allocate pinned input") &&
             check(cudaMallocHost(reinterpret_cast<void **>(&asyncOutputs[i]),
                                  sizeof(*asyncOutputs[i])), "allocate pinned output");
    }
    if (ok) {
        asyncInputs[0][0] = inputs[0];
        asyncInputs[0][0].origin = {0, 3, 3};
        asyncInputs[0][0].farT = 5;
        asyncInputs[1][0] = inputs[0];
        asyncInputs[1][0].origin = {5, 3, 3};
        asyncInputs[1][0].farT = 0.001f;
        parameters.amplitude = 0;
        ok = check(batcher.enqueue(asyncInputs[0], asyncOutputs[0], 1, parameters,
                                   completeAsyncBatch, &callbacks[0], stream), "enqueue first batch") &&
             check(batcher.enqueue(asyncInputs[1], asyncOutputs[1], 1, parameters,
                                   completeAsyncBatch, &callbacks[1], stream), "enqueue second batch");
    }
    if (ok) {
        ok = completed0.wait_for(std::chrono::seconds(5)) == std::future_status::ready &&
             completed1.wait_for(std::chrono::seconds(5)) == std::future_status::ready &&
             completed0.get() == cudaSuccess && completed1.get() == cudaSuccess &&
             asyncOutputs[0][0].status == Tungsten::Gpu::RaySegmentStatus::Ok &&
             asyncOutputs[0][0].hit == 1 &&
             std::abs(asyncOutputs[0][0].t - 2.5f) < 0.01f &&
             asyncOutputs[1][0].status == Tungsten::Gpu::RaySegmentStatus::Ok &&
             asyncOutputs[1][0].hit == 0 &&
             std::abs(asyncOutputs[1][0].value - 2.501f) < 1e-5f;
        if (!ok)
            std::cerr << "Asynchronous batch completion failed\n";
    }
    cudaStreamSynchronize(stream);
    for (int i = 0; i < 2; ++i) {
        cudaFreeHost(asyncOutputs[i]);
        cudaFreeHost(asyncInputs[i]);
    }
    cudaStreamDestroy(stream);
    return ok ? 0 : 1;
}
