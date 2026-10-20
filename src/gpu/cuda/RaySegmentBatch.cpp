#include "cuda/RaySegmentBatch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Tungsten::Gpu {

Se1DRayBatcher::~Se1DRayBatcher()
{
    cudaFree(outputs_);
    cudaFree(inputs_);
    cudaFree(gridVoxels_);
}

cudaError_t Se1DRayBatcher::setMeanGrid(const HostMeanGrid &grid)
{
    std::size_t expected = 1;
    const float clampMin[3] = {grid.clampMin.x, grid.clampMin.y, grid.clampMin.z};
    const float clampMax[3] = {grid.clampMax.x, grid.clampMax.y, grid.clampMax.z};
    for (int axis = 0; axis < 3; ++axis) {
        if (grid.dimensions[axis] < 2 ||
            expected > std::numeric_limits<std::size_t>::max() /
                       static_cast<std::size_t>(grid.dimensions[axis]) ||
            !std::isfinite(clampMin[axis]) ||
            !std::isfinite(clampMax[axis]) ||
            clampMin[axis] > clampMax[axis])
            return cudaErrorInvalidValue;
        expected *= static_cast<std::size_t>(grid.dimensions[axis]);
    }
    if (grid.voxels.size() != expected ||
        expected > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        !std::isfinite(grid.offset) || !std::isfinite(grid.scale))
        return cudaErrorInvalidValue;
    for (float element : grid.worldToIndex.m)
        if (!std::isfinite(element))
            return cudaErrorInvalidValue;

    float *newVoxels = nullptr;
    const std::size_t bytes = expected * sizeof(float);
    cudaError_t result = cudaMalloc(reinterpret_cast<void **>(&newVoxels), bytes);
    if (result != cudaSuccess)
        return result;
    result = cudaMemcpy(newVoxels, grid.voxels.data(), bytes, cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        cudaFree(newVoxels);
        return result;
    }

    cudaFree(gridVoxels_);
    gridVoxels_ = newVoxels;
    grid_.voxels = gridVoxels_;
    std::copy_n(grid.origin, 3, grid_.origin);
    std::copy_n(grid.dimensions, 3, grid_.dimensions);
    grid_.worldToIndex = grid.worldToIndex;
    grid_.clampMin = grid.clampMin;
    grid_.clampMax = grid.clampMax;
    grid_.offset = grid.offset;
    grid_.scale = grid.scale;
    hasGrid_ = true;
    return cudaSuccess;
}

cudaError_t Se1DRayBatcher::reserve(std::size_t count)
{
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(RaySegmentInput) ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(RaySegmentOutput))
        return cudaErrorInvalidValue;

    if (count > capacity_) {
        RaySegmentInput *newInputs = nullptr;
        RaySegmentOutput *newOutputs = nullptr;
        cudaError_t result = cudaMalloc(reinterpret_cast<void **>(&newInputs), count * sizeof(RaySegmentInput));
        if (result != cudaSuccess)
            return result;
        result = cudaMalloc(reinterpret_cast<void **>(&newOutputs), count * sizeof(RaySegmentOutput));
        if (result != cudaSuccess) {
            cudaFree(newInputs);
            return result;
        }
        cudaFree(inputs_);
        cudaFree(outputs_);
        inputs_ = newInputs;
        outputs_ = newOutputs;
        capacity_ = count;
    }
    return cudaSuccess;
}

cudaError_t Se1DRayBatcher::enqueue(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                                    std::size_t count, const Se1DRayParameters &parameters,
                                    cudaStreamCallback_t callback, void *userData,
                                    cudaStream_t stream)
{
    if (!hasGrid_)
        return cudaErrorInvalidValue;
    return enqueueWithGrid(inputs, outputs, count, grid_, parameters, callback, userData, stream);
}

cudaError_t Se1DRayBatcher::enqueueWithGrid(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                                    std::size_t count, const DeviceMeanGrid &grid,
                                    const Se1DRayParameters &parameters,
                                    cudaStreamCallback_t callback, void *userData,
                                    cudaStream_t stream)
{
    if (!grid.voxels || !inputs || !outputs || !callback || count == 0 || count > capacity_)
        return cudaErrorInvalidValue;

    cudaError_t result = cudaMemcpyAsync(inputs_, inputs, count * sizeof(RaySegmentInput),
                                          cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess)
        return result;
    result = launchSe1DRaySegments(inputs_, outputs_, count, grid, parameters, stream);
    if (result != cudaSuccess)
        return result;
    result = cudaMemcpyAsync(outputs, outputs_, count * sizeof(RaySegmentOutput),
                             cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess)
        return result;
    return cudaStreamAddCallback(stream, callback, userData, 0);
}

cudaError_t Se1DRayBatcher::trace(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                                  std::size_t count, const Se1DRayParameters &parameters,
                                  cudaStream_t stream)
{
    if (!hasGrid_ || (count && (!inputs || !outputs)))
        return cudaErrorInvalidValue;
    if (count == 0)
        return cudaSuccess;
    cudaError_t result = reserve(count);
    if (result != cudaSuccess)
        return result;
    result = cudaMemcpyAsync(inputs_, inputs, count * sizeof(RaySegmentInput),
                             cudaMemcpyHostToDevice, stream);
    if (result != cudaSuccess)
        return result;
    result = launchSe1DRaySegments(inputs_, outputs_, count, grid_, parameters, stream);
    if (result != cudaSuccess) {
        cudaStreamSynchronize(stream);
        return result;
    }
    result = cudaMemcpyAsync(outputs, outputs_, count * sizeof(RaySegmentOutput),
                             cudaMemcpyDeviceToHost, stream);
    if (result != cudaSuccess)
        return result;
    return cudaStreamSynchronize(stream);
}

} // namespace Tungsten::Gpu
