#ifndef TUNGSTEN_GPU_CUDA_RAY_SEGMENT_BATCH_HPP
#define TUNGSTEN_GPU_CUDA_RAY_SEGMENT_BATCH_HPP

#include <cstddef>
#include <cstdint>
#include <vector>

#include <cuda_runtime_api.h>

namespace Tungsten::Gpu {

struct Float3 {
    float x, y, z;
};

// Row-major 3x4 matrix, mapping Tungsten world coordinates to voxel coordinates.
struct Affine3x4 {
    float m[12];
};

struct HostMeanGrid {
    std::vector<float> voxels; // x varies fastest, then y, then z.
    int origin[3] = {0, 0, 0};
    int dimensions[3] = {0, 0, 0};
    Affine3x4 worldToIndex{};
    Float3 clampMin{0, 0, 0};
    Float3 clampMax{0, 0, 0};
    float offset = 0.0f;
    float scale = 1.0f;
};

struct Se1DRayParameters {
    float lengthScale = 1.0f;  // GP covariance lengthScale; noise scale is lengthScale / sqrt(2).
    float amplitude = 1.0f;    // GP sigma, applied after normalized sparse noise.
    float kernelRadius = 3.0f; // Same as StationaryCovariance::_kernelScale.
    float impulseDensity = 3.0f;
    float maxStepSize = 0.01f;
    unsigned int minSteps = 8;
    unsigned int maxSteps = 16384;
};

struct RaySegmentInput {
    Float3 origin;
    Float3 direction; // Unit-length, matching Tungsten::Ray::dir().
    float nearT;
    float farT;
    float jitter; // One sample from the caller's PathSampleGenerator, in [0, 1).
    std::uint32_t noiseSeed; // Existing SparseConvolutionNoiseRealization::computeSeed() result.
    std::uint32_t normalSeed; // xxhash32(pixelSampleSegment) + xxhash32(sceneSeed) + 1.
    std::uint32_t firstScatter; // Nonzero means there is no preceding GPIS event.
    float previousValue;
    Float3 previousGradient;
};

enum class RaySegmentStatus : std::uint32_t {
    Ok = 0,
    InvalidInput = 1,
    StepLimit = 2
};

struct RaySegmentOutput {
    float t;
    float value; // Zero at a hit; field value at farT on a miss.
    Float3 gradient; // Sampled at a hit; zero on a miss.
    std::uint32_t hit;
    std::uint32_t steps;
    RaySegmentStatus status;
};

struct DeviceMeanGrid {
    const float *voxels = nullptr;
    int origin[3];
    int dimensions[3];
    Affine3x4 worldToIndex;
    Float3 clampMin;
    Float3 clampMax;
    float offset;
    float scale;
};

// Low-level asynchronous API. Pointers and the grid must remain valid until the
// stream completes. One CUDA thread processes one complete ray segment.
cudaError_t launchSe1DRaySegments(const RaySegmentInput *deviceInputs,
                                  RaySegmentOutput *deviceOutputs,
                                  std::size_t count,
                                  const DeviceMeanGrid &deviceGrid,
                                  const Se1DRayParameters &parameters,
                                  cudaStream_t stream = nullptr);

// CPU-facing submission API. Owns and reuses GPU buffers; the grid is uploaded
// once by setMeanGrid(). trace() copies a whole batch and synchronizes once.
class Se1DRayBatcher {
    float *gridVoxels_ = nullptr;
    RaySegmentInput *inputs_ = nullptr;
    RaySegmentOutput *outputs_ = nullptr;
    std::size_t capacity_ = 0;
    DeviceMeanGrid grid_{};
    bool hasGrid_ = false;

public:
    Se1DRayBatcher() = default;
    ~Se1DRayBatcher();
    Se1DRayBatcher(const Se1DRayBatcher &) = delete;
    Se1DRayBatcher &operator=(const Se1DRayBatcher &) = delete;

    cudaError_t setMeanGrid(const HostMeanGrid &grid);
    const DeviceMeanGrid &deviceMeanGrid() const { return grid_; }
    // Call before asynchronous work starts. Buffers must not be resized while
    // earlier submissions on the same stream still use them.
    cudaError_t reserve(std::size_t count);
    // Enqueue copies, tracing and completion notification without waiting.
    // Host arrays must be pinned and remain valid until the callback runs.
    cudaError_t enqueue(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                        std::size_t count, const Se1DRayParameters &parameters,
                        cudaStreamCallback_t callback, void *userData,
                        cudaStream_t stream = nullptr);
    // Uses this slot's ray buffers with a borrowed, immutable medium grid.
    // The grid owner must outlive every submission using it.
    cudaError_t enqueueWithGrid(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                        std::size_t count, const DeviceMeanGrid &grid,
                        const Se1DRayParameters &parameters,
                        cudaStreamCallback_t callback, void *userData,
                        cudaStream_t stream);
    cudaError_t trace(const RaySegmentInput *inputs, RaySegmentOutput *outputs,
                      std::size_t count, const Se1DRayParameters &parameters,
                      cudaStream_t stream = nullptr);
};

} // namespace Tungsten::Gpu

#endif
