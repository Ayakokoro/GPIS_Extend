#ifndef GP_RAY_BATCH_BACKEND_HPP_
#define GP_RAY_BATCH_BACKEND_HPP_

#include "math/Vec.hpp"

#include <cstdint>
#include <future>
#include <functional>
#include <vector>

namespace Tungsten {

class SparseConvolutionNoiseMedium;

struct GpRaySegmentRequest {
    enum class Purpose { Path, Shadow };
    Purpose purpose = Purpose::Path;
    const SparseConvolutionNoiseMedium *medium = nullptr;
    Vec3f origin;
    Vec3f direction;
    float nearT = 0.0f;
    float farT = 0.0f;
    float jitter = 0.0f;
    std::uint32_t noiseSeed = 0;
    std::uint32_t normalSeed = 0;
    bool firstScatter = true;
    float previousValue = 0.0f;
    Vec3f previousGradient = Vec3f(0.0f);
};

struct GpRaySegmentResult {
    float t = 0.0f;
    float value = 0.0f;
    bool hit = false;
};

struct GpRaySegmentReply {
    bool valid = false;
    GpRaySegmentResult result;
};

// A failed submission leaves the caller free to trace the same segment on CPU.
class GpRayBatchBackend {
public:
    virtual ~GpRayBatchBackend() = default;
    virtual bool trace(const GpRaySegmentRequest &request,
                       GpRaySegmentResult &result) = 0;
    // Requests are copied and retained until all replies are published.
    // Reply order matches request order, including unsupported/failed segments.
    // onComplete runs once after publication, possibly inline or on a CUDA
    // callback thread. Use it only to notify a scheduler, without CUDA calls.
    virtual std::future<std::vector<GpRaySegmentReply>> submitBatch(
        const std::vector<GpRaySegmentRequest> &requests,
        std::function<void()> onComplete = {}) = 0;
};

} // namespace Tungsten

#endif
