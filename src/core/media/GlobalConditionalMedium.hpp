#pragma once
#include "ClassicGlobalMedium.hpp"
#include "math/conditional/transport/ConditionalSegment.h"
#include <atomic>
#include <cstdint>

namespace Tungsten {
enum class ConditionalNonGpSurfacePolicy {
    RetainEndpoint,
    RestartExterior
};

struct ConditionalPathContext final : MediumPathContext {
    const Medium *owner = nullptr;
    ConditionalGpis::FlightState birth;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Vec3f rayOrigin;
};
struct ConditionalEventContext final : MediumEventContext {
    const Medium *owner = nullptr;
    Eigen::Vector3d position;
    ConditionalGpis::Gaussian<3> gradient;
};

// Uses the same SDF and material loading as classic_global. Extinction and
// scattering instead depend on the latest full (F, gradient F) observation,
// renewed at every GP collision and geometry endpoint. Ordinary non-GP
// scattering can optionally restart this observation from the exterior prior.
class GlobalConditionalMedium final : public ClassicGlobalMedium {
    ConditionalGpis::GPSSField _field;
    ConditionalGpis::ConditionalTrackingSettings _tracking;
    ConditionalNonGpSurfacePolicy _nonGpSurfacePolicy = ConditionalNonGpSurfacePolicy::RetainEndpoint;
    mutable std::atomic<std::uint64_t> _discardedSurfacePaths{0};
    mutable std::atomic<std::uint64_t> _discardedSurfaceConnections{0};
    std::shared_ptr<const ConditionalPathContext> context(
        PathSampleGenerator &,const Ray &,const MediumState &) const;
public:
    GlobalConditionalMedium();
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &) const override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    // Called once at a non-forward geometry event, before direct-light and
    // continuation branches copy the state. Does not reset medium counters.
    void prepareNonGpSurface(PathSampleGenerator &,const Vec3f &position,
        const Vec3f &incomingDirection,MediumState &) const;
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &,const Ray &,bool,bool,MediumState *) const override;
    float pdf(PathSampleGenerator &,const Ray &,bool,bool) const override;
    Vec3f sigmaA(Vec3f) const override { return Vec3f(0.0f); }
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f) const override;
    const ConditionalGpis::GPSSField &field() const { return _field; }
    const ConditionalGpis::ConditionalTrackingSettings &trackingSettings() const { return _tracking; }
    ConditionalNonGpSurfacePolicy nonGpSurfacePolicy() const { return _nonGpSurfacePolicy; }
    std::uint64_t discardedSurfacePaths() const { return _discardedSurfacePaths.load(std::memory_order_relaxed); }
    std::uint64_t discardedSurfaceConnections() const { return _discardedSurfaceConnections.load(std::memory_order_relaxed); }
};
}
