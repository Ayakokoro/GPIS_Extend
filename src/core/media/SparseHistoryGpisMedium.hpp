#pragma once

#include "ClassicGlobalMedium.hpp"
#include "math/SparseHistoryFlight.hpp"
#include "math/conditional/transport/FixedSpeedGradientDistribution.hpp"
#include "io/Path.hpp"
#include <atomic>
#include <cstdint>
#include <optional>

namespace Tungsten {

// Immutable snapshots are copied to visibility branches. A transparent surface
// advances age but does not renew the GP observation or the exponential clock.
struct SparseHistoryPathContext final : MediumPathContext {
    const Medium *owner = nullptr;
    SparseHistoryGpis::Start start;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Vec3f rayOrigin = Vec3f(0.f);
    double age = 0;
    std::optional<double> residualOpticalDepth;
};

struct SparseHistoryEventContext final : MediumEventContext {
    const Medium *owner = nullptr;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::optional<ConditionalGpis::FixedSpeedGradientDistribution> gradient;
};

class SparseHistoryGpisMedium final : public ClassicGlobalMedium {
    ConditionalGpis::GPSSField _field;
    SparseHistoryGpis::FlightSettings _settings;
    std::string _probabilityMethod="numeric";
    PathPtr _cdfTableFile;
    std::shared_ptr<PhaseFunction> _deltaPhase;
    mutable std::atomic<std::uint64_t> _discardedSurfacePaths{0};
    mutable std::atomic<std::uint64_t> _discardedSurfaceConnections{0};
    std::shared_ptr<const SparseHistoryPathContext> context(const Ray &,const MediumState &) const;
public:
    SparseHistoryGpisMedium();
    void fromJson(JsonPtr value,const Scene &scene) override;
    rapidjson::Value toJson(Allocator &) const override;
    void loadResources() override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &,const Ray &,bool,bool,MediumState *) const override;
    float pdf(PathSampleGenerator &,const Ray &,bool,bool) const override;
    Vec3f sigmaA(Vec3f) const override { return Vec3f(0.f); }
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f p) const override { return sigmaS(p); }
    const ConditionalGpis::GPSSField &field() const { return _field; }
    const SparseHistoryGpis::FlightSettings &flightSettings() const { return _settings; }
    std::uint64_t discardedSurfacePaths() const { return _discardedSurfacePaths.load(std::memory_order_relaxed); }
    std::uint64_t discardedSurfaceConnections() const { return _discardedSurfaceConnections.load(std::memory_order_relaxed); }
};

} // namespace Tungsten
