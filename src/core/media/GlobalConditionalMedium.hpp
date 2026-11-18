#pragma once
#include "ClassicGlobalMedium.hpp"
#include "ClassicLocalGrid.hpp"
#include "math/FirstPassageGradient.hpp"
#include "math/conditional/transport/ConditionalSegment.h"
#include "math/conditional/transport/FixedSpeedGradientDistribution.hpp"
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
    // Experimental axial first-passage observation, shared by both MIS strategies.
    double derivative = 0;
    std::optional<ConditionalGpis::FixedSpeedGradientDistribution> correctedGradient;
};

// Uses the same SDF and material loading as classic_global. Extinction and
// scattering instead depend on the latest full (F, gradient F) observation,
// renewed at every GP collision and geometry endpoint. Ordinary non-GP
// scattering can optionally restart this observation from the exterior prior.
class GlobalConditionalMedium final : public ClassicGlobalMedium {
    ConditionalGpis::GPSSField _field;
    ConditionalGpis::ConditionalTrackingSettings _tracking;
    double _collisionStartStep = 0.0;
    bool _jitterCollisionStart = true;
    std::optional<ClassicLocalField> _bandField;
    std::unique_ptr<ClassicLocalGrid> _bandGrid;
    ConditionalNonGpSurfacePolicy _nonGpSurfacePolicy = ConditionalNonGpSurfacePolicy::RetainEndpoint;
    bool _fpGradient = false, _bridgeCorrection = true;
    // 1 samples both direct proposals, 0 samples the light alone. Larger
    // periods keep both proposals in one path sample per group of this size.
    uint32 _directPhaseSamplePeriod = 1;
    bool _binaryShadowVisibility = false;
    FirstPassageGradient::AnchorSelection _anchorSelection = FirstPassageGradient::AnchorSelection::Fixed5Snr;
    std::shared_ptr<PhaseFunction> _deltaPhase;
    mutable std::atomic<std::uint64_t> _correctedCollisions{0}, _gradientFallbacks{0};
    mutable std::atomic<std::uint64_t> _discardedSurfacePaths{0};
    mutable std::atomic<std::uint64_t> _discardedSurfaceConnections{0};
    bool _singlePointNeural = false;
    PathPtr _singlePointModelFile;
    std::optional<double> _singlePointCap;
    std::shared_ptr<const ConditionalGpis::SinglePointNeuralModel> _singlePointModel;
    mutable std::atomic<std::uint64_t> _networkQueries{0}, _correctionCapHits{0};
    mutable std::atomic<std::uint64_t> _neuralOutsideDomain{0}, _neuralBaselineFallbacks{0};
    mutable std::atomic<std::uint64_t> _neuralOutsideFeatureRange{0};
    void recordNeuralWork(const ConditionalGpis::ConditionalTrackingWork &) const;
    std::shared_ptr<const ConditionalPathContext> context(
        PathSampleGenerator &,const Ray &,const MediumState &) const;
public:
    explicit GlobalConditionalMedium(bool singlePointNeural = false);
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &) const override;
    void loadResources() override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    // Called once at a non-forward geometry event, before direct-light and
    // continuation branches copy the state. Does not reset medium counters.
    void prepareNonGpSurface(PathSampleGenerator &,const Vec3f &position,
        const Vec3f &incomingDirection,MediumState &) const;
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &,const Ray &,bool,bool,MediumState *) const override;
    Vec3f shadowTransmittance(PathSampleGenerator &,const Ray &,MediumState *,bool retainEndpoint) const;
    float pdf(PathSampleGenerator &,const Ray &,bool,bool) const override;
    Vec3f sigmaA(Vec3f) const override { return Vec3f(0.0f); }
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f) const override;
    const ConditionalGpis::GPSSField &field() const { return _field; }
    const ConditionalGpis::ConditionalTrackingSettings &trackingSettings() const { return _tracking; }
    ConditionalNonGpSurfacePolicy nonGpSurfacePolicy() const { return _nonGpSurfacePolicy; }
    bool gradientCorrectionEnabled() const { return _fpGradient; }
    bool lightOnlyDirectSampling(uint32 sampleIndex) const {
        if (_directPhaseSamplePeriod<=1) return _directPhaseSamplePeriod==0;
        // Alternate the selected slot so both odd/even variance buffers see MIS.
        const uint32 slot=(sampleIndex/_directPhaseSamplePeriod)&1u;
        return sampleIndex%_directPhaseSamplePeriod!=slot;
    }
    bool neuralCorrectionEnabled() const { return _singlePointNeural; }
    bool bridgeCorrectionEnabled() const { return _bridgeCorrection; }
    FirstPassageGradient::AnchorSelection anchorSelection() const { return _anchorSelection; }
    std::uint64_t correctedCollisions() const { return _correctedCollisions.load(std::memory_order_relaxed); }
    std::uint64_t gradientFallbacks() const { return _gradientFallbacks.load(std::memory_order_relaxed); }
    std::uint64_t discardedSurfacePaths() const { return _discardedSurfacePaths.load(std::memory_order_relaxed); }
    std::uint64_t discardedSurfaceConnections() const { return _discardedSurfaceConnections.load(std::memory_order_relaxed); }
};
}
