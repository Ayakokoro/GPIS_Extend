#pragma once

#include "ClassicGlobalMedium.hpp"
#include "math/FirstPassageGradient.hpp"
#include "math/conditional/transport/FixedSpeedGradientDistribution.hpp"
#include <atomic>
#include <optional>

namespace Tungsten {

struct FpGradientPathContext final : MediumPathContext {
    const Medium *owner = nullptr;
    Eigen::Vector3d origin, direction, gradient;
    double value = 0, age = 0, derivative = 0;
    Vec3f rayOrigin;
};
struct FpGradientEventContext final : MediumEventContext {
    const Medium *owner = nullptr;
    Eigen::Vector3d position;
    double derivative = 0;
    std::optional<ConditionalGpis::FixedSpeedGradientDistribution> gradient;
};

// Experimental hybrid: classic distance/visibility, explicit starting jet,
// single-anchor first-passage gradient. No corrected extinction or hazard.
class ClassicGlobalFpGradientMedium final : public ClassicGlobalMedium {
    bool _enabled = true, _bridgeCorrection = true;
    FirstPassageGradient::AnchorSelection _anchorSelection = FirstPassageGradient::AnchorSelection::Fixed5Snr;
    ConditionalGpis::CovarianceKernel _kernel;
    std::shared_ptr<PhaseFunction> _classicPhase, _correctedPhase, _deltaPhase;
    mutable std::atomic<std::uint64_t> _collisions{0}, _fallbacks{0}, _exteriorStarts{0};
    std::shared_ptr<const FpGradientPathContext> context(
        PathSampleGenerator &, const Ray &, const MediumState &) const;
public:
    ClassicGlobalFpGradientMedium();
    void fromJson(JsonPtr, const Scene &) override;
    rapidjson::Value toJson(Allocator &) const override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    bool sampleDistance(PathSampleGenerator &, const Ray &, MediumState &, MediumSample &) const override;
    bool gradientCorrectionEnabled() const { return _enabled; }
    bool bridgeCorrectionEnabled() const { return _bridgeCorrection; }
    FirstPassageGradient::AnchorSelection anchorSelection() const { return _anchorSelection; }
};

} // namespace Tungsten
