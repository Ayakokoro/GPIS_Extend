#pragma once

#include "SparseConvolutionNoiseMedium.hpp"
#include "math/FirstPassageGradient.hpp"
#include "math/conditional/gpss/CovarianceKernel.h"
#include <atomic>

namespace Tungsten {

// Diagnostic hybrid: native 3D sparse-convolution renewal+ flights and shadow
// visibility, with an independently resampled fixed5 gradient at accepted hits.
// The complete gradient is observed before the existing BRDF phase is sampled.
class RenewalPlusFpGradientMedium final : public SparseConvolutionNoiseMedium {
    bool _enabled = true, _bridgeCorrection = true;
    FirstPassageGradient::AnchorSelection _selection = FirstPassageGradient::AnchorSelection::Fixed5Snr;
    ConditionalGpis::CovarianceKernel _kernel;
    std::shared_ptr<GaussianProcess> _process;
    mutable std::atomic<std::uint64_t> _collisions{0}, _corrected{0}, _escapes{0};
    mutable std::atomic<std::uint64_t> _exteriorBirths{0}, _surfaceBirths{0}, _fallbacks{0};
    mutable std::atomic<std::uint64_t> _unsupportedBirths{0}, _zeroLength{0};

    void onDistanceSample(PathSampleGenerator &, const Ray &, double,
        MediumState &, MediumSample &) const override;
public:
    void fromJson(JsonPtr, const Scene &) override;
    rapidjson::Value toJson(Allocator &) const override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    bool gradientCorrectionEnabled() const { return _enabled; }
    std::uint64_t correctedCollisions() const { return _corrected.load(); }
    std::uint64_t unsupportedBirths() const { return _unsupportedBirths.load(); }
};

} // namespace Tungsten
