#pragma once

#include "GaussianProcessMedium.hpp"
#include "math/AdaptiveGpis.hpp"
#include <atomic>

namespace Tungsten {

// CPU reference: full online history within each flight, Renewal+ across GP hits.
class AdaptiveFunctionSpaceGaussianProcessMedium : public GaussianProcessMedium {
public:
    AdaptiveFunctionSpaceGaussianProcessMedium() = default;
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &allocator) const override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    bool intersectGP(PathSampleGenerator &, const Ray &, MediumState &, double &, bool) const override;
    bool sampleGradient(PathSampleGenerator &, const Ray &, const Vec3d &, const float &,
                        MediumState &, Vec3d &) const override;
    bool sampleDistance(PathSampleGenerator &, const Ray &, MediumState &, MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &, const Ray &, bool, bool, MediumState *) const override;
    float pdf(PathSampleGenerator &, const Ray &, bool, bool) const override;
    const AdaptiveGpis::Settings &settings() const { return _settings; }
private:
    struct Context;
    std::shared_ptr<Context> trace(PathSampleGenerator &, const Ray &, MediumState &, double &) const;
    double mean(const Vec3d &) const;
    Vec3d meanGradient(const Vec3d &) const;
    double meanLipschitz(const Vec3d &, const Vec3d &, double, double) const;
    AdaptiveGpis::Settings _settings;
    std::shared_ptr<GaussianProcess> _process;
    double _sigma = 0, _length = 0;
    bool _statistics = false;
    bool _oneDSampling = false;
    SparseConv1DSamplingScheme _samplingScheme = SparseConv1DSamplingScheme::UNI;
    mutable std::atomic<std::uint64_t> _rays{0}, _hits{0}, _queries{0}, _redundant{0},
        _steps{0}, _minimumSteps{0}, _refinements{0}, _invalidBirths{0}, _invalidGradients{0}, _rankDecompositions{0};
    mutable std::atomic<std::uint64_t> _misEvents{0}, _neePdfQueries{0}, _neeGradientQueries{0}, _degenerateGradients{0};
    mutable std::atomic<std::uint64_t> _choleskyRebuilds{0}, _varianceClamps{0}, _jointCovarianceClamps{0};
    mutable std::atomic<double> _largestVarianceCorrection{0};
};
}
