#pragma once

#include "ClassicGlobalMedium.hpp"
#include "math/FirstHitGpis.hpp"
#include "io/Path.hpp"

namespace Tungsten {

// A single-event preview: sample the first GP zero from an empty origin, shade
// it with the unmodified classic_global phase, and stop the path afterwards.
// Shadow connections retain geometry visibility but deliberately omit GP
// attenuation; no collision-origin or multi-bounce GP law is implied.
class FirstHitGpisMedium final : public ClassicGlobalMedium {
    double _lengthScale = 0.0;
    FirstHitGpis::Settings _firstHit;
    bool _statisticsEnabled = false;
    std::string _statisticsOutputPrefix = "first-hit-stats";
    const Scene *_statisticsScene = nullptr;
    Path _statisticsInputDirectory;
    struct Statistics;
    std::unique_ptr<Statistics> _statistics;

public:
    FirstHitGpisMedium();
    ~FirstHitGpisMedium() override;
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &allocator) const override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    bool sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
                        MediumState &state, MediumSample &sample) const override;
    Vec3f transmittance(PathSampleGenerator &, const Ray &, bool, bool,
                       MediumState *) const override;
    float pdf(PathSampleGenerator &, const Ray &, bool, bool) const override;

    double lengthScale() const { return _lengthScale; }
    const FirstHitGpis::Settings &firstHitSettings() const { return _firstHit; }
};

}
