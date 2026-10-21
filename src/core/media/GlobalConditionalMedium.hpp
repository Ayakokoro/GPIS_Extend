#pragma once
#include "ClassicGlobalMedium.hpp"
#include "math/conditional/transport/ConditionalSegment.h"

namespace Tungsten {
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
// renewed at every GP collision and geometry endpoint.
class GlobalConditionalMedium final : public ClassicGlobalMedium {
    ConditionalGpis::GPSSField _field;
    ConditionalGpis::ConditionalTrackingSettings _tracking;
    std::shared_ptr<const ConditionalPathContext> context(
        PathSampleGenerator &,const Ray &,const MediumState &) const;
public:
    GlobalConditionalMedium();
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &) const override;
    void prepareForRender() override;
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &,const Ray &,bool,bool,MediumState *) const override;
    float pdf(PathSampleGenerator &,const Ray &,bool,bool) const override;
    Vec3f sigmaA(Vec3f) const override { return Vec3f(0.0f); }
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f) const override;
    const ConditionalGpis::GPSSField &field() const { return _field; }
    const ConditionalGpis::ConditionalTrackingSettings &trackingSettings() const { return _tracking; }
};
}
