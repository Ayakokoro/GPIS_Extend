#pragma once
#include "Medium.hpp"
#include "io/Path.hpp"
#include <memory>

namespace Tungsten {

// CPU adapter for macrofacet's exported A/B causal Renewal model. Geometry,
// textures, lights and scheduling remain ordinary Tungsten path tracing.
class NeuralRenewalMedium final : public Medium {
    struct Impl;
    std::unique_ptr<Impl> _impl;
    PathPtr _modelFile, _gridFile;
    double _sigmaOverride = 0, _roughness = 0.5, _profileStep = 0.25, _tolerance = 1e-9;
    std::string _materialType = "conductor";
    Vec3d _eta = Vec3d(0.2,0.9,1.1), _k = Vec3d(3.9,2.5,2.2), _albedo = Vec3d(1.0);
    bool _unitFresnel = false;
public:
    NeuralRenewalMedium();
    ~NeuralRenewalMedium() override;
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &allocator) const override;
    void loadResources() override;
    void prepareForRender() override;
    bool isHomogeneous() const override { return false; }
    bool sampleDistance(PathSampleGenerator &, const Ray &, MediumState &, MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &, const Ray &, bool, bool, MediumState *) const override;
    float pdf(PathSampleGenerator &, const Ray &, bool, bool) const override;
    Vec3f sigmaA(Vec3f) const override;
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f) const override;
    Vec3f reflectance(double cosine) const;
};
}
