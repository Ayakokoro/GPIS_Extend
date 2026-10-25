#ifndef CLASSICGLOBALMEDIUM_HPP_
#define CLASSICGLOBALMEDIUM_HPP_

#include "Medium.hpp"
#include "math/ClassicGlobalMath.hpp"

namespace Tungsten {
class MeanFunction;
class CovarianceFunction;
struct GpisSdfGrid;

// Memoryless global_pointwise Gaussian microfacet medium. Only the SDF is
// spatially tabulated; density and directional extinction are evaluated online.
class ClassicGlobalMedium : public Medium {
protected:
    std::shared_ptr<MeanFunction> _mean;
    std::shared_ptr<CovarianceFunction> _covariance;
    std::unique_ptr<GpisSdfGrid> _sdfGrid;
    std::unique_ptr<ClassicGlobal::GaussianNdf> _ndf;
    std::string _materialType;
    Vec3d _mirrorAlbedo;
    Vec3d _eta, _k;
    bool _forceUnitFresnel;
    double _sigma, _lipschitz, _gradientBound, _trackingStep;
    double _configuredLipschitz, _configuredGradientBound;

    double track(PathSampleGenerator &sampler, const Ray &ray,
                 bool sampleCollision, double &eventT) const;
public:
    ClassicGlobalMedium();
    ~ClassicGlobalMedium() override;
    void fromJson(JsonPtr value, const Scene &scene) override;
    rapidjson::Value toJson(Allocator &allocator) const override;
    void loadResources() override;
    void prepareForRender() override;
    bool isHomogeneous() const override { return false; }
    Vec3f sigmaA(Vec3f p) const override;
    Vec3f sigmaS(Vec3f p) const override;
    Vec3f sigmaT(Vec3f p) const override;
    bool sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
                        MediumState &state, MediumSample &sample) const override;
    Vec3f transmittance(PathSampleGenerator &sampler, const Ray &ray,
                       bool startOnSurface, bool endOnSurface, MediumState *state) const override;
    float pdf(PathSampleGenerator &sampler, const Ray &ray,
              bool startOnSurface, bool endOnSurface) const override;

    double meanValue(const Vec3d &p) const;
    Vec3d meanGradient(const Vec3d &p) const;
    double extinction(const Vec3d &p, const Vec3d &travelDirection) const;
    double fieldSigma() const { return _sigma; }
    const ClassicGlobal::GaussianNdf &ndf() const { return *_ndf; }
    Vec3f reflectance(double cosine) const;
    bool hasDiffuseMicroMaterial() const { return _materialType == "lambert"; }
};
}
#endif
