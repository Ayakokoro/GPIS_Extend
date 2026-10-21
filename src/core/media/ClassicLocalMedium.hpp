#pragma once
#include "Medium.hpp"
#include "math/ClassicGlobalMath.hpp"
#include "math/Mat4f.hpp"
#include "io/Path.hpp"

namespace Tungsten {
class ClassicLocalGrid;
class CovarianceFunction;

// Baked density with a local unit mean gradient and one stationary covariance.
class ClassicLocalMedium final : public Medium {
    PathPtr _file;
    std::string _sdfName="sdf", _densityName="density", _materialType="conductor";
    Mat4f _fieldToWorld;
    int _macrocellSize=8;
    std::shared_ptr<CovarianceFunction> _covariance;
    std::unique_ptr<ClassicLocalGrid> _grid;
    std::unique_ptr<ClassicGlobal::GaussianNdf> _ndf;
    Vec3d _mirrorAlbedo=Vec3d(1.0), _eta=Vec3d(0.2,0.9,1.1), _k=Vec3d(3.9,2.5,2.2);
    bool _forceUnitFresnel=false;
    double track(PathSampleGenerator &sampler,const Ray &ray,bool collision,double &hit) const;
public:
    ClassicLocalMedium();
    ~ClassicLocalMedium() override;
    void fromJson(JsonPtr value,const Scene &scene) override;
    rapidjson::Value toJson(Allocator &allocator) const override;
    void loadResources() override;
    void prepareForRender() override;
    bool isHomogeneous() const override { return false; }
    Vec3f sigmaA(Vec3f p) const override;
    Vec3f sigmaS(Vec3f p) const override;
    Vec3f sigmaT(Vec3f p) const override;
    bool sampleDistance(PathSampleGenerator &sampler,const Ray &ray,MediumState &state,MediumSample &sample) const override;
    Vec3f transmittance(PathSampleGenerator &sampler,const Ray &ray,bool startOnSurface,bool endOnSurface,MediumState *state) const override;
    float pdf(PathSampleGenerator &sampler,const Ray &ray,bool startOnSurface,bool endOnSurface) const override;
    Eigen::Vector3d localMeanGradient(const Vec3d &position) const;
    double extinction(const Vec3d &position,const Vec3d &travelDirection) const;
    const ClassicGlobal::GaussianNdf &ndf() const { return *_ndf; }
    Vec3f reflectance(double cosine) const;
};
}
