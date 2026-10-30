#pragma once

#include "Medium.hpp"
#include "math/LocalEllipsoid.hpp"

namespace Tungsten {
class MeanFunction;

class PoissonEllipsoidMedium final : public Medium {
    struct Material {
        std::string type="mirror";
        Vec3d albedo=Vec3d(1.0), eta=Vec3d(0.2,0.9,1.1), k=Vec3d(3.9,2.5,2.2);
        bool unitFresnel=false;
    };
    struct Particle {
        Vec3d semiaxes, rotation=Vec3d(0.0);
        double density=0;
        Material material;
    };
    std::shared_ptr<MeanFunction> _mean;
    std::shared_ptr<const ConditionalGpis::MeanField> _field;
    std::unique_ptr<LocalEllipsoid::Model> _model;
    std::vector<Particle> _particles;
    double _decayLength=0, _trackingStep=0;
    std::string _frame="sdf";
    static Material readMaterial(JsonPtr value,Material defaults);
    double track(PathSampleGenerator &,const Ray &,bool,LocalEllipsoid::Collision &,
                 const LocalEllipsoid::CandidateCallback & = {}) const;
public:
    PoissonEllipsoidMedium();
    ~PoissonEllipsoidMedium() override;
    void fromJson(JsonPtr,const Scene &) override;
    rapidjson::Value toJson(Allocator &) const override;
    void loadResources() override;
    void prepareForRender() override;
    bool isHomogeneous() const override { return false; }
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &) const override;
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &,
                        const LocalEllipsoid::CandidateCallback &) const;
    Vec3f candidateScattering(const LocalEllipsoid::Candidate &,const Vec3f &travel,
                             const Vec3f &outgoing,double &density) const;
    bool hasCandidateLighting(const MediumSample &) const;
    double candidateDirectionPdf(const MediumSample &,const Vec3f &travel,const Vec3f &outgoing) const;
    Vec3f transmittance(PathSampleGenerator &,const Ray &,bool,bool,MediumState *) const override;
    float pdf(PathSampleGenerator &,const Ray &,bool,bool) const override;
    Vec3f sigmaA(Vec3f) const override;
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f) const override;
    double extinction(const Vec3d &,const Vec3d &,int quadratureOrder=32) const;
    SdfJet localPlane(const Vec3d &) const;
    Vec3f reflectance(std::size_t type,double cosine) const;
    const LocalEllipsoid::Model &model() const { return *_model; }
};

} // namespace Tungsten
