#include "ClassicLocalMedium.hpp"
#include "ClassicLocalGrid.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/transport/MajorantOpticalDepth.h"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace Tungsten {
namespace {
using V=Eigen::Vector3d;
V eigen(const Vec3d &p) { return V(p.x(),p.y(),p.z()); }
Vec3d vec(const V &p) { return Vec3d(p.x(),p.y(),p.z()); }

class ClassicLocalPhase final : public PhaseFunction {
    const ClassicLocalMedium &_medium;
public:
    explicit ClassicLocalPhase(const ClassicLocalMedium &medium) : _medium(medium) {}
    float pdf(const Vec3f &wi,const Vec3f &wo,const MediumSample &s) const override {
        return float(_medium.ndf().phasePdf(_medium.localMeanGradient(Vec3d(s.p)),
                     eigen(Vec3d(wi)).normalized(),eigen(Vec3d(wo)).normalized()));
    }
    Vec3f eval(const Vec3f &wi,const Vec3f &wo,const MediumSample &s) const override {
        const V a=eigen(Vec3d(wi)).normalized(),b=eigen(Vec3d(wo)).normalized();
        if ((b-a).squaredNorm()==0) return Vec3f(0.0f);
        return _medium.reflectance(std::abs(a.dot((b-a).normalized())))*pdf(wi,wo,s);
    }
    bool sample(PathSampleGenerator &sampler,const Vec3f &wi,const MediumSample &s,PhaseSample &result) const override {
        const V w=eigen(Vec3d(wi)).normalized();
        const V mean=_medium.localMeanGradient(Vec3d(s.p));
        const V n=_medium.ndf().sampleVisibleNormal(mean,w,sampler);
        result.w=Vec3f(vec(w-2*w.dot(n)*n)).normalized();
        result.pdf=float(_medium.ndf().phasePdf(mean,w,eigen(Vec3d(result.w)).normalized()));
        result.weight=_medium.reflectance(std::abs(w.dot(n)));
        return result.pdf>0 && std::isfinite(result.pdf);
    }
};
}

ClassicLocalMedium::ClassicLocalMedium() {
    _maxBounce=64;
    _phaseFunction=std::make_shared<ClassicLocalPhase>(*this);
}
ClassicLocalMedium::~ClassicLocalMedium()=default;

void ClassicLocalMedium::fromJson(JsonPtr value,const Scene &scene) {
    JsonSerializable::fromJson(value,scene);
    for (const char *key : {"gaussian_process","phase_function","transmittance","sigma_a","sigma_s","alpha_grid"})
        if (value[key]) value.parseError(std::string("classic_local does not use '")+key+"'; use field and stationary covariance");
    const auto field=value.getRequiredMember("field");
    _file=scene.fetchResource(field.getRequiredMember("file"));
    field.getField("sdf_grid",_sdfName); field.getField("density_grid",_densityName);
    field.getField("transform",_fieldToWorld); field.getField("macrocell_size",_macrocellSize);
    if (field["alpha_grid"] || field["use_alpha_grid"])
        field.parseError("classic_local stationary mode takes gradient covariance from the kernel, not an alpha grid");
    if (_sdfName.empty() || _densityName.empty() || _macrocellSize<1 || _macrocellSize>64)
        field.parseError("classic_local: nonempty grid names and macrocell_size in [1,64] are required");
    _covariance=scene.fetchCovarianceFunction(value.getRequiredMember("covariance"));
    if (!_covariance->isStationaryKernel()) value.parseError("classic_local requires a stationary kernel");
    if (auto material=value["material"]) {
        material.getField("type",_materialType);
        if (material["roughness"] || material["alpha"])
            material.parseError("classic_local: set roughness through the stationary kernel sigma/lengthScale");
        if (_materialType=="mirror") material.getField("albedo",_mirrorAlbedo);
        else if (_materialType=="conductor") {
            material.getField("eta",_eta); material.getField("k",_k);
            material.getField("force_unit_fresnel",_forceUnitFresnel);
        } else material.parseError("classic_local material must be mirror or conductor");
    }
    value.getField("max_bounces",_maxBounce);
    if (_maxBounce<0) value.parseError("classic_local: max_bounces must be nonnegative");
    for (int i=0; i<3; ++i) {
        if (!std::isfinite(_mirrorAlbedo[i]) || _mirrorAlbedo[i]<0 || _mirrorAlbedo[i]>1 ||
            !std::isfinite(_eta[i]) || !std::isfinite(_k[i]) || _eta[i]<=0 || _k[i]<0)
            value.parseError("classic_local: invalid material parameters");
    }
}

rapidjson::Value ClassicLocalMedium::toJson(Allocator &allocator) const {
    JsonObject material{allocator,"type",_materialType};
    if (_materialType=="mirror") material.add("albedo",_mirrorAlbedo);
    else material.add("eta",_eta,"k",_k,"force_unit_fresnel",_forceUnitFresnel);
    return JsonObject{JsonSerializable::toJson(allocator),allocator,
        "type","classic_local","field",JsonObject{allocator,"file",*_file,
            "sdf_grid",_sdfName,"density_grid",_densityName,"transform",_fieldToWorld,"macrocell_size",_macrocellSize},
        "covariance",*_covariance,"material",rapidjson::Value(material),"max_bounces",_maxBounce};
}

void ClassicLocalMedium::loadResources() {
    _covariance->loadResources();
    _grid=std::make_unique<ClassicLocalGrid>(_file->absolute().asString(),_sdfName,_densityName,_fieldToWorld,_macrocellSize);
}
void ClassicLocalMedium::prepareForRender() {
    if (!_grid) throw std::runtime_error("classic_local: loadResources must precede prepareForRender");
    if (!_covariance->isStationaryKernel()) throw std::runtime_error("classic_local requires a stationary kernel");
    const Vec3d zero(0.0);
    const double variance=(*_covariance)(Derivative::None,Derivative::None,zero,zero,zero,zero);
    if (!(variance>0) || !std::isfinite(variance)) throw std::runtime_error("classic_local: invalid field variance");
    Eigen::Matrix3d covariance;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        Vec3d a(0.0),b(0.0); a[i]=1; b[j]=1;
        covariance(i,j)=(*_covariance)(Derivative::First,Derivative::First,zero,zero,a,b);
    }
    _ndf=std::make_unique<ClassicGlobal::GaussianNdf>(covariance);
}

V ClassicLocalMedium::localMeanGradient(const Vec3d &position) const {
    V gradient=_grid->meanGradient(eigen(position));
    const double norm=gradient.stableNorm();
    if (!std::isfinite(norm)) throw std::runtime_error("classic_local: nonfinite SDF gradient");
    // Match macrofacet's local tangent model, including a constant mean.
    if (norm>0) gradient/=norm;
    return gradient;
}
double ClassicLocalMedium::extinction(const Vec3d &position,const Vec3d &travelDirection) const {
    const double density=_grid->density(eigen(position));
    if (!(density>=0) || !std::isfinite(density)) throw std::runtime_error("classic_local: invalid density");
    if (density==0) return 0;
    const V direction=eigen(travelDirection);
    const double speed=direction.stableNorm();
    if (!(speed>0) || !std::isfinite(speed)) throw std::runtime_error("classic_local: invalid direction");
    return density*_ndf->projectedArea(localMeanGradient(position),direction/speed);
}

double ClassicLocalMedium::track(PathSampleGenerator &sampler,const Ray &ray,bool collision,double &hit) const {
    const double begin=ray.nearT(),end=ray.farT();
    const V origin=eigen(Vec3d(ray.pos())),direction=eigen(Vec3d(ray.dir()));
    const double speed=direction.stableNorm();
    if (!origin.allFinite() || !(speed>0) || !std::isfinite(speed) ||
        !(begin>=0) || !(end>=0) || !std::isfinite(begin) || !std::isfinite(end))
        throw std::runtime_error("classic_local requires a finite ray segment and a nonzero direction");
    hit=end;
    if (end<=begin) return 1;
    const V w=direction/speed;
    // Every local mean has norm <= 1. Monotonicity of the negative-part mean
    // gives a tighter directional bound than sqrt(1+trace(C)).
    const double areaMaximum=ClassicGlobal::negativePartMean(-1,std::sqrt(w.dot(_ndf->covariance()*w)))*(1+1e-12);
    ClassicLocalGrid::Cursor cursor(*_grid,origin,direction,begin,end);
    ClassicLocalGrid::Segment segment;
    ConditionalGpis::Random random(sampler);
    ConditionalGpis::MajorantOpticalDepth optical;
    double weight=1;
    while (cursor.next(segment)) {
        const double maximum=segment.densityMaximum*areaMaximum;
        if (!std::isfinite(maximum) || !std::isfinite(maximum*speed))
            throw std::runtime_error("classic_local: nonfinite extinction majorant");
        double t=segment.begin,candidate;
        while (optical.advance(random,maximum*speed,t,segment.end,candidate)) {
            t=candidate;
            const double hazard=extinction(vec(origin+t*direction),vec(w));
            if (!(hazard>=0) || !std::isfinite(hazard) || hazard>maximum*(1+1e-10))
                throw std::runtime_error("classic_local: extinction exceeds its majorant");
            const double acceptance=std::min(1.0,hazard/maximum);
            if (collision) {
                if (random.openUniform01()<acceptance) { hit=t; return 1; }
            } else {
                weight*=1-acceptance;
                if (weight<0.1) {
                    if (random.openUniform01()>=weight/0.1) return 0;
                    weight=0.1;
                }
            }
        }
    }
    return weight;
}

bool ClassicLocalMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,
        MediumState &state,MediumSample &sample) const {
    if (state.bounce>_maxBounce) return false;
    double hit; track(sampler,ray,true,hit);
    sample=MediumSample{};
    sample.t=float(hit); sample.exited=hit==double(ray.farT());
    if (!sample.exited && sample.t>=ray.farT()) sample.t=std::nextafter(ray.farT(),ray.nearT());
    sample.p=ray.pos()+sample.t*ray.dir(); sample.phase=_phaseFunction.get();
    sample.weight=sample.continuedWeight=Vec3f(1.0f); sample.continuedT=sample.t;
    sample.emission=Vec3f(0.0f); sample.pdf=1;
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI; sample.rayInfo=state.info;
    if (!sample.exited) state.advance();
    return true;
}
Vec3f ClassicLocalMedium::transmittance(PathSampleGenerator &sampler,const Ray &ray,bool,bool,MediumState *) const {
    double hit; return Vec3f(float(track(sampler,ray,false,hit)));
}
float ClassicLocalMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const {
    throw std::runtime_error("classic_local supports path_tracer; marginalized distance PDFs for BDPT are unavailable");
}
Vec3f ClassicLocalMedium::sigmaA(Vec3f) const { throw std::runtime_error("classic_local absorption is part of phase reflectance"); }
Vec3f ClassicLocalMedium::sigmaS(Vec3f) const { throw std::runtime_error("classic_local requires directional extinction"); }
Vec3f ClassicLocalMedium::sigmaT(Vec3f) const { throw std::runtime_error("classic_local requires directional extinction"); }
Vec3f ClassicLocalMedium::reflectance(double cosine) const {
    if (_materialType=="mirror") return Vec3f(_mirrorAlbedo);
    if (_forceUnitFresnel) return Vec3f(1.0f);
    const double c=std::clamp(std::abs(cosine),0.0,1.0);
    if (c==0) return Vec3f(1.0f);
    Vec3f result;
    for (int i=0; i<3; ++i) {
        const std::complex<double> index(_eta[i],_k[i]),n2=index*index;
        const auto gamma=std::sqrt(n2-(1-c*c));
        result[i]=float(0.5*(std::norm((c-gamma)/(c+gamma))+std::norm((n2*c-gamma)/(n2*c+gamma))));
    }
    return result;
}
}
