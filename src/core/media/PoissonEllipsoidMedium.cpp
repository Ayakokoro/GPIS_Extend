#include "PoissonEllipsoidMedium.hpp"
#include "GpisSdfGrid.hpp"
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include <complex>
#include <sstream>
#include <stdexcept>

namespace Tungsten {
namespace {
namespace cg=ConditionalGpis;
using V=Eigen::Vector3d;
V eigen(const Vec3d &v) { return V(v.x(),v.y(),v.z()); }
Vec3f vec(const V &v) { return Vec3f(float(v.x()),float(v.y()),float(v.z())); }

class LinearField final : public cg::MeanField {
    V reference_,gradient_;
    double minimum_;
public:
    explicit LinearField(const LinearMean &mean)
        : reference_(eigen(mean.referencePoint())),gradient_(eigen(mean.gradient())),minimum_(mean.minimum()) {
        if (!reference_.allFinite() || !gradient_.allFinite() || !std::isfinite(minimum_))
            throw std::invalid_argument("poisson_ellipsoid_local: invalid linear SDF");
    }
    const char *typeName() const override { return "linear"; }
    SdfJet evaluate(const V &p) const override {
        const double raw=gradient_.dot(p-reference_);
        return {std::max(minimum_,raw),raw<minimum_ ? V::Zero().eval() : gradient_};
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &box) const override {
        V p;
        for (int i=0;i<3;++i) p[i]=gradient_[i]>=0 ? box.minimum[i] : box.maximum[i];
        return {std::max(minimum_,gradient_.dot(p-reference_)),gradient_.norm(),true};
    }
};

struct EllipsoidEvent final : MediumEventContext {
    const PoissonEllipsoidMedium *owner=nullptr;
    LocalEllipsoid::Collision collision;
    V incoming;
};

// Condition on the sampled particle normal. Its smooth mirror/conductor law is
// a delta event; the random normal has already been drawn by marked tracking.
class EllipsoidPhase final : public PhaseFunction {
    const PoissonEllipsoidMedium &_medium;
public:
    explicit EllipsoidPhase(const PoissonEllipsoidMedium &medium) : _medium(medium) {}
    bool isSpecular() const override { return true; }
    float pdf(const Vec3f &,const Vec3f &,const MediumSample &) const override { return 0; }
    Vec3f eval(const Vec3f &,const Vec3f &,const MediumSample &) const override { return Vec3f(0.f); }
    bool sample(PathSampleGenerator &,const Vec3f &wi,const MediumSample &sample,PhaseSample &result) const override {
        const auto *event=dynamic_cast<const EllipsoidEvent *>(sample.eventContext.get());
        if (!event || event->owner!=&_medium)
            throw std::runtime_error("poisson_ellipsoid_local: missing sampled collision normal");
        const V w=eigen(Vec3d(wi)).normalized(),n=event->collision.normal;
        if ((w-event->incoming).norm()>1e-5)
            throw std::runtime_error("poisson_ellipsoid_local: collision used with a different incoming direction");
        result.w=vec(w-2*w.dot(n)*n).normalized();
        result.weight=_medium.reflectance(event->collision.type,-w.dot(n));
        result.pdf=1; // Discrete mass conditional on this sampled normal.
        return true;
    }
};
}

PoissonEllipsoidMedium::PoissonEllipsoidMedium() {
    _maxBounce=128;
    _phaseFunction=std::make_shared<EllipsoidPhase>(*this);
}
PoissonEllipsoidMedium::~PoissonEllipsoidMedium()=default;

PoissonEllipsoidMedium::Material PoissonEllipsoidMedium::readMaterial(JsonPtr value,Material material) {
    if (!value) return material;
    value.getField("type",material.type);
    value.getField("albedo",material.albedo);
    value.getField("eta",material.eta); value.getField("k",material.k);
    value.getField("force_unit_fresnel",material.unitFresnel);
    if (material.type!="mirror" && material.type!="conductor")
        value.parseError("poisson_ellipsoid_local supports smooth mirror or conductor particles");
    if (value["roughness"] || value["alpha"])
        value.parseError("poisson_ellipsoid_local uses ellipsoid dimensions for its normal distribution");
    for (int i=0;i<3;++i)
        if (!std::isfinite(material.albedo[i]) || material.albedo[i]<0 || material.albedo[i]>1 ||
            !std::isfinite(material.eta[i]) || material.eta[i]<=0 ||
            !std::isfinite(material.k[i]) || material.k[i]<0)
            value.parseError("poisson_ellipsoid_local: invalid particle material");
    return material;
}

void PoissonEllipsoidMedium::fromJson(JsonPtr value,const Scene &scene) {
    JsonSerializable::fromJson(value,scene);
    for (const char *key:{"gaussian_process","covariance","phase_function","transmittance","sigma_a","sigma_s","density_grid"})
        if (value[key]) value.parseError(std::string("poisson_ellipsoid_local does not use '")+key+"'");
    const auto sdf=value.getRequiredMember("sdf");
    if (auto grid=sdf["grid"]) {
        bool generate=false; grid.getField("request_sdf",generate);
        std::string integration; grid.getField("integration_method",integration);
        if (generate || integration=="residual_ratio")
            grid.parseError("use an existing SDF with request_sdf=false and integration_method=raymarching");
    }
    _mean=scene.fetchMeanFunction(sdf);
    _decayLength=value.castField<double>("decay_length");
    _trackingStep=_decayLength;
    value.getField("tracking_step",_trackingStep);
    value.getField("frame",_frame);
    value.getField("max_bounces",_maxBounce);
    if (!(_decayLength>0) || !std::isfinite(_decayLength) || !(_trackingStep>0) ||
        !std::isfinite(_trackingStep) || _maxBounce<0 || (_frame!="sdf" && _frame!="world"))
        value.parseError("positive finite decay_length/tracking_step, nonnegative max_bounces, and frame=sdf or world are required");
    const Material defaults=readMaterial(value["material"],Material{});
    const auto particles=value.getRequiredMember("particles");
    if (!particles.isArray() || particles.size()==0) particles.parseError("particles must be a nonempty array");
    _particles.clear();
    for (unsigned i=0;i<particles.size();++i) {
        const auto entry=particles[i];
        Particle p;
        p.semiaxes=entry.castField<Vec3d>("semiaxes");
        p.density=entry.castField<double>("density");
        entry.getField("rotation_degrees",p.rotation);
        p.material=readMaterial(entry["material"],defaults);
        try { LocalEllipsoid::Grain check(eigen(p.semiaxes),eigen(p.rotation),p.density); }
        catch (const std::exception &error) { entry.parseError(error.what()); }
        _particles.push_back(p);
    }
}

rapidjson::Value PoissonEllipsoidMedium::toJson(Allocator &allocator) const {
    rapidjson::Value particles(rapidjson::kArrayType);
    for (const auto &p:_particles) {
        const auto &m=p.material;
        JsonObject material{allocator,"type",m.type};
        if (m.type=="mirror") material.add("albedo",m.albedo);
        else material.add("eta",m.eta,"k",m.k,"force_unit_fresnel",m.unitFresnel);
        particles.PushBack(rapidjson::Value(JsonObject{allocator,
            "semiaxes",p.semiaxes,"rotation_degrees",p.rotation,"density",p.density,
            "material",rapidjson::Value(material)}),allocator);
    }
    return JsonObject{JsonSerializable::toJson(allocator),allocator,
        "type","poisson_ellipsoid_local","sdf",*_mean,"decay_length",_decayLength,
        "tracking_step",_trackingStep,"frame",_frame,"max_bounces",_maxBounce,"particles",std::move(particles)};
}

void PoissonEllipsoidMedium::loadResources() { _mean->loadResources(); }

void PoissonEllipsoidMedium::prepareForRender() {
    _field.reset();
    if (auto tabulated=dynamic_cast<TabulatedMean *>(_mean.get())) {
        if (tabulated->isVolume()) throw std::runtime_error("poisson_ellipsoid_local needs an SDF, not a density grid");
#if OPENVDB_AVAILABLE
        if (auto vdb=dynamic_cast<VdbGrid *>(tabulated->grid().get())) {
            auto grid=std::make_shared<GpisSdfGrid>();
            grid->initialize(*tabulated,*vdb);
            _field=grid;
        }
#endif
    } else if (auto sphere=dynamic_cast<SphericalMean *>(_mean.get()))
        _field=std::make_shared<cg::SphereMean>(eigen(sphere->center()),sphere->radius());
    else if (auto linear=dynamic_cast<LinearMean *>(_mean.get()))
        _field=std::make_shared<LinearField>(*linear);
    else if (dynamic_cast<HomogeneousMean *>(_mean.get()))
        _field=std::make_shared<cg::ConstantMean>((*_mean)(Derivative::None,Vec3d(0.0),Vec3d(0.0)));
    else if (_mean->sdfQueries())
        _field=std::make_shared<cg::SdfMeanField>(_mean);
    if (!_field) throw std::runtime_error("poisson_ellipsoid_local requires a linear/spherical/constant SDF, a trilinear VDB, or certified procedural SDF queries");
    std::vector<LocalEllipsoid::Grain> grains;
    for (const auto &p:_particles) grains.emplace_back(eigen(p.semiaxes),eigen(p.rotation),p.density);
    _model=std::make_unique<LocalEllipsoid::Model>(std::move(grains),_decayLength,_frame=="sdf");
}

SdfJet PoissonEllipsoidMedium::localPlane(const Vec3d &position) const {
    if (!_field) throw std::runtime_error("poisson_ellipsoid_local: prepareForRender must precede queries");
    return _field->evaluate(eigen(position));
}
double PoissonEllipsoidMedium::extinction(const Vec3d &position,const Vec3d &travel,int order) const {
    const auto jet=localPlane(position);
    if (!_model) throw std::runtime_error("poisson_ellipsoid_local: prepareForRender must precede queries");
    return _model->extinction(jet,eigen(travel),order);
}

double PoissonEllipsoidMedium::track(PathSampleGenerator &sampler,const Ray &ray,bool collision,
                                    LocalEllipsoid::Collision &result) const {
    if (!_model || !_field) throw std::runtime_error("poisson_ellipsoid_local: prepareForRender must precede tracing");
    const V direction=eigen(Vec3d(ray.dir()));
    const double speed=direction.stableNorm();
    if (!(speed>0) || !std::isfinite(speed) || !(ray.farT()>=0) || !std::isfinite(ray.farT())) {
        std::ostringstream message;
        message.precision(10);
        message << "poisson_ellipsoid_local needs a finite boundary-enclosed ray segment: origin="
                << ray.pos() << ", direction=" << ray.dir() << ", near=" << ray.nearT() << ", far=" << ray.farT();
        throw std::runtime_error(message.str());
    }
    cg::Random rng(sampler);
    // nearT is a geometry self-intersection epsilon, not a vacuum gap in this
    // continuous local law. Integrate from the actual current ray origin.
    return _model->track(*_field,eigen(Vec3d(ray.pos())),direction/speed,0,double(ray.farT())*speed,
                         _trackingStep,rng,collision,result);
}

bool PoissonEllipsoidMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,
    MediumState &state,MediumSample &sample) const {
    if (state.bounce>=_maxBounce) return false;
    LocalEllipsoid::Collision collision;
    track(sampler,ray,true,collision);
    sample=MediumSample{};
    sample.t=sample.continuedT=ray.farT(); sample.p=ray.hitpoint(); sample.exited=!collision.hit;
    sample.weight=sample.continuedWeight=Vec3f(1.f); sample.emission=Vec3f(0.f); sample.pdf=1;
    sample.phase=_phaseFunction.get(); sample.rayInfo=state.info;
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    state.pathContext.reset();
    if (collision.hit) {
        const V direction=eigen(Vec3d(ray.dir()));
        sample.t=float(collision.distance/direction.norm());
        if (sample.t>=ray.farT()) sample.t=std::nextafter(ray.farT(),0.f);
        sample.continuedT=sample.t; sample.p=vec(collision.position);
        sample.aniso=Vec3d(collision.normal.x(),collision.normal.y(),collision.normal.z());
        auto event=std::make_shared<EllipsoidEvent>();
        event->owner=this; event->collision=collision; event->incoming=direction.normalized();
        sample.eventContext=event; state.advance();
    }
    return true;
}

Vec3f PoissonEllipsoidMedium::transmittance(PathSampleGenerator &sampler,const Ray &ray,bool,bool,MediumState *) const {
    LocalEllipsoid::Collision unused;
    return Vec3f(float(track(sampler,ray,false,unused)));
}
float PoissonEllipsoidMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const {
    throw std::runtime_error("poisson_ellipsoid_local supports path_tracer; marginalized distance PDFs for BDPT are unavailable");
}
Vec3f PoissonEllipsoidMedium::sigmaA(Vec3f) const { throw std::runtime_error("particle absorption is applied at reflection"); }
Vec3f PoissonEllipsoidMedium::sigmaS(Vec3f) const { throw std::runtime_error("poisson_ellipsoid_local requires directional extinction"); }
Vec3f PoissonEllipsoidMedium::sigmaT(Vec3f p) const { return sigmaS(p); }
Vec3f PoissonEllipsoidMedium::reflectance(std::size_t type,double cosine) const {
    const auto &material=_particles.at(type).material;
    if (material.type=="mirror") return Vec3f(material.albedo);
    if (material.unitFresnel) return Vec3f(1.f);
    const double c=std::clamp(std::abs(cosine),0.0,1.0);
    if (c==0) return Vec3f(1.f);
    Vec3f result;
    for (int i=0;i<3;++i) {
        const std::complex<double> index(material.eta[i],material.k[i]),n2=index*index;
        const auto g=std::sqrt(n2-(1-c*c));
        result[i]=float(0.5*(std::norm((c-g)/(c+g))+std::norm((n2*c-g)/(n2*c+g))));
    }
    return result;
}

} // namespace Tungsten
