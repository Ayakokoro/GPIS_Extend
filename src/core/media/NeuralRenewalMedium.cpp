#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
#include "NeuralRenewalMedium.hpp"
#include "media/NeuralSdfGrid.hpp"
#include "math/neural/RenewalSampler.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include <algorithm>
#include <complex>

namespace Tungsten {
namespace {
using V = ConditionalGpis::Vector3;
V eigen(const Vec3f &p) { return V(p.x(),p.y(),p.z()); }
Vec3f vec(const V &p) { return Vec3f(float(p.x()),float(p.y()),float(p.z())); }

struct NeuralPathContext final : MediumPathContext {
    const NeuralRenewalMedium *owner = nullptr;
    V position = V::Zero();
    Vec3f rayOrigin = Vec3f(0.f);
    std::optional<V> gradient;
    std::shared_ptr<const NeuralGpis::RenewalFlight> flight;
    double distance = 0;
};
struct NeuralEventContext final : MediumEventContext {
    const NeuralRenewalMedium *owner = nullptr;
    V position;
    std::optional<NeuralGpis::RenewalGradientDistribution> gradient;
};

std::shared_ptr<const NeuralPathContext> successor(const NeuralEventContext &event,
    const V &gradient, const Vec3f &position) {
    auto next=std::make_shared<NeuralPathContext>();
    next->owner=event.owner; next->position=event.position; next->rayOrigin=position;
    next->gradient=gradient;
    return next;
}

class NeuralPhase final : public PhaseFunction {
    const NeuralRenewalMedium &_medium;
    const bool _delta;
    const NeuralEventContext &event(const Vec3f &wi, const MediumSample &s) const {
        const auto *e=dynamic_cast<const NeuralEventContext *>(s.eventContext.get());
        if (!e || e->owner!=&_medium || !e->gradient)
            throw std::runtime_error("neural_causal_renewal: missing collision context");
        if ((eigen(wi).normalized()-e->gradient->incomingDirection()).squaredNorm()>1e-12)
            throw std::runtime_error("neural phase requires its incoming collision direction");
        return *e;
    }
public:
    explicit NeuralPhase(const NeuralRenewalMedium &medium, bool delta=false) : _medium(medium),_delta(delta) {}
    bool isSpecular() const override { return _delta; }
    float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample &s) const override {
        const auto &e=event(wi,s);
        if (_delta) return 0;
        const float value=float(e.gradient->phasePdf(eigen(wo)));
        if (!(value>=0) || !std::isfinite(value)) throw std::runtime_error("neural phase PDF exceeds float range");
        return value;
    }
    Vec3f eval(const Vec3f &wi, const Vec3f &wo, const MediumSample &s) const override {
        const float density=pdf(wi,wo,s);
        if (density==0) return Vec3f(0.f);
        const V w=eigen(wi).normalized(), h=(eigen(wo).normalized()-w).normalized();
        return _medium.reflectance(std::abs(w.dot(h)))*density;
    }
    Vec3f evalConnection(PathSampleGenerator &, const Vec3f &wi, const Vec3f &wo,
        const MediumSample &s, std::shared_ptr<const MediumPathContext> &state) const override {
        const Vec3f value=eval(wi,wo,s);
        if (value==0.f) return value;
        const auto &e=event(wi,s);
        // The learned inward derivative is shared by both strategies. A light
        // direction then fixes the complete gradient, including its length.
        const auto gradient=e.gradient->gradientForDirection(eigen(wo));
        if (!gradient) throw std::runtime_error("neural connection lost gradient support");
        state=successor(e,*gradient,s.p);
        return value;
    }
    bool sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample &s, PhaseSample &r) const override {
        const auto &e=event(wi,s);
        ConditionalGpis::Random rng(sampler);
        const V gradient=e.gradient->sample(rng);
        const V w=e.gradient->incomingDirection(), n=gradient.normalized();
        r.w = vec(ConditionalGpis::reflectTravelDirection(w,n)).normalized();
        if (!(eigen(r.w).dot(gradient)>0)) return false;
        r.weight = _medium.reflectance(-w.dot(n));
        r.pdf = _delta ? 1.f : float(e.gradient->phasePdfForGradient(gradient));
        if (!(r.pdf>=0) || !std::isfinite(r.pdf)) throw std::runtime_error("neural sampled PDF exceeds float range");
        r.pathContext = successor(e,gradient,s.p);
        return true;
    }
};

struct Segment {
    std::shared_ptr<const NeuralGpis::RenewalFlight> flight;
    V origin, direction;
    double speed = 1, entry = 0, begin = 0, end = 0;
    bool blocked = false;
};
}

struct NeuralRenewalMedium::Impl {
    ConditionalGpis::GPSSField field;
    std::optional<NeuralGpis::CausalRenewalModel> model;
    std::unique_ptr<NeuralGpis::RenewalSampler> medium;
    std::shared_ptr<PhaseFunction> deltaPhase;

    Segment segment(const NeuralRenewalMedium *owner, const Ray &ray, const MediumState *state) const {
        if (!medium) throw std::runtime_error("neural_causal_renewal: prepareForRender must precede tracing");
        Segment s; s.origin = eigen(ray.pos()); s.direction = eigen(ray.dir()); s.speed = s.direction.norm();
        if (!s.origin.allFinite() || !(s.speed>0) || !std::isfinite(s.speed) || !(ray.farT()>=0))
            throw std::runtime_error("neural_causal_renewal: invalid ray");
        s.direction /= s.speed;
        auto old = state ? std::dynamic_pointer_cast<const NeuralPathContext>(state->pathContext) : nullptr;
        if (old && (old->owner != owner || old->rayOrigin != ray.pos())) old.reset();
        if (old) s.origin = old->position;
        const double length = double(ray.farT())*s.speed;
        if (length == 0) return s;
        // Passing through a forward surface must not reset a surviving flight's
        // GRU history or recondition it as an independent exterior point.
        if (old && old->flight && (s.direction-old->flight->ray.direction).squaredNorm()<1e-14 &&
            old->distance < old->flight->maximumDistance()) {
            s.flight = old->flight; s.begin = old->distance;
            s.end = std::min(s.begin+length,s.flight->maximumDistance());
            return s;
        }
        if (old && old->gradient && !(s.direction.dot(*old->gradient)>0)) { s.blocked=true; return s; }
        const auto domain = field.activeDomain.intersect({s.origin,s.direction});
        if (!domain.hit || !(std::min(domain.exit,length)>domain.entry)) return s;
        s.entry = domain.entry;
        V entry = s.origin+s.entry*s.direction;
        entry = entry.cwiseMax(field.activeDomain.minimum).cwiseMin(field.activeDomain.maximum);
        const auto gradient = old && old->gradient && s.entry==0 ? old->gradient : std::nullopt;
        auto flight = medium->beginFlight({entry,s.direction},gradient);
        if (!flight) return s;
        s.flight = std::make_shared<NeuralGpis::RenewalFlight>(std::move(*flight));
        s.end = std::min(length-s.entry,s.flight->maximumDistance());
        return s;
    }

    std::shared_ptr<const NeuralPathContext> endpoint(const NeuralRenewalMedium *owner,
        const Ray &ray, const Segment &s) const {
        auto c = std::make_shared<NeuralPathContext>(); c->owner=owner;
        if (!std::isfinite(ray.farT())) { c->position=s.origin; c->rayOrigin=ray.pos(); return c; }
        const double length=double(ray.farT())*s.speed;
        c->position=s.origin+length*s.direction; c->rayOrigin=ray.hitpoint();
        if (s.flight && length<=s.entry+s.flight->maximumDistance()-s.begin) {
            c->flight=s.flight; c->distance=s.end;
        }
        // A changed direction at an ordinary surface uses a fresh exterior A
        // start. Such a surface supplies no zero-value/full-gradient observation.
        return c;
    }
};

NeuralRenewalMedium::NeuralRenewalMedium() : _impl(std::make_unique<Impl>()) {
    _maxBounce=128; _phaseFunction=std::make_shared<NeuralPhase>(*this);
    _impl->deltaPhase=std::make_shared<NeuralPhase>(*this,true);
}
NeuralRenewalMedium::~NeuralRenewalMedium() = default;

void NeuralRenewalMedium::fromJson(JsonPtr value, const Scene &scene) {
    JsonSerializable::fromJson(value,scene);
    for (const char *key : {"phase_function","gaussian_process","transmittance","sigma_a","sigma_s"})
        if (value[key]) value.parseError(std::string("neural_causal_renewal does not use '")+key+"'");
    std::string backend="scalar"; value.getField("backend",backend);
    if (backend!="scalar") value.parseError("neural_causal_renewal currently supports backend=scalar only");
    _modelFile=scene.fetchResource(value.getRequiredMember("model"));
    const auto field=value.getRequiredMember("field");
    _gridFile=scene.fetchResource(field.getRequiredMember("file"));
    if (field["transform"])
        field.parseError("neural_causal_renewal uses the native NVDB world transform");
    for (const char *key : {"sigma","sigma_grid","alpha_grid","use_alpha_grid"})
        if (field[key]) field.parseError("neural_causal_renewal loads only the SDF; set covariance.sigma and covariance.lengthScale");
    const auto covariance=value.getRequiredMember("covariance");
    const auto kernelType=covariance.getRequiredMember("type").cast<std::string>();
    if (kernelType!="squared_exponential")
        covariance.parseError("neural_causal_renewal requires a stationary squared_exponential covariance");
    _sigma=covariance.getRequiredMember("sigma").cast<double>();
    _lengthScale=covariance.getRequiredMember("lengthScale").cast<double>();
    if (!(_sigma>0 && std::isfinite(_sigma)) || !(_lengthScale>0 && std::isfinite(_lengthScale)))
        covariance.parseError("covariance.sigma and covariance.lengthScale must be finite positive scalars");
    if (auto material=value["material"]) {
        if (material["roughness"] || material["alpha"])
            material.parseError("neural_causal_renewal takes gradient covariance from covariance.sigma and covariance.lengthScale");
        material.getField("type",_materialType);
        material.getField("eta",_eta); material.getField("k",_k);
        material.getField("albedo",_albedo); material.getField("force_unit_fresnel",_unitFresnel);
    }
    value.getField("profile_maximum_step",_profileStep);
    value.getField("distance_tolerance",_tolerance); value.getField("max_bounces",_maxBounce);
    if ((_materialType!="mirror" && _materialType!="conductor") || _maxBounce<0 ||
        !(_profileStep>0 && std::isfinite(_profileStep)) ||
        !(_tolerance>0 && std::isfinite(_tolerance))) value.parseError("invalid neural_causal_renewal settings");
    for (int i=0;i<3;++i)
        if (!std::isfinite(_eta[i]) || _eta[i]<=0 || !std::isfinite(_k[i]) || _k[i]<0 ||
            !std::isfinite(_albedo[i]) || _albedo[i]<0 || _albedo[i]>1) value.parseError("invalid neural material parameters");
}

rapidjson::Value NeuralRenewalMedium::toJson(Allocator &allocator) const {
    JsonObject field{allocator,"file",*_gridFile};
    return JsonObject{JsonSerializable::toJson(allocator),allocator,
        "type","neural_causal_renewal","backend","scalar","model",*_modelFile,
        "field",rapidjson::Value(field),
        "covariance",JsonObject{allocator,"type","squared_exponential","sigma",_sigma,"lengthScale",_lengthScale},
        "material",JsonObject{allocator,"type",_materialType,
            "eta",_eta,"k",_k,"albedo",_albedo,"force_unit_fresnel",_unitFresnel},
        "profile_maximum_step",_profileStep,"distance_tolerance",_tolerance,"max_bounces",_maxBounce};
}

void NeuralRenewalMedium::loadResources() {
    const auto file=_gridFile->absolute().asString();
    auto mean=NeuralSdfGrid::open(file);
    _impl->medium.reset();
    _impl->field.mean=mean; _impl->field.activeDomain=mean->activeNodeBounds();
    _impl->field.kernel=ConditionalGpis::CovarianceKernel::fromCorrelationLengths(
        ConditionalGpis::CovarianceKernelType::SquaredExponential,_sigma,V::Constant(_lengthScale));
    _impl->model=NeuralGpis::CausalRenewalModel::load(_modelFile->absolute().asString());
}
void NeuralRenewalMedium::prepareForRender() {
    if (!_impl->model || !_impl->field.mean) throw std::runtime_error("neural_causal_renewal: resources not loaded");
    _impl->medium=std::make_unique<NeuralGpis::RenewalSampler>(_impl->field,*_impl->model,_profileStep);
}

bool NeuralRenewalMedium::sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
    MediumState &state, MediumSample &sample) const {
    if (state.bounce>=_maxBounce) return false;
    const auto segment=_impl->segment(this,ray,&state);
    if (segment.blocked) return false;
    sample=MediumSample{};
    sample.t=sample.continuedT=ray.farT(); sample.exited=true;
    sample.p=std::isfinite(ray.farT()) ? ray.hitpoint() : ray.pos();
    sample.weight=sample.continuedWeight=Vec3f(1.f); sample.emission=Vec3f(0.f);
    sample.pdf=1; sample.phase=_phaseFunction.get(); sample.rayInfo=state.info;
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    if (segment.flight && segment.end>segment.begin) {
        // Use the path-owned supplemental generator; no global/thread RNG state.
        const auto high=sampler.uniformGenerator().nextI(), low=sampler.uniformGenerator().nextI();
        ConditionalGpis::Random rng((std::uint64_t(high)<<32)|low);
        const auto &f=*segment.flight;
        const std::array<double,3> u{rng.openUniform01(),rng.openUniform01(),rng.openUniform01()};
        const auto hit=_impl->model->sampleInterval(f.mean,f.start,u,segment.begin/f.ell,
            std::min(segment.end/f.ell,f.mean.maximumX()),_tolerance/f.ell);
        if (hit.hit) {
            auto e=std::make_shared<NeuralEventContext>(); e->owner=this;
            e->position=f.ray.origin+(hit.q*f.ell)*f.ray.direction;
            const auto &parts=f.mean.segments();
            auto part=std::lower_bound(parts.begin(),parts.end(),hit.q,[](const auto &s,double q){return s.end<q;});
            if (part==parts.end()) part=std::prev(parts.end());
            const auto interior=f.ray.origin+(0.5*(part->begin+part->end)*f.ell)*f.ray.direction;
            const V meanGradient=_impl->field.mean->evaluateInCell(e->position,interior).gradient;
            // Keep the learned speed as the common conditioning variable. NEE
            // and phase sampling integrate the remaining transverse Gaussian.
            e->gradient.emplace(_impl->field.kernel,f.ray.direction,hit.q,
                -f.sigma/f.ell*hit.speed,meanGradient,f.birthMeanGradient,f.birthGradient);
            if (e->gradient->isDelta()) sample.phase=_impl->deltaPhase.get();
            sample.exited=false; sample.t=sample.continuedT=float(
                (segment.entry+hit.q*f.ell-segment.begin)/segment.speed);
            sample.p=vec(e->position);
            sample.eventContext=e; state.advance();
            return true;
        }
    }
    state.pathContext=_impl->endpoint(this,ray,segment);
    return true;
}

Vec3f NeuralRenewalMedium::transmittance(PathSampleGenerator &, const Ray &ray, bool, bool, MediumState *state) const {
    const auto s=_impl->segment(this,ray,state);
    if (s.blocked) return Vec3f(0.f);
    double tr=1;
    if (s.flight && s.end>s.begin) {
        const auto &f=*s.flight;
        const auto values=_impl->model->query(f.mean,f.start,
            {s.begin/f.ell,std::min(s.end/f.ell,f.mean.maximumX())});
        const double mass=values[1].cumulativeHazard-values[0].cumulativeHazard;
        if (!std::isfinite(mass) || mass < -1e-9) throw std::runtime_error("invalid neural transmittance");
        tr=std::exp(-std::max(0.,mass));
    }
    if (state) state->pathContext=_impl->endpoint(this,ray,s);
    return Vec3f(float(tr));
}
float NeuralRenewalMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const {
    throw std::runtime_error("neural_causal_renewal supports unidirectional path_tracer; state-free distance PDFs are undefined");
}
Vec3f NeuralRenewalMedium::sigmaA(Vec3f) const { return Vec3f(0.f); }
Vec3f NeuralRenewalMedium::sigmaS(Vec3f) const { throw std::runtime_error("neural extinction requires a flight state and direction"); }
Vec3f NeuralRenewalMedium::sigmaT(Vec3f p) const { return sigmaS(p); }
Vec3f NeuralRenewalMedium::reflectance(double cosine) const {
    if (_materialType=="mirror") return Vec3f(_albedo);
    if (_unitFresnel) return Vec3f(1.f);
    const double c=std::clamp(std::abs(cosine),0.,1.);
    if (c==0) return Vec3f(1.f);
    Vec3f result;
    for (int i=0;i<3;++i) {
        const std::complex<double> n(_eta[i],_k[i]), n2=n*n;
        const auto g=std::sqrt(n2-(1-c*c));
        result[i]=float(0.5*(std::norm((c-g)/(c+g))+std::norm((n2*c-g)/(n2*c+g))));
    }
    return result;
}
}
#endif
