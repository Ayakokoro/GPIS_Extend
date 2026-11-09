#include "ClassicGlobalFpGradientMedium.hpp"
#include "math/FirstPassageGradient.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "io/JsonObject.hpp"
#include "phasefunctions/ObservedNormalScattering.hpp"
#include <iostream>

namespace Tungsten {
namespace cg = ConditionalGpis;
namespace {
Eigen::Vector3d eigen(const Vec3f &v) { return {v.x(),v.y(),v.z()}; }
Eigen::Vector3d eigen(const Vec3d &v) { return {v.x(),v.y(),v.z()}; }
Vec3d vec(const Eigen::Vector3d &v) { return {v.x(),v.y(),v.z()}; }

std::shared_ptr<const FpGradientPathContext> successor(const FpGradientEventContext &event,
        const Eigen::Vector3d &gradient,const Vec3f &wo,const Vec3f &position,
        std::optional<Eigen::Vector3d> reflection=std::nullopt) {
    auto next=std::make_shared<FpGradientPathContext>();
    next->owner=event.owner; next->origin=event.position;
    next->direction=reflection ? reflection->normalized() : eigen(wo).normalized();
    next->gradient=gradient;
    next->value=0; next->rayOrigin=position;
    // At grazing incidence the reflected angle can be below a float ULP.
    // Preserve the sampled double direction and its exact axial observation;
    // recomputing their dot product after ray rounding can reverse its sign.
    next->derivative=reflection ? -event.derivative : next->direction.dot(gradient);
    if(!(next->derivative>0))
        throw std::runtime_error("FP gradient outgoing direction lost exterior support");
    return next;
}

// The same fixed-axial-gradient construction as SingleHistoryPhase: sample the
// axial latent once with the collision, then both MIS strategies condition on
// it. The transverse Gaussian is exact for a stationary anisotropic SE kernel.
class FpGradientPhase final : public PhaseFunction {
    const ClassicGlobalFpGradientMedium &_medium;
    bool _delta;
    const FpGradientEventContext &event(const Vec3f &wi,const MediumSample &s) const {
        const auto *e=dynamic_cast<const FpGradientEventContext *>(s.eventContext.get());
        if(!e || e->owner!=&_medium || !e->gradient ||
            (eigen(wi).normalized()-e->gradient->incomingDirection()).squaredNorm()>1e-12)
            throw std::runtime_error("missing FP gradient collision context");
        return *e;
    }
public:
    explicit FpGradientPhase(const ClassicGlobalFpGradientMedium &m,bool delta=false)
        : _medium(m),_delta(delta) {}
    bool isSpecular() const override { return _delta && !_medium.hasDiffuseMicroMaterial(); }
    float pdf(const Vec3f &wi,const Vec3f &wo,const MediumSample &s) const override {
        const auto &e=event(wi,s);
        if(_medium.hasDiffuseMicroMaterial()) return observedCosinePdf(wo,s);
        if(_delta) return 0;
        const float value=float(e.gradient->phasePdf(eigen(wo)));
        if(!(value>=0) || !std::isfinite(value)) throw std::runtime_error("nonfinite FP phase PDF");
        return value;
    }
    Vec3f eval(const Vec3f &wi,const Vec3f &wo,const MediumSample &s) const override {
        const float density=pdf(wi,wo,s);
        if(density==0) return Vec3f(0.f);
        if(_medium.hasDiffuseMicroMaterial()) return _medium.reflectance(1)*density;
        const auto &w=event(wi,s).gradient->incomingDirection();
        const Eigen::Vector3d h=(eigen(wo).normalized()-w).normalized();
        return _medium.reflectance(std::abs(w.dot(h)))*density;
    }
    Vec3f evalConnection(PathSampleGenerator &,const Vec3f &wi,const Vec3f &wo,
            const MediumSample &s,std::shared_ptr<const MediumPathContext> &state) const override {
        const Vec3f value=eval(wi,wo,s);
        if(value==0.f) return value;
        const auto &e=event(wi,s);
        if(_medium.hasDiffuseMicroMaterial()) state=successor(e,eigen(s.aniso),wo,s.p);
        else {
            const auto g=e.gradient->gradientForDirection(eigen(wo));
            if(!g) throw std::runtime_error("FP connection lost gradient support");
            state=successor(e,*g,wo,s.p,Eigen::Vector3d(eigen(wo).normalized()));
        }
        return value;
    }
    bool sample(PathSampleGenerator &sampler,const Vec3f &wi,const MediumSample &s,PhaseSample &r) const override {
        const auto &e=event(wi,s);
        if(_medium.hasDiffuseMicroMaterial()) {
            r.w=sampleObservedCosine(sampler,s); r.pdf=pdf(wi,r.w,s);
            r.weight=_medium.reflectance(1);
            if(!(r.pdf>0)) return false;
            r.pathContext=successor(e,eigen(s.aniso),r.w,s.p);
        } else {
            cg::Random rng(sampler);
            const Eigen::Vector3d g=e.gradient->sample(rng);
            const Eigen::Vector3d w=e.gradient->incomingDirection();
            const Eigen::Vector3d reflected=w-(2*e.derivative/g.squaredNorm())*g;
            r.w=Vec3f(vec(reflected)).normalized();
            r.weight=_medium.reflectance(-e.derivative/g.norm());
            // Evaluate the Jacobian at the actual sampled gradient. Inverting
            // a float-rounded direction can land outside a very narrow lobe.
            r.pdf=_delta ? 1. : e.gradient->phasePdfForGradient(g);
            if(!(r.pdf>0) || !std::isfinite(r.pdf))
                throw std::runtime_error("invalid FP sampled phase PDF");
            r.pathContext=successor(e,g,r.w,s.p,reflected);
        }
        return true;
    }
};
} // namespace

ClassicGlobalFpGradientMedium::ClassicGlobalFpGradientMedium() {
    _classicPhase=_phaseFunction;
    _correctedPhase=std::make_shared<FpGradientPhase>(*this);
    _deltaPhase=std::make_shared<FpGradientPhase>(*this,true);
    _phaseFunction=_correctedPhase;
}
void ClassicGlobalFpGradientMedium::fromJson(JsonPtr value,const Scene &scene) {
    ClassicGlobalMedium::fromJson(value,scene);
    _enabled=true; _bridgeCorrection=true;
    _anchorSelection=FirstPassageGradient::AnchorSelection::Fixed5Snr;
    if(auto settings=value["fp_gradient"]) {
        settings.getField("enabled",_enabled);
        settings.getField("bridge_correction",_bridgeCorrection);
        std::string selection="fixed5_snr";
        settings.getField("anchor_selection",selection);
        try { _anchorSelection=FirstPassageGradient::parseAnchorSelection(selection); }
        catch(const std::invalid_argument &e) { settings.parseError(e.what()); }
    }
    _phaseFunction=_enabled ? _correctedPhase : _classicPhase;
}
rapidjson::Value ClassicGlobalFpGradientMedium::toJson(Allocator &a) const {
    auto value=ClassicGlobalMedium::toJson(a);
    value["type"].SetString("classic_global_fp_gradient",a);
    value.AddMember("fp_gradient",rapidjson::Value(JsonObject{a,"enabled",_enabled,
        "bridge_correction",_bridgeCorrection,
        "anchor_selection",FirstPassageGradient::anchorSelectionName(_anchorSelection)}),a);
    return value;
}
void ClassicGlobalFpGradientMedium::prepareForRender() {
    if(_enabled && !dynamic_cast<SquaredExponentialCovariance *>(_covariance.get()))
        throw std::runtime_error("classic_global_fp_gradient requires stationary squared_exponential covariance");
    ClassicGlobalMedium::prepareForRender();
    if(_enabled) _kernel=cg::CovarianceKernel(_sigma,_ndf->covariance()/(_sigma*_sigma));
    _collisions=0; _fallbacks=0; _exteriorStarts=0;
}
void ClassicGlobalFpGradientMedium::teardownAfterRender() {
    std::cout<<"GPIS FP-gradient statistics: {\"collisions\":"<<_collisions.load()
        <<",\"sampling_fallbacks\":"<<_fallbacks.load()<<",\"exterior_starts\":"<<_exteriorStarts.load()<<"}\n";
    ClassicGlobalMedium::teardownAfterRender();
}

std::shared_ptr<const FpGradientPathContext> ClassicGlobalFpGradientMedium::context(
        PathSampleGenerator &sampler,const Ray &ray,const MediumState &state) const {
    const auto old=std::dynamic_pointer_cast<const FpGradientPathContext>(state.pathContext);
    const Eigen::Vector3d w=eigen(ray.dir()).normalized();
    if(old && old->owner==this) {
        if(old->rayOrigin!=ray.pos()) throw std::runtime_error("FP gradient path lost its ray origin");
        if((old->direction-w).squaredNorm()<1e-14) return old;
    }
    // No GP observation is supplied by a camera or a turn at ordinary mesh
    // geometry. This explicitly defines a sampled-positive-jet hybrid, NOT
    // the distribution obtained by marginalizing the start at a first hit.
    auto next=std::make_shared<FpGradientPathContext>();
    next->owner=this; next->origin=eigen(ray.pos()); next->direction=w;
    next->rayOrigin=ray.pos();
    const double m=meanValue(Vec3d(ray.pos()));
    cg::Random rng(sampler);
    const double z=-cg::normalQuantileFromLogCdf(cg::normalLogCdf(m/_sigma)+std::log(rng.openUniform01()));
    next->value=std::max(std::nextafter(0.0,1.0),m+_sigma*z);
    const Eigen::Vector3d noise(rng.standardNormal(),rng.standardNormal(),rng.standardNormal());
    next->gradient=eigen(meanGradient(Vec3d(ray.pos())))+_kernel.gradientCovarianceAtZero().llt().matrixL()*noise;
    next->derivative=w.dot(next->gradient);
    _exteriorStarts.fetch_add(1,std::memory_order_relaxed);
    return next;
}

bool ClassicGlobalFpGradientMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,
        MediumState &state,MediumSample &sample) const {
    if(!_enabled) return ClassicGlobalMedium::sampleDistance(sampler,ray,state,sample);
    // All classic proposal/acceptance draws occur before any new random draw.
    // The same incoming sampler state therefore gives exactly the same event.
    if(!sampleDistanceEvent(sampler,ray,state,sample)) return false;
    const auto current=context(sampler,ray,state);
    const double age=current->age+double(sample.t)*eigen(ray.dir()).norm();
    if(sample.exited) {
        auto next=std::make_shared<FpGradientPathContext>(*current);
        next->age=age; next->rayOrigin=sample.p;
        state.pathContext=next;
        return true;
    }
    const Eigen::Vector3d w=current->direction;
    const double gamma=std::sqrt(w.dot(_kernel.metric()*w));
    const Eigen::Vector3d birthMeanGradient=eigen(meanGradient(vec(current->origin)));
    const Eigen::Vector3d endMeanGradient=eigen(meanGradient(Vec3d(sample.p)));
    const FirstPassageGradient::Input input{_sigma,gamma,age,current->value,current->derivative,
        meanValue(vec(current->origin)),w.dot(birthMeanGradient),w.dot(endMeanGradient),
        [&](double s) { return meanValue(vec(current->origin+s*w)); }};
    const FirstPassageGradient::Law law(input,_bridgeCorrection,_anchorSelection);
    cg::Random rng(sampler); bool fallback=false;
    const double derivative=law.sample(rng,&fallback);
    auto event=std::make_shared<FpGradientEventContext>();
    event->owner=this; event->position=eigen(sample.p);
    event->derivative=derivative;
    event->gradient.emplace(_kernel,w,gamma*age,derivative,endMeanGradient,birthMeanGradient,current->gradient);
    sample.eventContext=event;
    if(event->gradient->isDelta()) sample.phase=_deltaPhase.get();
    if(hasDiffuseMicroMaterial()) sample.aniso=vec(event->gradient->sample(rng));
    state.advance(); state.pathContext.reset();
    _collisions.fetch_add(1,std::memory_order_relaxed);
    if(fallback) _fallbacks.fetch_add(1,std::memory_order_relaxed);
    return true;
}

} // namespace Tungsten
