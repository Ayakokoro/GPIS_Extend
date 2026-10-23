#include "GlobalConditionalMedium.hpp"
#include "GpisSdfGrid.hpp"
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "math/conditional/transport/CollisionGradientSampler.h"
#include "math/conditional/transport/DirectionalGradient.h"
#include "math/ConditionalProfile.hpp"
#include "io/JsonObject.hpp"
#include <iostream>

namespace Tungsten {
namespace cg = ConditionalGpis;
namespace {
Eigen::Vector3d eigen(const Vec3f &p) { return {p.x(),p.y(),p.z()}; }
Eigen::Vector3d eigen(const Vec3d &p) { return {p.x(),p.y(),p.z()}; }
Vec3f vec(const Eigen::Vector3d &p) { return Vec3f(float(p.x()),float(p.y()),float(p.z())); }
class LinearSdf final : public cg::MeanField {
    Eigen::Vector3d _ref,_g;
    double _minimum;
public:
    explicit LinearSdf(const LinearMean &m) : _ref(eigen(m.referencePoint())),_g(eigen(m.gradient())),_minimum(m.minimum()) {}
    const char *typeName() const override { return "linear"; }
    cg::MeanJet evaluate(const Eigen::Vector3d &x) const override {
        double f=_g.dot(x-_ref);
        return {std::max(f,_minimum), f<_minimum ? Eigen::Vector3d::Zero().eval():_g};
    }
    double valueDifference(const Eigen::Vector3d &x,const Eigen::Vector3d &d) const override {
        double a=_g.dot(x-_ref),b=a+_g.dot(d);
        return a>=_minimum&&b>=_minimum ? _g.dot(d):std::max(b,_minimum)-std::max(a,_minimum);
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override {
        Eigen::Vector3d x;
        for(int i=0;i<3;++i) x[i]=_g[i]>=0?b.minimum[i]:b.maximum[i];
        return {evaluate(x).value,_g.norm(),true};
    }
    void appendRayBreakpoints(const Eigen::Vector3d &o,const Eigen::Vector3d &w,double lo,double hi,std::vector<double> &knots) const override {
        double slope=_g.dot(w);
        if(slope!=0) { double t=(_minimum-_g.dot(o-_ref))/slope; if(t>lo&&t<hi) knots.push_back(t); }
    }
    cg::MeanRayBounds rayBounds(const Eigen::Vector3d &o,const Eigen::Vector3d &w,double lo,double hi,
                               const SdfRayQueryOptions &options = {}) const override {
        auto r=MeanField::rayBounds(o,w,lo,hi,options);
        std::vector<double> knots; appendRayBreakpoints(o,w,lo,hi,knots);
        if(knots.empty()) {
            r.minimumDerivative=r.maximumDerivative=evaluate(o+0.5*(lo+hi)*w).gradient.dot(w);
            if(options.computeBeginDerivative) r.beginDerivative=r.minimumDerivative;
            r.maximumSecondDerivative=0;
        }
        return r;
    }
};
std::shared_ptr<const ConditionalPathContext> successor(const ConditionalEventContext &e,
        const Eigen::Vector3d &g,const Vec3f &wo,const Vec3f &p)
{
    auto c=std::make_shared<ConditionalPathContext>();
    c->owner=e.owner; c->position=e.position; c->rayOrigin=p;
    c->birth.birthPosition=e.position; c->birth.birthValue=0;
    c->birth.birthGradient=g; c->birth.direction=eigen(wo).normalized();
    return c;
}
// Continuation and light connections use the same joint distribution of
// reflection direction and unnormalized collision gradient.
class ConditionalPhase : public PhaseFunction {
    const GlobalConditionalMedium &_medium;
    const ConditionalEventContext &event(const MediumSample &s) const {
        const auto *e=dynamic_cast<const ConditionalEventContext *>(s.eventContext.get());
        if(!e || e->owner!=&_medium) throw cg::NumericError(cg::NumericStatus::InvalidInput,"missing conditional collision state");
        return *e;
    }
public:
    explicit ConditionalPhase(const GlobalConditionalMedium &m):_medium(m) {}
    float pdf(const Vec3f &wi,const Vec3f &wo,const MediumSample &s) const override {
        cg::Profile::Scope profile(cg::Profile::Stage::PhasePdf,cg::Profile::Domain::Phase);
        const auto &e=event(s);
        try {
            const float result=float(ClassicGlobal::GaussianNdf(e.gradient.covariance).phasePdf(e.gradient.mean,eigen(wi).normalized(),eigen(wo).normalized()));
            if(!(result>=0) || !std::isfinite(result))
                throw cg::NumericError(cg::NumericStatus::NeedHigherPrecision,"conditional phase PDF overflow");
            return result;
        } catch(const std::runtime_error &error) {
            throw cg::NumericError(cg::NumericStatus::InvalidCovariance,error.what());
        }
    }
    Vec3f eval(const Vec3f &wi,const Vec3f &wo,const MediumSample &s) const override {
        cg::Profile::Scope profile(cg::Profile::Stage::PhaseEval,cg::Profile::Domain::Phase);
        const Eigen::Vector3d h=eigen(wo).normalized()-eigen(wi).normalized();
        if(h.squaredNorm()==0) return Vec3f(0.0f);
        return _medium.reflectance(std::abs(eigen(wi).normalized().dot(h.normalized())))*pdf(wi,wo,s);
    }
    Vec3f evalConnection(PathSampleGenerator &sampler,const Vec3f &wi,const Vec3f &wo,
            const MediumSample &s,std::shared_ptr<const MediumPathContext> &state) const override {
        cg::Profile::Scope profile(cg::Profile::Stage::DirectionalSample,cg::Profile::Domain::Connection);
        const auto &e=event(s);
        const Vec3f value=eval(wi,wo,s);
        if(value==0.0f) return value;
        const Eigen::Vector3d h=(eigen(wo).normalized()-eigen(wi).normalized()).normalized();
        cg::Random rng(sampler);
        const Eigen::Vector3d g=cg::sampleDirectionalGradient(e.gradient,h,rng);
        state=successor(e,g,wo,s.p);
        return value;
    }
    bool sample(PathSampleGenerator &sampler,const Vec3f &wi,const MediumSample &s,PhaseSample &r) const override {
        cg::Profile::Scope profile(cg::Profile::Stage::PhaseSample,cg::Profile::Domain::Phase);
        const auto &e=event(s); cg::Random rng(sampler);
        const Eigen::Vector3d w=eigen(wi).normalized();
        const Eigen::Vector3d g=cg::sampleFluxWeightedGradient(e.gradient,w,rng);
        const Eigen::Vector3d n=g.normalized();
        r.w=vec(w-2*w.dot(n)*n).normalized();
        if(!(eigen(r.w).dot(g)>0))
            throw cg::NumericError(cg::NumericStatus::NeedHigherPrecision,"reflected float direction lost positive-side support");
        r.weight=_medium.reflectance(std::abs(w.dot(n)));
        r.pdf=pdf(wi,r.w,s);
        r.pathContext=successor(e,g,r.w,s.p);
        return true;
    }
};
}
GlobalConditionalMedium::GlobalConditionalMedium() {
    _phaseFunction=std::make_shared<ConditionalPhase>(*this);
}
void GlobalConditionalMedium::fromJson(JsonPtr value,const Scene &scene) {
    ClassicGlobalMedium::fromJson(value,scene);
    _tracking=cg::ConditionalTrackingSettings{};
    std::string majorantMode="adaptive";
    value.getField("tracking_majorant",majorantMode);
    if(majorantMode=="two_constants") _tracking.majorantMode=cg::ConditionalMajorantMode::TwoConstants;
    else if(majorantMode!="adaptive") value.parseError("tracking_majorant must be adaptive or two_constants");
    value.getField("tracking_constant_birth_length",_tracking.constantBirthLength);
    _tracking.step=_trackingStep;
    value.getField("tracking_step_scale",_tracking.stepScale);
    value.getField("tracking_age_scale",_tracking.ageScale);
    value.getField("tracking_max_expected_events",_tracking.maxExpectedEvents);
    value.getField("tracking_tighten_expected_events",_tracking.tightenExpectedEvents);
    value.getField("tracking_macrocell_size",_tracking.macrocellSize);
    value.getField("tracking_macrocell_max_expected_events",_tracking.macrocellMaxExpectedEvents);
    try { _tracking.validate(); }
    catch(const cg::NumericError &error) { value.parseError(error.what()); }
}
rapidjson::Value GlobalConditionalMedium::toJson(Allocator &allocator) const {
    auto value=ClassicGlobalMedium::toJson(allocator);
    value["type"].SetString("global_conditional",allocator);
    value.AddMember("tracking_majorant",rapidjson::Value(
        _tracking.majorantMode==cg::ConditionalMajorantMode::TwoConstants ? "two_constants" : "adaptive",allocator),allocator);
    value.AddMember("tracking_constant_birth_length",_tracking.constantBirthLength,allocator);
    value.AddMember("tracking_step_scale",_tracking.stepScale,allocator);
    value.AddMember("tracking_age_scale",_tracking.ageScale,allocator);
    value.AddMember("tracking_max_expected_events",_tracking.maxExpectedEvents,allocator);
    value.AddMember("tracking_tighten_expected_events",_tracking.tightenExpectedEvents,allocator);
    value.AddMember("tracking_macrocell_size",_tracking.macrocellSize,allocator);
    value.AddMember("tracking_macrocell_max_expected_events",_tracking.macrocellMaxExpectedEvents,allocator);
    return value;
}
void GlobalConditionalMedium::prepareForRender() {
    _discardedSurfacePaths.store(0,std::memory_order_relaxed);
    _discardedSurfaceConnections.store(0,std::memory_order_relaxed);
    if(!dynamic_cast<SquaredExponentialCovariance *>(_covariance.get()))
        throw std::runtime_error("global_conditional requires a stationary squared_exponential covariance");
    ClassicGlobalMedium::prepareForRender();
    _field.kernel=cg::CovarianceKernel(_sigma,_ndf->covariance()/(_sigma*_sigma));
    _field.activeDomain={Eigen::Vector3d::Constant(-1e30),Eigen::Vector3d::Constant(1e30)};
    if(_sdfGrid) {
#if OPENVDB_AVAILABLE
        // The base medium owns the immutable SDF for the whole render lifetime.
        _sdfGrid->buildMacrocells(_tracking.majorantMode==cg::ConditionalMajorantMode::Adaptive ? _tracking.macrocellSize : 0);
        _field.mean=std::shared_ptr<const cg::MeanField>(_sdfGrid.get(),[](const cg::MeanField *){});
#endif
    } else if(auto sphere=dynamic_cast<SphericalMean *>(_mean.get()))
        _field.mean=std::make_shared<cg::SphereMean>(eigen(sphere->center()),sphere->radius());
    else if(auto linear=dynamic_cast<LinearMean *>(_mean.get()))
        _field.mean=std::make_shared<LinearSdf>(*linear);
    else if(dynamic_cast<HomogeneousMean *>(_mean.get()))
        _field.mean=std::make_shared<cg::ConstantMean>(meanValue(Vec3d(0.0)));
    else if(_mean->sdfQueries())
        _field.mean=std::make_shared<cg::SdfMeanField>(_mean);
    else throw std::runtime_error("global_conditional: this scene mean does not provide certified SDF queries");
    if(_tracking.majorantMode==cg::ConditionalMajorantMode::TwoConstants && !_field.mean->globalMagnitudeBounds())
        throw std::runtime_error("tracking_majorant two_constants requires a clamped VDB SDF or constant mean with global bounds");
}
void GlobalConditionalMedium::teardownAfterRender() {
    std::cout<<"GPIS conditional discarded surface births: {\"paths\":"<<discardedSurfacePaths()
             <<",\"shadow_connections\":"<<discardedSurfaceConnections()<<"}\n";
    ClassicGlobalMedium::teardownAfterRender();
}
std::shared_ptr<const ConditionalPathContext> GlobalConditionalMedium::context(
        PathSampleGenerator &sampler,const Ray &ray,const MediumState &s) const
{
    const auto old=std::dynamic_pointer_cast<const ConditionalPathContext>(s.pathContext);
    if(old&&old->owner==this) {
        if(old->rayOrigin!=ray.pos()) throw cg::NumericError(cg::NumericStatus::InvalidInput,"conditional segment origin lost its precise position");
        return old;
    }
    cg::Random rng(sampler);
    auto c=std::make_shared<ConditionalPathContext>();
    c->owner=this; c->rayOrigin=ray.pos(); c->position=eigen(ray.pos());
    c->birth=cg::startConditionalExterior(_field,c->position,eigen(ray.dir()).normalized(),rng);
    return c;
}
bool GlobalConditionalMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,MediumState &state,MediumSample &sample) const
try
{
    cg::Profile::Scope profile(cg::Profile::Stage::Medium,cg::Profile::Domain::Path);
    if(state.bounce>_maxBounce) return false;
    if(!std::isfinite(ray.farT()) || ray.farT()<0)
        throw cg::NumericError(cg::NumericStatus::InvalidInput,"global_conditional needs a finite boundary-enclosed segment");
    const auto c=context(sampler,ray,state);
    const double speed=eigen(ray.dir()).norm();
    const Eigen::Vector3d w=eigen(ray.dir())/speed;
    // Geometry nearT suppresses self-intersections only. It must not cut out
    // part of the medium next to a collision or a transparent boundary.
    cg::ConditionalSegment segment(_field,c->birth,c->position,w,ray.farT()*speed,_tracking);
    cg::Random rng(sampler); double t;
    const bool escaped=segment.track(rng,true,t)!=0;
    const Eigen::Vector3d p=c->position+t*w;
    sample.t=float(t/speed); sample.continuedT=sample.t;
    sample.p=escaped ? ray.hitpoint():vec(p);
    sample.weight=sample.continuedWeight=Vec3f(1.0f); sample.emission=Vec3f(0.0f);
    sample.pdf=1; sample.exited=escaped; sample.phase=_phaseFunction.get();
    sample.rayInfo=state.info;
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    sample.eventContext.reset();
    auto next=std::make_shared<ConditionalPathContext>(*c);
    next->position=p; next->rayOrigin=sample.p;
    if(escaped) {
        // A geometry hit is an exterior GP point, not a zero crossing. Renew
        // before surface NEE and BSDF continuation copy this observation.
        next->birth=segment.samplePositiveEndpoint(rng);
    } else {
        auto e=std::make_shared<ConditionalEventContext>();
        e->owner=this; e->position=p; e->gradient=segment.gradient(t);
        sample.eventContext=e;
        state.advance();
    }
    state.pathContext=next;
    return true;
} catch(const cg::NumericError &error) {
    if(error.status()!=cg::NumericStatus::NonPositiveSurfaceBirth) throw;
    // A deliberate absorbing approximation at unsupported surface births.
    // Returning false preserves radiance already accumulated by the integrator;
    // do not resample, move the ray, or swallow unrelated numerical failures.
    _discardedSurfacePaths.fetch_add(1,std::memory_order_relaxed);
    return false;
}
Vec3f GlobalConditionalMedium::transmittance(PathSampleGenerator &sampler,const Ray &ray,bool,bool,MediumState *state) const
try
{
    cg::Profile::Scope profile(cg::Profile::Stage::Medium,cg::Profile::Domain::Shadow);
    if(!state) throw cg::NumericError(cg::NumericStatus::InvalidInput,"conditional transmittance requires path state");
    const auto c=context(sampler,ray,*state);
    const double speed=eigen(ray.dir()).norm();
    const Eigen::Vector3d w=eigen(ray.dir())/speed;
    cg::ConditionalSegment segment(_field,c->birth,c->position,w,ray.farT()*speed,_tracking);
    cg::Random rng(sampler); double t;
    const double tr=segment.track(rng,false,t);
    auto next=std::make_shared<ConditionalPathContext>(*c);
    next->position=c->position+double(ray.farT())*speed*w; next->rayOrigin=ray.hitpoint();
    // Shadow traversal uses the same endpoint renewal as distance sampling,
    // including forward surfaces inside the same medium.
    if(tr>0) next->birth=segment.samplePositiveEndpoint(rng);
    state->pathContext=next;
    return Vec3f(float(tr));
} catch(const cg::NumericError &error) {
    if(error.status()!=cg::NumericStatus::NonPositiveSurfaceBirth) throw;
    // Apply the same termination rule to NEE/visibility branches.
    _discardedSurfaceConnections.fetch_add(1,std::memory_order_relaxed);
    return Vec3f(0.0f);
}
float GlobalConditionalMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const {
    throw std::runtime_error("global_conditional supports unidirectional path tracing; a state-free distance PDF is undefined");
}
Vec3f GlobalConditionalMedium::sigmaS(Vec3f) const {
    throw std::runtime_error("global_conditional extinction requires the path observation and direction");
}
Vec3f GlobalConditionalMedium::sigmaT(Vec3f p) const { return sigmaS(p); }
}
