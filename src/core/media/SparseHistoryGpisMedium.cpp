#include "SparseHistoryGpisMedium.hpp"
#include "GpisSdfGrid.hpp"
#include "math/BivariateNormalCdfTable.hpp"
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "math/conditional/mathutility/NumericPolicy.h"
#include "io/JsonObject.hpp"
#include "phasefunctions/ObservedNormalScattering.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace Tungsten {
namespace cg = ConditionalGpis;
namespace sh = SparseHistoryGpis;
namespace {
Eigen::Vector3d eigen(const Vec3f &p) { return {p.x(),p.y(),p.z()}; }
Eigen::Vector3d eigen(const Vec3d &p) { return {p.x(),p.y(),p.z()}; }
Vec3f vec(const Eigen::Vector3d &p) { return Vec3f(float(p.x()),float(p.y()),float(p.z())); }

class LinearSdf final : public cg::MeanField {
    Eigen::Vector3d _reference,_gradient;
    double _minimum;
public:
    explicit LinearSdf(const LinearMean &m)
        : _reference(eigen(m.referencePoint())),_gradient(eigen(m.gradient())),_minimum(m.minimum()) {}
    const char *typeName() const override { return "linear"; }
    cg::MeanJet evaluate(const Eigen::Vector3d &x) const override {
        const double value=_gradient.dot(x-_reference);
        return {std::max(value,_minimum),value<_minimum ? Eigen::Vector3d::Zero().eval() : _gradient};
    }
    double valueDifference(const Eigen::Vector3d &x,const Eigen::Vector3d &d) const override {
        const double a=_gradient.dot(x-_reference),b=a+_gradient.dot(d);
        return a>=_minimum && b>=_minimum ? _gradient.dot(d) : std::max(b,_minimum)-std::max(a,_minimum);
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override {
        Eigen::Vector3d p;
        for (int i=0;i<3;++i) p[i]=_gradient[i]>=0 ? b.minimum[i] : b.maximum[i];
        return {evaluate(p).value,_gradient.norm(),true};
    }
    void appendRayBreakpoints(const Eigen::Vector3d &o,const Eigen::Vector3d &w,double lo,double hi,
                              std::vector<double> &knots) const override {
        const double slope=_gradient.dot(w);
        if (slope!=0) {
            const double t=(_minimum-_gradient.dot(o-_reference))/slope;
            if (t>lo && t<hi) knots.push_back(t);
        }
    }
    cg::MeanRayBounds rayBounds(const Eigen::Vector3d &o,const Eigen::Vector3d &w,double lo,double hi,
                               const SdfRayQueryOptions &options = {}) const override {
        auto result=MeanField::rayBounds(o,w,lo,hi,options);
        std::vector<double> knots;
        appendRayBreakpoints(o,w,lo,hi,knots);
        if (knots.empty()) {
            result.minimumDerivative=result.maximumDerivative=evaluate(o+0.5*(lo+hi)*w).gradient.dot(w);
            if (options.computeBeginDerivative) result.beginDerivative=result.minimumDerivative;
            result.maximumSecondDerivative=0;
        }
        return result;
    }
};

std::shared_ptr<const SparseHistoryPathContext> successor(const SparseHistoryEventContext &event,
    const Eigen::Vector3d &gradient,const Vec3f &outgoing,const Vec3f &position)
{
    auto next=std::make_shared<SparseHistoryPathContext>();
    next->owner=event.owner;
    next->position=event.position;
    next->rayOrigin=position;
    next->start.kind=sh::Start::SurfaceValueGradient;
    next->start.origin=event.position;
    next->start.direction=eigen(outgoing).normalized();
    next->start.gradient=gradient;
    if (!(next->start.direction.dot(gradient)>0))
        throw cg::NumericError(cg::NumericStatus::NeedHigherPrecision,
            "sparse_history_gpis outgoing float direction lost positive-side support");
    return next;
}

class SparseHistoryPhase final : public PhaseFunction {
    const SparseHistoryGpisMedium &_medium;
    bool _delta;
    const SparseHistoryEventContext &event(const Vec3f &wi,const MediumSample &sample) const {
        const auto *result=dynamic_cast<const SparseHistoryEventContext *>(sample.eventContext.get());
        if (!result || result->owner!=&_medium || !result->gradient)
            throw std::runtime_error("sparse_history_gpis: missing collision context");
        if ((eigen(wi).normalized()-result->gradient->incomingDirection()).squaredNorm()>1e-12)
            throw std::runtime_error("sparse_history_gpis phase requires its incoming direction");
        return *result;
    }
public:
    explicit SparseHistoryPhase(const SparseHistoryGpisMedium &medium,bool delta=false)
        : _medium(medium),_delta(delta) {}
    bool isSpecular() const override { return _delta && !_medium.hasDiffuseMicroMaterial(); }
    float pdf(const Vec3f &wi,const Vec3f &wo,const MediumSample &sample) const override {
        if (_medium.hasDiffuseMicroMaterial()) return observedCosinePdf(wo,sample);
        const auto &e=event(wi,sample);
        if (_delta) return 0;
        const float result=float(e.gradient->phasePdf(eigen(wo)));
        if (!(result>=0) || !std::isfinite(result))
            throw std::runtime_error("sparse_history_gpis phase PDF exceeds float range");
        return result;
    }
    Vec3f eval(const Vec3f &wi,const Vec3f &wo,const MediumSample &sample) const override {
        const float density=pdf(wi,wo,sample);
        if (density==0) return Vec3f(0.f);
        if (_medium.hasDiffuseMicroMaterial()) return _medium.reflectance(1)*density;
        const Eigen::Vector3d w=eigen(wi).normalized(),h=(eigen(wo).normalized()-w).normalized();
        return _medium.reflectance(std::abs(w.dot(h)))*density;
    }
    Vec3f evalConnection(PathSampleGenerator &,const Vec3f &wi,const Vec3f &wo,
        const MediumSample &sample,std::shared_ptr<const MediumPathContext> &state) const override {
        const Vec3f value=eval(wi,wo,sample);
        if (value==0.f) return value;
        const auto &e=event(wi,sample);
        if (_medium.hasDiffuseMicroMaterial()) {
            state=successor(e,eigen(sample.aniso),wo,sample.p);
        } else {
            const auto gradient=e.gradient->gradientForDirection(eigen(wo));
            if (!gradient) throw std::runtime_error("sparse_history_gpis connection lost gradient support");
            state=successor(e,*gradient,wo,sample.p);
        }
        return value;
    }
    bool sample(PathSampleGenerator &sampler,const Vec3f &wi,const MediumSample &sample,PhaseSample &result) const override {
        const auto &e=event(wi,sample);
        if (_medium.hasDiffuseMicroMaterial()) {
            result.w=sampleObservedCosine(sampler,sample);
            result.pdf=pdf(wi,result.w,sample);
            result.weight=_medium.reflectance(1);
            if (!(result.pdf>0)) return false;
            result.pathContext=successor(e,eigen(sample.aniso),result.w,sample.p);
            return true;
        }
        cg::Random rng(sampler);
        const Eigen::Vector3d gradient=e.gradient->sample(rng);
        const Eigen::Vector3d w=e.gradient->incomingDirection(),normal=gradient.normalized();
        result.w=vec(w-2*w.dot(normal)*normal).normalized();
        result.weight=_medium.reflectance(-w.dot(normal));
        result.pdf=_delta ? 1.f : float(e.gradient->phasePdfForGradient(gradient));
        if (!(result.pdf>=0) || !std::isfinite(result.pdf))
            throw std::runtime_error("sparse_history_gpis sampled phase PDF exceeds float range");
        result.pathContext=successor(e,gradient,result.w,sample.p);
        return true;
    }
};

double physicalLength(const Ray &ray)
{
    const double speed=eigen(ray.dir()).norm();
    if (!(speed>0) || !std::isfinite(speed) || !std::isfinite(ray.farT()) || ray.farT()<0)
        throw std::invalid_argument("sparse_history_gpis requires a finite boundary-enclosed ray");
    const double length=double(ray.farT())*speed;
    if (!std::isfinite(length)) throw std::invalid_argument("sparse_history_gpis ray length overflow");
    return length;
}
} // namespace

SparseHistoryGpisMedium::SparseHistoryGpisMedium()
{
    _phaseFunction=std::make_shared<SparseHistoryPhase>(*this);
    _deltaPhase=std::make_shared<SparseHistoryPhase>(*this,true);
}

void SparseHistoryGpisMedium::fromJson(JsonPtr value,const Scene &scene)
{
    ClassicGlobalMedium::fromJson(value,scene);
    if (value["tracking_step"])
        value.parseError("sparse_history_gpis uses sparse_history integration settings, not tracking_step");
    _settings=sh::FlightSettings{};
    _probabilityMethod="numeric";
    _cdfTableFile.reset();
    if (auto options=value["sparse_history"]) {
        options.getField("points",_settings.historyPoints);
        options.getField("max_step_over_length",_settings.maxStepOverLength);
        options.getField("max_mean_range_over_sigma",_settings.maxMeanRangeOverSigma);
        options.getField("optical_depth_tolerance",_settings.opticalDepthTolerance);
        options.getField("distance_tolerance",_settings.distanceTolerance);
        options.getField("max_subdivisions",_settings.maxSubdivisions);
        options.getField("probability_absolute_tolerance",_settings.probability.absoluteTolerance);
        options.getField("probability_relative_tolerance",_settings.probability.relativeTolerance);
        options.getField("probability_initial_samples",_settings.probability.initialSamples);
        options.getField("probability_max_samples",_settings.probability.maxSamples);
        options.getField("probability_method",_probabilityMethod);
        if(_probabilityMethod!="numeric" && _probabilityMethod!="lut")
            options.parseError("probability_method must be numeric or lut");
        if(_probabilityMethod=="lut")
            _cdfTableFile=scene.fetchResource(options.getRequiredMember("cdf_table"));
        else if(options["cdf_table"])
            options.parseError("cdf_table requires probability_method=lut");
    }
    try { _settings.validate(); }
    catch (const std::exception &error) { value.parseError(error.what()); }
}

rapidjson::Value SparseHistoryGpisMedium::toJson(Allocator &allocator) const
{
    auto value=ClassicGlobalMedium::toJson(allocator);
    value["type"].SetString("sparse_history_gpis",allocator);
    value.RemoveMember("tracking_step");
    value.AddMember("sparse_history",rapidjson::Value(JsonObject{allocator,
        "points",_settings.historyPoints,
        "max_step_over_length",_settings.maxStepOverLength,
        "max_mean_range_over_sigma",_settings.maxMeanRangeOverSigma,
        "optical_depth_tolerance",_settings.opticalDepthTolerance,
        "distance_tolerance",_settings.distanceTolerance,
        "max_subdivisions",_settings.maxSubdivisions,
        "probability_absolute_tolerance",_settings.probability.absoluteTolerance,
        "probability_relative_tolerance",_settings.probability.relativeTolerance,
        "probability_initial_samples",_settings.probability.initialSamples,
        "probability_max_samples",_settings.probability.maxSamples,
        "probability_method",_probabilityMethod}),allocator);
    if(_cdfTableFile)
        value["sparse_history"].AddMember("cdf_table",JsonUtils::toJson(*_cdfTableFile,allocator),allocator);
    return value;
}

void SparseHistoryGpisMedium::loadResources()
{
    ClassicGlobalMedium::loadResources();
    if(_cdfTableFile) {
        _settings.probability.bivariateTable=sh::BivariateNormalCdfTable::loadShared(_cdfTableFile->absolute().asString());
        _settings.probability.statistics=std::make_shared<sh::ProbabilityStatistics>();
    }
}

void SparseHistoryGpisMedium::prepareForRender()
{
    _settings.validate();
    if(_probabilityMethod=="lut" && !_settings.probability.bivariateTable)
        throw std::runtime_error("sparse_history_gpis CDF table resource was not loaded");
    if(_settings.probability.statistics) {
        _settings.probability.statistics->tableAttempts.store(0,std::memory_order_relaxed);
        _settings.probability.statistics->tableAccepted.store(0,std::memory_order_relaxed);
    }
    _discardedSurfacePaths.store(0,std::memory_order_relaxed);
    _discardedSurfaceConnections.store(0,std::memory_order_relaxed);
    const auto *kernel=dynamic_cast<const SquaredExponentialCovariance *>(_covariance.get());
    if (!kernel || !kernel->isStationaryKernel() || !kernel->isIsotropic())
        throw std::invalid_argument("sparse_history_gpis requires an isotropic stationary squared_exponential covariance");
    ClassicGlobalMedium::prepareForRender();
    _field.kernel=cg::CovarianceKernel(_sigma,_ndf->covariance()/(_sigma*_sigma));
    _field.activeDomain={Eigen::Vector3d::Constant(-1e30),Eigen::Vector3d::Constant(1e30)};
    if (_sdfGrid) {
#if OPENVDB_AVAILABLE
        _field.mean=std::shared_ptr<const cg::MeanField>(_sdfGrid.get(),[](const cg::MeanField *){});
#endif
    } else if (const auto *sphere=dynamic_cast<const SphericalMean *>(_mean.get())) {
        _field.mean=std::make_shared<cg::SphereMean>(eigen(sphere->center()),sphere->radius());
    } else if (const auto *linear=dynamic_cast<const LinearMean *>(_mean.get())) {
        _field.mean=std::make_shared<LinearSdf>(*linear);
    } else if (dynamic_cast<const HomogeneousMean *>(_mean.get())) {
        _field.mean=std::make_shared<cg::ConstantMean>(meanValue(Vec3d(0.0)));
    } else if (_mean->sdfQueries()) {
        _field.mean=std::make_shared<cg::SdfMeanField>(_mean);
    } else {
        throw std::invalid_argument("sparse_history_gpis requires an analytic SDF, trilinear VDB, or procedural SdfQueries");
    }
    _field.validate();
}

void SparseHistoryGpisMedium::teardownAfterRender()
{
    if(const auto &statistics=_settings.probability.statistics) {
        const auto attempts=statistics->tableAttempts.load(std::memory_order_relaxed);
        const auto accepted=statistics->tableAccepted.load(std::memory_order_relaxed);
        std::cout<<"GPIS sparse-history CDF table: {\"accepted\":"<<accepted
                 <<",\"numerical_fallback\":"<<(attempts-accepted)<<"}\n";
    }
    std::cout<<"GPIS sparse-history discarded surface births: {\"paths\":"<<discardedSurfacePaths()
             <<",\"shadow_connections\":"<<discardedSurfaceConnections()<<"}\n";
    ClassicGlobalMedium::teardownAfterRender();
}

std::shared_ptr<const SparseHistoryPathContext> SparseHistoryGpisMedium::context(
    const Ray &ray,const MediumState &state) const
{
    const Eigen::Vector3d direction=cg::normalizedOrThrow(eigen(ray.dir()));
    const auto old=std::dynamic_pointer_cast<const SparseHistoryPathContext>(state.pathContext);
    if (old && old->owner==this && old->rayOrigin!=ray.pos())
        throw std::invalid_argument("sparse_history_gpis segment lost its precise origin");
    if (old && old->owner==this && (direction-old->start.direction).squaredNorm()<1e-14)
        return old;
    auto next=std::make_shared<SparseHistoryPathContext>();
    next->owner=this;
    next->position=old && old->owner==this ? old->position : eigen(ray.pos());
    next->rayOrigin=ray.pos();
    next->start.origin=next->position;
    next->start.direction=direction;
    if (old && old->owner==this && old->age==0 && old->start.kind==sh::Start::SurfaceValueGradient) {
        next->start.kind=sh::Start::SurfaceValueGradient;
        next->start.gradient=old->start.gradient;
        if (!(direction.dot(next->start.gradient)>0))
            throw cg::NumericError(cg::NumericStatus::NonPositiveSurfaceBirth,
                "sparse_history_gpis surface birth points into the solid side");
    }
    // A turn at an ordinary geometry surface supplies no GP zero observation.
    // It starts a new exterior-positive flight. Straight forward surfaces above
    // retain the existing start, physical age, history and remaining clock.
    return next;
}

bool SparseHistoryGpisMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,
    MediumState &state,MediumSample &sample) const
try {
    if (state.bounce>_maxBounce) return false;
    const double length=physicalLength(ray),speed=eigen(ray.dir()).norm();
    const auto current=context(ray,state);
    sample=MediumSample{};
    sample.t=sample.continuedT=ray.farT();
    sample.p=ray.hitpoint();
    sample.exited=true;
    sample.weight=sample.continuedWeight=Vec3f(1.f);
    sample.emission=Vec3f(0.f);
    sample.pdf=1;
    sample.phase=_phaseFunction.get();
    sample.rayInfo=state.info;
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    auto next=std::make_shared<SparseHistoryPathContext>(*current);
    if (length==0) { state.pathContext=next; return true; }
    sh::Flight flight(_field,current->start,_settings);
    cg::Random rng(sampler);
    const double target=current->residualOpticalDepth ? *current->residualOpticalDepth : -std::log(rng.openUniform01());
    const auto hit=flight.sampleDistance(current->age,current->age+length,target);
    if (hit.hit) {
        const auto law=flight.law(hit.distance);
        const double inwardSpeed=law.sampleSpeed(rng);
        auto event=std::make_shared<SparseHistoryEventContext>();
        event->owner=this;
        event->position=current->start.origin+hit.distance*current->start.direction;
        const std::optional<Eigen::Vector3d> birthGradient=current->start.kind==sh::Start::SurfaceValueGradient
            ? std::optional<Eigen::Vector3d>(current->start.gradient) : std::nullopt;
        event->gradient.emplace(_field.kernel,current->start.direction,hit.distance/flight.lengthScale(),
            -inwardSpeed,law.meanGradient(),flight.meanGradient(0),birthGradient);
        sample.t=float((hit.distance-current->age)/speed);
        if (sample.t>=ray.farT()) sample.t=std::nextafter(ray.farT(),0.f);
        sample.continuedT=sample.t;
        sample.p=vec(event->position);
        sample.exited=false;
        sample.eventContext=event;
        if (event->gradient->isDelta()) sample.phase=_deltaPhase.get();
        if (hasDiffuseMicroMaterial()) {
            const auto gradient=event->gradient->sample(rng);
            sample.aniso=Vec3d(gradient.x(),gradient.y(),gradient.z());
        }
        state.advance();
        // The phase installs the outgoing birth observation. Neither visibility
        // nor continuation is allowed to reuse the incoming flight after a hit.
        state.pathContext.reset();
    } else {
        next->age=current->age+length;
        next->position=current->start.origin+next->age*current->start.direction;
        next->rayOrigin=sample.p;
        next->residualOpticalDepth=std::max(0.0,target-hit.traversedOpticalDepth);
        state.pathContext=next;
    }
    return true;
} catch (const cg::NumericError &error) {
    if (error.status()!=cg::NumericStatus::NonPositiveSurfaceBirth) throw;
    _discardedSurfacePaths.fetch_add(1,std::memory_order_relaxed);
    return false;
}

Vec3f SparseHistoryGpisMedium::transmittance(PathSampleGenerator &,const Ray &ray,bool,bool,MediumState *state) const
try {
    if (!state) throw std::invalid_argument("sparse_history_gpis transmittance requires path state");
    const double length=physicalLength(ray);
    const auto current=context(ray,*state);
    auto next=std::make_shared<SparseHistoryPathContext>(*current);
    double opticalDepth=0;
    if (length>0) {
        sh::Flight flight(_field,current->start,_settings);
        opticalDepth=flight.opticalDepth(current->age,current->age+length);
    }
    if (!(opticalDepth>=0) || !std::isfinite(opticalDepth))
        throw std::runtime_error("sparse_history_gpis has invalid shadow optical depth");
    next->age=current->age+length;
    next->position=current->start.origin+next->age*current->start.direction;
    next->rayOrigin=ray.hitpoint();
    // Shadow traversal carries observations but has no sampled distance clock.
    next->residualOpticalDepth.reset();
    state->pathContext=next;
    return Vec3f(float(std::exp(-opticalDepth)));
} catch (const cg::NumericError &error) {
    if (error.status()!=cg::NumericStatus::NonPositiveSurfaceBirth) throw;
    _discardedSurfaceConnections.fetch_add(1,std::memory_order_relaxed);
    return Vec3f(0.f);
}

float SparseHistoryGpisMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const
{
    throw std::runtime_error("sparse_history_gpis supports unidirectional path tracing; a state-free distance PDF is undefined");
}
Vec3f SparseHistoryGpisMedium::sigmaS(Vec3f) const
{
    throw std::runtime_error("sparse_history_gpis extinction requires the flight state and direction");
}

} // namespace Tungsten
