#include "SingleHistoryGpisMedium.hpp"
#include "GpisSdfGrid.hpp"
#include "math/BivariateNormalCdfTable.hpp"
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "math/conditional/mathutility/NumericPolicy.h"
#include "io/JsonObject.hpp"
#include "phasefunctions/ObservedNormalScattering.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

namespace Tungsten {
namespace cg = ConditionalGpis;
namespace sh = SparseHistoryGpis;
namespace ih = SingleHistoryGpis;
namespace {
using Clock = std::chrono::steady_clock;
double elapsed(Clock::time_point begin) {
    return std::chrono::duration<double>(Clock::now()-begin).count();
}
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

std::shared_ptr<const SingleHistoryPathContext> successor(const SingleHistoryEventContext &event,
    const Eigen::Vector3d &gradient,const Vec3f &outgoing,const Vec3f &position)
{
    auto next=std::make_shared<SingleHistoryPathContext>();
    next->owner=event.owner;
    next->position=event.position;
    next->rayOrigin=position;
    next->start.kind=sh::Start::SurfaceValueGradient;
    next->start.origin=event.position;
    next->start.direction=eigen(outgoing).normalized();
    next->start.gradient=gradient;
    if (!(next->start.direction.dot(gradient)>0))
        throw cg::NumericError(cg::NumericStatus::NeedHigherPrecision,
            "single_history_gpis outgoing float direction lost positive-side support");
    return next;
}

class SingleHistoryPhase final : public PhaseFunction {
    const SingleHistoryGpisMedium &_medium;
    bool _delta;
    const SingleHistoryEventContext &event(const Vec3f &wi,const MediumSample &sample) const {
        const auto *result=dynamic_cast<const SingleHistoryEventContext *>(sample.eventContext.get());
        if (!result || result->owner!=&_medium || !result->gradient)
            throw std::runtime_error("single_history_gpis: missing collision context");
        if ((eigen(wi).normalized()-result->gradient->incomingDirection()).squaredNorm()>1e-12)
            throw std::runtime_error("single_history_gpis phase requires its incoming direction");
        return *result;
    }
public:
    explicit SingleHistoryPhase(const SingleHistoryGpisMedium &medium,bool delta=false)
        : _medium(medium),_delta(delta) {}
    bool isSpecular() const override { return _delta && !_medium.hasDiffuseMicroMaterial(); }
    float pdf(const Vec3f &wi,const Vec3f &wo,const MediumSample &sample) const override {
        if (_medium.hasDiffuseMicroMaterial()) return observedCosinePdf(wo,sample);
        const auto &e=event(wi,sample);
        if (_delta) return 0;
        const float result=float(e.gradient->phasePdf(eigen(wo)));
        if (!(result>=0) || !std::isfinite(result))
            throw std::runtime_error("single_history_gpis phase PDF exceeds float range");
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
            if (!gradient) throw std::runtime_error("single_history_gpis connection lost gradient support");
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
            throw std::runtime_error("single_history_gpis sampled phase PDF exceeds float range");
        result.pathContext=successor(e,gradient,result.w,sample.p);
        return true;
    }
};

double physicalLength(const Ray &ray)
{
    const double speed=eigen(ray.dir()).norm();
    if (!(speed>0) || !std::isfinite(speed) || !std::isfinite(ray.farT()) || ray.farT()<0)
        throw std::invalid_argument("single_history_gpis requires a finite boundary-enclosed ray");
    const double length=double(ray.farT())*speed;
    if (!std::isfinite(length)) throw std::invalid_argument("single_history_gpis ray length overflow");
    return length;
}
} // namespace

SingleHistoryGpisMedium::SingleHistoryGpisMedium()
{
    _phaseFunction=std::make_shared<SingleHistoryPhase>(*this);
    _deltaPhase=std::make_shared<SingleHistoryPhase>(*this,true);
}

void SingleHistoryGpisMedium::fromJson(JsonPtr value,const Scene &scene)
{
    ClassicGlobalMedium::fromJson(value,scene);
    if (value["tracking_step"])
        value.parseError("single_history_gpis uses single_history tracking settings, not tracking_step");
    if (value["sparse_history"])
        value.parseError("single_history_gpis uses the single_history settings object");
    _settings=ih::Settings{};
    _probabilityMethod="numeric";
    _cdfTableFile.reset();
    if (auto options=value["single_history"]) {
        std::string trackingMethod="delta";
        options.getField("tracking_method",trackingMethod);
        if(trackingMethod=="delta") _settings.trackingMethod=ih::TrackingMethod::Delta;
        else if(trackingMethod=="inversion") _settings.trackingMethod=ih::TrackingMethod::Inversion;
        else if(trackingMethod=="regular") _settings.trackingMethod=ih::TrackingMethod::Regular;
        else options.parseError("tracking_method must be delta, inversion or regular");
        options.getField("regular_subdivisions",_settings.regularSubdivisions);
        options.getField("selection_step_over_length",_settings.selectionStepOverLength);
        options.getField("history_groups",_settings.historyGroups);
        options.getField("history_enabled",_settings.historyEnabled);
        options.getField("shadow_roulette_threshold",_settings.shadowRouletteThreshold);
        options.getField("max_work",_settings.maxWork);
        options.getField("profile",_settings.profile);
        options.getField("profile_detail",_settings.profileDetail);
        options.getField("normalized_bound_only",_settings.normalizedBoundOnly);
        options.getField("far_bound_start_scale",_settings.farBoundStartScale);
        options.getField("tighten_expected_events",_settings.tightenExpectedEvents);
        options.getField("macrocell_size",_settings.macrocellSize);
        options.getField("macrocell_max_expected_events",_settings.macrocellMaxExpectedEvents);
        options.getField("candidate_macro_pruning",_settings.candidateMacroPruning);
        std::string candidateCollection="fine";
        options.getField("candidate_collection",candidateCollection);
        if(candidateCollection=="fine") _settings.candidateCollection=ih::CandidateCollection::Fine;
        else if(candidateCollection=="macro") _settings.candidateCollection=ih::CandidateCollection::Macro;
        else options.parseError("candidate_collection must be fine or macro");
        options.getField("candidate_macrocell_size",_settings.candidateMacrocellSize);
        options.getField("inversion_max_step_over_length",_settings.inversionMaxStepOverLength);
        options.getField("inversion_mean_range_over_sigma",_settings.inversionMeanRangeOverSigma);
        options.getField("inversion_absolute_tolerance",_settings.inversionAbsoluteTolerance);
        options.getField("inversion_relative_tolerance",_settings.inversionRelativeTolerance);
        options.getField("inversion_distance_tolerance_over_length",_settings.inversionDistanceToleranceOverLength);
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

rapidjson::Value SingleHistoryGpisMedium::toJson(Allocator &allocator) const
{
    auto value=ClassicGlobalMedium::toJson(allocator);
    value["type"].SetString("single_history_gpis",allocator);
    value.RemoveMember("tracking_step");
    value.AddMember("single_history",rapidjson::Value(JsonObject{allocator,
        "tracking_method",_settings.trackingMethod==ih::TrackingMethod::Regular ? "regular" :
            (_settings.trackingMethod==ih::TrackingMethod::Inversion ? "inversion":"delta"),
        "regular_subdivisions",_settings.regularSubdivisions,
        "selection_step_over_length",_settings.selectionStepOverLength,
        "history_groups",_settings.historyGroups,
        "history_enabled",_settings.historyEnabled,
        "shadow_roulette_threshold",_settings.shadowRouletteThreshold,
        "max_work",_settings.maxWork,
        "profile",_settings.profile,
        "profile_detail",_settings.profileDetail,
        "normalized_bound_only",_settings.normalizedBoundOnly,
        "far_bound_start_scale",_settings.farBoundStartScale,
        "tighten_expected_events",_settings.tightenExpectedEvents,
        "macrocell_size",_settings.macrocellSize,
        "macrocell_max_expected_events",_settings.macrocellMaxExpectedEvents,
        "candidate_macro_pruning",_settings.candidateMacroPruning,
        "candidate_collection",_settings.candidateCollection==ih::CandidateCollection::Macro ? "macro":"fine",
        "candidate_macrocell_size",_settings.candidateMacrocellSize,
        "inversion_max_step_over_length",_settings.inversionMaxStepOverLength,
        "inversion_mean_range_over_sigma",_settings.inversionMeanRangeOverSigma,
        "inversion_absolute_tolerance",_settings.inversionAbsoluteTolerance,
        "inversion_relative_tolerance",_settings.inversionRelativeTolerance,
        "inversion_distance_tolerance_over_length",_settings.inversionDistanceToleranceOverLength,
        "probability_absolute_tolerance",_settings.probability.absoluteTolerance,
        "probability_relative_tolerance",_settings.probability.relativeTolerance,
        "probability_initial_samples",_settings.probability.initialSamples,
        "probability_max_samples",_settings.probability.maxSamples,
        "probability_method",_probabilityMethod}),allocator);
    if(_cdfTableFile)
        value["single_history"].AddMember("cdf_table",JsonUtils::toJson(*_cdfTableFile,allocator),allocator);
    return value;
}

void SingleHistoryGpisMedium::loadResources()
{
    ClassicGlobalMedium::loadResources();
    if(_cdfTableFile) {
        _settings.probability.bivariateTable=sh::BivariateNormalCdfTable::loadShared(_cdfTableFile->absolute().asString());
        _settings.probability.statistics=std::make_shared<sh::ProbabilityStatistics>();
    }
}

void SingleHistoryGpisMedium::prepareForRender()
{
    _settings.validate();
    if(_probabilityMethod=="lut" && !_settings.probability.bivariateTable)
        throw std::runtime_error("single_history_gpis CDF table resource was not loaded");
    if(_settings.probability.statistics) {
        _settings.probability.statistics->tableAttempts.store(0,std::memory_order_relaxed);
        _settings.probability.statistics->tableAccepted.store(0,std::memory_order_relaxed);
    }
    _discardedSurfacePaths.store(0,std::memory_order_relaxed);
    _discardedSurfaceConnections.store(0,std::memory_order_relaxed);
    for (auto &p:_profile)
        for (auto *counter:{&p.calls,&p.nanoseconds,&p.speedNanoseconds,
                &p.selectionNanoseconds,&p.majorantNanoseconds,&p.probabilityNanoseconds,
                &p.quadratureNanoseconds,&p.hazardEvaluations,&p.inversionRefinements,
                &p.intervals,&p.candidates,&p.selections,&p.cells,
                &p.rouletteTests,&p.rouletteTerminations,
                &p.boundaryNanoseconds,
                &p.collectNanoseconds,
                &p.groupNanoseconds,
                &p.candidateStatisticsNanoseconds,
                &p.scoringNanoseconds,
                &p.baselineSetupNanoseconds,
                &p.historyFactorNanoseconds,
                &p.conditionalBoundNanoseconds,
                &p.exteriorBoundsNanoseconds,
                &p.exteriorAlgebraNanoseconds,
                &p.localLawNanoseconds,
                &p.hazardNanoseconds,
                &p.processedIntervals,
                &p.emptyIntervals,
                &p.acceptedCollisions,
                &p.rejectedCandidates,
                &p.zeroHazardCandidates,
                &p.majorantCalls,
                &p.eventSplits,
                &p.candidateStatistics,
                &p.scoreEvaluations,&p.macroBoundNanoseconds,
                &p.macroAttempts,&p.macroAccepted,&p.macroRejected,
                &p.macroUnavailable,&p.macroRetrySkips,
                &p.candidateMacroBoundNanoseconds,&p.candidateMacroQueries,
                &p.candidateMacroPrunedFragments,&p.candidateMacroUnprunedFragments,
                &p.candidateMacroUnavailable,
                &p.fineBoundaryQueries,&p.macroBoundaryQueries,&p.macroCandidateFragments})
            counter->store(0,std::memory_order_relaxed);
    const auto *kernel=dynamic_cast<const SquaredExponentialCovariance *>(_covariance.get());
    if (!kernel || !kernel->isStationaryKernel())
        throw std::invalid_argument("single_history_gpis requires a stationary squared_exponential covariance");
    ClassicGlobalMedium::prepareForRender();
    _field.kernel=cg::CovarianceKernel(_sigma,_ndf->covariance()/(_sigma*_sigma));
    _field.activeDomain={Eigen::Vector3d::Constant(-1e30),Eigen::Vector3d::Constant(1e30)};
    if (_sdfGrid) {
#if OPENVDB_AVAILABLE
        _sdfGrid->buildMacrocells(_settings.trackingMethod==ih::TrackingMethod::Delta
            ? _settings.macrocellSize : 0);
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
        throw std::invalid_argument("single_history_gpis requires an analytic SDF, trilinear VDB, or procedural SdfQueries");
    }
    _field.validate();
}

void SingleHistoryGpisMedium::recordWork(bool surface,bool shadow,double seconds,
    double speedSeconds,const ih::Work &work) const
{
    if (!_settings.profile) return;
    auto &p=_profile[(surface ? 2:0)+(shadow ? 1:0)];
    const auto add=[](std::atomic<std::uint64_t> &target,std::uint64_t value) {
        target.fetch_add(value,std::memory_order_relaxed);
    };
    const auto nanos=[](double value) { return static_cast<std::uint64_t>(value*1e9); };
    add(p.calls,1); add(p.nanoseconds,nanos(seconds)); add(p.speedNanoseconds,nanos(speedSeconds));
    add(p.selectionNanoseconds,nanos(work.selectionSeconds));
    add(p.majorantNanoseconds,nanos(work.majorantSeconds));
    add(p.probabilityNanoseconds,nanos(work.probabilitySeconds));
    add(p.quadratureNanoseconds,nanos(work.quadratureSeconds));
    add(p.hazardEvaluations,work.hazardEvaluations);add(p.inversionRefinements,work.inversionRefinements);
    add(p.intervals,work.intervals); add(p.candidates,work.candidates);
    add(p.selections,work.selections); add(p.cells,work.cells);
    add(p.rouletteTests,work.rouletteTests); add(p.rouletteTerminations,work.rouletteTerminations);
    if (_settings.profileDetail) {
        add(p.boundaryNanoseconds,nanos(work.boundarySeconds));
        add(p.collectNanoseconds,nanos(work.collectSeconds));
        add(p.groupNanoseconds,nanos(work.groupSeconds));
        add(p.candidateStatisticsNanoseconds,nanos(work.candidateStatisticsSeconds));
        add(p.scoringNanoseconds,nanos(work.scoringSeconds));
        add(p.baselineSetupNanoseconds,nanos(work.baselineSetupSeconds));
        add(p.historyFactorNanoseconds,nanos(work.historyFactorSeconds));
        add(p.conditionalBoundNanoseconds,nanos(work.conditionalBoundSeconds));
        add(p.exteriorBoundsNanoseconds,nanos(work.exteriorBoundsSeconds));
        add(p.exteriorAlgebraNanoseconds,nanos(work.exteriorAlgebraSeconds));
        add(p.localLawNanoseconds,nanos(work.localLawSeconds));
        add(p.hazardNanoseconds,nanos(work.hazardSeconds));
        add(p.processedIntervals,work.processedIntervals);
        add(p.emptyIntervals,work.emptyIntervals);
        add(p.acceptedCollisions,work.acceptedCollisions);
        add(p.rejectedCandidates,work.rejectedCandidates);
        add(p.zeroHazardCandidates,work.zeroHazardCandidates);
        add(p.majorantCalls,work.majorantCalls);
        add(p.eventSplits,work.eventSplits);
        add(p.candidateStatistics,work.candidateStatistics);
        add(p.scoreEvaluations,work.scoreEvaluations);
        add(p.macroBoundNanoseconds,nanos(work.macroBoundSeconds));
        add(p.macroAttempts,work.macroAttempts);
        add(p.macroAccepted,work.macroAccepted);
        add(p.macroRejected,work.macroRejected);
        add(p.macroUnavailable,work.macroUnavailable);
        add(p.macroRetrySkips,work.macroRetrySkips);
        add(p.candidateMacroBoundNanoseconds,nanos(work.candidateMacroBoundSeconds));
        add(p.candidateMacroQueries,work.candidateMacroQueries);
        add(p.candidateMacroPrunedFragments,work.candidateMacroPrunedFragments);
        add(p.candidateMacroUnprunedFragments,work.candidateMacroUnprunedFragments);
        add(p.candidateMacroUnavailable,work.candidateMacroUnavailable);
        add(p.fineBoundaryQueries,work.fineBoundaryQueries);
        add(p.macroBoundaryQueries,work.macroBoundaryQueries);
        add(p.macroCandidateFragments,work.macroCandidateFragments);
    }
}

void SingleHistoryGpisMedium::teardownAfterRender()
{
    if(const auto &statistics=_settings.probability.statistics) {
        const auto attempts=statistics->tableAttempts.load(std::memory_order_relaxed);
        const auto accepted=statistics->tableAccepted.load(std::memory_order_relaxed);
        std::cout<<"GPIS single-history CDF table: {\"accepted\":"<<accepted
                 <<",\"numerical_fallback\":"<<(attempts-accepted)<<"}\n";
    }
    if (_settings.profile) {
        // Seconds are summed worker time. Exterior includes camera entry and
        // ordinary geometry turns; only GP crossings enter the surface group.
        const char *names[]={"exterior_path","exterior_shadow","surface_path","surface_shadow"};
        const auto count=[](const std::atomic<std::uint64_t> &value) {
            return value.load(std::memory_order_relaxed);
        };
        std::cout<<"GPIS single-history profile: {";
        for (std::size_t i=0;i<_profile.size();++i) {
            const auto &p=_profile[i];
            if (i) std::cout<<',';
            std::cout<<'\"'<<names[i]<<"\":{\"calls\":"<<count(p.calls)
                <<",\"total_seconds\":"<<double(count(p.nanoseconds))*1e-9
                <<",\"speed_seconds\":"<<double(count(p.speedNanoseconds))*1e-9
                <<",\"selection_seconds\":"<<double(count(p.selectionNanoseconds))*1e-9
                <<",\"majorant_seconds\":"<<double(count(p.majorantNanoseconds))*1e-9
                <<",\"probability_seconds\":"<<double(count(p.probabilityNanoseconds))*1e-9
                <<",\"quadrature_seconds\":"<<double(count(p.quadratureNanoseconds))*1e-9
                <<",\"hazard_evaluations\":"<<count(p.hazardEvaluations)
                <<",\"inversion_refinements\":"<<count(p.inversionRefinements)
                <<",\"intervals\":"<<count(p.intervals)<<",\"candidates\":"<<count(p.candidates)
                <<",\"selections\":"<<count(p.selections)<<",\"cells\":"<<count(p.cells)
                <<",\"roulette_tests\":"<<count(p.rouletteTests)
                <<",\"roulette_terminations\":"<<count(p.rouletteTerminations);
            if (_settings.profileDetail) {
                std::cout
                    <<",\"boundary_seconds\":"<<double(count(p.boundaryNanoseconds))*1e-9
                    <<",\"collect_seconds\":"<<double(count(p.collectNanoseconds))*1e-9
                    <<",\"group_seconds\":"<<double(count(p.groupNanoseconds))*1e-9
                    <<",\"candidate_statistics_seconds\":"<<double(count(p.candidateStatisticsNanoseconds))*1e-9
                    <<",\"scoring_seconds\":"<<double(count(p.scoringNanoseconds))*1e-9
                    <<",\"baseline_setup_seconds\":"<<double(count(p.baselineSetupNanoseconds))*1e-9
                    <<",\"history_factor_seconds\":"<<double(count(p.historyFactorNanoseconds))*1e-9
                    <<",\"conditional_bound_seconds\":"<<double(count(p.conditionalBoundNanoseconds))*1e-9
                    <<",\"exterior_bounds_seconds\":"<<double(count(p.exteriorBoundsNanoseconds))*1e-9
                    <<",\"exterior_algebra_seconds\":"<<double(count(p.exteriorAlgebraNanoseconds))*1e-9
                    <<",\"local_law_seconds\":"<<double(count(p.localLawNanoseconds))*1e-9
                    <<",\"hazard_seconds\":"<<double(count(p.hazardNanoseconds))*1e-9
                    <<",\"processed_intervals\":"<<count(p.processedIntervals)
                    <<",\"empty_intervals\":"<<count(p.emptyIntervals)
                    <<",\"accepted_collisions\":"<<count(p.acceptedCollisions)
                    <<",\"rejected_candidates\":"<<count(p.rejectedCandidates)
                    <<",\"zero_hazard_candidates\":"<<count(p.zeroHazardCandidates)
                    <<",\"majorant_calls\":"<<count(p.majorantCalls)
                    <<",\"event_splits\":"<<count(p.eventSplits)
                    <<",\"candidate_statistics\":"<<count(p.candidateStatistics)
                    <<",\"score_evaluations\":"<<count(p.scoreEvaluations)
                    <<",\"macro_bound_seconds\":"<<double(count(p.macroBoundNanoseconds))*1e-9
                    <<",\"macro_attempts\":"<<count(p.macroAttempts)
                    <<",\"macro_accepted\":"<<count(p.macroAccepted)
                    <<",\"macro_rejected\":"<<count(p.macroRejected)
                    <<",\"macro_unavailable\":"<<count(p.macroUnavailable)
                    <<",\"macro_retry_skips\":"<<count(p.macroRetrySkips)
                    <<",\"candidate_macro_bound_seconds\":"<<double(count(p.candidateMacroBoundNanoseconds))*1e-9
                    <<",\"candidate_macro_queries\":"<<count(p.candidateMacroQueries)
                    <<",\"candidate_macro_pruned_fragments\":"<<count(p.candidateMacroPrunedFragments)
                    <<",\"candidate_macro_unpruned_fragments\":"<<count(p.candidateMacroUnprunedFragments)
                    <<",\"candidate_macro_unavailable\":"<<count(p.candidateMacroUnavailable)
                    <<",\"fine_boundary_queries\":"<<count(p.fineBoundaryQueries)
                    <<",\"macro_boundary_queries\":"<<count(p.macroBoundaryQueries)
                    <<",\"macro_candidate_fragments\":"<<count(p.macroCandidateFragments);
            }
            std::cout<<'}';
        }
        std::cout<<"}\n";
    }
    std::cout<<"GPIS single-history discarded surface births: {\"paths\":"<<discardedSurfacePaths()
             <<",\"shadow_connections\":"<<discardedSurfaceConnections()<<"}\n";
    ClassicGlobalMedium::teardownAfterRender();
}

std::shared_ptr<const SingleHistoryPathContext> SingleHistoryGpisMedium::context(
    const Ray &ray,const MediumState &state) const
{
    const Eigen::Vector3d direction=cg::normalizedOrThrow(eigen(ray.dir()));
    const auto old=std::dynamic_pointer_cast<const SingleHistoryPathContext>(state.pathContext);
    if (old && old->owner==this && old->rayOrigin!=ray.pos())
        throw std::invalid_argument("single_history_gpis segment lost its precise origin");
    if (old && old->owner==this && (direction-old->start.direction).squaredNorm()<1e-14)
        return old;
    auto next=std::make_shared<SingleHistoryPathContext>();
    next->owner=this;
    next->position=old && old->owner==this ? old->position : eigen(ray.pos());
    next->rayOrigin=ray.pos();
    next->start.origin=next->position;
    next->start.direction=direction;
    if (old && old->owner==this && old->snapshot.age==0 && old->start.kind==sh::Start::SurfaceValueGradient) {
        next->start.kind=sh::Start::SurfaceValueGradient;
        next->start.gradient=old->start.gradient;
        if (!(direction.dot(next->start.gradient)>0))
            throw cg::NumericError(cg::NumericStatus::NonPositiveSurfaceBirth,
                "single_history_gpis surface birth points into the solid side");
    }
    // A turn at an ordinary geometry surface supplies no GP zero observation.
    // It starts a new exterior-positive flight. Straight forward surfaces above
    // retain the existing start, physical age, history and proposal clock.
    return next;
}

bool SingleHistoryGpisMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,
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
    auto next=std::make_shared<SingleHistoryPathContext>(*current);
    if (length==0) { state.pathContext=next; return true; }
    const auto begin=_settings.profile ? Clock::now():Clock::time_point{};
    ih::Flight flight(_field,current->start,_settings,current->snapshot);
    cg::Random rng(sampler);
    const auto hit=flight.sampleDistance(current->snapshot.age+length,rng);
    double speedSeconds=0;
    if (hit.hit) {
        if (!hit.law) throw std::runtime_error("single_history_gpis collision lost its accepted local law");
        const auto speedBegin=_settings.profile ? Clock::now():Clock::time_point{};
        const double inwardSpeed=hit.law->sampleSpeed(rng);
        if (_settings.profile) speedSeconds=elapsed(speedBegin);
        auto event=std::make_shared<SingleHistoryEventContext>();
        event->owner=this;
        event->position=current->start.origin+hit.distance*current->start.direction;
        const std::optional<Eigen::Vector3d> birthGradient=current->start.kind==sh::Start::SurfaceValueGradient
            ? std::optional<Eigen::Vector3d>(current->start.gradient) : std::nullopt;
        event->gradient.emplace(_field.kernel,current->start.direction,hit.distance/flight.lengthScale(),
            -inwardSpeed,hit.law->meanGradient(),flight.meanGradient(0),birthGradient);
        sample.t=float((hit.distance-current->snapshot.age)/speed);
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
        next->snapshot=flight.snapshot();
        next->position=current->start.origin+next->snapshot.age*current->start.direction;
        next->rayOrigin=sample.p;
        state.pathContext=next;
    }
    if (_settings.profile)
        recordWork(current->start.kind==sh::Start::SurfaceValueGradient,false,elapsed(begin),speedSeconds,flight.work());
    return true;
} catch (const cg::NumericError &error) {
    if (error.status()!=cg::NumericStatus::NonPositiveSurfaceBirth) throw;
    _discardedSurfacePaths.fetch_add(1,std::memory_order_relaxed);
    return false;
}

Vec3f SingleHistoryGpisMedium::transmittance(PathSampleGenerator &sampler,const Ray &ray,bool,bool,MediumState *state) const
try {
    if (!state) throw std::invalid_argument("single_history_gpis transmittance requires path state");
    const double length=physicalLength(ray);
    const auto current=context(ray,*state);
    auto next=std::make_shared<SingleHistoryPathContext>(*current);
    if (length==0) { state->pathContext=next; return Vec3f(1.f); }
    const auto begin=_settings.profile ? Clock::now():Clock::time_point{};
    // A visibility branch starts its own Poisson proposals. Geometry and the
    // selected history remain a value copy of the incoming branch state.
    auto snapshot=current->snapshot;
    snapshot.clock={};
    snapshot.inversionRemaining=-1;
    ih::Flight flight(_field,current->start,_settings,snapshot);
    cg::Random rng(sampler);
    const double weight=flight.transmittance(snapshot.age+length,rng);
    if (!(weight>=0 && weight<=1) || !std::isfinite(weight))
        throw std::runtime_error("single_history_gpis has an invalid shadow weight");
    if (weight>0) {
        next->snapshot=flight.snapshot();
        next->snapshot.clock={};
        next->snapshot.inversionRemaining=-1;
        next->position=current->start.origin+next->snapshot.age*current->start.direction;
        next->rayOrigin=ray.hitpoint();
        state->pathContext=next;
    } else state->pathContext.reset();
    if (_settings.profile)
        recordWork(current->start.kind==sh::Start::SurfaceValueGradient,true,elapsed(begin),0,flight.work());
    return Vec3f(float(weight));
} catch (const cg::NumericError &error) {
    if (error.status()!=cg::NumericStatus::NonPositiveSurfaceBirth) throw;
    _discardedSurfaceConnections.fetch_add(1,std::memory_order_relaxed);
    return Vec3f(0.f);
}

float SingleHistoryGpisMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const
{
    throw std::runtime_error("single_history_gpis supports unidirectional path tracing; a state-free distance PDF is undefined");
}
Vec3f SingleHistoryGpisMedium::sigmaS(Vec3f) const
{
    throw std::runtime_error("single_history_gpis extinction requires the flight state and direction");
}

} // namespace Tungsten
