#include "ConditionalSegment.h"
#include "NonstationarySegment.h"
#include "media/ClassicLocalGrid.hpp"
#include "MajorantOpticalDepth.h"
#include "SinglePointNeural.h"
#include "math/ConditionalProfile.hpp"
#include "math/ConditionalRayProfile.hpp"
#include "ConditionalMajorantBounds.h"
#include <boost/container/small_vector.hpp>
#include <sstream>
#include <iomanip>
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
#include <atomic>
#include <cstdlib>
#include <iostream>
#endif

namespace Tungsten::ConditionalGpis {
namespace {
FlightState alongDirection(FlightState birth,const Vector3 &direction)
{
    birth.direction=direction;
    birth.age=0;
    return birth;
}
}
void ConditionalTrackingSettings::validate() const
{
    if (!(chapter2ParentEvents>=0) || !std::isfinite(chapter2ParentEvents) ||
        (chapter2ParentEvents>0 && !(chapter2StartScale>0)))
        throw NumericError(NumericStatus::InvalidInput,"chapter 2 parent budget requires enabled chapter 2 tracking");
    if(!(chapter2StartScale>=0 && chapter2FallbackEvents>=0) ||
       !std::isfinite(chapter2StartScale) || !std::isfinite(chapter2FallbackEvents) ||
       (chapter2StartScale>0 && !macrocellBounds))
        throw NumericError(NumericStatus::InvalidInput,"chapter 2 tracking requires finite nonnegative parameters and macrocell bounds");
    if (classicLocalBand && exteriorCutoffSigma>0)
        throw NumericError(NumericStatus::InvalidInput,"select either baked classic_local support or a one-sided exterior cutoff");
    if (!(exteriorCutoffSigma>=0) || !std::isfinite(exteriorCutoffSigma))
        throw NumericError(NumericStatus::InvalidInput,"exterior cutoff must be finite and nonnegative");
    if (!(classicStartScale>=0) || !std::isfinite(classicStartScale))
        throw NumericError(NumericStatus::InvalidInput,"tracking_classic_start_scale must be finite and nonnegative");
    if (!(shadowRouletteThreshold>=0 && shadowRouletteThreshold<=1) || !std::isfinite(shadowRouletteThreshold))
        throw NumericError(NumericStatus::InvalidInput,"tracking_shadow_roulette_threshold must be finite and between zero and one");
    if (!std::isfinite(farBoundStartScale) || (farBoundStartScale!=0 && farBoundStartScale<2))
        throw NumericError(NumericStatus::InvalidInput,"tracking_far_bound_start_scale must be zero or at least two");
    if (!std::isfinite(simpleBoundStartScale) || (simpleBoundStartScale!=0 && simpleBoundStartScale<2))
        throw NumericError(NumericStatus::InvalidInput,"tracking_simple_bound_start_scale must be zero or at least two");
    if (!std::isfinite(globalFarStartScale) || (globalFarStartScale!=0 && globalFarStartScale<2))
        throw NumericError(NumericStatus::InvalidInput,"tracking_global_far_start_scale must be zero or at least two");
    if (globalFarStartScale>0 && majorantMode!=ConditionalMajorantMode::Adaptive)
        throw NumericError(NumericStatus::InvalidInput,"global far tracking requires adaptive near tracking");
    if (!std::isfinite(localFarStartScale) || (localFarStartScale!=0 && localFarStartScale<2))
        throw NumericError(NumericStatus::InvalidInput,"tracking_local_far_start_scale must be zero or at least two");
    if(localFarStartScale>0 && (globalFarStartScale>0 || majorantMode!=ConditionalMajorantMode::Adaptive))
        throw NumericError(NumericStatus::InvalidInput,"local far tracking requires adaptive near tracking and excludes global far mode");
    if((transitionPointBounds || transitionIntegralBounds || coupledBounds) && majorantMode!=ConditionalMajorantMode::Adaptive)
        throw NumericError(NumericStatus::InvalidInput,"transition midpoint bounds require adaptive tracking");
    if(!(coupledLocalScale>=0) || !std::isfinite(coupledLocalScale) || !(proposalScale>0) || !std::isfinite(proposalScale))
        throw NumericError(NumericStatus::InvalidInput,"coupled local scale and proposal scale must be finite and nonnegative/positive");
    if(farIntegralBounds && !(localFarStartScale>=2))
        throw NumericError(NumericStatus::InvalidInput,"integrated far bounds require local far tracking");
    if(transitionVoxelBound!=TransitionVoxelBound::Disabled && (!coupledBounds ||
       (transitionVoxelBound!=TransitionVoxelBound::Bernstein5 && transitionMaxExpectedEvents!=0)))
        throw NumericError(NumericStatus::InvalidInput,"voxel transition requires coupled bounds; range/midpoint diagnostics require subdivision disabled");
    if(coupledBounds && (transitionPointBounds || transitionIntegralBounds))
        throw NumericError(NumericStatus::InvalidInput,"coupled bounds exclude legacy transition bounds");
    if(!(transitionMaxExpectedEvents>=0) || !std::isfinite(transitionMaxExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"tracking_transition_max_expected_events must be finite and nonnegative (zero disables transition subdivision)");
    if(majorantMode!=ConditionalMajorantMode::Adaptive && majorantMode!=ConditionalMajorantMode::TwoConstants)
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional majorant mode");
    if(majorantMode==ConditionalMajorantMode::TwoConstants && (geometryStep || pointBounds || simpleBoundStartScale>0))
        throw NumericError(NumericStatus::InvalidInput,"point/Lipschitz options require adaptive tracking");
    if(!(constantBirthLength>=0) || !std::isfinite(constantBirthLength))
        throw NumericError(NumericStatus::InvalidInput,"tracking_constant_birth_length must be finite and nonnegative");
    if(!(step>=0) || !std::isfinite(step))
        throw NumericError(NumericStatus::InvalidInput,"tracking_step must be finite and nonnegative");
    if(!(stepScale>0) || !std::isfinite(stepScale))
        throw NumericError(NumericStatus::InvalidInput,"tracking_step_scale must be finite and positive");
    if(!(ageScale>0) || !std::isfinite(ageScale))
        throw NumericError(NumericStatus::InvalidInput,"tracking_age_scale must be finite and positive");
    if(!(maxExpectedEvents>0) || !std::isfinite(maxExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"tracking_max_expected_events must be finite and positive");
    if(!(tightenExpectedEvents>=0) || !std::isfinite(tightenExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"tracking_tighten_expected_events must be finite and nonnegative");
    if(macrocellSize!=0 && (macrocellSize<2 || macrocellSize>64 || (macrocellSize&(macrocellSize-1))))
        throw NumericError(NumericStatus::InvalidInput,"tracking_macrocell_size must be 0 or a power of two from 2 to 64");
    if(!(macrocellMinExpectedEvents>=0) || !std::isfinite(macrocellMinExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"macrocell minimum event cost must be finite and nonnegative");
    if(!(macrocellStartScale>=0) || !std::isfinite(macrocellStartScale) ||
       (macrocellBounds && (!macrocellSize || !coupledBounds || farIntegralBounds)))
        throw NumericError(NumericStatus::InvalidInput,"macrocell bounds require a block size, coupled constants and a nonnegative start scale");
    if(!(macrocellMaxExpectedEvents>0) || !std::isfinite(macrocellMaxExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"tracking_macrocell_max_expected_events must be finite and positive");
    if(!(macrocellFarStartScale>=0) || !std::isfinite(macrocellFarStartScale))
        throw NumericError(NumericStatus::InvalidInput,"tracking_macrocell_far_start_scale must be finite and nonnegative");
    if(!(macrocellFarMaxExpectedEvents>=0) || !std::isfinite(macrocellFarMaxExpectedEvents) ||
       (macrocellFarMaxExpectedEvents>0 && macrocellFarMaxExpectedEvents<macrocellMaxExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"tracking_macrocell_far_max_expected_events must be zero or at least the near threshold");
}
ConditionalSegment::ConditionalSegment(const GPSSField &f,const FlightState &s,
        const Point3 &o,const Vector3 &w,double length,const ConditionalTrackingSettings &tracking,
        const SinglePointNeuralModel *neuralModel,double meanLipschitz,double meanGradientBound)
    : _field(f),_origin(o),_w(normalizedOrThrow(w)),_length(length),
      _age((o-s.birthPosition).dot(_w)),_inverseEll(0),
      _meanLipschitz(meanLipschitz),
      _meanGradientBound(std::isnan(meanGradientBound) ? meanLipschitz : meanGradientBound),_tracking(tracking),
      _neuralModel(neuralModel)
{
    _tracking.validate();
    if (f.nonstationary) {
        if (neuralModel || tracking.classicLocalBand || tracking.exteriorCutoffSigma>0 ||
            tracking.majorantMode!=ConditionalMajorantMode::Adaptive)
            throw NumericError(NumericStatus::InvalidInput,"proc_nonstationary requires full-domain numerical conditional tracking");
        const auto domain=f.activeDomain.intersect({s.birthPosition,_w});
        if (!domain.hit || _age+length>domain.exit)
            throw NumericError(NumericStatus::InvalidInput,"conditional segment exceeds the field domain");
        const double step=tracking.step>0 ? tracking.step : tracking.stepScale*f.nonstationary->minLength();
        _varying=std::make_unique<NonstationarySegment>(*f.nonstationary,*f.mean,s,o,_w,length,step);
        return;
    }
    _inverseEll=std::sqrt(_w.dot(f.kernel.precision()*_w));
    _straight.emplace(f,alongDirection(s,_w),tracking.normalizedBoundOnly,tracking.farBoundStartScale,
                      tracking.simpleBoundStartScale,tracking.tabulatedBounds);
    if(_tracking.exteriorCutoffSigma>0)
        _exteriorCutoff=_tracking.exteriorCutoffSigma*_field.kernel.sigma();
    if ((_tracking.geometryStep || _tracking.pointBounds || _tracking.simpleBoundStartScale>0 || _tracking.localFarStartScale>0 || _tracking.transitionPointBounds || _tracking.transitionIntegralBounds || _tracking.coupledBounds) &&
        (!(_meanLipschitz>=0) || !std::isfinite(_meanLipschitz)))
        throw NumericError(NumericStatus::InvalidInput,"point/Lipschitz tracking needs a finite world-space mean Lipschitz bound");
    if(_tracking.localFarStartScale>0 && (!(_meanGradientBound>=0) || !std::isfinite(_meanGradientBound)))
        throw NumericError(NumericStatus::InvalidInput,"local far tracking needs a finite world-space gradient bound");
    if(_tracking.geometryStep || _tracking.pointBounds)
        _classicBegin=_tracking.classicStartScale/_inverseEll;
    if (!(length>=0) || !std::isfinite(length))
        throw NumericError(NumericStatus::InvalidInput,"conditional segment must be finite");
    const Vector3 d=o-s.birthPosition;
    // Permit world-position subtraction roundoff when numerically subdividing
    // one flight, but require a fresh observation before changing its line.
    const double tolerance=32*std::numeric_limits<double>::epsilon()*
        (d.norm()+o.norm()+s.birthPosition.norm());
    if(!(_age>=-tolerance) || !((d-_age*_w).norm()<=tolerance))
        throw NumericError(NumericStatus::InvalidInput,
            "conditional segment requires an updated observation before changing direction");
    _age=std::max(0.0,_age);
    if(!std::isfinite(_age+length) || _age+length>_straight->maximumAgeInDomain())
        throw NumericError(NumericStatus::InvalidInput,"conditional segment exceeds the field domain");
    if(_tracking.globalFarStartScale>0) {
        if(!_field.mean->globalMagnitudeBounds())
            throw NumericError(NumericStatus::InvalidInput,"global far tracking needs a globally bounded mean");
        using I=MajorantBounds::Interval;
        const I inverseLength2(_w.dot(_field.kernel.precision()*_w));
        const double begin=(I(_tracking.globalFarStartScale)/MajorantBounds::sqrt(inverseLength2)).upper;
        _globalFarBegin=std::max(0.0,(I(begin)-I(_age)).upper);
    }
    if(_tracking.localFarStartScale>0) {
        using I=MajorantBounds::Interval;
        const I inverseLength2(_w.dot(_field.kernel.precision()*_w));
        const double begin=(I(_tracking.localFarStartScale)/MajorantBounds::sqrt(inverseLength2)*
            I(1+16*std::numeric_limits<double>::epsilon())).upper;
        _localFarBegin=std::max(0.0,(I(begin)-I(_age)).upper);
    }
    {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
    RayProfile::BoundScope birthProfile(0,length,true);
#endif
    if(_tracking.majorantMode==ConditionalMajorantMode::TwoConstants && _tracking.constantBirthLength>0) {
        const double split=std::min(_age+length,_tracking.constantBirthLength);
        // Keep the certified analytic birth interval, then bound its extension
        // before sampling. The maximum of the covering intervals is one valid
        // constant for the entire prefix; a Taylor certificate cannot simply
        // be extrapolated through grid knots. No observation is renewed here.
        const auto prefix=_straight->twoSegmentMajorants(split);
        _birthBound.split=split;
        _birthBound.maximum=std::max(prefix.nearMaximum,prefix.farMaximum);
    } else _birthBound=_straight->birthMajorant(_age+length,
        _tracking.coupledBounds ? _meanLipschitz : std::numeric_limits<double>::infinity());
    }
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
    if(auto *r=RayProfile::active()) {
        r->age=_age;r->effectiveDirection=_w;r->ell=1/_inverseEll;
        r->split=_birthBound.split;r->far=_localFarBegin;
    }
#endif
    if(_tracking.majorantMode==ConditionalMajorantMode::TwoConstants && _age+length>_birthBound.split)
        _tailMaximum=_straight->constantTailMajorant(_birthBound.split);
    if(_neuralModel) {
        _hazardEnvelope=_neuralModel->cap();
        _neuralFlight=std::make_unique<SinglePointNeuralFlight>(*_neuralModel,f,alongDirection(s,_w),*_straight);
    }
}
ConditionalSegment::~ConditionalSegment() = default;
Gaussian<3> ConditionalSegment::gradient(double t) const
{
    if (_varying) return _varying->gradient(t);
    return _straight->hitStatistics(_age+t).collisionGradient;
}
ConditionalScalarStatistics ConditionalSegment::statistics(double t) const
{
    if (_varying) return _varying->statistics(t);
    return _straight->scalarStatistics(_age+t);
}
double ConditionalSegment::hazard(double t) const
{
    return _varying ? _varying->hazard(t) : correctedHazard(t,nullptr);
}
double ConditionalSegment::correctedHazard(double t,ConditionalTrackingWork *work,bool knownInside) const
{
    if (_tracking.classicLocalBand) {
        Profile::Scope point(Profile::Stage::TrackBandPoint);
        if(!_tracking.classicLocalBand->hasDensity(_origin+t*_w)) return 0;
    }
    if(!knownInside && _tracking.exteriorCutoffSigma>0 && _straight->scalarMeanValue(_age+t)>_exteriorCutoff)
        return 0;
    double result=_straight->evaluate(_age+t).hazard.value;
    if(_neuralFlight && result>0) {
        const auto correction=_neuralFlight->correction(_age+t,*_straight);
        if(!(correction.ratio>=0 && correction.ratio<=_hazardEnvelope) || !std::isfinite(correction.ratio))
            throw NumericError(NumericStatus::InvalidMajorant,"neural correction exceeds its declared cap");
        result*=correction.ratio;
        if(work) {
            work->networkQueries+=correction.networkEvaluated;
            work->correctionCapHits+=correction.capped;
            work->neuralOutsideDomain+=correction.outsideDomain;
            work->neuralBaselineFallbacks+=!correction.networkEvaluated;
            work->neuralOutsideFeatureRange+=correction.outsideFeatureRange;
        }
    }
    if(!(result>=0) || !std::isfinite(result))
        throw NumericError(NumericStatus::NeedHigherPrecision,"nonfinite conditional hazard");
    return result;
}
double ConditionalSegment::majorant(double lo,double hi,double parentMaximum) const
{
    if (_varying) throw NumericError(NumericStatus::InvalidInput,"numerical nonstationary tracking does not provide a majorant");
    return _hazardEnvelope*baselineMajorant(lo,hi,parentMaximum/_hazardEnvelope);
}
double ConditionalSegment::baselineMajorant(double lo,double hi,double parentMaximum,bool permitMacro) const
{
    const bool macro=permitMacro && _tracking.macrocellBounds && _age+lo>=_tracking.macrocellStartScale/
        _inverseEll;
    if(lo>=_localFarBegin && _tracking.farIntegralBounds)
        return std::min(parentMaximum,_straight->farIntegralMajorant(_age+lo,_age+hi,_meanLipschitz,_meanGradientBound).value(_age+hi));
    if(lo>=_localFarBegin)
        return _straight->localFarIntervalMajorant(_age+lo,_age+hi,_meanLipschitz,_meanGradientBound,parentMaximum,_tracking.coupledBounds,macro);
    if(hi>_localFarBegin)
        return std::max(baselineMajorant(lo,_localFarBegin,parentMaximum,permitMacro),
                        baselineMajorant(_localFarBegin,hi,parentMaximum,permitMacro));
    if(lo>=_globalFarBegin) return std::min(parentMaximum,globalFarMaximum());
    if(hi>_globalFarBegin)
        return std::max(baselineMajorant(lo,_globalFarBegin,parentMaximum,permitMacro),
                        std::min(parentMaximum,globalFarMaximum()));
    const double a=_age+lo,b=_age+hi;
    if(_tracking.coupledBounds && a>=_birthBound.split) {
        // With refinement enabled, keep expensive cubic certificates inside
        // the existing local-mean prefix. Beyond it retain macrocell/geometric
        // proposals. A zero event limit retains the full-voxel diagnostic.
        const bool polynomialPrefix=_tracking.transitionVoxelBound==TransitionVoxelBound::Bernstein5 &&
            _tracking.transitionMaxExpectedEvents>0;
        if(_tracking.transitionVoxelBound!=TransitionVoxelBound::Disabled &&
           (!polynomialPrefix || lo<_tracking.coupledLocalScale/_inverseEll-_age))
            return _straight->coupledIntervalMajorant(a,b,_meanLipschitz,true,parentMaximum,false,
                _tracking.transitionVoxelBound!=TransitionVoxelBound::RangeOnly,
                _tracking.transitionVoxelBound==TransitionVoxelBound::Bernstein5);
        return _straight->coupledIntervalMajorant(a,b,_meanLipschitz,
            a<_tracking.coupledLocalScale/_inverseEll,parentMaximum,macro);
    }
    if(_tracking.transitionIntegralBounds && a>=_birthBound.split)
        return std::min(parentMaximum,_straight->transitionIntegralMajorant(a,b,_meanLipschitz).value(a));
    if(_tracking.transitionPointBounds && a>=_birthBound.split)
        return _straight->transitionIntervalMajorant(a,b,_meanLipschitz,_tracking.tightenExpectedEvents,parentMaximum);
    if(_tracking.majorantMode==ConditionalMajorantMode::TwoConstants) {
        if(b<=_birthBound.split) return _birthBound.maximum;
        if(a>=_birthBound.split) return _tailMaximum;
        return std::max(_birthBound.maximum,_tailMaximum);
    }
    if(b<=_birthBound.split*(1+1e-14)) return std::min(parentMaximum,_birthBound.maximum);
    if(b>_birthBound.smoothEnd) {
        if(a>=_straight->simpleBoundBegin()) {
            const double bound=_straight->simpleIntervalMajorant(a,b,_meanLipschitz,parentMaximum);
            if(std::isfinite(bound) && _tracking.tightenExpectedEvents>0 &&
                _hazardEnvelope*bound*(hi-lo)<=std::min(_tracking.tightenExpectedEvents,_tracking.maxExpectedEvents)) {
                Profile::count(Profile::Counter::SimpleBoundAccepted);
                return bound;
            }
            Profile::count(Profile::Counter::SimpleBoundFallbacks);
            // The point cone was already queried. Tighten with the original
            // spatial certificate rather than asking the point path again.
            return _straight->intervalMajorant(a,b,_tracking.tightenExpectedEvents,std::min(parentMaximum,bound));
        }
        if(_tracking.pointBounds && a>=_classicBegin) {
            const double bound=_straight->pointIntervalMajorant(a,b,_meanLipschitz,parentMaximum);
            if(std::isfinite(bound) && _tracking.tightenExpectedEvents>0 &&
                _hazardEnvelope*bound*(hi-lo)<=std::min(_tracking.tightenExpectedEvents,_tracking.maxExpectedEvents)) {
                Profile::count(Profile::Counter::PointBoundAccepted);
                return bound;
            }
            Profile::count(Profile::Counter::PointBoundFallbacks);
            parentMaximum=std::min(parentMaximum,bound);
        }
        return _straight->intervalMajorant(a,b,_tracking.tightenExpectedEvents,parentMaximum);
    }
    const double bound=std::min(parentMaximum,_straight->birthAdjacentMajorant(a,b,_birthBound));
    // A local interval certificate can be much tighter than the curvature
    // of the entire birth cell, e.g. near a CSG feature.
    if(bound*(hi-lo)>_tracking.maxExpectedEvents) {
        Profile::count(Profile::Counter::LocalBoundFallbacks);
        return _straight->intervalMajorant(a,b,_tracking.tightenExpectedEvents,bound);
    }
    return bound;
}

double ConditionalSegment::globalFarMaximum() const {
    if(std::isnan(_globalFarMaximum))
        _globalFarMaximum=_straight->globalFarMajorant(_tracking.globalFarStartScale);
    return _globalFarMaximum;
}
FlightState ConditionalSegment::samplePositiveEndpoint(Random &rng) const
{
    if (_varying) return _varying->samplePositiveEndpoint(rng);
    // At the birth point there is no new observation to sample.
    auto endpoint=_age+_length==0 ? _straight->state() :
        _straight->samplePositiveEndpoint(_age+_length,rng);
    // Use the same double-precision position as the renderer's path context.
    endpoint.birthPosition=_origin+_length*_w;
    return endpoint;
}
double ConditionalSegment::track(Random &rng,bool collision,double &hit,ConditionalTrackingWork *diagnostics,double begin) const
{
    Profile::Scope profile(Profile::Stage::Track);
    Profile::Scope setup(Profile::Stage::TrackSetup);
    Profile::FlightWork work;
    ConditionalTrackingWork neuralWork;
    if(!(begin>=0 && begin<=_length))
        throw NumericError(NumericStatus::InvalidInput,"conditional tracking start outside the segment");
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
    if(auto *record=RayProfile::active()) record->trackingStart=begin;
#endif
    double traversed=begin;
    struct CaptureWork {
        ConditionalTrackingWork *output;
        const Profile::FlightWork &work;
        const double &distance;
        const ConditionalTrackingWork &neural;
        ~CaptureWork() { if(output) *output={work.candidates,work.intervals,distance,
            neural.networkQueries,neural.correctionCapHits,neural.neuralOutsideDomain,neural.neuralBaselineFallbacks,
            neural.neuralOutsideFeatureRange}; }
    } capture{diagnostics,work,traversed,neuralWork};
    if (_varying) {
        const double tr=_varying->track(rng,collision,hit,begin);
        traversed=hit;
        return tr;
    }
    const bool twoConstants=_tracking.majorantMode==ConditionalMajorantMode::TwoConstants;
    hit=_length;
    const double ell=1/_inverseEll;
    const double maximumStep=_tracking.fixedStepLimit
        ? (_tracking.step>0 ? _tracking.step : _tracking.stepScale*ell)
        : std::numeric_limits<double>::infinity();
    double weight=1;
    MajorantOpticalDepth opticalDepth;
    double coarseRetryAt=0,coarseMaximum=std::numeric_limits<double>::infinity();
    const double chapter2Begin=_tracking.chapter2StartScale>0 ?
        std::max(_birthBound.split,std::max(_tracking.chapter2StartScale,_tracking.coupledLocalScale)*ell)-_age :
        std::numeric_limits<double>::infinity();
    double chapter2RetryAt=chapter2Begin;
    double parentRetryAt=chapter2Begin;
    struct PendingInterval {
        double begin,end,maximum;
        bool tighten;
        bool globalFar=false;
        bool transitionPoint=false;
        bool transitionIntegral=false;
        bool farFunction=false;
        ConditionalIntegralMajorant curve;
        bool fixedMajorant=false;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
        Profile::IntervalLimit limit=Profile::IntervalLimit::TwoConstants;
#endif
    };
    boost::container::small_vector<PendingInterval,4> pending;
    const auto proposalMean=[&](double age,bool transition) {
        Profile::Scope point(Profile::Stage::TrackProposalSdf);
        return _straight->scalarMeanValue(age,transition);
    };
    const auto constantBound=[&](double begin,double end,double parent,bool permitMacro=true) {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
        RayProfile::BoundScope scope(begin,end);
#endif
        Profile::Scope bound(Profile::Stage::TrackBoundDispatch);
        return baselineMajorant(begin,end,parent,permitMacro);
    };
    const auto integralBound=[&](double begin,double end) {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
        RayProfile::BoundScope scope(begin,end);
#endif
        Profile::Scope bound(Profile::Stage::TrackBoundDispatch);
        return _straight->transitionIntegralMajorant(_age+begin,_age+end,_meanLipschitz);
    };
    const auto clipMacro=[&](double lo,double &hi) {
        Profile::Scope clip(Profile::Stage::TrackMacroClip);
        if(_tracking.macrocellBounds && _age+lo>=_tracking.macrocellStartScale*ell) {
            double end=_age+hi;
            if(_straight->clipMacrocellInterval(_age+lo,end) && end<_age+hi) {
                const double clipped=end-_age;
                if(clipped>lo) hi=clipped;
            }
        }
    };
    const auto hybridConstantBound=[&](double lo,double &hi) {
        const bool hybrid=_tracking.macrocellBounds && _age+lo>=_tracking.macrocellStartScale*ell;
        double parent=std::numeric_limits<double>::infinity();
        if(hybrid && _tracking.macrocellMinExpectedEvents>0) {
            parent=constantBound(lo,hi,parent,false);
            if(_hazardEnvelope*parent*(hi-lo)<=_tracking.macrocellMinExpectedEvents) return parent;
        }
        if(hybrid)clipMacro(lo,hi);
        return constantBound(lo,hi,parent);
    };
    double insideUntil=-1;
    std::optional<ClassicLocalGrid::Cursor> bandCursor;
    double occupiedEnd=_length;
    if (_tracking.classicLocalBand) {
        Profile::Scope cursor(Profile::Stage::TrackBandCursor);
        bandCursor.emplace(*_tracking.classicLocalBand,_origin,_w,begin,_length);
        occupiedEnd=begin;
    }
    setup.finish();
    for(double lo=begin;lo<_length;) {
        if (bandCursor && lo>=occupiedEnd) {
            // Proposals and their pending halves are clipped to the previous range.
            // Carry the same optical clock and conditioning age across the gap.
            Profile::Scope traversal(Profile::Stage::TrackBandTraversal);
            if (!bandCursor->nextOccupiedRange(lo,occupiedEnd)) { traversed=_length; break; }
            traversed=lo;
        }
        if(_tracking.exteriorCutoffSigma>0 && pending.empty() && lo>insideUntil) {
            Profile::Scope exterior(Profile::Stage::TrackExterior);
            const double age=_age+lo;
            const double mean=_straight->scalarMeanValue(age);
            const double difference=mean-_exteriorCutoff;
            // Protect the spatial certificate at both endpoints, including
            // the world-space ray and its conversion to absolute flight age.
            const double pad=1024*std::numeric_limits<double>::epsilon()*(1+std::abs(mean)+_exteriorCutoff+
                _meanLipschitz*(1+_origin.norm()+_age+_length));
            double certifiedEnd=lo;
            if(std::abs(difference)>pad) {
                certifiedEnd=_meanLipschitz==0 ? _length : std::min(_length,
                    MajorantBounds::down(lo+(std::abs(difference)-pad)/(_meanLipschitz*(1+1e-12))));
            }
            if(difference>0) {
                // If the Lipschitz distance is tiny near the isosurface,
                // a whole macrocell can still be certified exterior.
                if(_tracking.macrocellSize && certifiedEnd-lo<.25*ell) {
                    double end=_age+_length;
                    if(_straight->exteriorMacrocell(age,end,_exteriorCutoff))
                        certifiedEnd=std::max(certifiedEnd,std::min(_length,MajorantBounds::down(end-_age)));
                }
                // Short unclassified remainders use the usual majorant plus a
                // point test. Do not move the exponential clock across vacuum.
                if(certifiedEnd>lo && (certifiedEnd==_length || certifiedEnd-lo>.025*ell)) {
                    lo=certifiedEnd;traversed=lo;
                    continue;
                }
            } else insideUntil=certifiedEnd>lo ? certifiedEnd:-1;
        }
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
        RayProfile::Timer phaseTimer(lo,0);
#endif
        Profile::Scope proposalTime(Profile::Stage::TrackProposal);
        double hi;
        double proposalEnd=lo<chapter2Begin ? std::min(occupiedEnd,chapter2Begin) : occupiedEnd;
        if(pending.empty() && lo>=chapter2RetryAt) {
            const auto tryBlock=[&](bool parent) {
                double endAge=_age+proposalEnd,maximum;
                MeanRayBounds parentMean;
                const bool available=parent ? _straight->chapter2ParentRange(_age+lo,endAge,parentMean)
                                            : _straight->chapter2MacrocellMajorant(_age+lo,endAge,maximum);
                if (!available) return;
                double end=endAge<_age+proposalEnd ? endAge-_age : proposalEnd;
                if(_age+end>endAge)end=std::nextafter(end,lo);
                if(end>lo) {
                    const double budget=parent ? _tracking.chapter2ParentEvents : _tracking.chapter2FallbackEvents;
                    if(parent) {
                        if(_tracking.chapter2ParentScreen && _straight->rejectChapter2Parent(
                                _age+lo,parentMean,end-lo,_hazardEnvelope,budget)) {
                            Profile::count(Profile::Counter::Chapter2ParentScreened);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
                            // Audit the counterfactual decision. This extra oracle query is
                            // excluded from production work counts and from uninstrumented timings.
                            if(Profile::registry().enabled &&
                               _straight->chapter2Majorant(_age+lo,endAge,parentMean,maximum) &&
                               maximum*_hazardEnvelope*(end-lo)<=budget)
                                Profile::count(Profile::Counter::Chapter2ParentFalseReject);
#endif
                            parentRetryAt=end;
                            return;
                        }
                        Profile::count(Profile::Counter::Chapter2ParentBounds);
                        if(!_straight->chapter2Majorant(_age+lo,endAge,parentMean,maximum)) return;
                    }
                    (parent ? parentRetryAt : chapter2RetryAt)=end;
                    if(budget==0 || maximum*_hazardEnvelope*(end-lo)<=budget) {
                        PendingInterval interval{lo,end,maximum,false};
                        interval.fixedMajorant=true;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
                        interval.limit=Profile::IntervalLimit::MacroBoundary;
#endif
                        pending.push_back(interval);
                        if(parent) Profile::count(Profile::Counter::Chapter2ParentAccepted);
                        Profile::count(Profile::Counter::ProposedIntervals);
                    }
                }
            };
            if(_tracking.chapter2ParentEvents>0 && lo>=parentRetryAt) tryBlock(true);
            if(pending.empty()) tryBlock(false);
        }
        if(lo>=chapter2Begin && lo<chapter2RetryAt)proposalEnd=std::min(proposalEnd,chapter2RetryAt);
        if(pending.empty() && lo>=_localFarBegin && _tracking.farIntegralBounds) {
            hi=proposalEnd;
            PendingInterval interval{lo,hi,0,false};
            interval.transitionIntegral=true;interval.farFunction=true;
            {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
                RayProfile::BoundScope scope(lo,hi);
#endif
                interval.curve=_straight->farIntegralMajorant(_age+lo,_age+hi,_meanLipschitz,_meanGradientBound);
            }
            interval.maximum=interval.curve.value(_age+hi);
            pending.push_back(interval);
            Profile::count(Profile::Counter::ProposedIntervals);Profile::count(Profile::Counter::LocalFarProposals);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            pending.back().limit=Profile::IntervalLimit::LocalFar;
#endif
        }
        if(pending.empty() && lo>=_localFarBegin) {
            double proposal=_length-lo;
            if(_meanLipschitz>0) {
                const double value=proposalMean(_age+lo,false);
                if(!std::isfinite(value))
                    throw NumericError(NumericStatus::InvalidMajorant,"nonfinite local far proposal mean");
                proposal=std::max(_field.kernel.sigma()/_meanLipschitz,
                                  0.5*std::max(0.0,value)/_meanLipschitz);
            }
            hi=std::min(proposalEnd,lo+proposal*_tracking.proposalScale);
            const double maximum=hybridConstantBound(lo,hi);
            if(!(hi>lo)) throw NumericError(NumericStatus::InvalidMajorant,"local far proposal cannot advance");
            pending.push_back({lo,hi,maximum,false});
            Profile::count(Profile::Counter::ProposedIntervals);
            Profile::count(Profile::Counter::LocalFarProposals);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            pending.back().limit=Profile::IntervalLimit::LocalFar;
            Profile::partition(Profile::PartitionPhase::Initial,Profile::IntervalLimit::LocalFar,(_age+lo)/ell,(hi-lo)/ell);
            Profile::partition(Profile::PartitionPhase::Proposed,Profile::IntervalLimit::LocalFar,(_age+lo)/ell,(hi-lo)/ell);
#endif
        }
        if(pending.empty() && lo>=_globalFarBegin) {
            hi=proposalEnd;
            pending.push_back({lo,hi,globalFarMaximum(),false,true});
            Profile::count(Profile::Counter::ProposedIntervals);
            Profile::count(Profile::Counter::GlobalFarIntervals);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            pending.back().limit=Profile::IntervalLimit::GlobalFar;
            Profile::partition(Profile::PartitionPhase::Initial,Profile::IntervalLimit::GlobalFar,(_age+lo)/ell,(hi-lo)/ell);
            Profile::partition(Profile::PartitionPhase::Proposed,Profile::IntervalLimit::GlobalFar,(_age+lo)/ell,(hi-lo)/ell);
#endif
        }
        if(pending.empty() && (_tracking.transitionPointBounds || _tracking.transitionIntegralBounds || _tracking.coupledBounds) && _age+lo>=_birthBound.split) {
            const bool polynomialPrefix=_tracking.transitionVoxelBound==TransitionVoxelBound::Bernstein5 &&
                _tracking.transitionMaxExpectedEvents>0;
            const double polynomialEnd=_tracking.coupledLocalScale/_inverseEll-_age;
            const bool voxel=_tracking.transitionVoxelBound!=TransitionVoxelBound::Disabled &&
                (!polynomialPrefix || lo<polynomialEnd);
            if(voxel) {
                hi=std::min({proposalEnd,_localFarBegin,_globalFarBegin});
                if(polynomialPrefix)hi=std::min(hi,polynomialEnd);
                const double face=_straight->nextMeanBoundary(_age+lo,_age+hi);
                if(face<_age+hi) hi=face-_age;
            } else {
                double proposal=_length-lo;
                if(_meanLipschitz>0) {
                    const double value=proposalMean(_age+lo,true);
                    if(!std::isfinite(value))
                        throw NumericError(NumericStatus::InvalidMajorant,"nonfinite transition proposal mean");
                    proposal=std::max(_field.kernel.sigma()/_meanLipschitz,
                                      0.5*std::max(0.0,value)/_meanLipschitz);
                }
                proposal*=_tracking.proposalScale;
                if(_tracking.coupledBounds) proposal=std::min(proposal,_tracking.ageScale*(_age+lo));
                hi=std::min({proposalEnd,lo+proposal,_localFarBegin,_globalFarBegin});
            }
            if(!(hi>lo)) throw NumericError(NumericStatus::InvalidMajorant,"transition proposal cannot advance");
            PendingInterval interval{lo,hi,0,false,false,true};
            interval.transitionIntegral=_tracking.transitionIntegralBounds;
            if(interval.transitionIntegral) {
                interval.curve=integralBound(lo,hi);
                interval.maximum=interval.curve.value(_age+lo);
            } else {
                interval.maximum=voxel ? constantBound(lo,hi,std::numeric_limits<double>::infinity(),false) : hybridConstantBound(lo,hi);
                interval.end=hi;
            }
            pending.push_back(interval);
            Profile::count(Profile::Counter::ProposedIntervals);
            Profile::count(Profile::Counter::TransitionProposals);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            pending.back().limit=Profile::IntervalLimit::Transition;
            Profile::partition(Profile::PartitionPhase::Initial,Profile::IntervalLimit::Transition,(_age+lo)/ell,(hi-lo)/ell);
            Profile::partition(Profile::PartitionPhase::Proposed,Profile::IntervalLimit::Transition,(_age+lo)/ell,(hi-lo)/ell);
#endif
        }
        if(pending.empty() && twoConstants) {
            // Both constants were computed at construction. Never walk cells,
            // tighten, or subdivide this proposal, even when it is very loose.
            // When the prefix covers the endpoint, avoid subtracting age back
            // out of age+length: that can leave a one-ulp, zero-rate tail.
            const double split=_birthBound.split>=_age+_length ? _length :
                std::max(0.0,_birthBound.split-_age);
            hi=lo<split ? std::min(proposalEnd,split) : proposalEnd;
            pending.push_back({lo,hi,lo<split ? _birthBound.maximum : _tailMaximum,false});
            Profile::count(Profile::Counter::ProposedIntervals);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            Profile::partition(Profile::PartitionPhase::Initial,Profile::IntervalLimit::TwoConstants,(_age+lo)/ell,(hi-lo)/ell);
            Profile::partition(Profile::PartitionPhase::Proposed,Profile::IntervalLimit::TwoConstants,(_age+lo)/ell,(hi-lo)/ell);
#endif
        }
        if(pending.empty()) {
            Profile::count(Profile::Counter::ProposedIntervals);
            double proposal=_tracking.ageScale*(_age+lo);
            const bool geometry=_tracking.geometryStep && _age+lo>=_birthBound.split &&
                _age+lo>=_classicBegin;
            if(geometry) {
                Profile::count(Profile::Counter::GeometryProposals);
                if(_meanLipschitz>0) {
                    const double base=_field.kernel.sigma()/_meanLipschitz;
                    Profile::Scope point(Profile::Stage::TrackProposalSdf);
                    const double value=_straight->meanValue(_age+lo);
                    if(!std::isfinite(value))
                        throw NumericError(NumericStatus::InvalidMajorant,"nonfinite geometric proposal mean");
                    proposal=std::max(base,0.5*std::max(0.0,value)/_meanLipschitz);
                } else proposal=_length-lo;
            }
            hi=std::min(proposalEnd,lo+std::min(maximumStep,proposal));
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            auto limit=hi==_length ? Profile::IntervalLimit::SegmentEnd :
                maximumStep<=proposal ? Profile::IntervalLimit::Step :
                geometry ? Profile::IntervalLimit::Geometry : Profile::IntervalLimit::Age;
#endif
            if(_age+lo<_birthBound.split) {
                hi=std::min(proposalEnd,_birthBound.split-_age);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
                limit=hi==_length ? Profile::IntervalLimit::SegmentEnd : Profile::IntervalLimit::Birth;
#endif
            }
            // Finish the near proposal exactly at the global-tail switch. The
            // same optical-depth clock continues across it without a new draw.
            hi=std::min({hi,_globalFarBegin,_localFarBegin});
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            Profile::partition(Profile::PartitionPhase::Initial,limit,(_age+lo)/ell,(hi-lo)/ell);
#endif
            bool coarse=false;
            if(_tracking.macrocellSize && _age+lo>=_birthBound.smoothEnd && lo>=coarseRetryAt) {
                double end=_age+hi,maximum;
                coarseRetryAt=hi;
                coarseMaximum=std::numeric_limits<double>::infinity();
                Profile::count(Profile::Counter::AdaptiveCoarseProbes);
                if(_straight->coarseMajorant(_age+lo,end,maximum)) {
                    const double clipped=end<_age+hi ? end-_age : hi;
                    if(clipped>lo) {
                        coarseRetryAt=clipped;
                        coarseMaximum=maximum;
                        // Only change the proposal partition. Keep the exact
                        // birth-conditioned hazard and the same certified bound.
                        // Age includes any straight continuation from the birth.
                        const double eventLimit=_tracking.macrocellFarMaxExpectedEvents>0 &&
                            _age+lo>=_tracking.macrocellFarStartScale*ell
                            ? _tracking.macrocellFarMaxExpectedEvents : _tracking.macrocellMaxExpectedEvents;
                        if(std::isfinite(maximum) && _hazardEnvelope*maximum*(clipped-lo)<=
                                std::min(eventLimit,_tracking.maxExpectedEvents)) {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
                            if(clipped<hi) limit=Profile::IntervalLimit::MacroBoundary;
#endif
                            hi=clipped;
                            coarse=true;
                            Profile::count(Profile::Counter::MacroAccepted);
                        } else Profile::count(Profile::Counter::MacroRejected);
                    }
                }
            }
            // A certified birth prefix already covers its complete span,
            // including grid faces for the positive-value Lipschitz proof.
            // Only ordinary adaptive proposals need interpolation-cell clips.
            if(!coarse && _age+lo>=_birthBound.split) {
                const double boundary=_straight->nextMeanBoundary(_age+lo,_age+hi);
                if(boundary<_age+hi) {
                    hi=boundary-_age;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
                    limit=Profile::IntervalLimit::FineBoundary;
#endif
                }
            }
            if(!(hi>lo)) {
                std::ostringstream message;
                message<<std::setprecision(17)<<"conditional segment cannot advance: lo="<<lo
                       <<" end="<<_length<<" age="<<_age<<" birthSplit="<<_birthBound.split;
                throw NumericError(NumericStatus::InvalidMajorant,message.str());
            }
            // A rejected coarse window still supplies a valid parent bound.
            // Do not retry that window for each of its fine interpolation cells.
            const double parent=hi<=coarseRetryAt ? coarseMaximum : std::numeric_limits<double>::infinity();
            pending.push_back({lo,hi,coarse ? coarseMaximum : constantBound(lo,hi,parent),false});
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            pending.back().limit=limit;
            Profile::partition(Profile::PartitionPhase::Proposed,limit,(_age+lo)/ell,(hi-lo)/ell);
#endif
        }
        proposalTime.finish();
        Profile::Scope refinement(Profile::Stage::TrackRefinement);
        auto b=pending.back(); pending.pop_back();
        lo=b.begin;
        hi=b.end;
        const auto tighten=[&]() {
            if(b.transitionIntegral) {
                b.curve=integralBound(lo,hi);
                b.maximum=b.curve.value(_age+lo);
            } else b.maximum=constantBound(lo,hi,b.maximum);
        };
        if(b.tighten) tighten();
        // Zero disables transition refinement, including the global event cap.
        // Proposal endpoints and the birth/far switches still apply.
        const double eventLimit=b.transitionPoint
            ? (_tracking.transitionMaxExpectedEvents==0 ? std::numeric_limits<double>::infinity()
                : std::min(_tracking.maxExpectedEvents,_tracking.transitionMaxExpectedEvents))
            : _tracking.maxExpectedEvents;
        const auto opticalCost=[&]() {
            return _hazardEnvelope*(b.transitionIntegral ? b.curve.integral(_age+lo,_age+hi) : b.maximum*(hi-lo));
        };
        while(!twoConstants && !b.globalFar && !b.fixedMajorant && !b.farFunction && opticalCost()>eventLimit) {
            Profile::count(Profile::Counter::Splits);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
            if(auto *r=RayProfile::active())++r->phases[r->phase(lo)].splits;
#endif
            const double mid=0.5*(lo+hi);
            if(!(mid>lo&&mid<hi)) throw NumericError(NumericStatus::InvalidMajorant,"cannot refine conditional interval");
            // The parent certificate already covers the right half. Tighten
            // it only if tracking reaches it; a collision in the left half
            // should not pay for SDF queries beyond the collision.
            auto right=b;right.begin=mid;right.end=hi;right.tighten=true;
            pending.push_back(right);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
            pending.back().limit=b.limit=Profile::IntervalLimit::EventThreshold;
#endif
            hi=mid;tighten();
        }
        refinement.finish();
        Profile::Scope leaf(Profile::Stage::TrackLeaf);
        ++work.intervals;
        Profile::count(Profile::Counter::ProcessedIntervals);
        if(b.maximum==0) Profile::count(Profile::Counter::ZeroRateIntervals);
        // Every pending, birth, tail and macrocell certificate above bounds the
        // analytic baseline. Scale exactly once here. The cap remains valid
        // even when the deterministic history anchor changes inside an interval.
        const double proposalMaximum=_hazardEnvelope*b.maximum;
        if(!(proposalMaximum>=0) || !std::isfinite(proposalMaximum))
            throw NumericError(NumericStatus::InvalidMajorant,"nonfinite neural conditional majorant");
        Profile::IntervalWork intervalWork;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
        Profile::ProcessedInterval partitionWork(b.limit,(_age+lo)/ell,lo,hi,
            b.transitionIntegral ? opticalCost()/(hi-lo) : proposalMaximum,ell,traversed,intervalWork.candidates);
#endif
        double t=lo;
        double candidateOffset=0;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
        RayProfile::IntervalScope tracedInterval(lo,hi,b.maximum,_hazardEnvelope,b.transitionIntegral,b.curve,traversed);
        // Opt-in diagnostic snapshot BEFORE a pathological proposal consumes
        // millions of samples. The normal renderer does not compile this code.
        static const bool dumpExpensive=std::getenv("TUNGSTEN_DUMP_EXPENSIVE_VOXEL")!=nullptr;
        static std::atomic<bool> dumped{false};
        if(dumpExpensive && b.maximum*_hazardEnvelope*(hi-lo)>1e5 && !dumped.exchange(true)) {
            if(const auto *r=RayProfile::active()) {
                double zlo=0,zhi=0;
                const bool polynomial=_straight->polynomialZBounds(_age+lo,_age+hi,zlo,zhi);
                double sampledZlo=INFINITY,sampledZhi=-INFINITY,sampledDlo=INFINITY,peak=0,peakAt=lo;
                for(int i=0;i<=2048;++i) {
                    const double t=lo+(hi-lo)*i/2048;
                    const auto p=_straight->scalarStatistics(_age+t);
                    const double sd=std::sqrt(p.varianceF),h=hazard(t);
                    sampledZlo=std::min(sampledZlo,p.meanF/sd);sampledZhi=std::max(sampledZhi,p.meanF/sd);
                    sampledDlo=std::min(sampledDlo,p.meanAtZero/sd);
                    if(h>peak) {peak=h;peakAt=t;}
                }
                std::ostringstream out;out<<std::setprecision(17);
                const auto vec=[&](const Vector3 &v){out<<'['<<v.x()<<','<<v.y()<<','<<v.z()<<']';};
                out<<"EXPENSIVE_VOXEL {\"birth_position\":";vec(r->birth.birthPosition);
                out<<",\"birth_gradient\":";vec(r->birth.birthGradient);
                out<<",\"origin\":";vec(r->origin);out<<",\"direction\":";vec(r->direction);
                out<<",\"effective_direction\":";vec(r->effectiveDirection);
                out<<",\"birth_value\":"<<r->birth.birthValue<<",\"birth_age\":"<<r->birth.age
                   <<",\"age_at_origin\":"<<_age<<",\"birth_split\":"<<_birthBound.split
                   <<",\"length\":"<<_length<<",\"bounce\":"<<r->bounce<<",\"collision\":"<<r->collision
                   <<",\"lo\":"<<lo<<",\"hi\":"<<hi<<",\"majorant\":"<<b.maximum*_hazardEnvelope
                   <<",\"expected_candidates\":"<<b.maximum*_hazardEnvelope*(hi-lo)
                   <<",\"polynomial_available\":"<<polynomial<<",\"z_lower\":"<<zlo<<",\"z_upper\":"<<zhi
                   <<",\"sampled_z_min\":"<<sampledZlo<<",\"sampled_z_max\":"<<sampledZhi
                   <<",\"sampled_d_min\":"<<sampledDlo<<",\"sampled_hazard_peak\":"<<peak<<",\"peak_at\":"<<peakAt<<'}';
                std::cerr<<out.str()<<std::endl;
            }
        }
#endif
        if(proposalMaximum>0) for(;;) {
            double next;
            bool candidate;
            {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
            RayProfile::Timer clockTimer(t,2);
#endif
            Profile::Scope clock(Profile::Stage::TrackOpticalClock);
            candidate=b.transitionIntegral
                ? opticalDepth.advanceIntegrated(rng,t,hi,_hazardEnvelope*b.curve.integral(_age+t,_age+hi),
                    [&](double depth) { return b.curve.inverseIntegral(_age+t,_age+hi,depth/_hazardEnvelope)-_age; },next)
                : opticalDepth.advanceFromOffset(rng,proposalMaximum,lo,hi,candidateOffset,next);
            }
            if(!candidate) break;
            t=next;
            traversed=t;
            ++work.candidates;++intervalWork.candidates;
            Profile::count(Profile::Counter::CandidateEvents);
            double h;
            {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
            RayProfile::Timer hazardTimer(t,1);
#endif
            Profile::Scope evaluation(Profile::Stage::TrackCandidateEval);
            h=correctedHazard(t,diagnostics ? &neuralWork : nullptr,t<=insideUntil);
            }
            Profile::Scope decision(Profile::Stage::TrackDecision);
            if(h==0) Profile::count(Profile::Counter::ZeroHazardCandidates);
            const double candidateMaximum=b.transitionIntegral ? _hazardEnvelope*b.curve.value(_age+t) : proposalMaximum;
            if(h>candidateMaximum*(1+1e-10)) throw NumericError(NumericStatus::InvalidMajorant,"conditional hazard exceeds its bound");
            if(!(candidateMaximum>0)) throw NumericError(NumericStatus::InvalidMajorant,"candidate has zero majorant density");
            const double acceptance=std::min(1.0,h/candidateMaximum);
            if(collision) {
                if(rng.openUniform01()<acceptance) {
                    Profile::count(Profile::Counter::AcceptedCollisions);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
                    RayProfile::event(t,h,candidateMaximum,0,1);
#endif
                    hit=t; return 0;
                }
                Profile::count(Profile::Counter::RejectedCandidates);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
                RayProfile::event(t,h,candidateMaximum,1,0);
#endif
            } else {
                Profile::count(Profile::Counter::RatioEvents);
                weight*=1-acceptance;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
                RayProfile::event(t,h,candidateMaximum,weight,2);
#endif
                if(weight==0) { Profile::count(Profile::Counter::ZeroWeight);return 0; }
                if(weight<_tracking.shadowRouletteThreshold) {
                    Profile::count(Profile::Counter::ShadowRouletteTests);
                    // Survive with weight/threshold and compensate the surviving
                    // estimator. This changes variance, not its expectation.
                    if(rng.openUniform01()>=weight/_tracking.shadowRouletteThreshold) {
                        Profile::count(Profile::Counter::ShadowRouletteTerminated);
                        return 0;
                    }
                    Profile::count(Profile::Counter::ShadowRouletteSurvived);
                    weight=_tracking.shadowRouletteThreshold;
                }
            }
            // Each candidate provides a new scalar anchor for the Lipschitz
            // cone. The target conditional observation is never renewed here.
            if(b.farFunction) { hi=t;break; }
        }
        lo=hi;
        traversed=hi;
    }
    Profile::count(Profile::Counter::Escapes);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
    if(auto *record=RayProfile::active()) record->visited=traversed;
#endif
    return weight;
}
}
