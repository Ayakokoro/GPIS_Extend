#include "ConditionalSegment.h"
#include "MajorantOpticalDepth.h"
#include "math/ConditionalProfile.hpp"
#include <sstream>
#include <iomanip>

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
    if(majorantMode!=ConditionalMajorantMode::Adaptive && majorantMode!=ConditionalMajorantMode::TwoConstants)
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional majorant mode");
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
    if(!(macrocellMaxExpectedEvents>0) || !std::isfinite(macrocellMaxExpectedEvents))
        throw NumericError(NumericStatus::InvalidInput,"tracking_macrocell_max_expected_events must be finite and positive");
}
ConditionalSegment::ConditionalSegment(const GPSSField &f,const FlightState &s,
        const Point3 &o,const Vector3 &w,double length,const ConditionalTrackingSettings &tracking)
    : _field(f),_origin(o),_w(normalizedOrThrow(w)),_length(length),
      _age((o-s.birthPosition).dot(_w)),_tracking(tracking),_straight(f,alongDirection(s,_w))
{
    _tracking.validate();
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
    if(!std::isfinite(_age+length) || _age+length>_straight.maximumAgeInDomain())
        throw NumericError(NumericStatus::InvalidInput,"conditional segment exceeds the field domain");
    if(_tracking.majorantMode==ConditionalMajorantMode::TwoConstants && _tracking.constantBirthLength>0) {
        const double split=std::min(_age+length,_tracking.constantBirthLength);
        // Keep the certified analytic birth interval, then bound its extension
        // before sampling. The maximum of the covering intervals is one valid
        // constant for the entire prefix; a Taylor certificate cannot simply
        // be extrapolated through grid knots. No observation is renewed here.
        const auto prefix=_straight.twoSegmentMajorants(split);
        _birthBound.split=split;
        _birthBound.maximum=std::max(prefix.nearMaximum,prefix.farMaximum);
    } else _birthBound=_straight.birthMajorant(_age+length);
    if(_tracking.majorantMode==ConditionalMajorantMode::TwoConstants && _age+length>_birthBound.split)
        _tailMaximum=_straight.constantTailMajorant(_birthBound.split);
}
Gaussian<3> ConditionalSegment::gradient(double t) const
{
    return _straight.hitStatistics(_age+t).collisionGradient;
}
ConditionalScalarStatistics ConditionalSegment::statistics(double t) const
{
    return _straight.scalarStatistics(_age+t);
}
double ConditionalSegment::hazard(double t) const
{
    const double result=_straight.evaluate(_age+t).hazard.value;
    if(!(result>=0) || !std::isfinite(result))
        throw NumericError(NumericStatus::NeedHigherPrecision,"nonfinite conditional hazard");
    return result;
}
double ConditionalSegment::majorant(double lo,double hi,double parentMaximum) const
{
    const double a=_age+lo,b=_age+hi;
    if(_tracking.majorantMode==ConditionalMajorantMode::TwoConstants) {
        if(b<=_birthBound.split) return _birthBound.maximum;
        if(a>=_birthBound.split) return _tailMaximum;
        return std::max(_birthBound.maximum,_tailMaximum);
    }
    if(b<=_birthBound.split*(1+1e-14)) return std::min(parentMaximum,_birthBound.maximum);
    if(b>_birthBound.smoothEnd)
        return _straight.intervalMajorant(a,b,_tracking.tightenExpectedEvents,parentMaximum);
    const double bound=std::min(parentMaximum,_straight.birthAdjacentMajorant(a,b,_birthBound));
    // A local interval certificate can be much tighter than the curvature
    // of the entire birth cell, e.g. near a CSG feature.
    if(bound*(hi-lo)>_tracking.maxExpectedEvents) {
        Profile::count(Profile::Counter::LocalBoundFallbacks);
        return _straight.intervalMajorant(a,b,_tracking.tightenExpectedEvents,bound);
    }
    return bound;
}
FlightState ConditionalSegment::samplePositiveEndpoint(Random &rng) const
{
    // At the birth point there is no new observation to sample.
    auto endpoint=_age+_length==0 ? _straight.state() :
        _straight.samplePositiveEndpoint(_age+_length,rng);
    // Use the same double-precision position as the renderer's path context.
    endpoint.birthPosition=_origin+_length*_w;
    return endpoint;
}
double ConditionalSegment::track(Random &rng,bool collision,double &hit,ConditionalTrackingWork *diagnostics) const
{
    Profile::Scope profile(Profile::Stage::Track);
    Profile::FlightWork work;
    double traversed=0;
    struct CaptureWork {
        ConditionalTrackingWork *output;
        const Profile::FlightWork &work;
        const double &distance;
        ~CaptureWork() { if(output) *output={work.candidates,work.intervals,distance}; }
    } capture{diagnostics,work,traversed};
    const bool twoConstants=_tracking.majorantMode==ConditionalMajorantMode::TwoConstants;
    hit=_length;
    const double ell=1/std::sqrt(_w.dot(_field.kernel.precision()*_w));
    const double maximumStep=_tracking.step>0 ? _tracking.step : _tracking.stepScale*ell;
    double weight=1;
    MajorantOpticalDepth opticalDepth;
    double coarseRetryAt=0,coarseMaximum=std::numeric_limits<double>::infinity();
    struct PendingInterval {
        double begin,end,maximum;
        bool tighten;
    };
    std::vector<PendingInterval> pending;
    for(double lo=0;lo<_length;) {
        double hi;
        if(pending.empty() && twoConstants) {
            // Both constants were computed at construction. Never walk cells,
            // tighten, or subdivide this proposal, even when it is very loose.
            // When the prefix covers the endpoint, avoid subtracting age back
            // out of age+length: that can leave a one-ulp, zero-rate tail.
            const double split=_birthBound.split>=_age+_length ? _length :
                std::max(0.0,_birthBound.split-_age);
            hi=lo<split ? std::min(_length,split) : _length;
            pending.push_back({lo,hi,lo<split ? _birthBound.maximum : _tailMaximum,false});
            Profile::count(Profile::Counter::ProposedIntervals);
        }
        if(pending.empty()) {
            Profile::count(Profile::Counter::ProposedIntervals);
            hi=std::min(_length,lo+std::min(maximumStep,_tracking.ageScale*(_age+lo)));
            if(_age+lo<_birthBound.split)
                hi=std::min(_length,_birthBound.split-_age);
            bool coarse=false;
            if(_tracking.macrocellSize && _age+lo>=_birthBound.smoothEnd && lo>=coarseRetryAt) {
                double end=_age+hi,maximum;
                coarseRetryAt=hi;
                coarseMaximum=std::numeric_limits<double>::infinity();
                if(_straight.coarseMajorant(_age+lo,end,maximum)) {
                    const double clipped=end<_age+hi ? end-_age : hi;
                    if(clipped>lo) {
                        coarseRetryAt=clipped;
                        coarseMaximum=maximum;
                        if(std::isfinite(maximum) && maximum*(clipped-lo)<=
                                std::min(_tracking.macrocellMaxExpectedEvents,_tracking.maxExpectedEvents)) {
                            hi=clipped;
                            coarse=true;
                            Profile::count(Profile::Counter::MacroAccepted);
                        } else Profile::count(Profile::Counter::MacroRejected);
                    }
                }
            }
            if(!coarse) {
                const double boundary=_straight.nextMeanBoundary(_age+lo,_age+hi);
                if(boundary<_age+hi) hi=boundary-_age;
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
            pending.push_back({lo,hi,coarse ? coarseMaximum : majorant(lo,hi,parent),false});
        }
        auto b=pending.back(); pending.pop_back();
        lo=b.begin;
        hi=b.end;
        if(b.tighten) b.maximum=majorant(lo,hi,b.maximum);
        while(!twoConstants && b.maximum*(hi-lo)>_tracking.maxExpectedEvents) {
            Profile::count(Profile::Counter::Splits);
            const double mid=0.5*(lo+hi);
            if(!(mid>lo&&mid<hi)) throw NumericError(NumericStatus::InvalidMajorant,"cannot refine conditional interval");
            // The parent certificate already covers the right half. Tighten
            // it only if tracking reaches it; a collision in the left half
            // should not pay for SDF queries beyond the collision.
            pending.push_back({mid,hi,b.maximum,true});
            hi=mid; b.maximum=majorant(lo,hi,b.maximum);
        }
        ++work.intervals;
        Profile::count(Profile::Counter::ProcessedIntervals);
        if(b.maximum==0) Profile::count(Profile::Counter::ZeroRateIntervals);
        Profile::IntervalWork intervalWork;
        double t=lo;
        if(b.maximum>0) for(;;) {
            double next;
            if(!opticalDepth.advance(rng,b.maximum,t,hi,next)) break;
            t=next;
            traversed=t;
            ++work.candidates;++intervalWork.candidates;
            Profile::count(Profile::Counter::CandidateEvents);
            const double h=hazard(t);
            if(h==0) Profile::count(Profile::Counter::ZeroHazardCandidates);
            if(h>b.maximum*(1+1e-10)) throw NumericError(NumericStatus::InvalidMajorant,"conditional hazard exceeds its bound");
            const double acceptance=std::min(1.0,h/b.maximum);
            if(collision) {
                if(rng.openUniform01()<acceptance) {
                    Profile::count(Profile::Counter::AcceptedCollisions);
                    hit=t; return 0;
                }
                Profile::count(Profile::Counter::RejectedCandidates);
            } else {
                Profile::count(Profile::Counter::RatioEvents);
                weight*=1-acceptance;
                if(weight==0) { Profile::count(Profile::Counter::ZeroWeight);return 0; }
            }
        }
        lo=hi;
        traversed=hi;
    }
    Profile::count(Profile::Counter::Escapes);
    return weight;
}
}
