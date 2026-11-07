#include "math/SingleHistoryFlight.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>

namespace Tungsten::SingleHistoryGpis {
namespace cg = ConditionalGpis;
namespace sh = SparseHistoryGpis;
namespace rb = cg::MajorantBounds;
namespace {
[[noreturn]] void fail(cg::NumericStatus status,const char *message) {
    throw cg::NumericError(status,message);
}
bool positive(double x) { return x>0 && std::isfinite(x); }
struct Timer {
    using Clock=std::chrono::steady_clock;
    double *destination;
    Clock::time_point begin;
    Timer(bool enabled,double &output):destination(enabled ? &output:nullptr) {
        if(destination) begin=Clock::now();
    }
    void stop() {
        if(destination) {
            *destination+=std::chrono::duration<double>(Clock::now()-begin).count();
            destination=nullptr;
        }
    }
    ~Timer() { stop(); }
};
struct QuadratureTimer {
    Work &work;
    bool enabled;
    double excluded;
    Timer::Clock::time_point begin;
    QuadratureTimer(bool active,Work &output):work(output),enabled(active),
        excluded(output.selectionSeconds+output.probabilitySeconds) {
        if(enabled) begin=Timer::Clock::now();
    }
    ~QuadratureTimer() {
        if(enabled) work.quadratureSeconds+=std::max(0.0,
            std::chrono::duration<double>(Timer::Clock::now()-begin).count()-
            (work.selectionSeconds+work.probabilitySeconds-excluded));
    }
};
}

void Settings::validate() const {
    if(!positive(selectionStepOverLength) || historyGroups<1 || historyGroups>256 ||
       !(shadowRouletteThreshold>=0 && shadowRouletteThreshold<=1) ||
       !std::isfinite(shadowRouletteThreshold) || maxWork<16 ||
       !std::isfinite(farBoundStartScale) || (farBoundStartScale!=0 && farBoundStartScale<2) ||
       !(tightenExpectedEvents>=0) || !std::isfinite(tightenExpectedEvents) ||
       (macrocellSize!=0 && (macrocellSize<2 || macrocellSize>64 || (macrocellSize&(macrocellSize-1)))) ||
       !positive(macrocellMaxExpectedEvents) ||
       (candidateMacroPruning && (macrocellSize==0 || trackingMethod!=TrackingMethod::Delta)) ||
       (candidateCollection!=CandidateCollection::Fine && candidateCollection!=CandidateCollection::Macro) ||
       candidateMacrocellSize<2 || candidateMacrocellSize>64 || (candidateMacrocellSize&(candidateMacrocellSize-1)) ||
       (candidateCollection==CandidateCollection::Macro &&
        (trackingMethod!=TrackingMethod::Delta || candidateMacroPruning)) ||
       (trackingMethod!=TrackingMethod::Delta && trackingMethod!=TrackingMethod::Inversion &&
        trackingMethod!=TrackingMethod::Regular) ||
       !positive(inversionMaxStepOverLength) || !positive(inversionMeanRangeOverSigma) ||
       !positive(inversionAbsoluteTolerance) || !positive(inversionRelativeTolerance) ||
       !positive(inversionDistanceToleranceOverLength) ||
       regularSubdivisions<1 || regularSubdivisions>4096 ||
       !positive(probability.absoluteTolerance) || !positive(probability.relativeTolerance) ||
       probability.initialSamples<16 || probability.maxSamples<probability.initialSamples)
        fail(cg::NumericStatus::InvalidInput,"single_history_gpis: invalid settings");
}

struct Flight::Impl {
    const cg::GPSSField &field;
    sh::Start birth;
    Settings settings;
    Snapshot state;
    Work work;
    double ell,sigma,selectionStep,originLogProbability=0;
    double statisticsInverseLength;
    double fragmentEnd=-1;
    double trackingFineEnd=-1;
    double regularCachedBegin=-1,regularCachedEnd=-1,regularCachedRate=0;
    std::optional<double> cachedLogHistoryFactor;
    std::unique_ptr<sh::HistoryCandidateEvaluator> candidateEvaluator;
    std::unique_ptr<cg::MeanRayCursor> geometry,queries;
    std::unique_ptr<cg::MeanRayCursor> trackingGeometry;
    // This certificate cursor follows exactly the normalized direction used
    // by standardizedMean, independently of geometry's polynomial anchors.
    std::unique_ptr<cg::MeanRayCursor> candidateBoundsCursor;
    cg::MeanRayBounds candidateBounds;
    double candidateBoundsBegin=-1,candidateBoundsEnd=-1,candidateBoundsRetryAt=0;
    double candidateBirthMean=0,candidateDeltaSlope=0;
    bool candidateBoundsUnavailableCounted=false;
    bool candidateOriginChecked=false,candidateOriginCertified=false;
    rb::Interval candidateOriginMean;
    std::unique_ptr<cg::ConditionalFlightKernel> baseline;
    cg::ConditionalBirthMajorant birthBound;

    Impl(const cg::GPSSField &f,const sh::Start &s,const Settings &opts,const Snapshot &snapshot)
        :field(f),birth(s),settings(opts),state(snapshot) {
        settings.validate();
        if(!field.mean || !birth.origin.allFinite() || !birth.direction.allFinite() ||
           std::abs(birth.direction.squaredNorm()-1)>1e-10 ||
           field.kernel.type()!=cg::CovarianceKernelType::SquaredExponential)
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: requires a unit ray and stationary SE kernel");
        const double precision=birth.direction.dot(field.kernel.precision()*birth.direction);
        sigma=field.kernel.sigma();ell=1/std::sqrt(precision);
        selectionStep=ell*settings.selectionStepOverLength;
        if(!positive(sigma) || !positive(ell) || !positive(selectionStep))
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: invalid kernel scale");
        if(birth.kind==sh::Start::SurfaceValueGradient &&
           (!birth.gradient.allFinite() || !(birth.gradient.dot(birth.direction)>0)))
            fail(cg::NumericStatus::NonPositiveSurfaceBirth,"single_history_gpis: surface birth must enter the empty side");
        if(!(state.age>=0) || !std::isfinite(state.age) || !(state.fragmentBegin>=0) ||
           state.fragmentBegin>state.age || state.age<groupBegin() || state.age>=groupEnd() ||
           (state.selected && !(*state.selected>0 && *state.selected<groupBegin())) ||
           !std::isfinite(state.inversionRemaining) ||
           !(state.inversionRemaining==-1 || state.inversionRemaining>=0) ||
           !(state.macroRetryAt>=0) || !std::isfinite(state.macroRetryAt) || state.macroRetryAt>groupEnd() ||
           state.completed.size()>std::size_t(settings.historyGroups))
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: invalid flight snapshot");
        geometry=field.mean->createRayCursor(birth.origin,birth.direction);
        queries=field.mean->createRayCursor(birth.origin,birth.direction);
        if(settings.candidateCollection==CandidateCollection::Macro) {
            if(!geometry)
                fail(cg::NumericStatus::InvalidInput,"single_history_gpis: macro candidates require macro-boundary support");
            trackingGeometry=field.mean->createRayCursor(birth.origin,birth.direction);
            if(!trackingGeometry)
                fail(cg::NumericStatus::InvalidInput,"single_history_gpis: macro candidates require a tracking cursor");
        }
        const auto statisticsDirection=cg::normalizedOrThrow(birth.direction);
        statisticsInverseLength=std::sqrt(statisticsDirection.dot(field.kernel.precision()*statisticsDirection));
        if(settings.candidateMacroPruning) {
            candidateBoundsCursor=field.mean->createRayCursor(birth.origin,statisticsDirection);
            if(candidateBoundsCursor && birth.kind==sh::Start::SurfaceValueGradient) {
                const auto birthMean=candidateBoundsCursor->evaluate(0);
                candidateBirthMean=birthMean.value;
                candidateDeltaSlope=statisticsDirection.dot(birth.gradient-birthMean.gradient);
            }
        }
        if(birth.kind==sh::Start::ExteriorPositiveOnly)
            originLogProbability=cg::normalLogCdf(mean(0).value/sigma);
    }
    double groupBegin() const { return double(state.group)*selectionStep; }
    double groupEnd() const { return double(state.group+1)*selectionStep; }
    bool detailedProfile() const { return settings.profile && settings.profileDetail; }
    bool detailedDelta() const { return detailedProfile() && settings.trackingMethod==TrackingMethod::Delta; }
    bool macroCandidates() const { return settings.candidateCollection==CandidateCollection::Macro; }
    void budget() const {
        if(work.intervals+work.candidates+work.cells+work.selections+
           work.hazardEvaluations+work.inversionRefinements>std::uint64_t(settings.maxWork)) {
            std::ostringstream message;message.precision(17);
            message<<"single_history_gpis: work budget exhausted (tracking or quadrature did not converge)"
                <<"; max_work="<<settings.maxWork<<", age="<<state.age
                <<", group="<<state.group<<", history="<<state.selected.value_or(-1)
                <<", log_history_factor="<<cachedLogHistoryFactor.value_or(-1)
                <<", intervals="<<work.intervals<<", candidates="<<work.candidates
                <<", hazards="<<work.hazardEvaluations<<", cells="<<work.cells
                <<", selections="<<work.selections<<", refinements="<<work.inversionRefinements
                <<", origin="<<birth.origin.transpose()<<", direction="<<birth.direction.transpose()
                <<", gradient="<<birth.gradient.transpose();
            throw cg::NumericError(cg::NumericStatus::SamplingNotConverged,message.str());
        }
    }
    void validateEnd(double end) const {
        if(!std::isfinite(end) || end<state.age)
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: queries must advance in physical distance");
    }
    cg::MeanJet mean(double t) {
        if(queries) return queries->evaluate(t);
        const auto p=(birth.origin+t*birth.direction).eval();
        if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get()))
            if((p-sphere->center()).squaredNorm()==0) return {-sphere->radius(),cg::Vector3::Zero()};
        return field.mean->evaluate(p);
    }
    double derivative(double t) {
        return geometry ? geometry->sample(t).derivative : mean(t).gradient.dot(birth.direction);
    }
    double nextFragment() {
        if(fragmentEnd>state.fragmentBegin) return fragmentEnd;
        Timer timer(settings.profile,work.selectionSeconds);
        Timer detailTimer(detailedProfile(),work.boundarySeconds);
        double end=groupEnd();
        if(macroCandidates()) {
            if(detailedDelta()) ++work.macroBoundaryQueries;
            if(!geometry->nextMacroBoundary(state.fragmentBegin,end,settings.candidateMacrocellSize))
                fail(cg::NumericStatus::InvalidInput,"single_history_gpis: mean does not support macro candidate boundaries");
        } else if(geometry) {
            if(detailedDelta()) ++work.fineBoundaryQueries;
            end=geometry->nextBoundary(state.fragmentBegin,end);
        }
        else {
            std::vector<double> knots;
            field.mean->appendRayBreakpoints(birth.origin,birth.direction,state.fragmentBegin,end,knots);
            for(double t:knots) if(t>state.fragmentBegin && t<end) end=t;
            if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
                const double t=(sphere->center()-birth.origin).dot(birth.direction);
                if(t>state.fragmentBegin && t<end) end=t;
            }
        }
        if(!(end>state.fragmentBegin) || !std::isfinite(end) || end<state.age || end>groupEnd())
            fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: geometry cursor cannot advance");
        return fragmentEnd=end;
    }
    double nextTrackingFineBoundary(double begin,double end) {
        if(trackingFineEnd>begin) return std::min(end,trackingFineEnd);
        Timer timer(settings.profile,work.selectionSeconds);
        Timer detailTimer(detailedProfile(),work.boundarySeconds);
        if(detailedDelta()) ++work.fineBoundaryQueries;
        trackingFineEnd=trackingGeometry->nextBoundary(begin,groupEnd());
        if(!(trackingFineEnd>begin) || !std::isfinite(trackingFineEnd))
            fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: tracking cursor cannot advance");
        return std::min(end,trackingFineEnd);
    }
    Candidate candidate(double t) {
        if(detailedDelta()) ++work.candidateStatistics;
        Timer detailTimer(detailedProfile(),work.candidateStatisticsSeconds);
        // Cache the flight's origin statistics, but start each point query
        // from a copy of its pristine cursor to preserve polynomial anchors
        // and roundoff ties between equal-height candidates.
        if(!candidateEvaluator)
            candidateEvaluator=std::make_unique<sh::HistoryCandidateEvaluator>(field,birth);
        const double z=candidateEvaluator->standardizedMean(t);
        Candidate c{t,z,cg::normalPdfOverCdf(z)};
        if(std::isnan(c.z)) fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: invalid candidate height");
        return c;
    }
    void consider(double t) {
        if(!(t>0) || t<groupBegin() || !(t<groupEnd())) return;
        if(state.current && state.current->distance==t) return;
        const auto c=candidate(t);
        if(!state.current || c.z<state.current->z ||
           (c.z==state.current->z && c.distance<state.current->distance)) state.current=c;
    }
    void fallbackRoots(double a,double b,std::vector<double> &roots) {
        // Procedural means without polynomial cursors use a deterministic
        // geometry heuristic. Only the trilinear cursor enumerates all extrema.
        const int count=std::max(1,int(std::ceil((b-a)/(ell/8))));
        for(int i=0;i<count;++i) {
            double lo=a+(b-a)*double(i)/count,hi=a+(b-a)*double(i+1)/count;
            const double dl=derivative(lo),dh=derivative(hi);
            if(!(dl<0 && dh>0)) continue;
            for(int k=0;k<48;++k) {
                const double mid=lo+0.5*(hi-lo);
                if(!(mid>lo && mid<hi)) break;
                if(derivative(mid)<0) lo=mid; else hi=mid;
            }
            roots.push_back(lo+0.5*(hi-lo));
        }
    }
    bool pruneCandidateFragment(double a,double b) {
        Timer detailTimer(detailedProfile(),work.candidateMacroBoundSeconds);
        const bool surface=birth.kind==sh::Start::SurfaceValueGradient;
        const auto unpruned=[&]() {
            if(detailedDelta()) ++work.candidateMacroUnprunedFragments;
            return false;
        };
        // Never replace an absent representative with a block statistic, and
        // leave the cancellation-sensitive birth prefix on its original path.
        if(!state.current || !std::isfinite(state.current->z) ||
           (surface && a*statisticsInverseLength<0.5)) return unpruned();
        if(!candidateBoundsCursor) {
            if(!candidateBoundsUnavailableCounted && detailedDelta()) ++work.candidateMacroUnavailable;
            candidateBoundsUnavailableCounted=true;
            return unpruned();
        }
        if(a<candidateBoundsRetryAt) return unpruned();
        if(!(candidateBoundsBegin<=a && b<=candidateBoundsEnd)) {
            double end=groupEnd();
            cg::MeanRayBounds bounds;
            if(detailedDelta()) ++work.candidateMacroQueries;
            bool available=false;
            try {
                available=candidateBoundsCursor->coarseRayBounds(a,end,bounds);
            } catch(const cg::NumericError &) {
                // This optional certificate is not the tracking bound. Keep
                // the original candidate path if it cannot certify a range.
            }
            if(!available) {
                // Numerical failures can be transient; retry in the next
                // selection group, without disabling this ray permanently.
                candidateBoundsRetryAt=groupEnd();
                if(detailedDelta()) ++work.candidateMacroUnavailable;
                return unpruned();
            }
            if(!bounds.certified || !(end>a && end<=groupEnd()) ||
               !std::isfinite(bounds.minimumValue) || !std::isfinite(bounds.maximumValue) ||
               bounds.minimumValue>bounds.maximumValue) return unpruned();
            candidateBoundsBegin=a;candidateBoundsEnd=end;candidateBounds=bounds;
            // Normalization can move a macro face slightly relative to the
            // geometry ray. A partially covered canonical fragment is retained.
            if(b>end) return unpruned();
        }
        rb::Interval mean(candidateBounds.minimumValue,candidateBounds.maximumValue);
        rb::Interval sd(1);
        double arithmeticScale=std::max(std::abs(mean.lower),std::abs(mean.upper))+sigma;
        if(surface) {
            if(!candidateOriginChecked) {
                candidateOriginChecked=true;
                // standardizedMean uses a cancellation-safe value difference
                // plus its original cursor's birth value. Enclose the true
                // birth value too, rather than assuming those floating point
                // evaluation paths coincide. This small raw-mean query is once
                // per eligible flight and never touches the geometry cursor.
                const double probeEnd=1e-6/statisticsInverseLength;
                if(positive(probeEnd)) {
                    SdfRayQueryOptions options;options.computeBeginDerivative=false;
                    try {
                        const auto originBounds=candidateBoundsCursor->rayBounds(0,probeEnd,options);
                        if(originBounds.certified && std::isfinite(originBounds.minimumValue) &&
                           std::isfinite(originBounds.maximumValue) &&
                           originBounds.minimumValue<=originBounds.maximumValue) {
                            candidateOriginMean={originBounds.minimumValue,originBounds.maximumValue};
                            candidateOriginCertified=true;
                        }
                    } catch(const cg::NumericError &) {
                        // An optional pruning certificate must not introduce a
                        // precision failure into an otherwise usable fine path.
                    }
                }
            }
            if(!candidateOriginCertified) return unpruned();
            const rb::Interval t(a,b),q=t*rb::Interval(statisticsInverseLength);
            const auto correction=rb::exp(rb::Interval(-0.5)*q*q)*
                (rb::Interval(-candidateBirthMean)+t*rb::Interval(candidateDeltaSlope));
            mean=mean+(rb::Interval(candidateBirthMean)-candidateOriginMean)+correction;
            arithmeticScale+=std::abs(candidateBirthMean)+b*std::abs(candidateDeltaSlope)+
                std::max(std::abs(candidateOriginMean.lower),std::abs(candidateOriginMean.upper));
            const auto variance=[&](double distance) {
                const auto q=rb::Interval(distance)*rb::Interval(statisticsInverseLength);
                const auto x=q*q;
                // The function 1-(1+x)exp(-x) is monotone for x>=0. Capping
                // the lower-distance evaluation at x=64 avoids inf*0 in the
                // far tail while retaining a safe, almost-unit lower bound.
                const rb::Interval y(std::min(64.0,x.lower),std::min(64.0,x.upper));
                const auto v=rb::Interval(1)-(rb::Interval(1)+y)*rb::exp(-y);
                constexpr double error=512*std::numeric_limits<double>::epsilon();
                return rb::Interval(std::max(0.0,rb::down(v.lower-error)),
                    x.upper>64 ? rb::up(1.0):std::min(rb::up(1.0),rb::up(v.upper+error)));
            };
            const auto lo=variance(a),hi=variance(b);
            sd=rb::sqrt(rb::Interval(lo.lower,hi.upper));
            if(!(sd.lower>0)) return unpruned();
        }
        // Include rounding in the oracle's value-difference/expm1 formula,
        // division, and (for first-hit) polynomial point evaluation. Strict
        // separation preserves the original equal-z, earliest-distance tie.
        const double error=8192*std::numeric_limits<double>::epsilon()*arithmeticScale;
        if(!std::isfinite(error)) return unpruned();
        mean.lower=rb::down(mean.lower-error);
        const auto z=rb::Interval(mean.lower)/rb::Interval(sigma)/sd;
        if(std::isfinite(z.lower) && z.lower>state.current->z) {
            if(detailedDelta()) ++work.candidateMacroPrunedFragments;
            return true;
        }
        return unpruned();
    }
    void collect(double a,double b) {
        ++work.cells;budget();
        if(!settings.historyEnabled) return;
        Timer timer(settings.profile,work.selectionSeconds);
        Timer detailTimer(detailedProfile(),work.collectSeconds);
        if(macroCandidates()) {
            if(detailedDelta()) ++work.macroCandidateFragments;
            // These are canonical geometry samples, committed only when the
            // whole fragment has been passed. Renderer cuts and null events
            // neither introduce candidates nor move this midpoint.
            consider(a);
            const double midpoint=a+0.5*(b-a);
            if(midpoint>a && midpoint<b) consider(midpoint);
            consider(b);
            return;
        }
        if(settings.candidateMacroPruning && pruneCandidateFragment(a,b)) return;
        consider(a);consider(b);
        std::vector<double> roots;
        if(!geometry || !geometry->appendStationaryPoints(a,b,roots)) fallbackRoots(a,b,roots);
        std::sort(roots.begin(),roots.end());
        roots.erase(std::unique(roots.begin(),roots.end()),roots.end());
        for(std::size_t i=0;i<roots.size();++i) {
            const double t=roots[i];
            if(!(t>a && t<b)) continue;
            const double left=i ? roots[i-1]:a,right=i+1<roots.size() ? roots[i+1]:b;
            const double dl=derivative(left+0.5*(t-left)),dr=derivative(t+0.5*(right-t));
            if(dl<=0 && dr>=0 && (dl<0 || dr>0)) consider(t);
        }
    }
    void finishGroup() {
        Timer timer(settings.profile,work.selectionSeconds);
        Timer detailTimer(detailedProfile(),work.groupSeconds);
        if(state.current) {
            state.completed.push_back(*state.current);
            if(state.completed.size()>std::size_t(settings.historyGroups)) state.completed.erase(state.completed.begin());
        }
        // Retention is by physical group age, including empty groups.
        ++state.group;
        state.macroRetryAt=0;
        if(!(groupEnd()>groupBegin()) || !std::isfinite(groupEnd()))
            fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: selection boundary resolution exhausted");
        const double oldest=double(state.group>std::uint64_t(settings.historyGroups)
            ? state.group-settings.historyGroups:0)*selectionStep;
        state.completed.erase(std::remove_if(state.completed.begin(),state.completed.end(),
            [&](const Candidate &c){return c.distance<oldest;}),state.completed.end());
        state.current.reset();state.selected.reset();
        cachedLogHistoryFactor.reset();
        if(!settings.historyEnabled) return;
        ++work.selections;budget();
        const double t=groupBegin(),reference=t-std::min(ell,t/2);
        Timer scorerTimer(detailedProfile(),work.scoringSeconds);
        const sh::HistoryCandidateScorer scorer(t*statisticsInverseLength,
            birth.kind==sh::Start::SurfaceValueGradient);
        scorerTimer.stop();
        double best=-1;
        auto score=[&](const Candidate &c) {
            const double u=c.distance;
            if(!(u>0 && u<t)) return;
            if(detailedDelta()) ++work.scoreEvaluations;
            double value;
            {
                Timer scoreTimer(detailedProfile(),work.scoringSeconds);
                value=scorer.score(u*statisticsInverseLength,c.inverseMills);
            }
            if(value>best || (value==best && (!state.selected || u<*state.selected))) {
                best=value;state.selected=u;
                cachedLogHistoryFactor=-originLogProbability-cg::normalLogCdf(c.z);
            }
        };
        bool haveReference=false;
        for(const auto &c:state.completed) {
            score(c);haveReference|=c.distance==reference;
        }
        if(!haveReference) score(candidate(reference));
    }
    void completeFragment(double end) {
        collect(state.fragmentBegin,end);
        state.fragmentBegin=end;fragmentEnd=-1;
        if(end==groupEnd()) finishGroup();
    }
    void advanceGeometry(double end) {
        validateEnd(end);
        while(state.age<end) {
            const double boundary=nextFragment();
            state.age=std::min(end,boundary);
            if(state.age==boundary) completeFragment(boundary);
        }
    }
    std::vector<double> history() const {
        return state.selected ? std::vector<double>{*state.selected}:std::vector<double>{};
    }
    void prepareBaseline() {
        if(baseline || birth.kind!=sh::Start::SurfaceValueGradient) return;
        Timer timer(settings.profile,work.majorantSeconds);
        Timer detailTimer(detailedProfile(),work.baselineSetupSeconds);
        cg::FlightState start;
        start.birthPosition=birth.origin;start.direction=birth.direction;start.birthGradient=birth.gradient;
        baseline=std::make_unique<cg::ConditionalFlightKernel>(field,start,
            settings.normalizedBoundOnly,settings.farBoundStartScale);
        birthBound=baseline->birthMajorant(std::min(selectionStep,baseline->maximumAgeInDomain()));
    }
    double logHistoryFactor() {
        if(cachedLogHistoryFactor) return *cachedLogHistoryFactor;
        Timer timer(settings.profile,work.majorantSeconds);
        Timer detailTimer(detailedProfile(),work.historyFactorSeconds);
        double result=-originLogProbability;
        if(state.selected) {
            const auto found=std::find_if(state.completed.begin(),state.completed.end(),
                [&](const Candidate &c){return c.distance==*state.selected;});
            result-=cg::normalLogCdf(found!=state.completed.end() ? found->z:candidate(*state.selected).z);
        }
        cachedLogHistoryFactor=result;
        return result;
    }
    double exteriorLogMajorant(const cg::MeanRayBounds &bounds) const {
        const double z=bounds.minimumValue/sigma;
        const double logMills=z<-8 ? std::log(cg::normalPdfOverCdf(z)) : cg::normalLogPdf(z)-cg::normalLogCdf(z);
        return logMills-std::log(sigma)+cg::negativePartMean(bounds.minimumDerivative,sigma/ell).logValue;
    }
    double maximum(double a,double b,double logFactor) {
        Timer timer(settings.profile,work.majorantSeconds);
        if(detailedDelta()) ++work.majorantCalls;
        double logBase;
        if(baseline) {
            Timer detailTimer(detailedProfile(),work.conditionalBoundSeconds);
            double bound;
            if(b<=birthBound.split) bound=birthBound.maximum;
            else if(b<=birthBound.smoothEnd) bound=baseline->birthAdjacentMajorant(a,b,birthBound);
            else {
                // The shortcut budget applies to the final history-amplified
                // rate, including its numerical guard. A vanishing threshold
                // conservatively requests the ordinary tightening path.
                const double threshold=settings.tightenExpectedEvents==0 ? 0 :
                    settings.tightenExpectedEvents*std::exp(-logFactor)/
                    (1+8*settings.probability.relativeTolerance+1e-10);
                bound=baseline->intervalMajorant(a,b,threshold);
            }
            logBase=std::log(bound);
        } else {
            Timer boundsTimer(detailedProfile(),work.exteriorBoundsSeconds);
            SdfRayQueryOptions options;options.computeBeginDerivative=false;
            cg::MeanRayBounds bounds;
            if(queries) bounds=queries->rayBounds(a,b,options);
            else if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
                const auto radial=(birth.origin-sphere->center()).eval();
                const double closest=std::clamp(-radial.dot(birth.direction),a,b);
                bounds.minimumValue=(radial+closest*birth.direction).norm()-sphere->radius();
                bounds.minimumDerivative=-1;bounds.certified=true;
            } else bounds=field.mean->rayBounds(birth.origin,birth.direction,a,b,options);
            if(!bounds.certified || !std::isfinite(bounds.minimumValue) || !std::isfinite(bounds.minimumDerivative))
                fail(cg::NumericStatus::InvalidMajorant,"single_history_gpis: missing finite certified mean bounds");
            boundsTimer.stop();
            Timer algebraTimer(detailedProfile(),work.exteriorAlgebraSeconds);
            logBase=exteriorLogMajorant(bounds);
        }
        // Analytic enclosure plus a guard for floating arithmetic and the
        // configured numerical CDF precision. Never clip a rare-sign factor.
        const double value=std::exp(logBase+logFactor)*(1+8*settings.probability.relativeTolerance+1e-10);
        if(!(value>=0) || !std::isfinite(value))
            fail(cg::NumericStatus::InvalidMajorant,"single_history_gpis: history majorant overflow");
        return std::nextafter(value,std::numeric_limits<double>::infinity());
    }
    bool macroMaximum(double a,double &b,double fineEnd,double logFactor,double &maximum) {
        if(state.macroUnavailable || a<state.macroRetryAt) {
            if(detailedDelta()) ++work.macroRetrySkips;
            return false;
        }
        ++work.intervals;budget();
        Timer timer(settings.profile,work.majorantSeconds);
        Timer detailTimer(detailedProfile(),work.macroBoundSeconds);
        if(detailedDelta()) {++work.macroAttempts;++work.majorantCalls;}
        const double attemptedEnd=b;
        if(!queries) {
            state.macroUnavailable=true;
            if(detailedDelta()) ++work.macroUnavailable;
            return false;
        }
        double logBase=0;
        bool available=false;
        if(baseline) {
            double base=0;
            try { available=baseline->coarseMajorant(a,b,base); }
            catch(const cg::NumericError &error) {
                // A valid but extremely loose conditional enclosure can
                // overflow before it is returned. A fine bound may resolve
                // it; uncertified inputs and other numerical errors propagate.
                const std::string message=error.what();
                if(error.status()!=cg::NumericStatus::InvalidMajorant ||
                   (message.find("overflow")==std::string::npos && message!="non-finite conditional majorant")) throw;
                state.macroRetryAt=std::min(attemptedEnd,std::max(a,b));
                if(detailedDelta()) ++work.macroRejected;
                return false;
            }
            if(available) {
                if(!(base>=0)) fail(cg::NumericStatus::InvalidMajorant,"single_history_gpis: invalid conditional macro bound");
                logBase=std::log(base);
            }
        } else {
            cg::MeanRayBounds bounds;
            available=queries->coarseRayBounds(a,b,bounds);
            if(available) {
                if(!bounds.certified || !std::isfinite(bounds.minimumValue) || !std::isfinite(bounds.minimumDerivative))
                    fail(cg::NumericStatus::InvalidMajorant,"single_history_gpis: missing finite certified macro mean bounds");
                logBase=exteriorLogMajorant(bounds);
            }
        }
        if(!available) {
            // A cursor can fail only for this window (for example a one-ULP
            // macro boundary). Do not disable later windows permanently.
            state.macroRetryAt=attemptedEnd;
            if(detailedDelta()) ++work.macroUnavailable;
            return false;
        }
        if(!(b>a && b<=attemptedEnd) || !std::isfinite(b))
            fail(cg::NumericStatus::InvalidMajorant,"single_history_gpis: invalid macro interval");
        const double value=std::exp(logBase+logFactor)*(1+8*settings.probability.relativeTolerance+1e-10);
        maximum=std::nextafter(value,std::numeric_limits<double>::infinity());
        if(!(b>fineEnd) || !(maximum>=0) || !std::isfinite(maximum) ||
           maximum*(b-a)>std::min(8.0,settings.macrocellMaxExpectedEvents)) {
            state.macroRetryAt=b;
            if(detailedDelta()) ++work.macroRejected;
            return false;
        }
        if(detailedDelta()) ++work.macroAccepted;
        return true;
    }
    double diagnosticMajorant(double begin,double end) {
        validateEnd(begin);
        if(!(end>begin) || !std::isfinite(end))
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: invalid diagnostic majorant interval");
        prepareBaseline();
        if(baseline && end>baseline->maximumAgeInDomain())
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: diagnostic interval exceeds conditional domain");
        advanceGeometry(begin);
        double boundary=nextFragment();
        if(macroCandidates()) boundary=nextTrackingFineBoundary(begin,boundary);
        if(end>boundary)
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: diagnostic interval crosses a canonical fragment");
        const double logFactor=logHistoryFactor();
        double result=0;
        for(double a=begin;a<end;) {
            double b=end;
            if(baseline) {
                if(a<birthBound.split) b=std::min(b,birthBound.split);
                else if(a<birthBound.smoothEnd) b=std::min(b,birthBound.smoothEnd);
            }
            if(!(b>a))
                fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: diagnostic majorant cannot advance");
            result=std::max(result,maximum(a,b,logFactor));
            a=b;
        }
        return result;
    }
    cg::MeanRayBounds integrationBounds(double a,double b) {
        SdfRayQueryOptions options;options.computeBeginDerivative=false;
        if(queries) return queries->rayBounds(a,b,options);
        // Avoid the sphere's undefined gradient at its center. Only the
        // complete value range is used here, never an extinction majorant.
        if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
            const auto radial=(birth.origin-sphere->center()).eval();
            const double closest=std::clamp(-radial.dot(birth.direction),a,b);
            cg::MeanRayBounds result;
            result.minimumValue=(radial+closest*birth.direction).norm()-sphere->radius();
            result.maximumValue=std::max((radial+a*birth.direction).norm(),
                (radial+b*birth.direction).norm())-sphere->radius();
            const double minimumRadius=(radial+closest*birth.direction).norm();
            // The canonical geometry split already isolates closest approach.
            // On a line through the center each open side is affine.
            const double impactSquared=radial.cross(birth.direction).squaredNorm();
            result.maximumSecondDerivative=impactSquared==0 ? 0 :
                std::nextafter(1/minimumRadius,std::numeric_limits<double>::infinity());
            result.certified=true;
            return result;
        }
        return field.mean->rayBounds(birth.origin,birth.direction,a,b,options);
    }
    double integrationEnd(double a,double end) {
        double b=std::min(end,a+ell*settings.inversionMaxStepOverLength);
        if(birth.kind==sh::Start::SurfaceValueGradient) {
            if(a==0) {
                // The near-birth return peak can occur at q=O(beta), even
                // when beta is arbitrarily small. An ell-sized first step
                // could put every quadrature node beyond that peak.
                const double beta=birth.gradient.dot(birth.direction)*ell/sigma;
                b=std::min(b,ell*std::min(settings.inversionMaxStepOverLength,0.25*beta));
                if(!(b>0))
                    fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: near-birth integration step underflow");
                auto initialBounds=integrationBounds(0,b);
                // A bound ending exactly on a cell face may conservatively
                // include the adjoining derivative jump. Query its interior.
                if(!std::isfinite(initialBounds.maximumSecondDerivative)) {
                    b*=0.5;
                    initialBounds=integrationBounds(0,b);
                }
                if(!initialBounds.certified || !(initialBounds.maximumSecondDerivative>=0) ||
                   !std::isfinite(initialBounds.maximumSecondDerivative))
                    fail(cg::NumericStatus::IntegrationNotConverged,
                        "single_history_gpis: inversion needs a finite one-sided near-birth curvature bound");
                const auto origin=mean(0);
                const double deltaSlope=birth.direction.dot(birth.gradient-origin.gradient);
                const double curvatureScale=1+std::abs(origin.value)/sigma+
                    initialBounds.maximumSecondDerivative*ell*ell/sigma+
                    2*std::abs(deltaSlope)*ell/sigma;
                b=std::min(b,ell*(0.25*beta/curvatureScale));
            } else if(a<ell/2) b=std::min(b,2*a);
        }
        for(;;) {
            if(!(b>a))
                fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: integration interval cannot advance");
            const auto bounds=integrationBounds(a,b);
            if(!bounds.certified || !std::isfinite(bounds.minimumValue) ||
               !std::isfinite(bounds.maximumValue) || bounds.maximumValue<bounds.minimumValue)
                fail(cg::NumericStatus::InvalidInput,"single_history_gpis: integration needs finite complete mean ranges");
            if(bounds.maximumValue-bounds.minimumValue<=sigma*settings.inversionMeanRangeOverSigma)
                return b;
            ++work.inversionRefinements;budget();
            b=a+0.5*(b-a);
        }
    }
    std::shared_ptr<sh::LocalLaw> evaluatedLaw(double t,double &hazard) {
        ++work.hazardEvaluations;budget();
        Timer timer(settings.profile,work.probabilitySeconds);
        std::shared_ptr<sh::LocalLaw> local;
        {
            Timer detailTimer(detailedProfile(),work.localLawSeconds);
            local=std::make_shared<sh::LocalLaw>(field,birth,t,history(),settings.probability);
        }
        {
            Timer detailTimer(detailedProfile(),work.hazardSeconds);
            hazard=local->hazard();
        }
        if(!(hazard>=0) || !std::isfinite(hazard))
            fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: invalid integration hazard");
        return local;
    }
    using HazardCache=std::map<double,double>;
    double integrationHazard(double t,HazardCache &cache) {
        const auto found=cache.find(t);
        if(found!=cache.end()) return found->second;
        double value;
        evaluatedLaw(t,value);
        cache.emplace(t,value);
        return value;
    }
    double integrationPoint(double a,double b) const {
        if(!(a>=0 && b>a) || !std::isfinite(b))
            fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: invalid quadrature interval");
        const double midpoint=a+0.5*(b-a);
        if(midpoint>a && midpoint<b) return midpoint;
        const double interior=std::nextafter(a,b);
        if(interior<b) return interior;
        // A cell/selection boundary may be only one ULP past the current
        // age. No distinct interior coordinate exists: retain its optical
        // depth using the positive left endpoint with this interval's fixed
        // history. At zero, only the positive right endpoint can be queried;
        // LocalLaw still reports any unresolved near-birth covariance.
        return a>0 ? a:b;
    }
    struct IntegrationLeaf { double begin,end,depth; };
    void integrate(double a,double b,double midpointHazard,HazardCache &cache,
                   std::vector<IntegrationLeaf> &leaves,int recursion=0) {
        ++work.intervals;budget();
        if(recursion>60)
            fail(cg::NumericStatus::IntegrationNotConverged,"single_history_gpis: quadrature depth exhausted");
        const double width=b-a,mid=a+0.5*width;
        const double left=a+0.25*width,right=a+0.75*width;
        if(!(a<left && left<mid && mid<right && right<b)) {
            const double depth=width*midpointHazard;
            if(!(depth>=0) || !std::isfinite(depth)) {
                std::ostringstream message;
                message.precision(17);
                message<<"single_history_gpis: unresolved machine-width optical depth at ["
                       <<a<<", "<<b<<"] width="<<width;
                throw cg::NumericError(cg::NumericStatus::NeedHigherPrecision,message.str());
            }
            leaves.push_back({a,b,depth});
            return;
        }
        const double leftHazard=integrationHazard(left,cache),rightHazard=integrationHazard(right,cache);
        const double coarse=width*midpointHazard;
        const double fine=(0.5*width)*leftHazard+(0.5*width)*rightHazard;
        const double error=std::abs(fine-coarse)/3;
        const double tolerance=settings.inversionAbsoluteTolerance*(width/ell)+
            settings.inversionRelativeTolerance*fine;
        if(std::isfinite(fine) && std::isfinite(coarse) && error<=tolerance) {
            // Retain the nonnegative fine midpoint estimate. Richardson
            // extrapolation could produce a negative optical depth.
            leaves.push_back({a,b,fine});
            return;
        }
        ++work.inversionRefinements;budget();
        integrate(a,mid,leftHazard,cache,leaves,recursion+1);
        integrate(mid,b,rightHazard,cache,leaves,recursion+1);
    }
    std::vector<IntegrationLeaf> integrationLeaves(double a,double b,HazardCache &cache) {
        std::vector<IntegrationLeaf> leaves;
        integrate(a,b,integrationHazard(integrationPoint(a,b),cache),cache,leaves);
        return leaves;
    }
    double integratedDepth(double a,double b,HazardCache &cache) {
        double result=0;
        for(const auto &leaf:integrationLeaves(a,b,cache)) result+=leaf.depth;
        return result;
    }
    TrackResult trackInversion(double end,cg::Random &random,bool collision,double &weight) {
        validateEnd(end);weight=1;
        if(state.age==end) return {false,end,{}};
        QuadratureTimer timer(settings.profile,work);
        // Match the conditional tracker's domain restriction without creating
        // a ConditionalFlightKernel or any conditional majorant.
        if(birth.kind==sh::Start::SurfaceValueGradient) {
            const auto domain=field.activeDomain.intersect({birth.origin,birth.direction});
            if(!domain.hit || end>domain.exit)
                fail(cg::NumericStatus::InvalidInput,"single_history_gpis: segment exceeds conditional domain");
        }
        if(collision && state.inversionRemaining<0)
            state.inversionRemaining=-std::log(random.openUniform01());
        while(state.age<end) {
            const double boundary=nextFragment(),a=state.age;
            const double b=integrationEnd(a,std::min(end,boundary));
            // Cached values belong to this fixed-history interval only. The
            // query cursor may move, but the candidate geometry cursor does
            // not observe any quadrature or inversion sample locations.
            HazardCache cache;
            const auto leaves=integrationLeaves(a,b,cache);
            for(const auto &leaf:leaves) {
                if(collision && state.inversionRemaining<=leaf.depth && leaf.depth>0) {
                    double lo=leaf.begin,hi=leaf.end;
                    while(hi-lo>settings.inversionDistanceToleranceOverLength*
                          std::min(ell,lo+0.5*(hi-lo))) {
                        const double mid=lo+0.5*(hi-lo);
                        if(!(mid>lo && mid<hi)) break;
                        ++work.inversionRefinements;budget();
                        const double partial=integratedDepth(leaf.begin,mid,cache);
                        if(partial<state.inversionRemaining) lo=mid; else hi=mid;
                    }
                    const double hit=integrationPoint(lo,hi);
                    double hazard;
                    auto local=evaluatedLaw(hit,hazard);
                    state.age=hit;state.inversionRemaining=0;
                    return {true,hit,std::move(local)};
                }
                if(collision) state.inversionRemaining-=leaf.depth;
                else weight*=std::exp(-leaf.depth);
                state.age=leaf.end;
                // Geometry is admitted only once an entire canonical
                // fragment has been passed, including at renderer cuts.
                if(state.age==boundary) completeFragment(boundary);
                if(!collision) {
                    if(weight==0) return {false,state.age,{}};
                    if(settings.shadowRouletteThreshold>0 && weight<settings.shadowRouletteThreshold) {
                        ++work.rouletteTests;
                        const double survive=weight/settings.shadowRouletteThreshold;
                        if(random.openUniform01()>=survive) {
                            ++work.rouletteTerminations;weight=0;
                            return {false,state.age,{}};
                        }
                        weight=settings.shadowRouletteThreshold;
                    }
                }
            }
        }
        return {false,end,{}};
    }
    TrackResult trackRegular(double end,cg::Random &random,bool collision,double &weight) {
        validateEnd(end);weight=1;
        if(state.age==end) return {false,end,{}};
        QuadratureTimer timer(settings.profile,work);
        if(birth.kind==sh::Start::SurfaceValueGradient) {
            const auto domain=field.activeDomain.intersect({birth.origin,birth.direction});
            if(!domain.hit || end>domain.exit)
                fail(cg::NumericStatus::InvalidInput,"single_history_gpis: segment exceeds conditional domain");
        }
        if(collision && state.inversionRemaining<0)
            state.inversionRemaining=-std::log(random.openUniform01());
        while(state.age<end) {
            const double boundary=nextFragment(),fragmentBegin=state.fragmentBegin;
            const int count=settings.regularSubdivisions;
            const auto subdivisionBoundary=[&](int i) {
                return i==count ? boundary : fragmentBegin+(boundary-fragmentBegin)*(double(i)/count);
            };
            // Find the first representable subdivision boundary after age.
            // This also skips collapsed subdivisions of ULP-sized fragments.
            int lo=1,hi=count;
            while(lo<hi) {
                const int mid=lo+(hi-lo)/2;
                if(subdivisionBoundary(mid)>state.age) hi=mid;
                else lo=mid+1;
            }
            const double pieceBegin=subdivisionBoundary(lo-1),pieceEnd=subdivisionBoundary(lo);
            if(!(pieceEnd>state.age) || pieceBegin>state.age)
                fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: regular subdivision cannot advance");
            ++work.intervals;budget();
            if(pieceBegin!=regularCachedBegin || pieceEnd!=regularCachedEnd) {
                double rate;
                evaluatedLaw(integrationPoint(pieceBegin,pieceEnd),rate);
                regularCachedBegin=pieceBegin;regularCachedEnd=pieceEnd;regularCachedRate=rate;
            }
            const double a=state.age,b=std::min(end,pieceEnd),rate=regularCachedRate;
            const double depth=rate*(b-a);
            if(collision && rate>0 && state.inversionRemaining<=depth) {
                // The exponential is inverted exactly for the approximate
                // piecewise constant rate. Clamp only arithmetic roundoff at
                // interval endpoints, never a rate or a sign probability.
                const double hit=std::clamp(a+state.inversionRemaining/rate,a,b);
                if(!(hit>0))
                    fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: regular hit has no positive representable distance");
                std::shared_ptr<sh::LocalLaw> local;
                {
                    Timer probabilityTimer(settings.profile,work.probabilitySeconds);
                    Timer detailTimer(detailedProfile(),work.localLawSeconds);
                    local=std::make_shared<sh::LocalLaw>(field,birth,hit,history(),settings.probability);
                }
                // The zero-value density can underflow even though the
                // conditional crossing-speed law has representable mass.
                // sampleSpeed validates that law's own normalization. This
                // regular hit needs no actual-position hazard evaluation.
                state.age=hit;state.inversionRemaining=0;
                return {true,hit,std::move(local)};
            }
            if(collision) state.inversionRemaining-=depth;
            else weight*=std::exp(-depth);
            state.age=b;
            // Renderer cuts within a piece preserve its canonical midpoint,
            // subdivision endpoints, and residual optical depth. Only passed
            // full geometry fragments contribute selection candidates.
            if(b==boundary) completeFragment(boundary);
            if(!collision) {
                if(weight==0) return {false,state.age,{}};
                if(settings.shadowRouletteThreshold>0 && weight<settings.shadowRouletteThreshold) {
                    ++work.rouletteTests;
                    const double survive=weight/settings.shadowRouletteThreshold;
                    if(random.openUniform01()>=survive) {
                        ++work.rouletteTerminations;weight=0;
                        return {false,state.age,{}};
                    }
                    weight=settings.shadowRouletteThreshold;
                }
            }
        }
        return {false,end,{}};
    }
    TrackResult track(double end,cg::Random &random,bool collision,double &weight) {
        if(settings.trackingMethod==TrackingMethod::Inversion)
            return trackInversion(end,random,collision,weight);
        if(settings.trackingMethod==TrackingMethod::Regular)
            return trackRegular(end,random,collision,weight);
        validateEnd(end);weight=1;
        if(state.age==end) return {false,end,{}};
        prepareBaseline();
        if(baseline && end>baseline->maximumAgeInDomain())
            fail(cg::NumericStatus::InvalidInput,"single_history_gpis: segment exceeds conditional domain");
        while(state.age<end) {
            const double boundary=nextFragment();
            double a=state.age,b=std::min(end,boundary);
            double bound=0;
            bool coarse=false;
            const double macroEnd=std::min(end,groupEnd());
            if(settings.macrocellSize && (!baseline || a>=birthBound.smoothEnd)) {
                if(macroCandidates()) {
                    // The collection boundary is already a cheap macro face.
                    // Attempt a coarse tracking bound before requesting any
                    // fine boundary. Tracking/candidate macro sizes may differ.
                    b=std::min(end,boundary);
                    coarse=macroMaximum(a,b,a,logHistoryFactor(),bound);
                } else if(boundary<macroEnd) {
                    b=macroEnd;
                    coarse=macroMaximum(a,b,boundary,logHistoryFactor(),bound);
                }
            }
            if(!coarse) {
                b=std::min(end,boundary);
                if(macroCandidates()) b=nextTrackingFineBoundary(a,b);
                if(baseline) {
                    if(a<birthBound.split) b=std::min(b,birthBound.split);
                    else if(a<ell/2) b=std::min(b,2*a);
                }
                const double logFactor=logHistoryFactor();
                const bool accelerateRefinement=state.selected && logFactor>0 &&
                    (!baseline || a>=birthBound.smoothEnd);
                for(int refinement=0;;++refinement) {
                    ++work.intervals;budget();
                    bound=maximum(a,b,logFactor);
                    if(bound*(b-a)<=8 || refinement>=32) break;
                    // A rare history sign can amplify the bound almost
                    // uniformly. Skip repeated halvings in that case, but
                    // limit the reduction: a loose bound may collapse on a
                    // short prefix. Keep ordinary bisection near birth and
                    // without history. Every shortened bound is recertified.
                    const double width=b-a;
                    const double length=accelerateRefinement
                        ? std::max(width/8,std::min(width/2,8.0/bound))
                        : width/2;
                    double mid=a+length;
                    if(mid-a>length) mid=std::nextafter(mid,a);
                    if(!(mid>a && mid<b)) break;
                    if(detailedDelta()) ++work.eventSplits;
                    b=mid;
                }
            }
            if(!(b>a)) fail(cg::NumericStatus::NeedHigherPrecision,"single_history_gpis: tracking interval cannot advance");
            if(detailedDelta()) ++work.processedIntervals;
            bool haveCandidate=false;
            double candidate;
            while(state.clock.advance(random,bound,a,b,candidate)) {
                haveCandidate=true;
                ++work.candidates;budget();
                std::shared_ptr<sh::LocalLaw> local;
                double hazard;
                {
                    ++work.hazardEvaluations;budget();
                    Timer timer(settings.profile,work.probabilitySeconds);
                    {
                        Timer detailTimer(detailedProfile(),work.localLawSeconds);
                        local=std::make_shared<sh::LocalLaw>(field,birth,candidate,history(),settings.probability);
                    }
                    {
                        Timer detailTimer(detailedProfile(),work.hazardSeconds);
                        hazard=local->hazard();
                    }
                }
                if(!(hazard>=0) || !std::isfinite(hazard) || hazard>bound*(1+1e-10)) {
                    std::ostringstream message;
                    message<<"single_history_gpis: invalid majorant at t="<<candidate
                           <<" hazard="<<hazard<<" bound="<<bound;
                    throw cg::NumericError(cg::NumericStatus::InvalidMajorant,message.str());
                }
                if(detailedDelta() && hazard==0) ++work.zeroHazardCandidates;
                const double acceptance=std::min(1.0,hazard/bound);
                if(collision) {
                    if(random.openUniform01()<acceptance) {
                        if(detailedDelta()) ++work.acceptedCollisions;
                        if(coarse) advanceGeometry(candidate); else state.age=candidate;
                        return {true,candidate,std::move(local)};
                    }
                    if(detailedDelta()) ++work.rejectedCandidates;
                } else {
                    weight*=1-acceptance;
                    if(weight==0) {
                        if(coarse) advanceGeometry(candidate); else state.age=candidate;
                        return {false,candidate,{}};
                    }
                    if(settings.shadowRouletteThreshold>0 && weight<settings.shadowRouletteThreshold) {
                        ++work.rouletteTests;
                        const double survive=weight/settings.shadowRouletteThreshold;
                        if(random.openUniform01()>=survive) {
                            ++work.rouletteTerminations;weight=0;
                            if(coarse) advanceGeometry(candidate); else state.age=candidate;
                            return {false,candidate,{}};
                        }
                        weight=settings.shadowRouletteThreshold;
                    }
                }
                a=candidate;
            }
            if(detailedDelta() && !haveCandidate) ++work.emptyIntervals;
            // Only complete canonical collection fragments contribute
            // candidates. Fine tracking subdivisions, null events and renderer
            // cuts never create candidates in the macro collection mode.
            if(coarse) advanceGeometry(b);
            else {
                state.age=b;
                if(b==boundary) completeFragment(boundary);
            }
        }
        return {false,end,{}};
    }
};

Flight::Flight(const cg::GPSSField &field,const sh::Start &birth,const Settings &settings,const Snapshot &snapshot)
    :_impl(std::make_unique<Impl>(field,birth,settings,snapshot)) {}
Flight::~Flight()=default;
const Snapshot &Flight::snapshot() const {return _impl->state;}
const Work &Flight::work() const {return _impl->work;}
double Flight::lengthScale() const {return _impl->ell;}
Eigen::Vector3d Flight::meanGradient(double distance) {return _impl->mean(distance).gradient;}
std::vector<double> Flight::selectHistory(double distance) {
    _impl->advanceGeometry(distance);return _impl->history();
}
sh::LocalLaw Flight::law(double distance) {
    const auto selected=selectHistory(distance);
    return sh::LocalLaw(_impl->field,_impl->birth,distance,selected,_impl->settings.probability);
}
double Flight::diagnosticMajorant(double begin,double end) {
    return _impl->diagnosticMajorant(begin,end);
}
TrackResult Flight::sampleDistance(double end,cg::Random &random) {
    double weight;return _impl->track(end,random,true,weight);
}
double Flight::transmittance(double end,cg::Random &random) {
    double weight;_impl->track(end,random,false,weight);return weight;
}

} // namespace Tungsten::SingleHistoryGpis
