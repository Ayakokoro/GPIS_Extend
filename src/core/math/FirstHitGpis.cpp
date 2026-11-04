#include "FirstHitGpis.hpp"
#include "FirstHitGaussian.hpp"
#include "math/ClassicGlobalMath.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <map>
#include <sstream>
#include <vector>

namespace Tungsten::FirstHitGpis {
namespace cg = ConditionalGpis;
namespace {
constexpr double Pi = 3.14159265358979323846;
class ScopedTimer {
    using Clock=std::chrono::steady_clock;
    std::uint64_t *_total, *_maximum;
    Clock::time_point _start;
public:
    explicit ScopedTimer(std::uint64_t *total, std::uint64_t *maximum=nullptr)
        : _total(total), _maximum(maximum) { if (_total) _start=Clock::now(); }
    ~ScopedTimer() {
        if (!_total) return;
        const auto elapsed=std::uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-_start).count());
        *_total+=elapsed;
        if (_maximum) *_maximum=std::max(*_maximum,elapsed);
    }
};
[[noreturn]] void unresolved(const char *message)
{
    throw cg::NumericError(cg::NumericStatus::SamplingNotConverged, message);
}
}

void Settings::validate() const
{
    if (!std::isfinite(probabilityTolerance) || probabilityTolerance <= 0.0 || probabilityTolerance >= 0.1 ||
        !std::isfinite(distanceTolerance) || distanceTolerance <= 0.0 ||
        maxNodes < 2 || maxNodes > 256 || qmcSamples < 16 || maxQmcSamples < qmcSamples ||
        maxQmcSamples > 1048576 || maxRefinements < 1 || maxRefinements > 100000)
        throw cg::NumericError(cg::NumericStatus::InvalidInput, "first_hit_gpis: invalid numerical settings");
}

struct Flight::Impl {
    struct Integral { double value = 0.0, error = 0.0; };
    struct Segment { double begin, end; Integral rice; };
    MeanRay mean;
    double sigma, ell, length, p0;
    Settings settings;
    FlightStatistics *statistics;
    std::map<double, MeanSample> meanCache;
    std::map<std::pair<double,double>,double> minimumCache;
    std::map<std::pair<double, double>, ProbabilityEstimate> pairCache;
    std::map<std::pair<double, double>, Integral> integralCache;
    std::vector<Segment> segments;
    std::vector<double> candidates, selected;
    bool prepared = false;

    Impl(MeanRay m, double s, double e, double l, Settings opts, FlightStatistics *stats)
        : mean(std::move(m)), sigma(s), ell(e), length(l), settings(opts), statistics(stats)
    {
        if (statistics) *statistics=FlightStatistics{};
        settings.validate();
        if (!mean.sample || !std::isfinite(sigma) || sigma <= 0.0 ||
            !std::isfinite(ell) || ell <= 0.0 || !std::isfinite(length) || length < 0.0 ||
            !std::isfinite(mean.gradientBound) || mean.gradientBound < 0.0)
            throw cg::NumericError(cg::NumericStatus::InvalidInput, "first_hit_gpis: invalid finite flight");
        p0 = cg::normalCdf(at(0.0).value/sigma);
        // Joint-probability arithmetic is deliberately not advertised as a
        // rare-event solver. Never silently replace this conditioning by Y0=0.
        if (p0 < 1e-10)
            throw cg::NumericError(cg::NumericStatus::UnresolvedRareEvent,
                                  "first_hit_gpis: origin positive probability below 1e-10");
    }

    MeanSample at(double t)
    {
        auto found = meanCache.find(t);
        if (found != meanCache.end()) return found->second;
        if (statistics) ++statistics->meanQueries;
        const auto result = mean.sample(t);
        if (!std::isfinite(result.value) || !std::isfinite(result.derivative) ||
            std::abs(result.derivative) > mean.gradientBound*(1.0+1e-5)+1e-12)
            throw cg::NumericError(cg::NumericStatus::InvalidInput,
                                  "first_hit_gpis: invalid mean or directional gradient bound");
        meanCache.emplace(t, result);
        return result;
    }

    double minimum(double a, double b)
    {
        const auto key=std::make_pair(a,b);
        const auto found=minimumCache.find(key);
        if (found!=minimumCache.end()) return found->second;
        const double value = mean.minimum ? mean.minimum(a,b)
            : at(0.5*(a+b)).value - mean.gradientBound*0.5*(b-a);
        if (!std::isfinite(value))
            throw cg::NumericError(cg::NumericStatus::InvalidInput, "first_hit_gpis: invalid interval mean bound");
        minimumCache.emplace(key,value);
        return value;
    }

    Integral integrate(double a, double b)
    {
        if (a == b) return {};
        const auto key = std::make_pair(a,b);
        auto found = integralCache.find(key);
        if (found != integralCache.end()) return found->second;
        const double lower = minimum(a,b);
        const double derivativeStddev = sigma/ell;
        // In an air interval bound the EXPECTED crossing count. The simpler
        // air-event probability is not an expected count and must not be
        // substituted here before subtracting the endpoint crossing q.
        const double upper=at(0.5*(a+b)).value+mean.gradientBound*0.5*(b-a);
        const double minimumMagnitude=lower>=0.0 ? lower : std::max(0.0,-upper);
        const double countBound = (b-a)*cg::normalPdf(minimumMagnitude/sigma)/sigma*
                ClassicGlobal::negativePartMean(-mean.gradientBound, derivativeStddev);
        Integral result;
        if (countBound <= settings.probabilityTolerance*p0*0.02*(b-a)/std::max(length,ell)) {
            result = {0.5*countBound, 0.5*countBound};
        } else {
            cg::NumericPolicy policy;
            policy.absoluteTolerance = std::max(1e-14, settings.probabilityTolerance*p0*1e-4*
                                                       (b-a)/std::max(length,ell));
            policy.relativeTolerance = 1e-8;
            policy.maxQuadratureSubdivisions = 512;
            auto f = [&](double t) {
                const auto m = at(t);
                return cg::normalPdf(m.value/sigma)/sigma *
                    ClassicGlobal::negativePartMean(m.derivative, derivativeStddev);
            };
            const auto r = cg::integrateFinite(f,a,b,policy);
            if (r.status != cg::NumericStatus::Ok)
                throw cg::NumericError(r.status, "first_hit_gpis: Rice quadrature did not converge");
            result = {r.value,r.absError};
        }
        integralCache.emplace(key,result);
        return result;
    }

    void prepare()
    {
        if (prepared) return;
        prepared = true;
        candidates.push_back(0.0);
        // Integration knots and potential CDF constraints are different sets.
        // Crossing smooth cells never automatically increases CDF dimension.
        for (double a = 0.0; a < length;) {
            double step = 0.5*ell;
            if (mean.gradientBound > 0.0) {
                // A narrow Rice peak has width sigma/|mean'|, which can be
                // much smaller than ell. Do not let all quadrature nodes jump
                // over it. Far from zero the Lipschitz bound allows long steps
                // on either side of the interface.
                step=std::min(step,sigma/mean.gradientBound);
                step=std::max(step,std::abs(at(a).value)/(2.0*mean.gradientBound));
            }
            else step = length;
            double b = std::min(length,a+step);
            if (mean.nextBoundary) b = mean.nextBoundary(a,b);
            if (!(b > a) || b > length || !std::isfinite(b))
                unresolved("first_hit_gpis: interpolation cursor made no progress");
            if (segments.size() >= 100000)
                unresolved("first_hit_gpis: exceeded 100000 integration intervals");
            const auto rice = integrate(a,b);
            segments.push_back({a,b,rice});
            candidates.push_back(0.5*(a+b));
            candidates.push_back(b);
            // A derivative sign change isolates a local SDF minimum. Midpoints
            // remain candidates even when a procedural mean has other extrema.
            const double inset = (b-a)*1e-6;
            if (at(a+inset).derivative < 0.0 && at(b-inset).derivative > 0.0) {
                double lo=a+inset, hi=b-inset;
                for (int k=0;k<24;++k) {
                    const double c=0.5*(lo+hi);
                    if (at(c).derivative < 0.0) lo=c; else hi=c;
                }
                candidates.push_back(0.5*(lo+hi));
            }
            a=b;
        }
        std::sort(candidates.begin(),candidates.end());
        candidates.erase(std::unique(candidates.begin(),candidates.end()),candidates.end());
        if (statistics) statistics->candidateLocations=candidates.size();
    }

    Integral rice(double a, double b)
    {
        Integral result;
        auto it=std::lower_bound(segments.begin(),segments.end(),a,
            [](const Segment &segment,double v) { return segment.end <= v; });
        for (;it!=segments.end() && it->begin<b;++it) {
            const double lo=std::max(a,it->begin), hi=std::min(b,it->end);
            const auto v=(lo==it->begin && hi==it->end) ? it->rice : integrate(lo,hi);
            result.value+=v.value; result.error+=v.error;
        }
        return result;
    }

    ProbabilityEstimate pair(double a,double b)
    {
        const auto key=std::make_pair(a,b);
        auto it=pairCache.find(key);
        if (it!=pairCache.end()) return it->second;
        const double meanA=at(a).value/sigma,meanB=at(b).value/sigma;
        ProbabilityEstimate q;
        {
            ScopedTimer timer(statistics ? &statistics->pairNanoseconds : nullptr);
            if (statistics) ++statistics->pairCalls;
            q=downcrossingPair(meanA,meanB,(b-a)/ell);
        }
        pairCache.emplace(key,q);
        return q;
    }

    double gapError(double a,double b)
    {
        const auto count=rice(a,b);
        const auto q=pair(a,b);
        if (q.value-q.error>count.value+count.error+1e-12)
            throw cg::NumericError(cg::NumericStatus::IntegrationNotConverged,
                                  "first_hit_gpis: Rice integral is below endpoint crossing probability");
        double error=std::max(0.0,count.value+count.error-std::max(0.0,q.value-q.error));
        const double k=minimum(a,b)/sigma;
        if (k>=0.0) {
            // This event bound covers the WHOLE active gap, including the
            // left endpoint. It can safely tighten that gap's hidden-event bound.
            const double air=cg::normalCdf(-k)+(b-a)/(2.0*Pi*ell)*std::exp(-0.5*k*k);
            error=std::min(error,air);
        }
        return error;
    }

    double chooseCandidate(double a,double b,double &gain)
    {
        std::vector<double> choices{0.5*(a+b)};
        double lowest=std::numeric_limits<double>::infinity(), lowT=choices.front();
        for (auto it=std::upper_bound(candidates.begin(),candidates.end(),a);
             it!=candidates.end() && *it<b;++it) {
            const double value=at(*it).value;
            if (value<lowest) { lowest=value; lowT=*it; }
        }
        choices.push_back(lowT);
        // Off-centre choices avoid a zero-gain symmetric split on long intervals.
        choices.push_back(a+0.25*(b-a)); choices.push_back(a+0.75*(b-a));
        double best=choices.front(); gain=-1.0;
        const auto old=pair(a,b);
        for (double c:choices) {
            if (!(c>a && c<b)) continue;
            if (statistics) ++statistics->candidateTests;
            const auto left=pair(a,c),right=pair(c,b);
            const double g=std::max(0.0,left.value+right.value-old.value);
            if (g>gain) { gain=g; best=c; }
        }
        return best;
    }

    bool precheck(double t, const std::vector<double> &nodes, double uniform,
                  Bounds &bounds, bool &riceChecked)
    {
        ScopedTimer timer(statistics ? &statistics->precheckNanoseconds : nullptr);
        if (statistics) ++statistics->precheckCalls;
        bounds.nodes=int(nodes.size());
        const double roundoff=32.0*std::numeric_limits<double>::epsilon();
        auto decision = [&](bool hit, bool marginal=false) {
            if (statistics) {
                if (hit) ++statistics->precheckHit; else ++statistics->precheckSurvived;
                if (marginal) ++statistics->precheckMarginalHits;
                if (nodes.size()>2) ++statistics->precheckWithExtraNodes;
            }
            return true;
        };
        auto validate = [&] {
            if (bounds.lower>bounds.upper+1e-12)
                unresolved("first_hit_gpis: inconsistent precheck probability envelope");
        };

        // Survival implies positivity at every retained location. Test the
        // one-dimensional upper bound before doing any additional pair work.
        for (size_t i=1;i<nodes.size();++i) {
            const double positive=cg::normalCdf(at(nodes[i]).value/sigma);
            bounds.upper=std::min(bounds.upper,std::min(p0,positive+roundoff)/p0);
            validate();
            if (uniform>=bounds.upper) return decision(true,true);
        }

        // With a positive origin, a collision requires a downward crossing.
        // Its probability is no larger than the unconditional expected count
        // divided by p0. The prepared integral and this lower bound are reused
        // when spatial refinement adds another node to the same prefix.
        if (!riceChecked) {
            const auto count=rice(0.0,t);
            bounds.lower=std::clamp(1.0-(count.value+count.error+
                roundoff*(1.0+std::abs(count.value)))/p0,0.0,1.0);
            riceChecked=true;
        }
        validate();
        if (uniform<bounds.lower) return decision(false);

        // q(0,v) is the joint event that the origin is positive and v is not.
        // Subtract its LOWER estimate to obtain an UPPER survival bound.
        // In particular, never replace the origin by the last bisection point.
        for (size_t i=1;i<nodes.size();++i) {
            const auto q=pair(0.0,nodes[i]);
            const double crossingLower=std::max(0.0,q.value-q.error-roundoff);
            bounds.upper=std::min(bounds.upper,std::clamp(1.0-crossingLower/p0,0.0,1.0));
            validate();
            if (uniform>=bounds.upper) return decision(true);
        }
        // An unresolved threshold must use the existing full refinement, even
        // if these cheap bounds happen to be within probabilityTolerance.
        return false;
    }

    Bounds query(double t, double uniform, bool thresholdQuery)
    {
        if (!std::isfinite(t) || t<0.0 || t>length)
            throw cg::NumericError(cg::NumericStatus::InvalidInput,"first_hit_gpis: prefix outside flight");
        if (statistics) ++statistics->prefixQueries;
        if (t==0.0) return {1.0,1.0,1};
        prepare();
        std::vector<double> nodes{0.0,t};
        for (double v:selected)
            if (v>0.0 && v<t && nodes.size()<size_t(settings.maxNodes)) nodes.push_back(v);
        std::sort(nodes.begin(),nodes.end());
        int samples=settings.qmcSamples;
        std::unique_ptr<OrthantIntegrator> orthant;
        ProbabilityEstimate probability{};
        Bounds cheapBounds;
        bool riceChecked=false;
        bool refresh=true;
        for (int iteration=0;iteration<settings.maxRefinements;++iteration) {
            if (refresh) {
                if (!orthant) {
                    if (thresholdQuery && settings.survivalPrecheck &&
                        precheck(t,nodes,uniform,cheapBounds,riceChecked)) return cheapBounds;
                    std::vector<double> positions, means;
                    for (double v:nodes) { positions.push_back(v/ell); means.push_back(at(v).value/sigma); }
                    orthant=std::make_unique<OrthantIntegrator>(positions,means);
                }
                if (statistics) {
                    ++statistics->orthantCalls;
                    statistics->maxActiveNodes=std::max(statistics->maxActiveNodes,std::uint64_t(nodes.size()));
                    OrthantDiagnostics diagnostics;
                    const auto previousTime=statistics->orthantNanoseconds;
                    {
                        ScopedTimer timer(&statistics->orthantNanoseconds,&statistics->maxOrthantCallNanoseconds);
                        probability=orthant->extendTo(samples,&diagnostics);
                    }
                    const auto elapsed=statistics->orthantNanoseconds-previousTime;
                    statistics->maxEffectiveDimension=std::max(statistics->maxEffectiveDimension,diagnostics.effectiveDimension);
                    if (diagnostics.effectiveDimension>=3) {
                        ++statistics->multivariateCalls;
                        statistics->multivariateNanoseconds+=elapsed;
                    }
                    if (diagnostics.usedQmc) {
                        ++statistics->qmcCalls;
                        statistics->qmcNanoseconds+=diagnostics.qmcNanoseconds;
                        statistics->maxQmcCallNanoseconds=std::max(statistics->maxQmcCallNanoseconds,diagnostics.qmcNanoseconds);
                        statistics->qmcPoints+=diagnostics.qmcPoints;
                        statistics->qmcReusedPoints+=diagnostics.qmcReusedPoints;
                        statistics->qmcPreparations+=diagnostics.qmcPreparations;
                    }
                } else probability=orthant->extendTo(samples);
                refresh=false;
            }
            const double pLower=std::max(0.0,probability.value-probability.error);
            const double pUpper=std::min(p0,probability.value+probability.error);
            double error=0.0;
            for (size_t i=1;i<nodes.size();++i) error+=gapError(nodes[i-1],nodes[i]);
            Bounds result{std::clamp((pLower-error)/p0,0.0,1.0),
                          std::clamp(pUpper/p0,0.0,1.0),int(nodes.size())};
            if (result.lower>result.upper+1e-12)
                unresolved("first_hit_gpis: inconsistent numerical probability envelope");
            if (thresholdQuery && (uniform<result.lower || uniform>=result.upper)) return result;
            if (result.upper-result.lower<=settings.probabilityTolerance) return result;

            // Resolve integration noise before asking for more spatial constraints.
            if (2.0*probability.error>0.25*settings.probabilityTolerance*p0 &&
                (error<4.0*probability.error || nodes.size()>=size_t(settings.maxNodes)) &&
                (samples<settings.maxQmcSamples ||
                 2.0*probability.error>=settings.probabilityTolerance*p0 ||
                 nodes.size()>=size_t(settings.maxNodes))) {
                if (samples>=settings.maxQmcSamples) {
                    std::ostringstream message;
                    message << "first_hit_gpis: QMC budget exhausted (nodes=" << nodes.size()
                            << ", QMC error=" << probability.error/p0 << ", missed crossings=" << error/p0
                            << ", prefix=" << t << "); raise max_qmc_samples or loosen probability_tolerance. Nodes (distance/ell,mean/sigma):";
                    for (double v:nodes) message << " (" << v/ell << "," << at(v).value/sigma << ")";
                    throw cg::NumericError(cg::NumericStatus::SamplingNotConverged,message.str());
                }
                samples=std::min(settings.maxQmcSamples,2*samples); refresh=true; continue;
            }
            if (nodes.size()>=size_t(settings.maxNodes))
                unresolved("first_hit_gpis: active node budget exhausted; raise max_nodes");
            double best=-1.0,chosen=0.0;
            for (size_t i=1;i<nodes.size();++i) {
                double gain;
                const double candidate=chooseCandidate(nodes[i-1],nodes[i],gain);
                if (gain>best) { best=gain; chosen=candidate; }
            }
            if (!(chosen>0.0 && chosen<t) || std::binary_search(nodes.begin(),nodes.end(),chosen))
                unresolved("first_hit_gpis: node refinement reached floating point resolution");
            nodes.insert(std::lower_bound(nodes.begin(),nodes.end(),chosen),chosen);
            // A changed node set can change the Genz ordering and every
            // conditional weight. Only budget increases reuse this state.
            orthant.reset();
            if (std::find(selected.begin(),selected.end(),chosen)==selected.end()) selected.push_back(chosen);
            if (statistics) {
                ++statistics->nodeInsertions;
                statistics->uniqueAddedNodes=selected.size();
            }
            refresh=true;
        }
        unresolved("first_hit_gpis: refinement budget exhausted");
    }

    bool survives(double t,double uniform)
    {
        const auto interval=query(t,uniform,true);
        if (uniform<interval.lower) return true;
        if (uniform>=interval.upper) return false;
        // Explicit finite-accuracy decision. We do not call a QMC error estimate
        // a rigorous bound, or claim an exact first-passage sampler at finite tol.
        return uniform<0.5*(interval.lower+interval.upper);
    }
};

Flight::Flight(MeanRay mean,double sigma,double ell,double length,const Settings &settings,FlightStatistics *statistics)
    : _impl(std::make_unique<Impl>(std::move(mean),sigma,ell,length,settings,statistics)) {}
Flight::~Flight()=default;
Bounds Flight::survivalBounds(double distance) { return _impl->query(distance,0.0,false); }
double Flight::sampleDistance(double uniform)
{
    ScopedTimer timer(_impl->statistics ? &_impl->statistics->flightNanoseconds : nullptr);
    if (!(uniform>0.0 && uniform<1.0))
        throw cg::NumericError(cg::NumericStatus::InvalidInput,"first_hit_gpis: uniform must be in (0,1)");
    const double length=_impl->length;
    if (length==0.0 || _impl->survives(length,uniform)) return length;
    double lo=0.0,hi=length;
    const double tolerance=std::max(_impl->settings.distanceTolerance,
                                   32.0*std::numeric_limits<double>::epsilon()*length);
    for (int iteration=0;hi-lo>tolerance;++iteration) {
        if (iteration==128) unresolved("first_hit_gpis: distance inversion did not converge");
        const double mid=0.5*(lo+hi);
        if (_impl->survives(mid,uniform)) lo=mid; else hi=mid;
    }
    return 0.5*(lo+hi);
}
}
