#pragma once
#include <cstdint>

// Keep diagnostics out of ordinary optimized builds: even disabled runtime
// scopes can change the compiler's inlining decisions in these hot functions.
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#endif

namespace Tungsten::ConditionalGpis::Profile {

// Opt-in diagnostics. Each worker writes its own counters; only registration
// takes a lock. Nested scopes report inclusive and non-overlapping exclusive time.
enum class Domain { Path, Shadow, Phase, Connection, Count };
enum class Stage { Medium, KernelSetup, Track, BirthBound, IntervalBound, TaylorBound,
    MomentBound, ExpPolynomial, SeVariance, SdfJet, SdfDifference, SdfRayBounds, SdfBounds,
    Hazard, ScalarStatistics, EndpointSample, CollisionGradient, PhasePdf, PhaseEval,
    PhaseSample, DirectionalSample,
    BoundSdfCoarse, BoundSdfFine, BoundSdfPoint, BoundSdfDifference, BoundAlgebra,
    BoundMomentPrepare, BoundMomentApply, BoundMills, BoundFlux, BoundNormalCdf, BoundFarTail, Count };
enum class Counter { ProposedIntervals, ProcessedIntervals, Splits, CandidateEvents,
    AcceptedCollisions, RejectedCandidates, RatioEvents, ZeroHazardCandidates,
    ZeroEventIntervals, ZeroRateIntervals, ZeroBirths, PositiveBirths, SmoothBirthHalvings,
    SeparationHalvings, LocalBoundFallbacks, CheapBoundAccepted, TightenedIntervals,
    CheapBoundFallbacks, PolynomialCells, PolynomialRefinements,
    PolynomialLimitCells, GridCellBuilds, GridPointFallbacks, GridBoundFallbacks,
    GridMacroQueries, MacroAccepted, MacroRejected,
    Escapes, ZeroWeight, FarBoundSetups, FarBoundAccepted, FarBoundFallbacks,
    ShadowRouletteTests, ShadowRouletteTerminated, ShadowRouletteSurvived, Count };
enum class Histogram { CandidatesPerFlight, IntervalsPerFlight, CandidatesPerInterval, Count };
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
constexpr size_t DomainCount=size_t(Domain::Count),StageCount=size_t(Stage::Count);
constexpr size_t CounterCount=size_t(Counter::Count),HistogramCount=size_t(Histogram::Count);
// Attribution follows the actual proposal construction, not a counterfactual
// traversal. A subdivided leaf (including the right child) is EventThreshold.
enum class IntervalLimit { Step, Age, FineBoundary, MacroBoundary, Birth, SegmentEnd,
    EventThreshold, TwoConstants, Count };
enum class PartitionPhase { Initial, Proposed, Processed, Count };
constexpr size_t IntervalLimitCount=size_t(IntervalLimit::Count),AgeBandCount=4;
constexpr size_t PartitionPhaseCount=size_t(PartitionPhase::Count);
struct PartitionStats {
    uint64_t intervals=0,candidates=0,zeroCandidateIntervals=0;
    double plannedLengthEll=0,traversedLengthEll=0,expectedCandidates=0;
};
using PartitionTable=std::array<std::array<std::array<PartitionStats,IntervalLimitCount>,AgeBandCount>,PartitionPhaseCount>;
inline size_t ageBand(double ageEll) {
    return ageEll<1 ? 0 : ageEll<4 ? 1 : ageEll<16 ? 2 : 3;
}
struct Scope;
struct Timing { uint64_t calls=0,inclusiveNs=0,exclusiveNs=0,maxNs=0; };
struct Distribution {
    uint64_t count=0,sum=0,maximum=0;
    // Bucket 0 is zero, bucket n>0 covers [2^(n-1), 2^n).
    std::array<uint64_t,65> bins{};
};
struct ThreadStats {
    Scope *top=nullptr;
    size_t domain=0;
    std::array<std::array<Timing,StageCount>,DomainCount> times{};
    std::array<std::array<uint64_t,CounterCount>,DomainCount> counts{};
    std::array<std::array<Distribution,HistogramCount>,DomainCount> histograms{};
    std::array<PartitionTable,DomainCount> partitions{};
};
struct Registry {
    const bool enabled=[] {
        const char *flag=std::getenv("TUNGSTEN_GPIS_CONDITIONAL_PROFILE");
        return flag && std::string(flag)=="1";
    }();
    std::mutex mutex;
    std::vector<std::unique_ptr<ThreadStats>> threads;
    ThreadStats &thread() {
        static thread_local ThreadStats *stats=nullptr;
        if(!stats) {
            auto item=std::make_unique<ThreadStats>();stats=item.get();
            std::lock_guard<std::mutex> lock(mutex);
            threads.push_back(std::move(item));
        }
        return *stats;
    }
    ~Registry() {
        if(!enabled) return;
        ThreadStats total;
        for(const auto &thread:threads) for(size_t d=0;d<DomainCount;++d) {
            for(size_t s=0;s<StageCount;++s) {
                auto &a=total.times[d][s];const auto &b=thread->times[d][s];
                a.calls+=b.calls;a.inclusiveNs+=b.inclusiveNs;a.exclusiveNs+=b.exclusiveNs;
                a.maxNs=std::max(a.maxNs,b.maxNs);
            }
            for(size_t c=0;c<CounterCount;++c) total.counts[d][c]+=thread->counts[d][c];
            for(size_t h=0;h<HistogramCount;++h) {
                auto &a=total.histograms[d][h];const auto &b=thread->histograms[d][h];
                a.count+=b.count;a.sum+=b.sum;a.maximum=std::max(a.maximum,b.maximum);
                for(size_t i=0;i<a.bins.size();++i) a.bins[i]+=b.bins[i];
            }
            for(size_t p=0;p<PartitionPhaseCount;++p) for(size_t a=0;a<AgeBandCount;++a)
                for(size_t l=0;l<IntervalLimitCount;++l) {
                    auto &v=total.partitions[d][p][a][l];const auto &w=thread->partitions[d][p][a][l];
                    v.intervals+=w.intervals;v.candidates+=w.candidates;
                    v.zeroCandidateIntervals+=w.zeroCandidateIntervals;
                    v.plannedLengthEll+=w.plannedLengthEll;v.traversedLengthEll+=w.traversedLengthEll;
                    v.expectedCandidates+=w.expectedCandidates;
                }
        }
        const char *domains[]={"path","shadow","phase","connection"};
        const char *stages[]={"medium","kernel_setup","track","birth_bound","interval_bound","taylor_bound",
            "moment_bound","exp_polynomial","se_variance","sdf_jet","sdf_difference","sdf_ray_bounds","sdf_bounds",
            "hazard","scalar_statistics","endpoint_sample","collision_gradient","phase_pdf","phase_eval",
            "phase_sample","directional_sample",
            "bound_sdf_coarse","bound_sdf_fine","bound_sdf_point","bound_sdf_difference","bound_algebra",
            "bound_moment_prepare","bound_moment_apply","bound_mills","bound_flux","bound_normal_cdf","bound_far_tail"};
        const char *counters[]={"proposed_intervals","processed_intervals","splits","candidate_events",
            "accepted_collisions","rejected_candidates","ratio_events","zero_hazard_candidates",
            "zero_event_intervals","zero_rate_intervals","zero_births","positive_births","smooth_birth_halvings",
            "separation_halvings","local_bound_fallbacks","cheap_bound_accepted","tightened_intervals",
            "cheap_bound_fallbacks","polynomial_cells","polynomial_refinements",
            "polynomial_limit_cells","grid_cell_builds","grid_point_fallbacks","grid_bound_fallbacks",
            "grid_macro_queries","macro_accepted","macro_rejected",
            "escapes","zero_weight","far_bound_setups","far_bound_accepted","far_bound_fallbacks",
            "shadow_roulette_tests","shadow_roulette_terminated","shadow_roulette_survived"};
        const char *histograms[]={"candidates_per_flight","intervals_per_flight","candidates_per_interval"};
        const char *limits[]={"step_cap","age_cap","fine_boundary","macro_boundary","birth",
            "segment_end","event_threshold","two_constants"};
        const char *phases[]={"initial","proposed","processed"};
        const char *ages[]={"q_lt1","q_1_4","q_4_16","q_ge16"};
        std::cout<<std::setprecision(9)<<"GPIS conditional profile: {\"unit\":\"sum_worker_ms\",\"domains\":{";
        for(size_t d=0;d<DomainCount;++d) {
            if(d) std::cout<<',';
            std::cout<<'\"'<<domains[d]<<"\":{\"stages\":{";
            for(size_t s=0;s<StageCount;++s) {
                if(s) std::cout<<',';
                const auto &t=total.times[d][s];
                std::cout<<'\"'<<stages[s]<<"\":{\"calls\":"<<t.calls<<",\"inclusive_ms\":"<<t.inclusiveNs*1e-6
                    <<",\"exclusive_ms\":"<<t.exclusiveNs*1e-6<<",\"max_ms\":"<<t.maxNs*1e-6<<'}';
            }
            std::cout<<"},\"counts\":{";
            for(size_t c=0;c<CounterCount;++c) {
                if(c) std::cout<<',';
                std::cout<<'\"'<<counters[c]<<"\":"<<total.counts[d][c];
            }
            std::cout<<"},\"histograms\":{";
            for(size_t h=0;h<HistogramCount;++h) {
                if(h) std::cout<<',';
                const auto &v=total.histograms[d][h];
                std::cout<<'\"'<<histograms[h]<<"\":{\"count\":"<<v.count<<",\"sum\":"<<v.sum
                    <<",\"max\":"<<v.maximum<<",\"log2_buckets\":{";
                bool first=true;
                for(size_t i=0;i<v.bins.size();++i) if(v.bins[i]) {
                    if(!first) std::cout<<',';
                    first=false;std::cout<<'\"'<<i<<"\":"<<v.bins[i];
                }
                std::cout<<"}}";
            }
            std::cout<<"},\"interval_partition\":{";
            for(size_t p=0;p<PartitionPhaseCount;++p) {
                if(p) std::cout<<',';
                std::cout<<'\"'<<phases[p]<<"\":{";
                for(size_t a=0;a<AgeBandCount;++a) {
                    if(a) std::cout<<',';
                    std::cout<<'\"'<<ages[a]<<"\":{";
                    bool first=true;
                    for(size_t l=0;l<IntervalLimitCount;++l) {
                        const auto &v=total.partitions[d][p][a][l];
                        if(!v.intervals) continue;
                        if(!first) std::cout<<',';
                        first=false;
                        std::cout<<'\"'<<limits[l]<<"\":{\"intervals\":"<<v.intervals
                            <<",\"planned_length_ell\":"<<v.plannedLengthEll
                            <<",\"traversed_length_ell\":"<<v.traversedLengthEll
                            <<",\"expected_candidates\":"<<v.expectedCandidates
                            <<",\"candidates\":"<<v.candidates
                            <<",\"zero_candidate_intervals\":"<<v.zeroCandidateIntervals<<'}';
                    }
                    std::cout<<'}';
                }
                std::cout<<'}';
            }
            std::cout<<"}}";
        }
        std::cout<<"}}\n";
    }
};
inline Registry &registry() { static Registry value;return value; }
struct Scope {
    using Clock=std::chrono::steady_clock;
    ThreadStats *stats=nullptr;
    Scope *parent=nullptr;
    Stage stage;
    size_t domain=0;
    uint64_t childrenNs=0;
    Clock::time_point start;
    explicit Scope(Stage stage,Domain root=Domain::Count):stage(stage) {
        if(!registry().enabled) return;
        auto &thread=registry().thread();
        if(!thread.top && root==Domain::Count) return;
        stats=&thread;parent=stats->top;
        domain=parent ? stats->domain : size_t(root);
        stats->domain=domain;stats->top=this;
        start=Clock::now();
    }
    Scope(const Scope &)=delete;
    Scope &operator=(const Scope &)=delete;
    ~Scope() {
        if(!stats) return;
        const auto ns=uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now()-start).count());
        auto &t=stats->times[domain][size_t(stage)];
        ++t.calls;t.inclusiveNs+=ns;t.exclusiveNs+=ns-childrenNs;t.maxNs=std::max(t.maxNs,ns);
        if(parent) parent->childrenNs+=ns;
        stats->top=parent;
    }
};
inline void count(Counter counter) {
    if(!registry().enabled) return;
    auto &thread=registry().thread();
    if(thread.top) ++thread.counts[thread.domain][size_t(counter)];
}
inline void partition(PartitionPhase phase,IntervalLimit limit,double ageEll,double lengthEll) {
    if(!registry().enabled) return;
    auto &thread=registry().thread();
    if(!thread.top) return;
    auto &v=thread.partitions[thread.domain][size_t(phase)][ageBand(ageEll)][size_t(limit)];
    ++v.intervals;v.plannedLengthEll+=lengthEll;
}
// Count only leaves actually reached by tracking. Keep their planned length
// separate from actual travel: a true collision may stop inside a leaf.
struct ProcessedInterval {
    PartitionStats *stats=nullptr;
    double begin,length,inverseEll;
    const double &traversed;
    const uint64_t &candidates;
    ProcessedInterval(IntervalLimit limit,double ageEll,double lo,double hi,double maximum,
                      double ell,const double &travel,const uint64_t &events)
        :begin(lo),length(hi-lo),inverseEll(1/ell),traversed(travel),candidates(events) {
        if(!registry().enabled) return;
        auto &thread=registry().thread();
        if(!thread.top) return;
        stats=&thread.partitions[thread.domain][size_t(PartitionPhase::Processed)][ageBand(ageEll)][size_t(limit)];
        ++stats->intervals;stats->plannedLengthEll+=length*inverseEll;
        stats->expectedCandidates+=maximum*length;
    }
    ProcessedInterval(const ProcessedInterval &)=delete;
    ProcessedInterval &operator=(const ProcessedInterval &)=delete;
    ~ProcessedInterval() {
        if(!stats) return;
        stats->candidates+=candidates;
        stats->zeroCandidateIntervals+=candidates==0;
        stats->traversedLengthEll+=std::clamp(traversed-begin,0.0,length)*inverseEll;
    }
};
inline void observe(Histogram histogram,uint64_t value) {
    if(!registry().enabled) return;
    auto &thread=registry().thread();
    if(!thread.top) return;
    auto &h=thread.histograms[thread.domain][size_t(histogram)];
    ++h.count;h.sum+=value;h.maximum=std::max(h.maximum,value);
    size_t bucket=0;
    for(auto bits=value;bits;bits>>=1) ++bucket;
    ++h.bins[bucket];
}
#else
struct Scope {
    explicit Scope(Stage,Domain=Domain::Count) {}
};
inline void count(Counter) {}
inline void observe(Histogram,uint64_t) {}
#endif
struct FlightWork {
    uint64_t candidates=0,intervals=0;
    ~FlightWork() {
        observe(Histogram::CandidatesPerFlight,candidates);
        observe(Histogram::IntervalsPerFlight,intervals);
    }
};
struct IntervalWork {
    uint64_t candidates=0;
    ~IntervalWork() {
        observe(Histogram::CandidatesPerInterval,candidates);
        if(candidates==0) count(Counter::ZeroEventIntervals);
    }
};
}
