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
    PhaseSample, DirectionalSample, Count };
enum class Counter { ProposedIntervals, ProcessedIntervals, Splits, CandidateEvents,
    AcceptedCollisions, RejectedCandidates, RatioEvents, ZeroHazardCandidates,
    ZeroEventIntervals, ZeroRateIntervals, ZeroBirths, PositiveBirths, SmoothBirthHalvings,
    SeparationHalvings, LocalBoundFallbacks, CheapBoundAccepted, TightenedIntervals,
    CheapBoundFallbacks, PolynomialCells, PolynomialRefinements,
    PolynomialLimitCells, GridCellBuilds, GridPointFallbacks, GridBoundFallbacks,
    GridMacroQueries, MacroAccepted, MacroRejected,
    Escapes, ZeroWeight, Count };
enum class Histogram { CandidatesPerFlight, IntervalsPerFlight, CandidatesPerInterval, Count };
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
constexpr size_t DomainCount=size_t(Domain::Count),StageCount=size_t(Stage::Count);
constexpr size_t CounterCount=size_t(Counter::Count),HistogramCount=size_t(Histogram::Count);
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
        }
        const char *domains[]={"path","shadow","phase","connection"};
        const char *stages[]={"medium","kernel_setup","track","birth_bound","interval_bound","taylor_bound",
            "moment_bound","exp_polynomial","se_variance","sdf_jet","sdf_difference","sdf_ray_bounds","sdf_bounds",
            "hazard","scalar_statistics","endpoint_sample","collision_gradient","phase_pdf","phase_eval",
            "phase_sample","directional_sample"};
        const char *counters[]={"proposed_intervals","processed_intervals","splits","candidate_events",
            "accepted_collisions","rejected_candidates","ratio_events","zero_hazard_candidates",
            "zero_event_intervals","zero_rate_intervals","zero_births","positive_births","smooth_birth_halvings",
            "separation_halvings","local_bound_fallbacks","cheap_bound_accepted","tightened_intervals",
            "cheap_bound_fallbacks","polynomial_cells","polynomial_refinements",
            "polynomial_limit_cells","grid_cell_builds","grid_point_fallbacks","grid_bound_fallbacks",
            "grid_macro_queries","macro_accepted","macro_rejected",
            "escapes","zero_weight"};
        const char *histograms[]={"candidates_per_flight","intervals_per_flight","candidates_per_interval"};
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
