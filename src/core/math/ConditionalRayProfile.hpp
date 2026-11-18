#pragma once
// Detailed traces are compiled only into the dedicated diagnostic executable.
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "sampling/PathSampleGenerator.hpp"
#include <algorithm>
#include <array>
#include <chrono>
#include <memory>
#include <mutex>
#include <vector>

namespace Tungsten::ConditionalGpis::RayProfile {
using Clock=std::chrono::steady_clock;
inline double elapsed(Clock::time_point start) {
    return std::chrono::duration<double,std::micro>(Clock::now()-start).count();
}
struct Phase {
    double totalUs=0,boundUs=0,hazardUs=0,clockUs=0;
    uint64_t bounds=0,intervals=0,empty=0,splits=0,candidates=0,nulls=0,accepted=0,ratio=0;
    Phase &operator+=(const Phase &b) {
        totalUs+=b.totalUs;boundUs+=b.boundUs;hazardUs+=b.hazardUs;clockUs+=b.clockUs;
        bounds+=b.bounds;intervals+=b.intervals;empty+=b.empty;splits+=b.splits;
        candidates+=b.candidates;nulls+=b.nulls;accepted+=b.accepted;ratio+=b.ratio;return *this;
    }
};
struct Bound { double begin,end,us;int phase;bool setup; };
struct Leaf {
    double begin,end,visitedEnd,maximum,cap;int phase;bool integral;
    ConditionalIntegralMajorant curve;uint64_t candidates=0;
};
struct Event { double t,hazard,majorant,weight;int phase,kind; }; // 0 null, 1 accepted, 2 ratio
struct Record {
    FlightState birth;Point3 origin;Vector3 direction,effectiveDirection;
    double length=0,trackingStart=0,age=0,ell=0,split=0,far=0,visited=0,result=0,hit=0;
    double setupUs=0,trackUs=0;int bounce=0;bool collision=false;
    std::array<Phase,3> phases{};
    std::vector<Bound> bounds;std::vector<Leaf> leaves;std::vector<Event> events;
    std::vector<float> uniforms;
    int phase(double t) const { return t>=far ? 2 : age+t>=split ? 1 : 0; }
    double score(int metric) const {
        if(metric==0)return setupUs+trackUs;
        double sum=0;for(const auto &p:phases) sum+=metric==1 ? p.bounds : collision ? p.nulls : p.ratio;
        return sum;
    }
};
constexpr size_t PoolSize=8;
struct Thread {
    Record scratch;
    std::array<std::array<std::vector<std::shared_ptr<Record>>,3>,2> best;
    std::array<std::array<Phase,3>,2> phases{};
    std::array<uint64_t,2> flights{};
    std::array<double,2> setupUs{},trackUs{};
};
struct Registry {
    bool enabled=false;
    std::mutex mutex;
    std::vector<std::unique_ptr<Thread>> threads;
    Thread &thread() {
        static thread_local Thread *local=nullptr;
        if(!local) { auto item=std::make_unique<Thread>();local=item.get();
            std::lock_guard<std::mutex> lock(mutex);threads.push_back(std::move(item)); }
        return *local;
    }
};
inline Registry &registry() { static Registry r;return r; }
inline Record *&active() { static thread_local Record *r=nullptr;return r; }
struct Flight {
    Thread *thread=nullptr;Record *r=nullptr;Clock::time_point start,tracking;
    Flight(const FlightState &birth,const Point3 &origin,const Vector3 &direction,double length,bool collision,int bounce) {
        if(!registry().enabled)return;
        thread=&registry().thread();r=&thread->scratch;
        // Retain capacities between flights; sampling and the renderer RNG are unchanged.
        r->phases={};r->bounds.clear();r->leaves.clear();r->events.clear();r->uniforms.clear();
        r->birth=birth;r->origin=origin;r->direction=direction;r->length=length;
        r->collision=collision;r->bounce=bounce;r->visited=0;r->setupUs=r->trackUs=0;
        active()=r;start=Clock::now();
    }
    void constructed() { if(r) { r->setupUs=elapsed(start);r->phases[0].totalUs+=r->setupUs;tracking=Clock::now(); } }
    void finish(double result,double hit) {
        if(!r)return;
        r->trackUs=elapsed(tracking);r->result=result;r->hit=hit;active()=nullptr;
        const int d=r->collision ? 0 : 1;
        ++thread->flights[d];thread->setupUs[d]+=r->setupUs;thread->trackUs[d]+=r->trackUs;
        for(int p=0;p<3;++p)thread->phases[d][p]+=r->phases[p];
        std::shared_ptr<Record> saved;
        for(int metric=0;metric<3;++metric) {
            auto &pool=thread->best[d][metric];
            auto worst=std::min_element(pool.begin(),pool.end(),[&](const auto &a,const auto &b){return a->score(metric)<b->score(metric);});
            if(pool.size()<PoolSize || r->score(metric)>(*worst)->score(metric)) {
                if(!saved)saved=std::make_shared<Record>(*r);
                if(pool.size()<PoolSize)pool.push_back(saved);else *worst=saved;
            }
        }
        r=nullptr;
    }
    ~Flight() { if(r)active()=nullptr; }
};
struct Sampler final : PathSampleGenerator {
    PathSampleGenerator &source;
    explicit Sampler(PathSampleGenerator &source):source(source) {}
    void startPath(uint32 a,uint32 b) override {source.startPath(a,b);}
    void advancePath() override {source.advancePath();}
    void saveState(OutputStreamHandle &s) override {source.saveState(s);}
    void loadState(InputStreamHandle &s) override {source.loadState(s);}
    bool nextBoolean(float p) override {return source.nextBoolean(p);}
    int nextDiscrete(int n) override {return source.nextDiscrete(n);}
    float next1D() override {const float u=source.next1D();if(active())active()->uniforms.push_back(u);return u;}
    Vec2f next2D() override {return source.next2D();}
    UniformSampler &uniformGenerator() override {return source.uniformGenerator();}
};
struct Timer {
    Record *r=active();int phase,kind;Clock::time_point start=Clock::now();
    Timer(double t,int kind):phase(r ? r->phase(t) : 0),kind(kind) {}
    ~Timer() {if(r) {const double us=elapsed(start);auto &p=r->phases[phase];
        if(kind==0)p.totalUs+=us;else if(kind==1)p.hazardUs+=us;else p.clockUs+=us;}}
};
struct BoundScope {
    Record *r=active();double begin,end;bool setup;int phase;Clock::time_point start=Clock::now();
    BoundScope(double a,double b,bool setup=false):begin(a),end(b),setup(setup),phase(setup ? 0 : r ? r->phase(a) : 0) {}
    ~BoundScope() {if(r) {const double us=elapsed(start);r->bounds.push_back({begin,end,us,phase,setup});
        r->phases[phase].boundUs+=us;++r->phases[phase].bounds;}}
};
struct IntervalScope {
    Record *r=active();size_t index=0,events=0;const double &traversed;
    IntervalScope(double lo,double hi,double maximum,double cap,bool integral,const ConditionalIntegralMajorant &curve,const double &travel):traversed(travel) {
        if(r) {index=r->leaves.size();events=r->events.size();r->leaves.push_back({lo,hi,lo,maximum,cap,r->phase(lo),integral,curve});}
    }
    ~IntervalScope() {if(r) {auto &v=r->leaves[index];v.visitedEnd=std::max(v.begin,std::min(v.end,traversed));
        v.candidates=r->events.size()-events;auto &p=r->phases[v.phase];++p.intervals;p.empty+=v.candidates==0;r->visited=traversed;}}
};
inline void event(double t,double h,double m,double weight,int kind) {
    if(auto *r=active()) {const int phase=r->phase(t);r->events.push_back({t,h,m,weight,phase,kind});
        auto &p=r->phases[phase];++p.candidates;if(kind==0)++p.nulls;else if(kind==1)++p.accepted;else ++p.ratio;}
}
}
#endif
