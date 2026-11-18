#pragma once
#include "math/ConditionalRayProfile.hpp"
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
#include <set>
#include <map>
#include <numeric>

inline nlohmann::json slowPhaseJson(const Tungsten::ConditionalGpis::RayProfile::Phase &p) {
    return {{"total_us",p.totalUs},{"bound_us",p.boundUs},{"hazard_us",p.hazardUs},{"clock_us",p.clockUs},
        {"bound_calls",p.bounds},{"intervals",p.intervals},{"empty_intervals",p.empty},{"splits",p.splits},
        {"candidates",p.candidates},{"nulls",p.nulls},{"accepted",p.accepted},{"ratio_events",p.ratio}};
}
class SlowReplaySampler final : public Tungsten::PathSampleGenerator {
    Tungsten::UniformSampler unused;
public:
    const std::vector<float> &values;size_t consumed=0;
    explicit SlowReplaySampler(const std::vector<float> &values):values(values) {}
    void startPath(Tungsten::uint32,Tungsten::uint32) override {}
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override {}
    void loadState(Tungsten::InputStreamHandle &) override {}
    bool nextBoolean(float) override {throw std::runtime_error("Unexpected boolean draw");}
    int nextDiscrete(int) override {throw std::runtime_error("Unexpected discrete draw");}
    float next1D() override {if(consumed>=values.size())throw std::runtime_error("Replay exhausted RNG tape");return values[consumed++];}
    Tungsten::Vec2f next2D() override {throw std::runtime_error("Unexpected 2D draw");}
    Tungsten::UniformSampler &uniformGenerator() override {return unused;}
};

inline void exportSlowRays(const std::filesystem::path &output,const Tungsten::ConditionalGpis::GPSSField &field,
    const Tungsten::ConditionalGpis::ConditionalTrackingSettings &settings,const nlohmann::json &profile) {
    namespace cg=Tungsten::ConditionalGpis;namespace rp=cg::RayProfile;using J=nlohmann::json;
    using Ptr=std::shared_ptr<rp::Record>;
    const auto check=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    const auto vec=[](const auto &v){return J::array({v.x(),v.y(),v.z()});};
    const char *names[]={"birth","transition","far"},*domains[]={"path","shadow"};
    const auto phaseJson=[&](const auto &phases){J j;for(int p=0;p<3;++p)j[names[p]]=slowPhaseJson(phases[p]);return j;};
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    J report={{"schema_version",1},{"phase_definition","birth: age+t < certified split; transition: split <= age+t < 4 ell; far: age+t >= 4 ell"},
        {"timing_unit","microseconds; elapsed worker time, diagnostic overhead included"},{"replay_repetitions",7},
        {"aggregate",J::object()},{"candidate_pool",J::array()},{"rays",J::array()}};
    size_t rayId=0;
    for(int d=0;d<2;++d) {
        std::array<rp::Phase,3> totals{};uint64_t flights=0;double setup=0,track=0;
        std::set<Ptr> pool;
        for(int metric=0;metric<3;++metric) {
            std::vector<Ptr> ranked;
            for(const auto &thread:rp::registry().threads)
                for(auto &p:thread->best[d][metric])ranked.push_back(p);
            std::sort(ranked.begin(),ranked.end(),[&](const auto &a,const auto &b){return a->score(metric)>b->score(metric);});
            for(size_t i=0;i<std::min(rp::PoolSize,ranked.size());++i)pool.insert(ranked[i]);
        }
        for(const auto &thread:rp::registry().threads) {
            flights+=thread->flights[d];setup+=thread->setupUs[d];track+=thread->trackUs[d];
            for(int p=0;p<3;++p)totals[p]+=thread->phases[d][p];
        }
        const auto &counts=profile.at("domains").at(domains[d]).at("counts");
        uint64_t candidates=0,intervals=0,nulls=0,accepted=0,ratio=0,splits=0;
        for(const auto &p:totals) {candidates+=p.candidates;intervals+=p.intervals;nulls+=p.nulls;accepted+=p.accepted;ratio+=p.ratio;splits+=p.splits;}
        check(candidates==counts.at("candidate_events").get<uint64_t>(),"Candidate aggregate mismatch");
        check(intervals==counts.at("processed_intervals").get<uint64_t>(),"Interval aggregate mismatch");
        check(nulls==counts.at("rejected_candidates").get<uint64_t>(),"Null aggregate mismatch");
        check(accepted==counts.at("accepted_collisions").get<uint64_t>(),"Accepted aggregate mismatch");
        check(ratio==counts.at("ratio_events").get<uint64_t>(),"Ratio aggregate mismatch");
        check(splits==counts.at("splits").get<uint64_t>(),"Split aggregate mismatch");
        report["aggregate"][domains[d]]={{"flights",flights},{"setup_us",setup},{"track_us",track},{"phases",phaseJson(totals)}};
        struct Replay {Ptr ray;std::vector<double> times;double median;};
        std::vector<Replay> replays;
        for(const auto &r:pool) {
            Replay replay;replay.ray=r;
            for(int repeat=0;repeat<7;++repeat) {
                SlowReplaySampler sampler(r->uniforms);cg::Random rng(sampler);cg::ConditionalTrackingWork work;double hit;
                cg::Profile::Scope root(cg::Profile::Stage::Medium,d==0 ? cg::Profile::Domain::Path : cg::Profile::Domain::Shadow);
                const auto start=rp::Clock::now();
                cg::ConditionalSegment segment(field,r->birth,r->origin,r->direction,r->length,settings,nullptr,L,L);
                const double result=segment.track(rng,r->collision,hit,&work,r->trackingStart);
                replay.times.push_back(rp::elapsed(start));
                check(hit==r->hit && result==r->result && work.distance==r->visited,"Replay endpoint/weight mismatch");
                check(work.candidates==r->events.size() && work.intervals==r->leaves.size(),"Replay work mismatch");
                check(sampler.consumed==r->uniforms.size(),"Replay RNG count mismatch");
            }
            auto sorted=replay.times;std::sort(sorted.begin(),sorted.end());replay.median=sorted[sorted.size()/2];
            replays.push_back(replay);
        }
        std::map<size_t,std::vector<std::string>> selected;
        for(int metric=0;metric<3;++metric) {
            std::vector<size_t> order(replays.size());std::iota(order.begin(),order.end(),0);
            std::sort(order.begin(),order.end(),[&](size_t a,size_t b){return metric==0 ? replays[a].median>replays[b].median : replays[a].ray->score(metric)>replays[b].ray->score(metric);});
            const int number=metric==0 ? 2 : 1;
            for(int rank=0;rank<number && rank<int(order.size());++rank)
                selected[order[rank]].push_back(metric==0 ? "replay_time_"+std::to_string(rank+1) : metric==1 ? "most_bound_calls" : d==0 ? "most_null_collisions" : "most_ratio_events");
        }
        for(size_t index=0;index<replays.size();++index) {
            const auto &replay=replays[index];const auto &r=*replay.ray;
            J candidate={{"domain",domains[d]},{"render_us",r.score(0)},{"replay_median_us",replay.median},
                {"bound_calls",r.score(1)},{"null_or_ratio_events",r.score(2)},{"selected",selected.count(index)!=0}};
            report["candidate_pool"].push_back(candidate);
            if(!selected.count(index))continue;
            const std::string id=std::string(d==0 ? "P" : "S")+std::to_string(++rayId);
            J ray={{"id",id},{"domain",domains[d]},{"selection",selected[index]},{"bounce",r.bounce},
                {"origin",vec(r.origin)},{"direction",vec(r.direction)},{"effective_direction",vec(r.effectiveDirection)},
                {"birth_position",vec(r.birth.birthPosition)},{"birth_value",r.birth.birthValue},{"birth_gradient",vec(r.birth.birthGradient)},
                {"birth_age",r.birth.age},{"age_at_origin",r.age},{"length",r.length},{"tracking_start_distance",r.trackingStart},{"visited",r.visited},{"result",r.result},{"hit",r.hit},
                {"ell",r.ell},{"sigma",field.kernel.sigma()},{"birth_split",std::max(0.0,r.split-r.age)},{"far_split",r.far},
                {"setup_us",r.setupUs},{"track_us",r.trackUs},{"replay_us",replay.times},{"replay_median_us",replay.median},
                {"phases",phaseJson(r.phases)},{"uniforms",r.uniforms},{"leaves",J::array()},{"bound_queries",J::array()},
                {"events",J::array()},{"curve",J::array()}};
            for(const auto &b:r.bounds)ray["bound_queries"].push_back({{"begin",b.begin},{"end",b.end},{"us",b.us},{"phase",names[b.phase]},{"setup",b.setup}});
            for(const auto &v:r.leaves)ray["leaves"].push_back({{"begin",v.begin},{"end",v.end},{"visited_end",v.visitedEnd},
                {"maximum",v.maximum},{"cap",v.cap},{"phase",names[v.phase]},{"integral",v.integral},
                {"C0",v.curve.constant},{"C1",v.curve.regressionCoefficient},{"inverse_length2",v.curve.inverseLength2},
                {"gaussian_linear",v.curve.gaussianLinear},{"curve_origin",v.curve.origin},{"curve_z0",v.curve.z0},
                {"curve_beta",v.curve.beta},{"candidates",v.candidates}});
            double maxRatio=0;uint64_t covered=0;
            for(const auto &e:r.events) {
                check(e.hazard<=e.majorant*(1+1e-10),"Captured event violates majorant");
                ray["events"].push_back({{"t",e.t},{"hazard",e.hazard},{"majorant",e.majorant},{"weight",e.weight},
                    {"phase",names[e.phase]},{"kind",e.kind==0 ? "null" : e.kind==1 ? "accepted" : "ratio"}});
            }
            // Use a fresh, monotonically sampled cursor, including interiors on
            // both sides of every boundary. Do not extend beyond actual travel.
            auto birth=r.birth;birth.direction=r.effectiveDirection;birth.age=0;
            cg::ConditionalFlightKernel kernel(field,birth);
            for(const auto &v:r.leaves)if(v.visitedEnd>v.begin) {
                const int n=std::max(8,int(std::ceil((v.visitedEnd-v.begin)/r.ell*128)));
                for(int i=0;i<n;++i) {
                    const double t=v.begin+(v.visitedEnd-v.begin)*(i+0.5)/n;
                    const double majorant=v.cap*(v.integral ? v.curve.value(r.age+t) : v.maximum);
                    const auto statistics=kernel.scalarStatistics(r.age+t);
                    const double h=kernel.evaluate(r.age+t).hazard.value;
                    check(h<=majorant*(1+1e-9)+1e-280,"Profile curve violates recorded leaf majorant");
                    if(majorant>0)maxRatio=std::max(maxRatio,h/majorant);++covered;
                    const double mean=kernel.scalarMeanValue(r.age+t);
                    ray["curve"].push_back({{"t",t},{"mean",mean},{"conditional_mean",statistics.meanF},
                        {"hazard",h},{"majorant",majorant},{"phase",names[v.phase]}});
                }
            }
            ray["validation"]={{"replay_exact",true},{"coverage_samples",covered},{"maximum_hazard_bound_ratio",maxRatio}};
            report["rays"].push_back(ray);
            std::ofstream detail(output/(id+".json"));detail<<ray.dump(2);check(bool(detail),"Cannot write ray trace");
            std::cout<<"Exported "<<id<<", "<<domains[d]<<", median "<<replay.median<<" us, events "<<r.events.size()<<std::endl;
        }
    }
    std::ofstream saved(output/"slow_rays.json");saved<<report.dump(2);check(bool(saved),"Cannot save ray report");
}
#endif
