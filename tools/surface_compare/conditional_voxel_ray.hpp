#pragma once
#include <array>

namespace VoxelRayExperiment {
using Json=nlohmann::json;
namespace cg=Tungsten::ConditionalGpis;

// The saved path contains exactly the uniforms consumed by the original tracker.
// A new partition may consume more. Extend the tape identically for each replay.
class Tape final : public Tungsten::PathSampleGenerator {
    const std::vector<float> &prefix;
    Tungsten::UniformSampler extra{71839};
public:
    size_t consumed=0;
    explicit Tape(const std::vector<float> &values):prefix(values) {}
    void startPath(Tungsten::uint32,Tungsten::uint32) override {}
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override {}
    void loadState(Tungsten::InputStreamHandle &) override {}
    bool nextBoolean(float) override {throw std::runtime_error("Unexpected boolean");}
    int nextDiscrete(int) override {throw std::runtime_error("Unexpected discrete draw");}
    float next1D() override {const auto i=consumed++;return i<prefix.size() ? prefix[i] : extra.next1D();}
    Tungsten::Vec2f next2D() override {throw std::runtime_error("Unexpected 2D draw");}
    Tungsten::UniformSampler &uniformGenerator() override {return extra;}
};

inline void compare(const Json &spec,const std::filesystem::path &out,const cg::GPSSField &field,
                    const cg::ConditionalTrackingSettings &original) {
    const auto check=[](bool ok,const char *msg){if(!ok) throw std::runtime_error(msg);};
    const auto vec=[](const Json &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    std::ifstream in(spec.at("voxel_ray_file").get<std::string>());
    const Json ray=Json::parse(in);
    cg::FlightState birth;
    birth.birthPosition=vec(ray["birth_position"]);birth.birthGradient=vec(ray["birth_gradient"]);
    birth.birthValue=ray["birth_value"];birth.age=ray["birth_age"];birth.direction=vec(ray["direction"]);
    const auto origin=vec(ray["origin"]);const auto direction=birth.direction;
    const double length=ray.at("length"),begin=ray.value("tracking_start_distance",0.);
    const double split=ray.value("tracking_split_distance",0.);
    const bool collision=ray["domain"]=="path";
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    const auto prefix=ray["uniforms"].get<std::vector<float>>();
    std::array<cg::ConditionalTrackingSettings,4> settings={original,original,original,original};
    const std::array<std::string,4> names={"baseline","voxel_range","voxel_midpoint","voxel_bernstein5"};
    settings[1].transitionVoxelBound=cg::TransitionVoxelBound::RangeOnly;
    settings[2].transitionVoxelBound=cg::TransitionVoxelBound::WithMidpoint;
    settings[3].transitionVoxelBound=cg::TransitionVoxelBound::Bernstein5;
    settings[1].transitionMaxExpectedEvents=settings[2].transitionMaxExpectedEvents=settings[3].transitionMaxExpectedEvents=0;
    Json report={{"source",spec.at("voxel_ray_file")},{"id",ray["id"]},{"prefix_uniforms",prefix.size()},
                 {"continuation_seed",71839},{"modes",Json::object()}};
    const auto save=[&] {std::ofstream f(out/"voxel_ray.json");f<<report.dump(2);check(bool(f),"Cannot save voxel replay");};
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
    using namespace Tungsten::ConditionalGpis::RayProfile;
    for(size_t mode=0;mode<settings.size();++mode) {
        registry().enabled=true;
        Flight trace(birth,origin,direction,length,collision,ray["bounce"].get<int>());
        cg::ConditionalSegment segment(field,birth,origin,direction,length,settings[mode],nullptr,L,L);
        Tape tape(prefix);cg::Random rng(tape);cg::ConditionalTrackingWork work;double hit;
        trace.constructed();const double result=segment.track(rng,collision,hit,&work,begin,split);
        trace.finish(result,hit);registry().enabled=false;
        const auto &r=registry().thread().scratch;
        auto &row=report["modes"][names[mode]];
        row={{"result",result},{"visited",work.distance},{"draws",tape.consumed},{"candidates",work.candidates},
             {"intervals",work.intervals},{"transition",slowPhaseJson(r.phases[1])},{"leaves",Json::array()}};
        if(mode==0) check(work.intervals==ray["leaves"].size() && work.candidates==ray["events"].size() &&
                result==ray["result"] && work.distance==ray["visited"] && hit==ray["hit"] && tape.consumed==prefix.size(),
                "Baseline replay does not match saved ray");
        if(mode!=0) check(r.phases[1].splits==0 && r.phases[1].bounds==r.phases[1].intervals,"Expected one bound per visited voxel");
        for(const auto &v:r.leaves) if(v.phase==1)
            row["leaves"].push_back({{"begin",v.begin},{"end",v.end},{"visited_end",v.visitedEnd},
                                     {"majorant",v.cap*v.maximum},{"candidates",v.candidates}});
        save();
    }
    // Compare equal physical lengths even when a realization ends early by roulette.
    auto obs=birth;obs.direction=vec(ray["effective_direction"]);obs.age=0;
    const double age=ray["age_at_origin"],far=ray["far_split"];
    const auto &baseline=report["modes"]["baseline"]["leaves"];
    check(!baseline.empty() && baseline.back()["end"].get<double>()>=far-1e-10,"Reference must traverse the transition");
    auto cursor=field.mean->createRayCursor(obs.birthPosition,obs.direction);
    cg::ConditionalFlightKernel kernel(field,obs,false,0,0,original.tabulatedBounds);
    cg::ConditionalSegment ref(field,birth,origin,direction,length,original,nullptr,L,L);
    std::vector<std::pair<double,double>> spans;
    for(const auto &v:baseline) {
        const double lo=v["begin"],hi=v["end"];
        if(!spans.empty() && spans.back().second==lo)spans.back().second=hi;
        else spans.emplace_back(lo,hi);
    }
    if(!spans.empty()) {
        // Export exact inputs for an independent scalar bound on the first traversed cell.
        const auto *grid=dynamic_cast<const Tungsten::GpisSdfGrid *>(field.mean.get());
        check(grid!=nullptr,"First-cell diagnostic requires a grid");
        const double lo=age+spans.front().first,hi=cursor->nextBoundary(lo,age+spans.front().second);
        const auto xyz=[](const cg::Vector3 &v){return Json::array({v.x(),v.y(),v.z()});};
        const cg::Vector3 q=grid->worldToIndex*obs.birthPosition+grid->translation,v=grid->worldToIndex*obs.direction;
        const Eigen::Vector3i cell=(q+(lo+hi)*.5*v).array().floor().cast<int>();
        const auto jet=field.mean->evaluate(obs.birthPosition);
        auto &c=report["first_cell_inputs"];
        c={{"lo",lo},{"hi",hi},{"q",xyz(q)},{"v",xyz(v)},{"cell",{cell.x(),cell.y(),cell.z()}},
           {"scale",grid->scale},{"offset",grid->offset},{"sigma",field.kernel.sigma()},
           {"a",obs.direction.dot(field.kernel.precision()*obs.direction)},
           {"delta_f",obs.birthValue-jet.value},{"delta_k",obs.direction.dot(obs.birthGradient-jet.gradient)},
           {"corners",Json::array()},{"endpoint_checks",Json::array()}};
        for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x)
            c["corners"].push_back(grid->at(cell.x()+x-grid->data.origin[0],cell.y()+y-grid->data.origin[1],
                                           cell.z()+z-grid->data.origin[2]));
        c["midpoint_ray_bound_curvature"]=cursor->rayBounds(lo,hi,{}).maximumSecondDerivative;
        for(double t:{lo,.5*(lo+hi),std::nextafter(hi,lo),hi}) {
            const auto p=kernel.scalarStatistics(t);
            c["endpoint_checks"].push_back({{"t",t},{"mu",p.meanF},{"sd",std::sqrt(p.varianceF)},
                                            {"z",p.meanF/std::sqrt(p.varianceF)},{"g",p.meanAtZero}});
        }
    }
    report["voxels"]=Json::array();
    for(const auto &span:spans) for(double lo=span.first;lo<span.second;) {
        const double hi=std::min(span.second,cursor->nextBoundary(age+lo,age+span.second)-age);
        check(hi>lo,"Voxel failed to advance");
        const auto spatial=cursor->rayBounds(age+lo,age+hi,{});
        const double range=kernel.coupledIntervalMajorant(age+lo,age+hi,L,true,INFINITY,false,false);
        const double mid=kernel.coupledIntervalMajorant(age+lo,age+hi,L,true,INFINITY,false,true);
        const double polynomial=kernel.coupledIntervalMajorant(age+lo,age+hi,L,true,INFINITY,false,true,true);
        double zlo=0,zhi=0;
        const bool hasPolynomial=kernel.polynomialZBounds(age+lo,age+hi,zlo,zhi);
        double baseDepth=0;
        for(const auto &v:baseline) {
            const double overlap=std::min(hi,v["end"].get<double>())-std::max(lo,v["begin"].get<double>());
            if(overlap>0)baseDepth+=overlap*v["majorant"].get<double>();
        }
        double peak=0,depths[2]={},muMin=INFINITY,muMax=-INFINITY;
        for(int pass=0;pass<2;++pass) {
            const int samples=pass ? 512 : 128;
            for(int i=0;i<samples;++i) {
                const double t=lo+(hi-lo)*(i+.5)/samples;
                const double h=ref.hazard(t);
                check(h<=std::min({range,mid,polynomial})*(1+1e-9)+1e-280,"Voxel majorant violated");
                if(hasPolynomial) {
                    const auto p=kernel.scalarStatistics(age+t);
                    const double z=p.meanF/std::sqrt(p.varianceF);
                    check(z>=zlo && z<=zhi,"Polynomial z enclosure violated");
                }
                peak=std::max(peak,h);depths[pass]+=h*((hi-lo)/samples);
                if(pass) {
                    const double mu=kernel.scalarStatistics(age+t).meanF;
                    muMin=std::min(muMin,mu);muMax=std::max(muMax,mu);
                }
            }
        }
        report["voxels"].push_back({{"begin",lo},{"end",hi},{"prior_min",spatial.minimumValue},{"prior_max",spatial.maximumValue},
            {"slope_min",spatial.minimumDerivative},{"slope_max",spatial.maximumDerivative},{"sampled_mu_min",muMin},{"sampled_mu_max",muMax},
            {"range_majorant",range},{"midpoint_majorant",mid},{"baseline_depth",baseDepth},{"range_depth",range*(hi-lo)},
            {"midpoint_depth",mid*(hi-lo)},{"polynomial_majorant",polynomial},{"polynomial_depth",polynomial*(hi-lo)},
            {"has_polynomial",hasPolynomial},{"z_lower",zlo},{"z_upper",zhi},
            {"hazard_peak_sampled",peak},{"hazard_depth",depths[1]},{"hazard_depth_128",depths[0]}});
        lo=hi;
    }
#else
    using Clock=std::chrono::steady_clock;
    const int repetitions=spec.value("voxel_repetitions",63);
    for(int repeat=-3;repeat<repetitions;++repeat)for(int j=0;j<int(settings.size());++j) {
        const int mode=(j+repeat+int(settings.size()))%int(settings.size());
        Tape tape(prefix);cg::Random rng(tape);cg::ConditionalTrackingWork work;double hit;
        const auto start=Clock::now();
        cg::ConditionalSegment segment(field,birth,origin,direction,length,settings[mode],nullptr,L,L);
        const double value=segment.track(rng,collision,hit,&work,begin,split);
        const double us=std::chrono::duration<double,std::micro>(Clock::now()-start).count();
        if(repeat<0) continue;
        auto &row=report["modes"][names[mode]];
        row["elapsed_us"].push_back(us);row["candidates"]=work.candidates;row["intervals"]=work.intervals;
        row["draws"]=tape.consumed;row["result"]=value;row["visited"]=work.distance;
        if(mode==0)check(work.candidates==ray["events"].size() && value==ray["result"] && tape.consumed==prefix.size(),
                         "Release baseline replay mismatch");
    }
#endif
    save();
}
}
