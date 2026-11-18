#pragma once
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
inline void exportOptimizationFixedRays(const nlohmann::json &files,const std::filesystem::path &output,
    const Tungsten::ConditionalGpis::GPSSField &field,const Tungsten::ConditionalGpis::ConditionalTrackingSettings &chosen) {
    namespace cg=Tungsten::ConditionalGpis;namespace rp=cg::RayProfile;using J=nlohmann::json;
    const auto vec=[](const J &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    J rows=J::array();
    for(const auto &file:files) {
        std::ifstream source(file.get<std::string>());const J input=J::parse(source);
        cg::FlightState birth;birth.birthPosition=vec(input["birth_position"]);birth.birthGradient=vec(input["birth_gradient"]);
        birth.birthValue=input["birth_value"];birth.age=input["birth_age"];
        const auto origin=vec(input["origin"]),direction=vec(input["direction"]);birth.direction=direction;
        const double length=input["length"];const bool collision=input["domain"]=="path";
        const auto tape=input["uniforms"].get<std::vector<float>>();
        for(bool optimized:{false,true}) {
            auto settings=chosen;
            if(!optimized) {
                settings=cg::ConditionalTrackingSettings{};settings.fixedStepLimit=false;
                settings.localFarStartScale=4;settings.transitionIntegralBounds=true;
            }
            cg::Profile::Scope root(cg::Profile::Stage::Medium,collision ? cg::Profile::Domain::Path:cg::Profile::Domain::Shadow);
            rp::registry().enabled=true;
            rp::Flight recording(birth,origin,direction,length,collision,input["bounce"].get<int>());
            cg::ConditionalSegment segment(field,birth,origin,direction,length,settings,nullptr,L,L);
            recording.constructed();TransitionBudgetSampler sampler(tape);cg::Random rng(sampler);
            cg::ConditionalTrackingWork work;double hit;const double result=segment.track(rng,collision,hit,&work);
            recording.finish(result,hit);rp::registry().enabled=false;const auto &r=rp::registry().thread().scratch;
            if(!optimized && (result!=input["result"].get<double>() || hit!=input["hit"].get<double>() || sampler.consumed!=tape.size()))
                throw std::runtime_error("Baseline fixed-ray replay changed");
            J row={{"id",input["id"]},{"optimized",optimized},{"domain",input["domain"]},
                {"elapsed_us",r.setupUs+r.trackUs},{"visited",work.distance},{"length",length},{"result",result},
                {"candidates",work.candidates},{"intervals",work.intervals},{"birth_split",r.split-r.age},{"far_split",r.far},
                {"phases",{{"birth",slowPhaseJson(r.phases[0])},{"transition",slowPhaseJson(r.phases[1])},{"far",slowPhaseJson(r.phases[2])}}},
                {"events",J::array()},{"curve",J::array()},{"leaves",J::array()}};
            for(const auto &e:r.events)row["events"].push_back({{"t",e.t},{"hazard",e.hazard},{"majorant",e.majorant},{"kind",e.kind},{"phase",e.phase}});
            auto observation=birth;observation.direction=r.effectiveDirection;observation.age=0;
            cg::ConditionalFlightKernel reference(field,observation);double maxRatio=0;size_t points=0;
            for(const auto &leaf:r.leaves) {
                row["leaves"].push_back({{"begin",leaf.begin},{"end",leaf.end},{"visited_end",leaf.visitedEnd},
                    {"maximum",leaf.maximum},{"integral",leaf.integral},{"phase",leaf.phase},{"candidates",leaf.candidates}});
                if(!(leaf.visitedEnd>leaf.begin))continue;
                const int n=std::max(16,int(std::ceil((leaf.visitedEnd-leaf.begin)/r.ell*256)));
                for(int j=0;j<n;++j) {
                    const double t=leaf.begin+(leaf.visitedEnd-leaf.begin)*(j+.5)/n;
                    const double h=reference.evaluate(r.age+t).hazard.value;
                    const double m=leaf.cap*(leaf.integral ? leaf.curve.value(r.age+t):leaf.maximum);
                    if(h>m*(1+1e-9)+1e-280)throw std::runtime_error("Optimized fixed ray exceeds its majorant");
                    if(m>0)maxRatio=std::max(maxRatio,h/m);++points;
                    row["curve"].push_back({{"t",t},{"hazard",h},{"majorant",m},{"phase",leaf.phase}});
                }
            }
            row["coverage_samples"]=points;row["maximum_hazard_bound_ratio"]=maxRatio;
            rows.push_back(row);
        }
    }
    std::ofstream saved(output/"fixed_rays.json");saved<<rows.dump(2);
    if(!saved)throw std::runtime_error("Cannot save fixed slow-ray profiles");
}
#endif
