#pragma once
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
// Diagnostic budgets only: an interrupted trial is explicitly incomplete and
// is never used as a transmittance estimate. Production tracking has no budget.
class TransitionBudgetSampler final : public Tungsten::PathSampleGenerator {
    Tungsten::UniformSampler fallback{71839};
    const std::vector<float> &tape;
    Tungsten::ConditionalGpis::RayProfile::Clock::time_point start;
public:
    size_t consumed=0;
    struct Budget { const char *reason; };
    explicit TransitionBudgetSampler(const std::vector<float> &values):tape(values),start(Tungsten::ConditionalGpis::RayProfile::Clock::now()) {}
    void startPath(Tungsten::uint32,Tungsten::uint32) override {}
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override {}
    void loadState(Tungsten::InputStreamHandle &) override {}
    bool nextBoolean(float) override {throw std::runtime_error("Unexpected boolean draw");}
    int nextDiscrete(int) override {throw std::runtime_error("Unexpected discrete draw");}
    float next1D() override {
        namespace rp=Tungsten::ConditionalGpis::RayProfile;
        if(rp::active()->events.size()>=200000)throw Budget{"candidate_budget_200000"};
        if(consumed%1024==0 && rp::elapsed(start)>3000000)throw Budget{"time_budget_3_seconds"};
        const size_t i=consumed++;
        return i<tape.size() ? tape[i] : fallback.next1D();
    }
    Tungsten::Vec2f next2D() override {throw std::runtime_error("Unexpected 2D draw");}
    Tungsten::UniformSampler &uniformGenerator() override {return fallback;}
};

inline void compareTransitionNoSplitRays(const nlohmann::json &files,const std::filesystem::path &output,
    const Tungsten::ConditionalGpis::GPSSField &field,const Tungsten::ConditionalGpis::ConditionalTrackingSettings &original) {
    namespace cg=Tungsten::ConditionalGpis;namespace rp=cg::RayProfile;using J=nlohmann::json;
    const auto vec=[](const J &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    const auto check=[](bool ok,const char *s){if(!ok)throw std::runtime_error(s);};
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    J report={{"candidate_limit",200000},{"time_limit_seconds",3},{"trials",J::array()}};
    for(const auto &file:files) {
        std::ifstream source(file.get<std::string>());const J input=J::parse(source);
        cg::FlightState birth;birth.birthPosition=vec(input["birth_position"]);birth.birthGradient=vec(input["birth_gradient"]);
        birth.birthValue=input["birth_value"];birth.age=input["birth_age"];
        const auto origin=vec(input["origin"]),direction=vec(input["direction"]);birth.direction=direction;
        const double length=input["length"];const bool collision=input["domain"]=="path";
        const std::vector<float> tape=input["uniforms"].get<std::vector<float>>();
        for(double threshold:{4.,0.}) {
            auto settings=original;settings.transitionIntegralBounds=true;settings.transitionPointBounds=false;
            settings.transitionMaxExpectedEvents=threshold;settings.validate();
            cg::Profile::Scope root(cg::Profile::Stage::Medium,collision ? cg::Profile::Domain::Path : cg::Profile::Domain::Shadow);
            rp::registry().enabled=true;
            rp::Flight recording(birth,origin,direction,length,collision,input["bounce"].get<int>());
            cg::ConditionalSegment segment(field,birth,origin,direction,length,settings,nullptr,L,L);
            recording.constructed();TransitionBudgetSampler sampler(tape);cg::Random random(sampler);
            cg::ConditionalTrackingWork work;double hit=length,result=0;std::string status="complete",error;
            try {result=segment.track(random,collision,hit,&work);}
            catch(const TransitionBudgetSampler::Budget &budget) {status=budget.reason;}
            catch(const cg::NumericError &problem) {status="numeric_error";error=problem.what();}
            recording.finish(result,hit);rp::registry().enabled=false;
            const auto &r=rp::registry().thread().scratch;
            if(threshold==4)check(status=="complete" && result==input["result"].get<double>() &&
                hit==input["hit"].get<double>() && work.distance==input["visited"].get<double>() &&
                work.candidates==input["events"].size() && work.intervals==input["leaves"].size() && sampler.consumed==tape.size(),
                "Refined slow-ray replay changed");
            else check(r.phases[1].splits==0,"Disabled transition still subdivides");
            J row={{"id",input["id"]},{"source",file},{"domain",input["domain"]},{"threshold",threshold},
                {"status",status},{"complete",status=="complete"},{"elapsed_us",r.setupUs+r.trackUs},
                {"visited",work.distance},{"length",length},{"intervals",work.intervals},{"candidates",work.candidates},
                {"random_draws",sampler.consumed},{"seeded_continuation_draws",sampler.consumed>tape.size() ? sampler.consumed-tape.size() : 0},
                {"phases",{{"birth",slowPhaseJson(r.phases[0])},{"transition",slowPhaseJson(r.phases[1])},{"far",slowPhaseJson(r.phases[2])}}}};
            if(status=="complete")row["transmittance_sample"]=result;
            if(!error.empty())row["error"]=error;
            for(const auto &leaf:r.leaves)if(leaf.phase==1) {
                row["first_transition"]={{"begin",leaf.begin},{"end",leaf.end},{"maximum",leaf.maximum},
                    {"C0",leaf.curve.constant},{"C1",leaf.curve.regressionCoefficient},
                    {"optical_depth",leaf.curve.integral(r.age+leaf.begin,r.age+leaf.end)}};
                break;
            }
            report["trials"].push_back(row);
            std::ofstream saved(output/"no_split_rays.json");saved<<report.dump(2);check(bool(saved),"Cannot save no-split diagnostics");
            std::cout<<"No-split comparison "<<input["id"]<<", threshold "<<threshold<<", "<<status
                <<", candidates "<<work.candidates<<", transition splits "<<r.phases[1].splits<<std::endl;
        }
    }
}
#endif
