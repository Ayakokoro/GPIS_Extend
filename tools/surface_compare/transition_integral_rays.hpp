#pragma once
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include <functional>

class IntegralLongClockSampler final : public Tungsten::PathSampleGenerator {
    Tungsten::UniformSampler unused;
public:
    void startPath(Tungsten::uint32,Tungsten::uint32) override {}
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override {}
    void loadState(Tungsten::InputStreamHandle &) override {}
    bool nextBoolean(float) override { return false; }
    int nextDiscrete(int) override { return 0; }
    float next1D() override { return std::numeric_limits<float>::min(); }
    Tungsten::Vec2f next2D() override { return Tungsten::Vec2f(next1D()); }
    Tungsten::UniformSampler &uniformGenerator() override { return unused; }
};

inline void exportIntegralRays(const nlohmann::json &spec,const std::filesystem::path &output,
                              const Tungsten::ConditionalGpis::GPSSField &loadedField) {
    namespace cg=Tungsten::ConditionalGpis;
    using J=nlohmann::json;
    const auto require=[](bool ok,const char *message) { if(!ok) throw std::runtime_error(message); };
    const auto vec=[](const J &x) { return cg::Vector3(x[0].get<double>(),x[1].get<double>(),x[2].get<double>()); };
    const auto global=loadedField.mean->globalMagnitudeBounds();require(bool(global),"Missing VDB gradient certificate");
    const double L=global->maximumGradientNorm;
    J results=J::array();
    for(const auto &file:spec.at("ray_files")) {
        std::ifstream input(file.get<std::string>());const J previous=J::parse(input);
        for(const auto &ray:previous.at("rays")) {
            const auto &profile=ray["profiles"][0];
            const double referenceEll=profile["length_scale"],referenceSigma=profile["sigma"];
            const auto &field=loadedField;
            cg::FlightState birth;birth.birthPosition=vec(ray["origin"]);birth.direction=vec(ray["direction"]);
            birth.birthGradient=vec(ray["birth_gradient"]);birth.birthValue=ray["birth_value"];
            const double end=ray["length"],ell=1/std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
            // Scene loading retains the renderer's float-derived covariance;
            // the old profile's length_scale is the requested JSON parameter.
            require(std::abs(ell-referenceEll)<1e-8 && field.kernel.sigma()==referenceSigma,
                    "Mismatched ray/field parameters");
            const cg::ConditionalFlightKernel kernel(field,birth);
            const cg::ConditionalFlightKernel profileKernel(field,birth);
            const auto initial=kernel.birthMajorant(end);
            const double far=profile["majorants"]["far_split"];
            cg::ConditionalTrackingSettings options;options.fixedStepLimit=false;options.localFarStartScale=4;
            options.transitionIntegralBounds=true;options.shadowRouletteThreshold=0;
            cg::ConditionalSegment segment(field,birth,birth.birthPosition,birth.direction,end,options,nullptr,L,L);
            J rows=J::array(),worstDifference;double cost=0,maxRatio=0,maxHazardError=0;std::uint64_t checked=0;
            const auto verify=[&](double t,double bound,double previousHazard=-1) {
                const auto &evaluationKernel=previousHazard>=0 ? profileKernel : kernel;
                const double actual=evaluationKernel.evaluate(t).hazard.value;
                require(actual<=bound*(1+1e-10),"VDB ray hazard exceeds integral envelope");
                if(bound>0) maxRatio=std::max(maxRatio,actual/bound);
                if(previousHazard>=0) {
                    const double error=std::abs(actual-previousHazard)/(1+actual);
                    if(error>maxHazardError) {
                        maxHazardError=error;
                        const auto stats=evaluationKernel.scalarStatistics(t);
                        worstDifference={{"t",t},{"actual_hazard",actual},{"previous_hazard",previousHazard},
                            {"conditional_mean",stats.meanF},{"conditional_variance",stats.varianceF},
                            {"mean_derivative",stats.meanK},{"crossing_mean",stats.meanAtZero},
                            {"actual_length_scale",ell},{"actual_sigma",field.kernel.sigma()}};
                    }
                }
                ++checked;
            };
            std::function<void(double,double,double)> refine;
            refine=[&](double lo,double hi,double parent) {
                const bool transition=lo>=initial.split && lo<far;
                cg::ConditionalIntegralMajorant curve;
                const double maximum=transition ? (curve=kernel.transitionIntegralMajorant(lo,hi,L)).value(lo)
                    : segment.majorant(lo,hi,parent);
                const double depth=transition ? curve.integral(lo,hi) : maximum*(hi-lo);
                if(depth>(transition ? options.transitionMaxExpectedEvents : options.maxExpectedEvents)) {
                    const double mid=lo+.5*(hi-lo);require(mid>lo&&mid<hi,"Cannot refine integral ray");
                    refine(lo,mid,maximum);refine(mid,hi,maximum);return;
                }
                for(int j=0;j<=16;++j) { const double t=lo+(hi-lo)*j/16;verify(t,transition?curve.value(t):maximum); }
                rows.push_back({{"begin",lo},{"end",hi},{"maximum",maximum},{"optical_depth",depth},
                    {"constant",transition?curve.constant:maximum},{"regression_coefficient",transition?curve.regressionCoefficient:0},
                    {"inverse_length2",transition?curve.inverseLength2:1/(ell*ell)},
                    {"phase",lo>=far?"far":transition?"transition":"birth"}});
                cost+=depth;
            };
            for(double lo=0;lo<end;) {
                double hi;
                if(lo<initial.split) hi=std::min(end,initial.split);
                else {
                    const double mean=kernel.scalarMeanValue(lo,lo<far);
                    const double step=L>0 ? std::max(field.kernel.sigma()/L,.5*std::max(0.,mean)/L) : end-lo;
                    hi=std::min(end,lo+step);if(lo<far) hi=std::min(hi,far);
                }
                require(hi>lo,"Integral replay cannot advance");refine(lo,hi,std::numeric_limits<double>::infinity());lo=hi;
            }
            J points=J::array();size_t leaf=0;
            for(const auto &p:profile["points"]) {
                const double t=p["t"];while(leaf+1<rows.size() && t>=rows[leaf]["end"].get<double>()) ++leaf;
                const auto &r=rows[leaf];
                const cg::ConditionalIntegralMajorant curve{r["constant"],r["regression_coefficient"],r["inverse_length2"]};
                const double bound=t==0 ? r["maximum"].get<double>() : curve.value(t);
                verify(t,bound,p["conditional"]);
                points.push_back({{"t",t},{"bound",bound},{"conditional",profileKernel.evaluate(t).hazard.value}});
            }
            IntegralLongClockSampler sampler;cg::Random rng(sampler);cg::ConditionalTrackingWork work;double hit;
            const double weight=segment.track(rng,false,hit,&work);
            require(weight>0 && work.distance==end && work.intervals==rows.size(),"Integral replay differs from production partition");
            J result={{"source",file},{"kind",ray["kind"]},{"length",end},{"birth_split",initial.split},{"far_split",far},
                {"intervals",rows},{"points",points},{"integrated_majorant",cost},{"coverage_points",checked},
                {"maximum_hazard_ratio",maxRatio},{"maximum_hazard_change",maxHazardError},{"worst_reference_difference",worstDifference},
                {"production_replay_intervals",work.intervals},{"production_replay_candidates",work.candidates}};
            results.push_back(result);
            std::ofstream saved(output/"rays.json");saved<<results.dump(2);
            std::cout<<"Verified integral ray "<<file<<" / "<<ray["kind"]<<", intervals "<<rows.size()<<", integral "<<cost
                     <<", previous hazard difference "<<maxHazardError<<std::endl;
        }
    }
}
