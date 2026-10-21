#pragma once
#include <chrono>
#include <iomanip>

// An explicit diagnostic command, not part of CTest. A draw budget reports
// unfinished flights without changing the renderer's tracking distribution.
inline void trackingProbe(const char *sceneFile,bool sweepSplits=false)
{
    using namespace Tungsten;
    using Clock=std::chrono::steady_clock;
#if OPENVDB_AVAILABLE
    openvdb::initialize();
#endif
    struct BudgetReached {};
    class CappedSampler final : public PathSampleGenerator {
        UniformSampler uniform{7391};
        std::uint64_t draws=0;
    public:
        void startPath(uint32,uint32) override {}
        void advancePath() override {}
        void saveState(OutputStreamHandle &) override { throw std::runtime_error("unused"); }
        void loadState(InputStreamHandle &) override { throw std::runtime_error("unused"); }
        bool nextBoolean(float p) override { return next1D()<p; }
        int nextDiscrete(int n) override { return int(n*next1D()); }
        float next1D() override { if(++draws>500000) throw BudgetReached{};return uniform.next1D(); }
        Vec2f next2D() override { const float x=next1D();return Vec2f(x,next1D()); }
        UniformSampler &uniformGenerator() override { return uniform; }
    };
    std::unique_ptr<Scene> scene(Scene::load(Path(sceneFile)));
    std::shared_ptr<GlobalConditionalMedium> medium;
    for(const auto &m:scene->media()) if(auto c=std::dynamic_pointer_cast<GlobalConditionalMedium>(m)) {
        check(!medium,"probe requires exactly one global_conditional medium");medium=c;
    }
    check(bool(medium),"probe requires a global_conditional medium");
    medium->loadResources();medium->prepareForRender();
    const auto &f=medium->field();
    const auto global=f.mean->globalMagnitudeBounds();
    check(bool(global),"probe needs global magnitude bounds");
    std::cout<<std::setprecision(17)<<"{\"S0\":"<<global->maximumAbsoluteValue
        <<",\"S1\":"<<global->maximumGradientNorm<<"}"<<std::endl;
    cg::Random observations(4221);
    const auto run=[&](const char *label,const cg::FlightState &birth,double length) {
        const cg::ConditionalFlightKernel kernel(f,birth);
        const auto b=kernel.birthMajorant(length);
        const double tail=b.split<length ? kernel.constantTailMajorant(b.split) : 0;
        std::cout<<"{\"flight\":\""<<label<<"\",\"h\":"<<b.split<<",\"M_birth\":"<<b.maximum
            <<",\"M_tail\":"<<tail<<",\"length\":"<<length<<",\"proposal_integral\":"
            <<b.split*b.maximum+(length-b.split)*tail<<"}"<<std::endl;
        const std::vector<double> requests=sweepSplits ?
            std::vector<double>{-1,0,0.02,0.04,0.1,0.2,0.5,1.0,3.0} : std::vector<double>{-1,0};
        for(double requested:requests) {
            const bool two=requested>=0;
            if(requested>0) {
                const auto beforeBounds=Clock::now();
                const double split=std::min(length,requested);
                const auto prefix=kernel.twoSegmentMajorants(split);
                const double maximum=std::max(prefix.nearMaximum,prefix.farMaximum);
                const double tailMaximum=split<length ? kernel.constantTailMajorant(split) : 0;
                const double boundMs=std::chrono::duration<double,std::milli>(Clock::now()-beforeBounds).count();
                std::cout<<"{\"flight\":\""<<label<<"\",\"requested_birth_length\":"<<requested
                    <<",\"h\":"<<split<<",\"M_birth\":"<<maximum<<",\"M_tail\":"<<tailMaximum
                    <<",\"length\":"<<length<<",\"proposal_integral\":"<<maximum*split+tailMaximum*(length-split)
                    <<",\"bound_intervals\":"<<prefix.boundIntervals<<",\"bound_ms\":"<<boundMs<<"}"<<std::endl;
            }
            for(bool collision : {true,false}) {
                auto settings=medium->trackingSettings();
                settings.majorantMode=two ? cg::ConditionalMajorantMode::TwoConstants : cg::ConditionalMajorantMode::Adaptive;
                settings.constantBirthLength=std::max(0.0,requested);
                const auto start=Clock::now();
                const cg::ConditionalSegment segment(f,birth,birth.birthPosition,birth.direction,length,settings);
                const auto prepared=Clock::now();
                CappedSampler source;cg::Random rng(source);cg::ConditionalTrackingWork work;
                double hit=0,weight=0;const char *status="complete";
                try { weight=segment.track(rng,collision,hit,&work); }
                catch(const BudgetReached &) { status="draw_budget"; }
                const auto end=Clock::now();
                std::cout<<"{\"flight\":\""<<label<<"\",\"mode\":\""<<(two ? "two_constants":"adaptive")
                    <<"\",\"estimator\":\""<<(collision ? "delta":"ratio")<<"\",\"status\":\""<<status
                    <<"\",\"requested_birth_length\":"<<settings.constantBirthLength
                    <<",\"setup_ms\":"<<std::chrono::duration<double,std::milli>(prepared-start).count()
                    <<",\"track_ms\":"<<std::chrono::duration<double,std::milli>(end-prepared).count()
                    <<",\"candidates\":"<<work.candidates<<",\"intervals\":"<<work.intervals
                    <<",\"distance\":"<<work.distance;
                if(std::string(status)=="complete") std::cout<<",\"weight\":"<<weight;
                std::cout<<"}"<<std::endl;
            }
        }
    };
    // Fixed front-entry ray through the knob, followed by an actual collision
    // sampled with the reference tracker and its reflected outgoing flight.
    const V position(0.137,0.923,1.5),direction(0,0,-1);
    auto birth=cg::startConditionalExterior(f,position,direction,observations);
    run("entry",birth,3.0);
    auto adaptive=medium->trackingSettings();adaptive.majorantMode=cg::ConditionalMajorantMode::Adaptive;
    const cg::ConditionalSegment incoming(f,birth,position,direction,3.0,adaptive);
    double hit;
    if(incoming.track(observations,true,hit)==0) {
        const auto gradient=cg::sampleFluxWeightedGradient(incoming.gradient(hit),direction,observations);
        const V normal=gradient.normalized();
        birth.birthPosition=position+hit*direction;birth.birthValue=0;birth.birthGradient=gradient;
        birth.direction=direction-2*direction.dot(normal)*normal;
        run("reflected",birth,0.5);
    }
}
