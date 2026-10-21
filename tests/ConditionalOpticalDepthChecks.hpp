#pragma once
#include "math/conditional/transport/MajorantOpticalDepth.h"

class OpticalDepthSampler final : public Tungsten::PathSampleGenerator {
    std::vector<float> _values;
    Tungsten::UniformSampler _unused{1};
public:
    size_t calls=0;
    explicit OpticalDepthSampler(std::vector<float> values):_values(std::move(values)) {}
    void startPath(Tungsten::uint32,Tungsten::uint32) override { calls=0; }
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override { throw std::runtime_error("unused sampler operation"); }
    void loadState(Tungsten::InputStreamHandle &) override { throw std::runtime_error("unused sampler operation"); }
    bool nextBoolean(float p) override { return next1D()<p; }
    int nextDiscrete(int n) override { return int(next1D()*n); }
    float next1D() override {
        check(calls<_values.size(),"optical-depth clock consumed an unexpected random draw");
        return _values[calls++];
    }
    Tungsten::Vec2f next2D() override { const float x=next1D();return Tungsten::Vec2f(x,next1D()); }
    Tungsten::UniformSampler &uniformGenerator() override { return _unused; }
};

inline void opticalDepthChecks()
{
    const double log2=std::log(2.0);
    OpticalDepthSampler sampler({0.5f,0.25f,0.000001f});
    cg::Random rng(sampler);
    cg::MajorantOpticalDepth clock;
    double t=-1;
    check(!clock.advance(rng,0,0,0.5,t) && sampler.calls==0,
        "initial zero-rate interval consumes no random sample");
    check(!clock.advance(rng,0.5,0.5,1,t) && sampler.calls==1,
        "first positive interval draws one exponential threshold");
    check(!clock.advance(rng,0,1,2,t) && sampler.calls==1,
        "zero-rate gap retains the pending threshold");
    check(clock.advance(rng,2,2,3,t) && sampler.calls==1,
        "rate change consumes residual optical depth without resampling");
    close(t,2+(log2-0.25)/2,1e-15,"candidate matches independent integrated-intensity inversion");
    const double first=t;
    check(clock.advance(rng,2,t,3,t) && sampler.calls==2,
        "each subsequent candidate draws a new exponential threshold");
    close(t,first+std::log(4.0)/2,1e-15,"second candidate follows its independent threshold");
    check(!clock.advance(rng,2,t,3,t) && sampler.calls==3,"final residual survives interval exit");

    OpticalDepthSampler boundarySampler({0.5f,0.5f});
    cg::Random boundaryRng(boundarySampler);
    cg::MajorantOpticalDepth boundary;
    check(!boundary.advance(boundaryRng,log2,0,1,t),"exact boundary candidate is deferred");
    check(!boundary.advance(boundaryRng,0,1,2,t),"boundary candidate survives a zero-rate gap");
    check(boundary.advance(boundaryRng,4,2,3,t) && t==2 && boundarySampler.calls==1,
        "boundary candidate is emitted exactly once without a new draw");
    check(boundary.advance(boundaryRng,4,t,3,t) && t>2 && boundarySampler.calls==2,
        "a positive increment follows a boundary candidate");

    OpticalDepthSampler roundedSampler({0.5f});
    cg::Random roundedRng(roundedSampler);
    cg::MajorantOpticalDepth rounded;
    const double face=std::nextafter(1.0,2.0);
    check(log2<4e15*(face-1),"rounded-face fixture contains the mathematical candidate");
    check(!rounded.advance(roundedRng,4e15,1,face,t),"rounded face carries the pending event");
    check(rounded.advance(roundedRng,1,face,2,t) && t==face && roundedSampler.calls==1,
        "rounded face neither discards nor resamples its event");

    // For a fixed rate, subdivision must preserve both candidate locations and
    // RNG consumption. This comparison uses no transport majorant implementation.
    const auto events=[&](int partitions) {
        OpticalDepthSampler source({0.5f,0.25f,0.125f,0.000001f});
        cg::Random random(source);
        cg::MajorantOpticalDepth process;
        std::vector<double> result;
        for(int i=0;i<partitions;++i) {
            const double end=4.0*(i+1)/partitions;
            double begin=4.0*i/partitions,event;
            while(process.advance(random,1.5,begin,end,event)) {
                result.push_back(event);
                begin=event;
            }
        }
        check(source.calls==4,"empty subintervals do not add exponential draws");
        return result;
    };
    const auto coarse=events(1),fine=events(400);
    check(coarse.size()==3 && fine.size()==coarse.size(),"subdivision preserves candidate count");
    for(size_t i=0;i<coarse.size();++i)
        close(coarse[i],fine[i],1e-13,"subdivision preserves candidate positions");
}
