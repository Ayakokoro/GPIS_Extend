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

    // Three mathematical events fit between adjacent representable coordinates.
    // Each gets a distinct offset and is evaluated inside its own half-open interval.
    OpticalDepthSampler subUlpSampler({.5f,.5f,.5f,.5f});
    cg::Random subUlpRng(subUlpSampler);
    cg::MajorantOpticalDepth localClock;
    double offset=0;
    for (int i=1;i<=3;++i) {
        check(localClock.advanceFromOffset(subUlpRng,1e16,1,face,offset,t) && t==1,
            "sub-ULP candidate remains in its interval");
        close(offset,i*log2/1e16,1e-30,"sub-ULP free paths accumulate");
    }
    check(!localClock.advanceFromOffset(subUlpRng,1e16,1,face,offset,t) && subUlpSampler.calls==4,
        "sub-ULP candidates consume each exponential once");

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

    // Independent intensity M(t)=2t, with primitive t^2: exercise a varying
    // intensity, subdivision and a subsequent constant interval on one clock.
    const auto variableEvents=[&](int partitions) {
        OpticalDepthSampler source({.5f,.25f,.125f,.000001f});
        cg::Random random(source);cg::MajorantOpticalDepth process;
        std::vector<double> result;
        for(int i=0;i<partitions;++i) {
            double begin=2.*i/partitions,end=2.*(i+1)/partitions,event;
            while(process.advanceIntegrated(random,begin,end,end*end-begin*begin,
                [&](double depth) { return std::sqrt(begin*begin+depth); },event)) {
                result.push_back(event);begin=event;
            }
        }
        double event;
        check(process.advance(random,1,2,3,event),"integrated clock retains its threshold into a constant segment");
        result.push_back(event);
        check(source.calls==3,"variable partitions reuse the exponential clock");
        return result;
    };
    const auto variableCoarse=variableEvents(1),variableFine=variableEvents(200);
    check(variableCoarse.size()==3 && variableFine.size()==3,"nonconstant clock event count");
    close(variableCoarse[0],std::sqrt(log2),1e-14,"nonconstant clock first event");
    close(variableCoarse[1],std::sqrt(3*log2),1e-14,"nonconstant clock second event");
    close(variableCoarse[2],2+6*log2-4,1e-14,"variable-to-constant residual event");
    for(size_t i=0;i<variableCoarse.size();++i)
        close(variableCoarse[i],variableFine[i],1e-13,"nonconstant clock subdivision invariance");
}
