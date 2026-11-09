#pragma once
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

inline void nonGpSurfaceChecks()
{
    using namespace Tungsten;
    using Policy = ConditionalNonGpSurfacePolicy;
    auto retained=makeMedium();
    auto restarted=makeMedium(R"("non_gp_surface_policy":"restart_exterior")");
    check(retained->nonGpSurfacePolicy()==Policy::RetainEndpoint,
        "omitted non-GP surface policy preserves the existing behavior");
    check(restarted->nonGpSurfacePolicy()==Policy::RestartExterior,
        "non-GP restart policy is loaded");
    for(const auto &medium : {retained,restarted}) {
        rapidjson::Document serialized;
        auto json=medium->toJson(serialized.GetAllocator());
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        json.Accept(writer);
        JsonDocument document(Path("non-gp-surface-roundtrip.json"),buffer.GetString());
        Scene scene;
        auto copy=std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(document));
        check(copy && copy->nonGpSurfacePolicy()==medium->nonGpSurfacePolicy(),
            "non-GP surface policy survives scene serialization");
    }
    // Invalid JSON options are checked separately in a renderer subprocess.

    const Vec3f position(0.2f,-0.1f,0.0f),incoming(0.0f,0.0f,-1.0f);
    auto observation=std::make_shared<ConditionalPathContext>();
    observation->owner=retained.get();observation->rayOrigin=position;
    observation->position=V(std::nextafter(double(position.x()),1.0),double(position.y()),0);
    observation->birth.birthPosition=observation->position;
    observation->birth.birthValue=7;observation->birth.birthGradient=V(3,-2,1);
    Medium::MediumState keep{};keep.reset();keep.pathContext=observation;keep.bounce=3;
    IndependentPathSampler keepSampler(199,false),untouchedSampler(199,false);
    keepSampler.startPath(2,0);untouchedSampler.startPath(2,0);
    retained->prepareNonGpSurface(keepSampler,position,incoming,keep);
    check(keep.pathContext==observation && keep.bounce==3 && keep.firstScatter,
        "retain policy does not replace the observation or change medium state");
    check(keepSampler.next1D()==untouchedSampler.next1D(),
        "retain policy does not consume random numbers");

    auto oldA=std::make_shared<ConditionalPathContext>(*observation);
    oldA->owner=restarted.get();
    auto oldB=std::make_shared<ConditionalPathContext>(*oldA);
    oldB->birth.birthPosition=V(-12,8,10);
    oldB->birth.birthValue=100;oldB->birth.birthGradient=V(-8,14,6);
    IndependentPathSampler samplerA(7191,false),samplerB(7191,false);
    double sumF=0,sumF2=0;
    V sumG=V::Zero(),sumG2=V::Zero();
    constexpr int count=20000;
    for(int i=0;i<count;++i) {
        samplerA.startPath(7,i);samplerB.startPath(7,i);
        Medium::MediumState a{};a.reset();a.pathContext=oldA;
        a.bounce=7;a.firstScatter=(i%2==0);a.lastVal=0.625f;
        auto b=a;b.pathContext=oldB;
        restarted->prepareNonGpSurface(samplerA,position,incoming,a);
        restarted->prepareNonGpSurface(samplerB,position,incoming,b);
        const auto ca=std::dynamic_pointer_cast<const ConditionalPathContext>(a.pathContext);
        const auto cb=std::dynamic_pointer_cast<const ConditionalPathContext>(b.pathContext);
        check(ca && cb && ca!=oldA && cb!=oldB,"restart replaces only the local observation");
        check(ca->owner==restarted.get() && ca->rayOrigin==position &&
            ca->position==oldA->position && ca->birth.birthPosition==ca->position,
            "restart preserves the precise geometry endpoint");
        check(ca->birth.birthValue==cb->birth.birthValue && ca->birth.birthGradient==cb->birth.birthGradient,
            "restarted observation is independent of old birth values and gradients");
        check(ca->birth.birthValue>0 && ca->birth.birthGradient.allFinite(),
            "restart retains positive exterior support and a finite gradient");
        check(a.bounce==7 && a.firstScatter==(i%2==0) && a.lastVal==0.625f,
            "restart must not clear unrelated medium state or require a previous GP hit");
        const auto branch=a;
        check(branch.pathContext==a.pathContext,"surface branches share the newly sampled observation");
        sumF+=ca->birth.birthValue;sumF2+=ca->birth.birthValue*ca->birth.birthValue;
        sumG+=ca->birth.birthGradient;sumG2+=ca->birth.birthGradient.array().square().matrix();
    }
    // At z=0 this fixture has F~N(0,1), conditioned to F>0, and G~N((0,0,1),I).
    // These analytic moments also distinguish an exterior restart from F=0
    // surface conditioning or sampling a non-truncated value distribution.
    close(sumF/count,std::sqrt(2.0/std::acos(-1.0)),0.025,"exterior restart half-normal value mean");
    close(sumF2/count,1.0,0.05,"exterior restart half-normal value second moment");
    check((sumG/count-V::UnitZ()).cwiseAbs().maxCoeff()<0.04,"exterior restart prior gradient mean");
    check((sumG2/count-V(1,1,2)).cwiseAbs().maxCoeff()<0.07,"exterior restart prior gradient second moment");
    check(oldA->birth.birthValue==7 && oldB->birth.birthValue==100,
        "restart does not mutate shared incoming observations");
    std::cout<<"non-GP surface policies: defaults, roundtrip, unchanged retain RNG, history-independent exterior moments and exact endpoints passed\n";
}
