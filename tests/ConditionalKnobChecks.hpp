#pragma once
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "math/GPFunctions.hpp"
#include "KnobLegacyFixtures.hpp"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

inline std::shared_ptr<Tungsten::ClassicGlobalMedium> knobMedium(const char *type,const char *mean)
{
    using namespace Tungsten;
    const std::string json=std::string("{\"type\":\"")+type+
        R"(","gaussian_process":{"type":"standard","mean":)"+mean+
        R"(,"covariance":{"type":"squared_exponential","sigma":0.02,"lengthScale":0.02}},
        "sdf_lipschitz":10,"gradient_bound":10,
        "material":{"type":"mirror","albedo":[0.15,0.8,0.5]}})";
    JsonDocument document(Path("knob-medium-test.json"),json);
    Scene scene;
    auto medium=std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(document));
    check(bool(medium),"knob medium factory");
    medium->loadResources(); medium->prepareForRender();
    return medium;
}

inline void sdfCapabilityChecks()
{
    using namespace Tungsten;
    const SdfFunctions::Function variants[]={
        SdfFunctions::Function::Knob,SdfFunctions::Function::KnobInner,SdfFunctions::Function::KnobOuter};
    for(const auto &p:LegacyKnobSamples) {
        const Vec3f x{float(p[0]),float(p[1]),float(p[2])};
        int material=0;
        for(int i=0;i<3;++i)
            close(SdfFunctions::eval(variants[i],x,material),p[3+i],3e-6,
                "single CSG definition preserves legacy knob geometry");
        close(SdfFunctions::evaluateJet(variants[0],V(p[0],p[1],p[2])).value,p[3],3e-6,
            "differential queries preserve legacy knob geometry");
    }
    // Unsupported functions retain their existing scalar/finite-difference
    // path, but must not pretend to provide conservative conditional bounds.
    for(const char *json : {
        R"({"type":"procedural","func":"knob_inner"})",
        R"({"type":"procedural","f":{"type":"constant","val":0.3}})"
    }) {
        JsonDocument document(Path("sdf-capability-test.json"),json);
        Scene scene;
        auto mean=scene.fetchMeanFunction(document);
        check(!mean->sdfQueries(),"unsupported procedural capability must be explicit");
        const Vec3d p(0.1,0.2,0.3);
        check(std::isfinite((*mean)(Derivative::None,p,Vec3d(0.0))),"legacy scalar evaluation");
        check(to_eigen3d(mean->dmean_da(p)).allFinite(),"legacy gradient evaluation");
        auto classic=knobMedium("classic_global",json);
        check(std::isfinite(classic->meanValue(p)),"legacy classic procedural mean");
        bool rejected=false;
        try { cg::SdfMeanField invalid(mean); } catch(const std::invalid_argument &) { rejected=true; }
        check(rejected,"generic adapter must reject missing SDF capabilities");
        rejected=false;
        try { knobMedium("global_conditional",json); } catch(const std::runtime_error &) { rejected=true; }
        check(rejected,"conditional medium must reject missing SDF capabilities");
    }
    // Zero scale and the minimum clamp must form a constant field, including
    // derivative and curvature bounds, even when the shape has CSG seams.
    JsonDocument constant(Path("sdf-constant-test.json"),
        R"({"type":"procedural","func":"knob","scale":0,"offset":-0.2,"min":0.1})");
    Scene scene;
    const auto mean=scene.fetchMeanFunction(constant);
    const auto *queries=mean->sdfQueries();
    check(queries!=nullptr,"knob exposes its SDF capabilities through MeanFunction");
    const V p(0.2,0.3,0.4),w(2,-1,0.5);
    close(queries->evaluateJet(p).value,double(0.1f),0,"zero scale clamp");
    close(queries->evaluateJet(p).gradient.norm(),0,0,"zero scale gradient");
    close(queries->valueDifference(p,w),0,0,"constant stable difference");
    const auto box=queries->bounds(p-w.cwiseAbs(),p+w.cwiseAbs());
    check(box.certified && box.minimumValue==box.maximumValue && box.maximumGradientNorm==0,
        "constant spatial bounds");
    const auto ray=queries->rayBounds(p,w,-2,3);
    check(ray.certified && ray.minimumValue==ray.maximumValue &&
        ray.minimumDerivative==0 && ray.maximumDerivative==0 && ray.maximumSecondDerivative==0,
        "constant directional bounds");
    close(ray.midpointValue,double(0.1f),0,"constant cached midpoint includes clamp");
    close(ray.midpointError,0,0,"constant cached midpoint is exact");
}

inline void knobChecks()
{
    using namespace Tungsten;
    sdfCapabilityChecks();
    cg::Random rng(29415);
    int smooth=0, refined=0;
    for(const char *json : {
        R"({"type":"procedural","func":"knob","transform":{"position":[0,0.8,0]}})",
        R"({"type":"procedural","func":"knob","transform":{"position":[0.2,-0.1,0.3],"rotation":[15,23,-11],"scale":[0.8,1.1,1.3]},"scale":-0.7,"offset":0.1,"min":-0.3})"
    }) {
        JsonDocument document(Path("knob-test.json"),json);
        Scene scene;
        auto mean=scene.fetchMeanFunction(document);
        check(mean->sdfQueries()!=nullptr,"procedural mean factory exposes SDF queries");
        auto knob=std::make_shared<cg::SdfMeanField>(mean);
        auto classic=knobMedium("classic_global",json);
        auto conditional=std::dynamic_pointer_cast<GlobalConditionalMedium>(knobMedium("global_conditional",json));
        check(bool(conditional),"conditional medium factory");
        rapidjson::Document serialized;
        auto value=mean->toJson(serialized.GetAllocator());
        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        value.Accept(writer);
        JsonDocument roundTrip(Path("knob-roundtrip.json"),buffer.GetString());
        auto copy=scene.fetchMeanFunction(roundTrip);
        const auto *before=dynamic_cast<const ProceduralMean *>(mean.get());
        const auto *after=dynamic_cast<const ProceduralMean *>(copy.get());
        check(before && after && before->scale()==after->scale() &&
            before->offset()==after->offset() && before->minimum()==after->minimum(),
            "procedural SDF scale/offset/clamp round trip");
        cg::GPSSField f{knob,cg::CovarianceKernel(0.02,M::Identity()/(0.02*0.02)),
                       {V::Constant(-100),V::Constant(100)}};
        for(int i=0;i<400;++i) {
            const V x(2*rng.uniform01()-1,2*rng.uniform01(),2*rng.uniform01()-1);
            const V w=V(rng.uniform01()-0.5,rng.uniform01()-0.5,rng.uniform01()-0.5).normalized();
            const auto jet=knob->evaluate(x);
            const auto cx=to_vec3d(x);
            close(jet.value,(*mean)(Derivative::None,cx,Vec3d(0.0)),0,"mean and adapter share values");
            close((jet.gradient-to_eigen3d(mean->dmean_da(cx))).norm(),0,0,"mean and adapter share gradients");
            close(jet.value,classic->meanValue(cx),0,"classic uses the same procedural value");
            close((jet.gradient-to_eigen3d(classic->meanGradient(cx))).norm(),0,0,"classic uses the same analytic gradient");
            const auto conditionalJet=conditional->field().mean->evaluate(x);
            close(jet.value,conditionalJet.value,0,"conditional uses the same procedural value");
            close((jet.gradient-conditionalJet.gradient).norm(),0,0,"conditional uses the same analytic gradient");
            close(copy->sdfQueries()->evaluateJet(x).value,jet.value,3e-6,"serialized procedural field");
            const double length=std::pow(10.0,-4+4*rng.uniform01());
            const auto bound=knob->rayBounds(x,w,0,length);
            check(bound.certified,"knob interval must be certified");
            SdfRayQueryOptions options;
            options.computeBeginDerivative=false;
            const auto fast=knob->rayBounds(x,w,0,length,options);
            check(fast.minimumValue==bound.minimumValue && fast.maximumValue==bound.maximumValue &&
                fast.minimumDerivative==bound.minimumDerivative && fast.maximumDerivative==bound.maximumDerivative &&
                fast.maximumSecondDerivative==bound.maximumSecondDerivative,
                "skipping beginning derivative preserves transformed certificates");
            close(bound.beginDerivative,jet.gradient.dot(w),0,"world beginning derivative is retained on request");
            close(fast.midpointValue,knob->evaluate(x+0.5*length*w).value,fast.midpointError,
                "transformed cached midpoint covers scale, offset and clamp");
            const auto box=mean->sdfQueries()->bounds(x-V::Constant(length),x+V::Constant(length));
            check(box.certified,"transformed knob box must be certified");
            double previousDerivative=knob->evaluate(x).gradient.dot(w);
            for(int j=1;j<=20;++j) {
                const auto value=knob->evaluate(x+(length*j/20)*w);
                const double derivative=value.gradient.dot(w);
                check(value.value>=box.minimumValue && value.value<=box.maximumValue &&
                    value.gradient.norm()<=box.maximumGradientNorm,"transformed spatial enclosure");
                check(value.value>=bound.minimumValue && value.value<=bound.maximumValue,"knob value enclosure");
                check(derivative>=bound.minimumDerivative && derivative<=bound.maximumDerivative,"knob derivative enclosure");
                if(std::isfinite(bound.maximumSecondDerivative)) {
                    check(std::abs(derivative-previousDerivative)<=
                        bound.maximumSecondDerivative*length/20+1e-10,"knob curvature enclosure");
                }
                previousDerivative=derivative;
            }
            const auto local=knob->rayBounds(x,w,0,1e-5);
            if(std::isfinite(local.maximumSecondDerivative)) {
                ++smooth;
                const double eps=1e-7;
                const double fd=(knob->evaluate(x+eps*w).value-knob->evaluate(x-eps*w).value)/(2*eps);
                close(jet.gradient.dot(w),fd,2e-7,"knob analytic directional gradient");
                const V d=1e-12*w;
                close(knob->valueDifference(x,d),jet.gradient.dot(d),
                      1e-21+local.maximumSecondDerivative*d.squaredNorm(),"stable knob mean difference");
            }
            if(i<100) {
                cg::FlightState birth;
                birth.birthPosition=x; birth.direction=w; birth.birthGradient=w;
                // Both a reflected zero and a positive entry observation.
                birth.birthValue=i%2?0.1:0;
                const cg::ConditionalFlightKernel flight(f,birth);
                const auto envelope=flight.birthMajorant(0.1);
                if(envelope.smoothEnd<0.01*(1-1e-6)) ++refined;
                for(int j=1;j<=30;++j) {
                    const double t=envelope.split*j/30;
                    check(flight.evaluate(t).hazard.value<=envelope.maximum*(1+1e-10),
                          "procedural knob birth envelope");
                }
                const double majorant=flight.intervalMajorant(0.03,0.04);
                for(int j=0;j<=20;++j)
                    check(flight.evaluate(0.03+0.01*j/20).hazard.value<=majorant,
                          "procedural knob far envelope");
            }
        }
    }
    check(smooth>300,"knob smooth branch coverage");
    check(refined>0,"knob CSG birth intervals must exercise adaptive refinement");
    // Actual shaderball paths that exposed an excessively loose cylinder-cap
    // curvature bound. Both leave their known zero with a positive slope, so
    // these tiny intervals must not require thousands of null collisions.
    JsonDocument preset(Path("knob-cap-test.json"),
        R"({"type":"procedural","func":"knob","transform":{"position":[0,0.8,0]}})");
    Scene scene;
    auto mean=scene.fetchMeanFunction(preset);
    cg::GPSSField f{std::make_shared<cg::SdfMeanField>(mean),
        cg::CovarianceKernel(0.02,M::Identity()/(0.02*0.02)),{V::Constant(-100),V::Constant(100)}};
    const V positions[]={
        V(-0.72625440319482504,0.3802830746543806,0.48301416317699419),
        V(0.53464407933850722,0.38012398347045723,0.53863862453771816)};
    const V gradients[]={
        V(-0.72591085862242466,-1.4203765063402567,-0.63986042512240449),
        V(-1.1809414428495326,1.6789163438921817,-1.2984677658498534)};
    const V directions[]={
        V(-0.54168610310683318,-0.78133729982861277,-0.30998095037835133),
        V(-0.14666873635352251,0.82822921256410675,0.5408554827603218)};
    for(int i=0;i<2;++i) {
        cg::FlightState birth;
        birth.birthPosition=positions[i]; birth.birthGradient=gradients[i]; birth.direction=directions[i];
        cg::ConditionalSegment segment(f,birth,positions[i],directions[i],0.01);
        const double lo=i?9e-5:2.8e-5,hi=lo+1e-7,bound=segment.majorant(lo,hi);
        check(bound*(hi-lo)<1,"knob cylinder cap must have an efficient conditional envelope");
        for(int j=0;j<=20;++j)
            check(segment.hazard(lo+(hi-lo)*j/20)<=bound,"knob cap envelope remains conservative");
        double t;
        for(int j=0;j<8;++j) segment.track(rng,true,t);
    }
    std::cout<<"knob: shared classic/conditional SDF queries, round trip, optional capabilities, transforms and "
             <<"certified value/derivative/curvature/hazard bounds passed; "<<refined<<" refined births\n";
}
