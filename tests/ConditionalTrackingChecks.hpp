#pragma once
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include "ConditionalOpticalDepthChecks.hpp"

inline void trackingChecks()
{
    using namespace Tungsten;
    using Settings = cg::ConditionalTrackingSettings;
    opticalDepthChecks();
    const auto defaults=makeMedium()->trackingSettings();
    check(defaults.step==0 && defaults.stepScale==0.5 && defaults.ageScale==0.25 &&
        defaults.maxExpectedEvents==64 && defaults.tightenExpectedEvents==4 &&
        defaults.macrocellSize==0 && defaults.macrocellMaxExpectedEvents==1,
        "omitted tracking options preserve defaults");
    auto medium=makeMedium(R"("tracking_step":0.2,"tracking_step_scale":2,
        "tracking_age_scale":0.6,"tracking_max_expected_events":0.5,"tracking_tighten_expected_events":0.1,
        "tracking_macrocell_size":4,"tracking_macrocell_max_expected_events":0.25)");
    const auto settings=medium->trackingSettings();
    check(settings.step==0.2 && settings.stepScale==2 && settings.ageScale==0.6 &&
        settings.maxExpectedEvents==0.5 && settings.tightenExpectedEvents==0.1 &&
        settings.macrocellSize==4 && settings.macrocellMaxExpectedEvents==0.25,"scene tracking options are loaded");
    rapidjson::Document serialized;
    auto json=medium->toJson(serialized.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    JsonDocument document(Path("tracking-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto copy=std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(document));
    const auto restored=copy->trackingSettings();
    check(restored.step==settings.step && restored.stepScale==settings.stepScale &&
        restored.ageScale==settings.ageScale && restored.maxExpectedEvents==settings.maxExpectedEvents &&
        restored.tightenExpectedEvents==settings.tightenExpectedEvents &&
        restored.macrocellSize==settings.macrocellSize &&
        restored.macrocellMaxExpectedEvents==settings.macrocellMaxExpectedEvents,
        "tracking settings survive scene serialization");
    check(makeMedium(R"("tracking_tighten_expected_events":0)")->trackingSettings().tightenExpectedEvents==0,
        "zero tightening threshold is a supported fine-bound reference mode");
    for(const char *invalid : {R"("tracking_step":-1)",R"("tracking_step_scale":0)",
            R"("tracking_age_scale":-1)",R"("tracking_max_expected_events":0)",
            R"("tracking_tighten_expected_events":-1)",R"("tracking_macrocell_size":3)",
            R"("tracking_macrocell_size":1)",R"("tracking_macrocell_size":128)",
            R"("tracking_macrocell_size":-8)",R"("tracking_macrocell_max_expected_events":0)"}) {
        bool rejected=false;
        try { makeMedium(invalid); } catch(const std::exception &) { rejected=true; }
        check(rejected,"invalid tracking configuration must fail during loading");
    }
    for(auto member : {&Settings::step,&Settings::stepScale,&Settings::ageScale,
            &Settings::maxExpectedEvents,&Settings::tightenExpectedEvents,&Settings::macrocellMaxExpectedEvents})
        for(double invalid : {-1.0,std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::quiet_NaN()}) {
            auto bad=defaults; bad.*member=invalid;
            bool rejected=false;
            try { bad.validate(); } catch(const cg::NumericError &) { rejected=true; }
            check(rejected,"direct tracking options reject negative and nonfinite values");
        }

    // Observe actual SDF queries on an empty flight, independently of the
    // settings getters. This catches options that parse but never reach track().
    class RecordingMean final : public cg::MeanField {
    public:
        mutable std::vector<double> lengths;
        const char *typeName() const override { return "tracking-test"; }
        cg::MeanJet evaluate(const V &) const override { return {10,V::Zero()}; }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {10,0,true}; }
        std::optional<V> affineGradient() const override { return V::Zero(); }
        void appendRayBreakpoints(const V &,const V &,double lo,double hi,std::vector<double> &) const override {
            lengths.push_back(hi-lo);
        }
    };
    const auto mean=std::make_shared<RecordingMean>();
    cg::GPSSField empty{mean,cg::CovarianceKernel(1,M::Identity()/(0.02*0.02)),
        {V::Constant(-10),V::Constant(10)}};
    cg::FlightState birth;birth.birthValue=10;
    const auto partitions=[&](Settings options) {
        cg::ConditionalSegment flight(empty,birth,V(0,0,0.2),V::UnitZ(),0.117,options);
        mean->lengths.clear(); // Exclude the birth certificate's queries.
        cg::Random rng(773);double hit;
        close(flight.track(rng,false,hit),1,1e-12,"empty flight transmittance");
        return mean->lengths;
    };
    check(partitions(defaults).size()==12,"default cap makes twelve ordinary intervals");
    auto wider=defaults;wider.stepScale=1;
    check(partitions(wider).size()==6,"one GP length halves interval queries");
    wider.stepScale=2;
    check(partitions(wider).size()==3,"two GP lengths quarter interval queries");
    wider.step=0.03;wider.stepScale=0.1;
    check(partitions(wider).size()==4,"absolute step overrides the kernel-scale setting");
    wider.ageScale=0.025;
    close(partitions(wider).front(),0.005,1e-14,"age growth still limits the ordinary step");

    // Force refinement on a constant far-from-birth field and observe actual
    // bound queries. A collision must not evaluate deferred right siblings;
    // an escaping flight still needs to tighten those siblings when visited.
    class RecordingRefinementMean final : public cg::MeanField {
    public:
        mutable std::vector<std::pair<double,double>> intervals;
        const char *typeName() const override { return "refinement-test"; }
        cg::MeanJet evaluate(const V &) const override { return {0,V::Zero()}; }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {0,0,true}; }
        std::optional<V> affineGradient() const override { return V::Zero(); }
        cg::MeanRayBounds rayBounds(const V &o,const V &w,double lo,double hi,
                                    const SdfRayQueryOptions &options = {}) const override {
            intervals.emplace_back(lo,hi);
            return MeanField::rayBounds(o,w,lo,hi,options);
        }
    };
    const auto refinementMean=std::make_shared<RecordingRefinementMean>();
    cg::GPSSField refinementField{refinementMean,cg::CovarianceKernel(1,M::Identity()),
        {V::Constant(-100),V::Constant(100)}};
    auto refinementOptions=defaults;
    refinementOptions.step=4;refinementOptions.ageScale=1;refinementOptions.maxExpectedEvents=0.2;
    cg::FlightState refinementBirth;refinementBirth.birthGradient=V::UnitZ();
    const cg::ConditionalSegment refined(refinementField,refinementBirth,V(0,0,10),V::UnitZ(),4,refinementOptions);
    bool earlyCollision=false,visitedRight=false;
    for(int seed=1;seed<=64 && !(earlyCollision && visitedRight);++seed) {
        refinementMean->intervals.clear();
        cg::Random rng(seed);double hit;
        const double survived=refined.track(rng,true,hit);
        if(!survived && hit<0.5) {
            earlyCollision=true;
            check(refinementMean->intervals.size()>1,"test flight actually refines its first interval");
            for(const auto &interval : refinementMean->intervals)
                check(interval.first<=10+hit,"collision skips all unvisited right-sibling bounds");
        }
        if(survived) {
            for(const auto &interval : refinementMean->intervals)
                visitedRight|=interval.first>10;
            check(hit==4 && visitedRight,"escaping flight reaches and tightens deferred intervals");
        }
    }
    check(earlyCollision && visitedRight,"refinement fixture exercises early collision and deferred traversal");

    // Exercise the real track() loop too: an exponential threshold larger than
    // the whole flight's majorant integral must survive every refined interval.
    for(bool collision : {false,true}) {
        refinementMean->intervals.clear();
        OpticalDepthSampler source({std::numeric_limits<float>::min()});
        cg::Random rng(source);double hit;
        close(refined.track(rng,collision,hit),1,0,"one large threshold escapes the refined flight");
        check(hit==4 && source.calls==1 && refinementMean->intervals.size()>4,
            "delta and ratio tracking carry one draw through deferred refinement");
    }
    auto smallSteps=refinementOptions;smallSteps.step=0.01;
    const cg::ConditionalSegment finelyPartitioned(refinementField,refinementBirth,
        V(0,0,10),V::UnitZ(),4,smallSteps);
    for(bool collision : {false,true}) {
        refinementMean->intervals.clear();
        OpticalDepthSampler source({std::numeric_limits<float>::min()});
        cg::Random rng(source);double hit;
        close(finelyPartitioned.track(rng,collision,hit),1,0,"one large threshold escapes fine steps");
        check(source.calls==1 && refinementMean->intervals.size()>=400,
            "hundreds of empty intervals use only one exponential draw");
    }

    // The configured medium must use the same options on collision and shadow
    // paths. Couple random streams to check the actual medium entry points.
    auto context=std::make_shared<ConditionalPathContext>();
    context->owner=medium.get();context->position=V(0,0,0.4);
    context->rayOrigin=Vec3f(0.0f,0.0f,0.4f);
    context->birth.birthPosition=context->position;
    context->birth.birthValue=0.4;context->birth.birthGradient=V(0.2,0.1,1);
    const Ray ray(context->rayOrigin,Vec3f(0.0f,0.0f,-1.0f),0,1.25f);
    const cg::ConditionalSegment expected(medium->field(),context->birth,context->position,
        V(0,0,-1),ray.farT(),settings);
    IndependentPathSampler directSampler(907,false),mediumSampler(907,false);
    for(int i=0;i<40;++i) for(bool collision : {false,true}) {
        directSampler.startPath(0,i);mediumSampler.startPath(0,i);
        cg::Random rng(directSampler);double hit;
        const double weight=expected.track(rng,collision,hit);
        Medium::MediumState state{};state.reset();state.pathContext=context;
        if(collision) {
            MediumSample sample{};
            check(medium->sampleDistance(mediumSampler,ray,state,sample),"configured distance sampling");
            check(sample.exited==(weight!=0) && sample.t==float(hit),
                "distance sampling forwards tracking settings");
        } else {
            const auto tr=medium->transmittance(mediumSampler,ray,true,true,&state);
            check(tr==Vec3f(float(weight)),"shadow tracking forwards tracking settings");
        }
    }

    // Longer intervals and stronger refinement must preserve the integrated
    // hazard law. Check delta and ratio tracking with an independent integral.
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(0);
    birth=cg::FlightState{};birth.birthGradient=V(0.2,-0.1,1);
    const V direction=V(0.2,0.3,1).normalized(),origin=0.4*direction;
    const cg::ConditionalSegment referenceFlight(f,birth,origin,direction,2);
    double integral=0;
    for(int i=0;i<10000;++i) integral+=referenceFlight.hazard(2*(i+0.5)/10000)*2/10000;
    const double survival=std::exp(-integral);
    for(bool absolute : {false,true}) for(double threshold : {0.0,4.0,1e6}) {
        auto options=defaults;options.stepScale=2;options.ageScale=0.5;
        options.tightenExpectedEvents=threshold;
        if(absolute) { options.step=0.2;options.maxExpectedEvents=0.25; }
        const cg::ConditionalSegment flight(f,birth,origin,direction,2,options);
        cg::Random rng(absolute?818:732);
        int escaped=0;double sum=0,sum2=0;
        constexpr int count=6000;
        for(int i=0;i<count;++i) {
            double hit;escaped+=flight.track(rng,true,hit)!=0;
            const double tr=flight.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/count,survival,6*std::sqrt(survival*(1-survival)/count),
            "configured delta tracking matches integrated hazard");
        const double variance=std::max(0.0,(sum2-sum*sum/count)/(count-1));
        close(sum/count,survival,6*std::sqrt(variance/count)+1e-4,
            "configured ratio tracking matches integrated hazard");
    }
    std::cout<<"tracking: carried optical depth, boundary events, draw reuse, JSON round trip, validation, "
        "actual interval sizes, deferred refinement, medium/shadow routing "
        "and fine/adaptive/cheap survival laws passed\n";
}
