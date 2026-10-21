#pragma once

inline void rayConstantChecks()
{
    using Mode=cg::ConditionalMajorantMode;
    class RecordingConstant final : public cg::MeanField {
    public:
        mutable int queries=0;
        const char *typeName() const override { return "recording-constant"; }
        cg::MeanJet evaluate(const V &) const override { return {0,V::Zero()}; }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {0,0,true}; }
        std::optional<cg::MeanMagnitudeBounds> globalMagnitudeBounds() const override {
            return cg::MeanMagnitudeBounds{0,0};
        }
        std::optional<V> affineGradient() const override { return V::Zero(); }
        cg::MeanRayBounds rayBounds(const V &o,const V &w,double lo,double hi,
                                    const Tungsten::SdfRayQueryOptions &options={}) const override {
            ++queries;
            return MeanField::rayBounds(o,w,lo,hi,options);
        }
    };
    auto f=field();
    auto mean=std::make_shared<RecordingConstant>();
    f.mean=mean;
    f.kernel=cg::CovarianceKernel(1,M::Identity());
    cg::ConditionalTrackingSettings options;
    options.majorantMode=Mode::TwoConstants;
    // These adaptive controls must have no effect in the two-constant mode.
    options.step=1e-10;options.maxExpectedEvents=1e-20;
    cg::FlightState birth;birth.birthGradient=V::UnitZ();
    const cg::ConditionalFlightKernel kernel(f,birth);
    const auto bound=kernel.birthMajorant(2);
    const double tail=kernel.constantTailMajorant(bound.split);
    check(mean->queries==1,"constant tail does not query any mean intervals");
    const cg::ConditionalSegment flight(f,birth,V::Zero(),V::UnitZ(),1.5,options);
    check(flight.majorant(0,bound.split)==bound.maximum &&
        flight.majorant(bound.split,1.5)==tail &&
        flight.majorant(0,1.5)==std::max(bound.maximum,tail),"two constant interval coverage");
    mean->queries=0;
    for(bool collision : {false,true}) {
        cg::Random rng(8481);
        cg::ConditionalTrackingWork work;
        double hit;
        flight.track(rng,collision,hit,&work);
        check(work.intervals<=2 && work.candidates>0,"two constants bypass refinement and step controls");
        check(work.distance>0 && work.distance<=1.5,"tracking work records travelled distance");
    }
    check(mean->queries==0,"neither delta nor ratio tracking queries a tail interval");
    const cg::ConditionalSegment empty(f,birth,V::Zero(),V::UnitZ(),0,options);
    cg::Random rng(931);
    cg::ConditionalTrackingWork emptyWork;double hit;
    check(empty.track(rng,false,hit,&emptyWork)==1 && emptyWork.intervals==0 &&
        emptyWork.candidates==0 && hit==0,"empty two-constant flight");
    for(double requested : {0.15,0.9,3.0}) {
        auto fixed=options;fixed.constantBirthLength=requested;
        const cg::ConditionalSegment extended(f,birth,V::Zero(),V::UnitZ(),1.5,fixed);
        const double split=std::min(requested,1.5);
        const double first=extended.majorant(0,split);
        const double last=split<1.5 ? extended.majorant(split,1.5) : first;
        mean->queries=0;
        for(int i=1;i<=2000;++i) {
            const double t=1.5*i/2000;
            check(extended.hazard(t)<=(t<=split ? first:last)*(1+1e-10),"extended constant prefix covers actual hazard");
        }
        for(bool collision : {false,true}) {
            cg::ConditionalTrackingWork work;
            extended.track(rng,collision,hit,&work);
            check(work.intervals<=(requested>=1.5 ? 1u:2u),"fixed split clips to ray and never refines during sampling");
        }
        check(mean->queries==0,"extended prefix does all interval queries before sampling");
        const cg::ConditionalSegment zero(f,birth,V::Zero(),V::UnitZ(),0,fixed);
        check(zero.track(rng,false,hit)==1 && hit==0,"fixed birth length supports empty rays");
    }

    // Test the actual hazard independently of the majorant formula: both birth
    // observations, anisotropic precision, and distances far below ell.
    for(double meanValue : {-0.7,0.0,0.7}) for(double birthValue : {0.0,0.2})
        for(double slope : {1e-4,0.1,2.0}) {
            auto varied=field();varied.mean=std::make_shared<cg::ConstantMean>(meanValue);
            auto state=birth;state.birthValue=birthValue;
            state.direction=V(0.3,-0.2,1).normalized();
            state.birthGradient=slope*state.direction;
            cg::ConditionalFlightKernel k(varied,state);
            const auto b=k.birthMajorant(4);
            const double maximum=k.constantTailMajorant(b.split);
            for(int i=0;i<=600;++i) {
                const double near=b.split*std::pow(10.0,-8.0+8.0*i/600);
                const double far=b.split*std::pow(4/b.split,double(i)/600);
                check(k.evaluate(near).hazard.value<=b.maximum*(1+1e-10),"constant birth covers tiny ages");
                check(k.evaluate(far).hazard.value<=maximum*(1+1e-10),"global constant covers anisotropic tail");
            }
            const double tiny=1e-10;
            const double tinyMaximum=k.constantTailMajorant(tiny);
            check(std::isfinite(tinyMaximum) && tinyMaximum>=k.evaluate(tiny).hazard.value,
                "tail variance stays positive when direct subtraction would cancel");
        }

    // Delta and ratio tracking must agree with independent numerical quadrature,
    // including a segment resumed beyond the single majorant switch.
    for(double requested : {0.0,0.9,3.0}) for(double start : {0.0,0.8}) {
        auto fixed=options;fixed.constantBirthLength=requested;
        const cg::ConditionalSegment segment(f,birth,start*V::UnitZ(),V::UnitZ(),1.5,fixed);
        double opticalDepth=0;
        for(int i=0;i<20000;++i) opticalDepth+=segment.hazard(1.5*(i+0.5)/20000)*1.5/20000;
        const double expected=std::exp(-opticalDepth);
        int escaped=0;double sum=0,sum2=0;
        constexpr int n=4000;
        for(int i=0;i<n;++i) {
            cg::ConditionalTrackingWork work;
            escaped+=segment.track(rng,true,hit,&work)!=0;
            const double split=requested>0 ? std::min(requested,start+1.5):bound.split;
            check(work.intervals<=(split<=start || split>=start+1.5 ? 1u:2u),"resumed flight uses its actual switch position");
            const double tr=segment.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/n,expected,6*std::sqrt(expected*(1-expected)/n),"two-constant delta survival");
        const double variance=std::max(0.0,(sum2-sum*sum/n)/(n-1));
        close(sum/n,expected,6*std::sqrt(variance/n)+1e-4,"two-constant ratio survival");
    }
    bool rejected=false;
    auto outsideOptions=options;outsideOptions.constantBirthLength=0.9;
    try { cg::ConditionalSegment outside(f,birth,V::Zero(),V::UnitZ(),101,outsideOptions); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"fixed prefix still validates the complete ray against the field domain");
    rejected=false;
    try { cg::ConditionalFlightKernel(field(),birth).constantTailMajorant(0.1); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"unbounded analytic SDF cannot silently use a finite global bound");
    std::cout<<"two_constants: global hazard coverage, stable tiny variance, no tail queries, "
        "two intervals and delta/ratio survival passed\n";
}
