#pragma once

inline void globalFarChecks()
{
    class RecordingConstant final : public cg::MeanField {
    public:
        mutable int queries=0,boundaries=0,globalQueries=0;
        const char *typeName() const override { return "global-far-recording"; }
        cg::MeanJet evaluate(const V &) const override { return {0,V::Zero()}; }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {0,0,true}; }
        std::optional<V> affineGradient() const override { return V::Zero(); }
        std::optional<cg::MeanMagnitudeBounds> globalMagnitudeBounds() const override {
            ++globalQueries;return cg::MeanMagnitudeBounds{0,0}; // legacy symmetric provider
        }
        void appendRayBreakpoints(const V &,const V &,double,double,std::vector<double> &) const override {
            ++boundaries;
        }
        cg::MeanRayBounds rayBounds(const V &o,const V &w,double lo,double hi,
                                    const Tungsten::SdfRayQueryOptions &options={}) const override {
            ++queries;return MeanField::rayBounds(o,w,lo,hi,options);
        }
    };
    // Independent point hazards cover anisotropy, both birth types, and large
    // observed gradients. Signed value bounds must work for either sign.
    int checked=0;
    for(double value:{-0.7,0.0,0.7}) for(double f0:{0.0,0.2})
        for(double slope:{1e-4,0.1,2.0,20.0}) for(double q:{2.0,4.0,8.0,12.0}) {
            auto f=field();f.mean=std::make_shared<cg::ConstantMean>(value);
            cg::FlightState birth;birth.direction=V(0.3,-0.2,1).normalized();
            birth.birthValue=f0;birth.birthGradient=slope*birth.direction;
            cg::ConditionalFlightKernel kernel(f,birth);
            const double ell=1/std::sqrt(birth.direction.dot(f.kernel.precision()*birth.direction));
            const double maximum=kernel.globalFarMajorant(q);
            for(int i=0;i<=100;++i) {
                const double t=(q+20.0*i/100)*ell;
                check(kernel.evaluate(t).hazard.value<=maximum*(1+1e-10),"global SE far certificate covers conditional hazard");
                ++checked;
            }
        }
    auto f=field();
    auto mean=std::make_shared<RecordingConstant>();f.mean=mean;
    f.kernel=cg::CovarianceKernel(1,M::Identity());
    cg::FlightState birth;birth.birthGradient=V::UnitZ();
    cg::ConditionalTrackingSettings settings;settings.globalFarStartScale=2;
    // Even absurdly small near-field controls cannot subdivide a global tail.
    settings.step=1e-12;settings.maxExpectedEvents=1e-12;settings.macrocellSize=8;
    cg::ConditionalSegment farFlight(f,birth,2.1*V::UnitZ(),V::UnitZ(),0.6,settings);
    mean->queries=mean->boundaries=mean->globalQueries=0;
    for(bool collision:{false,true}) {
        OpticalDepthSampler sampler({std::numeric_limits<float>::min()});cg::Random rng(sampler);
        cg::ConditionalTrackingWork work;double hit;
        close(farFlight.track(rng,collision,hit,&work),1,0,"far fixture survives without a candidate");
        check(work.intervals==1 && hit==0.6,"resumed global tail skips all near proposal limits");
    }
    check(mean->queries==0 && mean->boundaries==0 && mean->globalQueries==1,
        "global tail caches one bound and performs no spatial interval or boundary queries");
    settings={};settings.globalFarStartScale=2;
    cg::Random rng(84126);
    // Compare delta/ratio estimates to separately integrated exact hazards.
    // Starts cover crossing the switch and resuming on either side of it.
    for(double start:{0.0,1.8,2.2}) {
        cg::ConditionalSegment segment(f,birth,start*V::UnitZ(),V::UnitZ(),3,settings);
        cg::ConditionalFlightKernel exact(f,birth);
        double integral=0;
        for(int i=0;i<24000;++i) integral+=exact.evaluate(start+3.0*(i+0.5)/24000).hazard.value*3/24000;
        const double expected=std::exp(-integral);
        constexpr int n=6000;
        int escaped=0;double sum=0,sum2=0,hit;
        for(int i=0;i<n;++i) {
            escaped+=segment.track(rng,true,hit)!=0;
            const double tr=segment.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/n,expected,6*std::sqrt(expected*(1-expected)/n)+1e-4,"global far delta survival");
        const double variance=std::max(0.0,(sum2-sum*sum/n)/(n-1));
        close(sum/n,expected,6*std::sqrt(variance/n)+1e-4,"global far ratio survival");
        const double bound=segment.majorant(0.1,3);
        for(int i=1;i<=300;++i)
            check(segment.hazard(0.1+2.9*i/300)<=bound*(1+1e-10),"public majorant covers intervals crossing far switch");
    }
    // An unreachable switch must leave random-number use and the near path exact.
    settings.globalFarStartScale=4;
    cg::ConditionalSegment enabled(f,birth,V::Zero(),V::UnitZ(),1,settings);
    settings.globalFarStartScale=0;
    cg::ConditionalSegment disabled(f,birth,V::Zero(),V::UnitZ(),1,settings);
    for(bool collision:{false,true}) for(int i=0;i<100;++i) {
        cg::Random a(1000+i),b(1000+i);double ta,tb;
        const double va=enabled.track(a,collision,ta),vb=disabled.track(b,collision,tb);
        check(va==vb && ta==tb && a.uniform01()==b.uniform01(),"near-only flight unchanged including RNG state");
    }
    settings.globalFarStartScale=4;
    cg::ConditionalSegment empty(f,birth,V::Zero(),V::UnitZ(),0,settings);
    double hit;cg::ConditionalTrackingWork work;
    check(empty.track(rng,false,hit,&work)==1 && hit==0 && work.intervals==0,"empty hybrid flight");
    bool rejected=false;
    try { cg::ConditionalFlightKernel(field(),birth).globalFarMajorant(4); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"unbounded mean rejects global far certificate");
    for(double q:{0.,1.99,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        rejected=false;
        try { cg::ConditionalFlightKernel(f,birth).globalFarMajorant(q); }
        catch(const cg::NumericError &) { rejected=true; }
        check(rejected,"invalid direct far cutoff rejected");
    }
    std::cout<<"global far: "<<checked<<" hazard checks, cached spatial-query-free tail, near parity and delta/ratio survival passed\n";
}
