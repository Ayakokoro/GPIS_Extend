#pragma once

inline void pointLipschitzBoundChecks()
{
    using namespace Tungsten;
    const auto knob=std::dynamic_pointer_cast<GlobalConditionalMedium>(knobMedium("global_conditional",
        R"({"type":"procedural","func":"knob","scale":-0.7,"offset":0.1,"min":-10,
        "transform":{"position":[0.1,0.4,-0.2],"scale":[1.2,0.8,1.3]}})"));
    const std::vector<std::pair<std::shared_ptr<const cg::MeanField>,double>> providers{
        {field().mean,1}, {std::make_shared<cg::ConstantMean>(0),0},
        {std::make_shared<cg::SphereMean>(V::Zero(),1),1}, {knob->field().mean,10}};
    int checked=0;
    for(const auto &provider:providers) for(double sigma:{0.02,0.7}) for(double scale:{0.1,10.0})
        for(double f0:{0.0,0.3}) for(double slope:{0.02,0.7,6.0}) for(double tailCutoff:{0.0,4.0}) {
            auto f=field();f.mean=provider.first;
            f.kernel=cg::CovarianceKernel(sigma,scale*f.kernel.precision());
            f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            cg::FlightState birth;
            birth.birthPosition=V(0.2,-0.1,2);
            birth.direction=V(0.2,0.1,-1).normalized();
            const double ell=1/std::sqrt(birth.direction.dot(f.kernel.precision()*birth.direction));
            birth.birthValue=f0*sigma;
            birth.birthGradient=birth.direction*(slope*sigma/ell);
            const cg::ConditionalFlightKernel kernel(f,birth,false,tailCutoff);
            for(double start:{0.001,0.1,1.0,4.0,20.0}) for(double width:{0.2,2.0}) {
                const double lo=start*ell,hi=lo+width*ell;
                const double bound=kernel.pointIntervalMajorant(lo,hi,provider.second);
                check(bound>=0 && !std::isnan(bound),"point/Lipschitz returns a nonnegative certificate");
                if(start>=1) check(std::isfinite(bound),"ordinary point certificate is finite");
                for(int j=0;j<=16;++j) {
                    const double hazard=kernel.evaluate(lo+(hi-lo)*j/16).hazard.value;
                    check(hazard<=bound*(1+1e-10),"point/Lipschitz certificate covers actual conditional hazard");
                    ++checked;
                }
            }
        }
    auto f=field();
    auto counted=std::make_shared<CountingBoundMean>(f.mean);f.mean=counted;
    cg::FlightState birth;birth.birthValue=.3;birth.birthGradient=V::UnitZ();
    const cg::ConditionalFlightKernel kernel(f,birth);
    counted->pointQueries=counted->rayQueries=0;
    kernel.pointIntervalMajorant(1,1.3,1);
    check(counted->pointQueries==1 && counted->rayQueries==0,
        "point certificate uses one midpoint query without a spatial interval query");
    const double parent=kernel.intervalMajorant(1,1.3);
    check(kernel.pointIntervalMajorant(1,1.3,1,parent)<=parent,
        "point certificate retains an inherited valid parent bound");
    check(std::isinf(kernel.pointIntervalMajorant(1,4,std::numeric_limits<double>::max())),
        "overflowing cheap spatial enclosure requests fine fallback");
    std::cout<<"point/Lipschitz: "<<checked<<" coverage points, transformed CSG, anisotropic kernels, "
             <<"zero/positive births, tail envelopes and query routing passed\n";
}

inline void pointLipschitzTrackingChecks()
{
    auto f=field();
    auto counted=std::make_shared<CountingBoundMean>(std::make_shared<cg::ConstantMean>(10));
    f.mean=counted;f.kernel=cg::CovarianceKernel(.02,M::Identity()/(.02*.02));
    cg::FlightState birth;birth.birthValue=10;
    cg::ConditionalTrackingSettings options;
    options.fixedStepLimit=false;options.ageScale=.5;
    options.geometryStep=options.pointBounds=true;
    options.classicStartScale=1;
    const cg::ConditionalSegment empty(f,birth,V(0,0,1),V::UnitZ(),1,options,nullptr,1);
    counted->pointQueries=counted->rayQueries=0;
    cg::Random emptyRng(271);cg::ConditionalTrackingWork work;double hit;
    close(empty.track(emptyRng,false,hit,&work),1,0,"geometric empty flight survives");
    check(work.intervals==1 && counted->rayQueries==0,
        "geometric proposal crosses empty space with a point certificate");
    auto fineOptions=options;fineOptions.tightenExpectedEvents=0;
    const cg::ConditionalSegment fine(f,birth,V(0,0,1),V::UnitZ(),1,fineOptions,nullptr,1);
    counted->rayQueries=0;
    fine.track(emptyRng,false,hit);
    check(counted->rayQueries>0,"zero threshold forces a full spatial certificate fallback");
    bool rejected=false;
    try { cg::ConditionalSegment missing(f,birth,V::Zero(),V::UnitZ(),1,options); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"accelerated tracking requires a supplied Lipschitz certificate");

    for(bool constant:{false,true}) for(double f0:{0.0,0.3}) {
        auto target=field();
        if(constant) target.mean=std::make_shared<cg::ConstantMean>(0);
        const V w=V(0.2,0.1,-1).normalized();
        const double ell=1/std::sqrt(w.dot(target.kernel.precision()*w)),length=4*ell;
        cg::FlightState initial;initial.birthPosition=V(.2,-.1,1);
        initial.birthValue=f0*target.kernel.sigma();
        initial.birthGradient=w*(.6*target.kernel.sigma()/ell);
        initial.direction=w;
        const cg::ConditionalFlightKernel reference(target,initial);
        double integral=0;
        for(int i=0;i<12000;++i)
            integral+=reference.evaluate(length*(i+.5)/12000).hazard.value*length/12000;
        const double survival=std::exp(-integral);
        for(int mode=0;mode<3;++mode) {
            auto settings=options;
            settings.geometryStep=mode!=1;settings.pointBounds=mode!=0;
            settings.classicStartScale=.3;settings.maxExpectedEvents=2;
            settings.tightenExpectedEvents=1;
            const cg::ConditionalSegment flight(target,initial,initial.birthPosition,w,length,
                settings,nullptr,constant ? 0 : 1);
            cg::Random rng(7349+mode+int(100*f0)+1000*constant);
            constexpr int count=4000;
            int escaped=0;double sum=0,sum2=0;
            for(int i=0;i<count;++i) {
                escaped+=flight.track(rng,true,hit)!=0;
                const double tr=flight.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
            }
            close(double(escaped)/count,survival,6*std::sqrt(survival*(1-survival)/count)+1e-4,
                "geometric and point tracking preserve delta survival vs independent hazard integration");
            const double variance=std::max(0.0,(sum2-sum*sum/count)/(count-1));
            close(sum/count,survival,6*std::sqrt(variance/count)+1e-4,
                "geometric and point tracking preserve ratio survival vs independent hazard integration");
        }
    }
    std::cout<<"point/Lipschitz tracking: geometric proposals, fine fallback, delta/ratio survival passed\n";
}
