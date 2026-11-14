#pragma once

inline void simpleBoundChecks()
{
    using namespace Tungsten;
    const auto knob=std::dynamic_pointer_cast<GlobalConditionalMedium>(knobMedium("global_conditional",
        R"({"type":"procedural","func":"knob","scale":-0.7,"offset":0.1,
        "transform":{"position":[0.1,0.4,-0.2],"scale":[1.2,0.8,1.3]}})"));
    const std::vector<std::pair<std::shared_ptr<const cg::MeanField>,double>> providers{
        {field().mean,2}, {std::make_shared<cg::ConstantMean>(-0.2),0},
        {std::make_shared<cg::SphereMean>(V::Zero(),1),1}, {knob->field().mean,10}};
    int checked=0;
    for(const auto &provider:providers) for(double sigma:{0.02,0.7}) for(double scale:{0.1,10.0})
        for(double f0:{0.0,0.3}) for(double slope:{0.02,1.0,8.0}) for(double cutoff:{2.0,4.0,6.0,16.0}) {
            auto f=field();f.mean=provider.first;
            f.kernel=cg::CovarianceKernel(sigma,scale*f.kernel.precision());
            f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            cg::FlightState birth;birth.birthPosition=V(.2,-.1,.8);
            birth.direction=V(.2,.1,-1).normalized();birth.birthValue=f0*sigma;
            const double ell=1/std::sqrt(birth.direction.dot(f.kernel.precision()*birth.direction));
            birth.birthGradient=birth.direction*(slope*sigma/ell);
            const cg::ConditionalFlightKernel kernel(f,birth,false,0,cutoff),reference(f,birth);
            for(double offset:{0.0,4.0}) for(double width:{0.25,2.0}) {
                const double lo=kernel.simpleBoundBegin()+offset*ell,hi=lo+width*ell;
                const double bound=kernel.simpleIntervalMajorant(lo,hi,provider.second);
                check(std::isfinite(bound) && bound>=0,"simple SE certificate is finite and nonnegative");
                for(int j=0;j<=16;++j) {
                    const double t=lo+(hi-lo)*j/16,hazard=reference.evaluate(t).hazard.value;
                    check(hazard<=bound*(1+1e-10),"simple SE certificate covers the original conditional hazard");
                    close(kernel.evaluate(t).hazard.value,hazard,0,"simple option preserves exact pointwise hazard");
                    ++checked;
                }
            }
        }
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(0);
    f.kernel=cg::CovarianceKernel(1,M::Identity());
    cg::FlightState birth;birth.birthGradient=V::UnitZ();
    auto counted=std::make_shared<CountingBoundMean>(f.mean);f.mean=counted;
    const cg::ConditionalFlightKernel kernel(f,birth,false,0,4),reference(f,birth);
    counted->pointQueries=counted->rayQueries=0;
    check(kernel.simpleIntervalMajorant(4.1,4.3,0)>1/cg::kPi,
        "simple bound keeps the nonzero tail correction at four lengths");
    check(counted->pointQueries==1 && counted->rayQueries==0,"simple bound uses one point and no ray interval query");
    const double parent=reference.intervalMajorant(4.1,4.3);
    check(kernel.simpleIntervalMajorant(4.1,4.3,0,parent)<=parent,"simple bound retains a valid parent certificate");
    check(std::isinf(kernel.simpleIntervalMajorant(4.1,8,std::numeric_limits<double>::max())),
        "overflowing simple cone requests certified fallback");
    bool rejected=false;
    try {kernel.simpleIntervalMajorant(3.9,4.3,0);}catch(const cg::NumericError &){rejected=true;}
    check(rejected,"simple method cannot cover an interval starting before its cutoff");
    for(double invalid:{-1.0,1.99,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        rejected=false;
        try {cg::ConditionalFlightKernel bad(f,birth,false,0,invalid);}catch(const cg::NumericError &){rejected=true;}
        check(rejected,"simple kernel validates direct cutoffs");
    }
    std::cout<<"simple SE bound: "<<checked<<" coverage points, anisotropic kernels, transformed CSG, "
        "negative means, cutoff edges, exact hazard parity and point-query routing passed\n";
}

inline void simpleBoundTrackingChecks()
{
    for(bool positive:{false,true}) {
        auto f=field();f.mean=std::make_shared<cg::ConstantMean>(0.02);
        f.kernel=cg::CovarianceKernel(.2,M::Identity());
        cg::FlightState birth;birth.birthValue=positive ? .1 : 0;birth.birthGradient=V::UnitZ()*.2;
        const cg::ConditionalFlightKernel reference(f,birth);
        cg::ConditionalTrackingSettings options;options.fixedStepLimit=false;options.ageScale=.5;
        options.simpleBoundStartScale=2;options.tightenExpectedEvents=1;options.maxExpectedEvents=2;
        const V origin=V::UnitZ()*2.1;
        constexpr double length=3;
        const cg::ConditionalSegment flight(f,birth,origin,V::UnitZ(),length,options,nullptr,0);
        double integral=0;
        for(int i=0;i<12000;++i)integral+=reference.evaluate(2.1+length*(i+.5)/12000).hazard.value*length/12000;
        const double expected=std::exp(-integral);
        cg::Random rng(8429+positive);double sum=0,sum2=0,hit;int escaped=0;
        constexpr int count=8000;
        for(int i=0;i<count;++i) {
            escaped+=flight.track(rng,true,hit)!=0;
            const double tr=flight.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/count,expected,6*std::sqrt(expected*(1-expected)/count)+1e-4,
            "simple-bound delta survival agrees with independent hazard integration");
        const double variance=std::max(0.0,(sum2-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+1e-4,
            "simple-bound ratio survival agrees with independent hazard integration");
    }
    auto f=field();auto counted=std::make_shared<CountingBoundMean>(std::make_shared<cg::ConstantMean>(10));
    f.mean=counted;f.kernel=cg::CovarianceKernel(.02,M::Identity()/(.02*.02));
    cg::FlightState birth;birth.birthValue=10;
    cg::ConditionalTrackingSettings options;options.simpleBoundStartScale=4;
    const cg::ConditionalSegment flight(f,birth,V(0,0,1),V::UnitZ(),.1,options,nullptr,1);
    counted->rayQueries=counted->pointQueries=0;
    flight.majorant(0,.1);
    check(counted->rayQueries==0 && counted->pointQueries==1,"segment routes eligible bounds through the simple path");
    options.tightenExpectedEvents=0;
    const cg::ConditionalSegment fine(f,birth,V(0,0,1),V::UnitZ(),.1,options,nullptr,1);
    counted->rayQueries=0;fine.majorant(0,.1);
    check(counted->rayQueries>0,"zero tightening threshold retains the spatial fine fallback");
    bool rejected=false;
    try {cg::ConditionalSegment missing(f,birth,V(0,0,1),V::UnitZ(),.1,options);}catch(const cg::NumericError &){rejected=true;}
    check(rejected,"simple segment requires a supplied mean Lipschitz certificate");
    std::cout<<"simple SE tracking: retained birth age, fine fallback and independent delta/ratio survival passed\n";
}
