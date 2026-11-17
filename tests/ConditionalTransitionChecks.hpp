#pragma once

inline void transitionPointChecks()
{
    const std::vector<std::pair<std::shared_ptr<const cg::MeanField>,double>> providers{
        {field().mean,2}, {std::make_shared<cg::ConstantMean>(-.2),0},
        {std::make_shared<cg::SphereMean>(V::Zero(),1),1}};
    int checks=0;
    for(const auto &provider:providers) for(double sigma:{.04,.7}) for(double scale:{.1,10.})
        for(double f0:{0.,.2}) for(double slope:{1e-4,.3,8.}) {
            auto f=field();f.mean=provider.first;f.kernel=cg::CovarianceKernel(sigma,scale*f.kernel.precision());
            f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            cg::FlightState birth;birth.birthPosition=V(.2,-.1,.8);birth.direction=V(.2,.1,-1).normalized();
            const double ell=1/std::sqrt(birth.direction.dot(f.kernel.precision()*birth.direction));
            birth.birthValue=f0*sigma;birth.birthGradient=birth.direction*(slope*sigma/ell);
            const cg::ConditionalFlightKernel kernel(f,birth,false,2,2),reference(f,birth);
            for(double q:{1e-5,.001,.02,.1,.5,1.,1.9,2.1,3.9}) for(double width:{.01,.25,2.}) {
                const double lo=q*ell,hi=lo*(1+width);
                const double bound=kernel.transitionIntervalMajorant(lo,hi,provider.second);
                for(int j=0;j<=12;++j) {
                    check(reference.evaluate(lo+(hi-lo)*j/12).hazard.value<=bound*(1+1e-10),
                          "transition midpoint encloses independent conditional hazard including grazing and tiny ages");
                    ++checks;
                }
            }
        }
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(-.5);f.kernel=cg::CovarianceKernel(.2,M::Identity());
    cg::FlightState birth;birth.birthGradient=.2*V::UnitZ();
    const cg::ConditionalFlightKernel cancellation(f,birth,false,2,2);
    check(cancellation.transitionIntervalMajorant(1e-4,1.1e-4,0)<1e-100,
          "transition bound preserves cancellation of negative prior mean and positive birth correction");

    f.mean=std::make_shared<cg::ConstantMean>(.02);
    cg::Random rng(90371);
    for(bool positive:{false,true}) for(double start:{0.,.03,3.8}) {
        birth.birthValue=positive ? .1:0;
        cg::ConditionalTrackingSettings options;
        options.transitionPointBounds=true;options.transitionMaxExpectedEvents=.1;options.localFarStartScale=4;
        const double length=start<1 ? 3:1;
        const cg::ConditionalSegment segment(f,birth,start*V::UnitZ(),V::UnitZ(),length,options,nullptr,0,0);
        const cg::ConditionalFlightKernel reference(f,birth);
        double integral=0;
        for(int i=0;i<24000;++i) integral+=reference.evaluate(start+length*(i+.5)/24000).hazard.value*length/24000;
        const double expected=std::exp(-integral);
        constexpr int count=4000;
        int escaped=0;double sum=0,sum2=0,hit;
        for(int i=0;i<count;++i) {
            escaped+=segment.track(rng,true,hit)!=0;
            const double tr=segment.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/count,expected,6*std::sqrt(expected*(1-expected)/count)+1e-4,
              "transition delta survival across birth, resumption, refinement and far switch");
        const double variance=std::max(0.,(sum2-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+1e-4,"transition ratio survival");
    }
    birth.birthValue=0;
    cg::ConditionalTrackingSettings enabled;enabled.transitionPointBounds=true;enabled.localFarStartScale=4;
    auto disabled=enabled;disabled.transitionPointBounds=false;
    const cg::ConditionalSegment shortNew(f,birth,V::Zero(),V::UnitZ(),.01,enabled,nullptr,0,0);
    const cg::ConditionalSegment shortOld(f,birth,V::Zero(),V::UnitZ(),.01,disabled,nullptr,0,0);
    for(bool collision:{false,true}) for(int i=0;i<100;++i) {
        cg::Random a(100+i),b(100+i);double ta,tb;
        const double va=shortNew.track(a,collision,ta),vb=shortOld.track(b,collision,tb);
        check(va==vb && ta==tb && a.uniform01()==b.uniform01(),"transition mode preserves certified birth-only samples");
    }
    bool rejected=false;
    try { cg::ConditionalSegment missing(f,birth,V::Zero(),V::UnitZ(),3,enabled); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"transition requires a finite Lipschitz certificate");
    std::cout<<"transition midpoint: "<<checks<<" hazard checks, signed cancellation, birth parity and delta/ratio survival passed\n";
}
