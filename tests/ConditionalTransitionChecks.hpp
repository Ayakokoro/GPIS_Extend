#pragma once

inline void coupledTransitionChecks()
{
    const auto polynomialMedium=makeMedium(R"("tracking_coupled_bounds":true,"tracking_transition_max_expected_events":0,"tracking_transition_voxel_bound":"bernstein5")");
    check(polynomialMedium->trackingSettings().transitionVoxelBound==cg::TransitionVoxelBound::Bernstein5,
          "degree-five voxel option is parsed");
    rapidjson::Document polynomialDocument;
    const auto polynomialJson=polynomialMedium->toJson(polynomialDocument.GetAllocator());
    check(std::string(polynomialJson["tracking_transition_voxel_bound"].GetString())=="bernstein5",
          "degree-five voxel option survives serialization");
    class CornerMean final : public cg::MeanField {
    public:
        const char *typeName() const override { return "absolute-value-test"; }
        cg::MeanJet evaluate(const cg::Point3 &x) const override { return {std::abs(x.z()),V(0,0,x.z()<0 ? -1:1)}; }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {0,1,true}; }
    };
    for(double f0:{.01,.2,2.}) for(double slope:{-1.,0.,1.}) {
        auto f=field();f.mean=std::make_shared<CornerMean>();f.kernel=cg::CovarianceKernel(.2,M::Identity());
        cg::FlightState birth;birth.birthPosition=V(0,0,.0001);birth.direction=-V::UnitZ();
        birth.birthValue=f0;birth.birthGradient=slope*birth.direction;
        cg::ConditionalFlightKernel kernel(f,birth);const auto prefix=kernel.birthMajorant(1,1);
        check(prefix.split>.0001,"positive certificate crosses a nonsmooth mean corner");
        for(int i=1;i<=2048;++i)check(kernel.evaluate(prefix.split*i/2048).hazard.value<=prefix.maximum*(1+1e-10),
            "positive certificate covers a nonsmooth mean without curvature");
    }
    // Exact adjacent values, including subnormals, signed zeros and infinities.
    std::mt19937_64 bits(90218);
    for(int i=0;i<200000;++i) {
        std::uint64_t word=bits();double x;std::memcpy(&x,&word,sizeof x);
        if(i<8) x=std::array<double,8>{0.,-0.,std::numeric_limits<double>::denorm_min(),
            -std::numeric_limits<double>::denorm_min(),std::numeric_limits<double>::max(),
            -std::numeric_limits<double>::max(),INFINITY,-INFINITY}[i];
        if(std::isnan(x))continue;
        for(bool upper:{false,true}) {
            const double actual=upper ? cg::MajorantBounds::up(x) : cg::MajorantBounds::down(x);
            const double expected=std::nextafter(x,upper ? INFINITY : -INFINITY);
            check(actual==expected && std::signbit(actual)==std::signbit(expected),"inline outward rounding equals nextafter");
        }
    }
    int covered=0;
    for(double z:{-20.,-1.,0.,.01,1.,5.,20.}) for(double beta:{0.,1e-10,.1,10.,1000.}) {
        cg::ConditionalIntegralMajorant curve;curve.gaussianLinear=true;
        curve.constant=2.3;curve.z0=z;curve.beta=beta;curve.origin=.1;
        const double a=.1,b=a+std::min(10.,30./std::max(beta,1e-30));
        const double depth=curve.integral(a,b);double sum=0;constexpr int n=16000;
        for(int i=0;i<=n;++i)sum+=(i==0||i==n ? 1:i%2 ? 4:2)*curve.value(a+(b-a)*i/n);
        close(depth,sum*(b-a)/(3*n),2e-6*std::max(1.,depth),"Gaussian/linear primitive vs independent quadrature");
        for(double fraction:{1e-6,.01,.5,.99})if(depth>1e-250) {
            const double t=curve.inverseIntegral(a,b,depth*fraction);
            close(curve.integral(a,t),depth*fraction,1e-9*depth,"Gaussian/linear inverse optical depth");
        }
    }
    for(auto provider:std::vector<std::pair<std::shared_ptr<const cg::MeanField>,double>>{
            {field().mean,2},{std::make_shared<cg::ConstantMean>(-.2),0},
            {std::make_shared<cg::SphereMean>(V::Zero(),1),1}})
        for(bool tabulated:{false,true}) for(double sigma:{.04,.7}) for(double f0:{0.,.2}) for(double slope:{1e-4,.3,8.}) {
            auto f=field();f.mean=provider.first;f.kernel=cg::CovarianceKernel(sigma,f.kernel.precision());
            f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            cg::FlightState birth;birth.birthPosition=V(.2,-.1,.8);birth.direction=V(.2,.1,-1).normalized();
            const double ell=1/std::sqrt(birth.direction.dot(f.kernel.precision()*birth.direction));
            birth.birthValue=f0*sigma;birth.birthGradient=birth.direction*(slope*sigma/ell);
            const cg::ConditionalFlightKernel kernel(f,birth,false,0,0,tabulated),reference(f,birth);
            if(f0>0) {
                const auto prefix=kernel.birthMajorant(4*ell,provider.second);
                for(int j=0;j<256;++j) {
                    const double t=prefix.split*std::pow(10.,-8+8*(j+.5)/256);
                    check(reference.evaluate(t).hazard.value<=prefix.maximum*(1+1e-10),"positive birth certificate covers across cells");++covered;
                }
            }
            for(double q:{1e-6,.001,.02,.1,.5,1.,1.9,2.1,3.9}) for(double width:{.05,.5,2.}) {
                const double lo=q*ell,hi=lo*(1+width);
                for(bool local:{false,true}) {
                    const double bound=kernel.coupledIntervalMajorant(lo,hi,provider.second,local);
                    for(int j=0;j<=16;++j) {
                        check(reference.evaluate(lo+(hi-lo)*j/16).hazard.value<=bound*(1+1e-10),
                              "coupled interval covers conditional hazard including grazing births");++covered;
                    }
                }
            }
            // Seek forwards and backwards: cached tail factors must cover the
            // earliest requested age, even if the next query moves backwards.
            for(double q:{4.,8.,2.000000001,16.,3.}) {
                const double lo=q*ell,hi=(q+1)*ell;
                const double bound=kernel.localFarIntervalMajorant(lo,hi,provider.second,provider.second,
                    std::numeric_limits<double>::infinity(),true);
                const auto curve=kernel.farIntegralMajorant(lo,hi,provider.second,provider.second);
                for(int j=0;j<=16;++j) {
                    check(reference.evaluate(lo+(hi-lo)*j/16).hazard.value<=bound*(1+1e-10),"cached far envelope covers backwards seeks");++covered;
                    check(reference.evaluate(lo+(hi-lo)*j/16).hazard.value<=curve.value(lo+(hi-lo)*j/16)*(1+1e-10),"Gaussian/linear tail covers the hazard");++covered;
                }
            }
        }
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(.02);f.kernel=cg::CovarianceKernel(.2,M::Identity());
    cg::FlightState birth;birth.birthGradient=.2*V::UnitZ();
    for(double f0:{0.,.1}) for(bool integralFar:{false,true}) for(double far:{0.,2.,4.}) for(double start:{0.,.03,3.8}) {
        birth.birthValue=f0;
        if(integralFar && far==0)continue;
        cg::ConditionalTrackingSettings options;options.coupledBounds=true;options.localFarStartScale=far;
        options.farIntegralBounds=integralFar;
        options.shadowRouletteThreshold=0;options.ageScale=.5;
        const double length=start<1 ? 3:1;
        cg::ConditionalSegment segment(f,birth,start*V::UnitZ(),V::UnitZ(),length,options,nullptr,0,0);
        cg::ConditionalFlightKernel reference(f,birth);double depth=0;
        for(int i=0;i<24000;++i)depth+=reference.evaluate(start+length*(i+.5)/24000).hazard.value*length/24000;
        const double expected=std::exp(-depth);constexpr int count=4000;
        cg::Random rng(73521);int escaped=0;double sum=0,sum2=0,hit;
        for(int i=0;i<count;++i) {
            escaped+=segment.track(rng,true,hit)!=0;
            const double tr=segment.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/count,expected,6*std::sqrt(expected*(1-expected)/count)+1e-4,"coupled delta survival");
        close(sum/count,expected,6*std::sqrt(std::max(0.,(sum2-sum*sum/count)/(count-1))/count)+1e-4,"coupled ratio survival");
    }
    std::cout<<"coupled transition: "<<covered<<" coverage checks, cached tails and delta/ratio survival passed\n";
}

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
                const auto curve=kernel.transitionIntegralMajorant(lo,hi,provider.second);
                for(int j=0;j<=12;++j) {
                    check(reference.evaluate(lo+(hi-lo)*j/12).hazard.value<=bound*(1+1e-10),
                          "transition midpoint encloses independent conditional hazard including grazing and tiny ages");
                    check(reference.evaluate(lo+(hi-lo)*j/12).hazard.value<=curve.value(lo+(hi-lo)*j/12)*(1+1e-10),
                          "integrated transition envelope covers conditional hazard");
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
    for(double threshold:{.1,0.}) for(bool positive:{false,true}) for(double start:{0.,.03,3.8}) {
        birth.birthValue=positive ? .1:0;
        cg::ConditionalTrackingSettings options;
        options.transitionPointBounds=true;options.transitionMaxExpectedEvents=threshold;options.localFarStartScale=4;
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

inline void transitionIntegralChecks()
{
    // Compare the primitive with independent quadrature of the rate, in log
    // age so tiny birth ages and long intervals both receive good resolution.
    for(double scale:{.01,1.,400.}) for(double start:{1e-8,.02,1.,8.}) {
        cg::ConditionalIntegralMajorant curve{.7,.4839414490383,scale};
        const double a=start/std::sqrt(scale),b=7*a;
        double sum=0;constexpr int n=4096;
        const double step=std::log(b/a)/n;
        for(int i=0;i<=n;++i) {
            const double t=a*std::exp(i*step);
            sum+=(i==0||i==n ? 1 : i%2 ? 4 : 2)*curve.value(t)*t;
        }
        const double exact=curve.integral(a,b),quadrature=sum*step/3;
        close(exact,quadrature,2e-10*std::max(1.,exact),"integrated envelope vs independent log-age quadrature");
        for(double fraction:{1e-6,.1,.7,.999999}) {
            const double t=curve.inverseIntegral(a,b,fraction*exact);
            check(t>a && t<b,"integral inverse stays inside interval");
            close(curve.integral(a,t),fraction*exact,1e-12*exact,"inverse integrated transition intensity");
        }
        const double mid=.5*(a+b);
        close(curve.integral(a,mid)+curve.integral(mid,b),exact,1e-13*exact,"primitive is additive across partitions");
    }
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(.02);
    f.kernel=cg::CovarianceKernel(.2,M::Identity());
    cg::FlightState birth;birth.birthGradient=.2*V::UnitZ();
    for(double threshold:{.1,0.}) for(bool positive:{false,true}) for(double start:{0.,.03,3.8}) {
        birth.birthValue=positive ? .1 : 0;
        cg::ConditionalTrackingSettings options;
        options.transitionIntegralBounds=true;options.transitionMaxExpectedEvents=threshold;
        options.localFarStartScale=4;options.shadowRouletteThreshold=0;
        const double length=start<1 ? 3 : 1;
        cg::ConditionalSegment segment(f,birth,start*V::UnitZ(),V::UnitZ(),length,options,nullptr,0,0);
        cg::ConditionalFlightKernel reference(f,birth);
        double integral=0;
        for(int i=0;i<24000;++i) integral+=reference.evaluate(start+length*(i+.5)/24000).hazard.value*length/24000;
        const double expected=std::exp(-integral);
        constexpr int count=5000;int escaped=0;double sum=0,sum2=0,hit;
        cg::Random rng(73521+int(start*100)+positive);
        for(int i=0;i<count;++i) {
            escaped+=segment.track(rng,true,hit)!=0;
            const double tr=segment.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/count,expected,6*std::sqrt(expected*(1-expected)/count)+1e-4,
              "integrated transition delta law across birth, refinement, continuation and far switch");
        const double variance=std::max(0.,(sum2-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+1e-4,"integrated transition ratio law");
    }
    const auto medium=makeMedium(R"("tracking_transition_integral_bounds":true,"tracking_local_far_start_scale":4)");
    check(medium->trackingSettings().transitionIntegralBounds,"integrated transition scene option is parsed");
    const auto noSplit=makeMedium(R"("tracking_transition_integral_bounds":true,"tracking_transition_max_expected_events":0)");
    check(noSplit->trackingSettings().transitionMaxExpectedEvents==0,"zero disables transition subdivision");
    rapidjson::Document serialized;
    auto json=noSplit->toJson(serialized.GetAllocator());
    check(json["tracking_transition_max_expected_events"].GetDouble()==0,"disabled transition threshold survives serialization");
    // A transition-only ray must keep one proposal even when its optical
    // depth exceeds both four and the ordinary interval's global event cap.
    f.mean=std::make_shared<cg::ConstantMean>(-.5);birth.birthValue=0;
    for(bool integral:{false,true}) {
        cg::ConditionalTrackingSettings options;
        options.transitionIntegralBounds=integral;options.transitionPointBounds=!integral;
        options.transitionMaxExpectedEvents=0;options.maxExpectedEvents=1;
        options.shadowRouletteThreshold=0;
        const cg::ConditionalFlightKernel reference(f,birth);
        check(reference.birthMajorant(2).split<1,"no-subdivision fixture starts after birth");
        const double depth=integral ? reference.transitionIntegralMajorant(1,2,0).integral(1,2)
            : reference.transitionIntervalMajorant(1,2,0);
        check(depth>4 && depth<80,"no-subdivision fixture exceeds the old four-event cap");
        OpticalDepthSampler coarseSampler({std::numeric_limits<float>::min()});
        cg::Random coarseRandom(coarseSampler);cg::ConditionalTrackingWork coarseWork;double hit;
        const cg::ConditionalSegment coarse(f,birth,V::UnitZ(),V::UnitZ(),1,options,nullptr,0,0);
        check(coarse.track(coarseRandom,false,hit,&coarseWork)==1 && coarseWork.intervals==1 && coarseWork.distance==1,
              "zero threshold keeps one transition proposal despite global cap");
        options.transitionMaxExpectedEvents=.1;
        OpticalDepthSampler fineSampler({std::numeric_limits<float>::min()});cg::Random fineRandom(fineSampler);
        cg::ConditionalTrackingWork fineWork;
        const cg::ConditionalSegment fine(f,birth,V::UnitZ(),V::UnitZ(),1,options,nullptr,0,0);
        check(fine.track(fineRandom,false,hit,&fineWork)==1 && fineWork.intervals>1,
              "positive threshold restores transition refinement");
    }
    std::cout<<"integrated transition: independent quadrature, inverse, delta/ratio law, disabled subdivision and configuration passed\n";
}
