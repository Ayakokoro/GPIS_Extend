#pragma once
#include "math/ClassicGlobalMath.hpp"

inline void localFarChecks()
{
    const std::vector<std::pair<std::shared_ptr<const cg::MeanField>,double>> providers{
        {field().mean,2}, {std::make_shared<cg::ConstantMean>(-.2),0},
        {std::make_shared<cg::SphereMean>(V::Zero(),1),1}};
    int checks=0;
    for(const auto &provider:providers) for(double sigma:{.04,.7}) for(double scale:{.1,10.})
        for(double f0:{0.,.2}) for(double slope:{.1,8.}) for(double q:{2.001,4.,8.,16.}) {
            auto f=field();f.mean=provider.first;f.kernel=cg::CovarianceKernel(sigma,scale*f.kernel.precision());
            f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            cg::FlightState birth;birth.birthPosition=V(.2,-.1,.8);birth.direction=V(.2,.1,-1).normalized();
            const double ell=1/std::sqrt(birth.direction.dot(f.kernel.precision()*birth.direction));
            birth.birthValue=f0*sigma;birth.birthGradient=birth.direction*(slope*sigma/ell);
            const cg::ConditionalFlightKernel kernel(f,birth),reference(f,birth);
            for(double width:{.2,3.}) {
                const double lo=q*ell,hi=lo+width*ell;
                // Deliberately distinguish the scalar Lipschitz and gradient
                // certificates, as scene configuration is permitted to do.
                const double bound=kernel.localFarIntervalMajorant(lo,hi,provider.second,provider.second+1);
                for(int j=0;j<=16;++j) {
                    const double t=lo+(hi-lo)*j/16;
                    check(reference.evaluate(t).hazard.value<=bound*(1+1e-10),"local far bound covers independent conditional hazard");
                    ++checks;
                }
            }
        }
    // In the stationary limit, compare to the real classic_global functions,
    // including deep negative means that previously used the loose 1-z bound.
    for(double value:{-30.,-8.,0.,3.}) for(double gradient:{0.,1.}) {
        auto f=field();f.mean=std::make_shared<cg::ConstantMean>(value);f.kernel=cg::CovarianceKernel(1,M::Identity());
        cg::FlightState birth;birth.birthValue=.2;birth.birthGradient=5*V::UnitZ();
        cg::ConditionalFlightKernel kernel(f,birth);
        const double expected=Tungsten::ClassicGlobal::density(value,1)*
            Tungsten::ClassicGlobal::negativePartMean(-gradient,1)*(1+1e-5);
        const double at4=kernel.localFarIntervalMajorant(4,4.2,0,gradient);
        const double at8=kernel.localFarIntervalMajorant(8,8.2,0,gradient);
        check(at8<at4,"far residual envelope refreshes as the ray ages");
        for(double age:{16.,40.}) {
            const double bound=kernel.localFarIntervalMajorant(age,age+.2,0,gradient);
            close(bound,expected,1e-8*expected+1e-14,"local far limit agrees with classic_global majorant");
        }
    }
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(.02);f.kernel=cg::CovarianceKernel(.2,M::Identity());
    cg::Random rng(81057);
    for(bool positive:{false,true}) for(double start:{0.,1.8,2.2}) {
        cg::FlightState birth;birth.birthValue=positive ? .1:0;birth.birthGradient=.2*V::UnitZ();
        cg::ConditionalTrackingSettings options;options.localFarStartScale=2;options.maxExpectedEvents=.1;
        const cg::ConditionalSegment segment(f,birth,start*V::UnitZ(),V::UnitZ(),3,options,nullptr,0,0);
        const cg::ConditionalFlightKernel reference(f,birth);
        double integral=0;
        for(int i=0;i<24000;++i) integral+=reference.evaluate(start+3.*(i+.5)/24000).hazard.value*3/24000;
        const double expected=std::exp(-integral);
        constexpr int count=5000;
        int escaped=0;double sum=0,sum2=0,hit;
        for(int i=0;i<count;++i) {
            escaped+=segment.track(rng,true,hit)!=0;
            const double tr=segment.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/count,expected,6*std::sqrt(expected*(1-expected)/count)+1e-4,"local far delta survival including switch/refinement");
        const double variance=std::max(0.,(sum2-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+1e-4,"local far ratio survival including switch/refinement");
    }
    cg::FlightState birth;birth.birthGradient=V::UnitZ();
    cg::ConditionalTrackingSettings enabled;enabled.localFarStartScale=4;
    const cg::ConditionalSegment nearOnly(f,birth,V::Zero(),V::UnitZ(),1,enabled,nullptr,0,0);
    auto disabled=enabled;disabled.localFarStartScale=0;
    const cg::ConditionalSegment original(f,birth,V::Zero(),V::UnitZ(),1,disabled,nullptr,0,0);
    for(bool collision:{false,true}) for(int i=0;i<100;++i) {
        cg::Random a(100+i),b(100+i);double ta,tb;
        const double va=nearOnly.track(a,collision,ta),vb=original.track(b,collision,tb);
        check(va==vb && ta==tb && a.uniform01()==b.uniform01(),"local far switch preserves near-only samples and RNG");
    }
    bool rejected=false;
    try { cg::ConditionalSegment missing(f,birth,V::Zero(),V::UnitZ(),3,enabled); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"local far requires finite Lipschitz certificate");
    rejected=false;
    try { cg::ConditionalSegment missing(f,birth,V::Zero(),V::UnitZ(),3,enabled,nullptr,0,-1); }
    catch(const cg::NumericError &) { rejected=true; }
    check(rejected,"local far requires finite nonnegative gradient certificate");
    const cg::ConditionalSegment empty(f,birth,V::Zero(),V::UnitZ(),0,enabled,nullptr,0,0);
    double hit;cg::ConditionalTrackingWork work;
    check(empty.track(rng,false,hit,&work)==1 && hit==0 && work.intervals==0,"empty local far flight");
    std::cout<<"local far: "<<checks<<" hazard checks, classic_global limit, refreshed tail, near parity and delta/ratio survival passed\n";
}
