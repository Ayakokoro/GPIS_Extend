// Deterministic, complete-flight replay of the production majorant proposals.
// No random candidate, collision, roulette or observation renewal is performed.
// Supported settings are deliberately fixed: no macrocell, point or fixed-step
// cap; old three-phase enables transition midpoint bounds, current two-phase
// retains ordinary adaptive near tracking. Both switch to local SE at 4 ell.
#pragma once
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include "math/SdfFunctions.hpp"
#include <functional>

class LongClockSampler final : public PathSampleGenerator {
    UniformSampler uniform;
public:
    void startPath(uint32,uint32) override {}
    void advancePath() override {}
    void saveState(OutputStreamHandle &) override {}
    void loadState(InputStreamHandle &) override {}
    bool nextBoolean(float) override { return false; }
    int nextDiscrete(int) override { return 0; }
    float next1D() override { return std::numeric_limits<float>::min(); }
    Vec2f next2D() override { return Vec2f(next1D()); }
    UniformSampler &uniformGenerator() override { return uniform; }
};

J exportMajorants(const cg::GPSSField &field,const cg::FlightState &birth,
                  double end,const ClassicGlobalMedium &classic) {
    const auto global=field.mean->globalMagnitudeBounds();
    if(!global) throw std::runtime_error("Majorant export requires the bounded VDB mean");
    const double L=global->maximumGradientNorm,G=L,sigma=field.kernel.sigma();
    const double ell=1/std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
    using I=cg::MajorantBounds::Interval;
    const double absoluteFar=(I(4)/cg::MajorantBounds::sqrt(I(birth.direction.dot(field.kernel.precision()*birth.direction)))*
                      I(1+16*std::numeric_limits<double>::epsilon())).upper;
    const double far=std::max(0.,(I(absoluteFar)-I(0.)).upper);
    cg::ConditionalFlightKernel kernel(field,birth);
    const auto initial=kernel.birthMajorant(end);
    J result={{"birth_split",initial.split},{"birth_maximum",initial.maximum},
              {"far_split",far},{"lipschitz",L},{"gradient_bound",G},
              {"classic",J::array()},{"old_three",J::array()},{"current_two",J::array()}};
    const double area=ClassicGlobal::negativePartMean(-G,sigma/ell);
    for(double a=0;a<end;) {
        const auto mean=[&](double t){return classic.meanValue(vecd(birth.birthPosition+t*birth.direction));};
        const double step=L>0 ? std::max(sigma/L,.5*std::max(0.,mean(a))/L) : end-a;
        const double b=std::min(end,a+step),m=mean(.5*(a+b));
        if(!(b>a))throw std::runtime_error("Classic proposal cannot advance");
        const double lower=m-L*(b-a)*.5-1e-12*std::max(sigma,std::abs(m));
        const double bound=ClassicGlobal::density(lower,sigma)*area*(1+1e-5);
        result["classic"].push_back({{"begin",a},{"end",b},{"maximum",bound},{"phase","classic"}});
        a=b;
    }
    for(bool old:{true,false}) {
        cg::ConditionalTrackingSettings settings;
        settings.fixedStepLimit=false;settings.localFarStartScale=4;
        settings.transitionPointBounds=old;
        cg::ConditionalSegment segment(field,birth,birth.birthPosition,birth.direction,end,settings,nullptr,L,G);
        auto &rows=result[old?"old_three":"current_two"];
        std::function<void(double,double,double,bool)> refine;
        refine=[&](double a,double b,double bound,bool transition) {
            const double eventLimit=transition?4.:64.;
            if(bound*(b-a)>eventLimit) {
                const double mid=.5*(a+b);
                if(!(mid>a&&mid<b))throw std::runtime_error("Cannot refine majorant");
                refine(a,mid,segment.majorant(a,mid,bound),transition);
                refine(mid,b,segment.majorant(mid,b,bound),transition);
                return;
            }
            if(!(bound>=0)||!std::isfinite(bound))throw std::runtime_error("Invalid majorant");
            const std::string phase=a>=far?"far":old?(a<initial.split?"birth":"transition"):"near";
            rows.push_back({{"begin",a},{"end",b},{"maximum",bound},{"phase",phase}});
        };
        for(double a=0;a<end;) {
            double b;
            const bool transition=old&&a>=initial.split&&a<far;
            if(a>=far||transition) {
                const double m=field.mean->evaluateValue(birth.birthPosition+a*birth.direction);
                b=std::min(end,a+(L>0?std::max(sigma/L,.5*std::max(0.,m)/L):end-a));
                if(transition)b=std::min(b,far);
            } else {
                b=a<initial.split?std::min(end,initial.split):std::min(end,a+.25*a);
                b=std::min(b,far);
                b=kernel.nextMeanBoundary(a,b);
            }
            if(!(b>a))throw std::runtime_error("Conditional proposal cannot advance");
            refine(a,b,segment.majorant(a,b),transition);
            a=b;
        }
        // Run the actual ratio tracker over the SAME full endpoint. Truncating
        // at a replay leaf changes final-proposal refinement and can change its
        // count. A long exponential clock limits candidates; disable roulette
        // so the tracker visits every leaf, retaining its ordinary proposal law.
        double opticalDepth=0;
        for(const auto &r:rows) {
            const double cost=(r["end"].get<double>()-r["begin"].get<double>())*r["maximum"].get<double>();
            opticalDepth+=cost;
        }
        {
            settings.shadowRouletteThreshold=0;
            LongClockSampler sampler;cg::Random random(sampler);
            cg::ConditionalSegment actual(field,birth,birth.birthPosition,birth.direction,end,settings,nullptr,L,G);
            cg::ConditionalTrackingWork work;double hit;
            const double survival=actual.track(random,false,hit,&work);
            if(!(survival>0)||hit!=end||work.distance!=end||work.intervals!=rows.size())
                throw std::runtime_error(std::string(old?"Old":"Current")+" majorant replay: expected "+
                    std::to_string(rows.size())+" leaves, actual "+std::to_string(work.intervals)+
                    ", traversed "+std::to_string(work.distance)+" / "+std::to_string(end));
            result[old?"old_three_replay_check":"current_two_replay_check"]={
                {"passed",true},{"length",end},{"intervals",rows.size()},{"candidates",work.candidates},
                {"integrated_majorant",opticalDepth}};
        }
    }
    return result;
}
