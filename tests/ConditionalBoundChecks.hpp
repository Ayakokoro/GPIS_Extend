#pragma once
#include "math/conditional/transport/ConditionalMajorantBounds.h"

class CountingBoundMean final : public cg::MeanField {
    std::shared_ptr<const cg::MeanField> _mean;
public:
    mutable int pointQueries=0,rayQueries=0,beginDerivativeRequests=0;
    bool cacheMidpoint=true;
    double midpointBias=0;
    explicit CountingBoundMean(std::shared_ptr<const cg::MeanField> mean):_mean(std::move(mean)) {}
    const char *typeName() const override { return "counting_bound_mean"; }
    cg::MeanJet evaluate(const V &p) const override { ++pointQueries; return _mean->evaluate(p); }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override { return _mean->bounds(b); }
    double valueDifference(const V &p,const V &d) const override { return _mean->valueDifference(p,d); }
    cg::MeanRayBounds rayBounds(const V &o,const V &w,double lo,double hi,
                                const Tungsten::SdfRayQueryOptions &options = {}) const override {
        beginDerivativeRequests+=options.computeBeginDerivative;
        ++rayQueries;
        auto b=_mean->rayBounds(o,w,lo,hi,options);
        if(!cacheMidpoint) b.midpointValue=std::numeric_limits<double>::quiet_NaN();
        else {
            b.midpointValue+=midpointBias;
            b.midpointError+=std::abs(midpointBias);
        }
        return b;
    }
};

inline void midpointReuseChecks()
{
    auto f=field();
    auto counted=std::make_shared<CountingBoundMean>(f.mean);
    f.mean=counted;
    cg::FlightState birth;
    birth.birthPosition=V(0.2,-0.1,1);
    birth.direction=V(0.2,0.1,1).normalized();
    birth.birthGradient=birth.direction;
    birth.birthValue=0.1;
    const cg::ConditionalFlightKernel flight(f,birth);
    counted->pointQueries=0;
    const double cached=flight.intervalMajorant(0.3,0.4);
    check(counted->pointQueries==0 && counted->rayQueries==1 && counted->beginDerivativeRequests==0,
        "ordinary majorant reuses midpoint and does not request a beginning derivative");
    counted->rayQueries=0;
    const double cheap=flight.cheapIntervalMajorant(0.3,0.4);
    check(cheap>=cached && counted->rayQueries==1 && counted->pointQueries==0,
        "cheap certificate covers the fine one with a single SDF query");
    counted->rayQueries=0;
    close(flight.intervalMajorant(0.3,0.4,cheap*0.2),cheap,0,
        "small expected candidate count keeps the cheap bound");
    close(flight.intervalMajorant(0.3,0.4,0),cached,0,"zero threshold forces the fine bound");
    close(flight.intervalMajorant(0.3,0.4,cached*0.2,cached),cached,0,
        "an inherited certificate participates in screening");
    check(counted->rayQueries==3 && counted->pointQueries==0,
        "screening and tightening share each query's SDF certificate and midpoint");
    counted->cacheMidpoint=false;
    close(flight.intervalMajorant(0.3,0.4),cached,0,"uncached provider retains point-query fallback");
    check(counted->pointQueries==1,"missing cached midpoint performs exactly one point query");
    counted->cacheMidpoint=true;
    counted->midpointBias=0.25;
    const double uncertain=flight.intervalMajorant(0.3,0.4);
    for(int i=0;i<=30;++i)
        check(flight.evaluate(0.3+0.1*i/30).hazard.value<=uncertain,
            "midpoint error is included in conditional enclosure");
    flight.birthMajorant(0.1);
    check(counted->beginDerivativeRequests>0,"birth certificate still requests its beginning derivative");
}

inline void boundChecks()
{
    midpointReuseChecks();
    namespace b = cg::MajorantBounds;
    // Exact binary fixtures: endpoints, an interior vertex, a boundary vertex,
    // a linear/constant polynomial, and a very small quadratic coefficient.
    const double small=std::ldexp(1.0,-50);
    const double polynomialFixtures[][7]={
        {1,-4,2,0,3,-1,7}, {1,4,-2,0,3,-5,3},
        {2,-3,0,-1,2,-4,5}, {1,-2,1,1,2,0,1},
        {1,0,0,0,4,1,1}, {0,-2*small,small,0,2,-small,0}
    };
    for(const auto &p:polynomialFixtures) {
        const auto range=b::quadraticPolynomialRange(p[0],p[1],p[2],p[3],p[4]);
        check(range.lower<=p[5] && range.upper>=p[6],
            "cheap polynomial range includes exact extrema");
        check(range.lower>=p[5]-1e-12 && range.upper<=p[6]+1e-12,
            "fixed-coefficient quadratic range stays close to its exact extrema");
    }
    const auto uncertain=b::quadraticPolynomialRange(b::Interval(0.9,1.1),
        b::Interval(-2.1,-1.9),b::Interval(-1e-12,1e-12),0,2);
    for(double c0:{0.9,1.1}) for(double c1:{-2.1,-1.9}) for(double c2:{-1e-12,0.0,1e-12})
        for(int j=0;j<=40;++j) {
            const double t=j/20.0,value=c0+t*(c1+t*c2);
            check(value>=uncertain.lower && value<=uncertain.upper,
                "coefficient uncertainty spanning zero cannot lose a quadratic extremum");
        }
    // Independently evaluated with 100-decimal-digit arithmetic. The inputs
    // are exact binary doubles, including the cancellation-prone tiny q cases.
    const double fixtures[][4]={
        {9.0949470177292824e-13,0.49999999999969685,1.3786343542546282e-25,0.3333333333334344},
        {9.5367431640625e-7,0.4999996821086749,1.5158240210873324e-13,0.33333343929715636},
        {0.25,0.42398433857190265,0.0095869438467119199,0.36178482927943678},
        {1,0.26424111765711533,0.11995468382561243,0.45395919033792487},
        {2,0.14849853757254047,0.34731644978032195,0.58471358617026836},
        {8,0.015577825567951209,0.98082083771510198,0.98379106393563576},
        {64,0.000244140625,1,1}
    };
    for(const auto &fixture:fixtures) {
        const auto q=b::seVarianceBounds(b::Interval(fixture[0]));
        const b::Interval values[]={q.scaledValue,q.slope,q.normalized};
        for(int i=0;i<3;++i)
            check(values[i].lower<=fixture[i+1] && values[i].upper>=fixture[i+1],
                "SE variance bounds enclose high-precision fixture");
    }
    const auto zero=b::seVarianceBounds(b::Interval(0));
    check(zero.normalized.lower<=1.0/3 && zero.normalized.upper>=1.0/3,
        "normalized variance has finite birth limit");
    for(double q:{1e-200,1e-30,0.999999999999,1.000000000001,65.0,1e200}) {
        const auto v=b::seVarianceBounds(b::Interval(q));
        check(v.normalized.lower>=0 && v.normalized.upper>0 &&
            std::isfinite(v.normalized.upper),"SE ratio remains bounded across branches");
    }

    for(double a:{1e-6,1.0,1e6}) {
        const double root=std::sqrt(a),end=3/root;
        const auto linear=b::exponentialPolynomialRange(a,0,root,0,0,end);
        const double peak=std::exp(-0.5);
        check(linear.upper>=peak && linear.upper<peak*(1+1e-6),
            "exponential-linear interior maximum is tightly enclosed");
        const auto quadratic=b::exponentialPolynomialRange(a,1,0,-a,0,end);
        const double minimum=-2*std::exp(-1.5);
        check(quadratic.lower<=minimum && quadratic.lower>minimum-1e-6 && quadratic.upper>=1,
            "exponential-quadratic interior minimum is tightly enclosed");
        // Derivative is -(x-1)^2(x+1)*exp(-x^2/2): a repeated root.
        const auto repeated=b::exponentialPolynomialRange(a,1,-root,a,0,end);
        // An extremum exactly at the left endpoint, plus an almost-linear case.
        const auto boundary=b::exponentialPolynomialRange(a,0,root,0,1/root,end);
        const auto nearLinear=b::exponentialPolynomialRange(a,0,root,1e-14*a,0,end);
        for(int i=0;i<=200;++i) {
            const double t=end*i/200,x=root*t,e=std::exp(-0.5*x*x);
            const double value=e*(1-x+x*x);
            check(value>=repeated.lower && value<=repeated.upper,"multiple critical point coverage");
            const double perturbed=e*(x+1e-14*x*x);
            check(perturbed>=nearLinear.lower && perturbed<=nearLinear.upper,"almost-degenerate polynomial coverage");
            if(x>=1) check(e*x>=boundary.lower && e*x<=boundary.upper,"boundary critical point coverage");
        }
    }

    cg::Random rng(120535);
    for(int i=0;i<100;++i) {
        const double a=std::exp(8*(rng.uniform01()-0.5));
        const double c0=4*rng.uniform01()-2,c1=4*rng.uniform01()-2,c2=4*rng.uniform01()-2;
        const double lo=rng.uniform01()/std::sqrt(a),hi=lo+2/std::sqrt(a);
        const auto range=b::exponentialPolynomialRange(a,c0,c1,c2,lo,hi);
        const auto cheap=b::exp(b::Interval(-0.5)*b::Interval(a)*b::Interval(lo,hi)*b::Interval(lo,hi))*
            b::quadraticPolynomialRange(c0,c1,c2,lo,hi);
        for(int j=0;j<=100;++j) {
            const double t=j==100 ? hi : lo+(hi-lo)*j/100;
            const double value=std::exp(-0.5*a*t*t)*(c0+t*(c1+t*c2));
            check(value>=range.lower && value<=range.upper,"exponential polynomial enclosure");
            check(value>=cheap.lower && value<=cheap.upper,"cheap exponential-polynomial enclosure");
        }
    }

    int flights=0,intervals=0;
    for(int shape=0;shape<3;++shape) for(double scale:{1e-4,1.0,1e4})
        for(double sigma:{0.02,1.0}) for(double f0:{0.0,0.3}) for(double slope:{-0.5,0.02,1.5}) {
            if(f0==0 && slope<=0) continue;
            auto f=field();
            if(shape==0) f.mean=std::make_shared<cg::ConstantMean>(0);
            if(shape==2) f.mean=std::make_shared<cg::SphereMean>(V::Zero(),1);
            f.kernel=cg::CovarianceKernel(sigma,scale*f.kernel.precision());
            f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            cg::FlightState birth;
            birth.birthPosition=V(0.2,-0.1,2);
            birth.direction=V(0.2,0.1,1).normalized();
            const double a=birth.direction.dot(f.kernel.precision()*birth.direction),ell=1/std::sqrt(a);
            birth.birthValue=f0*sigma;
            birth.birthGradient=V(0.1,-0.2,0.3)*sigma/ell;
            birth.birthGradient+=birth.direction*(slope*sigma/ell-birth.direction.dot(birth.birthGradient));
            cg::ConditionalFlightKernel flight(f,birth);
            const auto near=flight.birthMajorant(3*ell);
            for(double fraction:{1e-6,0.001,0.1,0.5,1.0})
                check(flight.evaluate(near.split*fraction).hazard.value<=near.maximum*(1+1e-10),
                    "tightened birth envelope covers zero and positive births");
            for(const auto &interval:std::array<std::pair<double,double>,4>{{
                    {near.split,1.25*near.split},{0.05*ell,0.15*ell},{0.8*ell,1.2*ell},{2*ell,3*ell}}}) {
                const double lo=interval.first,hi=interval.second;
                const double bound=flight.intervalMajorant(lo,hi);
                const double cheap=flight.cheapIntervalMajorant(lo,hi);
                const double adaptive=flight.intervalMajorant(lo,hi,4);
                check(bound<=adaptive && adaptive<=cheap,"adaptive majorant lies between fine and cheap bounds");
                const double adjacent=hi<=near.smoothEnd ?
                    flight.birthAdjacentMajorant(lo,hi,near):bound;
                for(int j=0;j<=24;++j) {
                    const double hazard=flight.evaluate(lo+(hi-lo)*j/24).hazard.value;
                    check(hazard<=std::min(bound,adjacent)*(1+1e-10),
                        "normalized interval bounds cover the conditional hazard");
                    check(hazard<=cheap*(1+1e-10),"cheap interval bounds cover the conditional hazard");
                }
                ++intervals;
            }
            ++flights;
        }
    // This homogeneous example has a known birth envelope from the previous
    // implementation. The improved coefficient should materially tighten it.
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(0);
    f.kernel=cg::CovarianceKernel(1,M::Identity());
    cg::FlightState birth;birth.birthGradient=V::UnitZ();
    cg::ConditionalFlightKernel flight(f,birth);
    const auto near=flight.birthMajorant(2);
    check(near.maximum<0.5,"combined birth slope must avoid the former 3|g| coefficient");
    std::cout<<"bounds: cheap/fine certificates, screening, high-precision SE ratios, critical-point enclosures and "
             <<flights<<" flights / "<<intervals<<" intervals passed; simple birth bound "<<near.maximum<<"\n";
}
