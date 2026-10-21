#pragma once
#include "math/ConditionalProfile.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace Tungsten::ConditionalGpis::MajorantBounds {

// Small outward-rounded intervals for majorant construction. Transcendentals
// also receive a libm roundoff allowance; no changes to the global FP mode.
inline double down(double x) { return std::nextafter(x,-std::numeric_limits<double>::infinity()); }
inline double up(double x) { return std::nextafter(x,std::numeric_limits<double>::infinity()); }
struct Interval {
    double lower,upper;
    Interval(double x=0) : lower(x),upper(x) {}
    Interval(double lo,double hi) : lower(lo),upper(hi) {}
};
inline Interval operator+(Interval a,Interval b) {
    return {down(a.lower+b.lower),up(a.upper+b.upper)};
}
inline Interval operator-(Interval a) { return {-a.upper,-a.lower}; }
inline Interval operator-(Interval a,Interval b) { return a+(-b); }
inline Interval operator*(Interval a,Interval b) {
    if((a.lower==0 && a.upper==0) || (b.lower==0 && b.upper==0)) return {0};
    const std::array<double,4> p{{a.lower*b.lower,a.lower*b.upper,a.upper*b.lower,a.upper*b.upper}};
    return {down(*std::min_element(p.begin(),p.end())),up(*std::max_element(p.begin(),p.end()))};
}
inline Interval operator/(Interval a,Interval b) {
    if(b.lower<=0 && b.upper>=0)
        return {-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity()};
    return a*Interval(down(1/b.upper),up(1/b.lower));
}
inline Interval hull(Interval a,Interval b) {
    return {std::min(a.lower,b.lower),std::max(a.upper,b.upper)};
}
inline Interval exp(Interval x) {
    constexpr double error=32*std::numeric_limits<double>::epsilon();
    return {std::max(0.0,down(std::exp(x.lower)*(1-error))),
            up(std::exp(x.upper)*(1+error))};
}
inline Interval sqrt(Interval x) {
    return {std::max(0.0,down(std::sqrt(std::max(0.0,x.lower)))),
            up(std::sqrt(std::max(0.0,x.upper)))};
}

// A quadratic reaches its extrema at an endpoint or its vertex. Enclose the
// vertex as well as the coefficients so rounding cannot lose a boundary root.
// A coefficient interval spanning zero uses the safe Horner enclosure instead.
inline Interval quadraticPolynomialRange(Interval c0,Interval c1,Interval c2,
                                          double lo,double hi) {
    const auto value=[&](Interval t) { return c0+t*(c1+t*c2); };
    Interval result=hull(value(Interval(lo)),value(Interval(hi)));
    if(c2.lower==0 && c2.upper==0) return result;
    if(c2.lower<=0 && c2.upper>=0) return hull(result,value(Interval(lo,hi)));
    const Interval vertex=-c1/(Interval(2)*c2);
    const double left=std::max(lo,vertex.lower),right=std::min(hi,vertex.upper);
    if(left<=right) result=hull(result,value(Interval(left,right)));
    return result;
}

struct SeRemainders { Interval second,fourth; };
inline SeRemainders seRemainders(double q) {
    // E2=sum q^n/(n+2)!, E4=sum 2q^(2n)/(2n+4)!, 0<=q<=1.
    // Every term is positive. Bound both the arithmetic error and the omitted
    // tails by their next term times a geometric-series bound.
    double e2=0.5,t2=0.5,e4=1.0/12,t4=e4;
    for(int n=1;n<=20;++n) { t2*=q/(n+2); e2+=t2; }
    for(int n=1;n<=12;++n) { t4*=q*q/((2*n+3.0)*(2*n+4.0)); e4+=t4; }
    constexpr double error=256*std::numeric_limits<double>::epsilon();
    const double tail2=t2*q/23/(1-q/24);
    const double tail4=t4*q*q/(29*30)/(1-q*q/(31*32));
    return {{down(e2*(1-error)),up((e2+tail2)*(1+error))},
            {down(e4*(1-error)),up((e4+tail4)*(1+error))}};
}

struct SeVarianceBounds {
    Interval scaledValue; // V(q)/q^2, finite at q=0
    Interval slope;      // N(q)/V(q)
    Interval normalized; // N(q)/V(q)^2, finite at q=0
};
inline SeVarianceBounds seVarianceBounds(Interval q) {
    Profile::Scope profile(Profile::Stage::SeVariance);
    if(q.upper<=1) {
        const auto lo=seRemainders(std::max(0.0,q.lower)),hi=seRemainders(q.upper);
        const Interval e2(lo.second.lower,hi.second.upper),e4(lo.fourth.lower,hi.fourth.upper);
        return {exp(-q)*e2,q*q*e4/e2,exp(q)*e4/(e2*e2)};
    }
    // Both variance ratios are <=1. Avoid q^2*exp(-q) overflow/underflow
    // products in the far tail; the universal bounds are already very tight.
    if(q.upper>64) return {{0,up(1/(q.lower*q.lower))},{0,1},{0,1}};
    const Interval e=exp(-q);
    const Interval v=Interval(1)-(Interval(1)+q)*e;
    const Interval n=Interval(1)-(Interval(2)+q*q)*e+exp(Interval(-2)*q);
    return {v/(q*q),n/v,n/(v*v)};
}
inline Interval squaredKernelDistance(double a,double t) {
    return Interval(a)*Interval(t)*Interval(t);
}
inline double slopeStddevUpper(double sigma2,double a,double end) {
    const auto bounds=seVarianceBounds(squaredKernelDistance(a,end));
    return sqrt(Interval(sigma2)*Interval(a)*Interval(std::min(1.0,bounds.slope.upper))).upper;
}
// Enclose exp(-a*t^2/2)*(c0+c1*t+c2*t^2), t>=0, including all extrema.
// Its derivative has the sign of a polynomial of degree at most three.
// Interval sign exclusion isolates possible critical points without relying
// on approximate quadratic/cubic roots (including multiple or boundary roots).
inline Interval exponentialPolynomialRange(double a,Interval c0,Interval c1,Interval c2,
                                           double lo,double hi) {
    Profile::Scope profile(Profile::Stage::ExpPolynomial);
    if(c0.lower==0 && c0.upper==0 && c1.lower==0 && c1.upper==0 && c2.lower==0 && c2.upper==0)
        return {0};
    const auto value=[&](Interval t) {
        return exp(Interval(-0.5)*Interval(a)*t*t)*(c0+t*(c1+t*c2));
    };
    const Interval d0=c1,d1=Interval(2)*c2-Interval(a)*c0,
                   d2=-Interval(a)*c1,d3=-Interval(a)*c2;
    const auto criticalPolynomial=[&](Interval t) { return d0+t*(d1+t*(d2+t*d3)); };
    const auto criticalDerivative=[&](Interval t) { return d1+t*(Interval(2)*d2+t*Interval(3)*d3); };
    Interval result=hull(value(Interval(lo)),value(Interval(hi)));
    struct Cell { double lo,hi; int depth; };
    // Depth-first traversal needs at most depth+1 entries.
    std::array<Cell,64> pending;
    int pendingCount=1;
    pending[0]={lo,hi,0};
    int refinements=0;
    while(pendingCount>0) {
        Profile::count(Profile::Counter::PolynomialCells);
        const auto cell=pending[--pendingCount];
        const Interval t(cell.lo,cell.hi),d=criticalPolynomial(t);
        // All partition endpoints are already in result.
        if(d.lower>=0 || d.upper<=0) continue;
        const Interval enclosure=value(t);
        if(enclosure.lower>=result.lower && enclosure.upper<=result.upper) continue;
        const double middle=0.5*cell.lo+0.5*cell.hi;
        const double scale=std::max({std::abs(enclosure.lower),std::abs(enclosure.upper),1e-300});
        if(refinements>=64 || cell.depth>=48 || !(middle>cell.lo && middle<cell.hi) ||
           enclosure.upper-enclosure.lower<=1e-10*scale) {
            // A work limit only loosens the certificate. It never drops a root.
            if(refinements>=64 || cell.depth>=48)
                Profile::count(Profile::Counter::PolynomialLimitCells);
            result=hull(result,enclosure);
            continue;
        }
        ++refinements;
        Profile::count(Profile::Counter::PolynomialRefinements);
        const Interval curvature=criticalDerivative(t);
        if(curvature.lower>0 || curvature.upper<0) {
            // Every root r satisfies r=middle-Q(middle)/Q'(xi) for some xi
            // in this cell. This contraction cannot lose a critical point.
            const Interval roots=Interval(middle)-criticalPolynomial(Interval(middle))/curvature;
            const double left=std::max(cell.lo,roots.lower),right=std::min(cell.hi,roots.upper);
            if(left>right) continue;
            if(right-left<0.75*(cell.hi-cell.lo)) {
                result=hull(result,hull(value(Interval(left)),value(Interval(right))));
                pending[pendingCount++]={left,right,cell.depth+1};
                continue;
            }
        }
        result=hull(result,value(Interval(middle)));
        pending[pendingCount++]={middle,cell.hi,cell.depth+1};
        pending[pendingCount++]={cell.lo,middle,cell.depth+1};
    }
    return result;
}
}
