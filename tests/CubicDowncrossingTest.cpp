#include "math/CubicDowncrossing.hpp"
#include <cstdlib>
#include <iomanip>
#include <iostream>

using namespace Tungsten::CubicDowncrossing;

namespace {
void verify(bool ok,const char *message) {
    if (!ok) { std::cerr<<message<<'\n'; std::exit(1); }
}

void hit(Cubic p,double expected,double tolerance=1e-9,bool fromPositive=true) {
    double x=-1;
    if (!first(p,fromPositive,x) || std::abs(x-expected)>tolerance) {
        std::cerr<<std::setprecision(17)<<"expected "<<expected<<", got "<<x<<" for "<<p[0]<<", "<<p[1]
                 <<", "<<p[2]<<", "<<p[3]<<'\n';
        std::exit(1);
    }
}

void miss(Cubic p,bool &fromPositive,bool expectedSign) {
    double x;
    verify(!first(p,fromPositive,x),"tangency or upcrossing misclassified");
    verify(fromPositive==expectedSign,"wrong one-sided sign on exit");
}
}

int main() {
    hit({0.45*0.55,-1.,1.,0.},0.45);
    // Dyadic coordinates specify a narrow negative slot without rounding its roots.
    constexpr double halfWidth=1./65536;
    hit({0.25-halfWidth*halfWidth,-1.,1.,0.},0.5-halfWidth);

    // p=(x-1/4)(x-1/2)(x-3/4). It starts negative, goes positive, then down.
    const Cubic cubic{-3./32,11./16,-1.5,1.};
    for (double scale:{1.,1e200,1e-200}) {
        auto p=cubic;
        for (auto &c:p) c*=scale;
        hit(p,0.5,1e-9,false);
        for (auto &c:p) c=-c;
        hit(p,0.25);
    }
    // Decreasing stationary inflection, -(x-1/4)^3.
    hit({1./64,-3./16,0.75,-1.},0.25);
    hit({0.125,-0.5,0.,0.},0.25);

    bool positive=true;
    miss({0.25,-1.,1.,0.},positive,true);    // touch from above inside a segment
    miss({1.,-1.,0.,0.},positive,true);     // deferred shared endpoint
    miss({0.,1.,0.,0.},positive,true);      // positive on both sides of the corner
    miss({1.,-1.,0.,0.},positive,true);
    hit({0.,-1.,0.,0.},0.,0.,positive);     // slopes point down on both sides
    positive=false;
    miss({-0.25,1.,-1.,0.},positive,false); // touch from below
    miss({-1.,1.,0.,0.},positive,false);
    miss({0.,-1.,0.,0.},positive,false);    // negative on both sides of the corner
    miss({0.,0.,0.,0.},positive,false);
    miss({0.,1.,0.,0.},positive,true);
    miss({3.,0.,0.,0.},positive,true);
    miss({-3.,0.,0.,0.},positive,false);

    // Cutting a cubic at a knot must reproduce its first root after remapping.
    for (double cut:{0.25,0.375,0.5,0.625,0.75}) {
        Cubic left{},right{};
        addRestricted(left,cubic,0,cut);
        addRestricted(right,cubic,cut,1-cut);
        positive=false;
        double u;
        if (first(left,positive,u)) {
            verify(std::abs(cut*u-0.5)<1e-9,"restricted left segment hit");
        } else {
            verify(first(right,positive,u),"missed downcrossing after splitting cubic");
            verify(std::abs(cut+(1-cut)*u-0.5)<1e-9,"restricted right segment hit");
        }
    }
    std::cout<<"Cubic first downcrossings, narrow slots, tangencies, splits and scale checks passed\n";
    return 0;
}
