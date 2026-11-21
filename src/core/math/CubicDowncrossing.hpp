#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace Tungsten::CubicDowncrossing {

using Cubic = std::array<double,4>;

inline double value(const Cubic &c, double x) {
    return ((c[3]*x+c[2])*x+c[1])*x+c[0];
}

inline double slope(const Cubic &c, double x) {
    return (3*c[3]*x+2*c[2])*x+c[1];
}

// Compose p(offset + scale*u), then add the resulting polynomial to c.
inline void addRestricted(Cubic &c, const Cubic &p, double offset, double scale) {
    c[0] += value(p,offset);
    c[1] += scale*slope(p,offset);
    c[2] += scale*scale*(p[2]+3*offset*p[3]);
    c[3] += scale*scale*scale*p[3];
}

// Find the first positive-to-negative transition in [0,1). At a shared
// endpoint defer the decision until the sign on the right is available.
inline bool first(const Cubic &c, bool &positiveOnLeft, double &root) {
    const double end=value(c,1);
    const double b1=c[0]+c[1]/3, b2=c[0]+(2*c[1]+c[2])/3;
    if (std::min({c[0],b1,b2,end})>0) { positiveOnLeft=true; return false; }
    if (std::max({c[0],b1,b2,end})<0) { positiveOnLeft=false; return false; }
    double knots[3];
    int count=0;
    auto insert=[&](double x) { if (x>0 && x<1) knots[count++]=x; };
    double a=3*c[3], b=2*c[2], d=c[1];
    if (a==0) {
        if (b!=0) insert(-d/b);
    } else {
        // Rescale unusually small/large polynomials before squaring. Normal
        // rendering ranges take the direct path without any extra division.
        const double scale=std::max({std::abs(a),std::abs(b),std::abs(d)});
        if (scale<1e-100 || scale>1e100) { a/=scale; b/=scale; d/=scale; }
        const double discriminant=b*b-4*a*d;
        if (discriminant>0) {
            const double q=-0.5*(b+std::copysign(std::sqrt(discriminant),b));
            insert(q/a);
            insert(d/q);
            if (count==2 && knots[0]>knots[1]) std::swap(knots[0],knots[1]);
        } else if (discriminant==0) {
            // A stationary inflection can be a triple downcrossing. Testing it
            // directly avoids slow Newton convergence near the flat root.
            insert(-0.5*b/a);
        }
    }
    knots[count++]=1;
    double lo=0, left=c[0];
    for (int i=0; i<count; ++i) {
        double hi=knots[i];
        const double right=value(c,hi);
        if (right<0 && (left>0 || (left==0 && positiveOnLeft))) {
            if (left==0) { root=lo; return true; }
            // Monotone bracket, ordinary precision. Newton is fast for simple
            // roots; a midpoint step also handles a small/zero derivative.
            double x=lo+(hi-lo)*(left/(left-right));
            for (int j=0; j<28; ++j) {
                const double f=value(c,x);
                if (f==0) break;
                if (f>0) lo=x; else hi=x;
                const double dx=f/slope(c,x);
                if (std::abs(dx)<=1e-9) break;
                const double next=x-dx;
                x=next>lo && next<hi ? next : 0.5*(lo+hi);
            }
            root=x;
            return true;
        }
        if (right!=0) positiveOnLeft=right>0;
        else if (left!=0) positiveOnLeft=left>0;
        left=right;
        lo=hi;
    }
    return false;
}

}
