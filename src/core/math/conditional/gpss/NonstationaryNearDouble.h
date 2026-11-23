#pragma once

#include "math/conditional/core/Types.h"
#include <algorithm>
#include <array>
#include <cmath>

namespace Tungsten::ConditionalGpis {

struct NearGibbsCovariance {
    double variance;
    Vector3 regression;
    Matrix3 gradientGivenValueCovariance;
    double slopeVarianceGivenValue;
};

namespace NearHermite {
// For a Hermite coefficient q of nominal order k in flight age t, axial is
// dq/dt - k*q/t. Carrying it alongside the spatial gradient retains the
// longitudinal residual even when it is far smaller than the transverse one.
struct Coefficient {
    double value;
    Vector3 gradient;
    double axial;
};

inline Coefficient combine(double a, const Coefficient &x, double b, const Coefficient &y) {
    return {a*x.value + b*y.value, a*x.gradient + b*y.gradient, a*x.axial + b*y.axial};
}

inline Coefficient product(const Coefficient &x, const Coefficient &y) {
    return {x.value*y.value, x.gradient*y.value + y.gradient*x.value,
            x.axial*y.value + y.axial*x.value};
}

// expm1(z) - 2*(expm1(z)-z)/z, which vanishes to second order.
inline double exponentialSlopeRemainder(double z) {
    if (std::abs(z) >= .02)
        return std::expm1(z) - 2*(std::expm1(z)-z)/z;
    return z*z*(1./6 + z*(1./12 + z*(1./40 + z*(1./180 + z*(1./1008 + z/6720)))));
}
}

// The Gibbs kernel is the inner product of normalized Gaussian functions.
// Expand the target Gaussian in normalized Hermite functions centered and
// scaled at birth. Conditioning removes the observed directions in this
// orthonormal basis, so the resulting variances are sums of squares.
inline bool nearGibbsCovariance(const Vector3 &lengthY, const std::array<Vector3,3> &gradientY,
        const Vector3 &deltaLength, const Vector3 &linearRemainder,
        const std::array<Vector3,3> &deltaGradient, const Vector3 &displacement,
        const Vector3 &direction, double age, NearGibbsCovariance &out) {
    using namespace NearHermite;
    static constexpr int maxDegree=20;
    static const auto roots=[]() {
        std::array<double,maxDegree+1> r{};
        for (int i=0; i<=maxDegree; ++i) r[i]=std::sqrt(double(i));
        return r;
    }();
    const double invRoot2=1/roots[2];
    const Vector3 a=deltaLength.cwiseQuotient(lengthY), v=displacement.cwiseQuotient(lengthY);
    const Vector3 den=(1 + (1+a.array()).square()).matrix();
    const Vector3 s=(a.array()*(2+a.array())/den.array()).matrix();
    const Vector3 p=(2*v.array()/den.array()).matrix();
    const double stretch=s.cwiseAbs().maxCoeff();
    if (stretch > .16) return false; // Well-separated widths, use the closed form.

    // Shape changes generate degree-two modes at first order. The first
    // omitted slope residual has order ceil((D+1)/2)-1, whereas the retained
    // one starts at order two. Increase the band as the parameters separate.
    const double radius=std::max(stretch,p.cwiseAbs2().maxCoeff());
    const int degree=radius<1e-7 ? 6 : radius<1e-4 ? 8 : radius<.002 ? 10 :
                     radius<.015 ? 12 : radius<.05 ? 14 : radius<.09 ? 16 : maxDegree;
    std::array<std::array<Coefficient,maxDegree+1>,3> terms;
    std::array<Coefficient,3> first, shape, low;
    std::array<Vector3,3> observation;
    double logC=0;
    Vector3 dlogC=Vector3::Zero();
    for (int i=0; i<3; ++i) {
        const double A=a[i], D=den[i], S=s[i], V=v[i], P=p[i];
        const Vector3 h=gradientY[i]/lengthY[i], dh=deltaGradient[i]/lengthY[i];
        const Vector3 da=h+dh;
        const double rem=linearRemainder[i]/lengthY[i], adot=da.dot(direction);
        const double a1=dh.dot(direction)-rem/age;
        const double e=-A*A*(1+A)/D; // s-a
        const double de=-A*(4+A*(8+A*(4+A)))/(D*D);
        const Vector3 ds=(4*(1+A)/(D*D))*da;
        const double s1=a1+de*adot-e/age;
        const Vector3 dp=(2/(D*lengthY[i]))*Vector3::Unit(i)-V*ds;
        const double p1=-V*ds.dot(direction);
        first[i]={P,dp,p1};
        shape[i]={S,ds,s1};
        logC+=-.5*std::log1p(A*A/(2*(1+A)))-V*V/D;
        dlogC+=(-.5*S/(1+A)+2*(1+A)*V*V/(D*D))*da
                - (P/lengthY[i])*Vector3::Unit(i);

        double rem2=dh.dot(direction)-2*rem/age;
        for (int k=0; k<3; ++k) {
            if (h[k]!=0 && da[k]!=0) {
                // Both points lie in the same exponential ramp. Its endpoint
                // gradient ratio gives the exponent along this exact segment.
                const double z=std::log1p(dh[k]/h[k]);
                rem2=h[k]*direction[k]*exponentialSlopeRemainder(z);
                break;
            }
        }
        // Pure second-order mode, after eliminating the three pure first
        // modes via the birth gradient observations. Subtracting raw Hermite
        // coefficients here would lose the O(t^2) width remainder.
        const double e2=-2*A*((1+A)/D)*a1 + A*A*(S/D)*adot;
        low[i]={P*P+e+rem,2*P*dp+de*da+dh,2*P*p1+e2+rem2};
        observation[i]=invRoot2*h.cwiseProduct(lengthY);
    }
    for (int i=0; i<3; ++i) {
        const Vector3 h=gradientY[i]/lengthY[i];
        for (int j=0; j<3; ++j) {
            const double w=h[j]*displacement[j];
            low[i].value+=w*shape[j].value;
            low[i].gradient+=h[j]*shape[j].value*Vector3::Unit(j)+w*shape[j].gradient;
            low[i].axial+=w*shape[j].axial;
        }
        low[i].value*=invRoot2;
        low[i].gradient*=invRoot2;
        low[i].axial*=invRoot2;
    }
    // The low residual above is in oblique coordinates. Givens rotations
    // remove the observed part and express it in an orthonormal complement.
    for (int j=0; j<3; ++j) {
        Vector3 top=Vector3::Unit(j);
        Coefficient observed{0,Vector3::Zero(),0};
        for (int i=0; i<3; ++i) {
            const double b=observation[i][j];
            if (b==0) continue;
            const double norm=std::hypot(top[j],b);
            const double c=top[j]/norm, sn=b/norm;
            const Vector3 rotated=c*top+sn*observation[i];
            observation[i]=c*observation[i]-sn*top;
            top=rotated;
            const Coefficient next=combine(c,observed,sn,low[i]);
            low[i]=combine(c,low[i],-sn,observed);
            observed=next;
        }
    }

    // Generating function exp(p*z + s*z^2/2) for the normalized 1D
    // coefficients, after removing the common degree-zero factor C.
    for (int i=0; i<3; ++i) {
        auto &t=terms[i];
        t[0]={1,Vector3::Zero(),0};
        t[1]=first[i];
        for (int n=2; n<=degree; ++n) {
            auto u=product(first[i],t[n-1]);
            if (n%2==0) u.axial+=u.value/age;
            t[n]=combine(1/roots[n],u,roots[n-1]/roots[n],product(shape[i],t[n-2]));
        }
    }
    constexpr int maxCount=(maxDegree+1)*(maxDegree+2)*(maxDegree+3)/6;
    std::array<Coefficient,maxCount> residual;
    int count=0;
    for (const auto &q : low) residual[count++]=q;
    // Mixed second-order and all higher modes are already orthogonal to the
    // birth observations. Retain their spatial gradients and dq/dt-2*q/t.
    for (int i=0; i<=degree; ++i) {
        for (int j=0; i+j<=degree; ++j) {
            const auto xy=product(terms[0][i],terms[1][j]);
            for (int k=0; i+j+k<=degree; ++k) {
                if (i+j+k<=1 || (i+j+k==2 && (i==2 || j==2 || k==2))) continue;
                auto q=product(xy,terms[2][k]);
                const int order=(i+1)/2+(j+1)/2+(k+1)/2;
                q.axial+=(order-2)*q.value/age;
                residual[count++]=q;
            }
        }
    }
    double vf=0, slope=0;
    Vector3 fg=Vector3::Zero();
    for (int i=0; i<count; ++i) {
        const auto &q=residual[i];
        vf+=q.value*q.value;
        fg+=q.value*q.gradient;
        slope+=q.value*q.axial;
    }
    const Vector3 beta=fg/vf;
    slope/=vf;
    Matrix3 gg=Matrix3::Zero();
    double vv=0;
    for (int i=0; i<count; ++i) {
        const auto &q=residual[i];
        const Vector3 g=q.gradient-beta*q.value;
        gg.noalias()+=g*g.transpose();
        const double u=q.axial-slope*q.value;
        vv+=u*u;
    }
    const double c2=std::exp(2*logC);
    out={c2*vf,dlogC+beta,c2*gg,c2*vv};
    return true;
}

} // namespace Tungsten::ConditionalGpis
