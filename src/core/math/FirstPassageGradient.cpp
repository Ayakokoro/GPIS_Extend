#include "FirstPassageGradient.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <sstream>
#include <iomanip>
#include <type_traits>

namespace Tungsten::FirstPassageGradient {
namespace {
namespace cg = ConditionalGpis;
constexpr double Infinity = std::numeric_limits<double>::infinity();

cg::NumericPolicy quadraturePolicy() {
    cg::NumericPolicy p;
    p.absoluteTolerance=1e-12;
    p.relativeTolerance=2e-9;
    return p;
}
double checkedIntegral(const std::function<double(double)> &f,double a,double b) {
    const auto result=cg::integrateFinite(f,a,b,quadraturePolicy());
    if(result.status!=cg::NumericStatus::Ok)
        throw std::runtime_error("FP gradient quadrature did not converge");
    return result.value;
}
double logCdfOffset(double z,double delta) {
    if(z < -8 && z+delta < -8)
        return -0.5*delta*(2*z+delta)+
            std::log(cg::normalPdfOverCdf(z)/cg::normalPdfOverCdf(z+delta));
    return cg::normalLogCdf(z+delta)-cg::normalLogCdf(z);
}

struct Statistics {
    double mean,sd;
    std::array<AnchorCandidate,5> candidates{};
    int count = 0, selected = 0;
};

struct ConditioningPrecisionLoss : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The unique positive root of x^3-P*x-Q=0, Q>0. Scaling bounds the
// discriminant; the P<0 branch avoids subtracting nearly equal cube roots.
double positiveCubicRoot(double P,double Q) {
    const double scale=std::max(std::sqrt(std::abs(P)),std::cbrt(Q));
    const double p=P/scale/scale,q=Q/scale/scale/scale;
    const double third=p/3,discriminant=q*q/4-third*third*third;
    double x;
    if(discriminant<0) {
        const double a=std::sqrt(third);
        x=2*a*std::cos(std::acos(std::clamp(q/(2*a*a*a),-1.0,1.0))/3);
    } else {
        const double u=std::cbrt(q/2+std::sqrt(discriminant));
        if(p>=0) x=u+third/u;
        else {
            const double v=-third/u;
            x=q/(u*u+u*v+v*v);
        }
    }
    return scale*x;
}

double anchorPosition(const Input &in,double mean,double sd) {
    // Magnitude of the uncorrected negative-flux Gaussian mode. Evaluate
    // the positive-mean tail without catastrophic cancellation.
    const double m=mean/sd,root=std::hypot(m,2.0);
    const double speed=sd*(m>=0 ? 2/(root+m) : (root-m)/2);
    const double P=(4*in.birthValue/in.distance+in.birthDerivative)/speed;
    const double Q=(2*in.birthValue/in.distance)/speed;
    // Surface starts have Q=0 and P>0: discard the zero root explicitly.
    const double x=in.birthValue==0 ? std::sqrt(P) : positiveCubicRoot(P,Q);
    const double theta=x>1 ? 1-1/(1+x) : x/(1+x);
    const double s=std::max(in.distance*theta,in.distance-1.5/in.gamma);
    if(!(s>0 && s<in.distance) || !std::isfinite(s))
        throw std::runtime_error("invalid FP single-anchor position");
    return s;
}

// Small separations have a fourth-order endpoint variance and an eighth-order
// bridge variance. Evaluate those Schur complements in 100 decimal digits,
// rather than clipping a negative variance or changing the physical distance.
template<class Real> Statistics statistics(const Input &in,AnchorSelection selection) {
    using std::exp; using std::sqrt;
    const Real t=Real(in.gamma)*in.distance;
    const Real df=(Real(in.birthValue)-in.birthMean)/in.sigma;
    const Real dg=(Real(in.birthDerivative)-in.birthMeanDerivative)/(Real(in.sigma)*in.gamma);
    auto ff=[](const Real &a,const Real &b)->Real {
        using std::exp;
        return exp(-(a-b)*(a-b)/2)-exp(-(a*a+b*b)/2)*(1+a*b);
    };
    auto fg=[](const Real &a,const Real &b)->Real {
        using std::exp;
        return -(b-a)*exp(-(b-a)*(b-a)/2)+exp(-(a*a+b*b)/2)*(b*(1+a*b)-a);
    };
    auto mf=[&](const Real &q,double mean)->Real {
        return Real(mean)/in.sigma+exp(-q*q/2)*(df+q*dg);
    };
    const Real fvar=ff(t,t),fcg=fg(t,t),fm=mf(t,in.mean(in.distance));
    const Real gm=Real(in.endMeanDerivative)/(Real(in.sigma)*in.gamma)+
        exp(-t*t/2)*(-t*df+(1-t*t)*dg);
    const Real gvar=1-(1-t*t+t*t*t*t)*exp(-t*t);
    const Real conditionalMean=gm-fcg*fm/fvar;
    const Real conditionalVariance=gvar-fcg*fcg/fvar;
    if(!(fvar>0) || !(conditionalVariance>0))
        throw ConditioningPrecisionLoss("FP gradient endpoint covariance lost precision");
    const Real sd=sqrt(conditionalVariance);
    Statistics result;
    result.mean=double(conditionalMean*in.sigma*in.gamma);
    result.sd=double(sd*in.sigma*in.gamma);
    const double c0=-result.mean/result.sd,root=std::hypot(c0,2.0);
    const double u0=c0>=0 ? -2/(c0+root) : .5*(c0-root);
    const double begin=std::max(0.0,in.distance-1.5/in.gamma);
    result.count=selection==AnchorSelection::Fixed5Snr ? 5 : 1;
    for(int i=0;i<result.count;++i) {
        auto &candidate=result.candidates[i];
        candidate.distance=selection==AnchorSelection::Fixed5Snr ?
            begin+(in.distance-begin)*((i+.5)/5) : anchorPosition(in,result.mean,result.sd);
        if(!(candidate.distance>0 && candidate.distance<in.distance))
            throw std::runtime_error("invalid FP single-anchor position");
        candidate.mean=in.mean(candidate.distance);
        const Real s=Real(in.gamma)*candidate.distance;
        const Real k=ff(s,t),c=fg(s,t)-k*fcg/fvar;
        const Real variance=ff(s,s)-k*k/fvar-c*c/conditionalVariance;
        // Retry the same candidates at high precision instead of clipping a
        // small Schur complement or silently skipping a difficult candidate.
        if(!(variance>0) || (std::is_same<Real,double>::value && variance<Real(1e-10)))
            throw ConditioningPrecisionLoss("FP gradient bridge covariance lost precision");
        const Real bridgeSd=sqrt(variance);
        candidate.p=double((mf(s,candidate.mean)-k*fm/fvar)/bridgeSd);
        candidate.q=double(c/(sd*bridgeSd));
        candidate.score=std::fma(candidate.q,u0,candidate.p);
        if(!std::isfinite(candidate.p) || !std::isfinite(candidate.q) ||
            !std::isfinite(candidate.score) || candidate.q>1e-7)
            throw std::runtime_error("invalid FP single-anchor bridge statistics");
        // Keep the earliest candidate on a tie. Selection is deterministic
        // and remains fixed throughout all gradient proposals at this event.
        if(candidate.score<result.candidates[result.selected].score) result.selected=i;
    }
    return result;
}
} // namespace

const char *anchorSelectionName(AnchorSelection selection) {
    switch(selection) {
        case AnchorSelection::Fixed5Snr: return "fixed5_snr";
        case AnchorSelection::Hermite: return "hermite";
    }
    throw std::invalid_argument("invalid FP anchor selection");
}
AnchorSelection parseAnchorSelection(const std::string &name) {
    if(name=="fixed5_snr") return AnchorSelection::Fixed5Snr;
    if(name=="hermite") return AnchorSelection::Hermite;
    throw std::invalid_argument("fp_gradient.anchor_selection must be fixed5_snr or hermite");
}

Law::Law(const Input &in,bool bridgeCorrection,AnchorSelection selection)
    : _bridge(bridgeCorrection),_selection(selection) {
    anchorSelectionName(selection); // Reject invalid enum values as well as JSON strings.
    if(!(in.sigma>0) || !(in.gamma>0) || !(in.distance>0) || !in.mean ||
        !std::isfinite(in.sigma) || !std::isfinite(in.gamma) || !std::isfinite(in.distance) ||
        !(in.birthValue>=0) || !std::isfinite(in.birthValue) ||
        !std::isfinite(in.birthDerivative) || !std::isfinite(in.birthMean) ||
        !std::isfinite(in.birthMeanDerivative) || !std::isfinite(in.endMeanDerivative) ||
        (in.birthValue==0 && !(in.birthDerivative>0)))
        throw std::invalid_argument("FP gradient requires a positive exterior or outgoing surface start");
    using HighPrecision=boost::multiprecision::cpp_dec_float_100;
    Statistics s;
    if(in.gamma*in.distance<0.5) s=statistics<HighPrecision>(in,selection);
    else {
        try { s=statistics<double>(in,selection); }
        catch(const ConditioningPrecisionLoss &) { s=statistics<HighPrecision>(in,selection); }
    }
    _mean=s.mean; _sd=s.sd;
    _candidates=s.candidates; _candidateCount=s.count; _selectedAnchor=s.selected;
    const auto &anchor=_candidates[_selectedAnchor];
    _anchorDistance=anchor.distance;
    if(!std::isfinite(_mean) || !(_sd>0) || !std::isfinite(_sd))
        throw std::runtime_error("invalid FP single-anchor bridge statistics");
    if(_bridge) { _p=anchor.p; _q=anchor.q; }

}

double Law::logShapeRatio(double x,double center) const {
    if(!(x>0)) return -Infinity;
    const double delta=x-center,u=-x-_mean/_sd,u0=-center-_mean/_sd;
    return std::log(x/center)+0.5*delta*(u+u0)+
        (_bridge ? logCdfOffset(std::fma(_q,u0,_p),-_q*delta) : 0);
}
double Law::shapeDerivative(double x) const {
    const double u=-x-_mean/_sd;
    return 1/x+(_bridge ? std::fma(-_q,cg::normalPdfOverCdf(std::fma(_q,u,_p)),u) : u);
}
double Law::mode() const {
    double lo=0,hi=std::max(1.0,1-_mean/_sd);
    for(int i=0;shapeDerivative(hi)>0 && i<1024;++i) hi*=2;
    if(!std::isfinite(hi) || !(shapeDerivative(hi)<0))
        throw std::runtime_error("FP gradient mode not bracketed");
    for(int i=0;i<160;++i) {
        const double mid=0.5*(lo+hi);
        if(mid==lo || mid==hi) break;
        if(shapeDerivative(mid)>0) lo=mid; else hi=mid;
    }
    return 0.5*(lo+hi);
}

double Law::sampleEnvelope(cg::Random &rng) const {
    const double m=mode();
    // Tangents bound a log-concave density. A constant middle interval plus
    // exponential tails remains valid even when the original Rice proposal
    // almost never survives. Failed draws never discard a medium collision.
    double width=std::min(m*0.5,1.0);
    for(int i=0;i<100 && logShapeRatio(m+width,m)<-2;++i) width*=0.5;
    const double l=m-width,r=m+width,dl=shapeDerivative(l),dr=shapeDerivative(r);
    if(!(l>0 && dl>0 && dr<0)) throw std::runtime_error("invalid FP sampling envelope");
    const double fl=logShapeRatio(l,m),fr=logShapeRatio(r,m);
    const double left=std::exp(fl)*(-std::expm1(-dl*l))/dl;
    const double middle=r-l,right=std::exp(fr)/(-dr),total=left+middle+right;
    for(int i=0;i<100000;++i) {
        const double choose=rng.openUniform01()*total;
        double x,envelope;
        if(choose<left) {
            x=l+std::log1p(-(1-rng.openUniform01())*(-std::expm1(-dl*l)))/dl;
            envelope=fl+dl*(x-l);
        } else if(choose<left+middle) {
            x=l+middle*rng.openUniform01(); envelope=0;
        } else {
            x=r+std::log(rng.openUniform01())/dr; envelope=fr+dr*(x-r);
        }
        const double acceptance=logShapeRatio(x,m)-envelope;
        if(acceptance>1e-6) {
            std::ostringstream msg;msg<<std::setprecision(17)<<"FP gradient envelope violated: mean="<<_mean<<" sd="<<_sd<<" p="<<_p<<" q="<<_q<<" mode="<<m<<" x="<<x<<" width="<<width<<" acceptance="<<acceptance;
            throw std::runtime_error(msg.str());
        }
        if(std::log(rng.openUniform01())<=acceptance) return -_sd*x;
    }
    throw std::runtime_error("FP gradient sampling did not converge");
}

double Law::sample(cg::Random &rng,bool *usedFallback) const {
    if(usedFallback) *usedFallback=false;
    for(int i=0;i<32;++i) {
        const double v=cg::sampleNegativeFluxNormal(_mean,_sd,rng);
        if(!_bridge || std::log(rng.openUniform01())<=
            cg::normalLogCdf(std::fma(_q,(v-_mean)/_sd,_p))) return v;
    }
    if(usedFallback) *usedFallback=true;
    return sampleEnvelope(rng);
}

Law::Integration Law::integration() const {
    Integration r;
    r.mode=mode();
    r.width=std::min(1.0,0.5*r.mode);
    for(int i=0;i<100 && logShapeRatio(r.mode+r.width,r.mode)<-2;++i) r.width*=0.5;
    r.lo=std::max(-r.mode/r.width,-1.0); r.hi=1;
    while(r.lo>-r.mode/r.width && logShapeRatio(r.mode+r.width*r.lo,r.mode)>-45)
        r.lo=std::max(-r.mode/r.width,2*r.lo);
    while(logShapeRatio(r.mode+r.width*r.hi,r.mode)>-45) r.hi*=2;
    r.mass=integrate(r,r.lo,0)+integrate(r,0,r.hi);
    if(!(r.mass>0)) throw std::runtime_error("FP gradient normalization failed");
    return r;
}
double Law::integrate(const Integration &r,double a,double b) const {
    if(!(b>a)) return 0;
    return checkedIntegral([&](double z) { return std::exp(logShapeRatio(r.mode+r.width*z,r.mode)); },a,b);
}
double Law::pdf(double v) const {
    if(!(v<0)) return 0;
    const auto r=integration();
    return std::exp(logShapeRatio(-v/_sd,r.mode))/(r.mass*r.width*_sd);
}
double Law::cdf(double v) const {
    if(v>=0) return 1;
    const auto r=integration();
    const double z=std::clamp((-v/_sd-r.mode)/r.width,r.lo,r.hi);
    const double mass=z<0 ? integrate(r,z,0)+integrate(r,0,r.hi) : integrate(r,z,r.hi);
    return std::clamp(mass/r.mass,0.0,1.0);
}

} // namespace Tungsten::FirstPassageGradient
