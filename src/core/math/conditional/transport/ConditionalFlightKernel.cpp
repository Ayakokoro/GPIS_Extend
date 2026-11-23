// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "NonstationarySegment.h"
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include "math/ConditionalProfile.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/SmallGaussian.h"
#include "math/ClassicGlobalMath.hpp"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>
#include <boost/math/special_functions/erf.hpp>

namespace Tungsten::ConditionalGpis {
namespace {

constexpr double boundGuard = 1.0 + 1e-11;
namespace bounds = MajorantBounds;

double upward(double x) {
    if (!(x >= 0.0) || !std::isfinite(x))
        throw NumericError(NumericStatus::InvalidMajorant, "non-finite conditional majorant");
    const double result=bounds::up(x * boundGuard);
    if (!std::isfinite(result))
        throw NumericError(NumericStatus::InvalidMajorant, "conditional majorant overflow");
    return result;
}

using bounds::scaledExponentialRemainders;

struct SeFactors { double valueVariance, slopeVariance, regression; };

SeFactors seFactors(double a, double t) {
    const double u=a*t*t;
    if (u<=1.0) {
        const auto e=scaledExponentialRemainders(u);
        return {std::exp(-u)*u*u*e.first, u*u*e.second/e.first,
                1.0/(t*e.first)};
    }
    const double r=std::exp(-u);
    if (r==0.0) return {1.0,1.0,0.0};
    const double v=-std::expm1(-u)-u*r;
    return {v, (1.0-(2.0+u*u)*r+r*r)/v, a*t*u*r/v};
}

// M(z)=pdf(z)/cdf(z) and H(z)=E[(-z-N(0,1))_+] both decrease.
// A left endpoint therefore bounds the entire bin. Use binary-exact knots
// and floor before shifting the index, including for small negative inputs.
double logFluxUpper(double meanLower,double stddevUpper);
struct NormalEnvelopeTable {
    static constexpr int scale=64, limit=12, count=2*limit*scale+1;
    std::array<double,count> mills,flux;
    NormalEnvelopeTable() {
        for(int i=0;i<count;++i) {
            const double z=double(i-limit*scale)/scale;
            mills[i]=upward(normalPdfOverCdf(z));
            flux[i]=upward(negativePartMean(z,1).value);
        }
    }
    static int index(double z) { return int(std::floor(z*scale))+limit*scale; }
};

double tabulatedCrossingBound(double z,double meanLower,double deviation) {
    static const NormalEnvelopeTable table;
    const double q=bounds::down(meanLower/deviation);
    // For a negative tail M(z)<=1-z, H(q)<=-q+pdf(0).
    // Preserve the log-domain evaluation for extreme scales/positive tails.
    if(z<=12 && q<=12 && std::isfinite(q) && std::isfinite(z)) {
        const double m=z < -12 ? 1-z : table.mills[table.index(z)];
        const double h=q < -12 ? 0.4-q : table.flux[table.index(q)];
        const double maximum=m*(deviation*h);
        if(std::isnormal(maximum)) return upward(maximum);
    }
    const double logM=z < -8 ? std::log1p(-z) : normalLogPdf(z)-normalLogCdf(z);
    return upward(std::exp(logM+logFluxUpper(meanLower,deviation)));
}

double slopeStddevBound(double sigma2, double a, double end) {
    // N(q)/V(q) increases with q. Use its endpoint, including near birth.
    return upward(bounds::slopeStddevUpper(sigma2,a,end));
}

std::pair<double,double> quadraticRange(double c0, double c1, double c2,
                                        double lo, double hi) {
    const auto value=[&](double t) { return c0+t*(c1+t*c2); };
    double lower=std::min(value(lo),value(hi)), upper=std::max(value(lo),value(hi));
    if (c2!=0.0) {
        const double root=-c1/(2.0*c2);
        if (root>lo && root<hi) { lower=std::min(lower,value(root)); upper=std::max(upper,value(root)); }
    }
    const double error=64.0*std::numeric_limits<double>::epsilon()*
        (std::abs(c0)+std::abs(c1)*hi+std::abs(c2)*hi*hi);
    return {lower-error,upper+error};
}

double envelopeMaximum(double logC1, double c2, double power, double exponent, double end) {
    // Maximize C1 t^-power exp(-c2/t^exponent), including its zero limit at birth.
    if (!(c2>0.0)) throw NumericError(NumericStatus::InvalidMajorant, "birth envelope lost positive separation");
    const double t=std::min(end,std::pow(exponent*c2/power,1.0/exponent));
    const double logValue=logC1-power*std::log(t)-c2/std::pow(t,exponent);
    return upward(std::exp(logValue));
}

// Only the bound's CDF calls are timed here; endpoint/gradient sampling and
// candidate extinction retain their existing scopes. Ordinary builds keep
// the original direct calls, including the original CDF implementation.
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
double boundNormalCdf(double z) {
    Profile::Scope detail(Profile::Stage::BoundNormalCdf);
    return normalCdf(z);
}
double boundNormalLogCdf(double z) {
    Profile::Scope detail(Profile::Stage::BoundNormalCdf);
    return normalLogCdf(z);
}
#else
#define boundNormalCdf normalCdf
#define boundNormalLogCdf normalLogCdf
#endif

double logFluxUpper(double meanLower,double stddevUpper) {
    Profile::Scope detail(Profile::Stage::BoundFlux);
    if(stddevUpper==0)
        return meanLower<0 ? std::log(upward(-meanLower)) : -std::numeric_limits<double>::infinity();
    // Negative-part expectation increases as its mean falls or its standard
    // deviation rises. Use its exact value where the subtraction is stable.
    const double q=meanLower/stddevUpper;
    return meanLower<0.0
        ? std::log(upward(-meanLower*boundNormalCdf(-q)+stddevUpper*normalPdf(-q)))
        : q<=4.0
            ? std::log(stddevUpper)+std::log(upward(normalPdf(q)-q*boundNormalCdf(-q)))
            : std::log(stddevUpper)+normalLogPdf(q)+std::log(boundGuard);
}

struct MomentBounds {
    bounds::Interval sd,regression;
    double slopeMax,kappaMax;
    bool normalizedOnly;
};

MomentBounds prepareMomentBounds(double sigma2, double a, double lo, double hi,
                                 bool normalizedOnly) {
    Profile::Scope profile(Profile::Stage::MomentBound);
    Profile::Scope detail(Profile::Stage::BoundMomentPrepare);
    const auto left=seFactors(a,lo), right=seFactors(a,hi);
    const double sdmin=std::sqrt(sigma2*left.valueVariance)/boundGuard;
    const double sdmax=upward(std::sqrt(sigma2*right.valueVariance));
    if(!(sdmin>0))
        throw NumericError(NumericStatus::NeedHigherPrecision,"conditional interval variance underflow");
    // The regression s'/s decreases along a forward birth flight.
    const bounds::Interval regression(std::max(0.0,bounds::down(right.regression/boundGuard)),
                                      upward(left.regression));
    const auto endpoint=bounds::seVarianceBounds(bounds::squaredKernelDistance(a,hi));
    const double slopeMax=normalizedOnly ? 0.0 : upward(bounds::sqrt(bounds::Interval(sigma2)*bounds::Interval(a)*
        bounds::Interval(std::min(1.0,endpoint.slope.upper))).upper);
    const double kappaMax=upward(bounds::sqrt(bounds::Interval(a)*
        bounds::Interval(std::min(1.0,endpoint.normalized.upper))).upper);
    return {{sdmin,sdmax},regression,slopeMax,kappaMax,normalizedOnly};
}

double applyMomentBounds(double fmin, double fmax, double kmin,
                         const MomentBounds &moments, bool allowInfinite=false) {
    Profile::Scope profile(Profile::Stage::MomentBound);
    Profile::Scope detail(Profile::Stage::BoundMomentApply);
    if(std::isnan(fmin) || std::isnan(fmax) || std::isnan(kmin))
        throw NumericError(NumericStatus::InvalidMajorant,"invalid conditional mean bounds");
    if(allowInfinite && (!std::isfinite(fmin) || !std::isfinite(fmax) || !std::isfinite(kmin)))
        return std::numeric_limits<double>::infinity();
    const double zmin=(bounds::Interval(fmin)/moments.sd).lower;
    const double muMin=(bounds::Interval(kmin)-moments.regression*bounds::Interval(fmax)).lower;
    if(allowInfinite && (!std::isfinite(zmin) || !std::isfinite(muMin)))
        return std::numeric_limits<double>::infinity();
    const double logMills=[&]() {
        Profile::Scope mills(Profile::Stage::BoundMills);
        return zmin>-8.0
            ? normalLogPdf(zmin)-boundNormalLogCdf(zmin) : std::log1p(-zmin);
    }();
    // Each form separately bounds the same hazard. Omitting the moment form
    // may loosen the envelope but does not change pointwise extinction.
    const double oldLog=moments.normalizedOnly ? 0.0 :
        logMills-std::log(moments.sd.lower)+logFluxUpper(muMin,moments.slopeMax);
    // z'=meanAtZero/s; division must use sdmax for a positive lower mean.
    // N/V^2 increases from 1/3 to 1, so kappa is bounded at the right endpoint.
    const double derivativeMin=(bounds::Interval(muMin)/moments.sd).lower;
    const double kappaMax=moments.kappaMax;
    const double normalizedLog=logMills+logFluxUpper(derivativeMin,kappaMax);
    if(std::isnan(oldLog) || std::isnan(normalizedLog))
        throw NumericError(NumericStatus::InvalidMajorant,"invalid conditional moment bound");
    const double upper=std::exp(moments.normalizedOnly ? normalizedLog : std::min(oldLog,normalizedLog));
    // A loose cheap enclosure may overflow even when the fine bound is usable.
    if(allowInfinite && upper>std::numeric_limits<double>::max()/boundGuard)
        return std::numeric_limits<double>::infinity();
    return upward(upper);
}

double momentIntervalMajorant(double fmin, double fmax, double kmin,
                              double sigma2, double a, double lo, double hi, bool normalizedOnly) {
    return applyMomentBounds(fmin,fmax,kmin,prepareMomentBounds(sigma2,a,lo,hi,normalizedOnly));
}

#if !defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) || !TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
#undef boundNormalCdf
#undef boundNormalLogCdf
#endif

} // namespace

FlightState startConditionalExterior(const GPSSField& field, const Point3& entry,
                                     const Vector3& direction, Random& rng) {
    if (field.nonstationary)
        return startNonstationaryExterior(*field.nonstationary,*field.mean,entry,direction,rng);
    const PointPrior prior = field.pointPrior(entry);
    const double sigma = std::sqrt(prior.varianceF);
    // At one point the SE value and gradient are independent.
    const double logTail = normalLogCdf(prior.meanF / sigma);
    const double z = -normalQuantileFromLogCdf(logTail +
                                                std::log(rng.openUniform01()));
    FlightState state = startExternalFlight(entry, direction);
    state.birthValue = std::max(std::nextafter(0.0, 1.0), prior.meanF + sigma * z);
    state.birthGradient = sampleGaussianPSD(
        Gaussian<3>{prior.meanG, prior.covarianceG}, rng);
    return state;
}

double ConditionalIntegralMajorant::value(double age) const {
    if(gaussianLinear) {
        const double z=z0-beta*(age-origin),c=std::sqrt(2/kPi);
        return constant*(z>=0 ? c*std::exp(-.5*z*z) : c-z);
    }
    return constant+regressionCoefficient*seFactors(inverseLength2,age).regression;
}

double ConditionalIntegralMajorant::integral(double begin,double end) const {
    if(!(begin>0 && end>=begin))
        throw NumericError(NumericStatus::InvalidInput,"invalid integral majorant ages");
    if(end==begin) return 0;
    if(gaussianLinear) {
        if(beta==0) return value(begin)*(end-begin);
        const double za=z0-beta*(begin-origin),zb=za-beta*(end-begin),c=std::sqrt(2/kPi);
        if(za<=0) return constant*(end-begin)*(c-za+.5*beta*(end-begin));
        const double positiveLength=std::min(end-begin,za/beta),zp=std::max(0.,za-beta*positiveLength);
        double depth;
        if(beta*positiveLength*(1+za)<1e-3) {
            // Stable difference of two almost equal erfc values: three-point
            // Gauss integrates this tiny Gaussian interval to double accuracy.
            const double m=.5*(za+zp),h=.5*(za-zp)*std::sqrt(3./5);
            depth=constant*c*positiveLength*(5*std::exp(-.5*(m-h)*(m-h))+
                8*std::exp(-.5*m*m)+5*std::exp(-.5*(m+h)*(m+h)))/18;
        } else depth=constant/beta*(std::erfc(zp/kSqrtTwo)-std::erfc(za/kSqrtTwo));
        if(zb<0) depth+=constant*(-zb/beta)*(c-.5*zb);
        return depth;
    }
    double logRatio=0;
    if(regressionCoefficient>0) {
        const double u=inverseLength2*begin*begin,v=inverseLength2*end*end;
        const double delta=inverseLength2*(end-begin)*(end+begin);
        if(v<=1) {
            const double e0=scaledExponentialRemainders(u).first;
            const double e1=scaledExponentialRemainders(v).first;
            // s(t) proportional to t^2*exp(-u/2)*sqrt(E2(u)).
            logRatio=2*std::log1p((end-begin)/begin)-0.5*delta+
                0.5*std::log1p((e1-e0)/e0);
        } else {
            // V(v)-V(u)=exp(-u)*(u*(1-exp(-delta))+V(delta)).
            // This avoids subtracting two variances rounded to one in the tail.
            const double vd=seFactors(1,std::sqrt(delta)).valueVariance;
            const double difference=std::exp(-u)*(u*(-std::expm1(-delta))+vd);
            logRatio=0.5*std::log1p(difference/seFactors(inverseLength2,begin).valueVariance);
        }
    }
    return constant*(end-begin)+regressionCoefficient*std::max(0.0,logRatio);
}

double ConditionalIntegralMajorant::inverseIntegral(double begin,double end,double depth) const {
    if(depth==0) return begin;
    const double total=integral(begin,end);
    if(!(depth>0 && depth<=total) || !std::isfinite(total))
        throw NumericError(NumericStatus::InvalidInput,"invalid inverse majorant optical depth");
    if(depth==total) return end;
    if(gaussianLinear) {
        if(beta==0) return begin+depth/value(begin);
        const double za=z0-beta*(begin-origin),c=std::sqrt(2/kPi);
        double result;
        if(za>0 && depth<constant/beta*std::erf(za/kSqrtTwo)) {
            const double p=std::erfc(za/kSqrtTwo)+beta*depth/constant;
            result=begin+(za-kSqrtTwo*boost::math::erfc_inv(p))/beta;
        } else {
            const double offset=za>0 ? za/beta:0;
            const double remaining=za>0 ? depth-constant/beta*std::erf(za/kSqrtTwo):depth;
            const double rate=constant*(c-std::min(0.,za));
            result=begin+offset+2*remaining/(rate+std::sqrt(rate*rate+2*constant*beta*remaining));
        }
        // Resolve cancellation for tiny steps. The ordinary case uses the
        // closed inverse; this fallback maintains the optical-depth contract.
        double left=begin,right=end;
        for(int i=0;i<64;++i) {
            if(!(result>left && result<right)) result=left+.5*(right-left);
            const double residual=integral(begin,result)-depth;
            if(std::abs(residual)<=64*std::numeric_limits<double>::epsilon()*depth) return result;
            if(residual<0) left=result;else right=result;
            if(!(std::nextafter(left,right)<right)) return right;
            result-=residual/value(result);
        }
        throw NumericError(NumericStatus::NeedHigherPrecision,"Gaussian envelope inverse did not converge");
    }
    if(regressionCoefficient==0) return begin+depth/constant;
    double left=begin,right=end;
    double x=begin+(end-begin)*(depth/total);
    for(int i=0;i<96;++i) {
        const double residual=integral(begin,x)-depth;
        if(std::abs(residual)<=32*std::numeric_limits<double>::epsilon()*depth) return x;
        if(residual<0) left=x; else right=x;
        const double middle=left+0.5*(right-left);
        if(!(middle>left && middle<right)) return right;
        const double next=x-residual/value(x);
        x=next>left && next<right ? next : middle;
    }
    throw NumericError(NumericStatus::NeedHigherPrecision,"integrated majorant inverse did not converge");
}

ConditionalFlightKernel::ConditionalFlightKernel(const GPSSField& field,
                                                 const FlightState& state, bool normalizedBoundOnly,
                                                 double farBoundStartScale, double simpleBoundStartScale, bool tabulatedBounds)
    : FlightKernel(field, state), normalizedBoundOnly_(normalizedBoundOnly), tabulatedBounds_(tabulatedBounds),
      farBoundStartScale_(farBoundStartScale), simpleBoundStartScale_(simpleBoundStartScale) {
    Profile::Scope profile(Profile::Stage::KernelSetup);
    if (field.nonstationary)
        throw std::invalid_argument("stationary SE majorants require a stationary field; use ConditionalSegment for proc_nonstationary");
    if (!std::isfinite(farBoundStartScale_) || (farBoundStartScale_!=0 && farBoundStartScale_<2))
        throw NumericError(NumericStatus::InvalidInput,"far SE bound requires zero or a cutoff of at least two");
    if (!std::isfinite(simpleBoundStartScale_) || (simpleBoundStartScale_!=0 && simpleBoundStartScale_<2))
        throw NumericError(NumericStatus::InvalidInput,"simple SE bound requires zero or a cutoff of at least two");
    if (!field.kernel.supportsAnalyticConditionalTransport()) {
        throw std::invalid_argument(
            std::string("global_conditional is not implemented for kernel '") +
            covarianceKernelTypeName(field.kernel.type()) +
            "'; the common KernelJet interface is available for a future conditioner");
    }
    if (!state.birthGradient.allFinite() || !(state.birthValue>=0.0) || !std::isfinite(state.birthValue))
        throw std::invalid_argument("conditional flight requires a finite birth observation");
    birthMean_=field_.mean->evaluate(state.birthPosition);
    meanRay_=field_.mean->createRayCursor(state.birthPosition,state.direction);
    deltaGradient_=state.birthGradient-birthMean_.gradient;
    deltaValue_=state.birthValue-birthMean_.value;
    deltaSlope_=state.direction.dot(deltaGradient_);
    precisionDirection_=field.kernel.precision()*state.direction;
    inverseLength2_=state.direction.dot(precisionDirection_);
    sigma2_=field.kernel.sigma()*field.kernel.sigma();
    if (!(inverseLength2_>0.0))
        throw NumericError(NumericStatus::UnsupportedSingularFlight, "conditional flight requires positive ray precision");
    if (farBoundStartScale_>0)
        farBoundBegin_=(bounds::Interval(farBoundStartScale_)/
            bounds::sqrt(bounds::Interval(inverseLength2_))).upper;
    if (simpleBoundStartScale_>0)
        simpleBoundBegin_=(bounds::Interval(simpleBoundStartScale_)/
            bounds::sqrt(bounds::Interval(inverseLength2_))).upper;
}

ConditionalScalarStatistics ConditionalFlightKernel::scalarStatistics(double age) const {
    Profile::Scope profile(Profile::Stage::ScalarStatistics);
    return pointStatistics(age).scalar;
}

ConditionalFlightKernel::PointStatistics &ConditionalFlightKernel::pointStatistics(double age) const {
    if(pointCache_[recentPoint_].age==age) {
        Profile::count(Profile::Counter::PointStatisticsHits);
        return pointCache_[recentPoint_];
    }
    const unsigned other=1-recentPoint_;
    auto &point=pointCache_[other];
    if(point.age==age) {
        recentPoint_=other;
        Profile::count(Profile::Counter::PointStatisticsHits);
        return point;
    }
    if (!(age > 0.0 && age <= maximumAgeInDomain()))
        throw std::out_of_range("conditional flight age outside domain");
    Profile::count(Profile::Counter::PointStatisticsMisses);
    // Do not publish a partially computed entry if a provider throws.
    point.age=std::numeric_limits<double>::quiet_NaN();
    point.hasJet=false;
    const double u=inverseLength2_*age*age, r=std::exp(-0.5*u);
    const MeanRaySample target=[&]() {
        if(meanRay_) return meanRay_->sample(age);
        point.jet=meanJet(age);
        point.hasJet=true;
        return MeanRaySample{point.jet.value,state_.direction.dot(point.jet.gradient)};
    }();
    const double meanF=u<1e-3
        ? state_.birthValue+age*deltaSlope_+
          meanValueDifference(age)+
          std::expm1(-0.5*u)*(deltaValue_+age*deltaSlope_)
        : target.value+r*(deltaValue_+age*deltaSlope_);
    const double meanK=target.derivative+
        r*((1.0-u)*deltaSlope_-inverseLength2_*age*deltaValue_);
    const SeFactors factors=seFactors(inverseLength2_,age);
    point.scalar={meanF,sigma2_*factors.valueVariance,meanK,
            meanK-factors.regression*meanF,sigma2_*inverseLength2_*factors.slopeVariance};
    point.u=u;point.r=r;
    point.regression=factors.regression;point.slopeVariance=factors.slopeVariance;
    point.age=age;recentPoint_=other;
    return point;
}

HazardEvaluation ConditionalFlightKernel::evaluate(double age) const {
    Profile::Scope profile(Profile::Stage::Hazard);
    if (age == 0.0) return {exactZero()};
    const auto m = scalarStatistics(age);
    if (!(m.varianceF > 0.0))
        throw NumericError(NumericStatus::NeedHigherPrecision,"conditional variance underflow");
    const PositiveResult flux = negativePartMean(m.meanAtZero, std::sqrt(m.varianceAtZero));
    if (flux.status == NumericStatus::ExactZero) return {exactZero()};
    const double sd = std::sqrt(m.varianceF);
    const double z = m.meanF / sd;
    const double logRatio=z < -8.0 ? std::log(normalPdfOverCdf(z))
                                  : normalLogPdf(z)-normalLogCdf(z);
    return {positiveFromLog(logRatio - std::log(sd) +
                            flux.logValue)};
}

HitStatistics ConditionalFlightKernel::hitStatistics(double age) const {
    return {gradientGivenValueDeviation(age,-scalarStatistics(age).meanF)};
}

Gaussian<3> ConditionalFlightKernel::gradientGivenValueDeviation(double age, double deviation) const {
    Profile::Scope profile(Profile::Stage::CollisionGradient);
    auto &point=pointStatistics(age);
    const double u=point.u,r=point.r;
    if(!point.hasJet) {
        point.jet=meanJet(age);
        point.hasJet=true;
        Profile::count(Profile::Counter::PointGradientLoads);
    } else Profile::count(Profile::Counter::PointGradientReuses);
    const auto &target=point.jet;
    Gaussian<3> g;
    g.mean=target.gradient+r*deltaGradient_-precisionDirection_*
        (age*r*(deltaValue_+age*deltaSlope_)-point.regression*deviation/inverseLength2_);
    const Matrix3 longitudinal=precisionDirection_*precisionDirection_.transpose()/inverseLength2_;
    g.covariance=sigma2_*(-std::expm1(-u)*(field_.kernel.precision()-longitudinal)+
                         point.slopeVariance*longitudinal);
    g.covariance = 0.5 * (g.covariance + g.covariance.transpose());
    return g;
}

FlightState ConditionalFlightKernel::samplePositiveEndpoint(double age, Random& rng) const {
    Profile::Scope profile(Profile::Stage::EndpointSample);
    const auto m=scalarStatistics(age);
    if (!(m.varianceF > 0.0))
        throw NumericError(NumericStatus::NeedHigherPrecision,"conditional endpoint variance underflow");
    const double sd=std::sqrt(m.varianceF);
    const double z=-normalQuantileFromLogCdf(
        normalLogCdf(m.meanF/sd)+std::log(rng.openUniform01()));
    FlightState endpoint=startExternalFlight(state_.birthPosition+age*state_.direction,state_.direction);
    endpoint.birthValue=std::max(std::nextafter(0.0,1.0),m.meanF+sd*z);
    // Use the sampled deviation directly to avoid subtracting nearly equal
    // endpoint and mean values when the segment is much shorter than ell.
    endpoint.birthGradient=sampleGaussianPSD(gradientGivenValueDeviation(age,sd*z),rng);
    return endpoint;
}

MeanJet ConditionalFlightKernel::meanJet(double age) const {
    return meanRay_ ? meanRay_->evaluate(age) :
        field_.mean->evaluate(state_.birthPosition+age*state_.direction);
}

double ConditionalFlightKernel::scalarMeanValue(double age,bool transition) const {
    Profile::count(transition ? Profile::Counter::TransitionScalarQueries : Profile::Counter::LocalFarScalarQueries);
    return field_.mean->evaluateValue(state_.birthPosition+age*state_.direction);
}

double ConditionalFlightKernel::meanValueDifference(double age,double *error) const {
    if(error) *error=0;
    return meanRay_ ? meanRay_->valueDifference(age,error) :
        field_.mean->valueDifference(state_.birthPosition,age*state_.direction);
}

MeanRayBounds ConditionalFlightKernel::meanRayBounds(double lo,double hi,const SdfRayQueryOptions &options) const {
    return meanRay_ ? meanRay_->rayBounds(lo,hi,options) :
        field_.mean->rayBounds(state_.birthPosition,state_.direction,lo,hi,options);
}

double ConditionalFlightKernel::nextMeanBoundary(double lo,double hi) const {
    if(meanRay_) return meanRay_->nextBoundary(lo,hi);
    std::vector<double> knots;
    field_.mean->appendRayBreakpoints(state_.birthPosition,state_.direction,lo,hi,knots);
    return knots.empty() ? hi : *std::min_element(knots.begin(),knots.end());
}

double ConditionalFlightKernel::intervalMajorant(double lo, double hi,
        double tightenExpectedEvents, double parentMaximum) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    if (!(lo>0.0 && hi>lo && hi<=maximumAgeInDomain()) ||
       !(tightenExpectedEvents>=0.0) || !(parentMaximum>=0.0))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional bound interval");
    SdfRayQueryOptions queryOptions;
    queryOptions.computeBeginDerivative=false;
    const auto mean=[&]() {
        Profile::Scope detail(Profile::Stage::BoundSdfFine);
        return meanRayBounds(lo,hi,queryOptions);
    }();
    return boundedIntervalMajorant(mean,lo,hi,tightenExpectedEvents,parentMaximum);
}

double ConditionalFlightKernel::pointIntervalMajorant(double lo,double hi,double lipschitz,
        double parentMaximum) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::count(Profile::Counter::PointBoundQueries);
    if (!(lo>0 && hi>lo && hi<=maximumAgeInDomain()) ||
        !(lipschitz>=0) || !std::isfinite(lipschitz) || !(parentMaximum>=0))
        throw NumericError(NumericStatus::InvalidInput,"invalid point/Lipschitz bound interval");
    const auto mean=pointMeanBounds(lo,hi,lipschitz);
    if(!std::isfinite(mean.minimumValue) || !std::isfinite(mean.maximumValue) || !std::isfinite(mean.midpointError))
        return std::numeric_limits<double>::infinity();
    return boundedIntervalMajorant(mean,lo,hi,
        std::numeric_limits<double>::infinity(),parentMaximum);
}

MeanRayBounds ConditionalFlightKernel::pointMeanBounds(double lo,double hi,double lipschitz,bool scalarOnly,bool transition) const {
    const double middle=0.5*lo+0.5*hi;
    double value;
    {
        Profile::Scope point(Profile::Stage::BoundSdfPoint);
        value=scalarOnly ? scalarMeanValue(middle,transition) : meanValue(middle);
    }
    if(!std::isfinite(value))
        throw NumericError(NumericStatus::InvalidMajorant,"nonfinite point/Lipschitz mean");
    namespace b=MajorantBounds;
    const double slope=b::up(lipschitz*state_.direction.norm());
    const double half=b::up(0.5*(hi-lo));
    // Include coordinate construction, midpoint arithmetic, provider evaluation
    // and final enclosure roundoff. L is for the final transformed scalar field.
    const double error=b::up(512*std::numeric_limits<double>::epsilon()*
        (1+std::abs(value)+lipschitz*state_.birthPosition.norm()+slope*hi));
    const double radius=b::up(slope*half+error);
    MeanRayBounds mean;
    mean.minimumValue=b::down(value-radius);
    mean.maximumValue=b::up(value+radius);
    mean.minimumDerivative=-slope;
    mean.maximumDerivative=slope;
    mean.midpointValue=value;
    mean.midpointError=error;
    mean.certified=true;
    return mean;
}

double ConditionalFlightKernel::transitionIntervalMajorant(double lo,double hi,double lipschitz,
        double tightenExpectedEvents,double parentMaximum) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::Scope detail(Profile::Stage::BoundTransition);
    Profile::count(Profile::Counter::TransitionBoundQueries);
    if(!(lo>0 && hi>lo && hi<=maximumAgeInDomain()) ||
       !(lipschitz>=0) || !std::isfinite(lipschitz) ||
       !(tightenExpectedEvents>=0) || !(parentMaximum>=0))
        throw NumericError(NumericStatus::InvalidInput,"invalid transition midpoint interval");
    const auto mean=pointMeanBounds(lo,hi,lipschitz,true,true);
    // The reducer intersects the signed prior/correction enclosure with a
    // cone about the cancellation-safe conditional midpoint. Keep actual
    // interval moments here; tail envelopes discard that near-birth structure.
    return boundedIntervalMajorant(mean,lo,hi,tightenExpectedEvents,parentMaximum,false);
}

ConditionalIntegralMajorant ConditionalFlightKernel::transitionIntegralMajorant(
        double lo,double hi,double lipschitz) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::Scope detail(Profile::Stage::BoundTransition);
    Profile::count(Profile::Counter::TransitionBoundQueries);
    if(!(lo>0 && hi>lo && hi<=maximumAgeInDomain()) ||
       !(lipschitz>=0) || !std::isfinite(lipschitz))
        throw NumericError(NumericStatus::InvalidInput,"invalid integrated transition interval");
    const auto mb=pointMeanBounds(lo,hi,lipschitz,true,true);
    using I=bounds::Interval;
    const double a=inverseLength2_;
    const I times(lo,hi),exponential=bounds::exp(I(-0.5)*I(a)*times*times);
    const I correction=exponential*bounds::quadraticPolynomialRange(deltaValue_,deltaSlope_,0,lo,hi);
    const I slopeCorrection=exponential*bounds::quadraticPolynomialRange(
        deltaSlope_,-I(a)*I(deltaValue_),-I(a)*I(deltaSlope_),lo,hi);
    const double kmin=(I(mb.minimumDerivative)+slopeCorrection).lower;
    const double kmax=(I(mb.maximumDerivative)+slopeCorrection).upper;
    double fmin=(I(mb.minimumValue)+correction).lower;
    double fmax=(I(mb.maximumValue)+correction).upper;
    // Retain the cancellation-safe midpoint enclosure used by the old
    // transition certificate. It matters when birth and prior means cancel.
    const double middle=0.5*lo+0.5*hi,um=a*middle*middle;
    double differenceError=0;
    const double center=um<1e-3
        ? state_.birthValue+middle*deltaSlope_+meanValueDifference(middle,&differenceError)+
          std::expm1(-0.5*um)*(deltaValue_+middle*deltaSlope_)
        : mb.midpointValue+std::exp(-0.5*um)*(deltaValue_+middle*deltaSlope_);
    const double roundoff=256*std::numeric_limits<double>::epsilon()*
        (std::abs(mb.midpointValue)+std::abs(birthMean_.value)+std::abs(deltaValue_)+
         std::abs(middle*deltaSlope_)+1e-30);
    const double radius=upward(0.5*(hi-lo)*std::max(std::abs(kmin),std::abs(kmax))+
        roundoff+std::max(differenceError,mb.midpointError));
    fmin=std::max(fmin,bounds::down(center-radius));
    fmax=std::min(fmax,bounds::up(center+radius));
    const auto moments=prepareMomentBounds(sigma2_,a,lo,hi,true);
    const I z=I(fmin,fmax)/moments.sd;
    const auto logR=[](double x) {
        return x>-8 ? normalLogPdf(x)-normalLogCdf(x) : std::log1p(-x);
    };
    // A(d,kappa)<=(-d)+ + kappa/sqrt(2*pi), d>=kmin/s-gamma*z.
    // Keep z+R(z) together, bounding it by min(C, zmax+ R(max(0,zmin))).
    // C=sqrt(2/(pi*e)) follows from Phi(z)>=1/2 and max z*phi(z)=phi(1).
    const double slope=std::max(0.0,-kmin);
    const double base=(I(slope)/I(moments.sd.lower)+I(moments.kappaMax)*I(kInvSqrtTwoPi)).upper;
    const double c0=base==0 ? 0 : upward(std::exp(logR(z.lower)+std::log(base)));
    double c1=0;
    if(z.upper>0) {
        const double cap=std::sqrt(2/(kPi*std::exp(1.0)));
        c1=upward(std::exp(std::min(std::log(cap),std::log(z.upper)+logR(std::max(0.0,z.lower)))));
    }
    return {c0,c1,a};
}

bool ConditionalFlightKernel::polynomialZBounds(double lo,double hi,double &lower,double &upper) const {
    if(!meanRay_ || !(lo>0 && hi>lo) || !std::isfinite(hi)) return false;
    MeanRayPolynomial mean;
    if(!meanRay_->polynomial(lo,hi,mean)) return false;
    // exp(-a*t^2/2) = 1-a*t^2/2+a^2*t^4/8 + R, |R| <= (a*hi^2/2)^3/6.
    // Add this correction to the mean cubic BEFORE enclosing the polynomial.
    // All coefficients below use x=(t-lo)/(hi-lo), avoiding a global-age
    // expansion of a tiny spatial cell far from the conditioning point.
    constexpr double choose[6][6]={{1},{1,1},{1,2,1},{1,3,3,1},{1,4,6,4,1},{1,5,10,10,5,1}};
    const double a=inverseLength2_,h=hi-lo,e=-.5*a,e2=.125*a*a;
    const std::array<double,6> correction={deltaValue_,deltaSlope_,e*deltaValue_,e*deltaSlope_,
                                          e2*deltaValue_,e2*deltaSlope_};
    std::array<double,6> power{},lp{},hp{};
    lp[0]=hp[0]=1;
    for(int j=1;j<6;++j) {lp[j]=lp[j-1]*lo;hp[j]=hp[j-1]*h;}
    for(int k=0;k<6;++k) {
        for(int j=k;j<6;++j)power[k]+=correction[j]*choose[j][k]*lp[j-k]*hp[k];
        if(k<4)power[k]+=mean.coefficients[k];
    }
    constexpr double eps=std::numeric_limits<double>::epsilon();
    const double x=.5*a*hi*hi,affineScale=std::abs(deltaValue_)+hi*std::abs(deltaSlope_);
    double absoluteScale=affineScale*(1+x+.5*x*x);
    for(double c:mean.coefficients)absoluteScale+=std::abs(c);
    const double meanError=(mean.error+4096*eps*absoluteScale)*boundGuard;
    double wlo=INFINITY,whi=-INFINITY;
    for(int i=0;i<6;++i) {
        double numerator=0;
        for(int k=0;k<=i;++k)numerator+=power[k]*choose[i][k]/choose[5][k];
        // Positive Bernstein controls of t^2, elevated to degree five.
        const double denominator=lo*lo+2*lo*h*(double(i)/5)+h*h*(double(i*(i-1))/20);
        // A uniform mean error is a constant polynomial: shift every control
        // BEFORE division, retaining its correlation with the t^2 denominator.
        // Subtracting error/lo^2 after taking the minimum can be catastrophically
        // loose when a cell begins very close to the conditioning point.
        wlo=std::min(wlo,(numerator-meanError)/denominator);
        whi=std::max(whi,(numerator+meanError)/denominator);
    }
    const double affineMaximum=std::max(std::abs(deltaValue_+lo*deltaSlope_),std::abs(deltaValue_+hi*deltaSlope_));
    // Divide the Taylor remainder by t^2 pointwise first: its O(t^6)
    // numerator becomes O(t^4), bounded at hi rather than divided by lo^2.
    const double remainder=x*x*(a/12)*(affineMaximum+128*eps*affineScale);
    const double error=bounds::up(remainder*boundGuard+
        128*eps*std::max(std::abs(wlo),std::abs(whi)));
    // sd(t)/t^2 = sigma*a*sqrt(H(a*t^2)), H(u)=integral_0^1 x*exp(-u*x) dx.
    // H is positive and decreasing. Use the small-u series without subtracting
    // nearly equal values, and the regular variance formula beyond u=1.
    const auto normalizedSd=[&](double t) {
        const double u=a*t*t;
        const double H=u<=1 ? std::exp(-u)*scaledExponentialRemainders(u).first :
            seFactors(a,t).valueVariance/(u*u);
        return field_.kernel.sigma()*a*std::sqrt(H);
    };
    const double dlo=bounds::down(normalizedSd(hi)/boundGuard),dhi=bounds::up(normalizedSd(lo)*boundGuard);
    if(!(dlo>0) || !std::isfinite(dhi) || !std::isfinite(error) || !std::isfinite(wlo) || !std::isfinite(whi)) return false;
    wlo=bounds::down(wlo-error);whi=bounds::up(whi+error);
    lower=bounds::down(wlo/(wlo<0 ? dlo:dhi));
    upper=bounds::up(whi/(whi<0 ? dhi:dlo));
    return std::isfinite(lower) && std::isfinite(upper) && lower<=upper;
}

double ConditionalFlightKernel::coupledIntervalMajorant(double lo,double hi,double lipschitz,
        bool localMean,double parentMaximum,bool macrocell,bool tightenAtMidpoint,bool polynomial5) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::Scope detail(Profile::Stage::BoundTransition);
    Profile::count(Profile::Counter::TransitionBoundQueries);
    if(!(lo>0 && hi>lo && hi<=maximumAgeInDomain()) ||
       !(lipschitz>=0) || !std::isfinite(lipschitz))
        throw NumericError(NumericStatus::InvalidInput,"invalid coupled transition interval");
    SdfRayQueryOptions options;options.computeBeginDerivative=false;
    MeanRayBounds mean;
    {
        Profile::Scope spatial(Profile::Stage::TransitionSpatial);
        bool covered=false;
        if(macrocell) {
            Profile::Scope macro(Profile::Stage::TransitionMacroMean);
            covered=macrocellMeanBounds(lo,hi,lipschitz,mean,true);
        }
        if(!covered) {
            Profile::Scope fallback(localMean ? Profile::Stage::TransitionLocalMean : Profile::Stage::TransitionPointMean);
            mean=localMean ? meanRayBounds(lo,hi,options) : pointMeanBounds(lo,hi,lipschitz,true,true);
        }
    }
    if(!mean.certified) throw NumericError(NumericStatus::InvalidMajorant,"uncertified coupled spatial bound");
    // Enclose the two exponential-polynomial corrections with endpoint/vertex
    // polynomial extrema. A final absolute Horner error covers coefficient,
    // multiplication and endpoint roundoff without per-operation CRT calls.
    Profile::Scope correctionTime(Profile::Stage::TransitionCorrection);
    const double a=inverseLength2_;
    const double rlo=std::exp(-.5*a*hi*hi),rhi=std::exp(-.5*a*lo*lo);
    const auto product=[&](std::pair<double,double> p) {
        const double v[4]={p.first*rlo,p.first*rhi,p.second*rlo,p.second*rhi};
        const double pad=1e-12*std::max(std::abs(p.first),std::abs(p.second))*rhi;
        return std::pair<double,double>{bounds::down(*std::min_element(v,v+4)-pad),
                                       bounds::up(*std::max_element(v,v+4)+pad)};
    };
    const auto correction=product(quadraticRange(deltaValue_,deltaSlope_,0,lo,hi));
    const auto slopeCorrection=product(quadraticRange(deltaSlope_,-a*deltaValue_,-a*deltaSlope_,lo,hi));
    const double kmin=bounds::down(mean.minimumDerivative+slopeCorrection.first);
    const double kmax=bounds::up(mean.maximumDerivative+slopeCorrection.second);
    double fmin=bounds::down(mean.minimumValue+correction.first),fmax=bounds::up(mean.maximumValue+correction.second);
    correctionTime.finish();
    if(tightenAtMidpoint) {
        Profile::Scope centerTime(Profile::Stage::TransitionCenter);
        const double middle=.5*lo+.5*hi,u=a*middle*middle;
        double differenceError=0;
        const double prior=std::isfinite(mean.midpointValue) ? mean.midpointValue : scalarMeanValue(middle,true);
        const double center=u<1e-3
            ? state_.birthValue+middle*deltaSlope_+meanValueDifference(middle,&differenceError)+
              std::expm1(-.5*u)*(deltaValue_+middle*deltaSlope_)
            : prior+std::exp(-.5*u)*(deltaValue_+middle*deltaSlope_);
        const double pad=1e-12*(std::abs(prior)+std::abs(birthMean_.value)+std::abs(deltaValue_)+
            std::abs(middle*deltaSlope_)+1e-30)+std::max(differenceError,mean.midpointError);
        const double radius=upward(.5*(hi-lo)*std::max(std::abs(kmin),std::abs(kmax))+pad);
        fmin=std::max(fmin,bounds::down(center-radius));fmax=std::min(fmax,bounds::up(center+radius));
    }
    Profile::Scope factorsTime(Profile::Stage::TransitionFactors);
    double smin,smax,gmin,gmax,kappa;
    const double rootA=std::sqrt(a);
    const double qlo=tabulatedBounds_ ? bounds::down(lo*rootA/boundGuard) : 0;
    const double qhi=tabulatedBounds_ ? bounds::up(hi*rootA*boundGuard) : 0;
    if(tabulatedBounds_ && qlo>=0.0625 && qhi<8) {
        // V and the normalized slope deviation increase with age; regression
        // decreases. Outward bin selection encloses both endpoints directly.
        struct Knot { double sd, regression, normalizedSlope; };
        static const auto knots=[] {
            std::array<Knot,2049> result{};
            for(size_t i=1;i<result.size();++i) {
                const auto f=seFactors(1,double(i)/256);
                result[i]={std::sqrt(f.valueVariance),f.regression,
                           std::sqrt(f.slopeVariance/f.valueVariance)};
            }
            return result;
        }();
        const auto &left=knots[int(qlo*256)], &right=knots[int(qhi*256)+1];
        smin=bounds::down(field_.kernel.sigma()*left.sd/boundGuard);
        smax=upward(field_.kernel.sigma()*right.sd);
        gmin=std::max(0.,bounds::down(rootA*right.regression/boundGuard));
        gmax=upward(rootA*left.regression);
        kappa=upward(rootA*right.normalizedSlope);
    } else {
        const auto left=seFactors(a,lo),right=seFactors(a,hi);
        smin=bounds::down(std::sqrt(sigma2_*left.valueVariance)/boundGuard);
        smax=upward(std::sqrt(sigma2_*right.valueVariance));
        gmin=std::max(0.,bounds::down(right.regression/boundGuard));
        gmax=upward(left.regression);
        kappa=upward(std::sqrt(a*right.slopeVariance/right.valueVariance));
    }
    factorsTime.finish();
    Profile::Scope crossingTime(Profile::Stage::TransitionCrossing);
    if(!(smin>0)) throw NumericError(NumericStatus::NeedHigherPrecision,"coupled variance underflow");
    double z=bounds::down(fmin/(fmin<0 ? smin:smax));
    if(polynomial5) {
        double lower,upper;
        if(polynomialZBounds(lo,hi,lower,upper))z=std::max(z,lower);
    }
    const double mu=bounds::down(kmin-(fmax<0 ? gmin:gmax)*fmax-
        1e-12*(std::abs(kmin)+gmax*std::abs(fmax)));
    const double d=bounds::down(mu/(mu<0 ? smin:smax));
    if(tabulatedBounds_) return std::min(parentMaximum,tabulatedCrossingBound(z,d,kappa));
    const double logR=z>-8 ? normalLogPdf(z)-normalLogCdf(z) : std::log1p(-z);
    return std::min(parentMaximum,upward(std::exp(logR+logFluxUpper(d,kappa))));
}

ConditionalIntegralMajorant ConditionalFlightKernel::farIntegralMajorant(double lo,double hi,
        double lipschitz,double gradientBound) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::Scope detail(Profile::Stage::BoundLocalFar);
    Profile::count(Profile::Counter::LocalFarBoundQueries);
    if(!(lo>0 && hi>lo && hi<=maximumAgeInDomain()) || !(lipschitz>=0) || !std::isfinite(lipschitz) ||
       !(gradientBound>=0) || !std::isfinite(gradientBound))
        throw NumericError(NumericStatus::InvalidInput,"invalid Gaussian/linear tail interval");
    using I=bounds::Interval;
    const double q=(I(lo)*bounds::sqrt(I(inverseLength2_))).lower;
    if(!(q>=2)) throw NumericError(NumericStatus::InvalidInput,"Gaussian/linear tail begins before two lengths");
    if(!localFarBounds_.ready || lo<localFarBoundsAge_) {
        localFarBounds_=makeFarBounds(q,32.0);localFarBoundsAge_=lo;
    }
    const auto &f=localFarBounds_;
    const double value=scalarMeanValue(lo),slope=lipschitz==0 ? 0 : upward(lipschitz*state_.direction.norm());
    const double error=upward(512*std::numeric_limits<double>::epsilon()*
        (1+std::abs(value)+lipschitz*state_.birthPosition.norm()+slope*hi));
    const double fmin=(I(value)-I(error)-I(f.valueCorrection)).lower;
    const double fmax=(I(value)+I(error)+I(slope)*I(hi-lo)+I(f.valueCorrection)).upper;
    const double mu=(I(-gradientBound*state_.direction.norm())-I(f.slopeCorrection)-
        I(f.regressionMax)*I(std::max(0.,fmax))).lower;
    const double slopeSd=upward(std::sqrt(sigma2_*inverseLength2_));
    ConditionalIntegralMajorant result;
    result.gaussianLinear=true;result.origin=lo;
    result.z0=(I(fmin)/I(f.minimumSd,f.maximumSd)).lower;
    result.beta=(I(slope)/I(f.minimumSd)).upper;
    if(slope==0) result.beta=0;
    result.constant=upward(ClassicGlobal::negativePartMean(mu,slopeSd)/f.minimumSd*(1+1e-5));
    return result;
}

double ConditionalFlightKernel::localFarIntervalMajorant(double lo,double hi,double lipschitz,
        double gradientBound,double parentMaximum,bool cache,bool macrocell) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::Scope region(Profile::Stage::BoundLocalFar);
    Profile::Scope detail(Profile::Stage::BoundSimple);
    Profile::count(Profile::Counter::LocalFarBoundQueries);
    Profile::Scope ageTime(Profile::Stage::LocalFarAge);
    if(!(lo>0 && hi>lo && hi<=maximumAgeInDomain()) ||
       !(lipschitz>=0) || !std::isfinite(lipschitz) ||
       !(gradientBound>=0) || !std::isfinite(gradientBound) || !(parentMaximum>=0))
        throw NumericError(NumericStatus::InvalidInput,"invalid local far interval");
    using I=bounds::Interval;
    const double q=(I(lo)*bounds::sqrt(I(inverseLength2_))).lower;
    if(!(q>=2) || !std::isfinite(q))
        throw NumericError(NumericStatus::InvalidInput,"local far interval must begin beyond two directional lengths");
    ageTime.finish();
    MeanRayBounds mean;
    bool useMacro;
    {
        Profile::Scope spatial(Profile::Stage::FarSpatial);
        useMacro=macrocell && macrocellMeanBounds(lo,hi,lipschitz,mean,false);
        if(!useMacro) mean=pointMeanBounds(lo,hi,lipschitz,true);
    }
    // Refresh from this interval's age, not the configured switching age.
    // Beyond 32 lengths retain the tiny certified residual at 32; this avoids
    // enormous polynomial arguments while preserving a conservative bound.
    Profile::Scope parametersTime(Profile::Stage::LocalFarParameters);
    if(cache && (!localFarBounds_.ready || lo<localFarBoundsAge_)) {
        Profile::Scope refresh(Profile::Stage::LocalFarRefresh);
        localFarBounds_=makeFarBounds(q,32.0);localFarBoundsAge_=lo;
    }
    const auto f=cache ? localFarBounds_ : makeFarBounds(q,32.0);
    parametersTime.finish();
    Profile::Scope combineTime(Profile::Stage::LocalFarCombine);
    const I sd(f.minimumSd,f.maximumSd);
    const double fmin=(I(mean.minimumValue)-I(f.valueCorrection)).lower;
    const double fmax=(I(mean.maximumValue)+I(f.valueCorrection)).upper;
    const double zmin=(I(fmin)/sd).lower;
    const double derivative=(I(useMacro ? std::max(-gradientBound,mean.minimumDerivative):-gradientBound)-I(f.slopeCorrection)-
        I(0,f.regressionMax)*I(fmax)).lower;
    const double slopeSd=upward(bounds::sqrt(I(sigma2_)*I(inverseLength2_)).upper);
    combineTime.finish();
    Profile::Scope crossingTime(Profile::Stage::LocalFarCrossing);
    if(tabulatedBounds_) return std::min(parentMaximum,upward(tabulatedCrossingBound(zmin,derivative,slopeSd)/f.minimumSd));
    // Use the same Mills ratio and crossing-flux functions as classic_global,
    // including its guard, instead of the looser 1-z negative-tail envelope.
    double density,flux;
    { Profile::Scope mills(Profile::Stage::BoundMills);density=ClassicGlobal::density(zmin,1.0); }
    { Profile::Scope area(Profile::Stage::BoundFlux);flux=ClassicGlobal::negativePartMean(derivative,slopeSd); }
    const double maximum=density/f.minimumSd*flux*(1.0+1e-5);
    if(!(maximum>=0) || !std::isfinite(maximum))
        throw NumericError(NumericStatus::InvalidMajorant,"nonfinite local far majorant");
    return std::min(parentMaximum,upward(maximum));
}

double ConditionalFlightKernel::simpleIntervalMajorant(double lo,double hi,double lipschitz,
        double parentMaximum) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    Profile::Scope detail(Profile::Stage::BoundSimple);
    Profile::count(Profile::Counter::SimpleBoundQueries);
    if (!(simpleBoundStartScale_>0 && lo>=simpleBoundBegin_ && hi>lo && hi<=maximumAgeInDomain()) ||
        !(lipschitz>=0) || !std::isfinite(lipschitz) || !(parentMaximum>=0))
        throw NumericError(NumericStatus::InvalidInput,"invalid simple SE bound interval");
    using I=bounds::Interval;
    if(!simpleBounds_.ready) {
        Profile::count(Profile::Counter::SimpleBoundSetups);
        // The envelope is decreasing beyond Q>=2. Q=8 also safely covers
        // every later cutoff, without forming huge polynomial arguments.
        const I q(std::min(simpleBoundStartScale_,8.0)),q2=q*q;
        const I rootA=bounds::sqrt(I(inverseLength2_));
        const I r=bounds::exp(I(-0.5)*q2),e=bounds::exp(-q2);
        const I v=I(1)-(I(1)+q2)*e;
        const I df(std::abs(deltaValue_)),dk(std::abs(deltaSlope_));
        simpleBounds_.valueCorrection=upward(((df+q*dk/rootA)*r).upper);
        simpleBounds_.slopeCorrection=upward(((rootA*q*df+(q2-I(1))*dk)*r).upper);
        simpleBounds_.minimumSd=bounds::down(bounds::sqrt(I(sigma2_)*v).lower/boundGuard);
        simpleBounds_.maximumSd=upward(bounds::sqrt(I(sigma2_)).upper);
        simpleBounds_.regressionMax=upward((rootA*q*q2*e/v).upper);
        simpleBounds_.slopeSd=upward(bounds::sqrt(I(sigma2_)*I(inverseLength2_)).upper);
        simpleBounds_.ready=true;
    }
    const auto mean=pointMeanBounds(lo,hi,lipschitz);
    const auto &f=simpleBounds_;
    if(!std::isfinite(mean.minimumValue) || !std::isfinite(mean.maximumValue) ||
        !std::isfinite(mean.midpointError) || !(f.minimumSd>0))
        return parentMaximum;
    const double fmin=(I(mean.minimumValue)-I(f.valueCorrection)).lower;
    const double fmax=(I(mean.maximumValue)+I(f.valueCorrection)).upper;
    const double muMin=(I(mean.minimumDerivative)-I(f.slopeCorrection)-
        I(f.regressionMax)*I(std::max(0.0,fmax))).lower;
    const double zmin=(I(fmin)/I(f.minimumSd,f.maximumSd)).lower;
    if(!std::isfinite(muMin) || !std::isfinite(zmin)) return parentMaximum;
    const double logMills=[&]() {
        Profile::Scope mills(Profile::Stage::BoundMills);
        if(zmin<=-8) return std::log1p(-zmin);
        Profile::Scope cdf(Profile::Stage::BoundNormalCdf);
        return normalLogPdf(zmin)-normalLogCdf(zmin);
    }();
    const double maximum=std::exp(logMills-std::log(f.minimumSd)+logFluxUpper(muMin,f.slopeSd));
    if(!std::isfinite(maximum) || maximum>std::numeric_limits<double>::max()/boundGuard)
        return parentMaximum;
    return std::min(parentMaximum,upward(maximum));
}

bool ConditionalFlightKernel::clipMacrocellInterval(double lo,double &hi) const {
    if(!meanRay_ || !(hi>lo)) return false;
    if(!(lo>=macrocellBegin_ && lo<macrocellEnd_)) {
        // The active GP domain may be a 1e30 sentinel. Query only a finite
        // forward cache window; coarseRayBounds clips it to the first block.
        double end=std::min(maximumAgeInDomain(),lo+8/std::sqrt(inverseLength2_));
        MeanRayBounds mean;
        Profile::Scope detail(Profile::Stage::BoundSdfCoarse);
        if(!meanRay_->coarseRayBounds(lo,end,mean) || !mean.certified || !(end>lo)) return false;
        macrocellMean_=mean;macrocellBegin_=lo;macrocellEnd_=end;
        macrocellMidpoint_=.5*lo+.5*end;
    }
    hi=std::min(hi,macrocellEnd_);
    return true;
}

bool ConditionalFlightKernel::chapter2MacrocellMajorant(double lo,double &hi,double &maximum) const {
    if(!(lo>0) || !clipMacrocellInterval(lo,hi)) return false;
    Profile::count(Profile::Counter::Chapter2ChildBounds);
    return chapter2Majorant(lo,hi,macrocellMean_,maximum);
}

bool ConditionalFlightKernel::chapter2ParentRange(double lo,double &hi,MeanRayBounds &parent) const {
    double childEnd=hi;
    if (!(lo>0) || !clipMacrocellInterval(lo,childEnd) || !(childEnd<hi)) return false;
    return meanRay_->parentRayBounds(lo,hi,parent) && parent.certified && hi>childEnd;
}

bool ConditionalFlightKernel::rejectChapter2Parent(double lo,const MeanRayBounds &mean,double length,
                                                   double envelope,double eventBudget) const {
    const double u=inverseLength2_*lo*lo;
    // e^(-u/2) >= max(0,1-u/2). Evaluating the two residual envelopes at the
    // left endpoint gives lower bounds on the corrections in chapter2Majorant.
    const double decay=std::max(0.0,1.0-.5*u);
    double minusF=-mean.minimumValue;
    double minusG=-mean.minimumDerivative;
    if (decay>0) {
        minusF+=decay*(std::abs(deltaValue_)+lo*std::abs(deltaSlope_));
        minusG+=decay*(inverseLength2_*lo*std::abs(deltaValue_)+std::abs(1-u)*std::abs(deltaSlope_));
    }
    if (!(minusF>=0 && minusG>=0)) return false;
    const double sigma=field_.kernel.sigma();
    // 1-(1+u)exp(-u) <= min(1,u^2/2). In this quadrant the inverse Mills
    // ratio is >= max(sqrt(2/pi),-z), and negative-part flux >= max(sd/sqrt(2*pi),-g).
    const double s=sigma*std::min(1.0,.708*u);
    const double flux=std::max(.398*sigma*std::sqrt(inverseLength2_),minusG);
    const double lower=std::max(.797*s,minusF)*flux*length*envelope;
    const double threshold=eventBudget*s*s;
    return std::isnormal(lower) && std::isnormal(threshold) && lower>threshold*(1+1e-9);
}

bool ConditionalFlightKernel::chapter2Majorant(double lo,double hi,const MeanRayBounds &mean,double &maximum) const {
    // Use the symmetric, elementary upper bounds of section 2.3. Work in
    // directional length units. All residual envelopes decrease beyond 24
    // lengths. Retain that residual so exp(-q^2) stays representable even
    // before multiplying the coupling by a potentially large slope deviation.
    const double gamma=std::sqrt(inverseLength2_),sigma=field_.kernel.sigma();
    constexpr double tailAge=24.;
    const double left=std::min(lo*gamma,tailAge),right=std::min(hi*gamma,tailAge);
    const auto decay=[](double q) {return std::exp(-.5*q*q);};
    const double peak1=std::clamp(1.,left,right);
    const double linear=peak1*decay(peak1);
    const auto quadratic=[&](double q) {return std::abs(1-q*q)*decay(q);};
    const double slopeShape=std::max({quadratic(left),quadratic(right),quadratic(std::clamp(std::sqrt(3.),left,right))});
    const double deltaF=upward(std::abs(deltaValue_)*decay(left)+std::abs(deltaSlope_)/gamma*linear);
    const double deltaG=upward(gamma*std::abs(deltaValue_)*linear+std::abs(deltaSlope_)*slopeShape);
    const double slopeSd=upward(sigma*gamma);
    const auto sdFactor=[](double q){return std::sqrt(seFactors(1.,q).valueVariance);};
    const double smin=bounds::down(sigma*sdFactor(left)/boundGuard);
    if(!(smin>0))return false;
    const double smax=upward(sigma*(right>=tailAge ? 1. : sdFactor(right)));
    const double fmin=bounds::down(mean.minimumValue-deltaF);
    const double fmax=bounds::up(mean.maximumValue+deltaF);
    const double zmin=bounds::down(fmin/(fmin<0 ? smin:smax));
    const double zmax=bounds::up(fmax/(fmax<0 ? smax:smin));
    // Bracket the unique maximum at q=0.8526354784... . The absolute
    // constant rounds upward; monotone branches are evaluated at an endpoint.
    double coupling=.737070;
    if(right<.8526354 || left>.8526356) {
        const double q=right<.8526354 ? right:left;
        coupling=q*q*q*std::exp(-q*q)/sdFactor(q);
    }
    const double regression=upward(slopeSd*coupling);
    const double gmin=bounds::down(bounds::down(mean.minimumDerivative-deltaG)-
        bounds::up(regression*std::max(0.,zmax)));
    maximum=upward(tabulatedCrossingBound(zmin,gmin,slopeSd)/smin);
    return true;
}

bool ConditionalFlightKernel::macrocellMeanBounds(double lo,double hi,double lipschitz,
        MeanRayBounds &mean,bool transition) const {
    double clipped=hi;
    if(!clipMacrocellInterval(lo,clipped) || clipped<hi) return false;
    mean=macrocellMean_;
    const double middle=.5*lo+.5*hi;
    if(middle!=macrocellMidpoint_ || !std::isfinite(mean.midpointValue)) {
        mean.midpointValue=scalarMeanValue(middle,transition);
        // Same conservative world-coordinate/provider allowance as pointMeanBounds.
        mean.midpointError=upward(512*std::numeric_limits<double>::epsilon()*
            (1+std::abs(mean.midpointValue)+lipschitz*state_.birthPosition.norm()+
             lipschitz*state_.direction.norm()*hi));
    }
    const double global=upward(lipschitz*state_.direction.norm());
    mean.minimumDerivative=std::max(mean.minimumDerivative,-global);
    mean.maximumDerivative=std::min(mean.maximumDerivative,global);
    const double slope=std::max(std::abs(mean.minimumDerivative),std::abs(mean.maximumDerivative));
    const double radius=upward(.5*(hi-lo)*slope+mean.midpointError);
    mean.minimumValue=std::max(mean.minimumValue,bounds::down(mean.midpointValue-radius));
    mean.maximumValue=std::min(mean.maximumValue,bounds::up(mean.midpointValue+radius));
    if(mean.minimumValue>mean.maximumValue || mean.minimumDerivative>mean.maximumDerivative)
        throw NumericError(NumericStatus::InvalidMajorant,"inconsistent macrocell mean enclosure");
    return true;
}

bool ConditionalFlightKernel::coarseMajorant(double lo,double &hi,double &maximum) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    MeanRayBounds mean;
    {
        Profile::Scope detail(Profile::Stage::BoundSdfCoarse);
        if(!meanRay_ || !meanRay_->coarseRayBounds(lo,hi,mean)) return false;
    }
    if(!(lo>0.0 && hi>lo && hi<=maximumAgeInDomain()))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional coarse interval");
    maximum=boundedIntervalMajorant(mean,lo,hi,std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity());
    return true;
}

double ConditionalFlightKernel::farIntervalMajorant(const MeanRayBounds &mb) const {
    Profile::Scope detail(Profile::Stage::BoundFarTail);
    if (!farBounds_.ready) farBounds_=makeFarBounds(farBoundStartScale_);
    return applyFarBounds(mb,farBounds_);
}

ConditionalFlightKernel::FarBounds ConditionalFlightKernel::makeFarBounds(double startScale,double maximumScale) const {
    using I=bounds::Interval;
    FarBounds f;
    Profile::count(Profile::Counter::FarBoundSetups);
    // All residual envelopes decrease for q>=Q>=2. Reuse a finite earlier
    // envelope beyond maximumScale, avoiding overflow at extreme settings.
    const I q(std::min(startScale,maximumScale)), q2=q*q;
    const I rootA=bounds::sqrt(I(inverseLength2_));
    const I c=bounds::exp(I(-0.5)*q2), e=bounds::exp(-q2);
    const I v=I(1)-(I(1)+q2)*e;
    const I df(std::abs(deltaValue_)), dk(std::abs(deltaSlope_));
    f.valueCorrection=upward(((df+q*dk/rootA)*c).upper);
    f.slopeCorrection=upward(((rootA*q*df+(q2-I(1))*dk)*c).upper);
    f.minimumSd=bounds::down(bounds::sqrt(I(sigma2_)*v).lower/boundGuard);
    f.maximumSd=upward(bounds::sqrt(I(sigma2_)).upper);
    f.regressionMax=upward((rootA*q*q2*e/v).upper);
    // The normalized conditional slope variance N/V^2 <= 1 exactly.
    // It follows from x-2+(x+2)exp(-x)>=0, x=q^2>=0.
    f.kappaMax=upward(rootA.upper);
    f.ready=true;
    return f;
}

double ConditionalFlightKernel::applyFarBounds(const MeanRayBounds &mb,const FarBounds &f) const {
    const MomentBounds moments{{f.minimumSd,f.maximumSd},{0,f.regressionMax},0,f.kappaMax,true};
    return applyMomentBounds(bounds::down(mb.minimumValue-f.valueCorrection),
        bounds::up(mb.maximumValue+f.valueCorrection),
        bounds::down(mb.minimumDerivative-f.slopeCorrection),moments,true);
}

double ConditionalFlightKernel::globalFarMajorant(double startScale) const {
    Profile::Scope detail(Profile::Stage::BoundFarTail);
    if(!(startScale>=2) || !std::isfinite(startScale))
        throw NumericError(NumericStatus::InvalidInput,"global far scale must be finite and at least two");
    const auto global=field_.mean->globalMagnitudeBounds();
    if(!global || !(global->maximumAbsoluteValue>=0) || !std::isfinite(global->maximumAbsoluteValue) ||
       !(global->maximumGradientNorm>=0) || !std::isfinite(global->maximumGradientNorm))
        throw NumericError(NumericStatus::InvalidInput,"global far tracking requires finite global mean bounds");
    MeanRayBounds mean;
    const bool symmetric=std::isnan(global->minimumValue) && std::isnan(global->maximumValue);
    mean.minimumValue=symmetric ? -global->maximumAbsoluteValue : global->minimumValue;
    mean.maximumValue=symmetric ? global->maximumAbsoluteValue : global->maximumValue;
    if(!std::isfinite(mean.minimumValue) || !std::isfinite(mean.maximumValue) ||
       mean.minimumValue>mean.maximumValue)
        throw NumericError(NumericStatus::InvalidInput,"invalid signed global mean bounds");
    mean.minimumDerivative=-global->maximumGradientNorm;
    mean.maximumDerivative=global->maximumGradientNorm;
    mean.certified=true;
    const double maximum=applyFarBounds(mean,makeFarBounds(startScale));
    if(!(maximum>=0) || !std::isfinite(maximum))
        throw NumericError(NumericStatus::InvalidMajorant,"nonfinite global far majorant");
    Profile::count(Profile::Counter::GlobalFarSetups);
    return maximum;
}

double ConditionalFlightKernel::boundedIntervalMajorant(const MeanRayBounds &mb,double lo,double hi,
        double tightenExpectedEvents,double parentMaximum,bool allowFarTail) const {
    Profile::Scope detail(Profile::Stage::BoundAlgebra);
    const double a=inverseLength2_;
    if (!mb.certified) throw NumericError(NumericStatus::InvalidMajorant,"uncertified ray mean bounds");
    if (allowFarTail && lo>=farBoundBegin_) {
        const double far=std::min(parentMaximum,farIntervalMajorant(mb));
        if (std::isfinite(far)) {
            Profile::count(Profile::Counter::FarBoundAccepted);
            return far;
        }
        Profile::count(Profile::Counter::FarBoundFallbacks);
    }
    const bounds::Interval times(lo,hi);
    const auto exponential=bounds::exp(bounds::Interval(-0.5)*bounds::Interval(a)*times*times);
    auto correction=exponential*bounds::quadraticPolynomialRange(deltaValue_,deltaSlope_,0,lo,hi);
    const auto slopeLinear=-bounds::Interval(a)*bounds::Interval(deltaValue_);
    const auto slopeQuadratic=-bounds::Interval(a)*bounds::Interval(deltaSlope_);
    auto slopeCorrection=exponential*bounds::quadraticPolynomialRange(
        deltaSlope_,slopeLinear,slopeQuadratic,lo,hi);
    // Bound the mean from its midpoint and a certified derivative range to
    // preserve cancellation between the prior and the birth correction.
    const double middle=0.5*lo+0.5*hi;
    const double um=a*middle*middle;
    const bool reuseMidpoint=std::isfinite(mb.midpointValue);
    const double targetValue=reuseMidpoint ? mb.midpointValue : [&]() {
        Profile::Scope point(Profile::Stage::BoundSdfPoint);
        return meanJet(middle).value;
    }();
    const double midpointError=reuseMidpoint ? mb.midpointError : 0;
    if(!(midpointError>=0) || !std::isfinite(midpointError))
        throw NumericError(NumericStatus::InvalidMajorant,"invalid cached SDF midpoint error");
    double differenceError=0;
    const double center=um<1e-3
        ? state_.birthValue+middle*deltaSlope_+
          [&]() {
              Profile::Scope difference(Profile::Stage::BoundSdfDifference);
              return meanValueDifference(middle,&differenceError);
          }()+
          std::expm1(-0.5*um)*(deltaValue_+middle*deltaSlope_)
        : targetValue+std::exp(-0.5*um)*(deltaValue_+middle*deltaSlope_);
    const double roundoff=256.0*std::numeric_limits<double>::epsilon()*
        (std::abs(targetValue)+std::abs(birthMean_.value)+std::abs(deltaValue_)+
         std::abs(middle*deltaSlope_)+1e-30);
    // Both stages share the SDF certificate, midpoint and conditional moments.
    const auto moments=prepareMomentBounds(sigma2_,a,lo,hi,normalizedBoundOnly_);
    const auto maximum=[&]() {
        double fmin=bounds::down(mb.minimumValue+correction.lower);
        double fmax=bounds::up(mb.maximumValue+correction.upper);
        const double kmin=bounds::down(mb.minimumDerivative+slopeCorrection.lower);
        const double kmax=bounds::up(mb.maximumDerivative+slopeCorrection.upper);
        const double centerError=um<1e-3 ? differenceError : midpointError;
        const double radius=0.5*(hi-lo)*std::max(std::abs(kmin),std::abs(kmax))*boundGuard+
            roundoff+(allowFarTail ? centerError : std::max(centerError,midpointError));
        fmin=std::max(fmin,center-radius);
        fmax=std::min(fmax,center+radius);
        return applyMomentBounds(fmin,fmax,kmin,moments,true);
    };
    const double cheapMaximum=maximum();
    double result=std::min(parentMaximum,cheapMaximum);
    if(tightenExpectedEvents==std::numeric_limits<double>::infinity() ||
       (tightenExpectedEvents>0 && result*(hi-lo)<=tightenExpectedEvents)) {
        Profile::count(Profile::Counter::CheapBoundAccepted);
        return result;
    }
    if(!std::isfinite(cheapMaximum)) Profile::count(Profile::Counter::CheapBoundFallbacks);
    Profile::count(Profile::Counter::TightenedIntervals);
    const auto directCorrection=bounds::exponentialPolynomialRange(a,deltaValue_,deltaSlope_,0,lo,hi);
    const auto directSlope=bounds::exponentialPolynomialRange(a,deltaSlope_,
        slopeLinear,slopeQuadratic,lo,hi);
    correction={std::max(correction.lower,directCorrection.lower),
                std::min(correction.upper,directCorrection.upper)};
    slopeCorrection={std::max(slopeCorrection.lower,directSlope.lower),
                     std::min(slopeCorrection.upper,directSlope.upper)};
    result=std::min(result,maximum());
    if(!std::isfinite(result))
        throw NumericError(NumericStatus::InvalidMajorant,"conditional majorant overflow");
    return result;
}

ConditionalBirthMajorant ConditionalFlightKernel::birthMajorant(double end,double positiveBirthLipschitz) const {
    Profile::Scope profile(Profile::Stage::BirthBound);
    Profile::count(state_.birthValue==0 ? Profile::Counter::ZeroBirths : Profile::Counter::PositiveBirths);
    if (!(end>=currentAge() && end<=maximumAgeInDomain()))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional majorant limit");
    ConditionalBirthMajorant result;
    if (end==0.0) return result;
    const double a=inverseLength2_, ell=1.0/std::sqrt(a), f0=state_.birthValue;
    if(f0>0 && std::isfinite(positiveBirthLipschitz)) {
        if(!(positiveBirthLipschitz>=0))throw NumericError(NumericStatus::InvalidInput,"negative positive-birth Lipschitz bound");
        // Unlike a zero birth, F0>0 needs no local curvature certificate.
        // |mu(t)-F0| <= (L+|deltaSlope|)t + a*t^2/2*(|deltaValue|+t*|deltaSlope|).
        // This remains valid across nonsmooth interpolation-cell boundaries.
        const double L=upward(positiveBirthLipschitz*state_.direction.norm());
        const double ds=std::abs(deltaSlope_),dv=std::abs(deltaValue_);
        double h=std::min(end,.5*ell),deviation;
        for(int i=0;;++i) {
            deviation=upward(h*(L+ds)+.5*a*h*h*(dv+h*ds));
            if(deviation<=.5*f0)break;
            h*=.5;
            if(i>=1024 || !(h>0))throw NumericError(NumericStatus::NeedHigherPrecision,"positive birth cannot advance");
        }
        const auto factors=bounds::seVarianceBounds(bounds::squaredKernelDistance(a,h));
        const auto varianceScale=bounds::Interval(sigma2_)*bounds::Interval(a)*bounds::Interval(a);
        const double vmin=(varianceScale*factors.scaledValue).lower/boundGuard;
        const double vmax=upward((varianceScale*bounds::Interval(.5)).upper);
        const double slope=upward(L+(1+a*h*h)*ds+a*h*dv);
        // gamma<=2/t and A<=(-meanAtZero)+ + sd/sqrt(2*pi).
        const double common=upward(2*(L+ds)+a*h*(dv+h*ds)+slope+
            slopeStddevBound(sigma2_,a,h)*kInvSqrtTwoPi);
        const double separation=(f0-deviation)/boundGuard;
        result.split=result.smoothEnd=h;result.slope=state_.birthGradient.dot(state_.direction);
        result.curvature=std::numeric_limits<double>::infinity();
        result.maximum=envelopeMaximum(std::log(std::sqrt(2/kPi)*(2*f0+h*common)/std::sqrt(vmin)),
            separation*separation/(2*vmax),3,4,h);
        return result;
    }
    // A monotone conditional mean yields a Gaussian-suppressed birth rate
    // even across trilinear cell faces. Both coarse and ray-local certificates
    // cover the one-sided derivatives throughout the proposed prefix.
    if(f0==0) {
        const auto certifyMonotonePrefix=[&](double h,double priorMin,double priorMax) {
            const double scale=std::abs(priorMin)+std::abs(priorMax)+std::abs(deltaSlope_);
            const double pad=1e-12*scale;
            const double initialLower=bounds::down(priorMin+deltaSlope_-pad);
            if(initialLower>0) {
                // Choose a useful span, then certify it at its actual endpoint.
                const double sigma=field_.kernel.sigma();
                h=std::min(h,.5*initialLower/(sigma*a));
                const double drift=upward(a*h*(std::abs(deltaValue_)+1.5*h*std::abs(deltaSlope_)));
                const double lower=bounds::down(initialLower-drift);
                const double upper=bounds::up(priorMax+deltaSlope_+drift+pad);
                if(lower>0 && h>0) {
                    // lower*t <= mu <= upper*t. gamma <= 2/t, and
                    // 0.5*u^2*exp(-u) <= V(u) <= 0.5*u^2.
                    const double sdScale=sigma*a/kSqrtTwo;
                    const double sdLow=bounds::down(sdScale*std::exp(-.5*a*h*h)/boundGuard);
                    const double sdHigh=upward(sdScale);
                    const double flux=upward(2*upper-lower+sigma/ell*kInvSqrtTwoPi+pad);
                    const double separation=(lower/sdHigh)/boundGuard;
                    if(sdLow>0 && std::isnormal(separation*separation)) {
                        const double rate=envelopeMaximum(std::log(upward(std::sqrt(2/kPi)*flux/sdLow)),
                            .5*separation*separation,2,2,h);
                        if(rate*h<=1) {
                            result.split=result.smoothEnd=h;
                            result.slope=state_.direction.dot(state_.birthGradient);
                            result.curvature=std::numeric_limits<double>::infinity();
                            result.maximum=rate;
                            return true;
                        }
                    }
                }
            }
            return false;
        };
        double h=std::min(.5*ell,end);
        if(tabulatedBounds_ && std::isfinite(positiveBirthLipschitz) && clipMacrocellInterval(0,h) &&
           certifyMonotonePrefix(h,macrocellMean_.minimumDerivative,macrocellMean_.maximumDerivative))
            return result;

        // A face arbitrarily close to the birth leaves a microscopic Taylor
        // prefix. Ray-local derivative bounds can certify both sides without
        // dividing field-value roundoff by the vanishing conditional variance.
        const double localEnd=std::min(end,.02*ell);
        if(localEnd>0 && nextMeanBoundary(0,localEnd)<std::min(localEnd,1e-4*ell)) {
            SdfRayQueryOptions options;
            options.computeBeginDerivative=false;
            h=localEnd;
            for(int attempt=0;attempt<8;++attempt,h*=.5) {
                const auto spatial=meanRayBounds(0,h,options);
                if(spatial.certified &&
                   certifyMonotonePrefix(h,spatial.minimumDerivative,spatial.maximumDerivative))
                    return result;
            }
        }
    }
    // The first Taylor interval must lie inside one smooth interpolation cell.
    double smoothEnd=std::min(end,0.5*ell);
    const double firstBoundary=nextMeanBoundary(0.0,smoothEnd);
    if(firstBoundary<smoothEnd) smoothEnd=0.5*firstBoundary;
    MeanRayBounds birth=meanRayBounds(0.0,smoothEnd);
    // Procedural CSG provides local curvature certificates instead of explicit
    // grid knots. Find a certified first interval; no part of the flight is
    // discarded, and the remaining intervals still use Lipschitz majorants.
    for(int refinements=0;birth.certified && !std::isfinite(birth.maximumSecondDerivative);++refinements) {
        Profile::count(Profile::Counter::SmoothBirthHalvings);
        smoothEnd*=0.5;
        if(refinements>=128 || !(smoothEnd>0) ||
           (state_.birthPosition+smoothEnd*state_.direction)==state_.birthPosition)
            throw NumericError(NumericStatus::InvalidMajorant,"mean has no certifiable smooth conditional birth interval");
        birth=meanRayBounds(0.0,smoothEnd);
    }
    if(!birth.certified)
        throw NumericError(NumericStatus::InvalidMajorant,"mean lacks certified conditional birth bounds");
    const double k=birth.beginDerivative+deltaSlope_;
    if (!std::isfinite(k))
        throw NumericError(NumericStatus::InvalidMajorant,"nonfinite conditional birth slope");
    if (f0==0.0 && !(k>0.0))
        throw NumericError(NumericStatus::NonPositiveSurfaceBirth,"surface birth must enter the positive side");
    const double curvature=birth.maximumSecondDerivative;
    result.smoothEnd=smoothEnd;
    result.slope=k;
    result.curvature=curvature;
    const auto remainder=[&](double h) {
        return upward(0.5*(curvature+a*(std::abs(deltaValue_)+h*std::abs(deltaSlope_))));
    };
    double h=f0==0.0 ? std::min(smoothEnd,k/(std::sqrt(sigma2_)*a)) : smoothEnd;
    for (int count=0;;++count) {
        const double error=remainder(h);
        if (f0==0.0 ? error*h<=0.5*k : std::abs(k)*h+error*h*h<=0.5*f0) break;
        Profile::count(Profile::Counter::SeparationHalvings);
        h*=0.5;
        if (count>=1024 || !(h>0.0))
            throw NumericError(NumericStatus::InvalidMajorant,"cannot certify conditional birth segment");
    }
    result.split=h;
    const double error=remainder(h);
    const auto kernelBounds=bounds::seVarianceBounds(bounds::squaredKernelDistance(a,h));
    // V(q)/q^2 decreases from 1/2; these bound s(t)^2/t^4 over (0,h].
    const auto varianceScale=bounds::Interval(sigma2_)*bounds::Interval(a)*bounds::Interval(a);
    const double vmin=(varianceScale*kernelBounds.scaledValue).lower/boundGuard;
    const double vmax=upward((varianceScale*bounds::Interval(0.5)).upper);
    const double smax=slopeStddevBound(sigma2_,a,h);
    // Combine gamma*meanF-meanK before bounding, using meanF>0 and gamma<=2/t.
    // The shared birth slope contributes (k)+, instead of 2|k|+|k|.
    const double common=upward(std::max(k,0.0)+
        (2*curvature+2*a*std::abs(deltaValue_)+2.5*a*h*std::abs(deltaSlope_))*h+
        smax*kInvSqrtTwoPi);
    if (f0==0.0) {
        const double b=(k-error*h)/boundGuard;
        result.maximum=envelopeMaximum(std::log(std::sqrt(2.0/kPi)*common/std::sqrt(vmin)),
                                           b*b/(2.0*vmax),2.0,2.0,h);
    } else {
        const double b=(f0-std::abs(k)*h-error*h*h)/boundGuard;
        const double fluxCoefficient=2.0*f0+h*common;
        result.maximum=envelopeMaximum(std::log(std::sqrt(2.0/kPi)*fluxCoefficient/std::sqrt(vmin)),
                                           b*b/(2.0*vmax),3.0,4.0,h);
    }
    return result;
}

double ConditionalFlightKernel::birthAdjacentMajorant(
    double lo, double hi, const ConditionalBirthMajorant& birth) const {
    Profile::Scope profile(Profile::Stage::TaylorBound);
    if (!(lo>0.0 && hi>lo && hi<=birth.smoothEnd))
        throw NumericError(NumericStatus::InvalidInput, "invalid birth-adjacent interval");
    const double a=inverseLength2_;
    const double rem=upward(0.5*(birth.curvature+a*(std::abs(deltaValue_)+
                                                     hi*std::abs(deltaSlope_))));
    const double slopeRem=upward(birth.curvature+a*std::abs(deltaValue_)+
                                 1.5*a*hi*std::abs(deltaSlope_));
    return momentIntervalMajorant(
        quadraticRange(state_.birthValue,birth.slope,-rem,lo,hi).first,
        quadraticRange(state_.birthValue,birth.slope, rem,lo,hi).second,
        birth.slope-slopeRem*hi,sigma2_,a,lo,hi,normalizedBoundOnly_);
}

double ConditionalFlightKernel::constantTailMajorant(double begin) const {
    if (!(begin>0) || !std::isfinite(begin))
        throw NumericError(NumericStatus::InvalidInput,"constant tail requires a finite positive birth split");
    const auto meanBounds=field_.mean->globalMagnitudeBounds();
    if (!meanBounds || !(meanBounds->maximumAbsoluteValue>=0) ||
            !std::isfinite(meanBounds->maximumAbsoluteValue) ||
            !(meanBounds->maximumGradientNorm>=0) || !std::isfinite(meanBounds->maximumGradientNorm))
        throw NumericError(NumericStatus::InvalidMajorant,
            "two_constants requires finite global mean bounds (a clamped VDB SDF or constant mean)");
    const double a=inverseLength2_, inverseSqrtE=std::exp(-0.5);
    const double valueMaximum=upward(meanBounds->maximumAbsoluteValue+std::abs(deltaValue_)+
        std::abs(deltaSlope_)*inverseSqrtE/std::sqrt(a));
    const double derivativeMaximum=upward(meanBounds->maximumGradientNorm+std::abs(deltaSlope_)+
        std::sqrt(a)*inverseSqrtE*std::abs(deltaValue_));
    // The series in seFactors preserves 1-(1+a*h*h)*exp(-a*h*h) near birth.
    const double minimumSd=std::sqrt(sigma2_*seFactors(a,begin).valueVariance)/boundGuard;
    if (!(minimumSd>0) || !std::isfinite(minimumSd))
        throw NumericError(NumericStatus::NeedHigherPrecision,"constant tail variance underflow");
    return upward((std::sqrt(2.0/kPi)+valueMaximum/minimumSd)*
        (std::sqrt(a)*kInvSqrtTwoPi+(derivativeMaximum+2*valueMaximum/begin)/minimumSd));
}

ConditionalMajorants ConditionalFlightKernel::twoSegmentMajorants(double end) const {
    const ConditionalBirthMajorant birth=birthMajorant(end);
    ConditionalMajorants result;
    result.split=birth.split;
    result.nearMaximum=birth.maximum;
    if (end==0.0) return result;
    const double a=inverseLength2_, ell=1.0/std::sqrt(a), h=birth.split;

    // Bound the entire far segment. Internal intervals only tighten the proof:
    // their maximum is the default far Poisson rate. Retain the intervals for
    // exact thinning when this constant would cause excessive null events.
    const auto intervalMaximum=[&](double lo, double hi) {
        ++result.boundIntervals;
        if (hi>birth.smoothEnd) return intervalMajorant(lo,hi);
        return birthAdjacentMajorant(lo,hi,birth);
    };
    for (double lo=h; lo<end;) {
        double hi=std::min(end,lo+std::min(0.5*lo,0.5*ell));
        if (!(hi>lo)) throw NumericError(NumericStatus::InvalidMajorant,"conditional bound interval cannot advance");
        double bound=intervalMaximum(lo,hi);
        // Refine expensive near-birth enclosures, where vF grows like t^4.
        // The work threshold only chooses between already valid bounds; it
        // does not limit the rate, discard an interval, or probe the hazard.
        while (lo<ell && hi-lo>lo/16.0 && bound*(end-h)>256.0) {
            const double refinedHi=lo+std::max(lo/16.0,0.5*(hi-lo));
            if (!(refinedHi<hi)) break;
            hi=refinedHi;
            bound=intervalMaximum(lo,hi);
        }
        result.farMaximum=std::max(result.farMaximum,bound);
        result.farIntervals.push_back({lo,hi,bound});
        lo=hi;
    }
    return result;
}

} // namespace Tungsten::ConditionalGpis
