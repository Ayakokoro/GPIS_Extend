// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include "math/ConditionalProfile.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/SmallGaussian.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <limits>
#include <vector>

namespace Tungsten::ConditionalGpis {
namespace {

constexpr double boundGuard = 1.0 + 1e-11;
namespace bounds = MajorantBounds;

double upward(double x) {
    if (!(x >= 0.0) || !std::isfinite(x))
        throw NumericError(NumericStatus::InvalidMajorant, "non-finite conditional majorant");
    const double result=std::nextafter(x * boundGuard, std::numeric_limits<double>::infinity());
    if (!std::isfinite(result))
        throw NumericError(NumericStatus::InvalidMajorant, "conditional majorant overflow");
    return result;
}

// Positive series: truncation after 20 terms is below double roundoff for u<=1.
// The bounds below allow for both this remainder and floating-point arithmetic.
std::pair<double,double> scaledExponentialRemainders(double u) {
    double e2=0.5, term=0.5;
    for (int n=1; n<=20; ++n) { term *= u/(n+2); e2 += term; }
    double e4=1.0/12.0;
    term=e4;
    for (int n=1; n<=12; ++n) {
        term *= u*u/((2*n+3.0)*(2*n+4.0)); e4 += term;
    }
    return {e2,e4};
}

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

double logFluxUpper(double meanLower,double stddevUpper) {
    if(stddevUpper==0)
        return meanLower<0 ? std::log(upward(-meanLower)) : -std::numeric_limits<double>::infinity();
    // Negative-part expectation increases as its mean falls or its standard
    // deviation rises. Use its exact value where the subtraction is stable.
    const double q=meanLower/stddevUpper;
    return meanLower<0.0
        ? std::log(upward(-meanLower*normalCdf(-q)+stddevUpper*normalPdf(-q)))
        : q<=4.0
            ? std::log(stddevUpper)+std::log(upward(normalPdf(q)-q*normalCdf(-q)))
            : std::log(stddevUpper)+normalLogPdf(q)+std::log(boundGuard);
}

struct MomentBounds {
    bounds::Interval sd,regression;
    double slopeMax,kappaMax;
};

MomentBounds prepareMomentBounds(double sigma2, double a, double lo, double hi) {
    Profile::Scope profile(Profile::Stage::MomentBound);
    const auto left=seFactors(a,lo), right=seFactors(a,hi);
    const double sdmin=std::sqrt(sigma2*left.valueVariance)/boundGuard;
    const double sdmax=upward(std::sqrt(sigma2*right.valueVariance));
    if(!(sdmin>0))
        throw NumericError(NumericStatus::NeedHigherPrecision,"conditional interval variance underflow");
    // The regression s'/s decreases along a forward birth flight.
    const bounds::Interval regression(std::max(0.0,bounds::down(right.regression/boundGuard)),
                                      upward(left.regression));
    const auto endpoint=bounds::seVarianceBounds(bounds::squaredKernelDistance(a,hi));
    const double slopeMax=upward(bounds::sqrt(bounds::Interval(sigma2)*bounds::Interval(a)*
        bounds::Interval(std::min(1.0,endpoint.slope.upper))).upper);
    const double kappaMax=upward(bounds::sqrt(bounds::Interval(a)*
        bounds::Interval(std::min(1.0,endpoint.normalized.upper))).upper);
    return {{sdmin,sdmax},regression,slopeMax,kappaMax};
}

double applyMomentBounds(double fmin, double fmax, double kmin,
                         const MomentBounds &moments, bool allowInfinite=false) {
    Profile::Scope profile(Profile::Stage::MomentBound);
    if(std::isnan(fmin) || std::isnan(fmax) || std::isnan(kmin))
        throw NumericError(NumericStatus::InvalidMajorant,"invalid conditional mean bounds");
    if(allowInfinite && (!std::isfinite(fmin) || !std::isfinite(fmax) || !std::isfinite(kmin)))
        return std::numeric_limits<double>::infinity();
    const double zmin=(bounds::Interval(fmin)/moments.sd).lower;
    const double muMin=(bounds::Interval(kmin)-moments.regression*bounds::Interval(fmax)).lower;
    if(allowInfinite && (!std::isfinite(zmin) || !std::isfinite(muMin)))
        return std::numeric_limits<double>::infinity();
    const double logMills=zmin>-8.0
        ? normalLogPdf(zmin)-normalLogCdf(zmin) : std::log1p(-zmin);
    const double oldLog=logMills-std::log(moments.sd.lower)+logFluxUpper(muMin,moments.slopeMax);
    // z'=meanAtZero/s; division must use sdmax for a positive lower mean.
    // N/V^2 increases from 1/3 to 1, so kappa is bounded at the right endpoint.
    const double derivativeMin=(bounds::Interval(muMin)/moments.sd).lower;
    const double kappaMax=moments.kappaMax;
    const double normalizedLog=logMills+logFluxUpper(derivativeMin,kappaMax);
    if(std::isnan(oldLog) || std::isnan(normalizedLog))
        throw NumericError(NumericStatus::InvalidMajorant,"invalid conditional moment bound");
    const double upper=std::exp(std::min(oldLog,normalizedLog));
    // A loose cheap enclosure may overflow even when the fine bound is usable.
    if(allowInfinite && upper>std::numeric_limits<double>::max()/boundGuard)
        return std::numeric_limits<double>::infinity();
    return upward(upper);
}

double momentIntervalMajorant(double fmin, double fmax, double kmin,
                              double sigma2, double a, double lo, double hi) {
    return applyMomentBounds(fmin,fmax,kmin,prepareMomentBounds(sigma2,a,lo,hi));
}

} // namespace

FlightState startConditionalExterior(const GPSSField& field, const Point3& entry,
                                     const Vector3& direction, Random& rng) {
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

ConditionalFlightKernel::ConditionalFlightKernel(const GPSSField& field,
                                                 const FlightState& state)
    : FlightKernel(field, state) {
    Profile::Scope profile(Profile::Stage::KernelSetup);
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
}

ConditionalScalarStatistics ConditionalFlightKernel::scalarStatistics(double age) const {
    Profile::Scope profile(Profile::Stage::ScalarStatistics);
    if (!(age > 0.0 && age <= maximumAgeInDomain()))
        throw std::out_of_range("conditional flight age outside domain");
    const double u=inverseLength2_*age*age, r=std::exp(-0.5*u);
    const MeanRaySample target=[&]() {
        if(meanRay_) return meanRay_->sample(age);
        const auto jet=meanJet(age);
        return MeanRaySample{jet.value,state_.direction.dot(jet.gradient)};
    }();
    const double meanF=u<1e-3
        ? state_.birthValue+age*deltaSlope_+
          meanValueDifference(age)+
          std::expm1(-0.5*u)*(deltaValue_+age*deltaSlope_)
        : target.value+r*(deltaValue_+age*deltaSlope_);
    const double meanK=target.derivative+
        r*((1.0-u)*deltaSlope_-inverseLength2_*age*deltaValue_);
    const SeFactors factors=seFactors(inverseLength2_,age);
    return {meanF,sigma2_*factors.valueVariance,meanK,
            meanK-factors.regression*meanF,sigma2_*inverseLength2_*factors.slopeVariance};
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
    const auto factors=seFactors(inverseLength2_,age);
    const double u=inverseLength2_*age*age, r=std::exp(-0.5*u);
    const auto target=meanJet(age);
    Gaussian<3> g;
    g.mean=target.gradient+r*deltaGradient_-precisionDirection_*
        (age*r*(deltaValue_+age*deltaSlope_)-factors.regression*deviation/inverseLength2_);
    const Matrix3 longitudinal=precisionDirection_*precisionDirection_.transpose()/inverseLength2_;
    g.covariance=sigma2_*(-std::expm1(-u)*(field_.kernel.precision()-longitudinal)+
                         factors.slopeVariance*longitudinal);
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
    return boundedIntervalMajorant(meanRayBounds(lo,hi,queryOptions),lo,hi,
                                   tightenExpectedEvents,parentMaximum);
}

bool ConditionalFlightKernel::coarseMajorant(double lo,double &hi,double &maximum) const {
    Profile::Scope profile(Profile::Stage::IntervalBound);
    MeanRayBounds mean;
    if(!meanRay_ || !meanRay_->coarseRayBounds(lo,hi,mean)) return false;
    if(!(lo>0.0 && hi>lo && hi<=maximumAgeInDomain()))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional coarse interval");
    maximum=boundedIntervalMajorant(mean,lo,hi,std::numeric_limits<double>::infinity(),
                                    std::numeric_limits<double>::infinity());
    return true;
}

double ConditionalFlightKernel::boundedIntervalMajorant(const MeanRayBounds &mb,double lo,double hi,
        double tightenExpectedEvents,double parentMaximum) const {
    const double a=inverseLength2_;
    if (!mb.certified) throw NumericError(NumericStatus::InvalidMajorant,"uncertified ray mean bounds");
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
    const double targetValue=reuseMidpoint ? mb.midpointValue :
        meanJet(middle).value;
    const double midpointError=reuseMidpoint ? mb.midpointError : 0;
    if(!(midpointError>=0) || !std::isfinite(midpointError))
        throw NumericError(NumericStatus::InvalidMajorant,"invalid cached SDF midpoint error");
    double differenceError=0;
    const double center=um<1e-3
        ? state_.birthValue+middle*deltaSlope_+
          meanValueDifference(middle,&differenceError)+
          std::expm1(-0.5*um)*(deltaValue_+middle*deltaSlope_)
        : targetValue+std::exp(-0.5*um)*(deltaValue_+middle*deltaSlope_);
    const double roundoff=256.0*std::numeric_limits<double>::epsilon()*
        (std::abs(targetValue)+std::abs(birthMean_.value)+std::abs(deltaValue_)+
         std::abs(middle*deltaSlope_)+1e-30);
    // Both stages share the SDF certificate, midpoint and conditional moments.
    const auto moments=prepareMomentBounds(sigma2_,a,lo,hi);
    const auto maximum=[&]() {
        double fmin=bounds::down(mb.minimumValue+correction.lower);
        double fmax=bounds::up(mb.maximumValue+correction.upper);
        const double kmin=bounds::down(mb.minimumDerivative+slopeCorrection.lower);
        const double kmax=bounds::up(mb.maximumDerivative+slopeCorrection.upper);
        const double radius=0.5*(hi-lo)*std::max(std::abs(kmin),std::abs(kmax))*boundGuard+
            roundoff+(um<1e-3 ? differenceError : midpointError);
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

ConditionalBirthMajorant ConditionalFlightKernel::birthMajorant(double end) const {
    Profile::Scope profile(Profile::Stage::BirthBound);
    Profile::count(state_.birthValue==0 ? Profile::Counter::ZeroBirths : Profile::Counter::PositiveBirths);
    if (!(end>=currentAge() && end<=maximumAgeInDomain()))
        throw NumericError(NumericStatus::InvalidInput,"invalid conditional majorant limit");
    ConditionalBirthMajorant result;
    if (end==0.0) return result;
    const double a=inverseLength2_, ell=1.0/std::sqrt(a), f0=state_.birthValue;
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
    if (f0==0.0 && !(k>0.0))
        throw NumericError(NumericStatus::UnsupportedSingularFlight,"surface birth must enter the positive side");
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
        birth.slope-slopeRem*hi,sigma2_,a,lo,hi);
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
