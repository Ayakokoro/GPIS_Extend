#include "math/conditional/transport/SinglePointHistoryTeacher.h"
#include "math/SparseHistoryGaussian.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include <algorithm>
#include <limits>
#include <vector>

namespace Tungsten::ConditionalGpis {
namespace {

const GPSSField& validatedField(const GPSSField& field)
{
    field.validate();
    return field;
}

double correlation(double c, double a, double b)
{
    if (!(a > 0 && b > 0))
        throw NumericError(NumericStatus::NeedHigherPrecision,
                           "single-point teacher has a degenerate conditional variance");
    const double result = (c/std::sqrt(a))/std::sqrt(b);
    if (!std::isfinite(result) || std::abs(result) > 1 + 2e-12)
        throw NumericError(NumericStatus::InvalidCovariance,
                           "single-point teacher has an invalid correlation");
    // Only repair a demonstrable floating-point excursion; singular samples
    // are rejected below rather than silently turned into regular labels.
    return std::clamp(result, -1.0, 1.0);
}

Eigen::VectorXd seFeatures(double q, bool negativeDerivative, int count)
{
    Eigen::VectorXd result = Eigen::VectorXd::Zero(count);
    double previous = 0, value = std::exp(-0.5*q*q);
    for (int n = 0; n < count; ++n) {
        if (n) { previous = value; value *= q/std::sqrt(double(n)); }
        // Conditioning on the complete birth removes the value and axial
        // derivative features n=0 and n=1. Transverse birth derivatives are
        // independent of the field restricted to this straight ray.
        if (n >= 2)
            result[n] = negativeDerivative ? q*value-std::sqrt(double(n))*previous : value;
    }
    return result;
}

double valueVariance(double q)
{
    const double x = q*q;
    if (x < 1) {
        double term = 0.5*x*x, sum = term;
        for (int n = 3; n <= 32; ++n) { term *= x/n; sum += term; }
        return std::exp(-x)*sum;
    }
    return -std::expm1(-x)-x*std::exp(-x);
}

double valueCovariance(double q, double r)
{
    const double x = q*r;
    if (x < 1) {
        double term = 0.5*x*x, sum = term;
        for (int n = 3; n <= 32; ++n) { term *= x/n; sum += term; }
        return std::exp(-0.5*(q*q+r*r))*sum;
    }
    return std::exp(-0.5*(q-r)*(q-r))-(1+x)*std::exp(-0.5*(q*q+r*r));
}

struct JointCovariance {
    Matrix3 value;
    Matrix2 atZero; // V,U given X=0
};

JointCovariance jointCovariance(double q, double r)
{
    JointCovariance result;
    if (q <= 8) {
        // This stable feature projection is intentionally an offline cost.
        // The new training oracle does not alter the renderer's hot path.
        const int count = std::max(24, int(std::ceil(q*q+14*q+50)));
        Eigen::MatrixXd features(3, count);
        features.row(0) = seFeatures(q, false, count);
        features.row(1) = seFeatures(r, false, count);
        features.row(2) = seFeatures(q, true, count);
        result.value = features*features.transpose();
        if (!(result.value(0,0) > 0))
            throw NumericError(NumericStatus::NeedHigherPrecision,
                               "single-point teacher current variance underflow");
        Eigen::MatrixXd residual(2, count);
        residual.row(0) = features.row(2)-(result.value(2,0)/result.value(0,0))*features.row(0);
        residual.row(1) = features.row(1)-(result.value(1,0)/result.value(0,0))*features.row(0);
        result.atZero = residual*residual.transpose();
    } else {
        const double e = std::exp(-q*q), qr = q*r;
        result.value.setZero();
        result.value(0,0) = valueVariance(q);
        result.value(1,1) = valueVariance(r);
        result.value(2,2) = 1-(1-q*q+q*q*q*q)*e;
        result.value(0,1) = result.value(1,0) = valueCovariance(q,r);
        result.value(0,2) = result.value(2,0) = -q*q*q*e;
        result.value(1,2) = result.value(2,1) =
            (q-r)*std::exp(-0.5*(q-r)*(q-r))+
            (r-q*(1+qr))*std::exp(-0.5*(q*q+r*r));
        const int indices[2] = {2,1};
        for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j)
            result.atZero(i,j) = double(static_cast<long double>(result.value(indices[i],indices[j]))-
                static_cast<long double>(result.value(indices[i],0))*result.value(0,indices[j])/result.value(0,0));
    }
    return result;
}

double logAdd(double a, double b)
{
    const double maximum = std::max(a,b);
    return std::isfinite(maximum) ? maximum+std::log1p(std::exp(std::min(a,b)-maximum)) : maximum;
}

// E[W_+ 1(Y>0)] for unit-variance jointly Gaussian W,Y with means a,b.
// A positive log-scaled quadrature avoids cancellation of the usual first
// bivariate orthant moment in negative tails. Its omitted tail is bounded by
// the UNCONDITIONED positive W flux, so rare selected fluxes are not clipped.
struct FluxEstimate { double logValue, relativeError; };

FluxEstimate selectedFlux(double a, double b, double rho, const NumericPolicy& settings)
{
    if (!(1-rho*rho > 1e-12))
        throw NumericError(NumericStatus::NeedHigherPrecision,
                           "single-point teacher crossing distribution is nearly singular");
    const double sd = std::sqrt((1-rho)*(1+rho));
    const auto logIntegrand = [=](double v) {
        if (!(v > 0)) return -std::numeric_limits<double>::infinity();
        return std::log(v)+normalLogPdf(v-a)+normalLogCdf((b+rho*(v-a))/sd);
    };
    const auto derivative = [=](double v) {
        const double z = (b+rho*(v-a))/sd;
        return double(1.0L/v-(static_cast<long double>(v)-a)+
                      (static_cast<long double>(rho)/sd)*normalPdfOverCdf(z));
    };
    const double root = std::hypot(a,2.0);
    double lo = 0, hi = a >= 0 ? 0.5*(a+root) : 2/(root-a);
    for (int i = 0; derivative(hi) > 0; ++i) {
        if (i == 128 || !std::isfinite(hi*2))
            throw NumericError(NumericStatus::RootNotBracketed,"single-point teacher flux mode not bracketed");
        hi *= 2;
    }
    for (int i = 0; i < 96; ++i) {
        const double mid = lo+0.5*(hi-lo);
        if (!(mid > lo && mid < hi)) break;
        if (derivative(mid) > 0) lo = mid; else hi = mid;
    }
    const double mode = lo+0.5*(hi-lo), logScale = logIntegrand(mode);
    if (!(mode > 0) || !std::isfinite(logScale))
        throw NumericError(NumericStatus::UnresolvedRareEvent,"single-point teacher unresolved flux mode");
    const auto integrand = [&](double v) { return std::exp(logIntegrand(v)-logScale); };
    NumericPolicy policy = settings;
    policy.absoluteTolerance = std::numeric_limits<double>::min();
    policy.relativeTolerance = std::min(1e-8, settings.relativeTolerance*0.1);
    double end = std::max(mode+12, a+12);
    std::vector<double> cuts{0,end};
    const auto add = [&](double x) { if (x > 0 && x < end && std::isfinite(x)) cuts.push_back(x); };
    for (double scale : {0.125,0.25,0.5,1.0,2.0,4.0,8.0}) add(mode*scale);
    for (double offset : {-12.,-4.,-1.,0.,1.,4.,12.}) { add(mode+offset); add(a+offset); }
    if (rho != 0) {
        const double center = a-b/rho, width = sd/std::abs(rho);
        for (double offset : {-12.,-4.,-1.,0.,1.,4.,12.}) add(center+offset*width);
    }
    std::sort(cuts.begin(),cuts.end());
    cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    double sum = 0, error = 0;
    const auto accumulate = [&](double begin, double limit) {
        const auto part = integrateFinite(integrand,begin,limit,policy);
        if (part.status != NumericStatus::Ok)
            throw NumericError(part.status,"single-point teacher flux quadrature did not converge");
        sum += part.value; error += part.absError;
    };
    for (std::size_t i = 1; i < cuts.size(); ++i) accumulate(cuts[i-1],cuts[i]);
    for (int iteration = 0; iteration < 64; ++iteration) {
        const double logTail = logAdd(positiveRawMoment(1,a-end,1).logValue,
                                      std::log(end)+normalLogCdf(a-end));
        if (sum > 0 && logTail-logScale <= std::log(sum)+std::log(settings.relativeTolerance*0.1)) {
            const double relativeError = error/sum+std::exp(logTail-logScale-std::log(sum));
            if (relativeError > settings.relativeTolerance)
                throw NumericError(NumericStatus::IntegrationNotConverged,"single-point teacher flux error budget exceeded");
            return {logScale+std::log(sum),relativeError};
        }
        const double next = end+std::max(4.0,0.25*end);
        accumulate(end,next); end = next;
    }
    throw NumericError(NumericStatus::IntegrationNotConverged,"single-point teacher flux tail did not converge");
}

} // namespace

SinglePointHistoryTeacher::SinglePointHistoryTeacher(const GPSSField& field,
    const FlightState& birth, NumericPolicy policy)
    : baseline_(validatedField(field),birth), policy_(policy), sigma_(field.kernel.sigma()),
      inverseLength_(std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction)))
{
    if (!birth.direction.allFinite() || std::abs(birth.direction.norm()-1) > 1e-10 ||
        !(sigma_ > 0) || !std::isfinite(sigma_) || !(inverseLength_ > 0) || !std::isfinite(inverseLength_))
        throw NumericError(NumericStatus::InvalidInput,"single-point teacher requires a unit ray and finite SE parameters");
    if (!(policy.relativeTolerance > 0) || !std::isfinite(policy.relativeTolerance) ||
        !(policy.absoluteTolerance > 0) || !std::isfinite(policy.absoluteTolerance) ||
        policy.maxQuadratureSubdivisions < 16)
        throw NumericError(NumericStatus::InvalidInput,"single-point teacher invalid numerical policy");
}

SinglePointHistorySample SinglePointHistoryTeacher::evaluate(double t, double u) const
{
    if (!(u > 0 && t > u && t <= baseline_.maximumAgeInDomain()) || !std::isfinite(t))
        throw NumericError(NumericStatus::InvalidInput,"single-point teacher requires 0 < historyAge < currentAge inside the domain");
    const auto current = baseline_.scalarStatistics(t), history = baseline_.scalarStatistics(u);
    const auto covariance = jointCovariance(t*inverseLength_,u*inverseLength_);
    const Matrix3& c = covariance.value;
    const Vector3 m(current.meanF/sigma_,history.meanF/sigma_,-current.meanK/(sigma_*inverseLength_));
    Vector3 standard;
    for (int i = 0; i < 3; ++i) {
        if (!(c(i,i) > 0)) throw NumericError(NumericStatus::NeedHigherPrecision,"single-point teacher variance underflow");
        standard[i] = m[i]/std::sqrt(c(i,i));
    }
    SinglePointHistorySample result;
    result.features = {standard[0],standard[1],standard[2],
        correlation(c(0,1),c(0,0),c(1,1)),correlation(c(0,2),c(0,0),c(2,2)),
        correlation(c(1,2),c(1,1),c(2,2))};
    for (double x : result.features)
        if (!std::isfinite(x)) throw NumericError(NumericStatus::NeedHigherPrecision,"single-point teacher nonfinite feature");
    if (!(result.features[3] >= 0 && result.features[3] < 1-1e-12))
        throw NumericError(NumericStatus::NeedHigherPrecision,"single-point teacher value signs are nearly singular");
    result.logHistoryPositiveProbability = normalLogCdf(standard[1]);
    result.historyPositiveProbability = std::exp(result.logHistoryPositiveProbability);
    const auto base = baseline_.evaluate(t).hazard;
    if (!std::isfinite(base.logValue))
        throw NumericError(NumericStatus::UnresolvedRareEvent,"single-point teacher baseline has no finite log hazard");
    result.baselineHazard = base.value;
    result.logBaselineHazard = base.logValue;

    Eigen::Vector2d signs(standard[0],standard[1]);
    Matrix2 signsCov; signsCov << 1,result.features[3],result.features[3],1;
    SparseHistoryGpis::ProbabilitySettings probability;
    // The public orthant interface permits dropping almost-certain signs
    // under an absolute budget. Labels are probability ratios, so retain
    // every representable sign tail instead of applying an absolute cutoff.
    probability.absoluteTolerance = std::numeric_limits<double>::min();
    probability.relativeTolerance = std::min(1e-8,policy_.relativeTolerance*0.1);
    const double denominator = SparseHistoryGpis::positiveOrthant(signs,signsCov,probability);
    if (!(denominator > 0) || !std::isfinite(denominator))
        throw NumericError(NumericStatus::UnresolvedRareEvent,"single-point teacher joint survival probability underflow");
    const Matrix2& atZero = covariance.atZero;
    const double rho = correlation(atZero(0,1),atZero(0,0),atZero(1,1));
    const double speedMean = m[2]-(c(2,0)/c(0,0))*m[0];
    const double historyMean = m[1]-(c(1,0)/c(0,0))*m[0];
    const auto selected = selectedFlux(speedMean/std::sqrt(atZero(0,0)),
        historyMean/std::sqrt(atZero(1,1)),rho,policy_);
    const double logMoment = selected.logValue+0.5*std::log(atZero(0,0));
    result.logHistoryHazard = normalLogPdf(standard[0])-0.5*std::log(c(0,0))+
        std::log(inverseLength_)+logMoment-std::log(denominator);
    result.historyHazard = std::exp(result.logHistoryHazard);
    result.logCorrection = result.logHistoryHazard-result.logBaselineHazard;
    result.momentRelativeError = selected.relativeError;
    if (!std::isfinite(result.logCorrection) || !std::isfinite(result.historyHazard) ||
        !std::isfinite(result.baselineHazard))
        throw NumericError(NumericStatus::UnresolvedRareEvent,"single-point teacher nonfinite hazard or correction");
    if (result.logCorrection+result.logHistoryPositiveProbability >
            std::max(1e-9,16*policy_.relativeTolerance))
        throw NumericError(NumericStatus::InvalidMajorant,"single-point teacher violates the positive-association bound");
    return result;
}

} // namespace Tungsten::ConditionalGpis
