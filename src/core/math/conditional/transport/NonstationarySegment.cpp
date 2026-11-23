#include "NonstationarySegment.h"
#include "ConditionalFlightKernel.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include <algorithm>
#include <cmath>

namespace Tungsten::ConditionalGpis {
namespace {

FlightState directedBirth(FlightState birth, const Vector3 &direction) {
    birth.direction = normalizedOrThrow(direction);
    birth.age = 0;
    return birth;
}

struct PositiveValue {
    double value, deviation;
};

PositiveValue positiveValue(double mean, double variance, Random &rng) {
    if (!(variance > 0.0) || !std::isfinite(variance))
        throw NumericError(NumericStatus::NeedHigherPrecision, "invalid conditional endpoint variance");
    const double sd = std::sqrt(variance);
    const double lower = -mean/sd;
    if (lower > 8) {
        // Sample the excess of the truncated standard normal directly. For a
        // rare exterior point mean+sd*z would cancel its entire positive value.
        const double rate = 0.5*lower + std::hypot(0.5*lower, 1.0);
        for (;;) {
            const double excess = -std::log(rng.openUniform01())/rate;
            const double shift = excess - 1/rate;
            if (std::log(rng.openUniform01()) <= -0.5*shift*shift) {
                const double value = std::max(std::nextafter(0.0, 1.0), sd*excess);
                return {value, value-mean};
            }
        }
    }
    const double logTail = normalLogCdf(mean/sd) + std::log(rng.openUniform01());
    const double deviation = -sd*normalQuantileFromLogCdf(logTail);
    return {std::max(std::nextafter(0.0, 1.0), mean+deviation), deviation};
}

FlightState samplePositive(const Point3 &position, const Vector3 &direction,
        double meanF, double varianceF, const Vector3 &meanG, const Vector3 &regression,
        const Matrix3 &residual, Random &rng) {
    const auto sampled = positiveValue(meanF, varianceF, rng);
    FlightState result = startExternalFlight(position, direction);
    result.birthValue = sampled.value;
    result.birthGradient = sampleGaussianPSD(Gaussian<3>{meanG + regression*sampled.deviation, residual}, rng);
    return result;
}

double crossingRate(const ConditionalScalarStatistics &s, bool upcrossing) {
    if (!(s.varianceF > 0.0))
        throw NumericError(NumericStatus::NeedHigherPrecision, "invalid conditional flight variance");
    const double sd = std::sqrt(s.varianceF);
    const double z = s.meanF/sd;
    const double logMills = z < -8.0 ? std::log(normalPdfOverCdf(z)) : normalLogPdf(z)-normalLogCdf(z);
    const double flux = negativePartMean(upcrossing ? -s.meanAtZero : s.meanAtZero,
                                         std::sqrt(s.varianceAtZero)).logValue;
    return std::exp(logMills - std::log(sd) + flux);
}
}

FlightState startNonstationaryExterior(const NonstationaryConditioning &kernel, const MeanField &mean,
        const Point3 &position, const Vector3 &direction, Random &rng) {
    const auto p = kernel.pointPrior(position, mean.evaluate(position));
    const Vector3 regression = p.covarianceFG/p.varianceF;
    const Matrix3 residual = p.covarianceG - regression*p.covarianceFG.transpose();
    return samplePositive(position, direction, p.meanF, p.varianceF, p.meanG, regression, residual, rng);
}

NonstationarySegment::NonstationarySegment(const NonstationaryConditioning &kernel, const MeanField &mean,
        const FlightState &birth, const Point3 &origin, const Vector3 &direction, double length,
        double trackingStep)
    : _kernel(kernel), _mean(mean), _birth(directedBirth(birth, direction)), _origin(origin),
      _length(length), _age((origin-birth.birthPosition).dot(_birth.direction)),
      _maximumStep(0.5*kernel.minLength()), _birthStep(_maximumStep),
      _flight(kernel, mean, _birth), _cursor(mean.createRayCursor(origin, _birth.direction)) {
    if (!(length >= 0.0 && trackingStep >= 0.0 && birth.birthValue >= 0.0) ||
        !std::isfinite(_age+length) || !std::isfinite(trackingStep))
        throw NumericError(NumericStatus::InvalidInput, "invalid nonstationary flight interval");
    if (trackingStep > 0.0) _maximumStep = std::min(_maximumStep, trackingStep);
    const Vector3 displacement = origin - birth.birthPosition;
    const double tolerance = 32*std::numeric_limits<double>::epsilon()*
        (origin.norm() + birth.birthPosition.norm() + displacement.norm());
    if (!(_age >= -tolerance) || !((displacement - _age*_birth.direction).norm() <= tolerance))
        throw NumericError(NumericStatus::InvalidInput, "conditional flight turned without an endpoint observation");
    _age = std::max(0.0, _age);
    if (length == 0) return;

    // A grazing surface birth or a tiny positive birth may have a first peak
    // much closer than one correlation length. Resolve the linear separation
    // from zero against the local O(t^2) residual before growing the steps.
    const auto &start = _birth.birthPosition;
    const auto jet = mean.evaluate(start);
    const auto prior = kernel.pointPrior(start, jet);
    const double firstEnd = std::min(_age+length, _maximumStep);
    // A segment can start part way along a straight flight. Its conditioning
    // observation and near-birth step are still measured from the old birth.
    auto originalCursor = _age > 0 ? mean.createRayCursor(start, _birth.direction) : nullptr;
    auto *birthCursor = _age > 0 ? originalCursor.get() : _cursor.get();
    const double firstCellEnd = birthCursor ? birthCursor->nextBoundary(0, firstEnd) : firstEnd;
    const auto range = birthCursor ? birthCursor->rayBounds(0, firstCellEnd) :
                                     mean.rayBounds(start, _birth.direction, 0, firstCellEnd);
    double slope = _birth.direction.dot(_birth.birthGradient);
    if (range.certified) slope += range.beginDerivative - _birth.direction.dot(jet.gradient);
    if (_birth.birthValue == 0.0 && !(slope > 0.0))
        throw NumericError(NumericStatus::NonPositiveSurfaceBirth, "surface birth must enter the positive side");
    const double sigma = std::sqrt(prior.varianceF);
    double curvature = (sigma + std::abs(_birth.birthValue-jet.value)) /
                       (kernel.minLength()*kernel.minLength());
    if (range.certified && std::isfinite(range.maximumSecondDerivative))
        curvature += range.maximumSecondDerivative;
    if (_birth.birthValue == 0.0) {
        _birthStep = std::min(_maximumStep, 0.25*slope/curvature);
    } else {
        _birthStep = std::min(_maximumStep, 0.25*std::sqrt(_birth.birthValue/curvature));
        if (slope != 0) _birthStep = std::min(_birthStep, 0.25*_birth.birthValue/std::abs(slope));
    }
}

ConditionalScalarStatistics NonstationarySegment::statistics(double t) const {
    if (!(t >= 0 && t <= _length && _age+t > 0))
        throw NumericError(NumericStatus::InvalidInput, "conditional statistics require a positive flight age");
    const auto m = _flight.moments(_age+t);
    const double slope = _birth.direction.dot(m.meanG);
    return {m.meanF, m.varianceF, slope, slope - _birth.direction.dot(m.regression)*m.meanF,
            m.slopeVarianceGivenValue};
}

double NonstationarySegment::hazard(double t) const {
    return _age+t == 0 ? 0.0 : crossingRate(statistics(t), false);
}

Gaussian<3> NonstationarySegment::gradient(double t) const {
    if (!(t >= 0 && t <= _length && _age+t > 0))
        throw NumericError(NumericStatus::InvalidInput, "collision gradient requires a positive flight age");
    const auto m = _flight.moments(_age+t);
    return {m.meanG - m.regression*m.meanF, m.gradientGivenValueCovariance};
}

FlightState NonstationarySegment::samplePositiveEndpoint(Random &rng) const {
    if (_age+_length == 0.0) return _birth;
    const auto m = _flight.moments(_age+_length);
    return samplePositive(_origin + _length*_birth.direction, _birth.direction,
                          m.meanF, m.varianceF, m.meanG, m.regression, m.gradientGivenValueCovariance, rng);
}

double NonstationarySegment::nextIntervalEnd(double begin) const {
    const double step = std::min(_maximumStep, std::max(_birthStep, 0.5*(_age+begin)));
    double end = std::min(_length, begin+step);
    end = std::min(end, _kernel.nextBreakpoint(_origin, _birth.direction, begin));
    if (_cursor) end = _cursor->nextBoundary(begin, end);
    else {
        std::vector<double> breaks;
        _mean.appendRayBreakpoints(_origin, _birth.direction, begin, end, breaks);
        for (double point : breaks) if (point > begin) end = std::min(end, point);
    }
    if (!(end > begin))
        throw NumericError(NumericStatus::IntegrationNotConverged, "nonstationary integration step cannot advance");

    SdfRayQueryOptions options;
    options.computeBeginDerivative = false;
    const auto range = _cursor ? _cursor->rayBounds(begin, end, options) :
                                 _mean.rayBounds(_origin, _birth.direction, begin, end, options);
    // Even well away from birth a steep mean can cross several standard
    // deviations within one voxel. Cap the change near the zero level.
    if (range.certified) {
        const double rate = std::max(std::abs(range.minimumDerivative), std::abs(range.maximumDerivative));
        const double distance = std::max(range.minimumValue, -range.maximumValue);
        const double scale = std::max(_kernel.minSigma(), 0.5*distance);
        if (rate > 0.0 && std::isfinite(rate)) end = std::min(end, begin + scale/rate);
    }
    return std::max(std::nextafter(begin, _length), end);
}

double NonstationarySegment::logPositive(double t) const {
    if (_age+t == 0) return 0;
    // The right endpoint is the next interval's left endpoint.
    if (t == _logPositiveAt) return _logPositive;
    const auto s = statistics(t);
    _logPositive = normalLogCdf(s.meanF/std::sqrt(s.varianceF));
    _logPositiveAt = t;
    return _logPositive;
}

double NonstationarySegment::opticalDepth(double begin, double end) const {
    if (begin == end) return 0;
    // h_down - h_up = -d log P(F>0)/dt. On a sharp downward crossing
    // integrate the smaller upcrossing rate and recover the steep peak from
    // its endpoint probabilities. This remains valid across trilinear cells.
    const double left = logPositive(begin);
    const double decrease = left - logPositive(end);
    const bool upward = decrease > 1.0;
    const auto rate = [&](double t) {
        if (_age+t == 0) return 0.0;
        return crossingRate(statistics(t), upward);
    };
    NumericPolicy policy;
    policy.relativeTolerance = 1e-5;
    policy.absoluteTolerance = 1e-6*(end-begin)/_length;
    const double midpoint = begin + 0.5*(end-begin);
    const double coarse = rate(midpoint)*(end-begin);
    const double fine = 0.5*(end-begin)*(rate(begin+0.25*(end-begin)) + rate(begin+0.75*(end-begin)));
    double depth;
    if (std::abs(fine-coarse) <= 3*std::max(policy.absoluteTolerance, policy.relativeTolerance*fine)) {
        depth = std::max(0.0, fine + (fine-coarse)/3);
    } else {
        const auto result = integrateFinite(rate, begin, end, policy);
        if (result.status != NumericStatus::Ok)
            throw NumericError(result.status, "nonstationary optical-depth integration did not converge");
        depth = result.value;
    }
    return depth + (upward ? decrease : 0.0);
}

double NonstationarySegment::track(Random &rng, bool collision, double &hit, double begin) const {
    if (!(begin >= 0 && begin <= _length))
        throw NumericError(NumericStatus::InvalidInput, "invalid nonstationary tracking start");
    hit = _length;
    if (begin == _length) return 1.0;
    const double target = collision ? -std::log(rng.openUniform01()) : std::numeric_limits<double>::infinity();
    double total = 0;
    for (double a = begin; a < _length;) {
        const double b = nextIntervalEnd(a);
        const double depth = opticalDepth(a, b);
        if (collision && depth >= target-total) {
            double lo = a, hi = b;
            double remaining = target-total, spanDepth = depth;
            for (int i=0; i<96; ++i) {
                const double alpha = std::clamp(remaining/spanDepth, 0.2, 0.8);
                const double mid = lo + alpha*(hi-lo);
                if (!(mid > lo && mid < hi)) break;
                const double leftDepth = opticalDepth(lo, mid);
                if (std::abs(leftDepth-remaining) <= 1e-8*target) { hit = mid; return 0.0; }
                if (leftDepth < remaining) {
                    remaining -= leftDepth;
                    spanDepth = std::max(remaining, spanDepth-leftDepth);
                    lo = mid;
                } else {
                    spanDepth = leftDepth;
                    hi = mid;
                }
                const double scale = std::min(_maximumStep, _age+hi);
                if (hi-lo <= scale*1e-7) break;
            }
            hit = lo + 0.5*(hi-lo);
            return 0.0;
        }
        total += depth;
        if (!collision && std::exp(-total) == 0) return 0.0;
        a = b;
    }
    return collision ? 1.0 : std::exp(-total);
}

} // namespace Tungsten::ConditionalGpis
