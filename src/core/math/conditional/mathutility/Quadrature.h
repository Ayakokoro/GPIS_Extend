// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/mathutility/NumericPolicy.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <queue>
#include <vector>

namespace Tungsten::ConditionalGpis {

namespace detail {

struct QuadratureInterval {
    double lo, hi, value, error;
};

struct ErrorLess {
    bool operator()(const QuadratureInterval& a, const QuadratureInterval& b) const {
        return a.error < b.error;
    }
};

inline QuadratureInterval gaussKronrod15(const std::function<double(double)>& f,
                                         double lo, double hi) {
    static constexpr std::array<double, 8> xgk{{
        0.9914553711208126392, 0.9491079123427585245,
        0.8648644233597690728, 0.7415311855993944399,
        0.5860872354676911303, 0.4058451513773971669,
        0.2077849550078984676, 0.0}};
    static constexpr std::array<double, 8> wgk{{
        0.02293532201052922496, 0.06309209262997855329,
        0.1047900103222501838, 0.1406532597155259187,
        0.1690047266392679028, 0.1903505780647854099,
        0.2044329400752988924, 0.2094821410847278280}};
    static constexpr std::array<double, 4> wg{{
        0.1294849661688696933, 0.2797053914892766679,
        0.3818300505051189449, 0.4179591836734693878}};

    const double center = 0.5 * (lo + hi);
    const double half = 0.5 * (hi - lo);
    const double fc = f(center);
    if (!std::isfinite(fc)) {
        throw NumericError(NumericStatus::InvalidInput, "non-finite quadrature integrand");
    }
    double kronrod = wgk[7] * fc;
    double gauss = wg[3] * fc;
    for (int i = 0; i < 7; ++i) {
        const double dx = half * xgk[static_cast<std::size_t>(i)];
        const double pair = f(center - dx) + f(center + dx);
        if (!std::isfinite(pair)) {
            throw NumericError(NumericStatus::InvalidInput, "non-finite quadrature integrand");
        }
        kronrod += wgk[static_cast<std::size_t>(i)] * pair;
        if (i == 1) gauss += wg[0] * pair;
        if (i == 3) gauss += wg[1] * pair;
        if (i == 5) gauss += wg[2] * pair;
    }
    kronrod *= half;
    gauss *= half;
    const double error = std::abs(kronrod - gauss) * 1.5 +
                         32.0 * std::numeric_limits<double>::epsilon() * std::abs(kronrod);
    return {lo, hi, kronrod, error};
}

} // namespace detail

inline IntegralResult integrateFinite(const std::function<double(double)>& f,
                                      double lo, double hi,
                                      const NumericPolicy& policy = defaultNumericPolicy()) {
    if (!std::isfinite(lo) || !std::isfinite(hi) || hi < lo) {
        throw NumericError(NumericStatus::InvalidInput, "invalid finite integration interval");
    }
    if (lo == hi) return {};
    std::priority_queue<detail::QuadratureInterval,
                        std::vector<detail::QuadratureInterval>, detail::ErrorLess> queue;
    auto initial = detail::gaussKronrod15(f, lo, hi);
    queue.push(initial);
    double value = initial.value;
    double error = initial.error;
    int subdivisions = 1;
    auto converged = [&]() {
        return error <= std::max(policy.absoluteTolerance,
                                 policy.relativeTolerance * std::abs(value));
    };
    while (!converged() && subdivisions < policy.maxQuadratureSubdivisions) {
        const auto parent = queue.top();
        queue.pop();
        const double mid = 0.5 * (parent.lo + parent.hi);
        const auto left = detail::gaussKronrod15(f, parent.lo, mid);
        const auto right = detail::gaussKronrod15(f, mid, parent.hi);
        value += left.value + right.value - parent.value;
        error += left.error + right.error - parent.error;
        error = std::max(0.0, error);
        queue.push(left);
        queue.push(right);
        ++subdivisions;
    }
    return {value, error, subdivisions,
            converged() ? NumericStatus::Ok : NumericStatus::IntegrationNotConverged};
}

inline IntegralResult integrateSemiInfinite(const std::function<double(double)>& f,
                                            double lower,
                                            const NumericPolicy& policy = defaultNumericPolicy()) {
    if (!std::isfinite(lower)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid semi-infinite lower bound");
    }
    auto transformed = [&](double u) {
        if (u <= 0.0) return f(lower);
        if (u >= 1.0) return 0.0;
        const double inv = 1.0 / (1.0 - u);
        const double x = lower + u * inv;
        return f(x) * inv * inv;
    };
    return integrateFinite(transformed, 0.0, 1.0, policy);
}

inline PositiveResult integratePositiveLog(const std::function<double(double)>& logf,
                                           double lower, double scaleHint,
                                           const NumericPolicy& policy = defaultNumericPolicy()) {
    if (!std::isfinite(lower) || !std::isfinite(scaleHint)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid log-integral parameters");
    }
    double logScale = logf(std::max(lower, scaleHint));
    if (!std::isfinite(logScale)) {
        logScale = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < 64; ++i) {
            const double x = lower + std::exp(-8.0 + i * 0.25);
            logScale = std::max(logScale, logf(x));
        }
    }
    if (!std::isfinite(logScale)) return exactZero();
    auto scaled = [&](double x) {
        const double y = logf(x) - logScale;
        if (y < std::log(std::numeric_limits<double>::min())) return 0.0;
        if (y > std::log(std::numeric_limits<double>::max())) {
            throw NumericError(NumericStatus::NeedHigherPrecision,
                               "log-integral scale hint missed the integrand mode");
        }
        return std::exp(y);
    };
    const double split = std::max(lower, scaleHint);
    IntegralResult integral;
    if (split > lower) {
        const IntegralResult left = integrateFinite(scaled, lower, split, policy);
        const IntegralResult right = integrateSemiInfinite(scaled, split, policy);
        integral.value = left.value + right.value;
        integral.absError = left.absError + right.absError;
        integral.subdivisions = left.subdivisions + right.subdivisions;
        integral.status = left.status == NumericStatus::Ok && right.status == NumericStatus::Ok
                              ? NumericStatus::Ok : NumericStatus::IntegrationNotConverged;
    } else {
        integral = integrateSemiInfinite(scaled, lower, policy);
    }
    if (!(integral.value > 0.0)) {
        if (integral.status == NumericStatus::IntegrationNotConverged) {
            throw NumericError(integral.status, "positive log integral did not converge");
        }
        return exactZero();
    }
    const double logValue = logScale + std::log(integral.value);
    const double relativeError = integral.absError / integral.value;
    return positiveFromLog(logValue, std::exp(logValue) * relativeError, integral.status);
}

} // namespace Tungsten::ConditionalGpis
