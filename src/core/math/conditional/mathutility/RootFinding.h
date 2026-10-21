// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/mathutility/NumericPolicy.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <utility>

namespace Tungsten::ConditionalGpis {

struct RootEvaluation {
    double value;
    double derivative;
    double error = 0.0;
};

// The caller supplies an integration error estimate when f is an integral.
// Only unambiguous signs may shrink the bracket. A failed solve is never escape.
inline RootResult solveMonotoneSafeguardedNewton(
    const std::function<RootEvaluation(double)>& evaluate, double target,
    double lo, double hi, double initial, const NumericPolicy& policy,
    int* bisectionSteps = nullptr) {
    const auto left = evaluate(lo);
    const auto right = evaluate(hi);
    if (!std::isfinite(target) || !(lo <= hi) ||
        !std::isfinite(left.value) || !std::isfinite(right.value) ||
        left.value > target || right.value < target) {
        return {lo, left.value - target, 0, NumericStatus::RootNotBracketed};
    }
    const double ftol = policy.absoluteTolerance + policy.relativeTolerance * std::abs(target);
    for (const auto& endpoint : {std::make_pair(lo, left), std::make_pair(hi, right)}) {
        const double residual = endpoint.second.value - target;
        const double uncertainty = std::abs(residual) + endpoint.second.error;
        const double xtol = policy.distanceAbsoluteTolerance +
                            policy.distanceRelativeTolerance * std::abs(endpoint.first);
        if (uncertainty <= ftol && endpoint.second.derivative > 0.0 &&
            uncertainty / endpoint.second.derivative <= xtol)
            return {endpoint.first, residual, 0, NumericStatus::Ok};
    }
    double x = initial > lo && initial < hi ? initial : lo + (hi - lo) * 0.5;
    double previousStep = hi - lo;
    double residual = 0.0;
    for (int iteration = 1; iteration <= policy.maxRootIterations; ++iteration) {
        const auto e = evaluate(x);
        residual = e.value - target;
        if (!std::isfinite(residual) || !std::isfinite(e.error) || e.error < 0.0 ||
            !std::isfinite(e.derivative) || e.derivative < 0.0) {
            throw NumericError(NumericStatus::InvalidInput, "invalid Newton evaluation");
        }
        const double xtol = policy.distanceAbsoluteTolerance +
                            policy.distanceRelativeTolerance * std::abs(x);
        const double correction = e.derivative > 0.0
            ? residual / e.derivative : std::numeric_limits<double>::infinity();
        if (std::abs(residual) + e.error <= ftol &&
            (residual == 0.0 || hi - lo <= 2.0 * xtol ||
             (e.derivative > 0.0 && (std::abs(residual) + e.error) / e.derivative <= xtol))) {
            return {x, residual, iteration, NumericStatus::Ok};
        }
        if (residual < -e.error) lo = x;
        else if (residual > e.error) hi = x;
        else return {x, residual, iteration, NumericStatus::NeedHigherPrecision};
        double candidate = x - correction;
        if (!(candidate > lo && candidate < hi) ||
            std::abs(correction) > 0.75 * previousStep || candidate == x) {
            candidate = lo + (hi - lo) * 0.5;
            if (bisectionSteps) ++*bisectionSteps;
        }
        previousStep = std::abs(candidate - x);
        if (candidate == x) return {x, residual, iteration, NumericStatus::NeedHigherPrecision};
        x = candidate;
    }
    return {x, residual, policy.maxRootIterations, NumericStatus::NeedHigherPrecision};
}

inline RootResult solveMonotoneIncreasing(const std::function<double(double)>& f,
                                          double target, double lo, double hi,
                                          const NumericPolicy& policy = defaultNumericPolicy()) {
    double flo = f(lo) - target;
    double fhi = f(hi) - target;
    if (!std::isfinite(flo) || !std::isfinite(fhi) || flo > 0.0 || fhi < 0.0) {
        return {lo, flo, 0, NumericStatus::RootNotBracketed};
    }
    if (flo == 0.0) return {lo, 0.0, 0, NumericStatus::Ok};
    if (fhi == 0.0) return {hi, 0.0, 0, NumericStatus::Ok};
    double mid = lo;
    double fm = flo;
    for (int iteration = 1; iteration <= policy.maxRootIterations; ++iteration) {
        mid = lo + 0.5 * (hi - lo);
        fm = f(mid) - target;
        if (!std::isfinite(fm)) {
            throw NumericError(NumericStatus::InvalidInput, "non-finite root function");
        }
        const double xTolerance = policy.absoluteTolerance +
                                  policy.relativeTolerance * std::max(1.0, std::abs(mid));
        if (fm == 0.0 || hi - lo <= 2.0 * xTolerance) {
            return {mid, fm, iteration, NumericStatus::Ok};
        }
        if (fm < 0.0) lo = mid;
        else hi = mid;
    }
    return {mid, fm, policy.maxRootIterations, NumericStatus::NeedHigherPrecision};
}

inline RootResult expandAndSolveNegativeDomain(const std::function<double(double)>& f,
                                               double target, double upper,
                                               const NumericPolicy& policy = defaultNumericPolicy()) {
    double width = 1.0;
    double lower = upper - width;
    for (int i = 0; i < 1024 && f(lower) > target; ++i) {
        width *= 2.0;
        lower = upper - width;
        if (!std::isfinite(lower)) break;
    }
    return solveMonotoneIncreasing(f, target, lower, upper, policy);
}

} // namespace Tungsten::ConditionalGpis
