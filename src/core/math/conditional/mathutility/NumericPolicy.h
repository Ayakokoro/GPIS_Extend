// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

namespace Tungsten::ConditionalGpis {

enum class NumericStatus {
    Ok,
    ExactZero,
    Degenerate,
    UnderflowValueWithFiniteLog,
    NeedHigherPrecision,
    InvalidCovariance,
    InvalidInput,
    IntegrationNotConverged,
    RootNotBracketed,
    SamplingNotConverged,
    InvalidMajorant,
    UnsupportedSingularFlight,
    UnsupportedDegenerateNdf,
    UnresolvedRareEvent
};

inline const char* toString(NumericStatus s) {
    switch (s) {
    case NumericStatus::Ok: return "ok";
    case NumericStatus::ExactZero: return "exact_zero";
    case NumericStatus::Degenerate: return "degenerate";
    case NumericStatus::UnderflowValueWithFiniteLog: return "underflow_with_finite_log";
    case NumericStatus::NeedHigherPrecision: return "need_higher_precision";
    case NumericStatus::InvalidCovariance: return "invalid_covariance";
    case NumericStatus::InvalidInput: return "invalid_input";
    case NumericStatus::IntegrationNotConverged: return "integration_not_converged";
    case NumericStatus::RootNotBracketed: return "root_not_bracketed";
    case NumericStatus::SamplingNotConverged: return "sampling_not_converged";
    case NumericStatus::InvalidMajorant: return "invalid_majorant";
    case NumericStatus::UnsupportedSingularFlight: return "unsupported_singular_flight";
    case NumericStatus::UnsupportedDegenerateNdf: return "unsupported_degenerate_ndf";
    case NumericStatus::UnresolvedRareEvent: return "unresolved_rare_event";
    }
    return "unknown";
}

struct PositiveResult {
    double value = 0.0;
    double logValue = -std::numeric_limits<double>::infinity();
    double absError = 0.0;
    NumericStatus status = NumericStatus::ExactZero;
};

struct IntegralResult {
    double value = 0.0;
    double absError = 0.0;
    int subdivisions = 0;
    NumericStatus status = NumericStatus::Ok;
};

struct RootResult {
    double value = 0.0;
    double residual = 0.0;
    int iterations = 0;
    NumericStatus status = NumericStatus::Ok;
};

struct NumericPolicy {
    double relativeTolerance = 1e-7;
    double absoluteTolerance = 1e-10;
    int maxQuadratureSubdivisions = 4096;
    int maxRootIterations = 128;
    double covarianceRoundoffMultiplier = 128.0;
    // Distance tolerances are used by Gaussian sampling root solvers.
    double distanceAbsoluteTolerance = 1e-10;
    double distanceRelativeTolerance = 1e-8;
};

inline NumericPolicy& defaultNumericPolicy() {
    static NumericPolicy policy;
    return policy;
}

class NumericError : public std::runtime_error {
public:
    NumericError(NumericStatus status, const std::string& message)
        : std::runtime_error(message), status_(status) {}
    NumericStatus status() const noexcept { return status_; }
private:
    NumericStatus status_;
};

inline PositiveResult exactZero() { return {}; }

inline PositiveResult positiveFromLog(double logValue, double absError = 0.0,
                                      NumericStatus status = NumericStatus::Ok) {
    if (logValue == -std::numeric_limits<double>::infinity()) return exactZero();
    PositiveResult result;
    result.logValue = logValue;
    result.absError = absError;
    result.value = std::exp(logValue);
    result.status = result.value == 0.0 ? NumericStatus::UnderflowValueWithFiniteLog : status;
    return result;
}

inline double covarianceTolerance(double scale, const NumericPolicy& policy = defaultNumericPolicy()) {
    return policy.covarianceRoundoffMultiplier * std::numeric_limits<double>::epsilon() *
           std::max(1.0, std::abs(scale));
}

inline double validateNonnegative(double value, double scale,
                                  const NumericPolicy& policy = defaultNumericPolicy()) {
    if (value >= 0.0) return value;
    if (std::abs(value) <= covarianceTolerance(scale, policy)) return 0.0;
    throw NumericError(NumericStatus::InvalidInput, "negative value exceeds roundoff tolerance");
}

} // namespace Tungsten::ConditionalGpis
