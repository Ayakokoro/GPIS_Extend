// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Types.h"
#include <string>
#include <utility>

namespace Tungsten::ConditionalGpis {

struct KernelJet {
    double valueValue = 0.0;
    Vector3 gradientXValueY = Vector3::Zero();
    Vector3 valueXGradientY = Vector3::Zero();
    Matrix3 gradientXGradientY = Matrix3::Zero();
};

enum class CovarianceKernelType {
    SquaredExponential,
    Matern32,
    Matern52
};

const char* covarianceKernelTypeName(CovarianceKernelType type);
CovarianceKernelType parseCovarianceKernelType(const std::string& name);

// Common value-type interface for stationary anisotropic covariance kernels.
//
// The spatial metric P defines q^2=(x-y)^T P (x-y).  It is deliberately
// separate from gradientCovarianceAtZero(): the latter includes the
// family-dependent curvature at q=0 and is the only kernel statistic used by
// classic transport.  evaluate() exposes the complete value/gradient
// two-point covariance needed by conditional transport.
class CovarianceKernel {
public:
    CovarianceKernel();
    CovarianceKernel(CovarianceKernelType type, double sigma, Matrix3 metric);
    // Compatibility constructor: historically every kernel was SE.
    CovarianceKernel(double sigma, Matrix3 metric)
        : CovarianceKernel(CovarianceKernelType::SquaredExponential,
                           sigma, std::move(metric)) {}

    static CovarianceKernel fromCorrelationLengths(
        CovarianceKernelType type, double sigma, const Vector3& lengths,
        const Matrix3& rotation = Matrix3::Identity());
    static CovarianceKernel fromCorrelationLengths(
        double sigma, const Vector3& lengths,
        const Matrix3& rotation = Matrix3::Identity()) {
        return fromCorrelationLengths(CovarianceKernelType::SquaredExponential,
                                      sigma, lengths, rotation);
    }
    static CovarianceKernel fromGradientCovariance(
        CovarianceKernelType type, double sigma, const Matrix3& covariance);

    KernelJet evaluate(const Point3& x, const Point3& y) const;
    double sigma() const { return sigma_; }
    CovarianceKernelType type() const { return type_; }
    const Matrix3& metric() const { return metric_; }
    // Compatibility name for existing SE callers. New code should use metric().
    const Matrix3& precision() const { return metric_; }
    const Matrix3& gradientCovarianceAtZero() const {
        return gradientCovarianceAtZero_;
    }
    double radialGradientFactor() const;
    bool supportsAnalyticConditionalTransport() const {
        return type_ == CovarianceKernelType::SquaredExponential;
    }

private:
    CovarianceKernel(CovarianceKernelType type, double sigma, Matrix3 metric,
                     Matrix3 gradientCovarianceAtZero);

    CovarianceKernelType type_ = CovarianceKernelType::SquaredExponential;
    double sigma_ = 1.0;
    Matrix3 metric_ = Matrix3::Identity();
    Matrix3 gradientCovarianceAtZero_ = Matrix3::Identity();
};

// Transitional spelling retained for source compatibility. New code should
// use CovarianceKernel and select a CovarianceKernelType explicitly.
using SquaredExponentialKernel = CovarianceKernel;

} // namespace Tungsten::ConditionalGpis
