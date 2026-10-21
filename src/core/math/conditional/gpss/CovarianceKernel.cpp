// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/gpss/CovarianceKernel.h"
#include "math/conditional/mathutility/NumericPolicy.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <stdexcept>

namespace Tungsten::ConditionalGpis {

namespace {

Matrix3 validatePsd(Matrix3 matrix, const char* name) {
    matrix = 0.5 * (matrix + matrix.transpose());
    if (!matrix.allFinite()) {
        throw std::invalid_argument(std::string("invalid ") + name);
    }
    Eigen::SelfAdjointEigenSolver<Matrix3> eig(matrix);
    if (eig.info() != Eigen::Success || eig.eigenvalues().minCoeff() <
        -covarianceTolerance(std::max(1.0, matrix.norm()))) {
        throw NumericError(NumericStatus::InvalidCovariance,
                           std::string(name) + " must be PSD");
    }
    if (eig.eigenvalues().minCoeff() >= 0.0) return matrix;
    return eig.eigenvectors() * eig.eigenvalues().cwiseMax(0.0).asDiagonal() *
           eig.eigenvectors().transpose();
}

double kernelRadialGradientFactor(CovarianceKernelType type) {
    switch (type) {
    case CovarianceKernelType::SquaredExponential: return 1.0;
    case CovarianceKernelType::Matern32: return 1.0;
    case CovarianceKernelType::Matern52: return 5.0 / 3.0;
    }
    throw std::invalid_argument("unknown covariance kernel type");
}

} // namespace

const char* covarianceKernelTypeName(CovarianceKernelType type) {
    switch (type) {
    case CovarianceKernelType::SquaredExponential: return "squared_exponential";
    case CovarianceKernelType::Matern32: return "matern_3_2";
    case CovarianceKernelType::Matern52: return "matern_5_2";
    }
    throw std::invalid_argument("unknown covariance kernel type");
}

CovarianceKernelType parseCovarianceKernelType(const std::string& name) {
    if (name == "squared_exponential") return CovarianceKernelType::SquaredExponential;
    if (name == "matern_3_2") return CovarianceKernelType::Matern32;
    if (name == "matern_5_2") return CovarianceKernelType::Matern52;
    throw std::invalid_argument("unknown covariance kernel type: " + name);
}

CovarianceKernel::CovarianceKernel()
    : CovarianceKernel(CovarianceKernelType::SquaredExponential,
                       1.0, Matrix3::Identity()) {}

CovarianceKernel::CovarianceKernel(CovarianceKernelType type, double sigma, Matrix3 metric)
    : type_(type), sigma_(sigma), metric_(validatePsd(std::move(metric), "kernel metric")) {
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("kernel sigma must be finite and positive");
    }
    gradientCovarianceAtZero_ =
        (sigma_ * sigma_ * radialGradientFactor()) * metric_;
    if (!gradientCovarianceAtZero_.allFinite()) {
        throw std::invalid_argument("kernel gradient covariance is not finite");
    }
}

CovarianceKernel::CovarianceKernel(CovarianceKernelType type, double sigma, Matrix3 metric,
                                   Matrix3 gradientCovarianceAtZero)
    : type_(type), sigma_(sigma), metric_(validatePsd(std::move(metric), "kernel metric")),
      gradientCovarianceAtZero_(validatePsd(std::move(gradientCovarianceAtZero),
                                            "kernel gradient covariance")) {
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("kernel sigma must be finite and positive");
    }
}

double CovarianceKernel::radialGradientFactor() const {
    return kernelRadialGradientFactor(type_);
}

CovarianceKernel CovarianceKernel::fromCorrelationLengths(
    CovarianceKernelType type, double sigma, const Vector3& lengths,
    const Matrix3& rotation) {
    if (!rotation.allFinite() ||
        (rotation.transpose() * rotation - Matrix3::Identity()).norm() > 1e-10 ||
        std::abs(rotation.determinant() - 1.0) > 1e-10) {
        throw std::invalid_argument("kernel rotation must be orthonormal and right-handed");
    }
    Vector3 inverseSquares;
    for (int i = 0; i < 3; ++i) {
        if (std::isinf(lengths[i]) && lengths[i] > 0.0) inverseSquares[i] = 0.0;
        else if (lengths[i] > 0.0 && std::isfinite(lengths[i])) {
            inverseSquares[i] = 1.0 / (lengths[i] * lengths[i]);
        } else {
            throw std::invalid_argument("correlation lengths must be positive or +infinity");
        }
    }
    return {type, sigma, rotation * inverseSquares.asDiagonal() * rotation.transpose()};
}

CovarianceKernel CovarianceKernel::fromGradientCovariance(
    CovarianceKernelType type, double sigma, const Matrix3& covariance) {
    if (!(sigma > 0.0) || !std::isfinite(sigma)) {
        throw std::invalid_argument("kernel sigma must be finite and positive");
    }
    const Matrix3 checked = validatePsd(covariance, "kernel gradient covariance");
    const double factor = kernelRadialGradientFactor(type);
    const double denominator = sigma * sigma * factor;
    if (!(denominator > 0.0) || !std::isfinite(denominator)) {
        throw std::invalid_argument("kernel parameters cannot represent gradient covariance");
    }
    const Matrix3 metric = checked / denominator;
    if (!metric.allFinite() ||
        (checked.cwiseAbs().maxCoeff() > 0.0 && metric.cwiseAbs().maxCoeff() == 0.0)) {
        throw std::invalid_argument(
            "kernel parameters cannot represent a finite nonzero spatial metric");
    }
    return {type, sigma, metric, checked};
}

KernelJet CovarianceKernel::evaluate(const Point3& x, const Point3& y) const {
    const Vector3 delta = x - y;
    const Vector3 v = metric_ * delta;
    const double q2 = std::max(0.0, delta.dot(v));
    const double sigma2 = sigma_ * sigma_;
    if (type_ == CovarianceKernelType::SquaredExponential) {
        const double k = sigma2 * std::exp(-0.5 * q2);
        return {k, -v * k, v * k, (metric_ - v * v.transpose()) * k};
    }

    const double q = std::sqrt(q2);
    if (type_ == CovarianceKernelType::Matern32) {
        const double a = 1.0;
        const double e = std::exp(-a * q);
        const double k = sigma2 * (1.0 + a * q) * e;
        const Vector3 gx = -sigma2 * a * a * e * v;
        Matrix3 gg = sigma2 * a * a * e * metric_;
        if (q > 0.0) gg -= sigma2 * a * a * a * e / q * (v * v.transpose());
        return {k, gx, -gx, gg};
    }

    const double a = std::sqrt(5.0);
    const double e = std::exp(-a * q);
    const double a2 = a * a;
    const double k = sigma2 * (1.0 + a * q + a2 * q2 / 3.0) * e;
    const Vector3 gx = -sigma2 * (a2 / 3.0) * (1.0 + a * q) * e * v;
    const Matrix3 gg = sigma2 * (a2 / 3.0) * e *
        ((1.0 + a * q) * metric_ - a2 * v * v.transpose());
    return {k, gx, -gx, gg};
}

} // namespace Tungsten::ConditionalGpis
