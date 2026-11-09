#include "FixedSpeedGradientDistribution.hpp"
#include <Eigen/Cholesky>
#include <cmath>
#include <stdexcept>

namespace Tungsten::ConditionalGpis {

FixedSpeedGradientDistribution::FixedSpeedGradientDistribution(const CovarianceKernel &kernel,
    const Vector3 &direction, double q, double derivative, const Vector3 &meanGradient,
    const Vector3 &birthMeanGradient, const std::optional<Vector3> &birthGradient)
    : _derivative(derivative)
{
    if (kernel.type()!=CovarianceKernelType::SquaredExponential || !(q>=0) || !std::isfinite(q) ||
        !(derivative<0) || !std::isfinite(derivative) || !meanGradient.allFinite() ||
        !birthMeanGradient.allFinite() || (birthGradient && !birthGradient->allFinite()))
        throw std::invalid_argument("invalid fixed-speed gradient inputs");
    _direction=normalizedOrThrow(direction);
    Vector3 e1,e2;
    orthonormalComplement(_direction,e1,e2);
    _basis << e1,e2;
    const Matrix3 covariance=kernel.gradientCovarianceAtZero();
    const Vector3 cw=covariance*_direction;
    const double variance=_direction.dot(cw);
    if (!(variance>0)) throw std::invalid_argument("degenerate directional gradient variance");
    const Vector3 regression=cw/variance;
    const Matrix3 residualCovariance=covariance-cw*regression.transpose();
    const Matrix2 transverse=_basis.transpose()*residualCovariance*_basis;
    const Eigen::LLT<Matrix2> chol(transverse);
    if (chol.info()!=Eigen::Success)
        throw std::invalid_argument("fixed-speed gradient requires positive transverse covariance");
    Vector2 residual=Vector2::Zero();
    double varianceFactor=1;
    if (birthGradient) {
        const Vector3 previous=*birthGradient-birthMeanGradient;
        const double decay=0.5*q*q;
        residual=std::exp(-decay)*(_basis.transpose()*(previous-regression*_direction.dot(previous)));
        varianceFactor=-std::expm1(-2*decay);
    }
    const Vector3 longitudinal=meanGradient+regression*(derivative-_direction.dot(meanGradient));
    _mean=_basis.transpose()*longitudinal+residual;
    _lower=std::sqrt(varianceFactor)*chol.matrixL().toDenseMatrix();
    _delta=varianceFactor==0;
    if (!_delta)
        _logNormalization=-std::log(2*kPi)-_lower.diagonal().array().log().sum();
}

Vector3 FixedSpeedGradientDistribution::sample(Random &rng) const
{
    const double z1=rng.standardNormal(),z2=rng.standardNormal();
    return _derivative*_direction+_basis*(_mean+_lower*Vector2(z1,z2));
}

std::optional<Vector3> FixedSpeedGradientDistribution::gradientForDirection(const Vector3 &outgoing) const
{
    const Vector3 difference=normalizedOrThrow(outgoing)-_direction;
    const double distanceSquared=difference.squaredNorm();
    if (distanceSquared==0 || _delta) return std::nullopt;
    const Vector2 transverse=(-2*_derivative/distanceSquared)*(_basis.transpose()*difference);
    const Vector3 gradient=_derivative*_direction+_basis*transverse;
    if (!gradient.allFinite()) throw std::runtime_error("nonfinite fixed-speed connection gradient");
    return gradient;
}

double FixedSpeedGradientDistribution::phasePdf(const Vector3 &outgoing) const
{
    const auto gradient=gradientForDirection(outgoing);
    return gradient ? phasePdfForGradient(*gradient) : 0;
}

double FixedSpeedGradientDistribution::phasePdfForGradient(const Vector3 &gradient) const
{
    if (_delta) return 0;
    const double length=gradient.norm();
    if (!(length>0) || !std::isfinite(length)) throw std::runtime_error("invalid fixed-speed phase gradient");
    const Vector2 residual=_basis.transpose()*gradient-_mean;
    const Vector2 standardized=_lower.triangularView<Eigen::Lower>().solve(residual);
    // Plane -> normal -> reflection has Jacobian |g|^4/(4*v^2).
    // The axial speed already includes crossing flux; do not add it again.
    const double logPdf=_logNormalization-0.5*standardized.squaredNorm()+4*std::log(length)-
        std::log(4.)-2*std::log(-_derivative);
    const double result=std::exp(logPdf);
    if (!(result>=0) || !std::isfinite(result)) throw std::runtime_error("nonfinite fixed-speed phase PDF");
    return result;
}

} // namespace Tungsten::ConditionalGpis
