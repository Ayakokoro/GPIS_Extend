#include "ClassicGlobalMath.hpp"
#include "sampling/PathSampleGenerator.hpp"
#include <boost/math/quadrature/gauss_kronrod.hpp>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Tungsten {
namespace ClassicGlobal {
namespace {
constexpr double Pi = 3.14159265358979323846;
constexpr double LogSqrtTwoPi = 0.91893853320467274178;
double normalPdf(double x) { return std::exp(-0.5*x*x - LogSqrtTwoPi); }
double normalCdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }

// Invert the flux-weighted negative normal, using the same two regimes as
// macrofacet's sampleNegativeFluxNormal. Rejection stays efficient in the tail.
double negativeFlux(double mean, double stddev, PathSampleGenerator &sampler)
{
    if (mean >= 0.0) {
        const double a = mean/stddev;
        for (;;) {
            double y, logAccept;
            if (a >= 1.0) {
                y = -std::log(openUniform(sampler)) - std::log(openUniform(sampler));
                logAccept = -0.5*(y/a)*(y/a);
            } else {
                y = std::sqrt(-2.0*std::log(openUniform(sampler)));
                logAccept = -a*y;
            }
            if (std::log(openUniform(sampler)) <= logAccept)
                return -(a >= 1.0 ? stddev/a : stddev)*y;
        }
    }
    const double a = -mean/stddev;
    const auto cdfNumerator = [a](double z) {
        return a >= 1.0 ? normalCdf(z) + normalPdf(z)/a
                        : a*normalCdf(z) + normalPdf(z);
    };
    const double target = openUniform(sampler)*cdfNumerator(a);
    double lo = -8.0, hi = std::min(8.0, a);
    while (cdfNumerator(lo) > target) lo *= 2.0;
    while (cdfNumerator(hi) < target) hi = std::min(a, hi*2.0);
    for (int i = 0; i < 55; ++i) {
        const double mid = 0.5*(lo + hi);
        if (cdfNumerator(mid) < target) lo = mid; else hi = mid;
    }
    return -stddev*(a - 0.5*(lo + hi));
}
}

double openUniform(PathSampleGenerator &sampler)
{
    // Path samplers return floats in [0, 1). Reject endpoints instead of
    // clamping them into an atom in the exponential/normal distributions.
    double u;
    do { u = sampler.next1D(); } while (u <= 0.0 || u >= 1.0);
    return u;
}

double density(double mean, double sigma)
{
    const double z = mean/sigma;
    if (z < -8.0) {
        double tail = 0.0;
        for (int n = 64; n > 0; --n) tail = n/(-z + tail);
        return (-z + tail)/sigma;
    }
    return normalPdf(z)/(sigma*normalCdf(z));
}

double logPositiveMoment(int order, double mean, double stddev)
{
    if ((order != 1 && order != 3) || !(stddev > 0.0))
        throw std::runtime_error("classic_global: invalid Gaussian moment");
    const double a = mean/stddev;
    if (a > -5.0) {
        // Work in standardized coordinates to avoid powers of stddev.
        const double moment = order == 1 ? a*normalCdf(a) + normalPdf(a)
            : (a*a*a + 3.0*a)*normalCdf(a) + (a*a + 2.0)*normalPdf(a);
        return order*std::log(stddev) + std::log(moment);
    }
    // Scale the far-tail integral before quadrature: no subtraction of almost
    // equal terms, and no underflow until the final result is exponentiated.
    const double integral = boost::math::quadrature::gauss_kronrod<double, 31>::integrate(
        [a, order](double y) {
            return (order == 1 ? y : y*y*y)*std::exp(-y - 0.5*(y/a)*(y/a));
        }, 0.0, 80.0, 8, 1e-12);
    return order*std::log(stddev) - 0.5*a*a - LogSqrtTwoPi
        - (order + 1)*std::log(-a) + std::log(integral);
}

double negativePartMean(double mean, double stddev)
{
    return std::exp(logPositiveMoment(1, -mean, stddev));
}

GaussianNdf::GaussianNdf(const Eigen::Matrix3d &covariance)
    : _covariance(0.5*(covariance + covariance.transpose()))
{
    Eigen::LLT<Eigen::Matrix3d> chol(_covariance);
    if (!_covariance.allFinite() || chol.info() != Eigen::Success)
        throw std::runtime_error("classic_global: gradient covariance must be positive definite");
    _lower = chol.matrixL();
    _logDet = 2.0*_lower.diagonal().array().log().sum();
}

double GaussianNdf::projectedArea(const Eigen::Vector3d &mean, const Eigen::Vector3d &w) const
{
    return negativePartMean(w.dot(mean), std::sqrt(w.dot(_covariance*w)));
}

double GaussianNdf::logD(const Eigen::Vector3d &mean, const Eigen::Vector3d &n) const
{
    const Eigen::Vector3d u = _lower.triangularView<Eigen::Lower>().solve(mean);
    const Eigen::Vector3d v = _lower.triangularView<Eigen::Lower>().solve(n);
    const double a = v.squaredNorm(), m = u.dot(v)/a, s = 1.0/std::sqrt(a);
    return -std::log(2.0*Pi) - 0.5*_logDet - 0.5*std::log(a)
        - 0.5*(u - m*v).squaredNorm() + logPositiveMoment(3, m, s);
}

double GaussianNdf::phasePdf(const Eigen::Vector3d &mean, const Eigen::Vector3d &wi,
                             const Eigen::Vector3d &wo) const
{
    const Eigen::Vector3d delta = wo - wi;
    if (delta.squaredNorm() == 0.0) return 0.0;
    Eigen::Vector3d n = delta.normalized();
    if (wi.dot(n) >= 0.0) n = -n;
    const double logArea = logPositiveMoment(1, -wi.dot(mean),
                                             std::sqrt(wi.dot(_covariance*wi)));
    return std::exp(logD(mean, n) - std::log(4.0) - logArea);
}

Eigen::Vector3d GaussianNdf::sampleVisibleNormal(const Eigen::Vector3d &mean,
        const Eigen::Vector3d &w, PathSampleGenerator &sampler) const
{
    const Eigen::Vector3d cw = _covariance*w;
    const double variance = w.dot(cw);
    const double k = negativeFlux(w.dot(mean), std::sqrt(variance), sampler);
    const Eigen::Vector3d conditionalMean = mean + cw*((k - w.dot(mean))/variance);
    Eigen::Matrix<double, 3, 2> basis;
    basis.col(0) = w.unitOrthogonal();
    basis.col(1) = w.cross(basis.col(0));
    const Eigen::Matrix2d c = basis.transpose()*(_covariance - cw*cw.transpose()/variance)*basis;
    Eigen::LLT<Eigen::Matrix2d> chol(0.5*(c + c.transpose()));
    if (chol.info() != Eigen::Success)
        throw std::runtime_error("classic_global: invalid conditional gradient covariance");
    const double r = std::sqrt(-2.0*std::log(openUniform(sampler)));
    const double phi = 2.0*Pi*openUniform(sampler);
    const Eigen::Vector2d z(r*std::cos(phi), r*std::sin(phi));
    const Eigen::Vector2d transverse = basis.transpose()*conditionalMean + chol.matrixL()*z;
    const Eigen::Vector3d gradient = k*w + basis*transverse;
    if (!(w.dot(gradient) < 0.0) || !gradient.allFinite())
        throw std::runtime_error("classic_global: sampled normal left crossing support");
    return gradient.normalized();
}
}
}
