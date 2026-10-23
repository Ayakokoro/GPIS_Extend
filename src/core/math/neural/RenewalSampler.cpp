#include "RenewalSampler.hpp"
#include <Eigen/Cholesky>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Tungsten::NeuralGpis {
using ConditionalGpis::Vector2;
using ConditionalGpis::Vector3;
using ConditionalGpis::Point3;
using ConditionalGpis::Matrix2;
using ConditionalGpis::Matrix3;
using ConditionalGpis::Ray;
using ConditionalGpis::Random;
using ConditionalGpis::GPSSField;
using ConditionalGpis::CovarianceKernel;
using ConditionalGpis::CovarianceKernelType;
using ConditionalGpis::normalizedOrThrow;
using ConditionalGpis::orthonormalComplement;

Vector3 sampleRenewalGradient(const CovarianceKernel& kernel, const Vector3& direction,
    double x, double derivative, const Vector3& meanGradient,
    const Vector3& birthMeanGradient, const std::optional<Vector3>& birthGradient, Random& rng) {
    if (kernel.type() != CovarianceKernelType::SquaredExponential || !(x >= 0) || !std::isfinite(x) ||
        !(derivative < 0) || !std::isfinite(derivative) || !meanGradient.allFinite() ||
        !birthMeanGradient.allFinite() || (birthGradient && !birthGradient->allFinite()))
        throw std::invalid_argument("invalid Renewal gradient reconstruction inputs");
    const Vector3 w = normalizedOrThrow(direction);
    Vector3 e1, e2;
    orthonormalComplement(w, e1, e2);
    Eigen::Matrix<double,3,2> basis;
    basis << e1, e2;
    const Matrix3 C = kernel.gradientCovarianceAtZero();
    const Vector3 cw = C*w;
    const double variance = w.dot(cw);
    if (!(variance > 0)) throw std::invalid_argument("degenerate Renewal directional gradient covariance");
    const Vector3 regression = cw/variance;
    const Matrix3 residualCovariance = C-cw*regression.transpose();
    const Matrix2 transverse = basis.transpose()*residualCovariance*basis;
    const Eigen::LLT<Matrix2> cholesky(transverse);
    if (cholesky.info() != Eigen::Success)
        throw std::invalid_argument("Renewal requires a positive-definite kernel metric");
    Vector2 residual = Vector2::Zero();
    double varianceFactor = 1;
    if (birthGradient) {
        const Vector3 previous = *birthGradient-birthMeanGradient;
        // Projecting out the longitudinal component removes the radial rank-one
        // term from the two-point gradient covariance. The remaining transverse
        // SE correlation is exp(-x*x/2).
        const double decay = 0.5*x*x;
        residual = std::exp(-decay)*(basis.transpose()*(previous-regression*w.dot(previous)));
        varianceFactor = -std::expm1(-2*decay);
    }
    // Explicit temporaries fix random draw order across compilers.
    const double z1 = rng.standardNormal(), z2 = rng.standardNormal();
    const Vector2 noise = cholesky.matrixL()*Vector2(z1,z2);
    residual += std::sqrt(varianceFactor)*noise;
    const Vector3 longitudinal = meanGradient+regression*(derivative-w.dot(meanGradient));
    // Construct in the ray frame so transverse roundoff cannot flip V's sign.
    return derivative*w + basis*(basis.transpose()*longitudinal+residual);
}

RenewalSampler::RenewalSampler(const GPSSField& field, const CausalRenewalModel& model,
                               double maximumStep)
    : field_(field), model_(model), maximumStep_(maximumStep) {
    field_.validate();
    if (field_.kernel.type() != CovarianceKernelType::SquaredExponential)
        throw std::invalid_argument("neural_causal_renewal requires the squared-exponential kernel");
    if (!(maximumStep_ > 0) || !std::isfinite(maximumStep_))
        throw std::invalid_argument("invalid Renewal profile maximum step");
    const Eigen::LLT<Matrix3> cholesky(field_.kernel.metric());
    if (cholesky.info() != Eigen::Success)
        throw std::invalid_argument("neural_causal_renewal requires a positive-definite kernel metric");
}

std::optional<RenewalFlight> RenewalSampler::beginFlight(const Ray& input,
    std::optional<Vector3> birthGradient) const {
    Ray ray{input.origin, normalizedOrThrow(input.direction)};
    if (!field_.activeDomain.contains(ray.origin))
        throw std::invalid_argument("Renewal flight origin must be inside the active domain");
    const auto interval = field_.activeDomain.intersect(ray);
    if (!interval.hit || !(interval.exit > 0)) return std::nullopt;
    const double ell = 1/std::sqrt(ray.direction.dot(field_.kernel.metric()*ray.direction));
    const double sigma = field_.kernel.sigma();
    RayStartCondition start;
    if (birthGradient) {
        start = {RayStartMode::SurfaceOutward, ell/sigma*ray.direction.dot(*birthGradient)};
        if (!birthGradient->allFinite() || !(start.outwardDerivative > 0))
            throw std::invalid_argument("Renewal surface flight must point outward with a finite full gradient");
    }
    auto mean = RayMeanProfile::fromField(*field_.mean,ray.origin,ray.direction,
        sigma,ell,interval.exit,maximumStep_);
    const auto& first = mean.segments().front();
    const Point3 interior = ray.origin+(0.5*first.end*ell)*ray.direction;
    const Vector3 birthMean = field_.mean->evaluateInCell(ray.origin,interior).gradient;
    return RenewalFlight{ray, std::move(birthGradient), std::move(mean), start, ell, sigma, birthMean};
}

RenewalSurfaceSample RenewalSampler::sample(const RenewalFlight& flight, Random& rng,
                                           double tolerance) const {
    if (!(tolerance > 0) || !std::isfinite(tolerance))
        throw std::invalid_argument("invalid Renewal sample tolerance");
    const std::array<double,3> uniforms{rng.openUniform01(),rng.openUniform01(),rng.openUniform01()};
    const auto s = model_.sample(flight.mean,flight.start,uniforms,tolerance/flight.ell);
    const auto& segments = flight.mean.segments();
    if (s.hit) {
        const auto it = std::lower_bound(segments.begin(),segments.end(),s.q,
            [](const auto& segment, double q) { return segment.end < q; });
        const auto index = std::min(static_cast<std::size_t>(it-segments.begin()),segments.size()-1);
        return reconstruct(flight,index,s.q,s.speed,rng);
    }
    RenewalSurfaceSample miss;
    miss.distance = flight.maximumDistance(); miss.segment = segments.size();
    miss.position = flight.ray.origin+miss.distance*flight.ray.direction;
    return miss;
}

RenewalSurfaceSample RenewalSampler::reconstruct(const RenewalFlight& flight, std::size_t index,
    double x, double speed, Random& rng) const {
    const auto& segment = flight.mean.segments().at(index);
    RenewalSurfaceSample result;
    result.hit = true; result.segment = index; result.distance = x*flight.ell;
    result.position = flight.ray.origin+result.distance*flight.ray.direction;
    result.position = result.position.cwiseMax(field_.activeDomain.minimum).cwiseMin(field_.activeDomain.maximum);
    result.speed = speed;
    const Point3 interior = flight.ray.origin+(0.5*(segment.begin+segment.end)*flight.ell)*flight.ray.direction;
    const Vector3 meanGradient = field_.mean->evaluateInCell(result.position,interior).gradient;
    result.gradient = sampleRenewalGradient(field_.kernel,flight.ray.direction,x,
        -flight.sigma/flight.ell*result.speed,meanGradient,flight.birthMeanGradient,flight.birthGradient,rng);
    result.normal = normalizedOrThrow(result.gradient);
    if (!(flight.ray.direction.dot(result.gradient) < 0))
        throw std::runtime_error("Renewal collision lost its inward directional derivative");
    return result;
}

double RenewalSampler::transmittance(const RenewalFlight& flight, double distance) const {
    if (!(distance >= 0) || !std::isfinite(distance))
        throw std::invalid_argument("invalid Renewal visibility distance");
    const double q = std::min(distance/flight.ell,flight.mean.maximumX());
    return q == 0 ? 1 : std::exp(-model_.query(flight.mean,flight.start,{q})[0].cumulativeHazard);
}

} // namespace Tungsten::NeuralGpis
