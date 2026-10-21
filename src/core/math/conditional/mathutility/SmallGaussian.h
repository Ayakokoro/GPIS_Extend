// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Random.h"
#include "math/conditional/mathutility/NumericPolicy.h"
#include <Eigen/Eigenvalues>
#include <cmath>
#include <limits>

namespace Tungsten::ConditionalGpis {

template<int N>
struct Gaussian {
    Eigen::Matrix<double, N, 1> mean = Eigen::Matrix<double, N, 1>::Zero();
    Eigen::Matrix<double, N, N> covariance = Eigen::Matrix<double, N, N>::Zero();
};

struct DynamicGaussian {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

struct FactoredGaussianPSD {
    Eigen::VectorXd mean;
    Eigen::MatrixXd factor;
};

inline DynamicGaussian conditionGaussianDynamic(const DynamicGaussian& target,
                                                const DynamicGaussian& observation,
                                                const Eigen::MatrixXd& targetObservationCovariance,
                                                const Eigen::VectorXd& observed,
                                                const NumericPolicy& policy = defaultNumericPolicy()) {
    const int a = static_cast<int>(target.mean.size());
    const int b = static_cast<int>(observation.mean.size());
    if (target.covariance.rows() != a || target.covariance.cols() != a ||
        observation.covariance.rows() != b || observation.covariance.cols() != b ||
        targetObservationCovariance.rows() != a || targetObservationCovariance.cols() != b ||
        observed.size() != b) {
        throw NumericError(NumericStatus::InvalidInput, "Gaussian conditioning dimension mismatch");
    }

    Eigen::VectorXd scales(b);
    for (int i = 0; i < b; ++i) {
        const double diagonal = observation.covariance(i, i);
        if (diagonal < -covarianceTolerance(observation.covariance.norm(), policy)) {
            throw NumericError(NumericStatus::InvalidCovariance, "negative observation variance");
        }
        scales[i] = std::sqrt(std::max(0.0, diagonal));
        if (scales[i] == 0.0) scales[i] = 1.0;
    }
    const Eigen::MatrixXd invScale = scales.cwiseInverse().asDiagonal();
    Eigen::MatrixXd standardized = invScale * observation.covariance * invScale;
    standardized = 0.5 * (standardized + standardized.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(standardized);
    if (eig.info() != Eigen::Success) {
        throw NumericError(NumericStatus::InvalidCovariance, "observation eigensolver failed");
    }
    const double tolerance = covarianceTolerance(std::max(1.0, standardized.norm()), policy);
    if (eig.eigenvalues().minCoeff() < -tolerance) {
        throw NumericError(NumericStatus::InvalidCovariance, "observation covariance is not PSD");
    }
    const Eigen::VectorXd residualStd = invScale * (observed - observation.mean);
    const Eigen::VectorXd projected = eig.eigenvectors().transpose() * residualStd;
    Eigen::VectorXd inverseValues = Eigen::VectorXd::Zero(b);
    for (int i = 0; i < b; ++i) {
        if (eig.eigenvalues()[i] > tolerance) inverseValues[i] = 1.0 / eig.eigenvalues()[i];
        else if (std::abs(projected[i]) > 32.0 * std::sqrt(tolerance)) {
            throw NumericError(NumericStatus::InvalidInput,
                               "observation is outside deterministic Gaussian support");
        }
    }
    const Eigen::MatrixXd pseudoInverse = invScale * eig.eigenvectors() *
                                          inverseValues.asDiagonal() *
                                          eig.eigenvectors().transpose() * invScale;
    DynamicGaussian result;
    result.mean = target.mean + targetObservationCovariance * pseudoInverse *
                                (observed - observation.mean);
    result.covariance = target.covariance - targetObservationCovariance * pseudoInverse *
                                            targetObservationCovariance.transpose();
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> resultEig(result.covariance);
    if (resultEig.info() != Eigen::Success) {
        throw NumericError(NumericStatus::InvalidCovariance, "conditional eigensolver failed");
    }
    // Forming a Schur complement amplifies covariance roundoff by the norm of
    // the standardized observation pseudo-inverse. This matters for nearby GP
    // observations, whose difference mode has a very small variance.
    const double conditioningAmplification = std::min(
        1e8, std::max(1.0, inverseValues.maxCoeff()));
    const double resultTolerance = conditioningAmplification * covarianceTolerance(
        std::max({1.0, target.covariance.norm(), result.covariance.norm()}), policy);
    if (resultEig.eigenvalues().minCoeff() < -resultTolerance) {
        throw NumericError(NumericStatus::InvalidCovariance, "conditional covariance is not PSD");
    }
    Eigen::VectorXd clipped = resultEig.eigenvalues().cwiseMax(0.0);
    result.covariance = resultEig.eigenvectors() * clipped.asDiagonal() *
                        resultEig.eigenvectors().transpose();
    result.covariance = 0.5 * (result.covariance + result.covariance.transpose());
    return result;
}

template<int A, int B, typename Derived>
Gaussian<A> conditionGaussian(const Gaussian<A>& target, const Gaussian<B>& observation,
                              const Eigen::Matrix<double, A, B>& targetObservationCovariance,
                              const Eigen::MatrixBase<Derived>& observedExpression,
                              const NumericPolicy& policy = defaultNumericPolicy()) {
    const Eigen::Matrix<double, B, 1> observed = observedExpression;
    DynamicGaussian dt{target.mean, target.covariance};
    DynamicGaussian db{observation.mean, observation.covariance};
    DynamicGaussian dynamic = conditionGaussianDynamic(dt, db, targetObservationCovariance,
                                                        observed, policy);
    Gaussian<A> result;
    result.mean = dynamic.mean;
    result.covariance = dynamic.covariance;
    return result;
}

template<int M, int N>
Gaussian<M> linearMap(const Gaussian<N>& input, const Eigen::Matrix<double, M, N>& matrix,
                      const Eigen::Matrix<double, M, 1>& offset) {
    return {matrix * input.mean + offset, matrix * input.covariance * matrix.transpose()};
}

inline FactoredGaussianPSD factorGaussianPSD(
    const DynamicGaussian& gaussian,
    const NumericPolicy& policy = defaultNumericPolicy()) {
    Eigen::MatrixXd covariance = 0.5 * (gaussian.covariance + gaussian.covariance.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(covariance);
    if (eig.info() != Eigen::Success) {
        throw NumericError(NumericStatus::InvalidCovariance, "PSD Gaussian eigensolver failed");
    }
    const double tolerance = covarianceTolerance(std::max(1.0, covariance.norm()), policy);
    if (eig.eigenvalues().minCoeff() < -tolerance) {
        throw NumericError(NumericStatus::InvalidCovariance, "Gaussian sample covariance is not PSD");
    }
    const Eigen::VectorXd roots = eig.eigenvalues().cwiseMax(0.0).cwiseSqrt();
    return {gaussian.mean, eig.eigenvectors() * roots.asDiagonal()};
}

inline Eigen::VectorXd sampleGaussianPSD(const FactoredGaussianPSD& gaussian, Random& rng) {
    Eigen::VectorXd z(gaussian.factor.cols());
    for (int i = 0; i < z.size(); ++i) z[i] = rng.standardNormal();
    return gaussian.mean + gaussian.factor * z;
}

inline Eigen::VectorXd sampleGaussianPSDDynamic(const DynamicGaussian& gaussian, Random& rng,
                                                const NumericPolicy& policy = defaultNumericPolicy()) {
    return sampleGaussianPSD(factorGaussianPSD(gaussian, policy), rng);
}

template<int N>
Eigen::Matrix<double, N, 1> sampleGaussianPSD(
    const Gaussian<N>& gaussian, Random& rng,
    const NumericPolicy& policy = defaultNumericPolicy()) {
    DynamicGaussian dynamic{gaussian.mean, gaussian.covariance};
    return sampleGaussianPSDDynamic(dynamic, rng, policy);
}

template<int N>
double logPdfGaussianSPD(const Gaussian<N>& gaussian,
                         const Eigen::Matrix<double, N, 1>& value) {
    Eigen::LLT<Eigen::Matrix<double, N, N>> llt(gaussian.covariance);
    if (llt.info() != Eigen::Success) {
        throw NumericError(NumericStatus::InvalidCovariance, "Gaussian PDF requires SPD covariance");
    }
    const auto delta = value - gaussian.mean;
    const auto solved = llt.matrixL().solve(delta);
    double logDeterminant = 0.0;
    for (int i = 0; i < N; ++i) logDeterminant += 2.0 * std::log(llt.matrixL()(i, i));
    return -0.5 * (N * std::log(2.0 * kPi) + logDeterminant + solved.squaredNorm());
}

template<int M, int N>
Gaussian<M> marginal(const Gaussian<N>& gaussian, const std::array<int, M>& indices) {
    Gaussian<M> result;
    for (int i = 0; i < M; ++i) {
        result.mean[i] = gaussian.mean[indices[static_cast<std::size_t>(i)]];
        for (int j = 0; j < M; ++j) {
            result.covariance(i, j) = gaussian.covariance(
                indices[static_cast<std::size_t>(i)], indices[static_cast<std::size_t>(j)]);
        }
    }
    return result;
}

} // namespace Tungsten::ConditionalGpis
