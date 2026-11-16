#pragma once

#include <Eigen/Core>
#include <functional>
#include <vector>
#include <cstddef>

namespace Tungsten::AdaptiveGpis {

// A value, or ell times a unit directional derivative, of a unit-variance SE GP.
struct Observation {
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::Zero();
    bool derivative = false;
};

// Pure-double online conditioning: incremental Cholesky with an eigen-based
// fallback for failed updates (optional). Tiny conditional variances/eigenvalues are treated
// as numerically redundant (no jitter). Retained observations are not evicted.
class OnlineSampler {
public:
    struct Prediction { double mean, variance; Eigen::VectorXd beta; };
    struct JointPrediction { Eigen::VectorXd mean; Eigen::MatrixXd covariance; };
    OnlineSampler(double lengthScale, double rankTolerance = 1e-10, int maxRank = 512,
                  bool numericalFallback = true);
    double covariance(const Observation &a, const Observation &b) const;
    Prediction predict(const Observation &observation) const;
    JointPrediction predictJoint(const std::vector<Observation> &observations) const;
    double sample(const Observation &observation, double normal);
    void condition(const Observation &observation, double value);
    std::size_t rank() const { return _basis.size(); }
    std::size_t redundant() const { return _redundant; }
    std::size_t queries() const { return _queries; }
    std::size_t rankDecompositions() const { return _rankDecompositions; }
    std::size_t choleskyRebuilds() const { return _choleskyRebuilds; }
    std::size_t varianceClamps() const { return _varianceClamps; }
    std::size_t jointCovarianceClamps() const { return _jointCovarianceClamps; }
    double largestVarianceCorrection() const { return _largestVarianceCorrection; }
private:
    void append(const Observation &, const Prediction &, double value);
    void retryCholesky() const;
    Prediction predictRankRevealing(const Observation &) const;
    double _length, _rankTolerance;
    int _maxRank;
    bool _numericalFallback;
    std::vector<Observation> _basis;
    mutable Eigen::MatrixXd _lower;
    mutable Eigen::VectorXd _innovations;
    std::vector<double> _values;
    // Pure-double rank-revealing solve for nearly dependent observations.
    // The full observation history is retained; no jitter or window eviction.
    mutable bool _rankRevealing = false;
    mutable Eigen::MatrixXd _whitener;
    mutable Eigen::VectorXd _whitenedValues;
    mutable std::size_t _rankDecompositions = 0;
    mutable std::size_t _choleskyRebuilds = 0, _varianceClamps = 0, _jointCovarianceClamps = 0;
    mutable double _largestVarianceCorrection = 0;
    std::size_t _redundant = 0, _queries = 0;
};

// Conditional transverse gradient after drawing the longitudinal derivative.
// Coordinates are an orthonormal [tangent, bitangent, incoming-ray] frame.
// The density includes the gradient-plane -> normal -> reflected-direction
// Jacobians. Degenerate transverse distributions retain ordinary specular sampling.
class GradientPlane {
public:
    GradientPlane() = default;
    GradientPlane(const Eigen::Matrix3d &frame, const Eigen::Vector3d &mean,
                  const Eigen::Matrix3d &covariance, double longitudinalNormal);
    Eigen::Vector3d sample(const Eigen::Vector2d &normal) const;
    Eigen::Vector3d gradientForNormal(const Eigen::Vector3d &normal) const;
    double reflectionPdf(const Eigen::Vector3d &normal) const;
    bool continuous() const { return _continuous; }
    double longitudinal() const { return _longitudinal; }
private:
    Eigen::Matrix3d _frame = Eigen::Matrix3d::Identity();
    Eigen::Vector2d _mean = Eigen::Vector2d::Zero();
    Eigen::Matrix2d _root = Eigen::Matrix2d::Zero(), _inverseRoot = Eigen::Matrix2d::Zero();
    double _longitudinal = 0, _logNormalizer = 0;
    bool _continuous = false;
};

struct Settings {
    double probabilityTolerance = 0.001;
    double minimumStep = 0.005;
    double rootTolerance = 1e-5;
    double rankTolerance = 1e-10;
    int maxSamples = 512;
    int maxSteps = 100000;
    bool adaptive = true;
    // False keeps incremental Cholesky and reports invalid covariance instead
    // of rebuilding the complete history with an eigensolver. No SVD is used.
    bool numericalFallback = true;
    void validate() const;
};

struct Bounds {
    double meanLipschitz, sigma, sqrtDerivativeBound;
    double c = 0, q = 0;
    // Split eta between the two inequalities (supplement, Section 1.5).
    Bounds(double lipschitz, double standardDeviation, double sqrtM, double eta);
    double step(double mean, double value, double minimumStep) const;
};

struct Intersection {
    bool hit = false;
    double t = 0, value = 0;
    int steps = 0, refinements = 0, minimumSteps = 0;
};

// Every root-refinement evaluation uses the same online realization as marching.
Intersection intersect(double begin, double end, double initialValue,
    const std::function<double(double)> &evaluate,
    const std::function<double(double)> &mean,
    const Bounds &bounds, const Settings &settings);

} // namespace Tungsten::AdaptiveGpis
