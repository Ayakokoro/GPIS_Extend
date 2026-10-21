#pragma once
#include <Eigen/Core>
#include <limits>
#include <optional>

namespace Tungsten {

struct SdfJet {
    double value = 0;
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
};

struct SdfBounds {
    double minimumValue = 0, maximumValue = 0, maximumGradientNorm = 0;
    bool certified = false;
};

struct SdfRayQueryOptions {
    bool computeBeginDerivative = true;
    // Preserve the caller's midpoint through coordinate transforms instead
    // of reconstructing it from an already transformed origin and direction.
    std::optional<Eigen::Vector3d> midpoint;
};

struct SdfRayBounds {
    double minimumValue = 0, maximumValue = 0;
    double minimumDerivative = 0, maximumDerivative = 0;
    // Infinite when the interval can cross a nonsmooth branch.
    double maximumSecondDerivative = std::numeric_limits<double>::infinity();
    // Only populated when computeBeginDerivative is requested.
    double beginDerivative = 0;
    bool certified = false;
    // Optional value at origin+(0.5*begin+0.5*end)*direction (or the supplied
    // midpoint). Error includes differences from the provider's point query.
    // NaN leaves the consumer's ordinary point-query fallback available.
    double midpointValue = std::numeric_limits<double>::quiet_NaN();
    double midpointError = 0;
};

// Optional geometric capabilities. No dependency on a medium or GP sampler.
// Providers must evaluate the same field in every query. Finite differences
// alone do not certify interval bounds or stable differences near a GP birth.
class SdfQueries {
public:
    virtual ~SdfQueries() = default;
    virtual SdfJet evaluateJet(const Eigen::Vector3d &p) const = 0;
    // f(p+d)-f(p), evaluated without cancellation for tiny displacements.
    virtual double valueDifference(const Eigen::Vector3d &p, const Eigen::Vector3d &d) const = 0;
    virtual SdfBounds bounds(const Eigen::Vector3d &minimum, const Eigen::Vector3d &maximum) const = 0;
    // Bounds on f(origin+t*direction) and its first two t derivatives.
    // Direction need not be normalized; intervals include both endpoints.
    virtual SdfRayBounds rayBounds(const Eigen::Vector3d &origin, const Eigen::Vector3d &direction,
                                  double begin, double end,
                                  const SdfRayQueryOptions &options = {}) const = 0;
};
}
