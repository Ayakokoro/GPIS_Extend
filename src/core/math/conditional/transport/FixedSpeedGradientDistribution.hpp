#pragma once

#include "math/conditional/core/Random.h"
#include "math/conditional/gpss/CovarianceKernel.h"
#include <optional>

namespace Tungsten::ConditionalGpis {

// The inward speed has already been sampled with its crossing-flux weight.
// Collinear SE value observations leave this transverse Gaussian unchanged.
// Keeping the speed in the collision state gives continuation and NEE the same
// conditional directional measure, without approximating the axial law.
class FixedSpeedGradientDistribution {
public:
    FixedSpeedGradientDistribution(const CovarianceKernel &kernel,
        const Vector3 &direction, double normalizedDistance, double derivative,
        const Vector3 &meanGradient, const Vector3 &birthMeanGradient,
        const std::optional<Vector3> &birthGradient);
    const Vector3 &incomingDirection() const { return _direction; }
    bool isDelta() const { return _delta; }
    Vector3 sample(Random &rng) const;
    std::optional<Vector3> gradientForDirection(const Vector3 &outgoing) const;
    double phasePdf(const Vector3 &outgoing) const;
    double phasePdfForGradient(const Vector3 &gradient) const;
private:
    Vector3 _direction;
    Eigen::Matrix<double,3,2> _basis;
    Vector2 _mean;
    Matrix2 _lower;
    double _derivative, _logNormalization = 0;
    bool _delta = false;
};

} // namespace Tungsten::ConditionalGpis
