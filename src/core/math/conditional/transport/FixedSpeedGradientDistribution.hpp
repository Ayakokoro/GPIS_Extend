#pragma once

#include "math/conditional/core/Random.h"
#include "math/conditional/gpss/CovarianceKernel.h"
#include <optional>

namespace Tungsten::ConditionalGpis {

// Condition the transverse Gaussian on the observed axial derivative.
// A first downcrossing has inward speed. An occupied entry probe can also
// observe an outward derivative and reflect on the back of the surface.
class FixedSpeedGradientDistribution {
public:
    // Independent transverse components with the same conditional variance.
    FixedSpeedGradientDistribution(const Vector3 &direction, double derivative,
        const Vector3 &gradientMean, double transverseStddev);
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
