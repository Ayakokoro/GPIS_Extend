// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/gpss/GPSSField.h"
#include <stdexcept>

namespace Tungsten::ConditionalGpis {

PointPrior GPSSField::pointPrior(const Point3& x) const {
    if (!mean) throw std::invalid_argument("GPSS field has no mean field");
    const MeanJet jet = mean->evaluate(x);
    const double variance = kernel.sigma() * kernel.sigma();
    return {jet.value, variance, jet.gradient,
            kernel.gradientCovarianceAtZero(), Vector3::Zero()};
}

void GPSSField::validate() const {
    if (!mean || !activeDomain.valid()) throw std::invalid_argument("invalid GPSS field");
}

} // namespace Tungsten::ConditionalGpis
