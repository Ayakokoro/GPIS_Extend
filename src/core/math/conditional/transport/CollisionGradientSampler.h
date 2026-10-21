// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Random.h"
#include "math/conditional/mathutility/SmallGaussian.h"

namespace Tungsten::ConditionalGpis {

class FlightKernel;

Vector3 sampleFluxWeightedGradient(const Gaussian<3>& gradient, const Vector3& w, Random& rng,
                                   const NumericPolicy& policy = defaultNumericPolicy());
Vector3 sampleCollisionGradient(const FlightKernel& kernel, double age, Random& rng,
                                const NumericPolicy& policy = defaultNumericPolicy());

} // namespace Tungsten::ConditionalGpis

