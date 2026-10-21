// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Random.h"
#include "math/conditional/mathutility/NumericPolicy.h"

namespace Tungsten::ConditionalGpis {

PositiveResult positiveRawMoment(int order, double mean, double stddev,
                                 const NumericPolicy& policy = defaultNumericPolicy());
PositiveResult negativePartMean(double mean, double stddev,
                                const NumericPolicy& policy = defaultNumericPolicy());
double negativeFluxNormalLogPdf(double k, double mean, double stddev,
                                const NumericPolicy& policy = defaultNumericPolicy());
PositiveResult negativeFluxNormalCdf(double k, double mean, double stddev,
                                     const NumericPolicy& policy = defaultNumericPolicy());
double sampleNegativeFluxNormal(double mean, double stddev, Random& rng,
                                const NumericPolicy& policy = defaultNumericPolicy());

} // namespace Tungsten::ConditionalGpis

