// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Random.h"

namespace Tungsten::ConditionalGpis {

double normalPdf(double z);
double normalLogPdf(double z);
double normalCdf(double z);
double normalLogCdf(double z);
// p (Z > z)
double normalSurvival(double z);
double normalLogSurvival(double z);
double normalPdfOverCdf(double z);
// phi^{-1}(u) for 0<u<1
double normalQuantile(double u);
double normalQuantileFromLogCdf(double logU);
double sampleStandardNormal(Random& rng);

} // namespace Tungsten::ConditionalGpis

