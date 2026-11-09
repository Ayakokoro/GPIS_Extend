#pragma once

#include "math/conditional/core/Random.h"
#include "math/conditional/mathutility/NumericPolicy.h"
#include <cmath>
#include <stdexcept>

namespace Tungsten::LocalEllipsoid {

// Gamma(shape=alpha + collisionOrigin, rate=alpha). All random draws come
// from the caller's path sampler. The collision law is density-weighted;
// its mean is (alpha+1)/alpha, and must not be normalized back to one.
inline double sampleGammaDensityMultiplier(ConditionalGpis::Random &rng,
                                           double alpha, bool collisionOrigin) {
    if (!(alpha>1) || !std::isfinite(alpha))
        throw std::invalid_argument("gamma_renewal requires finite alpha > 1");
    const double shape=alpha+(collisionOrigin ? 1.0 : 0.0);
    const double d=shape-1.0/3.0, c=1.0/(3.0*std::sqrt(d));
    for (;;) {
        const double x=rng.standardNormal(), offset=c*x;
        if (offset<=-1) continue;
        const double logV=3*std::log1p(offset);
        const double u=rng.openUniform01(), x2=x*x;
        if (u<1-0.0331*x2*x2 ||
            std::log(u)<0.5*x2+d*(logV-std::expm1(logV))) {
            const double scale=(d/alpha)*std::exp(logV);
            if (!(scale>0) || !std::isfinite(scale))
                throw ConditionalGpis::NumericError(ConditionalGpis::NumericStatus::NeedHigherPrecision,
                    "gamma density multiplier is not positive and finite");
            return scale;
        }
    }
}

} // namespace Tungsten::LocalEllipsoid
