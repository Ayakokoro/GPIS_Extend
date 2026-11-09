#ifndef PHASESAMPLE_HPP_
#define PHASESAMPLE_HPP_

#include "math/Vec.hpp"
#include "MediumPathContext.hpp"

namespace Tungsten {

struct PhaseSample
{
    std::shared_ptr<const MediumPathContext> pathContext;
    Vec3f w;
    Vec3f weight;
    // Conditioned grazing lobes can exceed the float density range while
    // their directions, weights and MIS ratios remain finite.
    double pdf;
};

}

#endif /* PHASESAMPLE_HPP_ */
