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
    float pdf;
};

}

#endif /* PHASESAMPLE_HPP_ */
