#ifndef MEDIUMSAMPLE_HPP_
#define MEDIUMSAMPLE_HPP_

#include "math/Vec.hpp"
#include "MediumPathContext.hpp"
#include <Eigen/Dense>

namespace Tungsten {

class PhaseFunction;
struct GPContext;
enum class SparseConv1DSamplingScheme;

struct RayInfo {
    Vec4u pixelSampleSegment;
    uint sceneSeed;
    float t;
};


struct MediumSample
{
    std::shared_ptr<const MediumEventContext> eventContext;
    PhaseFunction *phase;
    Vec3f p;
    float continuedT;
    Vec3f continuedWeight;
    float t;
    Vec3f weight;
    Vec3f emission;
    float pdf;
    bool exited;
    Vec3d aniso;
    int gpId;
    SparseConv1DSamplingScheme sparseConv1DSamplingScheme;
    GPContext* ctxt;
    RayInfo rayInfo;
};

}

#endif /* MEDIUMSAMPLE_HPP_ */
