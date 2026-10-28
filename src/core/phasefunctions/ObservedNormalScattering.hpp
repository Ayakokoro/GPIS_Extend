#pragma once
#include "math/TangentFrame.hpp"
#include "samplerecords/MediumSample.hpp"
#include "sampling/SampleWarp.hpp"
#include "sampling/PathSampleGenerator.hpp"
#include <stdexcept>

namespace Tungsten {
// The gradient is sampled once with the collision, before choosing between
// light sampling and continuation. Both MIS strategies condition on this same
// observation; their directional density is an ordinary cosine hemisphere.
inline Vec3f observedNormal(const MediumSample &s) {
    const double length=s.aniso.length();
    if(!(length>0) || !std::isfinite(length))
        throw std::runtime_error("diffuse micro-scattering requires a collision normal");
    return Vec3f(s.aniso/length).normalized();
}
inline float observedCosinePdf(const Vec3f &wo,const MediumSample &s) {
    return std::max(0.0f,observedNormal(s).dot(wo.normalized()))*INV_PI;
}
inline Vec3f sampleObservedCosine(PathSampleGenerator &sampler,const MediumSample &s) {
    return TangentFrame(observedNormal(s)).toGlobal(SampleWarp::cosineHemisphere(sampler.next2D())).normalized();
}
}
