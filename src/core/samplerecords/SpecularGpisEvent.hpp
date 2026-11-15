#pragma once
#include "MediumPathContext.hpp"
#include "math/Vec.hpp"

namespace Tungsten {
// Immutable conditional normal law at a GP hit. The longitudinal derivative
// is already sampled; light sampling integrates over the transverse gradient.
// This avoids treating every GP context as a sparse-convolution realization.
struct SpecularGpisEvent : MediumEventContext {
    virtual double reflectionPdf(const Vec3f &travel, const Vec3f &outgoing) const = 0;
    virtual Vec3d reflectionGradient(const Vec3f &travel, const Vec3f &outgoing) const = 0;
};
}
