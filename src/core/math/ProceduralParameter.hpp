#ifndef PROCEDURALPARAMETER_HPP_
#define PROCEDURALPARAMETER_HPP_

#include "Vec.hpp"
#include <array>

namespace Tungsten {
// World-space values and first derivatives of covariance parameter fields.
// Bounds cover the entire field, including its clamped exterior.
struct ParameterJet {
    double value = 0.0;
    Vec3d gradient = Vec3d(0.0);
};
struct ParameterBounds {
    double minimum = 0.0, maximum = 0.0, gradientNorm = 0.0;
};
using VectorParameterJet = std::array<ParameterJet, 3>;
using VectorParameterBounds = std::array<ParameterBounds, 3>;
}
#endif
