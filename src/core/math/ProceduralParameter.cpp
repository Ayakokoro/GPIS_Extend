#include "GPFunctions.hpp"
#include <algorithm>
#include <cmath>
#include <limits>

namespace Tungsten {
namespace {
bool rampBounds(double a, double b, double scale, double offset, ParameterBounds &bounds)
{
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(scale) ||
        !std::isfinite(offset) || a <= -1.0 || b <= -1.0) return false;
    // Match the existing log-square ramp, with outward roundoff allowance.
    const double slope = 0.5*(std::log((b+1.0)*(b+1.0))-std::log((a+1.0)*(a+1.0)))*scale;
    const double error = 32.0*std::numeric_limits<double>::epsilon()*std::max({1.0, std::abs(a), std::abs(b)});
    bounds = {std::min(a,b)-error, std::max(a,b)+error,
              (std::max(a,b)+1.0+error)*std::abs(slope)*(1.0+1e-12)};
    return std::isfinite(bounds.gradientNorm);
}
ParameterJet rampJet(double value, double coordinate, int axis,
                     double a, double b, double scale, double offset)
{
    ParameterJet jet{value, Vec3d(0.0)};
    const double t = coordinate*scale+offset;
    // The clamped ramp has a crease at each endpoint. Use the exterior
    // derivative on the crease; away from these planes the derivative is exact.
    if (t > 0.0 && t < 1.0)
        jet.gradient[axis] = (value+1.0)*0.5*
            (std::log((b+1.0)*(b+1.0))-std::log((a+1.0)*(a+1.0)))*scale;
    return jet;
}
}

bool ProceduralNoise::parameterBounds(ParameterBounds &b) const
{
    if (type != NoiseType::BottomTop && type != NoiseType::LeftRight && type != NoiseType::FrontBack)
        return false;
    return rampBounds(_min, _max, _scale, _offset, b);
}
ParameterJet ProceduralNoise::parameterJet(Vec3d p) const
{
    const int axis = type == NoiseType::LeftRight ? 0 : type == NoiseType::BottomTop ? 1 : 2;
    if (type != NoiseType::BottomTop && type != NoiseType::LeftRight && type != NoiseType::FrontBack)
        return ProceduralScalar::parameterJet(p);
    return rampJet((*this)(p), p[axis], axis, _min, _max, _scale, _offset);
}
bool ProceduralNoiseVec::parameterBounds(VectorParameterBounds &b) const
{
    if (type != NoiseType::BottomTop && type != NoiseType::LeftRight && type != NoiseType::FrontBack)
        return false;
    if (!rampBounds(_min, _max, _scale, _offset, b[0])) return false;
    b[1] = b[2] = b[0];
    return true;
}
VectorParameterJet ProceduralNoiseVec::parameterJet(Vec3d p) const
{
    const int axis = type == NoiseType::LeftRight ? 0 : type == NoiseType::BottomTop ? 1 : 2;
    if (type != NoiseType::BottomTop && type != NoiseType::LeftRight && type != NoiseType::FrontBack)
        return ProceduralVector::parameterJet(p);
    const auto j = rampJet((*this)(p)[0], p[axis], axis, _min, _max, _scale, _offset);
    return {j,j,j};
}
}
