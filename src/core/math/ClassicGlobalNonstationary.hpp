#ifndef CLASSICGLOBALNONSTATIONARY_HPP_
#define CLASSICGLOBALNONSTATIONARY_HPP_

#include "ProceduralParameter.hpp"
#include <Eigen/Dense>

namespace Tungsten {
class ProceduralNonstationaryCovariance;
namespace ClassicGlobal {
// G | F=0 = sigma * Z. These statistics describe Z, not the unconditional
// gradient. Positive rescaling leaves the visible-normal distribution intact.
struct PointStatistics {
    double sigma, normalizedMean;
    Eigen::Vector3d gradientMean;
    Eigen::Matrix3d gradientCovariance;
};

class NonstationaryKernel {
    const ProceduralNonstationaryCovariance &_kernel;
    ParameterBounds _amplitudeBounds;
    VectorParameterBounds _scaleBounds;
    double _baseSigma, _baseGamma, _maximumGradientVariance;
public:
    explicit NonstationaryKernel(const ProceduralNonstationaryCovariance &kernel);
    double minimumSigma() const { return _baseSigma*_amplitudeBounds.minimum; }
    double maximumSigma() const { return _baseSigma*_amplitudeBounds.maximum; }
    PointStatistics statistics(const Vec3d &p, double mean, const Eigen::Vector3d &gradient) const;
    // Conservative constant on a segment contained in this world-space ball.
    double majorant(const Vec3d &center, double radius, double meanLower,
                    double meanUpper, double meanGradientBound) const;
};
}
}
#endif
