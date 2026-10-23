#pragma once

#include "math/conditional/gpss/MeanField.h"
#include <vector>

namespace Tungsten::NeuralGpis {

// Cubic in the segment's local coordinate u in [0,1].
struct MeanPolynomial {
    double a = 0, b = 0, c = 0, d = 0;
    double value(double u) const { return ((a*u+b)*u+c)*u+d; }
    double derivative(double u) const { return (3*a*u+2*b)*u+c; }
};

// Normalized distance q=s/ell and mean b=m/sigma, as used during training.
struct RayMeanSegment {
    double begin = 0, end = 0;
    MeanPolynomial polynomial;
    double value(double q) const { return polynomial.value((q-begin)/(end-begin)); }
    double derivative(double q) const { return polynomial.derivative((q-begin)/(end-begin))/(end-begin); }
};

class RayMeanProfile {
public:
    explicit RayMeanProfile(std::vector<RayMeanSegment> segments);
    // Grid cell boundaries are preserved. Four samples recover the exact
    // ray cubic of a trilinear SDF, without ambiguous voxel-face gradients.
    static RayMeanProfile fromField(const ConditionalGpis::MeanField& mean,
        const Eigen::Vector3d& origin, const Eigen::Vector3d& direction,
        double sigma, double ell, double maximumDistance, double maximumStep = 0.25);
    const std::vector<RayMeanSegment>& segments() const { return segments_; }
    double maximumX() const { return segments_.back().end; }
private:
    std::vector<RayMeanSegment> segments_;
};

} // namespace Tungsten::NeuralGpis
