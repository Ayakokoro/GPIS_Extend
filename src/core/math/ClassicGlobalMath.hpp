#ifndef CLASSICGLOBALMATH_HPP_
#define CLASSICGLOBALMATH_HPP_

#include <Eigen/Dense>

namespace Tungsten {
class PathSampleGenerator;

// The global_pointwise / generalized_gaussian law from macrofacet. Directions
// are travel directions; the mean gradient is deliberately NOT normalized.
namespace ClassicGlobal {
double openUniform(PathSampleGenerator &sampler);
double density(double mean, double sigma);
double logPositiveMoment(int order, double mean, double stddev);
double negativePartMean(double mean, double stddev);

class GaussianNdf {
    Eigen::Matrix3d _covariance, _lower;
    double _logDet;
public:
    explicit GaussianNdf(const Eigen::Matrix3d &covariance);
    const Eigen::Matrix3d &covariance() const { return _covariance; }
    double projectedArea(const Eigen::Vector3d &mean, const Eigen::Vector3d &w) const;
    double logD(const Eigen::Vector3d &mean, const Eigen::Vector3d &n) const;
    double phasePdf(const Eigen::Vector3d &mean, const Eigen::Vector3d &wi,
                    const Eigen::Vector3d &wo) const;
    Eigen::Vector3d sampleVisibleNormal(const Eigen::Vector3d &mean,
                    const Eigen::Vector3d &w, PathSampleGenerator &sampler) const;
};
}
}
#endif
