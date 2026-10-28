#include "ClassicGlobalNonstationary.hpp"
#include "ClassicGlobalMath.hpp"
#include "GPFunctions.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace Tungsten { namespace ClassicGlobal {
namespace {
Eigen::Vector3d eigen(const Vec3d &v) { return Eigen::Vector3d(v.x(),v.y(),v.z()); }
void validate(const ParameterBounds &b, const char *name)
{
    if (!(b.minimum > 0.0) || b.maximum < b.minimum || b.gradientNorm < 0.0 ||
        !std::isfinite(b.minimum) || !std::isfinite(b.maximum) || !std::isfinite(b.gradientNorm))
        throw std::runtime_error(std::string("classic_global: ")+name+
            " needs finite, strictly positive values and a finite gradient bound");
}
}
NonstationaryKernel::NonstationaryKernel(const ProceduralNonstationaryCovariance &kernel)
    : _kernel(kernel), _amplitudeBounds{1.0,1.0,0.0}
{
    const auto *base = dynamic_cast<const SquaredExponentialCovariance *>(kernel.stationaryCovariance().get());
    if (!base || !base->isIsotropic() || !(base->lengthScale() > 0.0) || !std::isfinite(base->lengthScale()))
        throw std::runtime_error("classic_global: proc_nonstationary cov must be an isotropic squared_exponential with positive lengthScale");
    if (kernel.isNonstationaryAnisotropicKernel())
        throw std::runtime_error("classic_global: proc_nonstationary supports cov/var/ls; spatial aniso is not supported");
    const Vec3d zero(0.0);
    const double variance = (*base)(Derivative::None,Derivative::None,zero,zero,zero,zero);
    if (!(variance > 0.0) || !std::isfinite(variance))
        throw std::runtime_error("classic_global: cov variance must be positive and finite");
    _baseSigma = std::sqrt(variance);
    // Read the stationary jet once, also preserving the base implementation's
    // float parameter arithmetic in the constant-field limit.
    const Vec3d axis(1.0,0.0,0.0);
    _baseGamma = (*base)(Derivative::First,Derivative::First,zero,zero,axis,axis)/variance;
    if (!(_baseGamma > 0.0) || !std::isfinite(_baseGamma))
        throw std::runtime_error("classic_global: cov gradient variance must be positive and finite");
    if (kernel.amplitudeField() && !kernel.amplitudeField()->parameterBounds(_amplitudeBounds))
        throw std::runtime_error("classic_global: var requires analytic gradients and global bounds (constant or axis noise ramp)");
    validate(_amplitudeBounds,"var");
    for (auto &b : _scaleBounds) b = {1.0,1.0,0.0};
    if (kernel.lengthScaleField() && !kernel.lengthScaleField()->parameterBounds(_scaleBounds))
        throw std::runtime_error("classic_global: ls requires analytic gradients and global bounds (constant or axis noise ramp)");
    double baseBound = 0.0, scaleGradientBound = 0.0;
    for (const auto &b : _scaleBounds) {
        validate(b,"ls");
        baseBound = std::max(baseBound,_baseGamma/(b.minimum*b.minimum));
        scaleGradientBound += 0.5*std::pow(b.gradientNorm/b.minimum,2.0);
    }
    _maximumGradientVariance = baseBound+scaleGradientBound;
    if (!std::isfinite(_maximumGradientVariance) || !(minimumSigma() > 0.0) || !std::isfinite(maximumSigma()))
        throw std::runtime_error("classic_global: nonstationary parameter bounds overflow");
}

PointStatistics NonstationaryKernel::statistics(const Vec3d &p, double mean,
                                               const Eigen::Vector3d &gradient) const
{
    const auto a = _kernel.amplitudeField() ? _kernel.amplitudeField()->parameterJet(p) : ParameterJet{1.0,Vec3d(0.0)};
    VectorParameterJet scales;
    if (_kernel.lengthScaleField()) scales = _kernel.lengthScaleField()->parameterJet(p);
    else for (auto &s : scales) s = {1.0,Vec3d(0.0)};
    if (!(a.value > 0.0) || !std::isfinite(a.value) || !eigen(a.gradient).allFinite())
        throw std::runtime_error("classic_global: invalid var value/gradient");
    PointStatistics result;
    result.sigma = _baseSigma*a.value;
    result.normalizedMean = mean/result.sigma;
    result.gradientMean = (gradient-mean*eigen(a.gradient)/a.value)/result.sigma;
    result.gradientCovariance.setZero();
    for (int k = 0; k < 3; ++k) {
        const auto &s = scales[k];
        if (!(s.value > 0.0) || !std::isfinite(s.value) || !eigen(s.gradient).allFinite())
            throw std::runtime_error("classic_global: invalid ls value/gradient");
        const Eigen::Vector3d q = eigen(s.gradient)/s.value;
        result.gradientCovariance(k,k) += _baseGamma/(s.value*s.value);
        result.gradientCovariance += 0.5*q*q.transpose();
    }
    if (!std::isfinite(result.normalizedMean) || !result.gradientMean.allFinite() ||
        !result.gradientCovariance.allFinite() || !(result.sigma > 0.0) || !std::isfinite(result.sigma))
        throw std::runtime_error("classic_global: non-finite local Gaussian statistics");
    return result;
}

double NonstationaryKernel::majorant(const Vec3d &center, double radius, double meanLower,
                                     double meanUpper, double meanGradientBound) const
{
    // Global parameter/gradient bounds suffice. Intersect the amplitude range
    // with a Lipschitz ball to avoid mixing the two ends of a long ramp on every
    // tiny segment. No sampled maxima or majorant textures are involved.
    double amin = _amplitudeBounds.minimum, amax = _amplitudeBounds.maximum;
    if (_kernel.amplitudeField()) {
        const double value = (*_kernel.amplitudeField())(center);
        const double delta = _amplitudeBounds.gradientNorm*radius + 1e-12*amax;
        amin = std::max(amin,value-delta);
        amax = std::min(amax,value+delta);
    }
    const double sigmaMin = _baseSigma*amin, sigmaMax = _baseSigma*amax;
    const double uLower = meanLower/(meanLower >= 0.0 ? sigmaMax : sigmaMin);
    const double maxMean = std::max(std::abs(meanLower),std::abs(meanUpper));
    const double gradientBound = (meanGradientBound+maxMean*_amplitudeBounds.gradientNorm/amin)/sigmaMin;
    return density(uLower,1.0)*negativePartMean(-gradientBound,std::sqrt(_maximumGradientVariance))*(1.0+1e-5);
}
}}
