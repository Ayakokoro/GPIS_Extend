#ifndef SPARSECONVOLUTIONNOISE_HPP_
#define SPARSECONVOLUTIONNOISE_HPP_

#include <math/GaussianProcess.hpp>

namespace Tungsten {
    struct SparseConvConditioningCoefficients3D
    {
        // 3D Renewal+ conditioning, pathwise update
        float value_scale = 0.;
        Vec3f gradient_scale = Vec3f(0.);
        Vec3f ray_origin  = Vec3f(0.);
    };

    struct SparseConvConditioningCoefficients1D
    {
        // 1D Renewal+ conditioning, pathwise update
        float value_scale = 0.;
        Vec3f gradient_scale = Vec3f(0.);
        Vec3f ray_origin  = Vec3f(0.);
        // Observed noise gradient, before the world-to-isotropic transform.
        Vec3f target_gradient_noise = Vec3f(0.);
        bool has_gradient_observation = false;
    };

    struct SparseConvolutionNoiseRealization {
        std::shared_ptr<GaussianProcess> _gp;
        GPCorrelationContext _ctxt;

        uint _globalSeed;
        bool _useSingleRealization;
        float _impulseDensity;
        bool _isotropicSpace3DSampling;
        bool _isotropicRaySpace3DSampling;
        bool _1DSampling;
        bool _activateConditioning;
        bool _correlationXY;
        SparseConv1DSamplingScheme _1DsamplingScheme;
        bool _multiResolutionGrid;

        float _base;
        SparseConvConditioningCoefficients3D coeff_3D;
        SparseConvConditioningCoefficients1D coeff_1D;

        // Tricky way to assign different phase functions, only used for the coffee mug scene
        bool _surfVolPhaseSeparate;
        float _surfVolPhaseAmpThresh;

        SparseConvolutionNoiseRealization() {}

        SparseConvolutionNoiseRealization(std::shared_ptr<GaussianProcess> gp, GPCorrelationContext ctxt,
                                          uint globalSeed, float impulseDensity, bool useSingleRealization,
                                          bool isotropicSpace3DSampling, bool oneDSampling,
                                          SparseConv1DSamplingScheme oneDSamplingScheme,
                                          bool correlationXY,
                                          bool surfVolPhaseSeparate, float surfVolPhaseAmpThresh);

        inline uint computeSeed(const Vec4u pixelSampleSegment, const uint sceneSeed) const;

        SparseConv1DSamplingScheme samplingScheme(const Vec3f& p) const;

        // Exact transverse law for stationary SE, in isotropic ray coordinates
        // (prior variance 1/2). Shared by gradient sampling and the NEE PDF.
        bool seTransverseGradientMoments(const Vec3f& p, const Vec3f& rayDir,
            bool conditioning, Vec2d& mean, double& variance) const;

        float evaluateValue(const Vec3f& p, const float t, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler, int& GPId);
        Vec3f evaluateGradient(const Vec3f& p, const float t, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler);

        float evaluateValueNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) { return evaluateNoise3D(p, rayDir, seed, sampler, conditioning).x(); }
        Vec3f evaluateGradientNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning) { return evaluateNoise3D(p, rayDir, seed, sampler, conditioning).yzw(); }
        Vec4f evaluateNoise3D(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning);

        float evaluateValueNoise1D(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, bool conditioning);
        Vec3f evaluateGradientNoise1D(const Vec3f& p, const float tTotal, const float tSegment, const Vec3f& rayDir, const RayInfo info, UniformSampler& sampler, bool conditioning);

        Vec4f evaluateNoise3DNormalized(const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        inline Vec4f evaluateNoise3DIsotropicNormalizedSelect(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        Vec4f evaluateNoise3DIsotropicNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        Vec4f evaluateNoise3DIsotropicRayNormalized(const Vec3f& p, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning);
        Vec4f evaluateNoise1DNormalized(const Vec3f& p, const float t, const Vec3f& rayDir, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale, bool conditioning, bool multiResLowLevel = false);

        Vec4f noise3D(const Vec3f& p_world, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const;
        Vec4f cell3D(const Vec3f& p_world, const Vec3u& ijk, const Vec3f& p, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius, float kernelSpatialScale) const;
        Vec2f noise1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const;
        Vec2f cell1D(const Vec3f& p_world, const Vec3f& ray_dir_world, const uint i, const float t, const uint seed, UniformSampler& sampler, float impulseDensity, float kernelRadius) const;

        Vec4f kernelScaleLevelRatio(const Vec3f& p) const;

        void conditioning(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler);
        void conditioning3D(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler);
        void conditioning1D(const Vec3f& p, const Vec3f& rayDir, const float targetVal, const Vec3f& targetGrad, const RayInfo info, UniformSampler& sampler);

        void neeShared(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info, Eigen::Matrix3f& mtx_pt, Vec3f& grad_constraint_3d, Vec3f& grad_3d_iso_ray, Vec3f& plane_normal_world, float& plane_stretch_jacobian, float& isect_dist);
        Vec3f neeGrad(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const RayInfo info);
        float neePDF(const Vec3f& rayDir, const Vec3f& normal, const Vec3f& p, const float t, const RayInfo info);
    };
}

#endif /* SPARSECONVOLUTIONNOISE_HPP_ */
