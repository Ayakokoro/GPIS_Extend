#ifndef SPARSECONVOLUITONNOISEMEDIUM_HPP_
#define SPARSECONVOLUITONNOISEMEDIUM_HPP_

#include "GaussianProcessMedium.hpp"
#include "math/SparseConvolutionNoise.hpp"
#include "GpRayBatchBackend.hpp"

namespace Tungsten {

    class GaussianProcess;
    class VdbGrid;

    struct Se1DGpBatchConfig {
        std::shared_ptr<VdbGrid> meanGrid;
        float meanOffset = 0.0f;
        float meanScale = 1.0f;
        float lengthScale = 1.0f;
        float amplitude = 1.0f;
        float kernelRadius = 3.0f;
        float impulseDensity = 3.0f;
        float maxStepSize = 0.01f;
        unsigned int minSteps = 8;
    };

    struct GPContextSparseConvNoise : public GPContext {
        SparseConvolutionNoiseRealization noise;
        virtual void reset() override {
            // Don't reset the realization
        }
    };

    struct PreparedGpisSegment {
        GpRaySegmentRequest request;
        std::shared_ptr<GPContextSparseConvNoise> context;
        UniformSampler uniformSampler;
    };

    class SparseConvolutionNoiseMedium : public GaussianProcessMedium
    {
        float _rayMarchStepSize;
        uint _rayMarchMinStep;
        uint _globalSeed;
        float _impulseDensity;
        bool _useSingleRealization;
        bool _isotropicSpace3DSampling; // Switch between sampling 3D noise in isotropic vs world space
        bool _1DSampling; // 1D noise is sampled in isotropic space
        SparseConv1DSamplingScheme _1DsamplingScheme;
        bool _correlationXY; // Consider the correlation along XY axes for the gradient, when conditioning on 3D normal

        // Tricky way to assign different phase functions
        bool _surfVolPhaseSeparate;
        float _surfVolPhaseAmpThresh;

    public:

        SparseConvolutionNoiseMedium();
        SparseConvolutionNoiseMedium(std::shared_ptr<GaussianProcess> gp, std::vector<std::shared_ptr<PhaseFunction>> phases,
                                         float materialSigmaA, float materialSigmaS, float density, int numBasisFunctions, bool useSingleRealization, float rayMarchStepSize) :
                GaussianProcessMedium(gp, phases, materialSigmaA, materialSigmaS, density), _rayMarchStepSize(rayMarchStepSize)
        {}

        static SparseConv1DSamplingScheme stringToSamplingScheme1D(const std::string& name);
        static std::string samplingScheme1DToString(SparseConv1DSamplingScheme val);

        virtual void fromJson(JsonPtr value, const Scene &scene) override;
        virtual rapidjson::Value toJson(Allocator &allocator) const override;

        virtual bool sampleGradient(PathSampleGenerator& sampler, const Ray& ray, const Vec3d& isect_p, const float& isect_t, MediumState& state, Vec3d& grad) const override;

        virtual bool intersectGP(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool) const override;

        // Returns false for scene features outside the CUDA SE 1D implementation.
        bool se1DGpuBatchConfig(Se1DGpBatchConfig &config) const;
        void prepareGpisSegment(PathSampleGenerator &sampler, const Ray &ray,
            MediumState &state, PreparedGpisSegment &prepared) const;
        bool finishGpisSegment(const Ray &ray, MediumState &state,
            PreparedGpisSegment &prepared, const GpRaySegmentReply &reply, double &t) const;
    };

}

#endif /* SPARSECONVOLUITONNOISEMEDIUM_HPP_ */
