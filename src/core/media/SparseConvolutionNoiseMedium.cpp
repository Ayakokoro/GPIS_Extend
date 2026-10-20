#include "SparseConvolutionNoiseMedium.hpp"
#include "GpRayBatchBackend.hpp"
#include "WeightSpaceGaussianProcessMedium.hpp"

#include <cfloat>
#include <atomic>
#include <chrono>
#include <cstdlib>

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformPathSampler.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"
#include "grids/VdbGrid.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {
    namespace {
    struct GpisCpuProfile {
        bool enabled;
        std::atomic<std::uint64_t> conditioningNs{0}, gradientNs{0}, marchingNs{0};
        std::atomic<std::uint64_t> marchingCalls{0}, steps{0};
        GpisCpuProfile() {
            const char *flag = std::getenv("TUNGSTEN_GPIS_PROFILE");
            enabled = flag && std::string(flag) == "1";
        }
        ~GpisCpuProfile() {
            if (enabled)
                std::cout << "GPIS CPU profile (sum of worker times): conditioning_ms="
                          << conditioningNs.load()*1e-6 << ", gradient_ms=" << gradientNs.load()*1e-6
                          << ", marching_ms=" << marchingNs.load()*1e-6
                          << ", marching_calls=" << marchingCalls << ", steps=" << steps << '\n';
        }
    };
    GpisCpuProfile &cpuProfile() { static GpisCpuProfile profile; return profile; }
    struct GpisCpuTimer {
        GpisCpuProfile &profile;
        std::atomic<std::uint64_t> &total;
        std::chrono::steady_clock::time_point start;
        const int *steps = nullptr;
        GpisCpuTimer(std::atomic<std::uint64_t> &total, const int *steps = nullptr)
            : profile(cpuProfile()), total(total), steps(steps) {
            if (profile.enabled)
                start = std::chrono::steady_clock::now();
        }
        ~GpisCpuTimer() {
            if (profile.enabled) {
                total += std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - start).count();
                if (steps) {
                    ++profile.marchingCalls;
                    profile.steps += *steps;
                }
            }
        }
    };
    }
    SparseConvolutionNoiseMedium::SparseConvolutionNoiseMedium()
            : GaussianProcessMedium(
                    std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(),
                    std::make_shared<SquaredExponentialCovariance>()),
                    {}, 0.0f, 0.0f, 1.f),
              _rayMarchStepSize(0.01f),
              _rayMarchMinStep(8),
              _globalSeed(0u),
              _impulseDensity(3.0f),
              _useSingleRealization(false),
              _isotropicSpace3DSampling(false),
              _1DSampling(false),
              _1DsamplingScheme(SparseConv1DSamplingScheme::UNI),
              _correlationXY(false),
              _surfVolPhaseSeparate(false),
              _surfVolPhaseAmpThresh(0.0)
    {
    }

    SparseConv1DSamplingScheme SparseConvolutionNoiseMedium::stringToSamplingScheme1D(const std::string& name)
    {
        if (name == "uni" || name == "UNI")
            return SparseConv1DSamplingScheme::UNI;
        else if (name == "nee" || name == "NEE")
            return SparseConv1DSamplingScheme::NEE;
        else if (name == "mis" || name == "MIS")
            return SparseConv1DSamplingScheme::MIS;
        FAIL("Invalid sparse conv sampling scheme: '%s'", name);
    }

    std::string SparseConvolutionNoiseMedium::samplingScheme1DToString(SparseConv1DSamplingScheme val)
    {
        switch (val) {
            default:
            case SparseConv1DSamplingScheme::UNI:  return "uni";
            case SparseConv1DSamplingScheme::NEE:  return "nee";
            case SparseConv1DSamplingScheme::MIS:  return "mis";
        }
    }

    void SparseConvolutionNoiseMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        GaussianProcessMedium::fromJson(value, scene);
        value.getField("step_size", _rayMarchStepSize);
        value.getField("min_step", _rayMarchMinStep);
        value.getField("seed", _globalSeed);
        value.getField("impulse_density", _impulseDensity);
        value.getField("single_realization", _useSingleRealization);
        value.getField("isotropic_3D_sampling", _isotropicSpace3DSampling);
        value.getField("1D_sampling", _1DSampling);
        std::string scheme1DString = "uni";
        value.getField("1D_sampling_scheme", scheme1DString);
        _1DsamplingScheme = stringToSamplingScheme1D(scheme1DString);
        value.getField("1D_gradient_correlationXY", _correlationXY);
        value.getField("surf_vol_phase_separate", _surfVolPhaseSeparate);
        value.getField("surf_vol_phase_amp_thresh", _surfVolPhaseAmpThresh);
    }

    rapidjson::Value SparseConvolutionNoiseMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ GaussianProcessMedium::toJson(allocator), allocator,
                           "type", "sparse_conv_noise",
                           "step_size", _rayMarchStepSize,
                           "min_step", _rayMarchMinStep,
                           "seed", _globalSeed,
                           "impulse_density", _impulseDensity,
                           "single_realization", _useSingleRealization,
                           "isotropic_3D_sampling", _isotropicSpace3DSampling,
                           "1D_sampling", _1DSampling,
                           "1D_sampling_scheme", samplingScheme1DToString(_1DsamplingScheme),
                           "1D_gradient_correlationXY", _correlationXY,
                           "surf_vol_phase_separate", _surfVolPhaseSeparate,
                           "surf_vol_phase_amp_thresh", _surfVolPhaseAmpThresh,
        };
    }

    bool SparseConvolutionNoiseMedium::sampleGradient(PathSampleGenerator& sampler, const Ray& ray,
                                                      const Vec3d& isect_p, const float& isect_t, MediumState& state, Vec3d& grad) const {
        GpisCpuTimer timer(cpuProfile().gradientNs);
        auto gp = _gp->get_gaussian_process();
        UniformSampler uniformSampler;
        GPContextSparseConvNoise& ctxt = *(GPContextSparseConvNoise*)state.gpContext.get();
        grad = Vec3d(ctxt.noise.evaluateGradient(Vec3f(isect_p), isect_t, ray.dir(), state.info, uniformSampler));
        return true;
    }

    bool SparseConvolutionNoiseMedium::se1DGpuBatchConfig(Se1DGpBatchConfig &config) const {
#if OPENVDB_AVAILABLE
        if (_intersectMethod != GPIntersectMethod::GPDiscrete ||
            _ctxt != GPCorrelationContext::RenewalPlus || !_1DSampling ||
            _1DsamplingScheme != SparseConv1DSamplingScheme::UNI ||
            _correlationXY || _useSingleRealization || _surfVolPhaseSeparate ||
            _rayMarchMinStep == 0)
            return false;

        auto gp = std::dynamic_pointer_cast<GaussianProcess>(_gp);
        if (!gp || gp->_mean_additional || !gp->_globalCondPs.empty())
            return false;
        auto covariance = std::dynamic_pointer_cast<SquaredExponentialCovariance>(gp->_cov);
        auto mean = std::dynamic_pointer_cast<TabulatedMean>(gp->_mean);
        if (!covariance || !covariance->isIsotropic() || !mean || mean->isVolume())
            return false;
        auto grid = std::dynamic_pointer_cast<VdbGrid>(mean->grid());
        if (!grid)
            return false;

        config.meanGrid = std::move(grid);
        config.meanOffset = mean->offset();
        config.meanScale = mean->scale();
        config.lengthScale = covariance->lengthScale();
        config.amplitude = covariance->sparseConvNoiseAmplitude();
        config.kernelRadius = covariance->splattingKernelRadius(true, 1.0f);
        config.impulseDensity = _impulseDensity;
        config.maxStepSize = _rayMarchStepSize;
        config.minSteps = _rayMarchMinStep;
        return std::isfinite(config.lengthScale) && config.lengthScale > 0.0f &&
               std::isfinite(config.amplitude) && std::isfinite(config.kernelRadius) &&
               config.kernelRadius > 0.0f && std::isfinite(config.impulseDensity) &&
               config.impulseDensity > 0.0f && config.impulseDensity <= 64.0f &&
               std::isfinite(config.maxStepSize) && config.maxStepSize > 0.0f;
#else
        (void)config;
        return false;
#endif
    }

    bool SparseConvolutionNoiseMedium::intersectGP(PathSampleGenerator &sampler,
        const Ray &ray, MediumState &state, double &t, bool) const
    {
        PreparedGpisSegment prepared;
        prepareGpisSegment(sampler, ray, state, prepared);
        GpRaySegmentReply reply;
        if (state.gpRayBatchBackend)
            reply.valid = state.gpRayBatchBackend->trace(prepared.request, reply.result);
        return finishGpisSegment(ray, state, prepared, reply, t);
    }

    void SparseConvolutionNoiseMedium::prepareGpisSegment(PathSampleGenerator &sampler,
        const Ray &ray, MediumState &state, PreparedGpisSegment &prepared) const
    {
        GpisCpuTimer timer(cpuProfile().conditioningNs);
        auto gp = _gp->get_gaussian_process();
        SparseConvolutionNoiseRealization noise(gp, _ctxt, _globalSeed, _impulseDensity, _useSingleRealization, _isotropicSpace3DSampling, _1DSampling, _1DsamplingScheme, _correlationXY, _surfVolPhaseSeparate, _surfVolPhaseAmpThresh);

        prepared.uniformSampler = UniformSampler();
        auto &uniformSampler = prepared.uniformSampler;
        auto p = vec_conv<Vec3d>(ray.pos());

        if (!state.firstScatter) {
            double targetVal = state.lastVal;
            Vec3d targetGrad = state.lastAniso;
            // Perform conditioning at the start of the secondary ray
            // Supported memory models: Renewal, Renewal+
            noise.conditioning(Vec3f(p), ray.dir(), targetVal, Vec3f(targetGrad), state.info, uniformSampler);
        }

        prepared.context = std::make_shared<GPContextSparseConvNoise>();
        prepared.context->noise = noise;
        auto &request = prepared.request;
        request.purpose = GpRaySegmentRequest::Purpose::Path;
        request.medium = this;
        request.origin = ray.pos();
        request.direction = ray.dir();
        request.nearT = ray.nearT();
        request.farT = ray.farT();
        request.jitter = sampler.next1D();
        request.noiseSeed = _globalSeed + MathUtil::xxhash32(state.info.pixelSampleSegment) +
                            MathUtil::xxhash32(state.info.sceneSeed);
        request.normalSeed = MathUtil::xxhash32(state.info.pixelSampleSegment) +
                             MathUtil::xxhash32(state.info.sceneSeed) + 1u;
        request.firstScatter = state.firstScatter;
        request.previousValue = state.firstScatter ? 0.0f : state.lastVal;
        request.previousGradient = state.firstScatter ? Vec3f(0.0f) : Vec3f(state.lastAniso);
    }

    bool SparseConvolutionNoiseMedium::finishGpisSegment(const Ray &ray, MediumState &state,
        PreparedGpisSegment &prepared, const GpRaySegmentReply &reply, double &t) const
    {
        auto gp = _gp->get_gaussian_process();
        auto &noise = prepared.context->noise;
        auto p = vec_conv<Vec3d>(ray.pos());
        auto rd = vec_conv<Vec3d>(ray.dir());
        if (reply.valid) {
            const auto &result = reply.result;
            t = result.t;
            state.gpContext = prepared.context;
            if (result.hit) {
                state.lastGPId = static_cast<int>(gp->mean_weight_space(
                    Vec3d(p + t * rd), Derivative::None)[1]);
                state.sparseConv1DSamplingScheme = noise.samplingScheme(Vec3f(p + t * rd));
                state.lastVal = 0.0f;
            } else {
                state.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
                state.lastVal = result.value;
            }
            return result.hit;
        }
        int step = 0;
        GpisCpuTimer timer(cpuProfile().marchingNs, &step);
        float step_size = (ray.farT() - ray.nearT()) / float(_rayMarchMinStep);
        if (_rayMarchStepSize < step_size)
            step_size = _rayMarchStepSize;
        int GPId;
        auto &uniformSampler = prepared.uniformSampler;
        t = ray.nearT();
        double f0 = noise.evaluateValue(Vec3f(p + t * rd), t, ray.dir(), state.info, uniformSampler, GPId);
        int sign0 = f0 < 0 ? -1 : 1;
        double pf = f0;
        t = ray.nearT() + step_size * prepared.request.jitter;
        // Perform ray marching (with equal step size) to find the first zero-crossing
        while (t < ray.farT()) {
            step ++;
            auto p_c = p + t * rd;
            double f_c = noise.evaluateValue(Vec3f(p_c), t, ray.dir(), state.info, uniformSampler, GPId);
            int signc = f_c < 0 ? -1 : 1;

            if (!state.firstScatter && step == 1) {
                sign0 = signc;
            }
            else if (signc != sign0) {
                // Estimate the exact intersect location
                double intp_factor = pf / (pf - f_c);
                double t_test, t_test_prev;
                int sign_test;
                t_test_prev = lerp(t - step_size, t, intp_factor);
                do {
                    t_test = lerp(t - step_size, t, intp_factor); // intp_factor changes in each loop
                    double f_test = noise.evaluateValue(Vec3f(p + t_test * rd), t_test, ray.dir(), state.info, uniformSampler, GPId);
                    sign_test = f_test < 0 ? -1 : 1;
                    if (sign_test == sign0) {
                        break;
                    }
                    intp_factor *= 0.9;
                    if (intp_factor <= 0.01) {
                        t_test_prev = t_test = 0;
                        break;
                    }
                    t_test_prev = t_test;
                } while(true);

                t = t_test_prev;
                state.lastGPId = GPId;
                state.sparseConv1DSamplingScheme = noise.samplingScheme(Vec3f(p + t * rd));
                state.gpContext = prepared.context;
                state.lastVal = 0.0;
                return true;
            }

            pf = f_c;
            t += step_size;
        }

        t = ray.farT();
        state.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
        state.gpContext = prepared.context;
        state.lastVal = noise.evaluateValue(Vec3f(p + t * rd), t, ray.dir(), state.info, uniformSampler, GPId);
        return false;
    }
}
