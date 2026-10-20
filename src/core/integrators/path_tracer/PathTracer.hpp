#ifndef PATHTRACER_HPP_
#define PATHTRACER_HPP_

#include "PathTracerSettings.hpp"

#include "integrators/TraceBase.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"
#include "primitives/IntersectionTemporary.hpp"
#include "primitives/IntersectionInfo.hpp"

namespace Tungsten {

class GpRayBatchBackend;

class PathTracer : public TraceBase
{
    PathTracerSettings _settings;
    bool _trackOutputValues;
    GpRayBatchBackend *_gpRayBatchBackend;

public:
    enum class SampleStatus { NeedGpis, Finished };
    // A live path and its sampler must remain at stable addresses across yields.
    // Deferred AOVs are committed by the pool coordinator, together with color.
    struct OutputSample {
        bool depthValid = false, normalValid = false, albedoValid = false, visibilityValid = false;
        float depth = 0.0f, visibility = 0.0f;
        Vec3f normal = Vec3f(0.0f), albedo = Vec3f(0.0f);
    };
    struct PathState {
        enum class Stage { Start, Bounce, Distance, Waiting, Shade, Direct, Scatter, Finished };
        Stage stage = Stage::Start;
        Vec2u pixel;
        uint spp = 0;
        Vec3f throughput, emission = Vec3f(0.0f);
        Ray ray;
        MediumSample mediumSample;
        SurfaceScatterEvent surfaceEvent;
        IntersectionTemporary data;
        IntersectionInfo info;
        Medium::MediumState state;
        const Medium *medium = nullptr;
        const SparseConvolutionNoiseMedium *sparse = nullptr;
        GaussianProcessMedium::DistanceState distance;
        DirectLightingState direct;
        SurfaceState surface;
        Vec3f transmittance = Vec3f(-1.0f);
        bool hitSurface = true, enableDirect = false;
        PreparedGpisSegment prepared;
        GpRaySegmentReply reply;
        bool replyReady = false;
        bool recordedOutputValues = false;
        bool deferOutputValues = false;
        OutputSample output;
        float hitDistance = 0.0f;
        int mediumBounces = 0, bounce = 0;
        bool didHit = false, wasSpecular = true;
        PathState(Vec2u pixel, uint spp) : pixel(pixel), spp(spp) {}
        PathState(const PathState &) = delete;
        PathState &operator=(const PathState &) = delete;
    };

    PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId,
               GpRayBatchBackend *gpRayBatchBackend = nullptr);

    Vec3f traceSample(Vec2u pixel, PathSampleGenerator &sampler, uint spp = 0);
    SampleStatus advanceSample(PathState &path, PathSampleGenerator &sampler);
    static void resumeGpis(PathState &path, const GpRaySegmentReply &reply);
    void commitOutput(Vec2u pixel, const OutputSample &output);

    std::function<bool(const MediumSample& sample, Ray r)> _firstMediumBounceCb;
    std::function<bool(const SurfaceScatterEvent& ev, Ray r)> _firstSurfaceBounceCb;
};

}

#endif /* PATHTRACER_HPP_ */
