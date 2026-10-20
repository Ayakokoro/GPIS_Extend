#ifndef TRACEBASE_HPP_
#define TRACEBASE_HPP_

#include "TraceSettings.hpp"

#include "samplerecords/SurfaceScatterEvent.hpp"
#include "samplerecords/MediumSample.hpp"
#include "samplerecords/LightSample.hpp"
#include "samplerecords/PhaseSample.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformSampler.hpp"
#include "sampling/Distribution1D.hpp"
#include "sampling/SampleWarp.hpp"

#include "renderer/TraceableScene.hpp"

#include "cameras/Camera.hpp"

#include "media/Medium.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"

#include "math/TangentFrame.hpp"
#include "math/MathUtil.hpp"
#include "math/Angle.hpp"

#include "bsdfs/Bsdf.hpp"

#include <vector>
#include <memory>
#include <cmath>

namespace Tungsten {

    class MediumState;
    class Cube;

class TraceBase
{
protected:
    const TraceableScene *_scene;
    TraceSettings _settings;
    uint32 _threadId;

    // For computing direct lighting probabilities
    std::vector<float> _lightPdf;
    // For sampling light sources in adjoint light tracing
    std::unique_ptr<Distribution1D> _lightSampler;
    std::vector<const Cube *> _gpisBoundaryCubes;

    TraceBase(TraceableScene *scene, const TraceSettings &settings, uint32 threadId);

    bool isConsistent(const SurfaceScatterEvent &event, const Vec3f &w) const;
    bool intersectScene(Ray &ray, const Medium *medium,
        IntersectionTemporary &data, IntersectionInfo &info) const;

    // These continuations own every local needed while a shadow GP segment is
    // in flight. Intersection data for the light and the scene remain separate.
    struct ShadowState {
        enum class Stage { Segment, Medium, Intersection, Waiting, Boundary, Finished };
        Stage stage = Stage::Segment;
        Ray ray;
        const Medium *medium = nullptr;
        const Primitive *endCap = nullptr;
        const SparseConvolutionNoiseMedium *sparse = nullptr;
        Medium::MediumState state, blankState;
        IntersectionTemporary data;
        IntersectionInfo info;
        GaussianProcessMedium::TransmittanceState transmittance;
        Vec3f throughput = Vec3f(1.0f);
        float initialFarT = 0.0f;
        int bounce = 0;
        bool didHit = false, startsOnSurface = false;
    };
    struct DirectLightingState {
        enum class Stage { LightSample, LightShadow, ScatterSample, ScatterShadow, Finished };
        Stage stage = Stage::LightSample;
        const Primitive *light = nullptr;
        float lightWeight = 0.0f;
        int bounce = 0;
        bool surface = false;
        Medium::MediumState state;
        LightSample lightSample;
        PhaseSample phaseSample;
        Vec3f f, result = Vec3f(0.0f), firstResult = Vec3f(0.0f);
        Vec3f visibility = Vec3f(-1.0f);
        ShadowState shadow;
        IntersectionTemporary lightData;
        IntersectionInfo lightInfo;
    };
    struct SurfaceState {
        bool forward = false;
        Vec3f wo;
    };

    void beginDirectLighting(DirectLightingState &direct, PathSampleGenerator &sampler,
        const Vec3f &p, const Medium::MediumState &state, int bounce, bool surface);
    bool advanceDirectLighting(DirectLightingState &direct, PathSampleGenerator &sampler,
        MediumSample &sample, SurfaceScatterEvent &event, const Medium *medium,
        const Ray &parentRay, PreparedGpisSegment &prepared, const GpRaySegmentReply &reply,
        bool &replyReady);
    void beginShadow(ShadowState &shadow, const Ray &ray, const Medium *medium,
        const Primitive *endCap, int bounce, bool startsOnSurface,
        const Medium::MediumState &state);
    bool advanceShadow(ShadowState &shadow, PathSampleGenerator &sampler,
        PreparedGpisSegment &prepared, const GpRaySegmentReply &reply, bool &replyReady) const;
    bool beginLightShadow(DirectLightingState &direct, Ray ray, const Medium *medium,
        float expectedDist, bool startsOnSurface, const Medium::MediumState &state);
    void beginSurface(SurfaceState &surface, SurfaceScatterEvent &event,
        const Ray &ray, Vec3f &throughput) const;
    bool finishSurface(SurfaceState &surface, SurfaceScatterEvent &event,
        IntersectionTemporary &data, IntersectionInfo &info, const Medium *&medium,
        int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
        Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
        Medium::MediumState &mediumState) const;

    template<bool ComputePdfs>
    inline Vec3f generalizedShadowRayImpl(PathSampleGenerator &sampler,
                               Ray &ray,
                               const Medium *medium,
                               const Primitive *endCap,
                               int bounce,
                               bool startsOnSurface,
                               bool endsOnSurface,
                               Medium::MediumState* mediumState,
                               float &pdfForward,
                               float &pdfBackward) const;

    Vec3f attenuatedEmission(PathSampleGenerator &sampler,
                             const Primitive &light,
                             const Medium *medium,
                             float expectedDist,
                             IntersectionTemporary &data,
                             IntersectionInfo &info,
                             int bounce,
                             bool startsOnSurface,
                             Ray &ray,
                             Medium::MediumState* mediumState,
                             Vec3f *transmittance);

    bool volumeLensSample(const Camera &camera,
                    PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    Medium::MediumState* mediumState,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay,
                    Vec3f &weight,
                    Vec2f &pixel);

    bool surfaceLensSample(const Camera &camera,
                    SurfaceScatterEvent &event,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay,
                    Vec3f &weight,
                    Vec2f &pixel);

    Vec3f lightSample(const Primitive &light,
                      SurfaceScatterEvent &event,
                      Medium::MediumState* mediumState,
                      const Medium *medium,
                      int bounce,
                      const Ray &parentRay,
                      Vec3f *transmittance);

    Vec3f bsdfSample(const Primitive &light,
                     SurfaceScatterEvent &event,
                     Medium::MediumState* mediumState,
                     const Medium *medium,
                     int bounce,
                     const Ray &parentRay);

    Vec3f volumeLightSample(PathSampleGenerator &sampler,
                            MediumSample &mediumSample,
                        Medium::MediumState* mediumState,
                        const Primitive &light,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay);

    Vec3f volumePhaseSample(const Primitive &light,
                        PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        Medium::MediumState* mediumState,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay);

    Vec3f sampleDirect(const Primitive &light,
                       SurfaceScatterEvent &event,
                       Medium::MediumState* mediumState,
                       const Medium *medium,
                       int bounce,
                       const Ray &parentRay,
                       Vec3f *transmittance);

    Vec3f volumeSampleDirect(const Primitive &light,
                        PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        Medium::MediumState* mediumState,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay);

    const Primitive *chooseLight(PathSampleGenerator &sampler, const Vec3f &p, float &weight);
    const Primitive *chooseLightAdjoint(PathSampleGenerator &sampler, float &pdf);

    Vec3f volumeEstimateDirect(PathSampleGenerator &sampler,
                        MediumSample &mediumSample,
                        Medium::MediumState* mediumState,
                        const Medium *medium,
                        int bounce,
                        const Ray &parentRay);

    Vec3f estimateDirect(SurfaceScatterEvent &event,
                         Medium::MediumState* mediumState,
                         const Medium *medium,
                         int bounce,
                         const Ray &parentRay,
                         Vec3f *transmission);

public:
    SurfaceScatterEvent makeLocalScatterEvent(IntersectionTemporary &data, IntersectionInfo &info,
            Ray &ray, PathSampleGenerator *sampler) const;

    Vec3f generalizedShadowRay(PathSampleGenerator &sampler,
                               Ray &ray,
                               const Medium *medium,
                               const Primitive *endCap,
                               bool startsOnSurface,
                               bool endsOnSurface,
                               int bounce,
                               Medium::MediumState* mediumState) const;
    Vec3f generalizedShadowRayAndPdfs(PathSampleGenerator &sampler,
                               Ray &ray,
                               const Medium *medium,
                               const Primitive *endCap,
                               int bounce,
                               bool startsOnSurface,
                               bool endsOnSurface,
                               Medium::MediumState* mediumState,
                               float &pdfForward,
                               float &pdfBackward) const;

    bool handleVolume(PathSampleGenerator &sampler, MediumSample &mediumSample, Medium::MediumState& mediumState,
               const Medium *&medium, int bounce, bool adjoint, bool enableLightSampling,
               Ray &ray, Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
               bool directLightingDone = false);

    bool handleSurface(SurfaceScatterEvent &event, IntersectionTemporary &data,
               IntersectionInfo &info, const Medium *&medium,
               int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
               Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
               Medium::MediumState &mediumState, Vec3f *transmittance = nullptr);

    void handleInfiniteLights(IntersectionTemporary &data,
            IntersectionInfo &info, bool enableLightSampling, Ray &ray,
            Vec3f throughput, bool wasSpecular, Vec3f &emission);
};

}

#endif /* TRACEBASE_HPP_ */
