#include "TraceBase.hpp"
#include "primitives/Cube.hpp"
#include "media/ClassicGlobalMedium.hpp"
#include "media/ClassicLocalMedium.hpp"
#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
#include "media/NeuralRenewalMedium.hpp"
#endif

namespace Tungsten {

TraceBase::TraceBase(TraceableScene *scene, const TraceSettings &settings, uint32 threadId)
: _scene(scene),
  _settings(settings),
  _threadId(threadId)
{
    _scene = scene;
    _lightPdf.resize(scene->lights().size());

    std::vector<float> lightWeights(scene->lights().size());
    for (size_t i = 0; i < scene->lights().size(); ++i) {
        scene->lights()[i]->makeSamplable(*_scene, _threadId);
        lightWeights[i] = 1.0f; // TODO: Use light power here
    }
    _lightSampler.reset(new Distribution1D(std::move(lightWeights)));

    for (const auto &prim : scene->lights())
        prim->makeSamplable(*_scene, _threadId);
    for (const auto &prim : scene->primitives()) {
        auto *cube = dynamic_cast<Cube *>(prim.get());
        if (cube && (dynamic_cast<const GaussianProcessMedium *>(cube->intMedium().get()) ||
                     dynamic_cast<const ClassicLocalMedium *>(cube->intMedium().get()) ||
                     dynamic_cast<const ClassicGlobalMedium *>(cube->intMedium().get())
#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
                     || dynamic_cast<const NeuralRenewalMedium *>(cube->intMedium().get())
#endif
                     ) &&
            cube->bsdf(0)->lobes().isForward())
            _gpisBoundaryCubes.push_back(cube);
    }
}

bool TraceBase::intersectScene(Ray &ray, const Medium *medium,
        IntersectionTemporary &data, IntersectionInfo &info) const
{
    bool hit = _scene->intersect(ray, data, info);
    // BVH traversal also ignores exits before nearT. Query the current GP/classic
    // boundary directly so a transparent boundary still updates medium state.
    if (medium) {
        for (const auto *cube : _gpisBoundaryCubes) {
            if (cube->intMedium().get() == medium && cube->intersectMediumExit(ray, data)) {
                info.p = ray.hitpoint();
                info.w = ray.dir();
                info.epsilon = _scene->rayEpsilon();
                cube->intersectionInfo(data, info);
                hit = true;
            }
        }
    }
    return hit;
}

SurfaceScatterEvent TraceBase::makeLocalScatterEvent(IntersectionTemporary &data, IntersectionInfo &info,
        Ray &ray, PathSampleGenerator *sampler) const
{
    TangentFrame frame;
    info.primitive->setupTangentFrame(data, info, frame);

    bool hitBackside = frame.normal.dot(ray.dir()) > 0.0f;
    bool isTransmissive = info.bsdf->lobes().isTransmissive();

    bool flipFrame = _settings.enableTwoSidedShading && hitBackside && !isTransmissive;

    if (flipFrame) {
        // TODO: Should we flip info.Ns here too? It doesn't seem to be used at the moment,
        // but it may be in the future. Modifying the intersection info itself could be a bad
        // idea though
        frame.normal = -frame.normal;
        frame.tangent = -frame.tangent;
    }

    SurfaceScatterEvent event(
        &info,
        sampler,
        frame,
        frame.toLocal(-ray.dir()),
        BsdfLobes::AllLobes,
        flipFrame
    );
    // These are real geometry events. GP phase functions set their own sampling
    // scheme/context explicitly; leaving them uninitialized here makes ordinary
    // mirror/conductor triangles accidentally dereference a GP context.
    event.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
    event.ctxt = nullptr;
    return event;
}

bool TraceBase::isConsistent(const SurfaceScatterEvent &event, const Vec3f &w) const
{
    if (!_settings.enableConsistencyChecks)
        return true;
    bool geometricBackside = (w.dot(event.info->Ng) < 0.0f);
    bool shadingBackside = (event.wo.z() < 0.0f) ^ event.flippedFrame;
    return geometricBackside == shadingBackside;
}

template<bool ComputePdfs>
inline Vec3f TraceBase::generalizedShadowRayImpl(PathSampleGenerator &sampler,
                           Ray &ray,
                           const Medium *medium,
                           const Primitive *endCap,
                           int bounce,
                           bool startsOnSurface,
                           bool endsOnSurface,
                           Medium::MediumState* mediumState,
                           float &pdfForward,
                           float &pdfBackward) const
{
    IntersectionTemporary data;
    IntersectionInfo info;

    float initialFarT = ray.farT();
    Vec3f throughput(1.0f);
    Medium::MediumState mediumStateIter;
    Medium::MediumState mediumStateIterBlank;
    mediumStateIterBlank.reset();
    if (mediumState) {
        mediumStateIter = *mediumState;
        mediumStateIterBlank.info = mediumStateIter.info;
    }

    int count = 0;
    do {
        count ++;
        bool didHit = intersectScene(ray, medium, data, info) && info.primitive != endCap;
        if (didHit) {
            if (!info.bsdf->lobes().hasForward())
                return Vec3f(0.0f);

            SurfaceScatterEvent event = makeLocalScatterEvent(data, info, ray, nullptr);

            // For forward events, the transport direction does not matter (since wi = -wo)
            Vec3f transparency = info.bsdf->eval(event.makeForwardEvent(), false);
            if (transparency == 0.0f)
                return Vec3f(0.0f);

            if (ComputePdfs) {
                float transparencyScalar = transparency.avg();
                pdfForward  *= transparencyScalar;
                pdfBackward *= transparencyScalar;
            }

            throughput *= transparency;
            mediumStateIterBlank.info.pixelSampleSegment.w() ++;

            if (bounce >= _settings.maxBounces)
                return Vec3f(0.0f);
        }

        if (medium) {
            if (ComputePdfs) {
                float forward, backward;
                throughput *= medium->transmittanceAndPdfs(sampler, ray, startsOnSurface, didHit || endsOnSurface, mediumState, forward, backward);
                pdfForward *= forward;
                pdfBackward *= backward;
            } else {
                throughput *= medium->transmittance(sampler, ray, startsOnSurface, endsOnSurface, &mediumStateIter);
            }
        }
        if (info.primitive == nullptr || info.primitive == endCap)
            return bounce >= _settings.minBounces ? throughput : Vec3f(0.0f);
        const Medium *nextMedium = info.primitive->selectMedium(medium, !info.primitive->hitBackside(data));
        if (nextMedium != medium || !mediumStateIter.pathContext)
            mediumStateIter = mediumStateIterBlank;
        medium = nextMedium;
        startsOnSurface = true;

        ray.setPos(ray.hitpoint());
        initialFarT -= ray.farT();
        ray.setNearT(info.epsilon);
        ray.setFarT(initialFarT);
    } while(true);
    return Vec3f(0.0f);
}

Vec3f TraceBase::generalizedShadowRay(PathSampleGenerator &sampler, Ray &ray, const Medium *medium,
            const Primitive *endCap, bool startsOnSurface, bool endsOnSurface, int bounce, Medium::MediumState* mediumState) const
{
    float dummyA, dummyB;
    return generalizedShadowRayImpl<false>(sampler, ray, medium, endCap, bounce,
            startsOnSurface, endsOnSurface, mediumState, dummyA, dummyB);
}

Vec3f TraceBase::generalizedShadowRayAndPdfs(PathSampleGenerator &sampler, Ray &ray, const Medium *medium,
           const Primitive *endCap, int bounce, bool startsOnSurface, bool endsOnSurface, Medium::MediumState* mediumState,
           float &pdfForward, float &pdfBackward) const
{
    pdfForward = pdfBackward = 1.0f;
    return generalizedShadowRayImpl<true>(sampler, ray, medium, endCap, bounce,
            startsOnSurface, endsOnSurface, mediumState, pdfForward, pdfBackward);
}

void TraceBase::beginShadow(ShadowState &shadow, const Ray &ray, const Medium *medium,
        const Primitive *endCap, int bounce, bool startsOnSurface,
        const Medium::MediumState &state)
{
    shadow = ShadowState();
    shadow.ray = ray;
    shadow.initialFarT = ray.farT();
    shadow.medium = medium;
    shadow.endCap = endCap;
    shadow.bounce = bounce;
    shadow.startsOnSurface = startsOnSurface;
    shadow.state = state;
    shadow.state.gpRayBatchBackend = nullptr;
    shadow.blankState.reset();
    shadow.blankState.info = state.info;
}

bool TraceBase::advanceShadow(ShadowState &shadow, PathSampleGenerator &sampler,
        PreparedGpisSegment &prepared, const GpRaySegmentReply &reply, bool &replyReady) const
{
    using Stage = ShadowState::Stage;
    using Status = GaussianProcessMedium::TransmittanceStatus;
    while (shadow.stage != Stage::Finished) {
        if (shadow.stage == Stage::Segment) {
            shadow.didHit = intersectScene(shadow.ray, shadow.medium, shadow.data, shadow.info) &&
                shadow.info.primitive != shadow.endCap;
            if (shadow.didHit) {
                if (!shadow.info.bsdf->lobes().hasForward()) {
                    shadow.throughput = Vec3f(0.0f);
                    shadow.stage = Stage::Finished;
                    break;
                }
                auto event = makeLocalScatterEvent(shadow.data, shadow.info, shadow.ray, nullptr);
                Vec3f transparency = shadow.info.bsdf->eval(event.makeForwardEvent(), false);
                if (transparency == 0.0f || shadow.bounce >= _settings.maxBounces) {
                    shadow.throughput = Vec3f(0.0f);
                    shadow.stage = Stage::Finished;
                    break;
                }
                shadow.throughput *= transparency;
                shadow.blankState.info.pixelSampleSegment.w()++;
            }
            shadow.stage = Stage::Medium;
        }
        if (shadow.stage == Stage::Medium) {
            shadow.sparse = dynamic_cast<const SparseConvolutionNoiseMedium *>(shadow.medium);
            Se1DGpBatchConfig config;
            if (shadow.ray.farT() <= shadow.ray.nearT()) {
                shadow.stage = Stage::Boundary;
            } else if (shadow.sparse && shadow.sparse->se1DGpuBatchConfig(config)) {
                shadow.sparse->beginTransmittance(shadow.ray, shadow.transmittance);
                shadow.stage = Stage::Intersection;
            } else {
                if (shadow.medium)
                    shadow.throughput *= shadow.medium->transmittance(sampler, shadow.ray,
                        shadow.startsOnSurface, true, &shadow.state);
                shadow.stage = Stage::Boundary;
            }
        }
        if (shadow.stage == Stage::Waiting) {
            if (!replyReady)
                return false;
            double t = shadow.transmittance.t;
            bool hit = shadow.sparse->finishGpisSegment(shadow.transmittance.ray,
                shadow.state, prepared, reply, t);
            replyReady = false;
            auto status = shadow.sparse->resumeTransmittance(sampler, shadow.ray,
                shadow.state, shadow.transmittance, t, hit);
            prepared.context.reset();
            if (status == Status::NeedIntersection)
                shadow.stage = Stage::Intersection;
            else {
                shadow.throughput *= shadow.transmittance.value;
                shadow.stage = Stage::Boundary;
            }
        }
        if (shadow.stage == Stage::Intersection) {
            shadow.sparse->prepareGpisSegment(sampler, shadow.transmittance.ray,
                shadow.state, prepared);
            prepared.request.purpose = GpRaySegmentRequest::Purpose::Shadow;
            shadow.stage = Stage::Waiting;
            return false;
        }
        if (shadow.stage == Stage::Boundary) {
            if (!shadow.info.primitive || shadow.info.primitive == shadow.endCap) {
                if (shadow.bounce < _settings.minBounces)
                    shadow.throughput = Vec3f(0.0f);
                shadow.stage = Stage::Finished;
                break;
            }
            const Medium *nextMedium = shadow.info.primitive->selectMedium(shadow.medium,
                !shadow.info.primitive->hitBackside(shadow.data));
            if (nextMedium != shadow.medium || !shadow.state.pathContext)
                shadow.state = shadow.blankState;
            shadow.medium = nextMedium;
            shadow.startsOnSurface = true;
            shadow.ray.setPos(shadow.ray.hitpoint());
            shadow.initialFarT -= shadow.ray.farT();
            shadow.ray.setNearT(shadow.info.epsilon);
            shadow.ray.setFarT(shadow.initialFarT);
            shadow.stage = Stage::Segment;
        }
    }
    return true;
}

void TraceBase::beginDirectLighting(DirectLightingState &direct, PathSampleGenerator &sampler,
        const Vec3f &p, const Medium::MediumState &state, int bounce, bool surface)
{
    direct = DirectLightingState();
    direct.surface = surface;
    direct.bounce = bounce;
    direct.state = state;
    direct.state.info.pixelSampleSegment.w()++;
    direct.light = chooseLight(sampler, p, direct.lightWeight);
    if (!direct.light)
        direct.stage = DirectLightingState::Stage::Finished;
}

bool TraceBase::beginLightShadow(DirectLightingState &direct, Ray ray, const Medium *medium,
        float expectedDist, bool startsOnSurface, const Medium::MediumState &state)
{
    const auto &light = *direct.light;
    if (light.isDirac())
        ray.setFarT(expectedDist);
    else if (!light.intersect(ray, direct.lightData) || ray.farT()*(1.0f + 1e-3f) < expectedDist)
        return false;
    direct.lightInfo.p = ray.pos() + ray.dir()*ray.farT();
    direct.lightInfo.w = ray.dir();
    light.intersectionInfo(direct.lightData, direct.lightInfo);
    beginShadow(direct.shadow, ray, medium, &light, direct.bounce, startsOnSurface, state);
    return true;
}

bool TraceBase::advanceDirectLighting(DirectLightingState &direct, PathSampleGenerator &sampler,
        MediumSample &sample, SurfaceScatterEvent &event, const Medium *medium,
        const Ray &parentRay, PreparedGpisSegment &prepared, const GpRaySegmentReply &reply,
        bool &replyReady)
{
    using Stage = DirectLightingState::Stage;
    while (direct.stage != Stage::Finished) {
        const auto &light = *direct.light;
        if (direct.stage == Stage::LightSample) {
            direct.stage = Stage::ScatterSample;
            if (direct.surface && (event.info->bsdf->lobes().isPureSpecular() ||
                                  event.info->bsdf->lobes().isForward())) {
                direct.stage = Stage::Finished;
                break;
            }
            if (!direct.surface && sample.phase->isSpecular() &&
                sample.sparseConv1DSamplingScheme == SparseConv1DSamplingScheme::UNI)
                continue;
            const Vec3f p = direct.surface ? event.info->p : sample.p;
            if (!light.sampleDirect(_threadId, p, sampler, direct.lightSample))
                continue;
            auto &ls = direct.lightSample;
            const Medium *shadowMedium = medium;
            auto shadowState = direct.state;
            if (direct.surface) {
                event.wo = event.frame.toLocal(ls.d);
                if (!isConsistent(event, ls.d))
                    continue;
                shadowMedium = event.info->primitive->selectMedium(medium, ls.d.dot(event.info->Ng) < 0.0f);
                event.requestedLobe = BsdfLobes::AllButSpecular;
                direct.f = event.info->bsdf->eval(event, false);
                shadowState.firstScatter = true;
            } else {
                direct.f = sample.phase->evalConnection(sampler, parentRay.dir(), ls.d, sample, shadowState.pathContext);
                if (sample.phase->isSpecular())
                    shadowState.lastAniso = Vec3d(sample.phase->evalGrad(parentRay.dir(), ls.d, sample));
            }
            if (direct.f == 0.0f)
                continue;
            auto ray = parentRay.scatter(p, ls.d, direct.surface ? event.info->epsilon : 0.0f);
            ray.setPrimaryRay(false);
            if (beginLightShadow(direct, ray, shadowMedium, ls.dist, direct.surface, shadowState))
                direct.stage = Stage::LightShadow;
        }
        if (direct.stage == Stage::LightShadow) {
            if (!advanceShadow(direct.shadow, sampler, prepared, reply, replyReady))
                return false;
            direct.visibility = direct.shadow.throughput;
            if (direct.visibility != 0.0f) {
                Vec3f e = direct.visibility*light.evalDirect(direct.lightData, direct.lightInfo);
                if (e != 0.0f) {
                    direct.firstResult = direct.f*e/direct.lightSample.pdf;
                    if (direct.surface) {
                        if (!light.isDirac())
                            direct.firstResult *= SampleWarp::powerHeuristic(direct.lightSample.pdf,
                                event.info->bsdf->pdf(event));
                    } else {
                        bool disableMIS = sample.phase->isSpecular() &&
                            sample.sparseConv1DSamplingScheme == SparseConv1DSamplingScheme::NEE;
                        if (!light.isDirac() && !disableMIS)
                            direct.firstResult *= SampleWarp::powerHeuristic(direct.lightSample.pdf,
                                sample.phase->pdf(parentRay.dir(), direct.lightSample.d, sample));
                    }
                }
            }
            direct.stage = Stage::ScatterSample;
        }
        if (direct.stage == Stage::ScatterSample) {
            direct.stage = Stage::Finished;
            if (light.isDirac())
                break;
            const Medium *shadowMedium = medium;
            auto shadowState = direct.state;
            Vec3f p, wo;
            if (direct.surface) {
                event.requestedLobe = BsdfLobes::AllButSpecular;
                if (!event.info->bsdf->sample(event, false) || event.weight == 0.0f)
                    break;
                wo = event.frame.toGlobal(event.wo);
                if (!isConsistent(event, wo))
                    break;
                p = event.info->p;
                shadowMedium = event.info->primitive->selectMedium(medium, wo.dot(event.info->Ng) < 0.0f);
                shadowState.firstScatter = true;
            } else {
                if (sample.phase->isSpecular() &&
                    sample.sparseConv1DSamplingScheme == SparseConv1DSamplingScheme::NEE)
                    break;
                if (!sample.phase->sample(sampler, parentRay.dir(), sample, direct.phaseSample))
                    break;
                p = sample.p;
                wo = direct.phaseSample.w;
                if (direct.phaseSample.pathContext) shadowState.pathContext = direct.phaseSample.pathContext;
            }
            auto ray = parentRay.scatter(p, wo, direct.surface ? event.info->epsilon : 0.0f);
            ray.setPrimaryRay(false);
            if (beginLightShadow(direct, ray, shadowMedium, -1.0f, direct.surface, shadowState))
                direct.stage = Stage::ScatterShadow;
        }
        if (direct.stage == Stage::ScatterShadow) {
            if (!advanceShadow(direct.shadow, sampler, prepared, reply, replyReady))
                return false;
            Vec3f shadow = direct.shadow.throughput;
            if (shadow != 0.0f) {
                Vec3f e = shadow*light.evalDirect(direct.lightData, direct.lightInfo);
                if (e != 0.0f) {
                    if (direct.surface) {
                        direct.result = e*event.weight;
                        direct.result *= SampleWarp::powerHeuristic(event.pdf,
                            light.directPdf(_threadId, direct.lightData, direct.lightInfo, event.info->p));
                    } else {
                        direct.result = e*direct.phaseSample.weight;
                        if (!sample.phase->isSpecular() ||
                            sample.sparseConv1DSamplingScheme != SparseConv1DSamplingScheme::UNI)
                            direct.result *= SampleWarp::powerHeuristic(direct.phaseSample.pdf,
                                light.directPdf(_threadId, direct.lightData, direct.lightInfo, sample.p));
                    }
                }
            }
            direct.stage = Stage::Finished;
        }
    }
    direct.result = (direct.firstResult + direct.result)*direct.lightWeight;
    return true;
}

Vec3f TraceBase::attenuatedEmission(PathSampleGenerator &sampler,
                         const Primitive &light,
                         const Medium *medium,
                         float expectedDist,
                         IntersectionTemporary &data,
                         IntersectionInfo &info,
                         int bounce,
                         bool startsOnSurface,
                         Ray &ray,
                         Medium::MediumState* mediumState,
                         Vec3f *transmittance)
{
    CONSTEXPR float fudgeFactor = 1.0f + 1e-3f;

    if (light.isDirac()) {
        ray.setFarT(expectedDist);
    } else {
        if (!light.intersect(ray, data) || ray.farT()*fudgeFactor < expectedDist)
            return Vec3f(0.0f);
    }
    info.p = ray.pos() + ray.dir()*ray.farT();
    info.w = ray.dir();
    light.intersectionInfo(data, info);

    Vec3f shadow = generalizedShadowRay(sampler, ray, medium, &light, startsOnSurface, true, bounce, mediumState);
    if (transmittance)
        *transmittance = shadow;
    if (shadow == 0.0f)
        return Vec3f(0.0f);

    return shadow*light.evalDirect(data, info);
}

bool TraceBase::volumeLensSample(const Camera &camera,
                                 PathSampleGenerator &sampler,
                                 MediumSample &mediumSample,
                                 Medium::MediumState* mediumState,
                                 const Medium *medium,
                                 int bounce,
                                 const Ray &parentRay,
                                 Vec3f &weight,
                                 Vec2f &pixel)
{
    LensSample lensSample;
    if (!camera.sampleDirect(mediumSample.p, sampler, lensSample))
        return false;

    Vec3f f = mediumSample.phase->eval(parentRay.dir(), lensSample.d, mediumSample);
    if (f == 0.0f)
        return false;

    Ray ray = parentRay.scatter(mediumSample.p, lensSample.d, 0.0f);
    ray.setPrimaryRay(false);
    ray.setFarT(lensSample.dist);


    Vec3f transmittance = generalizedShadowRay(sampler, ray, medium, nullptr, false, true, bounce, mediumState);
    if (transmittance == 0.0f)
        return false;

    weight = f*transmittance*lensSample.weight;
    pixel = lensSample.pixel;

    return true;
}

bool TraceBase::surfaceLensSample(const Camera &camera,
                                  SurfaceScatterEvent &event,
                                  const Medium *medium,
                                  int bounce,
                                  const Ray &parentRay,
                                  Vec3f &weight,
                                  Vec2f &pixel)
{
    LensSample sample;
    if (!camera.sampleDirect(event.info->p, *event.sampler, sample))
        return false;

    event.wo = event.frame.toLocal(sample.d);
    if (!isConsistent(event, sample.d))
        return false;

    bool geometricBackside = (sample.d.dot(event.info->Ng) < 0.0f);
    medium = event.info->primitive->selectMedium(medium, geometricBackside);

    event.requestedLobe = BsdfLobes::AllButSpecular;

    Vec3f f = event.info->bsdf->eval(event, true);
    if (f == 0.0f)
        return false;

    Ray ray = parentRay.scatter(event.info->p, sample.d, event.info->epsilon);
    ray.setPrimaryRay(false);
    ray.setFarT(sample.dist);

    Vec3f transmittance = generalizedShadowRay(*event.sampler, ray, medium, nullptr, true, true, bounce, nullptr);
    if (transmittance == 0.0f)
        return false;

    weight = f*transmittance*sample.weight;
    pixel = sample.pixel;

    return true;
}

Vec3f TraceBase::lightSample(const Primitive &light,
                             SurfaceScatterEvent &event,
                             Medium::MediumState* mediumState,
                             const Medium *medium,
                             int bounce,
                             const Ray &parentRay,
                             Vec3f *transmittance)
{
    LightSample sample;
    if (!light.sampleDirect(_threadId, event.info->p, *event.sampler, sample))
        return Vec3f(0.0f);

    event.wo = event.frame.toLocal(sample.d);
    if (!isConsistent(event, sample.d))
        return Vec3f(0.0f);

    bool geometricBackside = (sample.d.dot(event.info->Ng) < 0.0f);
    medium = event.info->primitive->selectMedium(medium, geometricBackside);

    event.requestedLobe = BsdfLobes::AllButSpecular;

    Vec3f f = event.info->bsdf->eval(event, false);
    if (f == 0.0f)
        return Vec3f(0.0f);

    Ray ray = parentRay.scatter(event.info->p, sample.d, event.info->epsilon);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    Medium::MediumState mediumStateReplicate;
    mediumStateReplicate = *mediumState;
    mediumStateReplicate.firstScatter = true;
    Vec3f e = attenuatedEmission(*event.sampler, light, medium, sample.dist, data, info, bounce, true, ray, &mediumStateReplicate, transmittance);
    if (e == 0.0f)
        return Vec3f(0.0f);

    Vec3f lightF = f*e/sample.pdf;

    if (!light.isDirac())
        lightF *= SampleWarp::powerHeuristic(sample.pdf, event.info->bsdf->pdf(event));

    return lightF;
}

Vec3f TraceBase::bsdfSample(const Primitive &light,
                            SurfaceScatterEvent &event,
                            Medium::MediumState* mediumState,
                            const Medium *medium,
                            int bounce,
                            const Ray &parentRay)
{
    event.requestedLobe = BsdfLobes::AllButSpecular;
    if (!event.info->bsdf->sample(event, false))
        return Vec3f(0.0f);
    if (event.weight == 0.0f)
        return Vec3f(0.0f);

    Vec3f wo = event.frame.toGlobal(event.wo);
    if (!isConsistent(event, wo))
        return Vec3f(0.0f);

    bool geometricBackside = (wo.dot(event.info->Ng) < 0.0f);
    medium = event.info->primitive->selectMedium(medium, geometricBackside);

    Ray ray = parentRay.scatter(event.info->p, wo, event.info->epsilon);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    Medium::MediumState mediumStateReplicate;
    mediumStateReplicate = *mediumState;
    mediumStateReplicate.firstScatter = true;
    Vec3f e = attenuatedEmission(*event.sampler, light, medium, -1.0f, data, info, bounce, true, ray, &mediumStateReplicate, nullptr);

    if (e == Vec3f(0.0f))
        return Vec3f(0.0f);

    Vec3f bsdfF = e*event.weight;

    bsdfF *= SampleWarp::powerHeuristic(event.pdf, light.directPdf(_threadId, data, info, event.info->p));

    return bsdfF;
}

Vec3f TraceBase::volumeLightSample(PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    Medium::MediumState* mediumState,
                    const Primitive &light,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay)
{
    if (mediumSample.phase->isSpecular() && mediumSample.sparseConv1DSamplingScheme == SparseConv1DSamplingScheme::UNI)
        return Vec3f(0.f);

    LightSample lightSample;
    if (!light.sampleDirect(_threadId, mediumSample.p, sampler, lightSample))
        return Vec3f(0.0f);

    Medium::MediumState mediumStateCorrectNormal = *mediumState;
    Vec3f f = mediumSample.phase->evalConnection(sampler, parentRay.dir(), lightSample.d, mediumSample, mediumStateCorrectNormal.pathContext);
    if (f == 0.0f)
        return Vec3f(0.0f);

    Ray ray = parentRay.scatter(mediumSample.p, lightSample.d, 0.0f);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    if (mediumSample.phase->isSpecular()) {
        mediumStateCorrectNormal.lastAniso = Vec3d(mediumSample.phase->evalGrad(parentRay.dir(), lightSample.d, mediumSample));
    }
    Vec3f e = attenuatedEmission(sampler, light, medium, lightSample.dist, data, info, bounce, false, ray, &mediumStateCorrectNormal, nullptr);
    if (e == 0.0f)
        return Vec3f(0.0f);

    Vec3f lightF = f*e/lightSample.pdf;

    bool disableMIS = mediumSample.phase->isSpecular() && mediumSample.sparseConv1DSamplingScheme == SparseConv1DSamplingScheme::NEE;
    if (!light.isDirac() && !disableMIS) {
        lightF *= SampleWarp::powerHeuristic(lightSample.pdf, mediumSample.phase->pdf(parentRay.dir(), lightSample.d, mediumSample));
    }

    return lightF;
}

Vec3f TraceBase::volumePhaseSample(const Primitive &light,
                    PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    Medium::MediumState* mediumState,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay)
{
    if (mediumSample.phase->isSpecular() && mediumSample.sparseConv1DSamplingScheme == SparseConv1DSamplingScheme::NEE)
        return Vec3f(0.f);

    PhaseSample phaseSample;
    if (!mediumSample.phase->sample(sampler, parentRay.dir(), mediumSample, phaseSample))
        return Vec3f(0.0f);

    Ray ray = parentRay.scatter(mediumSample.p, phaseSample.w, 0.0f);
    ray.setPrimaryRay(false);

    IntersectionTemporary data;
    IntersectionInfo info;
    auto branchState = *mediumState;
    if (phaseSample.pathContext) branchState.pathContext = phaseSample.pathContext;
    Vec3f e = attenuatedEmission(sampler, light, medium, -1.0f, data, info, bounce, false, ray, &branchState, nullptr);

    if (e == Vec3f(0.0f))
        return Vec3f(0.0f);

    Vec3f phaseF = e*phaseSample.weight;

    if (!mediumSample.phase->isSpecular() || mediumSample.sparseConv1DSamplingScheme != SparseConv1DSamplingScheme::UNI) {
        phaseF *= SampleWarp::powerHeuristic(phaseSample.pdf, light.directPdf(_threadId, data, info, mediumSample.p));
    }

    return phaseF;
}

Vec3f TraceBase::sampleDirect(const Primitive &light,
                              SurfaceScatterEvent &event,
                              Medium::MediumState* mediumState,
                              const Medium *medium,
                              int bounce,
                              const Ray &parentRay,
                              Vec3f *transmittance)
{
    Vec3f result(0.0f);

    if (event.info->bsdf->lobes().isPureSpecular() || event.info->bsdf->lobes().isForward())
        return Vec3f(0.0f);
    
    result += lightSample(light, event, mediumState, medium, bounce, parentRay, transmittance);
    if (!light.isDirac())
        result += bsdfSample(light, event, mediumState, medium, bounce, parentRay);

    return result;
}

Vec3f TraceBase::volumeSampleDirect(const Primitive &light,
                    PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    Medium::MediumState* mediumState,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay)
{
    Vec3f result = volumeLightSample(sampler, mediumSample, mediumState, light, medium, bounce, parentRay);
    if (!light.isDirac())
        result += volumePhaseSample(light, sampler, mediumSample, mediumState, medium, bounce, parentRay);

    return result;
}

const Primitive *TraceBase::chooseLight(PathSampleGenerator &sampler, const Vec3f &p, float &weight)
{
    if (_scene->lights().empty())
        return nullptr;
    if (_scene->lights().size() == 1) {
        weight = 1.0f;
        return _scene->lights()[0].get();
    }

    float total = 0.0f;
    unsigned numNonNegative = 0;
    for (size_t i = 0; i < _lightPdf.size(); ++i) {
        _lightPdf[i] = _scene->lights()[i]->approximateRadiance(_threadId, p);
        if (_lightPdf[i] >= 0.0f) {
            total += _lightPdf[i];
            numNonNegative++;
        }
    }
    if (numNonNegative == 0) {
        for (size_t i = 0; i < _lightPdf.size(); ++i)
            _lightPdf[i] = 1.0f;
        total = _lightPdf.size();
    } else if (numNonNegative < _lightPdf.size()) {
        for (size_t i = 0; i < _lightPdf.size(); ++i) {
            float uniformWeight = (total == 0.0f ? 1.0f : total)/numNonNegative;
            if (_lightPdf[i] < 0.0f) {
                _lightPdf[i] = uniformWeight;
                total += uniformWeight;
            }
        }
    }
    if (total == 0.0f)
        return nullptr;
    float t = sampler.next1D()*total;
    for (size_t i = 0; i < _lightPdf.size(); ++i) {
        if (t < _lightPdf[i] || i == _lightPdf.size() - 1) {
            weight = total/_lightPdf[i];
            return _scene->lights()[i].get();
        } else {
            t -= _lightPdf[i];
        }
    }
    return nullptr;
}

const Primitive *TraceBase::chooseLightAdjoint(PathSampleGenerator &sampler, float &pdf)
{
    float u = sampler.next1D();
    int lightIdx;
    _lightSampler->warp(u, lightIdx);
    pdf = _lightSampler->pdf(lightIdx);
    return _scene->lights()[lightIdx].get();
}

Vec3f TraceBase::volumeEstimateDirect(PathSampleGenerator &sampler,
                    MediumSample &mediumSample,
                    Medium::MediumState* mediumState,
                    const Medium *medium,
                    int bounce,
                    const Ray &parentRay)
{
    float weight;
    const Primitive *light = chooseLight(sampler, mediumSample.p, weight);
    if (light == nullptr)
        return Vec3f(0.0f);
    return volumeSampleDirect(*light, sampler, mediumSample, mediumState, medium, bounce, parentRay)*weight;
}

Vec3f TraceBase::estimateDirect(SurfaceScatterEvent &event,
                                Medium::MediumState* mediumState,
                                const Medium *medium,
                                int bounce,
                                const Ray &parentRay,
                                Vec3f *transmittance)
{
    float weight;
    const Primitive *light = chooseLight(*event.sampler, event.info->p, weight);
    if (light == nullptr)
        return Vec3f(0.0f);
    return sampleDirect(*light, event, mediumState, medium, bounce, parentRay, transmittance)*weight;
}

bool TraceBase::handleVolume(PathSampleGenerator &sampler, MediumSample &mediumSample, 
                             Medium::MediumState& mediumState,
           const Medium *&medium, int bounce, bool adjoint, bool enableLightSampling,
           Ray &ray, Vec3f &throughput, Vec3f &emission, bool &wasSpecular, bool directLightingDone)
{
    wasSpecular = !enableLightSampling;

    if (!directLightingDone && !adjoint && enableLightSampling && bounce < _settings.maxBounces - 1) {
        Medium::MediumState mediumStateBounceIncr = mediumState;
        mediumStateBounceIncr.info.pixelSampleSegment.w() += 1;
        emission += throughput*volumeEstimateDirect(sampler, mediumSample, &mediumStateBounceIncr, medium, bounce + 1, ray);
    }

    PhaseSample phaseSample;
    if (!mediumSample.phase->sample(sampler, ray.dir(), mediumSample, phaseSample)) {
        std::cout << "Wrong normal direction in sampling phase function!\n";
        return false;
    }

    if (phaseSample.pathContext) mediumState.pathContext = phaseSample.pathContext;
    ray = ray.scatter(mediumSample.p, phaseSample.w, 0.f);
    ray.setPrimaryRay(false);
    throughput *= phaseSample.weight;

    return true;
}

void TraceBase::beginSurface(SurfaceState &surface, SurfaceScatterEvent &event,
        const Ray &ray, Vec3f &throughput) const
{
    Vec3f transparency = event.info->bsdf->eval(event.makeForwardEvent(), false);
    float transparencyScalar = transparency.avg();
    surface.forward = event.sampler->nextBoolean(transparencyScalar);
    if (surface.forward) {
        surface.wo = ray.dir();
        event.pdf = transparencyScalar;
        event.weight = transparency/transparencyScalar;
        event.sampledLobe = BsdfLobes::ForwardLobe;
        throughput *= event.weight;
    }
}

bool TraceBase::finishSurface(SurfaceState &surface, SurfaceScatterEvent &event,
        IntersectionTemporary &data, IntersectionInfo &info, const Medium *&medium,
        int bounce, bool adjoint, bool enableLightSampling, Ray &ray,
        Vec3f &throughput, Vec3f &emission, bool &wasSpecular,
        Medium::MediumState &mediumState) const
{
    if (!surface.forward) {
        if (!adjoint && info.primitive->isEmissive() && bounce >= _settings.minBounces) {
            if (!enableLightSampling || wasSpecular || !info.primitive->isSamplable())
                emission += info.primitive->evalDirect(data, info)*throughput;
        }
        event.requestedLobe = BsdfLobes::AllLobes;
        if (!info.bsdf->sample(event, adjoint))
            return false;
        surface.wo = event.frame.toGlobal(event.wo);
        if (!isConsistent(event, surface.wo))
            return false;
        throughput *= event.weight;
        wasSpecular = event.sampledLobe.hasSpecular();
        if (!wasSpecular)
            ray.setPrimaryRay(false);
    }
    Vec3f wi = event.frame.toGlobal(event.wi);
    bool geometricBackside = (surface.wo.dot(info.Ng) < 0.0f);
    const Medium *previousMedium = medium;
    medium = info.primitive->selectMedium(medium, geometricBackside);
    if (mediumState.pathContext ? medium != previousMedium : geometricBackside != (wi.dot(info.Ng) < 0.0f))
        mediumState.reset();
    ray = ray.scatter(ray.hitpoint(), surface.wo, info.epsilon);
    return true;
}

bool TraceBase::handleSurface(SurfaceScatterEvent &event, IntersectionTemporary &data,
        IntersectionInfo &info, const Medium *&medium, int bounce, bool adjoint,
        bool enableLightSampling, Ray &ray, Vec3f &throughput, Vec3f &emission,
        bool &wasSpecular, Medium::MediumState &mediumState, Vec3f *transmittance)
{
    SurfaceState surface;
    beginSurface(surface, event, ray, throughput);
    if (!surface.forward && !adjoint && enableLightSampling && bounce < _settings.maxBounces - 1) {
        Medium::MediumState mediumStateBounceIncr = mediumState;
        mediumStateBounceIncr.info.pixelSampleSegment.w() += 1;
        emission += estimateDirect(event, &mediumStateBounceIncr, medium, bounce + 1,
            ray, transmittance)*throughput;
    }
    return finishSurface(surface, event, data, info, medium, bounce, adjoint,
        enableLightSampling, ray, throughput, emission, wasSpecular, mediumState);
}

void TraceBase::handleInfiniteLights(IntersectionTemporary &data,
        IntersectionInfo &info, bool enableLightSampling, Ray &ray,
        Vec3f throughput, bool wasSpecular, Vec3f &emission)
{
    if (_scene->intersectInfinites(ray, data, info)) {
        if (!enableLightSampling || wasSpecular || !info.primitive->isSamplable())
            emission += throughput*info.primitive->evalDirect(data, info);
    }
}

}
