#include "PathTracer.hpp"

#include "bsdfs/TransparencyBsdf.hpp"

#include <fstream>
#include <nlohmann/json.hpp>

namespace Tungsten {

PathTracer::PathTracer(TraceableScene *scene, const PathTracerSettings &settings, uint32 threadId,
                       GpRayBatchBackend *gpRayBatchBackend)
: TraceBase(scene, settings, threadId),
  _settings(settings),
  _trackOutputValues(!scene->rendererSettings().renderOutputs().empty()),
  _gpRayBatchBackend(gpRayBatchBackend)
{
}

Vec3f PathTracer::traceSample(Vec2u pixel, PathSampleGenerator &sampler, uint spp)
{
    std::shared_ptr<DumpData> gpDumpDataPointer;
    if (pixel == _scene->rendererSettings().dataDumpPixel()) {
        gpDumpDataPointer = std::make_shared<DumpData>();
    }

    // TODO: Put diagnostic colors in JSON?
    const Vec3f nanDirColor = Vec3f(0.0f);
    const Vec3f nanEnvDirColor = Vec3f(0.0f);
    const Vec3f nanBsdfColor = Vec3f(0.0f);

    try {

    PositionSample point;
    if (!_scene->cam().samplePosition(sampler, point))
        return Vec3f(0.0f);
    DirectionSample direction;
    if (!_scene->cam().sampleDirection(sampler, point, pixel, direction))
        return Vec3f(0.0f);

    Vec3f throughput = point.weight*direction.weight;
    Ray ray(point.p, direction.d);
    ray.setPrimaryRay(true);

    MediumSample mediumSample;
    SurfaceScatterEvent surfaceEvent;
    IntersectionTemporary data;
    Medium::MediumState state;
    state.info.pixelSampleSegment = Vec4u(pixel.x(), pixel.y(), spp, 0u);
    state.info.sceneSeed = _scene->_seed;
    state.info.t = 0;
    state.reset();
    state.gpRayBatchBackend = _gpRayBatchBackend;
    IntersectionInfo info;
    Vec3f emission(0.0f);
    const Medium *medium = _scene->cam().medium().get();

    bool recordedOutputValues = false;
    float hitDistance = 0.0f;

    int mediumBounces = 0;
    int bounce = 0;
    bool didHit = intersectScene(ray, medium, data, info);
    bool wasSpecular = true;

    while ((didHit || medium) && bounce < _settings.maxBounces) {
        bool hitSurface = true;
        state.info.pixelSampleSegment.w() = bounce;
        if (medium) {
            mediumSample.continuedWeight = throughput;

            bool succeed = medium->sampleDistance(sampler, ray, state, mediumSample);
            if (!succeed)
                return emission;

            emission += throughput*mediumSample.emission;
            throughput *= mediumSample.weight;
            hitSurface = mediumSample.exited;
            if (hitSurface && !didHit)
                break;

            if (abs(mediumSample.t - ray.farT()) < 0.0000001f && !mediumSample.exited) {
                std::cerr << "Reached end of ray, but not counting it as exited...\n";
            }
        }

        if (hitSurface) {
            hitDistance += ray.farT();

            if (mediumBounces == 1 && !_settings.lowOrderScattering)
                return emission;

            surfaceEvent = makeLocalScatterEvent(data, info, ray, &sampler);
            Vec3f transmittance(-1.0f);

            bool terminate = !handleSurface(surfaceEvent, data, info, medium, bounce, false,
                    _settings.enableLightSampling && (mediumBounces > 0 || _settings.includeSurfaces), ray, throughput, emission, wasSpecular, state, &transmittance);

            if (_firstSurfaceBounceCb) {
                if (!_firstSurfaceBounceCb(surfaceEvent, ray)) {
                    return Vec3f(0.f);
                }
            }

            if (!info.bsdf->lobes().isPureDirac())
                if (mediumBounces == 0 && !_settings.includeSurfaces)
                    return emission;

            if (_trackOutputValues && !recordedOutputValues && (!wasSpecular || terminate)) {
                if (_scene->cam().depthBuffer())
                    _scene->cam().depthBuffer()->addSample(pixel, hitDistance);
                if (_scene->cam().normalBuffer())
                    _scene->cam().normalBuffer()->addSample(pixel, info.Ns);
                if (_scene->cam().albedoBuffer()) {
                    Vec3f albedo;
                    if (const TransparencyBsdf *bsdf = dynamic_cast<const TransparencyBsdf *>(info.bsdf))
                        albedo = (*bsdf->base()->albedo())[info];
                    else
                        albedo = (*info.bsdf->albedo())[info];
                    if (info.primitive->isEmissive())
                        albedo += info.primitive->evalDirect(data, info);
                    _scene->cam().albedoBuffer()->addSample(pixel, albedo);
                }
                if (_scene->cam().visibilityBuffer() && transmittance != -1.0f)
                    _scene->cam().visibilityBuffer()->addSample(pixel, transmittance.avg());
                recordedOutputValues = true;
            }

            if (terminate)
                return emission;
        } else {
            mediumBounces++;

            if (!handleVolume(sampler, mediumSample, state, medium, bounce, false,
                _settings.enableVolumeLightSampling && (mediumBounces > 1 || _settings.lowOrderScattering), ray, throughput, emission, wasSpecular))
                return emission;

            if (_firstMediumBounceCb) {
                if (!_firstMediumBounceCb(mediumSample, ray)) {
                    return Vec3f(0.f);
                }
            }
        }

        if (throughput.max() == 0.0f)
            break;

        float roulettePdf = std::abs(throughput).max();
        if (bounce > 2 && roulettePdf < 0.1f) {
            if (sampler.nextBoolean(roulettePdf))
                throughput /= roulettePdf;
            else
                return emission;
        }

        if (std::isnan(ray.dir().sum() + ray.pos().sum()))
            return nanDirColor;
        if (std::isnan(throughput.sum() + emission.sum()))
            return nanBsdfColor;

        if (!(hitSurface && info.bsdf->lobes().isForward())) {
            bounce++;
        }

        if (bounce < _settings.maxBounces)
            didHit = intersectScene(ray, medium, data, info);

        if (ray.farT() == 0.f && !(didHit && info.bsdf->lobes().isForward())) {
            std::cerr << "Path tracing bounces ended with farT == 0. Was surface event? " << (hitSurface ? "yes" : "no") << "\n";
        }
    }
    if (bounce >= _settings.minBounces && bounce < _settings.maxBounces) {
        handleInfiniteLights(data, info, _settings.enableLightSampling, ray, throughput, wasSpecular, emission);
        if (_firstMediumBounceCb) {
            mediumSample.p = ray.pos() + ray.dir() * 10;
            if (!_firstMediumBounceCb(mediumSample, ray.scatter(mediumSample.p, ray.dir(), 0))) {
                return Vec3f(0.f);
            }
        }

        if (_firstSurfaceBounceCb) {
            if (!_firstSurfaceBounceCb(surfaceEvent, ray.scatter(ray.pos() + ray.dir() * 10, ray.dir(), 0))) {
                return Vec3f(0.f);
            }
        }
    }
    if (std::isnan(throughput.sum() + emission.sum()))
        return nanEnvDirColor;

    if (_trackOutputValues && !recordedOutputValues) {
        if (_scene->cam().depthBuffer() && bounce == 0)
            _scene->cam().depthBuffer()->addSample(pixel, 0.0f);
        if (_scene->cam().normalBuffer())
            _scene->cam().normalBuffer()->addSample(pixel, -ray.dir());
        if (_scene->cam().albedoBuffer() && info.primitive && info.primitive->isInfinite())
            _scene->cam().albedoBuffer()->addSample(pixel, info.primitive->evalDirect(data, info));
    }

    return emission;

    } catch (std::runtime_error &e) {
        std::cout << tfm::format("Caught an internal error at pixel %s: %s", pixel, e.what()) << std::endl;

        return Vec3f(0.0f);
    }
}

void PathTracer::resumeGpis(PathState &path, const GpRaySegmentReply &reply)
{
    path.reply = reply;
    path.replyReady = true;
}

void PathTracer::commitOutput(Vec2u pixel, const OutputSample &output)
{
    if (output.depthValid && _scene->cam().depthBuffer())
        _scene->cam().depthBuffer()->addSample(pixel, output.depth);
    if (output.normalValid && _scene->cam().normalBuffer())
        _scene->cam().normalBuffer()->addSample(pixel, output.normal);
    if (output.albedoValid && _scene->cam().albedoBuffer())
        _scene->cam().albedoBuffer()->addSample(pixel, output.albedo);
    if (output.visibilityValid && _scene->cam().visibilityBuffer())
        _scene->cam().visibilityBuffer()->addSample(pixel, output.visibility);
}

PathTracer::SampleStatus PathTracer::advanceSample(PathState &path, PathSampleGenerator &sampler)
{
    auto finish = [&path](Vec3f value) {
        path.emission = value;
        path.stage = PathState::Stage::Finished;
        return SampleStatus::Finished;
    };
    if (path.stage == PathState::Stage::Finished)
        return SampleStatus::Finished;
    const auto pixel = path.pixel;
    const auto spp = path.spp;
    auto recordVisibility = [&](float value) {
        if (path.deferOutputValues) {
            path.output.visibilityValid = true;
            path.output.visibility = value;
        } else
            _scene->cam().visibilityBuffer()->addSample(pixel, value);
    };
    auto recordAlbedo = [&](Vec3f value) {
        if (path.deferOutputValues) {
            path.output.albedoValid = true;
            path.output.albedo = value;
        } else
            _scene->cam().albedoBuffer()->addSample(pixel, value);
    };
    auto recordNormal = [&](Vec3f value) {
        if (path.deferOutputValues) {
            path.output.normalValid = true;
            path.output.normal = value;
        } else
            _scene->cam().normalBuffer()->addSample(pixel, value);
    };
    auto recordDepth = [&](float value) {
        if (path.deferOutputValues) {
            path.output.depthValid = true;
            path.output.depth = value;
        } else
            _scene->cam().depthBuffer()->addSample(pixel, value);
    };
    auto &throughput = path.throughput;
    auto &ray = path.ray;
    auto &mediumSample = path.mediumSample;
    auto &surfaceEvent = path.surfaceEvent;
    auto &data = path.data;
    auto &state = path.state;
    auto &info = path.info;
    auto &emission = path.emission;
    auto &medium = path.medium;
    auto &recordedOutputValues = path.recordedOutputValues;
    auto &hitDistance = path.hitDistance;
    auto &mediumBounces = path.mediumBounces;
    auto &bounce = path.bounce;
    auto &didHit = path.didHit;
    auto &wasSpecular = path.wasSpecular;
    try {
    if (path.stage == PathState::Stage::Start) {
        PositionSample point;
        if (!_scene->cam().samplePosition(sampler, point))
            return finish(Vec3f(0.0f));
        DirectionSample direction;
        if (!_scene->cam().sampleDirection(sampler, point, pixel, direction))
            return finish(Vec3f(0.0f));

        throughput = point.weight*direction.weight;
        ray = Ray(point.p, direction.d);
        ray.setPrimaryRay(true);

        state.info.pixelSampleSegment = Vec4u(pixel.x(), pixel.y(), spp, 0u);
        state.info.sceneSeed = _scene->_seed;
        state.info.t = 0;
        state.reset();
        state.gpRayBatchBackend = nullptr; // GP requests are submitted by the coordinator.
        medium = _scene->cam().medium().get();


        didHit = intersectScene(ray, medium, data, info);

        path.stage = PathState::Stage::Bounce;
    }
    while ((didHit || medium) && bounce < _settings.maxBounces) {
        auto &hitSurface = path.hitSurface;
        state.info.pixelSampleSegment.w() = bounce;
        if (path.stage == PathState::Stage::Bounce ||
            path.stage == PathState::Stage::Distance || path.stage == PathState::Stage::Waiting) {
            hitSurface = true;
            if (medium) {
                using DistanceStatus = GaussianProcessMedium::DistanceStatus;
                if (path.stage == PathState::Stage::Bounce) {
                    mediumSample.continuedWeight = throughput;
                    path.sparse = dynamic_cast<const SparseConvolutionNoiseMedium *>(medium);
                    Se1DGpBatchConfig config;
                    if (path.sparse && path.sparse->se1DGpuBatchConfig(config)) {
                        auto status = path.sparse->beginDistance(sampler, ray, state,
                            mediumSample, path.distance);
                        if (status == DistanceStatus::Failed)
                            return finish(emission);
                        if (status == DistanceStatus::NeedIntersection)
                            path.stage = PathState::Stage::Distance;
                    } else {
                        path.sparse = nullptr;
                        if (!medium->sampleDistance(sampler, ray, state, mediumSample))
                            return finish(emission);
                    }
                }
                if (path.stage == PathState::Stage::Waiting) {
                    if (!path.replyReady)
                        return SampleStatus::NeedGpis;
                    double t = path.distance.t;
                    bool hit = path.sparse->finishGpisSegment(path.distance.ray, state,
                        path.prepared, path.reply, t);
                    path.replyReady = false;
                    auto status = path.sparse->resumeDistance(sampler, ray, state,
                        mediumSample, path.distance, t, hit);
                    path.prepared.context.reset();
                    if (status == DistanceStatus::Failed)
                        return finish(emission);
                    path.stage = status == DistanceStatus::NeedIntersection ?
                        PathState::Stage::Distance : PathState::Stage::Bounce;
                }
                if (path.stage == PathState::Stage::Distance) {
                    path.sparse->prepareGpisSegment(sampler, path.distance.ray, state, path.prepared);
                    path.stage = PathState::Stage::Waiting;
                    return SampleStatus::NeedGpis;
                }

                emission += throughput*mediumSample.emission;
                throughput *= mediumSample.weight;
                hitSurface = mediumSample.exited;
                if (hitSurface && !didHit)
                    break;

                if (abs(mediumSample.t - ray.farT()) < 0.0000001f && !mediumSample.exited) {
                    std::cerr << "Reached end of ray, but not counting it as exited...\n";
                }
            }

            path.stage = PathState::Stage::Shade;
        }

        if (path.stage == PathState::Stage::Shade) {
            path.transmittance = Vec3f(-1.0f);
            if (hitSurface) {
                hitDistance += ray.farT();
                if (mediumBounces == 1 && !_settings.lowOrderScattering)
                    return finish(emission);
                surfaceEvent = makeLocalScatterEvent(data, info, ray, &sampler);
                beginSurface(path.surface, surfaceEvent, ray, throughput);
                path.enableDirect = _settings.enableLightSampling &&
                    (mediumBounces > 0 || _settings.includeSurfaces);
            } else {
                mediumBounces++;
                path.enableDirect = _settings.enableVolumeLightSampling &&
                    (mediumBounces > 1 || _settings.lowOrderScattering);
            }
            if (path.enableDirect && bounce < _settings.maxBounces - 1 &&
                (!hitSurface || !path.surface.forward)) {
                beginDirectLighting(path.direct, sampler, hitSurface ? info.p : mediumSample.p,
                    state, bounce + 1, hitSurface);
                path.stage = PathState::Stage::Direct;
            } else
                path.stage = PathState::Stage::Scatter;
        }
        if (path.stage == PathState::Stage::Direct) {
            if (!advanceDirectLighting(path.direct, sampler, mediumSample, surfaceEvent,
                medium, ray, path.prepared, path.reply, path.replyReady))
                return SampleStatus::NeedGpis;
            if (hitSurface) {
                emission += path.direct.result*throughput;
                path.transmittance = path.direct.visibility;
            } else
                emission += throughput*path.direct.result;
            path.stage = PathState::Stage::Scatter;
        }

        if (hitSurface) {
            bool terminate = !finishSurface(path.surface, surfaceEvent, data, info, medium,
                bounce, false, path.enableDirect, ray, throughput, emission, wasSpecular, state);
            if (_firstSurfaceBounceCb) {
                if (!_firstSurfaceBounceCb(surfaceEvent, ray)) {
                    return finish(Vec3f(0.f));
                }
            }

            if (!info.bsdf->lobes().isPureDirac())
                if (mediumBounces == 0 && !_settings.includeSurfaces)
                    return finish(emission);

            if (_trackOutputValues && !recordedOutputValues && (!wasSpecular || terminate)) {
                if (_scene->cam().depthBuffer())
                    recordDepth(hitDistance);
                if (_scene->cam().normalBuffer())
                    recordNormal(info.Ns);
                if (_scene->cam().albedoBuffer()) {
                    Vec3f albedo;
                    if (const TransparencyBsdf *bsdf = dynamic_cast<const TransparencyBsdf *>(info.bsdf))
                        albedo = (*bsdf->base()->albedo())[info];
                    else
                        albedo = (*info.bsdf->albedo())[info];
                    if (info.primitive->isEmissive())
                        albedo += info.primitive->evalDirect(data, info);
                    recordAlbedo(albedo);
                }
                if (_scene->cam().visibilityBuffer() && path.transmittance != -1.0f)
                    recordVisibility(path.transmittance.avg());
                recordedOutputValues = true;
            }

            if (terminate)
                return finish(emission);
        } else {
            if (!handleVolume(sampler, mediumSample, state, medium, bounce, false,
                path.enableDirect, ray, throughput, emission, wasSpecular, true))
                return finish(emission);

            if (_firstMediumBounceCb) {
                if (!_firstMediumBounceCb(mediumSample, ray)) {
                    return finish(Vec3f(0.f));
                }
            }
        }

        if (throughput.max() == 0.0f)
            break;

        float roulettePdf = std::abs(throughput).max();
        if (bounce > 2 && roulettePdf < 0.1f) {
            if (sampler.nextBoolean(roulettePdf))
                throughput /= roulettePdf;
            else
                return finish(emission);
        }

        if (std::isnan(ray.dir().sum() + ray.pos().sum()))
            return finish(Vec3f(0.f));
        if (std::isnan(throughput.sum() + emission.sum()))
            return finish(Vec3f(0.f));

        if (!(hitSurface && info.bsdf->lobes().isForward())) {
            bounce++;
        }

        if (bounce < _settings.maxBounces)
            didHit = intersectScene(ray, medium, data, info);
        path.stage = PathState::Stage::Bounce;

        if (ray.farT() == 0.f && !(didHit && info.bsdf->lobes().isForward())) {
            std::cerr << "Path tracing bounces ended with farT == 0. Was surface event? " << (hitSurface ? "yes" : "no") << "\n";
        }
    }
    if (bounce >= _settings.minBounces && bounce < _settings.maxBounces) {
        handleInfiniteLights(data, info, _settings.enableLightSampling, ray, throughput, wasSpecular, emission);
        if (_firstMediumBounceCb) {
            mediumSample.p = ray.pos() + ray.dir() * 10;
            if (!_firstMediumBounceCb(mediumSample, ray.scatter(mediumSample.p, ray.dir(), 0))) {
                return finish(Vec3f(0.f));
            }
        }

        if (_firstSurfaceBounceCb) {
            if (!_firstSurfaceBounceCb(surfaceEvent, ray.scatter(ray.pos() + ray.dir() * 10, ray.dir(), 0))) {
                return finish(Vec3f(0.f));
            }
        }
    }
    if (std::isnan(throughput.sum() + emission.sum()))
        return finish(Vec3f(0.f));

    if (_trackOutputValues && !recordedOutputValues) {
        if (_scene->cam().depthBuffer() && bounce == 0)
            recordDepth(0.0f);
        if (_scene->cam().normalBuffer())
            recordNormal(-ray.dir());
        if (_scene->cam().albedoBuffer() && info.primitive && info.primitive->isInfinite())
            recordAlbedo(info.primitive->evalDirect(data, info));
    }

    return finish(emission);

    } catch (std::runtime_error &e) {
        std::cout << tfm::format("Caught an internal error at pixel %s: %s", pixel, e.what()) << std::endl;
        return finish(Vec3f(0.0f));
    }
}

}
