#include "PathTracerSeBatchBackend.hpp"
#include "integrators/path_tracer/PathTraceIntegrator.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/TextureCache.hpp"
#include "io/FileUtils.hpp"
#include "io/DirectoryChange.hpp"
#include "media/GpRayBatchBackend.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"
#include "renderer/TraceableScene.hpp"
#include "primitives/Cube.hpp"
#include "sampling/UniformPathSampler.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "thread/ThreadUtils.hpp"

#include <openvdb/openvdb.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <future>
#include <iostream>
#include <mutex>
#include <stdexcept>

using namespace Tungsten;

namespace {
void require(bool condition, const char *message)
{
    if (!condition)
        throw std::runtime_error(message);
}

class RejectingBackend final : public GpRayBatchBackend {
public:
    std::size_t batches = 0, largest = 0, pathSegments = 0, shadowSegments = 0;
    bool gate = false, gateShadowOnly = false;
    std::mutex mutex;
    std::condition_variable ready;
    struct Pending {
        std::unique_ptr<std::promise<std::vector<GpRaySegmentReply>>> promise;
        std::size_t count;
        std::function<void()> notification;
    };
    std::vector<Pending> pending;

    bool trace(const GpRaySegmentRequest &, GpRaySegmentResult &) override
    {
        throw std::runtime_error("wavefront worker called blocking trace()");
    }

    std::future<std::vector<GpRaySegmentReply>> submitBatch(
        const std::vector<GpRaySegmentRequest> &requests,
        std::function<void()> onComplete = {}) override
    {
        std::lock_guard<std::mutex> lock(mutex);
        ++batches;
        largest = std::max(largest, requests.size());
        bool hasShadow = false;
        for (const auto &request : requests) {
            if (request.purpose == GpRaySegmentRequest::Purpose::Shadow) {
                ++shadowSegments;
                hasShadow = true;
            } else
                ++pathSegments;
        }
        auto promise = std::make_unique<std::promise<std::vector<GpRaySegmentReply>>>();
        auto future = promise->get_future();
        if (gate && (!gateShadowOnly || hasShadow)) {
            pending.push_back({std::move(promise), requests.size(), std::move(onComplete)});
            ready.notify_one();
        } else {
            promise->set_value(std::vector<GpRaySegmentReply>(requests.size()));
            if (onComplete)
                onComplete();
        }
        return future;
    }

    void waitForSubmission(std::size_t count = 1)
    {
        std::unique_lock<std::mutex> lock(mutex);
        require(ready.wait_for(lock, std::chrono::seconds(10), [this, count] { return pending.size() >= count; }),
                "no asynchronous GPIS submission arrived");
    }

    void releaseReply()
    {
        std::lock_guard<std::mutex> lock(mutex);
        gate = false;
        for (auto it = pending.rbegin(); it != pending.rend(); ++it) {
            it->promise->set_value(std::vector<GpRaySegmentReply>(it->count));
            it->promise.reset();
        }
    }

    void notifyComplete()
    {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = pending.rbegin(); it != pending.rend(); ++it)
            if (it->notification)
                it->notification();
        pending.clear();
    }
};

std::shared_ptr<RejectingBackend> testBackend;
std::shared_ptr<GpRayBatchBackend> testFactory(const TraceableScene &) { return testBackend; }

std::unique_ptr<Scene> loadScene(const Path &path, bool wavefront, bool backend,
                               Vec2u resolution, uint32 spp, bool sobol, uint32 sppStep = 1,
                               int variant = 0, bool pathPool = false,
                               bool independent = false, uint32 poolSize = 4096)
{
    std::unique_ptr<Scene> scene;
    if (variant) {
        auto fixture = nlohmann::json::parse(FileUtils::loadText(path));
        fixture["camera"]["medium"] = "gp";
        fixture["camera"]["transform"]["position"] = {0.0, -0.04, 0.04};
        fixture["camera"]["transform"]["look_at"] = {0.0, 0.0, 0.04};
        fixture["media"][0]["phase_function"] = {{"type", "isotropic"}};
        if (variant == 2) {
            fixture["primitives"].push_back({{"type", "cube"},
                {"bsdf", {{"type", "lambert"}, {"albedo", 0.7}}},
                {"transform", {{"position", {0.0, -0.02, 0.04}}, {"scale", 0.025}}}});
        }
        JsonDocument document(path, fixture.dump());
        DirectoryChange context(path.parent());
        scene.reset(new Scene(path.parent(), std::make_shared<TextureCache>()));
        scene->fromJson(document, *scene);
        scene->setPath(path);
    } else
        scene.reset(Scene::load(path));
    scene->camera()->setResolution(resolution);
    auto &renderer = scene->rendererSettings();
    renderer.setSpp(spp);
    renderer.setSppStep(sppStep);
    nlohmann::json options = {{"type", "path_tracer"}, {"max_bounces", variant ? 4 : 2},
        {"use_cuda_gpis", backend}, {"wavefront_gpis", wavefront},
        {"gpis_path_pool", pathPool}, {"independent_path_sampling", independent},
        {"gpis_max_active_paths", poolSize}, {"gpis_batch_size", 64},
        {"gpis_inflight_batches", 4}};
    JsonDocument settings(path, options.dump());
    scene->integrator()->fromJson(settings, *scene);
    // Preserve the other renderer options while changing the sampler.
    JsonDocument sampling(path, std::string("{\"stratified_sampler\":") + (sobol ? "true}" : "false}"));
    renderer.fromJson(sampling, *scene);
    scene->loadResources();
    return scene;
}

std::vector<Vec3f> render(const Path &path, bool wavefront, bool backend,
                         Vec2u resolution, uint32 spp, bool sobol, uint32 sppStep = 1,
                         int variant = 0, bool pathPool = false,
                         bool independent = false, uint32 poolSize = 4096, bool aovs = false,
                         bool adaptive = false)
{
    auto scene = loadScene(path, wavefront, backend, resolution, spp, sobol, sppStep, variant,
        pathPool, independent, poolSize);
    if (adaptive) {
        JsonDocument sampling(path, R"({"adaptive_sampling":true})");
        scene->rendererSettings().fromJson(sampling, *scene);
    }
    if (aovs) {
        JsonDocument outputs(path, R"({"output_buffers":[{"type":"depth"},{"type":"normal"},{"type":"albedo"},{"type":"visibility"}]})");
        scene->rendererSettings().fromJson(outputs, *scene);
    }
    std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(12345u));
    while (!traceable->integrator().done()) {
        traceable->integrator().startRender([] {});
        traceable->integrator().waitForCompletion();
    }
    std::vector<Vec3f> image;
    for (uint32 i = 0; i < resolution.product(); ++i)
        image.push_back((*traceable->cam().colorBuffer())[i]);
    if (aovs) {
        for (uint32 i = 0; i < resolution.product(); ++i) {
            image.push_back(Vec3f((*traceable->cam().depthBuffer())[i]));
            image.push_back((*traceable->cam().normalBuffer())[i]);
            image.push_back((*traceable->cam().albedoBuffer())[i]);
            image.push_back(Vec3f((*traceable->cam().visibilityBuffer())[i]));
        }
    }
    return image;
}

class BoundaryProbe : public TraceBase {
public:
    explicit BoundaryProbe(TraceableScene *scene) : TraceBase(scene, TraceSettings(), 0) {}
    using TraceBase::intersectScene;
};

void verifyBoundaryExits(const Path &path)
{
    auto scene = loadScene(path, false, false, Vec2u(32, 32), 1, false);
    std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(12345u));
    BoundaryProbe probe(traceable.get());
    const auto *cube = dynamic_cast<const Cube *>(scene->findPrimitive("processBox"));
    require(cube != nullptr, "GP boundary fixture missing");
    const auto box = cube->bounds();
    const auto *medium = cube->intMedium().get();
    for (int axis = 0; axis < 3; ++axis) {
        for (int sign : {-1, 1}) {
            for (float distance : {0.0f, 0.000125f}) {
                Vec3f p = (box.min() + box.max())*0.5f;
                p[axis] = sign > 0 ? box.max()[axis] - distance : box.min()[axis] + distance;
                Vec3f direction(0.0f);
                direction[axis] = float(sign);
                Ray ray(p, direction, 0.0005f);
                IntersectionTemporary data;
                IntersectionInfo info;
                require(probe.intersectScene(ray, medium, data, info), "GP exit before nearT missed");
                require(info.primitive == cube && cube->hitBackside(data) &&
                        ray.farT() <= ray.nearT(), "recovered wrong GP boundary");
                require(info.Ng.dot(direction) > 0.0f &&
                        cube->selectMedium(medium, direction.dot(info.Ng) < 0.0f) == nullptr,
                        "GP exit normal did not select exterior");
                const auto *gp = dynamic_cast<const GaussianProcessMedium *>(medium);
                UniformPathSampler sampler(123);
                Medium::MediumState state;
                state.reset();
                MediumSample sample;
                GaussianProcessMedium::DistanceState distanceState;
                require(gp->beginDistance(sampler, ray, state, sample, distanceState) ==
                        GaussianProcessMedium::DistanceStatus::Finished && sample.exited,
                        "empty GP interval was submitted for marching");
                require(gp->transmittance(sampler, ray, false, false, &state) == Vec3f(1.0f),
                        "empty GP shadow interval was not transparent");
                Ray exterior(p, direction, 0.0005f);
                require(!probe.intersectScene(exterior, nullptr, data, info),
                        "boundary recovery changed exterior-ray semantics");
            }
            // A shadow origin can round one ULP outside a face, with nearT=0.
            Vec3f p = (box.min() + box.max())*0.5f;
            p[axis] = std::nextafter(sign > 0 ? box.max()[axis] : box.min()[axis],
                sign > 0 ? Ray::infinity() : -Ray::infinity());
            Vec3f direction(0.0f);
            direction[axis] = float(sign);
            Ray shadow(p, direction, 0.0f);
            IntersectionTemporary data;
            IntersectionInfo info;
            require(probe.intersectScene(shadow, medium, data, info) && shadow.farT() == 0.0f,
                    "rounded-outward GP shadow origin lost its boundary exit");
        }
    }
    // An inward component at a rounded-outward adjacent face must not replace
    // the exit face normal. Otherwise a zero-length forward event re-enters GP.
    for (int exitAxis = 0; exitAxis < 3; ++exitAxis)
        for (int adjacentAxis = 0; adjacentAxis < 3; ++adjacentAxis) {
            if (exitAxis == adjacentAxis)
                continue;
            for (int exitSign : {-1, 1})
                for (int adjacentSign : {-1, 1}) {
                    Vec3f p = (box.min() + box.max())*0.5f;
                    p[exitAxis] = exitSign > 0 ? box.max()[exitAxis] : box.min()[exitAxis];
                    p[adjacentAxis] = std::nextafter(adjacentSign > 0 ? box.max()[adjacentAxis] :
                        box.min()[adjacentAxis], adjacentSign > 0 ? Ray::infinity() : -Ray::infinity());
                    Vec3f direction(0.0f);
                    direction[exitAxis] = float(exitSign);
                    direction[adjacentAxis] = float(-adjacentSign);
                    direction = direction.normalized();
                    Ray ray(p, direction, 0.0005f);
                    IntersectionTemporary data;
                    IntersectionInfo info;
                    require(probe.intersectScene(ray, medium, data, info) && cube->hitBackside(data),
                            "GP edge exit was not recovered");
                    require(info.Ng.dot(direction) > 0.0f &&
                            cube->selectMedium(medium, direction.dot(info.Ng) < 0.0f) == nullptr,
                            "GP edge exit used the adjacent entry-face normal");
                }
        }
    // Exact stalled ray captured from 512x512, sample index 15, pixel (80,455).
    Ray stalled(Vec3f(0.0475450344f, -0.00568206655f, -0.0035218522f),
                Vec3f(-0.191235453f, 0.872217834f, -0.450183421f), 0.0005f);
    IntersectionTemporary data;
    IntersectionInfo info;
    require(probe.intersectScene(stalled, medium, data, info) && stalled.farT() <= stalled.nearT() &&
            info.Ng.dot(stalled.dir()) > 0.0f &&
            cube->selectMedium(medium, stalled.dir().dot(info.Ng) < 0.0f) == nullptr,
            "32 spp stalled boundary ray did not exit its medium");
    std::cout << "GP face/edge exits and the 32 spp stalled ray recovered\n";
}

void verifyCudaSubmission(const Path &path)
{
    auto scene = loadScene(path, false, false, Vec2u(32, 32), 1, false);
    JsonDocument original(path);
    auto second = std::make_shared<SparseConvolutionNoiseMedium>();
    second->fromJson(original["media"][0U], *scene);
    second->loadResources();
    scene->media().push_back(second);
    std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(12345u));
    auto backend = createPathTracerSeBatchBackend(*traceable);
    if (!backend) {
        std::cout << "CUDA submission checks skipped: backend unavailable\n";
        return;
    }
    std::vector<GpRaySegmentRequest> requests(6600);
    GpRaySegmentResult expected[2];
    for (std::size_t i = 0; i < requests.size(); ++i) {
        auto &r = requests[i];
        r.medium = i % 3 == 2 ? nullptr : dynamic_cast<SparseConvolutionNoiseMedium *>(
            scene->media()[i % 3 == 0 ? 0 : scene->media().size()-1].get());
        r.origin = Vec3f(i % 3 == 0 ? 0.02f : -0.02f, 0.0f, 0.04f);
        r.direction = Vec3f(1.0f, 0.0f, 0.0f);
        r.farT = 1e-6f;
        r.jitter = 0.2f;
        r.noiseSeed = 123;
        r.normalSeed = 456;
        r.previousGradient = Vec3f(0.0f);
    }
    require(backend->trace(requests[0], expected[0]) && backend->trace(requests[1], expected[1]),
            "reference CUDA trace failed");
    require(backend->submitBatch({}).get().empty(), "empty submission failed");
    // Submit multiple waves without waiting; their owners outlive caller arrays.
    auto first = backend->submitBatch(requests);
    auto next = backend->submitBatch(requests);
    requests.clear();
    for (auto *future : {&first, &next}) {
        auto replies = future->get();
        require(replies.size() == 6600, "reply count mismatch");
        for (std::size_t i = 0; i < replies.size(); ++i) {
            require(replies[i].valid == (i % 3 != 2), "reply order or rejection mismatch");
            if (!replies[i].valid)
                continue;
            const auto &r = replies[i].result;
            const auto &e = expected[i % 3];
            require(r.t == e.t && r.value == e.value && r.hit == e.hit, "CUDA reply reordered");
        }
    }
}
}

int main(int argc, char **argv)
{
    if (argc != 2)
        return 1;
    try {
        openvdb::initialize();
        EmbreeUtil::initDevice();
        ThreadUtils::startThreads(2);
        setPathTracerGpBatchFactory(testFactory);
        const Path path(argv[1]);
        struct Case { Vec2u resolution; uint32 spp, step; bool sobol; };
        for (const auto &test : {Case{Vec2u(32, 32), 1, 1, false},
                                 Case{Vec2u(17, 13), 2, 2, false},
                                 Case{Vec2u(17, 13), 2, 1, true}}) {
            const auto resolution = test.resolution;
            const auto spp = test.spp;
            const auto sobol = test.sobol;
            auto reference = render(path, false, false, resolution, spp, sobol, test.step);
            auto cpu = render(path, true, false, resolution, spp, sobol, test.step);
            testBackend = std::make_shared<RejectingBackend>();
            auto fallback = render(path, true, true, resolution, spp, sobol, test.step);
            require(reference == cpu, "CPU state-machine output differs from legacy traceSample");
            require(reference == fallback, "prepared CPU fallback changed output or RNG");
            require(testBackend->batches > 0, "no GPIS suspension exercised");
            require(testBackend->shadowSegments > 0, "shadow GPIS requests were not submitted asynchronously");
            if (resolution == Vec2u(32, 32))
                require(testBackend->largest > 2, "batching still limited to CPU worker count");
            std::cout << "Exact CPU/scheduler/fallback parity: " << resolution << ", " << spp
                      << " spp, Sobol=" << sobol << ", largest=" << testBackend->largest << '\n';
        }

        // Non-specular phase lighting exercises both light and phase directions.
        // The diffuse cube also exercises surface light/BSDF sampling in GP media.
        for (int variant : {1, 2}) {
            auto reference = render(path, false, false, Vec2u(17, 13), 2, false, 2, variant);
            auto cpu = render(path, true, false, Vec2u(17, 13), 2, false, 2, variant);
            testBackend = std::make_shared<RejectingBackend>();
            auto fallback = render(path, true, true, Vec2u(17, 13), 2, false, 2, variant);
            require(reference == cpu && reference == fallback,
                    "non-specular or surface shadow continuation changed output/RNG");
            require(testBackend->shadowSegments > 0 && testBackend->pathSegments > 0,
                    "mixed path/shadow submissions not exercised");
            std::cout << "Exact shadow fallback parity: variant=" << variant
                      << ", shadow segments=" << testBackend->shadowSegments << '\n';
        }

        for (bool sobol : {false, true}) {
            for (int variant : {0, 2}) {
                auto reference = render(path, false, false, Vec2u(17, 13), 3, sobol, 3,
                    variant, false, true, 512, true);
                auto cpu = render(path, true, false, Vec2u(17, 13), 3, sobol, 3,
                    variant, true, true, 512, true);
                testBackend = std::make_shared<RejectingBackend>();
                auto fallback = render(path, true, true, Vec2u(17, 13), 3, sobol, 3,
                    variant, true, true, 31, true);
                require(reference == cpu && reference == fallback,
                    "independent pool color/AOV output differs from blocking CPU or small-pool fallback");
                auto stepped = render(path, true, false, Vec2u(17, 13), 3, sobol, 1,
                    variant, true, true, 67, true);
                require(reference == stepped, "absolute sample indexing depends on spp_step");
                std::cout << "Independent pool color/AOV parity: Sobol=" << sobol
                          << ", variant=" << variant << '\n';
            }
        }
        // Keep four submissions outstanding and complete in reverse order.
        auto reference = render(path, false, false, Vec2u(32, 32), 1, false, 1, 0, false, true);
        testBackend = std::make_shared<RejectingBackend>();
        testBackend->gate = true;
        auto concurrentScene = loadScene(path, true, true, Vec2u(32, 32), 1, false, 1,
            0, true, true, 512);
        {
            std::unique_ptr<TraceableScene> traceable(concurrentScene->makeTraceable(12345u));
            traceable->integrator().startRender([] {});
            testBackend->waitForSubmission(4);
            testBackend->releaseReply();
            testBackend->notifyComplete();
            traceable->integrator().waitForCompletion();
            std::vector<Vec3f> image;
            for (uint32 i = 0; i < 32*32; ++i)
                image.push_back((*traceable->cam().colorBuffer())[i]);
            require(reference == image, "reverse completion reordered pool samples");
        }
        std::cout << "Four outstanding submissions, reverse completion passed\n";

        auto adaptiveReference = render(path, false, false, Vec2u(17, 13), 20, true, 4,
            2, false, true, 512, true, true);
        auto adaptivePool = render(path, true, false, Vec2u(17, 13), 20, true, 4,
            2, true, true, 31, true, true);
        require(adaptiveReference == adaptivePool, "adaptive pool statistics or absolute sample index changed");
        std::cout << "Adaptive 20 spp pool color/AOV parity passed\n";

        for (bool pathPool : {false, true})
        for (bool shadowOnly : {false, true}) {
            testBackend = std::make_shared<RejectingBackend>();
            testBackend->gate = true;
            testBackend->gateShadowOnly = shadowOnly;
            auto scene = loadScene(path, true, true, Vec2u(32, 32), 1, false, 1, shadowOnly ? 1 : 0,
                pathPool, pathPool, 512);
            std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(12345u));
            std::atomic<int> callbacks{0};
            traceable->integrator().startRender([&] { ++callbacks; });
            testBackend->waitForSubmission(pathPool ? 4 : 1);
            if (shadowOnly)
                require(testBackend->shadowSegments > 0, "abort did not gate a shadow submission");
            auto abort = std::async(std::launch::async, [&] { traceable->integrator().abortRender(); });
            require(abort.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
                    "abort freed pending submission before completion");
            testBackend->releaseReply();
            require(abort.wait_for(std::chrono::milliseconds(20)) == std::future_status::timeout,
                    "abort did not drain the asynchronous notification");
            testBackend->notifyComplete();
            abort.get();
            require(callbacks == 0, "aborted render invoked completion callback");
            require(traceable->integrator().currentSpp() == 0, "abort advanced completed spp");
        }
        testBackend.reset();
        verifyBoundaryExits(path);
        verifyCudaSubmission(path);
        std::cout << "Wavefront path tracer checks passed\n";
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
    return 0;
}
