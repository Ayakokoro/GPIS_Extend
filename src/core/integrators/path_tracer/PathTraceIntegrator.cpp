#include "PathTraceIntegrator.hpp"

#include "sampling/UniformPathSampler.hpp"
#include "sampling/SobolPathSampler.hpp"
#include "sampling/IndependentPathSampler.hpp"

#include <deque>
#include <map>
#include <unordered_map>

#include "cameras/Camera.hpp"

#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"

namespace Tungsten {

namespace {
PathTracerGpBatchFactory gpBatchFactory = nullptr;
}

void setPathTracerGpBatchFactory(PathTracerGpBatchFactory factory)
{
    gpBatchFactory = factory;
}

CONSTEXPR uint32 PathTraceIntegrator::TileSize;
CONSTEXPR uint32 PathTraceIntegrator::VarianceTileSize;
CONSTEXPR uint32 PathTraceIntegrator::AdaptiveThreshold;

PathTraceIntegrator::PathTraceIntegrator()
: Integrator(),
  _rect(Vec2u(), Vec2u()),
  _varianceW(0),
  _varianceH(0),
  _sampler(0xBA5EBA11)
{
}

PathTraceIntegrator::~PathTraceIntegrator()
{
    {
        std::lock_guard<std::mutex> lock(_wavefrontMutex);
        _abortWavefront = true;
    }
    _wavefrontReady.notify_one();
    if (_wavefrontThread.joinable())
        _wavefrontThread.join();
}

void PathTraceIntegrator::diceTiles()
{
    for (uint32 y = _rect.min().y(); y < _rect.max().y(); y += TileSize) {
        for (uint32 x = _rect.min().x(); x < _rect.max().x(); x += TileSize) {
            _tiles.emplace_back(
                x,
                y,
                min(TileSize, _rect.max().x() - x),
                min(TileSize, _rect.max().y() - y),
                _scene->rendererSettings().useSobol() ?
                    std::unique_ptr<PathSampleGenerator>(new SobolPathSampler(MathUtil::hash32(_sampler.nextI()))) :
                    std::unique_ptr<PathSampleGenerator>(new UniformPathSampler(MathUtil::hash32(_sampler.nextI())))
            );
        }
    }
}

float PathTraceIntegrator::errorPercentile95()
{
    std::vector<float> errors;
    errors.reserve(_samples.size());

    for (size_t i = 0; i < _samples.size(); ++i) {
        _samples[i].adaptiveWeight = _samples[i].errorEstimate();
        if (_samples[i].adaptiveWeight > 0.0f)
            errors.push_back(_samples[i].adaptiveWeight);
    }
    if (errors.empty())
        return 0.0f;
    std::sort(errors.begin(), errors.end());

    return errors[(errors.size()*95)/100];
}

void PathTraceIntegrator::dilateAdaptiveWeights()
{
    for (uint32 y = 0; y < _varianceH; ++y) {
        for (uint32 x = 0; x < _varianceW; ++x) {
            int idx = x + y*_varianceW;
            if (y < _varianceH - 1)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx + _varianceW].adaptiveWeight);
            if (x < _varianceW - 1)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx + 1].adaptiveWeight);
        }
    }
    for (int y = _varianceH - 1; y >= 0; --y) {
        for (int x = _varianceW - 1; x >= 0; --x) {
            int idx = x + y*_varianceW;
            if (y > 0)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx - _varianceW].adaptiveWeight);
            if (x > 0)
                _samples[idx].adaptiveWeight = max(_samples[idx].adaptiveWeight,
                        _samples[idx - 1].adaptiveWeight);
        }
    }
}

void PathTraceIntegrator::distributeAdaptiveSamples(int spp)
{
    double totalWeight = 0.0;
    for (SampleRecord &record : _samples)
        totalWeight += record.adaptiveWeight;

    int adaptiveBudget = (spp - 1)* _rect.max().x() * _rect.max().y();
    int budgetPerTile = adaptiveBudget/(VarianceTileSize*VarianceTileSize);
    float weightToSampleFactor = double(budgetPerTile)/totalWeight;

    float pixelPdf = 0.0f;
    for (SampleRecord &record : _samples) {
        float fractionalSamples = record.adaptiveWeight*weightToSampleFactor;
        int adaptiveSamples = int(fractionalSamples);
        pixelPdf += fractionalSamples - float(adaptiveSamples);
        if (_sampler.next1D() < pixelPdf) {
            adaptiveSamples++;
            pixelPdf -= 1.0f;
        }
        record.nextSampleCount = adaptiveSamples + 1;
    }
}

bool PathTraceIntegrator::generateWork()
{
    for (SampleRecord &record : _samples)
        record.sampleIndex += record.nextSampleCount;

    int sppCount = _nextSpp - _currentSpp;
    bool enableAdaptive = _scene->rendererSettings().useAdaptiveSampling();

    if (enableAdaptive && _currentSpp >= AdaptiveThreshold) {
        float maxError = errorPercentile95();
        if (maxError == 0.0f)
            return false;

        for (SampleRecord &record : _samples)
            record.adaptiveWeight = min(record.adaptiveWeight, maxError);

        dilateAdaptiveWeights();
        distributeAdaptiveSamples(sppCount);
    } else {
        for (SampleRecord &record : _samples)
            record.nextSampleCount = sppCount;
    }

    return true;
}

void PathTraceIntegrator::renderTile(uint32 id, uint32 tileId)
{
    ImageTile &tile = _tiles[tileId];
    IndependentPathSampler independent(_scene->_seed, _scene->rendererSettings().useSobol());
    PathSampleGenerator &sampler = _settings.independentPathSampling ?
        static_cast<PathSampleGenerator &>(independent) : *tile.sampler;
    for (uint32 y = 0; y < tile.h; ++y) {
        for (uint32 x = 0; x < tile.w; ++x) {
            Vec2u pixel(tile.x + x, tile.y + y);
            uint32 pixelIndex = pixel.x() + pixel.y()* _scene->cam().resolution().x();
            uint32 variancePixelIndex = pixel.x()/VarianceTileSize + pixel.y()/VarianceTileSize*_varianceW;

            SampleRecord &record = _samples[variancePixelIndex];
            int spp = record.nextSampleCount;
            for (uint32 i = 0; i < spp; ++i) {
                sampler.startPath(pixelIndex, record.sampleIndex + i);
                Vec3f c = _tracers[id]->traceSample(pixel, sampler,
                    _settings.independentPathSampling ? record.sampleIndex + i : _currentSpp + i);
                if (std::isinf(c) || c[0] > 65536 || c[1] > 65536 || c[2] > 65536) {
                    std::cerr << "Clamp one inf sample at pixel (" << pixel.x() << "," << pixel.y() << "): " << c << std::endl;
                    c = Vec3f(0.f);
                }
                else if (std::isnan(c)) {
                    std::cerr << "Clamp one nan sample at pixel (" << pixel.x() << "," << pixel.y() << ") " << c << std::endl;
                    c = Vec3f(0.f);
                }

                record.addSample(c);
                _scene->cam().colorBuffer()->addSample(pixel, c);
            }
        }
    }
}

void PathTraceIntegrator::renderWavefront()
{
    struct TileContinuation {
        uint32 pixelOffset = 0, sample = 0;
        std::unique_ptr<PathTracer::PathState> path;
    };
    std::vector<TileContinuation> continuations(_tiles.size());
    std::vector<std::unique_ptr<IndependentPathSampler>> samplers;
    for (std::size_t i = 0; i < _tiles.size(); ++i)
        samplers.emplace_back(new IndependentPathSampler(_scene->_seed,
            _scene->rendererSettings().useSobol()));
    std::vector<std::shared_ptr<TaskGroup>> jobs;
    std::vector<uint32> waiting, gpuTiles;
    std::size_t cpuLive = 0, finishedTiles = 0;
    std::size_t waves = 0, submittedRays = 0, largestWave = 0;
    std::size_t shadowRays = 0;
    bool gpuCompleted = false, gpuInFlight = false, firstBatch = true;
    std::exception_ptr cpuException;
    std::future<std::vector<GpRaySegmentReply>> submission;

    auto schedule = [&](const std::vector<uint32> &tileIds) {
        if (tileIds.empty())
            return;
        {
            std::lock_guard<std::mutex> lock(_wavefrontMutex);
            cpuLive += tileIds.size();
        }
        jobs.push_back(ThreadUtils::pool->enqueue(
            [&, tileIds](uint32 task, uint32, uint32 thread) {
                const uint32 tileId = tileIds[task];
                bool paused = false;
                std::exception_ptr error;
                try {
                    auto &tile = _tiles[tileId];
                    PathSampleGenerator &sampler = _settings.independentPathSampling ?
                        static_cast<PathSampleGenerator &>(*samplers[tileId]) : *tile.sampler;
                    auto &continuation = continuations[tileId];
                    while (continuation.pixelOffset < tile.w * tile.h && !_abortWavefront) {
                        Vec2u pixel(tile.x + continuation.pixelOffset % tile.w,
                                    tile.y + continuation.pixelOffset / tile.w);
                        auto &record = _samples[pixel.x()/VarianceTileSize +
                            pixel.y()/VarianceTileSize*_varianceW];
                        if (continuation.sample >= record.nextSampleCount) {
                            ++continuation.pixelOffset;
                            continuation.sample = 0;
                            continue;
                        }
                        if (!continuation.path) {
                            const uint32 pixelIndex = pixel.x() + pixel.y()*_scene->cam().resolution().x();
                            sampler.startPath(pixelIndex, record.sampleIndex + continuation.sample);
                            continuation.path.reset(new PathTracer::PathState(pixel,
                                _settings.independentPathSampling ? record.sampleIndex + continuation.sample :
                                    _currentSpp + continuation.sample));
                        }
                        auto &path = *continuation.path;
                        if (_tracers[thread]->advanceSample(path, sampler) ==
                            PathTracer::SampleStatus::NeedGpis) {
                            paused = true;
                            break;
                        }
                        Vec3f c = path.emission;
                        if (std::isinf(c) || c[0] > 65536 || c[1] > 65536 || c[2] > 65536 || std::isnan(c))
                            c = Vec3f(0.0f);
                        record.addSample(c);
                        _scene->cam().colorBuffer()->addSample(pixel, c);
                        continuation.path.reset();
                        ++continuation.sample;
                    }
                } catch (...) {
                    error = std::current_exception();
                }
                // After publication this task never accesses its tile again.
                {
                    std::lock_guard<std::mutex> lock(_wavefrontMutex);
                    --cpuLive;
                    if (error) {
                        if (!cpuException)
                            cpuException = error;
                        _abortWavefront = true;
                    } else if (paused)
                        waiting.push_back(tileId);
                    else
                        ++finishedTiles;
                }
                _wavefrontReady.notify_one();
            }, tileIds.size()));
    };

    std::vector<uint32> initial;
    for (uint32 i = 0; i < _tiles.size(); ++i)
        initial.push_back(i);
    try {
        schedule(initial);
        for (;;) {
            bool consume = false;
            {
                std::unique_lock<std::mutex> lock(_wavefrontMutex);
                _wavefrontReady.wait(lock, [&] {
                    // Fill the initial batch from all tiles. Subsequently submit
                    // once the CPU queue is nearly drained, or a full GPU chunk
                    // is ready. Long CPU paths cannot hold other tiles at a barrier.
                    bool canSubmit = !gpuInFlight && !waiting.empty() &&
                        (firstBatch ? cpuLive == 0 :
                            cpuLive <= ThreadUtils::pool->threadCount() || waiting.size() >= 256);
                    return _abortWavefront || gpuCompleted ||
                        finishedTiles == _tiles.size() || canSubmit;
                });
                if (_abortWavefront || finishedTiles == _tiles.size())
                    break;
                if (gpuCompleted) {
                    gpuCompleted = false;
                    consume = true;
                } else {
                    gpuTiles.swap(waiting);
                    gpuInFlight = true;
                    firstBatch = false;
                }
            }
            if (consume) {
                auto replies = submission.get();
                for (std::size_t i = 0; i < gpuTiles.size(); ++i)
                    PathTracer::resumeGpis(*continuations[gpuTiles[i]].path, replies[i]);
                gpuInFlight = false;
                schedule(gpuTiles);
                gpuTiles.clear();
            } else {
                std::vector<GpRaySegmentRequest> requests;
                for (uint32 tileId : gpuTiles)
                    requests.push_back(continuations[tileId].path->prepared.request);
                for (const auto &request : requests)
                    if (request.purpose == GpRaySegmentRequest::Purpose::Shadow)
                        ++shadowRays;
                ++waves;
                submittedRays += requests.size();
                largestWave = std::max(largestWave, requests.size());
                if (_gpRayBatchBackend) {
                    submission = _gpRayBatchBackend->submitBatch(requests, [&, this] {
                        {
                            std::lock_guard<std::mutex> lock(_wavefrontMutex);
                            gpuCompleted = true;
                            _wavefrontReady.notify_one();
                        }
                    });
                } else {
                    std::promise<std::vector<GpRaySegmentReply>> ready;
                    submission = ready.get_future();
                    ready.set_value(std::vector<GpRaySegmentReply>(requests.size()));
                    std::lock_guard<std::mutex> lock(_wavefrontMutex);
                    gpuCompleted = true;
                }
            }
            jobs.erase(std::remove_if(jobs.begin(), jobs.end(), [](const std::shared_ptr<TaskGroup> &job) {
                if (!job->isDone())
                    return false;
                job->wait();
                return true;
            }), jobs.end());
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(_wavefrontMutex);
        _abortWavefront = true;
        if (!cpuException)
            cpuException = std::current_exception();
    }
    // Drain the notification as well as the future: the callback captures this
    // coordinator's completion flag. No callback ever accesses a PathState.
    if (gpuInFlight && submission.valid()) {
        submission.wait();
        std::unique_lock<std::mutex> lock(_wavefrontMutex);
        _wavefrontReady.wait(lock, [&] { return gpuCompleted; });
    }
    for (const auto &job : jobs)
        job->wait();
    std::cout << "GPIS wavefront waves: " << waves << ", rays: " << submittedRays
              << ", largest wave: " << largestWave << '\n';
    std::cout << "GPIS wavefront segments: path=" << submittedRays - shadowRays
              << ", shadow=" << shadowRays << '\n';
    if (cpuException)
        std::rethrow_exception(cpuException);
}

void PathTraceIntegrator::renderPathPool()
{
    struct Handle { uint32 slot; uint64 generation; };
    struct Slot {
        uint64 generation = 0, sequence = 0;
        std::unique_ptr<IndependentPathSampler> sampler;
        std::unique_ptr<PathTracer::PathState> path;
    };
    struct CpuEvent { Handle path; bool paused; };
    struct Submission {
        std::vector<Handle> paths;
        std::future<std::vector<GpRaySegmentReply>> replies;
        bool completed = false; // Protected by _wavefrontMutex, including inline callbacks.
    };
    struct Result {
        Vec2u pixel;
        Vec3f color;
        PathTracer::OutputSample output;
    };

    const std::size_t capacity = _settings.gpisMaxActivePaths;
    std::vector<Slot> slots(capacity);
    std::vector<uint32> freeSlots;
    for (uint32 i = 0; i < capacity; ++i) {
        slots[i].sampler.reset(new IndependentPathSampler(_scene->_seed,
            _scene->rendererSettings().useSobol()));
        freeSlots.push_back(uint32(capacity - 1 - i));
    }
    std::vector<std::shared_ptr<TaskGroup>> jobs;
    std::vector<CpuEvent> cpuEvents;
    std::unordered_map<const SparseConvolutionNoiseMedium *, std::deque<Handle>> waiting;
    std::vector<std::shared_ptr<Submission>> submissions;
    std::map<uint64, Result> results;
    std::exception_ptr cpuException;
    std::size_t cpuLive = 0, active = 0, peakActive = 0, peakSubmissions = 0, peakResults = 0;
    std::size_t waves = 0, rays = 0, shadowRays = 0, largestWave = 0;
    uint64 issued = 0, committed = 0;
    uint32 tileId = 0, pixelOffset = 0, sample = 0;
    bool exhausted = false;

    auto checkedSlot = [&](Handle handle) -> Slot & {
        Slot &slot = slots.at(handle.slot);
        if (slot.generation != handle.generation || !slot.path)
            throw std::runtime_error("stale GPIS path handle");
        return slot;
    };
    auto schedule = [&](const std::vector<Handle> &paths) {
        if (paths.empty())
            return;
        {
            std::lock_guard<std::mutex> lock(_wavefrontMutex);
            cpuLive += paths.size();
        }
        jobs.push_back(ThreadUtils::pool->enqueue(
            [&, paths](uint32 task, uint32, uint32 thread) {
                bool paused = false;
                std::exception_ptr error;
                try {
                    if (!_abortWavefront) {
                        auto &slot = checkedSlot(paths[task]);
                        paused = _tracers[thread]->advanceSample(*slot.path, *slot.sampler) ==
                            PathTracer::SampleStatus::NeedGpis;
                    }
                } catch (...) {
                    error = std::current_exception();
                }
                // Publication transfers ownership back to the coordinator.
                // This task never touches the slot again after taking this lock.
                {
                    std::lock_guard<std::mutex> lock(_wavefrontMutex);
                    --cpuLive;
                    if (error) {
                        if (!cpuException)
                            cpuException = error;
                        _abortWavefront = true;
                    } else
                        cpuEvents.push_back({paths[task], paused});
                }
                _wavefrontReady.notify_one();
            }, paths.size()));
    };
    auto hasReadyBatch = [&] {
        for (const auto &queue : waiting)
            if (!queue.second.empty() &&
                (queue.second.size() >= _settings.gpisBatchSize || cpuLive == 0))
                return true;
        return false;
    };

    try {
        for (;;) {
            std::vector<CpuEvent> events;
            std::vector<std::shared_ptr<Submission>> completed;
            {
                std::lock_guard<std::mutex> lock(_wavefrontMutex);
                if (_abortWavefront)
                    break;
                events.swap(cpuEvents);
                for (const auto &submission : submissions)
                    if (submission->completed)
                        completed.push_back(submission);
            }
            std::vector<Handle> runnable;
            for (const auto &event : events) {
                auto &slot = checkedSlot(event.path);
                if (event.paused) {
                    waiting[slot.path->prepared.request.medium].push_back(event.path);
                } else {
                    Vec3f c = slot.path->emission;
                    if (std::isinf(c) || std::isnan(c) || c.max() > 65536.0f)
                        c = Vec3f(0.0f);
                    results.emplace(slot.sequence, Result{slot.path->pixel, c, slot.path->output});
                    peakResults = std::max(peakResults, results.size());
                    slot.path.reset();
                    freeSlots.push_back(event.path.slot);
                    --active;
                }
            }
            for (const auto &submission : completed) {
                auto replies = submission->replies.get();
                if (replies.size() != submission->paths.size())
                    throw std::runtime_error("GPIS reply count mismatch");
                for (std::size_t i = 0; i < replies.size(); ++i) {
                    auto &slot = checkedSlot(submission->paths[i]);
                    PathTracer::resumeGpis(*slot.path, replies[i]);
                    runnable.push_back(submission->paths[i]);
                }
                submissions.erase(std::find(submissions.begin(), submissions.end(), submission));
            }
            // Commit in tile/pixel/sample order, regardless of CPU or GPU completion
            // order. This also protects AOVs and shared 4x4 adaptive statistics.
            for (;;) {
                auto it = results.find(committed);
                if (it == results.end())
                    break;
                const auto &result = it->second;
                _samples[result.pixel.x()/VarianceTileSize +
                    result.pixel.y()/VarianceTileSize*_varianceW].addSample(result.color);
                _scene->cam().colorBuffer()->addSample(result.pixel, result.color);
                _tracers[0]->commitOutput(result.pixel, result.output);
                results.erase(it);
                ++committed;
            }
            // Lazy job generation: no image-sized path array. Bound the reorder
            // window as well as the live pool when an early path takes a long time.
            while (!exhausted && !freeSlots.empty() && issued - committed < capacity*2) {
                while (tileId < _tiles.size()) {
                    const auto &tile = _tiles[tileId];
                    if (pixelOffset >= tile.w*tile.h) {
                        ++tileId;
                        pixelOffset = sample = 0;
                        continue;
                    }
                    Vec2u pixel(tile.x + pixelOffset%tile.w, tile.y + pixelOffset/tile.w);
                    const auto &record = _samples[pixel.x()/VarianceTileSize +
                        pixel.y()/VarianceTileSize*_varianceW];
                    if (sample >= record.nextSampleCount) {
                        ++pixelOffset;
                        sample = 0;
                        continue;
                    }
                    const uint32 index = freeSlots.back();
                    freeSlots.pop_back();
                    auto &slot = slots[index];
                    ++slot.generation;
                    slot.sequence = issued++;
                    const uint32 absoluteSample = record.sampleIndex + sample++;
                    slot.sampler->startPath(pixel.x() + pixel.y()*_scene->cam().resolution().x(),
                        absoluteSample);
                    slot.path.reset(new PathTracer::PathState(pixel, absoluteSample));
                    slot.path->deferOutputValues = true;
                    runnable.push_back({index, slot.generation});
                    peakActive = std::max(peakActive, ++active);
                    break;
                }
                exhausted = tileId == _tiles.size();
            }
            schedule(runnable);

            while (submissions.size() < _settings.gpisInflightBatches) {
                bool canSubmit;
                {
                    std::lock_guard<std::mutex> lock(_wavefrontMutex);
                    canSubmit = !_abortWavefront && hasReadyBatch();
                }
                if (!canSubmit)
                    break;
                auto group = std::max_element(waiting.begin(), waiting.end(),
                    [](const auto &a, const auto &b) { return a.second.size() < b.second.size(); });
                auto submission = std::make_shared<Submission>();
                std::vector<GpRaySegmentRequest> requests;
                auto &queue = group->second;
                const std::size_t count = std::min<std::size_t>(queue.size(), _settings.gpisBatchSize);
                requests.reserve(count);
                submission->paths.reserve(count);
                for (std::size_t i = 0; i < count; ++i) {
                    const auto handle = queue.front();
                    queue.pop_front();
                    const auto &request = checkedSlot(handle).path->prepared.request;
                    requests.push_back(request);
                    submission->paths.push_back(handle);
                    if (request.purpose == GpRaySegmentRequest::Purpose::Shadow)
                        ++shadowRays;
                }
                submissions.push_back(submission);
                peakSubmissions = std::max(peakSubmissions, submissions.size());
                ++waves;
                rays += count;
                largestWave = std::max(largestWave, count);
                auto notify = [this, submission] {
                    {
                        std::lock_guard<std::mutex> lock(_wavefrontMutex);
                        submission->completed = true;
                        // All coordinator accesses happen before releasing the
                        // lock, so observing completion also drains notification.
                        _wavefrontReady.notify_one();
                    }
                };
                if (_gpRayBatchBackend) {
                    submission->replies = _gpRayBatchBackend->submitBatch(requests, notify);
                } else {
                    std::promise<std::vector<GpRaySegmentReply>> ready;
                    submission->replies = ready.get_future();
                    ready.set_value(std::vector<GpRaySegmentReply>(count));
                    notify();
                }
            }
            jobs.erase(std::remove_if(jobs.begin(), jobs.end(), [](const auto &job) {
                if (!job->isDone())
                    return false;
                job->wait();
                return true;
            }), jobs.end());
            if (exhausted && active == 0)
                break;
            std::unique_lock<std::mutex> lock(_wavefrontMutex);
            _wavefrontReady.wait(lock, [&] {
                if (_abortWavefront || !cpuEvents.empty())
                    return true;
                for (const auto &submission : submissions)
                    if (submission->completed)
                        return true;
                return submissions.size() < _settings.gpisInflightBatches && hasReadyBatch();
            });
        }
    } catch (...) {
        std::lock_guard<std::mutex> lock(_wavefrontMutex);
        _abortWavefront = true;
        if (!cpuException)
            cpuException = std::current_exception();
    }
    // A ready future does not guarantee its notification has returned. Drain
    // both before freeing paths, callbacks, or coordinator synchronization state.
    for (const auto &submission : submissions) {
        if (!submission->replies.valid())
            continue;
        submission->replies.wait();
        std::unique_lock<std::mutex> lock(_wavefrontMutex);
        _wavefrontReady.wait(lock, [&] { return submission->completed; });
    }
    for (const auto &job : jobs)
        job->wait();
    std::cout << "GPIS wavefront waves: " << waves << ", rays: " << rays
              << ", largest wave: " << largestWave << '\n';
    std::cout << "GPIS wavefront segments: path=" << rays - shadowRays
              << ", shadow=" << shadowRays << '\n';
    std::cout << "GPIS path pool: capacity=" << capacity << ", peak active=" << peakActive
              << ", peak submissions=" << peakSubmissions << ", peak buffered results=" << peakResults
              << ", committed=" << committed << '\n';
    if (cpuException)
        std::rethrow_exception(cpuException);
}

void PathTraceIntegrator::saveState(OutputStreamHandle &out)
{
    for (SampleRecord &s : _samples)
        s.saveState(out);
    for (ImageTile &i : _tiles)
        i.sampler->saveState(out);
}

void PathTraceIntegrator::loadState(InputStreamHandle &in)
{
    for (SampleRecord &s : _samples)
        s.loadState(in);
    for (ImageTile &i : _tiles)
        i.sampler->loadState(in);
}

void PathTraceIntegrator::fromJson(JsonPtr value, const Scene &/*scene*/)
{
    _settings.fromJson(value);
}

rapidjson::Value PathTraceIntegrator::toJson(Allocator &allocator) const
{
    return _settings.toJson(allocator);
}

void PathTraceIntegrator::prepareForRender(TraceableScene &scene, uint32 seed)
{
    _currentSpp = 0;
    _sampler = UniformSampler(MathUtil::hash32(seed));
    _scene = &scene;
    advanceSpp();
    scene.cam().requestColorBuffer();

    _gpRayBatchBackend = _settings.useCudaGpis && gpBatchFactory ?
        gpBatchFactory(scene) : nullptr;

    for (uint32 i = 0; i < ThreadUtils::pool->threadCount(); ++i)
        _tracers.emplace_back(new PathTracer(&scene, _settings, i, _gpRayBatchBackend.get()));

    // _rect = scene.cam().rect(); // Weirdly causing seg fault... use temporary workaround below
    _rect = Box2u({ 0,0 }, scene.cam().resolution());
    _varianceW = (scene.cam().resolution().x() + VarianceTileSize - 1)/VarianceTileSize;
    _varianceH = (scene.cam().resolution().y() + VarianceTileSize - 1)/VarianceTileSize;
    diceTiles();
    _samples.resize(_varianceW*_varianceH);
}

void PathTraceIntegrator::teardownAfterRender()
{
    waitForCompletion();
    _group.reset();

    _tracers.clear();
    _gpRayBatchBackend.reset();
    _samples.clear();
    _tiles  .clear();
    _tracers.shrink_to_fit();
    _samples.shrink_to_fit();
    _tiles  .shrink_to_fit();
}

bool PathTraceIntegrator::supportsResumeRender() const
{
    return true;
}

void PathTraceIntegrator::startRender(std::function<void()> completionCallback)
{
    waitForCompletion();
    if (done() || !generateWork()) {
        _currentSpp = _nextSpp;
        advanceSpp();
        completionCallback();
        return;
    }

    if (_settings.wavefrontGpis) {
        _abortWavefront = false;
        _wavefrontException = nullptr;
        _wavefrontThread = std::thread([this, completionCallback] {
            try {
                if (_settings.gpisPathPool)
                    renderPathPool();
                else
                    renderWavefront();
                if (!_abortWavefront) {
                    _currentSpp = _nextSpp;
                    advanceSpp();
                    completionCallback();
                }
            } catch (...) {
                _wavefrontException = std::current_exception();
            }
        });
        return;
    }

    using namespace std::placeholders;
    _group = ThreadUtils::pool->enqueue(
        std::bind(&PathTraceIntegrator::renderTile, this, _3, _1),
        _tiles.size(),
        [&, completionCallback]() {
            _currentSpp = _nextSpp;
            advanceSpp();
            completionCallback();
        }
    );
}

void PathTraceIntegrator::waitForCompletion()
{
    if (_wavefrontThread.joinable()) {
        _wavefrontThread.join();
        if (_wavefrontException)
            std::rethrow_exception(_wavefrontException);
    }
    if (_group) {
        _group->wait();
        _group.reset();
    }
}

void PathTraceIntegrator::abortRender()
{
    {
        std::lock_guard<std::mutex> lock(_wavefrontMutex);
        _abortWavefront = true;
    }
    _wavefrontReady.notify_one();
    if (_wavefrontThread.joinable()) {
        _wavefrontThread.join();
        if (_wavefrontException)
            std::rethrow_exception(_wavefrontException);
    }
    if (_group) {
        _group->abort();
        _group->wait();
        _group.reset();
    }
}

}
