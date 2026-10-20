#include "PathTracerSeBatchBackend.hpp"

#include "VdbMeanGrid.hpp"
#include "cuda/RaySegmentBatch.hpp"

#include "media/GpRayBatchBackend.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"
#include "renderer/TraceableScene.hpp"
#include "integrators/path_tracer/PathTraceIntegrator.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <future>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

namespace Tungsten {
namespace {

using BatchReply = GpRaySegmentReply;

struct PendingSegment {
    GpRaySegmentRequest request;
    std::function<void(BatchReply)> reply;
};

struct BatchSubmission {
    std::vector<PendingSegment> segments;
    std::vector<BatchReply> replies;
    std::promise<std::vector<BatchReply>> done;
    std::atomic<std::size_t> remaining;
    std::function<void()> notification;
    explicit BatchSubmission(std::size_t count)
        : segments(count), replies(count), remaining(count) {}
};

void publish(PendingSegment &pending, BatchReply reply)
{
    // Moving the callback releases its submission owner after publication.
    auto callback = std::move(pending.reply);
    callback(reply);
}

struct MediumBatch {
    Gpu::Se1DRayBatcher batcher;
    Gpu::Se1DRayParameters parameters;
};

class SeBatchBackend;

struct BatchSlot {
    SeBatchBackend *owner = nullptr;
    Gpu::RaySegmentInput *inputs = nullptr;
    Gpu::RaySegmentOutput *outputs = nullptr;
    Gpu::Se1DRayBatcher batcher;
    cudaStream_t stream = nullptr;
    std::vector<PendingSegment *> selected;

    ~BatchSlot()
    {
        if (stream)
            cudaStreamDestroy(stream);
        cudaFreeHost(outputs);
        cudaFreeHost(inputs);
    }
};

class SeBatchBackend final : public GpRayBatchBackend {
    std::size_t maxBatchSize_ = 1024;
    std::size_t slotCount_ = 4;

    std::unordered_map<const SparseConvolutionNoiseMedium *, std::unique_ptr<MediumBatch>> media_;
    std::vector<std::unique_ptr<BatchSlot>> slots_;
    std::vector<BatchSlot *> freeSlots_;
    std::mutex mutex_;
    std::condition_variable ready_;
    std::vector<PendingSegment *> pending_;
    std::thread worker_;
    bool stopping_ = false;
    std::size_t inFlight_ = 0;
    std::size_t peakInFlight_ = 0;
    std::atomic<bool> disabled_{false};
    std::uint64_t batches_ = 0;
    std::uint64_t rays_ = 0;
    std::size_t largestBatch_ = 0;
    std::atomic<std::uint64_t> fallbacks_{0};
    std::atomic<std::uint64_t> stepLimits_{0}, invalidInputs_{0}, deviceFailures_{0};
    std::atomic<std::uint64_t> totalSteps_{0};
    bool profile_ = false;
    std::atomic<unsigned int> printedFailures_{0};

    static void CUDART_CB onCudaComplete(cudaStream_t, cudaError_t status, void *userData)
    {
        auto *slot = static_cast<BatchSlot *>(userData);
        slot->owner->complete(*slot, status);
    }

    void complete(BatchSlot &slot, cudaError_t status)
    {
        if (status != cudaSuccess)
            disabled_ = true;
        for (std::size_t i = 0; i < slot.selected.size(); ++i) {
            BatchReply reply;
            const auto &output = slot.outputs[i];
            if (profile_ && status == cudaSuccess)
                totalSteps_ += output.steps;
            if (status == cudaSuccess && output.status == Gpu::RaySegmentStatus::Ok) {
                reply.valid = true;
                reply.result.t = output.t;
                reply.result.value = output.value;
                reply.result.hit = output.hit != 0;
            } else {
                ++fallbacks_;
                if (status != cudaSuccess)
                    ++deviceFailures_;
                else if (output.status == Gpu::RaySegmentStatus::StepLimit)
                    ++stepLimits_;
                else
                    ++invalidInputs_;
                if (profile_ && printedFailures_.fetch_add(1) < 8) {
                    const auto &r = slot.selected[i]->request;
                    std::cout << "GPIS rejected segment: reason=" <<
                        (status != cudaSuccess ? "device" :
                         output.status == Gpu::RaySegmentStatus::StepLimit ? "step_limit" : "invalid_input")
                        << ", purpose=" << (r.purpose == GpRaySegmentRequest::Purpose::Shadow ? "shadow" : "path")
                        << ", origin=" << r.origin << ", direction=" << r.direction
                        << ", near=" << r.nearT << ", far=" << r.farT;
                    if (status == cudaSuccess)
                        std::cout << ", steps=" << output.steps;
                    std::cout << '\n';
                }
            }
            publish(*slot.selected[i], reply);
        }
        slot.selected.clear();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            --inFlight_;
            freeSlots_.push_back(&slot);
        }
        ready_.notify_one();
    }

    BatchSlot *acquireSlot()
    {
        std::unique_lock<std::mutex> lock(mutex_);
        ready_.wait(lock, [this] { return disabled_ || !freeSlots_.empty(); });
        if (disabled_)
            return nullptr;
        BatchSlot *slot = freeSlots_.back();
        freeSlots_.pop_back();
        return slot;
    }

    void fail(std::vector<PendingSegment *>::const_iterator begin,
              std::vector<PendingSegment *>::const_iterator end)
    {
        for (auto it = begin; it != end; ++it) {
            ++fallbacks_;
            publish(**it, BatchReply{});
        }
    }

    void process(const std::vector<PendingSegment *> &batch)
    {
        // Snapshot every group before publishing replies, which can release
        // the submissions owning these request pointers.
        std::unordered_map<const SparseConvolutionNoiseMedium *, std::vector<PendingSegment *>> groups;
        for (PendingSegment *pending : batch)
            groups[pending->request.medium].push_back(pending);
        // Batches may contain different media, each with its own uploaded grid.
        for (auto &medium : media_) {
            const auto &selected = groups[medium.first];
            for (std::size_t offset = 0; offset < selected.size();) {
                const std::size_t count = std::min(maxBatchSize_, selected.size() - offset);
                BatchSlot *slot = acquireSlot();
                if (!slot) {
                    fail(selected.begin() + offset, selected.end());
                    break;
                }
                slot->selected.assign(selected.begin() + offset,
                                      selected.begin() + offset + count);
                for (std::size_t i = 0; i < count; ++i) {
                    const auto &request = slot->selected[i]->request;
                    auto &input = slot->inputs[i];
                    input.origin = {request.origin.x(), request.origin.y(), request.origin.z()};
                    input.direction = {request.direction.x(), request.direction.y(), request.direction.z()};
                    input.nearT = request.nearT;
                    input.farT = request.farT;
                    input.jitter = request.jitter;
                    input.noiseSeed = request.noiseSeed;
                    input.normalSeed = request.normalSeed;
                    input.firstScatter = request.firstScatter ? 1u : 0u;
                    input.previousValue = request.previousValue;
                    input.previousGradient = {request.previousGradient.x(),
                                              request.previousGradient.y(),
                                              request.previousGradient.z()};
                }
                {
                    std::lock_guard<std::mutex> lock(mutex_);
                    ++inFlight_;
                    peakInFlight_ = std::max(peakInFlight_, inFlight_);
                }
                const cudaError_t error = slot->batcher.enqueueWithGrid(
                    slot->inputs, slot->outputs, count, medium.second->batcher.deviceMeanGrid(),
                    medium.second->parameters, onCudaComplete, slot, slot->stream);
                ++batches_;
                rays_ += count;
                largestBatch_ = std::max(largestBatch_, count);
                if (error != cudaSuccess) {
                    disabled_ = true;
                    cudaStreamSynchronize(slot->stream);
                    fail(slot->selected.begin(), slot->selected.end());
                    slot->selected.clear();
                    {
                        std::lock_guard<std::mutex> lock(mutex_);
                        --inFlight_;
                        freeSlots_.push_back(slot);
                    }
                    ready_.notify_one();
                }
                offset += count;
            }
        }
    }

    void run()
    {
        for (;;) {
            std::vector<PendingSegment *> batch;
            {
                std::unique_lock<std::mutex> lock(mutex_);
                ready_.wait(lock, [this] {
                    return (stopping_ && pending_.empty()) ||
                           (!pending_.empty() && (disabled_ || !freeSlots_.empty()));
                });
                if (stopping_ && pending_.empty())
                    return;
                batch.swap(pending_);
            }
            process(batch);
        }
    }

public:
    explicit SeBatchBackend(const TraceableScene &scene)
    {
        if (const auto *integrator = dynamic_cast<const PathTraceIntegrator *>(&scene.integrator())) {
            maxBatchSize_ = integrator->settings().gpisBatchSize;
            slotCount_ = integrator->settings().gpisInflightBatches;
        }
        const char *profile = std::getenv("TUNGSTEN_GPIS_PROFILE");
        profile_ = profile && std::string(profile) == "1";
        for (const auto &medium : scene.media()) {
            auto *sparse = dynamic_cast<SparseConvolutionNoiseMedium *>(medium.get());
            if (!sparse)
                continue;
            Se1DGpBatchConfig config;
            if (!sparse->se1DGpuBatchConfig(config)) {
                std::cout << "GPIS CUDA: scene medium is outside the supported SE 1D configuration\n";
                continue;
            }
            Gpu::HostMeanGrid mean;
            if (!Gpu::makeHostMeanGrid(*config.meanGrid, config.meanOffset,
                                       config.meanScale, mean)) {
                std::cout << "GPIS CUDA: could not export the mean VDB grid\n";
                continue;
            }
            auto batch = std::make_unique<MediumBatch>();
            batch->parameters.lengthScale = config.lengthScale;
            batch->parameters.amplitude = config.amplitude;
            batch->parameters.kernelRadius = config.kernelRadius;
            batch->parameters.impulseDensity = config.impulseDensity;
            batch->parameters.maxStepSize = config.maxStepSize;
            batch->parameters.minSteps = config.minSteps;
            if (batch->batcher.setMeanGrid(mean) == cudaSuccess)
                media_.emplace(sparse, std::move(batch));
            else
                std::cout << "GPIS CUDA: could not upload the mean VDB grid\n";
        }
        if (!media_.empty()) {
            for (std::size_t i = 0; i < slotCount_; ++i) {
                auto slot = std::make_unique<BatchSlot>();
                slot->owner = this;
                slot->selected.reserve(maxBatchSize_);
                if (cudaStreamCreateWithFlags(&slot->stream, cudaStreamNonBlocking) != cudaSuccess ||
                    slot->batcher.reserve(maxBatchSize_) != cudaSuccess ||
                    cudaMallocHost(reinterpret_cast<void **>(&slot->inputs),
                                   maxBatchSize_ * sizeof(Gpu::RaySegmentInput)) != cudaSuccess ||
                    cudaMallocHost(reinterpret_cast<void **>(&slot->outputs),
                                   maxBatchSize_ * sizeof(Gpu::RaySegmentOutput)) != cudaSuccess) {
                    media_.clear();
                    return;
                }
                freeSlots_.push_back(slot.get());
                slots_.push_back(std::move(slot));
            }
            worker_ = std::thread([this] { run(); });
        }
    }

    ~SeBatchBackend() override
    {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        ready_.notify_one();
        if (worker_.joinable())
            worker_.join();
        for (auto &slot : slots_)
            cudaStreamSynchronize(slot->stream);
        if (!media_.empty())
            std::cout << "GPIS CUDA batches: " << batches_ << ", rays: " << rays_
                      << ", CPU fallbacks: " << fallbacks_
                      << ", largest batch: " << largestBatch_ << '\n';
        if (!media_.empty())
            std::cout << "GPIS CUDA execution: slots=" << slots_.size()
                      << ", peak in flight=" << peakInFlight_ << '\n';
        if (profile_ && !media_.empty())
            std::cout << "GPIS CUDA profile: step_limit=" << stepLimits_
                      << ", invalid_input=" << invalidInputs_ << ", device_callback_failure=" << deviceFailures_
                      << ", total_steps=" << totalSteps_ << '\n';
    }

    bool empty() const { return media_.empty(); }

    bool trace(const GpRaySegmentRequest &request, GpRaySegmentResult &result) override
    {
        if (media_.find(request.medium) == media_.end())
            return false;
        PendingSegment pending;
        pending.request = request;
        std::promise<BatchReply> promise;
        std::future<BatchReply> future = promise.get_future();
        pending.reply = [&promise](BatchReply reply) { promise.set_value(reply); };
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (stopping_ || disabled_)
                return false;
            pending_.push_back(&pending);
        }
        ready_.notify_one();
        const BatchReply reply = future.get();
        if (reply.valid)
            result = reply.result;
        return reply.valid;
    }

    std::future<std::vector<GpRaySegmentReply>> submitBatch(
        const std::vector<GpRaySegmentRequest> &requests,
        std::function<void()> onComplete = {}) override
    {
        auto submission = std::make_shared<BatchSubmission>(requests.size());
        auto future = submission->done.get_future();
        submission->notification = std::move(onComplete);
        if (requests.empty()) {
            submission->done.set_value({});
            if (submission->notification)
                submission->notification();
            return future;
        }
        for (std::size_t i = 0; i < requests.size(); ++i) {
            auto &pending = submission->segments[i];
            pending.request = requests[i];
            pending.reply = [submission, i](BatchReply reply) {
                submission->replies[i] = reply;
                if (submission->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                    submission->done.set_value(std::move(submission->replies));
                    if (submission->notification)
                        submission->notification();
                }
            };
        }
        std::vector<PendingSegment *> rejected;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            for (auto &pending : submission->segments) {
                if (stopping_ || disabled_ || media_.find(pending.request.medium) == media_.end())
                    rejected.push_back(&pending);
                else
                    pending_.push_back(&pending);
            }
        }
        ready_.notify_one();
        for (auto *pending : rejected)
            publish(*pending, BatchReply{});
        return future;
    }
};

} // namespace

std::shared_ptr<GpRayBatchBackend> createPathTracerSeBatchBackend(const TraceableScene &scene)
{
    const char *enabled = std::getenv("TUNGSTEN_GPIS_CUDA");
    if (enabled && enabled[0] == '0' && enabled[1] == '\0')
        return nullptr;
    std::cout << "GPIS CUDA: probing " << scene.media().size() << " scene media\n";
    int deviceCount = 0;
    if (cudaGetDeviceCount(&deviceCount) != cudaSuccess || deviceCount == 0)
        return nullptr;
    auto backend = std::make_shared<SeBatchBackend>(scene);
    if (backend->empty())
        return nullptr;
    std::cout << "GPIS CUDA batch backend enabled\n";
    return backend;
}

} // namespace Tungsten
