#ifndef PATHTRACEINTEGRATOR_HPP_
#define PATHTRACEINTEGRATOR_HPP_

#include "PathTracerSettings.hpp"
#include "SampleRecord.hpp"
#include "PathTracer.hpp"

#include "integrators/Integrator.hpp"
#include "integrators/ImageTile.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/UniformSampler.hpp"

#include "thread/TaskGroup.hpp"

#include "math/MathUtil.hpp"

#include <thread>
#include <memory>
#include <vector>
#include <atomic>

namespace Tungsten {

class GpRayBatchBackend;
class TraceableScene;

using PathTracerGpBatchFactory = std::shared_ptr<GpRayBatchBackend> (*)(const TraceableScene &);
void setPathTracerGpBatchFactory(PathTracerGpBatchFactory factory);

class PathTraceIntegrator : public Integrator
{
    static CONSTEXPR uint32 TileSize = 16;
    static CONSTEXPR uint32 VarianceTileSize = 4;
    static CONSTEXPR uint32 AdaptiveThreshold = 16;

    PathTracerSettings _settings;

    std::shared_ptr<TaskGroup> _group;
    std::thread _wavefrontThread;
    std::atomic<bool> _abortWavefront{false};
    std::exception_ptr _wavefrontException;
    std::mutex _wavefrontMutex;
    std::condition_variable _wavefrontReady;

    Box2u _rect;
    uint32 _varianceW;
    uint32 _varianceH;

    UniformSampler _sampler;
    std::shared_ptr<GpRayBatchBackend> _gpRayBatchBackend;
    std::vector<std::unique_ptr<PathTracer>> _tracers;

    std::vector<SampleRecord> _samples;
    std::vector<ImageTile> _tiles;

    void diceTiles();

    float errorPercentile95();
    void dilateAdaptiveWeights();
    void distributeAdaptiveSamples(int spp);
    bool generateWork();

    void renderTile(uint32 id, uint32 tileId);
    void renderWavefront();
    void renderPathPool();

    virtual void saveState(OutputStreamHandle &out) override;
    virtual void loadState(InputStreamHandle &in) override;

public:
    PathTraceIntegrator();
    ~PathTraceIntegrator() override;

    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual void prepareForRender(TraceableScene &scene, uint32 seed) override;
    virtual void teardownAfterRender() override;

    virtual bool supportsResumeRender() const override;

    virtual void startRender(std::function<void()> completionCallback) override;
    virtual void waitForCompletion() override;
    virtual void abortRender() override;
    
    const PathTracerSettings &settings() const
    {
        return _settings;
    }
};

}

#endif /* PATHTRACEINTEGRATOR_HPP_ */
