#ifndef TUNGSTEN_GPU_PATH_TRACER_SE_BATCH_BACKEND_HPP
#define TUNGSTEN_GPU_PATH_TRACER_SE_BATCH_BACKEND_HPP

#include <memory>

namespace Tungsten {

class GpRayBatchBackend;
class TraceableScene;

std::shared_ptr<GpRayBatchBackend> createPathTracerSeBatchBackend(const TraceableScene &scene);

} // namespace Tungsten

#endif
