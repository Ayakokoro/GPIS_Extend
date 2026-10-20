#ifndef PATHTRACERSETTINGS_HPP_
#define PATHTRACERSETTINGS_HPP_

#include "integrators/TraceSettings.hpp"

#include "io/JsonObject.hpp"
#include <stdexcept>

namespace Tungsten {

struct PathTracerSettings : public TraceSettings
{
    bool enableLightSampling;
    bool enableVolumeLightSampling;
    bool lowOrderScattering;
    bool includeSurfaces;
    bool useCudaGpis;
    bool wavefrontGpis;
    bool gpisPathPool;
    bool independentPathSampling;
    uint32 gpisMaxActivePaths, gpisBatchSize, gpisInflightBatches;

    PathTracerSettings()
    : enableLightSampling(true),
      enableVolumeLightSampling(true),
      lowOrderScattering(true),
      includeSurfaces(true),
      useCudaGpis(false),
      wavefrontGpis(false),
      gpisPathPool(true),
      independentPathSampling(false),
      gpisMaxActivePaths(4096),
      gpisBatchSize(1024),
      gpisInflightBatches(4)
    {
    }

    void fromJson(JsonPtr value)
    {
        TraceSettings::fromJson(value);
        value.getField("enable_light_sampling", enableLightSampling);
        value.getField("enable_volume_light_sampling", enableVolumeLightSampling);
        value.getField("low_order_scattering", lowOrderScattering);
        value.getField("include_surfaces", includeSurfaces);
        value.getField("use_cuda_gpis", useCudaGpis);
        // CUDA uses resumable paths by default; explicit false selects blocking tracing.
        wavefrontGpis = useCudaGpis;
        value.getField("wavefront_gpis", wavefrontGpis);
        value.getField("gpis_path_pool", gpisPathPool);
        independentPathSampling = wavefrontGpis && gpisPathPool;
        value.getField("independent_path_sampling", independentPathSampling);
        value.getField("gpis_max_active_paths", gpisMaxActivePaths);
        value.getField("gpis_batch_size", gpisBatchSize);
        value.getField("gpis_inflight_batches", gpisInflightBatches);
        if (wavefrontGpis && gpisPathPool && !independentPathSampling)
            throw std::runtime_error("gpis_path_pool requires independent_path_sampling");
        if (!gpisMaxActivePaths || gpisMaxActivePaths > 1048576 ||
            !gpisBatchSize || gpisBatchSize > 65536 ||
            !gpisInflightBatches || gpisInflightBatches > 64)
            throw std::runtime_error("invalid GPIS pool/batch capacity");
    }

    rapidjson::Value toJson(rapidjson::Document::AllocatorType &allocator) const
    {
        return JsonObject{TraceSettings::toJson(allocator), allocator,
            "type", "path_tracer",
            "enable_light_sampling", enableLightSampling,
            "enable_volume_light_sampling", enableVolumeLightSampling,
            "low_order_scattering", lowOrderScattering,
            "include_surfaces", includeSurfaces,
            "use_cuda_gpis", useCudaGpis,
            "wavefront_gpis", wavefrontGpis,
            "gpis_path_pool", gpisPathPool,
            "independent_path_sampling", independentPathSampling,
            "gpis_max_active_paths", gpisMaxActivePaths,
            "gpis_batch_size", gpisBatchSize,
            "gpis_inflight_batches", gpisInflightBatches
        };
    }
};

}

#endif /* PATHTRACERSETTINGS_HPP_ */
