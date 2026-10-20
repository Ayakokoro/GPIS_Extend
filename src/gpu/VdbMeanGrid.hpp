#ifndef TUNGSTEN_GPU_VDB_MEAN_GRID_HPP
#define TUNGSTEN_GPU_VDB_MEAN_GRID_HPP

#include "cuda/RaySegmentBatch.hpp"

namespace Tungsten {
class VdbGrid;
}

namespace Tungsten::Gpu {

// Export an already loaded VdbGrid in the same coordinates as TabulatedMean.
// Only linear interpolation without a detail/emission grid is currently supported.
bool makeHostMeanGrid(const VdbGrid &source, float meanOffset, float meanScale,
                      HostMeanGrid &destination);

} // namespace Tungsten::Gpu

#endif
