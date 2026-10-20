#include "VdbMeanGrid.hpp"

#include "grids/VdbGrid.hpp"

#include <utility>

namespace Tungsten::Gpu {

bool makeHostMeanGrid(const VdbGrid &source, float meanOffset, float meanScale,
                      HostMeanGrid &destination)
{
    VdbGrid::DenseDensity dense;
    if (!source.exportLinearDensity(dense))
        return false;

    HostMeanGrid result;
    result.voxels = std::move(dense.values);
    for (int axis = 0; axis < 3; ++axis) {
        result.origin[axis] = dense.origin[axis];
        result.dimensions[axis] = dense.dimensions[axis];
    }
    const Mat4f &transform = dense.worldToIndex;
    for (int row = 0; row < 3; ++row)
        for (int column = 0; column < 4; ++column)
            result.worldToIndex.m[row * 4 + column] = transform[row * 4 + column];
    result.clampMin = {dense.clampMin.x(), dense.clampMin.y(), dense.clampMin.z()};
    result.clampMax = {dense.clampMax.x(), dense.clampMax.y(), dense.clampMax.z()};
    result.offset = meanOffset;
    result.scale = meanScale;
    destination = std::move(result);
    return true;
}

} // namespace Tungsten::Gpu
