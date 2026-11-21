#pragma once

#if OPENVDB_AVAILABLE
#include <Eigen/Core>
#include <array>

namespace Tungsten {
struct GpisSdfGrid;

// Lightweight value-only traversal for intersection with a piecewise cubic field.
class GpisSdfCubicRay {
    const GpisSdfGrid &_grid;
    Eigen::Vector3d _q, _v;

    struct Block {
        double begin=0, end=0, lower=0;
    };
    std::array<Block,2> _blocks;

public:
    GpisSdfCubicRay(const GpisSdfGrid &grid, const Eigen::Vector3d &origin,
                    const Eigen::Vector3d &direction);

    // Skip a block when its mean lower bound exceeds noiseBound. A bound on
    // the realized noise certifies the skip. The three-sigma cutoff is approximate.
    double skipPositive(double begin, double end, double noiseBound);

    // Restrict the clamped trilinear mean to [begin, min(limit, next voxel face)].
    // Returns the right endpoint. Coefficients are in ascending powers of
    // u = (t-begin)/(end-begin). Both methods permit arbitrary seeks.
    double polynomial(double begin, double limit, std::array<double,4> &coefficients) const;
};
}
#endif
