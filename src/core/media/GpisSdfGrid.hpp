#pragma once
#include "math/GPFunctions.hpp"
#include "grids/VdbGrid.hpp"
#include "math/conditional/gpss/MeanField.h"
#include <algorithm>
namespace Tungsten {
// A copy of the INPUT SDF, never a density/extinction grid. Double precision
// trilinear interpolation provides its exact piecewise gradient (including the
// VdbGrid clamp), instead of the generic MeanFunction's fixed 0.001 difference.
struct GpisSdfGrid : ConditionalGpis::MeanField {
#if OPENVDB_AVAILABLE
    VdbGrid::DenseDensity data;
    Eigen::Matrix3d worldToIndex;
    Eigen::Vector3d translation;
    double scale, offset, gradientBound;
    double magnitudeBound = std::numeric_limits<double>::quiet_NaN();

    struct Macrocell {
        double minimumValue,maximumValue,maximumNodeMagnitude;
        Eigen::Vector3d minimumGradient,maximumGradient; // unscaled index derivatives
    };
    int macrocellSize=0;
    Vec3i macrocellOrigin=Vec3i(0),macrocellDimensions=Vec3i(0);
    std::vector<Macrocell> macrocells;
    // Built once before rendering; interpolation still uses the original nodes.
    void buildMacrocells(int cellSize);
    const Macrocell &macrocell(int x,int y,int z) const {
        return macrocells[(size_t(z)*macrocellDimensions.y()+y)*macrocellDimensions.x()+x];
    }

    double at(int x, int y, int z) const
    {
        return data.values[(size_t(z)*data.dimensions.y() + y)*data.dimensions.x() + x];
    }
    double initialize(const TabulatedMean &mean, const VdbGrid &grid)
    {
        if (!grid.exportLinearDensity(data))
            throw std::runtime_error("classic_global: VDB SDF needs linear interpolation, no detail/emission grid");
        scale = mean.scale(); offset = mean.offset();
        for (int i = 0; i < 3; ++i) {
            translation[i] = data.worldToIndex(i, 3);
            for (int j = 0; j < 3; ++j) worldToIndex(i, j) = data.worldToIndex(i, j);
        }
        if (!worldToIndex.allFinite() || !translation.allFinite() ||
            !std::isfinite(scale) || !std::isfinite(offset))
            throw std::runtime_error("classic_global: invalid SDF transform/scale");
        for (float value : data.values)
            if (!std::isfinite(value)) throw std::runtime_error("classic_global: non-finite SDF voxel");
        initializeMagnitudeBounds();
        return gradientBound;
    }
    void initializeMagnitudeBounds()
    {
        Vec3i lo, hi;
        for (int i = 0; i < 3; ++i) {
            lo[i] = int(std::floor(data.clampMin[i])) - data.origin[i];
            hi[i] = int(std::floor(data.clampMax[i])) - data.origin[i] + 1;
        }
        Eigen::Vector3d edge = Eigen::Vector3d::Zero();
        double magnitude=0, arithmeticScale=0;
        // Ignore the unused padding/background discontinuity outside the
        // clamped interpolation domain; it cannot affect any field query.
        for (int z = lo.z(); z <= hi.z(); ++z)
            for (int y = lo.y(); y <= hi.y(); ++y)
                for (int x = lo.x(); x <= hi.x(); ++x) {
                    const double v = at(x, y, z);
                    magnitude=std::max(magnitude,std::abs((v+offset)*scale));
                    arithmeticScale=std::max(arithmeticScale,(std::abs(v)+std::abs(offset))*std::abs(scale));
                    if (x < hi.x()) edge.x() = std::max(edge.x(), std::abs(at(x+1, y, z) - v));
                    if (y < hi.y()) edge.y() = std::max(edge.y(), std::abs(at(x, y+1, z) - v));
                    if (z < hi.z()) edge.z() = std::max(edge.z(), std::abs(at(x, y, z+1) - v));
                }
        // Each partial derivative of a trilinear field is a convex combination
        // of edge differences. Include mean scale and the complete world map.
        gradientBound = std::abs(scale)*(worldToIndex.cwiseAbs().transpose()*edge).norm();
        // Include roundoff from the value/gradient interpolants and transforms.
        const double pad=128*std::numeric_limits<double>::epsilon();
        magnitudeBound=std::nextafter(magnitude+pad*arithmeticScale,std::numeric_limits<double>::infinity());
        // The direct gradient query sums signed node contributions. Its
        // cancellation error depends on node magnitude even for a flat grid.
        const double gradientPad=pad*arithmeticScale*
            (worldToIndex.cwiseAbs().transpose()*Eigen::Vector3d::Ones()).norm();
        gradientBound=std::nextafter(gradientBound*(1+pad)+gradientPad,std::numeric_limits<double>::infinity());
    }
    double evaluate(const Vec3d &p, Vec3d *gradient) const
    {
        const Eigen::Vector3d raw = worldToIndex*Eigen::Vector3d(p.x(), p.y(), p.z()) + translation;
        Eigen::Vector3d q, f, mask;
        int cell[3];
        for (int i = 0; i < 3; ++i) {
            q[i] = std::clamp(raw[i], double(data.clampMin[i]), double(data.clampMax[i]));
            mask[i] = raw[i] < data.clampMin[i] || raw[i] > data.clampMax[i] ? 0.0 : 1.0;
            cell[i] = int(std::floor(q[i])) - data.origin[i];
            f[i] = q[i] - std::floor(q[i]);
        }
        double value = 0.0;
        Eigen::Vector3d g = Eigen::Vector3d::Zero();
        for (int z = 0; z < 2; ++z)
            for (int y = 0; y < 2; ++y)
                for (int x = 0; x < 2; ++x) {
                    const double v = at(cell[0]+x, cell[1]+y, cell[2]+z);
                    const double wx = x ? f.x() : 1.0-f.x();
                    const double wy = y ? f.y() : 1.0-f.y();
                    const double wz = z ? f.z() : 1.0-f.z();
                    value += v*wx*wy*wz;
                    if (gradient) {
                        g.x() += v*(x ? 1.0 : -1.0)*wy*wz;
                        g.y() += v*wx*(y ? 1.0 : -1.0)*wz;
                        g.z() += v*wx*wy*(z ? 1.0 : -1.0);
                    }
                }
        if (gradient) { const Eigen::Vector3d v = scale*worldToIndex.transpose()*g.cwiseProduct(mask); *gradient = Vec3d(v.x(), v.y(), v.z()); }
        return (value + offset)*scale;
    }
    const char *typeName() const override { return "vdb_sdf"; }
    std::optional<ConditionalGpis::MeanMagnitudeBounds> globalMagnitudeBounds() const override {
        return ConditionalGpis::MeanMagnitudeBounds{magnitudeBound,gradientBound};
    }
    ConditionalGpis::MeanJet evaluate(const Eigen::Vector3d &p) const override;
    std::unique_ptr<ConditionalGpis::MeanRayCursor> createRayCursor(
        const Eigen::Vector3d &origin,const Eigen::Vector3d &direction) const override;
    double valueDifference(const Eigen::Vector3d &p, const Eigen::Vector3d &d) const override;
    ConditionalGpis::BoundsSummary bounds(const ConditionalGpis::Bounds3 &) const override;
    ConditionalGpis::MeanRayBounds rayBounds(const Eigen::Vector3d &o, const Eigen::Vector3d &w,
                                           double begin, double end,
                                           const SdfRayQueryOptions &options = {}) const override;
    void appendRayBreakpoints(const Eigen::Vector3d &o, const Eigen::Vector3d &w,
                              double begin, double end, std::vector<double> &knots) const override;
#endif
};

}
