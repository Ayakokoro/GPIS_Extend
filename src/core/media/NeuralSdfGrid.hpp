#pragma once

#include "math/conditional/gpss/MeanField.h"
#include <filesystem>

namespace Tungsten {

// Geometry-only NanoVDB input for neural GP transport. Covariance comes from
// the medium; no density/alpha/sigma grid, sidecar, or field factory is used.
class NeuralSdfGrid final : public ConditionalGpis::MeanField {
public:
    static std::shared_ptr<const NeuralSdfGrid> open(const std::filesystem::path& file);
    ~NeuralSdfGrid() override;
    const char* typeName() const override { return "nanovdb"; }
    SdfJet evaluate(const Eigen::Vector3d& p) const override;
    SdfJet evaluateInCell(const Eigen::Vector3d& p, const Eigen::Vector3d& interior) const override;
    ConditionalGpis::BoundsSummary bounds(const ConditionalGpis::Bounds3& domain) const override;
    double voxelSizeHint() const override;
    const ConditionalGpis::Bounds3& activeNodeBounds() const;
    void appendRayBreakpoints(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction,
        double begin, double end, std::vector<double>& knots) const override;
    void requireFullRayCoverage(const Eigen::Vector3d& origin, const Eigen::Vector3d& direction,
        double maximumDistance) const override;
private:
    NeuralSdfGrid();
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Tungsten
