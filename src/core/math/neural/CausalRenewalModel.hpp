#pragma once

#include "RayMeanProfile.hpp"
#include <array>
#include <cstddef>
#include <filesystem>
#include <iosfwd>
#include <memory>
#include <string>

namespace Tungsten::NeuralGpis {

enum class RayStartMode { PositiveExterior, SurfaceOutward };
struct RayStartCondition {
    RayStartMode mode = RayStartMode::PositiveExterior;
    double outwardDerivative = 0; // ell/sigma * dot(omega, gradient), for B starts
};

// Each Gaussian component is normalized on W > 0, without Rice reweighting.
struct RenewalSpeedMixture {
    std::vector<double> weights, means, scales;
};

struct CausalRenewalPrediction {
    double q = 0, cumulativeHazard = 0, logHazard = 0;
    RenewalSpeedMixture mixture;
};
struct CausalRenewalSample {
    bool hit = false;
    double q = 0, speed = 0;
    std::size_t intervals = 0;
};

// Native float32 MLP/GRU inference for the export-joint JSON format.
// Weights are immutable and shared; every flight query owns its recurrent state.
// Geometry, hazard integration and inverse sampling use double precision.
class CausalRenewalModel {
public:
    static CausalRenewalModel load(const std::filesystem::path& path);
    static CausalRenewalModel load(std::istream& input);
    double step() const;
    double integrationStep() const;
    const std::string& checkpointSha256() const;
    std::vector<CausalRenewalPrediction> query(const RayMeanProfile& profile,
        const RayStartCondition& start, const std::vector<double>& q) const;
    // Uniforms select optical depth, mixture component and component quantile.
    CausalRenewalSample sample(const RayMeanProfile& profile, const RayStartCondition& start,
        const std::array<double,3>& uniforms, double normalizedTolerance = 1e-9) const;
    // Condition on survival to beginQ without resetting the birth/grid/GRU state.
    CausalRenewalSample sampleInterval(const RayMeanProfile& profile, const RayStartCondition& start,
        const std::array<double,3>& uniforms, double beginQ, double endQ,
        double normalizedTolerance = 1e-9) const;
private:
    struct Impl;
    std::shared_ptr<const Impl> impl_;
    explicit CausalRenewalModel(std::shared_ptr<const Impl> impl) : impl_(std::move(impl)) {}
};

} // namespace Tungsten::NeuralGpis
