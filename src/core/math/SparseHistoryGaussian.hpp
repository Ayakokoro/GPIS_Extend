#pragma once

#include "math/conditional/gpss/GPSSField.h"
#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

namespace Tungsten::ConditionalGpis { class Random; }

namespace Tungsten::SparseHistoryGpis {

class BivariateNormalCdfTable;

struct ProbabilityStatistics {
    std::atomic<std::uint64_t> tableAttempts{0};
    std::atomic<std::uint64_t> tableAccepted{0};
};

struct Start {
    enum Kind { ExteriorPositiveOnly, SurfaceValueGradient };
    Kind kind = ExteriorPositiveOnly;
    Eigen::Vector3d origin = Eigen::Vector3d::Zero();
    Eigen::Vector3d direction = Eigen::Vector3d::UnitZ();
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
};

struct ProbabilitySettings {
    double absoluteTolerance = 1e-9;
    double relativeTolerance = 2e-4;
    int initialSamples = 256;
    int maxSamples = 8192;
    // Optional shared table. An absent table retains the numerical reference.
    std::shared_ptr<const BivariateNormalCdfTable> bivariateTable;
    std::shared_ptr<ProbabilityStatistics> statistics;
};

// Generic covariance, including negative correlations. Numerical estimates,
// not certified bounds. At most four positive sign constraints are supported.
double positiveOrthant(const Eigen::VectorXd &mean,
                       const Eigen::MatrixXd &covariance,
                       const ProbabilitySettings &settings = {});

double standardizedMean(const ConditionalGpis::GPSSField &field,
                        const Start &start, double distance);

// One immutable local sparse-history model. Internal positions are physical
// distances strictly between zero and distance; at most two are supported.
class LocalLaw {
    struct Impl;
    std::shared_ptr<Impl> _impl;
public:
    LocalLaw(const ConditionalGpis::GPSSField &field, const Start &start,
             double distance, const std::vector<double> &history,
             const ProbabilitySettings &settings = {});
    double hazard() const;
    double speedCdf(double speed) const;
    double sampleSpeed(double openUniform) const;
    // Flux-normal proposal with history acceptance; a bounded number of
    // failed proposals falls back to the same inverse CDF, never a new hit.
    double sampleSpeed(ConditionalGpis::Random &random) const;
    // Raw SDF gradient, before any GP conditioning or crossing weighting.
    Eigen::Vector3d meanGradient() const;
};

} // namespace Tungsten::SparseHistoryGpis
