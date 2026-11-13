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
    std::atomic<std::uint64_t> analyticCovariances{0};
    std::atomic<std::uint64_t> featureCovariances{0};
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
    // C++ numerical-reference switch for validation; not a scene setting.
    bool referenceCovariance = false;
};

// Generic covariance, including negative correlations. Numerical estimates,
// not certified bounds. At most four positive sign constraints are supported.
double positiveOrthant(const Eigen::VectorXd &mean,
                       const Eigen::MatrixXd &covariance,
                       const ProbabilitySettings &settings = {});

double standardizedMean(const ConditionalGpis::GPSSField &field,
                        const Start &start, double distance);

// Per-flight origin statistics for candidate heights. Independent cursor
// copies preserve the reference point-query rounding and near-birth algebra;
// query order never changes which polynomial anchor is used. The field must
// outlive this evaluator and remain immutable for the flight.
class HistoryCandidateEvaluator {
    struct Impl;
    std::unique_ptr<Impl> _impl;
public:
    HistoryCandidateEvaluator(const ConditionalGpis::GPSSField &field, const Start &start);
    ~HistoryCandidateEvaluator();
    double standardizedMean(double distance) const;
};

struct HistoryCandidateStatistics {
    double standardizedMean = 0;
    double score = 0;
};

// Cheap single-point selection heuristic, not an error or majorant bound:
// Mills(z_u)^2 * [r_tu^2 * (1-r_tu^2) + gamma_tu^2]. Camera starts use
// the original Gaussian here; their positive start sign remains in LocalLaw.
// Surface starts use the value-and-gradient conditioned Gaussian. Positions
// are physical distances with 0 < historyDistance < currentDistance.
HistoryCandidateStatistics historyCandidateStatistics(
    const ConditionalGpis::GPSSField &field, const Start &start,
    double currentDistance, double historyDistance);

// Reuses the current-position part of the same selection score across a
// candidate group, without querying the mean field or constructing a cursor.
// Distances are normalized by the directional SE correlation length. Pass
// Mills(z_u), not its square; the caller may cache it with the candidate.
class HistoryCandidateScorer {
    double _q;
    bool _surface;
    bool _scalarSurface = false;
    // Lazily built only when a covariance subtraction loses precision.
    // A scorer belongs to one candidate group, not to shared field state.
    mutable int _featureCount = 0;
    mutable Eigen::VectorXd _currentFeature, _speedFeature;
    double _logCurrentVariance = 0, _currentVariance = 1;
    double _speedVariance = 1, _speedRegression = 0;
    double _currentSd = 1, _speedSd = 1, _birthFactor = 1;
    double featureScore(double historyNormalizedDistance, double inverseMills) const;
public:
    HistoryCandidateScorer(double currentNormalizedDistance, bool surface);
    double score(double historyNormalizedDistance, double inverseMills) const;
};

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
