#pragma once

#include "math/SparseHistoryGaussian.hpp"
#include <memory>

namespace Tungsten::SparseHistoryGpis {

struct FlightSettings {
    int historyPoints = 2; // Internal sign constraints, excluding the exterior origin.
    double maxStepOverLength = 0.25;
    double maxMeanRangeOverSigma = 0.25;
    double opticalDepthTolerance = 1e-4;
    double distanceTolerance = 1e-5; // Physical distance.
    int maxSubdivisions = 4096;
    ProbabilitySettings probability;
    void validate() const;
};

struct DistanceSample {
    bool hit = false;
    double distance = 0;
    double traversedOpticalDepth = 0;
};

// A single straight flight, with physical distances measured from Start.origin.
// The field outlives this object. A flight owns a mutable geometry cursor/cache;
// construct a separate Flight for each renderer branch or shadow connection.
class Flight {
    struct Impl;
    std::unique_ptr<Impl> _impl;
public:
    Flight(const ConditionalGpis::GPSSField &field,const Start &start,
           const FlightSettings &settings = {});
    ~Flight();
    Flight(const Flight &) = delete;
    Flight &operator=(const Flight &) = delete;
    const Start &start() const;
    double lengthScale() const;
    // K=1: the lowest standardized mean in the fixed geometry pool plus the
    // recent point. K=2: the recent point and the lowest representative from
    // a different correlation-length bucket. Returned positions are sorted.
    std::vector<double> selectHistory(double distance);
    LocalLaw law(double distance);
    Eigen::Vector3d meanGradient(double distance);
    double opticalDepth(double begin,double end);
    DistanceSample sampleDistance(double begin,double end,double targetOpticalDepth);
};

} // namespace Tungsten::SparseHistoryGpis
