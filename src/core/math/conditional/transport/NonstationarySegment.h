#pragma once

#include "math/conditional/gpss/NonstationaryConditioning.h"
#include "math/conditional/mathutility/SmallGaussian.h"
#include "math/conditional/transport/FlightState.h"

namespace Tungsten::ConditionalGpis {

struct ConditionalScalarStatistics;
struct ConditionalTrackingWork;
FlightState startNonstationaryExterior(const NonstationaryConditioning &kernel,
    const MeanField &mean, const Point3 &position, const Vector3 &direction, Random &rng);

// Integrate the original birth's downcrossing hazard, optionally switching
// to null-collision tracking on intervals with a certified local bound.
class NonstationarySegment {
    const NonstationaryConditioning &_kernel;
    const MeanField &_mean;
    FlightState _birth;
    Point3 _origin;
    double _length, _age, _maximumStep, _birthStep, _coarseStep;
    bool _hybrid;
    NonstationaryFlight _flight;
    std::unique_ptr<MeanRayCursor> _cursor;
    mutable double _logPositiveAt = -1, _logPositive = 0;

    double nextIntervalEnd(double begin, MeanRayBounds *bounds) const;
    double coarseMajorant(double begin, double &end, double &retryAt) const;
    double logPositive(double t) const;
    double opticalDepth(double begin, double end) const;
    double invertOpticalDepth(double lo, double hi, double remaining, double spanDepth,
                              double depthTolerance) const;
public:
    NonstationarySegment(const NonstationaryConditioning &kernel, const MeanField &mean,
        const FlightState &birth, const Point3 &origin, const Vector3 &direction,
        double length, double trackingStep = 0.0, bool hybrid = false,
        bool coarse = false, double coarseStep = std::numeric_limits<double>::infinity());
    ConditionalScalarStatistics statistics(double t) const;
    double hazard(double t) const;
    Gaussian<3> gradient(double t) const;
    FlightState samplePositiveEndpoint(Random &rng) const;
    double track(Random &rng, bool collision, double &hit, double begin = 0.0,
                 double splitDistance = 0.0, ConditionalTrackingWork *work = nullptr) const;
};

} // namespace Tungsten::ConditionalGpis
