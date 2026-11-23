#pragma once

#include "math/conditional/gpss/NonstationaryConditioning.h"
#include "math/conditional/mathutility/SmallGaussian.h"
#include "math/conditional/transport/FlightState.h"

namespace Tungsten::ConditionalGpis {

struct ConditionalScalarStatistics;
FlightState startNonstationaryExterior(const NonstationaryConditioning &kernel,
    const MeanField &mean, const Point3 &position, const Vector3 &direction, Random &rng);

// Numerical optical-depth integration for the same latest-observation law.
// Both the distance sampler and shadow rays integrate its downcrossing hazard.
class NonstationarySegment {
    const NonstationaryConditioning &_kernel;
    const MeanField &_mean;
    FlightState _birth;
    Point3 _origin;
    double _length, _age, _maximumStep, _birthStep;
    NonstationaryFlight _flight;
    std::unique_ptr<MeanRayCursor> _cursor;
    mutable double _logPositiveAt = -1, _logPositive = 0;

    double nextIntervalEnd(double begin) const;
    double logPositive(double t) const;
    double opticalDepth(double begin, double end) const;
public:
    NonstationarySegment(const NonstationaryConditioning &kernel, const MeanField &mean,
        const FlightState &birth, const Point3 &origin, const Vector3 &direction,
        double length, double trackingStep = 0.0);
    ConditionalScalarStatistics statistics(double t) const;
    double hazard(double t) const;
    Gaussian<3> gradient(double t) const;
    FlightState samplePositiveEndpoint(Random &rng) const;
    double track(Random &rng, bool collision, double &hit, double begin = 0.0) const;
};

} // namespace Tungsten::ConditionalGpis
