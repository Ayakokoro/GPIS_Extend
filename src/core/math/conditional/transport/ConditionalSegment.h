#pragma once
#include "ConditionalFlightKernel.h"

namespace Tungsten::ConditionalGpis {
enum class ConditionalMajorantMode { Adaptive, TwoConstants };

struct ConditionalTrackingSettings {
    ConditionalMajorantMode majorantMode = ConditionalMajorantMode::Adaptive;
    double constantBirthLength = 0.0;  // two_constants: world distance to switch; 0 uses analytic birth split
    double step = 0.0;                 // world distance; 0 selects stepScale*ell
    double stepScale = 0.5;            // multiple of the directional GP length
    double ageScale = 0.25;            // maximum step / distance from birth
    double maxExpectedEvents = 64.0;   // refine when majorant*length exceeds this
    double tightenExpectedEvents = 4.0;// run expensive bound above this; 0 always
    int macrocellSize = 0;             // 0 disables; power-of-two cell width, 2..64
    double macrocellMaxExpectedEvents = 1.0; // otherwise descend to fine cells
    void validate() const;
};

// Optional per-call diagnostics, including partial work if sampling throws.
struct ConditionalTrackingWork {
    std::uint64_t candidates=0, intervals=0;
    double distance=0;
};

// A finite, forward part of a straight conditional flight. Geometry endpoints
// renew the observation before a continuation or light connection changes direction.
class ConditionalSegment {
    const GPSSField &_field;
    Point3 _origin;
    Vector3 _w;
    double _length, _age;
    ConditionalTrackingSettings _tracking;
    ConditionalFlightKernel _straight;
    ConditionalBirthMajorant _birthBound;
    double _tailMaximum=0;
public:
    ConditionalSegment(const GPSSField &field, const FlightState &birth,
                       const Point3 &origin, const Vector3 &direction, double length,
                       const ConditionalTrackingSettings &tracking = {});
    ConditionalScalarStatistics statistics(double t) const;
    Gaussian<3> gradient(double t) const;
    double hazard(double t) const;
    double majorant(double begin, double end,
                    double parentMaximum=std::numeric_limits<double>::infinity()) const;
    // Joint endpoint observation conditioned on the old birth and F(endpoint)>0.
    FlightState samplePositiveEndpoint(Random &rng) const;
    // Collision tracking returns survival 0/1; ratio tracking returns a weight.
    // hit is the physical distance from this segment's origin.
    double track(Random &rng, bool collision, double &hit, ConditionalTrackingWork *work=nullptr) const;
};
}
