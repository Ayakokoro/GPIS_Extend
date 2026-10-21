// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Types.h"

namespace Tungsten::ConditionalGpis {
struct FlightState {
    Point3 birthPosition = Point3::Zero();
    Vector3 direction = Vector3::UnitZ();
    double age = 0.0;
    double birthValue = 0.0;
    Vector3 birthGradient = Vector3::Zero();
};

FlightState startExternalFlight(const Point3& x, const Vector3& w);
FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w);

} // namespace Tungsten::ConditionalGpis
