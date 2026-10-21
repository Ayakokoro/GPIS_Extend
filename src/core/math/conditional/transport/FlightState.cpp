// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/transport/FlightState.h"
#include <stdexcept>

namespace Tungsten::ConditionalGpis {

FlightState startExternalFlight(const Point3& x, const Vector3& w) {
    return {x, normalizedOrThrow(w), 0.0};
}

FlightState startClassicCollisionFlight(const Point3& x, const Vector3& w) {
    return {x, normalizedOrThrow(w), 0.0};
}

} // namespace Tungsten::ConditionalGpis
