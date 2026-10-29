#ifndef BOUNCECONTRIBUTIONS_HPP_
#define BOUNCECONTRIBUTIONS_HPP_

#include "math/Vec.hpp"
#include <array>
#include <string>

namespace Tungsten {

// Optional radiance AOV for the blocking tracer. Bin k counts k scattering
// vertices between camera and light, including the current NEE vertex. Forward
// medium boundaries and the light endpoint are not scattering vertices.
struct BounceContributions {
    static constexpr int BinCount = 6; // 0, 1, 2, 3, 4, 5+
    std::array<Vec3d, BinCount> radiance;
    std::string termination;

    BounceContributions() { reset(); }
    void reset() {
        for (auto &c : radiance) c = Vec3d(0.0);
        termination.clear();
    }
    void add(int order, const Vec3f &value) {
        radiance[order < 5 ? order : 5] += Vec3d(value);
    }
};

// Thread-local so an image diagnostic can observe production paths without
// allocating JSON events or changing the sampler / ordinary render settings.
inline thread_local BounceContributions *activeBounceContributions = nullptr;
inline void recordBounceContribution(int order, const Vec3f &value) {
    if (activeBounceContributions) activeBounceContributions->add(order, value);
}

class ScopedBounceContributions {
    BounceContributions *_previous;
public:
    explicit ScopedBounceContributions(BounceContributions *value)
        : _previous(activeBounceContributions) { activeBounceContributions = value; }
    ~ScopedBounceContributions() { activeBounceContributions = _previous; }
    ScopedBounceContributions(const ScopedBounceContributions &) = delete;
    ScopedBounceContributions &operator=(const ScopedBounceContributions &) = delete;
};

}
#endif
