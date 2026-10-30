#pragma once

#include "math/conditional/gpss/MeanField.h"
#include "math/conditional/core/Random.h"
#include <functional>
#include <vector>

namespace Tungsten::LocalEllipsoid {

using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;

// Full ellipsoids. transform maps a unit sphere to the particle, before the
// optional SDF tangent-frame rotation. Density is a number per unit volume.
struct Grain {
    M transform, inverse;
    double determinant, radius, normalRadius, maximumArea, density;
    Grain(const V &semiaxes, const V &rotationDegrees, double numberDensity);
};

struct NormalSample {
    std::size_t type = 0;
    V normal = V::UnitZ();
};

struct Collision : NormalSample {
    bool hit = false;
    double distance = 0;
    V position = V::Zero();
    SdfJet jet;
    double directRate = 0;
};

// A thinned majorant event, before deciding whether a real collision occurs.
// rate is the event intensity per distance, not the physical extinction.
struct Candidate {
    V position;
    SdfJet jet;
    double rate;
};
using CandidateCallback = std::function<void(const Candidate &)>;

struct TrackingStatistics {
    std::size_t intervals = 0, coarseIntervals = 0, candidates = 0, collisions = 0, directCandidates = 0;
};

// A memoryless local law. At EVERY candidate x the plane is
// phi_x(c) = phi(x) + gradient(phi(x)).dot(c-x).
// No particle centers, previous tangent planes or empty-path history persist.
class Model {
    std::vector<Grain> grains_;
    double decayLength_;
    bool sdfFrame_;
    M frame(const V &gradient) const;
    struct Projection {
        M transform, normalTransform;
        V incoming, tangent, bitangent;
        double area;
        V spherePoint(double cosine, double azimuth) const;
    };
    Projection projection(std::size_t type, const V &gradient, const V &w) const;
public:
    Model(std::vector<Grain> grains, double decayLength, bool sdfFrame = true);
    const std::vector<Grain> &grains() const { return grains_; }
    double decayLength() const { return decayLength_; }

    // Deterministic quadrature for inspection/validation, not used to decide
    // transport events. Raising order refines this numerical integral.
    double extinction(const SdfJet &jet, const V &travel, int order = 32) const;
    double typeExtinction(const SdfJet &jet, const V &travel, std::size_t type,
                          int order = 32) const;
    double normalIntensity(const SdfJet &jet, const V &travel,
                           std::size_t type, const V &normal) const;
    // Unnormalized angular scattering rate, without material reflectance.
    // Its integral over outgoing directions equals typeExtinction().
    double directionIntensity(const SdfJet &jet, const V &travel,
                              const V &outgoing, std::size_t type) const;
    double visibleMajorant(const SdfJet &jet, const V &travel) const;
    static double misWeight(double sampledDensity, double otherDensity);
    NormalSample sampleNormal(const SdfJet &jet, const V &travel,
                              ConditionalGpis::Random &rng) const;

    // Finite physical-distance interval, unit travel direction. Marked delta
    // tracking samples distance/type/normal together; shadow calls use the
    // same marked process for ratio tracking. Neither uses angular quadrature.
    double track(const ConditionalGpis::MeanField &sdf, const V &origin,
                 const V &travel, double begin, double end, double maximumStep,
                 ConditionalGpis::Random &rng, bool sampleCollision,
                 Collision &result, TrackingStatistics *statistics = nullptr,
                 const CandidateCallback &onCandidate = {}) const;
};

} // namespace Tungsten::LocalEllipsoid
