#pragma once

#include "math/conditional/core/Random.h"
#include <array>
#include <functional>
#include <string>

namespace Tungsten::FirstPassageGradient {

// Stationary SE, explicit starting value and longitudinal derivative. Lengths
// and derivatives are in world units; mean(s) is the actual SDF along the ray.
struct Input {
    double sigma, gamma, distance;
    double birthValue, birthDerivative;
    double birthMean, birthMeanDerivative, endMeanDerivative;
    std::function<double(double)> mean;
};

enum class AnchorSelection { Fixed5Snr, Hermite };
const char *anchorSelectionName(AnchorSelection);
AnchorSelection parseAnchorSelection(const std::string &);

struct AnchorCandidate {
    double distance = 0, mean = 0, p = 0, q = 0, score = 0;
};

// Select one anchor from five fixed midpoints in the last 1.5 correlation
// lengths, minimizing p(s)+q(s)*u0 using the actual SDF. The original Hermite
// selector remains available for comparison. The selected positivity factor
// Phi(p+q*u) changes the normalized gradient law ONLY.
class Law {
    double _mean, _sd, _p = 0, _q = 0, _anchorDistance;
    bool _bridge;
    AnchorSelection _selection;
    std::array<AnchorCandidate,5> _candidates{};
    int _candidateCount = 0, _selectedAnchor = 0;
    struct Integration { double mode, width, lo, hi, mass; };
    double logShapeRatio(double x,double center) const;
    double shapeDerivative(double x) const;
    double mode() const;
    Integration integration() const;
    double integrate(const Integration &, double lo, double hi) const;
    double sampleEnvelope(ConditionalGpis::Random &) const;
public:
    explicit Law(const Input &, bool bridgeCorrection = true,
        AnchorSelection selection = AnchorSelection::Fixed5Snr);
    double sample(ConditionalGpis::Random &, bool *usedFallback = nullptr) const;
    // Deterministic diagnostic integrals; rendering does not need these.
    double pdf(double derivative) const;
    double cdf(double derivative) const;
    double gaussianMean() const { return _mean; }
    double gaussianStddev() const { return _sd; }
    double probitIntercept() const { return _p; }
    double probitSlope() const { return _q; }
    double anchorDistance() const { return _anchorDistance; }
    AnchorSelection anchorSelection() const { return _selection; }
    int candidateCount() const { return _candidateCount; }
    int selectedAnchorIndex() const { return _selectedAnchor; }
    const std::array<AnchorCandidate,5> &anchorCandidates() const { return _candidates; }
};

} // namespace Tungsten::FirstPassageGradient
