#pragma once

#include "math/conditional/transport/ConditionalFlightKernel.h"
#include <array>
#include <memory>
#include <string>
#include <vector>

namespace Tungsten::ConditionalGpis {

struct SinglePointHistoryAnchor {
    double t, z, mills, mean;
};

// The same geometry-driven pool and deterministic selection are used by data
// generation and rendering. The horizon is the fixed training-domain ray exit,
// never a numerical tracking interval or the next random Poisson proposal.
std::vector<SinglePointHistoryAnchor> buildSinglePointHistoryAnchors(
    const GPSSField&, const FlightState&, double horizon, double minimumSeparation,
    const ConditionalFlightKernel&);
double selectSinglePointHistoryAnchor(const std::vector<SinglePointHistoryAnchor>&,
    double currentAge, double lengthScale, double minimumSeparation);

// Cheap full-birth-conditioned Gaussian features. No orthant probabilities,
// quadrature, gradient correction, or neural marching grid is evaluated here.
std::array<double,6> singlePointNeuralFeatures(const GPSSField&, const FlightState&,
    const ConditionalFlightKernel&, double currentAge, double historyAge);

struct SinglePointNeuralEvaluation {
    double ratio = 1, logCorrection = 0;
    bool networkEvaluated = false, capped = false;
    bool outsideDomain = false, beforeMinimumAge = false, outsideFeatureRange = false;
};

class SinglePointNeuralModel final {
public:
    static std::shared_ptr<const SinglePointNeuralModel> load(
        const std::string& filename, std::optional<double> capOverride = {});
    void validateField(const GPSSField&) const;
    // Compare the serialized, resolved native mean description to the training
    // scene. Asset content hashes remain part of experiment provenance.
    void validateMeanDescription(const std::string& json) const;
    SinglePointNeuralEvaluation evaluate(const std::array<double,6>& features) const;
    double cap() const { return cap_; }
    double sigma() const { return sigma_; }
    double lengthScale() const { return ell_; }
    double minimumAgeOverLength() const { return minimumAge_; }
    double minimumSeparationOverLength() const { return minimumSeparation_; }
    const Bounds3& domain() const { return domain_; }
private:
    struct Linear { unsigned in=0, out=0; std::vector<float> weights,bias; };
    std::array<Linear,3> layers_;
    std::array<float,6> mean_, scale_;
    std::array<double,6> minimumFeatures_, maximumFeatures_;
    Bounds3 domain_;
    bool asinhZ_=false;
    std::string meanDescription_;
    double cap_=2, sigma_=0, ell_=0, minimumAge_=.05, minimumSeparation_=.01;
};

class SinglePointNeuralFlight final {
public:
    SinglePointNeuralFlight(const SinglePointNeuralModel&, const GPSSField&,
        const FlightState&, const ConditionalFlightKernel&);
    SinglePointNeuralEvaluation correction(double age, const ConditionalFlightKernel&) const;
    double horizon() const { return horizon_; }
    const std::vector<SinglePointHistoryAnchor>& anchors() const { return anchors_; }
private:
    const SinglePointNeuralModel& model_;
    const GPSSField& field_;
    FlightState birth_;
    std::vector<SinglePointHistoryAnchor> anchors_;
    double horizon_=0;
    bool outsideDomain_=false;
};

} // namespace Tungsten::ConditionalGpis
