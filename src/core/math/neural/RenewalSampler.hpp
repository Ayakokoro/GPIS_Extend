#pragma once

#include "CausalRenewalModel.hpp"
#include "math/conditional/core/Random.h"
#include "math/conditional/gpss/GPSSField.h"

namespace Tungsten::NeuralGpis {

struct RenewalFlight {
    ConditionalGpis::Ray ray;
    std::optional<Eigen::Vector3d> birthGradient;
    RayMeanProfile mean;
    RayStartCondition start;
    double ell, sigma;
    Eigen::Vector3d birthMeanGradient;
    double maximumDistance() const { return mean.maximumX()*ell; }
};

struct RenewalSurfaceSample {
    bool hit = false;
    double distance = 0, speed = 0;
    std::size_t segment = 0;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Eigen::Vector3d gradient = Eigen::Vector3d::Zero(), normal = Eigen::Vector3d::Zero();
};

// Condition on the network's inward derivative. The two transverse gradient
// components remain Gaussian, including full-gradient correlation at B starts.
// Reflection maps this plane bijectively to outgoing directions except wi.
class RenewalGradientDistribution {
public:
    RenewalGradientDistribution(const ConditionalGpis::CovarianceKernel& kernel,
        const Eigen::Vector3d& direction, double normalizedDistance, double derivative,
        const Eigen::Vector3d& meanGradient, const Eigen::Vector3d& birthMeanGradient,
        const std::optional<Eigen::Vector3d>& birthGradient);
    const Eigen::Vector3d& incomingDirection() const { return direction_; }
    bool isDelta() const { return delta_; }
    Eigen::Vector3d sample(ConditionalGpis::Random& rng) const;
    std::optional<Eigen::Vector3d> gradientForDirection(const Eigen::Vector3d& outgoing) const;
    double phasePdf(const Eigen::Vector3d& outgoing) const;
    // Density at a sampled reflection, before float ray storage loses precision.
    double phasePdfForGradient(const Eigen::Vector3d& gradient) const;
private:
    Eigen::Vector3d direction_;
    Eigen::Matrix<double,3,2> basis_;
    Eigen::Vector2d mean_;
    Eigen::Matrix2d lower_;
    double derivative_, logNormalization_ = 0;
    bool delta_ = false;
};

// Reconstruct the full SE gradient using the same law as the renderer's phase.
Eigen::Vector3d sampleRenewalGradient(const ConditionalGpis::CovarianceKernel& kernel,
    const Eigen::Vector3d& direction, double normalizedDistance, double derivative,
    const Eigen::Vector3d& meanGradient, const Eigen::Vector3d& birthMeanGradient,
    const std::optional<Eigen::Vector3d>& birthGradient, ConditionalGpis::Random& rng);

class RenewalSampler {
public:
    RenewalSampler(const ConditionalGpis::GPSSField& field, const CausalRenewalModel& model,
                   double maximumStep = 0.25);
    // Origin is inside the active domain; an empty flight exits immediately.
    std::optional<RenewalFlight> beginFlight(const ConditionalGpis::Ray& ray,
        std::optional<Eigen::Vector3d> birthGradient = std::nullopt) const;
    RenewalSurfaceSample sample(const RenewalFlight& flight, ConditionalGpis::Random& rng,
        double physicalTolerance = 1e-9) const;
    double transmittance(const RenewalFlight& flight, double maximumDistance) const;
private:
    const ConditionalGpis::GPSSField& field_;
    const CausalRenewalModel& model_;
    double maximumStep_;
    RenewalSurfaceSample reconstruct(const RenewalFlight& flight, std::size_t segment,
        double q, double speed, ConditionalGpis::Random& rng) const;
};

} // namespace Tungsten::NeuralGpis
