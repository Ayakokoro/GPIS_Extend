#pragma once

#include "math/conditional/transport/ConditionalFlightKernel.h"
#include <array>

namespace Tungsten::ConditionalGpis {

// All distributions condition on the SAME complete (F, gradient F) birth as
// global_conditional. The added history observation is only F(u)>0.
struct SinglePointHistorySample {
    // X=F(t), U=F(u), V=-dF(t)/dt. Each z is mean / standard deviation
    // BEFORE conditioning on X=0. Correlations also condition only on birth.
    std::array<double, 6> features; // zX,zU,zV,rhoXU,rhoXV,rhoUV
    double baselineHazard = 0;
    double logBaselineHazard = 0;
    double historyHazard = 0;
    double logHistoryHazard = 0;
    double historyPositiveProbability = 0;
    double logHistoryPositiveProbability = 0;
    double logCorrection = 0;
    double momentRelativeError = 0;
};

// Numerical training-data oracle, not a renderer or a neural runtime.
// Ages are physical distances from the complete-observation birth. The field
// must outlive this object. Cursors belong to this object; use one per worker.
class SinglePointHistoryTeacher {
public:
    SinglePointHistoryTeacher(const GPSSField& field, const FlightState& birth,
                             NumericPolicy policy = NumericPolicy{});
    SinglePointHistorySample evaluate(double currentAge, double historyAge) const;
    const ConditionalFlightKernel& baseline() const { return baseline_; }

private:
    ConditionalFlightKernel baseline_;
    NumericPolicy policy_;
    double sigma_, inverseLength_;
};

} // namespace Tungsten::ConditionalGpis
