// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/transport/FlightKernel.h"
#include "math/conditional/core/Random.h"
#include <limits>

namespace Tungsten::ConditionalGpis {

FlightState startConditionalExterior(const GPSSField& field, const Point3& entry,
                                     const Vector3& direction, Random& rng);

struct ConditionalMajorantInterval {
    double begin, end, maximum;
};

struct ConditionalMajorants {
    double split = 0.0;
    double nearMaximum = 0.0;
    double farMaximum = 0.0;
    std::uint64_t boundIntervals = 0;
    std::vector<ConditionalMajorantInterval> farIntervals;
};

struct ConditionalBirthMajorant {
    double split = 0.0;
    double maximum = 0.0;
    double smoothEnd = 0.0;
    double slope = 0.0;
    double curvature = 0.0;
};

struct ConditionalScalarStatistics {
    double meanF, varianceF, meanK, meanAtZero, varianceAtZero;
};

// SE GP conditioned on the complete (value, gradient) observation at birth.
// Surface flights have birthValue=0; exterior flights start from F>0.
class ConditionalFlightKernel final : public FlightKernel {
public:
    ConditionalFlightKernel(const GPSSField& field, const FlightState& state);
    HazardEvaluation evaluate(double age) const override;
    HitStatistics hitStatistics(double age) const override;
    // Joint endpoint sample conditioned on birth and F(age)>0. Geometry renews
    // the latest observation this way; this is not a survival-conditioned GP.
    FlightState samplePositiveEndpoint(double age, Random& rng) const;
    ConditionalScalarStatistics scalarStatistics(double age) const;
    ConditionalBirthMajorant birthMajorant(double end) const;
    double birthAdjacentMajorant(double begin, double end,
                                const ConditionalBirthMajorant& birth) const;
    ConditionalMajorants twoSegmentMajorants(double end) const;
    // Tighten only when the cheap (or inherited) rate times interval length
    // exceeds the threshold. Zero always tightens; infinity requests cheap only.
    double intervalMajorant(double begin, double end, double tightenExpectedEvents=0.0,
                            double parentMaximum=std::numeric_limits<double>::infinity()) const;
    // May return infinity when the cheap enclosure overflows. The adaptive
    // entry point above then tries the fine bound before reporting a failure.
    double cheapIntervalMajorant(double begin, double end) const {
        return intervalMajorant(begin,end,std::numeric_limits<double>::infinity());
    }
    double nextMeanBoundary(double begin,double end) const;
    bool coarseMajorant(double begin,double &end,double &maximum) const;
private:
    double boundedIntervalMajorant(const MeanRayBounds &mean,double begin,double end,
                                  double tightenExpectedEvents,double parentMaximum) const;
    MeanRayBounds meanRayBounds(double begin,double end,const SdfRayQueryOptions &options = {}) const;
    double meanValueDifference(double age,double *error=nullptr) const;
    MeanJet meanJet(double age) const;
    Gaussian<3> gradientGivenValueDeviation(double age, double deviation) const;
    std::unique_ptr<MeanRayCursor> meanRay_;
    MeanJet birthMean_;
    Vector3 deltaGradient_, precisionDirection_;
    double deltaValue_, deltaSlope_, inverseLength2_, sigma2_;
};

} // namespace Tungsten::ConditionalGpis
