#pragma once

#include "math/conditional/gpss/GPSSField.h"
#include "math/conditional/transport/FlightState.h"

namespace Tungsten {
class ProceduralNonstationaryCovariance;
namespace ConditionalGpis {

struct NonstationaryParameters;
struct NonstationaryBoundData;

// Paciorek-Schervish covariance with a squared-exponential base and diagonal
// local length tensor. The amplitude and widths are measured in world space.
class NonstationaryConditioning {
    std::shared_ptr<const NonstationaryParameters> _parameters;
    friend class NonstationaryFlight;
public:
    explicit NonstationaryConditioning(const ProceduralNonstationaryCovariance &source);
    double minLength() const;
    double minSigma() const;
    double nextBreakpoint(const Point3 &origin, const Vector3 &direction, double age) const;
    KernelJet evaluate(const Point3 &x, const Point3 &y) const;
    PointPrior pointPrior(const Point3 &x, const MeanJet &mean) const;
};

struct NonstationaryMoments {
    double meanF, varianceF;
    Vector3 meanG, regression;
    Matrix3 gradientGivenValueCovariance;
    double slopeVarianceGivenValue;
};

// Condition on the full value and gradient at the original flight birth.
// Age is a physical distance along the unit birth direction.
class NonstationaryFlight {
    const NonstationaryConditioning &_kernel;
    const MeanField &_mean;
    FlightState _birth;
    MeanJet _birthMean;
    std::unique_ptr<MeanRayCursor> _cursor;
    mutable std::unique_ptr<NonstationaryBoundData> _boundData;
    double _nearDistance;
public:
    NonstationaryFlight(const NonstationaryConditioning &kernel, const MeanField &mean,
                        const FlightState &birth);
    NonstationaryFlight(NonstationaryFlight &&) noexcept;
    ~NonstationaryFlight();
    // Return +infinity when the interval needs direct integration.
    double hazardMajorant(double beginAge, double endAge, const MeanRayBounds &meanBounds) const;
    bool coarseMeanBounds(double beginAge, double &endAge, MeanRayBounds &range) const {
        return _cursor && _cursor->coarseRayBounds(beginAge,endAge,range);
    }
    NonstationaryMoments moments(double age) const;
    PointPrior at(double age) const;
};

}}
