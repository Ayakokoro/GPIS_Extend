// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Types.h"
#include "math/SdfQueries.hpp"
#include <memory>
#include <optional>
#include <vector>

namespace Tungsten::ConditionalGpis {

using MeanJet = SdfJet;

struct MeanRayPoint {
    MeanJet jet;
    double end = 0; // physical ray distance; first cell boundary or requested end
};

struct BoundsSummary {
    double minimumValue = 0.0;
    double maximumGradientNorm = 0.0;
    bool certified = false;
};

using MeanRayBounds = SdfRayBounds;

struct MeanRaySample {
    double value, derivative;
};

// Optional finite bounds for the entire mean, including its exterior/clamp.
// An unbounded analytic SDF must not advertise this capability.
struct MeanMagnitudeBounds {
    double maximumAbsoluteValue, maximumGradientNorm;
};

// Optional per-flight query state. The field remains immutable and shared;
// cursors belong to one ray and may seek backwards for interval refinement.
class MeanRayCursor {
public:
    struct Statistics {
        std::size_t cellBuilds=0, pointFallbacks=0, boundFallbacks=0, coarseQueries=0;
    };
    virtual ~MeanRayCursor() = default;
    virtual double nextBoundary(double begin,double end) = 0;
    virtual MeanRaySample sample(double t) = 0;
    virtual MeanJet evaluate(double t) = 0;
    // Optional error for cancellation-safe midpoint enclosures near birth.
    virtual double valueDifference(double t,double *error=nullptr) = 0;
    virtual MeanRayBounds rayBounds(double begin,double end,
                                    const SdfRayQueryOptions &options = {}) = 0;
    // Optional precomputed block certificate; clips end to the block boundary.
    // It may span nonsmooth interpolation cells and cannot certify birth curvature.
    virtual bool coarseRayBounds(double, double &, MeanRayBounds &) { return false; }
    virtual Statistics statistics() const = 0;
};

class MeanField {
public:
    virtual ~MeanField() = default;
    // The config's "mean_type" string. This is what lets a resolved config echo
    // itself back out through the same registry that built it, so the core
    // library needs no dynamic_cast per field type -- and a type it has never
    // heard of, such as a baked NanoVDB grid, still round-trips.
    virtual const char* typeName() const = 0;
    // The statistical scale the field fixes by itself, when it has one. A
    // procedural mean is scale-free -- the same sphere renders at any sigma, so
    // sigma there is a model knob and the config is its only source. A *baked*
    // field is not: sigma is frozen into its data, since the band spans +-3
    // sigma, the background comes from the grid and dx was chosen against it. The
    // config may then omit "sigma" and read it from here, which is why this is
    // a capability of the field rather than a field of its own: nothing
    // downstream can tell the two sources apart, because kernel.sigma() is
    // derived from the resolved value either way.
    virtual std::optional<double> intrinsicSigma() const { return std::nullopt; }
    virtual MeanJet evaluate(const Point3& x) const = 0;
    virtual std::unique_ptr<MeanRayCursor> createRayCursor(const Point3&,const Vector3&) const {
        return nullptr;
    }
    // Compatibility query for Tungsten function-space rendering. Analytic
    // means keep their analytic derivatives; tabulated means override this.
    virtual MeanJet evaluateTungsten(const Point3& x) const { return evaluate(x); }
    // Evaluate the interpolant selected by an interior point of the ray segment.
    // At a grid face this preserves that segment's one-sided gradient.
    virtual MeanJet evaluateInCell(const Point3& x, const Point3&) const { return evaluate(x); }
    // Query only the current point, choosing the forward cell at a boundary.
    // The returned interval never crosses an interpolation-cell boundary.
    virtual MeanRayPoint queryRayPoint(const Point3& origin, const Vector3& direction,
                                      double begin, double maximumEnd) const;
    virtual double valueDifference(const Point3& x, const Vector3& displacement) const {
        if (const auto gradient = affineGradient()) return gradient->dot(displacement);
        return evaluate(x + displacement).value - evaluate(x).value;
    }
    virtual BoundsSummary bounds(const Bounds3& domain) const = 0;
    virtual std::optional<MeanMagnitudeBounds> globalMagnitudeBounds() const { return std::nullopt; }
    virtual MeanRayBounds rayBounds(const Point3& origin, const Vector3& direction,
                                    double begin, double end,
                                    const SdfRayQueryOptions& options = {}) const;
    // If the mean is affine, returns its constant gradient. Otherwise returns
    virtual std::optional<Vector3> affineGradient() const { return std::nullopt; }
    // Nonzero for a sampled grid. The integrator partitions at interpolation
    // cell boundaries rather than using a loose global gradient bound.
    virtual double voxelSizeHint() const { return 0.0; }
    virtual void appendRayBreakpoints(const Point3&, const Vector3&, double, double,
                                      std::vector<double>&) const {}
    // Analytic means are defined everywhere. Sampled means must certify that
    // this entire segment is covered by stored data, not background values.
    // Used by GP first-passage experiments; does not change renderer vacuum rules.
    virtual void requireFullRayCoverage(const Point3&, const Vector3&, double) const {}
};

class PlaneMean final : public MeanField {
public:
    const char* typeName() const override { return "plane"; }
    PlaneMean(Vector3 normal, double offset);
    MeanJet evaluate(const Point3& x) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
    std::optional<Vector3> affineGradient() const override { return normal_; }
    const Vector3& normal() const { return normal_; }
    double offset() const { return offset_; }
private:
    Vector3 normal_;
    double offset_;
};

class SphereMean final : public MeanField {
public:
    const char* typeName() const override { return "sphere"; }
    SphereMean(Point3 center, double radius);
    MeanJet evaluate(const Point3& x) const override;
    double valueDifference(const Point3& x, const Vector3& displacement) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
    MeanRayBounds rayBounds(const Point3& origin, const Vector3& direction,
                            double begin, double end,
                            const SdfRayQueryOptions& options = {}) const override;
private:
    Point3 center_;
    double radius_;
};

// A solid sphere with the quarter-sector x > 0, y > 0 removed outside
// innerRadius. Coordinates are relative to center; the inner sphere is kept.
// evaluate() returns the exact signed Euclidean distance. At sharp edges or
// medial-axis points its unit gradient is a deterministic one-sided choice.
class CutawaySphereMean final : public MeanField {
public:
    const char* typeName() const override { return "cutaway_sphere"; }
    CutawaySphereMean(Point3 center, double innerRadius, double outerRadius);
    MeanJet evaluate(const Point3& x) const override;
    BoundsSummary bounds(const Bounds3& domain) const override;
    const Point3& center() const { return center_; }
    double innerRadius() const { return innerRadius_; }
    double outerRadius() const { return outerRadius_; }
private:
    Point3 center_;
    double innerRadius_;
    double outerRadius_;
};

class ConstantMean final : public MeanField {
public:
    const char* typeName() const override { return "constant"; }
    explicit ConstantMean(double value) : value_(value) {}
    MeanJet evaluate(const Point3&) const override { return {value_, Vector3::Zero()}; }
    BoundsSummary bounds(const Bounds3&) const override { return {value_, 0.0, true}; }
    std::optional<MeanMagnitudeBounds> globalMagnitudeBounds() const override {
        return MeanMagnitudeBounds{std::abs(value_),0.0};
    }
    std::optional<Vector3> affineGradient() const override { return Vector3::Zero(); }
private:
    double value_;
};

using MeanFieldPtr = std::shared_ptr<const MeanField>;

} // namespace Tungsten::ConditionalGpis
