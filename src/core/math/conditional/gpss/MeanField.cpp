// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/gpss/MeanField.h"
#include <algorithm>
#include <array>
#include <limits>
#include <stdexcept>

namespace Tungsten::ConditionalGpis {

MeanRayPoint MeanField::queryRayPoint(const Point3& origin, const Vector3& direction,
                                    double begin, double maximumEnd) const {
    if (!origin.allFinite() || !direction.allFinite() || !(begin >= 0) ||
        !std::isfinite(maximumEnd) || !(maximumEnd > begin))
        throw std::invalid_argument("invalid point-query interval");
    std::vector<double> knots;
    appendRayBreakpoints(origin,direction,begin,maximumEnd,knots);
    double end = maximumEnd;
    for (double t : knots) if (t > begin && t < end) end = t;
    const Point3 point = origin+begin*direction;
    requireFullRayCoverage(point,direction,end-begin);
    return {evaluateInCell(point,origin+(begin+0.5*(end-begin))*direction),end};
}

MeanRayBounds MeanField::rayBounds(const Point3& origin, const Vector3& w,
                                    double begin, double end,const SdfRayQueryOptions& options) const {
    const Point3 first = origin + begin * w, last = origin + end * w;
    const Vector3 pad = Vector3::Constant(1e-12 * (1.0 + first.norm() + last.norm()));
    const BoundsSummary b = bounds({first.cwiseMin(last) - pad, first.cwiseMax(last) + pad});
    MeanRayBounds result;
    if (!b.certified) return result;
    const Point3 rayMidpoint = origin + (0.5*begin + 0.5*end)*w;
    const Point3 midpoint = options.midpoint.value_or(rayMidpoint);
    const double middle = evaluate(midpoint).value;
    result.midpointValue = middle;
    const double speed = w.norm();
    const double delta = b.maximumGradientNorm *
        (speed * (end - begin) * 0.5 + (midpoint-rayMidpoint).norm());
    result.minimumValue = std::max(b.minimumValue, middle - delta);
    result.maximumValue = middle + delta;
    result.minimumDerivative = -b.maximumGradientNorm * speed;
    result.maximumDerivative = b.maximumGradientNorm * speed;
    if (options.computeBeginDerivative)
        result.beginDerivative = evaluate(first).gradient.dot(w);
    if (const auto gradient = affineGradient()) {
        result.minimumDerivative = result.maximumDerivative = gradient->dot(w);
        result.maximumSecondDerivative = 0.0;
    }
    const double error = 128.0 * std::numeric_limits<double>::epsilon() *
        (1.0 + std::abs(middle) + delta + b.maximumGradientNorm * speed);
    result.minimumValue -= error;
    result.maximumValue += error;
    result.minimumDerivative -= error;
    result.maximumDerivative += error;
    result.certified = true;
    return result;
}

PlaneMean::PlaneMean(Vector3 normal, double offset)
    : normal_(normalizedOrThrow(normal)), offset_(offset) {
    if (!std::isfinite(offset)) throw std::invalid_argument("plane offset must be finite");
}

MeanJet PlaneMean::evaluate(const Point3& x) const {
    return {normal_.dot(x) - offset_, normal_};
}

BoundsSummary PlaneMean::bounds(const Bounds3& domain) const {
    Point3 minimizing;
    for (int i = 0; i < 3; ++i) {
        minimizing[i] = normal_[i] >= 0.0 ? domain.minimum[i] : domain.maximum[i];
    }
    return {normal_.dot(minimizing) - offset_, 1.0, true};
}

SphereMean::SphereMean(Point3 center, double radius) : center_(std::move(center)), radius_(radius) {
    if (!(radius > 0.0) || !center_.allFinite()) {
        throw std::invalid_argument("invalid sphere mean");
    }
}

MeanJet SphereMean::evaluate(const Point3& x) const {
    const Vector3 displacement = x - center_;
    const double length = displacement.norm();
    if (length == 0.0) {
        throw std::domain_error("sphere mean is nondifferentiable at its center");
    }
    return {length - radius_, displacement / length};
}

BoundsSummary SphereMean::bounds(const Bounds3& domain) const {
    Point3 closest = center_.cwiseMax(domain.minimum).cwiseMin(domain.maximum);
    return {(closest - center_).norm() - radius_, 1.0, true};
}

double SphereMean::valueDifference(const Point3& x, const Vector3& displacement) const {
    const Vector3 radial = x - center_;
    const double denominator = radial.norm() + (radial + displacement).norm();
    if (denominator == 0.0) return 0.0;
    return (2.0 * radial.dot(displacement) + displacement.squaredNorm()) / denominator;
}

MeanRayBounds SphereMean::rayBounds(const Point3& origin, const Vector3& w,
                                    double begin, double end,const SdfRayQueryOptions& options) const {
    MeanRayBounds result = MeanField::rayBounds(origin, w, begin, end, options);
    const Vector3 radial = origin - center_;
    const double closest = std::clamp(-radial.dot(w) / w.squaredNorm(), begin, end);
    const double minimumRadius = (radial + closest * w).norm();
    if (minimumRadius > 0.0)
        result.maximumSecondDerivative = std::nextafter(w.squaredNorm() / minimumRadius,
                                                        std::numeric_limits<double>::infinity());
    return result;
}
CutawaySphereMean::CutawaySphereMean(Point3 center, double innerRadius, double outerRadius)
    : center_(std::move(center)), innerRadius_(innerRadius), outerRadius_(outerRadius) {
    if (!center_.allFinite() || !std::isfinite(innerRadius_) || !std::isfinite(outerRadius_) ||
        !(innerRadius_ > 0.0) || !(outerRadius_ > innerRadius_)) {
        throw std::invalid_argument("cutaway sphere requires finite center and 0 < inner radius < outer radius");
    }
}

MeanJet CutawaySphereMean::evaluate(const Point3& x) const {
    const Vector3 p = x - center_;
    const double radius = p.norm();
    const bool inside = radius <= innerRadius_ ||
        (radius <= outerRadius_ && (p.x() <= 0.0 || p.y() <= 0.0));

    // The boundary consists of two restricted spherical patches and two
    // half-annuli. Project onto each patch to recover both distance and gradient.
    double closestDistance = std::numeric_limits<double>::infinity();
    Vector3 closestOffset = Vector3::Zero();
    Vector3 boundaryNormal = Vector3::UnitZ();
    const auto consider = [&](const Vector3& point, const Vector3& normal) {
        const Vector3 offset = p - point;
        const double distance = offset.norm();
        if (distance < closestDistance) {
            closestDistance = distance;
            closestOffset = offset;
            boundaryNormal = normal;
        }
    };

    // Inner sphere: only the x >= 0, y >= 0 patch is exposed.
    Vector3 innerDirection(std::max(p.x(), 0.0), std::max(p.y(), 0.0), p.z());
    const double innerLength = innerDirection.norm();
    if (innerLength > 0.0) innerDirection /= innerLength;
    else innerDirection = Vector3::UnitZ();
    consider(innerRadius_ * innerDirection, innerDirection);

    // Outer sphere: the x > 0, y > 0 patch is absent. Inside that quadrant,
    // the nearest retained point lies on the closer of its two bounding planes.
    Vector3 outerDirection = p;
    if (p.x() > 0.0 && p.y() > 0.0) {
        if (p.x() <= p.y()) outerDirection.x() = 0.0;
        else outerDirection.y() = 0.0;
    }
    const double outerLength = outerDirection.norm();
    if (outerLength > 0.0) outerDirection /= outerLength;
    else outerDirection = -Vector3::UnitX();
    consider(outerRadius_ * outerDirection, outerDirection);

    // Each cut is a planar half-annulus bounded by the two sphere radii.
    Vector3 cutX(0.0, std::max(p.y(), 0.0), p.z());
    const double cutXLength = cutX.norm();
    if (cutXLength > 0.0) {
        cutX *= std::clamp(cutXLength, innerRadius_, outerRadius_) / cutXLength;
    } else {
        cutX = innerRadius_ * Vector3::UnitZ();
    }
    consider(cutX, Vector3::UnitX());

    Vector3 cutY(std::max(p.x(), 0.0), 0.0, p.z());
    const double cutYLength = cutY.norm();
    if (cutYLength > 0.0) {
        cutY *= std::clamp(cutYLength, innerRadius_, outerRadius_) / cutYLength;
    } else {
        cutY = innerRadius_ * Vector3::UnitZ();
    }
    consider(cutY, Vector3::UnitY());

    // Strict comparisons above fix the choice at equally close boundary patches:
    // inner sphere, outer sphere, x cut, then y cut. On or within roundoff of a
    // boundary, use its outward normal rather than normalize cancellation noise
    // from the projection. At seams this selects an adjacent one-sided normal.
    const double sign = inside ? -1.0 : 1.0;
    const double normalTolerance = 32.0 * std::numeric_limits<double>::epsilon() *
        std::max(outerRadius_, radius);
    if (closestDistance <= normalTolerance) return {sign * closestDistance, boundaryNormal};
    return {sign * closestDistance, sign * closestOffset / closestDistance};
}

BoundsSummary CutawaySphereMean::bounds(const Bounds3& domain) const {
    const Point3 midpoint = 0.5 * domain.minimum + 0.5 * domain.maximum;
    const double halfDiagonal = (0.5 * (domain.maximum - domain.minimum)).norm();
    // Exact signed distance is globally 1-Lipschitz. The solid is also a subset
    // of the outer ball, so that ball's minimum SDF supplies another lower bound.
    const Point3 closestToCenter = center_.cwiseMax(domain.minimum).cwiseMin(domain.maximum);
    const double outerBallLowerBound = (closestToCenter - center_).norm() - outerRadius_;
    const double lipschitzLowerBound = evaluate(midpoint).value - halfDiagonal;
    return {std::max(outerBallLowerBound, lipschitzLowerBound), 1.0, true};
}

} // namespace Tungsten::ConditionalGpis
