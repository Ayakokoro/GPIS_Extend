// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>

namespace Tungsten::ConditionalGpis {

using Vector2 = Eigen::Vector2d;
using Vector3 = Eigen::Vector3d;
using Point3 = Eigen::Vector3d;
using Matrix2 = Eigen::Matrix2d;
using Matrix3 = Eigen::Matrix3d;
using Spectrum = Eigen::Vector3d;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kSqrtTwo = 1.414213562373095048801688724209698079;
constexpr double kInvSqrtTwoPi = 0.398942280401432677939946059934381868;

inline Vector3 normalizedOrThrow(const Vector3& v) {
    const double n = v.norm();
    if (!(n > 0.0) || !std::isfinite(n)) {
        throw std::invalid_argument("expected a finite nonzero vector");
    }
    return v / n;
}

inline Vector3 reflectTravelDirection(const Vector3& w, const Vector3& n) {
    return w - 2.0 * w.dot(n) * n;
}

struct Ray {
    Point3 origin = Point3::Zero();
    Vector3 direction = Vector3(0.0, 0.0, 1.0);
};

struct DomainInterval {
    bool hit = false;
    double entry = 0.0;
    double exit = 0.0;
};

struct Bounds3 {
    Point3 minimum = Point3::Constant(-1.0);
    Point3 maximum = Point3::Constant(1.0);

    bool valid() const {
        return minimum.allFinite() && maximum.allFinite() &&
               (minimum.array() < maximum.array()).all();
    }

    bool contains(const Point3& p, double tolerance = 0.0) const {
        return (p.array() >= minimum.array() - tolerance).all() &&
               (p.array() <= maximum.array() + tolerance).all();
    }

    DomainInterval intersect(const Ray& ray) const {
        double lo = 0.0;
        double hi = std::numeric_limits<double>::infinity();
        for (int i = 0; i < 3; ++i) {
            if (ray.direction[i] == 0.0) {
                if (ray.origin[i] < minimum[i] || ray.origin[i] > maximum[i]) {
                    return {};
                }
                continue;
            }
            double a = (minimum[i] - ray.origin[i]) / ray.direction[i];
            double b = (maximum[i] - ray.origin[i]) / ray.direction[i];
            if (a > b) std::swap(a, b);
            lo = std::max(lo, a);
            hi = std::min(hi, b);
            if (hi < lo) return {};
        }
        return {true, lo, hi};
    }
};

inline void orthonormalComplement(const Vector3& w, Vector3& u, Vector3& v) {
    const Vector3 wn = normalizedOrThrow(w);
    const Vector3 helper = std::abs(wn.z()) < 0.9 ? Vector3::UnitZ() : Vector3::UnitX();
    u = normalizedOrThrow(helper.cross(wn));
    v = wn.cross(u);
}

} // namespace Tungsten::ConditionalGpis

