#include "RayMeanProfile.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace Tungsten::NeuralGpis {
using ConditionalGpis::MeanField;
using ConditionalGpis::Point3;
using ConditionalGpis::Vector3;

namespace {
void positive(double x, const char* name) {
    if (!(x>0.0) || !std::isfinite(x)) throw std::invalid_argument(name);
}

std::vector<double> partition(std::vector<double> knots, double maximumStep) {
    positive(maximumStep, "invalid profile maximum step");
    std::sort(knots.begin(), knots.end());
    knots.erase(std::unique(knots.begin(), knots.end()), knots.end());
    std::vector<double> result{knots.front()};
    for (std::size_t i=1; i<knots.size(); ++i) {
        const double width=knots[i]-knots[i-1];
        const double requested=std::ceil(width/maximumStep);
        if (!(requested>=1.0) || requested>10000000.0)
            throw std::invalid_argument("ray mean profile has too many segments");
        const auto count=static_cast<std::size_t>(requested);
        for (std::size_t j=1; j<count; ++j)
            result.push_back(knots[i-1]+width*(static_cast<double>(j)/count));
        result.push_back(knots[i]);
    }
    return result;
}
}

RayMeanProfile::RayMeanProfile(std::vector<RayMeanSegment> segments)
    : segments_(std::move(segments)) {
    if (segments_.empty()) throw std::invalid_argument("empty ray mean profile");
    double previous=0.0;
    for (const auto& segment : segments_) {
        if (segment.begin!=previous || !(segment.end>segment.begin) ||
            !std::isfinite(segment.end) || !std::isfinite(segment.polynomial.a) ||
            !std::isfinite(segment.polynomial.b) || !std::isfinite(segment.polynomial.c) ||
            !std::isfinite(segment.polynomial.d))
            throw std::invalid_argument("invalid or non-contiguous ray mean profile");
        previous=segment.end;
    }
}

RayMeanProfile RayMeanProfile::fromField(const MeanField& mean, const Point3& origin,
    const Vector3& direction, double sigma, double ell, double maximumDistance,
    double maximumStep) {
    positive(sigma,"invalid profile sigma");
    positive(ell,"invalid profile ell");
    positive(maximumDistance,"invalid profile physical distance");
    if (!origin.allFinite() || !direction.allFinite() ||
        std::abs(direction.norm()-1.0)>1e-10)
        throw std::invalid_argument("profile requires finite origin and unit direction");
    mean.requireFullRayCoverage(origin,direction,maximumDistance);
    std::vector<double> knots{0.0,maximumDistance};
    mean.appendRayBreakpoints(origin,direction,0.0,maximumDistance,knots);
    for (double& t : knots) t/=ell;
    knots=partition(std::move(knots),maximumStep);
    std::vector<RayMeanSegment> segments;
    const auto sample=[&](double x) { return mean.evaluate(origin+(ell*x)*direction).value/sigma; };
    for (std::size_t i=1; i<knots.size(); ++i) {
        const double lo=knots[i-1], hi=knots[i], width=hi-lo;
        const double v0=sample(lo), v1=sample(lo+width/3.0);
        const double v2=sample(lo+2.0*width/3.0), v3=sample(hi);
        // Four values recover the cubic without querying ambiguous gradients
        // on a voxel face. Both endpoint derivatives belong to this cell.
        segments.push_back({lo,hi,{
            (-9*v0+27*v1-27*v2+9*v3)/2,
            (18*v0-45*v1+36*v2-9*v3)/2,
            (-11*v0+18*v1-9*v2+2*v3)/2,v0}});
    }
    return RayMeanProfile(std::move(segments));
}

} // namespace Tungsten::NeuralGpis
