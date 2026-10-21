#pragma once
#include "MeanField.h"

namespace Tungsten {
class MeanFunction;
namespace ConditionalGpis {

// Adapts any scene mean with SdfQueries; geometry and transforms stay in math/.
class SdfMeanField final : public MeanField {
    std::shared_ptr<const MeanFunction> _mean;
    const SdfQueries *_queries;
public:
    explicit SdfMeanField(std::shared_ptr<const MeanFunction> mean);
    const char *typeName() const override { return "scene_sdf"; }
    MeanJet evaluate(const Point3 &x) const override { return _queries->evaluateJet(x); }
    double valueDifference(const Point3 &x,const Vector3 &d) const override {
        return _queries->valueDifference(x,d);
    }
    BoundsSummary bounds(const Bounds3 &domain) const override {
        const auto b=_queries->bounds(domain.minimum,domain.maximum);
        return {b.minimumValue,b.maximumGradientNorm,b.certified};
    }
    MeanRayBounds rayBounds(const Point3 &o,const Vector3 &w,double lo,double hi,
                           const SdfRayQueryOptions &options = {}) const override {
        return _queries->rayBounds(o,w,lo,hi,options);
    }
};
}
}
