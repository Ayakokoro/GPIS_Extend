// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/gpss/MeanField.h"
#include "math/conditional/gpss/CovarianceKernel.h"
#include <memory>

namespace Tungsten::ConditionalGpis {

// value-gradient joint distribution at a single point, with optional prior information
struct PointPrior {
    double meanF = 0.0;
    double varianceF = 0.0;
    Vector3 meanG = Vector3::Zero();
    Matrix3 covarianceG = Matrix3::Zero();
    Vector3 covarianceFG = Vector3::Zero();
};

struct GPSSField {
    MeanFieldPtr mean;
    CovarianceKernel kernel;
    Bounds3 activeDomain;
    PointPrior pointPrior(const Point3& x) const;
    void validate() const;
};

} // namespace Tungsten::ConditionalGpis
