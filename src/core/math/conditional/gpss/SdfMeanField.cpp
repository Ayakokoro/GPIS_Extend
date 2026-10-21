#include "SdfMeanField.hpp"
#include "math/GPFunctions.hpp"
#include <stdexcept>

namespace Tungsten::ConditionalGpis {
SdfMeanField::SdfMeanField(std::shared_ptr<const MeanFunction> mean)
    : _mean(std::move(mean)),_queries(_mean ? _mean->sdfQueries() : nullptr)
{
    if(!_queries)
        throw std::invalid_argument("scene mean does not provide certified SDF queries");
}
}
