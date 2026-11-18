#ifndef TRACESETTINGS_HPP_
#define TRACESETTINGS_HPP_

#include "io/JsonPtr.hpp"

namespace Tungsten {

struct TraceSettings
{
    bool enableConsistencyChecks;
    bool enableTwoSidedShading;
    bool countForwardBounces = false;
    int minBounces;
    int maxBounces;

    TraceSettings()
    : enableConsistencyChecks(false),
      enableTwoSidedShading(true),
      minBounces(0),
      maxBounces(64)
    {
    }

    void fromJson(JsonPtr value)
    {
        value.getField("min_bounces", minBounces);
        value.getField("max_bounces", maxBounces);
        value.getField("enable_consistency_checks", enableConsistencyChecks);
        value.getField("enable_two_sided_shading", enableTwoSidedShading);
        value.getField("count_forward_bounces", countForwardBounces);
    }

    rapidjson::Value toJson(rapidjson::Document::AllocatorType &allocator) const
    {
        rapidjson::Value v(rapidjson::kObjectType);
        v.AddMember("min_bounces", minBounces, allocator);
        v.AddMember("max_bounces", maxBounces, allocator);
        v.AddMember("enable_consistency_checks", enableConsistencyChecks, allocator);
        v.AddMember("enable_two_sided_shading", enableTwoSidedShading, allocator);
        v.AddMember("count_forward_bounces", countForwardBounces, allocator);
        return std::move(v);
    }
};

}

#endif /* TRACESETTINGS_HPP_ */
