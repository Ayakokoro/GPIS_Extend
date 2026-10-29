#ifndef PATHTRACEDIAGNOSTICS_HPP_
#define PATHTRACEDIAGNOSTICS_HPP_

#include "math/Vec.hpp"
#include <nlohmann/json.hpp>

namespace Tungsten {

// Opt-in observer for the blocking path tracer. Recording never draws samples
// or changes transport. Connection branches are distinct from continuation
// vertices: their contributions must not be assigned the final path length.
struct PathTraceDiagnostics {
    using Json = nlohmann::json;
    Json events = Json::array();
    Json result;
    int mediumScatters = 0;
    int surfaceReflections = 0;

    template<typename V> static Json vector(const V &v)
    { return Json::array({v.x(), v.y(), v.z()}); }

    size_t add(const char *type, Json event = Json::object())
    {
        event["type"] = type;
        event["index"] = events.size();
        events.push_back(std::move(event));
        return events.size() - 1;
    }

    Vec3f finish(const std::string &reason, const Vec3f &radiance)
    {
        result = {{"termination", reason}, {"radiance", vector(radiance)},
            {"medium_scatters", mediumScatters}, {"surface_reflections", surfaceReflections}};
        return radiance;
    }
};

}
#endif
