#pragma once

#include "io/Scene.hpp"
#include "math/conditional/gpss/GPSSField.h"
#include <nlohmann/json.hpp>
#include <memory>
#include <optional>

namespace Tungsten::SinglePointTraining {

// Owns the parsed Tungsten scene and an immutable view of its single GPIS SDF.
// Loading this object does not prepare a renderer or load neural model weights.
struct SceneField {
    std::unique_ptr<Scene> scene;
    std::shared_ptr<const ConditionalGpis::MeanField> mean;
    ConditionalGpis::GPSSField field;
    nlohmann::json metadata;
};

SceneField loadSceneField(const Path &sceneFile,
    const std::optional<ConditionalGpis::Bounds3> &domainOverride = std::nullopt);

} // namespace Tungsten::SinglePointTraining
