#pragma once

#include "ClassicGlobalMedium.hpp"
#include "math/LatticeGaussianNoise.hpp"
#include <optional>

namespace Tungsten {

struct VaryingLatticeModel;

class LatticeCubicGpisMedium final : public ClassicGlobalMedium {
    float _spacingInEll=0.5f, _radiusInEll=3.0f;
    float _entryStepSize=0.01f;
    bool _prune=true, _threeSigmaPruning=false;
    bool _inheritFullGradient=false;
    std::unique_ptr<LatticeGaussianKernel> _kernel;
    std::unique_ptr<VaryingLatticeModel> _varying;
    std::shared_ptr<PhaseFunction> _deltaPhase;

    std::optional<float> trace(PathSampleGenerator &, const Ray &, MediumState *, MediumSample *) const;
    template<class Noise, class Model>
    std::optional<float> traceField(const Model &, PathSampleGenerator &, const Ray &, MediumState *, MediumSample *) const;
public:
    LatticeCubicGpisMedium();
    ~LatticeCubicGpisMedium() override;
    void fromJson(JsonPtr, const Scene &) override;
    rapidjson::Value toJson(Allocator &) const override;
    void prepareForRender() override;
    // Geometric first contact, including a negative entry probe seen from the back.
    std::optional<float> firstHitDistance(PathSampleGenerator &, const Ray &, MediumState *) const;
    bool sampleDistance(PathSampleGenerator &, const Ray &, MediumState &, MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &, const Ray &, bool, bool, MediumState *) const override;
    float pdf(PathSampleGenerator &, const Ray &, bool, bool) const override;
};

}
