#ifndef PHASEFUNCTION_HPP_
#define PHASEFUNCTION_HPP_

#include "samplerecords/PhaseSample.hpp"

#include "sampling/WritablePathSampleGenerator.hpp"

#include "io/JsonSerializable.hpp"

namespace Tungsten {

class PathSampleGenerator;
class Scene;
class MediumSample;

class PhaseFunction : public JsonSerializable
{
public:
    virtual void fromJson(JsonPtr value, const Scene &scene) override;
    virtual rapidjson::Value toJson(Allocator &allocator) const override;

    virtual Vec3f eval(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const = 0;
    // Evaluate a light connection and sample any latent state needed on that
    // branch. Ordinary phase functions have no additional state.
    virtual Vec3f evalConnection(PathSampleGenerator &, const Vec3f &wi, const Vec3f &wo,
            const MediumSample &s, std::shared_ptr<const MediumPathContext> &) const {
        return eval(wi, wo, s);
    }
    virtual Vec3f evalGrad(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const {
        std::cout << "PhaseFunction::evalGrad() not implemented!\n";
        return Vec3f(0.f);
    };
    virtual bool sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const = 0;
    virtual bool invert(WritablePathSampleGenerator &sampler, const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const;
    virtual float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const = 0;

    virtual bool isSpecular() const { return false; }
};

}

#endif /* PHASEFUNCTION_HPP_ */
