// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#pragma once

#include "math/conditional/core/Types.h"
#include <cstdint>
#include <limits>
#include <random>
#include "sampling/PathSampleGenerator.hpp"

namespace Tungsten::ConditionalGpis {

class Random {
public:
    explicit Random(std::uint64_t seed = 1) : engine_(seed) {}

    explicit Random(PathSampleGenerator &sampler) : sampler_(&sampler) {}

    double uniform01() {
        if (sampler_) return sampler_->next1D();
        return std::generate_canonical<double, 64>(engine_);
    }

    double openUniform01() {
        double u = 0.0;
        do { u = uniform01(); } while (!(u > 0.0 && u < 1.0));
        return u;
    }

    double standardNormal() { return sampler_ ? std::sqrt(-2.0*std::log(openUniform01()))*std::cos(2.0*kPi*openUniform01()) : normal_(engine_); }
    std::uint64_t integer() { return engine_(); }

private:
    PathSampleGenerator *sampler_ = nullptr;
    std::mt19937_64 engine_;
    std::normal_distribution<double> normal_{0.0, 1.0};
};

} // namespace Tungsten::ConditionalGpis

