#ifndef INDEPENDENTPATHSAMPLER_HPP_
#define INDEPENDENTPATHSAMPLER_HPP_

#include "PathSampleGenerator.hpp"
#include "UniformSampler.hpp"
#include "math/MathUtil.hpp"
#include <sobol/sobol.h>

namespace Tungsten {

// The entire random state belongs to one (scene, pixel, absolute sample) tuple.
// Sobol's scramble stays constant per pixel; its supplemental PCG is reset per
// sample as well, including boolean/discrete and uniformGenerator() draws.
class IndependentPathSampler final : public PathSampleGenerator {
    uint32 _seed, _scramble = 0, _index = 0, _dimension = 0;
    bool _sobol;
    UniformSampler _uniform;

public:
    IndependentPathSampler(uint32 seed, bool sobol) : _seed(seed), _sobol(sobol) {}

    void startPath(uint32 pixelId, uint32 sample) override
    {
        _scramble = MathUtil::hash32(_seed ^ 0x9E3779B9u) ^ MathUtil::hash32(pixelId);
        _index = sample;
        _dimension = 0;
        const uint64 seed = (uint64(_scramble) << 32) |
            MathUtil::hash32(sample ^ 0x85EBCA6Bu);
        _uniform = UniformSampler(seed);
    }
    void advancePath() override {}
    bool nextBoolean(float pTrue) override { return _uniform.next1D() < pTrue; }
    int nextDiscrete(int choices) override { return int(_uniform.next1D()*choices); }
    float next1D() override
    {
        if (!_sobol || _dimension >= 1024)
            return _uniform.next1D();
        const uint32 index = (_index & ~0xFFu) | ((_index + _scramble) & 0xFFu);
        return BitManip::normalizedUint(sobol::sample(index, _dimension++, _scramble));
    }
    Vec2f next2D() override
    {
        const float a = next1D(), b = next1D();
        return Vec2f(a, b);
    }
    UniformSampler &uniformGenerator() override { return _uniform; }
    void saveState(OutputStreamHandle &out) override
    {
        FileUtils::streamWrite(out, _seed);
        FileUtils::streamWrite(out, _scramble);
        FileUtils::streamWrite(out, _index);
        FileUtils::streamWrite(out, _dimension);
        _uniform.saveState(out);
    }
    void loadState(InputStreamHandle &in) override
    {
        FileUtils::streamRead(in, _seed);
        FileUtils::streamRead(in, _scramble);
        FileUtils::streamRead(in, _index);
        FileUtils::streamRead(in, _dimension);
        _uniform.loadState(in);
    }
};

}
#endif
