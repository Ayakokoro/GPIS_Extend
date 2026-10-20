#include "VdbMeanGrid.hpp"

#include "grids/VdbGrid.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/TextureCache.hpp"
#include "math/GaussianProcess.hpp"
#include "math/MathUtil.hpp"
#include "math/SparseConvolutionNoise.hpp"
#include "sampling/UniformSampler.hpp"

#include <openvdb/openvdb.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

int main(int argc, char **argv)
{
    if (argc != 2)
        return 1;
    int deviceCount = 0;
    const cudaError_t deviceStatus = cudaGetDeviceCount(&deviceCount);
    if (deviceStatus == cudaErrorNoDevice || (deviceStatus == cudaSuccess && deviceCount == 0))
        return 77;
    if (deviceStatus != cudaSuccess)
        return 1;

    openvdb::initialize();

    const Tungsten::Path scenePath(argv[1]);
    const Tungsten::JsonDocument document(scenePath);
    const Tungsten::Scene scene(scenePath.parent(), std::make_shared<Tungsten::TextureCache>());
    Tungsten::VdbGrid vdb;
    vdb.fromJson(document["media"][0U]["gaussian_process"]["mean"]["grid"], scene);
    vdb.loadResources();

    Tungsten::Gpu::HostMeanGrid grid;
    if (!Tungsten::Gpu::makeHostMeanGrid(vdb, 0.0f, 1.0f, grid)) {
        std::cerr << "Could not export shaderball VDB mean\n";
        return 1;
    }
    Tungsten::Gpu::Se1DRayBatcher batcher;
    if (batcher.setMeanGrid(grid) != cudaSuccess)
        return 1;

    std::vector<Tungsten::Gpu::RaySegmentInput> inputs;
    std::vector<float> expected;
    const auto bounds = vdb.bounds();
    const Tungsten::Vec3f span = bounds.max() - bounds.min();
    for (int z = 0; z < 5; ++z)
        for (int y = 0; y < 5; ++y)
            for (int x = 0; x < 5; ++x) {
                const Tungsten::Vec3f index = bounds.min() +
                    span * Tungsten::Vec3f((x + 1.0f) / 6.0f,
                                           (y + 1.0f) / 6.0f,
                                           (z + 1.0f) / 6.0f);
                const Tungsten::Vec3f p = vdb.naturalTransform() * index;
                const Tungsten::Vec3f endpoint = p + Tungsten::Vec3f(1e-6f, 0.0f, 0.0f);
                const float value = vdb.density(vdb.invNaturalTransform() * endpoint);
                if (!std::isfinite(value) || std::abs(value) < 0.005f)
                    continue;
                Tungsten::Gpu::RaySegmentInput ray{};
                ray.origin = {p.x(), p.y(), p.z()};
                ray.direction = {1, 0, 0};
                ray.nearT = 0;
                ray.farT = 1e-6f;
                ray.jitter = 0.2f;
                ray.noiseSeed = 123;
                ray.normalSeed = 456;
                ray.firstScatter = 1;
                inputs.push_back(ray);
                expected.push_back(value);
            }
    if (inputs.empty())
        return 1;

    std::vector<Tungsten::Gpu::RaySegmentOutput> outputs(inputs.size());
    Tungsten::Gpu::Se1DRayParameters parameters;
    parameters.amplitude = 0;
    if (batcher.trace(inputs.data(), outputs.data(), inputs.size(), parameters) != cudaSuccess)
        return 1;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        if (outputs[i].status != Tungsten::Gpu::RaySegmentStatus::Ok || outputs[i].hit ||
            !std::isfinite(outputs[i].value) || std::abs(outputs[i].value - expected[i]) > 2e-4f) {
            std::cerr << "VDB mean mismatch at ray " << i << ": CPU " << expected[i]
                      << ", CUDA " << outputs[i].value << '\n';
            return 1;
        }
    }

    auto sampledGrid = std::make_shared<Tungsten::VdbGrid>();
    sampledGrid->fromJson(document["media"][0U]["gaussian_process"]["mean"]["grid"], scene);
    sampledGrid->loadResources();
    auto covariance = std::make_shared<Tungsten::SquaredExponentialCovariance>();
    covariance->fromJson(document["media"][0U]["gaussian_process"]["covariance"], scene);
    auto gp = std::make_shared<Tungsten::GaussianProcess>(
        std::make_shared<Tungsten::TabulatedMean>(sampledGrid), covariance);
    Tungsten::SparseConvolutionNoiseRealization noise(
        gp, Tungsten::GPCorrelationContext::RenewalPlus, 17429u, 3.0f,
        false, false, true, Tungsten::SparseConv1DSamplingScheme::UNI,
        false, false, 0.0f);
    parameters.lengthScale = covariance->lengthScale();
    parameters.amplitude = covariance->sparseConvNoiseAmplitude();
    parameters.kernelRadius = covariance->splattingKernelRadius(true, 1.0f);
    parameters.impulseDensity = 3.0f;
    const Tungsten::Vec3f directions[] = {
        Tungsten::Vec3f(1.0f, 0.0f, 0.0f),
        Tungsten::Vec3f(-0.3f, 0.8f, 0.52f).normalized(),
        Tungsten::Vec3f(0.27f, -0.4f, 0.86f).normalized(),
        Tungsten::Vec3f(-0.81f, -0.21f, -0.55f).normalized()
    };
    const std::vector<Tungsten::Gpu::RaySegmentInput> baseInputs = inputs;
    inputs.clear();
    for (const auto &base : baseInputs)
        for (const auto &direction : directions) {
            auto ray = base;
            ray.direction = {direction.x(), direction.y(), direction.z()};
            inputs.push_back(ray);
        }
    outputs.resize(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto &ray = inputs[i];
        const Tungsten::Vec4u pixelSampleSegment(
            static_cast<unsigned int>((i * 17) % 128),
            static_cast<unsigned int>((i * 29) % 128),
            static_cast<unsigned int>(i % 2),
            static_cast<unsigned int>(i % 3));
        const unsigned int sceneSeed = 12345u;
        ray.noiseSeed = 17429u + Tungsten::MathUtil::xxhash32(pixelSampleSegment) +
                        Tungsten::MathUtil::xxhash32(sceneSeed);
        ray.normalSeed = Tungsten::MathUtil::xxhash32(pixelSampleSegment) +
                         Tungsten::MathUtil::xxhash32(sceneSeed) + 1u;
    }
    if (batcher.trace(inputs.data(), outputs.data(), inputs.size(), parameters) != cudaSuccess)
        return 1;
    float maximumNoiseDifference = 0.0f;
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        const auto &ray = inputs[i];
        const Tungsten::Vec3f direction(ray.direction.x, ray.direction.y, ray.direction.z);
        const Tungsten::Vec3f p(ray.origin.x + ray.farT * direction.x(),
                                ray.origin.y + ray.farT * direction.y(),
                                ray.origin.z + ray.farT * direction.z());
        Tungsten::RayInfo info{};
        info.pixelSampleSegment = Tungsten::Vec4u(
            static_cast<unsigned int>((i * 17) % 128),
            static_cast<unsigned int>((i * 29) % 128),
            static_cast<unsigned int>(i % 2),
            static_cast<unsigned int>(i % 3));
        info.sceneSeed = 12345u;
        info.t = 0.0f;
        Tungsten::UniformSampler sampler;
        int gpId = 0;
        const float cpu = noise.evaluateValue(p, ray.farT, direction,
                                              info, sampler, gpId);
        maximumNoiseDifference = std::max(maximumNoiseDifference,
                                           std::abs(cpu - outputs[i].value));
        if (outputs[i].status != Tungsten::Gpu::RaySegmentStatus::Ok || outputs[i].hit ||
            std::abs(cpu - outputs[i].value) > 2e-4f) {
            std::cerr << "Shaderball GP value mismatch at ray " << i << ": CPU "
                      << cpu << ", CUDA " << outputs[i].value << '\n';
            return 1;
        }
    }
    std::cout << "Maximum shaderball GP value difference: " << maximumNoiseDifference << '\n';
    std::cout << "Compared " << inputs.size() << " shaderball VDB samples\n";
    return 0;
}
