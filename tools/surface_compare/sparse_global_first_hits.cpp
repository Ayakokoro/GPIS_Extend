// Query production global 3D sparse-convolution distance samples on fixed exterior rays.
#include "io/Scene.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include <nlohmann/json.hpp>
#include <openvdb/openvdb.h>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace Tungsten;
using Json = nlohmann::json;

namespace {
Vec3f vector3(const Json &v) {
    const auto xyz = v.get<std::array<float, 3>>();
    return Vec3f(xyz[0], xyz[1], xyz[2]);
}

struct Result {
    std::string status;
    float distance = 0;
    double originValue = 0, left = 0, right = 0, leftValue = 0, rightValue = 0;
    double initialFraction = 0;
    bool cutoff = false;
};

// Replay the first sign bracket on the already selected field. This diagnostic
// cannot affect sampleDistance's return, sampler, or field state.
void inspectZero(Result &r, const Ray &ray, const Medium::MediumState &state,
                 float jitter, float step) {
    auto context = std::static_pointer_cast<GPContextSparseConvNoise>(state.gpContext);
    auto noise = context->noise;
    UniformSampler random;
    int gpId;
    const Vec3d origin(ray.pos()), direction(ray.dir());
    auto value = [&](double t) {
        return double(noise.evaluateValue(Vec3f(origin + t*direction), t, ray.dir(),
                                          state.info, random, gpId));
    };
    r.originValue = value(0);
    r.leftValue = r.originValue;
    for (double t = step*jitter; t < ray.farT(); t += step) {
        r.right = t;
        r.rightValue = value(t);
        if ((r.leftValue < 0) != (r.rightValue < 0)) {
            double fraction = r.initialFraction = r.leftValue / (r.leftValue-r.rightValue);
            while ((value(lerp(t-step, t, fraction)) < 0) != (r.leftValue < 0)) {
                fraction *= 0.9;
                if (fraction <= 0.01) {
                    r.cutoff = true;
                    break;
                }
            }
            return;
        }
        r.left = t;
        r.leftValue = r.rightValue;
    }
}
}

int main(int argc, char **argv) try {
    if (argc != 4 && argc != 5)
        throw std::runtime_error("Usage: sparse_global_first_hits scene.json rays.json hits.csv [zero_checks.csv]");
    std::ifstream sceneInput(argv[1]), rayInput(argv[2]);
    const Json sceneConfig = Json::parse(sceneInput), spec = Json::parse(rayInput);
    const auto &config = sceneConfig.at("media").at(0);
    if (sceneConfig.at("media").size() != 1 || config.at("type") != "sparse_conv_noise" ||
            config.at("correlation_context") != "global" || config.at("1D_sampling") != false ||
            config.at("single_realization") != false)
        throw std::runtime_error("Expected an ensemble of global 3D sparse-convolution fields");
    openvdb::initialize();
    const std::unique_ptr<Scene> scene(Scene::load(Path(argv[1]).absolute()));
    const auto medium = std::dynamic_pointer_cast<SparseConvolutionNoiseMedium>(scene->media().at(0));
    medium->loadResources();
    medium->prepareForRender();

    const int samples = spec.at("samples");
    const uint32 seed = spec.at("seed");
    if (samples < 1) throw std::runtime_error("Expected at least one draw per ray");
    std::ofstream out(argv[3]);
    out.exceptions(std::ios::failbit | std::ios::badbit);
    out << "ray,realization,distance,status\n" << std::setprecision(12);
    std::ofstream checks;
    if (argc == 5) {
        checks.exceptions(std::ios::failbit | std::ios::badbit);
        checks.open(argv[4]);
        checks << "ray,realization,origin_value,left_t,right_t,left_value,right_value,initial_fraction,cutoff\n"
               << std::setprecision(12);
    }
    for (uint32 index = 0; index < spec.at("rays").size(); ++index) {
        const auto &r = spec.at("rays").at(index);
        const uint32 rayId = r.value("ray_id", index);
        if (r.contains("start_derivative"))
            throw std::runtime_error("Global sparse mode requires an observed path for a surface birth");
        const std::string name = r.at("name");
        if (name.empty() || name.find_first_of(",\"\r\n") != std::string::npos)
            throw std::runtime_error("Invalid CSV ray label");
        const Vec3f origin = vector3(r.at("origin")), dir = vector3(r.at("direction"));
        const float length = r.at("length");
        if (!std::isfinite(length) || length <= 0 || !std::isfinite(dir.length()) || dir.lengthSq() == 0)
            throw std::runtime_error("Invalid ray extent or direction");
        const Ray ray(origin, dir.normalized(), 0, length);
        const auto ids = r.value("realization_ids", std::vector<int>());
        const int count = r.contains("realization_ids") ? int(ids.size()) : samples;
        const float step = std::min(config.at("step_size").get<float>(),
                                    length/config.at("min_step").get<unsigned>());
        std::vector<Result> results(count);
        #pragma omp parallel for schedule(static)
        for (int index = 0; index < count; ++index) {
            const int i = ids.empty() ? index : ids[index];
            IndependentPathSampler sampler(seed, false);
            sampler.startPath(rayId, i);
            Medium::MediumState state{};
            state.reset();
            state.info = {Vec4u(rayId, 0u, unsigned(i), 0u), seed, 0.0f};
            Medium::DistanceDiagnostic diagnostic;
            state.distanceDiagnostic = &diagnostic;
            MediumSample sample;
            auto &result = results[index];
            if (!medium->sampleDistance(sampler, ray, state, sample)) {
                result.status = diagnostic.failure;
            } else if (sample.exited) {
                result.status = "escaped";
            } else {
                result.status = "hit";
                result.distance = sample.t;
                if (argc == 5 && sample.t == 0) {
                    sampler.startPath(rayId, i);
                    inspectZero(result, ray, state, sampler.next1D(), step);
                }
            }
        }
        for (int index = 0; index < count; ++index) {
            const int i = ids.empty() ? index : ids[index];
            const auto &result = results[index];
            out << name << ',' << i << ',';
            if (result.status == "hit") {
                if (!(result.distance >= 0 && result.distance <= length))
                    throw std::runtime_error("Production medium reported invalid distance");
                out << result.distance;
                if (argc == 5 && result.distance == 0)
                    checks << name << ',' << i << ',' << result.originValue << ',' << result.left << ','
                           << result.right << ',' << result.leftValue << ',' << result.rightValue << ','
                           << result.initialFraction << ',' << result.cutoff << '\n';
            }
            out << ',' << result.status << '\n';
        }
        out.flush();
        std::cout << name << " finished " << count << " draws\n" << std::flush;
    }
    return 0;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
}
