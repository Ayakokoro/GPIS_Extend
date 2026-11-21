// Link against bounded-gpis' core, using its tungsten-intersection-test build flags.
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "media/PraticalGaussianProcessMedium.hpp"
#include "sampling/UniformPathSampler.hpp"
#include <openvdb/openvdb.h>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

using namespace Tungsten;

// A different fully mixed 64-bit PCG initial state for every ray and realization.
static uint64_t mix(uint64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

int main(int argc, char **argv)
{
    try {
        if (argc != 4)
            throw std::runtime_error("usage: bounded_first_hit_probe scene.json rays.json output.csv");
        openvdb::initialize();
        const JsonDocument spec{Path(argv[2])};
        const auto seed = spec.castField<uint64>("seed");
        const auto samples = spec.castField<uint32>("samples");
        const auto rays = spec.getRequiredMember("rays");
        if (samples == 0 || rays.size() == 0)
            throw std::runtime_error("expected at least one ray and one realization");
        std::unique_ptr<Scene> scene(Scene::load(Path(argv[1])));
        if (scene->media().size() != 1)
            throw std::runtime_error("expected one shaderball medium");
        auto medium = std::dynamic_pointer_cast<PraticalGaussianProcessMedium>(scene->media().front());
        if (!medium)
            throw std::runtime_error("expected pratical_gaussian_process");
        medium->loadResources();
        medium->prepareForRender();
        auto gp = std::dynamic_pointer_cast<GaussianProcess>(medium->_gp);
        if (!gp)
            throw std::runtime_error("expected one standard Gaussian process");

        std::ofstream output(argv[3]);
        output.exceptions(std::ios::failbit | std::ios::badbit);
        output << "ray,realization,distance\n" << std::setprecision(10);
        for (unsigned r = 0; r < rays.size(); ++r) {
            const auto row = rays[r];
            const auto name = row.castField<std::string>("name");
            if (name.empty() || name.find_first_of(",\"\r\n") != std::string::npos)
                throw std::runtime_error("expected a simple nonempty ray name");
            const auto origin = row.castField<Vec3f>("origin");
            const auto direction = row.castField<Vec3f>("direction").normalized();
            const auto length = row.castField<float>("length");
            if (!(length > 0 && std::isfinite(length) && std::isfinite(direction.length())))
                throw std::runtime_error("expected finite ray length and direction");
            const Ray ray(origin, direction, 0.0f, length);
            std::shared_ptr<GPContextPratical1D> birth;
            double slope = 0;
            if (row.getField("start_derivative", slope)) {
                const auto ro = vec_conv<Vec3d>(origin);
                const auto rd = vec_conv<Vec3d>(direction).normalized();
                const auto meanGrad = gp->_mean->dmean_da(ro);
                std::cout << std::setprecision(12) << name << " mean directional derivative "
                          << meanGrad.dot(rd) << ", prescribed field derivative " << slope << '\n';
                birth = std::make_shared<GPContextPratical1D>();
                birth->intersection = ro;
                birth->gradient = meanGrad + (slope - meanGrad.dot(rd)) * rd;
            }
            for (unsigned i = 0; i < samples; ++i) {
                UniformSampler engine(mix(mix(seed) ^ ((uint64_t(r) << 32) | i)));
                UniformPathSampler sampler(engine);
                Medium::MediumState state{};
                state.reset();
                state.gpContext = birth;
                double t = length;
                const bool hit = medium->intersectGP(sampler, ray, state, t);
                output << name << ',' << i << ',';
                if (hit) {
                    if (!(t >= 0 && t <= length && std::isfinite(t)))
                        throw std::runtime_error("intersection outside requested interval");
                    output << t;
                }
                output << '\n';
            }
        }
        output.close();
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
