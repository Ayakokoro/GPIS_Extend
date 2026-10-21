#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "media/GpisSdfGrid.hpp"
#include <openvdb/io/File.h>
#include <openvdb/tools/Dense.h>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>
#include <tbb/parallel_for.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

using namespace Tungsten;
using V = Eigen::Vector3d;
namespace fs = std::filesystem;

namespace {
std::string jsonText(const rapidjson::Value &value)
{
    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    return buffer.GetString();
}

rapidjson::Value vectorJson(const V &v, rapidjson::Document::AllocatorType &allocator)
{
    rapidjson::Value array(rapidjson::kArrayType);
    for (int a = 0; a < 3; ++a) array.PushBack(v[a], allocator);
    return array;
}

struct Errors {
    std::vector<double> absolute;
    double squared = 0, gradientSquared = 0;
    void add(const SdfJet &reference, const ConditionalGpis::MeanJet &baked)
    {
        const double e = std::abs(reference.value - baked.value);
        const double g = (reference.gradient - baked.gradient).squaredNorm();
        if (!std::isfinite(e) || !std::isfinite(g))
            throw std::runtime_error("Non-finite validation query");
        absolute.push_back(e);
        squared += e*e;
        gradientSquared += g;
    }
    rapidjson::Value json(rapidjson::Document::AllocatorType &allocator)
    {
        rapidjson::Value result(rapidjson::kObjectType);
        result.AddMember("samples", uint64_t(absolute.size()), allocator);
        if (absolute.empty()) return result;
        std::sort(absolute.begin(), absolute.end());
        result.AddMember("value_rmse", std::sqrt(squared/absolute.size()), allocator);
        result.AddMember("value_p99_abs", absolute[size_t(0.99*(absolute.size()-1))], allocator);
        result.AddMember("value_max_abs", absolute.back(), allocator);
        result.AddMember("gradient_vector_rmse", std::sqrt(gradientSquared/absolute.size()), allocator);
        return result;
    }
};
}

int main(int argc, char **argv)
{
    if (argc < 3 || argc > 5) {
        std::cerr << "Usage: bake_sdf_mean scene.json output.vdb [voxel_size=0.01] [medium=gp]\n";
        return 2;
    }
    try {
        const auto start = std::chrono::steady_clock::now();
        const fs::path input = fs::absolute(argv[1]), output = fs::absolute(argv[2]);
        const double spacing = argc > 3 ? std::stod(argv[3]) : 0.01;
        const std::string mediumName = argc > 4 ? argv[4] : "gp";
        if (!(spacing > 0) || !std::isfinite(spacing))
            throw std::runtime_error("voxel_size must be finite and positive");
        if (fs::exists(output) || fs::exists(output.string()+".json"))
            throw std::runtime_error("Output already exists; choose a new output filename");

        JsonDocument document(Path(input.generic_string()));
        JsonPtr selected;
        const auto media = document.getRequiredMember("media");
        for (unsigned i = 0; i < media.size(); ++i) {
            std::string name;
            media[i].getField("name", name);
            if (name == mediumName) {
                if (selected) throw std::runtime_error("Duplicate medium name");
                selected = media[i];
            }
        }
        if (!selected) throw std::runtime_error("Requested medium was not found");
        const auto meanJson = selected.getRequiredMember("gaussian_process").getRequiredMember("mean");
        if (meanJson.castField<std::string>("type") != "procedural")
            throw std::runtime_error("This baker expects a procedural SDF mean");
        Scene scene;
        const auto mean = scene.fetchMeanFunction(meanJson);
        mean->loadResources();
        const auto *queries = mean->sdfQueries();
        if (!queries) throw std::runtime_error("Mean does not expose SDF queries");

        // Cover all cube boundaries carrying this medium, in world coordinates.
        // Use the renderer's JSON transform parser, including rotation/scale.
        V minimum = V::Constant(std::numeric_limits<double>::infinity());
        V maximum = -minimum;
        const auto primitives = document.getRequiredMember("primitives");
        for (unsigned i = 0; i < primitives.size(); ++i) {
            const auto interior = primitives[i]["int_medium"];
            if (!interior || !interior.isString() || interior.cast<std::string>() != mediumName)
                continue;
            if (primitives[i].castField<std::string>("type") != "cube")
                throw std::runtime_error("Automatic bake bounds require cube medium boundaries");
            Mat4f transform;
            primitives[i].getField("transform", transform);
            for (int mask = 0; mask < 8; ++mask) {
                const Vec3f p = transform*Vec3f(mask&1 ? 0.5f : -0.5f,
                                               mask&2 ? 0.5f : -0.5f,
                                               mask&4 ? 0.5f : -0.5f);
                const V v(p.x(), p.y(), p.z());
                minimum = minimum.cwiseMin(v);
                maximum = maximum.cwiseMax(v);
            }
        }
        if (!minimum.allFinite() || !maximum.allFinite() ||
            !((maximum-minimum).array() > 0).all())
            throw std::runtime_error("No finite, nondegenerate cube domain for this medium");

        // VdbGrid clamps two voxel centers inward at both ends. Three padding
        // cells keep all medium queries strictly inside that unclamped domain.
        constexpr int padding = 3;
        const V origin = minimum - V::Constant(padding*spacing);
        Eigen::Vector3i dimensions;
        size_t count = 1;
        for (int a = 0; a < 3; ++a) {
            const double n = std::ceil((maximum[a]-minimum[a])/spacing) + 2*padding + 1;
            if (!std::isfinite(n) || n >= std::numeric_limits<int>::max() ||
                n > std::numeric_limits<size_t>::max()/sizeof(float)/count)
                throw std::runtime_error("Grid dimensions overflow; increase voxel_size");
            dimensions[a] = int(n);
            count *= size_t(dimensions[a]);
        }
        std::cout << "World domain: " << minimum.transpose() << " to " << maximum.transpose()
                  << "\nVoxel size: " << spacing << ", dimensions: " << dimensions.transpose()
                  << ", dense float storage: " << count*sizeof(float)/(1024.0*1024.0)
                  << " MiB\nSampling the original SDF mean..." << std::endl;

        openvdb::initialize();
        const openvdb::Coord last(dimensions.x()-1, dimensions.y()-1, dimensions.z()-1);
        const openvdb::CoordBBox box(openvdb::Coord(0), last);
        openvdb::tools::Dense<float, openvdb::tools::LayoutXYZ> dense(box);
        tbb::parallel_for(0, dimensions.z(), [&](int z) {
            for (int y = 0; y < dimensions.y(); ++y)
                for (int x = 0; x < dimensions.x(); ++x) {
                    const V p = origin + spacing*V(x,y,z);
                    const float value = float(queries->evaluateJet(p).value);
                    if (!std::isfinite(value)) throw std::runtime_error("Non-finite baked voxel");
                    dense.setValue(openvdb::Coord(x,y,z), value);
                }
        });
        auto grid = openvdb::FloatGrid::create(0.0f);
        grid->setName("sdf");
        auto transform = openvdb::math::Transform::createLinearTransform(spacing);
        transform->postTranslate(openvdb::Vec3d(origin.x(),origin.y(),origin.z()));
        grid->setTransform(transform);
        // Preserve the sampled mean; no distance reinitialization or narrow-band
        // truncation, which would change smoothed CSG and conditional statistics.
        openvdb::tools::copyFromDense(dense, *grid, 0.0f);
        auto accessor = grid->getAccessor();
        accessor.setValueOn(openvdb::Coord(0), dense.getValue(openvdb::Coord(0)));
        accessor.setValueOn(last, dense.getValue(last));
        grid->insertMeta("source_scene", openvdb::StringMetadata(input.generic_string()));
        grid->insertMeta("source_medium", openvdb::StringMetadata(mediumName));
        grid->insertMeta("field_semantics", openvdb::StringMetadata("world-space GP mean samples"));
        fs::create_directories(output.parent_path());
        std::cout << "Writing " << output.generic_string() << std::endl;
        openvdb::io::File file(output.string());
        file.write(openvdb::GridPtrVec{grid});
        file.close();
        grid.reset();

        // Reload through the actual renderer adapter to validate file I/O,
        // float transforms, clamping and interpolation, not a separate sampler.
        rapidjson::Document configuration;
        configuration.Parse(R"({"type":"tabulated","volume":false,"scale":1,"offset":0,"grid":{
            "type":"vdb","file":"","density_name":"sdf","density_scale":1,"emission_name":"",
            "normalize_size":false,"request_sdf":false,"interpolate":"linear",
            "integration_method":"raymarching"}})");
        auto &ca = configuration.GetAllocator();
        configuration["grid"]["file"].SetString(output.generic_string().c_str(), ca);
        JsonDocument bakedDocument(Path("baked-mean.json"), jsonText(configuration));
        const auto bakedMean = std::dynamic_pointer_cast<TabulatedMean>(scene.fetchMeanFunction(bakedDocument));
        bakedMean->loadResources();
        GpisSdfGrid baked;
        const double bound = baked.initialize(*bakedMean, *std::dynamic_pointer_cast<VdbGrid>(bakedMean->grid()));
        const V lower = baked.worldToIndex*minimum+baked.translation;
        const V upper = baked.worldToIndex*maximum+baked.translation;
        for (int a = 0; a < 3; ++a)
            if (!(lower[a] > baked.data.clampMin[a] && upper[a] < baked.data.clampMax[a]))
                throw std::runtime_error("Bake padding does not cover the medium domain");

        Errors uniform, nearSurface, nodes;
        std::uint64_t state = 0x713b5ab43cf01631ULL;
        const auto random = [&]() {
            state ^= state<<13; state ^= state>>7; state ^= state<<17;
            return double(state>>11)*0x1p-53;
        };
        double sigma = 0;
        selected.getRequiredMember("gaussian_process").getRequiredMember("covariance").getField("sigma", sigma);
        const double band = std::max(3*std::abs(sigma), 3*spacing);
        for (int i = 0; i < 100000; ++i) {
            V p;
            for (int a = 0; a < 3; ++a) p[a] = minimum[a]+random()*(maximum[a]-minimum[a]);
            const auto reference = queries->evaluateJet(p);
            const auto q = baked.evaluate(p);
            uniform.add(reference, q);
            if (std::abs(reference.value) <= band) nearSurface.add(reference, q);
            if (i < 10000) {
                for (int a = 0; a < 3; ++a)
                    p[a] = origin[a]+spacing*(padding+int(random()*(dimensions[a]-2*padding)));
                const auto ref = queries->evaluateJet(p);
                const auto v = baked.evaluate(p);
                nodes.add(ref,v);
                if (std::abs(ref.value-v.value) > 2e-5*std::max(1.0,std::abs(ref.value)))
                    throw std::runtime_error("Reloaded voxel values do not match the source mean");
            }
        }

        rapidjson::Document report(rapidjson::kObjectType);
        auto &ra = report.GetAllocator();
        report.AddMember("source_scene", rapidjson::Value(input.generic_string().c_str(),ra),ra);
        report.AddMember("source_mean", mean->toJson(ra), ra);
        report.AddMember("baked_mean", rapidjson::Value(configuration,ra),ra);
        report.AddMember("voxel_size", spacing, ra);
        report.AddMember("padding_voxels", padding, ra);
        report.AddMember("dimensions", vectorJson(dimensions.cast<double>(),ra),ra);
        report.AddMember("world_domain_min", vectorJson(minimum,ra),ra);
        report.AddMember("world_domain_max", vectorJson(maximum,ra),ra);
        report.AddMember("grid_origin", vectorJson(origin,ra),ra);
        report.AddMember("dense_bytes", uint64_t(count*sizeof(float)),ra);
        report.AddMember("vdb_bytes", uint64_t(fs::file_size(output)),ra);
        report.AddMember("automatic_gradient_bound", bound,ra);
        report.AddMember("near_surface_band", band,ra);
        report.AddMember("uniform_error", uniform.json(ra),ra);
        report.AddMember("near_surface_error", nearSurface.json(ra),ra);
        report.AddMember("voxel_node_error", nodes.json(ra),ra);
        report.AddMember("seconds", std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count(),ra);
        std::ofstream summary(output.string()+".json");
        summary << jsonText(report) << '\n';
        if (!summary) throw std::runtime_error("Unable to write bake report");
        std::cout << jsonText(report) << "\nBake and reload validation passed.\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "SDF mean bake failed: " << error.what() << '\n';
        return 1;
    }
}
