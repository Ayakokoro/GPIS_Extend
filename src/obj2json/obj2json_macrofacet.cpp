#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <rapidjson/prettywriter.h>
#include <rapidjson/stringbuffer.h>

#include "Version.hpp"

#include "io/CliParser.hpp"
#include "io/FileUtils.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "cameras/PinholeCamera.hpp"
#include "media/GaussianProcessMedium.hpp"
#include "math/GaussianProcess.hpp"
#include "thread/ThreadUtils.hpp"
#include "primitives/EmbreeUtil.hpp"
#include "primitives/TriangleMesh.hpp"

#ifdef OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/math/Transform.h>
#endif

//#define NANOVDB_USE_ZIP 1
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>

using namespace Tungsten;

namespace {

static const int OPT_VERSION = 0;
static const int OPT_HELP = 1;
static const int OPT_RESOLUTION = 2;
static const int OPT_PADDING = 3;
static const int OPT_FORMAT = 4;
static const int OPT_GRID_PREFIX = 5;
static const int OPT_MEDIUM_INDEX = 6;
static const int OPT_VOXEL_SIZE = 7;
static const int OPT_INCLUDE_DENSITY = 8;
static const int OPT_INCLUDE_NANOVDB = 9;
static const int OPT_INCLUDE_STATS = 10;
static const int OPT_PBRT_SCENE = 11;
static const int OPT_BAKED_JSON = 12;

struct Options {
    Path scenePath;
    Path outputPrefix;
    int resolution = 64;
    double paddingSigma = 3.0;
    std::string format = "both";
    std::string gridPrefix;
    int mediumIndex = 0;
    double voxelSize = 0.0;
    bool includeDensity = true;
    bool includeStats = false;
    bool verbose = false;
    bool emitPbrtScene = false;
    bool emitBakedJson = false;
};

struct ExportConfig {
    Box3d bounds;
    Vec3d voxelSize;
    int nx = 0;
    int ny = 0;
    int nz = 0;
};

struct MeshDump {
    std::vector<Vec3f> positions;
    std::vector<TriangleI> triangles;
};

bool primitiveUsesMedium(const Primitive &primitive, const Medium *medium);

std::string defaultBakedSceneName(const Path &outputPrefix)
{
    return outputPrefix.fileName().baseName().asString() + "-baked.json";
}

Path defaultBakedScenePath(const Scene &scene, const Path &outputPrefix)
{
    return scene.path().parent() / defaultBakedSceneName(outputPrefix);
}

std::string formatFloat(float value)
{
    std::ostringstream stream;
    stream << std::fixed << std::setprecision(6) << value;
    return stream.str();
}

std::string formatVec3(const Vec3f &value)
{
    return formatFloat(value.x()) + " " + formatFloat(value.y()) + " " + formatFloat(value.z());
}

Vec3f averagePrimitiveAlbedo(const Primitive &primitive)
{
    if (primitive.numBsdfs() <= 0)
        return Vec3f(0.2f);

    const auto &bsdf = const_cast<Primitive &>(primitive).bsdf(0);
    if (!bsdf || !bsdf->albedo())
        return Vec3f(0.2f);

    return max(bsdf->albedo()->average(), Vec3f(0.0f));
}

MeshDump dumpPrimitiveMesh(const Primitive &primitive)
{
    const TriangleMesh &mesh = const_cast<Primitive &>(primitive).asTriangleMesh();

    MeshDump result;
    result.positions.reserve(mesh.verts().size());
    result.triangles.reserve(mesh.tris().size());

    for (const Vertex &vertex : mesh.verts())
        result.positions.push_back(primitive.transform().transformPoint(vertex.pos()));

    result.triangles = mesh.tris();
    return result;
}

void writeTriangleMesh(std::ostream &out, const MeshDump &mesh, const std::string &indent, bool flipWinding = false)
{
    out << indent << "Shape \"trianglemesh\"\n";
    out << indent << "  \"integer indices\" [";
    for (size_t index = 0; index < mesh.triangles.size(); ++index) {
        const TriangleI &triangle = mesh.triangles[index];
        if (index > 0)
            out << " ";
        if (flipWinding)
            out << triangle.v0 << " " << triangle.v2 << " " << triangle.v1;
        else
            out << triangle.v0 << " " << triangle.v1 << " " << triangle.v2;
    }
    out << "]\n";
    out << indent << "  \"point3 P\" [";
    for (size_t index = 0; index < mesh.positions.size(); ++index) {
        if (index > 0)
            out << "    ";
        out << formatVec3(mesh.positions[index]);
        if (index + 1 != mesh.positions.size())
            out << "\n";
    }
    out << "]\n";
}

void writeGeneratedPbrtScene(
    const Scene &scene,
    const Options &options,
    const std::shared_ptr<GaussianProcessMedium> &gpMedium,
    const ExportConfig &config)
{
    auto camera = scene.camera();
    auto pinholeCamera = std::dynamic_pointer_cast<PinholeCamera>(camera);
    if (!pinholeCamera)
        throw std::runtime_error("Automatic PBRT scene generation currently requires a pinhole camera.");

    const Path pbrtPath = options.outputPrefix.setExtension(".pbrt");
    std::ofstream out(pbrtPath.asString(), std::ios::out | std::ios::trunc);
    if (!out)
        throw std::runtime_error("Unable to write generated PBRT scene to '" + pbrtPath.asString() + "'.");

    const Vec3f cameraPos = camera->pos();
    const Vec3f cameraLookAt = camera->lookAt();
    const Vec3f cameraUp = camera->up();
    const Vec2u resolution = camera->resolution();
    const Path nvdbFile = options.outputPrefix.fileName().setExtension(".nvdb");
    const Path exrFile = options.outputPrefix.fileName().setExtension(".exr");
    bool hasExplicitEmitter = false;
    for (const auto &primitive : scene.primitives()) {
        if (!primitiveUsesMedium(*primitive, gpMedium.get()) && primitive->isEmissive()) {
            hasExplicitEmitter = true;
            break;
        }
    }

    out << "Sampler \"halton\"\n";
    out << "  \"integer pixelsamples\" " << scene.rendererSettings().spp() << "\n";
    out << "LookAt " << formatVec3(cameraPos) << "\n";
    out << "  " << formatVec3(cameraLookAt) << "\n";
    out << "  " << formatVec3(cameraUp) << "\n\n";

    out << "Camera \"perspective\"\n";
    out << "  \"float fov\" " << formatFloat(pinholeCamera->fovDeg()) << "\n\n";

    out << "Film \"rgb\"\n";
    out << "  \"integer xresolution\" " << resolution.x() << "\n";
    out << "  \"integer yresolution\" " << resolution.y() << "\n";
    out << "  \"string filename\" \"" << exrFile.asString() << "\"\n\n";

    out << "Integrator \"volpath\"\n\n";
    out << "WorldBegin\n\n";

    if (!hasExplicitEmitter) {
        out << "AttributeBegin\n";
        out << "  LightSource \"infinite\" \"rgb L\" [ 1 1 1 ]\n";
        out << "AttributeEnd\n\n";
    }

    bool wroteSupportGeometry = false;
    MeshDump mediumBoundary;
    bool hasMediumBoundary = false;

    for (const auto &primitive : scene.primitives()) {
        if (primitiveUsesMedium(*primitive, gpMedium.get())) {
            if (!hasMediumBoundary) {
                mediumBoundary = dumpPrimitiveMesh(*primitive);
                hasMediumBoundary = true;
            }
            continue;
        }

        MeshDump mesh = dumpPrimitiveMesh(*primitive);
        out << "AttributeBegin\n";
        if (primitive->isEmissive()) {
            const Vec3f power = primitive->emission() ? primitive->emission()->average() : Vec3f(1.0f);
            out << "  AreaLightSource \"diffuse\" \"rgb L\" [ " << formatVec3(power) << " ]\n";
            writeTriangleMesh(out, mesh, "  ", true);
        } else {
            const Vec3f albedo = averagePrimitiveAlbedo(*primitive);
            out << "  Material \"diffuse\" \"rgb reflectance\" [ " << formatVec3(albedo) << " ]\n";
            writeTriangleMesh(out, mesh, "  ");
        }
        out << "AttributeEnd\n\n";
        wroteSupportGeometry = true;
    }

    if (!wroteSupportGeometry) {
        out << "AttributeBegin\n";
        out << "  Material \"diffuse\" \"rgb reflectance\" [ 0.2 0.2 0.2 ]\n";
        out << "  Shape \"trianglemesh\"  \"integer indices\" [0 2 1 0 3 2 ]\n";
        out << "    \"point3 P\" [1 0 -1    -1 0 -1    -1 0 1    1 0 1 ]\n";
        out << "AttributeEnd\n\n";
    }

    if (!hasMediumBoundary)
        throw std::runtime_error("Unable to find a primitive bound to the selected GP medium for PBRT scene generation.");

    out << "AttributeBegin\n";
    out << "  MakeNamedMedium \"macrofacet\" \"string type\" \"macrofacetvdb\"\n";
    out << "    \"string filename\" \"" << nvdbFile.asString() << "\"\n";
    out << "    \"string ndf\" \"gp\"\n";
    out << "    \"rgb albedo\" [ 1 1 1 ]\n";
    out << "  Material \"interface\"\n";
    out << "  MediumInterface \"macrofacet\" \"\"\n";
    writeTriangleMesh(out, mediumBoundary, "  ");
    out << "AttributeEnd\n";
}

rapidjson::Value makeVdbGridReference(
    rapidjson::Document::AllocatorType &allocator,
    const std::string &vdbFile,
    const std::string &gridName)
{
    return JsonObject{allocator,
        "type", "vdb",
        "file", vdbFile,
        "density_name", gridName,
        "normalize_size", false,
        "integration_method", "raymarching"
    };
}

void writeBakedSceneJson(
    const Scene &scene,
    const Options &options,
    const std::string &meanGridName,
    const std::string &varianceGridName)
{
    rapidjson::Document document;
    document.SetObject();
    rapidjson::Document::AllocatorType &allocator = document.GetAllocator();
    document.CopyFrom(scene.toJson(allocator), allocator);

    if (!document.HasMember("media") || !document["media"].IsArray())
        throw std::runtime_error("Serialized scene is missing a media array.");
    if (options.mediumIndex >= int(document["media"].Size()))
        throw std::runtime_error("Selected medium index is out of range during baked JSON export.");

    auto &media = document["media"][options.mediumIndex];
    if (!media.IsObject())
        throw std::runtime_error("Selected medium is not an object in serialized scene.");

    const std::string vdbFile = options.outputPrefix.setExtension(".vdb").absolute().asString();
    rapidjson::Value bakedGp(rapidjson::kObjectType);
    bakedGp.AddMember("id", 0, allocator);
    bakedGp.AddMember("type", rapidjson::Value("standard", allocator), allocator);
    bakedGp.AddMember("covariance_epsilon", 0, allocator);
    rapidjson::Value meanValue = JsonObject{allocator,
        "type", "tabulated",
        "grid", makeVdbGridReference(allocator, vdbFile, meanGridName)
    };
    bakedGp.AddMember("mean", std::move(meanValue), allocator);

    rapidjson::Value covariance(rapidjson::kObjectType);
    covariance.AddMember("type", rapidjson::Value("nonstationary", allocator), allocator);
    rapidjson::Value stationaryCov = JsonObject{allocator,
        "type", "squared_exponential",
        "lengthScale", 0.5,
        "sigma", 1.0
    };
    covariance.AddMember("cov", std::move(stationaryCov), allocator);
    covariance.AddMember("grid", makeVdbGridReference(allocator, vdbFile, varianceGridName), allocator);
    bakedGp.AddMember("covariance", std::move(covariance), allocator);

    media["gaussian_process"].Swap(bakedGp);

    if (document.HasMember("renderer") && document["renderer"].IsObject()) {
        auto &renderer = document["renderer"];
        const std::string baseName = options.outputPrefix.fileName().baseName().asString() + "-baked";
        renderer["output_file"] = rapidjson::Value((baseName + ".png").c_str(), allocator);
        renderer["hdr_output_file"] = rapidjson::Value((baseName + ".exr").c_str(), allocator);
    }

    const Path bakedPath = defaultBakedScenePath(scene, options.outputPrefix);
    std::ofstream out(bakedPath.asString(), std::ios::out | std::ios::trunc);
    if (!out)
        throw std::runtime_error("Unable to write baked scene JSON to '" + bakedPath.asString() + "'.");

    rapidjson::StringBuffer buffer;
    rapidjson::PrettyWriter<rapidjson::StringBuffer> writer(buffer);
    document.Accept(writer);
    out << buffer.GetString();
}

double densityFromSignedDistance(double signedDistance, double sigma)
{
    const double safeSigma = std::max(sigma, 1e-6);
    const double pdf = std::exp(-(signedDistance * signedDistance) / (2.0 * safeSigma * safeSigma)) /
        std::sqrt(2.0 * PI * safeSigma * safeSigma);
    const double cdf = 0.5 * (1.0 + std::erf(signedDistance / (safeSigma * std::sqrt(2.0))));
    return pdf / std::max(cdf, 1e-8);
}

std::shared_ptr<GPSampleNode> extractSampleNode(const std::shared_ptr<GaussianProcessMedium> &medium)
{
    return medium->_gp;
}

double localSigma(const GPSampleNode &gp, const Vec3d &point)
{
    constexpr double oneSigmaQuantile = 0.8413447460685429;
    return std::max(gp.noIntersectBound(point, oneSigmaQuantile), 1e-6);
}

double localAlpha(const GPSampleNode &gp, const Vec3d &point)
{
    return std::max(gp.compute_beckmann_roughness(point), 1e-6);
}

void prepareSceneForExport(Scene &scene)
{
    for (const auto &medium : scene.media())
        medium->prepareForRender();

    for (const auto &bsdf : scene.bsdfs())
        bsdf->prepareForRender();

    for (const auto &primitive : scene.primitives()) {
        primitive->prepareForRender();
        for (int index = 0; index < primitive->numBsdfs(); ++index)
            if (primitive->bsdf(index)->unnamed())
                primitive->bsdf(index)->prepareForRender();
    }
}

bool primitiveUsesMedium(const Primitive &primitive, const Medium *medium)
{
    return primitive.intMedium().get() == medium || primitive.extMedium().get() == medium;
}

ExportConfig buildConfig(const Scene &scene, const Options &options, const Medium *targetMedium)
{
    Box3d bounds;
    bool hasBounds = false;
    for (const auto &primitive : scene.primitives()) {
        if (targetMedium && !primitiveUsesMedium(*primitive, targetMedium))
            continue;

        Box3f primitiveBounds = primitive->bounds();
        if (primitiveBounds.empty())
            continue;
        if (!hasBounds) {
            bounds = Box3d(vec_conv<Vec3d>(primitiveBounds.min()), vec_conv<Vec3d>(primitiveBounds.max()));
            hasBounds = true;
        } else {
            bounds.grow(vec_conv<Vec3d>(primitiveBounds.min()));
            bounds.grow(vec_conv<Vec3d>(primitiveBounds.max()));
        }
    }

    if (!hasBounds && targetMedium) {
        for (const auto &primitive : scene.primitives()) {
            Box3f primitiveBounds = primitive->bounds();
            if (primitiveBounds.empty())
                continue;
            if (!hasBounds) {
                bounds = Box3d(vec_conv<Vec3d>(primitiveBounds.min()), vec_conv<Vec3d>(primitiveBounds.max()));
                hasBounds = true;
            } else {
                bounds.grow(vec_conv<Vec3d>(primitiveBounds.min()));
                bounds.grow(vec_conv<Vec3d>(primitiveBounds.max()));
            }
        }
    }

    if (!hasBounds)
        throw std::runtime_error("Scene does not contain any bounded primitives to infer export bounds.");

    const Vec3d diagonal = bounds.diagonal();
    const double longestAxis = std::max({diagonal.x(), diagonal.y(), diagonal.z(), 1e-3});

    ExportConfig config;
    config.bounds = bounds;

    if (options.voxelSize > 0.0) {
        config.voxelSize = Vec3d(options.voxelSize);
        config.nx = std::max(1, int(std::ceil(diagonal.x() / options.voxelSize)));
        config.ny = std::max(1, int(std::ceil(diagonal.y() / options.voxelSize)));
        config.nz = std::max(1, int(std::ceil(diagonal.z() / options.voxelSize)));
    } else {
        const double step = longestAxis / std::max(options.resolution, 2);
        config.voxelSize = Vec3d(step);
        config.nx = std::max(1, int(std::ceil(diagonal.x() / step)));
        config.ny = std::max(1, int(std::ceil(diagonal.y() / step)));
        config.nz = std::max(1, int(std::ceil(diagonal.z() / step)));
    }

    return config;
}

Options parseArgs(int argc, const char *argv[])
{
    CliParser parser("scene2nanovdb", "[options] scene.json output_prefix");
    parser.addOption('h', "help", "Prints this help text", false, OPT_HELP);
    parser.addOption('v', "version", "Prints version information", false, OPT_VERSION);
    parser.addOption('r', "resolution", "Approximate resolution along the longest axis", true, OPT_RESOLUTION);
    parser.addOption('p', "padding-sigma", "Padding multiplier for density export", true, OPT_PADDING);
    parser.addOption('f', "format", "Output format: vdb|nvdb|both", true, OPT_FORMAT);
    parser.addOption('g', "grid-prefix", "Optional prefix for auxiliary stats grids only", true, OPT_GRID_PREFIX);
    parser.addOption('m', "medium-index", "Medium index to export", true, OPT_MEDIUM_INDEX);
    parser.addOption('x', "voxel-size", "Explicit voxel size in world units", true, OPT_VOXEL_SIZE);
    parser.addOption('d', "include-density", "Bake Macrofacet-compatible density/alpha grids", false, OPT_INCLUDE_DENSITY);
    parser.addOption('n', "verbose", "Print export statistics", false, OPT_INCLUDE_NANOVDB);
    parser.addOption('s', "include-stats", "Also emit mean/variance grids for debugging", false, OPT_INCLUDE_STATS);
    parser.addOption('\0', "emit-pbrt", "Also emit a Macrofacet validation PBRT scene next to the NVDB output", false, OPT_PBRT_SCENE);
    parser.addOption('\0', "emit-baked-json", "Also emit an equivalent baked GPIS JSON scene next to the output", false, OPT_BAKED_JSON);

    parser.parse(argc, argv);

    if (parser.isPresent(OPT_VERSION)) {
        std::cout << "scene2nanovdb, version " << VERSION_STRING << std::endl;
        std::exit(0);
    }
    if (parser.operands().size() != 2 || parser.isPresent(OPT_HELP)) {
        parser.printHelpText();
        std::exit(0);
    }

    Options options;
    options.scenePath = Path(parser.operands()[0]);
    options.outputPrefix = Path(parser.operands()[1]);
    if (parser.isPresent(OPT_RESOLUTION))
        options.resolution = std::max(2, std::stoi(parser.param(OPT_RESOLUTION)));
    if (parser.isPresent(OPT_PADDING))
        options.paddingSigma = std::max(0.0, std::stod(parser.param(OPT_PADDING)));
    if (parser.isPresent(OPT_FORMAT))
        options.format = parser.param(OPT_FORMAT);
    if (parser.isPresent(OPT_GRID_PREFIX))
        options.gridPrefix = parser.param(OPT_GRID_PREFIX);
    if (parser.isPresent(OPT_MEDIUM_INDEX))
        options.mediumIndex = std::max(0, std::stoi(parser.param(OPT_MEDIUM_INDEX)));
    if (parser.isPresent(OPT_VOXEL_SIZE))
        options.voxelSize = std::max(0.0, std::stod(parser.param(OPT_VOXEL_SIZE)));
    options.includeDensity = true;
    options.includeStats = parser.isPresent(OPT_INCLUDE_STATS) || parser.isPresent(OPT_BAKED_JSON);
    options.verbose = parser.isPresent(OPT_INCLUDE_NANOVDB);
    options.emitPbrtScene = parser.isPresent(OPT_PBRT_SCENE);
    options.emitBakedJson = parser.isPresent(OPT_BAKED_JSON);
    return options;
}

} // namespace

int main(int argc, const char *argv[])
{
#ifndef OPENVDB_AVAILABLE
    std::cerr << "scene2nanovdb requires OPENVDB_AVAILABLE." << std::endl;
    return -1;
#else
    try {
        Options options = parseArgs(argc, argv);

        ThreadUtils::startThreads(1);
        EmbreeUtil::initDevice();
        openvdb::initialize();

        Path dstDir = options.outputPrefix.parent();
        if (!dstDir.empty() && !FileUtils::createDirectory(dstDir)) {
            std::cerr << "Unable to create target directory '" << dstDir << "'" << std::endl;
            return -1;
        }

        std::unique_ptr<Scene> scene(Scene::load(options.scenePath));
        if (!scene)
            throw std::runtime_error("Unable to load input scene.");
        scene->loadResources();
        prepareSceneForExport(*scene);

        if (options.mediumIndex >= int(scene->media().size()))
            throw std::runtime_error("Requested medium index is out of range.");

        auto gpMedium = std::dynamic_pointer_cast<GaussianProcessMedium>(scene->media()[options.mediumIndex]);
        if (!gpMedium)
            throw std::runtime_error("Selected medium is not a GaussianProcessMedium.");

        auto gpNode = extractSampleNode(gpMedium);
        if (!gpNode)
            throw std::runtime_error("Selected medium does not expose a GP sample node.");

        ExportConfig config = buildConfig(*scene, options, gpMedium.get());
        const std::string prefix = options.gridPrefix.empty() ? options.outputPrefix.baseName().asString() : options.gridPrefix;
        const std::string meanGridName = prefix + "_mean";
        const std::string varianceGridName = prefix + "_variance";

        auto gridTransform = openvdb::math::Transform::createLinearTransform(
            std::min({config.voxelSize.x(), config.voxelSize.y(), config.voxelSize.z()}));
        gridTransform->postTranslate(vec_conv<openvdb::Vec3d>(config.bounds.min()));

        auto meanGrid = openvdb::FloatGrid::create(0.0f);
        meanGrid->setName(meanGridName);
        meanGrid->setTransform(gridTransform);

        auto varianceGrid = openvdb::FloatGrid::create(0.0f);
        varianceGrid->setName(varianceGridName);
        varianceGrid->setTransform(gridTransform->copy());

        auto sdfGrid = openvdb::FloatGrid::create(0.0f);
        sdfGrid->setName("sdf");
        sdfGrid->setTransform(gridTransform->copy());

        auto densityGrid = openvdb::FloatGrid::create(0.0f);
        densityGrid->setName("density");
        densityGrid->setTransform(gridTransform->copy());

        auto alphaGrid = openvdb::FloatGrid::create(0.0f);
        alphaGrid->setName("alpha");
        alphaGrid->setTransform(gridTransform->copy());

        auto meanAccessor = meanGrid->getAccessor();
        auto varianceAccessor = varianceGrid->getAccessor();
        auto sdfAccessor = sdfGrid->getAccessor();
        auto densityAccessor = densityGrid->getAccessor();
        auto alphaAccessor = alphaGrid->getAccessor();

        const Derivative deriv = Derivative::None;

        std::cout << "nx: " << config.nx << std::endl;
        std::cout << "ny: " << config.ny << std::endl;
        std::cout << "nz: " << config.nz << std::endl;

        for (int ix = 0; ix < config.nx; ++ix) {
            std::cout << ix << "/" << config.nx << std::endl;
            for (int iy = 0; iy < config.ny; ++iy) {
                for (int iz = 0; iz < config.nz; ++iz) {
                    Vec3d local(
                        (double(ix) + 0.5) / config.nx,
                        (double(iy) + 0.5) / config.ny,
                        (double(iz) + 0.5) / config.nz);
                    Vec3d point = lerp(config.bounds.min(), config.bounds.max(), local);

                    Eigen::VectorXd meanVec = gpNode->mean(&point, &deriv, nullptr, Vec3d(0.0), 1);

                    const double mean = meanVec(0);
                    const double sigma = localSigma(*gpNode, point);
                    const double variance = sigma * sigma;

                    openvdb::Coord ijk(ix, iy, iz);
                    meanAccessor.setValue(ijk, float(mean));
                    varianceAccessor.setValue(ijk, float(variance));
                    sdfAccessor.setValue(ijk, float(mean));

                    if (options.includeDensity) {
                        const double alpha = localAlpha(*gpNode, point);
                        const double cutoff = options.paddingSigma * sigma;
                        if (sigma > 1e-5 && std::abs(mean) <= cutoff) {
                            densityAccessor.setValue(ijk, float(densityFromSignedDistance(mean, std::max(sigma, 1e-5))));
                            alphaAccessor.setValue(ijk, float(alpha));
                            sdfAccessor.setValue(ijk, float(mean));
                        }
                    }
                }
            }
        }

        const std::string base = options.outputPrefix.asString();
        const bool writeVdb = options.format == "vdb" || options.format == "both";
        const bool writeNvdb = options.format == "nvdb" || options.format == "both";

        if (writeVdb) {
            openvdb::GridPtrVec grids;
            if (options.includeStats) {
                grids.push_back(meanGrid);
                grids.push_back(varianceGrid);
            }
            if (options.includeDensity) {
                grids.push_back(densityGrid);
                grids.push_back(alphaGrid);
                grids.push_back(sdfGrid);
            } else {
                grids.push_back(sdfGrid);
            }
            openvdb::io::File file(base + ".vdb");
            file.write(grids);
            file.close();
        }

        if (writeNvdb) {
            const float voxel = float(std::min({config.voxelSize.x(), config.voxelSize.y(), config.voxelSize.z()}));
            const nanovdb::Vec3d origin(
                config.bounds.min().x(),
                config.bounds.min().y(),
                -config.bounds.max().z());

            nanovdb::tools::build::Grid<float> densityBuilder(0.0f, "density");
            nanovdb::tools::build::Grid<float> alphaBuilder(0.0f, "alpha");
            nanovdb::tools::build::Grid<float> sdfBuilder(0.0f, "sdf");
            nanovdb::tools::build::Grid<float> meanBuilder(0.0f, prefix + "_mean");
            nanovdb::tools::build::Grid<float> varianceBuilder(0.0f, prefix + "_variance");
            densityBuilder.setTransform(voxel, origin);
            alphaBuilder.setTransform(voxel, origin);
            sdfBuilder.setTransform(voxel, origin);
            meanBuilder.setTransform(voxel, origin);
            varianceBuilder.setTransform(voxel, origin);

            nanovdb::Map map;
            map.set(voxel, origin);
            nanovdb::Coord minCoord = nanovdb::Coord::Floor(
                map.applyInverseMapF(nanovdb::Vec3f(
                    float(config.bounds.min().x()), float(config.bounds.min().y()), float(-config.bounds.min().z()))));

            auto &densityBuildAccessor = densityBuilder.getAccessor();
            auto &alphaBuildAccessor = alphaBuilder.getAccessor();
            auto &sdfBuildAccessor = sdfBuilder.getAccessor();
            auto &meanBuildAccessor = meanBuilder.getAccessor();
            auto &varianceBuildAccessor = varianceBuilder.getAccessor();

            for (int ix = 0; ix < config.nx; ++ix) {
                std::cout << ix << "/" << config.nx << std::endl;
                for (int iy = 0; iy < config.ny; ++iy) {
                    for (int iz = 0; iz < config.nz; ++iz) {
                        openvdb::Coord vdbCoord(ix, iy, iz);
                        nanovdb::Coord ijk(minCoord.x() + ix, minCoord.y() + iy, minCoord.z() - iz);
                        const float meanValue = meanAccessor.getValue(vdbCoord);
                        const float varianceValue = varianceAccessor.getValue(vdbCoord);
                        const float sdfValue = sdfAccessor.getValue(vdbCoord);
                        if (options.includeStats) {
                            meanBuildAccessor.setValue(ijk, meanValue);
                            varianceBuildAccessor.setValue(ijk, varianceValue);
                        }
                        sdfBuildAccessor.setValue(ijk, sdfValue);
                        if (options.includeDensity) {
                            const float densityValue = densityAccessor.getValue(vdbCoord);
                            const float alphaValue = alphaAccessor.getValue(vdbCoord);
                            if (densityValue != 0.0f || alphaValue != 0.0f || sdfValue != 0.0f) {
                                densityBuildAccessor.setValue(ijk, densityValue);
                                alphaBuildAccessor.setValue(ijk, alphaValue);
                            }
                        }
                    }
                }
            }

            std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles;
            if (options.includeDensity) {
                handles.emplace_back(nanovdb::tools::createNanoGrid(alphaBuilder));
                handles.emplace_back(nanovdb::tools::createNanoGrid(sdfBuilder));
                handles.emplace_back(nanovdb::tools::createNanoGrid(densityBuilder));
            } else {
                handles.emplace_back(nanovdb::tools::createNanoGrid(sdfBuilder));
            }
            if (options.includeStats) {
                handles.emplace_back(nanovdb::tools::createNanoGrid(meanBuilder));
                handles.emplace_back(nanovdb::tools::createNanoGrid(varianceBuilder));
            }
            nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(base + ".nvdb", handles);
        }

        if (options.emitPbrtScene)
            writeGeneratedPbrtScene(*scene, options, gpMedium, config);

        if (options.emitBakedJson) {
            if (!writeVdb)
                throw std::runtime_error("Baked JSON export requires VDB output. Use format vdb or both.");
            writeBakedSceneJson(*scene, options, meanGridName, varianceGridName);
        }

        if (options.verbose) {
            std::cout << "Exported scene '" << options.scenePath << "' to prefix '" << options.outputPrefix << "'\n";
            std::cout << "Bounds: " << config.bounds << "\n";
            std::cout << "Resolution: " << config.nx << " x " << config.ny << " x " << config.nz << "\n";
            if (options.emitBakedJson)
                std::cout << "Baked scene: " << defaultBakedScenePath(*scene, options.outputPrefix) << "\n";
        }

        return 0;
    } catch (const std::exception &e) {
        std::cerr << "scene2nanovdb: " << e.what() << std::endl;
        return -1;
    }
#endif
}
