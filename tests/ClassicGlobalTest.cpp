#include "media/ClassicGlobalMedium.hpp"
#include "math/ClassicGlobalMath.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#if OPENVDB_AVAILABLE
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#endif
#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>

using namespace Tungsten;
namespace {
constexpr double Pi = 3.14159265358979323846;
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance, const char *message)
{
    if (!(std::abs(actual-expected) <= tolerance)) {
        std::cerr << message << ": " << actual << " vs " << expected << '\n';
        throw std::runtime_error(message);
    }
}
std::shared_ptr<ClassicGlobalMedium> makeMedium(const std::string &mean, const std::string &extra = "",
        const std::string &material = R"({"force_unit_fresnel":true})")
{
    const std::string json = R"({"type":"classic_global","gaussian_process":{
        "type":"standard","mean":)" + mean + R"(,
        "covariance":{"type":"squared_exponential","sigma":1,"lengthScale":1}},
        "material":)" + material + extra + "}";
    JsonDocument document(Path("classic-global-test.json"), json);
    Scene scene;
    auto medium = std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(document));
    require(bool(medium), "factory registration");
    medium->loadResources();
    medium->prepareForRender();
    return medium;
}
void coefficients()
{
    close(ClassicGlobal::density(0.0, 2.0), std::sqrt(2.0/Pi)/2.0, 1e-14, "density at zero");
    close(ClassicGlobal::density(-40.0, 1.0), 40.024968847207264, 1e-12, "stable density tail");
    // High-precision integral fixtures, not obtained from the implementation.
    close(ClassicGlobal::logPositiveMoment(1, -10.0, 1.0), -55.553122036122355, 1e-11, "first tail moment");
    close(ClassicGlobal::logPositiveMoment(3, -10.0, 1.0), -58.432500457047366, 1e-10, "third tail moment");
    auto m = makeMedium(R"({"type":"linear","direction":[0,0,1],"scale":2})");
    close(m->meanGradient(Vec3d(0.0)).z(), 2.0, 0.0, "global gradient must retain magnitude");
    const double expectedArea = 2.0*0.5*std::erfc(-2.0/std::sqrt(2.0)) + std::exp(-2.0)/std::sqrt(2.0*Pi);
    close(m->extinction(Vec3d(0.0), Vec3d(0.0,0.0,-1.0)), std::sqrt(2.0/Pi)*expectedArea, 1e-12, "directional extinction");
    require(m->extinction(Vec3d(0.0), Vec3d(0.0,0.0,-1.0)) > 100*m->extinction(Vec3d(0.0), Vec3d(0.0,0.0,1.0)), "extinction depends on direction");
    rapidjson::Document doc;
    auto value = m->toJson(doc.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    JsonDocument roundTrip(Path("classic-global-roundtrip.json"), buffer.GetString());
    Scene scene;
    auto copy = std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(roundTrip));
    copy->loadResources(); copy->prepareForRender();
    close(copy->extinction(Vec3d(0.0), Vec3d(0.0,0.0,-1.0)), m->extinction(Vec3d(0.0), Vec3d(0.0,0.0,-1.0)), 0.0, "round trip");
}
void flights()
{
    auto m = makeMedium(R"({"type":"linear","direction":[0,0,1]})");
    IndependentPathSampler rng(481, false);
    // Deliberately non-unit ray: geometry parameter t and physical distance
    // differ. The finite endpoint represents the nearest triangle.
    const Ray ray(Vec3f(0.0f,0.0f,1.0f), Vec3f(0.0f,0.0f,-2.0f), 0.125f, 0.875f);
    const double area = std::exp(-0.5)/std::sqrt(2.0*Pi) + 0.5*std::erfc(-1.0/std::sqrt(2.0));
    const double cdfStart = 0.5*std::erfc(-0.75/std::sqrt(2.0));
    const double cdfEnd = 0.5*std::erfc(0.75/std::sqrt(2.0));
    const double expectedT = std::exp(-area*std::log(cdfStart/cdfEnd));
    int exits = 0;
    double tr = 0.0;
    constexpr int count = 40000;
    for (int i = 0; i < count; ++i) {
        rng.startPath(0, i);
        Medium::MediumState state{}; state.reset();
        MediumSample s{};
        require(m->sampleDistance(rng, ray, state, s), "sampleDistance failed");
        require(s.t >= ray.nearT() && s.t <= ray.farT(), "event escaped finite segment");
        close(s.weight.x(), 1.0, 0.0, "delta tracking weight");
        if (s.exited) { ++exits; close(s.t, ray.farT(), 0.0, "surface endpoint"); }
        tr += m->transmittance(rng, ray, false, true, &state).x();
    }
    close(double(exits)/count, expectedT, 0.009, "free-flight survival vs analytic integral");
    close(tr/count, expectedT, 0.009, "ratio tracking vs analytic integral");
    Medium::MediumState state{}; state.reset();
    MediumSample s{};
    const Ray empty(Vec3f(0.0f), Vec3f(0.0f,0.0f,1.0f), 0.5f, 0.5f);
    m->sampleDistance(rng, empty, state, s);
    require(s.exited && state.bounce == 0, "zero segment produces no collision");
    close(m->transmittance(rng, empty, true, true, &state).x(), 1.0, 0.0, "empty transmittance");
    const Ray shortExit(Vec3f(0.0f), Vec3f(0.0f,0.0f,1.0f), 0.001f, 0.00001f);
    m->sampleDistance(rng, shortExit, state, s);
    require(s.exited && s.t == shortExit.farT(), "boundary exit inside geometry epsilon");
    bool rejected = false;
    try { m->transmittance(rng, Ray(Vec3f(0.0f), Vec3f(0.0f,0.0f,1.0f)), false, false, &state); }
    catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "unbounded segment must be rejected");
    rejected = false;
    try { m->pdf(rng, ray, false, true); } catch (const std::runtime_error &) { rejected = true; }
    require(rejected, "unsupported distance PDF must not silently return 1");
    std::cout << "flight survival=" << double(exits)/count << ", transmittance=" << tr/count << ", expected=" << expectedT << '\n';
}
void phase()
{
    const Eigen::Vector3d mean(0.2, -0.1, 0.8);
    Eigen::Matrix3d c;
    c << 0.7,0.1,0.05, 0.1,0.9,-0.1, 0.05,-0.1,0.6;
    ClassicGlobal::GaussianNdf ndf(c);
    IndependentPathSampler rng(7383, false);
    for (const Eigen::Vector3d &w : {Eigen::Vector3d(0,0,-1), Eigen::Vector3d(0,0,1), Eigen::Vector3d(1,0,0)}) {
        Eigen::Vector3d integrated = Eigen::Vector3d::Zero(), sampled = Eigen::Vector3d::Zero();
        double mass = 0.0;
        constexpr int nz = 100, nphi = 160, count = 30000;
        for (int z = 0; z < nz; ++z) for (int p = 0; p < nphi; ++p) {
            const double cosine = -1.0 + 2.0*(z+0.5)/nz, phi = 2.0*Pi*(p+0.5)/nphi;
            const double sine = std::sqrt(1.0-cosine*cosine);
            const Eigen::Vector3d wo(sine*std::cos(phi), sine*std::sin(phi), cosine);
            const double prob = ndf.phasePdf(mean,w,wo)*4.0*Pi/(nz*nphi);
            mass += prob; integrated += prob*wo;
        }
        for (int i = 0; i < count; ++i) {
            rng.startPath(0, i);
            const Eigen::Vector3d n = ndf.sampleVisibleNormal(mean,w,rng);
            require(w.dot(n) < 0.0, "visible normal support");
            sampled += w - 2.0*w.dot(n)*n;
        }
        close(mass, 1.0, 0.002, "phase PDF normalization");
        require((sampled/count - integrated).norm() < 0.018, "VNDF samples disagree with integrated phase PDF");
    }
    auto m = makeMedium(R"({"type":"linear","direction":[0,0,1]})");
    MediumSample ms{}; ms.p = Vec3f(0.0f);
    const auto *p = m->phaseFunction(ms.p);
    for (int i = 0; i < 500; ++i) {
        rng.startPath(1, i);
        PhaseSample s;
        require(p->sample(rng, Vec3f(0.0f,0.0f,-1.0f), ms, s), "phase sample");
        close(s.weight.x(), 1.0, 0.0, "unit-Fresnel furnace weight");
        close(p->eval(Vec3f(0.0f,0.0f,-1.0f), s.w, ms).x(), s.pdf, 1e-6, "phase eval/pdf consistency");
    }
}
void mirrorMaterial()
{
    auto mirror = makeMedium(R"({"type":"linear","direction":[0,0,1]})", "",
        R"({"type":"mirror","albedo":[0.15,0.8,0.5]})");
    auto white = makeMedium(R"({"type":"linear","direction":[0,0,1]})");
    const Vec3f albedo(0.15f, 0.8f, 0.5f);
    MediumSample ms{}; ms.p = Vec3f(0.0f);
    const auto *phase = mirror->phaseFunction(ms.p);
    IndependentPathSampler rng(917, false);
    for (const Vec3f &wi : {Vec3f(0.0f,0.0f,-1.0f), Vec3f(1.0f,0.0f,0.0f), Vec3f(0.0f,0.0f,1.0f)}) {
        close(mirror->extinction(Vec3d(ms.p), Vec3d(wi)), white->extinction(Vec3d(ms.p), Vec3d(wi)),
              0.0, "mirror material must not change free-flight law");
        for (int i = 0; i < 200; ++i) {
            rng.startPath(0, i);
            PhaseSample s;
            require(phase->sample(rng, wi, ms, s), "mirror phase sampling");
            const Vec3f energy = phase->eval(wi, s.w, ms);
            close(s.pdf, white->phaseFunction(ms.p)->pdf(wi, s.w, ms), 0.0, "mirror keeps VNDF PDF");
            for (int channel = 0; channel < 3; ++channel) {
                close(s.weight[channel], albedo[channel], 0.0, "mirror tint at every angle");
                close(energy[channel], albedo[channel]*s.pdf, 1e-6, "mirror NEE eval/sample agreement");
            }
        }
    }
    rapidjson::Document document;
    auto value = mirror->toJson(document.GetAllocator());
    require(std::string(value["material"]["type"].GetString()) == "mirror", "serialize mirror type");
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    JsonDocument json(Path("classic-global-mirror-roundtrip.json"), buffer.GetString());
    Scene scene;
    auto copy = std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(json));
    copy->loadResources(); copy->prepareForRender();
    require(copy->reflectance(0.3) == albedo, "mirror round trip preserves tint");
    // Existing configs without material.type must retain conductor Fresnel.
    auto conductor = makeMedium(R"({"type":"homogeneous"})", "", R"({"eta":[0.2,0.9,1.1],"k":[3.9,2.5,2.2]})");
    const double f0 = ((0.2-1)*(0.2-1)+3.9*3.9)/((0.2+1)*(0.2+1)+3.9*3.9);
    close(conductor->reflectance(1.0).x(), f0, 1e-7, "legacy conductor normal-incidence Fresnel");
    close(conductor->reflectance(0.0).x(), 1.0, 0.0, "legacy conductor grazing Fresnel");
}
#if OPENVDB_AVAILABLE
void sdfGrid(const std::filesystem::path &directory)
{
    std::filesystem::create_directories(directory);
    const auto path = directory / "classic-global-test-sdf.vdb";
    auto grid = openvdb::FloatGrid::create(0.0f);
    grid->setName("sdf");
    auto accessor = grid->getAccessor();
    for (int z=0; z<10; ++z) for (int y=0; y<10; ++y) for (int x=0; x<10; ++x)
        accessor.setValue(openvdb::Coord(x,y,z), float(x+2*y-3*z));
    openvdb::io::File file(path.string());
    file.write(openvdb::GridPtrVec{grid}); file.close();
    const std::string mean = R"({"type":"tabulated","scale":2,"offset":0.5,"grid":{
      "type":"vdb","file":")" + path.generic_string() + R"(","density_name":"sdf","emission_name":"",
      "normalize_size":false,"request_sdf":false,"interpolate":"linear","density_scale":1,
      "transform":{"position":[1,-2,3],"scale":[2,3,4]}}})";
    auto medium = makeMedium(mean);
    // Index position (4.25, 4.5, 4.75), then the grid's world transform.
    const Vec3d p(9.5,11.5,22.0);
    close(medium->meanValue(p), -1.0, 1e-5, "SDF interpolation/offset/transform");
    const Vec3d g = medium->meanGradient(p);
    close(g.x(), 1.0, 1e-6, "SDF world gradient x");
    close(g.y(), 4.0/3.0, 1e-6, "SDF world gradient y");
    close(g.z(), -1.5, 1e-6, "SDF world gradient z");
    close(medium->meanGradient(Vec3d(-100.0,11.5,22.0)).x(), 0.0, 0.0, "clamped SDF derivative");
    // Only this generated SDF is an input resource; no density or majorant file.
}
#endif
}
int main(int argc, char **argv)
{
    try {
        coefficients(); flights(); phase(); mirrorMaterial();
#if OPENVDB_AVAILABLE
        openvdb::initialize();
        sdfGrid(argc > 1 ? std::filesystem::path(argv[1]) : std::filesystem::current_path());
#endif
        std::cout << "classic_global tests passed\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "classic_global test failed: " << e.what() << '\n';
        return 1;
    }
}
