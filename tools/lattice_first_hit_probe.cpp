// Trace fixed world-space rays through the scene's actual lattice medium.
// A blank distance denotes survival through the end of the segment.
#include "io/Scene.hpp"
#include "media/ClassicGlobalFpGradientMedium.hpp"
#include "media/LatticeCubicGpisMedium.hpp"
#include "sampling/IndependentPathSampler.hpp"

#include <nlohmann/json.hpp>
#include <openvdb/openvdb.h>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace Tungsten;
using Json = nlohmann::json;
namespace {
Vec3f vector3(const Json &v) {
    if (!v.is_array() || v.size()!=3)
        throw std::runtime_error("expected a 3-component vector");
    Vec3f result(v.at(0).get<float>(),v.at(1).get<float>(),v.at(2).get<float>());
    for (int i=0;i<3;++i)
        if (!std::isfinite(result[i])) throw std::runtime_error("nonfinite vector");
    return result;
}
Eigen::Vector3d eigen(const Vec3f &v) { return {v.x(),v.y(),v.z()}; }
}

int main(int argc,char **argv) try {
    if (argc!=4 && argc!=5)
        throw std::runtime_error("lattice_first_hit_probe scene.json rays.json hits.csv [mean.csv]");
    std::ifstream input(argv[2]);
    if (!input) throw std::runtime_error("cannot read rays.json");
    const Json spec=Json::parse(input);
    const int samples=spec.at("samples").get<int>();
    if (samples<0) throw std::runtime_error("negative sample count");
    openvdb::initialize();
    const std::unique_ptr<Scene> scene(Scene::load(Path(argv[1]).absolute()));
    std::shared_ptr<LatticeCubicGpisMedium> medium;
    for (const auto &m : scene->media()) {
        auto lattice=std::dynamic_pointer_cast<LatticeCubicGpisMedium>(m);
        if (!lattice) continue;
        if (medium) throw std::runtime_error("expected a single lattice medium");
        medium=std::move(lattice);
    }
    if (!medium) throw std::runtime_error("scene has no lattice_cubic_gpis medium");
    medium->loadResources();
    medium->prepareForRender();
    std::ofstream hits(argv[3]);
    hits.exceptions(std::ios::failbit | std::ios::badbit);
    hits<<"ray,realization,distance\n"<<std::setprecision(12);
    std::ofstream profile;
    double step=0;
    if (argc==5) {
        step=spec.at("mean_step").get<double>();
        if (!(step>0) || !std::isfinite(step)) throw std::runtime_error("invalid mean_step");
        profile.exceptions(std::ios::failbit | std::ios::badbit);
        profile.open(argv[4]);
        profile<<"ray,t,mean,mean_slope\n"<<std::setprecision(17);
    }
    IndependentPathSampler sampler(spec.at("seed").get<uint32>(),false);
    uint32 rayIndex=0;
    for (const auto &r : spec.at("rays")) {
        const std::string name=r.at("name").get<std::string>();
        if (name.empty() || name.find_first_of(",\"\r\n")!=std::string::npos)
            throw std::runtime_error("ray name must be a nonempty CSV-safe label");
        const Vec3f origin=vector3(r.at("origin"));
        const Vec3f dir=vector3(r.at("direction"));
        const float length=r.at("length").get<float>();
        if (!(length>0) || !std::isfinite(length) || !(dir.lengthSq()>0))
            throw std::runtime_error("invalid ray");
        const Ray ray(origin,dir.normalized(),0,length);
        const auto direction=eigen(ray.dir()).normalized();
        std::shared_ptr<FpGradientPathContext> birth;
        if (r.contains("start_derivative")) {
            const double derivative=r.at("start_derivative").get<double>();
            if (derivative==0 || !std::isfinite(derivative))
                throw std::runtime_error("a surface birth needs a finite nonzero outgoing derivative");
            birth=std::make_shared<FpGradientPathContext>();
            birth->owner=medium.get();
            birth->origin=eigen(origin);
            birth->rayOrigin=origin;
            birth->direction=direction;
            birth->derivative=derivative;
            birth->gradient=direction*derivative;
        }
        if (profile.is_open()) {
            // Include both endpoints and align the half-resolution reference.
            const auto intervals=2*size_t(std::ceil(double(length)/(2*step)));
            // Use the outgoing side at an exact voxel face. In that voxel
            // the analytic gradient projected onto the ray is quadratic.
            const auto slope=[&](double t) {
                const Eigen::Vector3d p=eigen(origin)+t*eigen(ray.dir());
                const Vec3d g=medium->meanGradient(Vec3d(p.x(),p.y(),p.z()));
                return Eigen::Vector3d(g.x(),g.y(),g.z()).dot(eigen(ray.dir()));
            };
            const double h=64*std::numeric_limits<double>::epsilon()
                *std::max(1.0,eigen(origin).cwiseAbs().maxCoeff());
            const double slopeAtStart=3*(slope(h)-slope(2*h))+slope(3*h);
            for (size_t i=0;i<=intervals;++i) {
                const double t=double(length)*i/intervals;
                const Eigen::Vector3d p=eigen(origin)+t*eigen(ray.dir());
                profile<<name<<','<<t<<','<<medium->meanValue(Vec3d(p.x(),p.y(),p.z()))<<',';
                if (i==0) profile<<slopeAtStart;
                profile<<'\n';
            }
        }
        for (int i=0;i<samples;++i) {
            sampler.startPath(rayIndex,i);
            Medium::MediumState state{};
            state.reset();
            state.pathContext=birth;
            const auto hit=medium->firstHitDistance(sampler,ray,&state);
            hits<<name<<','<<i<<',';
            if (hit) {
                if (!(*hit>=0 && *hit<length))
                    throw std::runtime_error("invalid first-hit distance");
                hits<<*hit;
            }
            hits<<'\n';
        }
        ++rayIndex;
    }
    hits.close();
    if (profile.is_open()) profile.close();
    return 0;
} catch (const std::exception &error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
