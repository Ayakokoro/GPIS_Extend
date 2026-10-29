// Replay exact production CPU paths, including direct-light RNG draws.
#include "integrators/path_tracer/PathTraceIntegrator.hpp"
#include "integrators/path_tracer/PathTracer.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/Scene.hpp"
#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"
#include <openvdb/openvdb.h>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace Tungsten;
using J = nlohmann::json;
namespace fs = std::filesystem;
using D = PathTraceDiagnostics;

void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}

// Reference mean geometry only. Transport vertices below come from the actual
// production samplers, never from this visualization mesh.
J meanMesh(const MeanFunction &mean, int n) {
    const Vec3d lo(-.85, -.02, -.85), hi(.85, 1.64, .85);
    std::vector<Vec3d> positions(n*n*n);
    std::vector<double> values(n*n*n);
    auto idx = [=](int x, int y, int z) { return x+n*(y+n*z); };
    for (int z=0; z<n; ++z) for (int y=0; y<n; ++y) for (int x=0; x<n; ++x) {
        int i=idx(x,y,z);
        positions[i]=lo+(hi-lo)*Vec3d(double(x),double(y),double(z))/double(n-1);
        values[i]=mean(Derivative::None, positions[i], Vec3d(0.0));
    }
    const int corners[8][3]={{0,0,0},{1,0,0},{1,1,0},{0,1,0},{0,0,1},{1,0,1},{1,1,1},{0,1,1}};
    const int tetra[6][4]={{0,5,1,6},{0,1,2,6},{0,2,3,6},{0,3,7,6},{0,7,4,6},{0,4,5,6}};
    J triangles=J::array();
    for (int z=0; z<n-1; ++z) for (int y=0; y<n-1; ++y) for (int x=0; x<n-1; ++x) {
        int ids[8];
        for(int c=0;c<8;++c) ids[c]=idx(x+corners[c][0],y+corners[c][1],z+corners[c][2]);
        auto edge=[&](int a,int b) {
            a=ids[a];b=ids[b];double t=values[a]/(values[a]-values[b]);
            return positions[a]+t*(positions[b]-positions[a]);
        };
        auto tri=[&](Vec3d a,Vec3d b,Vec3d c) {
            triangles.push_back({D::vector(a),D::vector(b),D::vector(c)});
        };
        for(const auto &t:tetra) {
            int in[4],out[4],ni=0,no=0;
            for(int v:t) {if(values[ids[v]]<0) in[ni++]=v;else out[no++]=v;}
            if(ni==1) tri(edge(in[0],out[0]),edge(in[0],out[1]),edge(in[0],out[2]));
            if(ni==3) tri(edge(out[0],in[0]),edge(out[0],in[1]),edge(out[0],in[2]));
            if(ni==2) {
                auto a=edge(in[0],out[0]),b=edge(in[0],out[1]),c=edge(in[1],out[0]),d=edge(in[1],out[1]);
                tri(a,b,c);tri(b,d,c);
            }
        }
    }
    return {{"kind","mean_zero_isosurface_visualization_only"},{"grid",n},{"triangles",triangles}};
}

int main(int argc, char **argv) {
    try {
        require(argc==3,"Usage: surface_path_trace spec.json output.json");
        std::ifstream input(argv[1]);require(bool(input),"Missing spec");
        const J spec=J::parse(input);
        const uint32 seed=spec.at("seed").get<uint32>();
        const auto scenePath=fs::absolute(spec.at("scene").get<std::string>());
        std::cout<<"Initialize replay"<<std::endl;
        openvdb::initialize();ThreadUtils::startThreads(1);EmbreeUtil::initDevice();
        std::cout<<"Load scene "<<scenePath.string()<<std::endl;
        std::unique_ptr<Scene> scene(Scene::load(Path(scenePath.string())));
        scene->loadResources();
        std::cout<<"Prepare traceable scene"<<std::endl;
        std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(seed));
        std::cout<<"Prepare observer"<<std::endl;
        auto *integrator=dynamic_cast<PathTraceIntegrator *>(scene->integrator());
        require(integrator!=nullptr,"Expected path tracer");
        require(integrator->settings().independentPathSampling,"Replay requires independent_path_sampling");
        require(!integrator->settings().wavefrontGpis,"This observer supports blocking CPU paths");
        PathTracer tracer(traceable.get(),integrator->settings(),0);
        IndependentPathSampler sampler(seed,scene->rendererSettings().useSobol());
        const auto resolution=scene->camera()->resolution();
        J result={{"schema",1},{"spec",spec},{"paths",J::array()},
            {"resolution",{resolution.x(),resolution.y()}},{"observer_mismatches",0}};
        int verified=0;
        for(const auto &p:spec.at("pixels")) {
            const auto x=p.at("xy")[0].get<uint32>(),y=p.at("xy")[1].get<uint32>();
            require(x<resolution.x() && y<resolution.y(),"Pixel outside image");
            const int start=spec.value("sample_start",0),count=spec.at("samples").get<int>();
            require(start>=0 && count>0,"Invalid sample range");
            for(int sample=start;sample<start+count;++sample) {
                D diagnostics;
                tracer.setPathDiagnostics(&diagnostics);
                sampler.startPath(x+y*resolution.x(),sample);
                Vec3f c=tracer.traceSample(Vec2u(x,y),sampler,sample);
                // Verify every requested sample (or a specified prefix) against
                // an unobserved run, including the next RNG output.
                if(sample-start<spec.value("verify_samples",count)) {
                    const float next=sampler.next1D();
                    tracer.setPathDiagnostics(nullptr);
                    sampler.startPath(x+y*resolution.x(),sample);
                    const Vec3f plain=tracer.traceSample(Vec2u(x,y),sampler,sample);
                    require(plain==c && sampler.next1D()==next,"Observer changed path or RNG state");
                    ++verified;
                }
                tracer.setPathDiagnostics(nullptr);
                bool clamped=std::isinf(c)||std::isnan(c)||c.max()>65536;
                if(clamped) c=Vec3f(0.f);
                diagnostics.result["pixel"]=p.at("xy");
                diagnostics.result["pixel_label"]=p.at("label");
                diagnostics.result["sample"]=sample;
                diagnostics.result["renderer_clamped"]=clamped;
                diagnostics.result["radiance"]=D::vector(c);
                diagnostics.result["events"]=std::move(diagnostics.events);
                result["paths"].push_back(std::move(diagnostics.result));
            }
            std::cout<<"Replayed "<<p.at("label")<<", "<<count<<" paths"<<std::endl;
        }
        result["verified_samples"]=verified;
        if(spec.value("export_mean_mesh",false)) {
            auto gp=std::dynamic_pointer_cast<GaussianProcessMedium>(scene->media().at(0));
            require(bool(gp),"Mean mesh requires a GP medium");
            result["geometry"]=meanMesh(*gp->_gp->get_gaussian_process()->_mean,spec.value("mesh_grid",48));
        }
        fs::path output=fs::absolute(argv[2]);fs::create_directories(output.parent_path());
        std::ofstream stream(output);stream<<result.dump();require(bool(stream),"Output failed");
        traceable.reset();scene.reset();delete ThreadUtils::pool;ThreadUtils::pool=nullptr;
        std::cout<<"Saved "<<result["paths"].size()<<" paths; verified "<<verified<<std::endl;
        return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<std::endl;return 1; }
}
