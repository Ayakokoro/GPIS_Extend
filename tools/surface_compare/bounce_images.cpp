// One production path supplies all bounce-order images. No depth-limit changes.
#include "integrators/BounceContributions.hpp"
#include "integrators/path_tracer/PathTraceIntegrator.hpp"
#include "integrators/path_tracer/PathTracer.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/Scene.hpp"
#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"
#include <openvdb/openvdb.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <chrono>
#include <atomic>
#include <mutex>

using namespace Tungsten;
using J=nlohmann::json;
namespace fs=std::filesystem;
constexpr int Layers=7;
const char *Names[Layers]={"order00","order01","order02","order03","order04","order05plus","beauty"};

void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
void writePfm(const fs::path &path,const std::vector<Vec3d> &pixels,int w,int h) {
    std::ofstream f(path,std::ios::binary);
    f<<"PF\n"<<w<<" "<<h<<"\n-1\n";
    for(int y=h-1;y>=0;--y) for(int x=0;x<w;++x) {
        const auto &v=pixels[x+y*w];
        const float rgb[3]={float(v.x()),float(v.y()),float(v.z())};
        f.write(reinterpret_cast<const char *>(rgb),sizeof(rgb));
    }
    require(bool(f),"Failed to write image");
}

int main(int argc,char **argv) {
    try {
        require(argc==3,"Usage: surface_bounce_images spec.json output-directory");
        std::ifstream file(argv[1]);require(bool(file),"Missing spec");const J spec=J::parse(file);
        const int samples=spec.at("samples").get<int>(),threads=spec.value("threads",24);
        require(samples>1 && threads>0,"Invalid sample/thread count");
        const uint32 seed=spec.at("seed").get<uint32>();
        const fs::path output=fs::absolute(argv[2]);fs::create_directories(output);
        const auto start=std::chrono::steady_clock::now();
        openvdb::initialize();ThreadUtils::startThreads(threads);EmbreeUtil::initDevice();
        std::unique_ptr<Scene> scene(Scene::load(Path(fs::absolute(spec.at("scene").get<std::string>()).string())));
        scene->loadResources();
        std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(seed));
        auto *integrator=dynamic_cast<PathTraceIntegrator *>(scene->integrator());
        require(integrator && integrator->settings().independentPathSampling,"Independent CPU path tracer required");
        require(!integrator->settings().wavefrontGpis,"Wavefront mode is not supported by this diagnostic");
        const auto res=scene->camera()->resolution();const int w=res.x(),h=res.y();
        std::array<int,4> bounds{0,0,w,h};
        if(spec.contains("bounds")) for(int i=0;i<4;++i) bounds[i]=spec.at("bounds")[i].get<int>();
        require(bounds[0]>=0 && bounds[1]>=0 && bounds[2]<=w && bounds[3]<=h && bounds[0]<bounds[2] && bounds[1]<bounds[3],"Invalid bounds");
        std::array<std::vector<Vec3d>,Layers> mean,variance;
        for(int k=0;k<Layers;++k) {mean[k].assign(w*h,Vec3d(0.0));variance[k].assign(w*h,Vec3d(0.0));}
        std::vector<Vec3d> ends(w*h,Vec3d(0.0));
        std::vector<std::unique_ptr<PathTracer>> tracers;
        for(int t=0;t<threads;++t) tracers.emplace_back(new PathTracer(traceable.get(),integrator->settings(),t));
        std::vector<bool> verify(w*h,false);
        if(spec.contains("verify_pixels")) for(const auto &p:spec.at("verify_pixels")) {
            int x=p[0].get<int>(),y=p[1].get<int>();require(x>=0&&x<w&&y>=0&&y<h,"Invalid verify pixel");verify[x+y*w]=true;
        }
        const int tileSize=8,tw=(bounds[2]-bounds[0]+tileSize-1)/tileSize,th=(bounds[3]-bounds[1]+tileSize-1)/tileSize;
        std::atomic<int> done{0},verified{0};std::mutex outputMutex;
        double reconstructionError=0.0;
        auto group=ThreadUtils::pool->enqueue([&](uint32 task,uint32,uint32 thread) {
            auto &tracer=*tracers[thread];
            IndependentPathSampler sampler(seed,scene->rendererSettings().useSobol());
            const int x0=bounds[0]+int(task%tw)*tileSize,y0=bounds[1]+int(task/tw)*tileSize;
            BounceContributions contribution;ScopedBounceContributions observe(&contribution);
            double localError=0.0;
            for(int y=y0;y<std::min(y0+tileSize,bounds[3]);++y) for(int x=x0;x<std::min(x0+tileSize,bounds[2]);++x) {
                const int pixel=x+y*w;
                for(int s=0;s<samples;++s) {
                    contribution.reset();sampler.startPath(pixel,s);
                    Vec3f value=tracer.traceSample(Vec2u(uint32(x),uint32(y)),sampler,s);
                    if(verify[pixel] && s<spec.value("verify_samples",16)) {
                        const float next=sampler.next1D();
                        ScopedBounceContributions unobserved(nullptr);
                        sampler.startPath(pixel,s);
                        const Vec3f plain=tracer.traceSample(Vec2u(uint32(x),uint32(y)),sampler,s);
                        require(plain==value && sampler.next1D()==next,"AOV changed radiance or random-number sequence");
                        ++verified;
                    }
                    const bool clamped=std::isinf(value)||std::isnan(value)||value.max()>65536;
                    if(clamped) {value=Vec3f(0.f);for(auto &v:contribution.radiance)v=Vec3d(0.0);}
                    require(contribution.termination.find("runtime_error")!=0 && contribution.termination.find("nonfinite")!=0,"Path encountered an internal error");
                    Vec3d sum(0.0);for(const auto &v:contribution.radiance)sum+=v;
                    const double error=std::abs(sum-Vec3d(value)).max();
                    localError=std::max(localError,error);
                    require(error<=2e-6*(1.0+std::abs(Vec3d(value)).max()),"Bounce contributions do not sum to radiance");
                    for(int k=0;k<Layers;++k) {
                        const Vec3d c=k==Layers-1?Vec3d(value):contribution.radiance[k];
                        auto &m=mean[k][pixel];const Vec3d delta=c-m;
                        m+=delta/double(s+1);variance[k][pixel]+=delta*(c-m);
                    }
                    ends[pixel]+=Vec3d(contribution.termination=="outward_crossing_rejected"?1.0:0.0,
                        contribution.termination=="bounce_limit"?1.0:0.0,clamped?1.0:0.0);
                }
                for(int k=0;k<Layers;++k)variance[k][pixel]/=double(samples-1);
                ends[pixel]/=double(samples);
            }
            const int count=++done;
            std::lock_guard<std::mutex> lock(outputMutex);
            reconstructionError=std::max(reconstructionError,localError);
            if(count%16==0 || count==tw*th) {
                const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
                std::cout<<"Tiles "<<count<<"/"<<tw*th<<", seconds "<<seconds<<std::endl;
            }
        },tw*th);
        group->wait();
        for(int k=0;k<Layers;++k) {
            writePfm(output/(std::string(Names[k])+".pfm"),mean[k],w,h);
            writePfm(output/(std::string(Names[k])+"_variance.pfm"),variance[k],w,h);
        }
        writePfm(output/"termination_fractions.pfm",ends,w,h);
        J result={{"spec",spec},{"resolution",{w,h}},{"bounds",bounds},{"samples",samples},
            {"seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},
            {"verified_samples",int(verified)},{"observer_mismatches",0},
            {"max_sample_reconstruction_error",reconstructionError},
            {"definition","Number of scattering vertices on each contributing camera-to-light path. Includes floor scattering and NEE vertex; excludes transparent boundaries and light endpoint."},
            {"bins",{"0","1","2","3","4","5+"}}};
        std::ofstream info(output/"result.json");info<<result.dump(2);require(bool(info),"Failed to save result");
        std::cout<<"Finished bounce images; verified "<<verified<<" paths; max reconstruction error "<<reconstructionError<<std::endl;
        tracers.clear();traceable.reset();scene.reset();delete ThreadUtils::pool;ThreadUtils::pool=nullptr;
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 1;}
}
