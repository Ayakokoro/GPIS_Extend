// Experiment-only batch runner: production rendering with one loaded VDB.
#include "media/GlobalConditionalMedium.hpp"
#include "integrators/path_tracer/PathTraceIntegrator.hpp"
#include "io/Scene.hpp"
#include "thread/ThreadUtils.hpp"
#include "thread/ThreadPool.hpp"
#include "math/ConditionalProfile.hpp"
#include <openvdb/openvdb.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <chrono>
#include <iostream>

using namespace Tungsten;
namespace cg=Tungsten::ConditionalGpis;
namespace fs=std::filesystem;
using J=nlohmann::json;
using Clock=std::chrono::steady_clock;
void require(bool valid,const char *message) {if(!valid)throw std::runtime_error(message);}
double seconds(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}

void clearProfile() {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
    auto &registry=cg::Profile::registry();
    require(registry.enabled,"Profile executable requires TUNGSTEN_GPIS_CONDITIONAL_PROFILE=1");
    for(auto &stats:registry.threads) {
        require(stats->top==nullptr,"Cannot reset a live profiling scope");
        *stats=cg::Profile::ThreadStats{};
    }
#endif
}
J snapshotProfile() {
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_PROFILE
    // Workers are joined. Reuse the profiler's serializer on copied statistics;
    // keep the original allocations alive because workers cache their addresses.
    std::ostringstream buffer;
    auto *previous=std::cout.rdbuf(buffer.rdbuf());
    {
        cg::Profile::Registry snapshot;
        for(const auto &stats:cg::Profile::registry().threads)
            snapshot.threads.push_back(std::make_unique<cg::Profile::ThreadStats>(*stats));
    }
    std::cout.rdbuf(previous);
    const std::string prefix="GPIS conditional profile: ";
    require(buffer.str().find(prefix)==0,"Missing profile snapshot");
    return J::parse(buffer.str().substr(prefix.size()));
#else
    return nullptr;
#endif
}

J writeImage(const fs::path &path,const Camera &camera) {
    const auto res=camera.resolution();
    std::ofstream out(path,std::ios::binary);
    out<<"PF\n"<<res.x()<<" "<<res.y()<<"\n-1\n";
    double sum=0;float maximum=0;
    for(int y=int(res.y())-1;y>=0;--y)for(unsigned x=0;x<res.x();++x) {
        const auto c=camera.getLinear(x,y);
        const float rgb[3]={c.x(),c.y(),c.z()};
        for(float value:rgb) {require(std::isfinite(value)&&value>=0,"Invalid image value");sum+=value;maximum=std::max(maximum,value);}
        out.write(reinterpret_cast<const char *>(rgb),sizeof(rgb));
    }
    require(bool(out),"Cannot write image");
    return {{"mean_rgb",sum/(3*res.product())},{"maximum",maximum},{"finite_nonnegative",true}};
}

int main(int argc,char **argv) {
    try {
        require(argc==3,"Usage: conditional_threshold_sweep spec.json output-directory");
        std::ifstream input(argv[1]);require(bool(input),"Missing specification");const J spec=J::parse(input);
        const fs::path output=fs::absolute(argv[2]);fs::create_directories(output);
        require(!fs::exists(output/"runs.json"),"Output already contains runs.json");
        const unsigned seed=spec.at("seed"),spp=spec.at("spp"),size=spec.at("size"),threads=spec.at("threads");
        require(spp>0&&size>0&&threads>0,"Invalid render dimensions");
        const auto loading=Clock::now();
        openvdb::initialize();ThreadUtils::startThreads(threads);EmbreeUtil::initDevice();
        std::unique_ptr<Scene> scene(Scene::load(Path(fs::absolute(spec.at("scene").get<std::string>()).string())));
        scene->camera()->setResolution(Vec2u(size));
        scene->rendererSettings().setSpp(spp);scene->rendererSettings().setSppStep(1);
        require(!scene->rendererSettings().useAdaptiveSampling(),"Adaptive sampling is not supported");
        scene->loadResources();
        std::unique_ptr<TraceableScene> traceable(scene->makeTraceable(seed));
        auto *integrator=dynamic_cast<PathTraceIntegrator *>(scene->integrator());
        require(integrator&&integrator->settings().independentPathSampling,"Independent path sampling required");
        require(!integrator->settings().useCudaGpis&&!integrator->settings().wavefrontGpis,"CPU sequential paths required");
        GlobalConditionalMedium *medium=nullptr;
        for(auto &m:scene->media())if(auto *g=dynamic_cast<GlobalConditionalMedium *>(m.get())) {
            require(!medium,"Expected exactly one conditional medium");medium=g;
        }
        require(medium,"Missing conditional medium");
        // The owned Scene and medium are non-const. Change this experiment's
        // existing setting only between joined renders; no production API or
        // transport algorithm is altered to support the sweep.
        auto &tracking=const_cast<cg::ConditionalTrackingSettings &>(medium->trackingSettings());
        const double originalThreshold=tracking.tightenExpectedEvents;
        J report={{"spec",spec},{"initialization_seconds",seconds(loading)},{"runs",J::array()}};
        std::cout<<"Loaded scene once in "<<report["initialization_seconds"]<<" s"<<std::endl;
        for(const auto &run:spec.at("runs")) {
            const std::string label=run.at("label");
            require(!label.empty()&&label.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")==std::string::npos,"Unsafe run label");
            integrator->teardownAfterRender();scene->camera()->teardownAfterRender();
            scene->camera()->prepareForRender();scene->camera()->requestOutputBuffers(scene->rendererSettings().renderOutputs());
            tracking.tightenExpectedEvents=run.at("threshold").get<double>();tracking.validate();
            const unsigned runSeed=run.value("seed",seed);traceable->_seed=runSeed;
            integrator->prepareForRender(*traceable,runSeed);clearProfile();
            const auto pathsBefore=medium->discardedSurfacePaths(),shadowsBefore=medium->discardedSurfaceConnections();
            std::cout<<"Starting "<<label<<", threshold "<<tracking.tightenExpectedEvents<<std::endl;
            const auto start=Clock::now();
            while(!integrator->done()) {integrator->startRender([](){});integrator->waitForCompletion();}
            const double elapsed=seconds(start);
            require(integrator->currentSpp()==spp,"Incomplete render");
            J row=run;row["render_seconds"]=elapsed;row["profile"]=snapshotProfile();
            row["image"]=writeImage(output/(label+".pfm"),*scene->camera());
            row["pfm"]=(output/(label+".pfm")).string();
            row["discarded_surface_paths"]=medium->discardedSurfacePaths()-pathsBefore;
            row["discarded_surface_connections"]=medium->discardedSurfaceConnections()-shadowsBefore;
            report["runs"].push_back(row);
            std::ofstream saved(output/"runs.json");saved<<report.dump(2);require(bool(saved),"Cannot write report");
            std::cout<<"Completed "<<label<<", seconds "<<elapsed<<", mean "<<row["image"]["mean_rgb"]<<std::endl;
        }
        tracking.tightenExpectedEvents=originalThreshold;clearProfile();
        traceable.reset();scene.reset();delete ThreadUtils::pool;ThreadUtils::pool=nullptr;
        return 0;
    }catch(const std::exception &e) {std::cerr<<"Threshold experiment failed: "<<e.what()<<std::endl;return 1;}
}
