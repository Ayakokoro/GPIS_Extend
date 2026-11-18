// Compare integral, midpoint and adaptive transitions with one loaded VDB.
#include "media/GlobalConditionalMedium.hpp"
#include "media/GpisSdfGrid.hpp"
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
#include "transition_integral_rays.hpp"
#include "conditional_slow_rays.hpp"
#include "transition_no_split_rays.hpp"
#include "conditional_optimization_rays.hpp"
#include "conditional_macrocell_checks.hpp"
#include "conditional_fixed_optimization_rays.hpp"

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
        require(argc==3,"Usage: conditional_transition_experiment spec.json output-directory");
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
        // tracking modes only between joined renders.
        auto &tracking=const_cast<cg::ConditionalTrackingSettings &>(medium->trackingSettings());
        const auto originalTracking=tracking;
        const double initializationSeconds=seconds(loading);
        if(spec.contains("ray_files"))exportIntegralRays(spec,output,medium->field());
        J report={{"spec",spec},{"initialization_seconds",initializationSeconds},{"runs",J::array()}};
        std::cout<<"Loaded scene once in "<<report["initialization_seconds"]<<" s"<<std::endl;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
        if(spec.contains("no_split_ray_files"))compareTransitionNoSplitRays(spec["no_split_ray_files"],output,medium->field(),tracking);
#endif
        for(const auto &run:spec.at("runs")) {
            const std::string label=run.at("label");
            require(!label.empty()&&label.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")==std::string::npos,"Unsafe run label");
            integrator->teardownAfterRender();scene->camera()->teardownAfterRender();
            scene->camera()->prepareForRender();scene->camera()->requestOutputBuffers(scene->rendererSettings().renderOutputs());
            tracking=originalTracking;
            const std::string mode=run.at("mode");
            require(mode=="current_two"||mode=="old_three"||mode=="integral_three"||mode=="coupled","Unknown transition mode");
            tracking.transitionPointBounds=mode=="old_three";
            tracking.transitionIntegralBounds=mode=="integral_three";
            tracking.coupledBounds=mode=="coupled";
            tracking.farIntegralBounds=run.value("far_integral",originalTracking.farIntegralBounds);
            tracking.coupledLocalScale=run.value("local_scale",originalTracking.coupledLocalScale);
            tracking.localFarStartScale=run.value("far_scale",originalTracking.localFarStartScale);
            tracking.proposalScale=run.value("proposal_scale",originalTracking.proposalScale);
            tracking.ageScale=run.value("age_scale",originalTracking.ageScale);
            tracking.normalizedBoundOnly=run.value("normalized_only",originalTracking.normalizedBoundOnly);
            tracking.transitionMaxExpectedEvents=run.value("transition_max_expected_events",originalTracking.transitionMaxExpectedEvents);
            tracking.macrocellBounds=run.value("macro_bounds",originalTracking.macrocellBounds);
            tracking.macrocellMinExpectedEvents=run.value("macro_min_events",originalTracking.macrocellMinExpectedEvents);
            tracking.macrocellStartScale=run.value("macro_start",originalTracking.macrocellStartScale);
            tracking.macrocellSize=run.value("macro_size",originalTracking.macrocellSize);
            tracking.validate();
            double macroBuildSeconds=0;
            if(tracking.macrocellBounds) {
                auto *grid=const_cast<GpisSdfGrid *>(dynamic_cast<const GpisSdfGrid *>(medium->field().mean.get()));
                require(grid,"Macrocell benchmark requires a tabulated SDF");
                if(grid->macrocellSize!=tracking.macrocellSize) {
                    const auto build=Clock::now();grid->buildMacrocells(tracking.macrocellSize);
                    macroBuildSeconds=seconds(build);
                }
            }
            J macroChecks=nullptr;
            if(run.value("macro_checks",false))
                macroChecks=macrocellCoverageChecks(spec.at("optimization_ray_files"),medium->field());
            const unsigned runSeed=run.value("seed",seed);traceable->_seed=runSeed;
            integrator->prepareForRender(*traceable,runSeed);clearProfile();
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
            require(spec.at("runs").size()==1,"Detailed ray recording currently supports one run");
            cg::RayProfile::registry().enabled=true;
#endif
            const auto pathsBefore=medium->discardedSurfacePaths(),shadowsBefore=medium->discardedSurfaceConnections();
            std::cout<<"Starting "<<label<<", mode "<<mode<<std::endl;
            const auto start=Clock::now();
            while(!integrator->done()) {integrator->startRender([](){});integrator->waitForCompletion();}
            const double elapsed=seconds(start);
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
            cg::RayProfile::registry().enabled=false;
#endif
            require(integrator->currentSpp()==spp,"Incomplete render");
            J row=run;row["render_seconds"]=elapsed;row["profile"]=snapshotProfile();
            row["macro_build_seconds"]=macroBuildSeconds;
            row["macro_checks"]=macroChecks;
            row["image"]=writeImage(output/(label+".pfm"),*scene->camera());
            row["pfm"]=(output/(label+".pfm")).string();
            row["discarded_surface_paths"]=medium->discardedSurfacePaths()-pathsBefore;
            row["discarded_surface_connections"]=medium->discardedSurfaceConnections()-shadowsBefore;
            const unsigned raySamples=run.value("ray_samples",spec.value("ray_samples",64u));
            if(spec.contains("optimization_ray_files") && raySamples>0)
                row["slow_rays"]=optimizationRays(spec["optimization_ray_files"],medium->field(),tracking,raySamples);
            report["runs"].push_back(row);
            std::ofstream saved(output/"runs.json");saved<<report.dump(2);require(bool(saved),"Cannot write report");
            std::cout<<"Completed "<<label<<", seconds "<<elapsed<<", mean "<<row["image"]["mean_rgb"]<<std::endl;
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
            exportSlowRays(output,medium->field(),tracking,row["profile"]);
            if(spec.contains("fixed_ray_files"))exportOptimizationFixedRays(spec["fixed_ray_files"],output,medium->field(),tracking);
#endif
        }
        tracking=originalTracking;clearProfile();
        traceable.reset();scene.reset();delete ThreadUtils::pool;ThreadUtils::pool=nullptr;
        return 0;
    }catch(const std::exception &e) {std::cerr<<"Transition experiment failed: "<<e.what()<<std::endl;return 1;}
}
