// Compare integral, midpoint and adaptive transitions with one loaded VDB.
#include "media/GlobalConditionalMedium.hpp"
#include "media/GpisSdfGrid.hpp"
#include "integrators/path_tracer/PathTraceIntegrator.hpp"
#include "io/Scene.hpp"
#include "io/JsonDocument.hpp"
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
#include "conditional_voxel_ray.hpp"
#include "chapter2_macrocell_rays.hpp"

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

// Time the real tracker over a fixed near-birth window. Every variant starts
// with the same conditioning observation, including replays begun at positive age.
J chapter2WindowReplay(const J &config,const cg::GPSSField &field,cg::ConditionalTrackingSettings settings) {
    std::ifstream stream(config.at("file").get<std::string>());const J ray=J::parse(stream);
    const auto vec=[](const J &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    cg::FlightState birth;birth.birthPosition=vec(ray["birth_position"]);birth.birthValue=ray["birth_value"];
    birth.birthGradient=vec(ray["birth_gradient"]);birth.direction=vec(ray["direction"]).normalized();
    const auto origin=vec(ray["origin"]);const double age=(origin-birth.birthPosition).dot(birth.direction);
    const double ell=1/std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
    const double end=std::min(ray["length"].get<double>(),config.value("end_scale",2.)*ell-age);
    const double begin=std::max({0.,ray.value("tracking_start_distance",0.),config.value("begin_scale",0.)*ell-age});
    require(begin<end,"Empty chapter 2 replay window");
    if(config.at("raw").get<bool>()) {
        settings.coupledLocalScale=0;settings.chapter2StartScale=1e-12;settings.chapter2FallbackEvents=0;
    } else settings.chapter2StartScale=0;
    const bool collision=ray["domain"]=="path";const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    J trials=J::array();clearProfile();
    for(int i=0;i<config.value("trials",3);++i) {
        const unsigned seed=config.value("seed",20261024u)+i;cg::Random rng(seed);
        cg::ConditionalTrackingWork work;double hit,weight;
        const auto start=Clock::now();
        {
            cg::Profile::Scope medium(cg::Profile::Stage::Medium,collision ? cg::Profile::Domain::Path:cg::Profile::Domain::Shadow);
            cg::ConditionalSegment seg(field,birth,origin,birth.direction,end,settings,nullptr,L,L);
            weight=seg.track(rng,collision,hit,&work,begin);
        }
        const double elapsed=seconds(start);
        trials.push_back({{"seed",seed},{"seconds",elapsed},{"candidates",work.candidates},{"intervals",work.intervals},
                          {"visited",work.distance},{"result",weight},{"hit",hit}});
    }
    return {{"config",config},{"ray",ray["id"]},{"begin",begin},{"end",end},{"trials",trials},{"profile",snapshotProfile()}};
}

// First occupied coarse interval of a saved ray. Refine using only certified
// majorants, with no candidate or point-hazard samples in the decision.
J certifyFirstBlock(const J &config,const cg::GPSSField &field,const cg::ConditionalTrackingSettings &settings) {
    namespace mb=cg::MajorantBounds;
    std::ifstream file(config.at("file").get<std::string>());const J ray=J::parse(file);
    const auto vec=[](const J &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    cg::FlightState birth;birth.birthPosition=vec(ray["birth_position"]);birth.birthGradient=vec(ray["birth_gradient"]);
    birth.direction=vec(ray["direction"]).normalized();birth.birthValue=ray["birth_value"];
    const auto origin=vec(ray["origin"]);const double age=(origin-birth.birthPosition).dot(birth.direction);
    const double ell=1/std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
    double begin=std::max({0.,ray.value("tracking_start_distance",0.),ray.at("birth_split").get<double>()-age});
    double end=std::min(ray["length"].get<double>(),2*ell-age);
    require(settings.classicLocalBand && end>begin,"Missing band or empty window");
    ClassicLocalGrid::Cursor band(*settings.classicLocalBand,origin,birth.direction,begin,end);
    require(band.nextOccupiedRange(begin,end),"No occupied interval");
    cg::ConditionalFlightKernel kernel(field,birth,false,0,0,settings.tabulatedBounds);
    double hi=age+end,lo=age+begin,coarse;
    require(kernel.chapter2MacrocellMajorant(lo,hi,coarse),"Missing coarse certificate");
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    if(config.value("fine_cells",false)) {
        const auto *grid=dynamic_cast<const GpisSdfGrid *>(field.mean.get());
        require(grid!=nullptr,"Fine-cell comparison requires a trilinear grid");
        auto cursor=field.mean->createRayCursor(birth.birthPosition,birth.direction);
        const cg::Vector3 q=grid->worldToIndex*birth.birthPosition+grid->translation;
        const cg::Vector3 v=grid->worldToIndex*birth.direction;
        const auto jet=field.mean->evaluate(birth.birthPosition);
        cg::MeanRayBounds macro;
        double clipped=hi;
        require(cursor->coarseRayBounds(lo,clipped,macro) && clipped==hi,"Coarse extent mismatch");
        J cells=J::array();
        for(double a=lo;a<hi;) {
            const double b=cursor->nextBoundary(a,hi);
            require(b>a && b<=hi,"Fine-cell traversal failed");
            const Eigen::Vector3i cell=(q+(.5*a+.5*b)*v).array().floor().cast<int>();
            for(int axis=0;axis<3;++axis)
                require(cell[axis]>=grid->data.clampMin[axis] && cell[axis]+1<=grid->data.clampMax[axis],
                        "Fine-cell audit requires an interior cell");
            J nodes=J::array();
            for(int z=0;z<2;++z)for(int y=0;y<2;++y)for(int x=0;x<2;++x) {
                const auto &o=grid->data.origin;
                nodes.push_back(grid->scale*(grid->at(cell.x()+x-o.x(),cell.y()+y-o.y(),cell.z()+z-o.z())+grid->offset));
            }
            const auto m=cursor->rayBounds(a,b);
            require(m.certified,"Missing ray-local fine-cell bound");
            const auto coupled=kernel.coupledIntervalMajorant(a,b,L,true,coarse,false,true,true);
            cells.push_back({{"lo",a},{"hi",b},{"index",{cell.x(),cell.y(),cell.z()}},{"nodes_zyx",nodes},
                             {"ray_bounds",{m.minimumValue,m.maximumValue,m.minimumDerivative}},
                             {"coupled_rate",coupled}});
            a=b;
        }
        return {{"id",ray["id"]},{"lo",lo},{"hi",hi},{"coarse_rate",coarse},{"cells",cells},
                {"q",{q.x(),q.y(),q.z()}},{"v",{v.x(),v.y(),v.z()}},{"sigma",field.kernel.sigma()},{"ell",ell},
                {"df",birth.birthValue-jet.value},{"dk",birth.direction.dot(birth.birthGradient-jet.gradient)},
                {"coarse_bounds",{macro.minimumValue,macro.maximumValue,macro.minimumDerivative}}};
    }
    const double target=config.value("survival",.99);
    require(target>0 && target<1,"Invalid survival target");
    const double budget=mb::down(-std::log(target)/(1+1e-12));
    const auto bound=[&](double a,double b,double parent){
        return kernel.coupledIntervalMajorant(a,b,L,true,parent,false,true,true);
    };
    const auto cost=[](double a,double b,double m){return mb::up(mb::up(b-a)*m);};
    struct Leaf {double a,b,maximum,depth;};
    std::vector<Leaf> leaves;
    int queries=0;J trace=J::array();
    double root=bound(lo,hi,coarse);++queries;leaves.push_back({lo,hi,root,cost(lo,hi,root)});
    const int limit=config.value("max_leaves",128);
    for(;;) {
        double sum=0;for(const auto &leaf:leaves)sum=mb::up(sum+leaf.depth);
        trace.push_back({{"leaves",leaves.size()},{"queries",queries},{"upper_depth",sum}});
        if(sum<=budget || int(leaves.size())>=limit)break;
        auto worst=std::max_element(leaves.begin(),leaves.end(),[](const Leaf &a,const Leaf &b){return a.depth<b.depth;});
        const Leaf old=*worst;const double mid=.5*old.a+.5*old.b;
        if(!(mid>old.a && mid<old.b))break;
        const double ml=bound(old.a,mid,old.maximum),mr=bound(mid,old.b,old.maximum);queries+=2;
        *worst={old.a,mid,ml,cost(old.a,mid,ml)};
        leaves.push_back({mid,old.b,mr,cost(mid,old.b,mr)});
    }
    return {{"id",ray["id"]},{"lo",lo},{"hi",hi},{"coarse_depth",cost(lo,hi,coarse)},
            {"target",target},{"trace",trace},{"confirmed",trace.back()["upper_depth"].get<double>()<=budget}};
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
        std::unique_ptr<ClassicLocalGrid> alternateBand;
        if (spec.contains("alternate_band")) {
            JsonDocument fieldDocument(Path(argv[1]),spec.at("alternate_band").dump());
            ClassicLocalField field;
            field.fromJson(fieldDocument,*scene);
            alternateBand=field.load();
        }
        const double initializationSeconds=seconds(loading);
        if(spec.contains("voxel_ray_file"))VoxelRayExperiment::compare(spec,output,medium->field(),tracking);
        if(spec.contains("ray_files"))exportIntegralRays(spec,output,medium->field());
        J report={{"spec",spec},{"initialization_seconds",initializationSeconds},{"runs",J::array()}};
        if(spec.contains("certify_first_blocks")) {
            J certificates=J::array();
            for(const auto &config:spec.at("certify_first_blocks"))
                certificates.push_back(certifyFirstBlock(config,medium->field(),tracking));
            std::ofstream saved(output/"survival_bounds.json");saved<<certificates.dump(2);
            require(bool(saved),"Cannot write first-block survival bounds");
        }
        if(spec.contains("chapter2_ray_files")) {
            const auto audit=chapter2RayChecks(spec.at("chapter2_ray_files"),medium->field(),tracking,
                                               spec.at("chapter2_audit_scale").get<double>());
            std::ofstream saved(output/"chapter2_rays.json");saved<<audit.dump(2);
            require(bool(saved),"Cannot write chapter 2 ray audit");
        }
        if(spec.contains("chapter2_factor_scales")) {
            J audits=J::array();
            for(double scale:spec.at("chapter2_factor_scales"))
                audits.push_back(chapter2RayChecks(spec.at("optimization_ray_files"),medium->field(),tracking,scale,true,
                    spec.value("chapter2_factor_end_scale",std::numeric_limits<double>::infinity())));
            std::ofstream saved(output/"chapter2_factors.json");saved<<audits.dump(2);
            require(bool(saved),"Cannot write chapter 2 factor audit");
        }
        std::cout<<"Loaded scene once in "<<report["initialization_seconds"]<<" s"<<std::endl;
        if(spec.contains("chapter2_window_replays")) {
            J windows=J::array();
            for(const auto &config:spec.at("chapter2_window_replays")) {
                windows.push_back(chapter2WindowReplay(config,medium->field(),tracking));
                std::ofstream saved(output/"chapter2_timing.json");saved<<windows.dump(2);
                require(bool(saved),"Cannot write chapter 2 timings");
            }
        }
#if defined(TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE) && TUNGSTEN_ENABLE_CONDITIONAL_RAY_PROFILE
        if(spec.contains("no_split_ray_files"))compareTransitionNoSplitRays(spec["no_split_ray_files"],output,medium->field(),tracking);
#endif
        for(const auto &run:spec.at("runs")) {
            const std::string label=run.at("label");
            require(!label.empty()&&label.find_first_not_of("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-")==std::string::npos,"Unsafe run label");
            integrator->teardownAfterRender();scene->camera()->teardownAfterRender();
            scene->camera()->prepareForRender();scene->camera()->requestOutputBuffers(scene->rendererSettings().renderOutputs());
            tracking=originalTracking;
            tracking.chapter2StartScale=run.value("chapter2_start",originalTracking.chapter2StartScale);
            tracking.chapter2FallbackEvents=run.value("chapter2_fallback",originalTracking.chapter2FallbackEvents);
            tracking.chapter2ParentEvents=run.value("chapter2_parent_events",originalTracking.chapter2ParentEvents);
            tracking.chapter2ParentScreen=run.value("chapter2_parent_screen",originalTracking.chapter2ParentScreen);
            const std::string mode=run.at("mode");
            require(mode=="current_two"||mode=="old_three"||mode=="integral_three"||mode=="coupled","Unknown transition mode");
            tracking.transitionPointBounds=mode=="old_three";
            tracking.transitionIntegralBounds=mode=="integral_three";
            tracking.tabulatedBounds=run.value("tabulated",originalTracking.tabulatedBounds);
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
            tracking.exteriorCutoffSigma=run.value("exterior_cutoff_sigma",originalTracking.exteriorCutoffSigma);
            if (run.value("use_alternate_band",false)) {
                require(bool(alternateBand),"Run requires an alternate_band field in the specification");
                tracking.classicLocalBand=alternateBand.get();
            }
            if (!run.value("use_classic_local_band",tracking.classicLocalBand!=nullptr))
                tracking.classicLocalBand=nullptr;
            else require(tracking.classicLocalBand!=nullptr,"Requested a baked band absent from the scene");
            tracking.validate();
            double macroBuildSeconds=0;
            if(tracking.macrocellBounds) {
                auto *grid=const_cast<GpisSdfGrid *>(dynamic_cast<const GpisSdfGrid *>(medium->field().mean.get()));
                require(grid,"Macrocell benchmark requires a tabulated SDF");
                const bool parents=tracking.chapter2ParentEvents>0;
                if(grid->macrocellSize!=tracking.macrocellSize || (parents && grid->parents.empty())) {
                    const auto build=Clock::now();grid->buildMacrocells(tracking.macrocellSize,parents);
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
