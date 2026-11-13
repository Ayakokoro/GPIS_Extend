// Scene-native training probes. No rendered images or fixed neural ray grid.
#include "SceneField.hpp"
#include "io/Scene.hpp"
#include "math/SparseHistoryGaussian.hpp"
#include "math/conditional/transport/SinglePointHistoryTeacher.h"
#include "math/conditional/transport/SinglePointNeural.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "sampling/IndependentPathSampler.hpp"
#include <openvdb/openvdb.h>
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#ifdef _OPENMP
#include <omp.h>
#endif

using namespace Tungsten;
namespace cg = Tungsten::ConditionalGpis;
namespace sp = Tungsten::SinglePointTraining;
using J = nlohmann::json;
using V = Eigen::Vector3d;
namespace fs = std::filesystem;

namespace {
struct Options {
    std::string scene, output;
    int views = 12, rays = 128, queries = 8, threads = 4;
    std::uint64_t seed = 20261113;
    double minimumAge = .05, minimumSeparation = .01;
    bool inspect = false, cameraOnly = false;
    std::optional<cg::Bounds3> bounds;
};

void usage() {
    std::cout << "Usage: gpis_single_point_data --scene scene.json --output directory [options]\n"
        "  --inspect                 Validate/load the one SDF, write manifest only\n"
        "  --views N                 Native camera plus orbit views (default 12, >=3)\n"
        "  --rays-per-view N         Camera ray probes per view (default 128)\n"
        "  --queries-per-ray N       Continuous random queries per ray (default 8)\n"
        "  --threads N               CPU workers (default 4)\n"
        "  --seed N                  Deterministic seed\n"
        "  --camera-only             One original camera, diagnostic train split only\n"
        "  --bounds xmin ymin zmin xmax ymax zmax  Override sampling domain\n"
        "  --minimum-age-over-length X           Default .05\n"
        "  --minimum-separation-over-length X    Default .01\n";
}

Options parse(int argc, char **argv) {
    Options o;
    auto number = [&](int &i) -> std::string {
        if (++i >= argc) throw std::invalid_argument("Missing argument value");
        return argv[i];
    };
    for (int i=1;i<argc;++i) {
        const std::string a=argv[i];
        if (a=="--scene") o.scene=number(i);
        else if(a=="--output") o.output=number(i);
        else if(a=="--views") o.views=std::stoi(number(i));
        else if(a=="--rays-per-view") o.rays=std::stoi(number(i));
        else if(a=="--queries-per-ray") o.queries=std::stoi(number(i));
        else if(a=="--threads") o.threads=std::stoi(number(i));
        else if(a=="--seed") o.seed=std::stoull(number(i));
        else if(a=="--minimum-age-over-length") o.minimumAge=std::stod(number(i));
        else if(a=="--minimum-separation-over-length") o.minimumSeparation=std::stod(number(i));
        else if(a=="--inspect") o.inspect=true;
        else if(a=="--camera-only") o.cameraOnly=true;
        else if(a=="--bounds") {
            cg::Bounds3 b;
            for(int j=0;j<3;++j) b.minimum[j]=std::stod(number(i));
            for(int j=0;j<3;++j) b.maximum[j]=std::stod(number(i));
            if(!b.valid()) throw std::invalid_argument("Invalid --bounds");
            o.bounds=b;
        } else throw std::invalid_argument("Unknown option: "+a);
    }
    if(o.scene.empty() || o.output.empty()) throw std::invalid_argument("--scene and --output are required");
    if(o.cameraOnly) o.views=1;
    if((!o.cameraOnly && o.views<3) || o.views>10000 || o.rays<1 || o.queries<1 ||
       o.threads<1 || o.threads>1024 || !(o.minimumAge>0) || !std::isfinite(o.minimumAge) ||
       !(o.minimumSeparation>0) || !std::isfinite(o.minimumSeparation))
        throw std::invalid_argument("Invalid sampling settings");
    return o;
}

V eigen(const Vec3f &v) { return {v.x(),v.y(),v.z()}; }
Vec3f vec(const V &v) { return {float(v.x()),float(v.y()),float(v.z())}; }
J jsonVector(const V &v) { return J::array({v.x(),v.y(),v.z()}); }
std::uint64_t mixed(std::uint64_t x) {
    x+=0x9e3779b97f4a7c15ull; x=(x^(x>>30))*0xbf58476d1ce4e5b9ull;
    x=(x^(x>>27))*0x94d049bb133111ebull; return x^(x>>31);
}
void save(const fs::path &path,const J &value) {
    std::ofstream f(path); f<<value.dump(2)<<'\n';
    if(!f) throw std::runtime_error("Cannot write "+path.string());
}

struct Probe { bool valid=false; V origin,direction; double length=0; };
struct Result {
    std::string csv;
    std::size_t rows=0, queries=0;
    std::map<std::string,std::size_t> rejected;
};

Result generateRay(const Options &o,const cg::GPSSField &field,const Probe &ray,
                   int view,std::uint64_t rayId,int split,double ell) {
    Result out;
    if(!ray.valid) {out.rejected["camera_ray_missed_sdf_domain"]=1;return out;}
    try {
        field.mean->requireFullRayCoverage(ray.origin,ray.direction,ray.length);
        cg::Random rng(mixed(o.seed^mixed(rayId)));
        const auto birth=cg::startConditionalExterior(field,ray.origin,ray.direction,rng);
        cg::SinglePointHistoryTeacher teacher(field,birth);
        const double separation=o.minimumSeparation*ell;
        const double minimum=std::max(o.minimumAge*ell,2*separation);
        if(!(ray.length>minimum+separation)) {out.rejected["short_ray"]=1;return out;}
        const auto pool=cg::buildSinglePointHistoryAnchors(field,birth,ray.length,separation,teacher.baseline());
        std::vector<std::size_t> focus;
        for(std::size_t i=0;i<pool.size();++i)
            if(std::abs(pool[i].mean)<=3*field.kernel.sigma()) focus.push_back(i);
        std::ostringstream csv;csv<<std::setprecision(17);
        for(int query=0;query<o.queries;++query) {
            ++out.queries;
            double t=minimum+(ray.length-minimum)*rng.openUniform01();
            // A continuous mixture improves coverage near the SDF zero band;
            // neither component defines a rendering/network marching step.
            if(query%2 && !focus.empty()) {
                const auto k=focus[std::min(focus.size()-1,std::size_t(rng.uniform01()*focus.size()))];
                const double proposed=pool[k].t+.25*ell*rng.standardNormal();
                if(proposed>minimum && proposed<ray.length) t=proposed;
            }
            try {
                const double u=cg::selectSinglePointHistoryAnchor(pool,t,ell,separation);
                const auto sample=teacher.evaluate(t,u);
                csv<<view<<','<<rayId<<','<<split;
                for(double x:sample.features) csv<<','<<x;
                csv<<','<<sample.logBaselineHazard<<','<<sample.logHistoryHazard
                   <<','<<sample.logHistoryPositiveProbability<<','<<sample.logCorrection
                   <<','<<t<<','<<u;
                for(int i=0;i<3;++i) csv<<','<<ray.origin[i];
                for(int i=0;i<3;++i) csv<<','<<ray.direction[i];
                csv<<','<<birth.birthValue;
                for(int i=0;i<3;++i) csv<<','<<birth.birthGradient[i];
                csv<<','<<sample.momentRelativeError<<'\n';
                ++out.rows;
            } catch(const cg::NumericError &e) {++out.rejected[std::string("query: ")+e.what()];}
        }
        out.csv=csv.str();
    } catch(const std::exception &e) {
        // A failed ray never publishes a partial buffer or counts unpublished
        // rows. Keep its attempted-query count and explicit rejection reason.
        out.rows=0;out.csv.clear();
        ++out.rejected[std::string("ray: ")+e.what()];
    }
    return out;
}
}

int main(int argc,char **argv) {
    if(argc==1 || (argc==2 && std::string(argv[1])=="--help")) {usage();return 0;}
    try {
        const auto o=parse(argc,argv);
        const auto output=fs::absolute(o.output);
        if(fs::exists(output/"manifest.json") || fs::exists(output/"samples.csv") || fs::exists(output/"samples.partial.csv"))
            throw std::runtime_error("Output already contains a dataset; choose a new output directory");
        openvdb::initialize();
        auto source=sp::loadSceneField(Path(fs::absolute(o.scene).string()),o.bounds);
        const auto &field=source.field;
        const double ell=1/std::sqrt(field.kernel.metric()(0,0));
        const auto start=std::chrono::steady_clock::now();
        J manifest={{"schema","tungsten.single_point_history.v1"},{"status","loaded"},
            {"sigma",field.kernel.sigma()},{"length_scale",ell},{"source",source.metadata},
            {"feature_names",{"z_x","z_u","z_v","rho_xu","rho_xv","rho_uv"}},
            {"birth_condition","sample full F0,gradient0 given F0>0; same as global_conditional"},
            {"history_condition","one passed point F(u)>0"},
            {"teacher","numerical Gaussian single-history flux / joint survival; full birth retained"},
            {"sampling",{{"views",o.views},{"rays_per_view",o.rays},{"queries_per_ray",o.queries},
                {"seed",o.seed},{"threads",o.threads},{"camera_only",o.cameraOnly},
                {"minimum_age_over_length",o.minimumAge},{"minimum_separation_over_length",o.minimumSeparation},
                {"query_proposal","half uniform physical age, half zero-band geometry + normal(0,0.25*ell)"},
                {"anchor_selection","max existing single-point correlation/Mills score among passed SDF cell endpoints, midpoints, available stationary roots; t/2 fallback"},
                {"anchor_horizon","fixed sampling-domain ray exit; same shared implementation as renderer"},
                {"anchor_implementation","shared_geometry_pool_v2"},
                {"unsupported_stationary_root_behavior","uses available endpoints/midpoints only; no claim to enumerate all mean extrema"},
                {"network_step",nullptr},{"ordinary_geometry_occlusion",false},
                {"path_distribution","independent exterior camera-flight probes, not survival-weighted rendered paths"}}}};
        fs::create_directories(output);
        if(o.inspect) {
            manifest["status"]="inspected";save(output/"manifest.json",manifest);
            std::cout<<"Loaded one "<<field.mean->typeName()<<" SDF; sigma="<<field.kernel.sigma()<<", length="<<ell<<'\n';return 0;
        }
        auto camera=source.scene->camera();
        if(!camera) throw std::runtime_error("Scene needs a camera for ray probes");
        camera->loadResources();
        const V center=.5*(field.activeDomain.minimum+field.activeDomain.maximum);
        const V originalPosition=eigen(camera->pos()),originalUp=eigen(camera->up());
        const double radius=.5*(field.activeDomain.maximum-field.activeDomain.minimum).norm();
        const double orbitRadius=std::max((originalPosition-center).norm(),3*radius);
        V axis=originalUp.normalized(),basisA,basisB;
        if(!axis.allFinite()) axis=V::UnitY();
        cg::orthonormalComplement(axis,basisA,basisB);
        std::ofstream csv(output/"samples.partial.csv");
        if(!csv) throw std::runtime_error("Cannot open samples.partial.csv");
        csv<<"view_id,ray_id,split,z_x,z_u,z_v,rho_xu,rho_xv,rho_uv,log_alpha_base,log_alpha_history,log_p_history,log_correction,t,u,origin_x,origin_y,origin_z,dir_x,dir_y,dir_z,birth_f,birth_gx,birth_gy,birth_gz,moment_relative_error\n";
        std::array<std::size_t,3> counts{};
        std::size_t total=0,queries=0;std::map<std::string,std::size_t> rejected;
        J views=J::array();
        manifest["status"]="generating";save(output/"manifest.json",manifest);
#ifdef _OPENMP
        omp_set_num_threads(o.threads);
#endif
        const int validationViews=o.cameraOnly?0:std::max(1,o.views/5);
        std::vector<int> viewSplits(o.views,0),heldOutOrder;
        for(int i=1;i<o.views;++i) heldOutOrder.push_back(i);
        std::mt19937_64 splitRandom(mixed(o.seed^0x53504c4954ull));
        std::shuffle(heldOutOrder.begin(),heldOutOrder.end(),splitRandom);
        for(int i=0;i<validationViews;++i) {
            viewSplits[heldOutOrder[i]]=1;
            viewSplits[heldOutOrder[i+validationViews]]=2;
        }
        for(int view=0;view<o.views;++view) {
            const int split=viewSplits[view];
            // Keep the original camera matrix exactly, including any explicit
            // scale/handedness; rebuilding it from look-at loses that state.
            if(view!=0) {
                const double z=1-2*(double(view)-.5)/double(o.views-1);
                const double phi=view*2.399963229728653;
                const V radial=z*axis+std::sqrt(std::max(0.,1-z*z))*(std::cos(phi)*basisA+std::sin(phi)*basisB);
                const V up=std::abs(radial.dot(axis))>.97?basisA:axis;
                camera->setTransform(vec(center+orbitRadius*radial),vec(center),vec(up));
            }
            camera->prepareForRender();
            views.push_back({{"view_id",view},{"split",split},{"position",jsonVector(eigen(camera->pos()))},
                {"look_at",jsonVector(eigen(camera->lookAt()))},{"up",jsonVector(eigen(camera->up()))}});
            std::vector<Probe> rays(o.rays);std::vector<Result> results(o.rays);
            for(int i=0;i<o.rays;++i) {
                const std::uint64_t id=std::uint64_t(view)*o.rays+i;
                IndependentPathSampler sampler(std::uint32_t(mixed(o.seed)),false);sampler.startPath(std::uint32_t(id),0);
                const auto res=camera->resolution();
                if(!res.x() || !res.y()) throw std::runtime_error("Camera resolution is zero");
                const Vec2u pixel(std::min(res.x()-1,unsigned(sampler.next1D()*res.x())),
                                  std::min(res.y()-1,unsigned(sampler.next1D()*res.y())));
                PositionSample p;DirectionSample d;
                if(!camera->samplePosition(sampler,p) || !camera->sampleDirection(sampler,p,pixel,d)) continue;
                const V direction=eigen(d.d).normalized(),origin=eigen(p.p);
                const auto interval=field.activeDomain.intersect({origin,direction});
                if(!interval.hit || !(interval.exit>interval.entry) || !std::isfinite(interval.exit)) continue;
                // Move into the represented node domain, avoiding float camera
                // roundoff at an exact boundary; record the actual new birth.
                const double pad=std::min(1e-7*std::max(ell,radius),.01*(interval.exit-interval.entry));
                const V birthOrigin=origin+(interval.entry+pad)*direction;
                const auto canonical=field.activeDomain.intersect({birthOrigin,direction});
                rays[i]={true,birthOrigin,direction,canonical.exit};
            }
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
            for(int i=0;i<o.rays;++i)
                results[i]=generateRay(o,field,rays[i],view,std::uint64_t(view)*o.rays+i,split,ell);
            for(const auto &result:results) {
                csv<<result.csv;total+=result.rows;counts[split]+=result.rows;queries+=result.queries;
                for(const auto &entry:result.rejected) rejected[entry.first]+=entry.second;
            }
            csv.flush();if(!csv) throw std::runtime_error("Dataset write failed");
            camera->teardownAfterRender();
            std::cout<<"View "<<view+1<<'/'<<o.views<<": "<<total<<" retained rows\n"<<std::flush;
            manifest["rows"]=total;manifest["split_counts"]=counts;manifest["views"]=views;
            manifest["rejections"]=rejected;save(output/"manifest.json",manifest);
        }
        csv.close();
        if(!csv) throw std::runtime_error("Dataset close/flush failed");
        manifest["query_attempts"]=queries;
        manifest["elapsed_seconds"]=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
        if(!total || (!o.cameraOnly && (!counts[0] || !counts[1] || !counts[2]))) {
            manifest["status"]="failed_empty_partition";save(output/"manifest.json",manifest);
            throw std::runtime_error("No retained rows in one or more partitions; inspect rejections and camera/domain coverage");
        }
        fs::rename(output/"samples.partial.csv",output/"samples.csv");
        manifest["status"]=o.cameraOnly?"complete_diagnostic":"complete";save(output/"manifest.json",manifest);
        std::cout<<"Wrote "<<total<<" rows to "<<(output/"samples.csv").string()<<'\n';
        return 0;
    } catch(const std::exception &e) {std::cerr<<"gpis_single_point_data: "<<e.what()<<'\n';return 1;}
}
