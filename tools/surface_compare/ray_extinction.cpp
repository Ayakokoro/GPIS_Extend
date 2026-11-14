// Export the actual classic and conditional laws on a frozen camera path.
// Distance/phase sampling only selects the birth observations; curve evaluation
// never samples or renews them. No production transport behavior is changed.
#include "media/GlobalConditionalMedium.hpp"
#include "cameras/PinholeCamera.hpp"
#include "samplerecords/PositionSample.hpp"
#include "samplerecords/DirectionSample.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <openvdb/openvdb.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <vector>

using namespace Tungsten;
namespace cg=Tungsten::ConditionalGpis;
using J=nlohmann::json;
using V=Eigen::Vector3d;
J read(const std::string &path) {
    std::ifstream f(path);if(!f)throw std::runtime_error("Cannot open "+path);
    return J::parse(f);
}
J json(const V &v) {return J::array({v.x(),v.y(),v.z()});}
V vector(const J &v) {return {v[0].get<double>(),v[1].get<double>(),v[2].get<double>()};}
V eigen(const Vec3f &v) {return {v.x(),v.y(),v.z()};}
Vec3f vec(const V &v) {return Vec3f(float(v.x()),float(v.y()),float(v.z()));}
Vec3d vecd(const V &v) {return Vec3d(v.x(),v.y(),v.z());}
std::shared_ptr<ClassicGlobalMedium> medium(Scene &scene,const J &definition,const std::string &path) {
    JsonDocument document(Path(path),definition.dump());
    auto result=std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(document));
    if(!result)throw std::runtime_error("Expected classic or conditional medium");
    result->loadResources();result->prepareForRender();return result;
}
bool interval(const V &o,const V &w,const V &lo,const V &hi,double &begin,double &end) {
    begin=0;end=1e30;
    for(int i=0;i<3;++i) {
        if(w[i]==0) {if(o[i]<lo[i]||o[i]>hi[i])return false;continue;}
        double a=(lo[i]-o[i])/w[i],b=(hi[i]-o[i])/w[i];if(a>b)std::swap(a,b);
        begin=std::max(begin,a);end=std::min(end,b);
    }
    return end>begin;
}
double horizon(const V &o,const V &w,const V &lo,const V &hi) {
    double a,b;if(!interval(o,w,lo,hi,a,b))throw std::runtime_error("Ray misses process box");
    // The frozen shaderball scene has a floor at y=0. Stop before it, rather
    // than extending a conditional flight through an ordinary surface event.
    if(w.y()<0 && o.y()>0)b=std::min(b,-o.y()/w.y());
    return b;
}
struct FrozenRay {cg::FlightState birth;double length=0;J sampledEvent;};
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: ray_extinction_probe spec.json curves.json");
        openvdb::initialize();const J spec=read(argv[1]);
        const std::string path=spec.at("scene");const J source=read(path);
        J definition=source.at("media").at(0);definition["type"]="global_conditional";
        definition["fp_gradient"]={{"enabled",false}};
        Scene scene;auto conditional=std::dynamic_pointer_cast<GlobalConditionalMedium>(medium(scene,definition,path));
        const V lo=vector(spec.at("box_min")),hi=vector(spec.at("box_max"));
        JsonDocument cd(Path(path),source.at("camera").dump());
        PinholeCamera camera;camera.fromJson(cd,scene);camera.prepareForRender();
        const unsigned x=spec.at("pixel")[0],y=spec.at("pixel")[1];
        const unsigned sample=spec.value("sample",0u),seed=spec.value("seed",20261024u);
        if(x>=camera.resolution().x()||y>=camera.resolution().y())throw std::runtime_error("Pixel outside image");
        IndependentPathSampler rng(seed,false);rng.startPath(x+y*camera.resolution().x(),sample);
        PositionSample position;DirectionSample direction;
        camera.samplePosition(rng,position);camera.sampleDirection(rng,position,Vec2u(x,y),direction);
        const V cameraOrigin=eigen(position.p),cameraDirection=eigen(direction.d).normalized();
        double entry,exit;if(!interval(cameraOrigin,cameraDirection,lo,hi,entry,exit))throw std::runtime_error("Camera ray misses box");
        const Vec3f firstOrigin=vec(cameraOrigin+entry*cameraDirection);
        const V firstDirection=eigen(direction.d.normalized()).normalized();
        cg::Random random(rng);
        auto context=std::make_shared<ConditionalPathContext>();context->owner=conditional.get();
        context->position=eigen(firstOrigin);context->rayOrigin=firstOrigin;
        context->birth=cg::startConditionalExterior(conditional->field(),context->position,firstDirection,random);
        Medium::MediumState state{};state.reset();state.pathContext=context;
        state.info.pixelSampleSegment=Vec4u(x,y,sample,0u);state.info.sceneSeed=seed;state.info.t=0;
        Vec3f origin=firstOrigin,travel=direction.d.normalized();
        std::vector<FrozenRay> rays;
        for(int bounce=0;bounce<2;++bounce) {
            auto birth=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext)->birth;
            birth.direction=eigen(travel).normalized();birth.age=0;
            const double length=horizon(birth.birthPosition,birth.direction,lo,hi);
            MediumSample hit{};
            if(!conditional->sampleDistance(rng,Ray(origin,travel,0,float(length)),state,hit))
                throw std::runtime_error("Production distance sample rejected the selected path");
            J event={{"exited",hit.exited}};
            if(!hit.exited) {
                const auto e=std::dynamic_pointer_cast<const ConditionalEventContext>(hit.eventContext);
                event["distance"]=(e->position-birth.birthPosition).dot(birth.direction);
                event["position"]=json(e->position);
            }
            rays.push_back({birth,length,event});
            if(hit.exited)break;
            PhaseSample phase{};
            if(!hit.phase->sample(rng,travel,hit,phase))throw std::runtime_error("Production mirror phase rejected selected path");
            state.pathContext=phase.pathContext;origin=hit.p;travel=phase.w;
        }
        if(rays.size()!=2)throw std::runtime_error("Selected camera ray has no first GP reflection; choose another pixel/sample");
        J output={{"schema",1},{"spec",spec},{"source_medium",definition},
            {"camera_origin",json(cameraOrigin)},{"camera_direction",json(cameraDirection)},
            {"birth_source_length_scale",definition.at("gaussian_process").at("covariance").at("lengthScale")},
            {"note","Birth jets and rays sampled once with the production camera, exterior, distance and mirror phase laws. "
                "The two length scales reuse these exact birth observations and geometric rays. No NEE/roulette RNG draws. "
                "Profiles continue virtually beyond a sampled GP event without renewal, until the next box/floor boundary. "
                "At t=0 the conditional renderer returns its exact-zero convention; positive t is exported separately."},
            {"rays",J::array()},{"checks",{{"all_finite_nonnegative",true}}}};
        const int denominator=spec.value("denominator",16384);
        if(denominator<256||denominator>262144)throw std::runtime_error("Invalid curve denominator");
        for(size_t ri=0;ri<rays.size();++ri) {
            const auto &ray=rays[ri];
            J record={{"kind",ri==0?"camera_entry":"first_reflection"},
                {"origin",json(ray.birth.birthPosition)},{"direction",json(ray.birth.direction)},
                {"birth_value",ray.birth.birthValue},{"birth_gradient",json(ray.birth.birthGradient)},
                {"birth_axial_gradient",ray.birth.direction.dot(ray.birth.birthGradient)},
                {"length",ray.length},{"sampled_event",ray.sampledEvent},{"profiles",J::array()}};
            for(double ell:spec.at("length_scales")) {
                auto modified=definition;modified["gaussian_process"]["covariance"]["lengthScale"]=ell;
                auto cm=std::dynamic_pointer_cast<GlobalConditionalMedium>(medium(scene,modified,path));
                modified["type"]="classic_global";auto classic=medium(scene,modified,path);
                const auto &field=cm->field();cg::ConditionalFlightKernel kernel(field,ray.birth);
                std::set<double> times{0,ray.length};
                for(int i=1;i<denominator;++i)times.insert(ray.length*i/denominator);
                // Resolve both the near-birth singular scaling and SDF surface
                // width without assigning the t=0 convention to finite ages.
                for(int i=0;i<=512;++i) {
                    const double t=ell*std::pow(10.,-7.+8.*i/512);if(t<ray.length)times.insert(t);
                }
                if(ray.sampledEvent.contains("distance"))times.insert(ray.sampledEvent.at("distance").get<double>());
                const double axialVariance=ray.birth.direction.dot(field.kernel.gradientCovarianceAtZero()*ray.birth.direction);
                J points=J::array();double maxMeanError=0,maxHazardError=0;
                for(double t:times) {
                    const V p=ray.birth.birthPosition+t*ray.birth.direction;const auto mean=field.mean->evaluate(p);
                    const double a=classic->extinction(vecd(p),vecd(ray.birth.direction));
                    const auto evaluation=kernel.evaluate(t).hazard;const double h=evaluation.value;
                    if(!std::isfinite(a)||a<0||!std::isfinite(h)||h<0)throw std::runtime_error("Invalid curve value");
                    maxMeanError=std::max(maxMeanError,std::abs(mean.value-classic->meanValue(vecd(p))));
                    J row={{"t",t},{"mean",mean.value},{"mean_derivative",ray.birth.direction.dot(mean.gradient)},
                        {"mean_gradient_norm",mean.gradient.norm()},{"classic",a},{"conditional",h}};
                    if(t>0) {
                        const auto s=kernel.scalarStatistics(t);
                        const double sd=std::sqrt(s.varianceF),sdAtZero=std::sqrt(s.varianceAtZero);
                        row["conditional_mean"]=s.meanF;row["conditional_sd"]=sd;
                        row["crossing_mean"]=s.meanAtZero;row["crossing_sd"]=sdAtZero;
                        const double independent=ClassicGlobal::density(s.meanF,sd)*ClassicGlobal::negativePartMean(s.meanAtZero,sdAtZero);
                        if(!std::isfinite(independent))throw std::runtime_error("Nonfinite independent hazard check");
                        maxHazardError=std::max(maxHazardError,std::abs(h-independent)/(1+std::abs(h)));
                    }
                    const double priorFlux=ClassicGlobal::negativePartMean(ray.birth.direction.dot(mean.gradient),std::sqrt(axialVariance));
                    if(t==0)row["classic_crossing_flux"]=priorFlux;
                    points.push_back(std::move(row));
                }
                if(maxMeanError>1e-10||maxHazardError>1e-8)throw std::runtime_error("Shared mean / scalar hazard cross-check failed");
                record["profiles"].push_back({{"length_scale",ell},{"sigma",field.kernel.sigma()},
                    {"max_mean_error",maxMeanError},{"max_hazard_formula_error",maxHazardError},{"points",points}});
                cm->teardownAfterRender();classic->teardownAfterRender();
            }
            output["rays"].push_back(std::move(record));
        }
        // A real SDF section through the incoming and reflected lines locates
        // the selected ray and makes the surface/sign changes inspectable.
        const V center=rays[1].birth.birthPosition,e1=rays[0].birth.direction;
        V e2=rays[1].birth.direction-e1*e1.dot(rays[1].birth.direction);
        if(e2.norm()<1e-10)e2=e1.unitOrthogonal();else e2.normalize();
        const int n=spec.value("slice_denominator",256);const double extent=spec.value("slice_extent",1.5);
        if(n<16||n>1024||!(extent>0)||!std::isfinite(extent))throw std::runtime_error("Invalid SDF slice grid");
        J values=J::array();
        for(int y=0;y<=n;++y)for(int x=0;x<=n;++x)
            values.push_back(conditional->field().mean->evaluate(center+extent*(2.*x/n-1)*e1+extent*(2.*y/n-1)*e2).value);
        output["slice"]={{"center",json(center)},{"basis_u",json(e1)},{"basis_v",json(e2)},
            {"denominator",n},{"extent",extent},{"values",values}};
        const auto target=std::filesystem::absolute(argv[2]);std::filesystem::create_directories(target.parent_path());
        std::ofstream f(target);f<<output.dump();if(!f)throw std::runtime_error("Cannot write curves");
        conditional->teardownAfterRender();std::cout<<"Saved frozen camera and first-reflection curves to "<<target<<'\n';
        return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}
}
