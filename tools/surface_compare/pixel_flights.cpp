// Controlled native renewal+ versus conditional transport at frozen birth jets.
#include "media/GlobalConditionalMedium.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"
#include "math/conditional/transport/CollisionGradientSampler.h"
#include "math/conditional/mathutility/Quadrature.h"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <openvdb/openvdb.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <algorithm>

using namespace Tungsten;
namespace cg=Tungsten::ConditionalGpis;
namespace fp=Tungsten::FirstPassageGradient;
using J=nlohmann::json;
using V=Eigen::Vector3d;
J read(const std::string &p) {std::ifstream f(p);if(!f)throw std::runtime_error("Missing input: "+p);return J::parse(f);}
J vector(const V &p) {return J::array({p.x(),p.y(),p.z()});}
V vector(const J &p) {return {p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};}
Vec3f vec(const V &p) {return Vec3f(float(p.x()),float(p.y()),float(p.z()));}
V eigen(const Vec3f &p) {return {p.x(),p.y(),p.z()};}
std::shared_ptr<Medium> makeMedium(Scene &s,const J &j) {
    JsonDocument doc(Path("pixel-flight-medium.json"),j.dump());
    auto m=s.fetchMedium(doc);m->loadResources();m->prepareForRender();return m;
}
J parameters(const fp::Law &law) {
    return {{"mu",law.gaussianMean()},{"sd",law.gaussianStddev()},
        {"p",law.probitIntercept()},{"q",law.probitSlope()},
        {"anchor_distance",law.anchorDistance()},{"anchor_selection",fp::anchorSelectionName(law.anchorSelection())}};
}
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: pixel_flights spec.json result.json");
        openvdb::initialize();const J spec=read(argv[1]);Scene scene;
        J nd=read(spec.at("native_scene")) .at("media").at(0);
        J cd=read(spec.at("conditional_scene")).at("media").at(0);
        cd["fp_gradient"]={{"enabled",false}};
        if(spec.contains("step"))nd["step_size"]=spec["step"];
        auto native=makeMedium(scene,nd);
        auto conditional=std::dynamic_pointer_cast<GlobalConditionalMedium>(makeMedium(scene,cd));
        if(!conditional)throw std::runtime_error("Expected global_conditional");
        auto r=spec.at("record");const unsigned id=r.at("id"),seed=spec.value("seed",20261109u);
        const int count=spec.value("samples",8192),shadowCount=spec.value("shadow_samples",4096);
        const V origin=eigen(vec(vector(r.at("birth_position"))));
        const Vec3f direction=vec(vector(r.at("wi"))).normalized();
        const V w=eigen(direction).normalized();
        const double length=float(r.at("length").get<double>());
        cg::FlightState birth;birth.birthPosition=origin;birth.direction=w;
        birth.birthValue=float(r.at("f0").get<double>());
        birth.birthGradient=eigen(vec(vector(r.at("birth_gradient"))));
        if(birth.birthValue<0||(birth.birthValue==0&&w.dot(birth.birthGradient)<=0))
            throw std::runtime_error("Unsupported frozen birth");
        r["birth_position"]=vector(origin);r["wi"]=vector(w);r["f0"]=birth.birthValue;
        r["birth_gradient"]=vector(birth.birthGradient);r["g0"]=w.dot(birth.birthGradient);
        r["length"]=length;
        const auto &field=conditional->field();cg::ConditionalFlightKernel kernel(field,birth);
        const auto startMean=field.mean->evaluate(origin);
        const double gamma=std::sqrt(w.dot(field.kernel.metric()*w));
        r["sigma"]=field.kernel.sigma();r["gamma"]=gamma;
        r["birth_mean"]=startMean.value;r["birth_mean_gradient"]=vector(startMean.gradient);
        r["birth_mean_derivative"]=w.dot(startMean.gradient);
        auto input=[&](double t) {
            const auto end=field.mean->evaluate(origin+t*w);
            return fp::Input{field.kernel.sigma(),gamma,t,birth.birthValue,w.dot(birth.birthGradient),
                startMean.value,w.dot(startMean.gradient),w.dot(end.gradient),
                [&](double s){return field.mean->evaluate(origin+s*w).value;}};
        };
        auto state=[&](unsigned sample) {
            Medium::MediumState s{};s.reset();s.firstScatter=false;s.lastVal=float(birth.birthValue);
            s.lastAniso=Vec3d(birth.birthGradient.x(),birth.birthGradient.y(),birth.birthGradient.z());
            s.info.pixelSampleSegment=Vec4u(id,0u,sample,1u);s.info.sceneSeed=seed;s.info.t=0;
            return s;
        };
        if(spec.contains("replay_realization")) {
            const auto replay=spec.at("replay_realization");auto s=state(0);
            s.info.pixelSampleSegment=Vec4u(replay.at("pixel")[0].get<unsigned>(),replay.at("pixel")[1].get<unsigned>(),
                replay.at("sample").get<unsigned>(),replay.at("bounce").get<unsigned>());
            s.info.sceneSeed=replay.at("scene_seed");
            auto *m=dynamic_cast<SparseConvolutionNoiseMedium *>(native.get());
            IndependentPathSampler sampler(seed,false);sampler.startPath(0,0);PreparedGpisSegment prepared;
            m->prepareGpisSegment(sampler,Ray(vec(origin),direction,0,float(length)),s,prepared);
            const double horizon=replay.at("horizon");J points=J::array();UniformSampler uniform;int gpId;
            auto observe=[&](double t) {
                Vec3f position=vec(origin+t*w);
                double f=prepared.context->noise.evaluateValue(position,float(t),direction,s.info,uniform,gpId);
                auto g=prepared.context->noise.evaluateGradient(position,float(t),direction,s.info,uniform);
                return J{{"t",t},{"value",f},{"axial_gradient",w.dot(eigen(g))},
                    {"mean",field.mean->evaluate(origin+t*w).value}};
            };
            for(int i=0;i<=4096;++i)points.push_back(observe(horizon*i/4096));
            J out={{"spec",spec},{"record",r},{"points",points},{"at_candidate",observe(replay.at("candidate_distance"))}};
            std::ofstream f(argv[2]);f<<out.dump();if(!f)throw std::runtime_error("Write failed");return 0;
        }
        std::vector<double> times(count,-1);std::vector<V> gradients(count,V::Zero());
        std::vector<int> accepted;J statuses=J::array(),failures=J::object();int escaped=0;
        IndependentPathSampler sampler(seed,false);
        for(int i=0;i<count;++i) {
            sampler.startPath(id,unsigned(i));auto s=state(unsigned(i));MediumSample hit{};
            Medium::DistanceDiagnostic diagnostic;s.distanceDiagnostic=&diagnostic;
            const Ray ray(vec(origin),direction,0,float(length));
            if(!native->sampleDistance(sampler,ray,s,hit)) {
                std::string why=diagnostic.failure;failures[why]=failures.value(why,0)+1;statuses.push_back(why);
            } else if(hit.exited) {times[i]=length;++escaped;statuses.push_back("escape");}
            else {times[i]=hit.t;gradients[i]=V(hit.aniso.x(),hit.aniso.y(),hit.aniso.z());accepted.push_back(i);statuses.push_back("hit");}
        }
        std::cout<<"Native flights: "<<accepted.size()<<" accepted, "<<escaped<<" escaped / "<<count<<std::endl;
        std::vector<double> positive;
        for(int i:accepted)if(times[i]>0)positive.push_back(times[i]);
        std::sort(positive.begin(),positive.end());
        const double focus=r.value("focus_distance",positive.empty()?length*.5:positive[positive.size()/2]);
        r["focus_distance"]=focus;
        J gradientSamples=J::array();double halfWidth=0;int unsupported=0;
        if(!positive.empty()) {
            std::vector<double> distances;for(double t:positive)distances.push_back(std::abs(t-focus));
            std::sort(distances.begin(),distances.end());
            const size_t wanted=std::min<size_t>(1024,std::max<size_t>(128,positive.size()/4));
            halfWidth=distances[std::min(wanted-1,distances.size()-1)]+1e-12;
            for(int i:accepted)if(times[i]>0&&std::abs(times[i]-focus)<=halfWidth) {
                double t=times[i];sampler.startPath(id+100000u,unsigned(i));cg::Random rng(sampler);
                const auto gaussian=kernel.hitStatistics(t).collisionGradient;
                const V gc=cg::sampleFluxWeightedGradient(gaussian,w,rng);
                const fp::Law law(input(t),true,fp::AnchorSelection::Fixed5Snr);
                const double speed=law.sample(rng);
                const auto end=field.mean->evaluate(origin+t*w);
                const cg::FixedSpeedGradientDistribution fixed(field.kernel,w,gamma*t,speed,
                    end.gradient,startMean.gradient,birth.birthGradient);
                const V gf=fixed.sample(rng);
                if(!gc.allFinite()||!gf.allFinite())throw std::runtime_error("Nonfinite matched gradient");
                gradientSamples.push_back({{"t",t},{"renewal",vector(gradients[i])},
                    {"conditional",vector(gc)},{"fixed5",vector(gf)},
                    {"mean_gradient",vector(end.gradient)}});
            } else if(times[i]<=0)++unsupported;
        }
        std::cout<<"Matched gradients: "<<gradientSamples.size()<<", focus "<<focus<<", half-width "<<halfWidth<<std::endl;
        // The selected collision distance also gets an exact fixed-t SE-GP
        // bridge diagnostic, separate from finite-density native bin samples.
        if(focus>0) {
            r["t"]=focus;const fp::Law law(input(focus));r["production"]=parameters(law);
            const auto end=field.mean->evaluate(origin+focus*w);
            r["end_mean_derivative"]=w.dot(end.gradient);r["end_mean_gradient"]=vector(end.gradient);
            J means=J::array();const int n=spec.value("gradient_denominator",832);
            for(int i=0;i<=n;++i)means.push_back(field.mean->evaluate(origin+(focus*i/n)*w).value);
            r["profile_denominator"]=n;r["mean_profile"]=means;
        }
        J ages=J::array(),hazard=J::array(),cdf=J::array(),means=J::array();
        double integrated=0,error=0;const int n=spec.value("denominator",512);
        cg::NumericPolicy policy;policy.absoluteTolerance=1e-10;policy.relativeTolerance=1e-8;
        std::vector<double> grid;
        for(int i=0;i<=n;++i)grid.push_back(length*i/n);
        grid.push_back(std::min(length,std::max(0.,focus)));
        if(!positive.empty()) {
            const double lo=std::max(0.,positive[positive.size()/100]-.01);
            const double hi=std::min(length,positive[positive.size()*99/100]+.01);
            for(int i=0;i<=512;++i)grid.push_back(lo+(hi-lo)*i/512);
        }
        std::sort(grid.begin(),grid.end());grid.erase(std::unique(grid.begin(),grid.end()),grid.end());
        for(size_t i=0;i<grid.size();++i) {
            const double t=grid[i];
            if(i&&integrated<40) {
                auto q=cg::integrateFinite([&](double s){return kernel.evaluate(s).hazard.value;},grid[i-1],t,policy);
                if(q.status!=cg::NumericStatus::Ok)throw std::runtime_error("Hazard integral failed");
                integrated+=q.value;error+=q.absError;
            }
            ages.push_back(t);hazard.push_back(i?kernel.evaluate(t).hazard.value:0);
            means.push_back(field.mean->evaluate(origin+t*w).value);cdf.push_back(-std::expm1(-integrated));
        }
        J shadows=J::array();
        std::vector<double> queries={std::min(length,std::max(1e-7,focus)),length};
        for(size_t q=0;q<queries.size();++q) {
            const double d=queries[q];double nativeSum=0,conditionalSum=0,conditionalSquared=0;
            int positiveEndpoint=0,escapedNegativeEndpoint=0;
            for(int i=0;i<shadowCount;++i) {
                const unsigned sample=unsigned(count+i+q*shadowCount);
                sampler.startPath(id+200000u,sample);auto s=state(sample);
                const Ray ray(vec(origin),direction,0,float(d));
                const auto tn=native->transmittance(sampler,ray,false,false,&s);
                if(!(tn.x()==0||tn.x()==1))throw std::runtime_error("Expected native 0/1 visibility");
                nativeSum+=tn.x();
                auto *context=dynamic_cast<GPContextSparseConvNoise *>(s.gpContext.get());
                if(context) {
                    auto noise=context->noise;UniformSampler uniform;int gpId;
                    const double f=noise.evaluateValue(vec(origin+double(float(d))*eigen(direction)),float(d),direction,context->birthRayInfo,uniform,gpId);
                    positiveEndpoint+=f>=0;
                    escapedNegativeEndpoint+=tn.x()>0 && f<0;
                }
                sampler.startPath(id+300000u,sample);cg::Random rng(sampler);double hit;
                cg::ConditionalSegment segment(field,birth,origin,w,double(float(d)),conditional->trackingSettings());
                const double tc=segment.track(rng,false,hit);conditionalSum+=tc;conditionalSquared+=tc*tc;
            }
            double value=conditionalSum/shadowCount;
            shadows.push_back({{"distance",double(float(d))},{"samples",shadowCount},
                {"renewal_mean",nativeSum/shadowCount},{"conditional_mean",value},
                {"native_positive_endpoint_fraction",double(positiveEndpoint)/shadowCount},
                {"native_escaped_negative_endpoint",escapedNegativeEndpoint},
                {"conditional_standard_error",std::sqrt(std::max(0.,(conditionalSquared-shadowCount*value*value)/(shadowCount-1)/shadowCount))}});
            std::cout<<"Shadow at "<<d<<": "<<nativeSum/shadowCount<<" versus "<<value<<std::endl;
        }
        J allGradients=J::array();for(int i=0;i<count;++i)allGradients.push_back(statuses[i]=="hit"?vector(gradients[i]):J());
        J out={{"spec",spec},{"record",r},{"native_medium",nd},{"conditional_medium",cd},
            {"samples",count},{"escaped",escaped},{"failures",failures},{"times",times},{"statuses",statuses},
            {"native_gradients",allGradients},{"gradient_samples",gradientSamples},{"gradient_half_width",halfWidth},
            {"gradient_zero_distance_excluded",unsupported},{"ages",ages},{"hazard",hazard},{"conditional_cdf",cdf},
            {"flight_mean",means},{"quadrature_error_estimate",error},{"shadow_queries",shadows}};
        auto target=std::filesystem::absolute(argv[2]);std::filesystem::create_directories(target.parent_path());
        std::ofstream f(target);f<<out.dump();if(!f)throw std::runtime_error("Write failed");
        native->teardownAfterRender();conditional->teardownAfterRender();return 0;
    }catch(const std::exception &e){std::cerr<<e.what()<<std::endl;return 1;}
}
