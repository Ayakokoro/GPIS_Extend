// Fixed-event gradient diagnostics using the scene's actual SDF and production
// camera / medium / phase samplers. No production transport code is modified.
#include "media/GlobalConditionalMedium.hpp"
#include "media/SparseConvolutionNoiseMedium.hpp"
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/mathutility/Quadrature.h"
#include "math/FirstPassageGradient.hpp"
#include "cameras/PinholeCamera.hpp"
#include "samplerecords/PositionSample.hpp"
#include "samplerecords/DirectionSample.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <nlohmann/json.hpp>
#include <openvdb/openvdb.h>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace Tungsten;
namespace fp = Tungsten::FirstPassageGradient;
using J = nlohmann::json;
using V = Eigen::Vector3d;

J read(const char *path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Cannot open input");
    return J::parse(f);
}
void save(const char *path, const J &j) {
    auto p = std::filesystem::absolute(path);
    std::filesystem::create_directories(p.parent_path());
    std::ofstream f(p); f << j.dump();
    if (!f) throw std::runtime_error("Cannot write output");
}
V eigen(const Vec3f &p) { return {p.x(),p.y(),p.z()}; }
Vec3f vec(const V &p) { return {float(p.x()),float(p.y()),float(p.z())}; }
J vector(const V &p) { return J::array({p.x(),p.y(),p.z()}); }
V vector(const J &j) { return {j[0].get<double>(),j[1].get<double>(),j[2].get<double>()}; }
bool interval(const V &o, const V &w, const V &lo, const V &hi, double &a, double &b) {
    a=0; b=1e30;
    for (int i=0;i<3;++i) {
        if (w[i]==0) { if (o[i]<lo[i] || o[i]>hi[i]) return false; continue; }
        double x=(lo[i]-o[i])/w[i],y=(hi[i]-o[i])/w[i];
        if (x>y) std::swap(x,y);
        a=std::max(a,x); b=std::min(b,y);
    }
    return b>a;
}
std::shared_ptr<GlobalConditionalMedium> medium(const J &definition, Scene &scene) {
    JsonDocument doc(Path("gradient-ray-medium.json"),definition.dump());
    auto m=std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(doc));
    if (!m) throw std::runtime_error("Expected global_conditional");
    m->loadResources(); m->prepareForRender(); return m;
}
fp::Input input(const J &r, const ConditionalGpis::GPSSField &field) {
    const V origin=vector(r.at("birth_position")),w=vector(r.at("wi"));
    const double t=r.at("t");
    const auto a=field.mean->evaluate(origin),b=field.mean->evaluate(origin+t*w);
    return {field.kernel.sigma(),std::sqrt(w.dot(field.kernel.metric()*w)),t,
        r.at("f0"),r.at("g0"),a.value,w.dot(a.gradient),w.dot(b.gradient),
        [&field,origin,w](double s) {return field.mean->evaluate(origin+s*w).value;}};
}
J parameters(const fp::Law &law) {
    J candidates=J::array();
    for(int i=0;i<law.candidateCount();++i) {
        const auto &a=law.anchorCandidates()[i];
        candidates.push_back({{"distance",a.distance},{"mean",a.mean},{"p",a.p},{"q",a.q},{"score",a.score}});
    }
    return {{"mu",law.gaussianMean()},{"sd",law.gaussianStddev()},
        {"p",law.probitIntercept()},{"q",law.probitSlope()},
        {"approximation",law.anchorSelection()==fp::AnchorSelection::Hermite ? "single_anchor_v2" : "single_anchor_fixed5_snr"},
        {"anchor_selection",fp::anchorSelectionName(law.anchorSelection())},
        {"anchor_candidates",candidates},{"selected_anchor_index",law.selectedAnchorIndex()},
        {"anchor_distance",law.anchorDistance()},
        {"anchor_p",J::array({law.probitIntercept()})},
        {"anchor_q",J::array({law.probitSlope()})}};
}
void scan(const J &spec, const char *output) {
    Scene scene;
    const J original=read(spec.at("scene").get<std::string>().c_str());
    J definition=original.at("media").at(0);
    if (definition.at("material").at("type")!="mirror")
        throw std::runtime_error("Diagnostic continuation currently expects a mirror material");
    definition["type"]="global_conditional";
    JsonDocument cd(Path("gradient-ray-camera.json"),original.at("camera").dump());
    PinholeCamera camera; camera.fromJson(cd,scene); camera.prepareForRender();
    const auto resolution=camera.resolution();
    const V lo=vector(spec.at("box_min")),hi=vector(spec.at("box_max"));
    const int stride=spec.value("stride",8),samples=spec.value("samples",2);
    const int maxBounces=spec.value("max_bounces",6),denominator=spec.value("profile_denominator",416);
    const unsigned seed=spec.value("seed",20261108u);
    J records=J::array(),counts=J::array();
    for (bool corrected:{false,true}) {
        if(spec.contains("anchor_selection")) definition["fp_gradient"]["anchor_selection"]=spec["anchor_selection"];
        definition["fp_gradient"]["enabled"]=corrected;
        definition["fp_gradient"]["bridge_correction"]=true;
        auto m=medium(definition,scene);
        const auto &field=m->field();
        IndependentPathSampler rng(seed,false);
        int paths=0,hits=0,excluded=0,failed=0,phaseFailed=0;
        for (unsigned y=stride/2;y<resolution.y();y+=stride)
        for (unsigned x=stride/2;x<resolution.x();x+=stride)
        for (int sample=0;sample<samples;++sample) {
            ++paths; rng.startPath(x+y*resolution.x(),sample);
            PositionSample p; DirectionSample d;
            camera.samplePosition(rng,p); camera.sampleDirection(rng,p,Vec2u(x,y),d);
            const V cameraOrigin=eigen(p.p),direction=eigen(d.d).normalized();
            double a,b;
            if (!interval(cameraOrigin,direction,lo,hi,a,b)) continue;
            Vec3f origin=vec(cameraOrigin+a*direction),travel=vec(direction).normalized();
            Medium::MediumState state{};
            state.info.pixelSampleSegment=Vec4u(x,y,unsigned(sample),0u);
            state.info.sceneSeed=seed; state.info.t=0; state.reset();
            J vertices=J::array({vector(cameraOrigin),vector(eigen(origin))});
            for (int bounce=0;bounce<maxBounces;++bounce) {
                if (!interval(eigen(origin),eigen(travel),lo,hi,a,b)) break;
                // Stop at the real scene floor. We diagnose GP-to-GP flights;
                // ordinary surface reflection and NEE are not part of this probe.
                if (travel.y()<0 && origin.y()>0) b=std::min(b,-double(origin.y())/travel.y());
                state.info.pixelSampleSegment.w()=unsigned(bounce+1);
                Ray ray(origin,travel,0,float(b)); MediumSample hit{};
                if (!m->sampleDistance(rng,ray,state,hit)) {++failed;break;}
                if (hit.exited) break;
                ++hits;
                const auto e=std::dynamic_pointer_cast<const ConditionalEventContext>(hit.eventContext);
                const auto c=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
                if (!e || !c) throw std::runtime_error("Missing production collision state");
                const V w=eigen(travel).normalized();
                const double t=(e->position-c->birth.birthPosition).dot(w);
                const double gamma=std::sqrt(w.dot(field.kernel.metric()*w));
                vertices.push_back(vector(e->position));
                // QR reference conditioning is intentionally not used at
                // nearly coincident jets. Count these exclusions explicitly.
                if (t*gamma>=spec.value("minimum_normalized_age",.05)) {
                    const auto start=field.mean->evaluate(c->birth.birthPosition);
                    const auto end=field.mean->evaluate(e->position);
                    J r={{"id",int(records.size())},{"source",corrected?"bridge":"conditional"},
                        {"pixel",{x,y}},{"sample",sample},{"bounce",bounce},{"vertices",vertices},
                        {"birth_position",vector(c->birth.birthPosition)},{"position",vector(e->position)},
                        {"wi",vector(w)},{"birth_gradient",vector(c->birth.birthGradient)},
                        {"f0",c->birth.birthValue},{"g0",w.dot(c->birth.birthGradient)},
                        {"t",t},{"sigma",field.kernel.sigma()},{"gamma",gamma},
                        {"birth_mean",start.value},{"birth_mean_gradient",vector(start.gradient)},
                        {"birth_mean_derivative",w.dot(start.gradient)},
                        {"end_mean_gradient",vector(end.gradient)},{"end_mean_derivative",w.dot(end.gradient)}};
                    const fp::Law law(input(r,field),true,m->anchorSelection()); r["production"]=parameters(law);
                    if (!corrected) {
                        r["conditional_projection"]={{"mu",w.dot(e->gradient.mean)},
                            {"sd",std::sqrt(std::max(0.,w.dot(e->gradient.covariance*w)))}};
                    } else r["sampled_derivative"]=e->derivative;
                    J means=J::array();
                    for (int i=0;i<=denominator;++i)
                        means.push_back(field.mean->evaluate(c->birth.birthPosition+(t*i/denominator)*w).value);
                    r["profile_denominator"]=denominator; r["mean_profile"]=std::move(means);
                    records.push_back(std::move(r));
                } else ++excluded;
                PhaseSample phase{};
                if (!hit.phase->sample(rng,travel,hit,phase)) {++phaseFailed;break;}
                state.pathContext=phase.pathContext; origin=hit.p; travel=phase.w;
            }
            if (paths%256==0) std::cout<<(corrected?"bridge":"conditional")<<": "<<paths
                <<" paths, "<<hits<<" collisions, "<<records.size()<<" retained total\n"<<std::flush;
        }
        counts.push_back({{"source",corrected?"bridge":"conditional"},{"paths",paths},{"hits",hits},
            {"small_age_excluded",excluded},{"distance_failed",failed},{"phase_failed",phaseFailed},
            {"discarded_surface_paths",m->discardedSurfacePaths()}});
        m->teardownAfterRender();
    }
    save(output,{{"schema",1},{"spec",spec},{"counts",counts},{"records",records},
        {"sampling","Production camera, global_conditional distance and mirror phase. Sources with/without bridge. "
        "Diagnostic continuations omit NEE, roulette and ordinary mesh reflection; stop at floor or process box. "
        "Each comparison holds the recorded birth jet, direction and collision distance fixed."}});
}
void detail(const J &spec,const char *output) {
    Scene scene;
    const J original=read(spec.at("scene").get<std::string>().c_str());
    auto definition=original.at("media").at(0); definition["type"]="global_conditional";
    if(spec.contains("anchor_selection")) definition["fp_gradient"]["anchor_selection"]=spec["anchor_selection"];
    auto m=medium(definition,scene); J result=spec;
    for (auto &r:result["records"]) {
        const auto in=input(r,m->field());
        const fp::Law law(in,true,m->anchorSelection());
        r["production"]=parameters(law);
        const double anchor=law.anchorDistance();
        r["anchor_positions"]={anchor}; r["anchor_means"]={in.mean(anchor)};
        if (spec.contains("profile_denominator")) {
            const int denominator=spec.at("profile_denominator");
            if (denominator<16 || denominator>16384) throw std::runtime_error("Invalid profile resolution");
            J means=J::array();
            for (int i=0;i<=denominator;++i) means.push_back(in.mean(in.distance*i/denominator));
            r["profile_denominator"]=denominator; r["mean_profile"]=std::move(means);
        }
        J pdf=J::array(),cdf=J::array();
        for (const auto &v:r.at("query_derivatives")) {
            pdf.push_back(law.pdf(v.get<double>())); cdf.push_back(law.cdf(v.get<double>()));
        }
        r["production_pdf"]=pdf; r["production_cdf"]=cdf;
    }
    save(output,result);
}
// Compare flights with the same birth observation. This is a diagnostic, not a
// new rendering mode. Sparse samples use the actual production distance code.
void flights(const J &spec,const char *output) {
    namespace cg=ConditionalGpis;
    Scene scene;
    const J original=read(spec.at("scene").get<std::string>().c_str());
    J definition=original.at("media").at(0);
    definition["type"]="global_conditional";
    definition["fp_gradient"]={{"enabled",false}};
    auto conditional=medium(definition,scene);
    std::shared_ptr<Medium> sparse;
    const int samples=spec.value("samples",0);
    const unsigned seed=spec.value("seed",20261110u);
    const double step=spec.value("step",.01),density=spec.value("impulse_density",10.);
    J sparseDefinition={{"name","groove_renewal_plus"},{"type","sparse_conv_noise"},
        {"max_bounces",1024},{"gaussian_process",definition.at("gaussian_process")},
        {"phase_function",{{"type","brdf"},{"bsdf",definition.at("material")}}},
        {"sigma_a",0},{"sigma_s",1},{"density",1},{"step_size",step},{"min_step",8},
        {"seed",17429},{"impulse_density",density},{"single_realization",false},
        {"correlation_context","renewal+"},{"1D_sampling",false},
        {"isotropic_3D_sampling",false},{"1D_gradient_correlationXY",true},
        {"intersect_method","gp_discrete"},{"normal_method","conditioned_gaussian"}};
    if(samples) {
        JsonDocument doc(Path("groove-renewal-medium.json"),sparseDefinition.dump());
        sparse=scene.fetchMedium(doc); sparse->loadResources(); sparse->prepareForRender();
    }
    J records=J::array();
    for(const auto &source:spec.at("records")) {
        J r=source;
        const V origin=vector(r.at("birth_position")),w=vector(r.at("wi")).normalized();
        cg::FlightState birth;
        birth.birthPosition=origin; birth.direction=w;
        birth.birthValue=r.at("f0"); birth.birthGradient=vector(r.at("birth_gradient"));
        cg::ConditionalFlightKernel kernel(conditional->field(),birth);
        const double length=r.at("length");
        if(length>kernel.maximumAgeInDomain()) throw std::runtime_error("Flight exceeds field domain");
        if(spec.value("export_profile",true)) {
            const int n=spec.value("denominator",1024);
            J ages=J::array(),mean=J::array(),derivative=J::array(),hazard=J::array(),stats=J::array(),cdf=J::array();
            double integrated=0,integrationError=0;
            const bool integrate=spec.value("integrate_hazard",false);
            cg::NumericPolicy policy; policy.absoluteTolerance=1e-10; policy.relativeTolerance=1e-8;
            for(int i=0;i<=n;++i) {
                const double t=length*i/n;
                const auto jet=conditional->field().mean->evaluate(origin+t*w);
                ages.push_back(t); mean.push_back(jet.value); derivative.push_back(w.dot(jet.gradient));
                if(i) {
                    const auto s=kernel.scalarStatistics(t);
                    hazard.push_back(kernel.evaluate(t).hazard.value);
                    stats.push_back({s.meanF,s.varianceF,s.meanAtZero,s.varianceAtZero});
                } else { hazard.push_back(0); stats.push_back({birth.birthValue,0,0,0}); }
                if(i && integrate && integrated<40) {
                    const auto integral=cg::integrateFinite([&](double age) {return kernel.evaluate(age).hazard.value;},
                        length*(i-1)/n,t,policy);
                    if(integral.status!=cg::NumericStatus::Ok) throw std::runtime_error("Hazard quadrature did not converge");
                    integrated+=integral.value; integrationError+=integral.absError;
                }
                cdf.push_back(-std::expm1(-integrated));
            }
            r["ages"]=ages; r["flight_mean"]=mean; r["flight_mean_derivative"]=derivative;
            r["hazard"]=hazard; r["scalar_statistics"]=stats;
            if(integrate) {r["conditional_cdf"]=cdf;r["quadrature_error_estimate"]=integrationError;}
        }
        if(samples) {
            IndependentPathSampler rng(seed,false);
            J times=J::array(),gradients=J::array(),failures=J::object();
            int escaped=0,failed=0;
            for(int i=0;i<samples;++i) {
                const unsigned id=r.at("id").get<unsigned>(); rng.startPath(id,i);
                Medium::MediumState state{}; state.reset(); state.firstScatter=false;
                state.lastVal=float(birth.birthValue);
                state.lastAniso=Vec3d(birth.birthGradient.x(),birth.birthGradient.y(),birth.birthGradient.z());
                state.info.pixelSampleSegment=Vec4u(id,0u,unsigned(i),1u); state.info.sceneSeed=seed; state.info.t=0;
                Medium::DistanceDiagnostic diagnostic; state.distanceDiagnostic=&diagnostic;
                const Ray ray(vec(origin),vec(w).normalized(),0,float(length));
                MediumSample hit{};
                if(!sparse->sampleDistance(rng,ray,state,hit)) {
                    ++failed; const std::string why=diagnostic.failure;
                    failures[why]=failures.value(why,0)+1; times.push_back(-1); gradients.push_back(nullptr);
                } else if(hit.exited) { ++escaped; times.push_back(length); gradients.push_back(nullptr); }
                else {
                    times.push_back(hit.t);
                    gradients.push_back({hit.aniso.x(),hit.aniso.y(),hit.aniso.z()});
                }
            }
            r["sparse"]={{"samples",samples},{"escaped",escaped},{"failed",failed},
                {"failures",failures},{"times",times},{"gradients",gradients}};
        }
        records.push_back(std::move(r));
        std::cout<<"Flight "<<source.at("id")<<" complete ("<<samples<<" sparse samples)\n"<<std::flush;
    }
    save(output,{{"spec",spec},{"renewal_medium",sparseDefinition},{"records",records}});
}
int main(int argc,char **argv) {
    try {
        if (argc!=4) throw std::runtime_error("Usage: fp_gradient_rays scan|detail|flights input.json output.json");
        openvdb::initialize(); const auto spec=read(argv[2]);
        const std::string command=argv[1];
        if (command=="scan") scan(spec,argv[3]);
        else if (command=="detail") detail(spec,argv[3]);
        else if (command=="flights") flights(spec,argv[3]);
        else throw std::runtime_error("Unknown command");
        return 0;
    } catch (const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
