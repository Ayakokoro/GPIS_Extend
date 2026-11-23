#include "bsdfs/Fresnel.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "integrators/TraceBase.hpp"
#include "media/ClassicGlobalFpGradientMedium.hpp"
#include "media/LatticeCubicGpisMedium.hpp"
#include "media/LatticeNonstationaryParameters.hpp"
#include "primitives/Cube.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "thread/ThreadUtils.hpp"
#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>

using namespace Tungsten;
namespace fs = std::filesystem;
namespace {
constexpr float Ell=0.25f, Sigma=0.02f;
void require(bool ok,const char *message) {
    if (!ok) throw std::runtime_error(message);
}
void close(double a,double b,double tolerance,const char *message) {
    if (!(std::abs(a-b)<=tolerance)) {
        std::cerr<<message<<" ("<<a<<" vs "<<b<<")\n";
        throw std::runtime_error(message);
    }
}
double mean(double z) { return std::min(z,4-z); }

void testProcessBoxExit() {
    const Cube box(Vec3f(0.f,1.001f,0.f),Vec3f(4.f,2.f,2.f),Mat4f(),"processBox",nullptr);
    // Dragon's failed path starts on the -Z face and exits through +Y just past nearT.
    const Vec3f origin(0.929988623f,2.0007081f,-1.f), direction(0.626499593f,0.583716333f,0.516501248f);
    const float nearT=0.000500000024f;
    Ray exiting(origin,direction,nearT);
    IntersectionTemporary data{};
    require(box.intersectMediumExit(exiting,data),"recover Dragon process box exit past nearT");
    require(exiting.farT()>nearT && data.primitive==&box && box.hitBackside(data),"recovered exit is a backside intersection");
    close(exiting.farT(),0.0005001463f,1e-9,"Dragon process box exit distance");
    IntersectionInfo info{};
    info.p=exiting.hitpoint();
    box.intersectionInfo(data,info);
    require(info.Ng==Vec3f(0.f,1.f,0.f),"recovered exit uses the top face");

    const Cube blocker;
    data.primitive=&blocker;
    const float closer=(nearT+exiting.farT())*.5f;
    Ray occluded(origin,direction,nearT,closer);
    require(!box.intersectMediumExit(occluded,data) && occluded.farT()==closer && data.primitive==&blocker,
            "preserve a closer hit between nearT and the box exit");
    Ray outside(Vec3f(3.f,1.f,0.f),Vec3f(-1.f,0.f,0.f),nearT);
    require(!box.intersectMediumExit(outside,data) && outside.farT()==Ray::infinity() && data.primitive==&blocker,
            "an incoming exterior ray must not recover the far exit");
    Ray onFace(origin,Vec3f(0.f,0.f,-1.f),nearT);
    require(box.intersectMediumExit(onFace,data) && onFace.farT()==0.f && box.hitBackside(data),
            "an outward ray on the box face exits inside nearT");
}

void writeMeanGrid(const fs::path &path,std::optional<float> constant=std::nullopt) {
    auto vdb=openvdb::FloatGrid::create(constant.value_or(0.f));
    vdb->setName("sdf");
    vdb->setTransform(openvdb::math::Transform::createLinearTransform(1./16));
    auto access=vdb->getAccessor();
    for (int z=-16; z<=80; ++z)
        for (int y=-8; y<=8; ++y)
            for (int x=-8; x<=8; ++x)
                access.setValue(openvdb::Coord(x,y,z),constant.value_or(float(mean(z/16.))));
    openvdb::io::File file(path.string());
    file.write(openvdb::GridPtrVec{vdb});
    file.close();
}
std::shared_ptr<LatticeCubicGpisMedium> load(const fs::path &path,bool prune,bool varying=false,
                                             const std::string &pruningMode="",
                                             const std::string &correlation="",
                                             std::optional<float> stepSize=std::nullopt,
                                             const std::string &material=R"({"type":"mirror","albedo":1})") {
    const std::string covariance=varying
        ? R"({"type":"proc_nonstationary","cov":{"type":"squared_exponential","sigma":1,"lengthScale":1},"var":{"type":"noise","noise":"left_right","min":0.012,"max":0.035,"start":-0.25,"end":0.23},"ls":{"type":"noise","noise":"front_back","min":0.09,"max":0.25,"start":0.018,"end":1.38}})"
        : R"({"type":"squared_exponential","sigma":0.02,"lengthScale":0.25})";
    const std::string json=R"({"type":"lattice_cubic_gpis","geometry_pruning":)"+
        std::string(prune ? "true" : "false")+
        (pruningMode.empty() ? "" : ",\"geometry_pruning_mode\":\""+pruningMode+"\"")+
        (correlation.empty() ? "" : ",\"correlation_context\":\""+correlation+"\"")+
        (stepSize ? ",\"step_size\":"+std::to_string(*stepSize) : "")+
        ",\"material\":"+material+R"(,"gaussian_process":{"type":"standard",
            "covariance":)"+covariance+R"(,"mean":{"type":"tabulated","scale":1,"grid":{"type":"vdb","file":")"+
        path.generic_string()+R"(","density_name":"sdf","emission_name":"","request_sdf":false,
            "interpolate":"linear","normalize_size":false,"integration_method":"raymarching"}}}})";
    JsonDocument document(Path("lattice-medium-test.json"),json);
    Scene scene;
    auto medium=std::dynamic_pointer_cast<LatticeCubicGpisMedium>(scene.fetchMedium(document));
    require(bool(medium),"medium factory");
    medium->loadResources();
    medium->prepareForRender();
    return medium;
}
Medium::MediumState freshState() {
    Medium::MediumState state{};
    state.reset();
    return state;
}
IndependentPathSampler sampler(unsigned sample) {
    IndependentPathSampler rng(317,false);
    rng.startPath(3,sample);
    return rng;
}

class SurfaceLightingProbe : public TraceBase {
public:
    using TraceBase::bsdfSample;
    using TraceBase::lightSample;
    explicit SurfaceLightingProbe(TraceableScene *scene) : TraceBase(scene,TraceSettings{},0) {}

    Vec3f queued(PathSampleGenerator &rng, SurfaceScatterEvent &event, const Medium &medium,
                 const Ray &incoming, const Medium::MediumState &state) {
        DirectLightingState direct;
        beginDirectLighting(direct,rng,event.info->p,state,1,true);
        MediumSample sample{};
        PreparedGpisSegment prepared;
        GpRaySegmentReply reply{};
        bool ready=false;
        require(advanceDirectLighting(direct,rng,sample,event,&medium,incoming,prepared,reply,ready),
                "CPU lattice direct-light queries should finish synchronously");
        return direct.result;
    }
};

void testOrdinaryDirectLighting(const fs::path &zeroMean) {
    const auto medium=load(zeroMean,false);
    if (!ThreadUtils::pool) ThreadUtils::startThreads(1);
    if (!EmbreeUtil::getDevice()) EmbreeUtil::initDevice();
    Scene scene;
    JsonDocument geometry(Path("surface-lighting-test.json"),R"({"primitives":[
        {"type":"quad","bsdf":{"type":"lambert","albedo":1}},
        {"type":"quad","emission":1,"bsdf":{"type":"lambert","albedo":1},
         "transform":{"position":[0,0.2,0],"rotation":[180,0,0]}}]})");
    scene.fromJson(geometry,scene);
    scene.loadResources();
    std::unique_ptr<TraceableScene> traceable(scene.makeTraceable());
    SurfaceLightingProbe probe(traceable.get());
    const auto &floor=*scene.primitives().front();
    const auto &light=*traceable->lights().front();
    Ray incoming(Vec3f(0.f,.1f,0.f),Vec3f(0.f,-1.f,0.f),0,1);
    IntersectionTemporary intersection;
    require(floor.intersect(incoming,intersection),"incoming ray must reach the ordinary floor");
    IntersectionInfo info{};
    floor.intersectionInfo(intersection,info);
    info.epsilon=1e-5f;
    int clear[3]={}, blocked[3]={}, endpointChangesVisibility=0;
    for (unsigned i=0;i<128;++i) {
        auto retained=freshState();
        auto prefixRng=sampler(i);
        MediumSample prefix;
        if (!medium->sampleDistance(prefixRng,incoming,retained,prefix) || !prefix.exited) continue;
        retained.startsAtScatteringSurface=true;
        require(bool(retained.pathContext),"a surviving incoming path has a floor observation");
        const auto endpoint=retained.pathContext;
        auto fresh=retained;
        fresh.pathContext.reset();
        // Establish that retaining this observation would visibly change the
        // actual floor-to-light transmittance for some matched seeds.
        auto controlRng=sampler(3000+i);
        LightSample lightSample;
        require(light.sampleDirect(0,info.p,controlRng,lightSample),"sample floor-to-light control");
        Ray shadow(info.p,lightSample.d,info.epsilon,lightSample.dist);
        auto conditioned=retained, independent=fresh;
        auto pairedRng=controlRng;
        if (medium->transmittance(controlRng,shadow,true,true,&conditioned)!=
            medium->transmittance(pairedRng,shadow,true,true,&independent))
            ++endpointChangesVisibility;
        for (unsigned strategy=0;strategy<3;++strategy) {
            auto evaluate=[&](Medium::MediumState &state) {
                auto rng=sampler(3000+i);
                auto event=probe.makeLocalScatterEvent(intersection,info,incoming,&rng);
                if (strategy==0) return probe.lightSample(light,event,&state,medium.get(),1,incoming,nullptr);
                if (strategy==1) return probe.bsdfSample(light,event,&state,medium.get(),1,incoming);
                return probe.queued(rng,event,*medium,incoming,state);
            };
            const auto expected=evaluate(fresh);
            require(evaluate(retained)==expected,"ordinary direct lighting must match a fresh GP shadow");
            require(retained.pathContext==endpoint,"lighting must preserve the main path endpoint");
            if (expected==0.f) ++blocked[strategy]; else ++clear[strategy];
        }
    }
    for (unsigned i=0;i<3;++i)
        require(clear[i]>0 && blocked[i]>0,"direct-light check must exercise clear and blocked paths");
    require(endpointChangesVisibility>0,"a stale floor observation must change some control shadows");
}

void testPruningModes(const fs::path &path) {
    // The flat mean exceeds 3 times the stationary sigma, but is below
    // 3 times the maximum sigma of the spatially varying model.
    constexpr float plateau=.061f;
    writeMeanGrid(path,plateau);
    const Ray ray(Vec3f(.3f,0.f,0.f),Vec3f(0.f,0.f,1.f),0,4);
    for (bool varying : {false,true}) {
        const auto conservative=load(path,true,varying);
        const auto approximate=load(path,true,varying,"three_sigma");
        const auto disabled=load(path,false,varying,"three_sigma");
        rapidjson::Document document;
        auto &allocator=document.GetAllocator();
        require(std::string(conservative->toJson(allocator)["geometry_pruning_mode"].GetString())=="conservative",
                "default pruning mode serialization");
        require(std::string(approximate->toJson(allocator)["geometry_pruning_mode"].GetString())=="three_sigma",
                "three-sigma pruning mode serialization");
        if (!varying) {
            // This realization starts in the negative tail on the >3 sigma plateau.
            constexpr unsigned tailSeed=2753;
            auto oracle=sampler(tailSeed), plain=sampler(tailSeed), pruned=sampler(tailSeed);
            const LatticeGaussianKernel kernel(Sigma,Ell);
            const LatticeGaussianRay noise(kernel,oracle.uniformGenerator().nextI());
            require(plateau+noise.segment(0)[0]<0,"entry must lie in the negative tail");
            const auto unprunedHit=disabled->firstHitDistance(plain,ray,nullptr);
            const auto prunedHit=approximate->firstHitDistance(pruned,ray,nullptr);
            require(unprunedHit && prunedHit && *unprunedHit>0,"negative tail must have a later upcrossing");
            close(*prunedHit,*unprunedHit,1e-7,"three-sigma pruning changed the upcrossing");
        }
        bool testedHit=false;
        for (unsigned i=0; i<4096 && !testedHit; ++i) {
            auto a=sampler(i), b=sampler(i), c=sampler(i);
            auto sa=freshState(), sb=freshState(), sc=freshState();
            MediumSample x,y,z;
            const bool scattered=conservative->sampleDistance(a,ray,sa,x);
            if (!scattered || x.exited) continue;
            require(approximate->sampleDistance(b,ray,sb,y),"three-sigma flight");
            require(disabled->sampleDistance(c,ray,sc,z) && !z.exited,"disabled pruning lost a tail hit");
            close(z.t,x.t,1e-6,"disabled pruning changed a tail hit");
            if (varying) {
                require(!y.exited,"three-sigma threshold must use the global physical maximum");
                close(y.t,x.t,1e-6,"three-sigma changed an unskipped root");
            } else {
                require(y.exited,"three-sigma must skip the positive plateau above 3 sigma");
            }
            testedHit=true;
        }
        require(testedHit,"expected a reference downcrossing on the plateau");
    }
}
// Independent scalar sign scan and bisection for these separated planar
// crossings. This uses neither the geometry traversal nor the cubic root solver.
double reference(const LatticeGaussianRay &noise,double z,double dz,double spacing,double limit,
                 double left=1e-4) {
    const auto value=[&](double t) {
        const int cell=int(std::floor(t/spacing));
        const double u=t/spacing-cell;
        const auto p=noise.segment(cell);
        return mean(z+dz*t)+p[0]+u*(p[1]+u*(p[2]+u*p[3]));
    };
    const bool inside=value(left)<0;
    for (double right=left+1./128; right<=limit; right+=1./128) {
        if ((value(right)<0)!=inside) {
            double hi=right;
            for (int i=0; i<32; ++i) {
                const double mid=(left+hi)/2;
                if ((value(mid)<0)==inside) left=mid; else hi=mid;
            }
            return (left+hi)/2;
        }
        left=right;
    }
    throw std::runtime_error("reference missed the opposing plate");
}

void testFreshEntry(const fs::path &path) {
    constexpr float step=.0625f;
    const auto medium=load(path,true,false,"","",step);
    rapidjson::Document document;
    close(medium->toJson(document.GetAllocator())["step_size"].GetFloat(),step,0,
          "custom entry step must survive serialization");
    const LatticeGaussianKernel kernel(Sigma,Ell);
    const Ray ray(Vec3f(0.f,0.f,.0234375f),Vec3f(0.f,0.f,-1.f),.00390625f,.25f);
    unsigned negative=0,positive=0;
    for (unsigned i=0;i<64;++i) {
        auto oracle=sampler(i), draw=sampler(i), shadow=sampler(i);
        const LatticeGaussianRay noise(kernel,oracle.uniformGenerator().nextI());
        const auto p=noise.segment(0);
        const double u=ray.nearT()/kernel.spacing;
        const double value=ray.pos().z()-ray.nearT()+p[0]+u*(p[1]+u*(p[2]+u*p[3]));
        auto state=freshState();
        MediumSample hit;
        require(medium->sampleDistance(draw,ray,state,hit),"descending plate ray");
        const auto geometric=medium->firstHitDistance(shadow,ray,nullptr);
        if (value<0) {
            ++negative;
            require(hit.exited && !geometric,"a ray remaining inside has no crossing");
        } else {
            ++positive;
            const double root=reference(noise,ray.pos().z(),-1,kernel.spacing,ray.farT(),ray.nearT());
            require(!hit.exited && geometric,"both queries must find the downcrossing");
            close(hit.t,root,1e-6,"fresh ray downcrossing");
            close(*geometric,hit.t,1e-7,"shadow downcrossing");
        }
    }
    require(negative && positive,"exercise both signs at nearT");

    // The whole ray exits the negative plate. Transparent cuts on either side
    // of nearT must preserve its original sign test, direction and distance.
    const Ray whole(Vec3f(0.f,0.f,-.25f),Vec3f(0.f,0.f,2.f),.015625f,.25f);
    for (bool varying : {false,true}) {
        const auto bounded=load(path,true,varying,"three_sigma","",step);
        for (unsigned i=0;i<8;++i) {
            auto query=sampler(i), main=sampler(i);
            const auto root=bounded->firstHitDistance(query,whole,nullptr);
            require(root && *root>.08f && *root<.18f,"geometric ray must reach the upper side of the plate");
            if (!varying) {
                auto oracle=sampler(i);
                const LatticeGaussianRay noise(kernel,oracle.uniformGenerator().nextI());
                close(*root,reference(noise,-.25,1,kernel.spacing,.5,2*whole.nearT())/2,1e-6,
                      "upcrossing on a non-unit ray");
            }
            auto state=freshState();
            Medium::DistanceDiagnostic diagnostic;
            state.distanceDiagnostic=&diagnostic;
            MediumSample hit;
            require(!bounded->sampleDistance(main,whole,state,hit) && !hit.exited &&
                    std::string(diagnostic.failure)=="outward_crossing_rejected","reject outward main path");
            close(hit.t,*root,1e-7,"main path and geometry must select the same outward root");
            for (float cut : {.0078125f,.0625f}) {
                const Ray prefix(whole.pos(),whole.dir(),whole.nearT(),cut);
                const Ray suffix(prefix.hitpoint(),whole.dir(),.0125f,whole.farT()-cut);
                auto shadow=sampler(i), split=sampler(i);
                auto shadowState=freshState(), splitState=freshState();
                require(bounded->transmittance(shadow,prefix,false,true,&shadowState)==1.f &&
                        bounded->transmittance(shadow,suffix,true,false,&shadowState)==0.f,
                        "only the segment with the outward root occludes");
                require(bounded->sampleDistance(split,prefix,splitState,hit) && hit.exited,"clear prefix");
                splitState.distanceDiagnostic=&diagnostic;
                require(!bounded->sampleDistance(split,suffix,splitState,hit) && !hit.exited,
                        "transparent cut must retain outward rejection");
                close(cut+hit.t,*root,1e-6,"transparent cut moved the upcrossing");
            }
        }
    }
}

void testBackfaceScattering(const fs::path &path) {
    constexpr float step=.0625f;
    const Ray incoming(Vec3f(0.f,0.f,.5f),Vec3f(0.f,0.f,1.f),0,.25f);
    // An inward GPIS birth still uses an occupancy probe. Its field can
    // already be increasing while the probe is on the negative side.
    auto inwardState=[&](const LatticeCubicGpisMedium &medium) {
        auto birth=std::make_shared<FpGradientPathContext>();
        birth->owner=&medium; birth->rayOrigin=incoming.pos();
        birth->origin=Eigen::Vector3d(0,0,.5); birth->direction=Eigen::Vector3d::UnitZ();
        birth->value=0; birth->derivative=-.12; birth->gradient=birth->derivative*birth->direction;
        auto state=freshState();
        state.pathContext=birth;
        return state;
    };
    unsigned diffuseSeed=0;
    const std::string mirror=R"({"type":"mirror","albedo":1})";
    const std::string copper=R"({"type":"conductor","eta":[0.200438,0.924033,1.10221],"k":[3.91295,2.45285,2.14219]})";
    for (const auto &material : {mirror,copper}) {
        const auto medium=load(path,true,material==copper,"","",step,material);
        unsigned backfaces=0;
        for (unsigned i=0;i<2048 && backfaces<8;++i) {
            auto oracle=sampler(i), query=sampler(i), draw=sampler(i), shadow=sampler(i);
            oracle.uniformGenerator().nextI();
            const double probe=step*oracle.next1D();
            auto state=inwardState(*medium), shadowState=state, queryState=state;
            MediumSample surface;
            require(medium->sampleDistance(draw,incoming,state,surface),"conditioned specular flight");
            const auto event=std::dynamic_pointer_cast<const FpGradientEventContext>(surface.eventContext);
            if (!event || event->derivative<=0) continue;
            ++backfaces;
            if (material==mirror) diffuseSeed=i;
            const auto hit=medium->firstHitDistance(query,incoming,&queryState);
            require(hit.has_value(),"negative back face must give a geometric intersection");
            close(*hit,probe,1e-7,"back-face first-hit distance");
            require(medium->transmittance(shadow,incoming,false,false,&shadowState)==0.f,
                    "back face must still block shadows");
            require(!surface.exited,"conditioned specular probe can scatter on the back face");
            close(surface.t,probe,1e-7,"scattering distance at a negative probe");
            close(surface.p.z(),incoming.pos().z()+probe,1e-7,"back-face collision position");

            PhaseSample reflected;
            require(surface.phase->sample(draw,incoming.dir(),surface,reflected),"sample back-face reflection");
            const auto birth=std::dynamic_pointer_cast<const FpGradientPathContext>(reflected.pathContext);
            require(birth && birth->value==0 && birth->owner==medium.get(),"renew at the reflected GP contact");
            close(birth->gradient.z(),event->derivative,1e-11,"the phase must retain the physical gradient");
            close(birth->derivative,-event->derivative,1e-11,"signed reflected slope at a back face");
            close(birth->gradient.dot(birth->direction),birth->derivative,1e-11,"reflected gradient projection");
            const Eigen::Vector3d normal=birth->gradient.normalized();
            const Eigen::Vector3d expected=Eigen::Vector3d::UnitZ()-2*normal.z()*normal;
            const Eigen::Vector3d outgoing(reflected.w.x(),reflected.w.y(),reflected.w.z());
            require((expected-outgoing).norm()<2e-7,"back-face mirror direction");
            require(reflected.pdf>0 && std::isfinite(reflected.pdf),"finite sampled back-face density");
            close(surface.phase->pdf(incoming.dir(),reflected.w,surface)/reflected.pdf,1,5e-4,
                  "directional PDF matches the sampled back-face density");
            const Vec3f expectedWeight=material==mirror ? Vec3f(1.f) : Fresnel::conductorReflectance(
                Vec3f(.200438f,.924033f,1.10221f),Vec3f(3.91295f,2.45285f,2.14219f),-float(normal.z()));
            for (int c=0;c<3;++c)
                close(reflected.weight[c],expectedWeight[c],1e-5,"bounded's signed micro-material weight");

            std::shared_ptr<const MediumPathContext> connection;
            require(surface.phase->evalConnection(draw,incoming.dir(),reflected.w,surface,connection)==0.f &&
                    !connection,"a light connection cannot be constructed from a back-facing gradient");

            // The signed reflection starts towards the negative region. Bounded
            // tests its first interior probe after the geometry epsilon.
            const Ray next(surface.p,reflected.w,.00390625f,.15f);
            auto resume=sampler(i+700), offsetOracle=resume, visibility=resume;
            offsetOracle.uniformGenerator().nextI();
            const double nextProbe=double(next.nearT())+step*offsetOracle.next1D()/outgoing.norm();
            state.pathContext=birth;
            shadowState=state;
            MediumSample follow;
            require(medium->sampleDistance(resume,next,state,follow),"trace conditioned interior continuation");
            require(follow.exited || follow.t+1e-7>=nextProbe,"conditioned hit precedes its probe");
            require(medium->transmittance(visibility,next,true,false,&shadowState)==(follow.exited ? 1.f : 0.f),
                    "shadow and sampled flights agree on an interior continuation");
        }
        require(backfaces==8,"exercise conditioned back-face scattering");
    }
    const auto diffuse=load(path,true,false,"","",step,R"({"type":"lambert","albedo":1})");
    auto state=inwardState(*diffuse);
    auto draw=sampler(diffuseSeed);
    MediumSample surface;
    const bool accepted=diffuse->sampleDistance(draw,incoming,state,surface);
    require(!surface.exited,"a diffuse back face still gives a geometric hit");
    if (accepted) {
        PhaseSample sample;
        require(!surface.phase->sample(draw,incoming.dir(),surface,sample),"Lambert rejects the back hemisphere");
    }
    const auto mixed=load(path,true,false,"","",step,
        R"({"type":"conductor","eta":[1,0.924033,1.10221],"k":[0,2.45285,2.14219]})");
    const auto weight=mixed->signedReflectance(-.75);
    close(weight.x(),0,0,"index-matched channel has zero reflection at signed incidence");
    close(weight.y(),Fresnel::conductorReflectance(.924033f,2.45285f,-.75f),1e-5,
          "mixed-material green channel retains signed copper Fresnel");
    close(weight.z(),Fresnel::conductorReflectance(1.10221f,2.14219f,-.75f),1e-5,
          "mixed-material blue channel retains signed copper Fresnel");
}

void testInwardZeroOffset(const fs::path &path) {
    constexpr float step=.0625f,cut=step*.5f;
    const auto medium=load(path,true,false,"","",step);
    const Ray ray(Vec3f(0.f),Vec3f(0.f,0.f,-1.f),0,2*step);
    auto contact=std::make_shared<FpGradientPathContext>();
    contact->owner=medium.get();
    contact->origin.setZero();
    contact->rayOrigin=ray.pos();
    contact->value=0;
    contact->gradient=Eigen::Vector3d::UnitZ();
    contact->direction=-Eigen::Vector3d::UnitZ();
    contact->derivative=contact->gradient.dot(contact->direction);
    auto fullState=freshState();
    fullState.pathContext=contact;
    auto splitState=fullState;
    auto full=sampler(11043206), split=full, oracle=full;
    oracle.uniformGenerator().nextI();
    require(oracle.next1D()==0.f,"inward boundary case must draw exactly zero jitter");

    const auto wholeHit=medium->firstHitDistance(full,ray,&fullState);
    require(wholeHit.has_value(),"inward birth must reach the negative field at the first distinct probe");
    close(*wholeHit,step,1e-7,"zero jitter must skip the observed zero and advance one step");

    const Ray prefix(ray.pos(),ray.dir(),0,cut), suffix(prefix.hitpoint(),ray.dir(),.001f,2*step-cut);
    require(!medium->firstHitDistance(split,prefix,&splitState),"transparent prefix ends before the distinct probe");
    const auto suffixHit=medium->firstHitDistance(split,suffix,&splitState);
    require(suffixHit.has_value(),"transparent continuation must reach the pending interior probe");
    close(*suffixHit+cut,*wholeHit,1e-7,"transparent split must preserve the distinct probe position");
}

void testOrdinarySurfaceRenewal(const LatticeCubicGpisMedium &medium) {
    const Vec3f start(0.f,0.f,2.f), direction(0.f,0.f,-1.f);
    constexpr float cut=1.875f;
    const Ray prefix(start,direction,0,cut);
    const Ray suffix(prefix.hitpoint(),direction,0,1);
    const LatticeGaussianKernel kernel(Sigma,Ell);
    bool distinct=false;
    for (unsigned i=0;i<16;++i) {
        auto draw=sampler(i), initialSeed=sampler(i);
        const LatticeGaussianRay initial(kernel,initialSeed.uniformGenerator().nextI());
        auto state=freshState();
        MediumSample sample;
        require(medium.sampleDistance(draw,prefix,state,sample) && sample.exited,"ordinary surface reached");
        const auto endpoint=state;
        auto forward=sampler(i+500);
        require(medium.sampleDistance(forward,suffix,state,sample) && !sample.exited,"transparent continuation");
        const double original=reference(initial,2,-1,kernel.spacing,3)-cut;
        close(sample.t,original,1e-6,"transparent continuation keeps the same flight");

        // The cut is exactly a noise knot. An ordinary scatter along the same
        // direction observes the old nonzero field and its axial derivative.
        const auto old=initial.segment(int(cut/kernel.spacing));
        const double oldValue=mean(2-cut)+old[0];
        require(oldValue>0.04,"ordinary surface must be away from the zero level");
        auto oracle=sampler(i+500), transport=sampler(i+500);
        LatticeGaussianRay renewed(kernel,oracle.uniformGenerator().nextI());
        renewed.condition(float(old[0]),float(old[1]/kernel.spacing));
        const double expected=reference(renewed,2-cut,-1,kernel.spacing,1);
        state=endpoint;
        state.startsAtScatteringSurface=true;
        require(medium.sampleDistance(transport,suffix,state,sample) && !sample.exited,"ordinary surface continuation");
        close(sample.t,expected,1e-6,"ordinary surface renews at its actual value and projected derivative");
        distinct|=std::abs(original-expected)>1e-4;
    }
    require(distinct,"ordinary and transparent continuations must exercise distinct realizations");
}

void testTransverseMemory(const fs::path &path) {
    writeMeanGrid(path,0.f);
    for (bool varying : {false,true}) {
        const auto medium=load(path,true,varying,"","renewal+");
        const Vec3f dir(0.f,0.f,1.f);
        const Ray ray(Vec3f(0.f),dir,0,.12f);
        auto birth=[&](double tangent) {
            auto contact=std::make_shared<FpGradientPathContext>();
            contact->owner=medium.get();
            contact->rayOrigin=ray.pos();
            contact->origin.setZero();
            contact->value=0;
            contact->direction=Eigen::Vector3d::UnitZ();
            if (tangent<0) contact->direction=-Eigen::Vector3d::UnitX();
            contact->gradient=Eigen::Vector3d(tangent,0,.02);
            contact->derivative=contact->gradient.dot(contact->direction);
            auto state=freshState();
            state.advance();
            state.pathContext=contact;
            return state;
        };
        unsigned collisions=0, visible=0;
        for (unsigned i=0;i<256 && (collisions<4 || !visible);++i) {
            auto a=birth(1), b=birth(-1), shadow=b;
            auto da=sampler(i), db=sampler(i), ds=sampler(i);
            MediumSample x,y;
            require(medium->sampleDistance(da,ray,a,x) && medium->sampleDistance(db,ray,b,y),
                    "sample both surface-conditioned flights");
            require(x.exited==y.exited && x.t==y.t,"equal projected observations must give the same free flight");
            require(medium->transmittance(ds,ray,true,true,&shadow)==(x.exited ? 1.f : 0.f),
                    "GPIS shadow samples the surface-conditioned flight");
            if (x.exited) { ++visible; continue; }
            require(x.t>0,"the first root after the observed GPIS surface must be distinct");
            PhaseSample px,py;
            require(x.phase->sample(da,dir,x,px) && y.phase->sample(db,dir,y,py),"sample renewed mirror gradients");
            const auto gx=std::dynamic_pointer_cast<const FpGradientPathContext>(px.pathContext);
            const auto gy=std::dynamic_pointer_cast<const FpGradientPathContext>(py.pathContext);
            require(gx && gy && px.w==py.w && (gx->gradient-gy->gradient).squaredNorm()<1e-22,
                    "short-flight collision gradient must forget the transverse birth gradient");
            ++collisions;
        }
        require(collisions>=4 && visible,"exercise both visibility outcomes next to the GPIS birth");
    }
}

void test(const LatticeCubicGpisMedium &plain,const LatticeCubicGpisMedium &pruned) {
    const Vec3f start(0.f,0.f,2.f),down(0.f,0.f,-1.f);
    const Ray whole(start,down,0,3);
    const LatticeGaussianKernel kernel(Sigma,Ell);
    for (unsigned i=0; i<32; ++i) {
        auto a=sampler(i), b=sampler(i), c=sampler(i), oracle=sampler(i);
        auto sa=freshState(), sb=freshState(), sc=freshState();
        MediumSample x,y,prefix,suffix;
        require(plain.sampleDistance(a,whole,sa,x) && !x.exited,"first plate collision");
        require(pruned.sampleDistance(b,whole,sb,y) && !y.exited,"pruned collision");
        const LatticeGaussianRay initial(kernel,oracle.uniformGenerator().nextI());
        close(x.t,reference(initial,2,-1,kernel.spacing,3),1e-6,"first downcrossing");
        close(y.t,x.t,1e-6,"pruning changed first downcrossing");

        const float cut=x.t-0.002f;
        const Ray first(start,down,0,cut), second(first.hitpoint(),down,0.01f,3-cut);
        require(pruned.sampleDistance(c,first,sc,prefix) && prefix.exited && sc.pathContext,
                "same-medium boundary should carry the positive flight");
        require(pruned.sampleDistance(c,second,sc,suffix) && !suffix.exited,
                "root inside geometry epsilon after a transparent cut");
        close(cut+suffix.t,x.t,1e-6,"split distance changed the realization");
        for (const auto *medium:{&plain,&pruned}) {
            auto r=sampler(i);
            auto state=freshState();
            require(medium->transmittance(r,first,false,true,&state)==1.f,"visible prefix");
            require(medium->transmittance(r,second,true,false,&state)==0.f,"occluded suffix");
            auto full=sampler(i);
            state=freshState();
            require(medium->transmittance(full,whole,false,false,&state)==0.f,"occluded whole ray");
        }

        PhaseSample phase;
        require(y.phase->sample(b,down,y,phase),"sampled reflection");
        const auto birth=std::dynamic_pointer_cast<const FpGradientPathContext>(phase.pathContext);
        require(birth && birth->owner==&pruned && birth->value==0 && birth->derivative>0,
                "phase must retain the zero value and outgoing surface gradient");
        std::shared_ptr<const MediumPathContext> connection;
        require(y.phase->evalConnection(b,down,phase.w,y,connection).max()>0 && connection,
                "a front-facing reflection supports the sampled light direction");
        const Ray outgoing(y.p,phase.w,0,8);
        const double speed=Eigen::Vector3d(phase.w.x(),phase.w.y(),phase.w.z()).norm();
        // Trace freshly seeded continuations through exactly the phase context
        // installed by the integrator. The opposite plate gives a second hit.
        auto transport=sampler(i+300), shadow=sampler(i+300), ref=sampler(i+300);
        sb.pathContext=phase.pathContext;
        auto branch=sb;
        MediumSample continuation;
        require(pruned.sampleDistance(transport,outgoing,sb,continuation) && !continuation.exited,
                "reflected ray reaches the other plate");
        require(pruned.transmittance(shadow,outgoing,false,false,&branch)==0.f,"continuation visibility");
        LatticeGaussianRay next(kernel,ref.uniformGenerator().nextI());
        const double z=birth->origin.z(), dz=birth->direction.z();
        next.condition(float(-mean(z)),float(birth->derivative-dz));
        close(continuation.t,reference(next,z,dz,kernel.spacing,8)/speed,2e-6,
              "conditioned second downcrossing");
    }
}

void testParameterFields() {
    using V=Eigen::Vector3d;
    auto loadCovariance=[](const std::string &json) {
        Scene scene;
        JsonDocument doc(Path("lattice-covariance.json"),json);
        return std::dynamic_pointer_cast<ProceduralNonstationaryCovariance>(scene.fetchCovarianceFunction(doc));
    };
    const auto k=loadCovariance(R"({"type":"proc_nonstationary","cov":{"type":"squared_exponential","sigma":2,"lengthScale":0.5},"var":{"type":"noise","noise":"left_right","min":0.4,"max":0.7,"start":-1,"end":1},"ls":{"type":"noise","noise":"front_back","min":0.2,"max":0.5,"start":0.1234567890123,"end":0.75}})");
    const LatticeNonstationaryParameters params(*k);
    const V origin=V::Zero(), forward=V::UnitZ();
    const auto split=params.cuts(origin,forward);
    close(split[0],0.1234567890123,1e-15,"precise right-hand width ramp plane");
    close(split[1],0.75,1e-15,"width ramp exit plane");
    require(!std::isfinite(split[2]),"transverse amplitude ramp adds no ray cut");
    const auto left=params.at(origin,forward,split[0],-1);
    const auto right=params.at(origin,forward,split[0],1);
    close(left.ell,right.ell,1e-15,"value continuity at a ramp split");
    close(right.ell,0.1,1e-12,"local physical length includes stationary base");
    close(right.sigma,2*(std::sqrt(1.4*1.7)-1),1e-12,"local physical amplitude includes stationary base");
    const double slope=.5*1.2*std::log(1.5/1.2)/(.75-split[0]);
    close(left.ellSlope,0,0,"outside width derivative");
    close(right.ellSlope,slope,1e-12,"inside width derivative");
    const V reverseOrigin(0,0,1);
    const auto back=params.cuts(reverseOrigin,-forward);
    close(params.at(reverseOrigin,-forward,back[1],-1).ellSlope,-slope,1e-12,"reverse width derivative");
    close(params.at(reverseOrigin,-forward,back[1],1).ellSlope,0,0,"reverse exterior derivative");

    const NonstationaryLatticeGaussianKernel kernel(params.minEll,params.maxEll,params.maxSigma,
        params.maxSigmaGradient,params.maxEllGradient);
    const NonstationaryLatticeGaussianRay noise(kernel,175);
    V mean=V::Zero();
    double sd;
    params.transverse(kernel,noise,origin,forward,0.,mean,sd);
    const auto atBirth=params.at(origin,forward,0.,1);
    close(sd,noise.originStddev(atBirth)/atBirth.ell,1e-12,"finite transverse birth variance");
    mean.setZero();
    params.transverse(kernel,noise,origin,forward,.3,mean,sd);
    auto value=[&](double x) {
        const auto local=params.at(V(x,0,0),forward,.3,1);
        double sum=0;
        kernel.forEachKnot(.3,local,[&](std::int64_t i,double b,double,double,double) {
            sum+=noise.coefficient(i)*b;
        });
        return sum;
    };
    close(mean.x(),(value(1e-5)-value(-1e-5))/(2e-5),1e-8,"transverse derivative includes amplitude variation");
    require(sd>0 && std::isfinite(sd),"finite transverse residual variance");
    const auto anisotropic=loadCovariance(R"({"type":"proc_nonstationary","cov":{"type":"squared_exponential"},"ls":{"type":"constant","val":[1,1,1.1]}})");
    bool rejected=false;
    try { LatticeNonstationaryParameters unsupported(*anisotropic); }
    catch (const std::invalid_argument &) { rejected=true; }
    require(rejected,"unequal local widths must be rejected");
}

void testVarying(const LatticeCubicGpisMedium &plain,const LatticeCubicGpisMedium &pruned) {
    const Ray whole(Vec3f(0.f,0.f,2.f),Vec3f(0.f,0.f,-1.f),0,3);
    for (unsigned i=0;i<64;++i) {
        auto a=sampler(i), b=sampler(i), split=sampler(i), vis=sampler(i);
        auto stateA=freshState(), stateB=freshState(), stateSplit=freshState(), stateVis=freshState();
        MediumSample h0,h1,prefix,suffix;
        require(plain.sampleDistance(a,whole,stateA,h0) && !h0.exited,"nonstationary first collision");
        require(pruned.sampleDistance(b,whole,stateB,h1) && !h1.exited,"nonstationary pruned collision");
        close(h0.t,h1.t,1e-6,"nonstationary pruning changed collision");
        const float cut=0.7f;
        const Ray first(whole.pos(),whole.dir(),0,cut), second(first.hitpoint(),whole.dir(),0,3-cut);
        require(pruned.sampleDistance(split,first,stateSplit,prefix) && prefix.exited,"nonstationary exterior prefix");
        require(pruned.sampleDistance(split,second,stateSplit,suffix) && !suffix.exited,"nonstationary suffix");
        close(h0.t,cut+suffix.t,1e-6,"nonstationary cut changed the original flight");
        require(pruned.transmittance(vis,whole,false,true,&stateVis)==0.f,"nonstationary visibility");

        PhaseSample phase;
        require(h1.phase->sample(b,whole.dir(),h1,phase),"nonstationary mirror phase");
        const auto birth=std::dynamic_pointer_cast<const FpGradientPathContext>(phase.pathContext);
        require(birth && birth->derivative>0 && birth->gradient.allFinite(),"nonstationary outgoing gradient");
        const Ray next(h1.p,phase.w,0,10);
        stateA.pathContext=phase.pathContext;
        auto stateShadow=stateA;
        auto r=sampler(700+i),shadow=sampler(700+i);
        MediumSample outgoing;
        require(pruned.sampleDistance(r,next,stateA,outgoing),"nonstationary conditioned free flight");
        require(outgoing.exited || outgoing.t>0,"nonstationary surface birth self-hit");
        require(pruned.transmittance(shadow,next,true,true,&stateShadow)==(outgoing.exited ? 1.f : 0.f),
                "nonstationary conditioned visibility");
    }
}
}

int main(int argc,char **argv) try {
    const auto dir=argc>1 ? fs::absolute(argv[1]) : fs::current_path();
    fs::create_directories(dir);
    const auto path=dir/"lattice-test-plates.vdb";
    openvdb::initialize();
    testProcessBoxExit();
    testParameterFields();
    writeMeanGrid(path);
    testFreshEntry(path);
    testBackfaceScattering(path);
    testInwardZeroOffset(path);
    const auto plain=load(path,false), pruned=load(path,true);
    rapidjson::Document doc;
    require(std::string(plain->toJson(doc.GetAllocator())["correlation_context"].GetString())=="renewal+",
            "default correlation context is renewal+");
    require(plain->toJson(doc.GetAllocator())["step_size"].GetFloat()==0.01f,
            "default entry probe step matches the Dragon scene");
    bool rejected=false;
    try { load(path,false,false,"","global"); }
    catch (const std::exception &) { rejected=true; }
    require(rejected,"unsupported correlation context must be rejected");
    test(*plain,*pruned);
    testOrdinarySurfaceRenewal(*pruned);
    const auto varyingPlain=load(path,false,true),varyingPruned=load(path,true,true);
    testVarying(*varyingPlain,*varyingPruned);
    testTransverseMemory(dir/"lattice-test-zero.vdb");
    testOrdinaryDirectLighting(dir/"lattice-test-zero.vdb");
    testPruningModes(dir/"lattice-test-plateau.vdb");
    std::cout<<"stationary and nonstationary lattice distance, visibility, split rays and reflection passed\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr<<"lattice medium test failed: "<<e.what()<<'\n';
    return 1;
}
