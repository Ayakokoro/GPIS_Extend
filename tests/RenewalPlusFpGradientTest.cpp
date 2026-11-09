#include "media/RenewalPlusFpGradientMedium.hpp"
#include "math/conditional/transport/FixedSpeedGradientDistribution.hpp"
#include "phasefunctions/PhaseFunction.hpp"
#include "samplerecords/PhaseSample.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <iostream>
#include <stdexcept>

using namespace Tungsten;
namespace cg = ConditionalGpis;
namespace fp = FirstPassageGradient;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void close(double a, double b, double tolerance, const char *message) {
    if (!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b) > tolerance) {
        std::cerr << message << ": " << a << " vs " << b << '\n';
        throw std::runtime_error(message);
    }
}
Eigen::Vector3d eigen(const Vec3d &v) { return {v.x(),v.y(),v.z()}; }
std::shared_ptr<SparseConvolutionNoiseMedium> medium(const std::string &type,
        const std::string &extra = "", const std::string &context = "renewal+") {
    const std::string json = R"({"type":")" + type + R"(","correlation_context":")" + context +
        R"(","gaussian_process":{"type":"standard","mean":{"type":"homogeneous","offset":0.1},
        "covariance":{"type":"squared_exponential","sigma":1,"lengthScale":1}},
        "phase_function":{"type":"brdf","bsdf":{"type":"mirror","albedo":0.8}},
        "sigma_a":0,"sigma_s":1,"step_size":0.05,"min_step":8,"seed":17429,
        "impulse_density":6)" + extra + "}";
    JsonDocument document(Path("renewal-plus-fp-gradient-test.json"), json);
    Scene scene;
    auto m = std::dynamic_pointer_cast<SparseConvolutionNoiseMedium>(scene.fetchMedium(document));
    require(bool(m), "native/hybrid medium factory");
    m->loadResources(); m->prepareForRender(); return m;
}
Medium::MediumState state(int i, int birth) {
    Medium::MediumState s{}; s.reset();
    s.info.pixelSampleSegment = Vec4u(23u,17u,unsigned(i),2u);
    s.info.sceneSeed = 41; s.info.t = 0;
    s.firstScatter = birth == 2;
    s.lastVal = birth == 0 ? 0.f : .2f;
    s.lastAniso = Vec3d(.2,-.1,.8);
    return s;
}
void sameEvent(const MediumSample &x, const MediumSample &y) {
    require(x.t == y.t && x.continuedT == y.continuedT && x.exited == y.exited &&
        x.p == y.p && x.pdf == y.pdf && x.weight == y.weight &&
        x.continuedWeight == y.continuedWeight && x.emission == y.emission,
        "native collision position, decision and weight changed");
}
void transport() {
    auto native = medium("sparse_conv_noise"), hybrid = medium("renewal_plus_fp_gradient");
    auto *corrected = dynamic_cast<RenewalPlusFpGradientMedium *>(hybrid.get());
    int hits=0, escapes=0, failures=0, blocked=0, clear=0;
    const Ray ray(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,3.5f);
    const auto kernel = cg::CovarianceKernel::fromCorrelationLengths(1,Eigen::Vector3d::Ones());
    for (int birth=0; birth<3; ++birth) for (int i=0; i<48; ++i) {
        IndependentPathSampler a(724,false), b(724,false); a.startPath(0,i); b.startPath(0,i);
        auto sa=state(i,birth), sb=state(i,birth);
        Medium::DistanceDiagnostic da,db; sa.distanceDiagnostic=&da; sb.distanceDiagnostic=&db;
        MediumSample x{},y{};
        const bool xa=native->sampleDistance(a,ray,sa,x), ya=hybrid->sampleDistance(b,ray,sb,y);
        require(xa==ya,"native accepted/rejected decision changed");
        if (!xa) { ++failures; close(a.next1D(),b.next1D(),0,"failed query RNG"); }
        else {
            sameEvent(x,y);
            require(sa.lastVal==sb.lastVal && sa.firstScatter==sb.firstScatter && sa.bounce==sb.bounce,
                "native endpoint state changed");
            if (x.exited) {
                ++escapes; require(x.aniso==y.aniso && sa.lastAniso==sb.lastAniso,"escape gradient changed");
                close(a.next1D(),b.next1D(),0,"escape RNG changed");
            } else {
                ++hits;
                auto *ct=dynamic_cast<GPContextSparseConvNoise *>(sa.gpContext.get());
                require(ct && ct->conditionedBirth==(birth!=2),"saved birth observation");
                double f0=ct->birthValue; Vec3d g0=ct->birthGradient;
                if (!ct->conditionedBirth) {
                    UniformSampler uniform; int id;
                    f0=ct->noise.evaluateValue(ray.pos(),0,ray.dir(),ct->birthRayInfo,uniform,id);
                    g0=Vec3d(ct->noise.evaluateGradient(ray.pos(),0,ray.dir(),ct->birthRayInfo,uniform));
                }
                // Reconstruct the law independently from the native hit and
                // its unchanged RNG state, including an observed primary jet.
                const double distance=da.position.z();
                if (distance>0 && f0>=0 && (f0!=0 || g0.z()>0)) {
                    const fp::Input input{1,1,distance,f0,g0.z(),double(.1f),0,0,
                        [](double) { return double(.1f); }};
                    cg::Random rng(a); const fp::Law law(input);
                    const double derivative=law.sample(rng);
                    const cg::FixedSpeedGradientDistribution gd(kernel,Eigen::Vector3d::UnitZ(),
                        distance,derivative,Eigen::Vector3d::Zero(),Eigen::Vector3d::Zero(),eigen(g0));
                    const auto expected=gd.sample(rng);
                    close((expected-eigen(y.aniso)).norm(),0,2e-11,"fixed5 uses actual birth and exact native hit");
                    close(a.next1D(),b.next1D(),0,"only fixed5 sampling consumes extra RNG");
                }
                require(y.aniso.z()<0 && y.aniso==sb.lastAniso,"full inward corrected gradient is retained");
                PhaseSample phase;
                require(y.phase->isSpecular() && y.phase->sample(b,ray.dir(),y,phase),"observed-normal mirror phase");
                const auto n=eigen(y.aniso).normalized();
                const Eigen::Vector3d reflected=Eigen::Vector3d::UnitZ()-2*n.z()*n;
                close((eigen(Vec3d(phase.w))-reflected).norm(),0,5e-7,"reflection uses replaced gradient");
                close(phase.pdf,1,0,"native conditional delta PDF");
                close(phase.weight.x(),double(.8f),0,"native reflectance weight");
                require(Vec3d(phase.w).dot(sb.lastAniso)>0,"outgoing renewal+ observation");
                PreparedGpisSegment next;
                const Ray nextRay(y.p,phase.w,0,.3f);
                hybrid->prepareGpisSegment(b,nextRay,sb,next);
                require(next.request.previousGradient==Vec3f(y.aniso) &&
                    next.context->birthGradient==Vec3d(Vec3f(y.aniso)),"next GP segment receives the corrected full jet");
            }
        }
        // Shadow queries do not run the replacement hook or consume gradient RNG.
        a.startPath(9,i); b.startPath(9,i); sa=state(i,birth); sb=state(i,birth);
        const auto before=corrected->correctedCollisions();
        const auto tx=native->transmittance(a,ray,false,false,&sa);
        const auto ty=hybrid->transmittance(b,ray,false,false,&sb);
        require(tx==ty && (tx.x()==0 || tx.x()==1),"identical native binary visibility");
        require(sa.lastVal==sb.lastVal && sa.lastAniso==sb.lastAniso && sa.firstScatter==sb.firstScatter,
            "identical native shadow endpoint");
        require(before==corrected->correctedCollisions(),"shadow does not resample fixed5 gradients");
        close(a.next1D(),b.next1D(),0,"identical shadow RNG");
        if (tx.x()==0) ++blocked; else ++clear;
    }
    require(hits>20 && escapes>10 && blocked>10 && clear>10 && corrected->correctedCollisions()>20,
        "collision/escape/shadow coverage");
    std::cout << "transport identity, full-jet reconstruction and reflection passed: hits=" << hits
        << " escapes=" << escapes << " native_rejections=" << failures << '\n';
    hybrid->teardownAfterRender();
}
void wavefront() {
    auto m=medium("renewal_plus_fp_gradient");
    const Ray ray(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,3.5f);
    int hits=0;
    for (int i=0;i<40;++i) {
        IndependentPathSampler a(947,false),b(947,false); a.startPath(0,i); b.startPath(0,i);
        auto sa=state(i,0), sb=state(i,0); MediumSample x{},y{};
        const bool xa=m->sampleDistance(a,ray,sa,x);
        GaussianProcessMedium::DistanceState distance;
        auto status=m->beginDistance(b,ray,sb,y,distance);
        while (status==GaussianProcessMedium::DistanceStatus::NeedIntersection) {
            PreparedGpisSegment prepared; GpRaySegmentReply reply;
            m->prepareGpisSegment(b,distance.ray,sb,prepared);
            double t=distance.t;
            const bool hit=m->finishGpisSegment(distance.ray,sb,prepared,reply,t);
            status=m->resumeDistance(b,ray,sb,y,distance,t,hit);
        }
        require(xa==(status==GaussianProcessMedium::DistanceStatus::Finished),"wavefront decision");
        if (xa) {
            sameEvent(x,y); require(x.aniso==y.aniso && sa.lastAniso==sb.lastAniso,"wavefront must invoke replacement hook");
            if (!x.exited) ++hits;
        }
        close(a.next1D(),b.next1D(),0,"wavefront RNG identity");
    }
    require(hits>5,"wavefront collision coverage");
    std::cout << "synchronous/wavefront identity passed\n";
}
void settings() {
    auto native=medium("sparse_conv_noise");
    auto disabled=medium("renewal_plus_fp_gradient",R"(,"fp_gradient":{"enabled":false})");
    for (int i=0;i<24;++i) {
        IndependentPathSampler a(196,false),b(196,false); a.startPath(0,i); b.startPath(0,i);
        auto sa=state(i,0),sb=state(i,0); MediumSample x{},y{};
        const Ray ray(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,3.5f);
        const bool xa=native->sampleDistance(a,ray,sa,x),ya=disabled->sampleDistance(b,ray,sb,y);
        require(xa==ya,"disabled decision identity");
        if (xa) { sameEvent(x,y); require(x.aniso==y.aniso,"disabled gradient identity"); }
        close(a.next1D(),b.next1D(),0,"disabled RNG identity");
    }
    auto enabled=medium("renewal_plus_fp_gradient"); rapidjson::Document json;
    auto encoded=enabled->toJson(json.GetAllocator());
    require(std::string(encoded["type"].GetString())=="renewal_plus_fp_gradient" &&
        encoded["fp_gradient"]["enabled"].GetBool() &&
        std::string(encoded["fp_gradient"]["anchor_selection"].GetString())=="fixed5_snr", "hybrid round trip");
    for (const std::string bad : {R"(,"1D_sampling":true)",R"(,"single_realization":true)",
            R"(,"1D_sampling_scheme":"mis")",R"(,"fp_gradient":{"anchor_selection":"unknown"})"}) {
        bool rejected=false;
        try { medium("renewal_plus_fp_gradient",bad); } catch (const std::exception &) { rejected=true; }
        require(rejected,"reject incompatible sampling mode");
    }
    bool rejected=false;
    try { medium("renewal_plus_fp_gradient","","global"); } catch (const std::exception &) { rejected=true; }
    require(rejected,"require true renewal+ memory");
    std::cout << "disabled identity, serialization and supported-mode checks passed\n";
}
}
int main(int argc,char **argv) {
    try {
        const std::string mode=argc>1 ? argv[1] : "all";
        if (mode=="all" || mode=="transport") transport();
        if (mode=="all" || mode=="wavefront") wavefront();
        if (mode=="all" || mode=="settings") settings();
        return 0;
    } catch (const std::exception &e) { std::cerr << e.what() << '\n'; return 1; }
}
