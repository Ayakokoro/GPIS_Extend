#include "media/GlobalConditionalMedium.hpp"
#include "math/FirstPassageGradient.hpp"
#include "phasefunctions/PhaseFunction.hpp"
#include "samplerecords/PhaseSample.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/ImageIO.hpp"
#include "io/Scene.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

using namespace Tungsten;
namespace cg = ConditionalGpis;
namespace fp = FirstPassageGradient;
namespace {
using V = Eigen::Vector3d;
V eigen(const Vec3f &v) { return {v.x(),v.y(),v.z()}; }
void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
void close(double a,double b,double tolerance,const char *message) {
    if(!std::isfinite(a) || !std::isfinite(b) || std::abs(a-b)>tolerance) {
        std::cerr<<message<<": "<<a<<" vs "<<b<<'\n'; throw std::runtime_error(message);
    }
}
std::shared_ptr<GlobalConditionalMedium> medium(const std::string &settings="",
        const std::string &material="mirror") {
    JsonDocument document(Path("conditional-fp-gradient-test.json"),
        std::string(R"({"type":"global_conditional","gaussian_process":{"type":"standard",
        "mean":{"type":"homogeneous","offset":0.1},
        "covariance":{"type":"squared_exponential","sigma":1,"lengthScale":1}},
        "material":{"type":")")+material+R"(","albedo":0.8},"max_bounces":8)"+settings+"}");
    Scene scene;
    auto m=std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(document));
    require(bool(m),"conditional factory"); m->loadResources(); m->prepareForRender(); return m;
}
Medium::MediumState state(const GlobalConditionalMedium &m,double value,double age=0) {
    Medium::MediumState s{};s.reset();
    auto c=std::make_shared<ConditionalPathContext>();c->owner=&m;
    c->position=V(0,0,age);c->rayOrigin=Vec3f(0,0,float(age));
    c->birth.birthPosition=V::Zero();c->birth.direction=V::UnitZ();
    c->birth.birthValue=value;c->birth.birthGradient=V(.2,-.1,.8);
    s.pathContext=c;return s;
}
void sameObservation(const Medium::MediumState &a,const Medium::MediumState &b) {
    const auto x=std::dynamic_pointer_cast<const ConditionalPathContext>(a.pathContext);
    const auto y=std::dynamic_pointer_cast<const ConditionalPathContext>(b.pathContext);
    require(x && y && x->position==y->position && x->rayOrigin==y->rayOrigin &&
        x->birth.birthPosition==y->birth.birthPosition && x->birth.birthValue==y->birth.birthValue &&
        x->birth.birthGradient==y->birth.birthGradient,"identical renewed conditional observation");
}
void transport() {
    auto base=medium(),on=medium(R"(,"fp_gradient":{"enabled":true})");
    require(!base->gradientCorrectionEnabled() && on->gradientCorrectionEnabled() &&
        on->bridgeCorrectionEnabled() && on->anchorSelection()==fp::AnchorSelection::Fixed5Snr,
        "default and enabled gradient settings");
    int hits=0,escapes=0;
    for(double f0:{0.,.2}) for(double age:{0.,.125}) for(int i=0;i<60;++i) {
        const Ray ray(Vec3f(0,0,float(age)),Vec3f(0.f,0.f,1.f),.001f,2.f);
        IndependentPathSampler a(814,false),b(814,false);a.startPath(0,i);b.startPath(0,i);
        auto sa=state(*base,f0,age),sb=state(*on,f0,age);MediumSample x{},y{};
        require(base->sampleDistance(a,ray,sa,x) && on->sampleDistance(b,ray,sb,y),"distance sampling");
        require(x.t==y.t && x.exited==y.exited && x.p==y.p && x.pdf==y.pdf &&
            x.weight==y.weight,"unchanged conditional collision event");
        if(y.exited) {
            ++escapes;sameObservation(sa,sb);close(a.next1D(),b.next1D(),0,"unchanged escape RNG");
        } else {
            ++hits;
            const auto *e=dynamic_cast<const ConditionalEventContext *>(y.eventContext.get());
            const auto *raw=dynamic_cast<const ConditionalEventContext *>(x.eventContext.get());
            require(e && e->correctedGradient && e->derivative<0,"corrected collision observation");
            // Reconstruct from the exact double hit position, not float sample.t.
            const fp::Input input{1,1,e->position.z(),f0,.8,double(.1f),0,0,[](double){return double(.1f);}};
            const fp::Law law(input);
            close(law.gaussianMean(),raw->gradient.mean.z(),2e-10,"same endpoint Gaussian mean");
            close(law.gaussianStddev()*law.gaussianStddev(),raw->gradient.covariance(2,2),2e-10,
                "same endpoint Gaussian variance");
            cg::Random rng(a);
            close(e->derivative,law.sample(rng),0,"actual birth and full flight distance feed corrected law");
        }
        a.startPath(9,i);b.startPath(9,i);sa=state(*base,f0,age);sb=state(*on,f0,age);
        const auto saved=sb.pathContext;
        close(base->transmittance(a,ray,false,false,&sa).x(),
              on->transmittance(b,ray,false,false,&sb).x(),0,"identical conditional shadow weight");
        sameObservation(sa,sb);close(a.next1D(),b.next1D(),0,"identical shadow RNG");
        require(saved!=sb.pathContext &&
            std::dynamic_pointer_cast<const ConditionalPathContext>(saved)->position==V(0,0,age),
            "shadow renewal preserves immutable input");
    }
    require(hits>30 && escapes>30 && on->correctedCollisions()==std::uint64_t(hits),"collision/escape coverage");
    for(const std::string material:{"mirror","lambert","conductor"}) {
        auto aMedium=medium("",material);
        auto bMedium=medium(R"(,"fp_gradient":{"enabled":false,"bridge_correction":false})",material);
        for(int i=0;i<30;++i) {
            IndependentPathSampler a(962,false),b(962,false);a.startPath(0,i);b.startPath(0,i);
            auto sa=state(*aMedium,.2),sb=state(*bMedium,.2);MediumSample x{},y{};
            const Ray ray(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,2);
            require(aMedium->sampleDistance(a,ray,sa,x) && bMedium->sampleDistance(b,ray,sb,y),"disabled sampling");
            require(x.t==y.t && x.exited==y.exited,"disabled event identity");
            if(!x.exited) {
                PhaseSample p,q;
                require(x.phase->sample(a,ray.dir(),x,p) && y.phase->sample(b,ray.dir(),y,q),"disabled phase");
                require(p.w==q.w && p.pdf==q.pdf && p.weight==q.weight,"disabled phase identity");
            }
            close(a.next1D(),b.next1D(),0,"disabled RNG identity");
        }
    }
    // The control is serialized, including the Rice-only ablation.
    auto rice=medium(R"(,"fp_gradient":{"enabled":true,"bridge_correction":false})");
    rapidjson::Document json;auto encoded=rice->toJson(json.GetAllocator());
    require(encoded["fp_gradient"]["enabled"].GetBool() &&
        !encoded["fp_gradient"]["bridge_correction"].GetBool() && !rice->bridgeCorrectionEnabled(),
        "gradient settings round trip");
    require(std::string(encoded["fp_gradient"]["anchor_selection"].GetString())=="fixed5_snr",
        "serialize default anchor selection");
    auto legacy=medium(R"(,"fp_gradient":{"enabled":true,"anchor_selection":"hermite"})");
    const auto legacyJson=legacy->toJson(json.GetAllocator());
    require(legacy->anchorSelection()==fp::AnchorSelection::Hermite &&
        std::string(legacyJson["fp_gradient"]["anchor_selection"].GetString())=="hermite",
        "explicit legacy selector survives serialization");
    bool rejected=false;
    try { medium(R"(,"fp_gradient":{"enabled":true,"anchor_selection":"unknown"})"); }
    catch(const std::exception &) { rejected=true; }
    require(rejected,"reject invalid anchor selection");
    std::cout<<"conditional transport, renewal, birth inputs and disabled identity passed; hits="<<hits<<'\n';
}
void phase() {
    const V w=V::UnitZ();const Vec3f wi(0.f,0.f,1.f);
    const auto kernel=cg::CovarianceKernel::fromCorrelationLengths(1,{.7,1.2,.9});
    for(const std::string material:{"mirror","lambert","conductor"}) {
        auto m=medium(R"(,"fp_gradient":{"enabled":true})",material);
        MediumSample s{};s.p=Vec3f(0.f);s.exited=false;
        const auto *phase=m->phaseFunction(s.p);
        auto e=std::make_shared<ConditionalEventContext>();e->owner=m.get();e->position=V::Zero();e->derivative=-.8;
        e->correctedGradient.emplace(kernel,w,1.3,-.8,V(.2,.1,.3),V(.1,0,.2),V(.2,-.3,.8));
        s.eventContext=e;cg::Random rng(123);
        const V observed=e->correctedGradient->sample(rng);s.aniso=Vec3d(observed.x(),observed.y(),observed.z());
        double mass=0;
        for(int z=0;z<160;++z) for(int a=0;a<240;++a) {
            const double c=-1+2*(z+.5)/160,angle=2*cg::kPi*(a+.5)/240,r=std::sqrt(1-c*c);
            mass+=phase->pdf(wi,Vec3f(float(r*std::cos(angle)),float(r*std::sin(angle)),float(c)),s)*
                4*cg::kPi/(160*240);
        }
        close(mass,1,.002,"conditional fixed-speed phase normalization");
        IndependentPathSampler sampler(768,false);sampler.startPath(0,0);
        for(int i=0;i<100;++i) {
            PhaseSample p;require(phase->sample(sampler,wi,s,p),"corrected conditional phase");
            const double density=phase->pdf(wi,p.w,s);
            close(p.pdf,density,2e-4*p.pdf,"conditional phase and NEE PDFs");
            const Vec3f value=phase->eval(wi,p.w,s);
            for(int k=0;k<3;++k) close(value[k]/density,p.weight[k],2e-5,"conditional eval and weight");
            std::shared_ptr<const MediumPathContext> connection;
            phase->evalConnection(sampler,wi,p.w,s,connection);
            for(const auto &branch:{p.pathContext,connection}) {
                const auto c=std::dynamic_pointer_cast<const ConditionalPathContext>(branch);
                require(c && c->owner==m.get() && c->birth.birthValue==0 && c->position==e->position &&
                    eigen(p.w).dot(c->birth.birthGradient)>0,"ordinary conditional outgoing observation");
                close(w.dot(c->birth.birthGradient),e->derivative,2e-14,"shared corrected axial observation");
                if(material=="lambert") close((c->birth.birthGradient-observed).norm(),0,1e-14,"shared Lambert gradient");
                if(i<8) {
                    auto st=state(*m,.2);st.pathContext=branch;
                    const auto saved=st.pathContext;
                    const Vec3f tr=m->transmittance(sampler,Ray(s.p,p.w,0,.25f),false,false,&st);
                    require(tr.x()>=0 && tr.x()<=1 && std::isfinite(tr.x()) && saved!=st.pathContext,
                        "corrected birth is consumed by conditional transmittance");
                }
            }
        }
    }
    std::cout<<"conditional gradient phase normalization, MIS and outgoing shadow state passed\n";
}
void image(const char *path) {
    int w,h;auto pixels=ImageIO::loadHdr(Path(path),TexelConversion::REQUEST_RGB,w,h);
    require(bool(pixels) && w>0 && h>0,"read render EXR");double sum=0,lo=1e30,hi=0;
    for(int i=0;i<w*h*3;++i) {
        require(std::isfinite(pixels[i]) && pixels[i]>=0,"finite nonnegative render");
        sum+=pixels[i];lo=std::min(lo,double(pixels[i]));hi=std::max(hi,double(pixels[i]));
    }
    require(sum>0 && hi-lo>.1,"visible render");
    std::cout<<w<<'x'<<h<<" finite image, mean="<<sum/(w*h*3)<<'\n';
}
}
int main(int argc,char **argv) {
    try {
        const std::string mode=argc>1 ? argv[1] : "all";
        if(mode=="image" && argc==3) {image(argv[2]);return 0;}
        if(mode=="all" || mode=="transport") transport();
        if(mode=="all" || mode=="phase") phase();
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
