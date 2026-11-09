#include "math/FirstPassageGradient.hpp"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "media/ClassicGlobalFpGradientMedium.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "sampling/SampleWarp.hpp"
#include "FpGradientDocumentExport.hpp"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace Tungsten;
namespace fp = FirstPassageGradient;
namespace cg = ConditionalGpis;
namespace {
void require(bool ok,const char *message) { if(!ok) throw std::runtime_error(message); }
void close(double actual,double expected,double tolerance,const char *message) {
    if(!(std::abs(actual-expected)<=tolerance)) {
        std::cerr<<message<<": "<<actual<<" != "<<expected<<" (tol "<<tolerance<<")\n";
        throw std::runtime_error(message);
    }
}
fp::Input linear(double t,double f,double g,double slope) {
    return {1,1,t,f,g,0,slope,slope,[slope](double s){ return slope*s; }};
}
void distributions() {
    // Independent SE feature/QR conditioning and SciPy PDF quadrature.
    // Anchor positions also checked by minimizing the Hermite SNR directly.
    // Cover P<0, a surface start, and the long-flight 1.5/ gamma window.
    const double hermiteFixtures[][12]={
        {1.2,.2,-1,.5,.87794825338329308,.46439129924335171,-8.0297689134880486,-2.7263788197111807,
            -.54646636648711344,.33817050014866706,.022207398251003145,.51366270353949561},
        {2,.2,-.8,-.3,-.036890116768143633,.85922216598295287,-1.3456149238025501,-1.0529928410182148,
            -1.3656934488087504,2.1891190413909425,.72225082012891773,.55049543924954469},
        {3,0,.5,-.3,-.36818396206671844,.99547976771391133,.33684178794982611,-.7122964357986008,
            -1.502010349832948,2.7483205464287468,.74133621773695602,1.5},
        {5,1,-.5,-.3,-.30000074272664479,.99999999582667287,-.69359663573463182,-.60061725128032717,
            -1.6119171724878352,3.1112998349551781,.79010350582970656,3.5}
    };
    // Independently condition the same GP at the five fixed midpoint
    // locations with SE features/QR, then integrate the selected density.
    const double fixedFixtures[][12]={
        {1.2,.2,-1,.5,.87794825338329308,.46439129924335171,-8.695974259366567,-3.1247863410024546,
            -.515046382038001,.302458550661892,.016726864038972823,.6},
        {2,.2,-.8,-.3,-.036890116768143633,.85922216598295287,-1.3165888188651749,-1.1469291298788078,
            -1.3522299842843446,2.1448051822860466,.7166011617488901,.65},
        {3,0,.5,-.3,-.36818396206671844,.99547976771391133,.30149599414677014,-.8373186846173183,
            -1.5057429163943232,2.754209062450844,.7452303845538738,1.65},
        {5,1,-.5,-.3,-.30000074272664479,.99999999582667287,-.6666712270649829,-.7359739730723779,
            -1.617710423393799,3.115935024263848,.7974478514684572,3.65}
    };
    for(auto selection:{fp::AnchorSelection::Hermite,fp::AnchorSelection::Fixed5Snr}) {
    const auto &fixtures=selection==fp::AnchorSelection::Hermite ? hermiteFixtures : fixedFixtures;
    for(const auto &f:fixtures) {
        const fp::Law law(linear(f[0],f[1],f[2],f[3]),true,selection);
        close(law.gaussianMean(),f[4],2e-10,"Gaussian mean");
        close(law.gaussianStddev(),f[5],2e-10,"Gaussian sd");
        close(law.probitIntercept(),f[6],2e-7,"probit intercept");
        close(law.probitSlope(),f[7],2e-7,"probit slope");
        close(law.anchorDistance(),f[11],2e-10,"single anchor SNR minimum / window");
        close(law.cdf(-1),f[10],2e-7,"independent CDF fixture");
        double mass=0;
        constexpr int cells=600;
        for(int k=0;k<cells;++k) mass+=law.pdf(-8.0*(k+.5)/cells)*8/cells;
        close(mass,1,1e-4,"normalized axial PDF");
        cg::Random rng(9187);
        double sum=0,sum2=0; int below=0;
        constexpr int count=12000;
        for(int i=0;i<count;++i) {
            const double v=law.sample(rng);
            require(v<0 && std::isfinite(v),"negative finite sampled derivative");
            sum+=v; sum2+=v*v; below+=v<=-1;
        }
        close(sum/count,f[8],.025,"axial sample mean");
        close(sum2/count,f[9],.085,"axial sample second moment");
        close(double(below)/count,f[10],.018,"sample CDF");
        const fp::Law rice(linear(f[0],f[1],f[2],f[3]),false);
        close(rice.cdf(-1),cg::negativeFluxNormalCdf(-1,rice.gaussianMean(),rice.gaussianStddev()).value,
            2e-8,"bridge disabled retains conditioned Rice law");
    }
    }
    {
        int queries=0;
        auto in=linear(5,1,-.5,-.3);
        in.mean=[&](double s) { ++queries; return -.3*s; };
        const fp::Law law(in);
        require(law.anchorSelection()==fp::AnchorSelection::Fixed5Snr && law.candidateCount()==5,
            "default selects one anchor from five fixed candidates");
        require(queries==6,"five actual SDF candidates plus endpoint; no refinement");
        cg::Random rng(164);
        for(int i=0;i<100;++i) law.sample(rng);
        require(queries==6,"gradient proposals keep the selected anchor fixed");
    }
    // Short flights need the high-precision bridge, including surface f0=0.
    for(double t:{1e-6,1e-4,.01,.499,.501,1.,8.,30.}) {
        for(bool surface:{false,true}) {
            const auto in=linear(t,surface ? 0 : .2,surface ? .5 : -.7,.3);
            const fp::Law law=[&]() {
                try { return fp::Law(in); }
                catch(const std::exception &e) {
                    throw std::runtime_error("short/long t="+std::to_string(t)+
                        " surface="+std::to_string(surface)+": "+e.what());
                }
            }();
            require(law.anchorDistance()>0 && law.anchorDistance()<t &&
                law.anchorDistance()>=t-1.5,"anchor remains inside the active window");
            cg::Random rng(193);
            for(int i=0;i<20;++i) require(law.sample(rng)<0,"short/long flight support");
            if(surface && t<1e-3) {
                close(law.gaussianMean(),-.5,3*t*t,"short surface mean limit");
                close(law.gaussianStddev()/(t*t),1/std::sqrt(6.),2*t*t,"short surface variance limit");
            }
        }
    }
    const fp::Input scaled{.02,5,.4,.004,-.08,0,-.03,-.03,[](double s){ return -.03*s; }};
    const fp::Law units(scaled),unit(linear(2,.2,-.8,-.3));
    close(units.gaussianMean(),.1*unit.gaussianMean(),1e-12,"physical gradient units");
    close(units.probitIntercept(),unit.probitIntercept(),1e-9,"dimensionless bridge invariance");
    close(units.anchorDistance(),unit.anchorDistance()/5,1e-12,"physical anchor units");
    // A nearly tangent outgoing birth places the adaptive anchor very close
    // to the start even though gamma*t=1. This exercises the high-precision
    // retry for an adaptive anchor, independently of the short-flight path.
    const fp::Law grazingBirth(linear(1,0,1e-12,0),true,fp::AnchorSelection::Hermite);
    require(grazingBirth.anchorDistance()>0 && grazingBirth.anchorDistance()<1e-4,
        "long flight with near-start anchor");
    {
        cg::Random grazingRng(6248); int below=0;
        const double threshold=-grazingBirth.gaussianStddev();
        for(int i=0;i<4000;++i) {
            const double v=grazingBirth.sample(grazingRng);
            require(v<0 && std::isfinite(v),"near-start anchor finite sampling");
            below+=v<=threshold;
        }
        close(double(below)/4000,grazingBirth.cdf(threshold),.035,
            "near-start anchor sample CDF");
    }
    // A deep interior mean well gives tiny proposal acceptance. Exercise the
    // log-concave envelope and its unchanged target, rather than losing a hit.
    auto rare=linear(2,.2,-.8,-.3);
    rare.mean=[](double s){return -.3*s-3*std::pow(std::sin(3.141592653589793*s/2),2);};
    const fp::Law tail(rare);
    cg::Random rng(775);
    int fallback=0,below=0;
    const double threshold=tail.gaussianMean()-3*tail.gaussianStddev();
    const double expected=tail.cdf(threshold);
    for(int i=0;i<2000;++i) {
        bool used=false; const double v=tail.sample(rng,&used);
        fallback+=used; below+=v<=threshold;
    }
    require(fallback>100,"rare bridge uses envelope fallback");
    close(double(below)/2000,expected,.04,"rare bridge sample CDF");
    // Close encounters across a mean well produce log probabilities around
    // -1e18. Direct single-anchor parameters must remain finite in this tail.
    for(double distance:{.002,.0002}) {
        auto shortWell=linear(distance,0,.5,0);
        // Match the actual sine-squared mean's zero endpoint derivative.
        // The adaptive anchor then probes the interior well in both cases.
        shortWell.mean=[distance](double s) {
            return -.001*std::pow(std::sin(3.141592653589793*s/distance),2);
        };
        const fp::Law law(shortWell);
        const double c=-law.gaussianMean()/law.gaussianStddev();
        const double u0=c>=0 ? -2/(c+std::hypot(c,2.)) : .5*(c-std::hypot(c,2.));
        const double a=std::fma(law.probitSlope(),u0,law.probitIntercept());
        require(a<-1e8,"extreme bridge tail regression coverage");
        cg::Random shortRng(751);
        for(int i=0;i<100;++i) {
            const double v=law.sample(shortRng);
            require(v<0 && std::isfinite(v),"rare short-flight sampling remains finite");
        }
    }
    std::cout<<"gradient distribution checks passed\n";
}

std::shared_ptr<ClassicGlobalMedium> medium(const std::string &type,const std::string &extra="",
        const std::string &material="mirror",const std::string &covariance="squared_exponential") {
    const std::string json=R"({"type":")"+type+R"(","gaussian_process":{"type":"standard",
        "mean":{"type":"homogeneous","offset":0.1},"covariance":{"type":")"+covariance+
        R"(","sigma":1,"lengthScale":1}},"material":{"type":")"+material+R"(","albedo":0.8},
        "max_bounces":8)"+extra+"}";
    JsonDocument document(Path("fp-gradient-test.json"),json);
    Scene scene;
    auto m=std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(document));
    require(bool(m),"factory registration"); m->loadResources(); m->prepareForRender(); return m;
}
Medium::MediumState state() { Medium::MediumState s{};s.reset();return s; }

void transport() {
    const auto classic=medium("classic_global"),corrected=medium("classic_global_fp_gradient");
    const Ray ray(Vec3f(0.f),Vec3f(0.f,0.f,1.f),.001f,2.f);
    int hits=0;
    for(int i=0;i<500;++i) {
        IndependentPathSampler a(814,false),b(814,false);
        a.startPath(0,i);b.startPath(0,i);
        auto sa=state(),sb=state(); MediumSample x,y;
        require(classic->sampleDistance(a,ray,sa,x) && corrected->sampleDistance(b,ray,sb,y),"distance sampling");
        close(x.t,y.t,0,"unchanged classic collision distance");
        require(x.exited==y.exited && x.p==y.p && x.weight==y.weight && x.pdf==y.pdf,"unchanged distance event");
        if(!y.exited) {
            ++hits;
            const auto *event=dynamic_cast<const FpGradientEventContext *>(y.eventContext.get());
            require(event && event->gradient,"corrected collision event");
            PhaseSample sample;
            require(y.phase->sample(b,ray.dir(),y,sample),"corrected phase sample");
            const auto next=std::dynamic_pointer_cast<const FpGradientPathContext>(sample.pathContext);
            require(next && next->value==0 && next->direction.dot(next->gradient)>0,"full outgoing surface observation");
            close(sample.pdf,y.phase->pdf(ray.dir(),sample.w,y),2e-4*sample.pdf,"phase sample/pdf agreement");
            sb.pathContext=sample.pathContext;
            const auto saved=sb.pathContext;
            corrected->transmittance(b,Ray(y.p,sample.w,0,1),false,false,&sb);
            require(saved==sb.pathContext,"classic visibility does not modify flight state");
        }
        a.startPath(9,i);b.startPath(9,i);
        close(classic->transmittance(a,ray,false,false,&sa).x(),
              corrected->transmittance(b,ray,false,false,&sb).x(),0,"identical classic shadow tracking");
        close(a.next1D(),b.next1D(),0,"identical shadow RNG consumption");
    }
    require(hits>100,"transport test covers collisions");
    {
        IndependentPathSampler rng(118,false);rng.startPath(0,0);
        auto s=state();MediumSample event;
        // Boundary pieces inside geometry epsilon are empty for classic, but
        // still move the flight origin. They must not renew the latent jet.
        const Ray empty(Vec3f(0.f),Vec3f(0.f,0.f,1.f),.001f,.001f);
        corrected->sampleDistance(rng,empty,s,event);
        const auto first=std::dynamic_pointer_cast<const FpGradientPathContext>(s.pathContext);
        corrected->sampleDistance(rng,Ray(event.p,empty.dir(),.001f,.001f),s,event);
        const auto next=std::dynamic_pointer_cast<const FpGradientPathContext>(s.pathContext);
        require(first && next && first!=next && first->origin==next->origin &&
            first->gradient==next->gradient && first->value==next->value,"straight subdivisions preserve birth");
        close(next->age,2*double(.001f),0,"straight physical age");
        const Vec3f turn=event.p;
        corrected->sampleDistance(rng,Ray(turn,Vec3f(1.f,0.f,0.f),.001f,.001f),s,event);
        const auto turned=std::dynamic_pointer_cast<const FpGradientPathContext>(s.pathContext);
        require(turned && turned->origin==Eigen::Vector3d(turn.x(),turn.y(),turn.z()) && turned->value>0,
            "mesh turn initializes a positive exterior jet");
        close(next->age,2*double(.001f),0,"branches keep immutable incoming history");
    }
    for(const std::string material:{"mirror","lambert","conductor"}) {
        const auto base=medium("classic_global","",material);
        const auto off=medium("classic_global_fp_gradient",R"(,"fp_gradient":{"enabled":false})",material);
        for(int i=0;i<30;++i) {
            IndependentPathSampler a(481,false),b(481,false);a.startPath(0,i);b.startPath(0,i);
            auto sa=state(),sb=state();MediumSample x,y;
            base->sampleDistance(a,ray,sa,x);off->sampleDistance(b,ray,sb,y);
            require(x.exited==y.exited && x.p==y.p,"disabled mode restores classic event");
            if(!x.exited) {
                PhaseSample p,q;x.phase->sample(a,ray.dir(),x,p);y.phase->sample(b,ray.dir(),y,q);
                require(p.w==q.w && p.weight==q.weight && p.pdf==q.pdf,"disabled mode restores classic phase");
            }
            close(a.next1D(),b.next1D(),0,"disabled mode exact random stream");
        }
    }
    bool rejected=false;
    try { medium("classic_global_fp_gradient","","mirror","rational_quadratic"); }
    catch(const std::exception &) { rejected=true; }
    require(rejected,"reject unsupported non-SE gradient model");
    rapidjson::Document doc;auto serialized=corrected->toJson(doc.GetAllocator());
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);serialized.Accept(writer);
    JsonDocument input(Path("roundtrip.json"),buffer.GetString());Scene scene;
    auto restored=std::dynamic_pointer_cast<ClassicGlobalFpGradientMedium>(scene.fetchMedium(input));
    require(restored && restored->gradientCorrectionEnabled() && restored->bridgeCorrectionEnabled() &&
        restored->anchorSelection()==fp::AnchorSelection::Fixed5Snr,"JSON roundtrip with default selector");
    const auto legacy=medium("classic_global_fp_gradient",R"(,"fp_gradient":{"anchor_selection":"hermite"})");
    const auto legacyJson=legacy->toJson(doc.GetAllocator());
    require(std::string(legacyJson["fp_gradient"]["anchor_selection"].GetString())=="hermite" &&
        dynamic_cast<ClassicGlobalFpGradientMedium *>(legacy.get())->anchorSelection()==fp::AnchorSelection::Hermite,
        "explicit legacy selector survives serialization");
    rejected=false;
    try { medium("classic_global_fp_gradient",R"(,"fp_gradient":{"anchor_selection":"unknown"})"); }
    catch(const std::exception &) { rejected=true; }
    require(rejected,"reject invalid anchor selection");
    std::cout<<"classic transport and disabled-mode checks passed\n";
}

void phase() {
    for(const std::string material:{"mirror","lambert","conductor"}) {
        const auto m=medium("classic_global_fp_gradient","",material);
        const Eigen::Vector3d w=Eigen::Vector3d(1,2,-3).normalized();
        const Vec3f wi(float(w.x()),float(w.y()),float(w.z()));
        const cg::CovarianceKernel kernel=cg::CovarianceKernel::fromCorrelationLengths(1,{.8,1.3,1.8});
        MediumSample event{};event.p=Vec3f(0.f);
        auto context=std::make_shared<FpGradientEventContext>();
        context->owner=m.get();context->position=Eigen::Vector3d::Zero();
        context->derivative=-.8;
        context->gradient.emplace(kernel,w,1.3,-.8,Eigen::Vector3d(.2,.1,.3),
            Eigen::Vector3d(.1,0,.2),Eigen::Vector3d(.2,-.3,.8));
        event.eventContext=context;
        cg::Random rng(123);
        const auto g=context->gradient->sample(rng);event.aniso=Vec3d(g.x(),g.y(),g.z());
        const auto *p=m->phaseFunction(event.p);
        constexpr double pi=3.141592653589793;
        double mass=0;
        for(int z=0;z<160;++z) for(int a=0;a<240;++a) {
            const double c=-1+2*(z+.5)/160,angle=2*pi*(a+.5)/240,r=std::sqrt(1-c*c);
            mass+=p->pdf(wi,Vec3f(float(r*std::cos(angle)),float(r*std::sin(angle)),float(c)),event)*4*pi/(160*240);
        }
        close(mass,1,.002,"anisotropic fixed-speed phase normalization");
        IndependentPathSampler sampler(768,false);sampler.startPath(0,0);
        for(int i=0;i<100;++i) {
            PhaseSample result;require(p->sample(sampler,wi,event,result),"phase sample");
            const float density=p->pdf(wi,result.w,event);
            close(result.pdf,density,2e-4*result.pdf,"sample PDF matches NEE PDF within float direction precision");
            const Vec3f value=p->eval(wi,result.w,event);
            for(int c=0;c<3;++c) close(value[c]/density,result.weight[c],2e-5,"phase evaluation/weight");
            std::shared_ptr<const MediumPathContext> branch;
            p->evalConnection(sampler,wi,result.w,event,branch);
            require(bool(branch),"NEE has independent outgoing observation");
            if(material=="lambert") {
                const auto next=std::dynamic_pointer_cast<const FpGradientPathContext>(branch);
                close((next->gradient-g).norm(),0,1e-14,"Lambert strategies share full gradient");
            }
        }
    }
    for(double derivative:{-1e-12,-1e-22}) {
        const auto m=medium("classic_global_fp_gradient");
        const Eigen::Vector3d w=Eigen::Vector3d(1,2,-3).normalized();
        const Vec3f wi(float(w.x()),float(w.y()),float(w.z()));
        const auto kernel=cg::CovarianceKernel::fromCorrelationLengths(.02,{.2,.2,.2});
        MediumSample event{};event.p=Vec3f(0.f);
        auto context=std::make_shared<FpGradientEventContext>();
        context->owner=m.get();context->position=Eigen::Vector3d::Zero();
        context->derivative=derivative;
        context->gradient.emplace(kernel,w,1e-3,context->derivative,Eigen::Vector3d(1,.5,.3),
            Eigen::Vector3d(1,.5,.3),Eigen::Vector3d(1,.5,.3));
        event.eventContext=context;
        const auto *p=m->phaseFunction(event.p);
        IndependentPathSampler sampler(712,false);sampler.startPath(0,0);
        int roundedOutside=0;
        for(int i=0;i<100;++i) {
            PhaseSample result;
            require(p->sample(sampler,wi,event,result),"sub-float grazing reflection survives");
            require(result.pdf>0 && std::isfinite(result.pdf),"finite grazing sample PDF");
            if(derivative==-1e-22)
                require(result.pdf>std::numeric_limits<float>::max(),"grazing regression exceeds float density range");
            close(SampleWarp::powerHeuristic(result.pdf,1),1,1e-12,"finite grazing MIS weight");
            roundedOutside+=p->pdf(wi,result.w,event)==0;
            const auto next=std::dynamic_pointer_cast<const FpGradientPathContext>(result.pathContext);
            require(next && next->derivative==-context->derivative,"preserve grazing axial observation");
        }
        require(roundedOutside>0,"grazing regression exercises float direction loss");
    }
    close(SampleWarp::powerHeuristic(1e250,1e250),.5,0,"large double MIS ratio");
    close(SampleWarp::powerHeuristic(1e-250,1e-250),.5,0,"small double MIS ratio");
    std::cout<<"phase normalization and MIS checks passed\n";
}
void image(const char *path) {
    int w,h;auto pixels=ImageIO::loadHdr(Path(path),TexelConversion::REQUEST_RGB,w,h);
    require(bool(pixels) && w>0 && h>0,"read render EXR");
    double sum=0,lo=1e30,hi=0;
    for(int i=0;i<w*h*3;++i) {
        require(std::isfinite(pixels[i]) && pixels[i]>=0,"finite nonnegative render");
        sum+=pixels[i];lo=std::min(lo,double(pixels[i]));hi=std::max(hi,double(pixels[i]));
    }
    require(sum>0 && hi-lo>.1,"visible object in render");
    std::cout<<w<<'x'<<h<<" finite image, mean="<<sum/(w*h*3)<<'\n';
}
}
int main(int argc,char **argv) {
    std::cout.setf(std::ios::unitbuf);
    try {
        const std::string mode=argc>1 ? argv[1] : "all";
        if(mode=="document" && (argc==3 || argc==4)) {
            FpGradientDocumentExport::run(argv[2],argc==4 ? std::stoi(argv[3]) : 100000);
            return 0;
        }
        if(mode=="image" && argc==3) { image(argv[2]);return 0; }
        if(mode=="all" || mode=="distribution") distributions();
        if(mode=="all" || mode=="transport") transport();
        if(mode=="all" || mode=="phase") phase();
        return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n';return 1; }
}
