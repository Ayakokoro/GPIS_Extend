#include "media/ClassicGlobalMedium.hpp"
#include "math/GPFunctions.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <cmath>
#include <iostream>
#include <stdexcept>

using namespace Tungsten;
namespace {
constexpr double Pi = 3.14159265358979323846;
const std::string Base = R"({"type":"squared_exponential","sigma":1.3,"lengthScale":0.8})";
const std::string RampA = R"({"type":"noise","noise":"left_right","min":0.5,"max":1.1,"start":-0.5,"end":0.7})";
const std::string RampS = R"({"type":"noise","noise":"bottom_top","min":0.7,"max":1.2,"start":-0.5,"end":0.7})";
const std::string Mean = R"({"type":"linear","direction":[0,0,1],"scale":0.7})";
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void close(double a, double b, double tolerance, const char *message) {
    if (!(std::abs(a-b) <= tolerance)) {
        std::cerr << message << ": " << a << " vs " << b << '\n';
        throw std::runtime_error(message);
    }
}
Eigen::Vector3d eigen(Vec3d v) { return Eigen::Vector3d(v.x(),v.y(),v.z()); }
std::string covariance(const std::string &a, const std::string &s, const std::string &base = Base) {
    return R"({"type":"proc_nonstationary","multiResolutionGrid":true,"cov":)"+base+
        (a.empty() ? "" : ",\"var\":"+a)+(s.empty() ? "" : ",\"ls\":"+s)+"}";
}
std::shared_ptr<ClassicGlobalMedium> medium(const std::string &cov, const std::string &mean = Mean,
                                          const std::string &material = R"({"type":"mirror","albedo":1})") {
    const std::string json = R"({"type":"classic_global","gaussian_process":{"type":"standard","mean":)"+
        mean+",\"covariance\":"+cov+"},\"material\":"+material+"}";
    JsonDocument doc(Path("nonstationary-test.json"),json);
    Scene scene;
    auto m = std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(doc));
    m->loadResources(); m->prepareForRender();
    return m;
}
std::shared_ptr<ProceduralNonstationaryCovariance> kernel(const std::string &cov) {
    JsonDocument doc(Path("nonstationary-kernel-test.json"),cov);
    Scene scene;
    auto k = std::dynamic_pointer_cast<ProceduralNonstationaryCovariance>(scene.fetchCovarianceFunction(doc));
    k->loadResources();
    return k;
}
double value(const CovarianceFunction &k, Vec3d a, Vec3d b) {
    return k(Derivative::None,Derivative::None,a,b,Vec3d(0.0),Vec3d(0.0));
}
std::string serialize(const JsonSerializable &object) {
    rapidjson::Document doc;
    auto json = object.toJson(doc.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    return buffer.GetString();
}
void parameters() {
    Scene scene;
    int axis = 0;
    for (const char *name : {"left_right","bottom_top","front_back"}) {
        const std::string json = std::string(R"({"type":"noise","noise":")")+name+
            R"(","min":0.2,"max":1.3,"start":-0.7,"end":0.9})";
        JsonDocument doc(Path("parameter-test.json"),json);
        auto a = scene.fetchProceduralScalar(doc);
        auto s = scene.fetchProceduralVector(doc);
        ParameterBounds bounds; VectorParameterBounds sb;
        require(a->parameterBounds(bounds) && s->parameterBounds(sb),"axis ramp needs certified bounds");
        for (double coordinate : {-1.2,-0.3,0.0,0.6,1.4}) {
            Vec3d p(0.0); p[axis] = coordinate;
            const auto j = a->parameterJet(p);
            const auto v = s->parameterJet(p);
            Vec3d d(0.0); d[axis] = 1e-5;
            close(j.gradient[axis],((*a)(p+d)-(*a)(p-d))/2e-5,2e-9,"analytic parameter gradient");
            require(j.value >= bounds.minimum && j.value <= bounds.maximum &&
                    j.gradient.length() <= bounds.gradientNorm,"parameter exceeds global bound");
            for (int k=0;k<3;++k) {
                close(v[k].value,j.value,0.0,"broadcast scalar ls");
                close((v[k].gradient-j.gradient).length(),0.0,0.0,"broadcast ls gradient");
            }
        }
        ++axis;
    }
}
void statistics() {
    for (const auto &cov : {covariance(RampA,RampS),covariance(RampA,""),covariance("",RampS),
                           covariance(RampA,R"({"type":"constant","val":[0.6,1.1,1.7]})")}) {
        auto m = medium(cov); auto k = kernel(cov);
        const Vec3d p(0.12,0.17,0.21);
        const auto s = m->pointStatistics(p);
        const double h=2e-4, v=value(*k,p,p);
        Eigen::Vector3d b; Eigen::Matrix3d c;
        // Deliberately differentiate the two-point *value*, not the existing
        // nonstationary AD path (which freezes the procedural parameters).
        for (int i=0;i<3;++i) {
            Vec3d di(0.0); di[i]=h;
            b[i]=(value(*k,p+di,p)-value(*k,p-di,p))/(2*h);
            for (int j=0;j<3;++j) {
                Vec3d dj(0.0); dj[j]=h;
                c(i,j)=(value(*k,p+di,p+dj)-value(*k,p+di,p-dj)-
                        value(*k,p-di,p+dj)+value(*k,p-di,p-dj))/(4*h*h);
            }
        }
        close(s.sigma*s.sigma,v,2e-12,"var is amplitude, not variance");
        const Eigen::Vector3d expectedMean = eigen(m->meanGradient(p))-m->meanValue(p)*b/v;
        const Eigen::Matrix3d expectedCov = c-b*b.transpose()/v;
        require((s.sigma*s.gradientMean-expectedMean).norm()<2e-6,"conditional gradient mean vs kernel derivatives");
        require((s.sigma*s.sigma*s.gradientCovariance-expectedCov).norm()<3e-6,"conditional covariance vs kernel derivatives");
        JsonDocument doc(Path("nonstationary-roundtrip.json"),serialize(*m));
        require(doc["gaussian_process"]["covariance"]["multiResolutionGrid"].cast<bool>(),"preserve multiResolutionGrid flag");
        Scene scene;
        auto copy=std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(doc));
        copy->loadResources(); copy->prepareForRender();
        close(copy->extinction(p,Vec3d(0.0,0.0,-1.0)),m->extinction(p,Vec3d(0.0,0.0,-1.0)),0,"nonstationary round trip");
    }
    auto stationary=medium(R"({"type":"squared_exponential","sigma":0.5,"lengthScale":0.75})");
    auto constant=medium(covariance(R"({"type":"constant","val":0.5})",R"({"type":"constant","val":[0.75,0.75,0.75]})",
                                   R"({"type":"squared_exponential","sigma":1,"lengthScale":1})"));
    auto omitted=medium(covariance("","",R"({"type":"squared_exponential","sigma":0.5,"lengthScale":0.75})"));
    for (const Vec3d p : {Vec3d(0.0),Vec3d(0.2,-0.1,0.4)}) {
        for (const Vec3d w : {Vec3d(0.0,0.0,-1.0),Vec3d(1.0,0.0,1.0)}) {
            close(constant->extinction(p,w),stationary->extinction(p,w),1e-12,"constant fields recover stationary extinction");
            close(omitted->extinction(p,w),stationary->extinction(p,w),1e-12,"omitted fields default to one");
        }
        const Eigen::Vector3d wi(0,0,-1), wo=Eigen::Vector3d(0.2,0.3,1).normalized();
        close(constant->phasePdf(p,wi,wo),stationary->phasePdf(p,wi,wo),1e-12,"stationary phase limit");
    }
    // Non-binary base parameters must retain the stationary implementation's
    // arithmetic too, not only the exactly representable fixture above.
    close(medium(covariance("",""))->extinction(Vec3d(0.2),Vec3d(0.0,0.0,-1.0)),
          medium(Base)->extinction(Vec3d(0.2),Vec3d(0.0,0.0,-1.0)),1e-12,"stationary base parameter precision");
}
void boundsAndFlights() {
    const std::string cov=covariance(
        R"({"type":"noise","noise":"front_back","min":0.65,"max":1.2,"start":-0.8,"end":0.8})",
        R"({"type":"noise","noise":"front_back","min":0.5,"max":1,"start":-0.8,"end":0.8})");
    auto m=medium(cov); auto k=kernel(cov);
    ClassicGlobal::NonstationaryKernel bound(*k);
    const double gradientBound=m->meanGradient(Vec3d(0.0)).length();
    for (double z : {-1.4,-0.78,-0.3,0.0,0.4,0.79,1.3}) {
        const Vec3d center(0,0,z); const double radius=0.19, mean=m->meanValue(center);
        const double majorant=bound.majorant(center,radius,mean-gradientBound*radius,mean+gradientBound*radius,gradientBound);
        for (int i=0;i<=40;++i) {
            const Vec3d p=center+Vec3d(0,0,-radius+2*radius*i/40);
            for (const Vec3d w : {Vec3d(0.0,0.0,-1.0),Vec3d(0.0,0.0,1.0),Vec3d(1.0,0.0,0.0),Vec3d(1.0,2.0,-3.0)})
                require(m->extinction(p,w)<=majorant,"nonstationary extinction exceeds conservative majorant");
        }
    }
    // Independent quadrature of the law obtained from two-point derivatives.
    const Ray ray(Vec3f(0.0f,0.0f,1.0f),Vec3f(0.0f,0.0f,-2.0f),0.125f,0.875f);
    double opticalDepth=0.0;
    constexpr int steps=1024;
    for (int i=0;i<steps;++i) {
        const Vec3d p(0,0,0.75-1.5*(i+0.5)/steps), d(0,0,-2e-4);
        const double h=2e-4,v=value(*k,p,p),mvalue=m->meanValue(p);
        const double b=(value(*k,p+d,p)-value(*k,p-d,p))/(2*h);
        const double c=(value(*k,p+d,p+d)-value(*k,p+d,p-d)-value(*k,p-d,p+d)+value(*k,p-d,p-d))/(4*h*h);
        const double mu=-m->meanGradient(p).z()-mvalue*b/v, sd=std::sqrt(c-b*b/v), u=mvalue/std::sqrt(v);
        const double area=sd*std::exp(-0.5*mu*mu/(sd*sd))/std::sqrt(2*Pi)-mu*0.5*std::erfc(mu/(sd*std::sqrt(2.0)));
        const double rho=std::exp(-0.5*u*u)/(std::sqrt(2*Pi*v)*0.5*std::erfc(-u/std::sqrt(2.0)));
        opticalDepth+=rho*area*1.5/steps;
    }
    const double expected=std::exp(-opticalDepth);
    IndependentPathSampler rng(41837,false);
    constexpr int count=30000;
    double transmission=0.0; int exits=0;
    for (int i=0;i<count;++i) {
        rng.startPath(0,i);
        Medium::MediumState state{}; state.reset(); MediumSample sample{};
        require(m->sampleDistance(rng,ray,state,sample),"nonstationary distance sample");
        require(sample.t>=ray.nearT() && sample.t<=ray.farT(),"nonstationary event escaped segment");
        exits+=sample.exited;
        transmission+=m->transmittance(rng,ray,false,true,&state).x();
    }
    close(double(exits)/count,expected,0.009,"nonstationary flight survival vs two-point quadrature");
    close(transmission/count,expected,0.009,"nonstationary ratio tracking vs two-point quadrature");
    std::cout<<"nonstationary survival="<<double(exits)/count<<", transmission="<<transmission/count<<", expected="<<expected<<'\n';
}
void phase() {
    auto m=medium(covariance(RampA,RampS));
    IndependentPathSampler rng(13789,false);
    for (Vec3f position : {Vec3f(-0.2f,0.1f,0.2f),Vec3f(0.4f,0.3f,-0.15f)}) {
        MediumSample event{}; event.p=position;
        const auto *phase=m->phaseFunction(position);
        const Vec3f wi(0.0f,0.0f,-1.0f);
        Vec3d integrated(0.0),sampled(0.0); double mass=0.0;
        constexpr int nz=100,np=160,count=20000;
        for (int z=0;z<nz;++z) for (int p=0;p<np;++p) {
            const double c=-1+2.0*(z+0.5)/nz, angle=2*Pi*(p+0.5)/np, s=std::sqrt(1-c*c);
            const Vec3f wo(float(s*std::cos(angle)),float(s*std::sin(angle)),float(c));
            const double probability=phase->pdf(wi,wo,event)*4*Pi/(nz*np);
            mass+=probability; integrated+=probability*Vec3d(wo);
        }
        for (int i=0;i<count;++i) {
            rng.startPath(1,i); PhaseSample sample{};
            require(phase->sample(rng,wi,event,sample),"nonstationary VNDF sample");
            sampled+=Vec3d(sample.w);
            close(sample.pdf,phase->pdf(wi,sample.w,event),0,"nonstationary sample PDF");
            close(phase->eval(wi,sample.w,event).x()/sample.pdf,1.0,2e-6,"nonstationary white furnace weight");
            close(sample.weight.x(),1.0,0,"nonstationary mirror weight");
        }
        close(mass,1.0,0.002,"nonstationary phase normalization");
        require((sampled/count-integrated).length()<0.022,"nonstationary VNDF moment vs integrated PDF");
    }
    auto diffuse=medium(covariance(RampA,RampS),Mean,R"({"type":"lambert","albedo":0.7})");
    Medium::MediumState state{}; state.reset(); MediumSample event{};
    const Ray ray(Vec3f(0.1f,0.2f,0.7f),Vec3f(0.0f,0.0f,-1.0f),0,4);
    for (int i=0;i<100;++i) {
        rng.startPath(2,i); state.reset();
        diffuse->sampleDistance(rng,ray,state,event);
        if (!event.exited) break;
    }
    require(!event.exited && event.aniso.z()>0,"Lambert collision observes local visible normal");
    const auto normal=event.aniso;
    for (int i=0;i<500;++i) {
        rng.startPath(3,i); PhaseSample sample{};
        require(event.phase->sample(rng,ray.dir(),event,sample),"nonstationary Lambert sample");
        require(Vec3d(sample.w).dot(normal)>0,"Lambert continuation hemisphere");
        close(event.phase->eval(ray.dir(),sample.w,event).x()/sample.pdf,0.7,2e-6,"Lambert NEE weight");
        require(event.aniso==normal,"NEE and continuation share observed normal");
    }
}
void rejection() {
    for (const auto &cov : {
        covariance(R"({"type":"constant","val":0})",RampS),
        covariance(RampA,R"({"type":"constant","val":[1,-1,1]})"),
        covariance(R"({"type":"noise","noise":"bottom_top","min":0,"max":1})",RampS),
        covariance(R"({"type":"noise","noise":"rust","min":0.5,"max":1})",RampS),
        covariance(RampA,R"({"type":"noise","noise":"bottom_top","min":1,"max":2,"start":1,"end":1})"),
        covariance(RampA,RampS,R"({"type":"squared_exponential","sigma":1,"lengthScale":1,"aniso":[1,2,1]})"),
        std::string(R"({"type":"proc_nonstationary","cov":{"type":"squared_exponential"},"aniso":{"type":"constant","val":1}})"),
        std::string(R"({"type":"proc_nonstationary"})")}) {
        bool failed=false;
        try { medium(cov); } catch (const std::runtime_error &) { failed=true; }
        require(failed,"unsupported or degenerate covariance must fail before rendering");
    }
}
}
int main(int argc, char **argv) {
    try {
        if (argc == 2) {
            int w,h;
            auto pixels=ImageIO::loadHdr(Path(argv[1]),TexelConversion::REQUEST_RGB,w,h);
            require(bool(pixels) && w>0 && h>0,"rendered EXR must be readable");
            double sum=0.0;
            for (int i=0;i<w*h*3;++i) {
                require(std::isfinite(pixels[i]) && pixels[i]>=0,"rendered pixels must be finite and nonnegative");
                sum+=pixels[i];
            }
            require(sum>0,"rendered image must not be black");
            std::cout<<w<<'x'<<h<<" finite image, mean="<<sum/(w*h*3)<<'\n';
            return 0;
        }
        parameters(); statistics(); boundsAndFlights(); phase(); rejection();
        std::cout<<"classic_global nonstationary tests passed\n"; return 0;
    } catch (const std::exception &e) {
        std::cerr<<"classic_global nonstationary test failed: "<<e.what()<<'\n'; return 1;
    }
}
