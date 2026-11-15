#include "math/AdaptiveGpis.hpp"
#include "media/AdaptiveFunctionSpaceGaussianProcessMedium.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "samplerecords/SpecularGpisEvent.hpp"
#include <Eigen/Dense>
#include <boost/math/special_functions/erf.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <random>
#include <iostream>
#include <stdexcept>
#include <array>

using namespace Tungsten;
namespace ag=Tungsten::AdaptiveGpis;
namespace {
void require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
void close(double a,double b,double tolerance,const char *message) {
    if (!(std::abs(a-b)<=tolerance)) {
        std::cerr<<message<<": "<<a<<" vs "<<b<<" (tol "<<tolerance<<")\n";
        throw std::runtime_error(message);
    }
}
ag::Observation point(double t) { return {{t,0,0},Eigen::Vector3d::Zero(),false}; }

void gaussian()
{
    const std::vector<ag::Observation> o={point(0),{{0,0,0},{1,0,0},true},
        {{0,0,0},{0,1,0},true},point(.47),{{.9,.2,.1},{0,0,1},true},point(1.2)};
    SquaredExponentialCovariance gp(1,1);
    Eigen::MatrixXd covariance(o.size(),o.size());
    for(int i=0;i<int(o.size());++i) for(int j=0;j<int(o.size());++j) {
        covariance(i,j)=gp(o[i].derivative?Derivative::First:Derivative::None,
            o[j].derivative?Derivative::First:Derivative::None,
            Vec3d(o[i].position.x(),o[i].position.y(),o[i].position.z()),
            Vec3d(o[j].position.x(),o[j].position.y(),o[j].position.z()),
            Vec3d(o[i].direction.x(),o[i].direction.y(),o[i].direction.z()),
            Vec3d(o[j].direction.x(),o[j].direction.y(),o[j].direction.z()));
    }
    ag::OnlineSampler conditioned(1);
    conditioned.condition(o[0],.3); conditioned.condition(o[1],-.2); conditioned.condition(o[2],.4);
    // A nearly determined birth query must stay finite in double without
    // changing the following conditional Gaussian. Check its analytic mean.
    const auto birth=conditioned.predict(point(1e-5));
    close(birth.mean,std::exp(-.5e-10)*(.3-.2e-5),1e-12,"near-birth conditional mean");
    require(birth.variance>=0 && birth.variance<1e-12,"near-birth variance in double");
    Eigen::Vector3d known(.3,-.2,.4);
    for(int i=3;i<int(o.size());++i) {
        auto p=conditioned.predict(o[i]);
        Eigen::Matrix3d a=covariance.topLeftCorner<3,3>();
        Eigen::Vector3d b=covariance.block<3,1>(0,i);
        close(p.mean,b.dot(a.ldlt().solve(known)),1e-12,"conditional mean vs dense LDLT");
        close(p.variance,covariance(i,i)-b.dot(a.ldlt().solve(b)),1e-12,"conditional variance vs dense LDLT");
    }
    const auto joint=conditioned.predictJoint({o[3],o[4],o[5]});
    const Eigen::MatrixXd cross=covariance.bottomLeftCorner(3,3);
    const Eigen::MatrixXd expectedCov=covariance.bottomRightCorner(3,3)-
        cross*covariance.topLeftCorner(3,3).ldlt().solve(cross.transpose());
    require((joint.covariance-expectedCov).cwiseAbs().maxCoeff()<2e-12,"joint posterior covariance vs dense LDLT");
    require((joint.mean-cross*covariance.topLeftCorner(3,3).ldlt().solve(known)).norm()<2e-12,
        "joint posterior mean vs dense LDLT");
    std::mt19937_64 random(1307); std::normal_distribution<double> normal;
    constexpr int trials=14000;
    Eigen::VectorXd sum=Eigen::VectorXd::Zero(o.size());
    Eigen::MatrixXd square=Eigen::MatrixXd::Zero(o.size(),o.size());
    for(int trial=0;trial<trials;++trial) {
        ag::OnlineSampler sampler(1);
        Eigen::VectorXd v(o.size());
        for(int i=0;i<int(o.size());++i) {
            // Half the trials exercise the double rank-revealing path. A
            // prediction alone adds no observation and must not change the law.
            if (i==3 && trial%2==0) sampler.predict(point(1e-5));
            v[i]=sampler.sample(o[i],normal(random));
        }
        sum+=v; square.noalias()+=v*v.transpose();
    }
    require((sum/trials).cwiseAbs().maxCoeff()<.035,"online sample mean");
    require((square/trials-covariance).cwiseAbs().maxCoeff()<.045,"online joint covariance");
    ag::OnlineSampler dense(1);
    double first=dense.sample(point(0),normal(random));
    close(dense.sample(point(0),normal(random)),first,1e-12,"duplicate observation reuses value");
    for(int i=1;i<=250;++i) require(std::isfinite(dense.sample(point(i*.003),normal(random))),"dense samples finite");
    require(dense.redundant()>0,"near-dependent observations counted");
    bool threw=false;
    try { dense.condition(point(0),first+1); } catch(const std::exception &) { threw=true; }
    require(threw,"inconsistent observations must fail");
    ag::OnlineSampler limit(1,1e-10,4);
    threw=false;
    try { for(int i=0;i<5;++i) limit.sample(point(i*2),normal(random)); }
    catch(const std::exception &) { threw=true; }
    require(threw,"rank limit is not a miss / pruning");
    std::cout<<"Gaussian checks passed; dense rank="<<dense.rank()<<", redundant="<<dense.redundant()<<'\n';
}

void bounds()
{
    for(double ratio:{.01,1.,1000.}) {
        ag::Bounds b(1,1,ratio,.001);
        double c1=0;
        for(int n=1;n<100;++n) c1+=std::ldexp(std::sqrt(2*n*std::log(2.0)),-n-1);
        close(b.c,c1+std::sqrt(-2*std::log(.0005)),2e-6,"local coefficient");
        // Independent log-grid minimization of the supplement's expression.
        double best=1e99;
        for(int i=0;i<12000;++i) {
            const double k=std::exp(-22.+i*.002);
            const double a=k*.0005*std::sqrt(2/3.14159265358979323846);
            if(a>=1) continue;
            best=std::min(best,k*(1+c1*ratio)+std::sqrt(2.0)*boost::math::erfc_inv(a));
        }
        close(b.q,best,3e-5,"long-range coefficient");
        require(b.step(-100,.01,.001)<.01,"opposite mean/sample signs cannot take abs(mean) step");
    }
    ag::Bounds constant(0,1,1,.001);
    require(std::isfinite(constant.step(0,1,.001)),"constant mean bound");
    ag::Settings settings;
    settings.minimumStep=.1; settings.rootTolerance=1e-7;
    auto hit=ag::intersect(0,2,1,[](double t){return 1-t;},[](double t){return 1-t;},
                          ag::Bounds(1,.01,.01,.001),settings);
    require(hit.hit,"deterministic crossing found"); close(hit.t,1,1e-7,"root refinement");
    auto miss=ag::intersect(0,1,1,[](double){return 1.;},[](double){return 1.;},constant,settings);
    require(!miss.hit && miss.t==1,"finite endpoint");
}

void firstHit()
{
    constexpr int n=161,trials=2500;
    constexpr double sigma=.2,length=.25,end=1.6;
    Eigen::MatrixXd k(n,n);
    for(int i=0;i<n;++i) for(int j=0;j<n;++j) {
        const double d=end*(i-j)/(n-1)/length;
        k(i,j)=sigma*sigma*std::exp(-.5*d*d);
    }
    // Independent dense eigensampling, including all grid points at once.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(k);
    require(eig.info()==Eigen::Success && eig.eigenvalues().minCoeff()>-1e-12,"dense oracle PSD");
    const Eigen::MatrixXd root=eig.eigenvectors()*eig.eigenvalues().cwiseMax(0).cwiseSqrt().asDiagonal();
    std::mt19937_64 random(91271); std::normal_distribution<double> normal;
    const std::array<double,5> cut{.25,.45,.6,.75,.9};
    std::array<int,5> expected{},actual{};
    std::size_t queries=0;
    ag::Settings settings; settings.minimumStep=.01; settings.rootTolerance=1e-5;
    const ag::Bounds range(1,sigma,sigma/length,settings.probabilityTolerance);
    for(int trial=0;trial<trials;++trial) {
        Eigen::VectorXd z(n);
        for(int i=0;i<n;++i) z[i]=normal(random);
        Eigen::VectorXd f=root*z;
        for(int i=0;i<n;++i) f[i]+=.6-end*i/(n-1);
        double td=end;
        if(f[0]>0) for(int i=1;i<n;++i) if(f[i]<=0) {
            td=end*(i-1+f[i-1]/(f[i-1]-f[i]))/(n-1); break;
        }
        ag::OnlineSampler sampler(length);
        auto value=[&](double t){return .6-t+sigma*sampler.sample(point(t),normal(random));};
        const double initial=value(0);
        const double ta=initial>0 ? ag::intersect(0,end,initial,value,
            [](double t){return .6-t;},range,settings).t : end;
        queries+=sampler.queries();
        for(int i=0;i<5;++i) { expected[i]+=td<=cut[i]; actual[i]+=ta<=cut[i]; }
    }
    for(int i=0;i<5;++i) {
        std::cout<<"CDF "<<cut[i]<<": dense="<<double(expected[i])/trials<<", adaptive="<<double(actual[i])/trials<<'\n';
        close(double(actual[i])/trials,double(expected[i])/trials,.045,"first-hit CDF vs dense GP");
    }
    require(double(queries)/trials < n*.65,"adaptive sampling reduces queries");
    std::cout<<"Mean adaptive queries "<<double(queries)/trials<<" vs dense "<<n<<'\n';
}

std::shared_ptr<AdaptiveFunctionSpaceGaussianProcessMedium> medium(const std::string &extra="",const std::string &material="lambert")
{
    std::string json=R"({"type":"adaptive_function_space_gaussian_process",
        "name":"gp","correlation_context":"renewal+","sigma_s":1,"sigma_a":0,"density":1,
        "gaussian_process":{"type":"standard","mean":{"type":"linear","direction":[0,0,1]},
        "covariance":{"type":"squared_exponential","sigma":0.12,"lengthScale":0.25}},
        "phase_function":{"type":"brdf","bsdf":{"type":"lambert","albedo":0.7}},
        "adaptive":{"minimum_step":0.005,"root_tolerance":0.00001})"+extra+"}";
    json.replace(json.find("\"lambert\""),9,"\""+material+"\"");
    JsonDocument doc(Path("adaptive-test.json"),json); Scene scene;
    auto m=std::dynamic_pointer_cast<AdaptiveFunctionSpaceGaussianProcessMedium>(scene.fetchMedium(doc));
    require(bool(m),"medium factory"); m->loadResources(); m->prepareForRender(); return m;
}

void misChecks()
{
    constexpr double pi=3.14159265358979323846;
    Eigen::Matrix3d covariance;
    covariance<<.25,.04,.03, .04,.16,-.015, .03,-.015,.09;
    const Eigen::Vector3d mean(.2,-.1,-1),travel(0,0,1);
    ag::GradientPlane plane(Eigen::Matrix3d::Identity(),mean,covariance,.3);
    require(plane.continuous(),"nondegenerate gradient plane");
    const auto density=[&](const Eigen::Vector3d &outgoing) {
        return plane.reflectionPdf(outgoing-travel);
    };
    const auto radiance=[](const Eigen::Vector3d &w) { return 1+.7*w.x()+.4*w.z()*w.z(); };
    double integral=0,reference=0;
    constexpr int nz=600,np=900;
    for(int iz=0;iz<nz;++iz) for(int ip=0;ip<np;++ip) {
        const double z=-1+(iz+.5)*2/nz,phi=(ip+.5)*2*pi/np;
        const Eigen::Vector3d wo(std::sqrt(1-z*z)*std::cos(phi),std::sqrt(1-z*z)*std::sin(phi),z);
        const double mass=density(wo)*4*pi/(nz*np);
        integral+=mass; reference+=mass*radiance(wo);
    }
    close(integral,1,2e-5,"reflected-direction PDF integrates to one");
    std::mt19937_64 random(58473); std::normal_distribution<double> gaussian;
    std::uniform_real_distribution<double> uniform;
    Eigen::Vector2d sum=Eigen::Vector2d::Zero();
    Eigen::Matrix2d second=Eigen::Matrix2d::Zero();
    double forward=0,nee=0,mis=0;
    constexpr int trials=100000;
    const double lightPdf=1/(4*pi);
    for(int i=0;i<trials;++i) {
        const Eigen::Vector3d g=plane.sample({gaussian(random),gaussian(random)}),n=g.normalized();
        close(g.z(),-.91,1e-12,"fixed longitudinal derivative");
        require((plane.gradientForNormal(n)-g).norm()<1e-12,"normal recovers full gradient");
        sum+=g.head<2>(); second+=g.head<2>()*g.head<2>().transpose();
        const Eigen::Vector3d wo=travel-2*travel.dot(n)*n;
        const double p=density(wo),l=radiance(wo);
        forward+=l; mis+=l*p*p/(p*p+lightPdf*lightPdf);
        const double z=2*uniform(random)-1,phi=2*pi*uniform(random);
        const Eigen::Vector3d light(std::sqrt(1-z*z)*std::cos(phi),std::sqrt(1-z*z)*std::sin(phi),z);
        const double q=density(light),term=radiance(light)*q/lightPdf;
        nee+=term; mis+=term*lightPdf*lightPdf/(q*q+lightPdf*lightPdf);
    }
    const Eigen::Vector2d expectedMean=mean.head<2>()+covariance.topRightCorner<2,1>();
    const Eigen::Matrix2d expectedCov=covariance.topLeftCorner<2,2>()-
        covariance.topRightCorner<2,1>()*covariance.bottomLeftCorner<1,2>()/.09;
    require((sum/trials-expectedMean).norm()<.006,"transverse sample mean");
    require((second/trials-expectedMean*expectedMean.transpose()-expectedCov).norm()<.007,
        "transverse sample covariance");
    close(forward/trials,reference,.012,"forward estimator vs quadrature");
    close(nee/trials,reference,.02,"NEE estimator vs quadrature");
    close(mis/trials,reference,.012,"MIS estimator vs quadrature");
    ag::GradientPlane degenerate(Eigen::Matrix3d::Identity(),mean,Eigen::Matrix3d::Zero(),0);
    require(!degenerate.continuous(),"degenerate law keeps delta reflection");

    auto m=medium(",\"1D_sampling\":true,\"1D_sampling_scheme\":\"mis\"","mirror");
    IndependentPathSampler rng(711,false); rng.startPath(0,0);
    Medium::MediumState state{}; state.reset(); MediumSample sample{};
    const Vec3f incoming(0.f,0.f,-1.f);
    require(m->sampleDistance(rng,Ray(Vec3f(0.f,0.f,1.f),incoming,0,2),state,sample) && !sample.exited,
        "MIS medium collision");
    require(sample.sparseConv1DSamplingScheme==SparseConv1DSamplingScheme::MIS &&
        dynamic_cast<const SpecularGpisEvent *>(sample.eventContext.get()),"MIS event context");
    PhaseSample phase;
    require(sample.phase->sample(rng,incoming,sample,phase),"MIS phase sample");
    require(phase.pdf>0 && std::isfinite(phase.pdf),"sampled reflection PDF finite");
    close(sample.phase->pdf(incoming,phase.w,sample),phase.pdf,2e-5*phase.pdf,"sample and evaluated PDF agree");
    const Vec3f light=Vec3f(.2f,.1f,1.f).normalized();
    const double pdf=sample.phase->pdf(incoming,light,sample);
    require(pdf>0,"light direction has density");
    close(sample.phase->eval(incoming,light,sample).x()/pdf,.7,3e-5,"mirror evaluation includes density");
    const Vec3d connection(sample.phase->evalGrad(incoming,light,sample));
    close(connection.dot(Vec3d(incoming)),sample.aniso.dot(Vec3d(incoming)),2e-6,
        "light gradient retains the sampled longitudinal derivative");
    const auto parent=state.pathContext;
    auto shadow=state; shadow.lastAniso=connection;
    const auto tr=m->transmittance(rng,Ray(sample.p,light,0,.1f),false,false,&shadow);
    require(tr.x()==0 || tr.x()==1,"MIS shadow visibility");
    require(state.pathContext==parent && shadow.pathContext!=parent,"light branch preserves parent history");
    auto flipped=state; flipped.lastAniso=-sample.aniso;
    close(m->transmittance(rng,Ray(sample.p,Vec3f(sample.aniso.normalized()),0,.1f),false,false,&flipped).x(),
        0,0,"Renewal+ birth uses the branch gradient, not the parent's gradient");
    rapidjson::Document doc; auto json=m->toJson(doc.GetAllocator());
    require(json["1D_sampling"].GetBool() && std::string(json["1D_sampling_scheme"].GetString())=="mis",
        "MIS JSON round trip fields");
    auto conductor=medium(",\"1D_sampling\":true,\"1D_sampling_scheme\":\"mis\"","conductor");
    state.reset(); rng.startPath(0,0);
    require(conductor->sampleDistance(rng,Ray(Vec3f(0.f,0.f,1.f),incoming,0,2),state,sample) && !sample.exited,
        "conductor MIS collision");
    require(sample.phase->sample(rng,incoming,sample,phase),"conductor MIS sample");
    const double conductorPdf=sample.phase->pdf(incoming,phase.w,sample);
    const Vec3f conductorEval=sample.phase->eval(incoming,phase.w,sample);
    for(int k=0;k<3;++k) close(conductorEval[k]/conductorPdf,phase.weight[k],3e-5,
        "conductor Fresnel and PDF use the same reflection normal");
    std::cout<<"MIS: PDF integral="<<integral<<", reference="<<reference<<", forward="<<forward/trials
        <<", NEE="<<nee/trials<<", MIS="<<mis/trials<<'\n';
}

void mediumChecks()
{
    std::cout<<"medium: create"<<std::endl;
    auto m=medium();
    IndependentPathSampler rng(711,false);
    Medium::MediumState state{}; state.reset(); rng.startPath(0,0);
    MediumSample sample{};
    const Ray ray(Vec3f(0.f,0.f,1.f),Vec3f(0.f,0.f,-2.f),.1f,.9f);
    require(m->sampleDistance(rng,ray,state,sample) && !sample.exited,"plane collision");
    std::cout<<"medium: collision"<<std::endl;
    require(sample.t>=ray.nearT() && sample.t<=ray.farT(),"nonunit ray parameter");
    close(sample.p.z(),1-2*sample.t,1e-7,"nonunit ray position");
    close(sample.weight.x(),1,1e-6,"unit scattering weight");
    require(sample.aniso.z()>0,"oriented collision gradient");
    const auto parent=state.pathContext;
    auto shadow=state;
    const Vec3f outward(sample.aniso.normalized());
    const Ray escape(sample.p,outward,0,1);
    const Vec3f tr=m->transmittance(rng,escape,false,false,&shadow);
    require(tr.x()==0 || tr.x()==1,"binary visibility");
    require(state.pathContext==parent && shadow.pathContext!=parent,"shadow copies do not mutate main state");
    auto invalid=state;
    close(m->transmittance(rng,Ray(sample.p,-outward,0,1),false,false,&invalid).x(),0,0,"inward birth blocked");
    auto branch1=state,branch2=state;
    IndependentPathSampler a(91,false),b(91,false); a.startPath(0,0); b.startPath(0,0);
    MediumSample s1{},s2{};
    const bool ok1=m->sampleDistance(a,escape,branch1,s1);
    // Reset a shared sibling; main continuation must remain unchanged.
    shadow=state; shadow.reset();
    const bool ok2=m->sampleDistance(b,escape,branch2,s2);
    require(ok1==ok2 && s1.t==s2.t && s1.exited==s2.exited,"branch reset preserves realization");
    std::cout<<"medium: branches"<<std::endl;
    rapidjson::Document json; auto value=m->toJson(json.GetAllocator());
    rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer); value.Accept(writer);
    JsonDocument doc(Path("adaptive-roundtrip.json"),buffer.GetString()); Scene scene;
    auto copy=std::dynamic_pointer_cast<AdaptiveFunctionSpaceGaussianProcessMedium>(scene.fetchMedium(doc));
    require(bool(copy),"round-trip medium type");
    copy->loadResources(); copy->prepareForRender();
    close(copy->settings().rootTolerance,m->settings().rootTolerance,0,"JSON round trip");
    std::cout<<"medium: roundtrip"<<std::endl;
    bool threw=false;
    try { m->pdf(rng,ray,false,false); } catch(const std::exception &) { threw=true; }
    require(threw,"unsupported distance PDF fails explicitly");
    // Splitting a transparent straight segment preserves the sampled prefix.
    Medium::MediumState split{}; split.reset();
    MediumSample endpoint{};
    require(m->sampleDistance(rng,Ray(Vec3f(0.f,0.f,2.f),Vec3f(0.f,0.f,-1.f),0,.2f),split,endpoint) && endpoint.exited,
        "transparent prefix exits");
    require(bool(split.pathContext) && !split.firstScatter,"prefix history retained");
    require(m->sampleDistance(rng,Ray(endpoint.p,Vec3f(0.f,0.f,-1.f),0,3),split,endpoint) && !endpoint.exited,
        "continued prefix collides");
}
}
int main(int argc,char **argv)
{
    try {
        const std::string mode=argc>1?argv[1]:"all";
        if(mode=="gaussian" || mode=="all") gaussian();
        if(mode=="bounds" || mode=="all") bounds();
        if(mode=="first_hit" || mode=="all") firstHit();
        if(mode=="medium" || mode=="all") mediumChecks();
        if(mode=="mis" || mode=="all") misChecks();
        std::cout<<"adaptive_gpis "<<mode<<" passed\n"; return 0;
    } catch(const std::exception &e) { std::cerr<<e.what()<<'\n'; return 1; }
}
