#include "math/SparseHistoryGaussian.hpp"
#include "math/SparseHistoryFlight.hpp"
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "media/GpisSdfGrid.hpp"
#include "media/SparseHistoryGpisMedium.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "phasefunctions/PhaseFunction.hpp"
#include "samplerecords/MediumSample.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

namespace sh = Tungsten::SparseHistoryGpis;
namespace cg = Tungsten::ConditionalGpis;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
namespace {
void require(bool ok, const char *message) { if (!ok) throw std::runtime_error(message); }
void close(double actual, double expected, double tolerance, const char *message) {
    if (!std::isfinite(actual) || std::abs(actual-expected)>tolerance) {
        std::cerr.precision(17);
        std::cerr<<message<<": "<<actual<<" vs "<<expected<<" (tol "<<tolerance<<")\n";
        throw std::runtime_error(message);
    }
}
cg::GPSSField makeField(std::shared_ptr<const cg::MeanField> mean, double sigma=1, double ell=1) {
    return {std::move(mean),cg::CovarianceKernel(sigma,M::Identity()/(ell*ell)),
        {V::Constant(-100),V::Constant(100)}};
}
class ProfileMean final : public cg::MeanField {
public:
    const char *typeName() const override { return "sparse-history-profile-test"; }
    cg::MeanJet evaluate(const V &p) const override {
        const double t=p.z();
        if(t<0.5) return {t,V(0,0,1)};
        if(t<1.5) return {0.5-1.3*(t-0.5),V(0,0,-1.3)};
        if(t<2.25) return {-0.8+(1.25/0.75)*(t-1.5),V(0,0,1.25/0.75)};
        return {0.45-0.5*(t-2.25),V(0,0,-0.5)};
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override {
        double lo=std::min(evaluate(b.minimum).value,evaluate(b.maximum).value);
        if(b.minimum.z()<=1.5 && b.maximum.z()>=1.5) lo=std::min(lo,-0.8);
        return {lo,1.25/0.75,true};
    }
    void appendRayBreakpoints(const V &o,const V &w,double lo,double hi,std::vector<double> &out) const override {
        if(w.z()==0) return;
        for(double knot:{0.5,1.5,2.25}) {
            double t=(knot-o.z())/w.z(); if(t>lo && t<hi) out.push_back(t);
        }
    }
};
void gaussianChecks() {
    sh::ProbabilitySettings settings;
    settings.absoluteTolerance=2e-10; settings.relativeTolerance=5e-5;
    settings.maxSamples=32768;
    for(int n=1;n<=4;++n) {
        Eigen::VectorXd mean(n); double expected=1;
        for(int i=0;i<n;++i) { mean[i]=0.3*i-0.6; expected*=cg::normalCdf(mean[i]); }
        close(sh::positiveOrthant(mean,Eigen::MatrixXd::Identity(n,n),settings),expected,2e-7,
              "independent Gaussian orthant");
    }
    for(double rho:{-0.95,-0.3,0.0,0.5,0.999999}) {
        Eigen::Matrix2d c; c<<1,rho,rho,1;
        close(sh::positiveOrthant(Eigen::Vector2d::Zero(),c,settings),
              0.25+std::asin(rho)/(2*cg::kPi),1e-6,"signed bivariate correlation");
    }
    Eigen::Matrix4d c=Eigen::Matrix4d::Constant(0.5); c.diagonal().setOnes();
    // Xi=Zi-Z0: all four Xi>0 iff Z0 is smallest among five exchangeable normals.
    close(sh::positiveOrthant(Eigen::Vector4d::Zero(),c,settings),0.2,4e-5,
          "four-dimensional orthant independent order-statistic oracle");
    Eigen::Matrix3d triple; triple<<1,.2,-.35, .2,1,.1, -.35,.1,1;
    close(sh::positiveOrthant(V::Zero(),triple,settings),
          .125+(std::asin(.2)+std::asin(-.35)+std::asin(.1))/(4*cg::kPi),2e-6,
          "three-dimensional signed correlation oracle");
    // Independent one-dimensional common-factor integration oracles.
    for(const auto fixture:{std::pair<double,double>(.4,.1445682326889272),
                            std::pair<double,double>(.9999,.1840601253467595)}) {
        M correlation=M::Constant(fixture.first); correlation.diagonal().setOnes();
        close(sh::positiveOrthant(V(-.9,.2,1.3),correlation,settings),fixture.second,2e-8,
              "nonzero trivariate orthant matches common-factor integral");
    }
    std::cout<<"Gaussian probability checks passed\n";
}
void localLawChecks() {
    const auto constant=makeField(std::make_shared<cg::ConstantMean>(0.0));
    sh::Start exterior;
    const sh::LocalLaw initial(constant,exterior,1,{});
    close(initial.hazard(),0.39659956942467517,2e-6,"positive-only origin hazard");
    close(initial.speedCdf(1),0.3410117508800749,5e-5,"positive-only origin speed law");
    exterior.gradient=V(99,-100,-3);
    close(sh::LocalLaw(constant,exterior,1,{}).hazard(),initial.hazard(),1e-12,
          "unknown exterior gradient is integrated out");
    const auto f=makeField(std::make_shared<cg::PlaneMean>(V(0.2,0.1,1),0.3),0.7,0.8);
    sh::Start surface; surface.kind=sh::Start::SurfaceValueGradient;
    surface.gradient=V(.2,-.1,.7);
    cg::FlightState baseline; baseline.birthPosition=surface.origin;
    baseline.direction=surface.direction; baseline.birthGradient=surface.gradient;
    const cg::ConditionalFlightKernel old(f,baseline);
    for(double t:{0.001,0.03,0.2,0.8,1.7,3.}) {
        const double expected=old.evaluate(t).hazard.value;
        close(sh::LocalLaw(f,surface,t,{}).hazard(),expected,2e-6*(1+expected),
              "K0 surface law regresses existing conditional kernel");
    }
    const auto profile=makeField(std::make_shared<ProfileMean>());
    surface.gradient=V::UnitZ();
    const sh::LocalLaw one(profile,surface,2.625,{1.5});
    close(one.hazard(),0.38547,2e-5,"single-history published profile fixture");
    close(sh::LocalLaw(profile,surface,2.625,{}).hazard(),0.44433,2e-5,
          "profile origin-only baseline fixture");
    const auto two=sh::LocalLaw(profile,surface,2.625,{1.0,1.5});
    const auto reversed=sh::LocalLaw(profile,surface,2.625,{1.5,1.0});
    close(two.hazard(),reversed.hazard(),3e-5,"history order must not change law");
    for(double uniform:{0.1,0.5,0.9}) {
        const double v=one.sampleSpeed(uniform);
        require(v>0,"sampled speed points into solid");
        close(one.speedCdf(v),uniform,8e-5,"speed CDF inversion uses crossing density");
    }
    cg::Random speedRng(2851);
    const double speedQuartile=one.sampleSpeed(.25),speedMedian=one.sampleSpeed(.5);
    int belowQuartile=0,belowMedian=0;
    for(int i=0;i<800;++i) {
        const double speed=one.sampleSpeed(speedRng);
        require(speed>0,"rejection sampler returns a positive crossing speed");
        belowQuartile+=speed<speedQuartile;
        belowMedian+=speed<speedMedian;
    }
    close(belowQuartile/800.0,.25,.055,"rejection speed samples match exact CDF quartile");
    close(belowMedian/800.0,.5,.065,"rejection speed samples match exact CDF median");
    // User's near-miss sphere: verify the local law independently of selector policy.
    const auto sphere=makeField(std::make_shared<cg::SphereMean>(V::Zero(),1),.05,.2);
    exterior.origin=V(1.02,0,-2); exterior.gradient.setZero();
    const sh::LocalLaw grazing(sphere,exterior,1.68928,{1.48928});
    close(grazing.hazard(),1.11982,1e-4,"grazing first-hit hazard fixture");
    close(grazing.sampleSpeed(.5),.45847,3e-5,"grazing flux-weighted median speed");
    const auto rare=makeField(std::make_shared<cg::ConstantMean>(-8));
    exterior=sh::Start{};
    const sh::LocalLaw rareLaw(rare,exterior,20,{10});
    close(rareLaw.hazard(),cg::normalPdfOverCdf(-8)/std::sqrt(2*cg::kPi),2e-5,
          "rare independent history cancels in hazard ratio");
    close(rareLaw.speedCdf(1),1-std::exp(-.5),5e-5,"rare independent history preserves flux speed CDF");
    bool underflowReported=false;
    try {
        const auto extreme=makeField(std::make_shared<cg::ConstantMean>(-40));
        sh::LocalLaw(extreme,exterior,20,{}).hazard();
    } catch(const cg::NumericError &) { underflowReported=true; }
    require(underflowReported,"negative rare survival underflow must not become vacuum");
    sh::Start weakBirth; weakBirth.kind=sh::Start::SurfaceValueGradient;
    weakBirth.gradient=V(0,0,1e-4);
    const sh::LocalLaw nearBirth(constant,weakBirth,1e-4,{5e-5,2.5e-5});
    require(std::isfinite(nearBirth.hazard())&&nearBirth.hazard()>0,
            "nearly rank-one surface history remains a valid positive law");
    const double median=nearBirth.sampleSpeed(.5);
    close(nearBirth.speedCdf(median),.5,5e-4,"near-birth speed and history conditioning agree");
    std::cout<<"Origin and crossing-law checks passed\n";
}
void flightChecks() {
    const auto f=makeField(std::make_shared<cg::ConstantMean>(.4));
    sh::Start start;
    sh::FlightSettings settings;
    settings.historyPoints=2;
    sh::Flight flight(f,start,settings);
    for(double t:{.01,.5,1.1,2.1,3.3}) {
        const auto selected=flight.selectHistory(t);
        require(selected.size()<=2,"bounded internal history count");
        for(double u:selected) require(u>0 && u<t,"history is strictly causal");
    }
    const auto before=flight.selectHistory(1.35);
    flight.selectHistory(4.25);
    require(flight.selectHistory(1.35)==before,"backtracking cannot leak future candidates");
    sh::FlightSettings refined=settings; refined.maxStepOverLength=.125;
    sh::Flight second(f,start,refined);
    require(second.selectHistory(1.35)==before,"history independent of integration step size");
    settings.historyPoints=0;
    settings.opticalDepthTolerance=2e-5;
    sh::Flight baseline(f,start,settings);
    const double full=baseline.opticalDepth(0,1.5);
    close(full,baseline.opticalDepth(0,.6)+baseline.opticalDepth(.6,1.5),8e-5,
          "optical depth survives transparent segment split");
    const double target=.65*full;
    const auto hit=baseline.sampleDistance(0,1.5,target);
    require(hit.hit && hit.distance>0 && hit.distance<1.5,"optical threshold gives collision");
    close(baseline.opticalDepth(0,hit.distance),target,1e-4,"distance inverts optical depth");
    const auto miss=baseline.sampleDistance(0,1.5,full+.1);
    require(!miss.hit,"unreached optical threshold gives escape");
    std::cout<<"Flight history and distance checks passed\n";
}
void gradientChecks() {
    const auto kernel=cg::CovarianceKernel(1,M::Identity());
    const V mean(.2,-.3,-.1),birthMean(.1,.2,.3),birth(.4,-.2,1.5);
    for(bool conditioned:{false,true}) {
        const std::optional<V> g0=conditioned?std::optional<V>(birth):std::nullopt;
        cg::FixedSpeedGradientDistribution law(kernel,V::UnitZ(),1.2,-.8,mean,birthMean,g0);
        const double c=std::exp(-1.2*1.2/2);
        V expected=mean;
        if(conditioned) expected+=c*(birth-birthMean);
        const double variance=conditioned?1-c*c:1;
        cg::Random rng(1457); V sum=V::Zero(),sum2=V::Zero();
        for(int i=0;i<12000;++i) {
            const V g=law.sample(rng);
            close(g.z(),-.8,1e-12,"fixed-speed transverse reconstruction");
            sum+=g; sum2+=g.cwiseProduct(g);
            if(i<30) {
                const V w=V::UnitZ()-2*g.z()*g/g.squaredNorm();
                const auto back=law.gradientForDirection(w);
                require(bool(back)&&(*back-g).norm()<1e-11,"reflection direction reconstructs full gradient");
                close(law.phasePdf(w),law.phasePdfForGradient(g),1e-10,
                      "sample and NEE evaluate identical directional density");
            }
        }
        for(int axis=0;axis<2;++axis) {
            close(sum[axis]/12000,expected[axis],.04,"transverse conditional mean");
            close(sum2[axis]/12000-std::pow(sum[axis]/12000,2),variance,.05,"transverse conditional variance");
        }
        double integral=0;
        constexpr int nz=128,nphi=256;
        for(int z=0;z<nz;++z) for(int p=0;p<nphi;++p) {
            const double cz=-1+2*(z+.5)/nz,phi=2*cg::kPi*(p+.5)/nphi;
            const double s=std::sqrt(1-cz*cz);
            integral+=law.phasePdf(V(s*std::cos(phi),s*std::sin(phi),cz));
        }
        close(integral*4*cg::kPi/(nz*nphi),1,3e-3,"reflection PDF integrates to one");
    }
    std::cout<<"Fixed-speed gradient and scattering checks passed\n";
}
std::shared_ptr<Tungsten::SparseHistoryGpisMedium> makeMedium(const std::string &material="mirror") {
    using namespace Tungsten;
    const std::string json=R"({"type":"sparse_history_gpis","max_bounces":8,
      "gaussian_process":{"type":"standard","mean":{"type":"homogeneous","offset":0.4},
      "covariance":{"type":"squared_exponential","sigma":1,"lengthScale":1}},
      "sparse_history":{"points":0,"optical_depth_tolerance":0.00002},
      "material":{"type":")"+material+R"(","albedo":[0.2,0.5,0.8]}})";
    JsonDocument document(Path("sparse-history-medium-test.json"),json);
    Scene scene;
    auto medium=std::dynamic_pointer_cast<SparseHistoryGpisMedium>(scene.fetchMedium(document));
    require(bool(medium),"sparse-history factory registration");
    medium->loadResources();medium->prepareForRender();return medium;
}
void mediumChecks() {
    using namespace Tungsten;
    auto medium=makeMedium();
    IndependentPathSampler sampler(8497,false);sampler.startPath(0,0);
    Medium::MediumState state{};state.reset();
    auto initial=std::make_shared<SparseHistoryPathContext>();
    initial->owner=medium.get();initial->residualOpticalDepth=10.;
    state.pathContext=initial;
    const Ray shortRay(Vec3f(0.f),Vec3f(0.f,0.f,1.f),.1f,.2f);
    MediumSample sample{};
    require(medium->sampleDistance(sampler,shortRay,state,sample)&&sample.exited,"positive flight crosses boundary");
    const auto next=std::dynamic_pointer_cast<const SparseHistoryPathContext>(state.pathContext);
    require(next&&next->start.kind==sh::Start::ExteriorPositiveOnly,"exterior remains positive-only at boundary");
    close(next->age,.2f,1e-12,"transparent boundary advances physical flight age");
    require(next->start.origin==V::Zero()&&next->residualOpticalDepth&&*next->residualOpticalDepth<10,
            "transparent boundary keeps original observation and residual exponential clock");
    const auto saved=state.pathContext;
    auto shadow=state;
    const Ray shadowRay(sample.p,Vec3f(0.f,0.f,1.f),0,.3f);
    const double tr=medium->transmittance(sampler,shadowRay,false,false,&shadow).x();
    require(tr>0&&tr<1&&state.pathContext==saved,"shadow state does not mutate parent");
    const auto shadowEnd=std::dynamic_pointer_cast<const SparseHistoryPathContext>(shadow.pathContext);
    close(shadowEnd->age,double(.2f)+double(.3f),1e-12,"shadow preserves straight flight age");
    auto turn=state;
    const Ray turned(sample.p,Vec3f(1.f,0.f,0.f),0,.1f);
    medium->transmittance(sampler,turned,false,false,&turn);
    const auto turnedEnd=std::dynamic_pointer_cast<const SparseHistoryPathContext>(turn.pathContext);
    require(turnedEnd->start.kind==sh::Start::ExteriorPositiveOnly,"ordinary surface turn is not GP zero");
    close(turnedEnd->age,.1f,1e-12,"ordinary turn starts exterior renewal");
    for(const std::string material:{"mirror","lambert"}) {
        auto m=material=="mirror"?medium:makeMedium(material);
        state.reset();initial=std::make_shared<SparseHistoryPathContext>();
        initial->owner=m.get();initial->residualOpticalDepth=.04;state.pathContext=initial;
        const Ray incoming(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,1);
        require(m->sampleDistance(sampler,incoming,state,sample)&&!sample.exited,"first GP collision");
        PhaseSample scatter;
        require(sample.phase->sample(sampler,incoming.dir(),sample,scatter),"history-aware phase sample");
        const auto successor=std::dynamic_pointer_cast<const SparseHistoryPathContext>(scatter.pathContext);
        require(successor&&successor->start.kind==sh::Start::SurfaceValueGradient,"GP successor switches to known value-gradient");
        const V outgoing(scatter.w.x(),scatter.w.y(),scatter.w.z());
        require(outgoing.dot(successor->start.gradient)>0,"successor full gradient points to outgoing empty side");
        require(std::abs(successor->start.gradient.norm()-1)>1e-5,"propagated gradient retains magnitude");
        std::shared_ptr<const MediumPathContext> connection;
        const Vec3f value=sample.phase->evalConnection(sampler,incoming.dir(),scatter.w,sample,connection);
        require(value.max()>0&&bool(connection),"NEE constructs conditional successor");
        const auto light=std::dynamic_pointer_cast<const SparseHistoryPathContext>(connection);
        require(light&&light->start.kind==sh::Start::SurfaceValueGradient,"NEE also uses collision-origin conditioning");
        if(material=="lambert")
            require(light->start.gradient==successor->start.gradient,"Lambert branches share observed full gradient");
        auto second=std::make_shared<SparseHistoryPathContext>(*successor);
        second->residualOpticalDepth=.1;state.pathContext=second;
        const Ray reflected(sample.p,scatter.w,0,5);
        require(m->sampleDistance(sampler,reflected,state,sample)&&!sample.exited,"conditioned subsequent GP collision");
    }
    std::cout<<"Medium boundary, shadow, NEE and multi-hit checks passed\n";
}
void imageCheck(const char *path) {
    using namespace Tungsten;
    int w=0,h=0;
    const auto pixels=ImageIO::loadHdr(Path(path),TexelConversion::REQUEST_RGB,w,h);
    require(bool(pixels)&&w>0&&h>0,"load sparse-history smoke render");
    double sum=0,lo=1e30,hi=0;
    for(int i=0;i<3*w*h;++i) {
        const double x=pixels[i]; require(x>=0&&std::isfinite(x),"finite nonnegative image");
        sum+=x;lo=std::min(lo,x);hi=std::max(hi,x);
    }
    require(sum/(3*w*h)>.02 && hi-lo>.2,"render has visible lit geometry");
    std::cout<<"Image "<<w<<"x"<<h<<" mean="<<sum/(3*w*h)<<" range="<<lo<<","<<hi<<"\n";
}
void compareImages(const char *reference,const char *candidate) {
    using namespace Tungsten;
    int aw=0,ah=0,bw=0,bh=0;
    const auto a=ImageIO::loadHdr(Path(reference),TexelConversion::REQUEST_RGB,aw,ah);
    const auto b=ImageIO::loadHdr(Path(candidate),TexelConversion::REQUEST_RGB,bw,bh);
    require(a&&b&&aw==bw&&ah==bh&&aw>0&&ah>0,"comparison images have equal dimensions");
    double squared=0,energy=0,maximum=0,meanA=0,meanB=0;
    const int n=3*aw*ah;
    for(int i=0;i<n;++i) {
        require(std::isfinite(a[i])&&std::isfinite(b[i])&&a[i]>=0&&b[i]>=0,"finite comparison images");
        const double d=double(b[i])-a[i];
        squared+=d*d;energy+=double(a[i])*a[i];maximum=std::max(maximum,std::abs(d));
        meanA+=a[i];meanB+=b[i];
    }
    std::cout.precision(12);
    std::cout<<"{\"rmse\":"<<std::sqrt(squared/n)
             <<",\"relative_rmse\":"<<std::sqrt(squared/std::max(energy,1e-300))
             <<",\"maximum_absolute_error\":"<<maximum
             <<",\"reference_mean\":"<<meanA/n<<",\"candidate_mean\":"<<meanB/n<<"}\n";
}
}
#include "SparseHistoryGridChecks.hpp"
#include "SparseHistoryLutChecks.hpp"
int main(int argc,char **argv) {
    std::cout.setf(std::ios::unitbuf);
    try {
        const std::string mode=argc>1?argv[1]:"all";
        if(mode=="image" && argc==3) {imageCheck(argv[2]);return 0;}
        if(mode=="compare-images" && argc==4) {compareImages(argv[2],argv[3]);return 0;}
        if(mode=="lut" && argc==3) {sparseHistoryLutChecks(argv[2]);return 0;}
        if(mode=="lut-benchmark" && argc==3) {sparseHistoryLutBenchmark(argv[2]);return 0;}
        if(mode=="all"||mode=="gaussian") gaussianChecks();
        if(mode=="all"||mode=="local") localLawChecks();
        if(mode=="all"||mode=="flight") flightChecks();
        if(mode=="all"||mode=="gradient") gradientChecks();
        if(mode=="all"||mode=="medium") mediumChecks();
        if(mode=="all"||mode=="grid") sparseHistoryGridChecks();
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
