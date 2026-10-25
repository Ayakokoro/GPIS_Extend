#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/transport/CollisionGradientSampler.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include "math/conditional/transport/DirectionalGradient.h"
#include "media/GlobalConditionalMedium.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/ImageIO.hpp"
#include "thread/TaskGroup.hpp"
#include <iostream>

namespace cg = Tungsten::ConditionalGpis;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
void check(bool ok, const char *s) { if (!ok) throw std::runtime_error(s); }
void close(double x, double y, double eps, const char *s) {
    if (!(std::abs(x-y) <= eps)) { std::cerr<<s<<": "<<x<<" vs "<<y<<"\n"; throw std::runtime_error(s); }
}
cg::GPSSField field() {
    M p; p<<1.4,0.2,0.1, 0.2,0.8,-0.1, 0.1,-0.1,1.1;
    return {std::make_shared<cg::PlaneMean>(V(0.2,0.1,1),0.3),cg::CovarianceKernel(0.7,p),
            {V::Constant(-100),V::Constant(100)}};
}
cg::Gaussian<4> reference(const cg::GPSSField &f, const cg::FlightState &s, const V &x)
{
    const auto j=f.kernel.evaluate(x,s.birthPosition);
    Eigen::Matrix4d base=Eigen::Matrix4d::Zero(),cross;
    base(0,0)=f.kernel.sigma()*f.kernel.sigma();
    base.bottomRightCorner<3,3>()=f.kernel.gradientCovarianceAtZero();
    cross(0,0)=j.valueValue;
    cross.block<1,3>(0,1)=j.valueXGradientY.transpose();
    cross.block<3,1>(1,0)=j.gradientXValueY;
    cross.bottomRightCorner<3,3>()=j.gradientXGradientY;
    const auto birth=f.mean->evaluate(s.birthPosition), target=f.mean->evaluate(x);
    Eigen::Vector4d obs,mean;
    obs<<s.birthValue-birth.value,s.birthGradient-birth.gradient;
    mean<<target.value,target.gradient;
    return {mean+cross*base.ldlt().solve(obs),base-cross*base.ldlt().solve(cross.transpose())};
}
void stage1()
{
    auto f=field();
    cg::FlightState s;
    s.birthPosition=V(0.1,-0.1,0.2);
    s.direction=V(0.2,-0.3,1).normalized();
    s.birthGradient=V(0.1,-0.2,1.2);
    cg::ConditionalFlightKernel k(f,s);
    for (double t : {0.03,0.1,0.4,1.,3.}) {
        const auto ref=reference(f,s,s.birthPosition+t*s.direction);
        const auto q=k.scalarStatistics(t);
        const V mu=ref.mean.tail<3>()-ref.covariance.block<3,1>(1,0)*ref.mean[0]/ref.covariance(0,0);
        const M cov=ref.covariance.bottomRightCorner<3,3>()-
            ref.covariance.block<3,1>(1,0)*ref.covariance.block<1,3>(0,1)/ref.covariance(0,0);
        close(q.meanF,ref.mean[0],1e-12,"conditional value vs joint Gaussian");
        close(q.varianceF,ref.covariance(0,0),1e-12,"conditional variance vs joint Gaussian");
        check((k.hitStatistics(t).collisionGradient.mean-mu).norm()<2e-7,"conditional gradient mean");
        check((k.hitStatistics(t).collisionGradient.covariance-cov).norm()<2e-9,"conditional gradient covariance");
    }
    auto survivalField=f;
    survivalField.mean=std::make_shared<cg::ConstantMean>(0.0);
    cg::ConditionalFlightKernel flight(survivalField,s);
    const double end=4;
    const auto bounds=flight.twoSegmentMajorants(end);
    for (int i=1;i<=2000;++i) {
        const double t=bounds.split*i/2000;
        check(flight.evaluate(t).hazard.value<=bounds.nearMaximum*(1+1e-10),"birth majorant");
    }
    for (const auto &b:bounds.farIntervals)
        for (int j=0;j<=20;++j)
            check(flight.evaluate(b.begin+(b.end-b.begin)*j/20).hazard.value<=b.maximum*(1+1e-10),"interval majorant");
    for (double t : {1e-3,1e-5,1e-8,1e-12}) {
        const auto stats=flight.scalarStatistics(t);
        check(stats.varianceF>0 && stats.varianceAtZero>0,"stable tiny-distance covariance");
        check(std::isfinite(flight.evaluate(t).hazard.value),"finite tiny-distance hazard");
    }
    // Independent composite midpoint integral is used only as a test oracle.
    double integral=0;
    for (int i=0;i<20000;++i) integral+=flight.evaluate(end*(i+0.5)/20000).hazard.value*end/20000;
    const double expected=std::exp(-integral);
    cg::Random rng(713);
    std::vector<cg::ConditionalMajorantInterval> intervals{{0,bounds.split,bounds.nearMaximum}};
    intervals.insert(intervals.end(),bounds.farIntervals.begin(),bounds.farIntervals.end());
    int escaped=0;
    constexpr int count=12000;
    for (int i=0;i<count;++i) {
        bool hit=false;
        for (auto b:intervals) {
            if (b.maximum==0) continue;
            double t=b.begin;
            while ((t-=std::log(rng.openUniform01())/b.maximum)<b.end) {
                if (rng.openUniform01()*b.maximum<flight.evaluate(t).hazard.value) { hit=true; break; }
            }
            if (hit) break;
        }
        escaped+=!hit;
    }
    close(double(escaped)/count,expected,0.018,"delta tracking survival");
    // Integrating g*(-w.g)+ N(g) gives the raw-gradient collision mean.
    const auto distribution=flight.hitStatistics(1.1).collisionGradient;
    const V w=s.direction, cw=distribution.covariance*w;
    const double mu=w.dot(distribution.mean), sd=std::sqrt(w.dot(cw));
    const double flux=cg::negativePartMean(mu,sd).value;
    const double meanK=-cg::positiveRawMoment(2,-mu,sd).value/flux;
    const V expectedG=distribution.mean+cw*(meanK-mu)/(sd*sd);
    V sum=V::Zero();
    for(int i=0;i<count;++i) {
        const V g=cg::sampleFluxWeightedGradient(distribution,w,rng);
        check(w.dot(g)<0,"collision gradient hemisphere");
        sum+=g;
    }
    check((sum/count-expectedG).norm()<0.025,"raw collision gradient moments");
    std::cout<<"stage1: joint Gaussian, birth bounds, tiny distances, flux moments passed; survival "
             <<double(escaped)/count<<" expected "<<expected<<"\n";
}
std::shared_ptr<Tungsten::GlobalConditionalMedium> makeMedium(const std::string &tracking = "") {
    using namespace Tungsten;
    JsonDocument document(Path("conditional-test.json"),std::string(R"({"type":"global_conditional",)")+
      (tracking.empty() ? "" : tracking+",")+R"("gaussian_process":{"type":"standard",
      "mean":{"type":"linear","direction":[0,0,1]},
      "covariance":{"type":"squared_exponential","sigma":1,"lengthScale":1}},
      "material":{"type":"mirror","albedo":[0.15,0.8,0.5]}})");
    Scene scene;
    auto m=std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(document));
    check(bool(m),"conditional factory");
    m->loadResources();m->prepareForRender();return m;
}
void endpointPrecision() {
    using namespace Tungsten;
    auto m=makeMedium();
    IndependentPathSampler sampler(9371,false);sampler.startPath(0,0);
    cg::Random directions(9371);
    // Non-axis-aligned float directions acquire another rounding on double
    // normalization. Cancellation at an endpoint near the world origin makes
    // independently recomputed endpoint coordinates especially sensitive.
    for(int i=0;i<128;++i) {
        const Vec3f raw(float(2*directions.uniform01()-1),float(2*directions.uniform01()-1),float(2*directions.uniform01()-1));
        const Vec3f w=raw.normalized();const float length=.375f;
        auto context=std::make_shared<ConditionalPathContext>();
        context->owner=m.get();context->position=-double(length)*V(w.x(),w.y(),w.z());
        context->rayOrigin=Vec3f(float(context->position.x()),float(context->position.y()),float(context->position.z()));
        context->birth.birthPosition=context->position;
        context->birth.birthValue=20;context->birth.birthGradient=V::Zero();
        const Ray incoming(context->rayOrigin,w,0,length);
        for(bool shadow : {false,true}) {
            Medium::MediumState state{};state.reset();state.pathContext=context;
            Vec3f endpoint;
            if(shadow) {
                check(m->transmittance(sampler,incoming,true,true,&state).min()>0,"positive precision-probe visibility");
                endpoint=incoming.hitpoint();
            } else {
                MediumSample ms{};
                check(m->sampleDistance(sampler,incoming,state,ms)&&ms.exited,"precision probe reaches geometry");
                endpoint=ms.p;
            }
            // An ordinary surface may change direction immediately after
            // endpoint renewal; it must not look like an unobserved turn.
            check(m->transmittance(sampler,Ray(endpoint,Vec3f(1.0f,0.0f,0.0f),0,.01f),true,true,&state).min()>0,
                "near-origin renewed endpoint permits a direction change");
            const auto next=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
            check(next->position==next->birth.birthPosition,"renewed endpoint uses one precise position");
        }
    }
    std::cout<<"endpoint precision: non-axis directions, near-origin cancellation, distance and shadow renewal passed\n";
}
void stage2() {
    auto f=field();
    f.mean=std::make_shared<cg::ConstantMean>(0.0);
    cg::FlightState s; s.birthPosition=V::Zero();s.birthGradient=V(0.2,-0.1,1.0);
    const V w=V(0.2,0.3,1).normalized(),origin=0.4*w;
    s.direction=w;
    cg::ConditionalSegment segment(f,s,origin,w,2);
    for(double t : {0.,0.1,0.5,1.,2.}) {
        const auto ref=reference(f,s,origin+t*w);
        const auto g=segment.gradient(t);
        const V mu=ref.mean.tail<3>()-ref.covariance.block<3,1>(1,0)*ref.mean[0]/ref.covariance(0,0);
        const M cov=ref.covariance.bottomRightCorner<3,3>()-
            ref.covariance.block<3,1>(1,0)*ref.covariance.block<1,3>(0,1)/ref.covariance(0,0);
        check((g.mean-mu).norm()<1e-11,"resumed straight mean vs joint Gaussian");
        check((g.covariance-cov).norm()<1e-11,"resumed straight covariance vs joint Gaussian");
    }
    for(int i=0;i<20;++i) {
        const double lo=0.1*i,hi=0.1*(i+1),bound=segment.majorant(lo,hi);
        for(int j=0;j<=50;++j) check(segment.hazard(lo+(hi-lo)*j/50)<=bound,"straight majorant");
    }
    double integral=0;
    for(int i=0;i<10000;++i) integral+=segment.hazard(2*(i+0.5)/10000)*2/10000;
    cg::Random rng(832);int escaped=0,splitEscaped=0;
    const cg::ConditionalSegment first(f,s,origin,w,0.7),second(f,s,origin+0.7*w,w,1.3);
    constexpr int count=12000;
    for(int i=0;i<count;++i) {
        double t;escaped+=segment.track(rng,true,t)!=0;
        splitEscaped+=(first.track(rng,true,t)!=0 && second.track(rng,true,t)!=0);
    }
    close(double(escaped)/count,std::exp(-integral),0.018,"straight finite survival");
    close(double(splitEscaped)/count,std::exp(-integral),0.018,"subdivision preserves birth conditioning");

    // The joint endpoint observation must retain the old observation's
    // correlations, including value-gradient covariance after F>0 truncation.
    const cg::ConditionalSegment arrival(f,s,s.birthPosition,w,1.1);
    const auto ref=reference(f,s,s.birthPosition+1.1*w);
    const double sd=std::sqrt(ref.covariance(0,0)),alpha=ref.mean[0]/sd;
    const double mills=cg::normalPdfOverCdf(alpha);
    const Eigen::Vector4d regression=ref.covariance.col(0)/sd;
    const Eigen::Vector4d expectedMean=ref.mean+regression*mills;
    const Eigen::Matrix4d expectedCovariance=ref.covariance-
        regression*regression.transpose()*mills*(alpha+mills);
    Eigen::Vector4d sum=Eigen::Vector4d::Zero();
    Eigen::Matrix4d centeredSecond=Eigen::Matrix4d::Zero(),centeredFourth=Eigen::Matrix4d::Zero();
    for(int i=0;i<count;++i) {
        const auto endpoint=arrival.samplePositiveEndpoint(rng);
        check(endpoint.birthValue>0 && endpoint.birthGradient.allFinite(),"positive complete endpoint observation");
        check((endpoint.birthPosition-1.1*w).norm()<1e-14,"endpoint becomes new birth");
        Eigen::Vector4d sample;sample<<endpoint.birthValue,endpoint.birthGradient;
        const Eigen::Vector4d residual=sample-expectedMean;
        const Eigen::Matrix4d product=residual*residual.transpose();
        sum+=sample;centeredSecond+=product;centeredFourth+=product.cwiseProduct(product);
    }
    for(int i=0;i<4;++i) {
        close(sum[i]/count,expectedMean[i],6*std::sqrt(expectedCovariance(i,i)/count)+1e-6,
            "renewed joint mean vs truncated Gaussian");
        for(int j=0;j<4;++j) {
            const double moment=centeredSecond(i,j)/count;
            const double error=std::sqrt(std::max(0.0,centeredFourth(i,j)/count-moment*moment)/count);
            close(moment,expectedCovariance(i,j),6*error+1e-6,"renewed joint covariance vs truncated Gaussian");
        }
    }
    // A turn without renewal is now a caller error, including an exact
    // reversal. A turn from a freshly observed positive endpoint is supported.
    for(const V &direction : {V::UnitX().eval(),V(-w)}) {
        bool rejected=false;
        try { cg::ConditionalSegment invalid(f,s,origin,direction,1); }
        catch(const cg::NumericError &e) { rejected=e.status()==cg::NumericStatus::InvalidInput; }
        check(rejected,"changed direction requires endpoint renewal");
        const auto endpoint=arrival.samplePositiveEndpoint(rng);
        cg::ConditionalSegment outgoing(f,endpoint,endpoint.birthPosition,direction,1);
        double t;outgoing.track(rng,true,t);
    }

    using namespace Tungsten;
    auto m=makeMedium();
    Medium::MediumState state{};state.reset();
    auto context=std::make_shared<ConditionalPathContext>();
    context->owner=m.get();context->position=V(0.2,0.3,1.0);
    context->birth.birthPosition=context->position;
    context->birth.birthValue=5;context->birth.birthGradient=V(0.2,-0.1,1);
    context->rayOrigin=Vec3f(0.2f,0.3f,1.0f);
    state.pathContext=context;
    auto branch=state;branch.reset();
    check(state.pathContext==context && !branch.pathContext,"branch reset must not mutate parent");
    IndependentPathSampler sampler(53,false);sampler.startPath(0,0);
    MediumSample ms{};
    Ray ray(context->rayOrigin,Vec3f(0.0f,0.0f,1.0f),0,0.2f);
    check(m->sampleDistance(sampler,ray,state,ms) && ms.exited,"reach ordinary geometry endpoint");
    const auto updated=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
    check(updated->birth.birthPosition==updated->position && updated->birth.birthValue>0,
        "ordinary surface must renew value and gradient before branching");
    check(updated->rayOrigin==ms.p && updated->rayOrigin==ray.hitpoint(),"renewal keeps the geometry ray origin");
    check(context->position==V(0.2,0.3,1.0) && context->birth.birthValue==5,"tracking must not mutate another branch");
    for(const Vec3f &direction : {Vec3f(1.0f,0.0f,0.0f),Vec3f(0.0f,0.0f,-1.0f),Vec3f(0.0f,0.0f,1.0f)}) {
        auto shadow=state;
        const Ray connection(ms.p,direction,0,0.1f);
        const Vec3f tr=m->transmittance(sampler,connection,true,true,&shadow);
        check(tr.min()>0 && tr.max()<=1,"changed-direction shadow from renewed surface");
        const auto endpoint=std::dynamic_pointer_cast<const ConditionalPathContext>(shadow.pathContext);
        check(endpoint->birth.birthPosition==endpoint->position && endpoint->birth.birthValue>0,
            "shadow endpoint must renew its observation");
        check(state.pathContext==updated,"shadow renewal must not mutate the continuation");
    }
    const Ray outgoing(ms.p,Vec3f(1.0f,0.0f,0.0f),0,0.1f);
    check(m->sampleDistance(sampler,outgoing,state,ms) && ms.exited,"surface continuation after direction change");
    const auto beforeEmpty=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
    check(beforeEmpty->birth.birthPosition==beforeEmpty->position,"continuation renews again");
    check(m->transmittance(sampler,Ray(ms.p,Vec3f(0.0f,1.0f,0.0f),0,0),true,true,&state)==Vec3f(1.0f),
        "zero-length segment transmits");
    const auto afterEmpty=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
    check(afterEmpty->birth.birthValue==beforeEmpty->birth.birthValue &&
        afterEmpty->birth.birthGradient==beforeEmpty->birth.birthGradient,"zero-length segment preserves observation");
    std::cout<<"stage2: straight subdivision, joint endpoint renewal, changed directions and immutable state passed; survival "
             <<double(escaped)/count<<", split "<<double(splitEscaped)/count<<", expected "<<std::exp(-integral)<<"\n";
}
#include "ConditionalKnobChecks.hpp"
#include "ConditionalSdfChecks.hpp"
#include "ConditionalSurfaceBirthChecks.hpp"
#include "ConditionalBoundChecks.hpp"
#include "ConditionalTrackingChecks.hpp"
#include "ConditionalTrackingProbe.hpp"
struct Moments {
    double sum=0,sum2=0;int n=0;
    void add(double v) {sum+=v;sum2+=v*v;++n;}
    double mean() const {return sum/n;}
    double variance() const {return std::max(0.0,(sum2-sum*sum/n)/(n-1));}
};
void stage3() {
    cg::Random rng(535);
    for(double mean : {-10.,-1.,0.,1.,10.}) {
        const double sd=0.7,mode=0.5*mean+0.5*std::hypot(mean,std::sqrt(12.0)*sd);
        const double peak=3*std::log(mode)-0.5*std::pow((mode-mean)/sd,2);
        double norm=0,moment=0;
        const double end=std::max(0.0,mean)+10*sd;
        for(int i=0;i<20000;++i) {
            const double r=end*(i+0.5)/20000;
            const double p=std::exp(3*std::log(r)-0.5*std::pow((r-mean)/sd,2)-peak);
            norm+=p;moment+=p*r;
        }
        Moments sampled;
        for(int i=0;i<20000;++i) sampled.add(cg::sampleRadialCubic(mean,sd,rng));
        close(sampled.mean(),moment/norm,6*std::sqrt(sampled.variance()/sampled.n)+1e-5,"direction-conditioned gradient length");
    }
    auto f=field();cg::FlightState birth;
    birth.birthValue=0.4;birth.birthGradient=V(0.2,0.1,0.6);
    const V origin(0.3,0.1,0.8),w=V(0.1,0.2,-0.5).normalized();
    birth.birthPosition=origin;
    cg::ConditionalSegment segment(f,birth,origin,w,1.5);
    double integral=0;
    for(int i=0;i<10000;++i) integral+=segment.hazard(1.5*(i+0.5)/10000)*1.5/10000;
    Moments ratio;
    for(int i=0;i<16000;++i) {double t;ratio.add(segment.track(rng,false,t));}
    close(ratio.mean(),std::exp(-integral),6*std::sqrt(ratio.variance()/ratio.n)+1e-4,"conditional ratio tracking");
    using namespace Tungsten;
    auto medium=makeMedium();
    birth.birthPosition=V(0,0,1.2);birth.birthValue=0.7;birth.birthGradient=V(0.2,0.1,1);
    const V position(0,0,0.4);
    cg::ConditionalSegment incoming(medium->field(),birth,birth.birthPosition,V(0,0,-1),0.8);
    auto event=std::make_shared<ConditionalEventContext>();
    event->owner=medium.get();event->position=position;event->gradient=incoming.gradient(0.8);
    MediumSample ms{};ms.p=Vec3f(0.0f,0.0f,0.4f);ms.eventContext=event;
    ms.phase=const_cast<PhaseFunction *>(medium->phaseFunction(ms.p));
    const Vec3f wi(0.0f,0.0f,-1.0f);
    IndependentPathSampler sampler(8331,false);
    Moments phaseEstimate,lightEstimate,misEstimate;
    const double lightPdf=1/(4*cg::kPi);
    for(int i=0;i<20000;++i) {
        sampler.startPath(0,i);
        Medium::MediumState state{};state.reset();
        PhaseSample ps;
        check(ms.phase->sample(sampler,wi,ms,ps),"conditional phase sample");
        state.pathContext=ps.pathContext;
        const auto phaseContext=std::dynamic_pointer_cast<const ConditionalPathContext>(ps.pathContext);
        check(phaseContext && phaseContext->birth.birthValue==0 &&
            phaseContext->birth.birthPosition==event->position && phaseContext->position==event->position,
            "GP continuation starts from a new zero observation");
        auto original=state.pathContext;
        const double tr=medium->transmittance(sampler,Ray(ms.p,ps.w,0,2),false,true,&state).y();
        check(ps.pathContext==original,"shadow cannot mutate phase successor");
        const double pp=ps.pdf;
        const double pWeight=pp*pp/(pp*pp+lightPdf*lightPdf);
        phaseEstimate.add(tr);
        const double z=2*rng.uniform01()-1,phi=2*cg::kPi*rng.uniform01(),r=std::sqrt(1-z*z);
        const Vec3f wo(float(r*std::cos(phi)),float(r*std::sin(phi)),float(z));
        state.reset();
        const Vec3f value=ms.phase->evalConnection(sampler,wi,wo,ms,state.pathContext);
        double light=0,mis=tr*pWeight;
        if(value.y()>0) {
            check(bool(state.pathContext),"light connection must supply a gradient observation");
            const auto context=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
            check(context->birth.birthValue==0 && context->birth.birthPosition==event->position &&
                context->position==event->position,"GP light connection starts from a new zero observation");
            check(context->birth.birthGradient.dot(V(wo.x(),wo.y(),wo.z()))>0,"light branch enters positive side");
            const double visibility=medium->transmittance(sampler,Ray(ms.p,wo,0,2),false,true,&state).y();
            light=value.y()/0.8/lightPdf*visibility;
            const double density=ms.phase->pdf(wi,wo,ms);
            mis+=light*lightPdf*lightPdf/(density*density+lightPdf*lightPdf);
        }
        lightEstimate.add(light);misEstimate.add(mis);
    }
    const double tolerance=6*std::sqrt((phaseEstimate.variance()+lightEstimate.variance())/phaseEstimate.n)+1e-4;
    close(phaseEstimate.mean(),lightEstimate.mean(),tolerance,"phase vs light connections with latent gradient");
    close(phaseEstimate.mean(),misEstimate.mean(),6*std::sqrt((phaseEstimate.variance()+misEstimate.variance())/phaseEstimate.n)+1e-4,"conditional MIS expectation");
    bool completed=false,rethrown=false;
    TaskGroup group([](uint32,uint32,uint32){throw cg::NumericError(cg::NumericStatus::InvalidMajorant,"test bound failure");},
                    [&]{completed=true;},2);
    std::thread a([&]{group.run(0,0);}),b([&]{group.run(1,1);});a.join();b.join();
    try {group.wait();} catch(const cg::NumericError &) {rethrown=true;}
    check(rethrown&&!completed,"numerical failure must propagate without successful completion");
    std::cout<<"stage3: radial law, ratio tracking, branch isolation, conditional NEE/MIS and error propagation passed; phase "
             <<phaseEstimate.mean()<<", light "<<lightEstimate.mean()<<", MIS "<<misEstimate.mean()
             <<"; ratio "<<ratio.mean()<<" expected "<<std::exp(-integral)<<"\n";
}
void imageCheck(const char *file,bool furnace) {
    using namespace Tungsten;
    int w,h;auto data=ImageIO::loadHdr(Path(file),TexelConversion::REQUEST_RGB,w,h);
    check(bool(data)&&w>0&&h>0,"rendered image must be readable");
    double sum=0,min=1e30,max=0;
    for(int i=0;i<w*h*3;++i) {
        check(std::isfinite(data[i])&&data[i]>=0,"rendered image must be finite and nonnegative");
        sum+=data[i];min=std::min(min,double(data[i]));max=std::max(max,double(data[i]));
    }
    check(max>0,"rendered image must not be black");
    if(furnace) close(sum/(w*h*3),1.0,0.025,"white furnace energy");
    std::cout<<"image "<<file<<": mean "<<sum/(w*h*3)<<", range ["<<min<<", "<<max<<"]\n";
}
int main(int argc,char **argv)
{
    try {
        std::string stage=argc>1?argv[1]:"all";
        if(stage=="stage1"||stage=="all") stage1();
        if(stage=="stage2"||stage=="all") stage2();
        if(stage=="endpoint_precision"||stage=="all") endpointPrecision();
        if(stage=="stage3"||stage=="all") stage3();
        if(stage=="surface_births"||stage=="all") surfaceBirthChecks();
        if(stage=="sdf"||stage=="all") sdfChecks();
        if(stage=="knob"||stage=="all") knobChecks();
        if(stage=="bounds"||stage=="all") boundChecks();
        if(stage=="tracking"||stage=="all") trackingChecks();
        if(stage=="tracking_probe") { check(argc>2,"missing probe scene");trackingProbe(argv[2]); }
        if(stage=="tracking_split_probe") { check(argc>2,"missing probe scene");trackingProbe(argv[2],true); }
        if(stage=="image"||stage=="furnace") {check(argc>2,"missing image");imageCheck(argv[2],stage=="furnace");}
        return 0;
    }
    catch (const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
}
