#include "media/NeuralRenewalMedium.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "media/NeuralSdfGrid.hpp"
#include "math/neural/RenewalSampler.hpp"
#include <nlohmann/json.hpp>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <iostream>
#include <cmath>

using namespace Tungsten;
namespace {
using V=ConditionalGpis::Vector3;
V eigen(const Vec3f &v) { return V(v.x(),v.y(),v.z()); }
void require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
void near(double a,double b,double tolerance,const char *message) {
    if (!(std::abs(a-b)<=tolerance)) {
        std::cerr<<message<<": "<<a<<" vs "<<b<<"\n"; throw std::runtime_error(message);
    }
}
std::shared_ptr<NeuralRenewalMedium> parseMedium(const nlohmann::json &spec) {
    Scene scene; JsonDocument doc(Path("neural-test.json"),spec.dump());
    return std::dynamic_pointer_cast<NeuralRenewalMedium>(scene.fetchMedium(doc));
}
void checkImage(const char *file, bool furnace) {
    int width=0,height=0;
    const auto data=ImageIO::loadHdr(Path(file),TexelConversion::REQUEST_RGB,width,height);
    require(bool(data) && width>0 && height>0,"rendered image must be readable");
    double sum=0,minimum=1e30,maximum=0;
    for (int i=0;i<width*height*3;++i) {
        require(std::isfinite(data[i]) && data[i]>=0,"rendered image must be finite and nonnegative");
        sum+=data[i]; minimum=std::min(minimum,double(data[i])); maximum=std::max(maximum,double(data[i]));
    }
    require(maximum>0,"rendered image must not be black");
    if (furnace) near(sum/(width*height*3),1,0.025,"white furnace energy");
    std::cout<<"image "<<file<<": mean "<<sum/(width*height*3)<<", range ["
        <<minimum<<", "<<maximum<<"]\n";
}
}
int main(int argc,char **argv) {
    try {
        require(argc==3,"usage: neural_renewal_test model.json field.nvdb");
        if (std::string(argv[1])=="--image" || std::string(argv[1])=="--furnace") {
            checkImage(argv[2],std::string(argv[1])=="--furnace"); return 0;
        }
        const double sigma=0.05,lengthScale=std::sqrt(2.)*sigma/0.5;
        nlohmann::json spec={{"type","neural_causal_renewal"},{"name","neural"},{"model",argv[1]},
            {"field",{{"file",argv[2]}}},
            {"covariance",{{"type","squared_exponential"},{"sigma",sigma},{"lengthScale",lengthScale}}},
            {"material",{{"force_unit_fresnel",true}}}};
        for (int invalid=0;invalid<6;++invalid) {
            auto bad=spec;
            if (invalid==0) bad.erase("covariance");
            if (invalid==1) bad["covariance"]["sigma"]=0;
            if (invalid==2) bad["covariance"]["lengthScale"]=-1;
            if (invalid==3) bad["covariance"]["type"]="matern32";
            if (invalid==4) bad["material"]["roughness"]=0.5;
            if (invalid==5) bad["field"]["sigma"]=0.05;
            bool rejected=false;
            try { parseMedium(bad); } catch (const std::exception &) { rejected=true; }
            require(rejected,"invalid or obsolete covariance configuration rejected");
        }
        auto adapter=parseMedium(spec);
        require(bool(adapter),"neural factory registration");
        adapter->loadResources(); adapter->prepareForRender();
        auto mean=NeuralSdfGrid::open(argv[2]);
        require(!mean->intrinsicSigma(),"neural SDF has no intrinsic sigma");
        ConditionalGpis::GPSSField field; field.mean=mean; field.activeDomain=mean->activeNodeBounds();
        field.kernel=ConditionalGpis::CovarianceKernel::fromCorrelationLengths(sigma,V::Constant(lengthScale));
        const auto model=NeuralGpis::CausalRenewalModel::load(argv[1]);
        const NeuralGpis::RenewalSampler medium(field,model);
        std::cout<<"SDF domain "<<field.activeDomain.minimum.transpose()<<" to "
            <<field.activeDomain.maximum.transpose()<<", sigma="<<sigma<<", lengthScale="<<lengthScale<<"\n";
        const Vec3f origin(0.01f,-3.f,2.27f);
        const Vec3f direction=(Vec3f(0.f,0.f,0.35f)-origin).normalized();
        const Ray ray(origin,direction,0.f,10.f);
        const V w=eigen(direction).normalized();
        const auto domain=field.activeDomain.intersect({eigen(origin),w});
        require(domain.hit,"camera ray enters test field");
        V entry=eigen(origin)+domain.entry*w;
        entry=entry.cwiseMax(field.activeDomain.minimum).cwiseMin(field.activeDomain.maximum);
        const auto flight=medium.beginFlight({entry,w});
        require(bool(flight),"reference flight");
        const auto median=model.sample(flight->mean,flight->start,{0.5,0.3,0.7});
        require(median.hit,"test ray has a median collision");
        const double endQ=median.q,beginQ=std::max(0.,endQ-0.39),targetQ=(beginQ+endQ)/2;
        const auto hazards=model.query(flight->mean,flight->start,{beginQ,targetQ,endQ});
        const double optical=hazards[1].cumulativeHazard-hazards[0].cumulativeHazard;
        require(optical>0 && optical<700,"nontrivial truncated flight");
        const auto interval=model.sampleInterval(flight->mean,flight->start,
            {std::exp(-optical),0.3,0.7},beginQ,endQ);
        require(interval.hit,"conditional interval event");
        near(interval.q,targetQ,2e-6,"bounded inverse matches conditional cumulative hazard");
        require(!model.sampleInterval(flight->mean,flight->start,{0.5,0.3,0.7},beginQ,beginQ).hit,
            "empty interval exits");
        bool rejected=false;
        try { model.sampleInterval(flight->mean,flight->start,{0.5,0.3,0.7},endQ,beginQ); }
        catch (const std::invalid_argument &) { rejected=true; }
        require(rejected,"reversed interval rejected");
        IndependentPathSampler sampler(819,false),reference(819,false);
        int hits=0;
        for (int i=0;i<12;++i) {
            sampler.startPath(0,i); reference.startPath(0,i);
            const auto high=reference.uniformGenerator().nextI(),low=reference.uniformGenerator().nextI();
            ConditionalGpis::Random rng((std::uint64_t(high)<<32)|low);
            const auto expected=medium.sample(*flight,rng);
            Medium::MediumState state{}; state.reset(); MediumSample actual{};
            require(adapter->sampleDistance(sampler,ray,state,actual),"adapter flight succeeded");
            require(actual.exited==!expected.hit,"adapter/reference event parity");
            if (!expected.hit) continue;
            ++hits;
            near(actual.t,(domain.entry+expected.distance)/eigen(direction).norm(),3e-6,"adapter world distance");
            require((eigen(actual.p)-expected.position).norm()<2e-6,"adapter hit position");
            require((V(actual.aniso.x(),actual.aniso.y(),actual.aniso.z())-expected.gradient).norm()<1e-8,
                "adapter/reference full gradient parity");
            PhaseSample phase;
            require(actual.phase->sample(sampler,direction,actual,phase),"sampled conductor reflection");
            near(phase.weight.x(),1,0,"unit Fresnel energy");
            require(bool(phase.pathContext),"B birth context is returned");
            state.pathContext=phase.pathContext;
            const Ray outgoing(actual.p,phase.w,0.01f,0.3f);
            const auto bFlight=medium.beginFlight({expected.position,eigen(phase.w).normalized()},expected.gradient);
            require(bool(bFlight),"B reference flight");
            const double expectedT=medium.transmittance(*bFlight,double(outgoing.farT())*eigen(phase.w).norm());
            Medium::MediumState branch=state;
            near(adapter->transmittance(sampler,outgoing,false,true,&branch).x(),expectedT,2e-6,
                "B transmittance retains full gradient and includes geometry nearT gap");
            require(state.pathContext==phase.pathContext,"shadow query leaves parent context unchanged");
            const Ray inward(actual.p,-phase.w,0.f,0.1f);
            near(adapter->transmittance(sampler,inward,false,true,&state).x(),0,0,"inward B visibility is zero");
        }
        require(hits>0,"collision coverage");
        const float endT=float((domain.entry+endQ*flight->ell)/eigen(direction).norm());
        const float splitT=float((domain.entry+beginQ*flight->ell)/eigen(direction).norm());
        Medium::MediumState whole{},split{}; whole.reset(); split.reset();
        const Ray wholeRay(origin,direction,0.f,endT),first(origin,direction,0.f,splitT);
        const double full=adapter->transmittance(sampler,wholeRay,false,true,&whole).x();
        const double front=adapter->transmittance(sampler,first,false,true,&split).x();
        const Ray tail(first.hitpoint(),direction,0.f,endT-splitT);
        const double back=adapter->transmittance(sampler,tail,true,true,&split).x();
        near(front*back,full,3e-6,"forward geometry preserves conditional survival");
        near(full,0.5,2e-5,"median visibility");
        rapidjson::Document serialized; auto value=adapter->toJson(serialized.GetAllocator());
        rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        value.Accept(writer);
        const auto roundTripSpec=nlohmann::json::parse(buffer.GetString());
        require(!roundTripSpec["field"].contains("sigma") && !roundTripSpec["material"].contains("roughness"),
            "serialization contains only explicit stationary covariance settings");
        auto roundTrip=parseMedium(roundTripSpec); roundTrip->loadResources(); roundTrip->prepareForRender();
        near(roundTrip->transmittance(sampler,wholeRay,false,true,nullptr).x(),full,0,"covariance JSON round trip");
        for (int parameter=0;parameter<2;++parameter) {
            const double s=parameter==0 ? 0.08 : sigma, ell=parameter==1 ? 0.07 : lengthScale;
            auto changed=spec; changed["covariance"]["sigma"]=s; changed["covariance"]["lengthScale"]=ell;
            auto configured=parseMedium(changed); configured->loadResources(); configured->prepareForRender();
            auto referenceField=field;
            referenceField.kernel=ConditionalGpis::CovarianceKernel::fromCorrelationLengths(s,V::Constant(ell));
            const NeuralGpis::RenewalSampler configuredReference(referenceField,model);
            const auto configuredFlight=configuredReference.beginFlight({entry,w});
            require(bool(configuredFlight),"configured reference flight");
            const double expected=configuredReference.transmittance(*configuredFlight,
                double(endT)*eigen(direction).norm()-domain.entry);
            const double actual=configured->transmittance(sampler,wholeRay,false,true,nullptr).x();
            near(actual,expected,2e-6,"manual sigma/lengthScale agrees with independent transport");
            require(std::abs(actual-full)>1e-4,"changing each covariance parameter affects transport");
        }
        int escaped=0; constexpr int count=128;
        for (int i=0;i<count;++i) {
            sampler.startPath(1,i); Medium::MediumState state{}; state.reset(); MediumSample s{};
            require(adapter->sampleDistance(sampler,wholeRay,state,s),"finite interval flight");
            require(s.t>=0 && s.t<=wholeRay.farT()+1e-6,"geometry endpoint caps collisions");
            near(s.weight.x(),1,0,"distance sampling does not double-count transmittance");
            escaped+=s.exited;
        }
        require(std::abs(double(escaped)/count-full)<0.15,"sampled escape agrees with visibility");
        Medium::MediumState empty{}; empty.reset(); MediumSample miss{};
        const Ray before(origin,direction,0.f,float(domain.entry*0.5));
        require(adapter->sampleDistance(sampler,before,empty,miss) && miss.exited,"geometry before SDF wins");
        near(adapter->transmittance(sampler,before,false,true,nullptr).x(),1,0,"outside SDF is vacuum");
        std::cout<<"PASS: shared model parity, bounded inversion, A/B gradients, geometry cutoff, "
            <<"forward survival, branch isolation, and visibility ("<<escaped<<"/"<<count<<")\n";
        return 0;
    } catch (const std::exception &e) { std::cerr<<e.what()<<"\n"; return 1; }
}
