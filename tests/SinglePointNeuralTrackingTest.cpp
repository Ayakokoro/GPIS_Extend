#include "math/conditional/transport/ConditionalSegment.h"
#include "math/conditional/transport/SinglePointNeural.h"
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "math/SdfFunctions.hpp"
#include "math/GPFunctions.hpp"
#include "media/GlobalConditionalMedium.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "SinglePointNeuralFixture.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace cg=Tungsten::ConditionalGpis;
using V=Eigen::Vector3d;
using J=nlohmann::json;
namespace {
void check(bool value,const char *message) { if(!value) throw std::runtime_error(message); }
void close(double a,double b,double tolerance,const char *message) {
    if(!(std::abs(a-b)<=tolerance)) {
        std::cerr<<message<<": "<<a<<" vs "<<b<<'\n';
        throw std::runtime_error(message);
    }
}
void geometryCandidates() {
    const auto mean=std::make_shared<Tungsten::ProceduralMean>(Tungsten::SdfFunctions::Function::TwoSlabs);
    cg::SdfMeanField field(mean);
    for(double speed:{1.,2.}) {
        std::vector<double> candidates;
        field.appendHistoryCandidatePoints(V(0,.3,0),V(0,-speed,0),0,.6/speed,candidates);
        check(candidates.size()==3,"two slabs expose precisely the two medial minima and gap maximum");
        for(int i=0;i<3;++i) close(candidates[i],(.28+.02*i)/speed,2e-15,"history geometry keeps physical ray parameter");
        std::vector<double> unchanged;
        field.appendRayBreakpoints(V(0,.3,0),V(0,-speed,0),0,.6/speed,unchanged);
        check(unchanged.empty(),"history geometry must not alter baseline majorant knots");
    }
}

void tracking(const std::filesystem::path &filename) {
    auto fixture=singlePointNeuralFixture(1,1,20);
    fixture["deployment"]["correction_cap"]=1.5;
    fixture["source_manifest"]["source"]["domain"]={{"minimum",{-20,-20,-20}},{"maximum",{20,20,20}}};
    std::ofstream(filename)<<fixture.dump();
    const auto model=cg::SinglePointNeuralModel::load(filename.string());
    cg::GPSSField field{std::make_shared<cg::ConstantMean>(0),cg::CovarianceKernel(1,Eigen::Matrix3d::Identity()),
        {V::Constant(-20),V::Constant(20)}};
    cg::FlightState birth;
    birth.birthValue=.3;birth.birthGradient=V::UnitZ();birth.direction=V::UnitZ();
    cg::ConditionalTrackingSettings settings;
    settings.shadowRouletteThreshold=0;
    settings.maxExpectedEvents=.4;
    const double age=4,length=3;
    const cg::ConditionalSegment baseline(field,birth,age*V::UnitZ(),V::UnitZ(),length,settings);
    const cg::ConditionalSegment neural(field,birth,age*V::UnitZ(),V::UnitZ(),length,settings,model.get());
    double integral=0;
    constexpr int integrationPoints=12000;
    for(int i=0;i<integrationPoints;++i) {
        const double t=length*(i+.5)/integrationPoints;
        const double h=neural.hazard(t),base=baseline.hazard(t);
        check(h<=1.5*base*(1+1e-12),"runtime hazard cap");
        integral+=h*length/integrationPoints;
    }
    close(neural.majorant(0,length),1.5*baseline.majorant(0,length),1e-12,"public scaled majorant");
    for(double t:{.1,.7,2.9}) {
        const auto a=baseline.gradient(t),b=neural.gradient(t);
        check(a.mean==b.mean && a.covariance==b.covariance,"neural extinction leaves collision gradient distribution unchanged");
    }
    const cg::ConditionalSegment split(field,birth,(age+.7)*V::UnitZ(),V::UnitZ(),length-.7,settings,model.get());
    close(neural.hazard(1.2),split.hazard(.5),1e-12,"straight subdivision keeps the same history anchor and hazard");
    const double expected=std::exp(-integral);
    for(auto mode:{cg::ConditionalMajorantMode::Adaptive,cg::ConditionalMajorantMode::TwoConstants}) {
        settings.majorantMode=mode;settings.constantBirthLength=4.8;
        const cg::ConditionalSegment segment(field,birth,age*V::UnitZ(),V::UnitZ(),length,settings,model.get());
        for(bool collision:{false,true}) {
            cg::Random rng(collision ? 681 : 927);
            double sum=0,sum2=0;
            std::uint64_t calls=0,caps=0;
            constexpr int count=5000;
            for(int i=0;i<count;++i) {
                double hit;cg::ConditionalTrackingWork work;
                const double weight=segment.track(rng,collision,hit,&work);
                sum+=weight;sum2+=weight*weight;
                calls+=work.networkQueries;caps+=work.correctionCapHits;
            }
            const double variance=std::max(0.,(sum2-sum*sum/count)/(count-1));
            close(sum/count,expected,6*std::sqrt(variance/count)+.0003,
                "scaled candidate rates preserve integrated collision and ratio transmittance");
            check(calls>0 && caps>0,"tracking must execute the network and exercise the configured cap");
        }
    }
}

void factory(const std::filesystem::path &filename) {
    Tungsten::Scene scene;
    J config={{"type","global_conditional_neural"},
        {"single_point_neural",{{"model",filename.generic_string()},{"correction_cap",1.5}}},
        {"gaussian_process",{{"type","standard"},{"mean",{{"type","homogeneous"},{"offset",0}}},
            {"covariance",{{"type","squared_exponential"},{"sigma",1},{"lengthScale",1}}}}}};
    Tungsten::JsonDocument document(Tungsten::Path("single-point-runtime-test.json"),config.dump());
    const auto medium=std::dynamic_pointer_cast<Tungsten::GlobalConditionalMedium>(scene.fetchMedium(document));
    check(medium && medium->neuralCorrectionEnabled() && !medium->gradientCorrectionEnabled(),
        "new factory mode uses existing conditional collision/phase implementation");
    rapidjson::Document owner;
    const auto serialized=medium->toJson(owner.GetAllocator());
    check(std::string(serialized["type"].GetString())=="global_conditional_neural" &&
        serialized["single_point_neural"]["correction_cap"].GetDouble()==1.5,
        "new mode and inference cap serialize independently of global_conditional");
}
}
int main() {
    const auto filename=std::filesystem::temp_directory_path()/
        ("single_point_neural_tracking_"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".json");
    try {
        geometryCandidates();tracking(filename);factory(filename);
        std::filesystem::remove(filename);
        std::cout<<"single-point neural tracking, gradient invariance and history geometry passed\n";
        return 0;
    } catch(const std::exception &error) {
        std::filesystem::remove(filename);
        std::cerr<<error.what()<<'\n';return 1;
    }
}
