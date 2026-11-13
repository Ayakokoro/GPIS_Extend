#include "math/conditional/transport/SinglePointNeural.h"
#include "math/conditional/transport/SinglePointHistoryTeacher.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "SinglePointNeuralFixture.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <chrono>

namespace cg=Tungsten::ConditionalGpis;
using V=Eigen::Vector3d;
namespace {
void check(bool value,const char* message) {if (!value) throw std::runtime_error(message);}
void close(double a,double b,double eps,const char* message) {
    if (!(std::abs(a-b)<=eps)) {std::cerr<<message<<": "<<a<<" versus "<<b<<'\n';throw std::runtime_error(message);}
}
std::filesystem::path temporary;
std::shared_ptr<const cg::SinglePointNeuralModel> load(const nlohmann::json& j) {
    std::ofstream(temporary)<<j.dump();
    return cg::SinglePointNeuralModel::load(temporary.string());
}
cg::GPSSField field() {
    return {std::make_shared<cg::PlaneMean>(V(.2,.1,1),.1),
        cg::CovarianceKernel::fromCorrelationLengths(.02,V::Constant(.2)),{V::Constant(-1),V::Constant(1)}};
}
void featuresMatchNumericalTeacher() {
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(0);
    f.activeDomain={V::Constant(-10),V::Constant(10)};
    cg::FlightState b;b.direction=V(.2,-.3,1).normalized();
    cg::SinglePointHistoryTeacher teacher(f,b);
    // Compare all six values across the near-birth series and closed-form
    // branches, including a nonzero complete birth observation.
    for (double q:{.05,.1,.4,1.,1.4,2.,4.,9.}) for (double fraction:{.2,.5,.9}) {
        const double t=.2*q,u=t*fraction;
        const auto actual=cg::singlePointNeuralFeatures(f,b,teacher.baseline(),t,u);
        const auto expected=teacher.evaluate(t,u).features;
        for (int i=0;i<6;++i) close(actual[i],expected[i],2e-9*std::max(1.,std::abs(expected[i])),"cheap features differ from teacher");
    }
    b.birthValue=.002;b.birthGradient=V(.002,-.001,.003);
    cg::SinglePointHistoryTeacher nonzero(f,b);
    for (double q:{.4,1.,2.,4.}) {
        const double t=.2*q,u=.5*t;
        const auto actual=cg::singlePointNeuralFeatures(f,b,nonzero.baseline(),t,u);
        const auto expected=nonzero.evaluate(t,u).features;
        for (int i=0;i<6;++i) close(actual[i],expected[i],2e-9*std::max(1.,std::abs(expected[i])),"full birth features differ from teacher");
    }
}
void inferenceAndValidation() {
    auto j=singlePointNeuralFixture();auto model=load(j);
    model->validateField(field());
    for (double z:{-12.,-2.,0.,2.,12.}) {
        const auto result=model->evaluate({.1,z,-.2,.2,.3,.4});
        close(result.ratio,1,1e-14,"zero residual must reproduce baseline");
    }
    j["layers"][4]["bias"][0]=10.;model=load(j);
    auto positive=model->evaluate({0,-4,0,.2,.3,.4});
    check(positive.capped,"positive residual should activate cap");close(positive.ratio,2,0,"cap exact");
    j["layers"][4]["bias"][0]=-3.;model=load(j);
    const auto negative=model->evaluate({0,-2,0,.2,.3,.4});
    const double p=std::exp(cg::normalLogCdf(-2));
    close(negative.ratio,1/(p+(1-p)*std::exp(3.)),1e-14,"no lower clipping");
    check(negative.ratio<.1,"negative residual should retain a small correction");
    j=singlePointNeuralFixture();j["feature_transform"]="asinh_z";
    j["layers"][0]["weights"][0][0]=1.;j["layers"][2]["weights"][0][0]=1.;j["layers"][4]["weights"][0][0]=1.;
    model=load(j);
    const auto transformed=model->evaluate({2,-1,0,.2,.3,.4});
    const double f=double(std::asinh(2.0f)),p1=std::exp(cg::normalLogCdf(-1));
    close(transformed.ratio,std::min(2.,1/(p1+(1-p1)*std::exp(-f))),1e-14,"float32 asinh transform");
    auto wrong=field();wrong.kernel=cg::CovarianceKernel::fromCorrelationLengths(.03,V::Constant(.2));
    bool caught=false;try {model->validateField(wrong);} catch(const std::exception&) {caught=true;}
    check(caught,"mismatched sigma must reject");
    j["deployment"]["correction_cap"]=.5;
    caught=false;try {load(j);} catch(const std::exception&) {caught=true;}
    check(caught,"cap below one must reject");
    j=singlePointNeuralFixture();j["feature_order"][0]="other";
    caught=false;try {load(j);} catch(const std::exception&) {caught=true;}
    check(caught,"wrong feature order must reject");
    j=singlePointNeuralFixture();
    j["source_manifest"]["source"]["source_medium"]["gaussian_process"]["mean"]=
        {{"type","procedural"},{"func","two_slabs"},{"name","training"}};
    model=load(j);
    model->validateMeanDescription(R"({"type":"procedural","func":"two_slabs","name":"render"})");
    caught=false;try {model->validateMeanDescription(R"({"type":"procedural","func":"sphere"})");}
    catch(const std::exception&) {caught=true;}
    check(caught,"different scene mean must reject");
}
void historyAndDomain() {
    auto f=field();auto model=load(singlePointNeuralFixture());
    cg::FlightState b;b.direction=V::UnitZ();b.birthValue=.2;b.birthGradient=V(.2,.1,1.);
    cg::ConditionalFlightKernel baseline(f,b);cg::SinglePointNeuralFlight flight(*model,f,b,baseline);
    close(flight.horizon(),1,1e-15,"canonical training bounds horizon");
    check(flight.correction(.001,baseline).beforeMinimumAge,"near-birth fallback must be explicit");
    check(flight.correction(1.01,baseline).outsideDomain,"outside training domain fallback must be explicit");
    for (double t:{.02,.1,.3,.7}) {
        const double u=cg::selectSinglePointHistoryAnchor(flight.anchors(),t,.2,.002);
        check(u>0 && u<t,"selected history must precede current query");
        const auto a=flight.correction(t,baseline),c=flight.correction(t,baseline);
        close(a.ratio,c.ratio,0,"history must not depend on previous queries");
    }
}

void predict(const std::string& modelPath,const std::string& csvPath,const std::string& outputPath) {
    auto model=cg::SinglePointNeuralModel::load(modelPath);
    std::ifstream input(csvPath);std::ofstream output(outputPath);
    if (!input || !output) throw std::runtime_error("Cannot open parity input/output");
    std::string row;std::getline(input,row);
    output<<"log_correction,capped,outside_feature_range\n"<<std::setprecision(17);
    while (std::getline(input,row)) {
        std::stringstream stream(row);std::string token;std::array<double,6> features;
        for (int column=0;column<9;++column) {
            if (!std::getline(stream,token,',')) throw std::runtime_error("Invalid parity CSV row");
            if (column>=3) features[column-3]=std::stod(token);
        }
        const auto r=model->evaluate(features);output<<r.logCorrection<<','<<r.capped<<','<<r.outsideFeatureRange<<'\n';
    }
}
}
int main(int argc,char** argv) {
    try {
        if (argc==5 && std::string(argv[1])=="--predict") {predict(argv[2],argv[3],argv[4]);return 0;}
        temporary=std::filesystem::temp_directory_path()/("single_point_neural_"+
            std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count())+".json");
        featuresMatchNumericalTeacher();inferenceAndValidation();historyAndDomain();
        std::filesystem::remove(temporary);std::cout<<"Single-point neural runtime tests passed\n";return 0;
    } catch(const std::exception& e) {
        if (!temporary.empty()) std::filesystem::remove(temporary);
        std::cerr<<e.what()<<'\n';return 1;
    }
}
