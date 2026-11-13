#include "math/conditional/transport/SinglePointNeural.h"
#include "math/SparseHistoryGaussian.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <fstream>
#include <filesystem>
#include <cctype>
#include <limits>

namespace Tungsten::ConditionalGpis {
namespace {
using Json = nlohmann::json;
const std::array<std::string,6> FeatureNames{{"z_x","z_u","z_v","rho_xu","rho_xv","rho_uv"}};

double finiteNumber(const Json& value)
{
    if (!value.is_number()) throw std::runtime_error("single-point neural model expects a number");
    const double result=value.get<double>();
    if (!std::isfinite(result)) throw std::runtime_error("single-point neural model contains a nonfinite number");
    return result;
}

template<std::size_t N> std::array<double,N> numbers(const Json& value)
{
    if (!value.is_array() || value.size()!=N)
        throw std::runtime_error("single-point neural model array has the wrong size");
    std::array<double,N> result;
    for (std::size_t i=0;i<N;++i) result[i]=finiteNumber(value[i]);
    return result;
}

float finiteFloat(const Json& value)
{
    const float result=float(finiteNumber(value));
    if (!std::isfinite(result)) throw std::runtime_error("single-point neural weight is not representable as float32");
    return result;
}

// exp(x)-1-x without cancellation. The small branch needs only a short series
// at the supported minimum t/ell=.05 and u/ell=.01, rather than a dense feature
// projection with dozens of allocated entries used by the numerical teacher.
double remainder(double x)
{
    double term=.5*x*x, sum=term;
    for (int n=3;n<=32;++n) {
        term*=x/n;
        const double previous=sum;
        sum+=term;
        if (sum==previous) break;
    }
    return sum;
}

double valueVariance(double q)
{
    if (q>40) return 1;
    const double x=q*q, e=std::exp(-x);
    return x<1 ? e*remainder(x) : -std::expm1(-x)-x*e;
}

Matrix3 covariance(double q,double r)
{
    Matrix3 c=Matrix3::Zero();
    c(0,0)=valueVariance(q); c(1,1)=valueVariance(r);
    const double x=q*q;
    if (q>40) { c(2,2)=1; c(0,2)=c(2,0)=0; }
    else {
        const double e=std::exp(-x);
        c(2,2)=-std::expm1(-x)+x*(1-x)*e;
        c(0,2)=c(2,0)=-q*x*e;
    }
    const double qr=q*r, d=q-r;
    if (qr<1) {
        const double e=std::exp(-.5*(q*q+r*r)), h=remainder(qr);
        c(0,1)=c(1,0)=e*h;
        c(1,2)=c(2,1)=e*(q*h-r*std::expm1(qr));
    } else {
        const double e=std::exp(-.5*d*d);
        const double birth=q>40 ? 0 : std::exp(-.5*(q*q+r*r));
        c(0,1)=c(1,0)=e-(birth==0 ? 0 : (1+qr)*birth);
        c(1,2)=c(2,1)=d*e+(birth==0 ? 0 : (r-q*(1+qr))*birth);
    }
    return c;
}

double correlation(double cov,double a,double b)
{
    if (!(a>0 && b>0))
        throw NumericError(NumericStatus::NeedHigherPrecision,"single-point neural feature variance underflow");
    const double r=(cov/std::sqrt(a))/std::sqrt(b);
    if (!std::isfinite(r) || std::abs(r)>1+1e-10)
        throw NumericError(NumericStatus::InvalidCovariance,"single-point neural feature correlation invalid");
    return std::clamp(r,-1.0,1.0);
}

double logAdd(double a,double b)
{
    const double hi=std::max(a,b),lo=std::min(a,b);
    return std::isfinite(hi) ? hi+std::log1p(std::exp(lo-hi)) : hi;
}

void normalizeDescription(Json& description)
{
    if (description.is_object()) {
        description.erase("name");
        for (auto it=description.begin();it!=description.end();++it) {
            if (it.key()=="file" && it.value().is_string()) {
                std::string name=it.value().get<std::string>();
                std::replace(name.begin(),name.end(),'\\','/');
                name=std::filesystem::path(name).lexically_normal().generic_string();
#ifdef _WIN32
                std::transform(name.begin(),name.end(),name.begin(),[](unsigned char c){return char(std::tolower(c));});
#endif
                it.value()=name;
            } else normalizeDescription(it.value());
        }
    } else if (description.is_array()) for (auto& child:description) normalizeDescription(child);
}
}

std::vector<SinglePointHistoryAnchor> buildSinglePointHistoryAnchors(
    const GPSSField& field,const FlightState& birth,double horizon,double minimum,
    const ConditionalFlightKernel& base)
{
    if (!(horizon>0 && std::isfinite(horizon) && minimum>0 && std::isfinite(minimum)))
        throw NumericError(NumericStatus::InvalidInput,"invalid single-point anchor horizon or separation");
    std::vector<double> knots{0,horizon};
    field.mean->appendRayBreakpoints(birth.birthPosition,birth.direction,0,horizon,knots);
    std::sort(knots.begin(),knots.end());
    knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
    if (knots.size()>1000000) throw std::runtime_error("single-point SDF ray has more than one million cell boundaries");
    std::vector<double> locations;
    auto cursor=field.mean->createRayCursor(birth.birthPosition,birth.direction);
    for (std::size_t i=1;i<knots.size();++i) {
        const double a=knots[i-1],b=knots[i];
        if (!(b>a)) continue;
        locations.push_back(a); locations.push_back(a+.5*(b-a));
        if (cursor) cursor->appendStationaryPoints(a,b,locations);
    }
    field.mean->appendHistoryCandidatePoints(birth.birthPosition,birth.direction,0,horizon,locations);
    std::sort(locations.begin(),locations.end());
    locations.erase(std::unique(locations.begin(),locations.end()),locations.end());
    std::vector<SinglePointHistoryAnchor> result;
    for (double t:locations) {
        if (!(t>=minimum && t<horizon)) continue;
        const auto s=base.scalarStatistics(t);
        if (!(s.varianceF>0)) continue;
        const double z=s.meanF/std::sqrt(s.varianceF);
        if (!std::isfinite(z)) continue;
        result.push_back({t,z,normalPdfOverCdf(z),
            field.mean->evaluate(birth.birthPosition+t*birth.direction).value});
    }
    return result;
}

double selectSinglePointHistoryAnchor(const std::vector<SinglePointHistoryAnchor>& anchors,
    double t,double ell,double separation)
{
    const SparseHistoryGpis::HistoryCandidateScorer scorer(t/ell,true);
    double u=.5*t,best=-1;
    for (const auto& anchor:anchors) {
        if (anchor.t>=t-separation) break;
        const double score=scorer.score(anchor.t/ell,anchor.mills);
        if (score>best) { best=score;u=anchor.t; }
    }
    return u;
}

std::array<double,6> singlePointNeuralFeatures(const GPSSField& field,const FlightState& birth,
    const ConditionalFlightKernel& baseline,double t,double u)
{
    if (!(u>0 && t>u && std::isfinite(t)))
        throw NumericError(NumericStatus::InvalidInput,"single-point neural features require 0 < u < t");
    const double sigma=field.kernel.sigma();
    const double inverseLength=std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
    const auto current=baseline.scalarStatistics(t),history=baseline.scalarStatistics(u);
    const Matrix3 c=covariance(t*inverseLength,u*inverseLength);
    const Vector3 means(current.meanF/sigma,history.meanF/sigma,-current.meanK/(sigma*inverseLength));
    std::array<double,6> result;
    for (int i=0;i<3;++i) {
        if (!(c(i,i)>0))
            throw NumericError(NumericStatus::NeedHigherPrecision,"single-point neural feature variance underflow");
        result[i]=means[i]/std::sqrt(c(i,i));
    }
    result[3]=correlation(c(0,1),c(0,0),c(1,1));
    result[4]=correlation(c(0,2),c(0,0),c(2,2));
    result[5]=correlation(c(1,2),c(1,1),c(2,2));
    for (double x:result) if (!std::isfinite(x))
        throw NumericError(NumericStatus::NeedHigherPrecision,"nonfinite single-point neural feature");
    return result;
}

std::shared_ptr<const SinglePointNeuralModel> SinglePointNeuralModel::load(
    const std::string& filename,std::optional<double> capOverride)
{
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Cannot open single-point neural model: "+filename);
    Json j; file>>j;
    if (j.at("schema")!="tungsten.single_point_history_mlp.v2" ||
        j.at("feature_order").get<std::array<std::string,6>>()!=FeatureNames ||
        j.at("mlp_dtype")!="float32" || j.at("probability_transform_dtype")!="float64")
        throw std::runtime_error("Unsupported single-point neural model schema, precision, or feature order");
    const auto& a=j.at("architecture");
    const unsigned width=a.at("width").get<unsigned>();
    if (a.at("input_features")!=6 || a.at("output_features")!=1 ||
        a.at("hidden_layers")!=2 || a.at("activation")!="relu" || width<1 || width>256)
        throw std::runtime_error("Unsupported single-point neural architecture");
    if (j.at("output").at("quantity")!="log_hazard_correction" ||
        j.at("output").at("raw_quantity")!="residual_f" ||
        j.at("deployment").at("stage")!="inference_only" ||
        j.at("deployment").at("applied_to_training_loss")!=false ||
        !j.at("deployment").at("lower_cap").is_null())
        throw std::runtime_error("Unsupported single-point neural output/deployment policy");
    auto model=std::shared_ptr<SinglePointNeuralModel>(new SinglePointNeuralModel);
    const std::string transform=j.value("feature_transform",std::string("identity"));
    if (transform!="identity" && transform!="asinh_z")
        throw std::runtime_error("Unsupported single-point neural feature transform");
    model->asinhZ_=transform=="asinh_z";
    const auto mean=numbers<6>(j.at("feature_mean")),scale=numbers<6>(j.at("feature_scale"));
    for (int i=0;i<6;++i) {
        model->mean_[i]=float(mean[i]);model->scale_[i]=float(scale[i]);
        if (!std::isfinite(model->mean_[i]) || !(model->scale_[i]>0) || !std::isfinite(model->scale_[i]) ||
            (i>=3 && (mean[i]!=0 || scale[i]!=1)))
            throw std::runtime_error("Invalid single-point neural feature normalization");
    }
    const auto& layers=j.at("layers");
    if (!layers.is_array() || layers.size()!=5 || layers[1].at("type")!="relu" || layers[3].at("type")!="relu")
        throw std::runtime_error("Invalid single-point neural layer sequence");
    const unsigned sizes[4]={6,width,width,1};
    for (int k=0;k<3;++k) {
        const auto& source=layers[2*k];auto& layer=model->layers_[k];
        layer.in=sizes[k];layer.out=sizes[k+1];
        if (source.at("type")!="linear" || source.at("weights").size()!=layer.out || source.at("bias").size()!=layer.out)
            throw std::runtime_error("Invalid single-point neural linear layer dimensions");
        for (unsigned row=0;row<layer.out;++row) {
            const auto& weights=source.at("weights")[row];
            if (!weights.is_array() || weights.size()!=layer.in)
                throw std::runtime_error("Invalid single-point neural weight row size");
            for (const auto& weight:weights) layer.weights.push_back(finiteFloat(weight));
            layer.bias.push_back(finiteFloat(source.at("bias")[row]));
        }
    }
    model->sigma_=finiteNumber(j.at("sigma"));model->ell_=finiteNumber(j.at("length_scale"));
    if (!(model->sigma_>0 && model->ell_>0)) throw std::runtime_error("Invalid single-point neural covariance parameters");
    model->cap_=capOverride ? *capOverride : finiteNumber(j.at("deployment").at("correction_cap"));
    if (!(model->cap_>=1 && std::isfinite(model->cap_)))
        throw std::runtime_error("Single-point neural rendering requires a finite correction cap >= 1");
    const auto& manifest=j.at("source_manifest");
    if (manifest.at("schema")!="tungsten.single_point_history.v1" || manifest.at("status")!="complete" ||
        finiteNumber(manifest.at("sigma"))!=model->sigma_ || finiteNumber(manifest.at("length_scale"))!=model->ell_)
        throw std::runtime_error("Single-point model has inconsistent source metadata");
    if (manifest.at("sampling").at("anchor_implementation")!="shared_geometry_pool_v2")
        throw std::runtime_error("Single-point model uses an incompatible history anchor policy");
    const auto& source=manifest.at("source");
    if (source.contains("source_medium") && source.at("source_medium").contains("gaussian_process") &&
        source.at("source_medium").at("gaussian_process").contains("mean")) {
        Json mean=source.at("source_medium").at("gaussian_process").at("mean");
        normalizeDescription(mean);model->meanDescription_=mean.dump();
    }
    const auto lo=numbers<3>(manifest.at("source").at("domain").at("minimum"));
    const auto hi=numbers<3>(manifest.at("source").at("domain").at("maximum"));
    for (int i=0;i<3;++i) {model->domain_.minimum[i]=lo[i];model->domain_.maximum[i]=hi[i];}
    if (!model->domain_.valid()) throw std::runtime_error("Single-point model has invalid training bounds");
    model->minimumAge_=finiteNumber(manifest.at("sampling").at("minimum_age_over_length"));
    model->minimumSeparation_=finiteNumber(manifest.at("sampling").at("minimum_separation_over_length"));
    if (!(model->minimumAge_>0 && model->minimumSeparation_>0))
        throw std::runtime_error("Single-point model has invalid history spacing policy");
    model->minimumFeatures_=numbers<6>(j.at("training_domain").at("minimum_features"));
    model->maximumFeatures_=numbers<6>(j.at("training_domain").at("maximum_features"));
    for (int i=0;i<6;++i) if (model->minimumFeatures_[i]>model->maximumFeatures_[i])
        throw std::runtime_error("Single-point model has invalid feature ranges");
    return model;
}

void SinglePointNeuralModel::validateField(const GPSSField& field) const
{
    const double expected=1/(ell_*ell_);
    if (std::abs(field.kernel.sigma()-sigma_)>1e-7*sigma_ ||
        (field.kernel.precision()-expected*Matrix3::Identity()).cwiseAbs().maxCoeff()>1e-7*expected)
        throw std::runtime_error("Single-point model requires its training sigma and isotropic length scale");
}

void SinglePointNeuralModel::validateMeanDescription(const std::string& serialized) const
{
    if (meanDescription_.empty())
        throw std::runtime_error("Single-point model lacks a native GP mean description for renderer validation");
    auto mean=Json::parse(serialized);normalizeDescription(mean);
    if (mean!=Json::parse(meanDescription_))
        throw std::runtime_error("Single-point model SDF mean differs from its training scene");
}

SinglePointNeuralEvaluation SinglePointNeuralModel::evaluate(const std::array<double,6>& features) const
{
    SinglePointNeuralEvaluation result;
    for (int i=0;i<6;++i) {
        if (!std::isfinite(features[i])) throw std::runtime_error("Nonfinite single-point neural input");
        if (features[i]<minimumFeatures_[i] || features[i]>maximumFeatures_[i]) result.outsideFeatureRange=true;
    }
    const double logp=normalLogCdf(features[1]);
    if (logp==0) return result; // The analytic architecture is exactly R=1.
    std::array<float,256> a{},b{};
    for (int i=0;i<6;++i) {
        const float value=float(features[i]);
        a[i]=((asinhZ_ && i<3 ? std::asinh(value) : value)-mean_[i])/scale_[i];
        if (!std::isfinite(a[i])) throw std::runtime_error("Single-point neural normalized input overflow");
    }
    for (int k=0;k<3;++k) {
        const auto& layer=layers_[k];
        for (unsigned i=0;i<layer.out;++i) {
            float value=layer.bias[i];
            for (unsigned n=0;n<layer.in;++n) value+=layer.weights[i*layer.in+n]*a[n];
            if (!std::isfinite(value)) throw std::runtime_error("Single-point neural float32 inference overflow");
            b[i]=k<2 ? std::max(0.0f,value) : value;
        }
        std::swap(a,b);
    }
    const double log1mp=logp < -std::log(2.) ? std::log1p(-std::exp(logp)) : std::log(-std::expm1(logp));
    const double raw=-logAdd(logp,log1mp-double(a[0]));
    result.networkEvaluated=true;result.capped=raw>std::log(cap_);
    result.logCorrection=std::min(raw,std::log(cap_));
    result.ratio=std::min(cap_,std::exp(result.logCorrection));
    if (!(result.ratio>=0 && std::isfinite(result.ratio))) throw std::runtime_error("Nonfinite single-point neural correction");
    return result;
}

SinglePointNeuralFlight::SinglePointNeuralFlight(const SinglePointNeuralModel& model,const GPSSField& field,
    const FlightState& birth,const ConditionalFlightKernel& baseline)
    : model_(model),field_(field),birth_(birth)
{
    const double tolerance=1e-6*std::max(model.lengthScale(),(model.domain().maximum-model.domain().minimum).norm());
    const auto domain=model.domain().intersect({birth.birthPosition,birth.direction});
    outsideDomain_=!model.domain().contains(birth.birthPosition,tolerance) || !domain.hit || domain.entry>tolerance;
    if (outsideDomain_) return;
    horizon_=std::min(domain.exit,baseline.maximumAgeInDomain());
    if (!(horizon_>0 && std::isfinite(horizon_))) {outsideDomain_=true;return;}
    anchors_=buildSinglePointHistoryAnchors(field,birth,horizon_,model.minimumSeparationOverLength()*model.lengthScale(),baseline);
}

SinglePointNeuralEvaluation SinglePointNeuralFlight::correction(double age,const ConditionalFlightKernel& baseline) const
{
    SinglePointNeuralEvaluation result;
    if (outsideDomain_ || age>horizon_) {result.outsideDomain=true;return result;}
    if (age<std::max(model_.minimumAgeOverLength(),2*model_.minimumSeparationOverLength())*model_.lengthScale()) {
        result.beforeMinimumAge=true;return result;
    }
    const double u=selectSinglePointHistoryAnchor(anchors_,age,model_.lengthScale(),model_.minimumSeparationOverLength()*model_.lengthScale());
    return model_.evaluate(singlePointNeuralFeatures(field_,birth_,baseline,age,u));
}

} // namespace Tungsten::ConditionalGpis
