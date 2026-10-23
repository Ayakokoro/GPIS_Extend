// Native inference for the macrofacet export-joint contract; see docs/neural-renewal.md.
#include "CausalRenewalModel.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <stdexcept>

namespace Tungsten::NeuralGpis {
namespace {
using ConditionalGpis::normalLogPdf;
using ConditionalGpis::normalLogCdf;
using ConditionalGpis::normalPdfOverCdf;
using ConditionalGpis::normalQuantileFromLogCdf;
struct DenseWeights {
    int input = 0, output = 0;
    std::vector<float> weight, bias; // output x input, row major
};
struct NetworkWeights {
    int hidden = 0, embedding = 0, components = 0;
    DenseWeights initial0,initial2,segment0,segment2,head0,head2;
    DenseWeights mixture0,mixture2,mixture4,gruInput,gruHidden;
};
using Json=nlohmann::json;
using Vec=Eigen::VectorXf;
using Mat=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
inline double logAdd(double a,double b) {
    const double m=std::max(a,b);
    return std::isinf(m) ? m : m+std::log(std::exp(a-m)+std::exp(b-m));
}
inline double logLocalRate(double b,double d) {
    const double phi=normalLogPdf(d);
    double negativeMean;
    if (d<=0) negativeMean=logAdd(phi,d<0 ? std::log(-d)+normalLogCdf(-d) : -INFINITY);
    else if (d<=26) negativeMean=phi+std::log1p(-d*std::exp(normalLogCdf(-d)-phi));
    else {
        const double v=1/(d*d);
        negativeMean=phi+std::log(v)+std::log(1+v*(-3+v*(15+v*(-105+v*(945+v*(-10395+v*135135))))));
    }
    return (b < -8 ? std::log(normalPdfOverCdf(b)) : normalLogPdf(b)-normalLogCdf(b))+negativeMean;
}
inline const RayMeanSegment& segmentAt(const RayMeanProfile& p,double q) {
    const auto& s=p.segments();
    auto it=std::lower_bound(s.begin(),s.end(),q,[](const auto& a,double x){return a.end<x;});
    return it==s.end() ? s.back() : *it;
}
inline void validateProfile(const RayMeanProfile& p) {
    if (p.segments().empty() || !(p.maximumX()>0) || !std::isfinite(p.maximumX()))
        throw std::invalid_argument("invalid causal renewal profile");
    double value=p.segments().front().value(0),end=0;
    for (const auto& s:p.segments()) {
        if (std::abs(s.begin-end)>1e-10 || !(s.end>s.begin) ||
            !std::isfinite(s.value(s.begin)) || !std::isfinite(s.value(s.end)) ||
            !std::isfinite(s.derivative(s.begin)) || !std::isfinite(s.derivative(s.end)) ||
            std::abs(s.value(s.begin)-value)>1e-8*(1+std::abs(value)))
            throw std::invalid_argument("causal renewal requires a continuous cubic profile from q=0");
        value=s.value(s.end); end=s.end;
    }
}
// Eight-point Gauss-Legendre. Every integration fragment stays inside a cubic
// and an integration-grid cell; reducing root tolerance never advances the GRU.
inline double integral(const RayMeanProfile& p,int sign,double a,double b) {
    static constexpr double x[]={.1834346424956498,.5255324099163290,.7966664774136267,.9602898564975363};
    static constexpr double w[]={.3626837833783620,.3137066458778873,.2223810344533745,.1012285362903763};
    const double mid=(a+b)/2,half=(b-a)/2;
    if (!(half>0)) return 0;
    const auto& segment=segmentAt(p,mid);
    double sum=0;
    for (int j=0;j<4;++j) for (int direction:{-1,1}) {
        const double q=mid+direction*half*x[j];
        sum+=w[j]*std::exp(logLocalRate(sign*segment.value(q),sign*segment.derivative(q)));
    }
    if (!std::isfinite(sum)) throw std::runtime_error("nonfinite causal local hazard");
    return half*sum;
}
struct Fragment { double a,b,mass; };
inline std::vector<Fragment> fragments(const RayMeanProfile& p,int sign,double a,double b,double fine) {
    std::vector<double> knots{a,b};
    for (double i=std::floor(a/fine)+1;i*fine<b;++i) if (i*fine>a) knots.push_back(i*fine);
    const auto& segments=p.segments();
    auto it=std::upper_bound(segments.begin(),segments.end(),a,[](double q,const auto& s){return q<s.end;});
    for (;it!=segments.end() && it->end<b;++it) if (it->end>a) knots.push_back(it->end);
    std::sort(knots.begin(),knots.end());
    knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
    std::vector<Fragment> result;
    for (std::size_t i=1;i<knots.size();++i) result.push_back({knots[i-1],knots[i],integral(p,sign,knots[i-1],knots[i])});
    return result;
}
inline double total(const std::vector<Fragment>& f) {
    double sum=0; for (const auto& v:f) sum+=v.mass; return sum;
}
inline void appendLeft(std::vector<float>& v,const RayMeanProfile& p,int sign,double q) {
    const auto& s=segmentAt(p,q);
    v.push_back(static_cast<float>(std::asinh(sign*s.value(q))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(q))));
    v.push_back(static_cast<float>(std::log1p(q)));
}
inline void appendToken(std::vector<float>& v,const RayMeanProfile& p,int sign,double a,double b) {
    for (double q:{a,(a+b)/2,b}) v.push_back(static_cast<float>(std::asinh(sign*segmentAt(p,q).value(q))));
    for (double q:{a,(a+b)/2,b}) v.push_back(static_cast<float>(std::asinh(sign*segmentAt(p,q).derivative(q))));
    v.push_back(static_cast<float>(std::log1p(b-a)));
}
inline void appendLocal(std::vector<float>& v,const RayMeanProfile& p,int sign,double q,double left,double step) {
    v.push_back(static_cast<float>(std::clamp((q-left)/step,0.,1.)));
    v.push_back(static_cast<float>(std::log1p(q)));
    const auto& s=segmentAt(p,q);
    v.push_back(static_cast<float>(std::asinh(sign*s.value(q))));
    v.push_back(static_cast<float>(std::asinh(sign*s.derivative(q))));
}
inline DenseWeights dense(const Json& root,const std::string& name,int input,int output,
    const std::string& weightSuffix=".weight",const std::string& biasSuffix=".bias") {
    if (input<1 || input>4096 || output<1 || output>4096) throw std::invalid_argument("invalid causal network size");
    const auto& w=root.at(name+weightSuffix); const auto& b=root.at(name+biasSuffix);
    DenseWeights result{input,output,w.at("values").get<std::vector<float>>(),b.at("values").get<std::vector<float>>()};
    if (w.at("shape")!=Json::array({output,input}) || b.at("shape")!=Json::array({output}) ||
        result.weight.size()!=static_cast<std::size_t>(input)*output || result.bias.size()!=static_cast<std::size_t>(output))
        throw std::invalid_argument("causal tensor shape mismatch: "+name);
    for (const auto* values:{&result.weight,&result.bias}) for (float x:*values)
        if (!std::isfinite(x)) throw std::invalid_argument("nonfinite causal weight");
    return result;
}
inline Vec linear(const DenseWeights& w,const Vec& x) {
    return Eigen::Map<const Mat>(w.weight.data(),w.output,w.input)*x+Eigen::Map<const Vec>(w.bias.data(),w.output);
}
inline Vec silu(Vec x) { for (int i=0;i<x.size();++i) x[i]=x[i]/(1.f+std::exp(-x[i])); return x; }
inline float softplus(float x) { return x>20.f ? x : std::log1p(std::exp(x)); }
// One recurrent state per flight; query batches share this entering state.
class NetworkState {
    const NetworkWeights& w_;
    Vec state_;
    Vec input(const std::vector<float>& local,std::size_t row,int columns) const {
        Vec v(w_.hidden+columns); v.head(w_.hidden)=state_;
        v.tail(columns)=Eigen::Map<const Vec>(local.data()+row*columns,columns); return v;
    }
public:
    explicit NetworkState(const NetworkWeights& w):w_(w) {}
    void initialize(const std::vector<float>& entry) {
        if (entry.size()!=static_cast<std::size_t>(w_.initial0.input))
            throw std::invalid_argument("invalid causal renewal entry size");
        state_=linear(w_.initial2,silu(linear(w_.initial0,
            Eigen::Map<const Vec>(entry.data(),w_.initial0.input))));
    }
    float correction(const std::vector<float>& left) const {
        return linear(w_.head2,silu(linear(w_.head0,input(left,0,3))))[0];
    }
    void advance(const std::vector<float>& token) {
        const Vec embedded=silu(linear(w_.segment2,silu(linear(w_.segment0,Eigen::Map<const Vec>(token.data(),7)))));
        const Vec a=linear(w_.gruInput,embedded),b=linear(w_.gruHidden,state_);
        auto& h=state_;
        // PyTorch r/z/n ordering, reset gate applied after the hidden affine.
        for (int j=0;j<w_.hidden;++j) {
            const float r=1.f/(1.f+std::exp(-a[j]-b[j]));
            const float z=1.f/(1.f+std::exp(-a[j+w_.hidden]-b[j+w_.hidden]));
            const float n=std::tanh(a[j+2*w_.hidden]+r*b[j+2*w_.hidden]);
            h[j]=(1-z)*n+z*h[j];
        }
        if (!h.allFinite()) throw std::runtime_error("nonfinite causal GRU state");
    }
    std::vector<float> mixture(const std::vector<float>& local) const {
        std::vector<float> out;
        for (std::size_t i=0;i<local.size()/4;++i) {
            Vec raw=linear(w_.mixture4,silu(linear(w_.mixture2,silu(linear(w_.mixture0,input(local,i,4))))));
            for (int j=2*w_.components;j<3*w_.components;++j) raw[j]=softplus(raw[j])+.01f;
            out.insert(out.end(),raw.data(),raw.data()+raw.size());
        }
        return out;
    }
};

inline RenewalSpeedMixture decode(const float* raw,int components,double signedDerivative) {
    RenewalSpeedMixture m;
    const double maximum=*std::max_element(raw,raw+components); double sum=0;
    for (int j=0;j<components;++j) {
        m.weights.push_back(std::exp(raw[j]-maximum)); sum+=m.weights.back();
        m.means.push_back(-signedDerivative+raw[components+j]); m.scales.push_back(raw[2*components+j]);
        if (!std::isfinite(m.means.back()) || !(m.scales.back()>0) || !std::isfinite(m.scales.back()))
            throw std::runtime_error("invalid causal speed parameters");
    }
    if (!(sum>0) || !std::isfinite(sum)) throw std::runtime_error("invalid causal mixture weights");
    for (auto& x:m.weights) x/=sum; return m;
}
inline double sampleSpeed(const RenewalSpeedMixture& m,double component,double quantile) {
    std::size_t j=0; double cumulative=m.weights[0];
    while (j+1<m.weights.size() && component>=cumulative) cumulative+=m.weights[++j];
    const double logTail=std::log1p(-quantile)+normalLogCdf(m.means[j]/m.scales[j]);
    return std::max(0.,m.means[j]-m.scales[j]*normalQuantileFromLogCdf(logTail));
}

} // namespace

namespace {
std::vector<float> renewalEntry(const RayMeanProfile& p, const RayStartCondition& start) {
    if (start.mode != RayStartMode::PositiveExterior && start.mode != RayStartMode::SurfaceOutward)
        throw std::invalid_argument("invalid causal renewal start mode");
    const bool surface = start.mode == RayStartMode::SurfaceOutward;
    if (surface && (!(start.outwardDerivative > 0) || !std::isfinite(start.outwardDerivative)))
        throw std::invalid_argument("causal renewal B requires a finite positive outward derivative");
    const auto& first = p.segments().front();
    const double b0 = first.value(0);
    std::vector<float> entry;
    for (double x : {surface ? 1. : 0., b0, surface ? -b0 : 0.,
                     surface ? start.outwardDerivative-first.derivative(0) : 0.}) {
        entry.push_back(static_cast<float>(std::asinh(x)));
        if (!std::isfinite(entry.back())) throw std::invalid_argument("nonfinite causal renewal entry");
    }
    return entry;
}
double correctionFactor(double delta) {
    const double factor = std::exp(delta);
    if (!(factor > 0) || !std::isfinite(factor))
        throw std::runtime_error("nonfinite causal renewal hazard correction");
    return factor;
}
}

struct CausalRenewalModel::Impl {
    NetworkWeights weights;
    double step = 0, fine = 0;
    std::string checkpoint;
};

CausalRenewalModel CausalRenewalModel::load(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open causal renewal model " + path.string());
    return load(input);
}

CausalRenewalModel CausalRenewalModel::load(std::istream& input) {
    Json r; input >> r;
    const Json conditioning = {
        {"modes", {"positive_exterior", "collision_state"}},
        {"initial", {"surface_mode", "b0", "known_value_residual", "known_derivative_residual"}},
        {"derivative_residual", "ell*g0/sigma - b_prime(0)"},
        {"encoding", "asinh; unknown A residuals masked"},
        {"event", "first_downcrossing_after_start"}, {"baseline", "positive_side_local_hazard"}};
    const Json target = {
        {"variable", "w = -ell * directional_derivative_at_hit / sigma"},
        {"support", "w > 0; no upper truncation"}, {"physical_derivative", "-sigma * w / ell"},
        {"event", "first_downcrossing_after_start"}, {"state", "left_limit_of_fixed_q_interval"}};
    if (r.at("format") != "macrofacet.causal_renewal" || r.at("version") != 1 ||
        r.at("kernel") != "squared_exponential" || r.at("profile_mode") != "cubic" ||
        r.at("activation_dtype") != "float32" || r.at("conditioning") != conditioning || r.at("target") != target ||
        r.at("gru_convention") != "pytorch_rzn_reset_after" ||
        r.at("distance_convention") != "q=distance_from_birth/ell" || r.at("mean_convention") != "b=m/sigma" ||
        r.at("state_convention") != "entering_fixed_q_interval_left_limit_gradient")
        throw std::invalid_argument("unsupported causal renewal model contract");
    auto p = std::make_shared<Impl>();
    const auto& c = r.at("model_config"); const auto& g = r.at("gradient_config");
    auto& w = p->weights;
    w.hidden = c.at("hidden"); w.embedding = c.at("embedding"); w.components = g.at("size");
    const int width = c.at("width"), gw = g.at("width"), h = w.hidden, e = w.embedding;
    p->step = c.at("step"); p->fine = c.at("integration_step");
    if (!(p->step > 0 && p->step <= 1 && p->fine > 0 && p->fine <= p->step) || c.at("quadrature_order") != 8 ||
        c.at("history") != true || g.at("history") != true || g.at("kind") != "mixture" || g.at("hidden") != h ||
        h < 1 || h > 1024 || e < 1 || e > 1024 || w.components < 1 || w.components > 1024)
        throw std::invalid_argument("invalid causal renewal network configuration");
    const auto& b = r.at("backbone"); const auto& m = r.at("gradient");
    w.initial0 = dense(b,"initial.0",4,h); w.initial2 = dense(b,"initial.2",h,h);
    w.segment0 = dense(b,"segment.0",7,e); w.segment2 = dense(b,"segment.2",e,e);
    w.head0 = dense(b,"head.0",h+3,width); w.head2 = dense(b,"head.2",width,1);
    w.gruInput = dense(b,"gru",e,3*h,".weight_ih_l0",".bias_ih_l0");
    w.gruHidden = dense(b,"gru",h,3*h,".weight_hh_l0",".bias_hh_l0");
    w.mixture0 = dense(m,"net.0",h+4,gw); w.mixture2 = dense(m,"net.2",gw,gw);
    w.mixture4 = dense(m,"net.4",gw,3*w.components);
    p->checkpoint = r.at("checkpoint_sha256").get<std::string>();
    return CausalRenewalModel(std::move(p));
}

double CausalRenewalModel::step() const { return impl_->step; }
double CausalRenewalModel::integrationStep() const { return impl_->fine; }
const std::string& CausalRenewalModel::checkpointSha256() const { return impl_->checkpoint; }

std::vector<CausalRenewalPrediction> CausalRenewalModel::query(const RayMeanProfile& p,
    const RayStartCondition& start, const std::vector<double>& q) const {
    validateProfile(p);
    NetworkState network(impl_->weights); network.initialize(renewalEntry(p,start));
    for (double x : q) if (!(x >= 0 && x <= p.maximumX()) || !std::isfinite(x))
        throw std::invalid_argument("causal renewal query outside profile");
    std::vector<CausalRenewalPrediction> result(q.size());
    if (q.empty()) return result;
    const auto& m = *impl_;
    const auto count = static_cast<std::size_t>(std::ceil(p.maximumX()/m.step));
    std::vector<double> prefix{0}, deltas;
    for (std::size_t cell = 0; cell < count; ++cell) {
        const double a = cell*m.step, b = std::min((cell+1)*m.step,p.maximumX());
        std::vector<float> left; appendLeft(left,p,1,a);
        const double delta = network.correction(left); deltas.push_back(delta);
        prefix.push_back(prefix.back()+correctionFactor(delta)*total(fragments(p,1,a,b,m.fine)));
        if (!std::isfinite(prefix.back())) throw std::runtime_error("nonfinite causal renewal cumulative hazard");
        std::vector<std::size_t> indices; std::vector<float> local;
        for (std::size_t j = 0; j < q.size(); ++j) {
            // A boundary's gradient uses the state before consuming that interval.
            const auto gradientCell = std::min(count-1,static_cast<std::size_t>(std::max(0.,std::ceil(q[j]/m.step)-1)));
            if (gradientCell == cell) { indices.push_back(j); appendLocal(local,p,1,q[j],a,m.step); }
        }
        if (!indices.empty()) {
            const auto raw = network.mixture(local);
            for (std::size_t i = 0; i < indices.size(); ++i) {
                const auto j = indices[i];
                result[j].mixture = decode(raw.data()+i*3*m.weights.components,m.weights.components,
                                          segmentAt(p,q[j]).derivative(q[j]));
            }
        }
        if (cell+1 < count) { std::vector<float> token; appendToken(token,p,1,a,b); network.advance(token); }
    }
    for (std::size_t j = 0; j < q.size(); ++j) {
        const auto cell = std::min(count-1,static_cast<std::size_t>(std::floor(q[j]/m.step)));
        auto& r = result[j]; r.q = q[j];
        r.cumulativeHazard = prefix[cell]+correctionFactor(deltas[cell])*total(fragments(p,1,cell*m.step,q[j],m.fine));
        const auto& s = segmentAt(p,q[j]); r.logHazard = logLocalRate(s.value(q[j]),s.derivative(q[j]))+deltas[cell];
    }
    return result;
}

CausalRenewalSample CausalRenewalModel::sample(const RayMeanProfile& p, const RayStartCondition& start,
    const std::array<double,3>& uniforms, double tolerance) const {
    return sampleInterval(p,start,uniforms,0,p.maximumX(),tolerance);
}

CausalRenewalSample CausalRenewalModel::sampleInterval(const RayMeanProfile& p, const RayStartCondition& start,
    const std::array<double,3>& uniforms, double beginQ, double endQ, double tolerance) const {
    validateProfile(p);
    if (!(tolerance > 0) || !std::isfinite(tolerance)) throw std::invalid_argument("invalid causal renewal tolerance");
    for (double u : uniforms) if (!(u > 0 && u < 1)) throw std::invalid_argument("causal renewal uniforms must be open");
    if (!(beginQ >= 0 && endQ >= beginQ && endQ <= p.maximumX()) || !std::isfinite(endQ))
        throw std::invalid_argument("invalid causal renewal sampling interval");
    CausalRenewalSample result; result.q = endQ;
    if (endQ == beginQ) return result;
    const auto& m = *impl_;
    NetworkState network(m.weights); network.initialize(renewalEntry(p,start));
    double remaining = -std::log(uniforms[0]);
    for (std::size_t cell = 0; cell*m.step < endQ; ++cell) {
        ++result.intervals;
        const double a = cell*m.step, b = std::min((cell+1)*m.step,p.maximumX());
        std::vector<float> left; appendLeft(left,p,1,a);
        const double factor = correctionFactor(network.correction(left));
        const auto parts = b <= beginQ ? std::vector<Fragment>{}
            : fragments(p,1,std::max(a,beginQ),std::min(b,endQ),m.fine);
        const double mass = factor*total(parts);
        if (!std::isfinite(mass)) throw std::runtime_error("nonfinite causal renewal integrated hazard");
        if (mass > 0 && remaining <= mass) {
            double residual = remaining/factor;
            const Fragment* selected = &parts.back();
            for (const auto& part : parts) {
                if ((residual <= part.mass && part.mass > 0) || &part == &parts.back()) { selected = &part; break; }
                residual -= part.mass;
            }
            residual = std::clamp(residual,0.,selected->mass);
            double lo = selected->a, hi = selected->b;
            for (int iteration = 0; iteration < 80 && hi-lo > tolerance; ++iteration) {
                const double mid = (lo+hi)/2;
                if (integral(p,1,selected->a,mid) < residual) lo = mid; else hi = mid;
            }
            result.hit = true; result.q = (lo+hi)/2;
            std::vector<float> local; appendLocal(local,p,1,result.q,a,m.step);
            const auto raw = network.mixture(local);
            result.speed = sampleSpeed(decode(raw.data(),m.weights.components,segmentAt(p,result.q).derivative(result.q)),
                                       uniforms[1],uniforms[2]);
            if (!(result.speed > 0) || !std::isfinite(result.speed)) throw std::runtime_error("invalid causal renewal speed");
            return result;
        }
        remaining -= mass;
        if (b < endQ) { std::vector<float> token; appendToken(token,p,1,a,b); network.advance(token); }
    }
    return result;
}
} // namespace Tungsten::NeuralGpis
