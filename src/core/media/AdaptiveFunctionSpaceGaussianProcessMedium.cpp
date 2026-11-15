#include "AdaptiveFunctionSpaceGaussianProcessMedium.hpp"
#include "io/JsonObject.hpp"
#include "math/SdfQueries.hpp"
#include "phasefunctions/BRDFPhaseFunction.hpp"
#include "samplerecords/SpecularGpisEvent.hpp"
#include "bsdfs/MirrorBsdf.hpp"
#include "bsdfs/ConductorBsdf.hpp"
#include <nlohmann/json.hpp>
#include <iostream>
#include <stdexcept>
#include <algorithm>

namespace Tungsten {
namespace ag = AdaptiveGpis;
namespace {
void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
Eigen::Vector3d eigen(const Vec3d &p) { return {p.x(),p.y(),p.z()}; }
ag::Observation valueObservation(const Vec3d &p) { return {eigen(p),Eigen::Vector3d::Zero(),false}; }
ag::Observation derivativeObservation(const Vec3d &p, int axis) {
    return {eigen(p),Eigen::Vector3d::Unit(axis),true};
}
}

struct AdaptiveFunctionSpaceGaussianProcessMedium::Context : GPContext, MediumPathContext, SpecularGpisEvent {
    ag::OnlineSampler sampler;
    bool hit = false, validBirth = true, hasBirth = false, hasGradient = false;
    Vec3d point = Vec3d(0.0), gradient = Vec3d(0.0), birth = Vec3d(0.0), birthB = Vec3d(0.0);
    double birthA = 0, value = 0, birthLipschitz = 0;
    ag::GradientPlane gradientPlane;
    bool misReady = false;
    const AdaptiveFunctionSpaceGaussianProcessMedium *owner = nullptr;
    Context(double length, const ag::Settings &s) : sampler(length,s.rankTolerance,s.maxSamples) {}
    // Published contexts are immutable. MediumState::reset drops pathContext;
    // this no-op prevents a reset on a copied branch from corrupting its parent.
    void reset() override {}
    double reflectionPdf(const Vec3f &travel, const Vec3f &outgoing) const override {
        ++owner->_neePdfQueries;
        const auto half=eigen(Vec3d(outgoing).normalized()-Vec3d(travel).normalized());
        return gradientPlane.reflectionPdf(half);
    }
    Vec3d reflectionGradient(const Vec3f &travel, const Vec3f &outgoing) const override {
        ++owner->_neeGradientQueries;
        const auto g=gradientPlane.gradientForNormal(eigen(Vec3d(outgoing).normalized()-Vec3d(travel).normalized()));
        return Vec3d(g.x(),g.y(),g.z());
    }
};

void AdaptiveFunctionSpaceGaussianProcessMedium::fromJson(JsonPtr value, const Scene &scene)
{
    GaussianProcessMedium::fromJson(value,scene);
    require(!value["additional_phase_functions"], "adaptive GP: multiple phase functions are unsupported");
    value.getField("1D_sampling",_oneDSampling);
    std::string scheme=_oneDSampling ? "mis" : "uni";
    value.getField("1D_sampling_scheme",scheme);
    require(scheme=="uni" || scheme=="mis" || scheme=="nee","adaptive GP: unknown 1D sampling scheme");
    require(_oneDSampling || scheme=="uni","adaptive GP: enable 1D_sampling for MIS/NEE");
    _samplingScheme=scheme=="mis" ? SparseConv1DSamplingScheme::MIS :
        scheme=="nee" ? SparseConv1DSamplingScheme::NEE : SparseConv1DSamplingScheme::UNI;
    if (auto a = value["adaptive"]) {
        a.getField("probability_tolerance",_settings.probabilityTolerance);
        a.getField("minimum_step",_settings.minimumStep);
        a.getField("root_tolerance",_settings.rootTolerance);
        a.getField("rank_tolerance",_settings.rankTolerance);
        a.getField("max_samples",_settings.maxSamples);
        a.getField("max_steps",_settings.maxSteps);
        a.getField("enabled",_settings.adaptive);
        a.getField("statistics",_statistics);
    }
    _settings.validate();
}

rapidjson::Value AdaptiveFunctionSpaceGaussianProcessMedium::toJson(Allocator &a) const
{
    auto base=GaussianProcessMedium::toJson(a);
    base.RemoveMember("type");
    return JsonObject{std::move(base),a,
        "type","adaptive_function_space_gaussian_process",
        "1D_sampling",_oneDSampling,"1D_sampling_scheme",
        _samplingScheme==SparseConv1DSamplingScheme::MIS ? "mis" :
            _samplingScheme==SparseConv1DSamplingScheme::NEE ? "nee" : "uni",
        "adaptive",JsonObject{a,
            "probability_tolerance",_settings.probabilityTolerance,
            "minimum_step",_settings.minimumStep,"root_tolerance",_settings.rootTolerance,
            "rank_tolerance",_settings.rankTolerance,"max_samples",_settings.maxSamples,
            "max_steps",_settings.maxSteps,"enabled",_settings.adaptive,"statistics",_statistics}};
}

void AdaptiveFunctionSpaceGaussianProcessMedium::prepareForRender()
{
    GaussianProcessMedium::prepareForRender();
    _settings.validate();
    _process = std::dynamic_pointer_cast<GaussianProcess>(_gp);
    require(bool(_process), "adaptive GP: only a standard Gaussian process is supported");
    const auto se = std::dynamic_pointer_cast<SquaredExponentialCovariance>(_process->_cov);
    require(se && se->isIsotropic(), "adaptive GP: stationary isotropic squared_exponential covariance required");
    require(!_process->_embedCov && !_process->_mean_additional && _process->_globalCondPs.empty() &&
        !_process->_conditioningDataPath && _process->_id == 0,
        "adaptive GP: embedded covariance, CSG and external conditioning are unsupported");
    require(_ctxt == GPCorrelationContext::RenewalPlus && _intersectMethod == GPIntersectMethod::GPDiscrete &&
        _normalSamplingMethod == GPNormalSamplingMethod::ConditionedGaussian,
        "adaptive GP: use renewal+, gp_discrete and conditioned_gaussian");
    require(bool(std::dynamic_pointer_cast<BRDFPhaseFunction>(_phaseFunction)),
        "adaptive GP: a brdf phase_function is required");
    const auto brdf=std::static_pointer_cast<BRDFPhaseFunction>(_phaseFunction);
    require(!_oneDSampling || dynamic_cast<const MirrorBsdf *>(brdf->bsdf()) ||
        dynamic_cast<const ConductorBsdf *>(brdf->bsdf()),"adaptive GP: 1D sampling supports mirror/conductor");
    _sigma = std::sqrt((*se)(Derivative::None,Derivative::None,Vec3d(0.0),Vec3d(0.0),Vec3d(0.0),Vec3d(0.0)));
    _length = se->lengthScale();
    require(std::isfinite(_sigma) && _sigma > 0 && std::isfinite(_length) && _length > 0,
        "adaptive GP: positive finite sigma and lengthScale required");
    const auto *m = _process->_mean.get();
    require(dynamic_cast<const SphericalMean *>(m) || dynamic_cast<const LinearMean *>(m) ||
        dynamic_cast<const HomogeneousMean *>(m) || m->sdfQueries(),
        "adaptive GP: use spherical/linear/homogeneous mean or a certified SdfQueries provider");
    for (int i=0;i<3;++i) require(std::isfinite(sigmaS()[i]) && sigmaS()[i]>0 &&
        std::isfinite(sigmaA()[i]) && sigmaA()[i]>=0,
        "adaptive GP: positive scattering and nonnegative absorption required");
    _rays=0; _hits=0; _queries=0; _redundant=0; _steps=0; _minimumSteps=0;
    _refinements=0; _invalidBirths=0; _invalidGradients=0;
    _rankDecompositions=0;
    _misEvents=0; _neePdfQueries=0; _neeGradientQueries=0; _degenerateGradients=0;
}

double AdaptiveFunctionSpaceGaussianProcessMedium::mean(const Vec3d &p) const
{
    return (*_process->_mean)(Derivative::None,p,Vec3d(0.0));
}
Vec3d AdaptiveFunctionSpaceGaussianProcessMedium::meanGradient(const Vec3d &p) const
{
    if (const auto *s = dynamic_cast<const SphericalMean *>(_process->_mean.get())) {
        const Vec3d d = p-s->center();
        return d.lengthSq()>0 ? d.normalized() : Vec3d(0.0);
    }
    return _process->_mean->dmean_da(p);
}
double AdaptiveFunctionSpaceGaussianProcessMedium::meanLipschitz(const Vec3d &o, const Vec3d &d,
                                                                 double lo, double hi) const
{
    const auto *m = _process->_mean.get();
    if (dynamic_cast<const SphericalMean *>(m)) return 1;
    if (const auto *l = dynamic_cast<const LinearMean *>(m)) return l->gradient().length();
    if (dynamic_cast<const HomogeneousMean *>(m)) return 0;
    const auto b = m->sdfQueries()->rayBounds(eigen(o),eigen(d),lo,hi);
    require(b.certified && std::isfinite(b.minimumDerivative) && std::isfinite(b.maximumDerivative),
        "adaptive GP: mean provider did not certify the ray derivative bounds");
    return std::max(std::abs(b.minimumDerivative),std::abs(b.maximumDerivative));
}

std::shared_ptr<AdaptiveFunctionSpaceGaussianProcessMedium::Context>
AdaptiveFunctionSpaceGaussianProcessMedium::trace(PathSampleGenerator &rng, const Ray &ray,
                                                 MediumState &state, double &t) const
{
    require(bool(_process), "adaptive GP: medium is not prepared");
    const Vec3d o(ray.pos()), d(ray.dir());
    const double speed=d.length();
    require(std::isfinite(speed) && speed>0 && std::isfinite(ray.farT()),
        "adaptive GP: a finite ray segment and nonzero direction are required");
    const Vec3d w=d/speed;
    const double end=ray.farT()*speed;
    double begin=std::max(0.0,double(ray.nearT())*speed);
    auto previous = std::dynamic_pointer_cast<const Context>(state.pathContext);
    auto c = std::make_shared<Context>(_length,_settings);
    c->owner=this;
    if (previous && !state.firstScatter) {
        if (previous->hit) {
            // Renderer positions are float. Anchor the renewed GP at the actual
            // outgoing origin, using the full unnormalized collision gradient.
            require((o-previous->point).length() <= 1e-4*(1+o.length()),
                "adaptive GP: surface birth origin does not match the previous collision");
            c->hasBirth=true; c->birth=o;
            c->birthA=-mean(o)/_sigma;
            // Light connections replace lastAniso with their constrained full
            // gradient. The published parent context must remain unchanged.
            const Vec3d birthGradient=state.lastAniso;
            c->birthB=(birthGradient-meanGradient(o))*(_length/_sigma);
            c->birthLipschitz=(_sigma/_length)*(std::abs(c->birthA)/std::sqrt(std::exp(1.0))+c->birthB.length());
            c->sampler.condition(valueObservation(o),c->birthA);
            for (int k=0;k<3;++k) c->sampler.condition(derivativeObservation(o,k),c->birthB[k]);
            c->validBirth=birthGradient.dot(w)>0;
            // Exclude the known root at the birth. This numerical escape is
            // bounded by root_tolerance, and is included in convergence tests.
            begin=std::max(begin,_settings.rootTolerance);
        } else {
            c=std::make_shared<Context>(*previous);
            c->hasGradient=false;
            c->misReady=false;
        }
    }
    const auto beforeQ=c->sampler.queries(), beforeR=c->sampler.redundant();
    const auto beforeD=c->sampler.rankDecompositions();
    auto value = [&](double s) {
        const Vec3d p=o+w*s;
        return mean(p)+_sigma*c->sampler.sample(valueObservation(p),rand_normal_2(rng).x());
    };
    auto boundMean = [&](double s) {
        const Vec3d p=o+w*s;
        double m=mean(p);
        if (c->hasBirth) {
            const Vec3d r=(p-c->birth)/_length;
            m+=_sigma*std::exp(-0.5*r.lengthSq())*(c->birthA+r.dot(c->birthB));
        }
        return m;
    };
    ag::Intersection result;
    result.t=end;
    if (c->validBirth && begin<end) {
        const double v=value(begin);
        c->validBirth=v>0; // An exterior query already inside solid is absorbed.
        if (c->validBirth) {
            const ag::Bounds bounds(meanLipschitz(o,w,begin,end)+c->birthLipschitz,
                _sigma,_sigma/_length,_settings.probabilityTolerance);
            result=ag::intersect(begin,end,v,value,boundMean,bounds,_settings);
        }
    }
    c->hit=result.hit; c->value=result.value; c->point=o+w*result.t;
    if (result.hit) {
        if (_oneDSampling) {
            Eigen::Matrix3d frame;
            frame.col(2)=eigen(w); frame.col(0)=frame.col(2).unitOrthogonal();
            frame.col(1)=frame.col(2).cross(frame.col(0));
            std::vector<ag::Observation> derivatives;
            for (int k=0;k<3;++k) derivatives.push_back({eigen(c->point),frame.col(k),true});
            const auto joint=c->sampler.predictJoint(derivatives);
            const double scale=_sigma/_length;
            const Eigen::Vector3d gradientMean=frame.transpose()*eigen(meanGradient(c->point))+scale*joint.mean;
            c->gradientPlane=ag::GradientPlane(frame,gradientMean,scale*scale*joint.covariance,rand_normal_2(rng).x());
            const auto xy=rand_normal_2(rng);
            const auto gradient=c->gradientPlane.sample(Eigen::Vector2d(xy.x(),xy.y()));
            c->gradient=Vec3d(gradient.x(),gradient.y(),gradient.z());
            c->misReady=c->gradientPlane.continuous() && _samplingScheme!=SparseConv1DSamplingScheme::UNI;
            if (c->misReady) ++_misEvents;
            else if (_samplingScheme!=SparseConv1DSamplingScheme::UNI) ++_degenerateGradients;
            _queries.fetch_add(3);
        } else {
            c->gradient=meanGradient(c->point);
            for (int k=0;k<3;++k) c->gradient[k]+=_sigma/_length*
                c->sampler.sample(derivativeObservation(c->point,k),rand_normal_2(rng).x());
        }
        c->hasGradient=true;
    }
    t=result.t/speed;
    state.pathContext=c; state.gpContext=c; state.lastGPId=0;
    state.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    ++_rays; if (result.hit) ++_hits;
    if (!c->validBirth) ++_invalidBirths;
    _queries.fetch_add(c->sampler.queries()-beforeQ);
    _redundant.fetch_add(c->sampler.redundant()-beforeR);
    _rankDecompositions.fetch_add(c->sampler.rankDecompositions()-beforeD);
    _steps.fetch_add(result.steps); _minimumSteps.fetch_add(result.minimumSteps);
    _refinements.fetch_add(result.refinements);
    return c;
}

bool AdaptiveFunctionSpaceGaussianProcessMedium::intersectGP(PathSampleGenerator &rng, const Ray &ray,
    MediumState &state, double &t, bool) const
{
    auto c=trace(rng,ray,state,t);
    return c->hit || !c->validBirth;
}

bool AdaptiveFunctionSpaceGaussianProcessMedium::sampleGradient(PathSampleGenerator &, const Ray &,
    const Vec3d &, const float &, MediumState &state, Vec3d &gradient) const
{
    const auto c=std::dynamic_pointer_cast<const Context>(state.pathContext);
    if (!c || !c->hasGradient || !c->validBirth) return false;
    gradient=c->gradient;
    return true;
}

bool AdaptiveFunctionSpaceGaussianProcessMedium::sampleDistance(PathSampleGenerator &rng, const Ray &ray,
    MediumState &state, MediumSample &sample) const
{
    sample={}; sample.phase=_phaseFunction.get(); sample.pdf=1;
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    if (state.bounce>=_maxBounce) return false;
    if (ray.farT()<=ray.nearT()) {
        sample.exited=true; sample.t=sample.continuedT=ray.farT();
        sample.p=ray.pos()+ray.dir()*sample.t;
        sample.weight=sample.continuedWeight=Vec3f(1.0f);
        return true;
    }
    double t;
    auto c=trace(rng,ray,state,t);
    if (!c->validBirth) {
        if (state.distanceDiagnostic) state.distanceDiagnostic->failure="adaptive_invalid_birth";
        return false;
    }
    sample.exited=!c->hit; sample.t=sample.continuedT=float(t);
    sample.p=ray.pos()+ray.dir()*sample.t;
    sample.weight=sample.continuedWeight=Vec3f(1.0f);
    if (c->hit) {
        if (!std::isfinite(c->gradient.lengthSq()) || c->gradient.lengthSq()==0 ||
            c->gradient.dot(Vec3d(ray.dir()))>=0) {
            ++_invalidGradients;
            if (state.distanceDiagnostic) state.distanceDiagnostic->failure="adaptive_invalid_hit_gradient";
            return false;
        }
        sample.aniso=state.lastAniso=c->gradient;
        if (c->misReady) {
            sample.eventContext=c;
            sample.sparseConv1DSamplingScheme=_samplingScheme;
        }
        sample.weight=sample.continuedWeight=Vec3f(_gp->color(c->point))*sigmaS()/ (sigmaS()+sigmaA());
        sample.emission=Vec3f(_gp->emission(c->point));
        state.advance();
    } else state.firstScatter=false;
    sample.ctxt=c.get(); sample.gpId=0;
    state.info.t+=sample.t; sample.rayInfo=state.info;
    if (state.distanceDiagnostic && state.distanceDiagnostic->collectFlight)
        state.distanceDiagnostic->flight={{"sampler_rank",c->sampler.rank()},
            {"redundant_queries",c->sampler.redundant()},{"sampled_distance",t}};
    return true;
}

Vec3f AdaptiveFunctionSpaceGaussianProcessMedium::transmittance(PathSampleGenerator &rng, const Ray &ray,
    bool, bool, MediumState *state) const
{
    if (ray.farT()<=ray.nearT()) return Vec3f(1.0f);
    MediumState local{};
    if (!state) { local.reset(); state=&local; }
    double t;
    const auto c=trace(rng,ray,*state,t);
    state->firstScatter=false;
    return Vec3f(c->validBirth && !c->hit ? 1.0f : 0.0f);
}

float AdaptiveFunctionSpaceGaussianProcessMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const
{
    throw std::runtime_error("adaptive GP: state-free distance PDFs / bidirectional transport are unsupported");
}

void AdaptiveFunctionSpaceGaussianProcessMedium::teardownAfterRender()
{
    if (_statistics) std::cout << "Adaptive GPIS statistics: " << nlohmann::json{
        {"rays",_rays.load()},{"hits",_hits.load()},{"queries",_queries.load()},
        {"numerically_redundant",_redundant.load()},{"steps",_steps.load()},
        {"minimum_steps",_minimumSteps.load()},{"root_refinements",_refinements.load()},
        {"invalid_births",_invalidBirths.load()},{"invalid_gradients",_invalidGradients.load()},
        {"double_rank_decompositions",_rankDecompositions.load()},{"arithmetic","double"},
        {"one_d_sampling",_oneDSampling},{"mis_events",_misEvents.load()},
        {"nee_pdf_queries",_neePdfQueries.load()},{"nee_gradient_queries",_neeGradientQueries.load()},
        {"degenerate_gradient_events",_degenerateGradients.load()}}.dump() << '\n';
}
}
