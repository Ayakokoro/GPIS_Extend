#include "ClassicGlobalMedium.hpp"
#include "math/GPFunctions.hpp"
#include "GpisSdfGrid.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "phasefunctions/ObservedNormalScattering.hpp"
#include <algorithm>
#include <cmath>
#include <complex>
#include <stdexcept>

namespace Tungsten {
namespace {
Eigen::Vector3d eigen(const Vec3d &v) { return Eigen::Vector3d(v.x(), v.y(), v.z()); }
Vec3d vec(const Eigen::Vector3d &v) { return Vec3d(v.x(), v.y(), v.z()); }

class ClassicGlobalPhase final : public PhaseFunction {
    const ClassicGlobalMedium &_medium;
public:
    explicit ClassicGlobalPhase(const ClassicGlobalMedium &medium) : _medium(medium) {}
    float pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample &s) const override
    {
        if(_medium.hasDiffuseMicroMaterial()) return observedCosinePdf(wo,s);
        return float(_medium.phasePdf(Vec3d(s.p),
                       eigen(Vec3d(wi)).normalized(), eigen(Vec3d(wo)).normalized()));
    }
    Vec3f eval(const Vec3f &wi, const Vec3f &wo, const MediumSample &s) const override
    {
        if(_medium.hasDiffuseMicroMaterial()) return _medium.reflectance(1)*pdf(wi,wo,s);
        const Eigen::Vector3d a = eigen(Vec3d(wi)).normalized();
        const Eigen::Vector3d b = eigen(Vec3d(wo)).normalized();
        if ((b - a).squaredNorm() == 0.0) return Vec3f(0.0f);
        return _medium.reflectance(std::abs(a.dot((b - a).normalized())))*pdf(wi, wo, s);
    }
    bool sample(PathSampleGenerator &sampler, const Vec3f &wi,
                const MediumSample &s, PhaseSample &result) const override
    {
        if(_medium.hasDiffuseMicroMaterial()) {
            result.w=sampleObservedCosine(sampler,s);
            result.pdf=pdf(wi,result.w,s);
            result.weight=_medium.reflectance(1);
            return result.pdf>0;
        }
        const Eigen::Vector3d w = eigen(Vec3d(wi)).normalized();
        const Eigen::Vector3d n = _medium.sampleVisibleNormal(Vec3d(s.p), w, sampler);
        result.w = Vec3f(vec(w - 2.0*w.dot(n)*n)).normalized();
        result.pdf = pdf(wi, result.w, s);
        result.weight = _medium.reflectance(std::abs(w.dot(n)));
        return result.pdf > 0.0f && std::isfinite(result.pdf);
    }
};
}

ClassicGlobalMedium::ClassicGlobalMedium()
    : _materialType("conductor"), _mirrorAlbedo(1.0),
      _eta(0.2, 0.9, 1.1), _k(3.9, 2.5, 2.2), _forceUnitFresnel(false),
      _sigma(0.0), _lipschitz(0.0), _gradientBound(0.0), _trackingStep(0.0),
      _configuredLipschitz(-1.0), _configuredGradientBound(-1.0)
{
    _phaseFunction = std::make_shared<ClassicGlobalPhase>(*this);
}
ClassicGlobalMedium::~ClassicGlobalMedium() = default;

void ClassicGlobalMedium::fromJson(JsonPtr value, const Scene &scene)
{
    JsonSerializable::fromJson(value, scene);
    // These settings would silently change the intended classic_global law.
    for (const char *key : {"phase_function", "transmittance", "density", "density_grid",
                            "majorant_grid", "sigma_a", "sigma_s"})
        if (value[key]) value.parseError(std::string("classic_global does not use '") + key + "'; use material and gaussian_process");
    const auto gp = value.getRequiredMember("gaussian_process");
    std::string type = "standard";
    gp.getField("type", type);
    bool embedded = false;
    gp.getField("embed_cov", embedded);
    if (type != "standard" || embedded || gp["conditioning_data"])
        gp.parseError("classic_global requires an unconditioned standard GP in world coordinates");
    const auto mean = gp.getRequiredMember("mean");
    if (auto grid = mean["grid"]) {
        bool requestSdf = false;
        grid.getField("request_sdf", requestSdf);
        std::string integration;
        grid.getField("integration_method", integration);
        if (requestSdf || integration == "residual_ratio")
            grid.parseError("classic_global reads an existing SDF; set request_sdf=false and integration_method=raymarching (no auxiliary grid baking)");
    }
    _mean = scene.fetchMeanFunction(mean);
    _covariance = scene.fetchCovarianceFunction(gp.getRequiredMember("covariance"));
    if (auto material = value["material"]) {
        material.getField("type", _materialType);
        if (_materialType == "mirror" || _materialType == "lambert") {
            material.getField("albedo", _mirrorAlbedo);
            for (int i = 0; i < 3; ++i)
                if (!std::isfinite(_mirrorAlbedo[i]) || _mirrorAlbedo[i] < 0.0 || _mirrorAlbedo[i] > 1.0)
                    material.parseError("classic_global: albedo must be finite and in [0,1]");
        } else if (_materialType == "conductor") {
            material.getField("eta", _eta);
            material.getField("k", _k);
            material.getField("force_unit_fresnel", _forceUnitFresnel);
        } else {
            material.parseError("classic_global material type must be conductor, mirror or lambert");
        }
    }
    value.getField("max_bounces", _maxBounce);
    value.getField("tracking_step", _trackingStep);
    if (value.getField("sdf_lipschitz", _configuredLipschitz) &&
        (!std::isfinite(_configuredLipschitz) || _configuredLipschitz < 0.0))
        value.parseError("sdf_lipschitz must be finite and nonnegative");
    if (value.getField("gradient_bound", _configuredGradientBound) &&
        (!std::isfinite(_configuredGradientBound) || _configuredGradientBound < 0.0))
        value.parseError("gradient_bound must be finite and nonnegative");
    if (!std::isfinite(_trackingStep) || _trackingStep < 0.0 || _maxBounce < 0)
        value.parseError("classic_global: invalid tracking_step/max_bounces");
    for (int i = 0; i < 3; ++i)
        if (!std::isfinite(_eta[i]) || !std::isfinite(_k[i]) || _eta[i] <= 0.0 || _k[i] < 0.0)
            value.parseError("classic_global: eta must be positive and k nonnegative");
}

rapidjson::Value ClassicGlobalMedium::toJson(Allocator &allocator) const
{
    JsonObject material{allocator, "type", _materialType};
    if (_materialType == "mirror" || _materialType == "lambert")
        material.add("albedo", _mirrorAlbedo);
    else
        material.add("eta", _eta, "k", _k, "force_unit_fresnel", _forceUnitFresnel);
    JsonObject result{JsonSerializable::toJson(allocator), allocator,
        "type", "classic_global",
        "gaussian_process", JsonObject{allocator, "type", "standard", "mean", *_mean, "covariance", *_covariance},
        "material", rapidjson::Value(material),
        "tracking_step", _trackingStep, "max_bounces", _maxBounce};
    if (_configuredLipschitz >= 0.0) result.add("sdf_lipschitz", _configuredLipschitz);
    if (_configuredGradientBound >= 0.0) result.add("gradient_bound", _configuredGradientBound);
    return result;
}

void ClassicGlobalMedium::loadResources()
{
    _mean->loadResources();
    _covariance->loadResources();
}

void ClassicGlobalMedium::prepareForRender()
{
    _nonstationary.reset();
    _ndf.reset();
    if (auto procedural = dynamic_cast<ProceduralNonstationaryCovariance *>(_covariance.get())) {
        _nonstationary = std::make_unique<ClassicGlobal::NonstationaryKernel>(*procedural);
        _sigma = _nonstationary->minimumSigma();
    } else {
        if (!_covariance->isStationaryKernel())
            throw std::runtime_error("classic_global requires a stationary differentiable covariance or proc_nonstationary cov/var/ls");
        const Vec3d zero(0.0);
        const double variance = (*_covariance)(Derivative::None, Derivative::None, zero, zero, zero, zero);
        if (!(variance > 0.0) || !std::isfinite(variance))
            throw std::runtime_error("classic_global: field variance must be positive and finite");
        _sigma = std::sqrt(variance);
        Eigen::Matrix3d c;
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) {
                Vec3d a(0.0), b(0.0); a[i] = 1.0; b[j] = 1.0;
                c(i, j) = (*_covariance)(Derivative::First, Derivative::First, zero, zero, a, b);
            }
        _ndf = std::make_unique<ClassicGlobal::GaussianNdf>(c);
    }
    _sdfGrid.reset();
    double automaticBound = -1.0;
    if (dynamic_cast<HomogeneousMean *>(_mean.get())) automaticBound = 0.0;
    if (dynamic_cast<SphericalMean *>(_mean.get())) automaticBound = 1.0;
    if (dynamic_cast<LinearMean *>(_mean.get())) automaticBound = std::abs(_mean->lipschitz());
    if (auto tabulated = dynamic_cast<TabulatedMean *>(_mean.get())) {
        if (tabulated->isVolume()) throw std::runtime_error("classic_global needs an SDF, not a volume density mean");
#if OPENVDB_AVAILABLE
        if (auto vdb = dynamic_cast<VdbGrid *>(tabulated->grid().get())) {
            _sdfGrid = std::make_unique<GpisSdfGrid>();
            automaticBound = _sdfGrid->initialize(*tabulated, *vdb);
        }
#endif
    }
    if (automaticBound < 0.0 && (_configuredLipschitz < 0.0 || _configuredGradientBound < 0.0))
        throw std::runtime_error("classic_global: this mean needs explicit sdf_lipschitz and gradient_bound in world units");
    if ((_configuredLipschitz >= 0.0 && _configuredLipschitz < automaticBound) ||
        (_configuredGradientBound >= 0.0 && _configuredGradientBound < automaticBound))
        throw std::runtime_error("classic_global: supplied bounds are smaller than the certified SDF bounds");
    _lipschitz = _configuredLipschitz >= 0.0 ? _configuredLipschitz : automaticBound;
    _gradientBound = _configuredGradientBound >= 0.0 ? _configuredGradientBound : automaticBound;
    if (!std::isfinite(_lipschitz) || !std::isfinite(_gradientBound))
        throw std::runtime_error("classic_global: non-finite SDF bounds");
}

double ClassicGlobalMedium::meanValue(const Vec3d &p) const
{
    if (!eigen(p).allFinite()) throw std::runtime_error("classic_global: non-finite query position");
    double value;
#if OPENVDB_AVAILABLE
    if (_sdfGrid) value = _sdfGrid->evaluate(p, nullptr); else
#endif
    value = (*_mean)(Derivative::None, p, Vec3d(0.0));
    if (!std::isfinite(value)) throw std::runtime_error("classic_global: non-finite SDF value");
    return value;
}

Vec3d ClassicGlobalMedium::meanGradient(const Vec3d &p) const
{
    if (!eigen(p).allFinite()) throw std::runtime_error("classic_global: non-finite query position");
    Vec3d g;
#if OPENVDB_AVAILABLE
    if (_sdfGrid) _sdfGrid->evaluate(p, &g); else
#endif
    g = _mean->dmean_da(p);
    if (!eigen(g).allFinite()) throw std::runtime_error("classic_global: undefined SDF gradient");
    if (g.length() > _gradientBound*(1.0 + 1e-5) + 1e-12)
        throw std::runtime_error("classic_global: SDF gradient exceeds gradient_bound");
    return g;
}

double ClassicGlobalMedium::extinction(const Vec3d &p, const Vec3d &travelDirection) const
{
    const Eigen::Vector3d w = eigen(travelDirection).normalized();
    if (_nonstationary) {
        const auto s = pointStatistics(p);
        return ClassicGlobal::density(s.normalizedMean,1.0)*ClassicGlobal::negativePartMean(
            w.dot(s.gradientMean),std::sqrt(w.dot(s.gradientCovariance*w)));
    }
    return ClassicGlobal::density(meanValue(p), _sigma)*_ndf->projectedArea(eigen(meanGradient(p)), w);
}

ClassicGlobal::PointStatistics ClassicGlobalMedium::pointStatistics(const Vec3d &p) const
{
    const double mean = meanValue(p);
    const Eigen::Vector3d gradient = eigen(meanGradient(p));
    if (_nonstationary) return _nonstationary->statistics(p,mean,gradient);
    return {_sigma, mean/_sigma, gradient/_sigma, _ndf->covariance()/(_sigma*_sigma)};
}

double ClassicGlobalMedium::phasePdf(const Vec3d &p, const Eigen::Vector3d &wi, const Eigen::Vector3d &wo) const
{
    if (!_nonstationary) return _ndf->phasePdf(eigen(meanGradient(p)),wi,wo);
    const auto s = pointStatistics(p);
    return ClassicGlobal::GaussianNdf(s.gradientCovariance).phasePdf(s.gradientMean,wi,wo);
}

Eigen::Vector3d ClassicGlobalMedium::sampleVisibleNormal(const Vec3d &p, const Eigen::Vector3d &w,
                                                       PathSampleGenerator &sampler) const
{
    if (!_nonstationary) return _ndf->sampleVisibleNormal(eigen(meanGradient(p)),w,sampler);
    const auto s = pointStatistics(p);
    return ClassicGlobal::GaussianNdf(s.gradientCovariance).sampleVisibleNormal(s.gradientMean,w,sampler);
}

double ClassicGlobalMedium::fieldSigma() const
{
    if (_nonstationary) throw std::runtime_error("classic_global: fieldSigma requires a position; use pointStatistics(p).sigma");
    return _sigma;
}
const ClassicGlobal::GaussianNdf &ClassicGlobalMedium::ndf() const
{
    if (_nonstationary) throw std::runtime_error("classic_global: nonstationary NDF requires pointStatistics(p)");
    return *_ndf;
}

double ClassicGlobalMedium::track(PathSampleGenerator &sampler, const Ray &ray,
                                  bool sampleCollision, double &eventT) const
{
    const double nearT = ray.nearT(), farT = ray.farT();
    const Vec3d origin(ray.pos()), direction(ray.dir());
    const double speed = direction.length();
    if (!std::isfinite(nearT) || !std::isfinite(farT) || nearT < 0.0 || farT < 0.0 ||
        !(speed > 0.0) || !std::isfinite(speed))
        throw std::runtime_error("classic_global needs a finite ray segment; enclose the medium in a null boundary");
    eventT = farT;
    // A recovered boundary exit can lie inside the geometry epsilon. It must
    // still update the current medium, but contains no sampleable flight.
    if (farT <= nearT) return 1.0;
    const Eigen::Vector3d w = eigen(direction)/speed;
    const double areaBound = _nonstationary ? 0.0 : ClassicGlobal::negativePartMean(-_gradientBound,
                              std::sqrt(w.dot(_ndf->covariance()*w)));
    const double baseStep = _trackingStep > 0.0 ? _trackingStep
        : (_lipschitz > 0.0 ? _sigma/_lipschitz : (farT-nearT)*speed);
    double weight = 1.0;
    for (double begin = nearT; begin < farT;) {
        // Longer empty-space segments remain conservative; no SDF threshold
        // truncates the Gaussian tail. Only efficiency depends on this step.
        const double m0 = meanValue(origin + direction*begin);
        const double length = _lipschitz > 0.0
            ? std::max(baseStep, 0.5*std::max(0.0, m0)/_lipschitz) : (farT-begin)*speed;
        const double end = std::min(farT, begin + length/speed);
        if (!(end > begin)) throw std::runtime_error("classic_global: tracking step below floating point resolution");
        const double mid = 0.5*(begin + end);
        const double m = meanValue(origin + direction*mid);
        const double lower = m - _lipschitz*(end-begin)*speed*0.5 - 1e-12*std::max(_sigma, std::abs(m));
        const double majorant = _nonstationary
            ? _nonstationary->majorant(origin+direction*mid, (end-begin)*speed*0.5,
                                       lower, 2.0*m-lower, _gradientBound)
            : ClassicGlobal::density(lower, _sigma)*areaBound*(1.0 + 1e-5);
        if (!std::isfinite(majorant)) throw std::runtime_error("classic_global: non-finite extinction bound");
        if (majorant > 0.0) {
            double t = begin;
            for (;;) {
                const double next = t - std::log(ClassicGlobal::openUniform(sampler))/(majorant*speed);
                if (next >= end) break;
                if (!(next > t)) throw std::runtime_error("classic_global: collision distance below floating point resolution");
                t = next;
                const double acceptance = extinction(origin + direction*t, direction)/majorant;
                if (!std::isfinite(acceptance) || acceptance < 0.0 || acceptance > 1.0)
                    throw std::runtime_error("classic_global: invalid extinction majorant; check SDF bounds");
                if (sampleCollision) {
                    if (ClassicGlobal::openUniform(sampler) < acceptance) { eventT = t; return 1.0; }
                } else {
                    weight *= 1.0 - acceptance;
                    // Unbiased roulette prevents long shadow walks in opaque interiors.
                    if (weight < 0.1) {
                        if (ClassicGlobal::openUniform(sampler) >= weight/0.1) return 0.0;
                        weight = 0.1;
                    }
                }
            }
        }
        begin = end;
    }
    return weight;
}

bool ClassicGlobalMedium::sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
                                         MediumState &state, MediumSample &sample) const
{
    if (state.bounce > _maxBounce) return false;
    double eventT;
    track(sampler, ray, true, eventT);
    sample = MediumSample{};
    sample.t = float(eventT);
    sample.exited = eventT == double(ray.farT());
    // Preserve an interior event when conversion rounds onto the geometry endpoint.
    if (!sample.exited && sample.t >= ray.farT())
        sample.t = std::nextafter(ray.farT(), ray.nearT());
    sample.p = ray.pos() + sample.t*ray.dir();
    sample.phase = _phaseFunction.get();
    sample.weight = sample.continuedWeight = Vec3f(1.0f);
    sample.continuedT = sample.t;
    sample.emission = Vec3f(0.0f);
    // Delta tracking cancels the sampled free-flight measure from throughput.
    // A marginalized distance PDF is unavailable; pdf() explicitly rejects BDPT.
    sample.pdf = 1.0f;
    sample.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
    sample.rayInfo = state.info;
    if (!sample.exited) {
        if(hasDiffuseMicroMaterial())
            sample.aniso=vec(sampleVisibleNormal(Vec3d(sample.p),
                            eigen(Vec3d(ray.dir())).normalized(),sampler));
        state.advance();
    }
    return true;
}

Vec3f ClassicGlobalMedium::transmittance(PathSampleGenerator &sampler, const Ray &ray,
        bool /*startOnSurface*/, bool /*endOnSurface*/, MediumState * /*state*/) const
{
    double unused;
    return Vec3f(float(track(sampler, ray, false, unused)));
}

float ClassicGlobalMedium::pdf(PathSampleGenerator &, const Ray &, bool, bool) const
{
    throw std::runtime_error("classic_global supports path_tracer; marginalized distance PDFs for BDPT are not implemented");
}
Vec3f ClassicGlobalMedium::sigmaA(Vec3f) const { throw std::runtime_error("classic_global uses directional extinction and phase reflectance for absorption"); }
Vec3f ClassicGlobalMedium::sigmaS(Vec3f) const { throw std::runtime_error("classic_global sigmaS requires a travel direction"); }
Vec3f ClassicGlobalMedium::sigmaT(Vec3f) const { throw std::runtime_error("classic_global sigmaT requires a travel direction; use extinction(p, w)"); }

Vec3f ClassicGlobalMedium::reflectance(double cosine) const
{
    // Mirror and Lambert micro-materials return constant albedo. Their angular
    // scattering differs; the conductor retains its angle-dependent Fresnel.
    if (_materialType == "mirror" || _materialType == "lambert") return Vec3f(_mirrorAlbedo);
    if (_forceUnitFresnel) return Vec3f(1.0f);
    const double c = std::clamp(std::abs(cosine), 0.0, 1.0);
    if (c == 0.0) return Vec3f(1.0f);
    Vec3f result;
    for (int i = 0; i < 3; ++i) {
        const std::complex<double> index(_eta[i], _k[i]);
        const auto n2 = index*index;
        const auto gamma = std::sqrt(n2 - (1.0-c*c));
        const auto rs = (c-gamma)/(c+gamma);
        const auto rp = (n2*c-gamma)/(n2*c+gamma);
        result[i] = float(0.5*(std::norm(rs) + std::norm(rp)));
    }
    return result;
}
}
