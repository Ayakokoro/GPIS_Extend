#include "RenewalPlusFpGradientMedium.hpp"
#include "io/JsonObject.hpp"
#include "math/conditional/transport/FixedSpeedGradientDistribution.hpp"
#include <iostream>
#include <stdexcept>

namespace Tungsten {
namespace cg = ConditionalGpis;
namespace fp = FirstPassageGradient;
namespace {
Eigen::Vector3d eigen(const Vec3d &v) { return {v.x(), v.y(), v.z()}; }
Vec3d vec(const Eigen::Vector3d &v) { return {v.x(), v.y(), v.z()}; }
}

void RenewalPlusFpGradientMedium::fromJson(JsonPtr value, const Scene &scene) {
    SparseConvolutionNoiseMedium::fromJson(value, scene);
    _enabled = true;
    _bridgeCorrection = true;
    _selection = fp::AnchorSelection::Fixed5Snr;
    if (auto settings = value["fp_gradient"]) {
        settings.getField("enabled", _enabled);
        settings.getField("bridge_correction", _bridgeCorrection);
        std::string name = "fixed5_snr";
        settings.getField("anchor_selection", name);
        try { _selection = fp::parseAnchorSelection(name); }
        catch (const std::invalid_argument &e) { settings.parseError(e.what()); }
    }
    // The native 1D NEE/MIS code evaluates a different normal law. Keep the
    // observed-normal BRDF estimator used by the existing 3D renewal+ image.
    bool oneD = false, single = false, separate = false;
    std::string scheme = "uni";
    value.getField("1D_sampling", oneD);
    value.getField("single_realization", single);
    value.getField("surf_vol_phase_separate", separate);
    value.getField("1D_sampling_scheme", scheme);
    if (_enabled && (_ctxt != GPCorrelationContext::RenewalPlus ||
            _intersectMethod != GPIntersectMethod::GPDiscrete ||
            _normalSamplingMethod != GPNormalSamplingMethod::ConditionedGaussian ||
            oneD || single || separate || stringToSamplingScheme1D(scheme) != SparseConv1DSamplingScheme::UNI))
        value.parseError("renewal_plus_fp_gradient requires renewal+, gp_discrete, conditioned_gaussian, 3D sampling, and uni");
}

rapidjson::Value RenewalPlusFpGradientMedium::toJson(Allocator &a) const {
    auto value = SparseConvolutionNoiseMedium::toJson(a);
    value["type"].SetString("renewal_plus_fp_gradient", a);
    value.AddMember("fp_gradient", rapidjson::Value(JsonObject{a,
        "enabled", _enabled, "bridge_correction", _bridgeCorrection,
        "anchor_selection", fp::anchorSelectionName(_selection)}), a);
    return value;
}

void RenewalPlusFpGradientMedium::prepareForRender() {
    SparseConvolutionNoiseMedium::prepareForRender();
    _collisions = 0; _corrected = 0; _escapes = 0;
    _exteriorBirths = 0; _surfaceBirths = 0; _fallbacks = 0;
    _unsupportedBirths = 0; _zeroLength = 0;
    if (!_enabled) return;
    _process = std::dynamic_pointer_cast<GaussianProcess>(_gp);
    if (!_process || _process->_mean_additional || !_process->_globalCondPs.empty() ||
            _process->_embedCov || !dynamic_cast<SquaredExponentialCovariance *>(_process->_cov.get()))
        throw std::runtime_error("renewal_plus_fp_gradient requires one stationary SE GP without embedding or extra observations");
    const Vec3d zero(0.0);
    const auto &cov = *_process->_cov;
    const double variance = cov(Derivative::None, Derivative::None, zero, zero, zero, zero);
    Eigen::Matrix3d gradients;
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) {
        Vec3d a(0.0), b(0.0); a[i] = 1; b[j] = 1;
        gradients(i,j) = cov(Derivative::First, Derivative::First, zero, zero, a, b);
    }
    _kernel = cg::CovarianceKernel(std::sqrt(variance), gradients / variance);
}

void RenewalPlusFpGradientMedium::onDistanceSample(PathSampleGenerator &sampler,
        const Ray &ray, double distance, MediumState &state, MediumSample &sample) const {
    if (!_enabled) return;
    if (sample.exited) { ++_escapes; return; }
    ++_collisions;
    // Preserve exceptional native events instead of changing the flight law.
    // These retained-native cases are explicitly counted in the render report.
    if (!(distance > 0)) { ++_zeroLength; return; }
    auto *context = dynamic_cast<GPContextSparseConvNoise *>(state.gpContext.get());
    if (!context) throw std::runtime_error("missing renewal+ segment context");
    const Vec3d origin(ray.pos());
    const Eigen::Vector3d w = eigen(Vec3d(ray.dir())).normalized();
    double value = context->birthValue;
    Vec3d gradient = context->birthGradient;
    if (!context->conditionedBirth) {
        // For camera/exterior starts, observe the SAME realization that found
        // this hit. Drawing an unrelated positive jet would change the target.
        UniformSampler uniform;
        int id;
        value = context->noise.evaluateValue(ray.pos(), 0, ray.dir(), context->birthRayInfo, uniform, id);
        gradient = Vec3d(context->noise.evaluateGradient(ray.pos(), 0, ray.dir(), context->birthRayInfo, uniform));
    }
    const double birthDerivative = w.dot(eigen(gradient));
    if (!(value >= 0) || !std::isfinite(value) || !eigen(gradient).allFinite() ||
            (value == 0 && !(birthDerivative > 0))) {
        ++_unsupportedBirths;
        return;
    }
    const auto &mean = *_process->_mean;
    const Vec3d position = origin + distance * vec(w);
    const Eigen::Vector3d startMeanGradient = eigen(mean.dmean_da(origin));
    const Eigen::Vector3d endMeanGradient = eigen(mean.dmean_da(position));
    const double gamma = std::sqrt(w.dot(_kernel.metric()*w));
    const fp::Input input{_kernel.sigma(), gamma, distance, value, birthDerivative,
        mean(Derivative::None, origin, Vec3d(0.0)), w.dot(startMeanGradient), w.dot(endMeanGradient),
        [&](double s) { return mean(Derivative::None, origin + s*vec(w), Vec3d(0.0)); }};
    const fp::Law law(input, _bridgeCorrection, _selection);
    cg::Random random(sampler);
    bool fallback = false;
    const double derivative = law.sample(random, &fallback);
    const cg::FixedSpeedGradientDistribution distribution(_kernel, w, gamma*distance,
        derivative, endMeanGradient, startMeanGradient, eigen(gradient));
    const Eigen::Vector3d corrected = distribution.sample(random);
    if (!corrected.allFinite() || !(w.dot(corrected) < 0))
        throw std::runtime_error("invalid renewal+ fixed5 gradient");
    // The existing BRDF phase treats this full vector as an observed latent.
    // Its conditional phase PDF/weight remain valid. Native renewal+ reads
    // lastAniso when conditioning the next path or shadow segment.
    sample.aniso = state.lastAniso = vec(corrected);
    if (state.distanceDiagnostic) state.distanceDiagnostic->gradient = sample.aniso;
    ++_corrected;
    if (value == 0) ++_surfaceBirths; else ++_exteriorBirths;
    if (fallback) ++_fallbacks;
}

void RenewalPlusFpGradientMedium::teardownAfterRender() {
    std::cout << "GPIS renewal+ FP-gradient statistics: {\"collisions\":" << _collisions.load()
        << ",\"corrected_collisions\":" << _corrected.load()
        << ",\"escapes\":" << _escapes.load()
        << ",\"exterior_births\":" << _exteriorBirths.load()
        << ",\"surface_births\":" << _surfaceBirths.load()
        << ",\"sampling_fallbacks\":" << _fallbacks.load()
        << ",\"retained_native_unsupported_births\":" << _unsupportedBirths.load()
        << ",\"retained_native_zero_length\":" << _zeroLength.load() << "}\n";
    SparseConvolutionNoiseMedium::teardownAfterRender();
}

} // namespace Tungsten
