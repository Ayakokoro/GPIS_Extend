#include "GaussianProcessMedium.hpp"
#include "math/FunctionSpaceProfile.hpp"

#include <cfloat>

#include "sampling/PathSampleGenerator.hpp"

#include "math/GaussianProcess.hpp"
#include "math/TangentFrame.hpp"
#include "math/Ray.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "bsdfs/Microfacet.hpp"
#include <bsdfs/NDFs/beckmann.h>
#include <bsdfs/NDFs/GGX.h>

namespace Tungsten {

    std::string GaussianProcessMedium::correlationContextToString(GPCorrelationContext ctxt)
    {
        switch (ctxt) {
        default:
        case GPCorrelationContext::Global:  return "global";
        case GPCorrelationContext::RenewalPlus:   return "renewal+";
        case GPCorrelationContext::Renewal:   return "renewal";
        case GPCorrelationContext::None:   return "none";
        }
    }

    GPCorrelationContext GaussianProcessMedium::stringToCorrelationContext(const std::string& name)
    {
        if (name == "global")
            return GPCorrelationContext::Global;
        else if (name == "renewal+")
            return GPCorrelationContext::RenewalPlus;
        else if (name == "renewal")
            return GPCorrelationContext::Renewal;
        else if (name == "none")
            return GPCorrelationContext::None;
        FAIL("Invalid correlation context: '%s'", name);
    }

    std::string GaussianProcessMedium::intersectMethodToString(GPIntersectMethod ctxt)
    {
        switch (ctxt) {
        default:
        case GPIntersectMethod::Mean:       return "mean";
        case GPIntersectMethod::GPDiscrete: return "gp_discrete";
        }
    }

    GPIntersectMethod GaussianProcessMedium::stringToIntersectMethod(const std::string& name)
    {
        if (name == "mean")
            return GPIntersectMethod::Mean;
        else if (name == "gp_discrete")
            return GPIntersectMethod::GPDiscrete;
        FAIL("Invalid intersect method: '%s'", name);
    }

    std::string GaussianProcessMedium::normalSamplingMethodToString(GPNormalSamplingMethod val)
    {
        switch (val) {
        default:
        case GPNormalSamplingMethod::FiniteDifferences:  return "finite_differences";
        case GPNormalSamplingMethod::ConditionedGaussian:   return "conditioned_gaussian";
        case GPNormalSamplingMethod::Beckmann:   return "beckmann";
        case GPNormalSamplingMethod::GGX:   return "ggx";
        }
    }

    GPNormalSamplingMethod GaussianProcessMedium::stringToNormalSamplingMethod(const std::string& name)
    {
        if (name == "finite_differences")
            return GPNormalSamplingMethod::FiniteDifferences;
        else if (name == "conditioned_gaussian")
            return GPNormalSamplingMethod::ConditionedGaussian;
        else if (name == "beckmann")
            return GPNormalSamplingMethod::Beckmann;
        else if (name == "ggx")
            return GPNormalSamplingMethod::GGX;
        FAIL("Invalid normal sampling method: '%s'", name);
    }


    GaussianProcessMedium::GaussianProcessMedium()
        : _materialSigmaA(0.0f),
        _materialSigmaS(0.0f),
        _density(1.0f),
        _gp(std::make_shared<GaussianProcess>(std::make_shared<SphericalMean>(), std::make_shared<SquaredExponentialCovariance>())),
        _ctxt(GPCorrelationContext::RenewalPlus),
        _intersectMethod(GPIntersectMethod::GPDiscrete),
        _normalSamplingMethod(GPNormalSamplingMethod::ConditionedGaussian)
    {
    }

    void GaussianProcessMedium::fromJson(JsonPtr value, const Scene& scene)
    {
        Medium::fromJson(value, scene);
        value.getField("sigma_a", _materialSigmaA);
        value.getField("sigma_s", _materialSigmaS);
        value.getField("density", _density);

        std::string ctxtString = "goldfish";
        value.getField("correlation_context", ctxtString);
        _ctxt = stringToCorrelationContext(ctxtString);

        std::string intersectString = "gp_discrete";
        value.getField("intersect_method", intersectString);
        _intersectMethod = stringToIntersectMethod(intersectString);

        std::string normalString = "conditioned_gaussian";
        value.getField("normal_method", normalString);
        _normalSamplingMethod = stringToNormalSamplingMethod(normalString);

        if (auto gp = value["gaussian_process"])
            _gp = scene.fetchGaussianProcess(gp);

        // We always have the default one
        _phaseFunctions.push_back(_phaseFunction);

        if (auto addPhaseFunctions = value["additional_phase_functions"]) {
            for (unsigned i = 0; i < addPhaseFunctions.size(); ++i)
                _phaseFunctions.emplace_back(scene.fetchPhase(addPhaseFunctions[i]));
        }
    }

    rapidjson::Value GaussianProcessMedium::toJson(Allocator& allocator) const
    {
        return JsonObject{ Medium::toJson(allocator), allocator,
            "type", "gaussian_process",
            "sigma_a", _materialSigmaA,
            "sigma_s", _materialSigmaS,
            "density", _density,
            "gaussian_process", *_gp,
            "correlation_context", correlationContextToString(_ctxt),
            "intersect_method", intersectMethodToString(_intersectMethod),
            "normal_method", normalSamplingMethodToString(_normalSamplingMethod),
        };
    }

    void GaussianProcessMedium::loadResources() {
        _gp->loadResources();
    }


    bool GaussianProcessMedium::isHomogeneous() const
    {
        return false;
    }

    void GaussianProcessMedium::prepareForRender()
    {
        _sigmaA = _materialSigmaA * _density;
        _sigmaS = _materialSigmaS * _density;
        _sigmaT = _sigmaA + _sigmaS;
        _absorptionOnly = _sigmaS == 0.0f;
    }

    Vec3f GaussianProcessMedium::sigmaA(Vec3f /*p*/) const
    {
        return _sigmaA;
    }

    Vec3f GaussianProcessMedium::sigmaS(Vec3f /*p*/) const
    {
        return _sigmaS;
    }

    Vec3f GaussianProcessMedium::sigmaT(Vec3f /*p*/) const
    {
        return _sigmaT;
    }

    bool GaussianProcessMedium::intersect(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t, bool firstIsectAlongRay) const {
        switch (_intersectMethod) {
        case GPIntersectMethod::Mean:
            return intersectMean(sampler, ray, state, t);
        case GPIntersectMethod::GPDiscrete:
            return intersectGP(sampler, ray, state, t, firstIsectAlongRay);
        default:
            return false;
        }
    }

    bool GaussianProcessMedium::intersectMean(PathSampleGenerator& sampler, const Ray& ray, MediumState& state, double& t) const {
        auto gp = _gp->get_gaussian_process();

        t = ray.nearT() + 0.0001f;
        auto deriv = Derivative::None;
        auto rp = vec_conv<Vec3d>(ray.pos());
        auto rd = vec_conv<Vec3d>(ray.dir());

        for(int i = 0; i < 2048*4; i++) {
            auto p = rp + t * rd;
            Eigen::Vector2d mean_and_id = gp->mean_weight_space(p, deriv);

            if(mean_and_id[0] < 0.00001f) {
                auto ctxt = std::make_shared<GPContextFunctionSpace>();
                ctxt->derivs = { Derivative::None };
                ctxt->points = { p };
                //ctxt->values = { 0. };
                state.gpContext = ctxt;
                state.lastGPId = mean_and_id[1];
                return true;
            }

            t += mean_and_id[0] * 0.4f;

            if(t >= ray.farT()) {
                return false;
            }
        }

        state.gpContext = std::make_shared<GPContextFunctionSpace>();
        std::cerr << "Ran out of iterations in mean intersect sphere trace." << std::endl;
        t = ray.farT();
        return false;
    }

    bool GaussianProcessMedium::sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
        MediumState &state, MediumSample &sample) const
    {
        FsProfile::PurposeScope purpose(false);
        DistanceState distance;
        auto status = beginDistance(sampler, ray, state, sample, distance);
        while (status == DistanceStatus::NeedIntersection) {
            double t = distance.t;
            bool hit = intersect(sampler, distance.ray, state, t, distance.firstIntersection);
            status = resumeDistance(sampler, ray, state, sample, distance, t, hit);
        }
        return status == DistanceStatus::Finished;
    }

    GaussianProcessMedium::DistanceStatus GaussianProcessMedium::beginDistance(
        PathSampleGenerator &sampler, const Ray &ray, MediumState &state,
        MediumSample &sample, DistanceState &distance) const
    {
        sample.emission = Vec3f(0.0f);
        auto r = ray;

        double startT = r.nearT();
        if (!std::isfinite(r.farT())) {
            r.setFarT(startT + 2000);
        }

        float maxT = r.farT();

        if (state.bounce >= _maxBounce) {
            if (state.distanceDiagnostic) state.distanceDiagnostic->failure = "medium_bounce_limit";
            return DistanceStatus::Failed;
        }

        if (maxT <= r.nearT()) {
            sample.t = maxT;
            sample.weight = Vec3f(1.f);
            sample.pdf = 1.0f;
            sample.exited = true;
            sample.p = ray.pos() + sample.t * ray.dir();
            sample.phase = _phaseFunction.get();
            sample.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
            return DistanceStatus::Finished;
        }

        if (_absorptionOnly) {
            if (maxT == Ray::infinity())
                return DistanceStatus::Failed;
            sample.t = maxT;
            sample.weight = transmittance(sampler, ray, false, false, &state);
            sample.pdf = 1.0f;
            sample.exited = true;
            sample.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
        }
        else {
            distance.ray = r;
            distance.maxT = maxT;
            distance.t = maxT;
            distance.origin = vec_conv<Vec3d>(r.pos());
            distance.direction = vec_conv<Vec3d>(r.dir()).normalized();
            distance.firstIntersection = true;
            return DistanceStatus::NeedIntersection;
        }
        finishDistance(ray, state, sample);
        return DistanceStatus::Finished;
    }

    GaussianProcessMedium::DistanceStatus GaussianProcessMedium::resumeDistance(
        PathSampleGenerator &sampler, const Ray &ray, MediumState &state,
        MediumSample &sample, DistanceState &distance, double t, bool hit) const
    {
        const auto &r = distance.ray;
        const auto &ro = distance.origin;
        const auto &rd = distance.direction;
        const float maxT = distance.maxT;
        distance.firstIntersection = false;
        distance.t = t;
        sample.exited = !hit;
        // We sample a gradient if:
        // (1) The ray did intersect a surface
        // (2) The ray did not intersect a surface, but we'll need to continue
        if (t < maxT) {

            Vec3d ip = ro + rd * t;

            Vec3d grad;
            if (!sampleGradient(sampler, r, ip, t, state, grad)) {
                if (state.distanceDiagnostic) state.distanceDiagnostic->failure = "gradient_sampling_failed";
                std::cout << "Failed to sample gradient.\n";
                return DistanceStatus::Failed;
            }

            state.lastAniso = sample.aniso = grad;
            if (state.distanceDiagnostic) {
                state.distanceDiagnostic->hasCandidate = true;
                state.distanceDiagnostic->position = ip;
                state.distanceDiagnostic->gradient = grad;
            }
            state.firstScatter = false;

            if (!std::isfinite(sample.aniso.avg())) {
                if (state.distanceDiagnostic) state.distanceDiagnostic->failure = "nonfinite_gradient";
                sample.aniso = Vec3d(1.f, 0.f, 0.f);
                std::cout << "Gradient invalid in sampleDistance: " << grad << std::endl;
                return DistanceStatus::Failed;
            }
        }


        if (t < maxT && sample.exited) {
            distance.ray.setNearT(float(t));
            return DistanceStatus::NeedIntersection;
        }
        if (!sample.exited) {
            if (sample.aniso.dot(vec_conv<Vec3d>(ray.dir())) > 0) {
                if (state.distanceDiagnostic) state.distanceDiagnostic->failure = "outward_crossing_rejected";
                return DistanceStatus::Failed;
            }

            if (sample.aniso.lengthSq() < 0.0000001f) {
                if (state.distanceDiagnostic) state.distanceDiagnostic->failure = "zero_gradient";
                sample.aniso = Vec3d(1.f, 0.f, 0.f);
                std::cout << "Gradient zero.\n";
                return DistanceStatus::Failed;
            }

            sample.weight = sample.continuedWeight = vec_conv<Vec3f>(_gp->color(ro + rd * t));
            sample.emission = vec_conv<Vec3f>(_gp->emission(ro + rd * t));
        } else {
            sampleGradient(sampler, r, ro + rd * t, t, state, sample.aniso);
            sample.weight = sample.continuedWeight = Vec3f(1.f);
        }

        sample.t = min(float(t), maxT);
        sample.continuedT = float(t);
        sample.weight *= sigmaS(ray.pos() + sample.t * ray.dir()) / sigmaT(ray.pos() + sample.t * ray.dir());
        sample.continuedWeight *= sigmaS(ray.pos() + sample.continuedT * ray.dir()) / sigmaT(ray.pos() + sample.continuedT * ray.dir());
        sample.pdf = 1;

        state.lastAniso = sample.aniso;
        sample.sparseConv1DSamplingScheme = state.sparseConv1DSamplingScheme;
        state.advance();
        finishDistance(ray, state, sample);
        return DistanceStatus::Finished;
    }

    void GaussianProcessMedium::finishDistance(const Ray &ray, MediumState &state,
        MediumSample &sample) const
    {
        sample.p = ray.pos() + sample.t * ray.dir();

        sample.phase = _phaseFunctions[state.lastGPId].get();
        sample.gpId = state.lastGPId;
        sample.ctxt = state.gpContext.get();
        state.info.t += sample.t;
        sample.rayInfo = state.info;
    }

void GaussianProcessMedium::beginTransmittance(const Ray &ray,
        TransmittanceState &transmittance) const
    {
        transmittance = TransmittanceState();
        transmittance.ray = ray;
        if (!std::isfinite(ray.farT()))
            transmittance.ray.setFarT(ray.nearT() + 2000);
        transmittance.maxT = transmittance.ray.farT();
        transmittance.t = transmittance.maxT;
    }

    GaussianProcessMedium::TransmittanceStatus GaussianProcessMedium::resumeTransmittance(
        PathSampleGenerator &sampler, const Ray &ray, MediumState &state,
        TransmittanceState &transmittance, double t, bool hit) const
    {
        transmittance.firstIntersection = false;
        transmittance.t = t;
        if (t < transmittance.maxT) {
            Vec3d ip = vec_conv<Vec3d>(ray.pos()) + vec_conv<Vec3d>(ray.dir()).normalized() * t;
            Vec3d grad;
            if (!sampleGradient(sampler, ray, ip, t, state, grad)) {
                std::cout << "Failed to sample gradient.\n";
                transmittance.value = Vec3f(0.0f);
                return TransmittanceStatus::Finished;
            }
            state.lastAniso = grad;
            state.firstScatter = false;
            if (!std::isfinite(grad.avg())) {
                std::cout << "Gradient invalid in transmittance: " << grad << std::endl;
                transmittance.value = Vec3f(0.0f);
                return TransmittanceStatus::Finished;
            }
        }
        if (t < transmittance.maxT && !hit) {
            transmittance.ray.setNearT(float(t));
            return TransmittanceStatus::NeedIntersection;
        }
        transmittance.value = hit ? Vec3f(0.0f) : Vec3f(1.0f);
        return TransmittanceStatus::Finished;
    }

    Vec3f GaussianProcessMedium::transmittance(PathSampleGenerator &sampler,
        const Ray &ray, bool, bool, MediumState *state) const
    {
        FsProfile::PurposeScope purpose(true);
        if (ray.farT() <= ray.nearT())
            return Vec3f(1.0f);
        TransmittanceState transmittance;
        beginTransmittance(ray, transmittance);
        TransmittanceStatus status;
        do {
            double t = transmittance.t;
            bool hit = intersect(sampler, transmittance.ray, *state, t,
                transmittance.firstIntersection);
            status = resumeTransmittance(sampler, ray, *state, transmittance, t, hit);
        } while (status == TransmittanceStatus::NeedIntersection);
        return transmittance.value;
    }

    float GaussianProcessMedium::pdf(PathSampleGenerator&/*sampler*/, const Ray & ray, bool startOnSurface, bool endOnSurface) const
    {
        return 1.0f;
    }
}
