#include "BRDFPhaseFunction.hpp"

#include "sampling/PathSampleGenerator.hpp"
#include "sampling/SampleWarp.hpp"
#include "samplerecords/SpecularGpisEvent.hpp"
#include "media/Medium.hpp"

#include "io/JsonObject.hpp"
#include "io/Scene.hpp"

namespace Tungsten {

void BRDFPhaseFunction::fromJson(JsonPtr value, const Scene& scene) {
    PhaseFunction::fromJson(value, scene);

    if (auto bsdf = value["bsdf"])
        _bsdf = scene.fetchBsdf(bsdf);
}


rapidjson::Value BRDFPhaseFunction::toJson(Allocator &allocator) const
{
    return JsonObject{PhaseFunction::toJson(allocator), allocator,
        "type", "brdf",
        "bsdf", * _bsdf,
    };
}

void BRDFPhaseFunction::setEventIsectInfo(const Vec3f &wi, const MediumSample &mediumSample, SurfaceScatterEvent& se, IntersectionInfo& info) const {
    auto normal = vec_conv<Vec3f>(mediumSample.aniso.normalized());
    se.requestedLobe = BsdfLobes::AllLobes;
    se.frame = TangentFrame(normal);
    se.wi = se.frame.toLocal(-wi).normalized();
    // Generic GP events are handled here; the legacy BSDF MIS branch assumes
    // GPContextSparseConvNoise and must never receive another context type.
    se.sparseConv1DSamplingScheme = dynamic_cast<const SpecularGpisEvent *>(mediumSample.eventContext.get())
        ? SparseConv1DSamplingScheme::UNI : mediumSample.sparseConv1DSamplingScheme;
    se.ctxt = mediumSample.ctxt;
    info.bsdf = _bsdf.get();
    info.Ng = normal;
    info.Ns = normal;
    info.p = mediumSample.p;
    info.primitive = nullptr;
    info.uv = Vec2f(0.f);
    info.w = wi;
    info.t = mediumSample.t;
    info.rayInfo = mediumSample.rayInfo;
    se.info = &info;
}

bool BRDFPhaseFunction::isSpecular() const {
    return _bsdf->lobes().isPureSpecular();
}

Vec3f BRDFPhaseFunction::eval(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const
{
    if (const auto *event=dynamic_cast<const SpecularGpisEvent *>(mediumSample.eventContext.get())) {
        const double density=event->reflectionPdf(wi,wo);
        if (!(density>0)) return Vec3f(0.0f);
        auto mapped=mediumSample;
        mapped.aniso=event->reflectionGradient(wi,wo);
        if (!mapped.aniso.lengthSq()) return Vec3f(0.0f);
        SurfaceScatterEvent se; IntersectionInfo info;
        setEventIsectInfo(wi,mapped,se,info);
        se.wo=se.frame.toLocal(wo).normalized();
        return _bsdf->eval(se)*float(density);
    }
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();
    return _bsdf->eval(se);
}

Vec3f BRDFPhaseFunction::evalGrad(const Vec3f &wi, const Vec3f &wo, const MediumSample &mediumSample) const
{
    if (const auto *event=dynamic_cast<const SpecularGpisEvent *>(mediumSample.eventContext.get()))
        return Vec3f(event->reflectionGradient(wi,wo));
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();
    return _bsdf->evalGrad(se);
}

bool BRDFPhaseFunction::sample(PathSampleGenerator &sampler, const Vec3f &wi, const MediumSample& mediumSample, PhaseSample &sample) const
{
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.sampler = &sampler;

    if (!_bsdf->sample(se)) return false;
    sample.w = se.frame.toGlobal(se.wo).normalized();
    sample.weight = se.weight;
    if (const auto *event=dynamic_cast<const SpecularGpisEvent *>(mediumSample.eventContext.get()))
        sample.pdf=event->reflectionPdf(wi,sample.w);
    else sample.pdf = se.pdf;
    return true;
}

bool BRDFPhaseFunction::invert(WritablePathSampleGenerator &sampler, const Vec3f &/*wi*/, const Vec3f &wo, const MediumSample& mediumSample) const
{
    return false;
}

float BRDFPhaseFunction::pdf(const Vec3f &wi, const Vec3f &wo, const MediumSample& mediumSample) const
{
    if (const auto *event=dynamic_cast<const SpecularGpisEvent *>(mediumSample.eventContext.get()))
        return float(event->reflectionPdf(wi,wo));
    SurfaceScatterEvent se;
    IntersectionInfo info;
    setEventIsectInfo(wi, mediumSample, se, info);
    se.wo = se.frame.toLocal(wo).normalized();

    return _bsdf->pdf(se);
}

}
