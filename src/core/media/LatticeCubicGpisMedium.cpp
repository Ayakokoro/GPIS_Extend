#include "LatticeCubicGpisMedium.hpp"
#include "ClassicGlobalFpGradientMedium.hpp"
#include "GpisSdfGrid.hpp"
#include "GpisSdfCubicRay.hpp"
#include "LatticeNonstationaryParameters.hpp"
#include "TwoLevelLatticeRayField.hpp"
#include "io/JsonObject.hpp"
#include "math/CubicDowncrossing.hpp"
#include "math/NonstationaryLatticeGaussianNoise.hpp"

namespace Tungsten {
namespace cg = ConditionalGpis;

struct VaryingLatticeModel {
    LatticeNonstationaryParameters parameters;
    NonstationaryLatticeGaussianKernel kernel;
    VaryingLatticeModel(const ProceduralNonstationaryCovariance &cov, float spacing, float radius)
    : parameters(cov), kernel(parameters.minEll,parameters.maxEll,parameters.maxSigma,
            parameters.maxSigmaGradient,parameters.maxEllGradient,spacing,radius) {}
};

namespace {
using Cubic = CubicDowncrossing::Cubic;
using Vector = Eigen::Vector3d;
Vector eigen(const Vec3f &p) { return {p.x(),p.y(),p.z()}; }
Vector eigen(const Vec3d &p) { return {p.x(),p.y(),p.z()}; }
Vec3d vec(const Vector &p) { return Vec3d(p.x(),p.y(),p.z()); }

using NoisePiece = LatticeNoisePiece;

struct StationaryRayNoise {
    const LatticeGaussianKernel &kernel;
    LatticeGaussianRay realization;
    StationaryRayNoise(const LatticeGaussianKernel &k, const Vector &, const Vector &, uint32 seed)
    : kernel(k), realization(k,seed) {}
    double kernelSpacing() const { return kernel.spacing; }
    NoisePiece piece(double t) const {
        return NoisePiece::containing(t,kernel.spacing);
    }
    Cubic segment(const NoisePiece &p) const { return realization.segment(p.cell); }
    double absoluteBound() const { return realization.absoluteBound(); }
    double boundUntil(double,double limit,double &end) const { end=limit; return absoluteBound(); }
    void condition(double value,std::optional<double> derivative) {
        realization.condition(float(value),float(derivative.value_or(0)),derivative.has_value());
    }
    void transverse(const NoisePiece &p,double t,Vector &,double &sd) const {
        sd=kernel.transverseStddev((t-p.start)/p.width());
    }
};

struct VaryingRayNoise {
    const VaryingLatticeModel &model;
    NonstationaryLatticeGaussianRay realization;
    Vector origin, direction;
    std::array<double,4> cuts;
    VaryingRayNoise(const VaryingLatticeModel &m,const Vector &o,const Vector &d,uint32 seed)
    : model(m), realization(m.kernel,seed), origin(o), direction(d), cuts(m.parameters.cuts(o,d)) {}
    double kernelSpacing() const { return model.kernel.spacing; }
    NoisePiece piece(double t) const {
        NoisePiece p=NoisePiece::containing(t,model.kernel.spacing);
        for (double cut : cuts) {
            if (cut>=p.end) break;
            if (cut<=t) p.start=std::max(p.start,cut);
            else { p.end=cut; break; }
        }
        return p;
    }
    LatticeNoiseParameters at(double t,int side) const { return model.parameters.at(origin,direction,t,side); }
    Cubic segment(const NoisePiece &p) const {
        return realization.segment(p.start,p.end,at(p.start,1),at(p.end,-1));
    }
    double absoluteBound() const { return realization.absoluteBound(); }
    double boundUntil(double begin,double limit,double &end) const {
        end=limit;
        for (double cut : cuts)
            if (cut>begin) { end=std::min(end,cut); break; }
        if (!model.parameters.constantAlongRay(origin,direction,begin)) return absoluteBound();
        const auto p=at(begin,1);
        return realization.absoluteBound(p.ell,p.ell,p.sigma,0,0);
    }
    void condition(double value,std::optional<double> derivative) {
        realization.condition(at(0,1),value,derivative);
    }
    void transverse(const NoisePiece &,double t,Vector &mean,double &sd) const {
        model.parameters.transverse(model.kernel,realization,origin,direction,t,mean,sd);
    }
};

template<class Noise> struct Flight {
    Vector origin, direction;
    Noise noise;
    // Absolute arclength from birth. A short transparent prefix can end before
    // the first probe, in which case its continuation inherits the same probe.
    std::optional<double> entryProbe;
    bool unconditioned=false;
    bool seekUpcrossing=false;
    // Round the two imposed observations only once, at the shared birth
    // knot. A Hermite correction vanishes with its slope at the next knot.
    double valueCorrection=0, slopeCorrection=0;

    Cubic segment(const NoisePiece &piece) const {
        Cubic c=noise.segment(piece);
        if (piece.start==0) {
            c[0]+=valueCorrection; c[1]+=slopeCorrection;
            c[2]-=3*valueCorrection+2*slopeCorrection;
            c[3]+=2*valueCorrection+slopeCorrection;
        }
        return c;
    }
    void condition(double residual, std::optional<double> derivative) {
        noise.condition(residual,derivative);
        const auto first=noise.piece(0);
        const auto p=noise.segment(first);
        valueCorrection=residual-p[0];
        if (derivative) slopeCorrection=(*derivative)*first.width()-p[1];
    }
};

// A forward subdivision continues this flight. An ordinary scattering
// surface observes its endpoint value and projected gradient for renewal+.
template<class Noise> struct Endpoint : MediumPathContext {
    const Medium *owner;
    Flight<Noise> flight;
    Vec3f rayOrigin, rayDirection;
    double age;
    uint32 transverseSeed;
    Endpoint(const Medium *m, Flight<Noise> &&f, const Ray &ray, double t, uint32 seed)
        : owner(m), flight(std::move(f)), rayOrigin(ray.hitpoint()), rayDirection(ray.dir()),
          age(t), transverseSeed(seed) {}
};

} // namespace

LatticeCubicGpisMedium::LatticeCubicGpisMedium() {
    _phaseFunction=makeFixedGradientPhase(*this,false);
    _deltaPhase=makeFixedGradientPhase(*this,true);
}
LatticeCubicGpisMedium::~LatticeCubicGpisMedium() = default;
void LatticeCubicGpisMedium::fromJson(JsonPtr value, const Scene &scene) {
    ClassicGlobalMedium::fromJson(value,scene);
    std::string correlationContext="renewal+";
    value.getField("correlation_context",correlationContext);
    if (correlationContext!="renewal+")
        value.parseError("lattice_cubic_gpis supports only correlation_context renewal+");
    value.getField("lattice_spacing_in_ell",_spacingInEll);
    value.getField("kernel_radius_in_ell",_radiusInEll);
    value.getField("lattice_multiresolution",_multiResolution);
    if (_multiResolution && _radiusInEll!=3.0f)
        value.parseError("lattice_multiresolution uses the sparse convolution support; remove the custom kernel_radius_in_ell");
    value.getField("step_size",_entryStepSize);
    if (!std::isfinite(_entryStepSize) || _entryStepSize<=0)
        value.parseError("lattice_cubic_gpis step_size must be finite and positive");
    value.getField("geometry_pruning",_prune);
    std::string mode="conservative";
    value.getField("geometry_pruning_mode",mode);
    if (mode!="conservative" && mode!="three_sigma")
        value.parseError("geometry_pruning_mode must be conservative or three_sigma");
    _threeSigmaPruning=mode=="three_sigma";
}
rapidjson::Value LatticeCubicGpisMedium::toJson(Allocator &allocator) const {
    auto value=ClassicGlobalMedium::toJson(allocator);
    value["type"].SetString("lattice_cubic_gpis",allocator);
    value.RemoveMember("tracking_step");
    value.AddMember("correlation_context","renewal+",allocator);
    value.AddMember("lattice_spacing_in_ell",_spacingInEll,allocator);
    if (_multiResolution) value.AddMember("lattice_multiresolution",true,allocator);
    else value.AddMember("kernel_radius_in_ell",_radiusInEll,allocator);
    value.AddMember("step_size",_entryStepSize,allocator);
    value.AddMember("geometry_pruning",_prune,allocator);
    value.AddMember("geometry_pruning_mode",
        JsonUtils::toJson(std::string(_threeSigmaPruning ? "three_sigma" : "conservative"),allocator),allocator);
    return value;
}
void LatticeCubicGpisMedium::prepareForRender() {
    const auto *se=dynamic_cast<const SquaredExponentialCovariance *>(_covariance.get());
    const auto *varying=dynamic_cast<const ProceduralNonstationaryCovariance *>(_covariance.get());
    if (!varying && (!se || !se->isIsotropic() || !se->isStationaryKernel()))
        throw std::invalid_argument("lattice_cubic_gpis requires isotropic squared_exponential or proc_nonstationary SE covariance");
    ClassicGlobalMedium::prepareForRender();
    if (!_sdfGrid) throw std::invalid_argument("lattice_cubic_gpis requires a trilinear OpenVDB SDF");
    _kernel.reset();
    _varying.reset();
    _twoLevel.reset();
    if (varying && _multiResolution) _twoLevel=std::make_unique<TwoLevelLatticeModel>(*varying,_spacingInEll);
    else if (varying) _varying=std::make_unique<VaryingLatticeModel>(*varying,_spacingInEll,_radiusInEll);
    else _kernel=std::make_unique<LatticeGaussianKernel>(float(_sigma),se->lengthScale(),_spacingInEll,_radiusInEll);
#if OPENVDB_AVAILABLE
    if (_prune) _sdfGrid->buildMacrocells(8,true);
#endif
}

std::optional<float> LatticeCubicGpisMedium::trace(PathSampleGenerator &sampler, const Ray &ray,
                                   MediumState *state, MediumSample *sample) const {
    if (_twoLevel) return traceField<TwoLevelLatticeRayField>(*_twoLevel,sampler,ray,state,sample);
    if (_varying) return traceField<VaryingRayNoise>(*_varying,sampler,ray,state,sample);
    return traceField<StationaryRayNoise>(*_kernel,sampler,ray,state,sample);
}

template<class Noise,class Model>
std::optional<float> LatticeCubicGpisMedium::traceField(const Model &model,PathSampleGenerator &sampler,
                                       const Ray &ray,MediumState *state,MediumSample *sample) const {
#if OPENVDB_AVAILABLE
    if (!std::isfinite(ray.farT()) || !std::isfinite(ray.nearT()) || ray.farT()<0)
        throw std::invalid_argument("lattice_cubic_gpis requires a finite boundary-enclosed ray");
    if (ray.farT()==0) return std::nullopt;
    const Vector travel=eigen(ray.dir());
    const double speed=travel.norm();
    if (!(speed>0) || !std::isfinite(speed)) throw std::invalid_argument("invalid lattice ray direction");
    const Vector direction=travel/speed;
    cg::Random rng(sampler);
    const auto *context=state ? state->pathContext.get() : nullptr;
    auto *end=dynamic_cast<const Endpoint<Noise> *>(context);
    if (end && (end->owner!=this || end->rayOrigin!=ray.pos())) end=nullptr;
    const auto *hit=dynamic_cast<const FpGradientPathContext *>(context);
    if (hit && (hit->owner!=this || hit->rayOrigin!=ray.pos())) hit=nullptr;
    const bool straight=end && !state->startsAtScatteringSurface && end->rayDirection==ray.dir();
    if (state) state->startsAtScatteringSurface=false;
    const double age=straight ? end->age : 0;
    // Parameter fields and their breakpoints are evaluated on the same
    // physical line as the SDF, including the phase's unrounded reflection.
    const Vector origin=hit ? hit->origin : end ? end->flight.origin+end->age*end->flight.direction : eigen(ray.pos());
    const bool retainedDirection=hit && (hit->direction-direction).squaredNorm()<1e-12;
    const Vector axis=retainedDirection ? hit->direction : direction;
    Flight<Noise> flight=straight ? end->flight : Flight<Noise>{origin,axis,
        Noise(model,origin,axis,sampler.uniformGenerator().nextI())};
    if (!straight) {
        flight.unconditioned=!end && !hit;
        bool inwardBirth=false;
        if (end || hit) {
            double observedValue=0, observedDerivative=0;
            if (end) {
                // A seed lookup may fill a cache. Keep those writes local
                // when several branches turn at the same immutable endpoint.
                Flight<Noise> prev=end->flight;
                const auto piece=prev.noise.piece(end->age);
                const double u=(end->age-piece.start)/piece.width();
                const Cubic p=prev.segment(piece);
                Vec3d g;
                observedValue=_sdfGrid->evaluate(vec(flight.origin),&g)+CubicDowncrossing::value(p,u);
                const double axial=eigen(g).dot(prev.direction)+CubicDowncrossing::slope(p,u)/piece.width();
                Vector mean=eigen(g), axis1, axis2;
                double sd;
                prev.noise.transverse(piece,end->age,mean,sd);
                cg::orthonormalComplement(prev.direction,axis1,axis2);
                UniformSampler gen(end->transverseSeed);
                const double r=sd*std::sqrt(-2*std::log((double(gen.nextI())+1)/4294967296.0));
                const double angle=2*cg::kPi*(double(gen.nextI())/4294967296.0);
                const Vector gradient=mean+(axial-mean.dot(prev.direction))*prev.direction
                    +r*(std::cos(angle)*axis1+std::sin(angle)*axis2);
                observedDerivative=gradient.dot(direction);
            } else {
                // The phase retains its unrounded reflection at grazing hits.
                observedValue=hit->value;
                observedDerivative=retainedDirection ? hit->derivative : hit->gradient.dot(direction);
                inwardBirth=observedDerivative<0;
            }
            // Renewal+ retains these two scalar observations. Transverse
            // gradient components at the next hit come from the new flight.
            const double m=_sdfGrid->evaluate(vec(flight.origin),nullptr);
            GpisSdfCubicRay birthGeometry(*_sdfGrid,flight.origin,flight.direction);
            Cubic birthMean;
            const double width=birthGeometry.polynomial(0,flight.noise.piece(0).end,birthMean);
            flight.condition(observedValue-m,observedDerivative-birthMean[1]/width);
        }
        // A turn before the old direction's first probe retains its endpoint
        // observations, and schedules the first sign test on the new flight.
        if (inwardBirth || (!hit && (!end || end->flight.entryProbe))) {
            const double nearDistance=std::max(0.0,double(ray.nearT())*speed);
            // Reserve the same entry-jitter dimension for both birth rules.
            const double offset=double(_entryStepSize)*sampler.next1D();
            flight.entryProbe=flight.unconditioned ? nearDistance : nearDistance+offset;
            // An inward reflection starts at an observed zero. For U=0,
            // the next marching point is the first distinct occupancy probe.
            if (inwardBirth && *flight.entryProbe==0) flight.entryProbe=_entryStepSize;
        }
    }
    const double limit=age+double(ray.farT())*speed;
    GpisSdfCubicRay geometry(*_sdfGrid,flight.origin,flight.direction);
    bool positiveOnLeft=true;
    double bound=0, boundEnd=0;
    if (_prune && _threeSigmaPruning) {
        // This scene-wide cutoff can discard tail crossings, including on
        // conditioned flights. It applies only to the first macrocell test.
        bound=3*(_twoLevel ? _twoLevel->parameters.maxSigma :
                 _varying ? _varying->parameters.maxSigma : _sigma);
        boundEnd=limit;
    }
    NoisePiece piece{0,0,0};
    Cubic noise{};
    double cellBound=0;
    double begin=flight.entryProbe.value_or(age);
    bool checkEntry=flight.entryProbe && begin<limit;
    if (checkEntry) {
        flight.entryProbe.reset();
        if (flight.unconditioned) {
            const auto first=flight.noise.piece(begin);
            const double u=(begin-first.start)/first.width();
            const double field=_sdfGrid->evaluate(vec(flight.origin+begin*flight.direction),nullptr)
                +CubicDowncrossing::value(flight.segment(first),u);
            flight.seekUpcrossing=field<0;
            checkEntry=false;
        } else if (_prune) {
            // Only a certified bound can decide the occupancy probe's sign.
            const double boundAtEntry=flight.noise.absoluteBound()
                +std::abs(flight.valueCorrection)+std::abs(flight.slopeCorrection);
            const double skipped=geometry.skipPositive(begin,limit,boundAtEntry);
            if (skipped>begin) { begin=skipped; checkEntry=false; }
        }
    }
    // Positive-only pruning cannot certify an unbroken negative interval.
    const bool prunePositive=_prune && !flight.seekUpcrossing;
    while (begin<limit) {
        if (prunePositive && !checkEntry) {
            if (begin>=boundEnd) {
                bound=flight.noise.boundUntil(begin,limit,boundEnd)
                    +std::abs(flight.valueCorrection)+std::abs(flight.slopeCorrection);
            }
            const double skipped=geometry.skipPositive(begin,boundEnd,bound);
            if (skipped>begin) { begin=skipped; positiveOnLeft=true; continue; }
        }
        if (begin>=piece.end) {
            piece=flight.noise.piece(begin);
            noise=flight.segment(piece);
            cellBound=std::max({std::abs(noise[0]),std::abs(noise[0]+noise[1]/3),
                std::abs(noise[0]+(2*noise[1]+noise[2])/3),std::abs(CubicDowncrossing::value(noise,1))});
            cellBound+=16*std::numeric_limits<double>::epsilon()*
                (std::abs(noise[0])+std::abs(noise[1])+std::abs(noise[2])+std::abs(noise[3]));
        }
        const double cellEnd=std::min(limit,piece.end);
        if (prunePositive && !checkEntry) {
            const double skipped=geometry.skipPositive(begin,cellEnd,cellBound);
            if (skipped>begin) { begin=skipped; positiveOnLeft=true; continue; }
        }
        Cubic c;
        const double next=geometry.polynomial(begin,cellEnd,c);
        if (!(next>begin)) throw std::runtime_error("lattice_cubic_gpis: voxel traversal stalled");
        CubicDowncrossing::addRestricted(c,noise,(begin-piece.start)/piece.width(),(next-begin)/piece.width());
        if (begin==0 && hit) {
            c[0]=0;
            positiveOnLeft=false;
        }
        const bool negativeEntry=checkEntry && c[0]<0;
        if (checkEntry && !negativeEntry && _prune && _threeSigmaPruning) {
            const double skipped=geometry.skipPositive(begin,limit,bound);
            if (skipped>begin) { begin=skipped; checkEntry=false; continue; }
        }
        checkEntry=false;
        double u=0;
        if (flight.seekUpcrossing) {
            for (double &coefficient : c) coefficient=-coefficient;
        }
        if (negativeEntry || CubicDowncrossing::first(c,positiveOnLeft,u)) {
            const double t=begin+u*(next-begin);
            float hitT=float((t-age)/speed);
            if (hitT>=ray.farT()) hitT=std::nextafter(ray.farT(),0.f);
            if (!sample) return hitT;
            const Vector position=flight.origin+t*flight.direction;
            sample->p=Vec3f(vec(position));
            const double orientedSlope=CubicDowncrossing::slope(c,u)/(next-begin);
            const double axial=flight.seekUpcrossing ? -orientedSlope : orientedSlope;
            if (flight.unconditioned && axial>0) {
                if (state->distanceDiagnostic)
                    state->distanceDiagnostic->failure="outward_crossing_rejected";
                // The shadow query already returned the hit. No scattering event
                // is created, so sampleDistance ends the main path at this root.
                return hitT;
            }
            const bool backface=negativeEntry && axial>0;
            if (backface && hasDiffuseMicroMaterial()) return hitT;
            Vector mean=eigen(meanGradient(vec(position)));
            double sd;
            flight.noise.transverse(piece,t,mean,sd);
            // bounded applies its front-face check immediately to an observed
            // deterministic normal. A nondegenerate specular event defers the
            // normal to the NEE phase, whose material can reflect either side.
            if (backface && sd==0) return hitT;
            auto event=std::make_shared<FpGradientEventContext>();
            event->owner=this;
            event->position=position;
            // Protect the specular Jacobian at flat crossings, preserving
            // orientation for negative probes with an outward gradient.
            const double minSpeed=std::numeric_limits<double>::epsilon()*_sigma/flight.noise.kernelSpacing();
            event->derivative=backface ? std::max(axial,minSpeed) : std::min(axial,-minSpeed);
            event->gradient.emplace(flight.direction,event->derivative,mean,sd);
            sample->eventContext=event;
            sample->phase=event->gradient->isDelta() ? _deltaPhase.get() : _phaseFunction.get();
            if (hasDiffuseMicroMaterial()) sample->aniso=vec(event->gradient->sample(rng));
            state->pathContext.reset();
            state->advance();
            return hitT;
        }
        begin=next;
    }
    if (state) state->pathContext=std::make_shared<Endpoint<Noise>>(this,std::move(flight),ray,limit,
                                                             sampler.uniformGenerator().nextI());
    return std::nullopt;
#else
    throw std::runtime_error("lattice_cubic_gpis requires OpenVDB support");
#endif
}

bool LatticeCubicGpisMedium::sampleDistance(PathSampleGenerator &sampler,const Ray &ray,
                                            MediumState &state,MediumSample &sample) const {
    if (state.bounce>_maxBounce) return false;
    sample=MediumSample{};
    sample.t=ray.farT(); sample.p=ray.hitpoint(); sample.phase=_phaseFunction.get();
    sample.pdf=1; sample.weight=sample.continuedWeight=Vec3f(1.0f);
    sample.emission=Vec3f(0.f);
    sample.sparseConv1DSamplingScheme=SparseConv1DSamplingScheme::UNI;
    sample.rayInfo=state.info;
    const auto hit=trace(sampler,ray,&state,&sample);
    sample.exited=!hit;
    sample.t=hit.value_or(ray.farT());
    sample.continuedT=sample.t;
    return sample.exited || bool(sample.eventContext);
}

std::optional<float> LatticeCubicGpisMedium::firstHitDistance(
        PathSampleGenerator &sampler,const Ray &ray,MediumState *state) const {
    return trace(sampler,ray,state,nullptr);
}

Vec3f LatticeCubicGpisMedium::transmittance(PathSampleGenerator &sampler,const Ray &ray,
                                            bool,bool,MediumState *state) const {
    return Vec3f(firstHitDistance(sampler,ray,state) ? 0.f : 1.f);
}

float LatticeCubicGpisMedium::pdf(PathSampleGenerator &,const Ray &,bool,bool) const {
    throw std::runtime_error("lattice_cubic_gpis supports unidirectional path tracing");
}
}
