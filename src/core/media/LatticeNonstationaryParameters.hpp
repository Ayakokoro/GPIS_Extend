#pragma once

#include "math/GPFunctions.hpp"
#include "math/NonstationaryLatticeGaussianNoise.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Tungsten {

// World-space covariance parameters along a ray. The split planes are
// fixed for the scene; each flight needs at most four split distances.
class LatticeNonstationaryParameters {
    using V = Eigen::Vector3d;

    static Eigen::Vector3d eigen(const Vec3d &v) { return {v.x(),v.y(),v.z()}; }
    static ParameterJet fieldJet(const ProceduralScalar &f, const Vec3d &p) {
        return f.parameterJet(p);
    }
    static ParameterJet fieldJet(const ProceduralVector &f, const Vec3d &p) {
        return f.parameterJet(p)[0];
    }

    template<class Field> struct Parameter {
        const Field *source=nullptr;
        ParameterBounds bounds{};
        double multiplier=1, constant=1, start=0, end=0, logSlope=0;
        double startValue=1, endValue=1;
        int axis=-1;

        void prepare(const Field *field, double scale, const ParameterBounds &b) {
            source=field;
            multiplier=scale;
            bounds={b.minimum*scale,b.maximum*scale,b.gradientNorm*scale};
            if (!(bounds.minimum>0 && bounds.maximum>=bounds.minimum && bounds.gradientNorm>=0) ||
                !std::isfinite(bounds.maximum) || !std::isfinite(bounds.gradientNorm))
                throw std::invalid_argument("lattice_cubic_gpis needs finite positive covariance parameters");
            if (!field || !b.gradientNorm) {
                constant=scale*(field ? fieldJet(*field, Vec3d(0.0)).value : 1.);
                return;
            }
            // The supported varying procedural fields expose their clamp
            // planes via their scene descriptors. Read this once at load time.
            rapidjson::Document doc;
            const auto j=field->toJson(doc.GetAllocator());
            if (!j.HasMember("noise") || !j.HasMember("start") || !j.HasMember("end"))
                throw std::invalid_argument("lattice_cubic_gpis needs covariance fields with known derivative breaks");
            const std::string n=j["noise"].GetString();
            if (n=="left_right") axis=0;
            else if (n=="bottom_top") axis=1;
            else if (n=="front_back") axis=2;
            else throw std::invalid_argument("lattice_cubic_gpis supports constant and single-axis covariance fields");
            start=j["start"].GetDouble();
            end=j["end"].GetDouble();
            Vec3d p(0.);
            p[axis]=start;
            startValue=fieldJet(*field,p).value;
            p[axis]=end;
            endValue=fieldJet(*field,p).value;
            p[axis]=0.5*start+0.5*end;
            const auto interior=fieldJet(*field,p);
            logSlope=interior.gradient[axis]/(1.+interior.value);
        }

        ParameterJet sample(const V &origin, const V &direction, double t, int side) const {
            if (axis<0) return {constant,Vec3d(0.)};
            const V world=origin+t*direction;
            Vec3d p(world.x(),world.y(),world.z());
            // Reconstruct the ramp value on its plane even if the ray multiply
            // rounds to the neighboring side. At a crease use the chosen
            // segment's one-sided derivative; tangent rays use the exterior jet.
            bool onStart=p[axis]==start, onEnd=p[axis]==end;
            if (direction[axis]!=0.) {
                onStart=t==(start-origin[axis])/direction[axis];
                onEnd=t==(end-origin[axis])/direction[axis];
            }
            if (onStart || onEnd) {
                const bool enters=side*direction[axis]*(end-start)>0;
                const bool leaves=side*direction[axis]*(end-start)<0;
                const double value=onStart ? startValue : endValue;
                Vec3d gradient(0.);
                if ((onStart && enters) || (onEnd && leaves))
                    gradient[axis]=multiplier*(value+1.)*logSlope;
                return {multiplier*value,gradient};
            }
            if ((end>start && p[axis]<start) || (start>end && p[axis]>start))
                return {multiplier*startValue,Vec3d(0.)};
            if ((end>start && p[axis]>end) || (start>end && p[axis]<end))
                return {multiplier*endValue,Vec3d(0.)};
            const auto jet=fieldJet(*source,p);
            return {multiplier*jet.value,jet.gradient*multiplier};
        }
    };
    Parameter<ProceduralScalar> _amplitude;
    Parameter<ProceduralVector> _width;

    struct Jet {
        ParameterJet sigma,ell;
        LatticeNoiseParameters along(const V &dir) const {
            return {sigma.value,ell.value,eigen(sigma.gradient).dot(dir),eigen(ell.gradient).dot(dir)};
        }
    };
    Jet sample(const V &origin, const V &dir, double t, int side) const {
        return {_amplitude.sample(origin,dir,t,side),_width.sample(origin,dir,t,side)};
    }

public:
    double minEll, maxEll, maxSigma, maxSigmaGradient, maxEllGradient;
    explicit LatticeNonstationaryParameters(const ProceduralNonstationaryCovariance &covariance) {
        const auto *base=dynamic_cast<const SquaredExponentialCovariance *>(covariance.stationaryCovariance().get());
        if (!base || !base->isIsotropic() || covariance.isNonstationaryAnisotropicKernel())
            throw std::invalid_argument("lattice_cubic_gpis needs a locally isotropic squared_exponential covariance");
        const Vec3d zero(0.);
        const double baseSigma=std::sqrt((*base)(Derivative::None,Derivative::None,zero,zero,zero,zero));
        ParameterBounds a{1,1,0};
        const auto &amp=covariance.amplitudeField();
        if (amp && !amp->parameterBounds(a))
            throw std::invalid_argument("lattice_cubic_gpis var needs a known positive bound and gradient");
        _amplitude.prepare(amp.get(),baseSigma,a);
        VectorParameterBounds scales;
        scales.fill({1,1,0});
        const auto &ls=covariance.lengthScaleField();
        if (ls && !ls->parameterBounds(scales))
            throw std::invalid_argument("lattice_cubic_gpis ls needs a bounded constant or scalar axis ramp");
        // Constant vectors and single-axis ramps are the only supported
        // fields with bounds. The latter produce three identical components.
        for (int k=1;k<3;++k) {
            if (scales[k].minimum!=scales[0].minimum || scales[k].maximum!=scales[0].maximum ||
                scales[k].gradientNorm!=scales[0].gradientNorm)
                throw std::invalid_argument("lattice_cubic_gpis needs equal length scales on all three axes");
        }
        _width.prepare(ls.get(),base->lengthScale(),scales[0]);
        minEll=_width.bounds.minimum;
        maxEll=_width.bounds.maximum;
        maxSigma=_amplitude.bounds.maximum;
        maxSigmaGradient=_amplitude.bounds.gradientNorm;
        maxEllGradient=_width.bounds.gradientNorm;
    }

    LatticeNoiseParameters at(const V &origin, const V &direction, double t, int side) const {
        return sample(origin,direction,t,side).along(direction);
    }

    // On a plateau, the whole Hermite piece (including its endpoint jets)
    // uses constant parameters. A tangent line also has zero axial slope.
    bool constantAlongRay(const V &origin,const V &direction,double t) const {
        const auto varying=[&](const auto &field) {
            if (field.axis<0 || direction[field.axis]==0) return false;
            const double a=(field.start-origin[field.axis])/direction[field.axis];
            const double b=(field.end-origin[field.axis])/direction[field.axis];
            return t>=std::min(a,b) && t<std::max(a,b);
        };
        return !varying(_amplitude) && !varying(_width);
    }

    std::array<double,4> cuts(const V &origin, const V &direction) const {
        std::array<double,4> result;
        result.fill(std::numeric_limits<double>::infinity());
        int count=0;
        auto add=[&](const auto &field) {
            if (field.axis<0 || direction[field.axis]==0.) return;
            for (double plane:{field.start,field.end}) {
                const double t=(plane-origin[field.axis])/direction[field.axis];
                if (t>0. && std::isfinite(t)) result[count++]=t;
            }
        };
        add(_amplitude);
        add(_width);
        std::sort(result.begin(),result.end());
        return result;
    }

    // Local parameter derivatives act on the current flight's coefficients.
    // The transverse residuals are independent of its two birth observations.
    void transverse(const NonstationaryLatticeGaussianKernel &kernel,
                    const NonstationaryLatticeGaussianRay &ray,
                    const V &origin, const V &direction, double t, V &mean, double &sd) const {
        const auto jet = sample(origin,direction,t,1);
        const auto local = jet.along(direction);
        const V amplitudeGradient = eigen(jet.sigma.gradient);
        const V widthGradient = eigen(jet.ell.gradient);
        const bool varying = amplitudeGradient.squaredNorm()+widthGradient.squaredNorm()>0.;
        double responseSigma=0, responseEll=0, variance=0;
        kernel.forEachKnot(t,local,[&](std::int64_t i,double b,double,double dSigma,double dEll) {
            if (varying) {
                const double z = ray.coefficient(i);
                responseSigma += dSigma*z;
                responseEll += dEll*z;
            }
            const double w = b/local.ell;
            variance += w*w;
        });
        mean += responseSigma*amplitudeGradient + responseEll*widthGradient;
        sd = std::sqrt(variance);
    }
};
}
