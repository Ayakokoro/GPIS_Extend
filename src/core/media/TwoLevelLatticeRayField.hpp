#pragma once

#include "LatticeNonstationaryParameters.hpp"
#include "math/TwoLevelLatticeGaussianNoise.hpp"

namespace Tungsten {

struct LatticeNoisePiece {
    int cell;
    double start, end;
    double width() const { return end-start; }

    static LatticeNoisePiece containing(double t,double spacing) {
        const double index=std::floor(t/spacing);
        if (!(index>=0 && index<std::numeric_limits<int>::max()-1))
            throw std::runtime_error("lattice_cubic_gpis: ray exceeds the lattice index range");
        int i=int(index);
        // Division may round a point less than a knot onto its integer index.
        if (i*spacing>t) --i;
        else if ((i+1)*spacing<=t) ++i;
        return {i,i*spacing,(i+1.)*spacing};
    }
};

struct TwoLevelLatticeModel {
    LatticeNonstationaryParameters parameters;
    TwoLevelLatticeKernel kernel;
    std::vector<LatticeNonstationaryParameters::WidthPlane> levelPlanes;

    TwoLevelLatticeModel(const ProceduralNonstationaryCovariance &cov,double spacing,
                         float radiusInEll=TwoLevelLatticeKernel::DefaultRadiusInEll)
    : parameters(cov),
      kernel(parameters.baseEll(),parameters.minEll,parameters.maxEll,parameters.maxSigma,
             parameters.maxSigmaGradient,parameters.maxEllGradient,spacing,radiusInEll),
      levelPlanes(parameters.levelPlanes(TwoLevelLatticeKernel::Base)) {}
};

class TwoLevelLatticeRayField {
    using V=Eigen::Vector3d;
    const TwoLevelLatticeModel &_model;
    TwoLevelLatticeRay _ray;
    V _origin, _direction;
    struct Cut { double distance, ell; };
    std::vector<Cut> _cuts;
    mutable std::optional<double> _cachedEnd;
    mutable TwoLevelLatticeRay::EndpointJet _endJet{};

    bool snapWidthAtCut(double t,LatticeNoiseParameters &p) const {
        bool atCut=false;
        for (const auto &cut : _cuts)
            if (cut.distance==t) {
                atCut=true;
                if (cut.ell>0) {
                    p.ell=cut.ell;
                    break;
                }
            }
        return atCut;
    }
public:
    LatticeNoiseParameters at(double t,int side) const {
        auto p=_model.parameters.at(_origin,_direction,t,side);
        snapWidthAtCut(t,p);
        return p;
    }

    TwoLevelLatticeRayField(const TwoLevelLatticeModel &model,const V &origin,
                            const V &direction,std::uint32_t seed)
    : _model(model), _ray(model.kernel,seed), _origin(origin), _direction(direction) {
        for (double t : model.parameters.cuts(origin,direction))
            if (std::isfinite(t)) _cuts.push_back({t,0});
        for (const auto &plane : model.levelPlanes) {
            const double dir=direction[plane.axis];
            if (dir==0.) continue;
            const double t=(plane.coordinate-origin[plane.axis])/dir;
            if (t>=0 && std::isfinite(t)) _cuts.push_back({t,plane.ell});
        }
        std::sort(_cuts.begin(),_cuts.end(),[](const Cut &a,const Cut &b) {
            return a.distance<b.distance;
        });
    }
    double kernelSpacing() const { return _model.kernel.spacing; }

    LatticeNoisePiece piece(double t) const {
        const auto p=at(t,1);
        const auto blend=_model.kernel.blend(p.ell,(p.ellSlope>0)-(p.ellSlope<0));
        auto interval=LatticeNoisePiece::containing(t,_model.kernel.spacingForLevel(blend.level));
        const auto coarse=LatticeNoisePiece::containing(t,_model.kernel.spacingForLevel(blend.level+1));
        interval.start=std::max(interval.start,coarse.start);
        interval.end=std::min(interval.end,coarse.end);
        for (const auto &cut : _cuts) {
            if (cut.distance>=interval.end) break;
            if (cut.distance>t) { interval.end=cut.distance; break; }
            interval.start=std::max(interval.start,cut.distance);
        }
        return interval;
    }
    std::array<double,4> segment(const LatticeNoisePiece &p) const {
        const auto left=_cachedEnd==p.start ? _endJet : _ray.endpoint(p.start,at(p.start,1),1);
        auto rightParameters=_model.parameters.at(_origin,_direction,p.end,-1);
        const bool atCut=snapWidthAtCut(p.end,rightParameters);
        _endJet=_ray.endpoint(p.end,rightParameters,-1);
        // A ramp cut or a width rounded onto an octave boundary can have
        // different incoming and outgoing slopes at the same distance.
        if (atCut || _endJet.atOctaveCrease) _cachedEnd.reset();
        else _cachedEnd=p.end;
        return _ray.hermite(p.width(),left,_endJet);
    }
    double absoluteBound() const { return _ray.absoluteBound(); }
    double boundUntil(double,double limit,double &end) const { end=limit; return absoluteBound(); }
    void condition(double value,std::optional<double> derivative) {
        _cachedEnd.reset();
        _ray.condition(at(0,1),value,derivative);
    }
    void transverse(const LatticeNoisePiece &,double t,V &mean,double &sd) const {
        const auto jet=_model.parameters.sample(_origin,_direction,t,1);
        auto p=jet.along(_direction);
        snapWidthAtCut(t,p);
        double dSigma,dEll;
        _ray.transverse(t,p,dSigma,dEll,sd);
        mean+=dSigma*V(jet.sigma.gradient.x(),jet.sigma.gradient.y(),jet.sigma.gradient.z())
              +dEll*V(jet.ell.gradient.x(),jet.ell.gradient.y(),jet.ell.gradient.z());
    }
};
}
