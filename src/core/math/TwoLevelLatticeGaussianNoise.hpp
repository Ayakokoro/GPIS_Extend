#pragma once

#include "NonstationaryLatticeGaussianNoise.hpp"

namespace Tungsten {

// Independent, unit-variance octave fields. Lengths here are world lengths;
// sparse convolution chooses its octave in units of the stationary base ell.
class TwoLevelLatticeKernel {
    friend class TwoLevelLatticeRay;
    double _baseEll, _maxEll, _maxSigma, _maxSigmaSlope, _maxEllSlope;
    LatticeGaussianKernel _shape;

    struct Coordinate { int cell; double u; };
    Coordinate coordinate(double t,double ell) const {
        const double h=ell*_shape.spacing;
        const double index=std::floor(t/h);
        if (!(index>std::numeric_limits<int>::min()+1. && index<std::numeric_limits<int>::max()-1.))
            throw std::runtime_error("two-level lattice ray exceeds its cell index range");
        int i=int(index);
        if (i*h>t) --i;
        else if ((i+1.)*h<=t) ++i;
        return {i,std::clamp((t-i*h)/h,0.,1.)};
    }

    double controlBound(double valueBound,double slopeBound) const {
        // Both endpoint jets are in the same octave pair. A piece is at
        // most h = spacingFraction*lowEll long. The derivative of the two
        // normalized weights has L1 norm <= sqrt(8)/((Base-1)*lowEll).
        constexpr double root2=1.4142135623730951;
        const double slopeTimesEll=root2*_maxSigmaSlope*_maxEll*valueBound
            +_maxSigma*(2*root2/(Base-1)*_maxEllSlope*valueBound+root2*slopeBound);
        return (1.+2e-6)*(root2*_maxSigma*valueBound+_shape.spacing*slopeTimesEll/3.);
    }
public:
    static constexpr double Base=2.5;
    static constexpr float DefaultRadiusInEll=2.1213203435596426f; // three convolution standard deviations
    struct Blend {
        int level;
        double lowEll,highEll,lowWeight,highWeight,lowDerivative,highDerivative;
        bool atBoundary;
    };
    const double spacing;

    TwoLevelLatticeKernel(double baseEll,double minEll,double maxEll,double maxSigma,
                          double maxSigmaSlope,double maxEllSlope,double spacingInEll=.5,
                          float radiusInEll=DefaultRadiusInEll)
    : _baseEll(baseEll),_maxEll(maxEll),_maxSigma(maxSigma),_maxSigmaSlope(maxSigmaSlope),
      _maxEllSlope(maxEllSlope),_shape(1.f,1.f,float(spacingInEll),radiusInEll),
      spacing(minEll/Base*_shape.spacing) {
        if (!(baseEll>0 && minEll>0 && minEll<=maxEll && maxSigma>0 &&
              maxSigmaSlope>=0 && maxEllSlope>=0 && std::isfinite(baseEll) &&
              std::isfinite(maxEll) && std::isfinite(maxSigma) && std::isfinite(maxSigmaSlope) &&
              std::isfinite(maxEllSlope)))
            throw std::invalid_argument("invalid two-level lattice covariance bounds");
    }

    double spacingForLevel(int level) const { return _shape.spacing*(_baseEll*std::pow(Base,level)); }

    // At an octave boundary choose the derivative on the requested side of ell.
    Blend blend(double ell,int side=0) const {
        const double level=std::log(ell/_baseEll)/std::log(Base);
        if (!(std::isfinite(level) && std::abs(level)<std::numeric_limits<int>::max()-2.))
            throw std::runtime_error("invalid two-level lattice scale");
        const double closest=std::round(level);
        const double tolerance=16*std::numeric_limits<double>::epsilon()*std::max(1.,std::abs(level));
        const bool atBoundary=std::abs(level-closest)<=tolerance;
        const int low=atBoundary ? int(closest)-(side<0) : int(std::floor(level));
        const double lo=_baseEll*std::pow(Base,low),hi=_baseEll*std::pow(Base,low+1);
        const double f=std::clamp((ell-lo)/(hi-lo),0.,1.),g=1.-f;
        const double norm=std::sqrt(g*g+f*f),scale=1./((hi-lo)*norm*norm*norm);
        return {low,lo,hi,g/norm,f/norm,-f*scale,g*scale,atBoundary};
    }

    // Coefficient rows for the mixed field and its ray derivative, before the
    // final Hermite reconstruction across a piece with varying parameters.
    template<class Callback>
    void forEachWeight(double t,const LatticeNoiseParameters &p,int side,Callback &&visit) const {
        const auto pair=blend(p.ell,(p.ellSlope>0 ? side : p.ellSlope<0 ? -side : 0));
        auto add=[&](int level,double ell,double w,double dw) {
            const auto c=coordinate(t,ell);
            _shape.forEachWeight(c.cell,c.u,[&](std::int64_t i,double b,double d) {
                visit(level,i,p.sigma*w*b,(p.sigmaSlope*w+p.sigma*p.ellSlope*dw)*b+p.sigma*w*d/ell);
            });
        };
        add(pair.level,pair.lowEll,pair.lowWeight,pair.lowDerivative);
        add(pair.level+1,pair.highEll,pair.highWeight,pair.highDerivative);
    }
};

class TwoLevelLatticeRay {
    const TwoLevelLatticeKernel &_kernel;
    std::uint32_t _seed;
    struct Layer {
        int level;
        LatticeGaussianRay ray;
        int cell=std::numeric_limits<int>::min();
        std::array<double,4> cubic{};
    };
    mutable std::vector<Layer> _layers;
    double _bound;

    Layer &layer(int level) const {
        for (auto &l : _layers)
            if (l.level==level) return l;
        _layers.push_back({level,LatticeGaussianRay(_kernel._shape,_seed+std::uint32_t(level))});
        return _layers.back();
    }
    struct Jet { double value,slope; };
    Jet evaluateLayer(int level,double ell,double t) const {
        const auto c=_kernel.coordinate(t,ell);
        Layer &l=layer(level);
        const double spacing=ell*_kernel._shape.spacing;
        if (l.cell!=c.cell) {
            // A boundary probe needs only this knot. Build the rest of the cell
            // if a later probe actually falls in its interior.
            if (c.u==0.) {
                const auto knot=l.ray.knot(c.cell);
                return {knot[0],knot[1]/spacing};
            }
            l.cubic=l.ray.segment(c.cell);
            l.cell=c.cell;
        }
        const auto &v=l.cubic;
        return {((v[3]*c.u+v[2])*c.u+v[1])*c.u+v[0],
                ((3*v[3]*c.u+2*v[2])*c.u+v[1])/spacing};
    }
public:
    struct EndpointJet {
        double value,slope;
        bool atOctaveCrease;
    };
    EndpointJet endpoint(double t,const LatticeNoiseParameters &p,int side) const {
        const auto b=_kernel.blend(p.ell,(p.ellSlope>0 ? side : p.ellSlope<0 ? -side : 0));
        const auto lo=evaluateLayer(b.level,b.lowEll,t),hi=evaluateLayer(b.level+1,b.highEll,t);
        const double unit=b.lowWeight*lo.value+b.highWeight*hi.value;
        const double width=b.lowDerivative*lo.value+b.highDerivative*hi.value;
        return {p.sigma*unit,p.sigmaSlope*unit+p.sigma*(p.ellSlope*width+
                b.lowWeight*lo.slope+b.highWeight*hi.slope),b.atBoundary && p.ellSlope!=0.};
    }
    TwoLevelLatticeRay(const TwoLevelLatticeKernel &kernel,std::uint32_t seed)
    : _kernel(kernel),_seed(seed) {
        LatticeGaussianRay prior(kernel._shape,seed);
        _bound=_kernel.controlBound(prior.absoluteBound(),prior.absoluteSlopeBound());
    }
    double absoluteBound() const { return _bound; }
    double value(double t,const LatticeNoiseParameters &p) const { return endpoint(t,p,1).value; }
    static std::array<double,4> hermite(double width,const EndpointJet &a,const EndpointJet &b) {
        const double da=a.slope*width,db=b.slope*width,delta=b.value-a.value;
        return {a.value,da,3*delta-2*da-db,da+db-2*delta};
    }
    std::array<double,4> segment(double lo,double hi,const LatticeNoiseParameters &left,
                                 const LatticeNoiseParameters &right) const {
        const auto a=endpoint(lo,left,1),b=endpoint(hi,right,-1);
        return hermite(hi-lo,a,b);
    }

    void transverse(double t,const LatticeNoiseParameters &p,double &responseSigma,
                    double &responseEll,double &sd) const {
        const auto b=_kernel.blend(p.ell,(p.ellSlope>0)-(p.ellSlope<0));
        const double lo=evaluateLayer(b.level,b.lowEll,t).value;
        const double hi=evaluateLayer(b.level+1,b.highEll,t).value;
        responseSigma=b.lowWeight*lo+b.highWeight*hi;
        responseEll=p.sigma*(b.lowDerivative*lo+b.highDerivative*hi);
        const auto &shape=_kernel._shape;
        const double lowSD=b.lowWeight*shape.transverseStddev(_kernel.coordinate(t,b.lowEll).u)/b.lowEll;
        const double highSD=b.highWeight*shape.transverseStddev(_kernel.coordinate(t,b.highEll).u)/b.highEll;
        sd=p.sigma*std::hypot(lowSD,highSD);
    }

    // Project the two independent level processes onto observations of their
    // sum. The stationary level's birth value and derivative are independent,
    // so the four-variable projection can be passed back to its exact spline
    // conditioner without constraining each octave to the total observation.
    void condition(const LatticeNoiseParameters &p,double value,std::optional<double> slope) {
        _layers.clear();
        const auto blend=_kernel.blend(p.ell,(p.ellSlope>0)-(p.ellSlope<0));
        const std::array<double,2> ell{blend.lowEll,blend.highEll};
        const std::array<double,2> w{blend.lowWeight,blend.highWeight};
        const std::array<double,2> dw{blend.lowDerivative,blend.highDerivative};
        const auto &shape=_kernel._shape;
        const double v=double(shape.originStddev())*shape.originStddev();
        double vv=0,cross=0,oldValue=0,oldSlope=0;
        std::array<double,2> a,d,vd;
        std::array<Jet,2> old;
        for (int i=0;i<2;++i) {
            old[i]=evaluateLayer(blend.level+i,ell[i],0.);
            a[i]=p.sigma*w[i];
            d[i]=p.sigmaSlope*w[i]+p.sigma*p.ellSlope*dw[i];
            vd[i]=shape.originSlopeVariance()/(ell[i]*ell[i]);
            vv+=a[i]*a[i]*v;
            cross+=a[i]*d[i]*v;
            oldValue+=a[i]*old[i].value;
            oldSlope+=d[i]*old[i].value+a[i]*old[i].slope;
        }
        const double valueGain=(value-oldValue)/vv,projection=cross/vv;
        double ss=0;
        for (int i=0;i<2;++i) {
            d[i]-=projection*a[i];
            ss+=d[i]*d[i]*v+a[i]*a[i]*vd[i];
        }
        const double slopeGain=slope ? (*slope-oldSlope-projection*(value-oldValue))/ss : 0.;
        double valueBound=0,slopeBound=0;
        for (int i=0;i<2;++i) {
            auto &l=layer(blend.level+i);
            const double targetValue=old[i].value+v*(a[i]*valueGain+d[i]*slopeGain);
            const double targetSlope=old[i].slope+vd[i]*a[i]*slopeGain;
            l.ray.condition(float(targetValue),float(ell[i]*targetSlope),slope.has_value());
            l.cell=std::numeric_limits<int>::min();
            valueBound=std::max(valueBound,double(l.ray.absoluteBound()));
            slopeBound=std::max(slopeBound,double(l.ray.absoluteSlopeBound()));
        }
        _bound=_kernel.controlBound(valueBound,slopeBound);
    }
};

}
