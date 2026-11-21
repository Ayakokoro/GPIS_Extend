#pragma once

#include "LatticeGaussianNoise.hpp"
#include <optional>

namespace Tungsten {

// Physical amplitude and length scale, with their one-sided derivatives per
// unit world distance on this ray.
struct LatticeNoiseParameters {
    double sigma, ell, sigmaSlope, ellSlope;
};

class NonstationaryLatticeGaussianKernel {
    static constexpr double Normalization = 0.8932438417380023; // (2/pi)^(1/4)
    double _taperWidth, _boundPerCoefficient;
public:
    const double spacing, radiusInEll;

    NonstationaryLatticeGaussianKernel(double minEll, double maxEll, double maxSigma,
            double maxSigmaSlope, double maxEllSlope, double spacingInEll=.5, double radius=3.)
        : _taperWidth(std::min(.5,radius*.5)), spacing(spacingInEll*minEll), radiusInEll(radius)
    {
        if (!(minEll>0 && maxEll>=minEll && maxSigma>0 && maxSigmaSlope>=0 && maxEllSlope>=0 &&
              spacing>0 && radius*minEll>spacing && std::isfinite(maxSigma) &&
              std::isfinite(maxSigmaSlope) && std::isfinite(maxEllSlope) &&
              std::isfinite(spacing) && std::isfinite(radius) &&
              radius*maxEll/spacing<std::numeric_limits<int>::max()/2))
            throw std::invalid_argument("invalid nonstationary lattice amplitude, spacing, or support");
        _boundPerCoefficient=boundPerCoefficient(minEll,maxEll,maxSigma,maxSigmaSlope,maxEllSlope);
        if (!std::isfinite(_boundPerCoefficient))
            throw std::invalid_argument("nonfinite nonstationary lattice bound");
    }

    double boundPerCoefficient() const { return _boundPerCoefficient; }

    // A sampled unimodal Gaussian exp(-c*z^2) has sum at most
    // 1 + sqrt(pi/c)*ell/spacing, at every lattice phase. Bound the value and
    // slope independently; Hermite Bezier controls add at most spacing*slope/3.
    double boundPerCoefficient(double minEll, double maxEll, double maxSigma,
                               double maxSigmaSlope, double maxEllSlope) const {
        constexpr double sqrtPi=1.7724538509055161, sqrtTwoPi=2.5066282746310005;
        const double scale=std::sqrt(spacing/minEll);
        const double valueSum=scale+sqrtPi*std::sqrt(maxEll/spacing);
        const double slopeSum=scale+sqrtTwoPi*std::sqrt(maxEll/spacing);
        const double inverseLengthSum=(scale+sqrtTwoPi*std::sqrt(minEll/spacing))/minEll;
        // exp(-z^2/2)*2|z| <= 2/sqrt(e), exp(-z^2/2)*2z^2 <= 4/e.
        // The taper's derivative is zero below radius-taperWidth.
        const double taperStart=radiusInEll-_taperWidth;
        const double taperSlope=(1.5/_taperWidth)*std::exp(-.5*taperStart*taperStart);
        const double maxShapeSlope=1.214+1.472*maxEllSlope+
            taperSlope*(1+radiusInEll*maxEllSlope)+.5*maxEllSlope;
        const double slope=maxSigmaSlope*slopeSum+maxSigma*maxShapeSlope*inverseLengthSum;
        // The endpoints of a nominal cell are rounded separately even at
        // large lattice indices (the renderer supports signed 32-bit cells).
        return (1.+2e-6)*Normalization*(maxSigma*valueSum+spacing*slope/3.);
    }

    // Pass each active coefficient's value weight, directional derivative,
    // partial derivative w.r.t. amplitude, and partial w.r.t. width.
    template<typename Callback>
    void forEachKnot(double t, const LatticeNoiseParameters &p, Callback &&visit) const {
        const double grid=t/spacing;
        if (!(p.ell>0 && p.sigma>0 && std::isfinite(p.ell) && std::isfinite(p.sigma) &&
              std::isfinite(p.sigmaSlope) && std::isfinite(p.ellSlope) &&
              std::abs(grid)<0x1p52))
            throw std::runtime_error("invalid nonstationary lattice knot");
        const auto cell=static_cast<std::int64_t>(std::floor(grid));
        const double fraction=grid-double(cell);
        const double h=spacing/p.ell, amplitude=Normalization*std::sqrt(h);
        const double ratioStep=std::exp(-2*h*h);
        // Walk outwards from the Gaussian's peak. The ratio of successive
        // Gaussian values itself forms a geometric progression on each side.
        for (int side : {-1,1}) {
            const double start=(side<0 ? fraction : 1-fraction)*h;
            const auto first=side<0 ? cell : cell+1;
            double gaussian=0, ratio=0;
            for (int k=0;;++k) {
                const double x=start+k*h;
                if (x>=radiusInEll) break;
                if ((k&63)==0) {
                    gaussian=amplitude*std::exp(-x*x);
                    ratio=std::exp(-(2*x+h)*h);
                }
                const double z=-side*x;
                double window=1, windowSlope=0;
                if (x>radiusInEll-_taperWidth) {
                    const double v=(radiusInEll-x)/_taperWidth;
                    window=v*v*(3-2*v);
                    windowSlope=side*6*v*(1-v)/_taperWidth;
                }
                const double ds=gaussian*window;
                const double dz=gaussian*(windowSlope-2*z*window);
                const double de=(-.5*ds-z*dz)*(p.sigma/p.ell);
                const double slope=p.sigmaSlope*ds+p.ellSlope*de+dz*(p.sigma/p.ell);
                visit(first+std::int64_t(side)*k,p.sigma*ds,slope,ds,de);
                gaussian*=ratio;
                ratio*=ratioStep;
            }
        }
    }
};

class NonstationaryLatticeGaussianRay : private LatticeGaussianSamples {
    const NonstationaryLatticeGaussianKernel &_kernel;
    std::int64_t _first=0;
    std::vector<double> _conditioned;
    double _maxCoefficient=PriorCoefficientBound;
    struct Endpoint { double value=0, slope=0; };
    mutable double _lastT=std::numeric_limits<double>::quiet_NaN();
    mutable LatticeNoiseParameters _lastParameters{};
    mutable Endpoint _lastEndpoint;

    Endpoint evaluate(double t, const LatticeNoiseParameters &p) const {
        if (_lastT==t && p.sigma==_lastParameters.sigma && p.ell==_lastParameters.ell &&
            p.sigmaSlope==_lastParameters.sigmaSlope && p.ellSlope==_lastParameters.ellSlope)
            return _lastEndpoint;
        Endpoint e;
        _kernel.forEachKnot(t,p,[&](std::int64_t j,double b,double d,double,double) {
            const double z=coefficient(j);
            e.value+=b*z;
            e.slope+=d*z;
        });
        _lastT=t; _lastParameters=p; _lastEndpoint=e;
        return e;
    }
public:
    NonstationaryLatticeGaussianRay(const NonstationaryLatticeGaussianKernel &kernel,std::uint32_t seed)
        : LatticeGaussianSamples(seed), _kernel(kernel) {}

    double coefficient(std::int64_t index) const {
        const auto offset=static_cast<std::uint64_t>(index-_first);
        return offset<_conditioned.size() ? _conditioned[size_t(offset)] : prior(index);
    }

    double originStddev(const LatticeNoiseParameters &p) const {
        double variance=0;
        _kernel.forEachKnot(0,p,[&](std::int64_t,double b,double,double,double) { variance+=b*b; });
        return std::sqrt(variance);
    }

    // Orthogonalize the derivative row against the value row. This realizes
    // A^T(AA^T)^-1 without subtracting nearly equal squared norms.
    void condition(const LatticeNoiseParameters &p, double value, std::optional<double> slope) {
        struct Weight { double b=0,d=0; };
        const int radius=int(std::ceil(_kernel.radiusInEll*p.ell/_kernel.spacing));
        _first=-radius;
        const size_t count=2*size_t(radius)+1;
        _conditioned.assign(count,0);
        std::vector<Weight> rows(count);
        double variance=0,cross=0,oldValue=0;
        // The two extreme indices can have zero weight at t=0 but contribute
        // just beyond the birth. Preserve their prior samples as well.
        for (size_t i=0;i<count;++i) _conditioned[i]=prior(_first+std::int64_t(i));
        _kernel.forEachKnot(0,p,[&](std::int64_t j,double b,double d,double,double) {
            const size_t i=size_t(j-_first);
            rows[i]={b,d};
            variance+=b*b;
            cross+=b*d;
            oldValue+=b*_conditioned[i];
        });
        if (!(variance>0)) throw std::runtime_error("zero lattice variance at conditioned birth");
        const double valueGain=(value-oldValue)/variance;
        double slopeGain=0;
        if (slope) {
            const double projection=cross/variance;
            double norm=0,oldOrthogonalSlope=0;
            for (size_t i=0;i<count;++i) {
                rows[i].d-=projection*rows[i].b;
                norm+=rows[i].d*rows[i].d;
                oldOrthogonalSlope+=rows[i].d*_conditioned[i];
            }
            if (!(norm>0)) throw std::runtime_error("singular lattice slope at conditioned birth");
            slopeGain=(*slope-projection*value-oldOrthogonalSlope)/norm;
        }
        _maxCoefficient=PriorCoefficientBound;
        for (size_t i=0;i<count;++i) {
            _conditioned[i]+=valueGain*rows[i].b+slopeGain*rows[i].d;
            _maxCoefficient=std::max(_maxCoefficient,std::abs(_conditioned[i]));
        }
        _lastT=std::numeric_limits<double>::quiet_NaN();
    }

    double absoluteBound() const { return _maxCoefficient*_kernel.boundPerCoefficient(); }
    // The supplied extrema must cover both endpoint jets of every cubic
    // intersecting the span, including the enclosing lattice knots.
    double absoluteBound(double minEll,double maxEll,double maxSigma,
                         double maxSigmaSlope,double maxEllSlope) const {
        return _maxCoefficient*_kernel.boundPerCoefficient(minEll,maxEll,maxSigma,maxSigmaSlope,maxEllSlope);
    }

    std::array<double,4> segment(double lo,double hi,const LatticeNoiseParameters &left,
                                const LatticeNoiseParameters &right) const {
        const auto a=evaluate(lo,left), b=evaluate(hi,right);
        const double width=hi-lo, da=width*a.slope, db=width*b.slope, delta=b.value-a.value;
        return {a.value,da,3*delta-2*da-db,da+db-2*delta};
    }
};

}
