#include "math/NonstationaryLatticeGaussianNoise.hpp"
#include <iostream>
#include <map>
#include <stdexcept>

using namespace Tungsten;
using Kernel = NonstationaryLatticeGaussianKernel;
using Params = LatticeNoiseParameters;
using RayNoise = NonstationaryLatticeGaussianRay;

namespace {
void check(bool ok,const char *message) {
    if (!ok) throw std::runtime_error(message);
}

std::map<std::int64_t,double> weights(const Kernel &kernel,double t,const Params &p) {
    std::map<std::int64_t,double> b;
    kernel.forEachKnot(t,p,[&](std::int64_t j,double value,double,double,double) { b[j]=value; });
    return b;
}

Params parameters(double t) {
    return {.045+.023*t,.077-.031*t,.023,-.031};
}

void checkWeights(const Kernel &kernel) {
    const double t=.083, h=3e-8;
    const auto p=parameters(t);
    auto earlier=weights(kernel,t-h,parameters(t-h)), later=weights(kernel,t+h,parameters(t+h));
    Params wider=p, narrower=p;
    wider.ell+=h; narrower.ell-=h;
    auto up=weights(kernel,t,wider), down=weights(kernel,t,narrower);
    int tapers=0;
    kernel.forEachKnot(t,p,[&](std::int64_t j,double b,double slope,double ds,double de) {
        const double z=(t-j*kernel.spacing)/p.ell;
        check(std::abs(z)<3.,"tap outside support");
        const double v=std::clamp((3-std::abs(z))/.5,0.,1.);
        if (v>0 && v<1) ++tapers;
        const double window=v*v*(3-2*v);
        const double normalization=std::pow(2./3.14159265358979323846,.25);
        const double reference=normalization*std::sqrt(kernel.spacing/p.ell)*std::exp(-z*z)*window;
        check(std::abs(ds-reference)<1e-13 && std::abs(b-p.sigma*reference)<1e-14,
              "normalized variable-width/tapered basis");
        check(std::abs(slope-(later[j]-earlier[j])/(2*h))<4e-9, "total knot directional derivative");
        check(std::abs(de-(up[j]-down[j])/(2*h))<4e-9, "knot width partial derivative");
    });
    check(tapers>=2, "taper test did not exercise both sides");
}

void checkRays(const Kernel &kernel) {
    const auto p=parameters(0);
    double var=0,cov=0,varSlope=0;
    kernel.forEachKnot(0,p,[&](std::int64_t,double b,double d,double,double) {
        var+=b*b; cov+=b*d; varSlope+=d*d;
    });
    check(cov/std::sqrt(var*varSlope)>.015, "test requires correlated birth observations");
    for (unsigned seed=0;seed<18;++seed) {
        RayNoise ray(kernel,seed);
        const double spacing=kernel.spacing;
        const auto polynomial=[&](double t) {
            return ray.segment(t,t+spacing,parameters(t),parameters(t+spacing));
        };
        const auto start=polynomial(0), middle=polynomial(.08), far=polynomial(1);
        check(polynomial(0)==start && polynomial(.08)==middle, "out-of-order deterministic coefficients");
        check(std::abs(ray.originStddev(p)-std::sqrt(var))<1e-15, "local marginal at birth");
        const double value=-.09, slope=.34;
        // Independently predict the corrected field using cross-covariances
        // at a non-birth evaluation. This detects an incorrect diagonal solve.
        auto remoteWeights=weights(kernel,.08,parameters(.08));
        double bv=0,bd=0;
        kernel.forEachKnot(0,p,[&](std::int64_t j,double b,double d,double,double) {
            bv+=remoteWeights[j]*b; bd+=remoteWeights[j]*d;
        });
        const double dv=value-start[0], dd=slope-start[1]/spacing;
        const double det=var*varSlope-cov*cov;
        const double expected=middle[0]+((bv*varSlope-bd*cov)*dv+(bd*var-bv*cov)*dd)/det;
        ray.condition(p,value,slope);
        const auto birth=polynomial(0);
        check(std::abs(birth[0]-value)<1e-12 && std::abs(birth[1]/spacing-slope)<1e-11,
              "conditioned birth value and slope");
        check(std::abs(polynomial(.08)[0]-expected)<1e-12, "conditional remote value");
        check(polynomial(1)==far, "compact birth update independent of distant coefficients");
        for (int i=0;i<55;++i) {
            const double t=i*spacing;
            const auto left=parameters(t), right=parameters(t+spacing);
            const double localBound=ray.absoluteBound(right.ell,left.ell,right.sigma,.023,.031);
            check(localBound<=ray.absoluteBound(), "a restricted bound should tighten the global bound");
            const auto c=polynomial(t);
            const double controls[]={c[0],c[0]+c[1]/3,c[0]+(2*c[1]+c[2])/3,c[0]+c[1]+c[2]+c[3]};
            for (const double v:controls) check(std::abs(v)<=localBound,"local Hermite noise bound");
        }
        // A constant exterior plateau should benefit from a small relative
        // amplitude even if conditioned coefficients have a large magnitude.
        const Params plateau{.002,.032,0,0};
        const double plateauBound=ray.absoluteBound(plateau.ell,plateau.ell,plateau.sigma,0,0);
        check(plateauBound<ray.absoluteBound()*.05, "low-amplitude plateau bound");
        for (int i=0;i<25;++i) {
            const auto c=ray.segment(i*spacing,(i+1)*spacing,plateau,plateau);
            const double controls[]={c[0],c[0]+c[1]/3,c[0]+(2*c[1]+c[2])/3,c[0]+c[1]+c[2]+c[3]};
            for (const double v:controls) check(std::abs(v)<=plateauBound,"constant plateau bound");
        }
        ray.condition(p,value,std::nullopt);
        const auto valueOnly=polynomial(0);
        check(std::abs(valueOnly[0]-value)<1e-12, "value-only update");
        check(std::abs(valueOnly[1]/spacing-(start[1]/spacing+cov/var*dv))<1e-11,
              "value-only update retains correct correlated slope");
    }
}

}

int main() try {
    const Kernel kernel(.032,.1,.08,.04,.05,.5,3);
    checkWeights(kernel);
    checkRays(kernel);
    std::cout<<"nonstationary lattice noise checks passed\n";
    return 0;
} catch (const std::exception &e) {
    std::cerr<<e.what()<<'\n';
    return 1;
}
