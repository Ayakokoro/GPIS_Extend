#include "math/LatticeGaussianNoise.hpp"
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace Tungsten;

namespace {
void require(bool ok, const char *message) {
    if (!ok) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}
double eval(const std::array<double,4> &p, double u) {
    return p[0]+u*(p[1]+u*(p[2]+u*p[3]));
}
double derivative(const std::array<double,4> &p, double u) {
    return p[1]+u*(2*p[2]+3*u*p[3]);
}

void testSharedKnotReuse() {
    const LatticeGaussianKernel kernel(.7f,.3f,.5f,2.5f);
    constexpr unsigned seed=1234;
    LatticeGaussianRay ray(kernel,seed);
    LatticeGaussianRay knotRay(kernel,seed);
    auto checkAgainstColdSegments = [&](const LatticeGaussianRay &reference) {
        for (int cell : {0,1,2,73,74,1,std::numeric_limits<int>::max(),
                         std::numeric_limits<int>::min(),-2,-1}) {
            auto cold=reference;
            const auto segment=cold.segment(cell);
            require(ray.segment(cell)==segment,
                    "segment coefficients depend on prior knot access");
            const auto knot=knotRay.knot(cell);
            require(knot[0]==segment[0] && knot[1]==segment[1] &&
                    knotRay.segment(cell)==segment,
                    "direct knot query must agree with cold and cached segments");
        }
    };
    checkAgainstColdSegments(LatticeGaussianRay(kernel,seed));
    for (bool withSlope : {false,true}) {
        // The last query for cell -1 cached knot zero. Conditioning changes
        // that knot immediately before the next query for cell zero.
        const float value=withSlope ? -.4f : .2f;
        LatticeGaussianRay reference(kernel,seed);
        reference.condition(value,.8f,withSlope);
        ray.condition(value,.8f,withSlope);
        knotRay.condition(value,.8f,withSlope);
        checkAgainstColdSegments(reference);
    }
}

void testTransverseConditioning() {
    // With this coarse stencil, the three nonzero weights at a knot are
    // proportional to (e^-1, 1, e^-1). Its covariance differs from the SE limit.
    const LatticeGaussianKernel kernel(1.f,1.f,1.f,2.f);
    const auto birth=kernel.conditionedTransverse(0,0);
    require(birth.meanScale==1 && birth.stddev==0,"full gradient reproduced at birth");
    const double v=std::exp(-1.), norm=1+2*v*v;
    const double gain=2*v/norm;
    const double marginal=kernel.transverseStddev(0);
    const auto next=kernel.conditionedTransverse(1,0);
    require(std::abs(next.meanScale-gain)<1e-7 &&
            std::abs(next.stddev-marginal*std::sqrt(1-gain*gain))<1e-7,
            "full gradient regression must use the finite knot covariance");
    const double u=1e-10;
    const auto near=kernel.conditionedTransverse(0,u);
    // The derivative row has weights proportional to (-2e^-1, 0, 2e^-1)
    // and is orthogonal to the birth value. Even here the variance is positive.
    require(std::abs(near.stddev/u-marginal*std::sqrt(8*v*v/norm))<1e-6,
            "short-flight conditional transverse spread lost to cancellation");
    const auto far=kernel.conditionedTransverse(1000000,.37);
    require(far.meanScale==0 && far.stddev==kernel.transverseStddev(.37),
            "finite support restores the marginal at the current knot phase");
}
}

int main() {
    testSharedKnotReuse();
    testTransverseConditioning();
    constexpr float ell = 0.2f, sigma = 0.02f;
    const LatticeGaussianKernel kernel(sigma,ell);
    require(std::abs(kernel.originStddev()/sigma-1)<1e-4, "SE variance at birth");
    for (unsigned seed=0; seed<64; ++seed) {
        LatticeGaussianRay ray(kernel,seed);
        const auto near = ray.segment(0), far = ray.segment(1000000);
        require(near==ray.segment(0) && far==ray.segment(1000000), "out-of-order seeds");
        ray.condition(-0.05f,-0.13f);
        require(far==ray.segment(1000000), "conditioning must have compact support");
        const auto birth = ray.segment(0);
        require(std::abs(birth[0]+0.05f)<2e-7f &&
                std::abs(birth[1]/kernel.spacing+0.13f)<2e-6f, "birth constraints");
        for (int cell=-16; cell<20; ++cell) {
            const auto a = ray.segment(cell), b = ray.segment(cell+1);
            require(std::abs(eval(a,1)-b[0])<1e-14 &&
                    std::abs(derivative(a,1)-b[1])<1e-14, "continuity of value and derivative");
            for (int k=0; k<=16; ++k)
                require(std::abs(eval(a,k/16.f))<=ray.absoluteBound(), "ray amplitude bound");
        }
    }
    require(std::abs(kernel.transverseStddev(0)-kernel.originStddev()/ell)<1e-7,
            "transverse variance at a scalar-conditioned birth");
    for (double u : {0.,0.25,0.5,0.75,1.})
        require(std::abs(kernel.transverseStddev(u)/(sigma/ell)-1)<0.005,
                "transverse variance across one lattice interval");

    // Estimated correlations and conditioned variance against the actual SE
    // target detect both weight normalization and coefficient reuse mistakes.
    double xx=0, yy=0, xy=0, conditionedSecond=0;
    constexpr int samples=65536;
    for (int seed=0; seed<samples; ++seed) {
        LatticeGaussianRay ray(kernel,seed);
        const double x = ray.segment(0)[0]/sigma;
        const double y = eval(ray.segment(2),0.5f)/sigma; // t = 1.25 ell
        xx+=x*x; yy+=y*y; xy+=x*y;
        ray.condition(0,0,true);
        const double z = eval(ray.segment(2),0.5f)/sigma;
        conditionedSecond+=z*z;
    }
    const double rho=std::exp(-0.5*1.25*1.25);
    require(std::abs(xx/samples-1)<0.025 && std::abs(yy/samples-1)<0.025 &&
            std::abs(xy/samples-rho)<0.015, "sampled SE moments");
    require(std::abs(conditionedSecond/samples-(1-rho*rho*(1+1.25*1.25)))<0.014,
            "sampled conditioned SE variance");
    std::cout << "lattice Gaussian noise checks passed\n";
}
