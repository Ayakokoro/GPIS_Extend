#pragma once
#include "sampling/SampleWarp.hpp"

struct ScatteringMoments {
    int n=0;
    double sum=0,square=0;
    void add(double x) { ++n;sum+=x;square+=x*x; }
    double mean() const { return sum/n; }
    double variance() const { return std::max(0.,(square-sum*sum/n)/(n-1)); }
};

void scatteringChecks() {
    namespace cg=ConditionalGpis;
    using Distribution=NeuralGpis::RenewalGradientDistribution;
    const auto isotropic=cg::CovarianceKernel::fromCorrelationLengths(1,V::Ones());
    const Distribution simple(isotropic,V::UnitZ(),1,-1,V::Zero(),V::Zero(),std::nullopt);
    checkNear(simple.phasePdf(-V::UnitZ()),1/(8*cg::kPi),1e-14,"backscatter plane/reflection Jacobian");
    checkNear(simple.phasePdf(V::UnitZ()),0,0,"forward singular direction has zero density");
    require(!simple.gradientForDirection(V::UnitZ()),"forward direction has no finite collision gradient");
    for (double z:{-0.9,-0.5,0.,0.5,0.9}) {
        const V wo(std::sqrt(1-z*z),0,z);
        // Independent closed form: squared transverse radius is chi-square(2),
        // and wo.z=(r^2-1)/(r^2+1) for v=-1 and wi=+z.
        const double exact=std::exp(-0.5*(1+z)/(1-z))/(2*cg::kPi*(1-z)*(1-z));
        checkNear(simple.phasePdf(wo),exact,1e-13,"analytic outgoing density");
    }
    cg::Matrix3 metric;
    metric<<1.1,0.2,0.1, 0.2,0.8,0.15, 0.1,0.15,1.3;
    const cg::CovarianceKernel anisotropic(0.7,metric);
    const V wi=V(0.3,-0.4,1).normalized();
    const Distribution correlated(anisotropic,wi,0.7,-0.8,V(0.2,0.1,-0.3),V(-0.1,0.2,0.3),V(0.5,-0.2,0.8));
    for (const Distribution *law:{&simple,&correlated}) {
        // Uniform solid-angle quadrature is independent of the phase sampler.
        double integral=0;
        constexpr int nz=1200,nphi=128;
        for (int iz=0;iz<nz;++iz) for (int ip=0;ip<nphi;++ip) {
            const double z=-1+2*(iz+0.5)/nz,phi=2*cg::kPi*(ip+0.5)/nphi;
            const double radius=std::sqrt(1-z*z);
            integral+=law->phasePdf(V(radius*std::cos(phi),radius*std::sin(phi),z));
        }
        integral*=4*cg::kPi/(nz*nphi);
        checkNear(integral,1,2e-4,"direction PDF integrates to one (A/B, anisotropic)");

        cg::Random rng(83742);
        ScatteringMoments phase,light,mis;
        constexpr int count=80000;
        int lowerHemisphere=0;
        const auto contribution=[](const V &out,const V &g) {
            // Exercise the gradient-dependent successor, as real visibility does.
            return (0.2+0.8*std::abs(out.z()))*std::exp(-0.3*g.squaredNorm());
        };
        for (int i=0;i<count;++i) {
            const V g=law->sample(rng),w=law->incomingDirection();
            const V out=cg::reflectTravelDirection(w,g.normalized()).normalized();
            const auto recovered=law->gradientForDirection(out);
            require(bool(recovered) && (*recovered-g).norm()<2e-10*(1+g.norm()),
                "NEE recovers the same full gradient as reflection sampling");
            require(w.dot(g)<0 && out.dot(g)>0,"phase and connection have crossing support");
            lowerHemisphere+=out.z()<0;
            const double p=law->phasePdf(out),q=1/(4*cg::kPi),f=contribution(out,g);
            require(p>0 && std::isfinite(p),"sampled direction has finite positive PDF");
            checkNear(law->phasePdfForGradient(g),p,1e-10*std::max(1.,p),"gradient and double direction PDFs agree");
            phase.add(f);
            double estimate=f*p*p/(p*p+q*q);
            const double z=2*rng.openUniform01()-1,phi=2*cg::kPi*rng.openUniform01();
            const double r=std::sqrt(1-z*z);
            const V connection(r*std::cos(phi),r*std::sin(phi),z);
            const auto cgValue=law->gradientForDirection(connection);
            require(bool(cgValue),"random light direction has finite gradient");
            const double lp=law->phasePdf(connection);
            const double lf=contribution(connection,*cgValue)*lp/q;
            light.add(lf); estimate+=lf*q*q/(lp*lp+q*q); mis.add(estimate);
        }
        const auto agrees=[&](const ScatteringMoments &other) {
            return 6*std::sqrt((phase.variance()+other.variance())/count)+2e-4;
        };
        checkNear(phase.mean(),light.mean(),agrees(light),"phase and NEE agree for gradient-dependent visibility");
        checkNear(phase.mean(),mis.mean(),agrees(mis),"phase and MIS have equal expectation");
        if (law==&simple)
            checkNear(double(lowerHemisphere)/count,1-std::exp(-0.5),0.01,"sampled angular mass matches analytic CDF");
        std::cout<<"neural phase: PDF integral="<<integral<<", phase="<<phase.mean()
            <<", NEE="<<light.mean()<<", MIS="<<mis.mean()<<"\n";
    }
    const Distribution deterministic(isotropic,V::UnitZ(),0,-1,V::Zero(),V::Zero(),V(0.2,0.3,1));
    require(deterministic.isDelta(),"zero-age B retains its deterministic transverse gradient");
    cg::Random rng(13);
    require((deterministic.sample(rng)-V(0.2,0.3,-1)).norm()<1e-14,"degenerate B sample stays exact");
    // Independent SE fixture: rho=1/2 gives transverse mean (1,-1/2)
    // from the previous (2,-1), and covariance (1-rho^2) I = 3/4 I.
    const Distribution decayed(isotropic,V::UnitZ(),std::sqrt(2*std::log(2.)),-1,
        V::Zero(),V::Zero(),V(2,-1,3));
    ScatteringMoments gx,gy;
    for (int i=0;i<50000;++i) {
        const V g=decayed.sample(rng);
        gx.add(g.x());gy.add(g.y());
        checkNear(g.z(),-1,1e-14,"learned inward derivative is retained exactly");
    }
    checkNear(gx.mean(),1,0.025,"B transverse mean retains SE correlation (x)");
    checkNear(gy.mean(),-0.5,0.025,"B transverse mean retains SE correlation (y)");
    checkNear(gx.variance(),0.75,0.03,"B transverse variance matches SE conditional law (x)");
    checkNear(gy.variance(),0.75,0.03,"B transverse variance matches SE conditional law (y)");
    const Distribution narrow(isotropic,V::UnitZ(),1e-10,-1,V::Zero(),V::Zero(),V::UnitZ());
    require(!narrow.isDelta() && std::isfinite(narrow.phasePdf(-V::UnitZ())),"tiny positive B ages remain continuous");
    const Distribution rotatedNarrow(isotropic,wi,1e-10,-1,V::Zero(),V::Zero(),V(0.2,0.3,1));
    for (int i=0;i<20;++i) {
        const V g=rotatedNarrow.sample(rng);
        const double density=rotatedNarrow.phasePdfForGradient(g);
        require(density>0 && std::isfinite(float(density)),"narrow rotated samples retain a representable density");
        checkNear(SampleWarp::powerHeuristic(float(density),1),1,1e-7,"narrow sample retains its MIS contribution");
    }
    checkNear(SampleWarp::powerHeuristic(1e30f,2e30f),0.2,1e-7,"MIS handles large finite densities");
    checkNear(SampleWarp::powerHeuristic(1e-30f,2e-30f),0.2,1e-7,"MIS handles small finite densities");
    std::cout<<"PASS: neural conditional-direction PDF, A/B sampling, NEE/MIS and sharp-density weights\n";
}
