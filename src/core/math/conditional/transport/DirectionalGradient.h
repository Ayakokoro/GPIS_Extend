#pragma once
#include "math/conditional/mathutility/SmallGaussian.h"

namespace Tungsten::ConditionalGpis {
// r^3 N(r; mean, stddev), r>0. The extra radial power includes both the
// spherical Jacobian and crossing flux; r^2 would give the wrong shadow state.
inline double sampleRadialCubic(double mean,double stddev,Random &rng)
{
    if(!(stddev>0) || !std::isfinite(mean) || !std::isfinite(stddev))
        throw NumericError(NumericStatus::InvalidInput,"invalid directional gradient distribution");
    const double alpha=mean/stddev;
    for(int i=0;i<1000000;++i) {
        double x,logAcceptance;
        if(alpha<=-1) {
            x=-(std::log(rng.openUniform01())+std::log(rng.openUniform01())+
                std::log(rng.openUniform01())+std::log(rng.openUniform01()))/(-alpha);
            logAcceptance=-0.5*x*x;
        } else if(alpha<=0) {
            x=std::sqrt(-2*(std::log(rng.openUniform01())+std::log(rng.openUniform01())));
            logAcceptance=alpha*x;
        } else {
            const double mode=0.5*alpha+0.5*std::hypot(alpha,std::sqrt(12.0));
            const double z=rng.standardNormal();
            x=mode+z;
            if(!(x>0)) continue;
            const double y=z/mode;
            // Gaussian tangent envelope, with cancellation removed analytically.
            logAcceptance=3*(std::log1p(y)-y);
        }
        if(std::log(rng.openUniform01())<=logAcceptance) return stddev*x;
    }
    throw NumericError(NumericStatus::SamplingNotConverged,"directional gradient rejection did not converge");
}

inline Vector3 sampleDirectionalGradient(const Gaussian<3> &g,const Vector3 &normal,Random &rng)
{
    const Vector3 h=normalizedOrThrow(normal);
    Eigen::LLT<Matrix3> chol(g.covariance);
    if(chol.info()!=Eigen::Success)
        throw NumericError(NumericStatus::InvalidCovariance,"directional gradient needs positive covariance");
    const Vector3 u=chol.matrixL().solve(g.mean),v=chol.matrixL().solve(h);
    const double a=v.squaredNorm(),mean=u.dot(v)/a,sd=1/std::sqrt(a);
    return sampleRadialCubic(mean,sd,rng)*h;
}
}
