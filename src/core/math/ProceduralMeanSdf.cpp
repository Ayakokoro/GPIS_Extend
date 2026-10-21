#include "GPFunctions.hpp"
#include <stdexcept>

namespace Tungsten {
namespace {
double queryGuard(double v) {
    return 128*std::numeric_limits<double>::epsilon()*(1+std::abs(v));
}
}

void ProceduralMean::prepareSdfQueries()
{
    for(int i=0;i<3;++i) {
        _sdfTranslation[i]=_invConfigTransform(i,3);
        for(int j=0;j<3;++j) _sdfWorldToLocal(i,j)=_invConfigTransform(i,j);
    }
    _sdfTransformBound=std::sqrt((_sdfWorldToLocal.transpose()*_sdfWorldToLocal)
        .cwiseAbs().rowwise().sum().maxCoeff())*(1+1e-14);
    if(sdfQueries() && (!_sdfWorldToLocal.allFinite() || !_sdfTranslation.allFinite() ||
        !(_sdfTransformBound>0) || !std::isfinite(_sdfTransformBound) ||
        !std::isfinite(_scale) || !std::isfinite(_offset) || !std::isfinite(_min)))
        throw std::invalid_argument("procedural SDF requires a finite invertible transform, scale, offset and minimum");
}

const SdfQueries &ProceduralMean::localSdfQueries() const
{
    const auto *queries=_f ? _f->sdfQueries() : nullptr;
    if(!queries) throw std::invalid_argument("procedural mean does not provide certified SDF queries");
    return *queries;
}

SdfJet ProceduralMean::evaluateJet(const Eigen::Vector3d &p) const
{
    const auto q=localSdfQueries().evaluateJet(_sdfWorldToLocal*p+_sdfTranslation);
    const double value=q.value*double(_scale)+double(_offset);
    if(value<_min || _scale==0) return {std::max(value,double(_min)),Eigen::Vector3d::Zero()};
    return {value,double(_scale)*_sdfWorldToLocal.transpose()*q.gradient};
}

double ProceduralMean::valueDifference(const Eigen::Vector3d &p,const Eigen::Vector3d &d) const
{
    const auto &queries=localSdfQueries();
    if(_scale==0) return 0;
    const Eigen::Vector3d local=_sdfWorldToLocal*p+_sdfTranslation;
    const double a=queries.evaluateJet(local).value*double(_scale)+double(_offset);
    const double delta=queries.valueDifference(local,_sdfWorldToLocal*d)*double(_scale);
    const double b=a+delta;
    if(a>=_min && b>=_min) return delta;
    if(a<=_min && b<=_min) return 0;
    // Preserve the small displacement even when crossing the clamp.
    return a<_min ? (a-double(_min))+delta : double(_min)-a;
}

SdfBounds ProceduralMean::bounds(const Eigen::Vector3d &minimum,const Eigen::Vector3d &maximum) const
{
    const auto &queries=localSdfQueries();
    if(_scale==0) {
        const double value=std::max(double(_offset),double(_min));
        return {value,value,0,true};
    }
    const Eigen::Vector3d center=_sdfWorldToLocal*(0.5*minimum+0.5*maximum)+_sdfTranslation;
    const Eigen::Vector3d halfExtent=_sdfWorldToLocal.cwiseAbs()*(0.5*(maximum-minimum));
    auto b=queries.bounds(center-halfExtent,center+halfExtent);
    if(!b.certified) return b;
    double lo=double(_scale)*b.minimumValue+double(_offset);
    double hi=double(_scale)*b.maximumValue+double(_offset);
    if(lo>hi) std::swap(lo,hi);
    const double error=queryGuard(std::max(std::abs(lo),std::abs(hi)));
    b.minimumValue=std::max(lo-error,double(_min));
    b.maximumValue=std::max(hi+error,double(_min));
    b.maximumGradientNorm=hi+error<_min ? 0 :
        b.maximumGradientNorm*std::abs(double(_scale))*_sdfTransformBound*(1+1e-12);
    return b;
}

SdfRayBounds ProceduralMean::rayBounds(const Eigen::Vector3d &o,const Eigen::Vector3d &w,
                                      double lo,double hi,const SdfRayQueryOptions &options) const
{
    const auto &queries=localSdfQueries();
    if(_scale==0) {
        SdfRayBounds b;
        b.minimumValue=b.maximumValue=std::max(double(_offset),double(_min));
        b.midpointValue=b.minimumValue;
        b.maximumSecondDerivative=0; b.certified=true;
        return b;
    }
    // Keep the world ray parameter: local direction is deliberately not
    // normalized. This also transforms both directional derivative orders.
    const Eigen::Vector3d midpoint=options.midpoint.value_or(o+(0.5*lo+0.5*hi)*w);
    SdfRayQueryOptions localOptions;
    // The local beginning derivative would be overwritten by the world-space
    // query below. Ordinary bounds need neither beginning-point query.
    localOptions.computeBeginDerivative=false;
    localOptions.midpoint=(_sdfWorldToLocal*midpoint+_sdfTranslation).eval();
    auto b=queries.rayBounds(_sdfWorldToLocal*o+_sdfTranslation,_sdfWorldToLocal*w,lo,hi,localOptions);
    if(!b.certified) return b;
    if(std::isfinite(b.midpointValue)) {
        const double value=b.midpointValue*double(_scale)+double(_offset);
        b.midpointError=std::abs(double(_scale))*b.midpointError+
            queryGuard(std::abs(b.midpointValue*double(_scale))+std::abs(double(_offset)));
        b.midpointValue=std::max(value,double(_min));
    }
    double a=double(_scale)*b.minimumValue+double(_offset);
    double z=double(_scale)*b.maximumValue+double(_offset);
    if(a>z) std::swap(a,z);
    const double valueError=queryGuard(std::max(std::abs(a),std::abs(z)));
    a-=valueError; z+=valueError;
    b.minimumValue=std::max(a,double(_min)); b.maximumValue=std::max(z,double(_min));
    const double dl=double(_scale)*b.minimumDerivative,dh=double(_scale)*b.maximumDerivative;
    const double derivativeError=queryGuard(std::max(std::abs(dl),std::abs(dh)));
    b.minimumDerivative=std::min(dl,dh)-derivativeError;
    b.maximumDerivative=std::max(dl,dh)+derivativeError;
    b.maximumSecondDerivative*=std::abs(double(_scale))*(1+1e-12);
    if(options.computeBeginDerivative)
        b.beginDerivative=evaluateJet(o+lo*w).gradient.dot(w);
    if(z<_min) {
        b.minimumDerivative=b.maximumDerivative=b.beginDerivative=0;
        b.maximumSecondDerivative=0;
    } else if(a<=_min) {
        b.minimumDerivative=std::min(0.0,b.minimumDerivative);
        b.maximumDerivative=std::max(0.0,b.maximumDerivative);
        b.maximumSecondDerivative=std::numeric_limits<double>::infinity();
    }
    return b;
}
}
