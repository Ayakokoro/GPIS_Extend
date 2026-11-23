#include "NonstationaryConditioning.h"
#include "math/ClassicGlobalNonstationary.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include <boost/multiprecision/cpp_dec_float.hpp>
#include <boost/multiprecision/eigen.hpp>

namespace Tungsten::ConditionalGpis {
namespace {
template<class T> using V = Eigen::Matrix<T,3,1>;
template<class T> using M = Eigen::Matrix<T,3,3>;
using Precise = boost::multiprecision::number<boost::multiprecision::cpp_dec_float<100>,
                                              boost::multiprecision::et_off>;

// Store the ramp once so its values and derivatives can also be evaluated
// with extended precision at nearly coincident birth and target positions.
struct Parameter {
    double scale=1., first=1., last=1., start=0., end=0., logSlope=0.;
    Precise preciseLogSlope=0;
    int axis=-1;

    template<class Field>
    void initialize(const Field *field, double base, double constant, const ParameterBounds &bound) {
        scale=base;
        first=constant;
        if (!field || bound.gradientNorm==0.) return;
        rapidjson::Document doc;
        const auto desc=field->toJson(doc.GetAllocator());
        if (!desc.HasMember("noise")) throw std::invalid_argument("conditional covariance needs a constant or axis ramp");
        const std::string noise=desc["noise"].GetString();
        if (noise=="left_right") axis=0;
        else if (noise=="bottom_top") axis=1;
        else if (noise=="front_back") axis=2;
        else throw std::invalid_argument("conditional covariance needs a constant or axis ramp");
        first=desc["min"].GetDouble(); last=desc["max"].GetDouble();
        start=desc["start"].GetDouble(); end=desc["end"].GetDouble();
        preciseLogSlope=log((1+Precise(last))/(1+Precise(first)))/(Precise(end)-Precise(start));
        logSlope=static_cast<double>(preciseLogSlope);
    }

    template<class T> void sample(const V<T> &p,T &value,V<T> &gradient) const {
        using std::exp;
        gradient.setZero();
        if (axis<0 || (start<end ? p[axis]<=start : p[axis]>=start)) value=T(first)*scale;
        else if (start<end ? p[axis]>=end : p[axis]<=end) value=T(last)*scale;
        else {
            T slope;
            if constexpr (std::is_same_v<T,Precise>) slope=preciseLogSlope;
            else slope=logSlope;
            const T shifted=(1+T(first))*exp(slope*(p[axis]-start));
            value=T(scale)*(shifted-1);
            gradient[axis]=T(scale)*shifted*slope;
        }
    }
    bool samePlateau(const Point3 &origin,const Vector3 &delta) const {
        if (axis<0) return true;
        const double lo=std::min(start,end)-origin[axis], hi=std::max(start,end)-origin[axis];
        return (lo>=0 && delta[axis]<=lo) || (hi<=0 && delta[axis]>=hi);
    }
};

template<class T> struct Local {
    T amplitude;
    V<T> amplitudeGradient, length;
    std::array<V<T>,3> lengthGradient;
    // The supported length fields vary along at most one world axis, so the
    // covariance of the unit-amplitude field's gradient is diagonal.
    V<T> gradientVariance;
};

template<class T> struct Correlation {
    T value;
    V<T> dx,dy;
    M<T> mixed;
};

// Logarithmic derivatives of the normalized Gibbs correlation. For axis i
// the log correlation is .5 log(2 lxi lyi/S) - (xi-yi)^2/S,
// with S=lxi^2+lyi^2. Include the determinant prefactor's derivatives.
template<class T>
Correlation<T> correlation(const Local<T> &x,const Local<T> &y,const V<T> &delta) {
    using std::sqrt;
    using std::exp;
    T factor=1, distance=0;
    Correlation<T> c;
    c.dx.setZero(); c.dy.setZero(); c.mixed.setZero();
    for (int i=0;i<3;++i) {
        const T lx=x.length[i], ly=y.length[i], d=delta[i];
        const T inv=1/(lx*lx+ly*ly), q=d*d*inv;
        factor*=2*lx*ly*inv;
        distance+=q;
        const V<T> &gx=x.lengthGradient[i], &gy=y.lengthGradient[i];
        c.dx[i]-=2*d*inv;
        c.dy[i]+=2*d*inv;
        c.dx+=(T(.5)/lx+(2*q-1)*lx*inv)*gx;
        c.dy+=(T(.5)/ly+(2*q-1)*ly*inv)*gy;
        c.mixed(i,i)+=2*inv;
        c.mixed.row(i)+=(4*d*ly*inv*inv)*gy.transpose();
        c.mixed.col(i)-=(4*d*lx*inv*inv)*gx;
        c.mixed+=(2*lx*ly*inv*inv*(1-4*q))*gx*gy.transpose();
    }
    c.value=sqrt(factor)*exp(-distance);
    c.mixed+=c.dx*c.dy.transpose();  // (d_x d_y C)/C
    return c;
}
}

struct NonstationaryParameters {
    Parameter amplitude;
    std::array<Parameter,3> length;
    double minLength, minSigma;

    explicit NonstationaryParameters(const ProceduralNonstationaryCovariance &source) {
        // Reuse classic_global's supported-field, derivative and positivity checks.
        const ClassicGlobal::NonstationaryKernel checked(source);
        minSigma=checked.minimumSigma();
        const auto &base=*source.stationaryCovariance();
        const Vec3d zero(0.), axis(1.,0.,0.);
        const double variance=base(Derivative::None,Derivative::None,zero,zero,zero,zero);
        const double width=std::sqrt(variance/base(Derivative::First,Derivative::First,zero,zero,axis,axis));
        const auto &amp=source.amplitudeField();
        ParameterBounds bound{1,1,0};
        if (amp) amp->parameterBounds(bound);
        amplitude.initialize(amp.get(),std::sqrt(variance),amp ? amp->parameterJet(zero).value : 1.,bound);
        const auto &widths=source.lengthScaleField();
        VectorParameterBounds bounds;
        bounds.fill({1,1,0});
        if (widths) widths->parameterBounds(bounds);
        minLength=std::numeric_limits<double>::infinity();
        for (int i=0;i<3;++i) {
            length[i].initialize(widths.get(),width,widths ? widths->parameterJet(zero)[i].value : 1.,bounds[i]);
            minLength=std::min(minLength,width*bounds[i].minimum);
        }
    }
    template<class T> Local<T> sample(const V<T> &p) const {
        Local<T> local;
        amplitude.sample(p,local.amplitude,local.amplitudeGradient);
        for (int i=0;i<3;++i) {
            length[i].sample(p,local.length[i],local.lengthGradient[i]);
            local.gradientVariance[i]=1/(local.length[i]*local.length[i]);
        }
        for (int i=0;i<3;++i)
            local.gradientVariance+=T(.5)*(local.lengthGradient[i]/local.length[i]).cwiseAbs2();
        return local;
    }
    template<class T> Local<T> target(const FlightState &birth,double age,const V<T> &delta) const {
        V<T> p=birth.birthPosition.cast<T>()+delta;
        auto snapToPlane=[&](const Parameter &field) {
            const int k=field.axis;
            if (k<0 || birth.direction[k]==0.) return;
            for (double plane : {field.start,field.end})
                if (age==(plane-birth.birthPosition[k])/birth.direction[k]) p[k]=plane;
        };
        // nextBreakpoint uses these same distances. Its endpoint takes the
        // exterior derivative even when the ray multiply rounds off the plane.
        snapToPlane(amplitude);
        snapToPlane(length[0]); // All three components share the width ramp.
        return sample(p);
    }
};

NonstationaryConditioning::NonstationaryConditioning(const ProceduralNonstationaryCovariance &source)
    : _parameters(std::make_shared<NonstationaryParameters>(source)) {}
double NonstationaryConditioning::minLength() const { return _parameters->minLength; }
double NonstationaryConditioning::minSigma() const { return _parameters->minSigma; }
double NonstationaryConditioning::nextBreakpoint(const Point3 &origin,const Vector3 &direction,double age) const {
    double next=std::numeric_limits<double>::infinity();
    auto inspect=[&](const Parameter &p) {
        if (p.axis<0 || direction[p.axis]==0.) return;
        for (double plane : {p.start,p.end}) {
            const double t=(plane-origin[p.axis])/direction[p.axis];
            if (t>age) next=std::min(next,t);
        }
    };
    inspect(_parameters->amplitude);
    for (const auto &p : _parameters->length) inspect(p);
    return next;
}
KernelJet NonstationaryConditioning::evaluate(const Point3 &x,const Point3 &y) const {
    const auto a=_parameters->sample(x), b=_parameters->sample(y);
    const auto c=correlation(a,b,(x-y).eval());
    const Vector3 ax=a.amplitudeGradient/a.amplitude, by=b.amplitudeGradient/b.amplitude;
    const double k=a.amplitude*b.amplitude*c.value;
    return {k,k*(ax+c.dx),k*(by+c.dy),
            k*(c.mixed+ax*c.dy.transpose()+(ax+c.dx)*by.transpose())};
}
PointPrior NonstationaryConditioning::pointPrior(const Point3 &x,const MeanJet &mean) const {
    const auto local=_parameters->sample(x);
    const double variance=local.amplitude*local.amplitude;
    Matrix3 covariance=local.amplitudeGradient*local.amplitudeGradient.transpose();
    covariance.diagonal()+=variance*local.gradientVariance;
    return {mean.value,variance,mean.gradient,covariance,local.amplitude*local.amplitudeGradient};
}

NonstationaryFlight::NonstationaryFlight(const NonstationaryConditioning &kernel,const MeanField &mean,
                                         const FlightState &birth)
    : _kernel(kernel),_mean(mean),_birth(birth),_birthMean(mean.evaluate(birth.birthPosition)),
      _cursor(mean.createRayCursor(birth.birthPosition,birth.direction)) {
    const auto y=kernel._parameters->sample(birth.birthPosition);
    const double invLength=birth.direction.cwiseQuotient(y.length).norm();
    if (!(invLength>0) || !std::isfinite(invLength) || !birth.birthGradient.allFinite() ||
        !std::isfinite(birth.birthValue) || std::abs(birth.direction.squaredNorm()-1)>1e-5)
        throw std::invalid_argument("invalid nonstationary conditional birth");
    // The variance after two value/gradient observations includes a sixth-
    // order cancellation along the flight. Switch before double loses it.
    _nearDistance=.08/invLength;
}

namespace {
// On a length-scale plateau the normalized field U is exactly stationary,
// even when its amplitude varies. Use the SE remainders directly, since
// most short dragon flights lie completely outside the length ramp.
NonstationaryMoments plateauMoments(const Local<double> &x,const Local<double> &y,
        const Parameter &amplitude,const FlightState &birth,const MeanJet &birthMean,
        const MeanJet &target,double age,double meanDifference) {
    const Vector3 p=y.gradientVariance.cwiseProduct(birth.direction);
    const double gamma2=birth.direction.dot(p), u=age*age*gamma2;
    const auto e=MajorantBounds::scaledExponentialRemainders(u);
    const double r=std::exp(-.5*u), a=x.amplitude, b=(birth.birthValue-birthMean.value)/y.amplitude;
    const Vector3 g=(birth.birthGradient-birthMean.gradient-b*y.amplitudeGradient)/y.amplitude;
    const double slope=g.dot(birth.direction), mu=r*(b+age*slope);
    double amplitudeDifference=0;
    if (amplitude.axis>=0) {
        const int k=amplitude.axis;
        const double lo=std::min(amplitude.start,amplitude.end)-birth.birthPosition[k];
        const double hi=std::max(amplitude.start,amplitude.end)-birth.birthPosition[k];
        const double dx=std::clamp(age*birth.direction[k],lo,hi)-std::clamp(0.,lo,hi);
        amplitudeDifference=(y.amplitude+amplitude.scale)*std::expm1(amplitude.logSlope*dx);
    }
    NonstationaryMoments result;
    result.meanF=birth.birthValue+meanDifference+b*amplitudeDifference+
            a*(b*std::expm1(-.5*u)+r*age*slope);
    result.meanG=target.gradient+x.amplitudeGradient*mu+a*(r*g-age*mu*p);
    result.regression=x.amplitudeGradient/a+p/(gamma2*age*e.first);
    result.varianceF=a*a*std::exp(-u)*u*u*e.first;
    const double axial=a*a*u*u*e.second/e.first;
    const Matrix3 longitudinal=p*p.transpose()/gamma2;
    const Matrix3 precision=y.gradientVariance.asDiagonal();
    result.gradientGivenValueCovariance=(-a*a*std::expm1(-u))*(precision-longitudinal)+axial*longitudinal;
    result.slopeVarianceGivenValue=axial*gamma2;
    return result;
}

template<class T>
NonstationaryMoments conditioned(const NonstationaryParameters &params,const FlightState &birth,
        const MeanJet &birthMean,const MeanJet &target,double age,double meanDifference,bool near) {
    const V<T> direction=birth.direction.cast<T>(), origin=birth.birthPosition.cast<T>();
    const V<T> delta=T(age)*direction;
    const auto x=params.target(birth,age,delta), y=params.sample(origin);
    const auto c=correlation(x,y,delta);
    // Write F=m+aU. U_y and grad U_y are independent, with variances
    // 1 and D_y respectively. This diagonalizes the full birth observation.
    const T birthResidual=T(birth.birthValue)-T(birthMean.value);
    const T b=birthResidual/y.amplitude;
    const V<T> g=((birth.birthGradient.cast<T>()-birthMean.gradient.cast<T>())-
                  b*y.amplitudeGradient)/y.amplitude;
    const V<T> inverse=y.gradientVariance.cwiseInverse();
    const V<T> weight=inverse.cwiseProduct(g), slope=inverse.cwiseProduct(c.dy);
    const T mu=c.value*(b+c.dy.dot(weight));
    const V<T> gradMu=c.value*(b*c.dx+c.mixed*weight);
    const T r2=c.value*c.value;
    const T vf=1-r2*(1+c.dy.dot(slope));
    if (!(vf>0)) throw std::runtime_error("nonstationary conditional value variance lost precision");
    const V<T> fg=-r2*(c.dx+c.mixed*slope);
    M<T> gg=-r2*(c.dx*c.dx.transpose()+c.mixed*inverse.asDiagonal()*c.mixed.transpose());
    gg.diagonal()+=x.gradientVariance;
    const T a=x.amplitude;
    const M<T> given=a*a*(gg-fg*fg.transpose()/vf);
    NonstationaryMoments result;
    const T f=near ? T(birth.birthValue)+T(meanDifference)+(a*mu-birthResidual) : T(target.value)+a*mu;
    result.meanF=static_cast<double>(f);
    result.varianceF=static_cast<double>(a*a*vf);
    result.meanG=(target.gradient.cast<T>()+x.amplitudeGradient*mu+a*gradMu).template cast<double>();
    result.regression=(x.amplitudeGradient/a+fg/vf).template cast<double>();
    result.gradientGivenValueCovariance=((given+given.transpose())*T(.5)).template cast<double>();
    result.slopeVarianceGivenValue=static_cast<double>(direction.dot(given*direction));
    if (!(result.varianceF>0 && result.slopeVarianceGivenValue>=0) ||
        !std::isfinite(result.meanF) || !std::isfinite(result.varianceF) ||
        !std::isfinite(result.slopeVarianceGivenValue) ||
        !result.meanG.allFinite() || !result.regression.allFinite() ||
        !result.gradientGivenValueCovariance.allFinite())
        throw std::runtime_error("non-finite or unresolved nonstationary conditional moments");
    return result;
}
}

NonstationaryMoments NonstationaryFlight::moments(double age) const {
    if (!(age>0.) || !std::isfinite(age))
        throw std::invalid_argument("nonstationary conditional query needs a positive finite age");
    const Vector3 displacement=age*_birth.direction;
    const MeanJet m=_cursor ? _cursor->evaluate(age) : _mean.evaluate(_birth.birthPosition+displacement);
    if (age<_nearDistance) {
        const double difference=_cursor ? _cursor->valueDifference(age) : _mean.valueDifference(_birth.birthPosition,displacement);
        const auto &params=*_kernel._parameters;
        // Both endpoints must use the plateau's spatial derivative. A ray
        // tangent to an interior ramp still observes transverse width changes.
        bool plateau=true;
        for (const auto &width : params.length)
            plateau=plateau && width.samePlateau(_birth.birthPosition,displacement);
        if (plateau) {
            const auto y=params.sample(_birth.birthPosition), x=params.target(_birth,age,displacement);
            return plateauMoments(x,y,params.amplitude,_birth,_birthMean,m,age,difference);
        }
        return conditioned<Precise>(*_kernel._parameters,_birth,_birthMean,m,age,difference,true);
    }
    return conditioned<double>(*_kernel._parameters,_birth,_birthMean,m,age,0.,false);
}
PointPrior NonstationaryFlight::at(double age) const {
    if (age==0.) return {_birth.birthValue,0.,_birth.birthGradient,Matrix3::Zero(),Vector3::Zero()};
    const auto m=moments(age);
    const Vector3 fg=m.varianceF*m.regression;
    return {m.meanF,m.varianceF,m.meanG,m.gradientGivenValueCovariance+fg*m.regression.transpose(),fg};
}
}
