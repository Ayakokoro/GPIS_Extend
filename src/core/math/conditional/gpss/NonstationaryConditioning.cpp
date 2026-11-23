#include "NonstationaryConditioning.h"
#include "NonstationaryNearDouble.h"
#include "math/ClassicGlobalNonstationary.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/transport/ConditionalMajorantBounds.h"

namespace Tungsten::ConditionalGpis {
namespace {
struct RampIncrement {
    double value = 0, beyondTangent = 0, gradient = 0, gradientChange = 0;
};

struct Parameter {
    double scale=1., first=1., last=1., start=0., end=0., logSlope=0.;
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
        const double logRatio=std::log1p(std::abs(last-first)/(1+std::min(first,last)));
        logSlope=std::copysign(logRatio,last-first)/(end-start);
    }

    void sample(const Point3 &p,double &value,Vector3 &gradient) const {
        gradient.setZero();
        if (axis<0 || (start<end ? p[axis]<=start : p[axis]>=start)) value=first*scale;
        else if (start<end ? p[axis]>=end : p[axis]<=end) value=last*scale;
        else {
            const double shifted=(1+first)*std::exp(logSlope*(p[axis]-start));
            value=scale*(shifted-1);
            gradient[axis]=scale*shifted*logSlope;
        }
    }
    bool samePlateau(const Point3 &origin,const Vector3 &delta) const {
        if (axis<0) return true;
        const double lo=std::min(start,end)-origin[axis], hi=std::max(start,end)-origin[axis];
        return (lo>=0 && delta[axis]<=lo) || (hi<=0 && delta[axis]>=hi);
    }
    RampIncrement increment(const FlightState &birth,const Vector3 &delta,double age,double initial) const {
        if (axis<0) return {};
        const double lo=std::min(start,end)-birth.birthPosition[axis];
        const double hi=std::max(start,end)-birth.birthPosition[axis];
        double travel=delta[axis];
        if (birth.direction[axis]!=0.) {
            if (age==lo/birth.direction[axis]) travel=lo;
            if (age==hi/birth.direction[axis]) travel=hi;
        }
        const double step=std::clamp(travel,lo,hi)-std::clamp(0.,lo,hi);
        const double z=logSlope*step, shifted=initial+scale;
        const double em1=std::expm1(z);
        // exp(z)-1-z has a double zero at the origin.
        const double remainder=std::abs(z)<0.01 ?
            z*z*(0.5+z*(1./6+z*(1./24+z*(1./120+z*(1./720+z/5040))))) : em1-z;
        const bool birthActive=lo<0 && 0<hi, targetActive=lo<travel && travel<hi;
        RampIncrement change;
        change.value=shifted*em1;
        change.beyondTangent=shifted*(remainder+logSlope*(birthActive ? step-delta[axis] : step));
        change.gradient=targetActive ? (shifted+change.value)*logSlope : 0.;
        change.gradientChange=birthActive && targetActive ? change.value*logSlope :
            change.gradient-(birthActive ? shifted*logSlope : 0.);
        return change;
    }
};

struct Local {
    double amplitude;
    Vector3 amplitudeGradient, length;
    std::array<Vector3,3> lengthGradient;
    // The supported length fields vary along at most one world axis, so the
    // covariance of the unit-amplitude field's gradient is diagonal.
    Vector3 gradientVariance;
    void updateGradientVariance() {
        gradientVariance=length.cwiseAbs2().cwiseInverse();
        for (int i=0;i<3;++i)
            gradientVariance+=0.5*(lengthGradient[i]/length[i]).cwiseAbs2();
    }
};

struct Correlation {
    double value, logValue;
    Vector3 dx,dy;
    Matrix3 mixed;
};

// Logarithmic derivatives of the normalized Gibbs correlation. For axis i
// the log correlation is .5 log(2 lxi lyi/S) - (xi-yi)^2/S,
// with S=lxi^2+lyi^2. Include the determinant prefactor's derivatives.
Correlation correlation(const Local &x,const Local &y,const Vector3 &delta,
        const Vector3 *lengthChange=nullptr) {
    using std::sqrt;
    using std::exp;
    double factor=1, distance=0;
    Correlation c;
    c.logValue=0;
    c.dx.setZero(); c.dy.setZero(); c.mixed.setZero();
    for (int i=0;i<3;++i) {
        const double lx=x.length[i], ly=y.length[i], d=delta[i];
        const double inv=1/(lx*lx+ly*ly), q=d*d*inv;
        factor*=2*lx*ly*inv;
        distance+=q;
        const Vector3 &gx=x.lengthGradient[i], &gy=y.lengthGradient[i];
        c.dx[i]-=2*d*inv;
        c.dy[i]+=2*d*inv;
        if (lengthChange) {
            const double dl=(*lengthChange)[i];
            c.logValue-=q+0.5*std::log1p(0.5*(dl/lx)*(dl/ly));
            const double widthTerm=0.5*dl*(lx+ly)*inv;
            c.dx+=(-widthTerm/lx+2*q*lx*inv)*gx;
            c.dy+=( widthTerm/ly+2*q*ly*inv)*gy;
        } else {
            c.dx+=(0.5/lx+(2*q-1)*lx*inv)*gx;
            c.dy+=(0.5/ly+(2*q-1)*ly*inv)*gy;
        }
        c.mixed(i,i)+=2*inv;
        c.mixed.row(i)+=(4*d*ly*inv*inv)*gy.transpose();
        c.mixed.col(i)-=(4*d*lx*inv*inv)*gx;
        c.mixed+=(2*lx*ly*inv*inv*(1-4*q))*gx*gy.transpose();
    }
    c.value=lengthChange ? exp(c.logValue) : sqrt(factor)*exp(-distance);
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
    Local sample(const Point3 &p) const {
        Local local;
        amplitude.sample(p,local.amplitude,local.amplitudeGradient);
        for (int i=0;i<3;++i)
            length[i].sample(p,local.length[i],local.lengthGradient[i]);
        local.updateGradientVariance();
        return local;
    }
    Local target(const FlightState &birth,double age,const Vector3 &delta) const {
        Point3 p=birth.birthPosition+delta;
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

namespace {
using MajorantBounds::Interval;
using MajorantBounds::up;
using MajorantBounds::down;
constexpr double infinity=std::numeric_limits<double>::infinity();
constexpr double roundoff=4096*std::numeric_limits<double>::epsilon();

struct ParameterRange { Interval value, derivative; };

// Enclose the same exponential and both exterior endpoint conventions as
// Parameter::sample, including when a coordinate rounds onto a ramp plane.
ParameterRange parameterRange(const Parameter &p,Interval coordinate,double direction) {
    const Interval first=Interval(p.first)*p.scale;
    if (p.axis<0) return {first,0.};
    const Interval last=Interval(p.last)*p.scale;
    const double left=std::min(p.start,p.end),right=std::max(p.start,p.end);
    const bool increasing=p.start<p.end;
    if (coordinate.upper<=left) return {increasing ? first : last,0.};
    if (coordinate.lower>=right) return {increasing ? last : first,0.};
    const Interval inside(std::max(left,coordinate.lower),std::min(right,coordinate.upper));
    const Interval shifted=(Interval(1)+p.first)*MajorantBounds::exp(p.logSlope*(inside-p.start));
    ParameterRange range{p.scale*(shifted-1.),p.scale*shifted*p.logSlope*direction};
    if (coordinate.lower<=left) range.value=hull(range.value,increasing ? first : last);
    if (coordinate.upper>=right) range.value=hull(range.value,increasing ? last : first);
    if (coordinate.lower<=left || coordinate.upper>=right) range.derivative=hull(range.derivative,0.);
    return range;
}
double magnitude(Interval x) { return std::max(std::abs(x.lower),std::abs(x.upper)); }

struct CorrelationEnvelope {
    double decay=0., slopeSd=0., gradientScale=0.;
    std::array<double,3> dx{},dy{},mixed{};
};

// For C=Corr(U_x,U_y), each logarithmic derivative is a quadratic in
// distance with bounded coefficients. dx bounds |d_t log C|, dy bounds
// ||D_y^(-1/2) grad_y log C|| and mixed bounds the corresponding mixed
// logarithmic derivative. Sums of absolute components bound Euclidean norms.
CorrelationEnvelope correlationEnvelope(const NonstationaryParameters &params,const Local &y,
                                        const Vector3 &w,bool plateau) {
    Interval decay,variance,trace;
    std::array<Interval,3> dx{},dy{},mixed{};
    for (int i=0;i<3;++i) {
        const auto &field=params.length[i];
        const int axis=field.axis;
        const ParameterRange range=plateau ? ParameterRange{y.length[i],0.} :
            parameterRange(field,Interval(-infinity,infinity),1.);
        if (!(range.value.lower>0. && std::isfinite(range.value.upper))) return {};
        const Interval wi=std::abs(w[i]),ly=y.length[i],lx=range.value;
        const Interval inv=Interval(1)/(lx*lx+ly*ly),q2=wi*wi*inv;
        const Interval invX=Interval(1)/lx;
        // D_y >= diag(1/l_y^2). For a uniaxial width gradient its
        // whitened norm is bounded by l_y[axis]*|grad l_yi|.
        const Interval gy=axis<0 ? Interval(0.) :
            Interval(y.length[axis])*std::abs(y.lengthGradient[i][axis]);
        const Interval gradient=magnitude(range.derivative);
        const Interval gx=axis<0 ? Interval(0.) : gradient*std::abs(w[axis]);
        decay=decay+Interval(q2.lower);
        variance=variance+(wi*wi+.5*gx*gx)*invX*invX;
        trace=trace+(Interval(1.)+.5*gradient*gradient)*invX*invX;
        // |1/(2*l)-l/S| <= 1/(2*l), S=lx^2+ly^2.
        dx[0]=dx[0]+.5*invX*gx;
        dx[1]=dx[1]+2.*wi*wi*inv;
        dx[2]=dx[2]+2.*q2*lx*inv*gx;
        dy[0]=dy[0]+.5/ly*gy;
        dy[1]=dy[1]+2.*wi*inv*ly;
        dy[2]=dy[2]+2.*q2*ly*inv*gy;
        mixed[0]=mixed[0]+2.*inv*wi*ly+2.*lx*ly*inv*inv*gx*gy;
        mixed[1]=mixed[1]+4.*wi*inv*inv*(wi*ly*gy+lx*ly*gx);
        mixed[2]=mixed[2]+8.*lx*ly*inv*inv*q2*gx*gy;
    }
    CorrelationEnvelope env;
    if (!std::isfinite(decay.lower) || !std::isfinite(variance.upper) ||
        !std::isfinite(trace.upper) || !(variance.upper>0. && trace.upper>0.))
        return {};
    env.decay=std::max(0.,down(decay.lower*(1-roundoff)));
    env.slopeSd=MajorantBounds::sqrt(variance+roundoff*trace).upper;
    env.gradientScale=MajorantBounds::sqrt(trace).upper;
    for (int j=0;j<3;++j) {
        if (!std::isfinite(dx[j].upper) || !std::isfinite(dy[j].upper) ||
            !std::isfinite(mixed[j].upper)) return {};
        env.dx[j]=std::max(0.,dx[j].upper);
        env.dy[j]=std::max(0.,dy[j].upper);
        env.mixed[j]=std::max(0.,mixed[j].upper);
    }
    return env;
}
}

struct NonstationaryBoundData {
    struct SnappedPlane { int axis; double coordinate,age; };
    std::array<SnappedPlane,4> planes{};
    int planeCount=0;
    CorrelationEnvelope global,plateau;
    const Parameter *widthRamp=nullptr;
    Interval constantDecay,rampSpeedSquared,birthRampWidthSquared;
    double observationNorm,meanGradientBound,plateauEnd;
    NonstationaryBoundData(const NonstationaryParameters &params,const MeanField &mean,
                           const FlightState &birth,const MeanJet &birthMean) {
        const Local y=params.sample(birth.birthPosition);
        const auto &w=birth.direction;
        // target() can snap both parameters' shared coordinate on either
        // ramp's quotient, including when two nearby planes round to one age.
        for (const Parameter *p : {&params.amplitude,&params.length[0]}) {
            if (p->axis<0 || w[p->axis]==0.) continue;
            for (double plane : {p->start,p->end})
                planes[planeCount++]={p->axis,plane,(plane-birth.birthPosition[p->axis])/w[p->axis]};
        }
        global=correlationEnvelope(params,y,w,false);
        plateauEnd=infinity;
        for (const auto &field:params.length) {
            const int axis=field.axis;
            if (axis<0) continue;
            const double left=std::min(field.start,field.end),right=std::max(field.start,field.end);
            const double coordinate=birth.birthPosition[axis];
            if (coordinate>left && coordinate<right) plateauEnd=0.;
            else if ((w[axis]>0. && coordinate<=left) || (w[axis]<0. && coordinate>=right)) {
                // Bound the rounded product t*w before addition to birth.
                // Addition is monotone and the ramp plane is representable.
                const Interval gap=w[axis]>0. ? Interval(left)-coordinate : Interval(coordinate)-right;
                const Interval travel=gap-std::numeric_limits<double>::denorm_min();
                const Interval speed=Interval(std::abs(w[axis]))*(1.+roundoff);
                plateauEnd=std::min(plateauEnd,(travel/speed).lower);
            }
            for (int i=0;i<planeCount;++i) {
                const auto &p=planes[i];
                if (p.age>=0. && p.axis==axis &&
                    ((coordinate<=left && p.coordinate>left) || (coordinate>=right && p.coordinate<right)))
                    plateauEnd=std::min(plateauEnd,p.age);
            }
        }
        if (plateauEnd>0.) plateau=correlationEnvelope(params,y,w,true);
        if (plateauEnd<infinity) {
            // Varying components share one vector field's ramp and base width.
            // Their Gibbs exponents therefore have the same denominator.
            for (int i=0;i<3;++i) {
                const Interval speed2=Interval(w[i])*w[i], width2=Interval(y.length[i])*y.length[i];
                if (params.length[i].axis<0) {
                    constantDecay=constantDecay+speed2/(2.*width2);
                } else {
                    widthRamp=&params.length[i];
                    birthRampWidthSquared=width2;
                    rampSpeedSquared=rampSpeedSquared+speed2;
                }
            }
        }
        const Interval b=(Interval(birth.birthValue)-birthMean.value)/y.amplitude;
        Interval squared=b*b;
        for (int i=0;i<3;++i) {
            const Interval g=(Interval(birth.birthGradient[i])-birthMean.gradient[i]-
                              b*y.amplitudeGradient[i])/y.amplitude;
            squared=squared+g*g/y.gradientVariance[i];
        }
        observationNorm=std::isfinite(squared.upper) && squared.upper>=0. ?
            up(MajorantBounds::sqrt(squared).upper*(1+roundoff)) : infinity;
        const auto globalMean=mean.globalMagnitudeBounds();
        meanGradientBound=globalMean ? globalMean->maximumGradientNorm : infinity;
    }

    Interval targetCoordinate(const FlightState &birth,Interval age,int axis) const {
        if (axis<0) return 0.;
        Interval coordinate=Interval(birth.birthPosition[axis])+age*birth.direction[axis];
        for (int i=0;i<planeCount;++i) {
            const auto &p=planes[i];
            if (p.axis==axis && age.lower<=p.age && p.age<=age.upper)
                coordinate=hull(coordinate,p.coordinate);
        }
        return coordinate;
    }

    double intervalDecay(const FlightState &birth,Interval age) const {
        // The Gibbs determinant prefactor is at most one. Bound the exponent
        // with the largest target width reachable in this interval on each axis.
        if (!widthRamp) return global.decay;
        const Interval coordinate=targetCoordinate(birth,age,widthRamp->axis);
        const double maximum=parameterRange(*widthRamp,coordinate,0.).value.upper;
        if (!(maximum>0. && std::isfinite(maximum))) return global.decay;
        const Interval denominator=birthRampWidthSquared+Interval(maximum)*maximum;
        const Interval decay=constantDecay+rampSpeedSquared/denominator;
        return std::max(global.decay,down(decay.lower*(1-roundoff)));
    }
};

NonstationaryFlight::~NonstationaryFlight()=default;
NonstationaryFlight::NonstationaryFlight(NonstationaryFlight &&) noexcept=default;

double NonstationaryFlight::hazardMajorant(double lo,double hi,const MeanRayBounds &mean) const {
    if (!(lo>=_nearDistance && hi>=lo && std::isfinite(hi) && mean.certified &&
          std::isfinite(mean.minimumValue) && std::isfinite(mean.maximumValue) &&
          std::isfinite(mean.minimumDerivative) && std::isfinite(mean.maximumDerivative) &&
          mean.minimumValue<=mean.maximumValue && mean.minimumDerivative<=mean.maximumDerivative))
        return infinity;
    const auto &params=*_kernel._parameters;
    if (!_boundData)
        _boundData=std::make_unique<NonstationaryBoundData>(params,_mean,_birth,_birthMean);
    const auto &data=*_boundData;
    const bool plateau=hi<data.plateauEnd;
    const auto &env=plateau ? data.plateau : data.global;
    if (!(env.decay>0. && std::isfinite(env.slopeSd) && std::isfinite(data.observationNorm)))
        return infinity;
    const Interval age(lo,hi);
    const double decay=plateau ? env.decay : data.intervalDecay(_birth,age);
    const Interval rho=MajorantBounds::exp(-Interval(decay)*Interval(lo)*lo);
    double projection,projectionSlope;
    if (plateau) {
        // With u=t^2 w^T D_y w, |k|^2=(1+u)e^-u decreases with t.
        const Interval uLow=2.*Interval(env.decay)*lo*lo;
        const Interval uHigh=Interval(env.slopeSd)*env.slopeSd*hi*hi;
        projection=(rho*MajorantBounds::sqrt(Interval(1.)+uLow)).upper;
        projectionSlope=(rho*env.slopeSd*MajorantBounds::sqrt(Interval(1.)+uHigh*uHigh)).upper;
    } else {
        auto polynomial=[hi](const std::array<double,3> &c) {
            return Interval(c[0])+hi*(Interval(c[1])+hi*Interval(c[2]));
        };
        const Interval dy=polynomial(env.dy),norm=MajorantBounds::sqrt(Interval(1.)+dy*dy);
        projection=(rho*norm).upper;
        projectionSlope=(rho*(polynomial(env.dx)*norm+polynomial(env.mixed))).upper;
    }
    // Whitened birth observation h, value cross covariance k, and k'=d_t k.
    // Var(U|h)=1-|k|^2. Cauchy-Schwarz gives |k'|<=sd(U'), even if
    // endpoint interval bounds for the kernel derivatives are very broad.
    if (!std::isfinite(projection) || !std::isfinite(projectionSlope)) return infinity;
    projection=up(projection*(1+roundoff));
    // The exact Cauchy cap need not bound rounding of cancelling mixed
    // derivative terms. Retain their absolute scale in both the slope and
    // the dot product with the rounded gradient covariance matrix.
    const Interval rawSlope=projectionSlope;
    projectionSlope=(Interval(std::min(env.slopeSd,projectionSlope))+roundoff*rawSlope).upper;
    const Interval slopeVariance=Interval(env.slopeSd)*env.slopeSd+roundoff*rawSlope*rawSlope;
    if (!std::isfinite(slopeVariance.upper)) return infinity;
    const double slopeSd=MajorantBounds::sqrt(slopeVariance).upper;
    const Interval v=Interval(1.-roundoff)-Interval(projection)*projection;
    if (!(v.lower>=.125 && std::isfinite(projectionSlope))) return infinity;
    const double sd=MajorantBounds::sqrt(v).lower;

    const auto &amp=params.amplitude;
    const int axis=amp.axis;
    const Interval coordinate=data.targetCoordinate(_birth,age,axis);
    const auto amplitude=parameterRange(amp,coordinate,axis<0 ? 0. : _birth.direction[axis]);
    if (!(amplitude.value.lower>0. && std::isfinite(amplitude.value.upper))) return infinity;
    // Level zero and its Rice hazard are invariant under F/a, as a>0.
    // Its posterior mean is mu=q+k.h, q=m/a; covariance with the slope is
    // -k.k'. Work in this normalized field so amplitude gradients cancel
    // exactly at the level crossing.
    const Interval q=Interval(mean.minimumValue,mean.maximumValue)/amplitude.value;
    const Interval qPrime=(Interval(mean.minimumDerivative,mean.maximumDerivative)-
                           q*amplitude.derivative)/amplitude.value;
    const Interval shift=Interval(data.observationNorm)*projection;
    const Interval mu=q+Interval(-shift.upper,shift.upper);
    const Interval regression=Interval(projection)*projectionSlope/v.lower;
    double meanGradientBound=data.meanGradientBound;
    if (!std::isfinite(meanGradientBound)) {
        Bounds3 box;
        for (int i=0;i<3;++i) {
            const Interval p=Interval(_birth.birthPosition[i])+age*_birth.direction[i];
            box.minimum[i]=down(p.lower);
            box.maximum[i]=up(p.upper);
        }
        const auto bounds=_mean.bounds(box);
        if (!bounds.certified) return infinity;
        meanGradientBound=bounds.maximumGradientNorm;
    }
    // Allow absolute error from the point evaluator's world-space dot,
    // Schur complement and amplitude-gradient cancellation. The variance
    // floor above keeps divisions in the Schur complement well conditioned.
    const Interval amplitudeSlope=magnitude(amplitude.derivative)/Interval(amplitude.value.lower);
    const Interval meanScale=magnitude(q)+shift;
    const Interval slopeScale=Interval(meanGradientBound)/amplitude.value.lower+
        (Interval(1.)+data.observationNorm)*env.gradientScale+
        (amplitudeSlope+regression)*meanScale;
    const Interval meanLower=Interval(mu.lower)-roundoff*meanScale;
    const double z=(meanLower/(meanLower.lower<0. ? sd : 1.+roundoff)).lower;
    const Interval betaLower=Interval(qPrime.lower)-Interval(data.observationNorm)*projectionSlope-
        regression*magnitude(mu)-roundoff*slopeScale;
    if (!(std::isfinite(z) && std::isfinite(betaLower.lower) && std::isfinite(meanGradientBound)))
        return infinity;
    // For z>=0, Phi(z)>=1/2; for z<0 the inverse Mills ratio is <=1-z.
    // E[(-N(beta,s^2))_+] <= max(-beta,0)+s/sqrt(2*pi).
    const Interval mills=z<0. ? Interval(1.)-z :
        Interval(.798)*MajorantBounds::exp(Interval(-.5)*z*z);
    const Interval flux=Interval(std::max(0.,-betaLower.lower))+.399*Interval(slopeSd);
    const double maximum=(mills*flux/sd*(1.+roundoff)).upper;
    return std::isfinite(maximum) && maximum>=0. ? maximum : infinity;
}

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
    // Use the near-field residual form before the ordinary Schur complement
    // loses the small conditional variance.
    _nearDistance=.08/invLength;
}

namespace {
// On a length-scale plateau the normalized field U is exactly stationary,
// even when its amplitude varies. Use the SE remainders directly, since
// most short dragon flights lie completely outside the length ramp.
NonstationaryMoments plateauMoments(const Local &x,const Local &y,
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

template<bool Near>
NonstationaryMoments conditioned(const NonstationaryParameters &params,const FlightState &birth,
        const MeanJet &birthMean,const MeanJet &target,double age,double meanDifference) {
    const Vector3 &direction=birth.direction;
    const Vector3 delta=age*direction;
    const auto y=params.sample(birth.birthPosition);
    auto x=params.target(birth,age,delta);
    Vector3 widthChange, widthRemainder;
    std::array<Vector3,3> gradientChange;
    RampIncrement amplitude;
    if constexpr (Near) {
        amplitude=params.amplitude.increment(birth,delta,age,y.amplitude);
        if (params.amplitude.axis>=0) x.amplitudeGradient[params.amplitude.axis]=amplitude.gradient;
        for (int i=0;i<3;++i) {
            const auto &width=params.length[i];
            const auto change=width.increment(birth,delta,age,y.length[i]);
            widthChange[i]=change.value;
            widthRemainder[i]=change.beyondTangent;
            gradientChange[i].setZero();
            if (width.axis>=0) {
                gradientChange[i][width.axis]=change.gradientChange;
                x.lengthGradient[i][width.axis]=change.gradient;
            }
        }
    }
    const auto c=correlation(x,y,delta,Near ? &widthChange : nullptr);
    // Write F=m+aU. U_y and grad U_y are independent, with variances
    // 1 and D_y respectively. This diagonalizes the full birth observation.
    const double a=x.amplitude, b=(birth.birthValue-birthMean.value)/y.amplitude;
    const Vector3 g=(birth.birthGradient-birthMean.gradient-b*y.amplitudeGradient)/y.amplitude;
    const Vector3 inverse=y.gradientVariance.cwiseInverse();
    const Vector3 weight=inverse.cwiseProduct(g);
    const double observationSlope=c.dy.dot(weight), mu=c.value*(b+observationSlope);
    const Vector3 gradMu=c.value*(b*c.dx+c.mixed*weight);
    NonstationaryMoments result;
    if constexpr (Near)
        result.meanF=birth.birthValue+meanDifference+b*amplitude.value+
            a*(b*std::expm1(c.logValue)+c.value*observationSlope);
    else
        result.meanF=target.value+a*mu;
    result.meanG=target.gradient+x.amplitudeGradient*mu+a*gradMu;
    NearGibbsCovariance covariance;
    bool haveResidual=false;
    if constexpr (Near)
        haveResidual=nearGibbsCovariance(y.length,y.lengthGradient,widthChange,widthRemainder,
            gradientChange,delta,direction,age,covariance);
    if (haveResidual) {
        result.varianceF=a*a*covariance.variance;
        result.regression=x.amplitudeGradient/a+covariance.regression;
        result.gradientGivenValueCovariance=(a*a)*covariance.gradientGivenValueCovariance;
        result.slopeVarianceGivenValue=(a*a)*covariance.slopeVarianceGivenValue;
    } else {
        if constexpr (Near) x.updateGradientVariance();
        const double r2=c.value*c.value;
        const Vector3 slope=inverse.cwiseProduct(c.dy);
        const double vf=1-r2*(1+c.dy.dot(slope));
        if (!(vf>0)) throw std::runtime_error("nonstationary conditional value variance lost precision");
        const Vector3 fg=-r2*(c.dx+c.mixed*slope);
        Matrix3 gg=-r2*(c.dx*c.dx.transpose()+c.mixed*inverse.asDiagonal()*c.mixed.transpose());
        gg.diagonal()+=x.gradientVariance;
        const Matrix3 given=a*a*(gg-fg*fg.transpose()/vf);
        result.varianceF=a*a*vf;
        result.regression=x.amplitudeGradient/a+fg/vf;
        result.gradientGivenValueCovariance=(given+given.transpose())*.5;
        result.slopeVarianceGivenValue=direction.dot(given*direction);
    }
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
        return conditioned<true>(*_kernel._parameters,_birth,_birthMean,m,age,difference);
    }
    return conditioned<false>(*_kernel._parameters,_birth,_birthMean,m,age,0.);
}
PointPrior NonstationaryFlight::at(double age) const {
    if (age==0.) return {_birth.birthValue,0.,_birth.birthGradient,Matrix3::Zero(),Vector3::Zero()};
    const auto m=moments(age);
    const Vector3 fg=m.varianceF*m.regression;
    return {m.meanF,m.varianceF,m.meanG,m.gradientGivenValueCovariance+fg*m.regression.transpose(),fg};
}
}
