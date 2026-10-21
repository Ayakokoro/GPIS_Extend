#include "SdfFunctions.hpp"
#include "math/ConditionalProfile.hpp"
#include "math/Mat4f.hpp"
#include "math/Angle.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace Tungsten {
namespace {
using V = Eigen::Vector3d;
constexpr double infinity = std::numeric_limits<double>::infinity();
double guard(double v) { return 128*std::numeric_limits<double>::epsilon()*(1+std::abs(v)); }

double minimum(double a,double b) { return std::min(a,b); }
double maximum(double a,double b) { return std::max(a,b); }
double norm(double a,double b,double c=0) { return std::hypot(a,b,c); }
double smoothUnion(double a,double b,double k) {
    const double h=std::clamp(0.5+0.5*(b-a)/k,0.0,1.0);
    return b+(a-b)*h-k*h*(1-h);
}

// Bounds on a ball: |grad f| <= lip and ||Hessian f|| <= curvature.
// A hard CSG branch change has infinite curvature, but retains a finite
// Lipschitz bound. The birth tracker shrinks until its active branch is smooth.
struct Jet {
    double value, lip=0, curvature=0, radius=0;
    double derivative=0, minimumDerivative=0, maximumDerivative=0, speed=0;
    V gradient=V::Zero();
    Jet(double v=0) : value(v) {}
    double error() const { return lip*radius+guard(value); }
};
Jet operator+(const Jet &a,const Jet &b) {
    Jet c(a.value+b.value); c.gradient=a.gradient+b.gradient;
    c.lip=a.lip+b.lip; c.curvature=a.curvature+b.curvature;
    c.radius=std::max(a.radius,b.radius); c.speed=std::max(a.speed,b.speed);
    c.derivative=a.derivative+b.derivative;
    c.minimumDerivative=a.minimumDerivative+b.minimumDerivative;
    c.maximumDerivative=a.maximumDerivative+b.maximumDerivative; return c;
}
Jet operator*(const Jet &a,double s) {
    if(s==0) return Jet(0);
    Jet c=a; c.value*=s; c.gradient*=s;
    c.lip*=std::abs(s); c.curvature*=std::abs(s); c.derivative*=s;
    c.minimumDerivative=std::min(s*a.minimumDerivative,s*a.maximumDerivative);
    c.maximumDerivative=std::max(s*a.minimumDerivative,s*a.maximumDerivative); return c;
}
Jet operator-(const Jet &a) { return a*(-1); }
Jet operator-(const Jet &a,const Jet &b) { return a+(-b); }
Jet minimum(const Jet &a,const Jet &b) {
    if(a.lip==0 && b.lip==0) return Jet(std::min(a.value,b.value));
    if(a.value+a.error()<b.value-b.error()) return a;
    if(b.value+b.error()<a.value-a.error()) return b;
    Jet c=a.value<=b.value?a:b;
    c.lip=std::max(a.lip,b.lip); c.radius=std::max(a.radius,b.radius);
    c.speed=std::max(a.speed,b.speed);
    c.minimumDerivative=std::min(a.minimumDerivative,b.minimumDerivative);
    c.maximumDerivative=std::max(a.maximumDerivative,b.maximumDerivative);
    c.curvature=infinity; return c;
}
Jet maximum(const Jet &a,const Jet &b) { return -minimum(-a,-b); }
void tightenDerivative(Jet &c) {
    if(std::isfinite(c.curvature)) {
        const double error=c.curvature*c.radius*c.speed+guard(c.derivative);
        c.minimumDerivative=std::max(c.minimumDerivative,c.derivative-error);
        c.maximumDerivative=std::min(c.maximumDerivative,c.derivative+error);
    }
}
Jet norm(const Jet &a,const Jet &b,const Jet &c=Jet(0)) {
    const auto zero=[](const Jet &v) { return v.value==0 && v.lip==0; };
    // Outside a cylinder cap the norm often has only one active component:
    // sqrt(y*y)=abs(y). Its smooth branch has the child's curvature, not
    // 1/abs(y), which would make the birth envelope unusable near the cap.
    if(zero(b)&&zero(c)) return maximum(a,-a);
    if(zero(a)&&zero(c)) return maximum(b,-b);
    if(zero(a)&&zero(b)) return maximum(c,-c);
    const std::array<Jet,3> q{a,b,c};
    Jet result(std::hypot(a.value,b.value,c.value));
    for(const auto &v:q) {
        result.lip+=v.lip*v.lip;
        result.radius=std::max(result.radius,v.radius);
        result.speed=std::max(result.speed,v.speed);
        if(result.value>0) {
            result.gradient+=v.gradient*(v.value/result.value);
            result.derivative+=v.derivative*(v.value/result.value);
        }
    }
    result.lip=std::sqrt(result.lip);
    if(a.curvature==0 && b.curvature==0 && c.curvature==0) {
        // Affine inputs: Gershgorin bounds the largest eigenvalue of J J^T.
        Eigen::Matrix3d j;
        for(int i=0;i<3;++i) j.row(i)=q[i].gradient.transpose();
        result.lip=std::sqrt((j*j.transpose()).cwiseAbs().rowwise().sum().maxCoeff())*(1+1e-14);
    }
    const double lower=result.value-result.error();
    if(result.lip==0) result.curvature=0;
    else result.curvature=lower>0 ?
        (a.curvature+b.curvature+c.curvature+result.lip*result.lip/lower)*(1+1e-14) : infinity;
    result.minimumDerivative=-result.lip*result.speed*(1+1e-12);
    result.maximumDerivative=-result.minimumDerivative;
    tightenDerivative(result);
    return result;
}
Jet smoothUnion(const Jet &a,const Jet &b,double k) {
    const double difference=b.value-a.value, error=a.error()+b.error();
    if(difference-error>=k) return a;
    if(difference+error<=-k) return b;
    const double h=std::clamp(0.5+0.5*difference/k,0.0,1.0);
    Jet c(b.value+(a.value-b.value)*h-k*h*(1-h));
    c.gradient=h*a.gradient+(1-h)*b.gradient;
    c.radius=std::max(a.radius,b.radius); c.lip=std::max(a.lip,b.lip);
    c.speed=std::max(a.speed,b.speed); c.derivative=h*a.derivative+(1-h)*b.derivative;
    c.minimumDerivative=std::min(a.minimumDerivative,b.minimumDerivative);
    c.maximumDerivative=std::max(a.maximumDerivative,b.maximumDerivative);
    // The blending gradient is a convex combination. Its Hessian adds
    // -(grad a-grad b)(grad a-grad b)^T/(2k), including across clamp endpoints.
    c.curvature=std::max(a.curvature,b.curvature)+std::pow(a.lip+b.lip,2)/(2*k);
    tightenDerivative(c);
    return c;
}

// Point queries propagate only value and gradient. These two structural flags
// preserve the full Jet's zero-Lipschitz/zero-curvature cases without computing
// either bound. A zero gradient at a stationary point is not a constant field.
struct PointJet {
    double value;
    V gradient=V::Zero();
    bool constant=true, affine=true;
    PointJet(double v=0) : value(v) {}
};
PointJet operator+(const PointJet &a,const PointJet &b) {
    PointJet c(a.value+b.value); c.gradient=a.gradient+b.gradient;
    c.constant=a.constant&&b.constant; c.affine=a.affine&&b.affine;
    return c;
}
PointJet operator*(const PointJet &a,double s) {
    if(s==0) return PointJet(0);
    PointJet c=a; c.value*=s; c.gradient*=s; return c;
}
PointJet operator-(const PointJet &a) { return a*(-1); }
PointJet operator-(const PointJet &a,const PointJet &b) { return a+(-b); }
PointJet minimum(const PointJet &a,const PointJet &b) {
    if(a.constant&&b.constant) return PointJet(std::min(a.value,b.value));
    // At radius zero the full Jet's error is just this roundoff guard.
    // Retain its ambiguous-branch metadata as well as its tie-breaking rule.
    if(a.value+guard(a.value)<b.value-guard(b.value)) return a;
    if(b.value+guard(b.value)<a.value-guard(a.value)) return b;
    PointJet c=a.value<=b.value?a:b;
    c.constant=false; c.affine=false; return c;
}
PointJet maximum(const PointJet &a,const PointJet &b) { return -minimum(-a,-b); }
PointJet norm(const PointJet &a,const PointJet &b,const PointJet &c=PointJet(0)) {
    const auto zero=[](const PointJet &v) { return v.value==0&&v.constant; };
    if(zero(b)&&zero(c)) return maximum(a,-a);
    if(zero(a)&&zero(c)) return maximum(b,-b);
    if(zero(a)&&zero(b)) return maximum(c,-c);
    const std::array<PointJet,3> q{a,b,c};
    PointJet result(std::hypot(a.value,b.value,c.value));
    // Keep the original accumulation order, including its convention at zero.
    if(result.value>0)
        for(const auto &v:q) result.gradient+=v.gradient*(v.value/result.value);
    result.constant=a.constant&&b.constant&&c.constant;
    if(a.affine&&b.affine&&c.affine) {
        // The full Jet's affine Jacobian bound is zero exactly when every
        // input gradient is zero. No matrix product is needed to recognize it.
        result.constant=a.gradient.isZero(0)&&b.gradient.isZero(0)&&c.gradient.isZero(0);
    }
    result.affine=result.constant;
    return result;
}
PointJet smoothUnion(const PointJet &a,const PointJet &b,double k) {
    const double difference=b.value-a.value, error=guard(a.value)+guard(b.value);
    if(difference-error>=k) return a;
    if(difference+error<=-k) return b;
    const double h=std::clamp(0.5+0.5*difference/k,0.0,1.0);
    PointJet c(b.value+(a.value-b.value)*h-k*h*(1-h));
    c.gradient=h*a.gradient+(1-h)*b.gradient;
    c.constant=a.constant&&b.constant; c.affine=c.constant;
    return c;
}

// A value and its finite displacement, propagated algebraically. Subtracting
// two final SDF values loses the mean change on very short conditional flights.
struct Difference {
    double value, delta=0;
    Difference(double v=0) : value(v) {}
};
Difference operator+(const Difference &a,const Difference &b) {
    Difference c(a.value+b.value); c.delta=a.delta+b.delta; return c;
}
Difference operator*(const Difference &a,double s) {
    Difference c(a.value*s); c.delta=a.delta*s; return c;
}
Difference operator-(const Difference &a) { return a*(-1); }
Difference operator-(const Difference &a,const Difference &b) { return a+(-b); }
Difference operator*(const Difference &a,const Difference &b) {
    Difference c(a.value*b.value);
    c.delta=a.delta*b.value+a.value*b.delta+a.delta*b.delta; return c;
}
Difference minimum(const Difference &a,const Difference &b) {
    const bool first=a.value<=b.value, last=a.value+a.delta<=b.value+b.delta;
    Difference c(first?a.value:b.value);
    if(first==last) c.delta=first?a.delta:b.delta;
    else c.delta=last?(a.value-b.value)+a.delta:(b.value-a.value)+b.delta;
    return c;
}
Difference maximum(const Difference &a,const Difference &b) { return -minimum(-a,-b); }
Difference norm(const Difference &a,const Difference &b,const Difference &c=Difference(0)) {
    Difference result(std::hypot(a.value,b.value,c.value));
    const double last=std::hypot(a.value+a.delta,b.value+b.delta,c.value+c.delta);
    if(result.value+last>0)
        result.delta=(a.delta*(2*a.value+a.delta)+b.delta*(2*b.value+b.delta)+
                      c.delta*(2*c.value+c.delta))/(result.value+last);
    return result;
}
Difference smoothUnion(const Difference &a,const Difference &b,double k) {
    const Difference h=minimum(maximum(Difference(0.5)+(b-a)*(0.5/k),Difference(0)),Difference(1));
    return b+(a-b)*h-h*(Difference(1)-h)*k;
}
template<class T> T subtract(const T &a,const T &b,double k) { return -smoothUnion(a,-b,k); }
template<class T> using P = std::array<T,3>;
template<class T> P<T> shifted(P<T> p,double x,double y,double z) {
    return {p[0]+T(x),p[1]+T(y),p[2]+T(z)};
}
template<class T> P<T> rotated(const P<T> &p,const Mat4f &m) {
    return {p[0]*m(0,0)+p[1]*m(0,1)+p[2]*m(0,2),
            p[0]*m(1,0)+p[1]*m(1,1)+p[2]*m(1,2),
            p[0]*m(2,0)+p[1]*m(2,1)+p[2]*m(2,2)};
}
template<class T> T sphere(const P<T> &p,double r) { return norm(p[0],p[1],p[2])-T(r); }
template<class T> T cone(const P<T> &p) { return (norm(p[0],p[1])+p[2])*double(PI/3.f); }
template<class T> T cylinder(const P<T> &p,double h,double r) {
    const T x=norm(p[0],p[2])-T(h), y=maximum(p[1],-p[1])-T(r);
    return minimum(maximum(x,y),T(0))+norm(maximum(x,T(0)),maximum(y,T(0)));
}
template<class T> T prism(const P<T> &p) {
    return maximum(maximum(p[2],-p[2])-T(0.3f),
        maximum(maximum(p[0],-p[0])*double(0.866025f)+p[1]*0.5,-p[1])-T(1.2f*0.5f));
}
// One CSG definition for the public float API and all differential queries.
// Shape formula attribution: Towaki Takikawa (2020), NVIDIA CORPORATION (2021),
// MIT license reproduced in SdfFunctions.cpp.
template<class T> T knobGeometry(P<T> p,SdfFunctions::Function variant=SdfFunctions::Function::Knob) {
    static const Mat4f plus90=Mat4f::rotAxis(Vec3f(1.0f,0.0f,0.0f),90);
    static const Mat4f minus90=Mat4f::rotAxis(Vec3f(1.0f,0.0f,0.0f),-90);
    static const Mat4f minus45=Mat4f::rotAxis(Vec3f(1.0f,0.0f,0.0f),-45);
    const double size=double(0.8f);
    for(auto &v:p) v=v*double(float(1.0/size));
    if(variant==SdfFunctions::Function::KnobInner) return sphere(p,0.75f)*size;
    const T outer=sphere(p,1.0);
    const T cutout=sphere(shifted(p,0,-double(0.5f),-double(0.5f)),double(0.7f));
    const auto torus=rotated(shifted(p,0,-double(0.2f),-double(0.2f)),minus45);
    const T etch=norm(norm(torus[0],torus[2])-T(1.0f),torus[1])-T(0.05f);
    T d=subtract(cutout,outer,double(0.1f));
    const T inner=sphere(p,0.75f);
    d=variant==SdfFunctions::Function::KnobOuter ? maximum(d,-inner) : minimum(d,inner);
    d=maximum(-etch,d);
    const auto b=shifted(p,0,double(0.775f),0);
    T base=smoothUnion(cone(rotated(shifted(b,0,double(0.9f),0),minus90)),
        cone(rotated(shifted(b,0,-double(0.9f),0),plus90)),double(0.02f));
    base=maximum(base,cylinder(b,double(1.1f),double(0.25f)))*double(0.7f);
    base=maximum(-cylinder(b,double(0.6f),double(0.3f)),base);
    base=maximum(-prism(rotated(shifted(b,0,0,-1.0),plus90)),base);
    return minimum(subtract(outer,base,double(0.1f)),d)*size;
}
Jet query(const V &local,double radius,const V &w=V::Zero()) {
    P<Jet> p;
    for(int i=0;i<3;++i) {
        p[i]=Jet(local[i]); p[i].gradient[i]=1; p[i].lip=1; p[i].radius=radius;
        p[i].derivative=p[i].minimumDerivative=p[i].maximumDerivative=w[i];
        p[i].speed=w.norm();
    }
    return knobGeometry(p);
}
PointJet queryPoint(const V &local) {
    P<PointJet> p;
    for(int i=0;i<3;++i) {
        p[i]=PointJet(local[i]); p[i].gradient[i]=1; p[i].constant=false;
    }
    return knobGeometry(p);
}
void requireQueries(SdfFunctions::Function fn) {
    if(!SdfFunctions::supportsQueries(fn))
        throw std::invalid_argument("procedural SDF does not provide certified differential queries");
}
}

float SdfFunctions::knob(Vec3f p,int &) {
    return float(knobGeometry(P<double>{p.x(),p.y(),p.z()}));
}
float SdfFunctions::knob_inner(Vec3f p,int &) {
    return float(knobGeometry(P<double>{p.x(),p.y(),p.z()},Function::KnobInner));
}
float SdfFunctions::knob_outer(Vec3f p,int &) {
    return float(knobGeometry(P<double>{p.x(),p.y(),p.z()},Function::KnobOuter));
}
bool SdfFunctions::supportsQueries(Function fn) { return fn==Function::Knob; }
SdfJet SdfFunctions::evaluateJet(Function fn,const V &x) {
    ConditionalGpis::Profile::Scope profile(ConditionalGpis::Profile::Stage::SdfJet);
    requireQueries(fn);
    const auto q=queryPoint(x);
    return {q.value,q.gradient};
}
double SdfFunctions::valueDifference(Function fn,const V &x,const V &d) {
    ConditionalGpis::Profile::Scope profile(ConditionalGpis::Profile::Stage::SdfDifference);
    requireQueries(fn);
    P<Difference> inputs;
    for(int i=0;i<3;++i) { inputs[i]=Difference(x[i]); inputs[i].delta=d[i]; }
    return knobGeometry(inputs).delta;
}
SdfBounds SdfFunctions::bounds(Function fn,const V &lo,const V &hi) {
    ConditionalGpis::Profile::Scope profile(ConditionalGpis::Profile::Stage::SdfBounds);
    requireQueries(fn);
    const auto q=query(0.5*lo+0.5*hi,0.5*(hi-lo).norm()*(1+1e-12));
    return {q.value-q.error(),q.value+q.error(),q.lip*(1+1e-12),true};
}
SdfRayBounds SdfFunctions::rayBounds(Function fn,const V &o,const V &w,double lo,double hi,
                                    const SdfRayQueryOptions &options) {
    ConditionalGpis::Profile::Scope profile(ConditionalGpis::Profile::Stage::SdfRayBounds);
    requireQueries(fn);
    const V rayMidpoint=o+(0.5*lo+0.5*hi)*w;
    const V midpoint=options.midpoint.value_or(rayMidpoint);
    // A transformed midpoint can differ from a reconstructed local-ray
    // midpoint by roundoff. Cover both when constructing the certificate.
    const double shift=(midpoint-rayMidpoint).norm(),speed=w.norm();
    const double radius=(0.5*(hi-lo)*speed+shift)*(1+1e-12);
    const auto q=query(midpoint,radius,w);
    SdfRayBounds r;
    // Keep directional enclosures through hard min/max branches. Using only
    // +/-|grad| after a nearby CSG seam is unnecessarily loose for grazing
    // flights, whose conditional value variance grows like distance^4.
    r.minimumDerivative=q.minimumDerivative-guard(q.lip*speed);
    r.maximumDerivative=q.maximumDerivative+guard(q.lip*speed);
    const double delta=std::max(std::abs(r.minimumDerivative),std::abs(r.maximumDerivative))*0.5*(hi-lo)+
        q.lip*shift+guard(q.value);
    const double valueError=std::min(q.error(),delta);
    r.minimumValue=q.value-valueError; r.maximumValue=q.value+valueError;
    r.maximumSecondDerivative=q.curvature*speed*speed*(1+1e-12);
    if(options.computeBeginDerivative)
        r.beginDerivative=evaluateJet(fn,o+lo*w).gradient.dot(w);
    r.midpointValue=q.value;
    // A finite-radius Jet can take a different smooth-union shortcut than a
    // PointJet. Enclose that evaluation roundoff rather than treating the
    // reused value as bit-identical to a separate point evaluation.
    r.midpointError=guard(std::abs(q.value)+midpoint.norm());
    r.certified=true;
    return r;
}
}
