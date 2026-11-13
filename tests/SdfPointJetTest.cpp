// Compile the implementation here to compare its private point and bound jets.
// This executable does not link core, so it defines the SDF functions only once.
#include "math/SdfFunctionsQueries.cpp"
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iomanip>

namespace Tungsten {
namespace {
bool identical(double a,double b) {
    return std::memcmp(&a,&b,sizeof(double))==0;
}
void comparePoint(const PointJet &point,const Jet &bound,const char *label) {
    bool same=identical(point.value,bound.value);
    for(int i=0;i<3;++i) same=same&&identical(point.gradient[i],bound.gradient[i]);
    if(!same || point.constant!=(bound.lip==0) || point.affine!=(bound.curvature==0)) {
        std::cerr<<std::setprecision(17)<<label<<"\npoint: "<<point.value<<" / "
                 <<point.gradient.transpose()<<" / "<<point.constant<<", "<<point.affine
                 <<"\nbound: "<<bound.value<<" / "<<bound.gradient.transpose()
                 <<" / "<<bound.lip<<", "<<bound.curvature<<'\n';
        throw std::runtime_error("point query differs from the radius-zero bound jet");
    }
}
struct Pair {
    PointJet point;
    Jet bound;
    explicit Pair(double v) : point(v),bound(v) {}
    Pair(double v,int axis) : point(v),bound(v) {
        point.gradient[axis]=bound.gradient[axis]=1;
        point.constant=false; bound.lip=1;
    }
};
void primitiveChecks() {
    const Pair zero(0),one(1);
    for(double value : {-1.0,-0.1,-1e-14,-1e-300,-0.0,0.0,1e-300,1e-14,0.1,1.0}) {
        const Pair x(value,0),y(0,1);
        comparePoint(norm(x.point,zero.point),norm(x.bound,zero.bound),"one active component / abs");
        comparePoint(norm(x.point,y.point),norm(x.bound,y.bound),"two variable components at the origin");
        const auto p=minimum(zero.point,x.point);
        const auto b=minimum(zero.bound,x.bound);
        comparePoint(p,b,"CSG constant/variable tie");
        comparePoint(norm(p,y.point),norm(b,y.bound),"zero gradient at a CSG seam is not constant");
        comparePoint(norm(x.point-x.point,one.point),norm(x.bound-x.bound,one.bound),
                     "cancelled affine input");
        comparePoint(maximum(x.point,-x.point),maximum(x.bound,-x.bound),"max tie");
    }
    Pair stationary(0);
    stationary.point.constant=stationary.point.affine=false;
    stationary.bound.lip=1; stationary.bound.curvature=2;
    const Pair y(0,1);
    comparePoint(norm(stationary.point,y.point),norm(stationary.bound,y.bound),
                 "stationary nonlinear input is not constant");
    for(double k : {0.02,0.1,1.0}) {
        for(double center : {-k,0.0,k}) {
            for(double value : {std::nextafter(center,-infinity),center,
                               std::nextafter(center,infinity),center-1e-12,center+1e-12}) {
                const Pair a(0,0),b(value,1);
                comparePoint(smoothUnion(a.point,b.point,k),smoothUnion(a.bound,b.bound,k),
                             "smooth union clamp and neighboring values");
            }
        }
        comparePoint(smoothUnion(zero.point,one.point,k),smoothUnion(zero.bound,one.bound,k),
                     "constant smooth union");
    }
}
void geometryChecks() {
    std::uint64_t state=0x92155a54d42d7c91ULL;
    const auto uniform=[&] {
        state^=state<<13; state^=state>>7; state^=state<<17;
        return double(state>>11)*0x1p-53;
    };
    std::size_t count=0;
    const auto compareAt=[&](const V &x) {
        try {
            comparePoint(queryPoint(x),query(x,0),"knob point");
            const auto exposed=SdfFunctions::evaluateJet(SdfFunctions::Function::Knob,x);
            const auto reference=query(x,0);
            if(!identical(exposed.value,reference.value) ||
               !(exposed.gradient.array()==reference.gradient.array()).all())
                throw std::runtime_error("public point query differs from reference");
        } catch(...) {
            std::cerr<<std::setprecision(17)<<"at "<<x.transpose()<<'\n'; throw;
        }
        ++count;
    };
    // Axes, cylinder caps, primitive centers and both sides of nearby seams.
    const double coordinates[]={-1.2,-0.9,-0.88,-0.82,-0.8,-0.775,-0.62,-0.6,
        -0.42,-0.4,-0.2,-0.16,-1e-14,-0.0,0.0,1e-14,0.16,0.2,0.4,0.6,0.8,0.88,1.2};
    for(double x:coordinates) for(double y:coordinates) for(double z:coordinates)
        compareAt(V(x,y,z));
    for(int i=0;i<20000;++i) {
        const double scale=i%3==0?1e-9:i%3==1?2.0:1e6;
        compareAt(V(2*uniform()-1,2*uniform()-1,2*uniform()-1)*scale);
    }
    const V caps[]={
        V(-0.72625440319482504,0.3802830746543806-0.8,0.48301416317699419),
        V(0.53464407933850722,0.38012398347045723-0.8,0.53863862453771816)};
    for(const auto &x:caps) {
        compareAt(x);
        for(int axis=0;axis<3;++axis) for(double offset:{-1e-12,1e-12}) {
            V near=x; near[axis]+=offset; compareAt(near);
        }
    }
    std::cout<<"point jet: primitive ties, zero norms, stationary points, smooth clamps; "
             <<count<<" knob points match full Jet values, gradients and branch flags exactly\n";
}

void rayReuseChecks() {
    const auto fn=SdfFunctions::Function::Knob;
    const V direction=V(0.4,-0.7,1.3).normalized();
    int count=0;
    for(const V &origin : {V::Zero().eval(),V(0.2,-0.62,0.1),V(-0.8,0,0),
                          V(0.6,-0.42,0),V(1e6,-2e6,3e6)}) {
        for(double begin : {0.0,1e-12,0.031,1.7}) {
            for(double length : {1e-12,1e-5,0.08,2.0}) {
                const double end=begin+length;
                const V midpoint=origin+(0.5*begin+0.5*end)*direction;
                const auto full=SdfFunctions::rayBounds(fn,origin,direction,begin,end);
                SdfRayQueryOptions options;
                options.computeBeginDerivative=false;
                const auto fast=SdfFunctions::rayBounds(fn,origin,direction,begin,end,options);
                if(!identical(full.minimumValue,fast.minimumValue) ||
                   !identical(full.maximumValue,fast.maximumValue) ||
                   !identical(full.minimumDerivative,fast.minimumDerivative) ||
                   !identical(full.maximumDerivative,fast.maximumDerivative) ||
                   !identical(full.maximumSecondDerivative,fast.maximumSecondDerivative))
                    throw std::runtime_error("optional beginning derivative changed ray certificate");
                if(full.beginDerivative!=queryPoint(origin+begin*direction).gradient.dot(direction))
                    throw std::runtime_error("requested beginning derivative changed");
                const double reference=queryPoint(midpoint).value;
                if(!(std::abs(fast.midpointValue-reference)<=fast.midpointError))
                    throw std::runtime_error("cached midpoint does not enclose point query");

                // Exercise the center displacement introduced by transforming
                // a world midpoint separately from a ray origin/direction.
                V supplied=midpoint;
                supplied.x()=std::nextafter(supplied.x(),infinity);
                options.midpoint=supplied;
                const auto shifted=SdfFunctions::rayBounds(fn,origin,direction,begin,end,options);
                if(!(std::abs(shifted.midpointValue-queryPoint(supplied).value)<=shifted.midpointError))
                    throw std::runtime_error("supplied midpoint was not preserved");
                for(int i=0;i<=8;++i) {
                    const auto point=queryPoint(origin+(begin+(end-begin)*i/8)*direction);
                    const double derivative=point.gradient.dot(direction);
                    if(!(point.value>=shifted.minimumValue && point.value<=shifted.maximumValue &&
                         derivative>=shifted.minimumDerivative && derivative<=shifted.maximumDerivative))
                        throw std::runtime_error("shifted midpoint invalidated ray certificate");
                }
                ++count;
            }
        }
    }
    std::cout<<"ray reuse: "<<count<<" finite-radius, seam, large-coordinate and shifted-midpoint certificates passed\n";
}

void roundedUGrooveChecks() {
    const auto fn=SdfFunctions::Function::RoundedUGroove;
    const auto check=[](bool ok,const char *label) { if(!ok) throw std::runtime_error(label); };
    // Known surface points and both sides of the circle/wall tangent join.
    for(double angle : {0.,0.3,0.9,1.57,2.4,3.141592653589793}) {
        const V p(std::cos(angle),-std::sin(angle),0.7);
        const auto j=SdfFunctions::evaluateJet(fn,p);
        check(std::abs(j.value)<1e-14,"U groove circular boundary");
        check((j.gradient+V(p.x(),p.y(),0)).norm()<1e-14,"U groove inward circular normal");
    }
    for(double sign : {-1.,1.}) {
        const auto a=SdfFunctions::evaluateJet(fn,V(sign,-1e-9,0));
        const auto b=SdfFunctions::evaluateJet(fn,V(sign,1e-9,0));
        check((a.gradient-b.gradient).norm()<2e-9,"U groove C1 tangent join");
        for(double delta : {1e-8,1e-12,1e-18}) {
            const double diff=SdfFunctions::valueDifference(fn,V(sign,0.4,0),V(sign*delta,0,0));
            check(std::abs(diff+delta)<1e-14*delta,"U groove stable sub-ulp wall difference");
        }
    }
    int certificates=0;
    for(const V &o : {V(1,0,0),V(-1,1e-12,0),V(0,-1,0),V(0,0.4,0),V(0.4,-0.2,0)}) {
        for(const V &w : {V(0.3,0.8,0.5),V(-0.7,-0.3,0),V(0,0,1)}) {
            for(double length : {1e-9,0.01,0.3,2.0}) {
                SdfRayQueryOptions options;
                options.midpoint=o+0.5*length*w+V(1e-15,-1e-15,0);
                const auto b=SdfFunctions::rayBounds(fn,o,w,0,length,options);
                const V lo=o.cwiseMin(o+length*w),hi=o.cwiseMax(o+length*w);
                const auto box=SdfFunctions::bounds(fn,lo,hi);
                for(int i=0;i<=100;++i) {
                    const V p=o+(length*i/100)*w;
                    const auto j=SdfFunctions::evaluateJet(fn,p);
                    const double derivative=j.gradient.dot(w);
                    check(j.value>=b.minimumValue && j.value<=b.maximumValue,"U groove ray value enclosure");
                    check(derivative>=b.minimumDerivative && derivative<=b.maximumDerivative,"U groove ray derivative enclosure");
                    check(j.value>=box.minimumValue && j.value<=box.maximumValue,"U groove box enclosure");
                    if(i>0 && std::isfinite(b.maximumSecondDerivative)) {
                        const auto prev=SdfFunctions::evaluateJet(fn,o+(length*(i-1)/100)*w);
                        check(std::abs((j.gradient-prev.gradient).dot(w))<=
                              b.maximumSecondDerivative*length/100+1e-12,"U groove curvature enclosure");
                    }
                }
                ++certificates;
            }
        }
    }
    std::cout<<"rounded U groove: boundary, normals, tiny differences and "<<certificates<<" certificates passed\n";
}

void twoSlabChecks() {
    const auto fn=SdfFunctions::Function::TwoSlabs;
    const auto check=[](bool ok,const char *label) { if(!ok) throw std::runtime_error(label); };
    const auto reference=[](const V &p) {
        const auto box=[](const V &q) {
            const V d=q.cwiseAbs()-V(.6,.01,.6);
            return d.maxCoeff();
        };
        return std::min(box(p-V(.18,.02,0)),box(p-V(-.18,-.02,0)));
    };
    for(double y : {-.03,-.01,.01,.03}) {
        const auto j=SdfFunctions::evaluateJet(fn,V(0,y,0));
        check(std::abs(j.value)<1e-16,"two slab face location");
        check(std::abs(j.gradient.y())==1,"two slab outward face normal");
    }
    for(double step : {1e-8,1e-12,1e-18})
        check(std::abs(SdfFunctions::valueDifference(fn,V(0,.03,0),V(0,step,0))-step)<1e-14*step,
              "two slab stable sub-ulp difference");
    int certificates=0;
    for(const V &o : {V(0,.3,0),V(0,0,0),V(.18,.02,0),V(.78,.03,.6),V(.5,0,.4),V(1e6,-2e6,3e6)}) {
        comparePoint(queryPoint(o,fn),query(o,0,V::Zero(),fn),"two slab point and bound jet");
        for(const V &w : {V(0,-1,0),V(.3,-.8,.5),V(0,0,1)}) {
            for(double length : {1e-9,.01,.08,.6,2.}) {
                const auto ray=SdfFunctions::rayBounds(fn,o,w,0,length);
                const auto box=SdfFunctions::bounds(fn,o.cwiseMin(o+length*w),o.cwiseMax(o+length*w));
                check(ray.certified && box.certified,"two slab certified bounds");
                for(int i=0;i<=100;++i) {
                    const V p=o+(length*i/100)*w;
                    const auto j=SdfFunctions::evaluateJet(fn,p);
                    check(std::abs(j.value-reference(p))<1e-9,"two slab independent union");
                    check(j.value>=ray.minimumValue && j.value<=ray.maximumValue,"two slab ray value bound");
                    check(j.gradient.dot(w)>=ray.minimumDerivative && j.gradient.dot(w)<=ray.maximumDerivative,
                          "two slab ray derivative bound");
                    check(j.value>=box.minimumValue && j.value<=box.maximumValue,"two slab box value bound");
                    check(j.gradient.norm()<=box.maximumGradientNorm+1e-12,"two slab gradient bound");
                }
                ++certificates;
            }
        }
    }
    std::cout<<"two slabs: independent box union, faces, sub-ulp differences and "
             <<certificates<<" certificates passed\n";
}

void twoSphereChecks() {
    const auto fn=SdfFunctions::Function::TwoSpheres;
    const auto check=[](bool ok,const char *label) { if(!ok) throw std::runtime_error(label); };
    const auto reference=[](const V &p) {
        return std::min((p-V(0,10,0)).norm(),(p-V(0,-10,0)).norm())-9.5;
    };
    check(SdfFunctions::supportsQueries(fn),"two spheres expose certified queries");
    for(double sign : {-1.,1.}) {
        for(const V &n : {V(1,0,0),V(0,1,0),V(0,-1,0),V(.6,.8,0)}) {
            const V p=V(0,10*sign,0)+9.5*n;
            const auto j=SdfFunctions::evaluateJet(fn,p);
            check(std::abs(j.value)<1e-13,"two sphere surface location");
            check((j.gradient-n).norm()<1e-13,"two sphere outward normal");
        }
        for(double d : {1e-8,1e-12,1e-18}) {
            const double difference=SdfFunctions::valueDifference(fn,V(0,19.5*sign,0),V(0,sign*d,0));
            check(std::abs(difference-d)<1e-14*d,"two sphere sub-ulp normal displacement");
        }
    }
    const V seam(2,0,3);
    const double tiny=1e-18;
    const double seamDifference=SdfFunctions::valueDifference(fn,seam,V(0,tiny,0));
    check(std::abs(seamDifference+10/std::sqrt(113.)*tiny)<1e-14*tiny,"two sphere tiny seam displacement");
    check(SdfFunctions::valueDifference(fn,V(2,1e-6,3),V(0,-2e-6,0))==0,"two sphere symmetric seam crossing");
    check(std::isinf(SdfFunctions::rayBounds(fn,V(0,-.1,0),V(0,1,0),0,.2).maximumSecondDerivative),
          "two sphere branch seam has no finite curvature certificate");
    int certificates=0;
    for(const V &o : {V(0,19.5,0),V(0,-19.5,0),V(6.5,0,0),V(0,10,0),V(0,0,0),V(1e6,-2e6,3e6)}) {
        comparePoint(queryPoint(o,fn),query(o,0,V::Zero(),fn),"two sphere point and bound jet");
        for(const V &w : {V(.3,.8,.5),V(-.7,-.3,0),V(0,0,1),V(0,1,0)}) {
            for(double length : {1e-9,.01,.3,2.,30.}) {
                SdfRayQueryOptions options;
                options.midpoint=o+.5*length*w+V(1e-15,-1e-15,0);
                const auto ray=SdfFunctions::rayBounds(fn,o,w,0,length,options);
                const auto box=SdfFunctions::bounds(fn,o.cwiseMin(o+length*w),o.cwiseMax(o+length*w));
                check(ray.certified && box.certified,"two sphere certificates enabled");
                for(int i=0;i<=100;++i) {
                    const V p=o+(length*i/100)*w;
                    const auto j=SdfFunctions::evaluateJet(fn,p);
                    const double derivative=j.gradient.dot(w);
                    check(std::abs(j.value-reference(p))<1e-9,"two sphere agreement with independent union");
                    check(j.value>=ray.minimumValue && j.value<=ray.maximumValue,"two sphere ray value enclosure");
                    check(derivative>=ray.minimumDerivative && derivative<=ray.maximumDerivative,"two sphere directional derivative enclosure");
                    check(j.value>=box.minimumValue && j.value<=box.maximumValue,"two sphere box value enclosure");
                    check(j.gradient.norm()<=box.maximumGradientNorm+1e-12,"two sphere box gradient enclosure");
                    if(i>0 && std::isfinite(ray.maximumSecondDerivative)) {
                        const auto previous=SdfFunctions::evaluateJet(fn,o+(length*(i-1)/100)*w);
                        check(std::abs((j.gradient-previous.gradient).dot(w))<=
                              ray.maximumSecondDerivative*length/100+1e-11,"two sphere curvature enclosure");
                    }
                }
                ++certificates;
            }
        }
    }
    std::cout<<"two spheres: independent union, outward normals, sub-ulp differences, seam and "
             <<certificates<<" certificates passed\n";
}
}
}
int main() {
    try {
        Tungsten::primitiveChecks();
        Tungsten::geometryChecks();
        Tungsten::rayReuseChecks();
        Tungsten::roundedUGrooveChecks();
        Tungsten::twoSphereChecks();
        Tungsten::twoSlabChecks();
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<'\n'; return 1;
    }
}
