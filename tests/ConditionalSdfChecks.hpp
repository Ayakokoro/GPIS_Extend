#pragma once
#include "media/GpisSdfGrid.hpp"
#include "ConditionalGridCursorChecks.hpp"
#include "ConditionalMacrocellChecks.hpp"

inline void sdfChecks()
{
#if OPENVDB_AVAILABLE
    using namespace Tungsten;
    auto grid=std::make_shared<GpisSdfGrid>();
    grid->data.dimensions=Vec3i(6); grid->data.origin=Vec3i(0);
    grid->data.clampMin=Vec3f(0.25f); grid->data.clampMax=Vec3f(4.75f);
    grid->scale=-0.3;grid->offset=0.2;
    grid->worldToIndex=M::Identity();
    grid->worldToIndex(0,0)=1.2;grid->worldToIndex(0,1)=0.1;
    grid->worldToIndex(2,2)=0.8;
    grid->translation=V(2,2,2);
    grid->gradientBound=20;
    for(int z=0;z<6;++z) for(int y=0;y<6;++y) for(int x=0;x<6;++x)
        grid->data.values.push_back(float(0.1*x*y*z+0.03*x*x-0.2*y+0.3*z));
    grid->initializeMagnitudeBounds();
    const auto global=grid->globalMagnitudeBounds();
    cg::Random globalRng(926);
    for(int i=0;i<3000;++i) {
        const V p(20*globalRng.uniform01()-10,20*globalRng.uniform01()-10,20*globalRng.uniform01()-10);
        const auto jet=grid->evaluate(p);
        check(std::abs(jet.value)<=global->maximumAbsoluteValue && jet.gradient.norm()<=global->maximumGradientNorm,
            "cached global bounds cover transformed negative-scale field, clamp and exterior");
    }
    auto nearlyConstant=*grid;
    for(auto &node:nearlyConstant.data.values) node=1e20f;
    nearlyConstant.initializeMagnitudeBounds();
    const auto constantBounds=nearlyConstant.globalMagnitudeBounds();
    for(int i=0;i<100;++i) {
        const V p(globalRng.uniform01(),globalRng.uniform01(),globalRng.uniform01());
        const auto jet=nearlyConstant.evaluate(p);
        check(jet.gradient.norm()<=constantBounds->maximumGradientNorm,
            "global gradient certificate includes roundoff in cancelling node contributions");
    }
    gridCursorChecks(grid);
    macrocellChecks(grid);
    for(const V &o : {V(-0.4,-0.2,-0.5),V(-3,-1,-2),V(2.1,0.1,0.4)}) {
        const V w=V(0.4,0.2,0.3).normalized();
        std::vector<double> knots{0,3};
        grid->appendRayBreakpoints(o,w,0,3,knots);
        std::sort(knots.begin(),knots.end());
        for(size_t i=1;i<knots.size();++i) {
            const double lo=knots[i-1],hi=knots[i];
            if(!(hi>lo)) continue;
            // Avoid asking a cell certificate across a representationally
            // ambiguous knot; coverage of complete segments is tested below.
            const double a=lo+1e-9*(hi-lo),b=hi-1e-9*(hi-lo);
            const auto bounds=grid->rayBounds(o,w,a,b);
            check(std::isfinite(bounds.maximumSecondDerivative),"single-cell SDF curvature certificate");
            close(bounds.midpointValue,grid->evaluate(o+(0.5*a+0.5*b)*w).value,0,
                "cached grid midpoint matches the ordinary point query exactly");
            close(bounds.midpointError,0,0,"cached grid midpoint needs no extra error allowance");
            SdfRayQueryOptions options;
            options.computeBeginDerivative=false;
            const auto fast=grid->rayBounds(o,w,a,b,options);
            check(fast.minimumValue==bounds.minimumValue && fast.maximumValue==bounds.maximumValue &&
                fast.minimumDerivative==bounds.minimumDerivative && fast.maximumDerivative==bounds.maximumDerivative &&
                fast.maximumSecondDerivative==bounds.maximumSecondDerivative,
                "optional beginning derivative preserves grid certificates");
            options.midpoint=o+(0.75*a+0.25*b)*w;
            close(grid->rayBounds(o,w,a,b,options).midpointValue,
                grid->evaluate(*options.midpoint).value,0,"grid cache honors the supplied world midpoint");
            for (double t : {a,b}) {
                const auto endpoint=grid->evaluate(o+t*w);
                check(endpoint.value>=bounds.minimumValue && endpoint.value<=bounds.maximumValue &&
                    endpoint.gradient.dot(w)>=bounds.minimumDerivative &&
                    endpoint.gradient.dot(w)<=bounds.maximumDerivative,"grid bounds include both endpoint queries");
            }
            for(int j=0;j<40;++j) {
                const double t=a+(b-a)*(j+0.5)/40;
                const auto jet=grid->evaluate(o+t*w);
                check(jet.value>=bounds.minimumValue && jet.value<=bounds.maximumValue,"trilinear value bound");
                if(!(jet.gradient.dot(w)>=bounds.minimumDerivative && jet.gradient.dot(w)<=bounds.maximumDerivative)) {
                    std::cerr.precision(17);
                    std::cerr<<"SDF bound origin "<<o.transpose()<<" t "<<t<<" interval "<<a<<" "<<b
                        <<" derivative "<<jet.gradient.dot(w)<<" bounds "<<bounds.minimumDerivative<<" "<<bounds.maximumDerivative<<"\n";
                    check(false,"trilinear directional bound");
                }
            }
            const double mid=0.5*(a+b),eps=std::min(1e-5,0.1*(b-a));
            const double curvature=std::abs((grid->evaluate(o+(mid+eps)*w).gradient-
                    grid->evaluate(o+(mid-eps)*w).gradient).dot(w)/(2*eps));
            check(curvature<=bounds.maximumSecondDerivative+1e-8,"trilinear second derivative bound");
        }
        const auto whole=grid->rayBounds(o,w,0,3);
        close(whole.midpointValue,grid->evaluate(o+1.5*w).value,0,"multi-cell cached midpoint");
        for(int j=0;j<100;++j) {
            const auto jet=grid->evaluate(o+3*(j+0.5)/100*w);
            check(jet.value>=whole.minimumValue && jet.value<=whole.maximumValue,"SDF clamp and multi-cell coverage");
        }
    }
    // No representable interior: the midpoint selects the cell on the other
    // side of the face from the left endpoint. Its derivative must be enclosed.
    auto face=*grid;
    face.worldToIndex=M::Identity();face.translation=V::Zero();
    const V faceOrigin(0,1.5,1.5);
    const double beforeFace=std::nextafter(1.0,0.0);
    const auto adjacent=face.rayBounds(faceOrigin,V::UnitX(),beforeFace,1);
    for(double t : {beforeFace,1.0}) {
        const auto point=face.evaluate(faceOrigin+t*V::UnitX());
        check(point.value>=adjacent.minimumValue && point.value<=adjacent.maximumValue &&
            point.gradient.x()>=adjacent.minimumDerivative && point.gradient.x()<=adjacent.maximumDerivative,
            "adjacent-double cell crossing keeps neighboring endpoint derivatives");
    }
    close(adjacent.midpointValue,face.evaluate(faceOrigin+(0.5*beforeFace+0.5)*V::UnitX()).value,0,
        "rounded cell-face midpoint matches point query");
    SdfRayQueryOptions outside;
    outside.midpoint=V(3.2,2.4,1.8);
    close(face.rayBounds(faceOrigin,V::UnitX(),0.5,0.75,outside).midpointValue,
        face.evaluate(*outside.midpoint).value,0,"midpoint outside the visited cell uses point-query fallback");
    auto shifted=*grid;
    shifted.data.origin+=Vec3i(-4,3,7);
    shifted.data.clampMin+=Vec3f(-4.0f,3.0f,7.0f);shifted.data.clampMax+=Vec3f(-4.0f,3.0f,7.0f);
    shifted.translation+=V(-4,3,7);
    const V shiftedOrigin(-0.4,-0.2,-0.5),shiftedDirection=V(0.4,0.2,0.3).normalized();
    close(shifted.rayBounds(shiftedOrigin,shiftedDirection,0,3).midpointValue,
        shifted.evaluate(shiftedOrigin+1.5*shiftedDirection).value,0,"cached midpoint handles nonzero grid origins");
    const V x(0.1,0.2,0.3),d=1e-10*V(0.5,-0.3,0.2);
    close(grid->valueDifference(x,d),grid->evaluate(x).gradient.dot(d),1e-18,"stable SDF value difference");
    cg::GPSSField f{grid,cg::CovarianceKernel(0.4,M::Identity()),{V::Constant(-100),V::Constant(100)}};
    cg::FlightState birth;birth.birthPosition=x;birth.birthGradient=V(0.2,0.1,1.0);birth.direction=V(0,0,1);
    // Include multi-cell and clamped intervals as well as individual cells:
    // both screening paths must cover the actual trilinear conditional hazard.
    for(const V &w:std::array<V,3>{{V(0,0,1),V(0.4,-0.3,0.8).normalized(),V(-1,0,0)}}) {
        auto observation=birth;observation.direction=w;observation.birthValue=0.1;
        const cg::ConditionalFlightKernel kernel(f,observation);
        const auto constantBirth=kernel.birthMajorant(3);
        const double constantTail=kernel.constantTailMajorant(constantBirth.split);
        for(int j=0;j<=200;++j) {
            const double t=constantBirth.split*std::pow(3/constantBirth.split,double(j)/200);
            check(kernel.evaluate(t).hazard.value<=constantTail*(1+1e-10),
                "global tail covers transformed grid cells and clamped exterior");
        }
        for(double requested : {0.6,1.5,4.0}) {
            cg::ConditionalTrackingSettings options;
            options.majorantMode=cg::ConditionalMajorantMode::TwoConstants;
            options.constantBirthLength=requested;
            const cg::ConditionalSegment extended(f,observation,x,w,3,options);
            const double split=std::min(requested,3.0);
            const double first=extended.majorant(0,split);
            const double last=split<3 ? extended.majorant(split,3):first;
            for(int i=1;i<=600;++i) {
                const double t=3.0*i/600;
                check(extended.hazard(t)<=(t<=split ? first:last)*(1+1e-10),
                    "extended prefix crosses nonlinear grid cells and clamps conservatively");
            }
        }
        for(const auto &span:std::array<std::pair<double,double>,3>{{{0.1,0.2},{0.2,1.5},{1.5,3.0}}}) {
            const double cheap=kernel.cheapIntervalMajorant(span.first,span.second);
            const double fine=kernel.intervalMajorant(span.first,span.second);
            const double adaptive=kernel.intervalMajorant(span.first,span.second,4);
            check(fine<=adaptive && adaptive<=cheap,"grid adaptive majorant stays within cheap/fine certificates");
            for(int i=0;i<=100;++i) {
                const double t=i==100 ? span.second : span.first+(span.second-span.first)*i/100;
                check(kernel.evaluate(t).hazard.value<=fine*(1+1e-10),
                    "cheap and fine majorants cover transformed grid cells and clamp boundaries");
            }
        }
    }
    cg::ConditionalSegment flight(f,birth,x,birth.direction,2);
    cg::Random rng(935);
    for(int i=0;i<200;++i) { double t;flight.track(rng,true,t); }
    std::cout<<"SDF: transformed nonlinear trilinear cells, cached midpoints, cell-face endpoints, "
        "clamp, curvature and conditional tracking passed\n";
#endif
}
