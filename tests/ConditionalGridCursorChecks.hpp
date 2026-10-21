#pragma once

#if OPENVDB_AVAILABLE
class UncachedGridMean final : public cg::MeanField {
    std::shared_ptr<const Tungsten::GpisSdfGrid> _grid;
public:
    explicit UncachedGridMean(std::shared_ptr<const Tungsten::GpisSdfGrid> grid):_grid(std::move(grid)) {}
    const char *typeName() const override { return "uncached-grid-reference"; }
    cg::MeanJet evaluate(const V &p) const override { return _grid->evaluate(p); }
    double valueDifference(const V &p,const V &d) const override { return _grid->valueDifference(p,d); }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override { return _grid->bounds(b); }
    cg::MeanRayBounds rayBounds(const V &o,const V &w,double lo,double hi,
            const Tungsten::SdfRayQueryOptions &options = {}) const override {
        return _grid->rayBounds(o,w,lo,hi,options);
    }
    void appendRayBreakpoints(const V &o,const V &w,double lo,double hi,std::vector<double> &out) const override {
        _grid->appendRayBreakpoints(o,w,lo,hi,out);
    }
};

inline void gridCursorChecks(const std::shared_ptr<Tungsten::GpisSdfGrid> &grid)
{
    using namespace Tungsten;
    int intervals=0,points=0;
    for(const V &o:{V(-0.4,-0.2,-0.5),V(-3,-1,-2),V(2.1,0.1,0.4),V::Zero().eval()})
        for(const V &w:{V(0.4,0.2,0.3).normalized().eval(),V(-0.7,0.2,-0.1),
                       V(0,0,-1),V(1,0,0),V::Zero().eval()}) {
            auto cursor=grid->createRayCursor(o,w);
            std::vector<double> knots{0,3};
            grid->appendRayBreakpoints(o,w,0,3,knots);
            std::sort(knots.begin(),knots.end());
            knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
            for(size_t i=1;i<knots.size();++i) {
                const double lo=knots[i-1],hi=knots[i];
                close(cursor->nextBoundary(lo,3),hi,0,"cursor agrees with independent grid crossings");
                const auto b=cursor->rayBounds(lo,hi);
                check(b.certified,"cursor returns a certified interval");
                const double midpoint=0.5*lo+0.5*hi;
                close(b.midpointValue,grid->evaluate(o+midpoint*w).value,b.midpointError+1e-14,
                    "polynomial midpoint includes its arithmetic error");
                for(int j=0;j<=24;++j) {
                    const double t=j==24 ? hi : lo+(hi-lo)*j/24;
                    const auto expected=grid->evaluate(o+t*w),actual=cursor->evaluate(t);
                    const auto scalar=cursor->sample(t);
                    close(actual.value,expected.value,2e-12*(1+std::abs(expected.value)),
                        "cached cubic matches trilinear point values");
                    check((actual.gradient-expected.gradient).norm()<=2e-11*(1+expected.gradient.norm()),
                        "cached quadratic gradient matches trilinear point gradients");
                    close(scalar.value,expected.value,2e-12*(1+std::abs(expected.value)),
                        "scalar ray query matches the independent field");
                    close(scalar.derivative,expected.gradient.dot(w),2e-11*(1+expected.gradient.norm()*w.norm()),
                        "cubic derivative matches the directional field gradient");
                    check(expected.value>=b.minimumValue && expected.value<=b.maximumValue &&
                          expected.gradient.dot(w)>=b.minimumDerivative &&
                          expected.gradient.dot(w)<=b.maximumDerivative,
                        "polynomial interval covers reference values and derivatives including faces");
                    ++points;
                }
                ++intervals;
            }
            // A complete multi-cell query and a later backwards seek must both
            // work; neither the interval builder nor candidates own the cursor.
            const auto whole=cursor->rayBounds(0,3);
            for(double t:{2.9,0.01,1.2,0.4}) {
                const auto p=grid->evaluate(o+t*w);
                close(cursor->evaluate(t).value,p.value,2e-12*(1+std::abs(p.value)),
                    "cursor supports non-monotone queries after a multi-cell bound");
                check(p.value>=whole.minimumValue && p.value<=whole.maximumValue &&
                    p.gradient.dot(w)>=whole.minimumDerivative && p.gradient.dot(w)<=whole.maximumDerivative,
                    "multi-cell polynomial certificate covers reference queries");
            }
        }

    auto exact=*grid;
    exact.worldToIndex=M::Identity();exact.translation=V::Zero();exact.scale=1;exact.offset=0;
    for(int z=0;z<6;++z) for(int y=0;y<6;++y) for(int x=0;x<6;++x)
        exact.data.values[(z*6+y)*6+x]=float(x*y*z);
    const V origin(1.1,1.3,1.5),direction(0.3,0.2,-0.4);
    auto cache=exact.createRayCursor(origin,direction);
    const double limit=cache->nextBoundary(0,0.2);
    close(limit,0.2,0,"reuse fixture remains within one cell");
    for(int i=0;i<20;++i) {
        const double t=0.005*i;
        const V p=origin+t*direction;
        const V expectedGradient(p.y()*p.z(),p.x()*p.z(),p.x()*p.y());
        const auto result=cache->evaluate(t);
        close(result.value,p.x()*p.y()*p.z(),1e-13,"independent cubic fixture");
        check((result.gradient-expectedGradient).norm()<1e-12,"independent gradient polynomial fixture");
        const auto b=cache->rayBounds(t,t+0.005);
        const double second=2*(direction.x()*direction.y()*p.z()+
            direction.x()*direction.z()*p.y()+direction.y()*direction.z()*p.x());
        check(std::abs(second)<=b.maximumSecondDerivative,"cubic curvature certificate");
    }
    for(double t:{1e-15,1e-12,1e-9}) {
        const double linear=V(origin.y()*origin.z(),origin.x()*origin.z(),origin.x()*origin.y()).dot(direction);
        const double quadratic=direction.x()*direction.y()*origin.z()+
            direction.x()*direction.z()*origin.y()+direction.y()*direction.z()*origin.x();
        const double expected=t*(linear+t*(quadratic+t*direction.x()*direction.y()*direction.z()));
        close(cache->valueDifference(t),expected,1e-12*std::abs(expected)+1e-30,
            "cached birth differences avoid subtractive cancellation");
    }
    check(cache->statistics().cellBuilds==1 && cache->statistics().pointFallbacks==0 &&
        cache->statistics().boundFallbacks==0,
        "same-cell bounds, candidate jets and birth differences reuse one eight-node load");
    auto other=exact.createRayCursor(V(3.1,2.2,2.3),V(-0.2,0.1,0.4));
    other->rayBounds(0,2);
    check(cache->statistics().cellBuilds==1,"independent ray cursors do not overwrite each other");
    // The value has an interior maximum at t=0.1.
    auto extremum=exact.createRayCursor(origin,V(1,-1,0));
    const auto extremumBounds=extremum->rayBounds(0,0.18);
    check(extremumBounds.maximumValue>=2.16,"Bernstein enclosure retains an interior extremum");

    for(const V &o:{V(1,1,1),V(0.25,1.5,1.5),V(4.75,1.5,1.5)})
        for(const V &w:{V(1,1,1),V(-1,-1,-1),V(0,1,0)}) {
            auto cursor=exact.createRayCursor(o,w);
            double t=0;
            for(int i=0;i<16 && t<2;++i) {
                const double end=cursor->nextBoundary(t,2);
                check(end>t,"face, corner and clamp starts advance in either direction");
                const auto b=cursor->rayBounds(t,end);
                for(double u:{t,0.5*t+0.5*end,end}) {
                    const auto reference=exact.evaluate(o+u*w);
                    const auto actual=cursor->evaluate(u);
                    close(actual.value,reference.value,1e-11,"boundary value convention");
                    check((actual.gradient-reference.gradient).norm()<1e-10,"boundary gradient convention");
                    check(reference.gradient.dot(w)>=b.minimumDerivative &&
                        reference.gradient.dot(w)<=b.maximumDerivative,"boundary derivative enclosure");
                }
                t=end;
            }
            close(t,2,0,"traversal reaches the requested endpoint");
        }
    auto tiny=exact.createRayCursor(V(0,1.5,1.5),V::UnitX());
    const double left=std::nextafter(1.0,0.0);
    const auto tinyBounds=tiny->rayBounds(left,1);
    for(double t:{left,1.0}) {
        const auto ref=exact.evaluate(V(t,1.5,1.5));
        check(ref.value>=tinyBounds.minimumValue && ref.value<=tinyBounds.maximumValue &&
            ref.gradient.x()>=tinyBounds.minimumDerivative && ref.gradient.x()<=tinyBounds.maximumDerivative,
            "adjacent-double interval preserves neighbouring cells");
    }
    check(tiny->statistics().boundFallbacks==1,"unrepresentable interior uses the reference certificate");
    SdfRayQueryOptions supplied;supplied.midpoint=V(3.2,2.4,1.8);
    close(cache->rayBounds(0,0.1,supplied).midpointValue,exact.evaluate(*supplied.midpoint).value,0,
        "cursor respects an explicitly supplied world midpoint");

    auto translated=exact;
    translated.translation=V(-1e12,1e12,-1e12);
    const V largeOrigin=origin-translated.translation;
    auto difficult=translated.createRayCursor(largeOrigin,direction);
    for(double t:{0.0,0.1,0.3})
        close(difficult->evaluate(t).value,translated.evaluate(largeOrigin+t*direction).value,0,
            "large-coordinate uncertainty retains exact point fallback");
    difficult->rayBounds(0,0.1);
    check(difficult->statistics().boundFallbacks>0,"uncertain transform retains the original bound");

    const cg::GPSSField cachedField{grid,cg::CovarianceKernel(0.4,M::Identity()),
        {V::Constant(-100),V::Constant(100)}};
    auto referenceField=cachedField;
    referenceField.mean=std::make_shared<UncachedGridMean>(grid);
    cg::FlightState birth;birth.birthPosition=V(0.1,0.2,0.3);
    birth.birthValue=0.1;birth.birthGradient=V(0.2,0.1,1);
    birth.direction=V(0.2,-0.1,1).normalized();
    const cg::ConditionalFlightKernel cached(cachedField,birth),reference(referenceField,birth);
    for(double t:{1e-8,1e-5,0.001,0.1,0.7,1.3,2.0}) {
        const auto a=cached.scalarStatistics(t),b=reference.scalarStatistics(t);
        close(a.meanF,b.meanF,1e-12*(1+std::abs(b.meanF)),"cached conditional mean matches reference");
        close(a.meanK,b.meanK,1e-11*(1+std::abs(b.meanK)),"cached conditional slope matches reference");
        const double h=reference.evaluate(t).hazard.value;
        close(cached.evaluate(t).hazard.value,h,1e-9*(1+h),"cached hazard matches uncached field");
    }
    cg::ConditionalTrackingSettings options;
    options.step=0.8;options.ageScale=1;
    const V segmentOrigin=birth.birthPosition+0.4*birth.direction;
    const cg::ConditionalSegment flight(cachedField,birth,segmentOrigin,birth.direction,1.6,options);
    double opticalDepth=0;
    for(int i=0;i<10000;++i)
        opticalDepth+=reference.evaluate(0.4+1.6*(i+0.5)/10000).hazard.value*1.6/10000;
    const double survival=std::exp(-opticalDepth);
    cg::Random rng(50483);
    int escaped=0;double sum=0,sum2=0;
    constexpr int samples=5000;
    for(int i=0;i<samples;++i) {
        double hit;escaped+=flight.track(rng,true,hit)!=0;
        const double weight=flight.track(rng,false,hit);
        sum+=weight;sum2+=weight*weight;
    }
    close(double(escaped)/samples,survival,6*std::sqrt(survival*(1-survival)/samples)+1e-4,
        "cached grid delta tracking preserves the uncached integrated hazard law");
    const double variance=std::max(0.0,(sum2-sum*sum/samples)/(samples-1));
    close(sum/samples,survival,6*std::sqrt(variance/samples)+1e-4,
        "cached grid ratio tracking preserves the uncached integrated hazard law");
    std::cout<<"grid cursor: "<<intervals<<" intervals / "<<points
        <<" reference points, cubic fixtures, reuse, boundaries, seeks and survival laws passed\n";
}
#endif
