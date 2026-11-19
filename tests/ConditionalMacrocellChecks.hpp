#pragma once
#include "ConditionalOpticalDepthChecks.hpp"

#if OPENVDB_AVAILABLE
class MacrocellRecordingGrid final : public Tungsten::GpisSdfGrid {
public:
    struct Counts { int fine=0,coarse=0,boundaries=0,jets=0,samples=0,scalar=0; };
    mutable Counts counts;
    double evaluateValue(const V &p) const override {
        ++counts.scalar;return GpisSdfGrid::evaluateValue(p);
    }
    std::unique_ptr<cg::MeanRayCursor> createRayCursor(const V &o,const V &w) const override {
        class Cursor final : public cg::MeanRayCursor {
            std::unique_ptr<cg::MeanRayCursor> _inner;
            Counts &_counts;
        public:
            Cursor(std::unique_ptr<cg::MeanRayCursor> inner,Counts &counts)
                :_inner(std::move(inner)),_counts(counts) {}
            double nextBoundary(double lo,double hi) override {
                ++_counts.boundaries;return _inner->nextBoundary(lo,hi);
            }
            cg::MeanRaySample sample(double t) override { ++_counts.samples;return _inner->sample(t); }
            cg::MeanJet evaluate(double t) override { ++_counts.jets;return _inner->evaluate(t); }
            double valueDifference(double t,double *error=nullptr) override { return _inner->valueDifference(t,error); }
            cg::MeanRayBounds rayBounds(double lo,double hi,const Tungsten::SdfRayQueryOptions &options) override {
                ++_counts.fine;return _inner->rayBounds(lo,hi,options);
            }
            bool coarseRayBounds(double lo,double &hi,cg::MeanRayBounds &result) override {
                ++_counts.coarse;return _inner->coarseRayBounds(lo,hi,result);
            }
            Statistics statistics() const override { return _inner->statistics(); }
        };
        return std::make_unique<Cursor>(GpisSdfGrid::createRayCursor(o,w),counts);
    }
};

inline void macrocellChecks(const std::shared_ptr<Tungsten::GpisSdfGrid> &source)
{
    using namespace Tungsten;
    {
        auto grid=std::make_shared<GpisSdfGrid>(*source);
        grid->buildMacrocells(4);
        auto f=field();f.mean=grid;f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
        M precision=M::Zero();precision.diagonal()=V(225,400,900);
        f.kernel=cg::CovarianceKernel(.04,precision);
        int checked=0;
        for(const V &direction:{V(.2,-.3,1).normalized().eval(),V(-.7,.1,-.2).normalized().eval()})
            for(double f0:{0.,.03}) for(double slope:{.001,2.}) {
                cg::FlightState birth;birth.birthPosition=V(.2,.1,-.3);birth.direction=direction;
                birth.birthValue=f0;birth.birthGradient=slope*direction;
                cg::ConditionalFlightKernel kernel(f,birth),reference(f,birth);
                const double ell=1/std::sqrt(direction.dot(precision*direction));
                for(double q:{4.,.08,.852635,1.,1.73205,32.,64.,.5}) {
                    const double begin=q*ell;
                    double end=begin+.8*ell,upper;
                    check(kernel.chapter2MacrocellMajorant(begin,end,upper),"chapter 2 coarse block available");
                    for(int i=0;i<=64;++i) {
                        const double t=i==64 ? end : begin+(end-begin)*(i/64.);
                        check(reference.evaluate(t).hazard.value<=upper*(1+1e-10)+1e-280,
                              "chapter 2 single-point bound covers independent point hazard");
                        ++checked;
                    }
                }
            }
        std::cout<<"chapter 2: "<<checked<<" coarse-block coverage checks passed\n";
    }
    // Compare both sides of the event threshold, including small conditional
    // variance, signed birth residuals, and parents with positive mean/slope.
    {
        const auto f=field();
        const auto w=V::UnitZ().eval();
        const double ell=1/std::sqrt(w.dot(f.kernel.metric()*w));
        int screened=0,kept=0;
        for(double value:{0.,.03}) for(double slope:{-.4,1.5}) {
            cg::FlightState birth;birth.direction=w;birth.birthValue=value;birth.birthGradient=slope*w;
            birth.birthPosition=V::Zero();
            cg::ConditionalFlightKernel kernel(f,birth);
            for(double q:{.08,.2,.75,1.,1.5,3.,25.}) for(double width:{.07,1.,2.5})
                for(double fmin:{-.25,.0,.2}) for(double gmin:{-1.,.25}) {
                    cg::MeanRayBounds mean;
                    mean.minimumValue=fmin;mean.maximumValue=fmin+.4;
                    mean.minimumDerivative=gmin;mean.maximumDerivative=gmin+2.;
                    const double lo=q*ell,hi=(q+width)*ell;
                    double upper;
                    check(kernel.chapter2Majorant(lo,hi,mean,upper),"parent oracle is available");
                    for(double budget:{1.,4.,16.,std::nextafter(upper*(hi-lo),INFINITY)}) {
                        if(kernel.rejectChapter2Parent(lo,mean,hi-lo,1.,budget)) {
                            check(upper*(hi-lo)>budget,"cheap screen only rejects over-budget parents");
                            ++screened;
                        } else ++kept;
                    }
                }
        }
        check(screened>0 && kept>0,"screen exercises both decisions");
    }
    // Independent point evaluations check the cached cubic, normalized z and
    // final extinction bounds, including anisotropy, grazing/positive births,
    // backward seeks, grid faces and the clamped exterior.
    int polynomialChecks=0;
    for(const V &origin:{V(.2,-.1,.8),V(-2,0,0),V(2.1,.1,.4)})
        for(const V &direction:{V(.2,.1,-1).normalized().eval(),V(-.7,.2,-.1).normalized().eval()})
        for(double f0:{0.,.02}) for(double slope:{1e-4,.3,8.}) for(bool tabulated:{false,true}) {
            auto f=field();f.mean=source;f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
            M precision=M::Zero();precision.diagonal()=V(400,225,900);
            f.kernel=cg::CovarianceKernel(.04,precision);
            cg::FlightState birth;birth.birthPosition=origin;birth.direction=direction;
            birth.birthValue=f0;birth.birthGradient=slope*direction;
            cg::ConditionalFlightKernel kernel(f,birth,false,0,0,tabulated),reference(f,birth);
            auto cursor=source->createRayCursor(origin,direction);
            for(double lo:{.1,.0000005,.03,.0005,.005}) {
                const double hi=cursor->nextBoundary(lo,lo+.07);
                cg::MeanRayPolynomial p;double zlo,zhi;
                check(cursor->polynomial(lo,hi,p),"single-cell cubic is available");
                check(kernel.polynomialZBounds(lo,hi,zlo,zhi),"degree-five z certificate is available");
                const double majorant=kernel.coupledIntervalMajorant(lo,hi,source->gradientBound,true,INFINITY,false,true,true);
                for(int i=0;i<=32;++i) {
                    const double x=double(i)/32,t=lo+(hi-lo)*x;
                    const auto &c=p.coefficients;
                    const double m=c[0]+x*(c[1]+x*(c[2]+x*c[3]));
                    check(std::abs(m-source->evaluate(origin+t*direction).value)<=p.error,"cached cubic covers direct grid query");
                    const auto stats=reference.scalarStatistics(t);
                    const double z=stats.meanF/std::sqrt(stats.varianceF);
                    check(z>=zlo && z<=zhi,"degree-five bounds cover exact normalized mean");
                    check(reference.evaluate(t).hazard.value<=majorant*(1+1e-10)+1e-280,"degree-five majorant covers exact hazard");
                    ++polynomialChecks;
                }
            }
        }
    std::cout<<"degree-five polynomial: "<<polynomialChecks<<" independent grid/z/hazard coverage checks passed\n";
    {
        // Regression: hi/lo=6500. A global Taylor error divided by lo^2
        // makes this positive, essentially transparent interval look opaque.
        auto flat=std::make_shared<GpisSdfGrid>(*source);
        std::fill(flat->data.values.begin(),flat->data.values.end(),0.f);
        flat->offset=0;flat->initializeMagnitudeBounds();
        auto f=field();f.mean=flat;f.kernel=cg::CovarianceKernel(.04,400*M::Identity());
        f.activeDomain={V::Constant(-1e6),V::Constant(1e6)};
        cg::FlightState birth;birth.direction=V::UnitZ();birth.birthGradient=.4*V::UnitZ();
        cg::ConditionalFlightKernel kernel(f,birth);
        double lo,hi;
        check(kernel.polynomialZBounds(1e-6,.0065,lo,hi) && lo>5,
              "degree-five error retains age dependence on a wide near-birth interval");
        cg::ConditionalTrackingSettings normal;normal.coupledBounds=true;normal.coupledLocalScale=.25;
        normal.localFarStartScale=4;normal.shadowRouletteThreshold=0;
        auto polynomial=normal;polynomial.transitionVoxelBound=cg::TransitionVoxelBound::Bernstein5;
        const V origin=birth.birthPosition+.03*birth.direction;
        cg::ConditionalSegment a(f,birth,origin,birth.direction,.1,normal,nullptr,0,0);
        cg::ConditionalSegment b(f,birth,origin,birth.direction,.1,polynomial,nullptr,0,0);
        cg::Random ra(783),rb(783);
        for(int i=0;i<64;++i) for(bool collision:{false,true}) {
            cg::ConditionalTrackingWork wa,wb;double ta,tb;
            const double va=a.track(ra,collision,ta,&wa),vb=b.track(rb,collision,tb,&wb);
            check(va==vb && ta==tb && wa.intervals==wb.intervals && wa.candidates==wb.candidates &&
                  ra.uniform01()==rb.uniform01(),"degree-five prefix preserves ordinary proposals beyond its cutoff");
        }
    }
    int intervals=0,points=0,parentSpans=0;
    for(int width:{2,4,8}) for(bool shifted:{false,true}) {
        auto grid=*source;
        if(shifted) {
            const Vec3i delta(-10,7,-3);
            grid.data.origin+=delta;
            grid.data.clampMin+=Vec3f(-10.0f,7.0f,-3.0f);grid.data.clampMax+=Vec3f(-10.0f,7.0f,-3.0f);
            grid.translation+=V(-10,7,-3);
        }
        grid.buildMacrocells(width,true);
        check(!grid.macrocells.empty(),"macrocell certificates are precomputed");
        for(const V &o:{V(-0.4,-0.2,-0.5),V(-3,-1,-2),V(2.1,0.1,0.4),V::Zero().eval()})
            for(const V &w:{V(0.4,0.2,0.3),V(-0.7,0.2,-0.1),V(0,0,-1),V::Zero().eval()}) {
                auto cursor=grid.createRayCursor(o,w);
                double lo=0;
                for(int steps=0;lo<3 && steps<64;++steps) {
                    double hi=3;
                    cg::MeanRayBounds b;
                    check(cursor->coarseRayBounds(lo,hi,b) && hi>lo && hi<=3,
                        "coarse traversal advances to a block or clamp boundary");
                    double parentEnd=3;
                    cg::MeanRayBounds parent;
                    check(cursor->parentRayBounds(lo,parentEnd,parent) && parent.certified && parentEnd>=hi,
                          "parent covers the first child up to the same clamp");
                    if(parentEnd>hi) ++parentSpans;
                    for(int j=0;j<=32;++j) {
                        const double t=j==32 ? parentEnd : lo+(parentEnd-lo)*j/32;
                        const auto jet=grid.evaluate(o+t*w);
                        const double slope=jet.gradient.dot(w);
                        check(parent.minimumValue<=jet.value && jet.value<=parent.maximumValue &&
                              parent.minimumDerivative<=slope && slope<=parent.maximumDerivative,
                              "parent extrema cover independent SDF samples across child faces");
                    }
                    check(b.certified && !std::isfinite(b.maximumSecondDerivative),
                        "coarse intervals certify values and slopes, not birth curvature");
                    const double mid=0.5*lo+0.5*hi;
                    close(b.midpointValue,grid.evaluate(o+mid*w).value,b.midpointError+1e-14,
                        "macrocell midpoint agrees with original trilinear interpolation");
                    for(int j=0;j<=32;++j) {
                        const double t=j==32 ? hi : lo+(hi-lo)*j/32;
                        const auto jet=grid.evaluate(o+t*w);
                        check(jet.value>=b.minimumValue && jet.value<=b.maximumValue &&
                            jet.gradient.dot(w)>=b.minimumDerivative && jet.gradient.dot(w)<=b.maximumDerivative,
                            "macrocell bounds cover fine-cell values, gradients and rounded faces");
                        close(jet.value,source->evaluate(o+t*w).value,1e-12,
                            "negative grid origins and acceleration preserve the field");
                        ++points;
                    }
                    ++intervals;lo=hi;
                }
                close(lo,3,0,"macrocell traversal reaches the requested endpoint");
                check(cursor->statistics().cellBuilds==0 && cursor->statistics().coarseQueries>0,
                    "coarse certificates do not traverse or build fine-cell polynomials");
            }
        auto difficult=grid;
        difficult.translation+=V(-1e12,1e12,-1e12);
        const V origin=difficult.worldToIndex.inverse()*(V(2,2,2)-difficult.translation);
        auto cursor=difficult.createRayCursor(origin,V::UnitZ());
        double end=0.1;cg::MeanRayBounds b;
        check(!cursor->coarseRayBounds(0,end,b) && end==0.1,
            "uncertain large transforms leave the fine fallback available");
        grid.buildMacrocells(0);
        cursor=grid.createRayCursor(V::Zero(),V::UnitZ());
        check(!cursor->coarseRayBounds(0,end,b),"zero macrocell size disables block queries");
        check(!cursor->parentRayBounds(0,end,b),"rebuilding without blocks clears parent bounds");
    }
    check(parentSpans>0,"parent bound tests cover intervals spanning multiple children");

    // Observe the actual transport routing, including one rejected coarse
    // window covering several fine cells.
    auto recording=std::make_shared<MacrocellRecordingGrid>();
    recording->data.dimensions=Vec3i(18);recording->data.origin=Vec3i(0);
    recording->data.clampMin=Vec3f(0.25f);recording->data.clampMax=Vec3f(16.75f);
    recording->data.values.assign(18*18*18,10.0f);
    recording->worldToIndex=20*M::Identity();recording->translation=V::Zero();
    recording->scale=1;recording->offset=0;recording->gradientBound=0;
    recording->buildMacrocells(8);
    cg::GPSSField f{recording,cg::CovarianceKernel(1,M::Identity()),{V::Constant(-100),V::Constant(100)}};
    cg::FlightState birth;birth.birthPosition=V(0.05,0.05,0.05);birth.birthValue=10;
    cg::ConditionalTrackingSettings settings;settings.step=0.8;settings.ageScale=1;settings.macrocellSize=8;
    const V origin=birth.birthPosition+0.4*V::UnitZ();
    const auto trace=[&](const cg::ConditionalTrackingSettings &options,bool collision) {
        cg::ConditionalSegment flight(f,birth,origin,V::UnitZ(),0.2,options);
        recording->counts={};
        OpticalDepthSampler sampler({std::numeric_limits<float>::min()});
        cg::Random rng(sampler);double hit;
        close(flight.track(rng,collision,hit),1,0,"macrocell routing fixture escapes without a candidate");
        close(hit,0.2,0,"macrocell routing preserves segment length");
        return recording->counts;
    };
    for(bool collision:{false,true}) {
        const auto coarse=trace(settings,collision);
        check(coarse.coarse==1 && coarse.fine==0 && coarse.boundaries==0,
            "accepted coarse interval skips fine bounds and fine DDA");
        auto disabled=settings;disabled.macrocellSize=0;
        const auto fine=trace(disabled,collision);
        check(fine.coarse==0 && fine.fine>=4,"disabled coarse mode retains interpolation-cell traversal");
    }
    recording->data.values.assign(18*18*18,0.0f);
    recording->buildMacrocells(8);
    birth.birthValue=0;birth.birthGradient=V::UnitZ();
    {
        auto surface=birth;
        surface.direction=V::UnitZ();
        const cg::ConditionalFlightKernel reference(f,surface), accelerated(f,surface,false,0,0,true);
        const auto oldPrefix=reference.birthMajorant(.3,0);
        recording->counts={};
        const auto prefix=accelerated.birthMajorant(.3,0);
        check(prefix.split>oldPrefix.split && recording->counts.fine==0,
              "positive directional slope certifies a prefix across interpolation cells");
        for(int i=1;i<=2048;++i)
            check(reference.evaluate(prefix.split*i/2048).hazard.value<=prefix.maximum*(1+1e-10),
                  "monotone surface prefix covers the unchanged conditional rate");
    }
    settings.macrocellMaxExpectedEvents=1e-12;
    for(bool collision:{false,true}) {
        const auto rejected=trace(settings,collision);
        check(rejected.coarse==1 && rejected.fine>=4,
            "rejected coarse window descends and is not retried per fine cell");
    }

    // A resumed segment starts at birth age 0.4, not age zero. With ell=0.5
    // along z, the same real grid must switch routing at the scaled birth age.
    M precision=M::Identity();precision(2,2)=4;
    f.kernel=cg::CovarianceKernel(1,precision);
    for(bool collision:{false,true}) {
        auto relaxed=settings;relaxed.macrocellFarMaxExpectedEvents=64;
        relaxed.macrocellFarStartScale=1.3; // 0.65 > entire [0.4,0.6] segment
        const auto nearTrace=trace(relaxed,collision);
        check(nearTrace.fine>=4,"far relaxation preserves near-field fine traversal");
        relaxed.macrocellFarStartScale=0.5; // 0.25 < resumed birth age
        const auto farTrace=trace(relaxed,collision);
        check(farTrace.coarse==1 && farTrace.fine==0 && farTrace.boundaries==0,
            "far relaxation uses directional ell and retained conditioning age");
    }

    // A true global tail bypasses both levels of the actual VDB cursor, even
    // with geometry proposals and tiny near-field step/event limits enabled.
    recording->initializeMagnitudeBounds();
    for(bool collision:{false,true}) {
        auto global=settings;global.globalFarStartScale=2;
        global.geometryStep=true;global.step=1e-12;global.maxExpectedEvents=1e-12;
        const cg::ConditionalSegment flight(f,birth,birth.birthPosition+1.1*V::UnitZ(),
            V::UnitZ(),0.2,global,nullptr,recording->gradientBound);
        recording->counts={};
        OpticalDepthSampler sampler({std::numeric_limits<float>::min()});cg::Random rng(sampler);
        cg::ConditionalTrackingWork work;double hit;
        close(flight.track(rng,collision,hit,&work),1,0,"VDB global-tail fixture escapes");
        check(work.intervals==1 && recording->counts.coarse==0 && recording->counts.fine==0 &&
            recording->counts.boundaries==0,"VDB global tail skips macro bounds, fine bounds and DDA");
    }

    // The local far tracker also bypasses grid traversal. Its two proposal
    // queries must use the new direct scalar path, not prepare a ray-cell jet.
    for(bool collision:{false,true}) {
        auto local=settings;local.localFarStartScale=2;local.step=1e-12;local.maxExpectedEvents=64;
        const cg::ConditionalSegment flight(f,birth,birth.birthPosition+1.1*V::UnitZ(),
            V::UnitZ(),0.2,local,nullptr,1,recording->gradientBound);
        recording->counts={};
        OpticalDepthSampler sampler({std::numeric_limits<float>::min()});cg::Random rng(sampler);
        cg::ConditionalTrackingWork work;double hit;
        close(flight.track(rng,collision,hit,&work),1,0,"VDB local-tail fixture escapes");
        check(work.intervals==1 && recording->counts.scalar==2 && recording->counts.jets==0 &&
            recording->counts.samples==0 && recording->counts.coarse==0 && recording->counts.fine==0 &&
            recording->counts.boundaries==0,"local far uses only scalar proposals/midpoints without ray-cell construction");
    }

    for(bool collision:{false,true}) {
        auto midpoint=settings;midpoint.transitionPointBounds=true;midpoint.localFarStartScale=4;
        midpoint.step=1e-12;midpoint.ageScale=1e-12;midpoint.maxExpectedEvents=64;
        midpoint.transitionMaxExpectedEvents=64;
        const cg::ConditionalSegment flight(f,birth,birth.birthPosition+.4*V::UnitZ(),
            V::UnitZ(),.2,midpoint,nullptr,1,recording->gradientBound);
        recording->counts={};
        OpticalDepthSampler sampler({std::numeric_limits<float>::min()});cg::Random rng(sampler);
        cg::ConditionalTrackingWork work;double hit;
        close(flight.track(rng,collision,hit,&work),1,0,"VDB transition fixture escapes");
        check(work.intervals==1 && recording->counts.scalar==2 && recording->counts.jets==0 &&
            recording->counts.samples==0 && recording->counts.coarse==0 && recording->counts.fine==0 &&
            recording->counts.boundaries==0,"transition scalar midpoint crosses cells without age cap or DDA");
    }

    // Independent uncached-hazard integration checks both tracking estimators.
    auto referenceField=f;referenceField.mean=std::make_shared<UncachedGridMean>(source);
    referenceField.kernel=cg::CovarianceKernel(0.4,M::Identity());
    birth={};birth.birthPosition=V(0.1,0.2,0.3);
    birth.birthValue=0.1;birth.birthGradient=V(0.2,0.1,1);
    birth.direction=V(0.2,-0.1,1).normalized();
    const cg::ConditionalFlightKernel reference(referenceField,birth);
    double optical=0;
    for(int i=0;i<10000;++i) optical+=reference.evaluate(0.4+1.6*(i+0.5)/10000).hazard.value*1.6/10000;
    const double expected=std::exp(-optical);
    for(bool normalizedOnly:{false,true}) for(int width:{2,4}) for(double threshold:{0.1,8.0}) for(bool relax:{false,true}) {
        auto grid=std::make_shared<GpisSdfGrid>(*source);grid->buildMacrocells(width);
        auto field=referenceField;field.mean=grid;
        const cg::ConditionalFlightKernel kernel(field,birth,normalizedOnly);
        for(double lo=0.4;lo<2;) {
            double hi=2,maximum;
            check(kernel.coarseMajorant(lo,hi,maximum),"conditional coarse majorant is available");
            for(int j=0;j<=32;++j) {
                const double t=lo+(hi-lo)*j/32;
                check(reference.evaluate(t).hazard.value<=maximum*(1+1e-10),
                    "coarse conditional majorant covers the independently evaluated hazard");
            }
            lo=hi;
        }
        if(normalizedOnly) {
            auto tailField=field;tailField.kernel=cg::CovarianceKernel(0.4,16*M::Identity());
            const cg::ConditionalFlightKernel tail(tailField,birth,true,2),exact(tailField,birth);
            for(double lo=0.51;lo<2;) {
                double hi=2,maximum;
                check(tail.coarseMajorant(lo,hi,maximum),"far tail uses a macrocell certificate");
                for(int j=0;j<=32;++j)
                    check(exact.evaluate(lo+(hi-lo)*j/32).hazard.value<=maximum*(1+1e-10),
                        "far macrocell covers exact conditional hazard across fine cells");
                lo=hi;
            }
        }
        settings.macrocellSize=width;settings.macrocellMaxExpectedEvents=threshold;
        settings.normalizedBoundOnly=normalizedOnly;
        settings.macrocellFarStartScale=0.8; // crosses the gate within this flight
        settings.macrocellFarMaxExpectedEvents=relax ? 16 : 0;
        const cg::ConditionalSegment flight(field,birth,birth.birthPosition+0.4*birth.direction,
            birth.direction,1.6,settings);
        cg::Random rng(2026+width);
        constexpr int n=4000;
        double sum=0,sum2=0;int escaped=0;
        for(int i=0;i<n;++i) {
            double hit;escaped+=flight.track(rng,true,hit)!=0;
            const double tr=flight.track(rng,false,hit);sum+=tr;sum2+=tr*tr;
        }
        close(double(escaped)/n,expected,6*std::sqrt(expected*(1-expected)/n)+1e-4,
            "macrocell delta tracking preserves integrated-hazard survival");
        const double variance=std::max(0.0,(sum2-sum*sum/n)/(n-1));
        close(sum/n,expected,6*std::sqrt(variance/n)+1e-4,
            "macrocell ratio tracking preserves integrated-hazard survival");
        if(relax) std::cout<<"far macrocell survival: width="<<width<<" near="<<threshold
            <<" delta="<<double(escaped)/n<<" ratio="<<sum/n<<" expected="<<expected<<"\n";
    }
    std::cout<<"macrocells: "<<intervals<<" intervals / "<<points
        <<" reference points, coarse/fine routing, rejection reuse and survival laws passed\n";
}
#endif
