#pragma once
#include "ConditionalOpticalDepthChecks.hpp"

#if OPENVDB_AVAILABLE
class MacrocellRecordingGrid final : public Tungsten::GpisSdfGrid {
public:
    struct Counts { int fine=0,coarse=0,boundaries=0; };
    mutable Counts counts;
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
            cg::MeanRaySample sample(double t) override { return _inner->sample(t); }
            cg::MeanJet evaluate(double t) override { return _inner->evaluate(t); }
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
    int intervals=0,points=0;
    for(int width:{2,4,8}) for(bool shifted:{false,true}) {
        auto grid=*source;
        if(shifted) {
            const Vec3i delta(-10,7,-3);
            grid.data.origin+=delta;
            grid.data.clampMin+=Vec3f(-10.0f,7.0f,-3.0f);grid.data.clampMax+=Vec3f(-10.0f,7.0f,-3.0f);
            grid.translation+=V(-10,7,-3);
        }
        grid.buildMacrocells(width);
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
    }

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
    settings.macrocellMaxExpectedEvents=1e-12;
    for(bool collision:{false,true}) {
        const auto rejected=trace(settings,collision);
        check(rejected.coarse==1 && rejected.fine>=4,
            "rejected coarse window descends and is not retried per fine cell");
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
    for(int width:{2,4}) for(double threshold:{0.1,8.0}) {
        auto grid=std::make_shared<GpisSdfGrid>(*source);grid->buildMacrocells(width);
        auto field=referenceField;field.mean=grid;
        const cg::ConditionalFlightKernel kernel(field,birth);
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
        settings.macrocellSize=width;settings.macrocellMaxExpectedEvents=threshold;
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
    }
    std::cout<<"macrocells: "<<intervals<<" intervals / "<<points
        <<" reference points, coarse/fine routing, rejection reuse and survival laws passed\n";
}
#endif
