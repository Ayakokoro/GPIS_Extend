#pragma once
#include "media/GpisSdfGrid.hpp"

struct PointReuseQueries { int points=0,samples=0,jets=0; };

class PointReuseMean final : public cg::MeanField {
    std::shared_ptr<const cg::MeanField> _source;
    std::shared_ptr<PointReuseQueries> _queries;
    class Cursor final : public cg::MeanRayCursor {
        std::unique_ptr<cg::MeanRayCursor> _source;
        std::shared_ptr<PointReuseQueries> _queries;
    public:
        Cursor(std::unique_ptr<cg::MeanRayCursor> source,std::shared_ptr<PointReuseQueries> queries)
            :_source(std::move(source)),_queries(std::move(queries)) {}
        double nextBoundary(double a,double b) override { return _source->nextBoundary(a,b); }
        cg::MeanRaySample sample(double t) override { ++_queries->samples;return _source->sample(t); }
        cg::MeanJet evaluate(double t) override { ++_queries->jets;return _source->evaluate(t); }
        double valueDifference(double t,double *error=nullptr) override { return _source->valueDifference(t,error); }
        cg::MeanRayBounds rayBounds(double a,double b,const Tungsten::SdfRayQueryOptions &options={}) override {
            return _source->rayBounds(a,b,options);
        }
        Statistics statistics() const override { return _source->statistics(); }
    };
public:
    PointReuseMean(std::shared_ptr<const cg::MeanField> source,std::shared_ptr<PointReuseQueries> queries)
        :_source(std::move(source)),_queries(std::move(queries)) {}
    const char *typeName() const override { return "point_reuse_test"; }
    cg::MeanJet evaluate(const V &p) const override { ++_queries->points;return _source->evaluate(p); }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override { return _source->bounds(b); }
    double valueDifference(const V &p,const V &d) const override { return _source->valueDifference(p,d); }
    std::unique_ptr<cg::MeanRayCursor> createRayCursor(const V &p,const V &w) const override {
        auto cursor=_source->createRayCursor(p,w);
        return cursor ? std::make_unique<Cursor>(std::move(cursor),_queries) : nullptr;
    }
};

inline void samePointStatisticsChecks()
{
    auto original=field(),counted=original;
    const auto queries=std::make_shared<PointReuseQueries>();
    counted.mean=std::make_shared<PointReuseMean>(original.mean,queries);
    cg::FlightState birth;birth.birthPosition=V(.2,-.1,.4);
    birth.direction=V(.2,.1,1).normalized();
    birth.birthValue=.3;birth.birthGradient=V(.1,-.2,.8);
    const cg::ConditionalFlightKernel kernel(counted,birth);
    *queries={};
    kernel.evaluate(.7);
    const auto gradient=kernel.hitStatistics(.7).collisionGradient;
    kernel.scalarStatistics(.7);
    check(queries->points==1,"hazard and collision gradient share one analytic point jet");
    const auto ref=reference(original,birth,birth.birthPosition+.7*birth.direction);
    const V expectedMean=ref.mean.tail<3>()-ref.covariance.block<3,1>(1,0)*ref.mean[0]/ref.covariance(0,0);
    const M expectedCov=ref.covariance.bottomRightCorner<3,3>()-
        ref.covariance.block<3,1>(1,0)*ref.covariance.block<1,3>(0,1)/ref.covariance(0,0);
    check((gradient.mean-expectedMean).norm()<2e-10 && (gradient.covariance-expectedCov).norm()<2e-10,
        "reused collision statistics agree with independent joint-Gaussian conditioning");

    kernel.scalarStatistics(1.1);
    kernel.hitStatistics(.7);kernel.hitStatistics(1.1);
    check(queries->points==2,"current/history pair survives interleaved accesses");
    kernel.scalarStatistics(2);kernel.scalarStatistics(.7);
    check(queries->points==4,"third age evicts the least recently used point");
    const double neighbor=std::nextafter(.7,1.0);
    kernel.scalarStatistics(neighbor);
    check(queries->points==5,"adjacent representable ages are never merged");

    const cg::ConditionalFlightKernel endpointKernel(counted,birth);
    *queries={};endpointKernel.evaluate(.9);
    cg::Random cachedRng(592),coldRng(592);
    for(int i=0;i<20;++i) {
        const cg::ConditionalFlightKernel cold(original,birth);
        const auto cached=endpointKernel.samplePositiveEndpoint(.9,cachedRng);
        const auto expected=cold.samplePositiveEndpoint(.9,coldRng);
        check(cached.birthValue==expected.birthValue && cached.birthGradient==expected.birthGradient &&
              cached.birthPosition==expected.birthPosition,"warm and cold endpoint samples are bit-identical");
    }
    check(queries->points==1,"endpoint value and sampled gradient reuse the analytic jet");
    close(cachedRng.uniform01(),coldRng.uniform01(),0,"statistics reuse preserves random draw order");
    close(endpointKernel.evaluate(0).hazard.value,0,0,"exact zero-age hazard stays zero");
    bool rejected=false;
    try { endpointKernel.scalarStatistics(0); } catch(const std::out_of_range &) { rejected=true; }
    check(rejected,"zero-age scalar statistics retain validation");
    auto turned=birth;turned.direction=V::UnitX();
    const cg::ConditionalFlightKernel other(counted,turned);
    *queries={};other.scalarStatistics(.9);
    check(queries->points==1,"a new flight never reuses another flight's entry");

#if OPENVDB_AVAILABLE
    auto grid=std::make_shared<Tungsten::GpisSdfGrid>();
    grid->data.dimensions=Tungsten::Vec3i(5);grid->data.origin=Tungsten::Vec3i(0);
    grid->data.clampMin=Tungsten::Vec3f(.1f);grid->data.clampMax=Tungsten::Vec3f(3.9f);
    grid->worldToIndex=M::Identity();grid->translation=V(.4,.3,.2);
    grid->scale=1;grid->offset=0;grid->gradientBound=10;
    for(int z=0;z<5;++z) for(int y=0;y<5;++y) for(int x=0;x<5;++x)
        grid->data.values.push_back(float(.1*x*y*z+.3*x-.2*y+.1*z));
    grid->initializeMagnitudeBounds();
    auto gridField=original;gridField.mean=std::make_shared<PointReuseMean>(grid,queries);
    cg::FlightState gridBirth=birth;gridBirth.birthPosition=V(.2,.1,.3);
    gridBirth.direction=V(.4,.2,.3).normalized();
    auto boundaryCursor=grid->createRayCursor(gridBirth.birthPosition,gridBirth.direction);
    const double face=boundaryCursor->nextBoundary(0,4);
    const cg::ConditionalFlightKernel cachedGrid(gridField,gridBirth);
    for(double t:{.05,face,std::nextafter(face,0.0),std::nextafter(face,4.0),1.7}) {
        const cg::ConditionalFlightKernel cold(gridField,gridBirth);
        *queries={};cachedGrid.scalarStatistics(t);
        check(queries->samples==1 && queries->jets==0,"grid scalar query keeps the scalar cursor path");
        // Move the ray cursor without replacing the statistical cache entry.
        cachedGrid.meanValue(t+.01);
        const auto value=cachedGrid.hitStatistics(t).collisionGradient;
        cachedGrid.hitStatistics(t);
        check(queries->samples==1 && queries->jets==2,"grid gradient is loaded once after an unrelated cursor seek");
        const auto expected=cold.hitStatistics(t).collisionGradient;
        check(value.mean==expected.mean && value.covariance==expected.covariance,
            "grid faces, neighboring doubles and backward seeks retain gradient convention");
    }
#endif
    std::cout<<"same-point statistics: analytic query reuse, two-age interleaving, exact keys, grid lazy jets, "
             <<"Gaussian moments and endpoint RNG parity passed\n";
}
