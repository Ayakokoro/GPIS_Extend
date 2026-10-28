#pragma once

// The value is continuous at x=0, but the outgoing cell has a different
// derivative from the point query used by the observed collision gradient.
inline void surfaceBirthChecks()
{
#if OPENVDB_AVAILABLE
    using namespace Tungsten;
    auto grid=std::make_shared<GpisSdfGrid>();
    grid->data.dimensions=Vec3i(3);grid->data.origin=Vec3i(-1);
    grid->data.clampMin=Vec3f(-1.0f);grid->data.clampMax=Vec3f(1.0f);
    grid->scale=1;grid->offset=0;
    grid->worldToIndex=M::Identity();grid->translation=V::Zero();
    for(int z=-1;z<=1;++z) for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x)
        grid->data.values.push_back(float(y-std::abs(x)));
    grid->initializeMagnitudeBounds();

    for(const std::string mode:{"adaptive","two_constants"}) {
        auto medium=makeMedium();
        // Install an in-memory VDB fixture without exposing a production setter.
        auto &f=const_cast<cg::GPSSField &>(medium->field());
        f.mean=grid;
        auto &tracking=const_cast<cg::ConditionalTrackingSettings &>(medium->trackingSettings());
        tracking.majorantMode=mode=="adaptive" ? cg::ConditionalMajorantMode::Adaptive :
                                               cg::ConditionalMajorantMode::TwoConstants;
        auto context=std::make_shared<ConditionalPathContext>();
        context->owner=medium.get();context->position=V::Zero();
        context->rayOrigin=Vec3f(0.0f);
        context->birth.birthPosition=V::Zero();context->birth.birthValue=0;
        context->birth.direction=V(-1,0,0);
        const Ray ray(context->rayOrigin,Vec3f(-1.0f,0.0f,0.0f),0,0.25f);

        // Both have a positive observed g.w; the true forward birth slopes
        // are respectively -1.5 and zero. Neither may start a positive flight.
        for(double gx:{-0.5,-2.0}) {
            context->birth.birthGradient=V(gx,1,0);
            check(context->birth.birthGradient.dot(context->birth.direction)>0,
                "crease regression starts with an outward observed gradient");
            bool rejected=false;
            try { cg::ConditionalFlightKernel(f,context->birth).birthMajorant(0.25); }
            catch(const cg::NumericError &e) { rejected=e.status()==cg::NumericStatus::NonPositiveSurfaceBirth; }
            check(rejected,"nonpositive forward slope has its own status");

            Medium::MediumState path{};path.reset();path.pathContext=context;
            auto shadow=path;
            MediumSample sample{};
            IndependentPathSampler sampler(712,false),referenceSampler(712,false);
            sampler.startPath(0,0);referenceSampler.startPath(0,0);
            check(!medium->sampleDistance(sampler,ray,path,sample),
                "crease continuation terminates without an exception");
            check(medium->transmittance(sampler,ray,false,true,&shadow)==Vec3f(0.0f),
                "crease light connection has zero transmittance");
            check(path.pathContext==context && shadow.pathContext==context && path.bounce==0,
                "discard does not renew or mutate branch state");
            check(sampler.next1D()==referenceSampler.next1D(),
                "discard does not consume random numbers or resample");
        }
        check(medium->discardedSurfacePaths()==2 && medium->discardedSurfaceConnections()==2,
            "discard counters distinguish paths from light connections");

        // A valid departure from the same crease still reaches the endpoint.
        context->birth.birthGradient=V(-0.5,1,0);
        const Ray outward(context->rayOrigin,Vec3f(0.0f,1.0f,0.0f),0,0.01f);
        Medium::MediumState valid{};valid.reset();valid.pathContext=context;
        MediumSample sample{};
        IndependentPathSampler sampler(345,false);sampler.startPath(0,0);
        check(medium->sampleDistance(sampler,outward,valid,sample) && sample.exited,
            "positive forward birth is still traced");

        // A negative slope at a positive-valued exterior point is supported.
        auto exterior=context->birth;exterior.birthValue=0.5;
        cg::ConditionalFlightKernel(f,exterior).birthMajorant(0.25);

        class UncertifiedMean final : public cg::MeanField {
        public:
            const char *typeName() const override { return "uncertified-test-mean"; }
            cg::MeanJet evaluate(const V &p) const override { return {p.y(),V::UnitY()}; }
            cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {}; }
        };
        f.mean=std::make_shared<UncertifiedMean>();
        Medium::MediumState invalid{};invalid.reset();invalid.pathContext=context;
        bool pathError=false,shadowError=false;
        try { medium->sampleDistance(sampler,ray,invalid,sample); }
        catch(const cg::NumericError &e) { pathError=e.status()==cg::NumericStatus::InvalidMajorant; }
        try { medium->transmittance(sampler,ray,false,true,&invalid); }
        catch(const cg::NumericError &e) { shadowError=e.status()==cg::NumericStatus::InvalidMajorant; }
        check(pathError && shadowError,"unrelated bound failures still propagate on both branches");
        check(medium->discardedSurfacePaths()==2 && medium->discardedSurfaceConnections()==2,
            "valid flights and numerical errors are not counted as discarded births");
    }
    std::cout<<"surface births: V-grid negative/zero slopes, branch termination, valid departures and error propagation passed\n";
#endif
}
