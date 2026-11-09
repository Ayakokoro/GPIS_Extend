#pragma once

// Included after the parent test's cg/sh/V aliases and require/close helpers.
inline void sparseHistoryGridChecks() {
#if OPENVDB_AVAILABLE
    auto grid=std::make_shared<Tungsten::GpisSdfGrid>();
    grid->data.origin=Tungsten::Vec3i(0);
    grid->data.dimensions=Tungsten::Vec3i(3);
    grid->data.clampMin=Tungsten::Vec3f(0.f);
    grid->data.clampMax=Tungsten::Vec3f(1);
    grid->worldToIndex=Eigen::Matrix3d::Identity();
    grid->translation=V::Zero();
    grid->scale=1; grid->offset=0;
    grid->data.values.resize(27);
    // A genuinely trilinear cell whose diagonal is the cubic
    // 64*s^3-108*s^2+42*s+3. Its endpoint derivatives are both positive,
    // yet it has a maximum at s=1/4 and a minimum at s=7/8.
    for(int z=0;z<3;++z) for(int y=0;y<3;++y) for(int x=0;x<3;++x)
        grid->data.values[(z*3+y)*3+x]=float(64*x*y*z-36*(x*y+x*z+y*z)+14*(x+y+z)+3);
    grid->initializeMagnitudeBounds();
    const double diagonal=std::sqrt(3.0);
    const V direction=V::Ones()/diagonal;
    auto cursor=grid->createRayCursor(V::Zero(),direction);
    std::vector<double> roots;
    require(cursor->appendStationaryPoints(0,diagonal,roots),"grid enumerates all stationary roots");
    std::sort(roots.begin(),roots.end());
    require(roots.size()==2,"one trilinear voxel can contain two ray stationary points");
    close(roots[0],diagonal/4,2e-13,"first cubic stationary root");
    close(roots[1],7*diagonal/8,2e-13,"second cubic stationary root");
    close(cursor->sample(0).value,3,1e-12,"positive first endpoint");
    close(cursor->sample(diagonal).value,1,1e-12,"positive second endpoint");
    close(cursor->sample(roots[1]).value,-.0625,1e-12,"hidden negative trough");
    require(cursor->sample(0).derivative>0 && cursor->sample(diagonal*.999999).derivative>0,
            "endpoint derivative signs alone miss the two extrema");
    const auto range=cursor->rayBounds(0,diagonal);
    require(range.certified&&range.minimumValue<=-.0625&&range.maximumValue>=7.75,
            "whole-step range encloses both hidden extrema");

    const double ell=diagonal/2;
    const cg::GPSSField field{grid,cg::CovarianceKernel(1,Eigen::Matrix3d::Identity()/(ell*ell)),
        {V::Zero(),V::Ones()}};
    sh::Start start;start.direction=direction;
    sh::FlightSettings settings;settings.historyPoints=2;
    sh::Flight flight(field,start,settings);
    const auto before=flight.selectHistory(1.0);
    const auto after=flight.selectHistory(1.7);
    require(after.size()==2,"recent point and different-bucket trough are retained");
    require(std::any_of(after.begin(),after.end(),[&](double t){return std::abs(t-roots[1])<2e-12;}),
            "second internal point discovers the hidden cubic trough");
    for(double t:before) require(t>0&&t<1.0,"future trough is not a past sign constraint");
    require(flight.selectHistory(1.0)==before,"cached future geometry cannot alter a past prefix");
    settings.maxStepOverLength=.0625;
    sh::Flight fine(field,start,settings);
    require(fine.selectHistory(1.7)==after,"quadrature step does not create history nodes");
    settings.historyPoints=1;
    sh::Flight one(field,start,settings);
    const auto minimum=one.selectHistory(1.7);
    require(minimum.size()==1,"one-point mode respects its internal constraint budget");
    close(minimum[0],roots[1],2e-12,"one-point selector keeps the hidden minimum");
    std::cout<<"Trilinear stationary-root and causal candidate checks passed\n";
#else
    std::cout<<"Grid stationary checks skipped: OpenVDB is unavailable\n";
#endif
}
