#pragma once
// Pointwise checks on the original VDB, including positive/grazing births,
// nonmonotone cache seeks and inherited parent certificates.
inline nlohmann::json macrocellCoverageChecks(const nlohmann::json &files,
    const Tungsten::ConditionalGpis::GPSSField &field) {
    namespace cg=Tungsten::ConditionalGpis;using J=nlohmann::json;
    const auto vec=[](const J &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    size_t points=0,blocks=0;double maximumRatio=0;
    for(const auto &file:files) {
        std::ifstream source(file.get<std::string>());const J input=J::parse(source);
        for(double f0:{0.,1e-6,.04})for(double gradientScale:{.001,1.}) {
            cg::FlightState birth;birth.birthPosition=vec(input["birth_position"]);
            birth.direction=vec(input["direction"]).normalized();birth.birthValue=f0;
            birth.birthGradient=gradientScale*vec(input["birth_gradient"]);
            cg::ConditionalFlightKernel kernel(field,birth),reference(field,birth);
            const double ell=1/std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
            const double end=std::min(kernel.maximumAgeInDomain(),input["length"].get<double>());
            if(!(end>1e-4*ell))continue;
            for(int j=0;j<96;++j) {
                const int index=(j*37)%96;
                const double lo=1e-5*ell*std::pow(end/(1e-5*ell),index/96.);
                double hi=std::min(end,lo*1.6+.05*ell);
                if(lo<4*ell)hi=std::min(hi,4*ell);
                if(kernel.clipMacrocellInterval(lo,hi))++blocks;
                if(!(hi>lo))continue;
                const auto bound=[&](double a,double b,double parent) {
                    return a<4*ell ? kernel.coupledIntervalMajorant(a,b,L,a<.25*ell,parent,true)
                        : kernel.localFarIntervalMajorant(a,b,L,L,parent,true,true);
                };
                const double parent=bound(lo,hi,INFINITY);
                for(int half=0;half<2;++half) {
                    const double a=half ? .5*(lo+hi):lo,b=half ? hi:.5*(lo+hi);
                    if(!(b>a))continue;
                    const double upper=bound(a,b,parent);
                    for(int k=0;k<=16;++k) {
                        const double t=k==16 ? b:a+(b-a)*k/16;
                        const double actual=reference.evaluate(t).hazard.value;
                        if(actual>upper*(1+1e-9)+1e-280)
                            throw std::runtime_error("Macrocell conditional hazard exceeds its certificate");
                        if(upper>0)maximumRatio=std::max(maximumRatio,actual/upper);
                        ++points;
                    }
                }
            }
        }
    }
    if(blocks==0)throw std::runtime_error("Macrocell coverage checks did not exercise macro certificates");
    return {{"coverage_points",points},{"macro_intervals",blocks},{"maximum_hazard_bound_ratio",maximumRatio},
        {"positive_and_surface_births",true},{"grazing_gradients",true},{"backwards_cache_seeks",true}};
}
