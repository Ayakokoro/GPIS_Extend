#pragma once
#include "math/conditional/transport/ConditionalMajorantBounds.h"

// Compare the integrated raw proposal with the point hazard on identical
// occupied intervals. Thinning/roulette would otherwise change visit lengths.
inline nlohmann::json chapter2RayChecks(const nlohmann::json &files,
    const Tungsten::ConditionalGpis::GPSSField &field,
    const Tungsten::ConditionalGpis::ConditionalTrackingSettings &settings,double startScale,bool factors=false,
    double endScale=std::numeric_limits<double>::infinity()) {
    namespace cg=Tungsten::ConditionalGpis;
    using Json=nlohmann::json;
    if(!settings.classicLocalBand || !(startScale>=0 && endScale>startScale))
        throw std::runtime_error("Chapter 2 occupied-block audit requires a support band and ordered age limits");
    Json result=Json::array();
    const auto vector=[](const Json &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    for(const auto &file:files) {
        std::ifstream input(file.get<std::string>());
        Json ray=Json::parse(input);
        cg::FlightState birth;
        birth.birthPosition=vector(ray.at("birth_position"));birth.birthGradient=vector(ray.at("birth_gradient"));
        birth.birthValue=ray.at("birth_value");birth.direction=vector(ray.at("direction")).normalized();
        const cg::Vector3 origin=vector(ray.at("origin"));
        const double age=(origin-birth.birthPosition).dot(birth.direction);
        const double ell=1/std::sqrt(birth.direction.dot(field.kernel.precision()*birth.direction));
        const double length=std::clamp(endScale*ell-age,0.,ray.at("length").get<double>());
        const double begin=std::max({0.,ray.value("tracking_start_distance",0.),startScale*ell-age,
                                     ray.at("birth_split").get<double>()-age});
        cg::ConditionalFlightKernel kernel(field,birth);
        auto cursor=factors ? field.mean->createRayCursor(birth.birthPosition,birth.direction) : nullptr;
        if(factors && !cursor)throw std::runtime_error("Factor audit requires a grid cursor");
        Json rayFactors;
        if(factors) {
            const auto prior=field.mean->evaluate(birth.birthPosition);
            rayFactors={{"sigma",field.kernel.sigma()},{"ell",ell},{"df",birth.birthValue-prior.value},
                        {"f0",birth.birthValue},{"m0",prior.value},{"k0",birth.direction.dot(birth.birthGradient)},
                        {"dm0",birth.direction.dot(prior.gradient)},{"domain",ray["domain"]},{"file",file},
                        {"birth_split",ray.at("birth_split")},{"begin",begin},{"length",length},
                        {"dk",birth.direction.dot(birth.birthGradient-prior.gradient)},{"age",age}};
        }
        Tungsten::ClassicLocalGrid::Cursor band(*settings.classicLocalBand,origin,birth.direction,std::min(begin,length),length);
        Json blocks=Json::array();
        double depth=0,actual=0,largest=0,ratio=0;
        for(double lo=begin,occupiedEnd=begin;lo<length;) {
            if(lo>=occupiedEnd && !band.nextOccupiedRange(lo,occupiedEnd))break;
            double endAge=age+occupiedEnd,maximum=0;
            if(!kernel.chapter2MacrocellMajorant(age+lo,endAge,maximum))
                throw std::runtime_error("Missing chapter 2 block in ray audit");
            const double hi=std::min(occupiedEnd,endAge-age);
            if(!(hi>lo)) throw std::runtime_error("Non-advancing chapter 2 audit");
            const double cost=maximum*(hi-lo);
            double sampledDepth=0;
            double minZ=INFINITY,maxZ=-INFINITY,minG=INFINITY,maxSdG=0;
            for(int i=0;i<=512;++i) {
                const double t=lo+(hi-lo)*(i/512.);
                const double h=kernel.evaluate(age+t).hazard.value;
                if(factors) {
                    const auto point=kernel.scalarStatistics(age+t);
                    const double z=point.meanF/std::sqrt(point.varianceF);
                    minZ=std::min(minZ,z);maxZ=std::max(maxZ,z);
                    minG=std::min(minG,point.meanAtZero);
                    maxSdG=std::max(maxSdG,std::sqrt(point.varianceAtZero));
                }
                if(h>maximum*(1+1e-10)+1e-280)throw std::runtime_error("Chapter 2 bound below sampled hazard");
                if(maximum>0)ratio=std::max(ratio,h/maximum);
                sampledDepth+=h*(i==0 || i==512 ? .5:1.)*(hi-lo)/512.;
            }
            blocks.push_back({{"lo",lo},{"hi",hi},{"expected_candidates",cost},{"sampled_depth",sampledDepth}});
            if(factors) {
                double end=endAge;cg::MeanRayBounds m;
                if(!cursor->coarseRayBounds(age+lo,end,m) || end<endAge)
                    throw std::runtime_error("Factor audit did not reproduce the same block");
                blocks.back()["factors"]={{"macro_m_lo",m.minimumValue},{"macro_m_hi",m.maximumValue},
                    {"macro_derivative_lo",m.minimumDerivative},{"sample_z_lo",minZ},{"sample_z_hi",maxZ},
                    {"sample_g_lo",minG},{"sample_sd_g_hi",maxSdG},{"points",513}};
            }
            largest=std::max(largest,cost);actual+=sampledDepth;depth+=cost;lo=hi;
        }
        result.push_back({{"id",ray["id"]},{"start_scale",startScale},{"blocks",blocks},{"majorant_depth",depth},
                          {"sampled_hazard_depth",actual},{"largest_block_depth",largest},{"max_sampled_ratio",ratio}});
        if(factors)result.back()["ray_factors"]=rayFactors;
    }
    return result;
}
