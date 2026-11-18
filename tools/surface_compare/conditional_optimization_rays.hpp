#pragma once
#include <map>
#include <tuple>
#include <boost/math/quadrature/gauss_kronrod.hpp>
// Small, independent slow-flight benchmark. Timing includes construction;
// Monte Carlo estimates are checked against refined quadrature of the actual
// conditional hazard, not against a majorant or another tracking mode.
inline nlohmann::json optimizationRays(const nlohmann::json &files,
    const Tungsten::ConditionalGpis::GPSSField &field,
    const Tungsten::ConditionalGpis::ConditionalTrackingSettings &settings,unsigned count) {
    namespace cg=Tungsten::ConditionalGpis;using J=nlohmann::json;
    using Clock=std::chrono::steady_clock;
    const auto vec=[](const J &v){return cg::Vector3(v[0].get<double>(),v[1].get<double>(),v[2].get<double>());};
    const double L=field.mean->globalMagnitudeBounds()->maximumGradientNorm;
    static std::map<std::tuple<std::string,double,const Tungsten::ClassicLocalGrid *>,J> references;
    J rows=J::array();
    for(const auto &file:files) {
        std::ifstream source(file.get<std::string>());const J input=J::parse(source);
        cg::FlightState birth;birth.birthPosition=vec(input["birth_position"]);birth.birthGradient=vec(input["birth_gradient"]);
        birth.birthValue=input["birth_value"];birth.age=input["birth_age"];
        const auto origin=vec(input["origin"]),direction=vec(input["direction"]);birth.direction=direction;
        const double length=input["length"];const bool collision=input["domain"]=="path";
        const double begin=input.value("tracking_start_distance",0.0);
        const auto key=std::make_tuple(file.get<std::string>(),settings.exteriorCutoffSigma,settings.classicLocalBand);
        if(!references.count(key)) {
            auto observation=birth;observation.direction=direction.normalized();observation.age=0;
            cg::ConditionalFlightKernel reference(field,observation);
            const double age=(origin-birth.birthPosition).dot(observation.direction);
            std::vector<double> knots;
            field.mean->appendRayBreakpoints(observation.birthPosition,observation.direction,age,age+length,knots);
            const auto prefix=reference.birthMajorant(age+length);
            const double tiny=std::min(length*1e-10,prefix.split*.001);
            const double omitted=age+begin==0 ? reference.birthMajorant(tiny).maximum*tiny:0;
            if(!(omitted<1e-10))throw std::runtime_error("Reference prefix is not negligible");
            const double start=age+begin>0 ? age+begin:tiny;
            knots.push_back(start);knots.push_back(age+length);
            for(int i=1;i<256;++i)knots.push_back(start*std::pow((age+length)/start,i/256.));
            if (settings.classicLocalBand) {
                // Nonnegative trilinear density has constant support on open voxel intervals.
                Tungsten::ClassicLocalGrid::Cursor cursor(*settings.classicLocalBand,origin,
                    observation.direction,0,length);
                std::vector<double> faces;
                cursor.appendVoxelBoundaries(faces);
                for (double t:faces) knots.push_back(age+t);
            }
            std::sort(knots.begin(),knots.end());
            const double cutoff=settings.exteriorCutoffSigma*field.kernel.sigma();
            if(cutoff>0) {
                // Split at SDF extrema and threshold crossings before integrating
                // the discontinuous mask. Shaderball is trilinear within each cell.
                auto cursor=field.mean->createRayCursor(observation.birthPosition,observation.direction);
                if(!cursor || !cursor->appendStationaryPoints(start,age+length,knots))
                    throw std::runtime_error("Cannot partition the reference SDF into monotone intervals");
                std::sort(knots.begin(),knots.end());
                std::vector<double> crossings;
                for(size_t i=1;i<knots.size();++i) {
                    double a=std::max(start,knots[i-1]), b=std::min(age+length,knots[i]);
                    if(!(b>a))continue;
                    const bool outside=reference.scalarMeanValue(a)>cutoff;
                    if(outside==(reference.scalarMeanValue(b)>cutoff))continue;
                    for(int step=0;step<60;++step) {
                        const double m=.5*(a+b);
                        if(!(m>a && m<b))break;
                        if((reference.scalarMeanValue(m)>cutoff)==outside)a=m;else b=m;
                    }
                    crossings.push_back(.5*(a+b));
                }
                knots.insert(knots.end(),crossings.begin(),crossings.end());
                std::sort(knots.begin(),knots.end());
            }
            double depths[2]={0,0};
            for(int pass=0;pass<2;++pass) for(size_t i=1;i<knots.size();++i) {
                const double a=std::max(start,knots[i-1]),b=std::min(age+length,knots[i]);if(!(b>a))continue;
                if(cutoff>0 && reference.scalarMeanValue(.5*(a+b))>cutoff)continue;
                if(settings.classicLocalBand && settings.classicLocalBand->density(
                        origin+(.5*(a+b)-age)*observation.direction)==0)continue;
                depths[pass]+=boost::math::quadrature::gauss_kronrod<double,31>::integrate(
                    [&](double t){return reference.evaluate(t).hazard.value;},a,b,8,pass ? 1e-11:1e-8);
            }
            const double expected=std::exp(-depths[1]),difference=std::abs(expected-std::exp(-depths[0]));
            if(difference>1e-7)throw std::runtime_error("Reference quadrature did not converge");
            references[key]={{"transmittance",expected},{"optical_depth",depths[1]},
                {"refinement_difference",difference},{"omitted_prefix_depth_bound",omitted}};
        }
        std::vector<double> times;double sum=0,sum2=0;uint64_t events=0,intervals=0;
        cg::Random rng(20261118);
        for(unsigned i=0;i<count;++i) {
            cg::ConditionalTrackingWork work;double hit;
            const auto start=Clock::now();
            cg::ConditionalSegment segment(field,birth,origin,direction,length,settings,nullptr,L,L);
            const double tr=segment.track(rng,collision,hit,&work,begin);
            times.push_back(std::chrono::duration<double,std::micro>(Clock::now()-start).count());
            sum+=tr;sum2+=tr*tr;events+=work.candidates;intervals+=work.intervals;
        }
        std::sort(times.begin(),times.end());
        const double variance=count>1 ? std::max(0.,(sum2-sum*sum/count)/(count-1)) : 0;
        const double expected=references[key]["transmittance"],standardError=std::sqrt(variance/count);
        const double tolerance=6*(collision ? std::sqrt(expected*(1-expected)/count):standardError)+1e-4;
        if(count>=1000 && std::abs(sum/count-expected)>tolerance)throw std::runtime_error("Slow-ray Monte Carlo law check failed");
        rows.push_back({{"id",input["id"]},{"domain",input["domain"]},{"samples",count},
            {"median_us",times[count/2]},{"p95_us",times[std::min(size_t(count-1),size_t(.95*count))]},
            {"mean_candidates",double(events)/count},{"mean_intervals",double(intervals)/count},
            {"mean_transmittance",sum/count},{"standard_error",standardError},{"reference",references[key]},
            {"within_six_sigma",std::abs(sum/count-expected)<=tolerance}});
    }
    return rows;
}
