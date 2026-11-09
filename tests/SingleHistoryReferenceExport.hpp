#pragma once

#include "math/SingleHistoryFlight.hpp"
#include "media/GpisSdfGrid.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// Offline distribution experiment: export the production selected-history
// law and its origin-only ablation on exactly the same trilinear mean field.
// No sampling or reference GP implementation is shared with the Python oracle.
namespace SingleHistoryReferenceExport {
namespace cg = Tungsten::ConditionalGpis;
namespace sh = Tungsten::SparseHistoryGpis;
namespace one = Tungsten::SingleHistoryGpis;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
constexpr double Ell=.2, Sigma=.02, End=6;

struct Case {
    const char *id;
    bool surface;
    double gradient;
    std::vector<std::pair<double,double>> knots;
    std::vector<double> probes;

    double mean(double q) const {
        if(knots.empty()) return .4+.3*(q-3)*(q-3);
        std::size_t hi=1;
        while(hi+1<knots.size()&&q>=knots[hi].first) ++hi;
        const auto a=knots[hi-1],b=knots[hi];
        return a.second+(q-a.first)*(b.second-a.second)/(b.first-a.first);
    }
};

inline std::vector<Case> cases() {
    return {
        {"first_grazing",false,0,{}, {2.25,4.25}},
        {"first_double",false,0,{{0,2},{1,.5},{2,-.6},{3,.7},{4,-.5},{5,1},{6,1.5}}, {1.65,3.65}},
        {"surface_groove",true,1,{{0,0},{.5,.5},{1.5,-.8},{2.25,.45},{3.25,-.05},{4.5,.6},{6,.9}}, {1,2.75}},
        {"surface_rising",true,.3,{{0,0},{6,.9}}, {1.5,4.5}}
    };
}

#if OPENVDB_AVAILABLE
inline std::shared_ptr<Tungsten::GpisSdfGrid> makeGrid(const Case &spec) {
    using namespace Tungsten;
    auto grid=std::make_shared<GpisSdfGrid>();
    // Four padding nodes precede the birth. q=6 remains strictly inside the
    // interpolation domain, including the right-hand derivative cell.
    grid->data.origin=Vec3i(0);
    grid->data.dimensions=Vec3i(8,8,132);
    grid->data.clampMin=Vec3f(.25f);
    grid->data.clampMax=Vec3f(6.75f,6.75f,130.75f);
    grid->worldToIndex=100*M::Identity();
    grid->translation=V(0,0,4);
    grid->scale=1;grid->offset=0;
    grid->data.values.resize(8*8*132);
    for(int z=0;z<132;++z) for(int y=0;y<8;++y) for(int x=0;x<8;++x)
        grid->data.values[(z*8+y)*8+x]=float(Sigma*spec.mean((z-4)/20.));
    grid->initializeMagnitudeBounds();
    grid->buildMacrocells(4);
    return grid;
}

inline void selected(std::ostream &out,const one::Flight &flight) {
    const auto history=flight.snapshot().selected;
    if(history) out<<*history/Ell;else out<<"null";
}

inline std::vector<double> boundaries(int subdivisions,int meanSubdivisions=20) {
    std::vector<double> result;
    for(int i=0;i<=int(End*subdivisions);++i) result.push_back(double(i)/subdivisions);
    for(int i=0;i<=int(End*meanSubdivisions);++i) result.push_back(double(i)/meanSubdivisions);
    for(int i=0;i<=12;++i) result.push_back(.5*i);
    std::sort(result.begin(),result.end());
    result.erase(std::unique(result.begin(),result.end()),result.end());
    return result;
}

inline void workStatistics(std::ostream &out,const one::Work &work) {
    out<<"{\"cells\":"<<work.cells<<",\"selections\":"<<work.selections
       <<",\"candidate_statistics\":"<<work.candidateStatistics
       <<",\"score_evaluations\":"<<work.scoreEvaluations
       <<",\"macro_candidate_fragments\":"<<work.macroCandidateFragments
       <<",\"fine_boundary_queries\":"<<work.fineBoundaryQueries
       <<",\"macro_boundary_queries\":"<<work.macroBoundaryQueries
       <<",\"selection_seconds\":"<<work.selectionSeconds
       <<",\"candidate_statistics_seconds\":"<<work.candidateStatisticsSeconds
       <<",\"scoring_seconds\":"<<work.scoringSeconds<<'}';
}

inline void model(std::ostream &out,const cg::GPSSField &field,const sh::Start &birth,
                  const one::Settings &settings,const Case &spec,const char *name,
                  int meanSubdivisions=20,bool exportStatistics=false,int coarseDivisions=64) {
    out<<"{\"integration\":[";
    bool firstIntegration=true;
    for(int divisions:{coarseDivisions,2*coarseDivisions}) {
        if(!firstIntegration) out<<',';firstIntegration=false;
        out<<"{\"subdivisions_per_length\":"<<divisions<<",\"segments\":[";
        const auto started=std::chrono::steady_clock::now();
        const auto points=boundaries(divisions,meanSubdivisions);
        one::Flight flight(field,birth,settings);
        for(std::size_t i=1;i<points.size();++i) {
            const double a=points[i-1],b=points[i],q=a+.5*(b-a);
            try {
                const double rate=Ell*flight.law(Ell*q).hazard();
                if(!std::isfinite(rate)||rate<0) throw std::runtime_error("invalid exported extinction");
                if(i>1) out<<',';
                out<<'['<<a<<','<<b<<','<<q<<','<<rate<<',';
                selected(out,flight);out<<']';
            } catch(const std::exception &error) {
                std::ostringstream message;
                message<<std::setprecision(17)<<spec.id<<'/'<<name<<" hazard q="<<q<<": "<<error.what();
                throw std::runtime_error(message.str());
            }
        }
        out<<']';
        if(exportStatistics) {
            out<<",\"hazard_queries\":"<<points.size()-1<<",\"work\":";
            workStatistics(out,flight.work());
            out<<",\"elapsed_seconds\":"
               <<std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        }
        out<<'}';
    }
    out<<"],\"probes\":[";
    for(std::size_t index=0;index<spec.probes.size();++index) {
        const double q=spec.probes[index];
        try {
            // Each probe advances its own fresh canonical candidate cache;
            // integration nodes never become history observations.
            one::Flight flight(field,birth,settings);
            const auto law=flight.law(Ell*q);
            const auto gradient=law.meanGradient();
            if(index) out<<',';
            out<<"{\"q\":"<<q<<",\"selected\":";selected(out,flight);
            out<<",\"mean_slope\":"<<gradient.dot(birth.direction)*Ell/Sigma
               <<",\"mean_gradient\":["<<gradient.x()<<','<<gradient.y()<<','<<gradient.z()<<']'
               <<",\"speed_cdf\":[";
            for(int speedIndex=0;speedIndex<=160;++speedIndex) {
                const double speed=speedIndex/20.;
                const double probability=law.speedCdf(Sigma/Ell*speed);
                if(!std::isfinite(probability)||probability<0||probability>1)
                    throw std::runtime_error("invalid exported speed CDF");
                if(speedIndex) out<<',';
                out<<'['<<speed<<','<<probability<<']';
            }
            out<<']';
            if(exportStatistics) {out<<",\"work\":";workStatistics(out,flight.work());}
            out<<'}';
        } catch(const std::exception &error) {
            std::ostringstream message;
            message<<std::setprecision(17)<<spec.id<<'/'<<name<<" speed probe q="<<q<<": "<<error.what();
            throw std::runtime_error(message.str());
        }
    }
    out<<"]}";
}

inline void run(const char *path) {
    std::ofstream out(path,std::ios::binary);
    if(!out) throw std::runtime_error("Cannot open single-history reference export output");
    out<<std::setprecision(17)
       <<"{\"ell\":"<<Ell<<",\"sigma\":"<<Sigma<<",\"q_end\":"<<End
       <<",\"settings\":{\"selection_step_over_length\":0.5,\"history_groups\":8,"
         "\"candidate_collection\":\"macro\",\"candidate_macrocell_size\":4,"
         "\"voxel_width\":0.01,\"probability_method\":\"numeric\",\"probability_relative_tolerance\":0.0002},"
         "\"segment_columns\":[\"a\",\"b\",\"mid\",\"ell_hazard\",\"selected_q\"],\"cases\":[";
    bool firstCase=true;
    for(const auto &spec:cases()) {
        std::cout<<"Exporting "<<spec.id<<"\n"<<std::flush;
        const auto grid=makeGrid(spec);
        const cg::GPSSField field{grid,cg::CovarianceKernel(Sigma,M::Identity()/(Ell*Ell)),
                                 {V::Constant(-100),V::Constant(100)}};
        sh::Start birth;birth.origin=V(.033,.037,0);birth.direction=V::UnitZ();
        if(spec.surface) birth.kind=sh::Start::SurfaceValueGradient;
        birth.gradient=(Sigma/Ell*spec.gradient)*birth.direction;
        const auto cursor=grid->createRayCursor(birth.origin,birth.direction);
        const auto birthMean=cursor->evaluate(0);
        one::Settings settings;
        settings.selectionStepOverLength=.5;settings.historyGroups=8;
        settings.candidateCollection=one::CandidateCollection::Macro;
        settings.candidateMacrocellSize=4;settings.macrocellSize=4;
        settings.maxWork=16000000;settings.profile=false;settings.profileDetail=false;
        settings.probability.relativeTolerance=2e-4;
        if(!firstCase) out<<',';firstCase=false;
        out<<"{\"id\":\""<<spec.id<<"\",\"start_kind\":\""<<(spec.surface?"surface":"first_hit")
           <<"\",\"start_value\":"<<birthMean.value/Sigma
           <<",\"start_slope\":"<<birthMean.gradient.dot(birth.direction)*Ell/Sigma
           <<",\"start_gradient\":"<<spec.gradient<<",\"mean_knots\":[";
        for(int i=0;i<=120;++i) {
            if(i) out<<',';
            out<<'['<<i/20.<<','<<grid->at(3,3,i+4)/Sigma<<']';
        }
        out<<"],\"models\":{\"baseline\":";
        settings.historyEnabled=false;model(out,field,birth,settings,spec,"baseline");
        out<<",\"selected\":";
        settings.historyEnabled=true;model(out,field,birth,settings,spec,"selected");
        out<<"}}";
        out.flush();
    }
    out<<"]}\n";
    if(!out) throw std::runtime_error("Failed writing single-history reference export output");
}

inline std::shared_ptr<Tungsten::GpisSdfGrid> makeNarrowGrid(const Case &spec) {
    using namespace Tungsten;
    auto grid=std::make_shared<GpisSdfGrid>();
    grid->data.origin=Vec3i(0);grid->data.dimensions=Vec3i(8,8,492);
    grid->data.clampMin=Vec3f(.25f);grid->data.clampMax=Vec3f(6.75f,6.75f,490.75f);
    grid->worldToIndex=400*M::Identity();grid->translation=V(0,0,4);
    grid->scale=1;grid->offset=0;grid->data.values.resize(8*8*492);
    for(int z=0;z<492;++z) for(int y=0;y<8;++y) for(int x=0;x<8;++x)
        grid->data.values[(z*8+y)*8+x]=float(Sigma*spec.mean((z-4)/80.));
    grid->initializeMagnitudeBounds();grid->buildMacrocells(16);
    return grid;
}

inline void runNarrow(const char *path) {
    std::ofstream out(path,std::ios::binary);
    if(!out) throw std::runtime_error("Cannot open narrow-groove reference export output");
    out<<std::setprecision(17)
       <<"{\"ell\":"<<Ell<<",\"sigma\":"<<Sigma<<",\"q_end\":"<<End
       <<",\"experiment\":\"narrow_groove_width_sweep\","
         "\"integration_levels\":[512,1024],"
         "\"settings\":{\"selection_step_over_length\":0.5,\"history_groups\":8,"
         "\"candidate_collection\":\"macro\",\"candidate_macrocell_size\":16,"
         "\"macro_width_over_length\":0.2,\"voxel_width\":0.0025,\"voxel_width_over_length\":0.0125,"
         "\"z_padding_nodes\":4,\"probability_method\":\"numeric\",\"probability_relative_tolerance\":0.0002,"
         "\"profile\":true,\"profile_detail\":true},"
         "\"model_settings\":[{\"name\":\"baseline\",\"history_enabled\":false,\"candidate_collection\":\"macro\"},"
         "{\"name\":\"selected\",\"history_enabled\":true,\"candidate_collection\":\"macro\"},"
         "{\"name\":\"fine_selected\",\"history_enabled\":true,\"candidate_collection\":\"fine\"}],"
         "\"segment_columns\":[\"a\",\"b\",\"mid\",\"ell_hazard\",\"selected_q\"],\"cases\":[";
    const std::vector<std::pair<double,const char *>> widths={{1,"1"},{.5,"0p5"},{.2,"0p2"},{.1,"0p1"},{.05,"0p05"}};
    bool firstCase=true;
    for(bool slopeLimited:{false,true}) for(bool surface:{false,true}) for(const auto &width:widths) {
        if(slopeLimited&&width.first>.2) continue;
        constexpr double center=1.7,plateau=.5;
        const double depth=slopeLimited?std::min(1.3,5*width.first):1.3;
        const double floor=plateau-depth;
        const std::string id=std::string(surface?"surface_narrow_w":"first_narrow_w")+width.second+
            (slopeLimited?"_slope_limited":"");
        const Case spec{id.c_str(),surface,surface?1.:0.,
            {{0,surface?0.:2.},{.5,plateau},{center-.5*width.first,plateau},
             {center,floor},{center+.5*width.first,plateau},{End,plateau}}, {2.25,2.75}};
        std::cout<<"Exporting "<<id<<"\n"<<std::flush;
        const auto grid=makeNarrowGrid(spec);
        const cg::GPSSField field{grid,cg::CovarianceKernel(Sigma,M::Identity()/(Ell*Ell)),
                                 {V::Constant(-100),V::Constant(100)}};
        sh::Start birth;birth.origin=V(.00825,.00925,0);birth.direction=V::UnitZ();
        if(surface) birth.kind=sh::Start::SurfaceValueGradient;
        birth.gradient=(Sigma/Ell*spec.gradient)*birth.direction;
        const auto cursor=grid->createRayCursor(birth.origin,birth.direction);
        const auto birthMean=cursor->evaluate(0);
        const double actualPlateau=grid->at(3,3,84)/Sigma;
        const double actualFloor=grid->at(3,3,140)/Sigma;
        double maximumPhysicalSlope=0;
        for(int i=0;i<480;++i)
            maximumPhysicalSlope=std::max(maximumPhysicalSlope,
                std::abs(grid->at(3,3,i+5)-grid->at(3,3,i+4))/.0025);
        one::Settings settings;
        settings.selectionStepOverLength=.5;settings.historyGroups=8;
        settings.candidateCollection=one::CandidateCollection::Macro;
        settings.candidateMacrocellSize=16;settings.macrocellSize=16;
        settings.maxWork=16000000;settings.profile=true;settings.profileDetail=true;
        settings.probability.relativeTolerance=2e-4;
        if(!firstCase) out<<',';firstCase=false;
        out<<"{\"id\":\""<<id<<"\",\"start_kind\":\""<<(surface?"surface":"first_hit")
           <<"\",\"family\":\""<<(slopeLimited?"slope_limited":"fixed_depth")<<'"'
           <<",\"width_over_length\":"<<width.first<<",\"center\":"<<center
           <<",\"plateau\":"<<actualPlateau<<",\"floor\":"<<actualFloor
           <<",\"depth\":"<<actualPlateau-actualFloor<<",\"max_physical_slope\":"<<maximumPhysicalSlope
           <<",\"start_value\":"<<birthMean.value/Sigma
           <<",\"start_slope\":"<<birthMean.gradient.dot(birth.direction)*Ell/Sigma
           <<",\"start_gradient\":"<<spec.gradient<<",\"mean_knots\":[";
        for(int i=0;i<=480;++i) {
            if(i) out<<',';
            out<<'['<<i/80.<<','<<grid->at(3,3,i+4)/Sigma<<']';
        }
        out<<"],\"models\":{\"baseline\":";
        settings.historyEnabled=false;model(out,field,birth,settings,spec,"baseline",80,true,512);
        out<<",\"selected\":";
        settings.historyEnabled=true;model(out,field,birth,settings,spec,"selected",80,true,512);
        out<<",\"fine_selected\":";
        settings.candidateCollection=one::CandidateCollection::Fine;
        model(out,field,birth,settings,spec,"fine_selected",80,true,512);
        out<<"}}";out.flush();
    }
    out<<"]}\n";
    if(!out) throw std::runtime_error("Failed writing narrow-groove reference export output");
}
#else
inline void run(const char *) {
    throw std::runtime_error("single-history reference_export requires OpenVDB support");
}
inline void runNarrow(const char *) {
    throw std::runtime_error("single-history narrow_reference_export requires OpenVDB support");
}
#endif
} // namespace SingleHistoryReferenceExport
