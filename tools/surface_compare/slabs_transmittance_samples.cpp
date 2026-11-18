// Sample first-hit distances on the previously exported, fixed double rays.
#include "media/SingleHistoryGpisMedium.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "math/conditional/core/Random.h"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <nlohmann/json.hpp>
#include <openvdb/openvdb.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <atomic>
#include <mutex>
#include <omp.h>

using namespace Tungsten;
using J=nlohmann::json;
using V=Eigen::Vector3d;
J read(const std::string &p) {
    std::ifstream f(p);if(!f)throw std::runtime_error("Missing input: "+p);return J::parse(f);
}
V vector(const J &p) {return {p[0].get<double>(),p[1].get<double>(),p[2].get<double>()};}
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: slabs_transmittance_samples spec.json output.json");
        openvdb::initialize();const J spec=read(argv[1]);
        const int count=spec.at("samples"),threads=spec.value("threads",4);
        if(count<1024||count>1048576||threads<1||threads>32)throw std::runtime_error("Invalid sample count or threads");
        const unsigned seed=spec.at("seed");omp_set_num_threads(threads);
        const std::string scenePath=spec.at("scene");const J source=read(scenePath);
        J records=J::array();
        for(const auto &methodJson:spec.at("methods")) {
            const std::string method=methodJson;
            J definition=source.at("media").at(0);
            definition["single_history"]["probability_method"]=method;
            definition["single_history"]["profile"]=false;
            definition["single_history"]["profile_detail"]=false;
            Scene scene;JsonDocument document(Path(scenePath),definition.dump());
            auto medium=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
            if(!medium)throw std::runtime_error("Expected single_history_gpis");
            medium->loadResources();medium->prepareForRender();
            for(const auto &path:spec.at("ray_inputs")) {
                const J input=read(path.get<std::string>());
                const std::string id=input.at("provenance").at("case").at("id");
                const V origin=vector(input.at("ray").at("origin")),direction=vector(input.at("ray").at("direction"));
                const double length=input.at("ray").at("t");
                SparseHistoryGpis::Start birth;birth.origin=origin;birth.direction=direction;
                birth.kind=SparseHistoryGpis::Start::ExteriorPositiveOnly;
                const auto pixel=input.at("provenance").at("case").at("pixel");
                const unsigned pixelId=pixel[0].get<unsigned>()+64*pixel[1].get<unsigned>();
                std::vector<double> distances(count,-1.);
                std::atomic<bool> failed{false};std::string error;std::mutex errorMutex;
                const double started=omp_get_wtime();
                for(int begin=0;begin<count;begin+=4096) {
                    const int end=std::min(count,begin+4096);
                    #pragma omp parallel for schedule(dynamic,16)
                    for(int i=begin;i<end;++i) {
                        if(failed.load())continue;
                        try {
                            IndependentPathSampler sampler(seed,false);sampler.startPath(pixelId,unsigned(i));
                            ConditionalGpis::Random random(sampler);
                            SingleHistoryGpis::Flight flight(medium->field(),birth,medium->flightSettings());
                            const auto hit=flight.sampleDistance(length,random);
                            if(hit.hit) {
                                if(!(hit.distance>=0&&hit.distance<=length)||!std::isfinite(hit.distance))
                                    throw std::runtime_error("Invalid first-hit distance");
                                distances[i]=hit.distance;
                            }
                        }catch(const std::exception &e) {
                            failed.store(true);std::lock_guard<std::mutex> lock(errorMutex);error=e.what();
                        }
                    }
                    if(failed.load())throw std::runtime_error(error);
                    std::cout<<method<<" / "<<id<<": "<<end<<" / "<<count<<std::endl;
                }
                records.push_back({{"id",id},{"method",method},{"ray",input.at("ray")},
                    {"input",path},{"samples",count},{"seed",seed},{"pixel_id",pixelId},
                    {"seconds",omp_get_wtime()-started},{"first_hit_distances",distances},
                    {"escaped_value",-1},{"source_medium",definition},
                    {"sampling","Production SingleHistoryGpis::Flight::sampleDistance; fresh ExteriorPositiveOnly start per draw; frozen double-precision ray; no hazard integration or camera resampling. Escape is -1. Phase is omitted because the original albedo is zero."}});
            }
            medium->teardownAfterRender();
        }
        const auto output=std::filesystem::absolute(argv[2]);std::filesystem::create_directories(output.parent_path());
        std::ofstream file(output);file<<J{{"spec",spec},{"records",records}}.dump();
        if(!file)throw std::runtime_error("Could not write output");return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<std::endl;return 1;}
}
