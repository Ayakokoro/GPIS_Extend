// Convert one existing float grid without resampling or changing its transform.
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/NanoToOpenVDB.h>
#include <openvdb/io/File.h>
#include <nlohmann/json.hpp>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_reduce.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace fs = std::filesystem;
using Json = nlohmann::json;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
bool sameBits(float a, float b) { return std::memcmp(&a, &b, sizeof(float)) == 0; }
struct Counts {
    uint64_t nodes=0, active=0, valueErrors=0, stateErrors=0, nonfinite=0;
    Counts operator+(const Counts &other) const {
        return {nodes+other.nodes, active+other.active, valueErrors+other.valueErrors,
                stateErrors+other.stateErrors, nonfinite+other.nonfinite};
    }
};

int main(int argc, char **argv) {
    try {
        require(argc==4, "Usage: gpis_nvdb_to_vdb input.nvdb output.vdb grid_name");
        const fs::path input=fs::absolute(argv[1]), output=fs::absolute(argv[2]);
        const std::string name=argv[3];
        require(fs::is_regular_file(input), "Input grid does not exist");
        require(!fs::exists(output), "Output already exists; refusing to overwrite");
        require(!fs::exists(output.string()+".json"), "Output metadata already exists");
        const auto start=std::chrono::steady_clock::now();
        tbb::global_control threads(tbb::global_control::max_allowed_parallelism, 16);
        openvdb::initialize();
        std::cout << "Reading named NanoVDB grid: " << name << std::endl;
        auto handle=nanovdb::io::readGrid<nanovdb::HostBuffer>(input.string(), name);
        const auto *nano=handle.grid<float>();
        require(nano!=nullptr, "Selected grid is missing or not float32");
        const auto box=nano->indexBBox();
        std::cout << "Converting existing nodes; no resampling" << std::endl;
        auto converted=nanovdb::tools::nanoToOpenVDB(*nano);
        require(converted && converted->getName()==name, "Grid name changed");
        converted->setSaveFloatAsHalf(false);
        fs::create_directories(output.parent_path());
        std::cout << "Writing lossless OpenVDB: " << output.string() << std::endl;
        {
            openvdb::io::File file(output.string());
            file.setCompression(openvdb::io::COMPRESS_ZIP | openvdb::io::COMPRESS_ACTIVE_MASK);
            file.write(openvdb::GridPtrVec{converted});
        }
        converted.reset();
        std::cout << "Reloading saved file and verifying every node" << std::endl;
        openvdb::io::File file(output.string());
        file.open(false);
        const auto grids=file.getGrids();
        require(grids->size()==1, "Output must contain exactly one grid");
        auto saved=openvdb::gridPtrCast<openvdb::FloatGrid>((*grids)[0]);
        require(saved && saved->getName()==name, "Saved grid type or name changed");
        file.close();
        const auto savedBox=saved->evalActiveVoxelBoundingBox();
        for (int axis=0; axis<3; ++axis) {
            require(savedBox.min()[axis]==box.min()[axis] && savedBox.max()[axis]==box.max()[axis],
                    "Active bounding box changed");
        }
        require(saved->activeVoxelCount()==nano->activeVoxelCount(), "Active voxel count changed");
        require(sameBits(saved->background(), nano->tree().root().background()), "Background changed");
        double transformError=0;
        for (int point=0; point<12; ++point) {
            nanovdb::Vec3d index(0);
            if (point<8) {
                for(int axis=0; axis<3; ++axis) index[axis]=(point&(1<<axis)) ? box.max()[axis] : box.min()[axis];
            } else if (point<11) index[point-8]=1;
            const auto a=nano->indexToWorld(index);
            const auto b=saved->indexToWorld(openvdb::Vec3d(index[0],index[1],index[2]));
            for(int axis=0; axis<3; ++axis) transformError=std::max(transformError,std::abs(a[axis]-b[axis]));
        }
        require(transformError<=1e-12, "Index-to-world transform changed");
        const auto counts=tbb::parallel_reduce(tbb::blocked_range<int>(box.min()[2],box.max()[2]+1), Counts{},
            [&](const tbb::blocked_range<int> &range, Counts result) {
                auto source=nano->getAccessor();
                auto target=saved->getConstAccessor();
                for(int z=range.begin(); z<range.end(); ++z)
                    for(int y=box.min()[1]; y<=box.max()[1]; ++y)
                        for(int x=box.min()[0]; x<=box.max()[0]; ++x) {
                            float a,b;
                            const bool aa=source.probeValue(nanovdb::Coord(x,y,z),a);
                            const bool ab=target.probeValue(openvdb::Coord(x,y,z),b);
                            ++result.nodes;
                            result.active+=aa;
                            result.valueErrors+=!sameBits(a,b);
                            result.stateErrors+=aa!=ab;
                            result.nonfinite+=!std::isfinite(a) || !std::isfinite(b);
                        }
                return result;
            }, [](Counts a, Counts b) { return a+b; });
        require(counts.valueErrors==0 && counts.stateErrors==0 && counts.nonfinite==0,
                "Node verification failed");
        const auto origin=nano->indexToWorld(nanovdb::Vec3d(0));
        const auto spacing=saved->voxelSize();
        const Json report={
            {"complete",true},{"source",input.string()},{"output",output.string()},
            {"grid_name",name},{"grid_count",1},{"value_type","float32"},{"resampled",false},
            {"compression","lossless ZIP + active mask"},
            {"index_min",{box.min()[0],box.min()[1],box.min()[2]}},
            {"index_max",{box.max()[0],box.max()[1],box.max()[2]}},
            {"origin",{origin[0],origin[1],origin[2]}},
            {"voxel_size",{spacing[0],spacing[1],spacing[2]}},
            {"background",saved->background()},
            {"verified_nodes",counts.nodes},{"active_nodes",counts.active},
            {"bitwise_value_mismatches",counts.valueErrors},{"active_state_mismatches",counts.stateErrors},
            {"nonfinite_nodes",counts.nonfinite},{"transform_max_abs_error",transformError},
            {"output_bytes",fs::file_size(output)},
            {"elapsed_seconds",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};
        std::ofstream metadata(output.string()+".json");
        metadata << report.dump(2) << '\n';
        require(bool(metadata), "Could not write verification report");
        std::cout << report.dump(2) << std::endl;
        return 0;
    } catch(const std::exception &error) {
        std::cerr << error.what() << std::endl;
        return 1;
    }
}
