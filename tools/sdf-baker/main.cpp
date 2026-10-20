#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tools/FastSweeping.h>
#include <openvdb/tools/MeshToVolume.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

template <typename T> T read(std::ifstream& input)
{
    T value;
    input.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!input) throw std::runtime_error("Unexpected end of PLY data");
    return value;
}

struct Mesh {
    std::vector<openvdb::Vec3s> vertices;
    std::vector<openvdb::Vec3I> triangles;
    openvdb::Vec3d min, max;
};

Mesh loadPly(const std::string& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open PLY: " + path);

    std::string line;
    std::size_t vertexCount = 0, faceCount = 0;
    bool binaryLittleEndian = false, headerEnded = false;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line == "format binary_little_endian 1.0") binaryLittleEndian = true;
        if (line.rfind("element vertex ", 0) == 0)
            vertexCount = std::stoull(line.substr(15));
        if (line.rfind("element face ", 0) == 0)
            faceCount = std::stoull(line.substr(13));
        if (line == "end_header") { headerEnded = true; break; }
    }
    if (!headerEnded || !binaryLittleEndian || !vertexCount || !faceCount)
        throw std::runtime_error("Expected binary little-endian PLY with vertices and faces");

    Mesh mesh;
    mesh.min = openvdb::Vec3d(std::numeric_limits<double>::max());
    mesh.max = openvdb::Vec3d(std::numeric_limits<double>::lowest());
    std::vector<std::uint32_t> remap(vertexCount);
    std::map<std::array<float, 3>, std::uint32_t> unique;
    for (std::size_t i = 0; i < vertexCount; ++i) {
        // This PLY has x, y, z, s, t, all float32. Weld UV seam vertices by position.
        std::array<float, 3> pos{read<float>(input), read<float>(input), read<float>(input)};
        read<float>(input); // s
        read<float>(input); // t
        auto [it, inserted] = unique.emplace(pos, static_cast<std::uint32_t>(mesh.vertices.size()));
        remap[i] = it->second;
        if (inserted) {
            mesh.vertices.emplace_back(pos[0], pos[1], pos[2]);
            for (int axis = 0; axis < 3; ++axis) {
                mesh.min[axis] = std::min(mesh.min[axis], double(pos[axis]));
                mesh.max[axis] = std::max(mesh.max[axis], double(pos[axis]));
            }
        }
    }

    mesh.triangles.reserve(faceCount);
    for (std::size_t i = 0; i < faceCount; ++i) {
        if (read<std::uint8_t>(input) != 3)
            throw std::runtime_error("Only triangle faces are supported");
        std::uint32_t a = read<std::uint32_t>(input);
        std::uint32_t b = read<std::uint32_t>(input);
        std::uint32_t c = read<std::uint32_t>(input);
        if (a >= vertexCount || b >= vertexCount || c >= vertexCount)
            throw std::runtime_error("Face index exceeds vertex count");
        mesh.triangles.emplace_back(remap[a], remap[b], remap[c]);
    }
    return mesh;
}

} // namespace

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "Usage: ply_sdf_baker input.ply output.vdb\n";
        return 2;
    }
    try {
        openvdb::initialize();
        const Mesh mesh = loadPly(argv[1]);
        constexpr int size = 100;
        constexpr double occupiedVoxels = 88.0; // 5.5 voxel centers of padding per side
        const auto center = (mesh.min + mesh.max) * 0.5;
        const auto extent = mesh.max - mesh.min;
        const double voxelSize = std::max({extent.x(), extent.y(), extent.z()}) / occupiedVoxels;
        const auto origin = center - openvdb::Vec3d((size - 1) * 0.5 * voxelSize);
        auto transform = openvdb::math::Transform::createLinearTransform(voxelSize);
        transform->postTranslate(origin);

        std::cout << "Welded vertices: " << mesh.vertices.size()
                  << ", triangles: " << mesh.triangles.size() << '\n';
        std::cout << "Voxel size: " << voxelSize << ", origin: " << origin << '\n';

        // OpenVDB computes accurate distances close to the surface and fills the
        // inactive interior with negative values. Sweep over the requested dense
        // domain to extend the signed distance throughout all 100^3 voxels.
        auto narrow = openvdb::tools::meshToLevelSet<openvdb::FloatGrid>(
            *transform, mesh.vertices, mesh.triangles, 3.0f);
        auto dense = openvdb::FloatGrid::create();
        dense->setTransform(transform->copy());
        auto source = narrow->getConstAccessor();
        auto target = dense->getAccessor();
        for (int z = 0; z < size; ++z)
            for (int y = 0; y < size; ++y)
                for (int x = 0; x < size; ++x) {
                    openvdb::Coord ijk(x, y, z);
                    target.setValue(ijk, source.getValue(ijk));
                }
        auto sdf = openvdb::tools::sdfToSdf(*dense, 0.0f, 2);
        sdf->setName("sdf");
        sdf->setGridClass(openvdb::GRID_LEVEL_SET);

        std::size_t negative = 0, positive = 0, nonfinite = 0;
        double minValue = std::numeric_limits<double>::infinity();
        double maxValue = -minValue;
        for (auto iter = sdf->cbeginValueOn(); iter; ++iter) {
            const double value = *iter;
            if (!std::isfinite(value)) ++nonfinite;
            if (value < 0) ++negative;
            if (value > 0) ++positive;
            minValue = std::min(minValue, value);
            maxValue = std::max(maxValue, value);
        }
        if (sdf->activeVoxelCount() != 1000000 || !negative || !positive || nonfinite)
            throw std::runtime_error("SDF validation failed");

        openvdb::GridPtrVec grids{sdf};
        openvdb::io::File output(argv[2]);
        output.write(grids);
        output.close();
        openvdb::io::File verify(argv[2]);
        verify.open();
        auto saved = openvdb::gridPtrCast<openvdb::FloatGrid>(verify.readGrid("sdf"));
        verify.close();
        if (!saved || saved->activeVoxelCount() != 1000000 ||
            saved->evalActiveVoxelBoundingBox().min() != openvdb::Coord(0, 0, 0) ||
            saved->evalActiveVoxelBoundingBox().max() != openvdb::Coord(99, 99, 99))
            throw std::runtime_error("Saved VDB read-back validation failed");
        std::cout << "Wrote " << argv[2] << ": " << sdf->activeVoxelCount()
                  << " active voxels, negative=" << negative << ", positive=" << positive
                  << ", value range=[" << minValue << ", " << maxValue << "]\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "SDF baking failed: " << error.what() << '\n';
        return 1;
    }
}
