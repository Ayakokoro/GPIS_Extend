#pragma once
#include "math/Mat4f.hpp"
#include "io/JsonPtr.hpp"
#include "io/Path.hpp"
#include <Eigen/Dense>
#include <memory>
#include <string>
#include <vector>

namespace Tungsten {

class Scene;
class ClassicLocalGrid;
struct ClassicLocalField {
    PathPtr file;
    std::string sdfName="sdf", densityName="density";
    Mat4f transform;
    int macrocellSize=8;
    int macrocellResolution=0; // Optional fixed number of blocks on each axis.
    void fromJson(JsonPtr field,const Scene &scene);
    rapidjson::Value toJson(rapidjson::Document::AllocatorType &allocator) const;
    std::unique_ptr<ClassicLocalGrid> load() const;
};

// Runtime reader for an existing baked field. No GP or field baking is done here.
class ClassicLocalGrid {
    struct Impl;
    std::unique_ptr<Impl> _impl;
public:
    struct Segment { double begin, end, densityMaximum; };
    class Cursor {
        const Impl *_grid;
        Eigen::Vector3d _origin, _direction;
        double _begin, _end;
    public:
        Cursor(const ClassicLocalGrid &grid, const Eigen::Vector3d &origin,
               const Eigen::Vector3d &direction, double begin, double end);
        bool next(Segment &segment);
        // Skip empty blocks and coalesce adjacent blocks with density support.
        bool nextOccupiedRange(double &begin, double &end);
        // Interpolation faces inside the clipped domain, for reference integration.
        void appendVoxelBoundaries(std::vector<double> &knots) const;
    };
    ClassicLocalGrid(const std::string &file, const std::string &sdfName,
                     const std::string &densityName, const Mat4f &fieldToWorld, int macrocellSize,
                     int macrocellResolution=0);
    ~ClassicLocalGrid();
    double density(const Eigen::Vector3d &world) const;
    Eigen::Vector3d meanGradient(const Eigen::Vector3d &world) const;
};
}
