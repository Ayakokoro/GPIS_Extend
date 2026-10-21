#pragma once
#include "math/Mat4f.hpp"
#include <Eigen/Dense>
#include <memory>
#include <string>

namespace Tungsten {

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
    };
    ClassicLocalGrid(const std::string &file, const std::string &sdfName,
                     const std::string &densityName, const Mat4f &fieldToWorld, int macrocellSize);
    ~ClassicLocalGrid();
    double density(const Eigen::Vector3d &world) const;
    Eigen::Vector3d meanGradient(const Eigen::Vector3d &world) const;
};
}
