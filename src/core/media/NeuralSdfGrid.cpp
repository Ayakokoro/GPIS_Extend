#include "NeuralSdfGrid.hpp"
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/NanoVDB.h>
#include <nanovdb/io/IO.h>
#include <nanovdb/tools/GridStats.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace Tungsten {
using ConditionalGpis::Bounds3;
using ConditionalGpis::BoundsSummary;
using ConditionalGpis::Point3;
using ConditionalGpis::Vector3;
namespace {
// Index-node interpolation in native NVDB world coordinates. The neural model
// consumes this exact cubic along each ray, rather than a converted density.
class GridView {
public:
    // Reads `gridName` out of a .nvdb and caches everything the sampler needs.
    // Throws when the file or the named grid is missing, or when it is not a
    // float grid.
    static GridView open(const std::filesystem::path& path, const std::string& gridName) {
        GridView view;
        try {
            view.handle_ = nanovdb::io::readGrid<nanovdb::HostBuffer>(path.string(), gridName);
        } catch (const std::exception& error) {
            throw std::runtime_error("cannot read grid \"" + gridName + "\" from " + path.string() +
                                     ": " + error.what());
        }
        view.grid_ = view.handle_.grid<float>();
        if (!view.grid_) {
            throw std::runtime_error("grid \"" + gridName + "\" in " + path.string() +
                                     " is not a float grid");
        }
        view.dx_ = view.grid_->voxelSize()[0];
        if (!(view.dx_ > 0.0) || !std::isfinite(view.dx_)) {
            throw std::runtime_error("grid \"" + gridName + "\" has a degenerate voxel size");
        }
        const nanovdb::Vec3d origin = view.grid_->map().applyMap(nanovdb::Vec3d(0.0));
        view.origin_ = Point3(origin[0], origin[1], origin[2]);
        view.background_ = view.grid_->tree().root().background();

        // Active-value extremes. updateGridStats walks the tree once; the root
        // then carries the min/max of the active values.
        nanovdb::tools::updateGridStats(view.handle_.grid<float>());
        view.minimum_ = static_cast<double>(view.grid_->tree().root().minimum());
        view.maximum_ = static_cast<double>(view.grid_->tree().root().maximum());
        if (!(view.maximum_ >= view.minimum_)) {
            // An empty tree (the constant-grid case) reports both as zero.
            view.minimum_ = view.maximum_ = static_cast<double>(view.background_);
        }
        // A sub-domain's extremes are a subset of the grid's, so folding the
        // background in keeps bounds() conservative for both narrow-band and
        // full-domain fields.
        view.maximum_ = std::max(view.maximum_, static_cast<double>(view.background_));
        view.minimum_ = std::min(view.minimum_, static_cast<double>(view.background_));

        const nanovdb::CoordBBox bbox = view.grid_->indexBBox();
        view.worldBounds_.minimum = view.origin_ + view.dx_ * Point3(bbox.min()[0], bbox.min()[1],
                                                                     bbox.min()[2]);
        view.worldBounds_.maximum = view.origin_ + view.dx_ * Point3(bbox.max()[0] + 1,
                                                                     bbox.max()[1] + 1,
                                                                     bbox.max()[2] + 1);
        view.activeNodeBounds_.minimum = view.worldBounds_.minimum;
        view.activeNodeBounds_.maximum = view.origin_ + view.dx_ * Point3(bbox.max()[0],
                                                                          bbox.max()[1],
                                                                          bbox.max()[2]);
        return view;
    }

    double voxelSize() const { return dx_; }
    const Point3& origin() const { return origin_; }
    double background() const { return static_cast<double>(background_); }
    double minimumValue() const { return minimum_; }
    double maximumValue() const { return maximum_; }
    const Bounds3& worldBounds() const { return worldBounds_; }
    const Bounds3& activeNodeBounds() const { return activeNodeBounds_; }

    bool hasFullActiveNodeDomain() const {
        const auto bbox = grid_->indexBBox();
        std::uint64_t count = 1;
        for (int axis = 0; axis < 3; ++axis) {
            const std::int64_t size = std::int64_t(bbox.max()[axis])-bbox.min()[axis]+1;
            if (size < 2 || count > std::numeric_limits<std::uint64_t>::max()/std::uint64_t(size))
                return false;
            count *= std::uint64_t(size);
        }
        // Active voxels (including active tiles) are a subset of indexBBox.
        // Equality proves there are no missing nodes inside that box.
        return grid_->activeVoxelCount() == count;
    }

    bool hasCompleteInterpolationCell(const Point3& x) const {
        const auto u = continuousIndex(x);
        const auto bbox = grid_->indexBBox();
        int base[3];
        for (int axis = 0; axis < 3; ++axis) {
            // Check bounds before conversion to avoid overflow for far queries.
            if (!std::isfinite(u[axis]) || u[axis] < bbox.min()[axis] ||
                u[axis] >= bbox.max()[axis]) return false;
            base[axis] = static_cast<int>(std::floor(u[axis]));
        }
        auto accessor = grid_->getAccessor();
        for (int k = 0; k < 2; ++k)
            for (int j = 0; j < 2; ++j)
                for (int i = 0; i < 2; ++i) {
                    const nanovdb::Coord node(base[0] + i, base[1] + j, base[2] + k);
                    if (!accessor.isActive(node) || !std::isfinite(accessor.getValue(node)))
                        return false;
                }
        return true;
    }

    void sampleWithGradient(const Point3& x, double& value, Vector3& gradient) const {
        sampleWithGradientInCell(x, x, value, gradient);
    }

    void sampleWithGradientInCell(const Point3& x, const Point3& interior,
                                  double& value, Vector3& gradient, bool requireComplete = false) const {
        const nanovdb::Vec3d u = continuousIndex(x);
        const nanovdb::Vec3d cell = continuousIndex(interior);
        if (requireComplete) {
            const auto bbox = grid_->indexBBox();
            for (int a = 0; a < 3; ++a)
                if (!std::isfinite(cell[a]) || cell[a] < bbox.min()[a] || cell[a] >= bbox.max()[a])
                    throw std::invalid_argument("point query reaches outside stored SDF interpolation cells");
        }
        const nanovdb::Coord base(static_cast<int>(std::floor(cell[0])),
                                  static_cast<int>(std::floor(cell[1])),
                                  static_cast<int>(std::floor(cell[2])));
        const nanovdb::Vec3d f(u[0] - base[0], u[1] - base[1], u[2] - base[2]);

        auto accessor = grid_->getAccessor();
        double corner[2][2][2];
        for (int dk = 0; dk < 2; ++dk) {
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    const nanovdb::Coord node(base[0] + di, base[1] + dj, base[2] + dk);
                    const double sample = static_cast<double>(accessor.getValue(node));
                    if (requireComplete && (!accessor.isActive(node) || !std::isfinite(sample)))
                        throw std::invalid_argument("point query reaches missing/non-finite SDF interpolation nodes");
                    corner[di][dj][dk] = sample;
                }
            }
        }

        const double wx[2] = {1.0 - f[0], f[0]};
        const double wy[2] = {1.0 - f[1], f[1]};
        const double wz[2] = {1.0 - f[2], f[2]};
        const double dwx[2] = {-1.0, 1.0};
        const double dwy[2] = {-1.0, 1.0};
        const double dwz[2] = {-1.0, 1.0};

        value = 0.0;
        double du[3] = {0.0, 0.0, 0.0};
        for (int dk = 0; dk < 2; ++dk) {
            for (int dj = 0; dj < 2; ++dj) {
                for (int di = 0; di < 2; ++di) {
                    const double v = corner[di][dj][dk];
                    value += wx[di] * wy[dj] * wz[dk] * v;
                    du[0] += dwx[di] * wy[dj] * wz[dk] * v;
                    du[1] += wx[di] * dwy[dj] * wz[dk] * v;
                    du[2] += wx[di] * wy[dj] * dwz[dk] * v;
                }
            }
        }
        const double inverseDx = 1.0 / dx_;
        gradient = Vector3(du[0] * inverseDx, du[1] * inverseDx, du[2] * inverseDx);
    }

private:
    // World point -> continuous index of the stored sample nodes.
    nanovdb::Vec3d continuousIndex(const Point3& x) const {
        return nanovdb::Vec3d((x.x() - origin_.x()) / dx_,
                              (x.y() - origin_.y()) / dx_,
                              (x.z() - origin_.z()) / dx_);
    }

    nanovdb::GridHandle<nanovdb::HostBuffer> handle_;
    const nanovdb::NanoGrid<float>* grid_ = nullptr;
    double dx_ = 1.0;
    Point3 origin_ = Point3::Zero();
    float background_ = 0.0f;
    double minimum_ = 0.0;
    double maximum_ = 0.0;
    Bounds3 worldBounds_;
    Bounds3 activeNodeBounds_;
};

} // namespace

struct NeuralSdfGrid::Impl {
    GridView view;
    std::filesystem::path path;
};
NeuralSdfGrid::NeuralSdfGrid() : impl_(std::make_unique<Impl>()) {}
NeuralSdfGrid::~NeuralSdfGrid() = default;

std::shared_ptr<const NeuralSdfGrid> NeuralSdfGrid::open(const std::filesystem::path& file) {
    auto result = std::shared_ptr<NeuralSdfGrid>(new NeuralSdfGrid());
    result->impl_->path = file;
    result->impl_->view = GridView::open(file,"sdf");
    if (!result->impl_->view.hasFullActiveNodeDomain())
        throw std::invalid_argument("neural SDF requires full-domain active nodes: " + file.string());
    return result;
}

BoundsSummary NeuralSdfGrid::bounds(const Bounds3&) const {
    // A conservative global trilinear bound; neural flight integration itself
    // uses cell cubics and does not build a tracking majorant.
    const auto& grid = impl_->view;
    const double bound = std::sqrt(3.)*((grid.maximumValue()-grid.minimumValue())/grid.voxelSize());
    if (!std::isfinite(bound)) throw std::runtime_error("nonfinite neural SDF bound");
    return {grid.minimumValue(),std::nextafter(bound,std::numeric_limits<double>::infinity()),true};
}
double NeuralSdfGrid::voxelSizeHint() const { return impl_->view.voxelSize(); }
const Bounds3& NeuralSdfGrid::activeNodeBounds() const { return impl_->view.activeNodeBounds(); }

SdfJet NeuralSdfGrid::evaluate(const Point3& x) const {
    SdfJet jet;
    impl_->view.sampleWithGradient(x, jet.value, jet.gradient);
    if (!std::isfinite(jet.value) || !jet.gradient.allFinite()) {
        // Only reachable through a corrupt grid; degrade to the vacuum value
        // rather than propagating NaNs into the transport algebra.
        jet.value = impl_->view.background();
        jet.gradient = Vector3::Zero();
    }
    return jet;
}

SdfJet NeuralSdfGrid::evaluateInCell(const Point3& x, const Point3& interior) const {
    SdfJet jet;
    impl_->view.sampleWithGradientInCell(x, interior, jet.value, jet.gradient);
    if (!std::isfinite(jet.value) || !jet.gradient.allFinite())
        throw std::runtime_error("nonfinite NanoVDB ray-cell mean");
    return jet;
}

void NeuralSdfGrid::appendRayBreakpoints(const Point3& origin, const Vector3& direction,
                                       double begin, double end,
                                       std::vector<double>& knots) const {
    const double dx = impl_->view.voxelSize();
    const Point3& gridOrigin = impl_->view.origin();
    for (int axis = 0; axis < 3; ++axis) {
        const double speed = direction[axis];
        if (std::abs(speed) < 1e-14) continue;
        const double u0 = (origin[axis] + begin * speed - gridOrigin[axis]) / dx;
        const double u1 = (origin[axis] + end * speed - gridOrigin[axis]) / dx;
        const long long first = static_cast<long long>(std::floor(std::min(u0, u1))) + 1;
        const long long last = static_cast<long long>(std::ceil(std::max(u0, u1))) - 1;
        if (last - first > 100000) {
            throw std::invalid_argument("ray crosses too many NVDB interpolation cells");
        }
        for (long long index = first; index <= last; ++index) {
            const double coordinate = gridOrigin[axis] + static_cast<double>(index) * dx;
            const double age = (coordinate - origin[axis]) / speed;
            if (age > begin && age < end) knots.push_back(age);
        }
    }
}

void NeuralSdfGrid::requireFullRayCoverage(const Point3& origin, const Vector3& direction,
                                        double maximumDistance) const {
    if (!origin.allFinite() || !direction.allFinite() ||
        !(direction.squaredNorm() > 0.0) || !std::isfinite(maximumDistance) ||
        !(maximumDistance > 0.0)) {
        throw std::invalid_argument("invalid NanoVDB first-passage ray");
    }
    const auto covered = [&](double t) {
        if (!impl_->view.hasCompleteInterpolationCell(origin + t * direction)) {
            throw std::invalid_argument("NanoVDB first-passage ray reaches missing/non-finite "
                "SDF interpolation nodes at distance " + std::to_string(t) +
                "; shorten the ray or re-bake a larger full domain: " + impl_->path.string());
        }
    };
    std::vector<double> knots{0.0, maximumDistance};
    appendRayBreakpoints(origin, direction, 0.0, maximumDistance, knots);
    std::sort(knots.begin(), knots.end());
    // Every open interval is one interpolation cell. Its eight stored corners
    // cover its CLOSED endpoints too. Do not select an unrelated outside cell
    // by flooring an endpoint on the outermost stored node / upper grid face.
    for (std::size_t i = 1; i < knots.size(); ++i) {
        if (knots[i] > knots[i - 1]) covered(0.5 * (knots[i] + knots[i - 1]));
    }
}

} // namespace Tungsten
