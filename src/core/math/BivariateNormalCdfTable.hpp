#pragma once

#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Tungsten::SparseHistoryGpis {

// A scene-independent, immutable table for Phi_2(a,b;rho). The interpolation
// error travels with each value; callers must still apply their own relative
// probability error budget before using it in a ratio or a signed moment.
class BivariateNormalCdfTable {
public:
    struct Estimate { double value; double error; };
    static std::shared_ptr<const BivariateNormalCdfTable>
        loadShared(const std::string &path);
    // A finite relative budget enables rejection before the 64-node lookup.
    // Returned estimates then satisfy error <= budget*(value - error).
    std::optional<Estimate> lookup(double a, double b, double rho,
        double relativeBudget = std::numeric_limits<double>::infinity()) const;
    std::size_t sizeBytes() const;
    const std::string &sourcePath() const { return _sourcePath; }

private:
    explicit BivariateNormalCdfTable(const std::string &path);
    std::string _sourcePath;
    unsigned _n = 0, _nt = 0;
    double _minimum = 0.0, _step = 0.0;
    double _thetaMinimum = 0.0, _thetaStep = 0.0;
    std::vector<float> _values, _errors;
    std::size_t nodeIndex(unsigned a, unsigned b, unsigned t) const;
    std::size_t cellIndex(unsigned a, unsigned b, unsigned t) const;
};

} // namespace Tungsten::SparseHistoryGpis
