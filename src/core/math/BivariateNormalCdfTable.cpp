#include "BivariateNormalCdfTable.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>

namespace Tungsten::SparseHistoryGpis {
namespace {
template<typename T> T read(std::istream &stream)
{
    T value;
    if (!stream.read(reinterpret_cast<char *>(&value), sizeof(T)))
        throw std::runtime_error("Truncated bivariate Gaussian CDF table");
    return value;
}

std::array<double, 4> weights(double t)
{
    return {{-t*(t - 1.0)*(t - 2.0)/6.0,
             (t + 1.0)*(t - 1.0)*(t - 2.0)/2.0,
            -(t + 1.0)*t*(t - 2.0)/2.0,
             (t + 1.0)*t*(t - 1.0)/6.0}};
}
}

BivariateNormalCdfTable::BivariateNormalCdfTable(const std::string &path)
: _sourcePath(path)
{
    const std::uint16_t endian = 1;
    if (*reinterpret_cast<const unsigned char *>(&endian) != 1)
        throw std::runtime_error("Bivariate Gaussian CDF table requires little-endian storage");
    std::ifstream input(path, std::ios::binary);
    if (!input)
        throw std::runtime_error("Cannot open bivariate Gaussian CDF table: " + path);
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    if (magic != std::array<char, 8>{{'G','P','B','C','D','F','1','\0'}} ||
        read<std::uint32_t>(input) != 1)
        throw std::runtime_error("Invalid bivariate Gaussian CDF table version");
    _n = read<std::uint32_t>(input);
    _nt = read<std::uint32_t>(input);
    if (read<std::uint32_t>(input) != 0)
        throw std::runtime_error("Unsupported bivariate Gaussian CDF table flags");
    _minimum = read<double>(input);
    _step = read<double>(input);
    _thetaMinimum = read<double>(input);
    _thetaStep = read<double>(input);
    const auto nodes = read<std::uint64_t>(input);
    const auto cells = read<std::uint64_t>(input);
    if (_n < 4 || _n > 4096 || _nt < 4 || _nt > 4096 ||
        !std::isfinite(_minimum) || !std::isfinite(_thetaMinimum) ||
        !std::isfinite(_step) || !(_step > 0.0) ||
        !std::isfinite(_thetaStep) || !(_thetaStep > 0.0) ||
        _thetaMinimum <= -1.5707963267948966 ||
        _thetaMinimum + (_nt - 1)*_thetaStep >= 1.5707963267948966 ||
        nodes != std::uint64_t(_n)*(_n + 1)/2*_nt ||
        cells != std::uint64_t(_n - 3)*(_n - 2)/2*(_nt - 3) ||
        nodes + cells > 64*1024*1024)
        throw std::runtime_error("Invalid bivariate Gaussian CDF table dimensions");
    _values.resize(static_cast<std::size_t>(nodes));
    _errors.resize(static_cast<std::size_t>(cells));
    if (!input.read(reinterpret_cast<char *>(_values.data()), _values.size()*sizeof(float)) ||
        !input.read(reinterpret_cast<char *>(_errors.data()), _errors.size()*sizeof(float)) ||
        input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Invalid bivariate Gaussian CDF table payload size");
    for (float value : _values)
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f)
            throw std::runtime_error("Invalid bivariate Gaussian CDF table value");
    for (float error : _errors)
        if (!std::isfinite(error) || error < 0.0f)
            throw std::runtime_error("Invalid bivariate Gaussian CDF table error estimate");
}

std::shared_ptr<const BivariateNormalCdfTable>
BivariateNormalCdfTable::loadShared(const std::string &path)
{
    static std::mutex mutex;
    static std::map<std::string, std::weak_ptr<const BivariateNormalCdfTable>> tables;
    const auto canonical = std::filesystem::absolute(path).lexically_normal().string();
    std::lock_guard<std::mutex> lock(mutex);
    if (auto existing = tables[canonical].lock())
        return existing;
    std::shared_ptr<const BivariateNormalCdfTable> result(new BivariateNormalCdfTable(canonical));
    tables[canonical] = result;
    return result;
}

std::size_t BivariateNormalCdfTable::nodeIndex(unsigned a, unsigned b, unsigned t) const
{
    if (a > b) std::swap(a, b);
    return std::size_t(t)*_n*(_n + 1)/2 + std::size_t(b)*(b + 1)/2 + a;
}

std::size_t BivariateNormalCdfTable::cellIndex(unsigned a, unsigned b, unsigned t) const
{
    --a; --b; --t;
    if (a > b) std::swap(a, b);
    return std::size_t(t)*(_n - 3)*(_n - 2)/2 + std::size_t(b)*(b + 1)/2 + a;
}

std::optional<BivariateNormalCdfTable::Estimate>
BivariateNormalCdfTable::lookup(double a, double b, double rho, double relativeBudget) const
{
    if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(rho) || std::abs(rho) >= 1.0 ||
        !(relativeBudget >= 0.0))
        return std::nullopt;
    const double x = (a - _minimum)/_step;
    const double y = (b - _minimum)/_step;
    const double z = (std::asin(rho) - _thetaMinimum)/_thetaStep;
    // Cubic stencils require one guard node on each side. The omitted domain,
    // including correlations close to +/-1, stays with the numerical backend.
    if (x < 1.0 || y < 1.0 || z < 1.0 || x >= _n - 2.0 ||
        y >= _n - 2.0 || z >= _nt - 2.0)
        return std::nullopt;
    const unsigned i = static_cast<unsigned>(std::floor(x));
    const unsigned j = static_cast<unsigned>(std::floor(y));
    const unsigned k = static_cast<unsigned>(std::floor(z));
    const double cellError = _errors[cellIndex(i, j, k)];
    const bool hasBudget = std::isfinite(relativeBudget);
    if (hasBudget) {
        // Phi_2 is increasing in a, b and rho. The upper cell corner therefore
        // bounds every exact probability in this cell; cellError also covers
        // the corner's float quantization and numerical node uncertainty.
        const double upper = _values[nodeIndex(i + 1, j + 1, k + 1)] + cellError;
        if (cellError > relativeBudget*upper)
            return std::nullopt;
    }
    const auto wa = weights(x - i), wb = weights(y - j), wt = weights(z - k);
    std::array<std::size_t, 16> planeIndices;
    for (unsigned jb = 0; jb < 4; ++jb)
        for (unsigned ia = 0; ia < 4; ++ia)
            planeIndices[4*jb + ia] = nodeIndex(i + ia - 1, j + jb - 1, 0);
    double value = 0.0, magnitude = 0.0;
    for (unsigned kt = 0; kt < 4; ++kt) {
        const std::size_t plane = std::size_t(k + kt - 1)*_n*(_n + 1)/2;
        double planeValue = 0.0, planeMagnitude = 0.0;
        for (unsigned jb = 0; jb < 4; ++jb) {
            double rowValue = 0.0, rowMagnitude = 0.0;
            for (unsigned ia = 0; ia < 4; ++ia) {
                const double term = wa[ia]*_values[plane + planeIndices[4*jb + ia]];
                rowValue += term;
                rowMagnitude += std::abs(term);
            }
            planeValue += wb[jb]*rowValue;
            planeMagnitude += std::abs(wb[jb])*rowMagnitude;
        }
        value += wt[kt]*planeValue;
        magnitude += std::abs(wt[kt])*planeMagnitude;
    }
    const double error = cellError + 128.0*std::numeric_limits<double>::epsilon()*magnitude;
    if (!(value > error) || !std::isfinite(value) ||
        (hasBudget && error > relativeBudget*(value - error)))
        return std::nullopt;
    // Projection onto the exact Frechet interval cannot increase the absolute
    // interpolation error. It also preserves the CDF bounds in nearly nested
    // events, where cubic interpolation can otherwise slightly overshoot.
    const double pa = 0.5*std::erfc(-a*0.70710678118654752440);
    const double pb = 0.5*std::erfc(-b*0.70710678118654752440);
    value = std::clamp(value, std::max(0.0, pa + pb - 1.0), std::min(pa, pb));
    if (!(value > error) || value > 1.0 ||
        (hasBudget && error > relativeBudget*(value - error)))
        return std::nullopt;
    return Estimate{value, error};
}

std::size_t BivariateNormalCdfTable::sizeBytes() const
{
    return (_values.size() + _errors.size())*sizeof(float);
}

} // namespace Tungsten::SparseHistoryGpis
