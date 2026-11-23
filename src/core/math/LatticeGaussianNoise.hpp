#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace Tungsten {

// The convolution kernel is tabulated in units of ell. Its Hermite knots
// coincide with the coefficient lattice. Its two outer knots have zero value
// and slope, so the compactly supported kernel is continuously differentiable.
class LatticeGaussianKernel {
    friend class LatticeGaussianRay;
    using Cubic = std::array<float,4>;
    struct Knot { float value, slope; }; // slope per lattice interval
    int _radius;
    float _sigma, _originStddev, _invValueVariance, _invSlopeVariance, _boundPerCoefficient;
    float _transverseScale;
    double _originWeightNormSq;
    std::vector<Knot> _origin;
    std::vector<Cubic> _weights;

public:
    const float spacing;
    float originStddev() const { return _originStddev; }

    LatticeGaussianKernel(float sigma, float ell, float spacingInEll=0.5f, float radiusInEll=3.0f)
        : _sigma(sigma), _transverseScale(sigma/ell), spacing(ell*spacingInEll)
    {
        const float cells = std::ceil(radiusInEll/spacingInEll);
        if (!std::isfinite(sigma) || sigma < 0 || !std::isfinite(ell) || ell <= 0 ||
            !std::isfinite(spacing) || spacing <= 0 || !(cells >= 2 && cells <= 32768))
            throw std::invalid_argument("lattice_gaussian: invalid amplitude, spacing or kernel support");
        _radius = int(cells);
        // sqrt(Delta) * (2 / (pi * ell^2))^(1/4). Keep sigma outside
        // the convolution so the conditioning solve uses dimensionless rows.
        const float amplitude = std::sqrt(spacingInEll)*0.893243841738f;
        auto knot = [&](int offset) -> Knot {
            if (std::abs(offset) == _radius) return {0,0};
            const float x = offset*spacingInEll;
            const float value = amplitude*std::exp(-x*x);
            return {value,-2*x*spacingInEll*value};
        };

        double originNormSq = 0;
        float slopeVariance = 0, powerNorm = 0;
        Cubic controlNorm{};
        for (int j=1-_radius; j<=_radius; ++j) {
            // For segment i, coefficient i+j contributes this same polynomial.
            const Knot a = knot(-j), b = knot(1-j);
            const float delta = b.value-a.value;
            Cubic w{a.value, a.slope, 3*delta-2*a.slope-b.slope,
                    a.slope+b.slope-2*delta};
            _weights.push_back({a.value,a.slope,b.value,b.slope});
            if (j<_radius) {
                _origin.push_back(a);
                originNormSq += double(a.value)*a.value;
                slopeVariance += a.slope*a.slope;
            }
            Cubic controls{a.value,a.value+a.slope/3,b.value-b.slope/3,b.value};
            for (int k=0; k<4; ++k) {
                controlNorm[k] += std::abs(controls[k]);
                powerNorm += std::abs(w[k]);
            }
        }
        _originStddev = sigma*float(std::sqrt(originNormSq));
        _originWeightNormSq = originNormSq;
        _invValueVariance = float(1/originNormSq);
        _invSlopeVariance = 1/slopeVariance;
        if (!std::isfinite(_invValueVariance) || !std::isfinite(_invSlopeVariance))
            throw std::invalid_argument("lattice_gaussian: kernel cannot resolve both origin observations");
        // Convex hull in Bernstein form, padded for float dot products,
        // basis conversion and evaluation of the assembled power polynomial.
        const float roundoff = (4*_weights.size()+32)*std::numeric_limits<float>::epsilon();
        _boundPerCoefficient = sigma*(*std::max_element(controlNorm.begin(),controlNorm.end()) + roundoff*powerNorm);
    }

    // Renewal+ observes only the value and axial slope. The two independent
    // transverse components each have variance K_lattice(t,t)/ell^2.
    double transverseStddev(double u) const {
        const double v = 1-u;
        const double h00 = (1+2*u)*v*v, h10 = u*v*v;
        const double h01 = (1+2*v)*u*u, h11 = -v*u*u;
        double variance = 0;
        for (const auto &w : _weights) {
            const double weight = h00*w[0] + h10*w[1] + h01*w[2] + h11*w[3];
            variance += weight*weight;
        }
        return _transverseScale*std::sqrt(variance);
    }

    struct TransverseCondition {
        double meanScale;
        double stddev;
    };

    // Each transverse residual has covariance K_lattice(t,s)/ell^2.
    // Condition it on its observed residual at the birth knot (cell 0, u=0).
    TransverseCondition conditionedTransverse(int cell, double u) const {
        if (_sigma == 0) return {0,0};
        const int originCount = int(_origin.size()), width = int(_weights.size());
        if (cell >= originCount || cell <= -width) return {0,transverseStddev(u)};

        const double v = 1-u;
        const double h00 = (1+2*u)*v*v, h10 = u*v*v;
        const double h01 = (1+2*v)*u*u, h11 = -v*u*u;
        // Both rows use the same source indices. The hit row starts 'cell'
        // indices later than the birth row, even when a flight was subdivided.
        double cross = 0;
        for (int k=std::max(0,-cell); k<std::min(width,originCount-cell); ++k) {
            const auto &w = _weights[k];
            const double weight = h00*w[0] + h10*w[1] + h01*w[2] + h11*w[3];
            cross += weight*_origin[cell+k].value;
        }
        const double gain = cross/_originWeightNormSq;

        // Take the norm of the perpendicular row directly. Subtracting the
        // two marginal variances would lose the small variance near birth.
        double variance = 0;
        for (int j=std::min(0,cell); j<std::max(originCount,cell+width); ++j) {
            const double birth = j>=0 && j<originCount ? _origin[j].value : 0.0;
            const int k = j-cell;
            double residual = -gain*birth;
            if (k>=0 && k<width) {
                const auto &w = _weights[k];
                // w(0)-gain*birth is exactly zero at birth. Keep the small
                // Hermite increment separate so it survives this subtraction.
                residual = (w[0]-gain*birth) + h10*w[1]
                         + h01*(double(w[2])-w[0]) + h11*w[3];
            }
            variance += residual*residual;
        }
        return {gain,_transverseScale*std::sqrt(variance)};
    }
};

// Cache misses depend only on (seed,index), including after revisiting an evicted cell.
class LatticeGaussianSamples {
    std::uint64_t _key;
    static constexpr unsigned CacheSize = 32;
    mutable std::array<std::int64_t,CacheSize> _indices;
    mutable std::array<float,CacheSize> _values{};

    static std::uint64_t mix(std::uint64_t x) {
        // SplitMix64 finalizer. The two halves supply radius and angle.
        x = (x^(x>>30))*0xbf58476d1ce4e5b9ULL;
        x = (x^(x>>27))*0x94d049bb133111ebULL;
        return x^(x>>31);
    }
protected:
    // U >= 2^-32 in Box-Muller, so sqrt(-2 log(U)) <= 6.661.
    static constexpr float PriorCoefficientBound = 6.67f;
    float prior(std::int64_t index) const {
        const unsigned slot = static_cast<unsigned>(index) & (CacheSize-1);
        if (_indices[slot] != index) {
            const std::uint64_t hash = mix(std::uint64_t(index)+_key);
            constexpr float unit = 1.0f/4294967296.0f;
            const float u = (float(std::uint32_t(hash))+1)*unit;
            const float angle = float(hash>>32)*(6.28318530718f*unit);
            _values[slot] = std::sqrt(-2*std::log(u))*std::cos(angle);
            _indices[slot] = index;
        }
        return _values[slot];
    }
    explicit LatticeGaussianSamples(std::uint32_t seed)
        : _key(mix(std::uint64_t(seed)+0x9e3779b97f4a7c15ULL))
    {
        _indices.fill(std::numeric_limits<std::int64_t>::min());
    }
};

class LatticeGaussianRay : private LatticeGaussianSamples {
    const LatticeGaussianKernel &_kernel;
    std::vector<float> _conditioned;
    float _maxCoefficient = PriorCoefficientBound;
public:
    LatticeGaussianRay(const LatticeGaussianKernel &kernel, std::uint32_t seed)
        : LatticeGaussianSamples(seed), _kernel(kernel) {}

    // Both observations are residuals after subtracting the mean. At the
    // birth knot the value row is even and the derivative row is odd. Thus
    // AA^T is diagonal for this very spline, including the compact cutoff.
    void condition(float valueNoise, float directionalDerivativeNoise, bool includeDerivative=true) {
        if (_kernel._sigma == 0) return;
        const auto &rows = _kernel._origin;
        _conditioned.resize(rows.size());
        float value = 0, slope = 0;
        for (std::size_t k=0; k<rows.size(); ++k) {
            const float z = prior(std::int64_t(k)+1-_kernel._radius);
            _conditioned[k] = z;
            value += rows[k].value*z;
            slope += rows[k].slope*z;
        }
        const float valueScale = (valueNoise/_kernel._sigma-value)*_kernel._invValueVariance;
        const float slopeScale = includeDerivative ?
            (directionalDerivativeNoise*_kernel.spacing/_kernel._sigma-slope)*_kernel._invSlopeVariance : 0;
        _maxCoefficient = PriorCoefficientBound;
        for (std::size_t k=0; k<rows.size(); ++k) {
            _conditioned[k] += rows[k].value*valueScale + rows[k].slope*slopeScale;
            _maxCoefficient = std::max(_maxCoefficient,std::abs(_conditioned[k]));
        }
    }

    float absoluteBound() const { return _maxCoefficient*_kernel._boundPerCoefficient; }

    // For t in [cell*spacing,(cell+1)*spacing], evaluate in u=t/spacing-cell.
    std::array<double,4> segment(int cell) const {
        std::array<float,4> endpoints{};
        if (_kernel._sigma == 0) return {};
        const auto &weights = _kernel._weights;
        for (std::size_t k=0; k<weights.size(); ++k) {
            const auto relative = std::int64_t(cell)+std::int64_t(k);
            const float z = std::uint64_t(relative)<_conditioned.size() ?
                _conditioned[std::size_t(relative)] : prior(relative+1-_kernel._radius);
            for (int d=0; d<4; ++d) endpoints[d] += weights[k][d]*z;
        }
        // Adjacent cells sum their shared knot in the same order. Convert to
        // powers once, after the dot products, to preserve C1 knot agreement.
        for (float &v : endpoints) v *= _kernel._sigma;
        const double a=endpoints[0], b=endpoints[2], da=endpoints[1], db=endpoints[3];
        return {a,da,3*(b-a)-2*da-db,da+db-2*(b-a)};
    }
};

}
