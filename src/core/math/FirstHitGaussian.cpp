#include "math/FirstHitGaussian.hpp"
#include "math/conditional/mathutility/Quadrature.h"

#include <sobol/sobol.h>
#include <boost/math/special_functions/erf.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace Tungsten::FirstHitGpis {
namespace {

constexpr double Pi = 3.1415926535897932384626433832795;
constexpr double SqrtTwo = 1.4142135623730950488016887242097;
constexpr double NormalLimit = 9.0;
constexpr int Replicates = 8;

class QmcTimer {
    OrthantDiagnostics* _diagnostics;
    std::chrono::steady_clock::time_point _begin;
public:
    explicit QmcTimer(OrthantDiagnostics* diagnostics) : _diagnostics(diagnostics)
    {
        if (_diagnostics) _begin = std::chrono::steady_clock::now();
    }

    ~QmcTimer() { stop(); }

    void stop()
    {
        if (_diagnostics) {
            _diagnostics->qmcNanoseconds += static_cast<std::uint64_t>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - _begin).count());
            _diagnostics = nullptr;
        }
    }
};

using ConditionalGpis::NumericError;
using ConditionalGpis::NumericStatus;

double density(double x)
{
    return std::exp(-0.5*x*x)/std::sqrt(2.0*Pi);
}

double correlationAngle(double separation)
{
    const double squaredDistance = separation*separation;
    return std::atan2(std::sqrt(-std::expm1(-squaredDistance)),
                      std::exp(-0.5*squaredDistance));
}

double intervalProbability(double lower, double upper)
{
    if (!(upper > lower)) return 0.0;
    const double width = upper - lower;
    const double midpoint = 0.5*lower + 0.5*upper;
    // CDF subtraction loses the narrow crossing bands occurring as rho -> 1.
    // The midpoint expansion retains their first-order probability mass.
    if (width*(1.0 + std::abs(midpoint)) < 1e-4) {
        return width*density(midpoint)*
            (1.0 + (midpoint*midpoint - 1.0)*width*width/24.0);
    }
    if (lower >= 0.0) return std::max(0.0, cdf(-lower) - cdf(-upper));
    return std::max(0.0, cdf(upper) - cdf(lower));
}

double quantile(double probability)
{
    // erfc_inv avoids forming 2*p-1 in the small tail. The Genz transform
    // only calls this with a strictly interior, representable probability.
    return probability <= 0.5
        ? -SqrtTwo*boost::math::erfc_inv(2.0*probability)
        :  SqrtTwo*boost::math::erfc_inv(2.0*(1.0 - probability));
}

double truncatedNormalMean(double normalizedMean)
{
    if (normalizedMean > -8.0) return density(normalizedMean)/cdf(normalizedMean);
    // E[Z | Z > -a] = phi(a)/Phi(a). Use the inverse Mills ratio's
    // continued fraction when numerator and denominator are tiny.
    const double x = -normalizedMean;
    double tail = 0.0;
    for (int k = 64; k >= 1; --k) tail = static_cast<double>(k)/(x + tail);
    return x + tail;
}

ProbabilityEstimate integrate(const std::function<double(double)>& function,
                              double lower, double upper)
{
    if (!(upper > lower)) return {};
    ConditionalGpis::NumericPolicy policy;
    policy.absoluteTolerance = 1e-13;
    policy.relativeTolerance = 1e-9;
    policy.maxQuadratureSubdivisions = 512;
    const auto result = ConditionalGpis::integrateFinite(function, lower, upper, policy);
    if (result.status != NumericStatus::Ok)
        throw NumericError(result.status, "first-hit Gaussian pair quadrature did not converge");
    return {result.value, result.absError};
}

void add(ProbabilityEstimate& a, const ProbabilityEstimate& b)
{
    a.value += b.value;
    a.error += b.error;
}

ProbabilityEstimate positivePair(double meanA, double meanB, double separation)
{
    if (separation == 0.0) return {cdf(std::min(meanA, meanB)), 0.0};
    const double rho = std::exp(-0.5*separation*separation);
    if (rho == 0.0) return {cdf(meanA)*cdf(meanB), 0.0};
    if (meanA == 0.0 && meanB == 0.0) {
        const double angle = correlationAngle(separation);
        return {0.5 - angle/(2.0*Pi), 8.0*std::numeric_limits<double>::epsilon()};
    }
    const double differenceStddev = std::sqrt(-0.5*std::expm1(-0.5*separation*separation));
    const double averageStddev = std::sqrt(1.0 - differenceStddev*differenceStddev);
    if (differenceStddev == 0.0) return {cdf(std::min(meanA, meanB)), 0.0};
    // The average and half-difference of two equal-variance Gaussians are
    // independent. Condition on the difference and integrate the average.
    const auto integrand = [&](double z) {
        const double threshold = std::min(meanA + differenceStddev*z,
                                          meanB - differenceStddev*z);
        return density(z)*cdf(threshold/averageStddev);
    };
    const double split = std::clamp((meanB - meanA)/(2.0*differenceStddev),
                                    -NormalLimit, NormalLimit);
    auto result = integrate(integrand, -NormalLimit, split);
    add(result, integrate(integrand, split, NormalLimit));
    result.error += 2.0*cdf(-NormalLimit);
    result.value = std::clamp(result.value, 0.0, std::min(cdf(meanA), cdf(meanB)));
    return result;
}

std::uint32_t hash(std::uint32_t x)
{
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    return x ^ (x >> 16);
}

} // namespace

double cdf(double x)
{
    return 0.5*std::erfc(-x/SqrtTwo);
}

ProbabilityEstimate downcrossingPair(double meanA, double meanB, double separation)
{
    if (!std::isfinite(meanA) || !std::isfinite(meanB) ||
        !std::isfinite(separation) || separation < 0.0)
        throw NumericError(NumericStatus::InvalidInput, "invalid first-hit Gaussian pair inputs");
    if (separation == 0.0) return {intervalProbability(-meanA, -meanB), 0.0};
    const double squaredDistance = separation*separation;
    const double rho = std::exp(-0.5*squaredDistance);
    if (rho == 0.0) return {cdf(meanA)*cdf(-meanB), 0.0};
    if (meanA == 0.0 && meanB == 0.0) {
        const double value = correlationAngle(separation)/(2.0*Pi);
        return {value, 8.0*std::numeric_limits<double>::epsilon()*value};
    }
    const double differenceStddev = std::sqrt(-0.5*std::expm1(-0.5*squaredDistance));
    const double averageStddev = std::sqrt(1.0 - differenceStddev*differenceStddev);
    if (differenceStddev == 0.0) return {intervalProbability(-meanA, -meanB), 0.0};
    const double threshold = (meanB - meanA)/(2.0*differenceStddev);
    if (threshold >= NormalLimit) return {0.0, cdf(-threshold)};
    const double lower = std::max(-NormalLimit, threshold);
    const auto integrand = [&](double z) {
        const double bandLower = (-meanA - differenceStddev*z)/averageStddev;
        const double bandUpper = (-meanB + differenceStddev*z)/averageStddev;
        return density(z)*intervalProbability(bandLower, bandUpper);
    };
    auto result = integrate(integrand, lower, NormalLimit);
    result.error += cdf(-NormalLimit) + (threshold < -NormalLimit ? cdf(-NormalLimit) : 0.0);
    result.value = std::clamp(result.value, 0.0, std::min(cdf(meanA), cdf(-meanB)));
    return result;
}

struct OrthantIntegrator::Impl {
    std::vector<double> positions;
    std::vector<double> means;
    int seed;
    bool prepared = false;
    bool usesQmc = false;
    ProbabilityEstimate analyticEstimate;
    double omittedProbability = 0.0;
    std::vector<std::size_t> order;
    std::vector<double> factor;
    std::vector<double> z;
    std::size_t rank = 0;
    double roundoff = 0.0;
    double factorError = 0.0;
    int computedSamples = 0;
    std::array<std::vector<std::uint32_t>, Replicates> shifts;
    // Entry n is the cumulative weight of exactly the first n points. Keep
    // all prefixes so arbitrary smaller/odd budgets retain the original
    // summation order and the original half-budget convergence check.
    std::array<std::vector<double>, Replicates> prefixSums;

    Impl(const std::vector<double>& positions_, const std::vector<double>& means_, int seed_)
    : positions(positions_), means(means_), seed(seed_)
    {
    }

    void prepare();
    ProbabilityEstimate extendTo(int samples, OrthantDiagnostics* diagnostics);
};

void OrthantIntegrator::Impl::prepare()
{
    std::size_t count = positions.size();
    if (count != means.size() || count > sobol::Matrices::num_dimensions)
        throw NumericError(NumericStatus::InvalidInput, "invalid first-hit Gaussian orthant dimensions or sample count");
    for (std::size_t i = 0; i < count; ++i)
        if (!std::isfinite(positions[i]) || !std::isfinite(means[i]))
            throw NumericError(NumericStatus::InvalidInput, "non-finite first-hit Gaussian orthant input");
    // Almost-certain air constraints waste integration dimensions and can
    // move important surface constraints into poorly conditioned late rows.
    // Their omission is charged explicitly by the union bound, even though
    // each individual probability is below 6.3e-16 at this threshold.
    // As in the one-shot implementation, original one- and two-dimensional
    // problems use their dedicated formula without removing air constraints.
    if (count >= 3) {
        std::vector<double> retainedPositions, retainedMeans;
        for (std::size_t i = 0; i < count; ++i) {
            if (means[i] >= 8.0) omittedProbability += cdf(-means[i]);
            else { retainedPositions.push_back(positions[i]); retainedMeans.push_back(means[i]); }
        }
        if (retainedPositions.size() != count) {
            positions.swap(retainedPositions);
            means.swap(retainedMeans);
            count = positions.size();
        }
    }
    if (count == 0) { analyticEstimate = {1.0, 0.0}; return; }
    if (count == 1) { analyticEstimate = {cdf(means[0]), 0.0}; return; }
    if (count == 2) {
        analyticEstimate = positivePair(means[0], means[1], std::abs(positions[0] - positions[1]));
        return;
    }

    std::vector<double> covariance(count*count);
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            const double separation = positions[i] - positions[j];
            covariance[i*count + j] = covariance[j*count + i] = std::exp(-0.5*separation*separation);
        }
    }
    if (count == 3 && means[0] == 0.0 && means[1] == 0.0 && means[2] == 0.0) {
        analyticEstimate = {0.5 - (correlationAngle(positions[0] - positions[1]) +
                       correlationAngle(positions[0] - positions[2]) +
                       correlationAngle(positions[1] - positions[2]))/(4.0*Pi),
                16.0*std::numeric_limits<double>::epsilon()};
        return;
    }

    order.resize(count);
    std::iota(order.begin(), order.end(), 0);
    factor.assign(count*count, 0.0);
    std::vector<double> residualVariance(count, 1.0);
    std::vector<double> proxyNormalMean(count, 0.0);
    roundoff = 64.0*std::numeric_limits<double>::epsilon()*count;
    rank = count;
    factorError = 0.0;
    for (std::size_t column = 0; column < count; ++column) {
        std::size_t pivot = column;
        for (std::size_t row = column + 1; row < count; ++row)
            if (residualVariance[row] > residualVariance[pivot]) pivot = row;
        const double maximumVariance = residualVariance[pivot];
        if (maximumVariance > roundoff) {
            // Genz ordering: constrain the most restrictive remaining
            // variable first, estimating earlier truncated normals by their
            // conditional means. Maximum-variance ordering alone places
            // innocuous air points first and leaves almost binary surface
            // constraints at the end, greatly increasing QMC variation.
            // Exclude unresolved pivots and extremely small pivots relative
            // to other available rows to retain numerical stability.
            const double pivotFloor = std::max(roundoff, maximumVariance*1e-6);
            double best = std::numeric_limits<double>::infinity();
            for (std::size_t row = column; row < count; ++row) {
                if (residualVariance[row] <= pivotFloor) continue;
                double conditionalMean = means[order[row]];
                for (std::size_t j = 0; j < column; ++j)
                    conditionalMean += factor[row*count + j]*proxyNormalMean[j];
                const double score = conditionalMean/std::sqrt(residualVariance[row]);
                if (score < best) { best = score; pivot = row; }
            }
        }
        std::swap(order[column], order[pivot]);
        std::swap(residualVariance[column], residualVariance[pivot]);
        for (std::size_t j = 0; j < column; ++j)
            std::swap(factor[column*count + j], factor[pivot*count + j]);
        if (residualVariance[column] < -roundoff)
            throw NumericError(NumericStatus::InvalidCovariance, "first-hit SE factorization lost positive semidefiniteness");
        if (residualVariance[column] <= roundoff) {
            // Dense SE samples have almost dependent values. Keep the
            // already factored linear constraints and treat only residual
            // variances below floating-point resolution as deterministic.
            // This is numerical rank truncation, NOT a diagonal jitter.
            rank = column;
            for (std::size_t row = column; row < count; ++row) {
                if (residualVariance[row] < -roundoff)
                    throw NumericError(NumericStatus::InvalidCovariance, "negative first-hit SE residual variance");
                const double residual = std::max(0.0, residualVariance[row]) + roundoff;
                // For an independent residual Z, the chance that adding Z
                // changes a sign is <= E|Z| * maximum Gaussian density.
                // Include this allowance in the numerical error estimate.
                factorError += std::sqrt(residual)/(Pi*std::sqrt(std::max(0.5, 1.0 - residual)));
            }
            break;
        }
        const double diagonal = std::sqrt(residualVariance[column]);
        factor[column*count + column] = diagonal;
        double proxyMean = means[order[column]];
        for (std::size_t j = 0; j < column; ++j)
            proxyMean += factor[column*count + j]*proxyNormalMean[j];
        proxyNormalMean[column] = truncatedNormalMean(proxyMean/diagonal);
        for (std::size_t row = column + 1; row < count; ++row) {
            double entry = covariance[order[row]*count + order[column]];
            for (std::size_t j = 0; j < column; ++j)
                entry -= factor[row*count + j]*factor[column*count + j];
            entry /= diagonal;
            factor[row*count + column] = entry;
            residualVariance[row] -= entry*entry;
        }
    }

    z.assign(rank, 0.0);
    for (int replicate = 0; replicate < Replicates; ++replicate) {
        shifts[replicate].resize(count);
        for (std::size_t dimension = 0; dimension < count; ++dimension)
            shifts[replicate][dimension] = hash(static_cast<std::uint32_t>(seed) ^
                hash(0x9e3779b9U*static_cast<std::uint32_t>(replicate + 1)) ^
                hash(0x85ebca6bU*static_cast<std::uint32_t>(dimension + 1)));
        prefixSums[replicate].assign(1, 0.0);
    }
    usesQmc = true;
}

ProbabilityEstimate OrthantIntegrator::Impl::extendTo(int samples,
                                                     OrthantDiagnostics* diagnostics)
{
    if (diagnostics) *diagnostics = {};
    if (samples < 1)
        throw NumericError(NumericStatus::InvalidInput, "invalid first-hit Gaussian orthant dimensions or sample count");
    if (!prepared) {
        prepare();
        prepared = true;
        if (diagnostics && usesQmc) diagnostics->qmcPreparations = 1;
    }
    const std::size_t count = positions.size();
    if (diagnostics) diagnostics->effectiveDimension = static_cast<std::uint64_t>(count);
    if (!usesQmc) {
        auto result = analyticEstimate;
        result.error = std::min(1.0, result.error + omittedProbability);
        return result;
    }

    double replicateMeans[Replicates] = {};
    double halfMeans[Replicates] = {};
    const int halfSamples = samples/2;
    if (diagnostics) {
        diagnostics->usedQmc = true;
        diagnostics->qmcPoints = static_cast<std::uint64_t>(Replicates)*
            std::max(0, samples - computedSamples);
        diagnostics->qmcReusedPoints = static_cast<std::uint64_t>(Replicates)*
            std::min(computedSamples, samples);
    }
    if (samples > computedSamples) {
        // Reserve before modifying the cached sums, both to keep allocation
        // outside the QMC timer and to preserve the previous cache on failure.
        for (auto& sums : prefixSums) sums.reserve(static_cast<std::size_t>(samples) + 1);
        QmcTimer qmcTimer(diagnostics);
        for (int replicate = 0; replicate < Replicates; ++replicate) {
            auto& sums = prefixSums[replicate];
            // On a retry after an evaluation exception, discard any partial
            // extension while preserving every previously completed budget.
            sums.resize(static_cast<std::size_t>(computedSamples) + 1);
            double sum = sums.back();
            for (int point = computedSamples; point < samples; ++point) {
                double weight = 1.0;
                for (std::size_t row = 0; row < count; ++row) {
                    double conditionalMean = means[order[row]];
                    for (std::size_t column = 0; column < std::min(row, rank); ++column)
                        conditionalMean += factor[row*count + column]*z[column];
                    if (row >= rank) {
                        if (!(conditionalMean > 0.0)) { weight = 0.0; break; }
                        continue;
                    }
                    const double probability = cdf(conditionalMean/factor[row*count + row]);
                    weight *= probability;
                    if (!(weight > std::numeric_limits<double>::min())) { weight = 0.0; break; }
                    // The final nondegenerate variable integrates out when no
                    // later constraints remain. Otherwise sample its truncated
                    // normal by a survival probability, avoiding 1-CDF loss.
                    if (row + 1 < count) {
                        const double u = (static_cast<double>(sobol::sample(
                            static_cast<std::uint64_t>(point), static_cast<unsigned>(row), shifts[replicate][row])) + 0.5)/4294967296.0;
                        const double tail = probability*(1.0 - u);
                        if (!(tail > 0.0)) { weight = 0.0; break; }
                        z[row] = -quantile(tail);
                    }
                }
                sum += weight;
                sums.push_back(sum);
            }
        }
        computedSamples = samples;
    }
    for (int replicate = 0; replicate < Replicates; ++replicate) {
        const auto& sums = prefixSums[replicate];
        replicateMeans[replicate] = sums[samples]/static_cast<double>(samples);
        halfMeans[replicate] = halfSamples ? sums[halfSamples]/static_cast<double>(halfSamples) : replicateMeans[replicate];
    }
    const double mean = std::accumulate(replicateMeans, replicateMeans + Replicates, 0.0)/Replicates;
    const double halfMean = std::accumulate(halfMeans, halfMeans + Replicates, 0.0)/Replicates;
    double variation = 0.0;
    for (double value : replicateMeans) variation += (value - mean)*(value - mean);
    const double standardError = std::sqrt(variation/(Replicates*(Replicates - 1.0)));
    // Three replicate standard errors and a half-budget convergence check
    // are useful diagnostics, but do not constitute a rigorous enclosure.
    const double error = std::max(3.0*standardError, std::abs(mean - halfMean)) +
                         factorError + roundoff;
    return {std::clamp(mean, 0.0, 1.0),
            std::min(1.0, std::min(1.0, error) + omittedProbability)};
}

OrthantIntegrator::OrthantIntegrator(const std::vector<double>& positions,
                                   const std::vector<double>& means, int seed)
: _impl(std::make_unique<Impl>(positions, means, seed))
{
}

OrthantIntegrator::~OrthantIntegrator() = default;

ProbabilityEstimate OrthantIntegrator::extendTo(int samples, OrthantDiagnostics* diagnostics)
{
    return _impl->extendTo(samples, diagnostics);
}

ProbabilityEstimate positiveOrthant(const std::vector<double>& positions,
                                    const std::vector<double>& means,
                                    int samples, int seed,
                                    OrthantDiagnostics* diagnostics)
{
    OrthantIntegrator integrator(positions, means, seed);
    return integrator.extendTo(samples, diagnostics);
}

} // namespace Tungsten::FirstHitGpis
