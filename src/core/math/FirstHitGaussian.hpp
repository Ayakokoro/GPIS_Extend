#pragma once

#include <cstdint>
#include <memory>
#include <vector>

namespace Tungsten::FirstHitGpis {

// Numerical estimates, not certified probability enclosures. In particular,
// independent digital shifts measure QMC variation but cannot bound its bias.
struct ProbabilityEstimate {
    double value = 0.0;
    double error = 0.0;
};

// Per-call diagnostics. Effective dimension includes analytic cases and is
// measured after any almost-certain air constraints have been removed.
// QMC time and points cover only newly evaluated points in the eight replicate
// integration loops, excluding covariance construction, ordering,
// factorization and error estimation. Cached estimates still set usedQmc.
struct OrthantDiagnostics {
    std::uint64_t effectiveDimension = 0;
    bool usedQmc = false;
    std::uint64_t qmcNanoseconds = 0;
    std::uint64_t qmcPoints = 0;
    std::uint64_t qmcReusedPoints = 0;
    std::uint64_t qmcPreparations = 0;
};

double cdf(double x);

// Means are divided by the field standard deviation; distance by its SE
// correlation length. Returns Pr(Y(a) > 0, Y(b) <= 0).
ProbabilityEstimate downcrossingPair(double meanA, double meanB,
                                    double distanceOverLength);

// A fixed orthant problem with incrementally evaluated Sobol sequences.
// Preparation is lazy: the first extendTo() prepares the same ordering and
// factorization as positiveOrthant(). Increasing the budget evaluates only
// new points; repeated or smaller budgets use cached cumulative sums. Each
// result (including the half-budget error check) uses exactly `samples` points
// per replicate. Changing nodes, means or seed requires a new integrator.
class OrthantIntegrator {
    struct Impl;
    std::unique_ptr<Impl> _impl;

public:
    OrthantIntegrator(const std::vector<double>& positionsOverLength,
                      const std::vector<double>& normalizedMeans, int seed = 0);
    ~OrthantIntegrator();
    OrthantIntegrator(const OrthantIntegrator&) = delete;
    OrthantIntegrator& operator=(const OrthantIntegrator&) = delete;

    ProbabilityEstimate extendTo(int samples,
                                OrthantDiagnostics* diagnostics = nullptr);
};

// SE covariance exp(-0.5*(positions[i]-positions[j])^2), unit field variance.
// `samples` is the number of Sobol points PER replicate; eight independently
// digitally shifted replicates are used. No renderer sample dimensions are
// consumed. The same seed and inputs always give the same estimate.
ProbabilityEstimate positiveOrthant(const std::vector<double>& positionsOverLength,
                                    const std::vector<double>& normalizedMeans,
                                    int samples, int seed = 0,
                                    OrthantDiagnostics* diagnostics = nullptr);

} // namespace Tungsten::FirstHitGpis
