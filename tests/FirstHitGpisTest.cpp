#include "math/FirstHitGpis.hpp"
#include "math/FirstHitGaussian.hpp"
#include "math/conditional/mathutility/NumericPolicy.h"
#include "media/FirstHitGpisMedium.hpp"
#include "io/ImageIO.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace Tungsten;
namespace fh = Tungsten::FirstHitGpis;
namespace cg = Tungsten::ConditionalGpis;
namespace {
constexpr double Pi = 3.1415926535897932384626433832795;
void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance, const char *message)
{
    if (!(std::abs(actual-expected) <= tolerance)) {
        std::cerr << message << ": " << actual << " vs " << expected
                  << " (tolerance " << tolerance << ")\n";
        throw std::runtime_error(message);
    }
}
double normalCdf(double x) { return 0.5*std::erfc(-x/std::sqrt(2.0)); }

// A distance decision must use the same random threshold for the whole prefix.
// Unlike the renderer sampler, this exposes accidental redraws to the test.
class FixedSampler final : public PathSampleGenerator {
    UniformSampler _uniform{419};
public:
    float value;
    int draws = 0;
    explicit FixedSampler(float v) : value(v) {}
    void startPath(uint32, uint32) override { draws = 0; }
    void advancePath() override {}
    void saveState(OutputStreamHandle &) override {}
    void loadState(InputStreamHandle &) override {}
    bool nextBoolean(float p) override { return next1D() < p; }
    int nextDiscrete(int n) override { return std::min(n-1, int(next1D()*n)); }
    float next1D() override { ++draws; return value; }
    Vec2f next2D() override { return Vec2f(next1D(), next1D()); }
    UniformSampler &uniformGenerator() override { return _uniform; }
};

fh::MeanRay affineMean(double intercept, double slope)
{
    fh::MeanRay mean;
    mean.sample = [=](double s) { return fh::MeanSample{intercept+slope*s, slope}; };
    mean.minimum = [=](double a, double b) {
        return intercept+slope*(slope < 0.0 ? b : a);
    };
    mean.nextBoundary = [](double, double end) { return end; };
    mean.gradientBound = std::abs(slope);
    return mean;
}

fh::MeanRay narrowPocketMean()
{
    fh::MeanRay mean;
    mean.sample = [](double s) {
        const double offset = s-0.21;
        return fh::MeanSample{offset*offset-1e-6, 2.0*offset};
    };
    mean.minimum = [](double a, double b) {
        const double offset = std::clamp(0.21, a, b)-0.21;
        return offset*offset-1e-6;
    };
    mean.nextBoundary = [](double, double end) { return end; };
    mean.gradientBound = 0.6;
    return mean;
}

void gaussianProbabilities()
{
    for (double d : {0.001, 0.1, 1.0, 8.0}) {
        const auto q = fh::downcrossingPair(0.0, 0.0, d);
        const double expected = std::acos(std::exp(-0.5*d*d))/(2.0*Pi);
        close(q.value, expected, 2e-10, "zero-mean bivariate orthant identity");
        require(q.error >= 0.0 && std::isfinite(q.error), "finite pair error estimate");
    }
    const auto independent = fh::downcrossingPair(0.7, -0.4, 12.0);
    close(independent.value, normalCdf(0.7)*normalCdf(0.4), 2e-10,
          "independent pair probability");
    const auto coincident = fh::downcrossingPair(0.2, -0.3, 0.0);
    close(coincident.value, normalCdf(0.3)-normalCdf(-0.2), 2e-10,
          "perfectly correlated unequal means");

    const std::vector<double> positions{0.0, 0.4, 1.1};
    double expected = 0.125;
    for (int i = 0; i < 3; ++i) for (int j = i+1; j < 3; ++j) {
        const double d = positions[j]-positions[i];
        expected += std::asin(std::exp(-0.5*d*d))/(4.0*Pi);
    }
    const auto triple = fh::positiveOrthant(positions, std::vector<double>(3, 0.0), 4096, 17);
    close(triple.value, expected, 0.002, "three-dimensional zero-mean orthant identity");
    const auto one = fh::positiveOrthant({0.0}, {0.8}, 256, 17);
    close(one.value, normalCdf(0.8), 1e-13, "one-dimensional positive probability");
    const auto independentTriple = fh::positiveOrthant({0.0, 12.0, 24.0}, {0.2, -0.3, 0.7}, 256, 17);
    close(independentTriple.value, normalCdf(0.2)*normalCdf(-0.3)*normalCdf(0.7),
          1e-12, "independent positive constraints");
}

void gaussianDiagnostics()
{
    fh::OrthantDiagnostics diagnostics;
    const std::vector<double> positions{0.0, 0.4, 1.1};
    const std::vector<double> means{0.2, -0.3, 0.7};
    const auto reference = fh::positiveOrthant(positions, means, 256, 17);
    const auto observed = fh::positiveOrthant(positions, means, 256, 17, &diagnostics);
    close(observed.value, reference.value, 0.0, "QMC diagnostics preserve the probability estimate");
    close(observed.error, reference.error, 0.0, "QMC diagnostics preserve the error estimate");
    require(diagnostics.effectiveDimension == 3 && diagnostics.usedQmc,
            "nonzero-mean triple records a genuine three-dimensional QMC call");
    require(diagnostics.qmcPoints == 8*256,
            "QMC work counts every point in all eight replicates");
    require(diagnostics.qmcReusedPoints == 0 && diagnostics.qmcPreparations == 1,
            "a standalone QMC call prepares once and has no previously integrated points");

    fh::positiveOrthant(positions, std::vector<double>(3, 0.0), 256, 17, &diagnostics);
    require(diagnostics.effectiveDimension == 3 && !diagnostics.usedQmc,
            "zero-mean triple is multivariate but analytic");
    require(diagnostics.qmcPoints == 0 && diagnostics.qmcNanoseconds == 0,
            "reusing diagnostics clears the preceding QMC counters");

    diagnostics.effectiveDimension = 99;
    diagnostics.usedQmc = true;
    diagnostics.qmcNanoseconds = 99;
    diagnostics.qmcPoints = 99;
    diagnostics.qmcReusedPoints = 99;
    diagnostics.qmcPreparations = 99;
    const auto pruned = fh::positiveOrthant({0.0, 0.4, 1.1, 2.0},
                                           {8.0, -0.3, 0.7, 9.0}, 256, 17, &diagnostics);
    const auto pair = fh::positiveOrthant({0.4, 1.1}, {-0.3, 0.7}, 256, 17);
    close(pruned.value, pair.value, 0.0, "air pruning retains the same pair probability");
    require(pruned.error >= pair.error, "air pruning retains its omission error allowance");
    require(diagnostics.effectiveDimension == 2 && !diagnostics.usedQmc &&
            diagnostics.qmcPoints == 0 && diagnostics.qmcNanoseconds == 0 &&
            diagnostics.qmcReusedPoints == 0 && diagnostics.qmcPreparations == 0,
            "recursive air pruning reports the actual analytic dimension and clears QMC counters");
    fh::positiveOrthant(positions, {8.0, 9.0, 10.0}, 256, 17, &diagnostics);
    require(diagnostics.effectiveDimension == 0 && !diagnostics.usedQmc &&
            diagnostics.qmcPoints == 0 && diagnostics.qmcNanoseconds == 0,
            "all-air pruning reports no remaining constraints");
}

void incrementalGaussianCase(const std::vector<double> &positions,
                            const std::vector<double> &means,
                            const std::vector<int> &budgets,
                            std::uint64_t effectiveDimension, bool usesQmc,
                            int seed = 17)
{
    fh::OrthantIntegrator integral(positions, means, seed);
    int previousMaximum = 0;
    fh::OrthantDiagnostics diagnostics;
    for (int budget : budgets) {
        const auto expected = fh::positiveOrthant(positions, means, budget, seed);
        const auto actual = integral.extendTo(budget, &diagnostics);
        close(actual.value, expected.value, 0.0,
              "incremental probability exactly matches a fresh same-budget integral");
        close(actual.error, expected.error, 0.0,
              "incremental half-budget and replicate error exactly match a fresh integral");
        require(diagnostics.effectiveDimension == effectiveDimension &&
                diagnostics.usedQmc == usesQmc,
                "incremental calls report their retained dimension and evaluation method");
        const auto newPoints = usesQmc ? 8ULL*std::max(0, budget-previousMaximum) : 0ULL;
        const auto reusedPoints = usesQmc ? 8ULL*std::min(previousMaximum, budget) : 0ULL;
        require(diagnostics.qmcPoints == newPoints && diagnostics.qmcReusedPoints == reusedPoints,
                "incremental diagnostics count only new integrand evaluations and requested cached points");
        require(diagnostics.qmcPreparations == (usesQmc && previousMaximum == 0 ? 1ULL : 0ULL),
                "an unchanged QMC problem is prepared exactly once");
        previousMaximum = std::max(previousMaximum, budget);
    }
    bool rejected = false;
    try { integral.extendTo(0); }
    catch (const cg::NumericError &) { rejected = true; }
    require(rejected, "a cached integral still rejects a nonpositive sample budget");
    const auto expected = fh::positiveOrthant(positions, means, budgets.back(), seed);
    const auto recovered = integral.extendTo(budgets.back(), &diagnostics);
    close(recovered.value, expected.value, 0.0, "invalid budget leaves cached probability unchanged");
    close(recovered.error, expected.error, 0.0, "invalid budget leaves cached error unchanged");
    require(diagnostics.qmcPoints == 0 && diagnostics.qmcPreparations == 0,
            "returning to an existing budget does not repeat integration or preparation");
}

void incrementalGaussian()
{
    const std::vector<double> positions{0.0, 0.3, 0.75, 1.2, 1.75, 2.4, 3.2, 4.1};
    const std::vector<double> means{0.4, -0.35, 0.25, -0.1, 0.8, -0.3, 0.5, 0.15};
    const std::vector<int> ladder{256, 512, 1024, 2048, 4096, 8192};
    incrementalGaussianCase(positions, means, ladder, 8, true);

    // This is the actual amount of integration saved by the production
    // 256 -> 8192 ladder, independently of wall-clock scheduling and CPU speed.
    fh::OrthantIntegrator ladderIntegral(positions, means, 17);
    std::uint64_t integrated = 0, reused = 0, preparations = 0;
    for (int budget : ladder) {
        fh::OrthantDiagnostics diagnostics;
        ladderIntegral.extendTo(budget, &diagnostics);
        integrated += diagnostics.qmcPoints;
        reused += diagnostics.qmcReusedPoints;
        preparations += diagnostics.qmcPreparations;
    }
    require(integrated == 65536 && reused == 63488 && preparations == 1,
            "full budget ladder evaluates each of 65536 points once, saving 63488 evaluations");

    // An arbitrary budget needs its exact floor(N/2) checkpoint. Reductions
    // must return that prefix, not an estimate using the largest cached N.
    const std::vector<int> irregular{1, 3, 9, 16, 23, 64, 65, 17, 65, 31,
                                      129, 128, 257, 509, 1024, 7, 1031, 515, 1031};
    incrementalGaussianCase({0.0, 0.4, 1.1}, {0.2, -0.3, 0.7}, irregular, 3, true);
    incrementalGaussianCase({0.0, 1e-9, 0.4, 0.400000001, 1.1},
                            {0.2, 0.2001, -0.3, -0.29, 0.7}, irregular, 5, true);
    incrementalGaussianCase({0.0, 0.2, 0.4, 1.1, 2.0},
                            {8.0, 0.2, -0.3, 0.7, 9.0}, irregular, 3, true);

    incrementalGaussianCase({}, {}, irregular, 0, false);
    incrementalGaussianCase({0.0}, {0.8}, irregular, 1, false);
    incrementalGaussianCase({0.0, 0.4}, {-0.3, 0.7}, irregular, 2, false);
    incrementalGaussianCase({0.0, 0.4, 1.1}, {0.0, 0.0, 0.0}, irregular, 3, false);
    incrementalGaussianCase({0.0, 0.4, 1.1, 2.0},
                            {8.0, -0.3, 0.7, 9.0}, irregular, 2, false);
    incrementalGaussianCase({0.0, 0.4, 1.1}, {8.0, 9.0, 10.0}, irregular, 0, false);

    // Workspaces with distinct problems and shifts must not share sample
    // progress or cached weights; diagnostics can be enabled after a warmup.
    fh::OrthantIntegrator first(positions, means, 23);
    auto changedMeans = means;
    changedMeans[3] = -0.6;
    fh::OrthantIntegrator second(positions, changedMeans, 29);
    first.extendTo(23);
    second.extendTo(65);
    fh::OrthantDiagnostics firstDiagnostics, secondDiagnostics;
    const auto firstResult = first.extendTo(129, &firstDiagnostics);
    const auto secondResult = second.extendTo(129, &secondDiagnostics);
    const auto firstReference = fh::positiveOrthant(positions, means, 129, 23);
    const auto secondReference = fh::positiveOrthant(positions, changedMeans, 129, 29);
    close(firstResult.value, firstReference.value, 0.0, "first interleaved workspace preserves its own seed and mean");
    close(firstResult.error, firstReference.error, 0.0, "first interleaved workspace preserves its own error estimate");
    close(secondResult.value, secondReference.value, 0.0, "second interleaved workspace preserves its own seed and mean");
    close(secondResult.error, secondReference.error, 0.0, "second interleaved workspace preserves its own error estimate");
    require(firstDiagnostics.qmcPoints == 8*(129-23) && firstDiagnostics.qmcReusedPoints == 8*23 &&
            secondDiagnostics.qmcPoints == 8*(129-65) && secondDiagnostics.qmcReusedPoints == 8*65 &&
            firstDiagnostics.qmcPreparations == 0 && secondDiagnostics.qmcPreparations == 0,
            "enabling diagnostics after warmup observes independent existing progress");
}

void checkFlightStatistics(const fh::FlightStatistics &statistics, const fh::Settings &settings)
{
    const auto precheckDecisions = statistics.precheckSurvived+statistics.precheckHit;
    require(statistics.prefixQueries > 0 &&
            statistics.orthantCalls+precheckDecisions >= statistics.prefixQueries,
            "every tested nonempty prefix records an orthant evaluation or a precheck decision");
    require((statistics.orthantCalls == 0 ? statistics.maxActiveNodes == 0 :
             statistics.maxActiveNodes >= 2) &&
            statistics.maxActiveNodes <= statistics.uniqueAddedNodes+2 &&
            statistics.uniqueAddedNodes <= statistics.nodeInsertions,
            "active constraints and unique inserted positions use consistent counting");
    require(precheckDecisions <= statistics.precheckCalls &&
            statistics.precheckWithExtraNodes <= precheckDecisions &&
            statistics.precheckMarginalHits <= statistics.precheckHit,
            "precheck outcomes and marginal-only hits are nested subsets of precheck attempts");
    if (!settings.survivalPrecheck)
        require(statistics.precheckCalls == 0 && statistics.precheckNanoseconds == 0,
                "disabling the precheck records no precheck work");
    require(statistics.candidateTests >= statistics.nodeInsertions && statistics.meanQueries > 0,
            "each insertion evaluates a candidate and records mean cache misses");
    require(statistics.maxEffectiveDimension <= statistics.maxActiveNodes &&
            statistics.qmcCalls <= statistics.multivariateCalls &&
            statistics.multivariateCalls <= statistics.orthantCalls,
            "QMC, multivariate and total orthant calls are nested subsets");
    require(statistics.orthantNanoseconds >= statistics.multivariateNanoseconds &&
            statistics.multivariateNanoseconds >= statistics.qmcNanoseconds &&
            statistics.maxOrthantCallNanoseconds <= statistics.orthantNanoseconds &&
            statistics.maxQmcCallNanoseconds <= statistics.qmcNanoseconds,
            "full backend timing contains QMC loop timing and per-call maxima");
    require(statistics.qmcPoints+statistics.qmcReusedPoints >= statistics.qmcCalls*8*settings.qmcSamples &&
            statistics.qmcPoints >= statistics.qmcPreparations*8*settings.qmcSamples &&
            statistics.qmcPreparations <= statistics.qmcCalls,
            "new and cached QMC points cover every requested budget without repeating preparations");
}

void flightStatistics()
{
    fh::Settings settings;
    settings.probabilityTolerance = 0.004;
    settings.distanceTolerance = 0.002;

    // A safe empty interval should be distinguishable from a ray that needs
    // adaptive constraints, even though prepare() produces potential candidates.
    fh::FlightStatistics airStatistics;
    fh::Flight air(affineMean(7.0, 0.0), 1.0, 1.0, 8.0, settings, &airStatistics);
    close(air.sampleDistance(0.999), 8.0, 0.0, "instrumented air ray remains a miss");
    checkFlightStatistics(airStatistics, settings);
    require(airStatistics.prefixQueries == 1 && airStatistics.orthantCalls == 0 &&
            airStatistics.precheckCalls == 1 && airStatistics.precheckSurvived == 1 &&
            airStatistics.precheckHit == 0 && airStatistics.maxActiveNodes == 0 &&
            airStatistics.uniqueAddedNodes == 0 &&
            airStatistics.nodeInsertions == 0 && airStatistics.candidateTests == 0,
            "safe air ray exits by its survival lower bound without an orthant evaluation");
    require(airStatistics.candidateLocations > 2 && airStatistics.qmcCalls == 0,
            "prepared candidates are distinct from active constraints and actual QMC work");
    require(airStatistics.flightNanoseconds >= airStatistics.orthantNanoseconds,
            "distance-inversion timing includes its backend evaluations");

    // Requesting an accurate probability interval forces refinement regardless
    // of a lucky distance uniform deciding the result from endpoints alone.
    fh::FlightStatistics refinedStatistics;
    fh::Flight refined(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings, &refinedStatistics);
    fh::Flight reference(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings);
    const auto refinedBounds = refined.survivalBounds(1.0);
    const auto referenceBounds = reference.survivalBounds(1.0);
    close(refinedBounds.lower, referenceBounds.lower, 0.0, "statistics preserve refined lower probability");
    close(refinedBounds.upper, referenceBounds.upper, 0.0, "statistics preserve refined upper probability");
    checkFlightStatistics(refinedStatistics, settings);
    require(refinedStatistics.prefixQueries == 1 && refinedStatistics.uniqueAddedNodes > 0 &&
            refinedStatistics.candidateTests > 0 && refinedStatistics.maxActiveNodes > 2,
            "uncertain zero-mean interval records its additional constraints");
    require(refinedStatistics.maxActiveNodes == static_cast<std::uint64_t>(refinedBounds.nodes) &&
            refinedStatistics.nodeInsertions == refinedStatistics.uniqueAddedNodes,
            "single-prefix refinement reports its final active size and unique insertions");
    require(refinedStatistics.multivariateCalls > 0 && refinedStatistics.qmcCalls > 0,
            "refinement exercises both multivariate evaluation and measured QMC work");
    require(refinedStatistics.precheckCalls == 0,
            "a probability-interval query retains the full refinement path");

    // Instrumentation may observe repeated prefix queries, but must neither
    // change their decisions nor consume an additional sampling dimension.
    for (bool precheck : {false, true}) for (double uniform : {0.5, 0.8, 0.95}) {
        settings.survivalPrecheck = precheck;
        fh::FlightStatistics statistics;
        fh::Flight measured(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings, &statistics);
        fh::Flight unmeasured(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings);
        close(measured.sampleDistance(uniform), unmeasured.sampleDistance(uniform), 0.0,
              "enabling statistics preserves fixed-uniform first-hit distance exactly");
        checkFlightStatistics(statistics, settings);
        require(statistics.flightNanoseconds >= statistics.orthantNanoseconds,
                "full sampled-flight timing includes all prefix orthant calls");
    }
}

void firstPassage()
{
    fh::Settings settings;
    settings.probabilityTolerance = 0.004;
    settings.distanceTolerance = 0.002;
    fh::Flight zero(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings);
    const auto bounds = zero.survivalBounds(1.0);
    // Rice plus the two endpoints give these independent, analytic bounds.
    const double endpointUpper = 1.0-std::acos(std::exp(-0.5))/Pi;
    const double riceLower = 1.0-1.0/Pi;
    require(bounds.lower >= riceLower-0.006 && bounds.upper <= endpointUpper+0.006,
            "refinement must respect the analytic two-endpoint envelope");
    require(bounds.lower <= bounds.upper && bounds.upper-bounds.lower <= 0.008,
            "requested survival tolerance");
    fh::Flight miss(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings);
    close(miss.sampleDistance(0.5), 1.0, 0.0, "known surviving quantile");
    fh::Flight hit(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings);
    const double hitDistance = hit.sampleDistance(0.8);
    require(hitDistance > 0.0 && hitDistance < 1.0, "known colliding quantile");
    fh::Flight repeat(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, settings);
    close(repeat.sampleDistance(0.8), hitDistance, 0.0,
          "same threshold must produce reproducible first passage");

    fh::Flight emptyAir(affineMean(7.0, 0.0), 1.0, 1.0, 8.0, settings);
    close(emptyAir.sampleDistance(0.999), 8.0, 0.0, "whole-air interval screening");
    fh::Flight plane(affineMean(1.0, -1.0), 0.005, 0.5, 2.0, settings);
    close(plane.sampleDistance(0.5), 1.0, 0.025, "small-noise plane first hit");

    // A quadratic mean is also a legal trilinear-cell ray restriction. Its
    // narrow negative interval falls between the fixed Gauss-Kronrod nodes;
    // zero quadrature samples must not certify that the whole prefix is empty.
    fh::Flight narrowDip(narrowPocketMean(), 1e-7, 1.0, 0.5, settings);
    close(narrowDip.sampleDistance(0.5), 0.209, 0.004,
          "resolve a thin negative mean pocket between quadrature nodes");

    fh::Settings limited = settings;
    limited.probabilityTolerance = 1e-12;
    limited.maxNodes = 2;
    limited.maxRefinements = 1;
    bool rejected = false;
    try {
        fh::Flight unresolved(affineMean(0.0, 0.0), 1.0, 1.0, 1.0, limited);
        unresolved.survivalBounds(1.0);
    } catch (const cg::NumericError &) { rejected = true; }
    require(rejected, "exhausted refinement must not silently return a miss");
    rejected = false;
    try { fh::Flight zeroVariance(affineMean(1.0, 0.0), 0.0, 1.0, 1.0, settings); }
    catch (const cg::NumericError &) { rejected = true; }
    require(rejected, "zero variance is outside this experimental sampler");
    rejected = false;
    try { fh::Flight rareOrigin(affineMean(-8.0, 0.0), 1.0, 1.0, 1.0, settings); }
    catch (const cg::NumericError &) { rejected = true; }
    require(rejected, "rare positive origins must not be silently clamped");
    std::cout << "first-hit zero-mean survival interval=[" << bounds.lower << ", "
              << bounds.upper << "], sampled distance=" << hitDistance << '\n';
}

void survivalPrecheck()
{
    fh::Settings settings;
    settings.probabilityTolerance = 0.004;
    settings.distanceTolerance = 0.002;
    fh::Bounds referenceBounds;
    for (bool enabled : {false, true}) {
        settings.survivalPrecheck = enabled;

        fh::FlightStatistics zeroMissStatistics;
        fh::Flight zeroMiss(affineMean(0.0, 0.0), 1.0, 1.0, 1.0,
                            settings, &zeroMissStatistics);
        // Independently of any discretization, the Rice lower bound is
        // 1 - 1/pi > 0.5, so this quantile must survive the whole prefix.
        close(zeroMiss.sampleDistance(0.5), 1.0, 0.0,
              "zero-mean Rice-certified miss agrees with precheck enabled and disabled");
        checkFlightStatistics(zeroMissStatistics, settings);
        require(enabled ? zeroMissStatistics.precheckSurvived == 1 &&
                          zeroMissStatistics.orthantCalls == 0 :
                          zeroMissStatistics.orthantCalls == 1,
                "Rice precheck actually avoids the otherwise required endpoint integral");

        fh::FlightStatistics zeroHitStatistics;
        fh::Flight zeroHit(affineMean(0.0, 0.0), 1.0, 1.0, 1.0,
                           settings, &zeroHitStatistics);
        const double hitDistance = zeroHit.sampleDistance(0.8);
        require(hitDistance > 0.0 && hitDistance < 1.0,
                "zero-mean pair upper bound identifies a hit with either precheck setting");
        checkFlightStatistics(zeroHitStatistics, settings);
        if (enabled)
            require(zeroHitStatistics.precheckHit > zeroHitStatistics.precheckMarginalHits,
                    "pair probabilities can reject a prefix whose marginal upper bound cannot");

        fh::FlightStatistics conditionedStatistics;
        fh::Flight conditioned(affineMean(-2.0, 0.0), 1.0, 1.0, 0.01,
                               settings, &conditionedStatistics);
        // The unconditional positive probability is only 0.02275, but the
        // ray is already known to start positive. Omitting division by p0
        // from the marginal upper bound would incorrectly turn this into a hit.
        close(conditioned.sampleDistance(0.5), 0.01, 0.0,
              "cheap bounds preserve conditioning on a positive origin");
        checkFlightStatistics(conditionedStatistics, settings);
        if (enabled)
            require(conditionedStatistics.precheckSurvived == 1 &&
                    conditionedStatistics.precheckHit == 0,
                    "rare positive-origin normalization does not create a false collision");

        fh::FlightStatistics descendingStatistics;
        fh::Flight descending(affineMean(2.0, -4.0), 1.0, 1.0, 1.0,
                              settings, &descendingStatistics);
        const double descendingDistance = descending.sampleDistance(0.9);
        require(descendingDistance > 0.0 && descendingDistance < 1.0,
                "negative endpoint marginal upper bound preserves a known hit");
        checkFlightStatistics(descendingStatistics, settings);
        if (enabled)
            require(descendingStatistics.precheckMarginalHits > 0,
                    "a sufficiently negative endpoint can decide before evaluating pair probabilities");

        fh::Flight plane(affineMean(1.0, -1.0), 0.005, 0.5, 2.0, settings);
        close(plane.sampleDistance(0.5), 1.0, 0.025,
              "small-noise plane remains at its analytic surface with either precheck setting");
        fh::Flight narrowDip(narrowPocketMean(), 1e-7, 1.0, 0.5, settings);
        close(narrowDip.sampleDistance(0.5), 0.209, 0.004,
              "precheck never skips a thin interior pocket with positive endpoints");

        fh::FlightStatistics intervalStatistics;
        fh::Flight interval(affineMean(0.0, 0.0), 1.0, 1.0, 1.0,
                            settings, &intervalStatistics);
        const auto bounds = interval.survivalBounds(1.0);
        require(intervalStatistics.precheckCalls == 0,
                "precheck is only used for threshold decisions, not requested probability intervals");
        if (enabled) {
            close(bounds.lower, referenceBounds.lower, 0.0,
                  "precheck setting leaves a requested survival lower bound unchanged");
            close(bounds.upper, referenceBounds.upper, 0.0,
                  "precheck setting leaves a requested survival upper bound unchanged");
        } else {
            referenceBounds = bounds;
        }

        fh::Settings limited = settings;
        limited.probabilityTolerance = 1e-12;
        limited.maxNodes = 2;
        limited.maxRefinements = 1;
        fh::FlightStatistics unresolvedStatistics;
        bool rejected = false;
        try {
            // 0.69 lies between the Rice lower bound (0.68169) and
            // the endpoint upper bound (about 0.707). Neither is decisive.
            fh::Flight unresolved(affineMean(0.0, 0.0), 1.0, 1.0, 1.0,
                                  limited, &unresolvedStatistics);
            unresolved.sampleDistance(0.69);
        } catch (const cg::NumericError &) { rejected = true; }
        require(rejected && unresolvedStatistics.orthantCalls == 1,
                "an undecided precheck falls back to integration and respects exhaustion errors");
        require(unresolvedStatistics.precheckSurvived == 0 && unresolvedStatistics.precheckHit == 0,
                "an overlapping cheap interval never chooses its midpoint as a survival decision");
    }
}

std::shared_ptr<ClassicGlobalMedium> makeMedium(const std::string &mean,
        const std::string &type = "first_hit_gpis", const std::string &extra = "",
        const std::string &kernel = "squared_exponential")
{
    const std::string json = "{\"type\":\""+type+R"(","gaussian_process":{
        "type":"standard","mean":)"+mean+R"(,
        "covariance":{"type":")"+kernel+R"(","sigma":1,"lengthScale":1}},
        "material":{"type":"mirror","albedo":[0.25,0.6,0.9]})"+extra+"}";
    JsonDocument document(Path("first-hit-gpis-test.json"), json);
    Scene scene;
    auto medium = std::dynamic_pointer_cast<ClassicGlobalMedium>(scene.fetchMedium(document));
    require(bool(medium), "first-hit medium factory registration");
    medium->loadResources();
    medium->prepareForRender();
    return medium;
}

void mediumIntegration()
{
    const std::string mean = R"({"type":"linear","direction":[0,0,1],"scale":0.7})";
    auto experimental = makeMedium(mean, "first_hit_gpis",
        R"(,"first_hit":{"probability_tolerance":0.004,"distance_tolerance":0.002,"survival_precheck":false})");
    auto classic = makeMedium(mean, "classic_global");
    auto first = std::dynamic_pointer_cast<FirstHitGpisMedium>(experimental);
    require(bool(first), "factory must instantiate the experimental medium");
    require(!first->firstHitSettings().survivalPrecheck,
            "scene configuration can disable the survival precheck for comparisons");
    close(first->lengthScale(), 1.0, 1e-12, "physical squared-exponential length scale");
    MediumSample sample{};
    sample.p = Vec3f(0.1f, -0.2f, 0.3f);
    const Vec3f incoming(0.0f, 0.0f, -1.0f);
    IndependentPathSampler rng(1931, false);
    for (int i = 0; i < 64; ++i) {
        rng.startPath(0, i);
        PhaseSample phaseSample;
        require(classic->phaseFunction(sample.p)->sample(rng, incoming, sample, phaseSample),
                "classic_global phase reference sample");
        close(first->phaseFunction(sample.p)->pdf(incoming, phaseSample.w, sample),
              classic->phaseFunction(sample.p)->pdf(incoming, phaseSample.w, sample), 0.0,
              "reuse classic_global phase density exactly");
        const Vec3f a = first->phaseFunction(sample.p)->eval(incoming, phaseSample.w, sample);
        const Vec3f b = classic->phaseFunction(sample.p)->eval(incoming, phaseSample.w, sample);
        for (int channel = 0; channel < 3; ++channel)
            close(a[channel], b[channel], 0.0, "reuse classic_global phase evaluation exactly");
    }

    rapidjson::Document document;
    auto value = first->toJson(document.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    JsonDocument roundTrip(Path("first-hit-gpis-roundtrip.json"), buffer.GetString());
    Scene scene;
    auto copy = std::dynamic_pointer_cast<FirstHitGpisMedium>(scene.fetchMedium(roundTrip));
    require(bool(copy), "first-hit type survives serialization");
    copy->loadResources(); copy->prepareForRender();
    close(copy->firstHitSettings().probabilityTolerance, 0.004, 0.0, "serialize probability tolerance");
    close(copy->firstHitSettings().distanceTolerance, 0.002, 0.0, "serialize distance tolerance");
    require(!copy->firstHitSettings().survivalPrecheck, "serialize disabled survival precheck");

    auto homogeneous = makeMedium(R"({"type":"homogeneous","offset":0})");
    require(std::dynamic_pointer_cast<FirstHitGpisMedium>(homogeneous)->firstHitSettings().survivalPrecheck,
            "survival precheck is enabled by default");
    FixedSampler fixed(0.95f);
    Medium::MediumState state{};
    state.reset();
    // Physical length is 1, not 0.5. nearT is a geometry epsilon, not a GP gap.
    const Ray ray(Vec3f(0.0f), Vec3f(0.0f,0.0f,2.0f), 0.4f, 0.5f);
    require(homogeneous->sampleDistance(fixed, ray, state, sample), "first empty-origin flight");
    require(!sample.exited && sample.t > 0.0f && sample.t < ray.nearT(),
            "first hit includes the segment hidden by the geometry near epsilon");
    require(state.bounce == 1 && !state.firstScatter, "record exactly one medium collision");
    require(fixed.draws == 1, "one random threshold per first-hit flight");
    close(sample.weight.x(), 1.0, 0.0, "unit first-hit distance weight");
    const int previousDraws = fixed.draws;
    require(!homogeneous->sampleDistance(fixed, ray, state, sample), "terminate after first hit");
    require(fixed.draws == previousDraws, "do not start a collision-origin random flight");
    close(homogeneous->transmittance(fixed, ray, true, false, &state).x(), 1.0, 0.0,
          "preview shadow connections omit GP attenuation");
    state.reset();
    state.info.pixelSampleSegment.w() = 1;
    require(!homogeneous->sampleDistance(fixed, ray, state, sample),
            "ordinary surface reflection must not restart a camera first-hit flight");
    state.info.pixelSampleSegment.w() = 0;

    bool rejected = false;
    try { homogeneous->pdf(fixed, ray, false, false); }
    catch (const cg::NumericError &) { rejected = true; }
    require(rejected, "unsupported marginalized distance PDF must fail explicitly");
    rejected = false;
    try { makeMedium(mean, "first_hit_gpis", "", "rational_quadratic"); }
    catch (const std::exception &) { rejected = true; }
    require(rejected, "reject non-SE covariance instead of changing its meaning");
    rejected = false;
    state.reset();
    try {
        homogeneous->sampleDistance(fixed,
            Ray(Vec3f(0.0f), Vec3f(0.0f,0.0f,1.0f)), state, sample);
    } catch (const cg::NumericError &) { rejected = true; }
    require(rejected, "unbounded first-hit segment must fail explicitly");
}

void imageCheck(const char *path)
{
    int width = 0, height = 0;
    auto data = ImageIO::loadHdr(Path(path), TexelConversion::REQUEST_RGB, width, height);
    require(bool(data) && width > 0 && height > 0, "read first-hit rendered image");
    double sum = 0.0, maximum = 0.0, minimum = 1e30;
    int dark = 0, bright = 0;
    for (int i = 0; i < width*height; ++i) {
        double luminance = 0.0;
        for (int channel = 0; channel < 3; ++channel) {
            const double value = data[3*i+channel];
            require(std::isfinite(value) && value >= 0.0, "finite nonnegative first-hit pixels");
            luminance += value/3.0;
        }
        sum += luminance;
        minimum = std::min(minimum, luminance);
        maximum = std::max(maximum, luminance);
        dark += luminance < 0.8;
        bright += luminance > 0.9;
    }
    require(sum/(width*height) > 0.05, "first-hit render must not be black");
    require(maximum-minimum > 0.2 && dark > width*height/50 && bright > width*height/50,
            "first-hit sphere must differ visibly from the background");
    std::cout << "first-hit image " << width << 'x' << height << ", mean="
              << sum/(width*height) << ", range=[" << minimum << ", " << maximum << "]\n";
}
}

int main(int argc, char **argv)
{
    std::cout.setf(std::ios::unitbuf);
    try {
        if (argc == 3 && std::string(argv[1]) == "image") { imageCheck(argv[2]); return 0; }
        gaussianProbabilities();
        gaussianDiagnostics();
        incrementalGaussian();
        firstPassage();
        flightStatistics();
        survivalPrecheck();
        mediumIntegration();
        std::cout << "first-hit GPIS checks passed\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
