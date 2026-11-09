#pragma once

#include <functional>
#include <memory>
#include <cstdint>

namespace Tungsten::FirstHitGpis {

// Experimental numerical tolerances, not a certificate for the QMC integral.
struct Settings {
    double probabilityTolerance = 0.002;
    double distanceTolerance = 0.001; // world distance
    int maxNodes = 48;
    int qmcSamples = 256;            // per scramble, eight scrambles
    int maxQmcSamples = 8192;
    int maxRefinements = 256;
    bool survivalPrecheck = true;   // cheap threshold bounds before the orthant backend
    void validate() const;
};

struct MeanSample { double value, derivative; }; // derivative along a unit ray
struct MeanRay {
    std::function<MeanSample(double)> sample;
    std::function<double(double, double)> minimum; // conservative interval lower bound
    std::function<double(double, double)> nextBoundary; // optional interpolation knots
    double gradientBound = 0.0;
};

struct Bounds {
    double lower = 0.0, upper = 1.0;
    int nodes = 0;
    // Rice quadrature and QMC errors are estimates; these are numerical
    // envelopes around mathematically valid continuous-path bounds.
};

// Optional per-flight observations. Counters include all original-origin
// prefix queries used to invert ONE distance uniform, never subsequent bounces.
struct FlightStatistics {
    std::uint64_t prefixQueries = 0;
    std::uint64_t precheckCalls = 0; // once per new node set, never for QMC budget increases
    std::uint64_t precheckSurvived = 0;
    std::uint64_t precheckHit = 0;
    std::uint64_t precheckMarginalHits = 0; // subset of precheckHit, without pair evaluation
    std::uint64_t precheckWithExtraNodes = 0; // decisions with >2 nodes, not necessarily QMC saved
    std::uint64_t precheckNanoseconds = 0; // includes pair cache misses, excludes prepare()
    std::uint64_t candidateLocations = 0; // prepared pool, including endpoints
    std::uint64_t candidateTests = 0;     // G(c) evaluations, including repeats
    std::uint64_t nodeInsertions = 0;
    std::uint64_t uniqueAddedNodes = 0;   // distinct selected internal positions
    std::uint64_t maxActiveNodes = 0;     // largest set passed to orthant; zero if none
    std::uint64_t maxEffectiveDimension = 0; // after backend air pruning
    std::uint64_t meanQueries = 0;        // cache misses
    std::uint64_t orthantCalls = 0;
    std::uint64_t orthantNanoseconds = 0; // full all-positive backend calls
    std::uint64_t maxOrthantCallNanoseconds = 0;
    std::uint64_t multivariateCalls = 0; // effective dimension >= 3, including analytic cases
    std::uint64_t multivariateNanoseconds = 0; // full backend call, when effective dimension >= 3
    std::uint64_t qmcCalls = 0;
    std::uint64_t qmcNanoseconds = 0;     // QMC integration loop only
    std::uint64_t maxQmcCallNanoseconds = 0;
    std::uint64_t qmcPoints = 0;          // newly evaluated points, all eight replicates
    std::uint64_t qmcReusedPoints = 0;    // cached points included in later estimates
    std::uint64_t qmcPreparations = 0;    // distinct QMC matrix/integrand preparations
    std::uint64_t pairCalls = 0;          // downcrossing pair cache misses
    std::uint64_t pairNanoseconds = 0;
    std::uint64_t flightNanoseconds = 0;  // distance inversion, excluding phase
};

// One finite, straight flight conditioned only on Y(0)>0. Positions and all
// tolerances use world distance. No sampled field value or outgoing history.
class Flight {
public:
    Flight(MeanRay mean, double sigma, double lengthScale, double length,
           const Settings &settings = {}, FlightStatistics *statistics = nullptr);
    ~Flight();
    Flight(const Flight &) = delete;
    Flight &operator=(const Flight &) = delete;

    // Refine to the configured probability tolerance or raise NumericError.
    Bounds survivalBounds(double distance);
    // A single open uniform is reused for every prefix query. length means miss.
    double sampleDistance(double uniform);

private:
    struct Impl;
    std::unique_ptr<Impl> _impl;
};
}
