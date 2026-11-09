#pragma once

#include "math/SparseHistoryGaussian.hpp"
#include "math/conditional/transport/MajorantOpticalDepth.h"
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace Tungsten::SingleHistoryGpis {

enum class TrackingMethod { Delta, Inversion, Regular };
enum class CandidateCollection { Fine, Macro };

struct Settings {
    double selectionStepOverLength = 0.5;
    int historyGroups = 8;
    bool historyEnabled = true;
    double shadowRouletteThreshold = 0.1;
    int maxWork = 1000000;
    bool profile = false;
    bool profileDetail = false;
    // Conditional baseline bound controls; the history law and the outer
    // event-budget subdivision remain unchanged. Zero retains legacy work.
    bool normalizedBoundOnly = false;
    double farBoundStartScale = 0;
    double tightenExpectedEvents = 0;
    int macrocellSize = 0;
    double macrocellMaxExpectedEvents = 8;
    // Delta-only conservative pruning of candidate work inside canonical
    // fine fragments. Requires macrocellSize; fine DDA boundaries stay intact.
    bool candidateMacroPruning = false;
    // An alternative history approximation: collect endpoints and midpoint of
    // each macro/selection fragment, independently of tracking subdivisions.
    CandidateCollection candidateCollection = CandidateCollection::Fine;
    int candidateMacrocellSize = 4;
    TrackingMethod trackingMethod = TrackingMethod::Delta;
    double inversionMaxStepOverLength = 0.25;
    double inversionMeanRangeOverSigma = 0.25;
    double inversionAbsoluteTolerance = 1e-5;
    double inversionRelativeTolerance = 1e-3;
    double inversionDistanceToleranceOverLength = 1e-4;
    // Fixed midpoint rates on equal subdivisions of each canonical
    // cell/selection fragment. No adaptive near-birth subdivision is applied.
    int regularSubdivisions = 1;
    SparseHistoryGpis::ProbabilitySettings probability;
    void validate() const;
};

struct Candidate {
    double distance = 0;
    double z = 0;
    double inverseMills = 0;
};

// Value-only state: branches never share a mutable geometry or random cursor.
// fragmentBegin is a canonical candidate-collection/selection boundary (fine
// cell or macro face), independent of renderer and tracking subdivisions.
struct Snapshot {
    double age = 0;
    std::uint64_t group = 0;
    double fragmentBegin = 0;
    std::optional<Candidate> current;
    std::vector<Candidate> completed;
    std::optional<double> selected;
    ConditionalGpis::MajorantOpticalDepth clock;
    // Residual optical depth for inversion or the regular piecewise constant
    // model, distinct from the delta proposal clock. Negative means no draw.
    double inversionRemaining = -1;
    // Proposal-only state copied across straight renderer continuation.
    // Retry windows never cross a history-selection group. A missing cursor
    // is permanently unavailable; a failed numerical query is only deferred.
    double macroRetryAt = 0;
    bool macroUnavailable = false;
};

struct Work {
    // cells counts completed candidate fragments, fine or macro depending on
    // candidateCollection; it is not a tracking DDA counter in macro mode.
    std::uint64_t intervals = 0, candidates = 0, selections = 0, cells = 0;
    std::uint64_t rouletteTests = 0, rouletteTerminations = 0;
    std::uint64_t hazardEvaluations = 0, inversionRefinements = 0;
    double selectionSeconds = 0, majorantSeconds = 0, probabilitySeconds = 0;
    // Exclusive integration/range/inversion overhead; probability and
    // candidate-selection timers are subtracted from this measurement.
    double quadratureSeconds = 0;
    // Optional detail timers (profile && profileDetail). The three selection
    // stages are disjoint; candidate statistics and scoring are nested costs.
    // Restored reference candidates can also occur inside historyFactorSeconds.
    double boundarySeconds = 0, collectSeconds = 0, groupSeconds = 0;
    double candidateStatisticsSeconds = 0, scoringSeconds = 0;
    double baselineSetupSeconds = 0, historyFactorSeconds = 0;
    double conditionalBoundSeconds = 0, exteriorBoundsSeconds = 0, exteriorAlgebraSeconds = 0;
    double macroBoundSeconds = 0;
    // Nested inside collectSeconds; includes cached certificate checks.
    double candidateMacroBoundSeconds = 0;
    double localLawSeconds = 0, hazardSeconds = 0;
    // Diagnostic delta-only counters, enabled by profile && profileDetail.
    // They do not participate in the work budget. Accept/reject counts refer
    // to path proposals; ratio-tracked shadows have no Bernoulli acceptance.
    std::uint64_t processedIntervals = 0, emptyIntervals = 0;
    std::uint64_t acceptedCollisions = 0, rejectedCandidates = 0, zeroHazardCandidates = 0;
    std::uint64_t majorantCalls = 0, eventSplits = 0;
    std::uint64_t candidateStatistics = 0, scoreEvaluations = 0;
    std::uint64_t macroAttempts = 0, macroAccepted = 0, macroRejected = 0;
    std::uint64_t macroUnavailable = 0, macroRetrySkips = 0;
    std::uint64_t candidateMacroQueries = 0, candidateMacroPrunedFragments = 0;
    std::uint64_t candidateMacroUnprunedFragments = 0, candidateMacroUnavailable = 0;
    std::uint64_t fineBoundaryQueries = 0, macroBoundaryQueries = 0;
    std::uint64_t macroCandidateFragments = 0;
};

struct TrackResult {
    bool hit = false;
    double distance = 0;
    // Crossing law at the actual event position, including the selected
    // history coordinate. Delta also uses this law for acceptance. Regular
    // uses a midpoint rate for distance and this actual-position law for speed.
    std::shared_ptr<SparseHistoryGpis::LocalLaw> law;
};

class Flight {
    struct Impl;
    std::unique_ptr<Impl> _impl;
public:
    Flight(const ConditionalGpis::GPSSField &, const SparseHistoryGpis::Start &,
           const Settings & = {}, const Snapshot & = {});
    ~Flight();
    const Snapshot &snapshot() const;
    const Work &work() const;
    double lengthScale() const;
    Eigen::Vector3d meanGradient(double distance);
    // Deterministic diagnostic/reference query; advances passed geometry only.
    // Calls must have nondecreasing distances. No random candidates are used.
    std::vector<double> selectHistory(double distance);
    SparseHistoryGpis::LocalLaw law(double distance);
    // Diagnostic bound query: advances only geometry to begin (never to end),
    // consumes no random numbers, and requires [begin,end) inside one canonical
    // cell/selection fragment. The fixed history is the one selected at begin.
    // Intended for a separate diagnostic Flight, not a resumed sampled path.
    double diagnosticMajorant(double begin, double end);
    // All distances are physical ages measured from the original flight birth.
    TrackResult sampleDistance(double end, ConditionalGpis::Random &);
    double transmittance(double end, ConditionalGpis::Random &);
};

} // namespace Tungsten::SingleHistoryGpis
