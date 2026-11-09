#pragma once

#include "ClassicGlobalMedium.hpp"
#include "math/SingleHistoryFlight.hpp"
#include "math/conditional/transport/FixedSpeedGradientDistribution.hpp"
#include "io/Path.hpp"
#include <array>
#include <atomic>
#include <cstdint>
#include <optional>

namespace Tungsten {

// Immutable, value-only snapshots. Visibility branches copy the candidate
// cache but never share a geometry cursor or a pending random proposal.
struct SingleHistoryPathContext final : MediumPathContext {
    const Medium *owner = nullptr;
    SparseHistoryGpis::Start start;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    Vec3f rayOrigin = Vec3f(0.f);
    SingleHistoryGpis::Snapshot snapshot;
};

struct SingleHistoryEventContext final : MediumEventContext {
    const Medium *owner = nullptr;
    Eigen::Vector3d position = Eigen::Vector3d::Zero();
    std::optional<ConditionalGpis::FixedSpeedGradientDistribution> gradient;
};

class SingleHistoryGpisMedium final : public ClassicGlobalMedium {
    struct ProfileRecord {
        std::atomic<std::uint64_t> calls{0}, nanoseconds{0}, speedNanoseconds{0};
        std::atomic<std::uint64_t> selectionNanoseconds{0}, majorantNanoseconds{0}, probabilityNanoseconds{0};
        std::atomic<std::uint64_t> quadratureNanoseconds{0}, hazardEvaluations{0}, inversionRefinements{0};
        std::atomic<std::uint64_t> intervals{0}, candidates{0}, selections{0}, cells{0};
        std::atomic<std::uint64_t> rouletteTests{0}, rouletteTerminations{0};
        std::atomic<std::uint64_t> boundaryNanoseconds{0};
        std::atomic<std::uint64_t> collectNanoseconds{0};
        std::atomic<std::uint64_t> groupNanoseconds{0};
        std::atomic<std::uint64_t> candidateStatisticsNanoseconds{0};
        std::atomic<std::uint64_t> scoringNanoseconds{0};
        std::atomic<std::uint64_t> baselineSetupNanoseconds{0};
        std::atomic<std::uint64_t> historyFactorNanoseconds{0};
        std::atomic<std::uint64_t> conditionalBoundNanoseconds{0};
        std::atomic<std::uint64_t> exteriorBoundsNanoseconds{0};
        std::atomic<std::uint64_t> exteriorAlgebraNanoseconds{0};
        std::atomic<std::uint64_t> localLawNanoseconds{0};
        std::atomic<std::uint64_t> hazardNanoseconds{0};
        std::atomic<std::uint64_t> processedIntervals{0};
        std::atomic<std::uint64_t> emptyIntervals{0};
        std::atomic<std::uint64_t> acceptedCollisions{0};
        std::atomic<std::uint64_t> rejectedCandidates{0};
        std::atomic<std::uint64_t> zeroHazardCandidates{0};
        std::atomic<std::uint64_t> majorantCalls{0};
        std::atomic<std::uint64_t> eventSplits{0};
        std::atomic<std::uint64_t> candidateStatistics{0};
        std::atomic<std::uint64_t> scoreEvaluations{0};
        std::atomic<std::uint64_t> macroBoundNanoseconds{0};
        std::atomic<std::uint64_t> macroAttempts{0}, macroAccepted{0}, macroRejected{0};
        std::atomic<std::uint64_t> macroUnavailable{0}, macroRetrySkips{0};
        std::atomic<std::uint64_t> candidateMacroBoundNanoseconds{0};
        std::atomic<std::uint64_t> candidateMacroQueries{0}, candidateMacroPrunedFragments{0};
        std::atomic<std::uint64_t> candidateMacroUnprunedFragments{0}, candidateMacroUnavailable{0};
        std::atomic<std::uint64_t> fineBoundaryQueries{0}, macroBoundaryQueries{0}, macroCandidateFragments{0};
    };
    ConditionalGpis::GPSSField _field;
    SingleHistoryGpis::Settings _settings;
    std::string _probabilityMethod = "numeric";
    PathPtr _cdfTableFile;
    std::shared_ptr<PhaseFunction> _deltaPhase;
    mutable std::array<ProfileRecord,4> _profile;
    mutable std::atomic<std::uint64_t> _discardedSurfacePaths{0};
    mutable std::atomic<std::uint64_t> _discardedSurfaceConnections{0};
    std::shared_ptr<const SingleHistoryPathContext> context(const Ray &,const MediumState &) const;
    void recordWork(bool surface,bool shadow,double seconds,double speedSeconds,
                    const SingleHistoryGpis::Work &) const;
public:
    SingleHistoryGpisMedium();
    void fromJson(JsonPtr value,const Scene &scene) override;
    rapidjson::Value toJson(Allocator &) const override;
    void loadResources() override;
    void prepareForRender() override;
    void teardownAfterRender() override;
    bool sampleDistance(PathSampleGenerator &,const Ray &,MediumState &,MediumSample &) const override;
    Vec3f transmittance(PathSampleGenerator &,const Ray &,bool,bool,MediumState *) const override;
    float pdf(PathSampleGenerator &,const Ray &,bool,bool) const override;
    Vec3f sigmaA(Vec3f) const override { return Vec3f(0.f); }
    Vec3f sigmaS(Vec3f) const override;
    Vec3f sigmaT(Vec3f p) const override { return sigmaS(p); }
    const ConditionalGpis::GPSSField &field() const { return _field; }
    const SingleHistoryGpis::Settings &flightSettings() const { return _settings; }
    std::uint64_t discardedSurfacePaths() const { return _discardedSurfacePaths.load(std::memory_order_relaxed); }
    std::uint64_t discardedSurfaceConnections() const { return _discardedSurfaceConnections.load(std::memory_order_relaxed); }
};

} // namespace Tungsten
