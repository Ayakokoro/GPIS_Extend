#ifndef FUNCTION_SPACE_PROFILE_HPP_
#define FUNCTION_SPACE_PROFILE_HPP_

#include <array>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace Tungsten { namespace FsProfile {

// Opt-in diagnostics only. Counters belong to individual rendering threads;
// the registry is locked once per thread, never in a matrix element loop.
enum class Stage { Intersect, Gradient, MeanCovariance, Mean, Covariance,
    Conditioning, PseudoInverse, Cholesky, EigenSquareRoot, Draw, Projection, Count };
enum class Counter { SampleDimension, ConditionDimension, PseudoInverseDimension,
    CholeskySuccess, CholeskyFallback, EigenFailure, Hit, Miss, Count };
constexpr std::size_t StageCount = static_cast<std::size_t>(Stage::Count);
constexpr std::size_t CounterCount = static_cast<std::size_t>(Counter::Count);
constexpr std::size_t DimensionCount = 66; // exact dimensions 0..64; 65 means >64
inline thread_local bool shadowQuery = false;
struct Scope;
struct Timing { std::uint64_t calls = 0, inclusiveNs = 0, exclusiveNs = 0; };
struct ThreadStats {
    Scope *top = nullptr;
    std::size_t domain = 0;
    std::array<std::array<Timing, StageCount>, 4> times{};
    std::array<std::array<std::array<std::uint64_t, DimensionCount>, CounterCount>, 4> counts{};
};

struct Profile {
    bool enabled;
    std::mutex mutex;
    std::vector<std::unique_ptr<ThreadStats>> threads;
    Profile() : enabled([] {
        const char *flag = std::getenv("TUNGSTEN_GPIS_FS_PROFILE");
        return flag && std::string(flag) == "1";
    }()) {}
    ThreadStats &thread() {
        static thread_local ThreadStats *stats = nullptr;
        if (!stats) {
            auto item = std::make_unique<ThreadStats>();
            stats = item.get();
            std::lock_guard<std::mutex> lock(mutex);
            threads.push_back(std::move(item));
        }
        return *stats;
    }
    ~Profile() {
        if (!enabled) return;
        ThreadStats total;
        for (const auto &thread : threads) {
            for (std::size_t d = 0; d < 4; ++d) {
                for (std::size_t s = 0; s < StageCount; ++s) {
                    auto &a = total.times[d][s];
                    const auto &b = thread->times[d][s];
                    a.calls += b.calls;
                    a.inclusiveNs += b.inclusiveNs;
                    a.exclusiveNs += b.exclusiveNs;
                }
                for (std::size_t c = 0; c < CounterCount; ++c)
                    for (std::size_t n = 0; n < DimensionCount; ++n)
                        total.counts[d][c][n] += thread->counts[d][c][n];
            }
        }
        const char *domains[] = {"path_intersect", "path_gradient", "shadow_intersect", "shadow_gradient"};
        const char *stages[] = {"intersect", "gradient", "mean_covariance", "mean", "covariance",
            "conditioning", "pseudo_inverse", "cholesky", "eigen_square_root", "draw", "projection"};
        const char *counters[] = {"sample_dimension", "condition_dimension", "pseudo_inverse_dimension",
            "cholesky_success", "cholesky_fallback", "eigen_failure", "hit", "miss"};
        std::cout << "GPIS function-space profile: {\"unit\":\"sum_worker_ms\",\"domains\":{";
        for (std::size_t d = 0; d < 4; ++d) {
            if (d) std::cout << ',';
            std::cout << '"' << domains[d] << "\":{\"stages\":{";
            for (std::size_t s = 0; s < StageCount; ++s) {
                if (s) std::cout << ',';
                const auto &t = total.times[d][s];
                std::cout << '"' << stages[s] << "\":{\"calls\":" << t.calls
                    << ",\"inclusive_ms\":" << t.inclusiveNs*1e-6
                    << ",\"exclusive_ms\":" << t.exclusiveNs*1e-6 << '}';
            }
            std::cout << "},\"counts\":{";
            for (std::size_t c = 0; c < CounterCount; ++c) {
                if (c) std::cout << ',';
                std::cout << '"' << counters[c] << "\":{";
                bool first = true;
                for (std::size_t n = 0; n < DimensionCount; ++n) {
                    const auto value = total.counts[d][c][n];
                    if (!value) continue;
                    if (!first) std::cout << ',';
                    first = false;
                    std::cout << '"' << n << "\":" << value;
                }
                std::cout << '}';
            }
            std::cout << "}}";
        }
        std::cout << "}}\n";
    }
};
inline Profile &profile() { static Profile value; return value; }

// Inclusive times explain call hierarchies. Exclusive times can be added
// without counting covariance/solver work twice inside conditioning.
struct Scope {
    using Clock = std::chrono::steady_clock;
    ThreadStats *stats = nullptr;
    Scope *parent = nullptr;
    Stage stage;
    std::size_t domain = 0, previousDomain = 0;
    std::uint64_t childrenNs = 0;
    Clock::time_point start;
    Scope(Stage stage, bool root = false) : stage(stage) {
        if (!profile().enabled) return;
        auto &thread = profile().thread();
        if (!root && !thread.top) return;
        stats = &thread;
        parent = stats->top;
        previousDomain = stats->domain;
        domain = root ? (shadowQuery ? 2 : 0) + (stage == Stage::Gradient ? 1 : 0) : previousDomain;
        stats->domain = domain;
        stats->top = this;
        start = Clock::now();
    }
    Scope(const Scope &) = delete;
    Scope &operator=(const Scope &) = delete;
    void stop() {
        if (!stats) return;
        const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start).count();
        auto &timing = stats->times[domain][static_cast<std::size_t>(stage)];
        ++timing.calls;
        timing.inclusiveNs += ns;
        timing.exclusiveNs += ns - childrenNs;
        if (parent) parent->childrenNs += ns;
        stats->top = parent;
        stats->domain = previousDomain;
        stats = nullptr;
    }
    ~Scope() { stop(); }
};
inline void count(Counter counter, std::size_t dimension = 0) {
    if (!profile().enabled) return;
    auto &thread = profile().thread();
    if (!thread.top) return;
    ++thread.counts[thread.domain][static_cast<std::size_t>(counter)][dimension < 65 ? dimension : 65];
}
struct PurposeScope {
    bool previous;
    explicit PurposeScope(bool shadow) : previous(shadowQuery) { shadowQuery = shadow; }
    ~PurposeScope() { shadowQuery = previous; }
};

} } // namespace Tungsten::FsProfile
#endif
