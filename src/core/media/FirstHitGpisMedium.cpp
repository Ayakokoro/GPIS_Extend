#include "FirstHitGpisMedium.hpp"
#include "GpisSdfGrid.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "io/FileUtils.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/mathutility/NumericPolicy.h"
#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <sstream>
#include <tuple>

namespace Tungsten {
namespace cg = ConditionalGpis;
namespace {
using FlightStatistics = FirstHitGpis::FlightStatistics;
struct StatisticField {
    const char *name;
    std::uint64_t FlightStatistics::*member;
};
constexpr StatisticField StatisticFields[] = {
    {"prefix_queries", &FlightStatistics::prefixQueries},
    {"precheck_calls", &FlightStatistics::precheckCalls},
    {"precheck_survived", &FlightStatistics::precheckSurvived},
    {"precheck_hit", &FlightStatistics::precheckHit},
    {"precheck_marginal_hits", &FlightStatistics::precheckMarginalHits},
    {"precheck_with_extra_nodes", &FlightStatistics::precheckWithExtraNodes},
    {"precheck_nanoseconds", &FlightStatistics::precheckNanoseconds},
    {"candidate_locations", &FlightStatistics::candidateLocations},
    {"candidate_tests", &FlightStatistics::candidateTests},
    {"node_insertions", &FlightStatistics::nodeInsertions},
    {"unique_added_nodes", &FlightStatistics::uniqueAddedNodes},
    {"max_active_nodes", &FlightStatistics::maxActiveNodes},
    {"max_effective_dimension", &FlightStatistics::maxEffectiveDimension},
    {"mean_queries", &FlightStatistics::meanQueries},
    {"orthant_calls", &FlightStatistics::orthantCalls},
    {"orthant_nanoseconds", &FlightStatistics::orthantNanoseconds},
    {"max_orthant_call_nanoseconds", &FlightStatistics::maxOrthantCallNanoseconds},
    {"multivariate_calls", &FlightStatistics::multivariateCalls},
    {"multivariate_nanoseconds", &FlightStatistics::multivariateNanoseconds},
    {"qmc_calls", &FlightStatistics::qmcCalls},
    {"qmc_nanoseconds", &FlightStatistics::qmcNanoseconds},
    {"max_qmc_call_nanoseconds", &FlightStatistics::maxQmcCallNanoseconds},
    {"qmc_points", &FlightStatistics::qmcPoints},
    {"qmc_reused_points", &FlightStatistics::qmcReusedPoints},
    {"qmc_preparations", &FlightStatistics::qmcPreparations},
    {"pair_calls", &FlightStatistics::pairCalls},
    {"pair_nanoseconds", &FlightStatistics::pairNanoseconds},
    {"flight_nanoseconds", &FlightStatistics::flightNanoseconds}
};
nlohmann::json statisticValues(const FlightStatistics &value)
{
    auto result = nlohmann::json::object();
    for (const auto &field : StatisticFields) result[field.name] = value.*field.member;
    return result;
}
}

struct FirstHitGpisMedium::Statistics {
    struct RayRecord {
        std::uint64_t id = 0;
        Vec4u identity = Vec4u(0u);
        unsigned sceneSeed = 0;
        Vec3d origin = Vec3d(0.0), direction = Vec3d(0.0);
        double length = 0.0, uniform = 0.0, distance = 0.0;
        const char *outcome = "miss";
        FlightStatistics work;

        nlohmann::json json() const
        {
            return {{"flight_id", id}, {"pixel_x", identity.x()}, {"pixel_y", identity.y()},
                {"sample_index", identity.z()}, {"path_segment", identity.w()}, {"scene_seed", sceneSeed},
                {"origin", {origin.x(), origin.y(), origin.z()}},
                {"unit_direction", {direction.x(), direction.y(), direction.z()}},
                {"length", length}, {"uniform", uniform}, {"sampled_distance", distance},
                {"outcome", outcome}, {"statistics", statisticValues(work)}};
        }
    };
    mutable std::mutex mutex;
    OutputStreamHandle csv, summary;
    Path csvPath, summaryPath;
    bool reported = false, csvFailed = false;
    std::uint64_t flights = 0, hits = 0, misses = 0, failures = 0;
    std::uint64_t classification[2][3] = {}; // extra nodes, hit/miss/failed
    FlightStatistics sum, maximum;
    std::map<std::uint64_t, std::uint64_t> uniqueHistogram, activeHistogram, effectiveHistogram;
    RayRecord mostSelected, mostActive, slowestOrthant, slowestFlight;
    std::string mediumName;
    FirstHitGpis::Settings settings;

    Statistics(const Path &prefix, std::string name, const FirstHitGpis::Settings &options)
        : csvPath(prefix + "-rays.csv"), summaryPath(prefix + "-summary.json"),
          mediumName(std::move(name)), settings(options)
    {
        const auto parent = prefix.parent();
        if (!parent.empty() && !FileUtils::createDirectory(parent, true))
            throw std::runtime_error("first_hit_gpis: cannot create statistics directory " + parent.asString());
        csv = FileUtils::openOutputStream(csvPath);
        summary = FileUtils::openOutputStream(summaryPath);
        if (!csv || !summary)
            throw std::runtime_error("first_hit_gpis: cannot open statistics output prefix " + prefix.asString());
        *csv << "flight_id,pixel_x,pixel_y,sample_index,path_segment,scene_seed,origin_x,origin_y,origin_z,"
                "direction_x,direction_y,direction_z,length,uniform,outcome,sampled_distance";
        for (const auto &field : StatisticFields) *csv << ',' << field.name;
        *csv << ",orthant_ms,multivariate_ms,qmc_ms,pair_ms,flight_ms\n";
        *csv << std::setprecision(17);
        if (!csv->good()) throw std::runtime_error("first_hit_gpis: cannot write statistics CSV header");
    }

    static bool precedes(const RayRecord &a, const RayRecord &b)
    {
        return std::make_tuple(a.identity.y(), a.identity.x(), a.identity.z(), a.identity.w(), a.id) <
               std::make_tuple(b.identity.y(), b.identity.x(), b.identity.z(), b.identity.w(), b.id);
    }

    void record(const RayInfo &identity, const Vec3d &origin, const Vec3d &direction,
                double length, double uniform, double distance, const char *outcome,
                const FlightStatistics &work)
    {
        std::lock_guard<std::mutex> lock(mutex);
        RayRecord ray;
        ray.id = flights++;
        ray.identity = identity.pixelSampleSegment;
        ray.sceneSeed = identity.sceneSeed;
        ray.origin = origin; ray.direction = direction;
        ray.length = length; ray.uniform = uniform; ray.distance = distance;
        ray.outcome = outcome; ray.work = work;
        const bool hit = std::string(outcome) == "hit";
        const bool failed = std::string(outcome) == "failed";
        hits += hit; failures += failed; misses += !hit && !failed;
        ++classification[work.uniqueAddedNodes != 0][failed ? 2 : (hit ? 0 : 1)];
        ++uniqueHistogram[work.uniqueAddedNodes];
        ++activeHistogram[work.maxActiveNodes];
        ++effectiveHistogram[work.maxEffectiveDimension];
        for (const auto &field : StatisticFields) {
            sum.*field.member += work.*field.member;
            maximum.*field.member = std::max(maximum.*field.member, work.*field.member);
        }
        auto update = [&](RayRecord &best, std::uint64_t FlightStatistics::*member) {
            if (flights == 1 || work.*member > best.work.*member ||
                (work.*member == best.work.*member && precedes(ray, best))) best = ray;
        };
        update(mostSelected, &FlightStatistics::uniqueAddedNodes);
        update(mostActive, &FlightStatistics::maxActiveNodes);
        update(slowestOrthant, &FlightStatistics::orthantNanoseconds);
        update(slowestFlight, &FlightStatistics::flightNanoseconds);
        *csv << ray.id << ',' << ray.identity.x() << ',' << ray.identity.y() << ',' << ray.identity.z()
             << ',' << ray.identity.w() << ',' << ray.sceneSeed
             << ',' << origin.x() << ',' << origin.y() << ',' << origin.z()
             << ',' << direction.x() << ',' << direction.y() << ',' << direction.z()
             << ',' << length << ',' << uniform << ',' << outcome << ',' << distance;
        for (const auto &field : StatisticFields) *csv << ',' << work.*field.member;
        *csv << ',' << work.orthantNanoseconds*1e-6 << ',' << work.multivariateNanoseconds*1e-6
             << ',' << work.qmcNanoseconds*1e-6
             << ',' << work.pairNanoseconds*1e-6 << ',' << work.flightNanoseconds*1e-6 << '\n';
        csvFailed = csvFailed || !csv->good();
    }

    void report() noexcept
    {
        try {
            std::lock_guard<std::mutex> lock(mutex);
            if (reported) return;
            reported = true;
            csv->flush();
            csvFailed = csvFailed || !csv->good();
            auto histogram = [](const std::map<std::uint64_t, std::uint64_t> &bins) {
                auto result = nlohmann::json::object();
                for (const auto &entry : bins) result[std::to_string(entry.first)] = entry.second;
                return result;
            };
            auto timing = [&](std::uint64_t calls, std::uint64_t ns, std::uint64_t maxNs) {
                return nlohmann::json{{"calls", calls}, {"total_ms", ns*1e-6},
                    {"mean_ms_per_call", calls ? ns*1e-6/double(calls) : 0.0},
                    {"mean_ms_per_flight", flights ? ns*1e-6/double(flights) : 0.0},
                    {"max_ms_per_flight", maxNs*1e-6}};
            };
            auto nodeClass = [&](int index) {
                return nlohmann::json{{"hits", classification[index][0]}, {"misses", classification[index][1]},
                    {"failed", classification[index][2]},
                    {"completed", classification[index][0] + classification[index][1]}};
            };
            const auto noExtra = classification[0][0] + classification[0][1];
            const auto completed = hits + misses;
            nlohmann::json result = {
                {"schema_version", 1}, {"medium", mediumName},
                {"csv_path", csvPath.absolute().asString()}, {"csv_write_failed", csvFailed},
                {"definitions", {
                    {"ray", "One finite first-hit flight; a pixel/sample may produce multiple flights at boundaries."},
                    {"without_extra_nodes", "No internal location was selected in any prefix query, including distance inversion."},
                    {"candidate_locations", "Prepared cheap candidate pool, including endpoints; not CDF dimensions or quadrature points."},
                    {"unique_added_nodes", "Distinct selected internal positions accumulated over the whole flight."},
                    {"max_active_nodes", "Largest node set passed to the orthant backend, including endpoints; zero if all queries were prechecked. Distinct selected points across prefixes can exceed this."},
                    {"precheck_calls", "Cheap threshold-bound attempts before a new orthant node set, excluding QMC budget increases and survivalBounds queries."},
                    {"precheck_survived", "Prefix queries decided positive by the Rice lower bound, avoiding the next orthant evaluation."},
                    {"precheck_hit", "Prefix queries decided negative by a marginal or origin-pair upper bound, avoiding the next orthant evaluation."},
                    {"precheck_marginal_hits", "Subset of precheck_hit decided by a one-dimensional marginal upper bound."},
                    {"precheck_with_extra_nodes", "Successful prechecks with more than two retained nodes; not necessarily avoided QMC because the backend also has analytic cases and air pruning."},
                    {"precheck_time", "Cheap-bound work, including pair cache misses but excluding prepare(); overlaps pair time and is a subset of flight time."},
                    {"max_effective_dimension", "Largest orthant dimension after omitting extreme-air constraints; does not imply QMC was used."},
                    {"time", "Milliseconds summed across workers, not renderer wall time. Orthant includes factorization and QMC; QMC is a subset, not additive."},
                    {"multivariate_time", "Full orthant calls whose effective dimension is at least three, including analytic three-dimensional cases."},
                    {"qmc_time", "Integration loops only for completed QMC calls; backend exceptions may leave partial QMC work unreported."},
                    {"qmc_points", "Newly evaluated Sobol points across all eight replicates; cached points are excluded."},
                    {"qmc_reused_points", "Previously evaluated Sobol points reused in later estimates. New plus reused equals the work requested without incremental integration."},
                    {"qmc_preparations", "QMC integrand/matrix preparations. Budget increases reuse these; changed nodes or query endpoints rebuild them."},
                    {"pair_time", "Separate downcrossing-pair evaluations for bounds and candidate scores, excluding their cache hits."},
                    {"flight_time", "Distance-sampler work, excluding phase sampling, CSV output, and statistics merging."},
                    {"pixel_coordinates", "Zero-based image x/y; sample_index is the renderer sample index."},
                    {"failed", "Incomplete numerical flights are counted separately and never treated as valid misses."}}},
                {"settings", {{"probability_tolerance", settings.probabilityTolerance},
                    {"distance_tolerance", settings.distanceTolerance}, {"max_nodes", settings.maxNodes},
                    {"qmc_samples", settings.qmcSamples}, {"max_qmc_samples", settings.maxQmcSamples},
                    {"max_refinements", settings.maxRefinements}, {"survival_precheck", settings.survivalPrecheck}}},
                {"counts", {{"flights", flights}, {"completed", completed}, {"hits", hits},
                    {"misses", misses}, {"failed", failures}, {"without_extra_nodes", noExtra},
                    {"with_extra_nodes", completed-noExtra},
                    {"without_extra_nodes_fraction", completed ? double(noExtra)/double(completed) : 0.0}}},
                {"classification", {{"without_extra_nodes", nodeClass(0)}, {"with_extra_nodes", nodeClass(1)}}},
                {"sum_per_flight_statistics", statisticValues(sum)},
                {"max_per_flight_statistics", statisticValues(maximum)},
                {"histograms", {{"unique_added_nodes", histogram(uniqueHistogram)},
                    {"max_active_nodes", histogram(activeHistogram)},
                    {"max_effective_dimension", histogram(effectiveHistogram)}}},
                {"timing", {{"orthant", timing(sum.orthantCalls, sum.orthantNanoseconds, maximum.orthantNanoseconds)},
                    {"precheck", timing(sum.precheckCalls, sum.precheckNanoseconds, maximum.precheckNanoseconds)},
                    {"multivariate", timing(sum.multivariateCalls, sum.multivariateNanoseconds, maximum.multivariateNanoseconds)},
                    {"qmc", timing(sum.qmcCalls, sum.qmcNanoseconds, maximum.qmcNanoseconds)},
                    {"pair", timing(sum.pairCalls, sum.pairNanoseconds, maximum.pairNanoseconds)},
                    {"flight", timing(flights, sum.flightNanoseconds, maximum.flightNanoseconds)}}}
            };
            result["timing"]["orthant"]["max_ms_per_call"] = maximum.maxOrthantCallNanoseconds*1e-6;
            result["timing"]["qmc"]["max_ms_per_call"] = maximum.maxQmcCallNanoseconds*1e-6;
            result["timing"]["orthant"]["fraction_of_flight_time"] = sum.flightNanoseconds ?
                double(sum.orthantNanoseconds)/double(sum.flightNanoseconds) : 0.0;
            result["maxima"] = flights ? nlohmann::json{{"unique_added_nodes", mostSelected.json()},
                {"active_nodes", mostActive.json()}, {"orthant_time_per_flight", slowestOrthant.json()},
                {"flight_time", slowestFlight.json()}} : nlohmann::json::object();
            *summary << result.dump(2) << '\n';
            summary->flush();
            if (!summary->good() || csvFailed)
                std::cerr << "first_hit_gpis: failed to write complete statistics output " << summaryPath << '\n';
            std::cout << "GPIS first-hit statistics: flights=" << flights << ", hits=" << hits
                      << ", misses=" << misses << ", failed=" << failures
                      << ", no_extra_nodes=" << noExtra << '/' << completed
                      << ", max_added_nodes=" << maximum.uniqueAddedNodes
                      << ", max_active_nodes=" << maximum.maxActiveNodes
                      << ", orthant_ms=" << sum.orthantNanoseconds*1e-6
                      << ", qmc_ms=" << sum.qmcNanoseconds*1e-6
                      << ", qmc_points=" << sum.qmcPoints
                      << ", qmc_reused_points=" << sum.qmcReusedPoints
                      << ", precheck_decisions=" << sum.precheckSurvived+sum.precheckHit
                      << ", summary=" << summaryPath.absolute().asString() << '\n';
            csv.reset(); summary.reset();
        } catch (const std::exception &error) {
            std::cerr << "first_hit_gpis: cannot finalize statistics: " << error.what() << '\n';
        } catch (...) {
            std::cerr << "first_hit_gpis: cannot finalize statistics (unknown error)\n";
        }
    }
};

FirstHitGpisMedium::FirstHitGpisMedium()
{
    _maxBounce = 1;
    // ClassicGlobalMedium installed its phase object with a reference to this
    // base subobject. Keep that same object and its normal/gradient sampling.
}

FirstHitGpisMedium::~FirstHitGpisMedium() = default;

void FirstHitGpisMedium::fromJson(JsonPtr value, const Scene &scene)
{
    ClassicGlobalMedium::fromJson(value, scene);
    if (_maxBounce != 1)
        value.parseError("first_hit_gpis supports exactly one medium event; max_bounces must be 1");
    if (value["tracking_step"])
        value.parseError("first_hit_gpis does not use classic_global tracking_step");
    _firstHit = FirstHitGpis::Settings{};
    _statisticsEnabled = false;
    _statisticsOutputPrefix = "first-hit-stats";
    _statisticsScene = &scene;
    _statisticsInputDirectory = Path(".");
    _statisticsInputDirectory.freezeWorkingDirectory();
    if (auto settings = value["first_hit"]) {
        settings.getField("probability_tolerance", _firstHit.probabilityTolerance);
        settings.getField("distance_tolerance", _firstHit.distanceTolerance);
        settings.getField("max_nodes", _firstHit.maxNodes);
        settings.getField("qmc_samples", _firstHit.qmcSamples);
        settings.getField("max_qmc_samples", _firstHit.maxQmcSamples);
        settings.getField("max_refinements", _firstHit.maxRefinements);
        settings.getField("survival_precheck", _firstHit.survivalPrecheck);
        if (auto statistics = settings["statistics"]) {
            statistics.getField("enabled", _statisticsEnabled);
            statistics.getField("output_prefix", _statisticsOutputPrefix);
            if (_statisticsEnabled && _statisticsOutputPrefix.empty())
                statistics.parseError("first_hit_gpis statistics output_prefix must not be empty");
        }
        if (!std::isfinite(_firstHit.probabilityTolerance) ||
            !(_firstHit.probabilityTolerance > 0.0 && _firstHit.probabilityTolerance < 1.0) ||
            !std::isfinite(_firstHit.distanceTolerance) || !(_firstHit.distanceTolerance > 0.0) ||
            _firstHit.maxNodes < 2 || _firstHit.qmcSamples < 1 ||
            _firstHit.maxQmcSamples < _firstHit.qmcSamples || _firstHit.maxRefinements < 1)
            settings.parseError("first_hit_gpis: invalid first_hit probability/distance tolerance or resource limit");
    }
}

rapidjson::Value FirstHitGpisMedium::toJson(Allocator &allocator) const
{
    auto value = ClassicGlobalMedium::toJson(allocator);
    value["type"].SetString("first_hit_gpis", allocator);
    value.RemoveMember("tracking_step");
    value.AddMember("first_hit", rapidjson::Value(JsonObject{allocator,
        "probability_tolerance", _firstHit.probabilityTolerance,
        "distance_tolerance", _firstHit.distanceTolerance,
        "max_nodes", _firstHit.maxNodes,
        "qmc_samples", _firstHit.qmcSamples,
        "max_qmc_samples", _firstHit.maxQmcSamples,
        "max_refinements", _firstHit.maxRefinements,
        "survival_precheck", _firstHit.survivalPrecheck,
        "statistics", JsonObject{allocator, "enabled", _statisticsEnabled,
            "output_prefix", _statisticsOutputPrefix}}), allocator);
    return value;
}

void FirstHitGpisMedium::prepareForRender()
{
    _statistics.reset();
    _firstHit.validate();
    const auto *kernel = dynamic_cast<const SquaredExponentialCovariance *>(_covariance.get());
    if (!kernel || !kernel->isStationaryKernel() || !kernel->isIsotropic())
        throw cg::NumericError(cg::NumericStatus::InvalidInput,
            "first_hit_gpis requires an isotropic stationary squared_exponential covariance");
    ClassicGlobalMedium::prepareForRender();
    if (!_sdfGrid && !_mean->sdfQueries() &&
        !dynamic_cast<const HomogeneousMean *>(_mean.get()) &&
        !dynamic_cast<const SphericalMean *>(_mean.get()) &&
        !dynamic_cast<const LinearMean *>(_mean.get()))
        throw cg::NumericError(cg::NumericStatus::InvalidInput,
            "first_hit_gpis requires an analytic mean, trilinear VDB, or procedural SdfQueries; finite-difference-only gradients do not define its Rice rate");
    const Eigen::Matrix3d covariance = _ndf->covariance();
    const double derivativeVariance = covariance.trace()/3.0;
    if (!(derivativeVariance > 0.0) || !std::isfinite(derivativeVariance) ||
        (covariance - derivativeVariance*Eigen::Matrix3d::Identity()).cwiseAbs().maxCoeff()
            > 1e-10*derivativeVariance)
        throw cg::NumericError(cg::NumericStatus::InvalidCovariance,
            "first_hit_gpis requires isotropic positive derivative covariance");
    _lengthScale = kernel->lengthScale();
    if (!std::isfinite(_lengthScale) || !(_lengthScale > 0.0))
        throw cg::NumericError(cg::NumericStatus::InvalidInput,
            "first_hit_gpis has an invalid physical correlation length");
    if (_statisticsEnabled) {
        Path directory = _statisticsScene ? _statisticsScene->rendererSettings().outputDirectory() : Path();
        if (directory.empty()) directory = _statisticsInputDirectory;
        Path prefix(_statisticsOutputPrefix);
        if (prefix.isRelative()) prefix.setWorkingDirectory(directory);
        prefix = prefix.absolute();
        _statistics = std::make_unique<Statistics>(prefix, name(), _firstHit);
    }
}

void FirstHitGpisMedium::teardownAfterRender()
{
    if (_statistics) _statistics->report();
    ClassicGlobalMedium::teardownAfterRender();
}

bool FirstHitGpisMedium::sampleDistance(PathSampleGenerator &sampler, const Ray &ray,
                                      MediumState &state, MediumSample &sample) const
try {
    // The first direct-light estimate has already been added by handleVolume.
    // Returning false terminates continuation without tracing another GP flight.
    if (state.bounce > 0 || state.info.pixelSampleSegment.w() > 0) return false;
    const Vec3d origin(ray.pos()), direction(ray.dir());
    const double speed = direction.length();
    if (!std::isfinite(ray.farT()) || ray.farT() < 0.0f ||
        !std::isfinite(speed) || !(speed > 0.0) ||
        !std::isfinite(origin.x()) || !std::isfinite(origin.y()) || !std::isfinite(origin.z()))
        throw cg::NumericError(cg::NumericStatus::InvalidInput,
            "first_hit_gpis needs a finite boundary-enclosed ray with a finite nonzero direction");
    const Vec3d unitDirection = direction/speed;
    const Eigen::Vector3d eigenOrigin(origin.x(), origin.y(), origin.z());
    const Eigen::Vector3d eigenDirection(unitDirection.x(), unitDirection.y(), unitDirection.z());
    const double length = double(ray.farT())*speed;
    if (!std::isfinite(length))
        throw cg::NumericError(cg::NumericStatus::InvalidInput, "first_hit_gpis ray length overflow");

    std::unique_ptr<cg::MeanRayCursor> cursor;
#if OPENVDB_AVAILABLE
    if (_sdfGrid) cursor = _sdfGrid->createRayCursor(eigenOrigin, eigenDirection);
#endif
    const auto *sphere = dynamic_cast<const SphericalMean *>(_mean.get());
    const auto *linear = dynamic_cast<const LinearMean *>(_mean.get());
    const bool constant = dynamic_cast<const HomogeneousMean *>(_mean.get()) != nullptr;
    FirstHitGpis::MeanRay mean;
    mean.gradientBound = std::max(_gradientBound, _lipschitz);
    mean.sample = [&, sphere, linear, constant](double distance) -> FirstHitGpis::MeanSample {
        if (cursor) {
            const auto point = cursor->sample(distance);
            return {point.value, point.derivative};
        }
        const Vec3d point = origin + distance*unitDirection;
        if (sphere) {
            const Vec3d radial = point - sphere->center();
            const double radius = radial.length();
            // The ray is split at closest approach. At the sphere center an
            // arbitrary finite derivative affects no probability integral.
            return {radius-sphere->radius(), radius > 0.0 ? radial.dot(unitDirection)/radius : 0.0};
        }
        if (linear) {
            const Vec3d gradient = linear->gradient();
            const double rawValue = gradient.dot(point-linear->referencePoint());
            return {std::max(rawValue, linear->minimum()),
                    rawValue > linear->minimum() ? gradient.dot(unitDirection) : 0.0};
        }
        const double value = meanValue(point);
        return {value, constant ? 0.0 : meanGradient(point).dot(unitDirection)};
    };
    mean.minimum = [&](double begin, double end) {
        if (cursor) {
            const auto bounds = cursor->rayBounds(begin, end);
            if (bounds.certified) return bounds.minimumValue;
        } else if (const auto *queries = _mean->sdfQueries()) {
            const auto bounds = queries->rayBounds(eigenOrigin, eigenDirection, begin, end);
            if (bounds.certified) return bounds.minimumValue;
        }
        double minimum;
        if (sphere) {
            const double closest = std::clamp((sphere->center()-origin).dot(unitDirection), begin, end);
            minimum = (origin+closest*unitDirection-sphere->center()).length()-sphere->radius();
        } else if (linear) {
            minimum = std::min(mean.sample(begin).value, mean.sample(end).value);
        } else {
            const double midpoint = begin+0.5*(end-begin);
            minimum = meanValue(origin+midpoint*unitDirection)-mean.gradientBound*0.5*(end-begin);
        }
        const double padding = 64.0*std::numeric_limits<double>::epsilon()*
            (1.0+std::abs(minimum)+mean.gradientBound*(end-begin));
        return minimum-padding;
    };
    mean.nextBoundary = [&](double begin, double end) {
        if (cursor) return cursor->nextBoundary(begin, end);
        double boundary = end;
        if (sphere) boundary = (sphere->center()-origin).dot(unitDirection);
        else if (linear) {
            const double slope = linear->gradient().dot(unitDirection);
            if (slope != 0.0)
                boundary = (linear->minimum()-linear->gradient().dot(origin-linear->referencePoint()))/slope;
        }
        return boundary > begin && boundary < end ? boundary : end;
    };

    // nearT belongs to geometry self-intersection handling. The GP first-hit
    // event is conditioned at the actual origin and includes [0, farT].
    double distance = length;
    if (_statistics) {
        FlightStatistics work;
        double uniform = std::numeric_limits<double>::quiet_NaN();
        try {
            if (length > 0.0) {
                FirstHitGpis::Flight flight(mean, _sigma, _lengthScale, length, _firstHit, &work);
                uniform = ClassicGlobal::openUniform(sampler);
                distance = flight.sampleDistance(uniform);
            }
        } catch (...) {
            _statistics->record(state.info, origin, unitDirection, length, uniform,
                std::numeric_limits<double>::quiet_NaN(), "failed", work);
            throw;
        }
        _statistics->record(state.info, origin, unitDirection, length, uniform, distance,
            distance == length ? "miss" : "hit", work);
    } else if (length > 0.0) {
        FirstHitGpis::Flight flight(mean, _sigma, _lengthScale, length, _firstHit);
        distance = flight.sampleDistance(ClassicGlobal::openUniform(sampler));
    }
    sample = MediumSample{};
    sample.exited = distance == length;
    sample.t = sample.exited ? ray.farT() : float(distance/speed);
    if (!sample.exited && sample.t >= ray.farT())
        sample.t = std::nextafter(ray.farT(), 0.0f);
    sample.p = ray.pos()+sample.t*ray.dir();
    sample.phase = _phaseFunction.get();
    sample.weight = sample.continuedWeight = Vec3f(1.0f);
    sample.continuedT = sample.t;
    sample.emission = Vec3f(0.0f);
    // Forward free-flight sampling cancels its sampled measure. This is not a
    // marginalized distance PDF; pdf() explicitly rejects such requests.
    sample.pdf = 1.0f;
    sample.sparseConv1DSamplingScheme = SparseConv1DSamplingScheme::UNI;
    sample.rayInfo = state.info;
    if (!sample.exited) {
        if (hasDiffuseMicroMaterial()) {
            const auto normal = sampleVisibleNormal(Vec3d(sample.p), eigenDirection, sampler);
            sample.aniso = Vec3d(normal.x(), normal.y(), normal.z());
        }
        state.advance();
    }
    return true;
} catch (const cg::NumericError &) {
    throw;
} catch (const std::exception &error) {
    // PathTracer must not swallow a numerical failure as a black sample.
    throw cg::NumericError(cg::NumericStatus::InvalidInput, error.what());
}

Vec3f FirstHitGpisMedium::transmittance(PathSampleGenerator &, const Ray &, bool,
                                     bool, MediumState *) const
{
    // These are light connections for a first-hit shading preview. The
    // integrator still traces opaque/transparent geometry. Applying the
    // empty-origin GP survival law here would incorrectly condition a ray
    // that was born at a collision; that law is intentionally not implemented.
    return Vec3f(1.0f);
}

float FirstHitGpisMedium::pdf(PathSampleGenerator &, const Ray &, bool, bool) const
{
    throw cg::NumericError(cg::NumericStatus::InvalidInput,
        "first_hit_gpis supports the CPU path_tracer preview; BDPT distance PDFs are not implemented");
}

}
