// Unconditioned 1D field covariance on an oblique ray through Dragon's
// procedural amplitude and isotropic length-scale fields. No VDB is needed.
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "math/CubicDowncrossing.hpp"
#include "math/SparseConvolutionNoise.hpp"
#include "media/TwoLevelLatticeRayField.hpp"
#include "sampling/UniformSampler.hpp"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <future>
#include <iostream>
#include <map>
#include <vector>

using namespace Tungsten;
using J = nlohmann::json;

namespace {
using Vector = Eigen::Vector3d;
using Weights = std::map<std::pair<int, std::int64_t>, double>;
const Vector origin(-0.6531, 0.37, 0.09);
const Vector direction = Vector(1.0, 0.21, -0.13).normalized();
Vec3f toFloat(const Vector &v) { return Vec3f(float(v.x()), float(v.y()), float(v.z())); }

std::shared_ptr<GaussianProcess> makeProcess() {
    JsonDocument doc(Path("lattice-multires-covariance.json"), R"({
      "type":"standard", "mean":{"type":"homogeneous","offset":0},
      "covariance":{"type":"proc_nonstationary", "multiResolutionGrid":true,
          "cov":{"type":"squared_exponential","sigma":1,"lengthScale":1},
          "var":{"type":"noise","noise":"bottom_top","min":3.0,"max":0.1,"start":0,"end":1.33},
          "ls":{"type":"noise","noise":"left_right","min":0.0316,"max":0.223,"start":-0.5,"end":0}}
    })");
    Scene scene;
    auto gp = scene.fetchGaussianProcess(doc)->get_gaussian_process();
    gp->loadResources();
    return gp;
}

// Hermite interpolation of the actual basis of a rendered piece. Its Gram
// matrix is the covariance for independent unit Gaussian lattice coefficients.
Weights coefficientRow(const TwoLevelLatticeModel &model, const TwoLevelLatticeRayField &field, double t) {
    const auto piece = field.piece(t);
    const double lo = piece.start, hi = piece.end;
    const double dt = hi-lo, u = (t-lo)/dt;
    const double leftValue = (1+2*u)*(1-u)*(1-u), leftSlope = u*(1-u)*(1-u)*dt;
    const double rightValue = u*u*(3-2*u), rightSlope = u*u*(u-1)*dt;
    Weights row;
    auto add = [&](double s, int side, double value, double slope) {
        auto p = field.at(s, side);
        model.kernel.forEachWeight(s, p, side, [&](int level, std::int64_t i, double b, double d) {
            row[{level,i}] += value*b + slope*d;
        });
    };
    add(lo, 1, leftValue, leftSlope);
    add(hi, -1, rightValue, rightSlope);
    return row;
}

double inner(const Weights &a, const Weights &b) {
    double result = 0;
    for (const auto &entry : a) {
        auto found = b.find(entry.first);
        if (found != b.end()) result += entry.second * found->second;
    }
    return result;
}

// Sparse signs have zero mean and unit variance. Independent uniform
// locations in each bin integrate to constant intensity. Only the same
// impulse contributes to the expectation of a product. Integrating the
// overlap of its two hard-truncated Gaussian kernels gives this expression.
double octaveOverlap(double distance, double ell) {
    const double x = std::abs(distance)/ell;
    const double overlap = 3.0-x/std::sqrt(2.0);
    return overlap > 0 ? std::exp(-0.5*x*x)*std::erf(overlap) : 0.0;
}

struct Site {
    double t, sigma;
    Vec3f position;
    Vec4f scales; // lower/upper lengths, then their weights from the sparse renderer
    Weights row;
};

double sparseCovariance(const Site &a, const Site &b, double baseEll) {
    double result = 0;
    for (int i=0; i<2; ++i) for (int j=0; j<2; ++j) {
        if (a.scales[i] == b.scales[j])
            result += a.scales[2+i]*b.scales[2+j]*octaveOverlap(a.t-b.t, baseEll*a.scales[i]);
    }
    return a.sigma*b.sigma*result;
}

// Dense sampling of position and lag to test dependence on lattice phase.
// Use the exact coefficient Gram so refinement differences have no MC noise.
J spacingSweep(const ProceduralNonstationaryCovariance &cov,
        SparseConvolutionNoiseRealization &ensemble, double spacing) {
    TwoLevelLatticeModel model(cov, spacing);
    TwoLevelLatticeRayField field(model, origin, direction, 0u);
    std::vector<Site> sites;
    for (int i=0; i<197; ++i) {
        const double t = 0.00231 + i*0.00721;
        const Vec3f pos = toFloat(origin+t*direction);
        sites.push_back({t, cov.sparseConvNoiseAmplitude(pos), pos, ensemble.kernelScaleLevelRatio(pos),
                coefficientRow(model,field,t)});
    }
    double maxError = 0, sumSq = 0;
    int pairs = 0;
    for (size_t i=0; i<sites.size(); ++i) for (size_t j=i; j<sites.size(); ++j) {
        const double error = (inner(sites[i].row,sites[j].row)-sparseCovariance(sites[i],sites[j],1.))
                / (sites[i].sigma*sites[j].sigma);
        maxError = std::max(maxError,std::abs(error));
        sumSq += error*error;
        ++pairs;
    }
    return {{"spacing_in_ell",spacing},{"pairs",pairs},{"max_abs_normalized_error",maxError},
            {"rms_normalized_error",std::sqrt(sumSq/pairs)}};
}

struct Moment {
    double sum = 0, squares = 0;
    void add(double x) { sum += x; squares += x*x; }
    void merge(const Moment &m) { sum += m.sum; squares += m.squares; }
    double mean(int n) const { return sum/n; }
    double se(int n) const {
        return std::sqrt(std::max(0.0,squares-sum*sum/n)/(double(n)*(n-1)));
    }
};
struct PairMoments {
    Moment sparse, lattice;
};

std::vector<PairMoments> sample(const std::shared_ptr<GaussianProcess> &gp,
        const TwoLevelLatticeModel &model, const std::vector<Site> &sites, int first, int end) {
    const size_t count = sites.size();
    std::vector<PairMoments> moments(count*(count+1)/2);
    std::vector<double> sparse(count), lattice(count);
    SparseConvolutionNoiseRealization ensemble(gp, GPCorrelationContext::RenewalPlus,
            23u, 10.0f, false, false, true, SparseConv1DSamplingScheme::MIS, false, false, 0.0f);
    UniformSampler sampler;
    const Vec3f dir = toFloat(direction);
    for (int draw=first; draw<end; ++draw) {
        // Hash the sample index before passing it as base seed. Consecutive
        // base seeds would reuse other samples' neighboring octave seeds.
        const uint32 seed = MathUtil::xxhash32(uint32(draw));
        TwoLevelLatticeRayField field(model, origin, direction, seed ^ 0x95b609d3u);
        for (size_t i=0; i<count; ++i) {
            const auto &site = sites[i];
            sparse[i] = site.sigma * ensemble.evaluateValueNoise1D(site.position, float(site.t),
                    dir, seed, sampler, false);
            const auto piece = field.piece(site.t);
            const auto cubic = field.segment(piece);
            lattice[i] = CubicDowncrossing::value(cubic, (site.t-piece.start)/piece.width());
        }
        size_t k = 0;
        for (size_t i=0; i<count; ++i) for (size_t j=i; j<count; ++j, ++k) {
            moments[k].sparse.add(sparse[i]*sparse[j]);
            moments[k].lattice.add(lattice[i]*lattice[j]);
        }
    }
    return moments;
}

J verify(int draws) {
    auto gp = makeProcess();
    const auto &cov = dynamic_cast<const ProceduralNonstationaryCovariance &>(*gp->_cov);
    if (!cov.useMultiResolutionGrid()) throw std::runtime_error("Sparse multiresolution disabled");
    constexpr double spacing = .5;
    TwoLevelLatticeModel model(cov, spacing);
    TwoLevelLatticeRayField field(model, origin, direction, 0u);
    SparseConvolutionNoiseRealization ensemble(gp, GPCorrelationContext::RenewalPlus,
            23u, 10.0f, false, false, true, SparseConv1DSamplingScheme::MIS, false, false, 0.0f);
    std::vector<Site> sites;
    std::map<int,float> seedOffsets;
    J points = J::array();
    // Pairs on either plateau, across the ramp's octave boundaries, on the
    // same two levels with varying weights, and with no shared octave.
    for (double x : {-0.6213,-0.604,-0.575,-0.501,-0.434,-0.414,-0.401,-0.373,
                     -0.257,-0.171,-0.144,-0.115,-0.009,0.013,0.081,0.177,0.305,0.893}) {
        const double t = (x-origin.x())/direction.x();
        const auto position = toFloat(origin+t*direction);
        auto scales = ensemble.kernelScaleLevelRatio(position);
        for (int j=0;j<2;++j) {
            const int offset = int(std::floor(std::log(scales[j])/std::log(ensemble._base)));
            auto entry = seedOffsets.emplace(offset, scales[j]);
            if (entry.first->second != scales[j])
                throw std::runtime_error("Two sparse octaves reuse a seed offset on this platform");
        }
        const auto piece = field.piece(t);
        const double sigma = cov.sparseConvNoiseAmplitude(position);
        sites.push_back({t,sigma,position,scales,coefficientRow(model,field,t)});
        points.push_back({{"world_x",x},{"t",t},{"sigma",sigma},{"ell",cov.sparseConvNoiseLateralScale(position)},
                {"levels",{scales[0],scales[1]}},{"weights",{scales[2],scales[3]}},
                {"local_piece_coordinate",(t-piece.start)/piece.width()}});
    }

    std::vector<std::future<std::vector<PairMoments>>> jobs;
    constexpr int workers = 4;
    for (int i=0; i<workers; ++i)
        jobs.push_back(std::async(std::launch::async, sample, gp, std::cref(model),
                std::cref(sites), draws*i/workers, draws*(i+1)/workers));
    auto moments = jobs.front().get();
    for (size_t i=1; i<jobs.size(); ++i) {
        auto block = jobs[i].get();
        for (size_t k=0; k<block.size(); ++k) {
            moments[k].sparse.merge(block[k].sparse);
            moments[k].lattice.merge(block[k].lattice);
        }
    }
    J pairs = J::array();
    double maxGram = 0, maxSparseZ = 0, maxLatticeZ = 0, maxMcDifference = 0;
    size_t k = 0;
    for (size_t i=0; i<sites.size(); ++i) for (size_t j=i; j<sites.size(); ++j, ++k) {
        const double target = sparseCovariance(sites[i], sites[j], 1.);
        const double gram = inner(sites[i].row, sites[j].row);
        const double scale = sites[i].sigma*sites[j].sigma;
        const auto &m = moments[k];
        maxGram = std::max(maxGram, std::abs(gram-target)/scale);
        maxSparseZ = std::max(maxSparseZ, std::abs(m.sparse.mean(draws)-target)/m.sparse.se(draws));
        maxLatticeZ = std::max(maxLatticeZ, std::abs(m.lattice.mean(draws)-gram)/m.lattice.se(draws));
        maxMcDifference = std::max(maxMcDifference, std::abs(m.sparse.mean(draws)-m.lattice.mean(draws))/scale);
        pairs.push_back({{"i",i},{"j",j},{"sparse_kernel_integral",target},{"lattice_gram",gram},
                {"normalized_gram_error",(gram-target)/scale},{"sparse_MC",m.sparse.mean(draws)},
                {"sparse_MC_SE",m.sparse.se(draws)},{"lattice_MC",m.lattice.mean(draws)},
                {"lattice_MC_SE",m.lattice.se(draws)}});
    }
    // Separate deterministic discretization error from sampling uncertainty.
    constexpr double gramTolerance = .01, zTolerance = 6.;
    J sweep = J::array();
    bool sweepPassed = true;
    for (double h : {spacing, spacing*.5}) {
        auto result = spacingSweep(cov,ensemble,h);
        sweepPassed &= result["max_abs_normalized_error"].get<double>()<=gramTolerance;
        sweep.push_back(std::move(result));
    }
    return {{"passed",sweepPassed && maxGram<=gramTolerance && maxSparseZ<=zTolerance && maxLatticeZ<=zTolerance},
            {"scope","Unconditioned field values along one oblique ray, 1D isotropic SE sparse convolution with hard radius 3 in isotropic coordinates. Sigma and length fields from Dragon. MC estimates E[f_i*f_j], using known zero ensemble means."},
            {"draws",draws},{"spacing_in_ell",spacing},{"impulses_per_sparse_bin",10},
            {"max_absolute_gram_error_divided_by_sigma_product",maxGram},
            {"max_absolute_MC_difference_divided_by_sigma_product",maxMcDifference},
            {"gram_tolerance",gramTolerance},{"z_tolerance",zTolerance},
            {"max_sparse_reference_error_in_SE",maxSparseZ},{"max_lattice_gram_error_in_SE",maxLatticeZ},
            {"phase_and_lag_sweep",sweep},{"points",points},{"pairs",pairs}};
}
} // namespace

int main(int argc, char **argv) try {
    if (argc>3) throw std::invalid_argument("Usage: lattice_multires_covariance [draws] [output.json]");
    const int draws = argc>1 ? std::stoi(argv[1]) : 32768;
    if (draws<128 || draws>100000000) throw std::invalid_argument("Invalid number of draws");
    auto result = verify(draws);
    if (argc==3) {
        std::ofstream out(argv[2]);
        out << result.dump(2) << '\n';
        if (!out) throw std::runtime_error("Failed to write covariance report");
    }
    result.erase("points");
    result.erase("pairs");
    std::cout << result.dump(2) << std::endl;
    return result.at("passed").get<bool>() ? 0 : 1;
} catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return 1;
}
