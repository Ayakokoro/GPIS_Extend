// Ported from macrofacet; see ../PORT.md for provenance and adaptations.
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include "math/conditional/mathutility/RootFinding.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

namespace Tungsten::ConditionalGpis {

PositiveResult positiveRawMoment(int order, double mean, double stddev,
                                 const NumericPolicy& policy) {
    if (order < 0 || order > 3 || stddev < 0.0 || !std::isfinite(mean) ||
        !std::isfinite(stddev)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid truncated Gaussian moment");
    }
    if (stddev == 0.0) {
        if (!(mean > 0.0)) return exactZero();
        const double value = order == 0 ? 1.0 : std::pow(mean, order);
        return {value, std::log(value), 0.0, NumericStatus::Degenerate};
    }
    const double a = mean / stddev;
    if (order == 0) return positiveFromLog(normalLogCdf(a));

    // Direct expressions are excellent except in the cancelling negative tail.
    if (a > -5.0) {
        const double p = normalCdf(a);
        const double d = normalPdf(a);
        double value = 0.0;
        if (order == 1) value = mean * p + stddev * d;
        if (order == 2) value = (mean * mean + stddev * stddev) * p + mean * stddev * d;
        if (order == 3) {
            value = (mean * mean * mean + 3.0 * mean * stddev * stddev) * p +
                    (mean * mean * stddev + 2.0 * stddev * stddev * stddev) * d;
        }
        value = validateNonnegative(value, std::pow(std::abs(mean) + stddev, order), policy);
        if (value > 0.0) return {value, std::log(value), 0.0, NumericStatus::Ok};
    }

    const double r = static_cast<double>(order);
    const double root = std::hypot(a, 2.0 * std::sqrt(r));
    const double mode = a >= 0.0 ? 0.5 * (a + root) : 2.0 * r / (root - a);
    auto logIntegrand = [=](double u) {
        if (!(u > 0.0)) return -std::numeric_limits<double>::infinity();
        return r * std::log(u) + normalLogPdf(u - a);
    };
    PositiveResult dimensionless = integratePositiveLog(logIntegrand, 0.0, mode, policy);
    if (dimensionless.status == NumericStatus::ExactZero) return dimensionless;
    const double logValue = r * std::log(stddev) + dimensionless.logValue;
    return positiveFromLog(logValue, dimensionless.absError * std::pow(stddev, order),
                           dimensionless.status);
}

PositiveResult negativePartMean(double mean, double stddev, const NumericPolicy& policy) {
    return positiveRawMoment(1, -mean, stddev, policy);
}

double negativeFluxNormalLogPdf(double k, double mean, double stddev,
                                const NumericPolicy& policy) {
    if (!(k < 0.0)) return -std::numeric_limits<double>::infinity();
    const PositiveResult normalizer = negativePartMean(mean, stddev, policy);
    if (normalizer.status == NumericStatus::ExactZero) {
        throw NumericError(NumericStatus::InvalidInput, "negative-flux distribution has no mass");
    }
    if (stddev == 0.0) {
        return k == mean ? std::numeric_limits<double>::infinity()
                         : -std::numeric_limits<double>::infinity();
    }
    const double z = (k - mean) / stddev;
    return std::log(-k) + normalLogPdf(z) - std::log(stddev) - normalizer.logValue;
}

PositiveResult negativeFluxNormalCdf(double k, double mean, double stddev,
                                     const NumericPolicy& policy) {
    if (stddev < 0.0 || !std::isfinite(mean) || !std::isfinite(stddev)) {
        throw NumericError(NumericStatus::InvalidInput, "invalid negative-flux Gaussian");
    }
    if (k >= 0.0) return {1.0, 0.0, 0.0, NumericStatus::Ok};
    if (stddev == 0.0) {
        if (!(mean < 0.0)) return exactZero();
        return k < mean ? exactZero() : PositiveResult{1.0, 0.0, 0.0, NumericStatus::Degenerate};
    }
    const PositiveResult denominator = negativePartMean(mean, stddev, policy);
    if (denominator.status == NumericStatus::ExactZero) return exactZero();
    const double z = (k - mean) / stddev;
    const double direct = stddev * normalPdf(z) - mean * normalCdf(z);
    if (direct > 0.0 && std::isfinite(direct) && denominator.value > 0.0 &&
        (mean <= 0.0 || z > -5.0)) {
        const double value = std::clamp(direct / denominator.value, 0.0, 1.0);
        return {value, value > 0.0 ? std::log(value) : -std::numeric_limits<double>::infinity(),
                0.0, value > 0.0 ? NumericStatus::Ok : NumericStatus::ExactZero};
    }

    // sigma*phi(z)-mean*Phi(z) = sigma*E[(N(z,1))_+] + (-k)*Phi(z).
    // Both terms are positive; log addition avoids cancelling Gaussian tails
    // and still works when the ordinary normalizer underflows.
    const double first = std::log(stddev) + positiveRawMoment(1, z, 1.0, policy).logValue;
    const double second = std::log(-k) + normalLogCdf(z);
    const double largest = std::max(first, second);
    if (!std::isfinite(largest)) return exactZero();
    const double logNumerator = largest + std::log1p(std::exp(std::min(first, second) - largest));
    return positiveFromLog(std::min(0.0, logNumerator - denominator.logValue));
}

double sampleNegativeFluxNormal(double mean, double stddev, Random& rng,
                                const NumericPolicy& policy) {
    if (stddev == 0.0) {
        if (mean < 0.0) return mean;
        throw NumericError(NumericStatus::InvalidInput, "negative-flux distribution has no mass");
    }
    if (!(stddev > 0.0) || !std::isfinite(stddev) || !std::isfinite(mean))
        throw NumericError(NumericStatus::InvalidInput, "invalid negative-flux Gaussian");
    if (mean>=0.0) {
        // For X=-K>0 the unnormalized density is
        // X exp(-mean*X/sigma^2-X^2/(2*sigma^2)). The common Gaussian tail
        // factor cancels. Rayleigh and Gamma(2) envelopes give exact rejection
        // samplers without subtracting enormous tail log probabilities.
        const double a=mean/stddev;
        const double scale=a>=1.0 ? stddev/a : stddev;
        if (!(scale>0.0)) throw NumericError(NumericStatus::NeedHigherPrecision,"flux sampling scale underflowed");
        for (;;) {
            const double y=a>=1.0
                ? -std::log(rng.openUniform01())-std::log(rng.openUniform01())
                : std::sqrt(-2.0*std::log(rng.openUniform01()));
            const double logAcceptance=a>=1.0 ? -0.5*(y/a)*(y/a) : -a*y;
            if (std::log(rng.openUniform01())>logAcceptance) continue;
            const double result=-scale*y;
            if (!(result<0.0)) throw NumericError(NumericStatus::NeedHigherPrecision,"flux sample underflowed");
            return result;
        }
    }
    const double u=rng.openUniform01(), a=-mean/stddev;
    // Invert directly in Z=(K-mean)/sigma. Converting to world K inside the
    // CDF erases Z when sigma is small relative to |mean|, making Newton stall.
    const double inverseA=1.0/a;
    const double denominator=a>=1.0 ? normalCdf(a)+normalPdf(a)*inverseA
                                   : a*normalCdf(a)+normalPdf(a);
    auto evaluate = [&](double z) {
        if (z>=a) return RootEvaluation{1.0,0.0};
        const double pdf=normalPdf(z), cdf=normalCdf(z);
        return RootEvaluation{
            (a>=1.0 ? cdf+pdf*inverseA : a*cdf+pdf)/denominator,
            (a>=1.0 ? 1.0-z*inverseA : a-z)*pdf/denominator};
    };
    double lower=-8.0, upper=std::min(8.0,a);
    for (int i=0; i<128 && evaluate(lower).value>=u; ++i) lower*=2.0;
    for (int i=0; i<128 && evaluate(upper).value<u; ++i) upper=std::min(2.0*upper,a);
    NumericPolicy inversePolicy = policy;
    inversePolicy.distanceAbsoluteTolerance = 1e-11;
    inversePolicy.distanceRelativeTolerance = 1e-10;
    const RootResult root = solveMonotoneSafeguardedNewton(evaluate, u, lower, upper,
                                                          0.5 * (lower + upper), inversePolicy);
    if (root.status != NumericStatus::Ok) {
        throw NumericError(root.status, "failed to sample negative-flux Gaussian");
    }
    const double result = a<1.0 ? -stddev*(a-root.value) : mean+stddev*root.value;
    if (!(result < 0.0)) throw NumericError(NumericStatus::NeedHigherPrecision, "flux sample rounded outside negative support");
    return result;
}

} // namespace Tungsten::ConditionalGpis
