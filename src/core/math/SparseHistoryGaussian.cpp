#include "math/SparseHistoryGaussian.hpp"
#include "math/BivariateNormalCdfTable.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "math/conditional/mathutility/GaussianMoments1D.h"
#include "math/conditional/mathutility/Quadrature.h"
#include <sobol/sobol.h>
#include <boost/math/special_functions/erf.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>

namespace Tungsten::SparseHistoryGpis {
namespace {
namespace cg = ConditionalGpis;
using cg::NumericError;
using cg::NumericStatus;
constexpr double Pi = 3.1415926535897932384626433832795;
constexpr double Epsilon = std::numeric_limits<double>::epsilon();
struct Estimate { double value = 0, error = 0; };

double quantileFromLog(double logProbability)
{
    if(logProbability < -700) return cg::normalQuantileFromLogCdf(logProbability);
    const double p=std::exp(logProbability);
    return p<=0.5 ? -std::sqrt(2.0)*boost::math::erfc_inv(2*p)
                  : std::sqrt(2.0)*boost::math::erfc_inv(-2*std::expm1(logProbability));
}

void validate(const ProbabilitySettings &s)
{
    if (!(s.absoluteTolerance > 0) || !std::isfinite(s.absoluteTolerance) ||
        !(s.relativeTolerance > 0) || !std::isfinite(s.relativeTolerance) ||
        s.initialSamples < 16 || s.maxSamples < s.initialSamples)
        throw NumericError(NumericStatus::InvalidInput, "invalid sparse-history probability settings");
}

Estimate integrate(const std::function<double(double)> &f, double lo, double hi,
                   const ProbabilitySettings &settings, double scale = 1.0)
{
    if (!(hi > lo)) return {};
    cg::NumericPolicy policy;
    policy.relativeTolerance = std::min(1e-7, settings.relativeTolerance*0.1);
    // These integrals feed probability ratios. An absolute probability
    // tolerance is insufficient when numerator and denominator are both rare.
    policy.absoluteTolerance = std::numeric_limits<double>::min();
    policy.maxQuadratureSubdivisions = 1024;
    const auto result = cg::integrateFinite(f, lo, hi, policy);
    if (result.status != NumericStatus::Ok)
        throw NumericError(result.status, "sparse-history Gaussian quadrature did not converge");
    return {result.value, result.absError};
}

Estimate pairProbability(double a, double b, double rho, const ProbabilitySettings &settings)
{
    rho = std::clamp(rho, -1.0, 1.0);
    if (rho == 1) return {cg::normalCdf(std::min(a, b)), 0};
    if (rho == -1) {
        if (!(a+b > 0)) return {};
        const double lo = -a, hi = b;
        return {lo > 0 ? cg::normalCdf(-lo)-cg::normalCdf(-hi)
                       : cg::normalCdf(hi)-cg::normalCdf(lo), 0};
    }
    if (rho == 0) return {cg::normalCdf(a)*cg::normalCdf(b), 0};
    if (a == 0 && b == 0) return {0.25+std::asin(rho)/(2*Pi), 8*Epsilon};
    if(settings.bivariateTable) {
        if(settings.statistics) settings.statistics->tableAttempts.fetch_add(1,std::memory_order_relaxed);
        const auto estimate=settings.bivariateTable->lookup(a,b,rho,0.25*settings.relativeTolerance);
        // Probabilities enter ratios and cancelling crossing moments. Reserve
        // most of the relative budget for that propagation, even when the
        // requested absolute tolerance would permit a much larger tail error.
        if(estimate && estimate->value>0 &&
           estimate->error<=0.25*settings.relativeTolerance*estimate->value) {
            if(settings.statistics) settings.statistics->tableAccepted.fetch_add(1,std::memory_order_relaxed);
            return {estimate->value,estimate->error};
        }
    }
    if (rho > 0) {
        // Plackett's correlation integral has only positive terms here.
        const auto f = [=](double theta) {
            const double c = std::cos(theta), s = std::sin(theta);
            const double d = (a-b*s)/c;
            return std::exp(-0.5*(d*d+b*b))/(2*Pi);
        };
        auto result = integrate(f, 0, std::asin(rho), settings,
                                std::min(cg::normalCdf(a), cg::normalCdf(b)));
        result.value += cg::normalCdf(a)*cg::normalCdf(b);
        return result;
    }
    // Independent average and half-difference. Unlike subtracting a
    // negative Plackett integral, this remains positive in a rare joint tail.
    const double sa = std::sqrt(0.5*(1+rho)), sd = std::sqrt(0.5*(1-rho));
    const double limit = std::min(38.0, std::max(10.0, 2+std::max(std::abs(a),std::abs(b))/sd));
    std::vector<double> cuts{-limit,limit};
    for (double center : {(b-a)/(2*sd), -a/sd, b/sd})
        for(double offset:{-12.0,-4.0,-1.0,0.0,1.0,4.0,12.0}) {
            const double x=center+offset*sa/sd;
            if(x>-limit && x<limit) cuts.push_back(x);
        }
    std::sort(cuts.begin(),cuts.end());
    cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
    Estimate result;
    const auto f = [=](double z) {
        return cg::normalPdf(z)*cg::normalCdf(std::min(a+sd*z,b-sd*z)/sa);
    };
    for (std::size_t i=1;i<cuts.size();++i) {
        const auto part=integrate(f,cuts[i-1],cuts[i],settings,
                                 std::min(cg::normalCdf(a),cg::normalCdf(b)));
        result.value+=part.value; result.error+=part.error;
    }
    result.error+=2*cg::normalCdf(-limit);
    return result;
}

std::optional<Estimate> positiveTriple(const Eigen::VectorXd &m,const Eigen::MatrixXd &c,
                                      const ProbabilitySettings &settings)
{
    // Along R(s)=I+s(R-I), Plackett's derivative is a bivariate boundary
    // density times the conditional third CDF. Nonnegative correlations make
    // every term nonnegative, so rare probabilities never require subtraction.
    for(int i=0;i<3;++i) for(int j=0;j<i;++j)
        if(c(i,j)<0 || c(i,j)>1-1e-10) return std::nullopt;
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(c,Eigen::EigenvaluesOnly);
    if(eigen.info()!=Eigen::Success || eigen.eigenvalues().minCoeff()<1e-10)
        return std::nullopt;
    const double minimumEigenvalue=eigen.eigenvalues().minCoeff();
    Estimate result{cg::normalCdf(m[0])*cg::normalCdf(m[1])*cg::normalCdf(m[2]),0};
    result.error=8*Epsilon*result.value;
    bool usable=true;
    try {
        for(int i=0;i<3;++i) for(int j=0;j<i;++j) {
            const double correlation=c(i,j);
            if(correlation==0) continue;
            const int k=3-i-j;
            // theta=asin(s*r_ij) cancels the pair density's singular
            // 1/sqrt(1-rho^2) factor near the perfectly correlated endpoint.
            const auto integrand=[&](double theta) {
                const double rho=std::sin(theta),cosine=std::cos(theta);
                const double pairVariance=cosine*cosine,s=rho/correlation;
                const double ci=s*c(k,i),cj=s*c(k,j);
                const double residual=cj-rho*ci;
                const double variance=std::fma(-ci,ci,1.0)-residual*residual/pairVariance;
                if(!(variance>0)) { usable=false;return 0.0; }
                const double conditionalMean=m[k]-ci*m[i]-residual*(m[j]-rho*m[i])/pairVariance;
                const double z=(m[i]-rho*m[j])/cosine;
                return std::exp(-0.5*(z*z+m[j]*m[j]))/(2*Pi)*
                    cg::normalCdf(conditionalMean/std::sqrt(variance));
            };
            const auto contribution=integrate(integrand,0,std::asin(correlation),settings);
            result.value+=contribution.value;result.error+=contribution.error;
        }
    } catch(const NumericError &error) {
        if(error.status()!=NumericStatus::IntegrationNotConverged) throw;
        return std::nullopt;
    }
    if(!usable) return std::nullopt;
    // Include base-product rounding and conditioning sensitivity, not just
    // the quadrature error of the three derivative terms. The general path
    // remains available whenever this estimate misses the requested accuracy.
    result.error+=64*Epsilon*result.value/minimumEigenvalue;
    return result;
}

std::uint32_t hash(std::uint32_t x)
{
    x ^= x>>16; x*=0x7feb352dU; x^=x>>15; x*=0x846ca68bU; return x^(x>>16);
}

Estimate orthant(const Eigen::VectorXd &, const Eigen::MatrixXd &,
                 const ProbabilitySettings &, bool deterministic = false,
                 bool relativeOnly = false);

// Deterministic fallback also handles the almost singular cases in which
// digital-shift variation ceases to give a useful convergence diagnostic.
Estimate recursiveOrthant(const Eigen::VectorXd &m, const Eigen::MatrixXd &c,
                          const ProbabilitySettings &settings,bool relativeOnly)
{
    const int n=int(m.size());
    int pivot=0;
    for(int i=1;i<n;++i) if(m[i]<m[pivot]) pivot=i;
    const double probability=cg::normalCdf(m[pivot]);
    if(probability==0) return {};
    Eigen::VectorXd remaining(n-1),cross(n-1);
    // The parent has unit marginal variances. Factor it before conditioning:
    // direct subtraction can leave O(epsilon) negative eigenvalues in a
    // conditional covariance whose entire scale is O(epsilon) itself.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> decomposition(c);
    if(decomposition.info()!=Eigen::Success || decomposition.eigenvalues().minCoeff() < -256*Epsilon)
        throw NumericError(NumericStatus::InvalidCovariance,"invalid recursive orthant covariance");
    const Eigen::MatrixXd features=decomposition.eigenvectors()*
        decomposition.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
    Eigen::MatrixXd residual(n-1,n);
    const double pivotVariance=features.row(pivot).squaredNorm();
    for(int i=0,ii=0;i<n;++i) if(i!=pivot) {
        remaining[ii]=m[i]; cross[ii]=c(i,pivot);
        residual.row(ii)=features.row(i)-(features.row(i).dot(features.row(pivot))/pivotVariance)*features.row(pivot);
        ++ii;
    }
    const Eigen::MatrixXd conditional=residual*residual.transpose();
    const auto f=[&](double u) {
        const double z=-quantileFromLog(std::log(probability)+std::log(u));
        return orthant(remaining+cross*z,conditional,settings,true,relativeOnly).value;
    };
    const auto result=integrate(f,0,1,settings);
    return {probability*result.value,probability*result.error};
}

Estimate orthant(const Eigen::VectorXd &inputMean,const Eigen::MatrixXd &inputCovariance,
                 const ProbabilitySettings &settings,bool deterministic,bool relativeOnly)
{
    const int inputN=int(inputMean.size());
    if(inputN>4 || inputCovariance.rows()!=inputN || inputCovariance.cols()!=inputN ||
       !inputMean.allFinite() || !inputCovariance.allFinite())
        throw NumericError(NumericStatus::InvalidInput,"invalid sparse-history orthant inputs");
    if(inputN==0) return {1,0};
    if(relativeOnly) {
        // Cheap/pruned evaluation is safe for a ratio only when its complete
        // error estimate (including the omitted-sign union bound) is small
        // relative to THIS joint probability. Otherwise retain every sign.
        const auto cheap=orthant(inputMean,inputCovariance,settings,deterministic,false);
        if(cheap.value>0 && cheap.error<=settings.relativeTolerance*cheap.value)
            return cheap;
    }
    const double scale=inputCovariance.cwiseAbs().maxCoeff();
    const double tolerance=128*Epsilon*std::max(scale,std::numeric_limits<double>::min());
    if((inputCovariance-inputCovariance.transpose()).cwiseAbs().maxCoeff()>tolerance)
        throw NumericError(NumericStatus::InvalidCovariance,"asymmetric sparse-history covariance");
    // Test the full covariance BEFORE dropping deterministic or certain signs.
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(inputCovariance,Eigen::EigenvaluesOnly);
    if(eig.info()!=Eigen::Success || eig.eigenvalues().minCoeff() < -tolerance)
        throw NumericError(NumericStatus::InvalidCovariance,"indefinite sparse-history covariance");
    std::vector<int> indices;
    double omitted=0;
    for(int i=0;i<inputN;++i) {
        const double variance=inputCovariance(i,i);
        if(variance<=0) { if(!(inputMean[i]>0)) return {}; continue; }
        const double z=inputMean[i]/std::sqrt(variance);
        const double tail=cg::normalCdf(-z);
        if(tail<=settings.absoluteTolerance/(8*inputN) && (!relativeOnly || tail==0)) omitted+=tail;
        else indices.push_back(i);
    }
    const int n=int(indices.size());
    if(n==0) return {1,omitted};
    Eigen::VectorXd m(n),sd(n);
    Eigen::MatrixXd c(n,n);
    for(int i=0;i<n;++i) { sd[i]=std::sqrt(inputCovariance(indices[i],indices[i])); m[i]=inputMean[indices[i]]/sd[i]; }
    for(int i=0;i<n;++i) for(int j=0;j<n;++j)
        c(i,j)=inputCovariance(indices[i],indices[j])/(sd[i]*sd[j]);
    if(n==1) return {cg::normalCdf(m[0]),omitted};
    if(n==2) { auto r=pairProbability(m[0],m[1],c(0,1),settings);r.error+=omitted;return r; }
    if(n==3 && m.isZero(0)) {
        double value=0.125;
        for(int i=0;i<3;++i) for(int j=0;j<i;++j) value+=std::asin(std::clamp(c(i,j),-1.0,1.0))/(4*Pi);
        if(value < -128*Epsilon) throw NumericError(NumericStatus::InvalidCovariance,"negative orthant probability");
        return {std::max(0.0,value),omitted+128*Epsilon};
    }
    if(n==3) {
        const auto fast=positiveTriple(m,c,settings);
        if(fast) {
            const double error=fast->error+omitted;
            const double allowed=relativeOnly ? settings.relativeTolerance*fast->value
                : std::max(settings.absoluteTolerance,settings.relativeTolerance*fast->value);
            if(error<=allowed) return {fast->value,error};
        }
    }
    if(deterministic) { auto r=recursiveOrthant(m,c,settings,relativeOnly);r.error+=omitted;return r; }
    std::vector<int> order(n); std::iota(order.begin(),order.end(),0);
    Eigen::MatrixXd factor=Eigen::MatrixXd::Zero(n,n);
    Eigen::VectorXd variance=Eigen::VectorXd::Ones(n),proxy=Eigen::VectorXd::Zero(n);
    int rank=n;
    for(int k=0;k<n;++k) {
        int pivot=k; double best=std::numeric_limits<double>::infinity();
        for(int i=k;i<n;++i) if(variance[i]>128*Epsilon) {
            const double z=(m[order[i]]+factor.row(i).head(k).dot(proxy.head(k)))/std::sqrt(variance[i]);
            if(z<best) { best=z;pivot=i; }
        }
        std::swap(order[k],order[pivot]); std::swap(variance[k],variance[pivot]);
        for(int j=0;j<k;++j) std::swap(factor(k,j),factor(pivot,j));
        if(variance[k] < -256*Epsilon)
            throw NumericError(NumericStatus::InvalidCovariance,"sparse-history Cholesky failed");
        if(variance[k]<=128*Epsilon) { rank=k;break; }
        factor(k,k)=std::sqrt(variance[k]);
        const double z=(m[order[k]]+factor.row(k).head(k).dot(proxy.head(k)))/factor(k,k);
        proxy[k]=cg::normalPdfOverCdf(z);
        for(int i=k+1;i<n;++i) {
            factor(i,k)=(c(order[i],order[k])-factor.row(i).head(k).dot(factor.row(k).head(k)))/factor(k,k);
            variance[i]-=factor(i,k)*factor(i,k);
        }
    }
    if(rank<n) { auto r=recursiveOrthant(m,c,settings,relativeOnly);r.error+=omitted;return r; }
    constexpr int replicates=8;
    std::array<double,replicates> sums{};
    int computed=0;
    for(int samples=settings.initialSamples;;samples=std::min(settings.maxSamples,samples*2)) {
        const auto previous=sums;
        for(int replicate=0;replicate<replicates;++replicate) {
            for(int point=computed;point<samples;++point) {
                Eigen::Vector4d z=Eigen::Vector4d::Zero(); double weight=1;
                for(int i=0;i<n;++i) {
                    double mean=m[order[i]];
                    for(int j=0;j<i;++j) mean+=factor(i,j)*z[j];
                    const double p=cg::normalCdf(mean/factor(i,i));
                    weight*=p;
                    if(weight==0) break;
                    if(i+1<n) {
                        const double u=(double(sobol::sample(std::uint64_t(point),unsigned(i),
                            hash(0x9e3779b9U*(replicate+1)^hash(0x85ebca6bU*(i+1)))))+0.5)/4294967296.0;
                        z[i]=-quantileFromLog(std::log(p)+std::log(u));
                    }
                }
                sums[replicate]+=weight;
            }
        }
        double mean=std::accumulate(sums.begin(),sums.end(),0.0)/(replicates*samples);
        double variation=0;
        for(double sum:sums) variation+=std::pow(sum/samples-mean,2);
        double error=3*std::sqrt(variation/(replicates*(replicates-1.0)));
        if(computed) error=std::max(error,std::abs(mean-std::accumulate(previous.begin(),previous.end(),0.0)/(replicates*computed)));
        const double allowed=relativeOnly ? settings.relativeTolerance*mean
                                         : std::max(settings.absoluteTolerance,settings.relativeTolerance*mean);
        if(mean>0 && error+omitted<=allowed)
            return {mean,error+omitted};
        if(samples==settings.maxSamples) { auto r=recursiveOrthant(m,c,settings,relativeOnly);r.error+=omitted;return r; }
        computed=samples;
    }
}

cg::MeanJet meanJet(const cg::GPSSField &field,const Eigen::Vector3d &point)
{
    if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get()))
        if(point==sphere->center()) return {-sphere->radius(),Eigen::Vector3d::Zero()};
    return field.mean->evaluate(point);
}

struct StartStatistics {
    Eigen::Vector3d direction;
    std::unique_ptr<cg::MeanRayCursor> ray;
    cg::MeanJet birth;
    double sigma, inverseLength, deltaSlope;
    bool surface;
    StartStatistics(const cg::GPSSField &field,const Start &start)
        : direction(cg::normalizedOrThrow(start.direction)),ray(field.mean->createRayCursor(start.origin,direction)),
          birth(ray ? ray->evaluate(0):meanJet(field,start.origin)),
          sigma(field.kernel.sigma()),inverseLength(std::sqrt(direction.dot(field.kernel.precision()*direction))),
          deltaSlope(direction.dot(start.gradient-birth.gradient)),surface(start.kind==Start::SurfaceValueGradient)
    {
        if(!start.origin.allFinite() || !start.gradient.allFinite() || !(sigma>0) || !(inverseLength>0) ||
           !field.kernel.supportsAnalyticConditionalTransport())
            throw NumericError(NumericStatus::InvalidInput,"invalid sparse-history start or covariance");
        if(start.kind!=Start::ExteriorPositiveOnly && start.kind!=Start::SurfaceValueGradient)
            throw NumericError(NumericStatus::InvalidInput,"invalid sparse-history start kind");
        if(surface && !(direction.dot(start.gradient)>0))
            throw NumericError(NumericStatus::NonPositiveSurfaceBirth,"sparse-history surface start must point to air");
    }
    double mean(const cg::GPSSField &field,const Start &start,double t) const {
        if(!surface) return (ray ? ray->sample(t).value:meanJet(field,start.origin+t*direction).value)/sigma;
        const double q=t*inverseLength;
        const double difference=ray ? ray->valueDifference(t):field.mean->valueDifference(start.origin,t*direction);
        return (difference+t*deltaSlope+
                std::expm1(-0.5*q*q)*(-birth.value+t*deltaSlope))/sigma;
    }
};

double valueCovariance(double q,double r,bool surface)
{
    const double prior=std::exp(-0.5*(q-r)*(q-r));
    if(!surface) return prior;
    const double x=q*r;
    if(x<1) {
        double term=0.5*x*x,sum=term;
        for(int n=3;n<=30;++n) { term*=x/n;sum+=term; }
        return std::exp(-0.5*(q*q+r*r))*sum;
    }
    return prior-(1+x)*std::exp(-0.5*(q*q+r*r));
}

Eigen::VectorXd feature(double q,bool derivative,bool surface,int count)
{
    Eigen::VectorXd f=Eigen::VectorXd::Zero(count);
    double previous=0,current=std::exp(-0.5*q*q);
    for(int n=0;n<count;++n) {
        if(n>0) { previous=current;current*=q/std::sqrt(double(n)); }
        if(!surface || n>=2) f[n]=derivative ? q*current-(n ? std::sqrt(double(n))*previous:0) : current;
    }
    return f;
}

} // namespace

double positiveOrthant(const Eigen::VectorXd &mean,const Eigen::MatrixXd &covariance,
                       const ProbabilitySettings &settings)
{
    validate(settings);return orthant(mean,covariance,settings).value;
}

double standardizedMean(const cg::GPSSField &field,const Start &start,double distance)
{
    if(!(distance>=0) || !std::isfinite(distance))
        throw NumericError(NumericStatus::InvalidInput,"invalid sparse-history position");
    const StartStatistics s(field,start);
    if(distance==0 && s.surface) return std::numeric_limits<double>::infinity();
    const double q=distance*s.inverseLength;
    return s.mean(field,start,distance)/std::sqrt(valueCovariance(q,q,s.surface));
}

struct LocalLaw::Impl {
    ProbabilitySettings settings;
    Eigen::Vector3d rawGradient;
    Eigen::VectorXd signsMean,crossingMean;
    Eigen::MatrixXd signsCovariance,crossingCovariance,crossingFeatures;
    double densityAtZero=0,normalizedCurrentMean=0,speedScale=1,hazardScale=1;
    bool atSurfaceOrigin=false;
    mutable double denominator=-1,moment=-1,speedNormalization=-1;

    double historyProbability(double speed) const {
        const int n=int(crossingMean.size())-1;
        if(n==0) return 1;
        const double variance=crossingCovariance(0,0);
        const Eigen::VectorXd cross=crossingCovariance.block(1,0,n,1);
        if(variance==0) return orthant(crossingMean.tail(n),crossingCovariance.bottomRightCorner(n,n),settings,false,true).value;
        const Eigen::VectorXd m=crossingMean.tail(n)+cross*((speed-crossingMean[0])/variance);
        Eigen::MatrixXd c;
        if(crossingFeatures.cols()) {
            Eigen::MatrixXd residual=crossingFeatures.bottomRows(n);
            for(int i=0;i<n;++i) residual.row(i)-=(cross[i]/variance)*crossingFeatures.row(0);
            c=residual*residual.transpose();
        } else c=crossingCovariance.bottomRightCorner(n,n)-cross*cross.transpose()/variance;
        return orthant(m,c,settings,false,true).value;
    }

    double integratedMoment(double upper) const {
        const double mu=crossingMean[0],variance=crossingCovariance(0,0);
        if(variance==0) return mu>0 && upper>=mu ? mu*historyProbability(mu) : 0;
        const double sd=std::sqrt(variance),a=mu/sd;
        // Integrate in speed/sd, scaled by its Gaussian flux mode. This
        // retains the negative-mean tail without subtracting two CDFs.
        const double root=std::hypot(a,2.0);
        const double mode=a>=0 ? 0.5*(a+root):2/(root-a);
        double maxU=std::isfinite(upper) ? upper/sd : std::max(mode+12.0,a+12.0);
        if(!(maxU>0)) return 0;
        const double logScale=std::log(mode)+cg::normalLogPdf(mode-a);
        if(logScale < std::log(std::numeric_limits<double>::min())) return 0;
        const auto f=[&](double u) {
            if(!(u>0)) return 0.0;
            return std::exp(std::log(u)+cg::normalLogPdf(u-a)-logScale)*historyProbability(sd*u);
        };
        std::vector<double> transitions;
        for(int i=1;i<crossingMean.size();++i) {
            const double slope=crossingCovariance(i,0)/sd;
            if(slope==0) continue;
            const double center=a-crossingMean[i]/slope;
            const double conditionalVariance=std::max(0.0,crossingCovariance(i,i)-slope*slope);
            const double width=std::sqrt(conditionalVariance)/std::abs(slope);
            if(center>0 && std::isfinite(center)) {
                if(!std::isfinite(upper) && slope>0) maxU=std::max(maxU,center+12*width+12);
                for(double offset:{-12.0,-4.0,-1.0,0.0,1.0,4.0,12.0}) {
                    transitions.push_back(center+offset*width);
                    transitions.push_back(center+offset/std::max(1.0,std::abs(center-a)));
                }
            }
        }
        std::vector<double> cuts{0,maxU};
        for(double x:{mode,std::max(0.0,a),std::max(0.0,a-8),std::max(0.0,a+8)})
            if(x>0 && x<maxU) cuts.push_back(x);
        for(double x:transitions) if(x>0 && x<maxU) cuts.push_back(x);
        std::sort(cuts.begin(),cuts.end());
        cuts.erase(std::unique(cuts.begin(),cuts.end()),cuts.end());
        double sum=0;
        for(std::size_t i=1;i<cuts.size();++i) sum+=integrate(f,cuts[i-1],cuts[i],settings).value;
        if(!std::isfinite(upper)) {
            // Acceptance is at most one, so the ordinary Gaussian flux tail
            // bounds the omitted selected flux even for an unusual history.
            for(int iteration=0;iteration<32;++iteration) {
                const double tail1=cg::positiveRawMoment(1,a-maxU,1).logValue;
                const double tail2=std::log(maxU)+cg::normalLogCdf(a-maxU);
                const double largest=std::max(tail1,tail2);
                const double logTail=std::isfinite(largest) ? largest+std::log1p(std::exp(std::min(tail1,tail2)-largest)) : largest;
                if(sum>0 && logTail-logScale<=std::log(settings.relativeTolerance*0.1*sum)) break;
                if(logTail < std::log(std::numeric_limits<double>::min()) && sum==0) break;
                if(iteration==31) throw NumericError(NumericStatus::IntegrationNotConverged,"sparse-history speed tail did not converge");
                const double next=maxU+std::max(4.0,0.25*maxU);
                sum+=integrate(f,maxU,next,settings).value;maxU=next;
            }
        }
        return sd*std::exp(logScale)*sum;
    }

    double crossingMomentWith(const ProbabilitySettings &probability) const {
        const int n=int(crossingMean.size());
        if(n==1) return cg::positiveRawMoment(1,crossingMean[0],std::sqrt(crossingCovariance(0,0))).value;
        // Gaussian integration by parts: exact first orthant moment. Each
        // term only needs an orthant probability, with one fewer variable
        // on its boundary. Fall back to a positive integrand on cancellation.
        const auto p=orthant(crossingMean,crossingCovariance,probability,false,true);
        long double value=static_cast<long double>(crossingMean[0])*p.value;
        double magnitude=std::abs(double(value)),error=std::abs(crossingMean[0])*p.error;
        for(int j=0;j<n;++j) {
            const double var=crossingCovariance(j,j);
            if(!(var>0)) continue;
            const double coefficient=crossingCovariance(0,j)/std::sqrt(var)*
                cg::normalPdf(crossingMean[j]/std::sqrt(var));
            if(coefficient==0) continue;
            Eigen::VectorXd m(n-1);Eigen::MatrixXd c(n-1,n-1);
            Eigen::MatrixXd residual;
            if(crossingFeatures.cols()) residual.resize(n-1,crossingFeatures.cols());
            for(int i=0,ii=0;i<n;++i) if(i!=j) {
                m[ii]=crossingMean[i]-crossingCovariance(i,j)*crossingMean[j]/var;
                if(crossingFeatures.cols()) residual.row(ii)=crossingFeatures.row(i)-
                    (crossingCovariance(i,j)/var)*crossingFeatures.row(j);
                for(int k=0,kk=0;k<n;++k) if(k!=j) {
                    c(ii,kk)=crossingCovariance(i,k)-crossingCovariance(i,j)*crossingCovariance(j,k)/var; ++kk;
                }
                ++ii;
            }
            if(crossingFeatures.cols()) c=residual*residual.transpose();
            const auto boundary=orthant(m,c,probability,false,true);
            value+=static_cast<long double>(coefficient)*boundary.value;
            magnitude+=std::abs(coefficient*boundary.value);error+=std::abs(coefficient)*boundary.error;
        }
        const double result=double(value);
        if(result>0 && result>1e-7*magnitude && error<=probability.relativeTolerance*result)
            return result;
        if(probability.bivariateTable) {
            // Cancellation can amplify an otherwise acceptable CDF lookup
            // error. Retry the inexpensive IBP identity numerically before
            // resorting to an outer speed integral with many CDF evaluations.
            auto numerical=probability;
            numerical.bivariateTable.reset();
            return crossingMomentWith(numerical);
        }
        return integratedMoment(std::numeric_limits<double>::infinity());
    }

    double crossingMoment() const {
        if(moment<0) moment=crossingMomentWith(settings);
        return moment;
    }
};

LocalLaw::LocalLaw(const cg::GPSSField &field,const Start &start,double distance,
                   const std::vector<double> &history,const ProbabilitySettings &settings)
    : _impl(std::make_shared<Impl>())
{
    validate(settings);
    if(!(distance>=0) || !std::isfinite(distance) || history.size()>2)
        throw NumericError(NumericStatus::InvalidInput,"invalid sparse-history query distance or history count");
    for(std::size_t i=0;i<history.size();++i) {
        if(!(history[i]>0 && history[i]<distance) || !std::isfinite(history[i]))
            throw NumericError(NumericStatus::InvalidInput,"history must precede the sparse-history query");
        for(std::size_t j=0;j<i;++j) if(history[i]==history[j])
            throw NumericError(NumericStatus::InvalidInput,"duplicate sparse-history position");
    }
    const StartStatistics s(field,start);
    auto &p=*_impl; p.settings=settings;p.speedScale=s.sigma*s.inverseLength;p.hazardScale=s.inverseLength;
    p.rawGradient=(s.ray ? s.ray->evaluate(distance):meanJet(field,start.origin+distance*s.direction)).gradient;
    if(distance==0 && s.surface) { p.atSurfaceOrigin=true;return; }
    std::vector<double> positions{distance};
    if(!s.surface && distance>0) positions.push_back(0);
    positions.insert(positions.end(),history.begin(),history.end());
    const int n=int(positions.size());
    // Variable order: current value, negative directional derivative, signs.
    Eigen::VectorXd means(n+1);Eigen::MatrixXd covariance(n+1,n+1);
    means[0]=s.mean(field,start,distance);
    const double q=distance*s.inverseLength,r=std::exp(-0.5*q*q);
    double derivative=p.rawGradient.dot(s.direction);
    if(s.surface) derivative+=r*((1-q*q)*s.deltaSlope+s.inverseLength*s.inverseLength*distance*s.birth.value);
    means[1]=-derivative/p.speedScale;
    for(int i=1;i<n;++i) means[i+1]=s.mean(field,start,positions[i]);
    Eigen::MatrixXd conditional;
    if(q<=8) {
        // The SE feature series makes conditioning a geometric projection.
        // It avoids subtracting nearly equal covariances near a surface birth.
        const int count=std::max(24,int(std::ceil(q*q+14*q+50)));
        Eigen::MatrixXd features(n+1,count);
        features.row(0)=feature(q,false,s.surface,count);
        features.row(1)=feature(q,true,s.surface,count);
        for(int i=1;i<n;++i) features.row(i+1)=feature(positions[i]*s.inverseLength,false,s.surface,count);
        covariance=features*features.transpose();
        Eigen::MatrixXd residual=features.bottomRows(n);
        for(int i=0;i<n;++i) residual.row(i)-=(covariance(i+1,0)/covariance(0,0))*features.row(0);
        conditional=residual*residual.transpose();
        p.crossingFeatures=std::move(residual);
    } else {
        for(int i=0;i<n;++i) for(int j=0;j<n;++j)
            covariance(i ? i+1:0,j ? j+1:0)=valueCovariance(positions[i]*s.inverseLength,positions[j]*s.inverseLength,s.surface);
        covariance(1,1)=s.surface ? 1-(1-q*q+q*q*q*q)*std::exp(-q*q):1;
        for(int i=0;i<n;++i) {
            const double u=positions[i]*s.inverseLength;
            double cross=(q-u)*std::exp(-0.5*(q-u)*(q-u));
            if(s.surface) cross+=(u-q*(1+q*u))*std::exp(-0.5*(q*q+u*u));
            covariance(1,i ? i+1:0)=covariance(i ? i+1:0,1)=cross;
        }
        conditional=covariance.bottomRightCorner(n,n)-covariance.block(1,0,n,1)*covariance.block(0,1,1,n)/covariance(0,0);
    }
    if(!(covariance(0,0)>0)) throw NumericError(NumericStatus::NeedHigherPrecision,"sparse-history field variance underflow");
    p.crossingMean=means.tail(n)-covariance.block(1,0,n,1)*(means[0]/covariance(0,0));
    p.crossingCovariance=0.5*(conditional+conditional.transpose());
    p.signsMean.resize(n);p.signsCovariance.resize(n,n);
    for(int i=0;i<n;++i) {
        const int ii=i ? i+1:0;p.signsMean[i]=means[ii];
        for(int j=0;j<n;++j) p.signsCovariance(i,j)=covariance(ii,j ? j+1:0);
    }
    p.normalizedCurrentMean=means[0]/std::sqrt(covariance(0,0));
    p.densityAtZero=cg::normalPdf(p.normalizedCurrentMean)/std::sqrt(covariance(0,0));
}

double LocalLaw::hazard() const
{
    auto &p=*_impl;
    if(p.atSurfaceOrigin) return 0;
    if(p.densityAtZero==0) {
        if(p.normalizedCurrentMean<0)
            throw NumericError(NumericStatus::UnresolvedRareEvent,"sparse-history negative field tail requires log-domain ratios");
        return 0;
    }
    if(p.denominator<0) p.denominator=orthant(p.signsMean,p.signsCovariance,p.settings,false,true).value;
    if(!(p.denominator>0)) throw NumericError(NumericStatus::UnresolvedRareEvent,"sparse-history survival probability underflow");
    const double result=p.densityAtZero*p.crossingMoment()*p.hazardScale/p.denominator;
    if(!(result>=0) || !std::isfinite(result)) throw NumericError(NumericStatus::NeedHigherPrecision,"nonfinite sparse-history hazard");
    return result;
}

double LocalLaw::speedCdf(double speed) const
{
    const auto &p=*_impl;
    if(!(speed>0)) return 0;
    if(p.atSurfaceOrigin) throw NumericError(NumericStatus::InvalidInput,"surface origin has no collision speed law");
    // A table approximates the IBP moment used by the hazard. Normalize speed
    // inversion with the same acceptance function as rejection sampling, so
    // a small table error cannot leave its finite-speed CDF below one.
    if(p.speedNormalization<0) p.speedNormalization=p.settings.bivariateTable
        ? p.integratedMoment(std::numeric_limits<double>::infinity()) : p.crossingMoment();
    const double total=p.speedNormalization;
    if(!(total>0)) throw NumericError(NumericStatus::UnresolvedRareEvent,"sparse-history crossing has no representable mass");
    if(!std::isfinite(speed)) return 1;
    const double result=p.integratedMoment(speed/p.speedScale)/total;
    if(result < -p.settings.relativeTolerance || result>1+5*p.settings.relativeTolerance)
        throw NumericError(NumericStatus::IntegrationNotConverged,"sparse-history speed CDF disagrees with crossing moment");
    return std::clamp(result,0.0,1.0);
}

double LocalLaw::sampleSpeed(double openUniform) const
{
    if(!(openUniform>0 && openUniform<1)) throw NumericError(NumericStatus::InvalidInput,"speed sampling requires an open uniform");
    const auto &p=*_impl;
    if(p.atSurfaceOrigin) throw NumericError(NumericStatus::InvalidInput,"surface origin has no collision speed law");
    const double conditionalSd=p.speedScale*std::sqrt(p.crossingCovariance(0,0));
    if(conditionalSd==0) {
        const double speed=p.speedScale*p.crossingMean[0];
        if(!(speed>0) || !(p.crossingMoment()>0))
            throw NumericError(NumericStatus::UnresolvedRareEvent,"sparse-history deterministic crossing has no positive mass");
        return speed;
    }
    double lo=0,hi=p.speedScale*std::max(0.0,p.crossingMean[0])+8*conditionalSd;
    if(!(hi>0)) throw NumericError(NumericStatus::UnresolvedRareEvent,"sparse-history crossing has no positive speed");
    for(int i=0;speedCdf(hi)<openUniform;++i) {
        if(i==32) throw NumericError(NumericStatus::RootNotBracketed,"sparse-history speed inversion could not bracket root");
        hi*=2;
    }
    for(int i=0;i<64;++i) {
        const double mid=0.5*(lo+hi);
        if(speedCdf(mid)<openUniform) lo=mid;else hi=mid;
        // A conditioned near-birth speed can be many orders of magnitude
        // narrower than sigma/ell. Stop on that conditional scale, allowing
        // only unavoidable floating-point resolution at a nonzero mean.
        const double tolerance=std::max(1e-9*conditionalSd,
            32*Epsilon*std::max(std::abs(lo),std::abs(hi)));
        if(hi-lo<=tolerance) return 0.5*(lo+hi);
    }
    throw NumericError(NumericStatus::SamplingNotConverged,"sparse-history speed inversion did not converge");
}

double LocalLaw::sampleSpeed(cg::Random &random) const
{
    const auto &p=*_impl;
    if(p.atSurfaceOrigin)
        throw NumericError(NumericStatus::InvalidInput,"surface origin has no collision speed law");
    const double sd=std::sqrt(p.crossingCovariance(0,0));
    if(sd==0) return sampleSpeed(random.openUniform01());
    for(int attempt=0;attempt<64;++attempt) {
        const double candidate=-cg::sampleNegativeFluxNormal(-p.crossingMean[0],sd,random);
        if(random.openUniform01()<p.historyProbability(candidate))
            return p.speedScale*candidate;
    }
    // Independent inversion after unsuccessful proposals preserves the
    // selected flux law even when the historical event is very rare.
    return sampleSpeed(random.openUniform01());
}

Eigen::Vector3d LocalLaw::meanGradient() const { return _impl->rawGradient; }

} // namespace Tungsten::SparseHistoryGpis
