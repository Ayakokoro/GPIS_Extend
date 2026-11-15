#include "AdaptiveGpis.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Tungsten::AdaptiveGpis {
namespace {
constexpr double C1 = 0.793135035671329;
constexpr double Pi = 3.14159265358979323846;
void require(bool valid, const char *message) {
    if (!valid) throw std::runtime_error(message);
}
}

OnlineSampler::OnlineSampler(double length, double tolerance, int maxRank)
    : _length(length), _rankTolerance(tolerance), _maxRank(maxRank)
{
    require(std::isfinite(length) && length > 0 && std::isfinite(tolerance) &&
        tolerance > 0 && tolerance < 1e-3 && maxRank >= 4,
        "adaptive GP: invalid online sampler settings");
}

double OnlineSampler::covariance(const Observation &a, const Observation &b) const
{
    const Eigen::Vector3d r = (a.position - b.position)/_length;
    const double k = std::exp(-0.5*r.squaredNorm());
    if (!a.derivative && !b.derivative) return k;
    if (a.derivative && !b.derivative) return -r.dot(a.direction)*k;
    if (!a.derivative && b.derivative) return r.dot(b.direction)*k;
    return (a.direction.dot(b.direction) - r.dot(a.direction)*r.dot(b.direction))*k;
}

OnlineSampler::Prediction OnlineSampler::predict(const Observation &o) const
{
    require(o.position.allFinite() && (!o.derivative ||
        (o.direction.allFinite() && std::abs(o.direction.norm()-1) < 1e-8)),
        "adaptive GP: invalid observation");
    const int n = int(_basis.size());
    for (int i=0;i<n;++i) {
        const auto &a=_basis[i];
        if (a.derivative==o.derivative && a.position==o.position &&
            (!o.derivative || a.direction==o.direction))
            return {_values[i],0,Eigen::VectorXd()};
    }
    if (_rankRevealing) return predictRankRevealing(o);
    Prediction p{0, 1, Eigen::VectorXd(n)};
    for (int i = 0; i < n; ++i) {
        double b = covariance(_basis[i], o);
        for (int j = 0; j < i; ++j) b = std::fma(-_lower(i,j),p.beta[j],b);
        p.beta[i] = b/_lower(i,i);
        p.mean = std::fma(p.beta[i],_innovations[i],p.mean);
        p.variance = std::fma(-p.beta[i],p.beta[i],p.variance);
    }
    // Resolve cancellation / near-singular pivots with a rank-revealing solve
    // in double. Previously drawn values are preserved, never resampled.
    if (p.variance < std::sqrt(std::numeric_limits<double>::epsilon()) ||
        !std::isfinite(p.variance) || !std::isfinite(p.mean)) {
        _rankRevealing=true;
        return predictRankRevealing(o);
    }
    require(std::isfinite(p.mean) && std::isfinite(p.variance) &&
        p.variance >= -std::max(1e-12, 0.1*_rankTolerance),
        "adaptive GP: Cholesky lost positive semidefiniteness");
    p.variance = std::max(0.0, p.variance);
    return p;
}

OnlineSampler::Prediction OnlineSampler::predictRankRevealing(const Observation &o) const
{
    const int n=int(_basis.size());
    if (_whitener.rows()!=n) {
        Eigen::MatrixXd k(n,n);
        for (int i=0;i<n;++i) for (int j=0;j<=i;++j)
            k(i,j)=k(j,i)=covariance(_basis[i],_basis[j]);
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(k);
        require(solver.info()==Eigen::Success && solver.eigenvalues().minCoeff()>=
            -std::max(1e-12,0.1*_rankTolerance)*n,
            "adaptive GP: double rank decomposition failed");
        const auto &eigenvalues=solver.eigenvalues();
        const int retained=int((eigenvalues.array()>_rankTolerance).count());
        _whitener=solver.eigenvectors().rightCols(retained)*
            eigenvalues.tail(retained).array().sqrt().inverse().matrix().asDiagonal();
        _whitenedValues=_whitener.transpose()*Eigen::Map<const Eigen::VectorXd>(_values.data(),n);
        ++_rankDecompositions;
    }
    Eigen::VectorXd k(n);
    for (int i=0;i<n;++i) k[i]=covariance(_basis[i],o);
    Prediction p{0,1,_whitener.transpose()*k};
    for (Eigen::Index i=0;i<p.beta.size();++i) {
        p.mean=std::fma(p.beta[i],_whitenedValues[i],p.mean);
        p.variance=std::fma(-p.beta[i],p.beta[i],p.variance);
    }
    require(std::isfinite(p.mean) && std::isfinite(p.variance),
        "adaptive GP: nonfinite double conditional moments");
    if (p.variance < -std::max(1e-12,0.1*_rankTolerance))
        return predictJointFactor(o);
    p.variance=std::max(0.0,p.variance);
    return p;
}

OnlineSampler::Prediction OnlineSampler::predictJointFactor(const Observation &o) const
{
    const auto p=predictJointFactor(std::vector<Observation>{o});
    return {p.mean[0],p.covariance(0,0),Eigen::VectorXd()};
}

OnlineSampler::JointPrediction OnlineSampler::predictJointFactor(const std::vector<Observation> &observations) const
{
    // Rare cancellation in 1 - beta^T beta: factor the augmented covariance
    // in double and obtain variance as a squared null-space projection.
    // No diagonal noise is added and no negative variance is silently hidden.
    const int n=int(_basis.size()), m=int(observations.size());
    const auto observation=[&](int i)->const Observation & { return i<n ? _basis[i] : observations[i-n]; };
    Eigen::MatrixXd k(n+m,n+m);
    for (int i=0;i<n+m;++i) for (int j=0;j<=i;++j)
        k(i,j)=k(j,i)=covariance(observation(i),observation(j));
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(k);
    require(eig.info()==Eigen::Success && eig.eigenvalues().minCoeff()>=
        -std::max(1e-12,0.1*_rankTolerance)*(n+m),
        "adaptive GP: joint double covariance is not positive semidefinite");
    const int rank=int((eig.eigenvalues().array()>_rankTolerance).count());
    const Eigen::MatrixXd factor=eig.eigenvectors().rightCols(rank)*
        eig.eigenvalues().tail(rank).array().sqrt().matrix().asDiagonal();
    if (n==0) return {Eigen::VectorXd::Zero(m),factor*factor.transpose()};
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(factor.topRows(n),Eigen::ComputeThinU|Eigen::ComputeFullV);
    require(svd.info()==Eigen::Success,"adaptive GP: joint double rank solve failed");
    const int known=int((svd.singularValues().array()>std::sqrt(_rankTolerance)).count());
    const Eigen::MatrixXd projected=factor.bottomRows(m)*svd.matrixV();
    const Eigen::VectorXd weights=(svd.matrixU().leftCols(known).transpose()*
        Eigen::Map<const Eigen::VectorXd>(_values.data(),n)).array()/svd.singularValues().head(known).array();
    JointPrediction p{projected.leftCols(known)*weights,
        projected.rightCols(rank-known)*projected.rightCols(rank-known).transpose()};
    require(p.mean.allFinite() && p.covariance.allFinite(),"adaptive GP: nonfinite joint double moments");
    ++_rankDecompositions;
    return p;
}

OnlineSampler::JointPrediction OnlineSampler::predictJoint(const std::vector<Observation> &observations) const
{
    require(!observations.empty(),"adaptive GP: empty joint prediction");
    std::vector<Prediction> predictions;
    const bool rankBefore=_rankRevealing;
    for (const auto &o:observations) predictions.push_back(predict(o));
    // All cross terms must use the same whitening coordinates.
    if (rankBefore!=_rankRevealing) {
        predictions.clear();
        for (const auto &o:observations) predictions.push_back(predict(o));
    }
    const auto betaSize=_rankRevealing ? _whitener.cols() : Eigen::Index(_basis.size());
    for (const auto &p:predictions)
        if (p.beta.size()!=betaSize) return predictJointFactor(observations);
    const int m=int(observations.size());
    JointPrediction joint{Eigen::VectorXd(m),Eigen::MatrixXd(m,m)};
    for (int i=0;i<m;++i) {
        joint.mean[i]=predictions[i].mean;
        for (int j=0;j<=i;++j) {
            double value=covariance(observations[i],observations[j]);
            for (Eigen::Index k=0;k<betaSize;++k)
                value=std::fma(-predictions[i].beta[k],predictions[j].beta[k],value);
            joint.covariance(i,j)=joint.covariance(j,i)=value;
        }
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(joint.covariance);
    require(eig.info()==Eigen::Success,"adaptive GP: joint prediction eigensolve failed");
    if (eig.eigenvalues().minCoeff() < -std::max(1e-12,0.1*_rankTolerance)*m)
        return predictJointFactor(observations);
    joint.covariance=eig.eigenvectors()*eig.eigenvalues().cwiseMax(0).asDiagonal()*eig.eigenvectors().transpose();
    return joint;
}

void OnlineSampler::append(const Observation &o, const Prediction &p, double value)
{
    const int n = int(_basis.size());
    require(n < _maxRank, "adaptive GP: max_samples reached; increase it (history is not discarded)");
    if (!_rankRevealing) {
        _lower.conservativeResize(n+1,n+1);
        _lower.row(n).head(n) = p.beta.transpose();
        _lower(n,n) = std::sqrt(p.variance);
        _innovations.conservativeResize(n+1);
        _innovations[n] = (value-p.mean)/std::sqrt(p.variance);
    }
    _basis.push_back(o);
    _values.push_back(value);
}

double OnlineSampler::sample(const Observation &o, double normal)
{
    require(std::isfinite(normal), "adaptive GP: nonfinite normal variate");
    const auto p = predict(o);
    ++_queries;
    if (p.variance <= _rankTolerance) { ++_redundant; return p.mean; }
    const double value=p.mean+std::sqrt(p.variance)*normal;
    append(o,p,value);
    return value;
}

void OnlineSampler::condition(const Observation &o, double value)
{
    require(std::isfinite(value), "adaptive GP: nonfinite constraint");
    const auto p = predict(o);
    if (p.variance <= _rankTolerance) {
        require(std::abs(value-p.mean) <= 8*std::sqrt(_rankTolerance),
            "adaptive GP: inconsistent redundant constraint");
        ++_redundant;
    } else append(o,p,value);
}

GradientPlane::GradientPlane(const Eigen::Matrix3d &frame, const Eigen::Vector3d &mean,
    const Eigen::Matrix3d &covariance, double longitudinalNormal) : _frame(frame)
{
    require(frame.allFinite() && mean.allFinite() && covariance.allFinite() &&
        std::isfinite(longitudinalNormal) &&
        (frame.transpose()*frame-Eigen::Matrix3d::Identity()).norm()<1e-8,
        "adaptive GP: invalid gradient-plane inputs");
    const double variance=std::max(0.0,covariance(2,2));
    _longitudinal=mean[2]+std::sqrt(variance)*longitudinalNormal;
    _mean=mean.head<2>();
    Eigen::Matrix2d conditional=covariance.topLeftCorner<2,2>();
    if (variance>0) {
        const Eigen::Vector2d cross=covariance.topRightCorner<2,1>()/std::sqrt(variance);
        _mean+=cross*longitudinalNormal;
        conditional-=cross*cross.transpose();
    }
    conditional=(conditional+conditional.transpose()).eval()*.5;
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d> eig(conditional);
    const double scale=covariance.cwiseAbs().maxCoeff();
    require(eig.info()==Eigen::Success && eig.eigenvalues().minCoeff()>=-1e-10*std::max(scale,1e-300),
        "adaptive GP: invalid transverse gradient covariance");
    const Eigen::Vector2d values=eig.eigenvalues().cwiseMax(0);
    _root=eig.eigenvectors()*values.cwiseSqrt().asDiagonal();
    _continuous=_longitudinal<0 && values.minCoeff()>std::max(1e-300,scale*1e-12);
    if (_continuous) {
        _inverseRoot=values.cwiseSqrt().cwiseInverse().asDiagonal()*eig.eigenvectors().transpose();
        _logNormalizer=-std::log(2*Pi)-.5*std::log(values[0])-.5*std::log(values[1]);
    }
}

Eigen::Vector3d GradientPlane::sample(const Eigen::Vector2d &normal) const
{
    const Eigen::Vector2d xy=_mean+_root*normal;
    return _frame*Eigen::Vector3d(xy[0],xy[1],_longitudinal);
}

Eigen::Vector3d GradientPlane::gradientForNormal(const Eigen::Vector3d &normal) const
{
    if (!normal.allFinite() || normal.squaredNorm()==0) return Eigen::Vector3d::Zero();
    const Eigen::Vector3d n=normal.normalized();
    const double cosine=n.dot(_frame.col(2));
    if (!(cosine<0) || !(_longitudinal<0)) return Eigen::Vector3d::Zero();
    return n*(_longitudinal/cosine);
}

double GradientPlane::reflectionPdf(const Eigen::Vector3d &normal) const
{
    if (!_continuous) return 0;
    const Eigen::Vector3d gradient=gradientForNormal(normal);
    if (!gradient.allFinite() || gradient.squaredNorm()==0) return 0;
    const double cosine=-normal.normalized().dot(_frame.col(2));
    const Eigen::Vector2d xy=(_frame.transpose()*gradient).head<2>();
    const double mahalanobis=(_inverseRoot*(xy-_mean)).squaredNorm();
    // dA/dOmega_normal = |g|^2/cos(theta),
    // dOmega_normal/dOmega_out = 1/(4*cos(theta)).
    const double logPdf=_logNormalizer-.5*mahalanobis+2*std::log(-_longitudinal)
        -std::log(4.0)-4*std::log(cosine);
    const double pdf=std::exp(logPdf);
    require(std::isfinite(pdf),"adaptive GP: nonfinite reflection PDF");
    return pdf;
}

void Settings::validate() const
{
    require(std::isfinite(probabilityTolerance) && probabilityTolerance > 0 && probabilityTolerance < 0.5 &&
        std::isfinite(minimumStep) && minimumStep > 0 && std::isfinite(rootTolerance) &&
        rootTolerance > 0 && rootTolerance <= minimumStep && std::isfinite(rankTolerance) &&
        rankTolerance > 0 && rankTolerance < 1e-3 && maxSamples >= 4 && maxSteps > 0,
        "adaptive GP: invalid adaptive settings");
}

Bounds::Bounds(double lipschitz, double standardDeviation, double sqrtM, double eta)
    : meanLipschitz(lipschitz), sigma(standardDeviation), sqrtDerivativeBound(sqrtM)
{
    require(std::isfinite(lipschitz) && lipschitz >= 0 && std::isfinite(sigma) && sigma > 0 &&
        std::isfinite(sqrtM) && sqrtM > 0 && std::isfinite(eta) && eta > 0 && eta < 0.5,
        "adaptive GP: invalid range bounds");
    const double risk = eta*0.5;
    c = C1 + std::sqrt(-2*std::log(risk));
    q = std::numeric_limits<double>::infinity();
    if (lipschitz == 0) return; // Only the local bound is needed for constant means.
    const double a = 1 + C1*sqrtM/lipschitz;
    // At the minimum of a*kappa + sqrt(2)*erfc_inv(kappa*risk*sqrt(2/pi)),
    // z=erfc_inv(...) satisfies exp(z^2)=a/risk. Avoid a per-ray iterative solve.
    const double z=std::sqrt(std::log(a/risk));
    const double kappa=std::erfc(z)/(risk*std::sqrt(2/Pi));
    q=kappa*a+std::sqrt(2.0)*z;
}

double Bounds::step(double mean, double value, double minimumStep) const
{
    require(std::isfinite(mean) && std::isfinite(value), "adaptive GP: nonfinite field");
    // Orient the mean by the observed sign; abs(mean) is unsafe when signs disagree.
    const double orientedMean = value >= 0 ? mean : -mean;
    const double distant = meanLipschitz > 0 ? (orientedMean-q*sigma)/meanLipschitz : 0;
    const double local = std::abs(value)/(meanLipschitz+c*sqrtDerivativeBound);
    return std::max({minimumStep, distant, local});
}

Intersection intersect(double begin, double end, double value,
    const std::function<double(double)> &evaluate, const std::function<double(double)> &mean,
    const Bounds &bounds, const Settings &settings)
{
    require(std::isfinite(begin) && std::isfinite(end) && end >= begin && std::isfinite(value),
        "adaptive GP: invalid finite ray interval");
    Intersection result;
    result.t = begin; result.value = value;
    while (result.t < end) {
        require(result.steps < settings.maxSteps, "adaptive GP: max_steps reached");
        const double dt = settings.adaptive ? bounds.step(mean(result.t),result.value,settings.minimumStep)
                                            : settings.minimumStep;
        if (dt == settings.minimumStep) ++result.minimumSteps;
        const double next = std::min(end,result.t+dt);
        require(next > result.t, "adaptive GP: marching failed to advance");
        const double v = evaluate(next);
        require(std::isfinite(v), "adaptive GP: nonfinite sample");
        ++result.steps;
        if ((result.value > 0 && v <= 0) || (result.value < 0 && v >= 0)) {
            double lo = result.t, hi = next, vl = result.value, vh = v;
            for (; hi-lo > settings.rootTolerance; ++result.refinements) {
                require(result.refinements < 80, "adaptive GP: root refinement failed");
                const double mid = lo+(hi-lo)*0.5;
                require(mid > lo && mid < hi, "adaptive GP: root interval below floating-point resolution");
                const double vm = evaluate(mid);
                require(std::isfinite(vm), "adaptive GP: nonfinite root sample");
                if ((vl > 0 && vm > 0) || (vl < 0 && vm < 0)) { lo=mid; vl=vm; }
                else { hi=mid; vh=vm; }
            }
            result.hit = true;
            result.t = lo+(hi-lo)*std::clamp(vl/(vl-vh),0.0,1.0);
            result.value = 0;
            return result;
        }
        result.t = next; result.value = v;
    }
    return result;
}
}
