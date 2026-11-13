#include "math/conditional/transport/SinglePointHistoryTeacher.h"
#include "math/SparseHistoryGaussian.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include <iostream>

namespace cg = Tungsten::ConditionalGpis;
namespace sh = Tungsten::SparseHistoryGpis;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;

namespace {
void check(bool valid, const char* message)
{
    if (!valid) throw std::runtime_error(message);
}
void close(double actual, double expected, double tolerance, const char* message)
{
    if (!(std::abs(actual-expected) <= tolerance)) {
        std::cerr << message << ": " << actual << " vs " << expected << '\n';
        throw std::runtime_error(message);
    }
}
cg::GPSSField field()
{
    M metric;
    metric << 1.4,0.2,0.1, 0.2,0.8,-0.1, 0.1,-0.1,1.1;
    return {std::make_shared<cg::PlaneMean>(V(0.2,0.1,1),0.3),
        cg::CovarianceKernel(0.7,metric), {V::Constant(-100),V::Constant(100)}};
}

void denseConditioningReference()
{
    const auto f = field();
    cg::FlightState birth;
    birth.birthPosition = V(0.1,-0.1,0.2);
    birth.direction = V(0.2,-0.3,1).normalized();
    birth.birthValue = 0.24;
    birth.birthGradient = V(0.1,-0.2,1.2);
    cg::SinglePointHistoryTeacher teacher(f,birth);
    const auto origin = f.mean->evaluate(birth.birthPosition);
    Eigen::Matrix4d c0 = Eigen::Matrix4d::Zero();
    c0(0,0) = f.kernel.sigma()*f.kernel.sigma();
    c0.bottomRightCorner<3,3>() = f.kernel.gradientCovarianceAtZero();
    Eigen::Vector4d observation;
    observation << birth.birthValue-origin.value,birth.birthGradient-origin.gradient;
    for (const auto ages : {std::pair<double,double>(0.8,0.35), {1.5,0.6}, {3.0,1.7}, {10.0,0.9}}) {
        const double t = ages.first, u = ages.second;
        const V x = birth.birthPosition+t*birth.direction;
        const V y = birth.birthPosition+u*birth.direction;
        const auto xb = f.kernel.evaluate(x,birth.birthPosition);
        const auto ub = f.kernel.evaluate(y,birth.birthPosition);
        const auto xu = f.kernel.evaluate(x,y);
        Eigen::Matrix<double,3,4> cross;
        cross.row(0) << xb.valueValue,xb.valueXGradientY.transpose();
        cross.row(1) << ub.valueValue,ub.valueXGradientY.transpose();
        cross.row(2) << -birth.direction.dot(xb.gradientXValueY),
                       -birth.direction.transpose()*xb.gradientXGradientY;
        M prior = M::Zero();
        prior(0,0) = prior(1,1) = c0(0,0);
        prior(2,2) = birth.direction.dot(f.kernel.gradientCovarianceAtZero()*birth.direction);
        prior(0,1) = prior(1,0) = xu.valueValue;
        prior(1,2) = prior(2,1) = -birth.direction.dot(xu.gradientXValueY);
        const auto mx = f.mean->evaluate(x), mu = f.mean->evaluate(y);
        const V mean = V(mx.value,mu.value,-birth.direction.dot(mx.gradient))+cross*c0.ldlt().solve(observation);
        const M covariance = prior-cross*c0.ldlt().solve(cross.transpose());
        std::array<double,6> expected;
        for (int i = 0; i < 3; ++i) expected[i] = mean[i]/std::sqrt(covariance(i,i));
        expected[3] = covariance(0,1)/std::sqrt(covariance(0,0)*covariance(1,1));
        expected[4] = covariance(0,2)/std::sqrt(covariance(0,0)*covariance(2,2));
        expected[5] = covariance(1,2)/std::sqrt(covariance(1,1)*covariance(2,2));
        const auto sample = teacher.evaluate(t,u);
        for (int i = 0; i < 6; ++i) close(sample.features[i],expected[i],2e-10,"features vs dense Gaussian conditioning");
        close(sample.baselineHazard,teacher.baseline().evaluate(t).hazard.value,1e-14,"native baseline preserved");
        check(sample.logCorrection+sample.logHistoryPositiveProbability <= 2e-7,"analytic correction envelope");
    }
}

void zeroMeanClosedForm()
{
    auto f = field();
    f.mean = std::make_shared<cg::ConstantMean>(0);
    f.kernel = cg::CovarianceKernel::fromCorrelationLengths(0.02,V::Constant(0.2));
    cg::FlightState birth;
    birth.direction = V(0.2,-0.1,1).normalized();
    cg::SinglePointHistoryTeacher teacher(f,birth);
    for (const auto ages : {std::pair<double,double>(0.24,0.08), {0.6,0.3}, {2.0,1.8}}) {
        const double t = ages.first,u = ages.second;
        const auto sample = teacher.evaluate(t,u);
        const double rxu = sample.features[3], rxv = sample.features[4], ruv = sample.features[5];
        const double rho = (ruv-rxu*rxv)/std::sqrt((1-rxu*rxu)*(1-rxv*rxv));
        // At zero means the selected Gaussian flux fraction is (1+rho)/2,
        // and P(X>0,U>0)=1/4+asin(rxu)/(2*pi). This is independent of the
        // numerical orthant/moment routines used by the teacher.
        const double ratio = (1+rho)/(1+2*std::asin(rxu)/cg::kPi);
        close(std::exp(sample.logCorrection),ratio,2e-7,"zero-mean analytic flux correction");
        close(sample.historyPositiveProbability,0.5,1e-15,"history sign probability");
    }
}

void nativeSurfaceCompatibility()
{
    auto f = field();
    cg::FlightState birth;
    birth.birthPosition = V(0.1,-0.1,0.2);
    birth.direction = V(0.2,-0.3,1).normalized();
    birth.birthValue = 0;
    birth.birthGradient = V(0.1,-0.2,1.2);
    cg::SinglePointHistoryTeacher teacher(f,birth);
    sh::Start surface;
    surface.kind = sh::Start::SurfaceValueGradient;
    surface.origin = birth.birthPosition;
    surface.direction = birth.direction;
    surface.gradient = birth.birthGradient;
    sh::ProbabilitySettings settings;
    settings.relativeTolerance = 1e-8;
    for (const auto ages : {std::pair<double,double>(0.8,0.35), {1.5,0.6}, {3.0,1.7}}) {
        const auto sample = teacher.evaluate(ages.first,ages.second);
        const double reference = sh::LocalLaw(f,surface,ages.first,{ages.second},settings).hazard();
        close(sample.historyHazard,reference,2e-7*std::max(1.0,reference),"native surface-history hazard");
    }
}

void invalidInputs()
{
    auto f = field();
    cg::FlightState birth;
    birth.birthValue = 0.4;
    cg::SinglePointHistoryTeacher teacher(f,birth);
    for (const auto ages : {std::pair<double,double>(0.4,0.4), {0.4,0}, {0.4,-0.1}, {200,0.1}}) {
        bool rejected = false;
        try { teacher.evaluate(ages.first,ages.second); }
        catch (const cg::NumericError&) { rejected = true; }
        check(rejected,"invalid history ages must fail");
    }
    bool rejected = false;
    try { teacher.evaluate(1,1-1e-14); }
    catch (const cg::NumericError&) { rejected = true; }
    check(rejected,"unresolved coincident history must fail explicitly");
    birth.direction *= 2;
    rejected = false;
    try { cg::SinglePointHistoryTeacher invalid(f,birth); }
    catch (const cg::NumericError&) { rejected = true; }
    check(rejected,"non-unit ray must fail");
}
}

int main()
{
    try {
        denseConditioningReference();
        zeroMeanClosedForm();
        nativeSurfaceCompatibility();
        invalidInputs();
        std::cout << "Single-point history teacher checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
