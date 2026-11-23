#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/gpss/NonstationaryConditioning.h"
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/transport/ConditionalSegment.h"
#include "math/conditional/transport/NonstationarySegment.h"
#if OPENVDB_AVAILABLE
#include "media/GpisSdfGrid.hpp"
#endif
#include <iostream>

namespace {
using namespace Tungsten;
namespace cg = ConditionalGpis;
using V = Eigen::Vector3d;
using Matrix4 = Eigen::Matrix4d;

void require(bool condition, const char *message)
{
    if (!condition) throw std::runtime_error(message);
}

std::shared_ptr<ProceduralNonstationaryCovariance> makeKernel(const char *fields)
{
    const std::string json = std::string(R"({"type":"proc_nonstationary","cov":)")+
        R"({"type":"squared_exponential","sigma":1.5,"lengthScale":0.5},)"+fields+"}";
    JsonDocument doc(Path("conditional-nonstationary-test.json"),json);
    Scene scene;
    auto kernel=std::dynamic_pointer_cast<ProceduralNonstationaryCovariance>(scene.fetchCovarianceFunction(doc));
    kernel->loadResources();
    return kernel;
}

Matrix4 jointMatrix(const cg::KernelJet &jet)
{
    Matrix4 result;
    result << jet.valueValue, jet.valueXGradientY.transpose(), jet.gradientXValueY, jet.gradientXGradientY;
    return result;
}

Matrix4 jointMatrix(const cg::PointPrior &prior)
{
    Matrix4 result;
    result << prior.varianceF, prior.covarianceFG.transpose(), prior.covarianceFG, prior.covarianceG;
    return result;
}

// Differentiate the original function-space scalar covariance at both points.
// Its legacy autodiff overloads hold the procedural parameters fixed.
Matrix4 referenceMatrix(const CovarianceFunction &kernel, const V &x, const V &y)
{
    const auto value=[&](const V &a,const V &b) {
        return kernel(Derivative::None,Derivative::None,Vec3d(a.x(),a.y(),a.z()),
            Vec3d(b.x(),b.y(),b.z()),Vec3d(0.0),Vec3d(0.0));
    };
    const double h=5e-5;
    Matrix4 matrix;
    matrix(0,0)=value(x,y);
    for (int i=0;i<3;++i) {
        const V di=V::Unit(i)*h;
        matrix(i+1,0)=(value(x+di,y)-value(x-di,y))/(2*h);
        matrix(0,i+1)=(value(x,y+di)-value(x,y-di))/(2*h);
        for (int j=0;j<3;++j) {
            const V dj=V::Unit(j)*h;
            matrix(i+1,j+1)=(value(x+di,y+dj)-value(x-di,y+dj)
                -value(x+di,y-dj)+value(x-di,y-dj))/(4*h*h);
        }
    }
    return matrix;
}

void conditioningChecks(const CovarianceFunction &source, const cg::NonstationaryConditioning &kernel,
                        const cg::MeanField &mean, const cg::FlightState &birth)
{
    const auto origin=birth.birthPosition;
    const auto birthJet=mean.evaluate(origin);
    const Matrix4 atBirth=referenceMatrix(source,origin,origin);
    const auto prior=kernel.pointPrior(origin,birthJet);
    require((jointMatrix(prior)-atBirth).norm()<2e-5,"nonstationary local value/gradient covariance");
    require(std::abs(atBirth(2,0))>0.25,"amplitude ramp must correlate value and gradient");
    const cg::NonstationaryFlight flight(kernel,mean,birth);
    Eigen::Vector4d residual;
    residual << birth.birthValue-birthJet.value,birth.birthGradient-birthJet.gradient;
    for (double t : {0.23,0.65,1.4}) {
        const V x=origin+t*birth.direction;
        const auto target=mean.evaluate(x);
        const Matrix4 cross=referenceMatrix(source,x,origin);
        const Matrix4 expectedCov=referenceMatrix(source,x,x)-cross*atBirth.ldlt().solve(cross.transpose());
        Eigen::Vector4d expectedMean;
        expectedMean << target.value,target.gradient;
        expectedMean+=cross*atBirth.ldlt().solve(residual);
        require((jointMatrix(kernel.evaluate(x,origin))-cross).norm()<2e-5,
            "two-point value/gradient covariance vs finite differences");
        const auto law=flight.at(t);
        require(std::abs(law.meanF-expectedMean[0])<2e-6 &&
                (law.meanG-expectedMean.tail<3>()).norm()<2e-6,"conditioned value and gradient means");
        require((jointMatrix(law)-expectedCov).norm()<4e-5,"conditioned joint covariance vs Gaussian regression");
    }
    const auto tiny=flight.at(1e-6);
    const auto twice=flight.at(2e-6);
    require(tiny.varianceF>0 && std::abs(twice.varianceF/tiny.varianceF-16)<0.05,
        "variance near a fully observed smooth birth is quartic in distance");
    const double slopeVariance=flight.moments(1e-9).slopeVarianceGivenValue;
    require(slopeVariance>0 && std::abs(flight.moments(2e-9).slopeVarianceGivenValue/slopeVariance-16)<0.05,
        "conditional slope retains its quartic residual at near-coincident points");
}

void constantFieldLimit(const cg::FlightState &birth)
{
    const auto source=makeKernel(R"("var":{"type":"constant","val":0.5},"ls":{"type":"constant","val":[1,1,1]})");
    const cg::NonstationaryConditioning kernel(*source);
    auto mean=std::make_shared<cg::ConstantMean>(0.2);
    const cg::GPSSField stationary{mean,cg::CovarianceKernel(0.75,4*Eigen::Matrix3d::Identity()),
        {V::Constant(-10),V::Constant(10)}};
    const cg::ConditionalFlightKernel reference(stationary,birth);
    const cg::NonstationaryFlight flight(kernel,*mean,birth);
    const cg::NonstationarySegment segment(kernel,*mean,birth,birth.birthPosition,birth.direction,2.0);
    for (double t : {0.07,0.29,0.8,1.9}) {
        const auto stats=reference.scalarStatistics(t);
        const auto law=flight.at(t);
        const auto gradient=segment.gradient(t);
        const auto expected=reference.hitStatistics(t).collisionGradient;
        require(std::abs(law.varianceF-stats.varianceF)<1e-10 && std::abs(law.meanF-stats.meanF)<1e-10,
            "constant parameter fields reproduce stationary value statistics");
        require((gradient.mean-expected.mean).norm()<1e-6 && (gradient.covariance-expected.covariance).norm()<1e-7,
            "constant parameter fields reproduce stationary collision gradients");
        require(std::abs(segment.hazard(t)-reference.evaluate(t).hazard.value)<1e-8,
            "constant parameter fields reproduce stationary crossing hazard");
    }
}

#if OPENVDB_AVAILABLE
template<class NodeValue>
GpisSdfGrid makeTestGrid(NodeValue nodeValue)
{
    GpisSdfGrid grid;
    grid.data.dimensions=Vec3i(5);
    grid.data.origin=Vec3i(0);
    grid.data.clampMin=Vec3f(0.125f);
    grid.data.clampMax=Vec3f(3.75f);
    grid.worldToIndex=Eigen::Matrix3d::Identity();
    grid.translation=V::Zero();
    grid.scale=1;
    grid.offset=0;
    for (int z=0;z<5;++z) for (int y=0;y<5;++y) for (int x=0;x<5;++x)
        grid.data.values.push_back(float(nodeValue(x,y,z)));
    grid.initializeMagnitudeBounds();
    return grid;
}

void gridIncrementsAcrossFaces()
{
    const auto grid=makeTestGrid([](int x,int y,int z) {
        return x*x/4.0+y*y/8.0-z*z/4.0+x*y*z/16.0;
    });
    const std::pair<V,V> endpoints[]={
        {V(0.3,0.4,0.5),V(3.4,3.1,2.9)},       // Several non-coplanar cells.
        {V(3.6,2.8,3.4),V(0.3,0.2,0.4)},       // Negative direction on every axis.
        {V(-1.5,2.3,4.1),V(4.5,-0.7,-0.1)},    // Both endpoints are clamped.
        {V(0.5,0.5,0.5),V(3.5,3.5,3.5)},       // Simultaneous face crossings.
    };
    for (const auto &[p,q] : endpoints) {
        const double expected=grid.evaluate(q).value-grid.evaluate(p).value;
        require(std::abs(grid.valueDifference(p,q-p)-expected)<2e-13,
            "cross-cell grid increments reproduce the trilinear endpoint difference");
    }

    // At y=z=1.5 the x-slopes on either side of x=2 are 3/4+yz/16 and
    // 5/4+yz/16. The first displacement crosses the face, but p+d rounds
    // onto it. The reverse query starts exactly on the face.
    const double left=3.0/4+2.25/16, right=5.0/4+2.25/16;
    const double x=std::nextafter(2.0,1.0), toFace=2.0-x;
    const double step=1.5*toFace;
    const double forward=grid.valueDifference(V(x,1.5,1.5),V(step,0,0));
    require(std::abs(forward-(left*toFace+right*(step-toFace)))<1e-29,
        "sub-ulp motion past a grid face uses both one-sided slopes");
    const double backward=grid.valueDifference(V(2.0,1.5,1.5),V(-0.25*toFace,0,0));
    require(std::abs(backward+0.25*toFace*left)<1e-29,
        "motion from a grid face uses the slope in the direction of travel");
}

void shallowBirthAcrossGridFace()
{
    // Both grid cells contain the same plane m(x)=x-2. Near a fully observed
    // birth, tiny errors in m(x+t)-m(x) are amplified by the O(t^2) value SD.
    const auto grid=makeTestGrid([](int x,int,int) { return x-2; });

    // The SE base in makeKernel has amplitude 1.5 and length 0.5. These
    // constant multipliers give unit amplitude and correlation length 0.1.
    const auto source=makeKernel(R"("var":{"type":"constant","val":0.6666666666666666},
        "ls":{"type":"constant","val":[0.2,0.2,0.2]})");
    const cg::NonstationaryConditioning kernel(*source);
    cg::FlightState birth;
    birth.birthPosition=V(2.0-1e-8,1.25,1.75);
    birth.birthGradient=V(1e-5,0,0);
    birth.direction=V::UnitX();
    const cg::NonstationarySegment flight(kernel,grid,birth,birth.birthPosition,birth.direction,0.05);

    // Independent reference: condition the analytic 1D SE covariance on F
    // and F' at birth and integrate the resulting Rice hazard in log distance
    // at 75-digit precision. Optical depth is 0.7708152798104461.
    cg::Random rng(7);
    double hit;
    const double transmittance=flight.track(rng,false,hit);
    require(std::abs(transmittance-0.4626357369408857)<1e-5,
        "near-birth grid-face crossing agrees with the analytic plane transmission");
}
#endif

void transportChecks(const cg::GPSSField &field, const cg::FlightState &birth)
{
    const double length=1.9;
    cg::ConditionalSegment segment(field,birth,birth.birthPosition,birth.direction,length);
    const cg::NonstationarySegment reference(*field.nonstationary,*field.mean,birth,birth.birthPosition,birth.direction,length);
    require(std::abs(segment.hazard(0.65)-reference.hazard(0.65))<1e-10,
        "public segment uses the varying kernel hazard");
    const auto midpointDepth=[&](double end) {
        double depth=0;
        for (double begin=0;begin<end;) {
            const double next=std::min(end,field.nonstationary->nextBreakpoint(birth.birthPosition,birth.direction,begin));
            require(next>begin,"procedural ramp breakpoint must advance");
            const int n=256;
            const double step=(next-begin)/n;
            for (int i=0;i<n;++i) depth+=step*reference.hazard(begin+(i+0.5)*step);
            begin=next;
        }
        return depth;
    };
    const double depth=midpointDepth(length);
    require(depth>0.05 && depth<10,"flight must exercise nontrivial survival");
    cg::Random random(2638);
    double hit=0;
    const double visibility=segment.track(random,false,hit);
    require(std::abs(visibility-std::exp(-depth))<5e-5,"shadow transmission vs integrated crossing hazard");
    // Replay one exponential threshold so inversion is checked independently
    // of Monte Carlo noise, including any procedural ramp clamp on its way.
    for (unsigned seed=1;seed<200;++seed) {
        cg::Random proposal(seed);
        const double threshold=-std::log(proposal.openUniform01());
        if (threshold<0.2*depth || threshold>0.9*depth) continue;
        cg::Random replay(seed);
        require(segment.track(replay,true,hit)==0 && hit>0 && hit<length,"collision threshold inside segment");
        require(std::abs(midpointDepth(hit)-threshold)<1e-4,"inverted collision distance vs crossing integral");
        const auto endpoint=segment.samplePositiveEndpoint(random);
        require(endpoint.birthValue>0 && endpoint.birthGradient.allFinite() &&
                (endpoint.birthPosition-birth.birthPosition-length*birth.direction).norm()<1e-12,
            "escaped segment renews complete positive endpoint");
        return;
    }
    throw std::runtime_error("failed to choose an interior collision threshold");
}

void exteriorBirthChecks(cg::GPSSField field, const cg::FlightState &birth)
{
    field.mean=std::make_shared<cg::ConstantMean>(0.0);
    const auto prior=field.pointPrior(birth.birthPosition);
    const auto reference=field.nonstationary->pointPrior(birth.birthPosition,field.mean->evaluate(birth.birthPosition));
    require((jointMatrix(prior)-jointMatrix(reference)).norm()<1e-12,"public field uses nonstationary prior");
    const double sigma=std::sqrt(prior.varianceF);
    const double halfNormalMean=std::sqrt(2/cg::kPi);
    const V expectedGradient=halfNormalMean*prior.covarianceFG/sigma;
    double valueSum=0;
    V gradientSum=V::Zero();
    cg::Random rng(2149);
    constexpr int samples=8192;
    for (int i=0;i<samples;++i) {
        const auto sample=cg::startConditionalExterior(field,birth.birthPosition,birth.direction,rng);
        require(sample.birthValue>0,"exterior birth has positive value");
        valueSum+=sample.birthValue;
        gradientSum+=sample.birthGradient;
    }
    require(std::abs(valueSum/samples-sigma*halfNormalMean)<6*sigma/std::sqrt(samples),
        "exterior values follow the truncated local prior");
    const V tolerance=(6/std::sqrt(samples))*prior.covarianceG.diagonal().cwiseSqrt();
    require(((gradientSum/samples-expectedGradient).cwiseAbs().array()<tolerance.array()).all(),
        "exterior gradient retains correlation with the positive field value");
}
}

void nonstationaryChecks()
{
    const auto source=makeKernel(R"("var":{"type":"noise","noise":"bottom_top","min":0.8,"max":0.35,"start":-0.5,"end":0.7},
        "ls":{"type":"noise","noise":"left_right","min":0.6,"max":1.1,"start":-0.5,"end":0.5})");
    const cg::GPSSField field{std::make_shared<cg::PlaneMean>(V(-0.1,0.25,0.7),0.2),{},
        {V::Constant(-10),V::Constant(10)},std::make_shared<cg::NonstationaryConditioning>(*source)};
    cg::FlightState birth;
    birth.birthPosition=V(-0.25,-0.15,0.05);
    birth.direction=V(1,0.22,-0.17).normalized();
    birth.birthValue=0.4;
    birth.birthGradient=V(0.25,-0.4,0.6);
    conditioningChecks(*source,*field.nonstationary,*field.mean,birth);
    constantFieldLimit(birth);
#if OPENVDB_AVAILABLE
    shallowBirthAcrossGridFace();
    gridIncrementsAcrossFaces();
#endif
    exteriorBirthChecks(field,birth);
    transportChecks(field,birth);
    std::cout<<"Nonstationary kernel derivatives, joint conditioning, stationary limit and transport passed\n";
}
