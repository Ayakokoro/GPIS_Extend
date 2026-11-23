#include "io/JsonDocument.hpp"
#include "io/JsonLoadException.hpp"
#include "io/Scene.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/gpss/NonstationaryConditioning.h"
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/transport/ConditionalSegment.h"
#include "math/conditional/transport/NonstationarySegment.h"
#include "media/GlobalConditionalMedium.hpp"
#if OPENVDB_AVAILABLE
#include "media/GpisSdfGrid.hpp"
#endif
#include <array>
#include <iostream>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>

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

void nearRampMoments()
{
    const auto source=makeKernel(R"("var":{"type":"constant","val":1},
        "ls":{"type":"noise","noise":"left_right","min":0.0316,"max":0.223,"start":-0.5,"end":0})");
    const cg::NonstationaryConditioning kernel(*source);
    const cg::ConstantMean mean(0);
    struct Reference {
        double birthX, age;
        V direction;
        double variance, slopeRegression, slopeVariance, transverseVariance;
    };
    // Differentiate K=2.25*(2*lx*ly/S)^(3/2)*exp(-|x-y|^2/S), S=lx^2+ly^2,
    // at 100 decimal digits. Condition the resulting 4-by-4 blocks on birth
    // value and gradient, then on target value. The final ray crosses the
    // lower clamp about a quarter of the way to the target.
    const Reference cases[]={
        {-0.25,1e-9,V(0.6,0.8,0),8.29800711249851058e-32,1999999998.1643518,
            7.99668902405315208e-30,1.61244129699627418e-13},
        {-0.25,0.004,V(0.6,0.8,0),2.08729811564701438e-5,497.446664457740699,
            0.00198417328894764701,2.5176527912633844},
        {-0.50000000025,1e-9,V::UnitX(),2.34424447242454984e-16,1333333471.53013568,
            6.9499910475088133e-12,3.76689863176141779e-11}
    };
    for (const auto &ref : cases) {
        cg::FlightState birth;
        birth.birthPosition=V(ref.birthX,0,0);
        birth.direction=ref.direction;
        const auto m=cg::NonstationaryFlight(kernel,mean,birth).moments(ref.age);
        require(std::abs(m.varianceF/ref.variance-1)<2e-6,"near-ramp field variance");
        require(std::abs(ref.direction.dot(m.regression)/ref.slopeRegression-1)<5e-10,
            "near-ramp directional value-to-slope regression");
        require(std::abs(m.slopeVarianceGivenValue/ref.slopeVariance-1)<5e-6,
            "near-ramp conditional slope variance");
        require(std::abs(m.gradientGivenValueCovariance(2,2)/ref.transverseVariance-1)<2e-6,
            "near-ramp conditional transverse gradient variance");
    }
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
    for (double split : {0.0, 1e-150, 1e-4, 0.05}) {
        const double transmittance=flight.track(rng,false,hit,0.0,split);
        require(std::abs(transmittance-0.4626357369408857)<1e-5,
            "splitting at a grazing birth preserves the analytic plane transmission");
    }
    require(flight.track(rng,false,hit,1e-4)>0.56,
        "this grazing flight has significant optical depth before the split");
}
#endif

// Distinguish an optical integration boundary (which evaluates the field)
// from a geometry integration cell (which queries the mean's breakpoints).
class SplitProbeMean final : public cg::MeanField {
    const cg::MeanField &_source;
    V _splitPoint;
public:
    mutable bool sawSplit=false;
    mutable std::vector<double> cellStarts;
    SplitProbeMean(const cg::MeanField &source,const V &splitPoint) : _source(source),_splitPoint(splitPoint) {}
    const char *typeName() const override { return "split_probe"; }
    cg::MeanJet evaluate(const V &x) const override {
        if ((x-_splitPoint).cwiseAbs().maxCoeff()<1e-14) sawSplit=true;
        return _source.evaluate(x);
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override { return _source.bounds(b); }
    std::optional<V> affineGradient() const override { return _source.affineGradient(); }
    void appendRayBreakpoints(const V &origin,const V &w,double begin,double end,std::vector<double> &out) const override {
        cellStarts.push_back(begin);
        _source.appendRayBreakpoints(origin,w,begin,end,out);
    }
};

void transportChecks(const cg::GPSSField &field, const cg::FlightState &birth)
{
    const double length=1.9, begin=0.25, split=0.7;
    auto observed=field;
    const auto probe=std::make_shared<SplitProbeMean>(*field.mean,birth.birthPosition+split*birth.direction);
    observed.mean=probe;
    cg::ConditionalSegment segment(observed,birth,birth.birthPosition,birth.direction,length);
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
    // An actual geometry clip removes [0,begin], while split only partitions
    // the remaining optical depth. Replay a threshold in each possible region.
    const double clippedDepth=midpointDepth(begin);
    const double prefix=midpointDepth(split)-clippedDepth, total=depth-clippedDepth;
    probe->cellStarts.clear();
    segment.track(random,false,hit,nullptr,begin);
    const auto originalStarts=probe->cellStarts;
    require(!probe->sawSplit,"fixture split must lie inside a natural integration interval");
    probe->cellStarts.clear();
    require(std::abs(segment.track(random,false,hit,nullptr,begin,split)-std::exp(-total))<5e-5,
        "split shadow integrates both pieces of the physical interval");
    require(probe->sawSplit,"optical depth evaluates the field at the requested split");
    require(!originalStarts.empty() && probe->cellStarts==originalStarts,
        "an optical split preserves all original geometry cells");
    bool covered[3]={};
    for (unsigned seed=1;seed<=1024 && !(covered[0] && covered[1] && covered[2]);++seed) {
        cg::Random proposal(seed);
        const double threshold=-std::log(proposal.openUniform01());
        if (std::abs(threshold-prefix)<1e-3 || std::abs(threshold-total)<1e-3) continue;
        const int region=threshold<prefix ? 0 : threshold<total ? 1 : 2;
        if (covered[region]) continue;
        cg::Random replay(seed);
        const double weight=segment.track(replay,true,hit,nullptr,begin,split);
        if (region==2) {
            require(weight==1 && hit==length,"threshold beyond both intervals escapes at the endpoint");
        } else {
            require(weight==0 && hit>begin && hit<length && (hit<split)==(region==0),
                "the collision lies on the expected side of the split");
            require(std::abs(midpointDepth(hit)-clippedDepth-threshold)<1e-4,
                "the sampled distance inverts the original crossing integral");
        }
        require(replay.openUniform01()==proposal.openUniform01(),"the whole flight consumes one exponential draw");
        covered[region]=true;
    }
    require(covered[0] && covered[1] && covered[2],"exercise collisions before and after the split and escape");
    const auto endpoint=segment.samplePositiveEndpoint(random);
    require(endpoint.birthValue>0 && endpoint.birthGradient.allFinite() &&
            (endpoint.birthPosition-birth.birthPosition-length*birth.direction).norm()<1e-12,
        "escaped segment renews complete positive endpoint");
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

void nonstationaryMajorantChecks()
{
    const auto ramp=makeKernel(R"("var":{"type":"noise","noise":"left_right","min":0.8,"max":0.35,"start":0.5,"end":-0.5},
        "ls":{"type":"noise","noise":"left_right","min":0.12,"max":0.30,"start":-0.5,"end":0.5})");
    const auto anisotropic=makeKernel(R"("var":{"type":"constant","val":0.5},
        "ls":{"type":"constant","val":[0.16,0.3,0.2]})");
    for (const auto &source : {ramp,anisotropic}) {
        const cg::NonstationaryConditioning kernel(*source);
        for (double startX : {-2.,0.,2.}) {
            const double sign=startX>0 ? -1. : 1.;
            const bool activeBirth=startX==0.;
            cg::FlightState birth;
            birth.direction=sign*(activeBirth ? V(0.4,1,0.3) : V(1,0.4,0.3)).normalized();
            birth.birthPosition=V(startX,0,0);
            birth.birthValue=0.3;
            birth.birthGradient=V(-2,3,1);
            const cg::PlaneMean mean(V(1,0,0),2*sign);
            const cg::NonstationaryFlight flight(kernel,mean,birth);
            const cg::NonstationarySegment segment(kernel,mean,birth,birth.birthPosition,birth.direction,4);
            const double enter=1.5/std::abs(birth.direction.x());
            const double leave=(activeBirth ? 0.5 : 2.5)/std::abs(birth.direction.x());
            require(std::isinf(flight.hazardMajorant(0.8,leave,cg::MeanRayBounds{})),
                "an uncertified mean requests numerical integration");
            // Include both sides of each clamp and the values exactly on it.
            using Knots=std::array<double,10>;
            const Knots knots=activeBirth ? Knots{0.1,0.2,0.3,0.4,0.55,0.8,leave-0.05,leave,leave+0.05,3.8}
                : Knots{0.25,0.45,enter-0.05,enter,enter+0.05,0.5*(enter+leave),leave-0.05,leave,leave+0.05,3.8};
            const double farAge=activeBirth ? 0.55 : enter-0.05;
            for (size_t i=1;i<knots.size();++i) {
                const double a=knots[i-1], b=knots[i];
                const auto range=mean.rayBounds(birth.birthPosition,birth.direction,a,b);
                const double bound=flight.hazardMajorant(a,b,range);
                require(bound>=0 && (a<farAge || std::isfinite(bound)),
                    "the decorrelated ramp and plateaus admit finite certificates");
                for (int j=0;j<=32;++j) {
                    const double age=j==32 ? b : a+(b-a)*(j/32.);
                    require(segment.hazard(age)<=bound,"pointwise crossing hazard exceeds its certificate");
                }
            }
        }
    }
}

void localWidthMajorantChecks()
{
    const auto source=makeKernel(R"("var":{"type":"constant","val":1},
        "ls":{"type":"noise","noise":"left_right","min":0.0316,"max":0.223,"start":-0.5,"end":0})");
    const cg::NonstationaryConditioning kernel(*source);
    const cg::ConstantMean mean(-1);
    cg::FlightState birth;
    birth.birthPosition=V(-0.42,0,0);
    birth.direction=V(-0.6,0.8,0);
    birth.birthGradient=V(-2,3,1);
    const cg::NonstationaryFlight flight(kernel,mean,birth);
    const cg::NonstationarySegment segment(kernel,mean,birth,birth.birthPosition,birth.direction,0.2);
    const double clamp=kernel.nextBreakpoint(birth.birthPosition,birth.direction,0);
    // The birth lies inside the length ramp. A global target-width range fails
    // the variance gate even after the target reaches the narrow plateau.
    for (double center : {clamp-0.025,clamp,clamp+0.025}) {
        const double a=center-0.005, b=center+0.005;
        const auto range=mean.rayBounds(birth.birthPosition,birth.direction,a,b);
        const double bound=flight.hazardMajorant(a,b,range);
        require(std::isfinite(bound),"locally decorrelated narrow widths admit a finite majorant");
        for (double age : {a,std::nextafter(center,a),center,std::nextafter(center,b),b}) {
            require(segment.hazard(age)<=bound,"local width majorant covers the ramp and its clamp");
        }
    }
}

// Withhold the certificate on a prescribed part of the ray while preserving
// point evaluations. This exercises transitions in both directions.
class PartiallyCertifiedMean final : public cg::MeanField {
    const cg::MeanField &_mean;
    double _minX, _maxX;
public:
    PartiallyCertifiedMean(const cg::MeanField &mean,double minX,double maxX)
        : _mean(mean), _minX(minX), _maxX(maxX) {}
    const char *typeName() const override { return "partially_certified_test_mean"; }
    cg::MeanJet evaluate(const V &x) const override { return _mean.evaluate(x); }
    cg::BoundsSummary bounds(const cg::Bounds3 &box) const override { return _mean.bounds(box); }
    std::optional<V> affineGradient() const override { return _mean.affineGradient(); }
    cg::MeanRayBounds rayBounds(const V &origin,const V &w,double begin,double end,
                                const SdfRayQueryOptions &options) const override {
        const double x0=origin.x()+w.x()*begin, x1=origin.x()+w.x()*end;
        if (std::min(x0,x1)<_maxX && std::max(x0,x1)>_minX) return {};
        return _mean.rayBounds(origin,w,begin,end,options);
    }
};

void hybridTransportChecks(cg::GPSSField field, const cg::FlightState &birth)
{
    // Start along an existing flight. Geometry clips the segment at begin,
    // while the original birth still determines the conditional hazard.
    const V origin=birth.birthPosition+3.8*birth.direction;
    const double begin=0.18, split=0.43, insideTail=0.72, end=1.04;
    cg::ConditionalTrackingSettings settings;
    settings.nonstationaryHybrid=true;
    auto mixedField=field;
    mixedField.mean=std::make_shared<PartiallyCertifiedMean>(*field.mean,
        origin.x()+0.70*birth.direction.x(),origin.x()+0.73*birth.direction.x());
    const cg::ConditionalSegment hybrid(mixedField,birth,origin,birth.direction,end,settings);
    cg::Random referenceRng(1);
    double hit;
    const auto referenceSurvival=[&](double limit) {
        const cg::ConditionalSegment reference(field,birth,origin,birth.direction,limit);
        return reference.track(referenceRng,false,hit,nullptr,begin,split);
    };
    const double prefixSurvival=referenceSurvival(split);
    const double tailSurvival=referenceSurvival(insideTail);
    const double escape=referenceSurvival(end);
    require(prefixSurvival<0.94 && escape>0.05 && prefixSurvival-escape>0.1,
        "fixture has appreciable early collisions, tail collisions, and escapes");

    constexpr int samples=1024;
    cg::Random collisions(4901), shadows(5309);
    int atSplit=0, atTailPoint=0, escaped=0;
    double shadowSum=0;
    bool collisionCandidate=false, shadowCandidate=false, shadowUsedBoth=false;
    double minimumWeight=1, maximumWeight=0;
    for (int i=0;i<samples;++i) {
        cg::ConditionalTrackingWork collisionWork, shadowWork;
        const double survived=hybrid.track(collisions,true,hit,&collisionWork,begin,split);
        require((survived==1 && hit==end) || (survived==0 && begin<hit && hit<end),
            "hybrid collision distance respects the clipped physical interval");
        atSplit+=hit>=split;
        atTailPoint+=hit>=insideTail;
        escaped+=survived==1;
        const double weight=hybrid.track(shadows,false,hit,&shadowWork,begin,split);
        require(weight>=0 && weight<=prefixSurvival+1e-6 && hit==end,
            "ratio weights include the integrated prefix");
        minimumWeight=std::min(minimumWeight,weight);
        maximumWeight=std::max(maximumWeight,weight);
        collisionCandidate|=collisionWork.nonstationaryBoundedIntervals && collisionWork.candidates;
        shadowCandidate|=shadowWork.nonstationaryBoundedIntervals && shadowWork.candidates;
        shadowUsedBoth|=shadowWork.nonstationaryPrefixIntervals && shadowWork.nonstationaryFallbackIntervals
            && shadowWork.nonstationaryBoundedIntervals;
        shadowSum+=weight;
    }
    const auto checkSurvival=[samples](double sum,double expected) {
        // All estimators lie in [0,1], so Var(X) <= E[X]*(1-E[X]).
        const double tolerance=5*std::sqrt(expected*(1-expected)/samples)+1e-5;
        require(std::abs(sum/samples-expected)<tolerance,"hybrid survival vs integrated hazard");
    };
    checkSurvival(atSplit,prefixSurvival);
    checkSurvival(atTailPoint,tailSurvival);
    checkSurvival(escaped,escape);
    checkSurvival(shadowSum,escape);
    require(shadowUsedBoth,"shadow transport includes the prefix, certified tail and fallback intervals");
    require(collisionCandidate && shadowCandidate && maximumWeight-minimumWeight>0.01,
        "both tail estimators visit candidates and the ratio weights vary across samples");

    auto fallbackField=field;
    fallbackField.mean=std::make_shared<PartiallyCertifiedMean>(*field.mean,-1e3,1e3);
    const cg::ConditionalSegment fallback(fallbackField,birth,origin,birth.direction,end,settings);
    cg::ConditionalTrackingWork work;
    require(std::abs(fallback.track(shadows,false,hit,&work,begin,split)-escape)<1e-5 &&
        work.nonstationaryFallbackIntervals && !work.nonstationaryBoundedIntervals && !work.candidates,
        "uncertified shadow intervals contribute their full integrated optical depth");
    std::cout<<"Hybrid survival ("<<samples<<" samples, measured/reference)"
             <<" split="<<double(atSplit)/samples<<"/"<<prefixSurvival
             <<" tail="<<double(atTailPoint)/samples<<"/"<<tailSurvival
             <<" escape="<<double(escaped)/samples<<"/"<<escape
             <<" ratio="<<shadowSum/samples<<"/"<<escape<<"\n";
}

void nonstationaryTrackingConfigChecks()
{
    Scene scene;
    const auto load=[&](const std::string &option) {
        const std::string json=std::string(R"({"type":"global_conditional",)")+option+R"(
            "gaussian_process":{"type":"standard","mean":{"type":"linear","direction":[0,0,1]},
            "covariance":{"type":"proc_nonstationary","cov":{"type":"squared_exponential","sigma":1,"lengthScale":0.2},
            "ls":{"type":"noise","noise":"left_right","min":0.2,"max":1,"start":-1,"end":1}}},
            "material":{"type":"mirror","albedo":1}})";
        JsonDocument input(Path("nonstationary-tracking.json"),json);
        return std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(input));
    };
    const auto hybrid=load(R"("nonstationary_tracking":"hybrid",)");
    require(hybrid->trackingSettings().nonstationaryHybrid && !load("")->trackingSettings().nonstationaryHybrid &&
        !load(R"("nonstationary_tracking":"quadrature",)")->trackingSettings().nonstationaryHybrid,
        "nonstationary tracking is opt-in");
    rapidjson::Document allocator;
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    hybrid->toJson(allocator.GetAllocator()).Accept(writer);
    JsonDocument roundTrip(Path("nonstationary-tracking-roundtrip.json"),buffer.GetString());
    const auto copy=std::dynamic_pointer_cast<GlobalConditionalMedium>(scene.fetchMedium(roundTrip));
    require(copy && copy->trackingSettings().nonstationaryHybrid,"hybrid tracking survives serialization");
    bool rejected=false;
    try { load(R"("nonstationary_tracking":"invalid",)"); }
    catch (const JsonLoadException &error) { rejected=error.description().find("nonstationary_tracking")!=std::string::npos; }
    require(rejected,"reject unknown nonstationary tracking modes at parse time");
}

#if OPENVDB_AVAILABLE
void nonstationaryCoarseGridChecks()
{
    const auto source=makeKernel(R"("var":{"type":"constant","val":1},
        "ls":{"type":"noise","noise":"left_right","min":0.4,"max":0.7,"start":-2,"end":-1})");
    const auto kernel=std::make_shared<cg::NonstationaryConditioning>(*source);
    cg::FlightState birth;
    birth.birthPosition=V(0.25,0.5,0.5);
    birth.direction=V::UnitX();
    birth.birthValue=0.4;
    birth.birthGradient=V::UnitX();
    const double initialAge=2.1, begin=0.06, end=0.66;
    const V origin=birth.birthPosition+initialAge*birth.direction;
    cg::ConditionalTrackingSettings settings;
    settings.nonstationaryHybrid=true;
    settings.macrocellSize=2;
    for (bool looseBlock : {false,true}) {
        // The ray interpolates only y=0 and y=1. Large off-ray values in
        // the same macrocell make the coarse certificate expensive while
        // preserving the exact point hazard and every fine-cell certificate.
        auto grid=std::make_shared<GpisSdfGrid>(makeTestGrid([&](int,int y,int) {
            return looseBlock && y>=2 ? -256. : 0.;
        }));
        grid->buildMacrocells(settings.macrocellSize);
        const cg::GPSSField field{grid,{}, {V::Constant(-10),V::Constant(10)},kernel};
        const cg::ConditionalSegment reference(field,birth,origin,birth.direction,end);
        const cg::ConditionalSegment hybrid(field,birth,origin,birth.direction,end,settings);
        cg::Random collisions(213),shadows(708);
        double hit;
        const double expected=reference.track(shadows,false,hit,nullptr,begin);
        constexpr int samples=384;
        double escaped=0, sum=0;
        std::uint64_t coarse=0, bounded=0, candidates=0;
        for (int i=0;i<samples;++i) {
            cg::ConditionalTrackingWork work;
            sum+=hybrid.track(shadows,false,hit,&work,begin);
            coarse+=work.nonstationaryCoarseIntervals;
            bounded+=work.nonstationaryBoundedIntervals;
            candidates+=work.candidates;
            escaped+=hybrid.track(collisions,true,hit,nullptr,begin);
            require(hit>=begin && hit<=end,"grid tracking respects the geometry clip on a continued flight");
        }
        require(candidates>0 && bounded>0 && (looseBlock ? coarse==0 : coarse==bounded),
            "a usable macrocell skips fine tracking and an expensive macrocell descends to fine tracking");
        const double tolerance=5*std::sqrt(expected*(1-expected)/samples);
        require(std::abs(escaped/samples-expected)<tolerance && std::abs(sum/samples-expected)<tolerance,
            "coarse and fine VDB tracking match integrated survival on the continued ray");
    }
}
#endif
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
    nearRampMoments();
    constantFieldLimit(birth);
#if OPENVDB_AVAILABLE
    shallowBirthAcrossGridFace();
    gridIncrementsAcrossFaces();
#endif
    exteriorBirthChecks(field,birth);
    transportChecks(field,birth);
    nonstationaryMajorantChecks();
    localWidthMajorantChecks();
    hybridTransportChecks(field,birth);
    nonstationaryTrackingConfigChecks();
#if OPENVDB_AVAILABLE
    nonstationaryCoarseGridChecks();
#endif
    std::cout<<"Nonstationary kernel derivatives, joint conditioning, majorants and transport passed\n";
}
