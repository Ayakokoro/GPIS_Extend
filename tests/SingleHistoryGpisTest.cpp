#include "math/SparseHistoryGaussian.hpp"
#include "math/SingleHistoryFlight.hpp"
#include "math/conditional/transport/ConditionalFlightKernel.h"
#include "math/conditional/mathutility/Gaussian1D.h"
#include "media/SingleHistoryGpisMedium.hpp"
#include "media/GpisSdfGrid.hpp"
#include "SingleHistoryReferenceExport.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "phasefunctions/PhaseFunction.hpp"
#include "samplerecords/MediumSample.hpp"
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace sh = Tungsten::SparseHistoryGpis;
namespace one = Tungsten::SingleHistoryGpis;
namespace cg = Tungsten::ConditionalGpis;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;

namespace {
void require(bool value,const char *message) {
    if(!value) throw std::runtime_error(message);
}
void close(double actual,double expected,double tolerance,const char *message) {
    if(!std::isfinite(actual)||std::abs(actual-expected)>tolerance) {
        std::cerr<<std::setprecision(17)<<message<<": "<<actual<<" vs "<<expected
                 <<" (tolerance "<<tolerance<<")\n";
        throw std::runtime_error(message);
    }
}
cg::GPSSField field(std::shared_ptr<const cg::MeanField> mean,double sigma=1,
                   M metric=M::Identity()) {
    return {std::move(mean),cg::CovarianceKernel(sigma,std::move(metric)),
        {V::Constant(-100),V::Constant(100)}};
}

// The stationary points are geometry knots, independent of random candidates
// and numerical integration nodes. This is the published piecewise mean profile.
class ProfileMean final : public cg::MeanField {
public:
    const char *typeName() const override { return "single-history-test-profile"; }
    cg::MeanJet evaluate(const V &p) const override {
        const double t=p.z();
        if(t<0.5) return {t,V(0,0,1)};
        if(t<1.5) return {0.5-1.3*(t-0.5),V(0,0,-1.3)};
        if(t<2.25) return {-0.8+(1.25/0.75)*(t-1.5),V(0,0,1.25/0.75)};
        return {0.45-0.5*(t-2.25),V(0,0,-0.5)};
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &b) const override {
        double lo=std::min(evaluate(b.minimum).value,evaluate(b.maximum).value);
        if(b.minimum.z()<=1.5&&b.maximum.z()>=1.5) lo=std::min(lo,-0.8);
        return {lo,1.25/0.75,true};
    }
    void appendRayBreakpoints(const V &o,const V &w,double lo,double hi,
                              std::vector<double> &out) const override {
        if(w.z()==0) return;
        for(double knot:{0.5,1.5,2.25}) {
            const double t=(knot-o.z())/w.z();
            if(t>lo&&t<hi) out.push_back(t);
        }
    }
};

// Smooth on-ray mean reconstructed from the rare crossing observed with
// ell=.2, sigma=.02. In normalized coordinates M(0)=0, M'(0)=1,
// M(U)=-.94512370494674536, M(Q)=0 and M'(Q)=2.9618342702559604.
class RareTailMean final : public cg::MeanField {
    double _a=-28.800854648826082,_b=67.11665172099852,_c=-40.55016060381338;
public:
    static constexpr double Ell=.2,Sigma=.02,Q=.737844290607805,U=.43042090545584255;
    explicit RareTailMean(double historyShift=0) {
        // x^2(x-Q)^2 changes only the interior mean, preserving the birth
        // value/slope and the current zero value/slope.
        const double scale=historyShift/(U*U*(U-Q)*(U-Q));
        _a+=scale*Q*Q;_b-=2*scale*Q;_c+=scale;
    }
    const char *typeName() const override {return "single-history-rare-tail-profile";}
    cg::MeanJet evaluate(const V &p) const override {
        const double x=p.z()/Ell;
        return {Sigma*x*(1+x*(_a+x*(_b+x*_c))),
                V(0,0,Sigma/Ell*(1+x*(2*_a+x*(3*_b+4*x*_c))))};
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &bounds) const override {
        const double r=std::max(std::abs(bounds.minimum.z()),std::abs(bounds.maximum.z()))/Ell;
        const double magnitude=Sigma*r*(1+r*(std::abs(_a)+r*(std::abs(_b)+r*std::abs(_c))));
        const double slope=Sigma/Ell*(1+r*(2*std::abs(_a)+r*(3*std::abs(_b)+4*r*std::abs(_c))));
        return {-magnitude,slope,true};
    }
};

void rareHistoryTailChecks() {
    const auto mean=std::make_shared<RareTailMean>();
    const auto f=field(mean,RareTailMean::Sigma,25*M::Identity());
    sh::Start start;start.kind=sh::Start::SurfaceValueGradient;
    start.gradient=mean->evaluate(V::Zero()).gradient;
    const double t=RareTailMean::Ell*RareTailMean::Q,u=RareTailMean::Ell*RareTailMean::U;
    close(mean->evaluate(V(0,0,u)).value/RareTailMean::Sigma,-.94512370494674536,1e-12,
          "rare-tail history point lies in the retained negative valley");
    close(mean->evaluate(V(0,0,t)).value/RareTailMean::Sigma,0,1e-12,
          "rare-tail query is on the mean zero surface");

    // Independent upper bound on the positive-speed moment: condition on
    // U>0 and bound E[V_+|U] by |conditional mean|+conditional sd/sqrt(2pi).
    // These joint conditional statistics are the reconstructed SE fixture.
    const double mu=-2.9618342702559604,muU=-.94512370494674536;
    const double variance=.041259974169068535,varianceU=.00052522140094730073;
    const double cross=.0045985963076722522,sdU=std::sqrt(varianceU);
    const double conditionalSd=std::sqrt(variance-cross*cross/varianceU);
    const double logP=cg::normalLogCdf(muU/sdU);
    const double terms[]={std::log(std::abs(mu))+logP,
        std::log(std::abs(cross)/sdU)+cg::normalLogPdf(muU/sdU),
        std::log(conditionalSd)-.5*std::log(2*cg::kPi)+logP};
    const double largest=*std::max_element(terms,terms+3);
    double scaled=0;for(double term:terms) scaled+=std::exp(term-largest);
    const double logMomentBound=largest+std::log(scaled);
    const auto varianceAt=[](double q) {return 1-(1+q*q)*std::exp(-q*q);};
    // Surface SE values at two positive ages are positively correlated, so
    // P(F_t>0,U>0) >= P(F_t>0)P(U>0). This yields a conservative hazard bound.
    const double logDenominatorLower=std::log(.5)+
        cg::normalLogCdf(muU/std::sqrt(varianceAt(RareTailMean::U)));
    const double logHazardBound=logMomentBound-.5*std::log(2*cg::kPi*varianceAt(RareTailMean::Q))-
        std::log(RareTailMean::Ell)-logDenominatorLower;
    require(logHazardBound<std::log(std::numeric_limits<double>::denorm_min())-20,
            "rare-tail physical extinction is independently bounded below double representability");
    const sh::LocalLaw rare(f,start,t,{u});
    close(rare.hazard(),0,0,"subnormal scaled flux terminates safely instead of exhausting tail expansion");

    // Moving the same valley upward leaves a very small but representable
    // crossing rate. The tail fix must not replace all rare fluxes by zero.
    const auto shallowerMean=std::make_shared<RareTailMean>(.34512370494674536);
    const auto shallower=field(shallowerMean,RareTailMean::Sigma,25*M::Identity());
    const double rate=sh::LocalLaw(shallower,start,t,{u}).hazard();
    require(rate>0&&std::isfinite(rate),"representable rare crossing flux remains positive");
    std::cout<<"Subnormal sparse-history flux tail regression passed\n";
}

void referenceChecks() {
    const auto f=field(std::make_shared<cg::ConstantMean>(0));
    sh::Start camera;
    const sh::LocalLaw startSign(f,camera,1,{});
    // Independent zero-mean, one-origin-sign fixture; it differs from the
    // unconditioned stationary intensity 1/pi even without an interior point.
    close(startSign.hazard(),0.39659956942467517,2e-6,
          "first hit retains the positive origin sign");
    require(std::abs(startSign.hazard()-1/cg::kPi)>.05,
            "first hit is not the unconditioned stationary baseline");
    camera.gradient=V(9,-8,-7);
    close(sh::LocalLaw(f,camera,1,{}).hazard(),startSign.hazard(),1e-12,
          "first-hit origin gradient is unknown and integrated out");
    const auto profile=field(std::make_shared<ProfileMean>());
    sh::Start surface;surface.kind=sh::Start::SurfaceValueGradient;
    surface.gradient=V::UnitZ();
    close(sh::LocalLaw(profile,surface,2.625,{1.5}).hazard(),.38547,2e-5,
          "single-history surface crossing fixture");
    std::cout<<"Independent first-hit and surface fixtures passed\n";
}

void localChecks() {
    referenceChecks();
    rareHistoryTailChecks();
    M metric;metric<<1.4,.15,.1, .15,.8,-.1, .1,-.1,1.1;
    const auto f=field(std::make_shared<cg::PlaneMean>(V(.2,.1,1),.3),.7,metric);
    for(bool surface:{false,true}) {
        sh::Start start;start.direction=V(.1,-.2,1).normalized();
        if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V(.2,-.1,.7);
        one::Flight flight(f,start);
        for(double t:{.02,.4,.9,1.6,2.6}) {
            const auto history=flight.selectHistory(t);
            const auto law=flight.law(t);
            const sh::LocalLaw reference(f,start,t,history);
            close(law.hazard(),reference.hazard(),1e-10*(1+reference.hazard()),
                  "new flight evaluates direct positive-domain local law");
            close(law.speedCdf(.6),reference.speedCdf(.6),1e-10,
                  "distance and crossing speed use identical selected history");
        }
        if(!surface) {
            auto changed=start;changed.gradient=V(-12,9,-11);
            one::Flight original(f,start),unknownGradient(f,changed);
            for(double t:{.2,.7,1.3,2.1}) {
                require(original.selectHistory(t)==unknownGradient.selectHistory(t),
                        "first-hit selector cannot observe a sampled origin gradient");
                close(original.law(t).hazard(),unknownGradient.law(t).hazard(),1e-11,
                      "first hit marginalizes the unknown gradient");
            }
        } else {
            one::Settings settings;settings.historyEnabled=false;
            one::Flight baseline(f,start,settings);
            cg::FlightState birth;birth.birthPosition=start.origin;
            birth.direction=start.direction;birth.birthGradient=start.gradient;
            const cg::ConditionalFlightKernel reference(f,birth);
            for(double t:{.002,.03,.2,.8,1.7,3.}) {
                const double expected=reference.evaluate(t).hazard.value;
                close(baseline.law(t).hazard(),expected,2e-6*(1+expected),
                      "surface K0 reduces to value-gradient conditional baseline");
            }
        }
    }
    std::cout<<"First-hit, surface and anisotropic local-law checks passed\n";
}

void scoreChecks() {
    const auto f=field(std::make_shared<cg::ConstantMean>(0));
    sh::Start start;
    const double expected=(2/cg::kPi)*std::exp(-1)*(2-std::exp(-1));
    close(sh::historyCandidateStatistics(f,start,6,5).score,expected,2e-12,
          "camera score matches the stationary closed expression");
    require(sh::historyCandidateStatistics(f,start,1,std::nextafter(1.,0.)).score<1e-12,
            "nearly coincident history has vanishing score");
    require(sh::historyCandidateStatistics(f,start,30,1).score<1e-20,
            "distant uncorrelated history has vanishing score");
    M metric=M::Identity();metric(0,0)=4;metric(1,1)=.5;metric(2,2)=.25;
    const auto anisotropic=field(std::make_shared<cg::ConstantMean>(0),1,metric);
    close(sh::historyCandidateStatistics(anisotropic,start,12,10).score,expected,2e-12,
          "score uses directional correlation length in physical coordinates");
    start.kind=sh::Start::SurfaceValueGradient;start.gradient=V::UnitZ();
    for(double t:{1e-6,1e-4,.1,1.,20.}) {
        const auto score=sh::historyCandidateStatistics(f,start,t,.5*t);
        require(std::isfinite(score.score)&&score.score>=0&&std::isfinite(score.standardizedMean),
                "surface score remains finite near the conditional birth");
    }
    close(sh::historyCandidateStatistics(f,start,20,19).score,expected,2e-12,
          "surface score converges to stationary form far from birth");
}

void cachedScoreChecks() {
    M metric;metric<<2,.2,-.1, .2,.8,.15, -.1,.15,.4;
    const auto isotropic=field(std::make_shared<cg::ConstantMean>(0));
    const auto anisotropic=field(std::make_shared<cg::PlaneMean>(V(.2,-.1,.6),-.3),.7,metric);
    for(const auto *f:{&isotropic,&anisotropic}) for(bool surface:{false,true}) {
        sh::Start start;start.origin=V(.07,-.03,.02);
        start.direction=V(.2,-.3,1).normalized();
        if(surface) start.kind=sh::Start::SurfaceValueGradient;
        // A small outgoing slope keeps the near-birth score nonzero, so the
        // test exercises the stable projection rather than only zero tails.
        start.gradient=1e-8*start.direction;
        const double inverseLength=std::sqrt(start.direction.dot(f->kernel.precision()*start.direction));
        for(double q:{1e-6,1e-4,.05,.5,1.,3.,7.99,8.,8.01,12.,30.,50.}) {
            const double t=q/inverseLength;
            const sh::HistoryCandidateScorer scorer(t*inverseLength,surface);
            const std::vector<double> positions={q*.01,q*.25,q*.5,q*.875,
                q-std::min(1.,q*.25),q*(1-1e-9),std::nextafter(q,0.)};
            for(double u:positions) {
                const double h=u/inverseLength;
                // Physical-to-normalized rounding may collapse one ULP.
                if(!(h<t && h*inverseLength<t*inverseLength)) continue;
                const auto reference=sh::historyCandidateStatistics(*f,start,t,h);
                const double z=sh::standardizedMean(*f,start,h);
                const double mills=cg::normalPdfOverCdf(z);
                const double actual=scorer.score(h*inverseLength,mills);
                require(std::isfinite(actual)&&actual>=0,"cached score is finite and nonnegative");
                // The reference subtracts nearly parallel feature vectors.
                // An O(eps) residual error becomes O(eps*sqrt(score)*Mills)
                // in its squared score; relative error alone is unsuitable
                // as u approaches q. Check the camera's closed form below
                // separately, so this allowance cannot hide score drift.
                const double eps=std::numeric_limits<double>::epsilon();
                const double projectionError=64*eps*mills*(std::sqrt(reference.score)+eps*mills);
                close(actual,reference.score,2e-9*std::max(reference.score,1e-16)+projectionError,
                      "cached scorer preserves the original selection score");
                if(!surface) {
                    const long double d=t*inverseLength-h*inverseLength;
                    const long double d2=d*d;
                    const double exact=double(static_cast<long double>(mills)*mills*
                        std::exp(-d2)*(-std::expm1(-d2)+d2));
                    close(actual,exact,2e-13*std::max(exact,1e-300),
                          "cached camera score agrees with the independent stationary closed form");
                }
            }
        }
    }
}

void cachedCandidateMeanChecks() {
    // An affine mean makes origin work observable without a timing assertion.
    // Its cancellation-safe valueDifference does not re-query the origin.
    class CountingMean final : public cg::MeanField {
        V _origin;
    public:
        mutable unsigned originQueries=0;
        explicit CountingMean(V origin):_origin(std::move(origin)) {}
        const char *typeName() const override { return "candidate-origin-counting-plane"; }
        cg::MeanJet evaluate(const V &p) const override {
            if(p==_origin) ++originQueries;
            const V gradient(.2,-.1,.6);
            return {.35+p.dot(gradient),gradient};
        }
        std::optional<V> affineGradient() const override { return V(.2,-.1,.6); }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {-100,1,true}; }
    };
    M metric;metric<<2,.2,-.1, .2,.8,.15, -.1,.15,.4;
    for(bool surface:{false,true}) {
        sh::Start start;start.origin=V(.07,-.03,.02);
        start.direction=V(.2,-.3,1).normalized();start.gradient=.4*start.direction;
        if(surface) start.kind=sh::Start::SurfaceValueGradient;
        const auto mean=std::make_shared<CountingMean>(start.origin);
        const auto f=field(mean,.7,metric);
        // Repeated and backwards requests include the small-variance birth
        // regime, normal distances and the far-field covariance branch.
        const std::vector<double> positions={1e-12,1e-6,.001,.07,.071,1.,.07,20.,.001,0};
        std::vector<double> expected;
        for(double t:positions) expected.push_back(sh::standardizedMean(f,start,t));
        mean->originQueries=0;
        const sh::HistoryCandidateEvaluator cached(f,start);
        require(mean->originQueries==1,"candidate evaluator initializes the origin once");
        for(std::size_t i=0;i+1<positions.size();++i)
            require(cached.standardizedMean(positions[i])==expected[i],
                    "cached analytic candidate heights exactly match uncached conditioning");
        require(mean->originQueries==1,"candidate queries reuse immutable flight origin statistics");
        require(cached.standardizedMean(0)==expected.back(),"cached candidate handles the birth limit");

        auto other=start;other.origin=V(-.31,.27,.12);
        other.direction=V(-.2,.1,1).normalized();other.gradient=.2*other.direction;
        const sh::HistoryCandidateEvaluator second(f,other);
        for(double t:{.002,.13,.002,2.}) {
            require(second.standardizedMean(t)==sh::standardizedMean(f,other,t),
                    "independent candidate evaluator uses its own flight origin");
            require(cached.standardizedMean(t)==sh::standardizedMean(f,start,t),
                    "interleaved flights cannot mutate cached origin statistics");
        }
    }
}

void checkSelectionReference(const cg::GPSSField &f,const sh::Start &start,
                             const one::Settings &settings,const one::Flight &flight) {
    const auto &snapshot=flight.snapshot();
    const double t=double(snapshot.group)*(flight.lengthScale()*settings.selectionStepOverLength);
    std::optional<double> selected;
    double best=-1;
    auto score=[&](double u) {
        if(!(u>0&&u<t)) return;
        const double value=sh::historyCandidateStatistics(f,start,t,u).score;
        if(value>best || (value==best&&(!selected||u<*selected))) {
            best=value;selected=u;
        }
    };
    auto checkCandidate=[&](const one::Candidate &candidate) {
        const double z=sh::standardizedMean(f,start,candidate.distance);
        close(candidate.z,z,2e-10*(1+std::abs(z)),"cached candidate height matches uncached conditioning");
        close(candidate.inverseMills,cg::normalPdfOverCdf(z),2e-10*(1+cg::normalPdfOverCdf(z)),
              "candidate retains the same inverse Mills ratio");
    };
    for(const auto &candidate:snapshot.completed) {
        checkCandidate(candidate);score(candidate.distance);
    }
    if(snapshot.current) checkCandidate(*snapshot.current);
    if(t>0) score(t-std::min(flight.lengthScale(),t/2));
    require(snapshot.selected==selected,"cached selector chooses the same point as original score oracle");
}

void cachedSelectionChecks() {
    M metric;metric<<2,.2,-.1, .2,.8,.15, -.1,.15,.4;
    const auto profile=field(std::make_shared<ProfileMean>());
    const auto anisotropic=field(std::make_shared<cg::PlaneMean>(V(.2,-.1,.6),-.3),.7,metric);
    for(const auto *f:{&profile,&anisotropic}) for(bool surface:{false,true}) {
        sh::Start start;start.origin=V(.07,-.03,.02);
        start.direction=V(.2,-.3,1).normalized();start.gradient=.4*start.direction;
        if(surface) start.kind=sh::Start::SurfaceValueGradient;
        one::Settings settings;one::Flight flight(*f,start,settings);
        const double step=settings.selectionStepOverLength*flight.lengthScale();
        // Includes cache eviction, conditional near-field and far-field score
        // branches, while checking the exact tie rule at every group boundary.
        for(int group=1;group<=32;++group) {
            flight.selectHistory((group+.01)*step);
            checkSelectionReference(*f,start,settings,flight);
        }
    }
    const auto sphere=field(std::make_shared<cg::SphereMean>(V::Zero(),1));
    sh::Start camera;camera.origin=V(0,0,-1.3);
    one::Flight crossingCenter(sphere,camera);
    crossingCenter.selectHistory(1.51);
    checkSelectionReference(sphere,camera,{},crossingCenter);
    require(std::any_of(crossingCenter.snapshot().completed.begin(),crossingCenter.snapshot().completed.end(),
        [](const one::Candidate &c){return c.distance==1.3;}),
        "camera candidate at sphere center reads the finite mean without requiring its gradient");
}

void compareCandidates(const one::Snapshot &,const one::Snapshot &);

void gridChecks() {
#if OPENVDB_AVAILABLE
    auto grid=std::make_shared<Tungsten::GpisSdfGrid>();
    grid->data.origin=Tungsten::Vec3i(0);grid->data.dimensions=Tungsten::Vec3i(3);
    grid->data.clampMin=Tungsten::Vec3f(0.f);grid->data.clampMax=Tungsten::Vec3f(1.f);
    grid->worldToIndex=M::Identity();grid->translation=V::Zero();grid->scale=1;grid->offset=0;
    grid->data.values.resize(27);
    for(int z=0;z<3;++z) for(int y=0;y<3;++y) for(int x=0;x<3;++x)
        grid->data.values[(z*3+y)*3+x]=float(64*x*y*z-36*(x*y+x*z+y*z)+14*(x+y+z)+3);
    grid->initializeMagnitudeBounds();
    // The ray's cubic has a hidden minimum at 7/8 of the voxel diagonal;
    // both voxel endpoints are positive and both endpoint slopes positive.
    const double trough=7*std::sqrt(3.)/8;
    const auto f=field(grid);sh::Start start;start.direction=V::Ones().normalized();
    one::Flight coarse(f,start),fine(f,start);
    coarse.selectHistory(2.01);
    for(double t:{.13,.49,.51,.87,1.13,1.499,1.52,1.7,1.93,2.01}) fine.selectHistory(t);
    compareCandidates(coarse.snapshot(),fine.snapshot());
    checkSelectionReference(f,start,{},coarse);
    require(std::any_of(coarse.snapshot().completed.begin(),coarse.snapshot().completed.end(),
            [&](const one::Candidate &c){return std::abs(c.distance-trough)<2e-12;}),
            "DDA candidate cache discovers the hidden cubic trough");
    auto surface=start;surface.kind=sh::Start::SurfaceValueGradient;
    surface.gradient=.3*surface.direction;
    one::Flight conditioned(f,surface);
    for(double t:{.51,1.01,1.51,2.01}) {
        conditioned.selectHistory(t);
        checkSelectionReference(f,surface,{},conditioned);
    }

    // The real grid cursor anchors a polynomial in the first evaluated cell.
    // Reusing a mutated cursor can subtly change rounding even within a cell;
    // a cached evaluator must reproduce a freshly origin-initialized cursor.
    class CountingGrid final : public Tungsten::GpisSdfGrid {
    public:
        mutable unsigned cursorCreates=0;
        std::unique_ptr<cg::MeanRayCursor> createRayCursor(const V &o,const V &w) const override {
            ++cursorCreates;return GpisSdfGrid::createRayCursor(o,w);
        }
    };
    auto counted=std::make_shared<CountingGrid>();
    static_cast<Tungsten::GpisSdfGrid &>(*counted)=*grid;
    counted->data.clampMin=Tungsten::Vec3f(.125f);
    counted->data.clampMax=Tungsten::Vec3f(1.875f);
    M metric;metric<<2,.2,-.1, .2,.8,.15, -.1,.15,.4;
    const auto gridField=field(counted,.7,metric);
    for(bool surface:{false,true}) for(bool reverse:{false,true}) {
        sh::Start birth;birth.origin=reverse?V(.4,.5,1.75):V(.4,.5,.25);
        birth.direction=V::UnitZ();if(reverse) birth.direction=-birth.direction;
        birth.gradient=.3*birth.direction;
        if(surface) birth.kind=sh::Start::SurfaceValueGradient;
        const std::vector<double> positions={1e-12,.01,.011,.02,.75,
            std::nextafter(.75,0.),std::nextafter(.75,1.),1.5,1.625,1.7,2.,.01,.75,0};
        std::vector<double> expected;
        for(double t:positions) expected.push_back(sh::standardizedMean(gridField,birth,t));
        counted->cursorCreates=0;
        const sh::HistoryCandidateEvaluator cached(gridField,birth);
        for(std::size_t i=0;i<positions.size();++i)
            require(cached.standardizedMean(positions[i])==expected[i],
                    "cached grid candidate is bit-identical across cells, clamp and query order");
        require(counted->cursorCreates==1,"grid candidate queries clone the initialized cursor without rebuilding it");
    }
#endif
}

void compareCandidates(const one::Snapshot &a,const one::Snapshot &b) {
    require(a.group==b.group&&a.selected==b.selected&&a.completed.size()==b.completed.size(),
            "same prefix has the same fixed-bin history");
    close(a.fragmentBegin,b.fragmentBegin,1e-12,"geometry cursor uses canonical fragment boundary");
    require(bool(a.current)==bool(b.current),"current candidate presence is partition independent");
    if(a.current) {
        close(a.current->distance,b.current->distance,1e-12,"current candidate position");
        close(a.current->z,b.current->z,1e-11,"current candidate standardized mean");
        close(a.current->inverseMills,b.current->inverseMills,1e-11,"current candidate cached inverse Mills ratio");
    }
    for(size_t i=0;i<a.completed.size();++i) {
        close(a.completed[i].distance,b.completed[i].distance,1e-12,"completed candidate position");
        close(a.completed[i].z,b.completed[i].z,1e-11,"completed candidate standardized mean");
        close(a.completed[i].inverseMills,b.completed[i].inverseMills,1e-11,
              "completed candidate cached inverse Mills ratio");
    }
}

void selectorChecks() {
    scoreChecks();
    cachedScoreChecks();
    cachedCandidateMeanChecks();
    cachedSelectionChecks();
    const auto f=field(std::make_shared<ProfileMean>());
    sh::Start start;start.kind=sh::Start::SurfaceValueGradient;start.gradient=V::UnitZ();
    one::Flight flight(f,start);
    require(flight.selectHistory(.49).empty(),"first half-length group has no interior history");
    for(int group=1;group<8;++group) {
        const double begin=group*.5;
        const auto selected=flight.selectHistory(begin+1e-6);
        require(selected.size()==1&&selected[0]>0&&selected[0]<begin,
                "selected history is strictly before current group");
        require(flight.selectHistory(begin+.13)==selected&&
                flight.selectHistory(begin+.49)==selected,
                "selected coordinate remains fixed for the full half-length group");
    }
    one::Flight direct(f,start),partitioned(f,start);
    direct.selectHistory(3.73);
    for(double t:{.11,.27,.499,.51,.83,1.18,1.52,1.97,2.28,2.81,3.33,3.73})
        partitioned.selectHistory(t);
    compareCandidates(direct.snapshot(),partitioned.snapshot());
    one::Flight prefix(f,start);prefix.selectHistory(1.23);
    const auto saved=prefix.snapshot();
    one::Flight resumed(f,start,{},saved);resumed.selectHistory(3.73);
    compareCandidates(direct.snapshot(),resumed.snapshot());
    require(saved.age==1.23,"resuming a snapshot cannot mutate the saved branch");

    // Ratio tracking visits random candidate locations, but those cannot
    // become geometry/history candidates. All variants end at the same prefix.
    const auto flat=field(std::make_shared<cg::ConstantMean>(.4));
    sh::Start camera;one::Settings settings;settings.shadowRouletteThreshold=0;
    one::Flight deterministic(flat,camera,settings);deterministic.selectHistory(2.73);
    for(int seed:{17,29,41}) {
        cg::Random rng(seed);one::Flight tracked(flat,camera,settings);
        tracked.transmittance(2.73,rng);
        compareCandidates(deterministic.snapshot(),tracked.snapshot());
    }
    cg::Random rng(751);one::Flight split(flat,camera,settings);
    split.transmittance(.37,rng);split.transmittance(1.23,rng);
    one::Flight restored(flat,camera,settings,split.snapshot());
    restored.transmittance(2.73,rng);
    compareCandidates(deterministic.snapshot(),restored.snapshot());
    gridChecks();
    std::cout<<"Fixed groups, causal geometry and continuation checks passed\n";
}

double opticalReference(const cg::GPSSField &f,const sh::Start &start,
                        const one::Settings &settings,double end,int count) {
    one::Flight flight(f,start,settings);
    double sum=0;
    for(int i=0;i<count;++i) sum+=flight.law(end*(i+.5)/count).hazard();
    return end*sum/count;
}

double groupedOpticalReference(const cg::GPSSField &f,const sh::Start &start,
                               const one::Settings &settings,double end,int perGroup) {
    one::Flight flight(f,start,settings);
    const double groupStep=flight.lengthScale()*settings.selectionStepOverLength;
    double opticalDepth=0;
    // Independent composite midpoint integration splits every model change;
    // its nodes never become history candidates.
    for(int group=0;group*groupStep<end;++group) {
        const double a=group*groupStep,b=std::min(end,(group+1)*groupStep);
        double sum=0;
        for(int i=0;i<perGroup;++i) sum+=flight.law(a+(b-a)*(i+.5)/perGroup).hazard();
        opticalDepth+=(b-a)*sum/perGroup;
    }
    return opticalDepth;
}

struct RegularReferencePiece { double begin,end,rate; };

std::vector<RegularReferencePiece> regularReference(const cg::GPSSField &f,const sh::Start &start,
                                                    int subdivisions,double end,bool profile=false) {
    // Test fields have unit directional length. These canonical boundaries
    // come from their known geometry plus fixed half-length selection bins,
    // independent of the transport implementation and renderer segment ends.
    std::vector<double> boundaries{0};
    for(int group=1;boundaries.back()<end;++group) boundaries.push_back(group*.5);
    const double lastBoundary=boundaries.back();
    if(profile) for(double knot:{.5,1.5,2.25})
        if(knot<lastBoundary) boundaries.push_back(knot);
    std::sort(boundaries.begin(),boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(),boundaries.end()),boundaries.end());
    one::Flight selector(f,start);
    std::vector<RegularReferencePiece> result;
    for(std::size_t cell=1;cell<boundaries.size();++cell) {
        const double begin=boundaries[cell-1],width=boundaries[cell]-begin;
        for(int part=0;part<subdivisions;++part) {
            const double a=begin+width*part/subdivisions,b=begin+width*(part+1)/subdivisions;
            const double midpoint=a+.5*(b-a);
            result.push_back({a,b,sh::LocalLaw(f,start,midpoint,selector.selectHistory(midpoint)).hazard()});
        }
    }
    return result;
}

double regularReferenceDepth(const std::vector<RegularReferencePiece> &pieces,double end) {
    double result=0;
    for(const auto &piece:pieces)
        if(end>piece.begin) result+=piece.rate*(std::min(end,piece.end)-piece.begin);
    return result;
}

double regularReferenceDistance(const std::vector<RegularReferencePiece> &pieces,double depth) {
    for(const auto &piece:pieces) {
        const double localDepth=piece.rate*(piece.end-piece.begin);
        if(piece.rate>0&&depth<=localDepth) return piece.begin+depth/piece.rate;
        depth-=localDepth;
    }
    throw std::runtime_error("regular reference threshold does not collide");
}

void regularChecks() {
    const auto flat=field(std::make_shared<cg::ConstantMean>(.4));
    const auto profile=field(std::make_shared<ProfileMean>());
    constexpr double end=2.73,target=.2;
    for(const auto *f:{&flat,&profile}) for(bool surface:{false,true}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        double convergence[3]{};
        const double continuousReference=f==&flat ?
            std::exp(-groupedOpticalReference(*f,start,{},end,64)):0;
        int index=0;
        for(int subdivisions:{1,2,4}) {
            one::Settings settings;settings.trackingMethod=one::TrackingMethod::Regular;
            settings.regularSubdivisions=subdivisions;settings.shadowRouletteThreshold=0;
            const auto reference=regularReference(*f,start,subdivisions,end,f==&profile);
            const double expectedDepth=regularReferenceDepth(reference,end);
            cg::Random rng(1381+subdivisions);
            one::Flight shadow(*f,start,settings);
            const double transmittance=shadow.transmittance(end,rng);
            close(transmittance,std::exp(-expectedDepth),3e-12,
                  "regular shadow equals the independently specified canonical midpoint model");
            one::Flight geometry(*f,start);geometry.selectHistory(end);
            compareCandidates(shadow.snapshot(),geometry.snapshot());
            require(shadow.work().hazardEvaluations>0,"regular backend reports its midpoint hazard queries");
            one::Snapshot initial;initial.inversionRemaining=target;
            one::Flight distance(*f,start,settings,initial);
            const auto hit=distance.sampleDistance(end,rng);
            require(hit.hit&&hit.law,"regular inversion returns a collision and its local law");
            close(hit.distance,regularReferenceDistance(reference,target),3e-11,
                  "regular collision distance inverts the piecewise constant optical depth");
            const auto history=distance.snapshot().selected;
            const sh::LocalLaw actual(*f,start,hit.distance,
                history?std::vector<double>{*history}:std::vector<double>{});
            close(hit.law->hazard(),actual.hazard(),1e-11*(1+actual.hazard()),
                  "regular collision gradient law is evaluated at the actual position with the same history");
            close(actual.speedCdf(hit.law->sampleSpeed(.5)),.5,3e-4,
                  "regular collision speed preserves its selected-history distribution");

            // A boundary inside a regular piece cannot move that piece's
            // midpoint to the shortened renderer segment.
            one::Snapshot longThreshold;longThreshold.inversionRemaining=expectedDepth+1;
            one::Flight first(*f,start,settings,longThreshold);
            require(!first.sampleDistance(.17,rng).hit,"regular prefix remains below its collision threshold");
            const auto saved=first.snapshot();
            close(saved.inversionRemaining,longThreshold.inversionRemaining-regularReferenceDepth(reference,.17),
                  3e-12,"regular prefix uses the complete canonical piece's rate");
            one::Flight resumed(*f,start,settings,saved);
            for(double boundary:{.33,.52,1.12,end})
                require(!resumed.sampleDistance(boundary,rng).hit,"regular transparent continuation escapes");
            close(resumed.snapshot().inversionRemaining,1,3e-11,
                  "regular transparent cuts preserve the original optical-depth threshold");
            compareCandidates(resumed.snapshot(),geometry.snapshot());
            require(saved.inversionRemaining==first.snapshot().inversionRemaining,
                    "regular continuation leaves its saved branch unchanged");
            if(f==&flat) {
                convergence[index++]=std::abs(transmittance-continuousReference);
            }
        }
        if(f==&flat) {
            require(convergence[2]<=convergence[0]+2e-4,
                    "four regular subdivisions improve this smooth fixture's transmission error");
            std::cout<<(surface?"surface":"first-hit")<<" regular T errors N=1/2/4: "
                     <<convergence[0]<<", "<<convergence[1]<<", "<<convergence[2]<<"\n";
        }
    }
    // A coarse midpoint proposal can land in an extremely rare early return.
    // The zero-value density underflows there, but the conditional crossing
    // speed still has a well-defined positive mass. It must remain sampleable.
    sh::Start rare;rare.kind=sh::Start::SurfaceValueGradient;rare.gradient=V::UnitZ();
    one::Settings rareSettings;rareSettings.trackingMethod=one::TrackingMethod::Regular;
    constexpr double rareDistance=1e-5;
    const double midpointRate=sh::LocalLaw(flat,rare,.25,{}).hazard();
    require(midpointRate>0,"regular rare-return fixture has positive canonical midpoint rate");
    one::Snapshot rareThreshold;rareThreshold.inversionRemaining=midpointRate*rareDistance;
    one::Flight rareFlight(flat,rare,rareSettings,rareThreshold);
    cg::Random rareRandom(7831);
    const auto rareHit=rareFlight.sampleDistance(.1,rareRandom);
    require(rareHit.hit&&rareHit.law,"regular retains a collision whose zero-density factor underflows");
    close(rareHit.distance,rareDistance,1e-15,"rare regular return uses the midpoint optical-depth inverse");
    require(rareHit.law->hazard()==0,"fixture isolates density underflow from the conditional speed law");
    const double speed=rareHit.law->sampleSpeed(rareRandom);
    require(std::isfinite(speed)&&speed>.99&&speed<1.01,
            "density underflow does not erase the normalized positive crossing-speed distribution");
    close(rareHit.law->speedCdf(std::numeric_limits<double>::infinity()),1,0,
          "rare-return conditional speed law still normalizes");
    // Regular tracking intentionally has no adaptive near-birth guarantee:
    // a grazing return peak may be narrower than one canonical midpoint cell.
    std::cout<<"Regular canonical-rate, distance, speed and continuation checks passed\n";
}

void inversionChecks() {
    const auto f=field(std::make_shared<cg::ConstantMean>(.4));
    constexpr double end=2.5,target=.2;
    for(bool surface:{false,true}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        one::Settings fine;fine.trackingMethod=one::TrackingMethod::Inversion;
        fine.shadowRouletteThreshold=0;fine.profile=true;
        fine.inversionMaxStepOverLength=.125;
        fine.inversionAbsoluteTolerance=1e-7;fine.inversionRelativeTolerance=2e-5;
        fine.inversionDistanceToleranceOverLength=2e-6;
        auto coarse=fine;coarse.inversionMaxStepOverLength=.5;
        coarse.inversionAbsoluteTolerance=3e-5;coarse.inversionRelativeTolerance=2e-3;
        coarse.inversionDistanceToleranceOverLength=1e-4;
        const double low=groupedOpticalReference(f,start,fine,end,32);
        const double reference=groupedOpticalReference(f,start,fine,end,64);
        close(low,reference,2e-4,"independent fixed-history midpoint reference converges");
        require(reference>target,"inversion fixture contains a collision at the fixed optical depth");
        cg::Random rng(surface?983:457);
        one::Flight shadow(f,start,fine),roughShadow(f,start,coarse),deterministic(f,start);
        const double transmittance=shadow.transmittance(end,rng);
        close(transmittance,std::exp(-reference),3e-4,"inversion shadow integrates the same first-hit or surface hazard");
        close(roughShadow.transmittance(end,rng),transmittance,5e-4,
              "inversion shadow converges as integration tolerances tighten");
        deterministic.selectHistory(end);
        compareCandidates(shadow.snapshot(),deterministic.snapshot());
        compareCandidates(roughShadow.snapshot(),deterministic.snapshot());
        require(shadow.work().hazardEvaluations>0,"inversion reports actual hazard queries");

        one::Snapshot initial;initial.inversionRemaining=target;
        one::Flight direct(f,start,fine,initial);
        const auto hit=direct.sampleDistance(end,rng);
        require(hit.hit&&bool(hit.law),"optical-depth inversion returns its collision law");
        const double hitDepth=groupedOpticalReference(f,start,fine,hit.distance,64);
        close(hitDepth,target,3e-4,"inverted distance meets an independent optical-depth target");
        const auto selected=direct.snapshot().selected;
        const sh::LocalLaw local(f,start,hit.distance,
            selected?std::vector<double>{*selected}:std::vector<double>{});
        close(hit.law->hazard(),local.hazard(),1e-10,"inversion freezes the same selected collision history");
        close(local.speedCdf(hit.law->sampleSpeed(.5)),.5,3e-4,
              "inversion collision speed uses the distance model");

        one::Flight prefix(f,start,fine,initial);
        const auto first=prefix.sampleDistance(.2,rng);
        require(!first.hit,"transparent-prefix fixture escapes its first segment");
        const auto saved=prefix.snapshot();
        const double prefixDepth=groupedOpticalReference(f,start,fine,.2,64);
        close(saved.inversionRemaining,target-prefixDepth,2e-5,
              "transparent continuation preserves remaining true optical depth");
        one::Flight restored(f,start,fine,saved);
        const auto resumed=restored.sampleDistance(end,rng);
        require(resumed.hit,"continued inversion still reaches the original collision threshold");
        close(resumed.distance,hit.distance,3e-5,
              "splitting a straight flight preserves the inverted collision distance");
        require(saved.inversionRemaining==prefix.snapshot().inversionRemaining,
                "inversion continuation cannot mutate its saved snapshot");
    }
    // A grazing birth has a narrow first-crossing peak at t ~ g0*ell^2/sigma.
    // Uniform midpoint reference nodes would miss the same peak as an unsafe
    // initial quadrature step, so integrate independently in log distance.
    const auto grazingField=field(std::make_shared<cg::ConstantMean>(0));
    sh::Start grazing;grazing.kind=sh::Start::SurfaceValueGradient;
    constexpr double beta=1e-6,grazingEnd=.2;
    grazing.gradient=beta*V::UnitZ();
    one::Settings fine;fine.trackingMethod=one::TrackingMethod::Inversion;
    fine.shadowRouletteThreshold=0;fine.inversionMaxStepOverLength=.125;
    fine.inversionAbsoluteTolerance=1e-7;fine.inversionRelativeTolerance=2e-5;
    fine.inversionDistanceToleranceOverLength=1e-10;
    const auto logarithmicDepth=[&](double end,int count) {
        const double lo=std::log(beta/256),hi=std::log(end),step=(hi-lo)/count;
        one::Flight reference(grazingField,grazing,fine);
        double sum=0;
        for(int i=0;i<count;++i) {
            const double t=std::exp(lo+(i+.5)*step);
            sum+=t*reference.law(t).hazard();
        }
        return step*sum;
    };
    const double low=logarithmicDepth(grazingEnd,256),reference=logarithmicDepth(grazingEnd,512);
    close(low,reference,2e-4,"log-distance grazing reference resolves the near-birth crossing peak");
    cg::Random grazingRandom(547);
    one::Flight fineGrazing(grazingField,grazing,fine);
    const double fineTr=fineGrazing.transmittance(grazingEnd,grazingRandom);
    close(fineTr,std::exp(-reference),3e-4,"grazing inversion preserves the narrow near-birth hazard mass");
    require(fineTr<.7&&fineTr>.25,"grazing fixture has substantial near-birth attenuation");
    auto coarse=fine;coarse.inversionMaxStepOverLength=.5;
    coarse.inversionAbsoluteTolerance=1e-5;coarse.inversionRelativeTolerance=1e-3;
    one::Flight coarseGrazing(grazingField,grazing,coarse);
    close(coarseGrazing.transmittance(grazingEnd,grazingRandom),fineTr,1e-3,
          "grazing shadow is stable under coarser maximum quadrature steps");
    one::Snapshot threshold;threshold.inversionRemaining=.2;
    one::Flight grazingDistance(grazingField,grazing,fine,threshold);
    const auto grazingHit=grazingDistance.sampleDistance(grazingEnd,grazingRandom);
    require(grazingHit.hit&&grazingHit.distance<20*beta,"inversion locates a collision inside the grazing peak");
    close(logarithmicDepth(grazingHit.distance,256),.2,3e-4,
          "grazing collision meets the true optical-depth target");

    // A renderer cut can leave only one or two representable distances before
    // a fixed model boundary. Such positive-width fragments still carry
    // optical depth, although exp(-depth) may round to exactly one.
    for(bool surface:{false,true}) for(int ulps:{1,2}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        double begin=.5;
        for(int i=0;i<ulps;++i) begin=std::nextafter(begin,0.);
        one::Snapshot snapshot;snapshot.age=begin;
        one::Settings settings;settings.trackingMethod=one::TrackingMethod::Inversion;
        settings.shadowRouletteThreshold=0;
        const sh::LocalLaw reference(f,start,begin,{});
        const double depth=(.5-begin)*reference.hazard();
        require(depth>0,"machine-width fixture has nonzero optical depth");
        one::Flight shadow(f,start,settings,snapshot);
        close(shadow.transmittance(.5,grazingRandom),std::exp(-depth),2e-15,
              "inversion handles a positive-width fragment with no quadrature interior nodes");
        require(shadow.work().hazardEvaluations>0,"machine-width fragment evaluates rather than drops its hazard");
        one::Flight deterministic(f,start,settings);deterministic.selectHistory(.5);
        compareCandidates(shadow.snapshot(),deterministic.snapshot());

        snapshot.inversionRemaining=4*depth;
        one::Flight continuation(f,start,settings,snapshot);
        require(!continuation.sampleDistance(.5,grazingRandom).hit,"machine-width escape keeps a positive threshold");
        close(continuation.snapshot().inversionRemaining,3*depth,1e-10*depth+1e-30,
              "machine-width fragment subtracts its nonzero true optical depth");
        snapshot.inversionRemaining=.5*depth;
        one::Flight collision(f,start,settings,snapshot);
        const auto tinyHit=collision.sampleDistance(.5,grazingRandom);
        require(tinyHit.hit&&tinyHit.law&&tinyHit.distance>=begin&&tinyHit.distance<.5,
                "machine-width collision uses the previous group's history");
        close(tinyHit.law->hazard(),reference.hazard(),1e-12*(1+reference.hazard()),
              "machine-width collision law remains in the selected model");
    }
    std::cout<<"Inversion first-hit, surface, shadow and continuation checks passed\n";
}

void trackingChecks() {
    const auto f=field(std::make_shared<cg::ConstantMean>(.4));
    constexpr double end=2.5;
    constexpr int count=1200;
    for(bool surface:{false,true}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        one::Settings settings;settings.shadowRouletteThreshold=0;
        const double coarse=opticalReference(f,start,settings,end,160);
        const double fine=opticalReference(f,start,settings,end,320);
        close(coarse,fine,2e-3,"independent midpoint integral converges at fixed selection boundaries");
        const double expected=std::exp(-fine);
        cg::Random rng(surface?173:241);
        int escaped=0,checkedSpeeds=0;
        for(int i=0;i<count;++i) {
            one::Flight flight(f,start,settings);
            const auto result=flight.sampleDistance(end,rng);
            escaped+=!result.hit;
            require(result.distance>=0&&result.distance<=end,"sample stays inside requested segment");
            if(result.hit) {
                require(bool(result.law),"accepted collision freezes its selected local law");
                if(checkedSpeeds++<4) {
                    const auto selected=flight.snapshot().selected;
                    const sh::LocalLaw exact(f,start,result.distance,
                        selected?std::vector<double>{*selected}:std::vector<double>{});
                    close(result.law->hazard(),exact.hazard(),1e-10,
                          "accepted hazard and collision law retain the same history");
                    const double median=result.law->sampleSpeed(.5);
                    require(median>0,"collision velocity points into the solid");
                    close(exact.speedCdf(median),.5,3e-4,
                          "collision speed inversion uses the accepted history model");
                }
            }
        }
        close(escaped/double(count),expected,6*std::sqrt(expected*(1-expected)/count)+.002,
              "delta escape agrees with independent integrated hazard");
        for(double roulette:{0.0,0.9}) {
            settings.shadowRouletteThreshold=roulette;
            double sum=0,square=0;std::uint64_t terminations=0;
            for(int i=0;i<count;++i) {
                one::Flight flight(f,start,settings);
                const double tr=flight.transmittance(end,rng);
                require(tr>=0&&std::isfinite(tr),"ratio estimate is finite and nonnegative");
                sum+=tr;square+=tr*tr;terminations+=flight.work().rouletteTerminations;
            }
            const double variance=std::max(0.0,(square-sum*sum/count)/(count-1));
            close(sum/count,expected,6*std::sqrt(variance/count)+.002,
                  "ratio expectation matches same-model numerical transmittance");
            if(roulette>0) require(terminations>0,"roulette fixture exercises early shadow termination");
        }
        std::cout<<(surface?"surface":"first-hit")<<" expected T="<<expected
                 <<" delta T="<<escaped/double(count)<<"\n";
    }
}

std::vector<one::Settings> majorantVariants() {
    one::Settings legacy;legacy.shadowRouletteThreshold=0;
    auto normalized=legacy;normalized.normalizedBoundOnly=true;
    auto far=legacy;far.farBoundStartScale=4;
    auto cheap=legacy;cheap.tightenExpectedEvents=8;
    auto combined=normalized;combined.farBoundStartScale=4;combined.tightenExpectedEvents=8;
    return {legacy,normalized,far,cheap,combined};
}

void majorantModelChecks() {
    // Proposal bounds may change candidate locations and random consumption.
    // They must not alter the causal history, hazard or collision-speed law.
    const auto variants=majorantVariants();
    M metric;metric<<1.4,.15,.1, .15,.8,-.1, .1,-.1,1.1;
    const auto f=field(std::make_shared<ProfileMean>(),.7,metric);
    for(bool surface:{false,true}) {
        sh::Start start;start.direction=V(.1,-.2,1).normalized();
        if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V(.2,-.1,.7);
        one::Flight reference(f,start,variants.front());
        const double ell=reference.lengthScale();
        std::vector<std::unique_ptr<one::Flight>> changed;
        for(std::size_t i=1;i<variants.size();++i)
            changed.emplace_back(std::make_unique<one::Flight>(f,start,variants[i]));
        for(double q:{.00001,.12,.49,.51,1.01,2.3,3.95,4.05,4.6,8.2}) {
            const double t=q*ell;
            const auto history=reference.selectHistory(t);
            const auto law=reference.law(t);
            const double hazard=law.hazard();
            for(auto &flight:changed) {
                require(flight->selectHistory(t)==history,"majorant options retain the selected history");
                compareCandidates(reference.snapshot(),flight->snapshot());
                const auto actual=flight->law(t);
                close(actual.hazard(),hazard,0,"majorant options do not alter extinction statistics");
                if(q>=.12) for(double speed:{.2,.8,1.7})
                    close(actual.speedCdf(speed),law.speedCdf(speed),0,
                          "majorant options do not alter the collision velocity distribution");
            }
        }
    }
}

void majorantCoverageChecks() {
    const auto variants=majorantVariants();
    // A difficult positive-history event can inflate the bound by orders of
    // magnitude; this also exercises scaling the cheap-accept event budget
    // by the selected history probability, not only by the baseline rate.
    for(double mean:{.4,-2.4}) for(bool surface:{false,true}) {
        const auto f=field(std::make_shared<cg::ConstantMean>(mean));
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        for(const auto &settings:variants) {
            one::Flight flight(f,start,settings);
            for(const auto &interval:std::vector<std::pair<double,double>>{
                    {1e-6,2e-6},{.001,.002},{.08,.12},{.51,.65},{1.05,1.23},
                    {3.81,3.97},{4.01,4.23},{4.55,4.78},{8.01,8.23}}) {
                const auto history=flight.selectHistory(interval.first);
                const double maximum=flight.diagnosticMajorant(interval.first,interval.second);
                require(std::isfinite(maximum)&&maximum>=0,"optimized history majorant is finite");
                // Use the same fixed history even at the right endpoint.
                // Calling Flight::law there would silently change the group
                // when a test interval happens to end on a selection boundary.
                for(int i=0;i<=12;++i) {
                    const double t=interval.first+(interval.second-interval.first)*i/12;
                    const sh::LocalLaw law(f,start,t,history,settings.probability);
                    require(law.hazard()<=maximum*(1+1e-10)+1e-12,
                            "optimized history majorant covers sampled extinction");
                }
            }
        }
    }
    // Surface conditioning has a narrow near-birth feature for a grazing
    // outgoing gradient. Logarithmic intervals resolve it; uniform samples at
    // ordinary distances would miss a broken birth majorant.
    const auto grazingField=field(std::make_shared<cg::ConstantMean>(0));
    sh::Start grazing;grazing.kind=sh::Start::SurfaceValueGradient;
    grazing.gradient=1e-6*V::UnitZ();
    double peak=0;
    for(const auto &settings:variants) {
        one::Flight flight(grazingField,grazing,settings);
        for(double a:{1e-8,1e-7,5e-7,1e-6,2e-6,1e-5,1e-4}) {
            const double b=1.5*a;
            const double maximum=flight.diagnosticMajorant(a,b);
            require(maximum>=0&&std::isfinite(maximum),"grazing birth majorant remains finite");
            for(int i=0;i<=8;++i) {
                const double t=a+(b-a)*i/8;
                const double hazard=sh::LocalLaw(grazingField,grazing,t,{}).hazard();
                peak=std::max(peak,hazard);
                require(hazard<=maximum*(1+1e-10)+1e-12,
                        "optimized birth majorant covers the narrow grazing peak");
            }
        }
    }
    require(peak>1e4,"grazing bound fixture reaches the narrow near-origin peak");
}

void majorantTrackingChecks() {
    const auto f=field(std::make_shared<cg::ConstantMean>(.4));
    const auto settings=majorantVariants().back();
    constexpr double end=5.25; // crosses the far-bound switch at four lengths
    constexpr int count=600;
    for(bool surface:{false,true}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        const double coarse=groupedOpticalReference(f,start,{},end,32);
        const double fine=groupedOpticalReference(f,start,{},end,64);
        close(coarse,fine,8e-4,"independent reference resolves the optimized tracking fixture");
        const double expected=std::exp(-fine);
        cg::Random rng(surface?37051:47087);
        int escaped=0,speedChecks=0,continued=0;
        double sum=0,square=0;
        for(int i=0;i<count;++i) {
            one::Flight flight(f,start,settings);
            one::TrackResult result;
            // Half of the samples cross a transparent cut inside a fixed
            // history group. Preserve the residual proposal clock at the cut.
            if(i%2==0) {
                result=flight.sampleDistance(.37,rng);
                if(!result.hit) {
                    const auto saved=flight.snapshot();
                    one::Flight resumed(f,start,settings,saved);
                    cg::Random probe=rng;
                    const auto uninterrupted=flight.sampleDistance(end,probe);
                    result=resumed.sampleDistance(end,rng);
                    require(result.hit==uninterrupted.hit,
                            "restored majorant clock preserves continuation outcome");
                    close(result.distance,uninterrupted.distance,0,
                          "restored majorant clock preserves continuation distance");
                    require(rng.uniform01()==probe.uniform01(),
                            "transparent continuation retains the residual exponential clock");
                    compareCandidates(resumed.snapshot(),flight.snapshot());
                    close(saved.age,.37,0,"resuming a path leaves its saved snapshot immutable");
                    ++continued;
                }
            } else result=flight.sampleDistance(end,rng);
            escaped+=!result.hit;
            require(result.distance>=0&&result.distance<=end,"optimized delta sample remains in the segment");
            if(result.hit&&speedChecks++<4) {
                require(bool(result.law),"optimized delta collision retains its local law");
                one::Flight model(f,start);const auto exact=model.law(result.distance);
                close(result.law->hazard(),exact.hazard(),1e-10*(1+exact.hazard()),
                      "optimized delta collision uses the unchanged hazard model");
                const double speed=result.law->sampleSpeed(.5);
                close(exact.speedCdf(speed),.5,3e-4,
                      "optimized delta collision velocity uses the same selected history");
            }
            one::Flight shadow(f,start,settings);
            const double tr=shadow.transmittance(end,rng);
            require(tr>=0&&tr<=1&&std::isfinite(tr),"optimized ratio tracking remains a valid weight");
            sum+=tr;square+=tr*tr;
        }
        require(continued>0,"optimized delta fixture exercises transparent continuation");
        close(escaped/double(count),expected,6*std::sqrt(expected*(1-expected)/count)+.002,
              "optimized delta agrees with independently integrated same-model transmittance");
        const double variance=std::max(0.0,(square-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+.002,
              "optimized shadow estimator agrees with independent transmittance");
        std::cout<<(surface?"surface":"first-hit")<<" optimized bounds expected T="<<expected
                 <<" delta T="<<escaped/double(count)<<" ratio T="<<sum/count<<"\n";
    }
}

// A large exponential threshold makes routing/history tests deterministic:
// every requested segment is traversed without a random proposal. The field
// fixture below has a total majorant optical depth far below -log(FLT_MIN).
class LargeOpticalDepthSampler final : public Tungsten::PathSampleGenerator {
    Tungsten::UniformSampler _unused{1};
public:
    void startPath(Tungsten::uint32,Tungsten::uint32) override {}
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override { throw std::runtime_error("unused sampler operation"); }
    void loadState(Tungsten::InputStreamHandle &) override { throw std::runtime_error("unused sampler operation"); }
    bool nextBoolean(float p) override { return next1D()<p; }
    int nextDiscrete(int n) override { return int(next1D()*n); }
    float next1D() override { return std::numeric_limits<float>::min(); }
    Tungsten::Vec2f next2D() override { return Tungsten::Vec2f(next1D()); }
    Tungsten::UniformSampler &uniformGenerator() override { return _unused; }
};

// A valid, reproducible proposal at optical depth log(2), followed by either
// a very small acceptance variate or a long residual exponential clock.
class LeadingEventSampler final : public Tungsten::PathSampleGenerator {
    Tungsten::UniformSampler _unused{1};
    std::size_t _calls=0;
public:
    void startPath(Tungsten::uint32,Tungsten::uint32) override {}
    void advancePath() override {}
    void saveState(Tungsten::OutputStreamHandle &) override { throw std::runtime_error("unused sampler operation"); }
    void loadState(Tungsten::InputStreamHandle &) override { throw std::runtime_error("unused sampler operation"); }
    bool nextBoolean(float p) override { return next1D()<p; }
    int nextDiscrete(int n) override { return int(next1D()*n); }
    float next1D() override { return _calls++==0 ? .5f:std::numeric_limits<float>::min(); }
    Tungsten::Vec2f next2D() override { return Tungsten::Vec2f(next1D()); }
    Tungsten::UniformSampler &uniformGenerator() override { return _unused; }
    std::size_t calls() const { return _calls; }
};

void largeHistoryBoundRefinementChecks() {
    // The retained positive sign is rare in this negative mean field. Its
    // certified bound is large even though the local crossing law is finite.
    // Start from a legitimate, deterministic prefix rather than manufacturing
    // a selected history point. At this distance the optimized constant far
    // enclosure also gives a simple independent optical-clock reference.
    const auto f=field(std::make_shared<cg::ConstantMean>(-3));
    auto settings=majorantVariants().back();
    settings.profile=true;settings.profileDetail=true;settings.maxWork=128;
    constexpr double begin=8.1,end=8.5;
    for(bool surface:{false,true}) {
        sh::Start start;start.gradient=V::UnitZ();
        if(surface) start.kind=sh::Start::SurfaceValueGradient;
        auto prefixSettings=settings;prefixSettings.maxWork=1000000;
        one::Flight prefix(f,start,prefixSettings);
        const auto history=prefix.selectHistory(begin);
        require(history.size()==1,"rare-sign refinement fixture has one causally selected history point");
        const auto saved=prefix.snapshot();

        one::Flight bounds(f,start,settings,saved);
        const double maximum=bounds.diagnosticMajorant(begin,end);
        require(std::isfinite(maximum)&&maximum*(end-begin)>128,
                "rare-sign fixture would require repeated binary bound refinements");
        const double shortEnd=begin+40/maximum;
        require(shortEnd<end,"clock reference remains within one fixed-history group");
        close(bounds.diagnosticMajorant(begin,shortEnd),maximum,1e-12*maximum,
              "constant far enclosure is unchanged by a shorter query interval");
        const double expectedDistance=begin+std::log(2.0)/maximum;
        const sh::LocalLaw exact(f,start,expectedDistance,history,settings.probability);
        const double hazard=exact.hazard(),acceptance=hazard/maximum;
        require(acceptance>std::numeric_limits<float>::min()&&acceptance<1,
                "scripted acceptance is a possible event under the certified bound");

        LeadingEventSampler distanceSource;cg::Random distanceRandom(distanceSource);
        one::Flight distance(f,start,settings,saved);
        const auto hit=distance.sampleDistance(end,distanceRandom);
        require(hit.hit&&hit.law,"large-bound first proposal returns a valid collision law");
        const auto binaryQueries=std::uint64_t(1+std::ceil(std::log2(maximum*(end-begin)/8)));
        require(distance.work().intervals<=8&&2*distance.work().intervals<=binaryQueries&&
                distance.work().eventSplits>0,
                "rare-sign bound uses at most half the bound queries of ordinary binary refinement");
        require(distance.work().candidates==1&&distanceSource.calls()==2,
                "event-budget refinement neither redraws the clock nor adds proposals");
        close(hit.distance,expectedDistance,8*std::numeric_limits<double>::epsilon()*begin,
              "refinement preserves the exponential proposal's optical distance");
        compareCandidates(distance.snapshot(),saved);
        const sh::LocalLaw actual(f,start,hit.distance,history,settings.probability);
        close(hit.law->hazard(),actual.hazard(),0,
              "large-bound collision retains the actual-position history law");
        const double speed=hit.law->sampleSpeed(.5);
        close(actual.speedCdf(speed),.5,3e-4,
              "large-bound collision samples the unchanged history-conditioned speed");

        // After the first event, -log(FLT_MIN)>40 leaves no more proposals.
        // Ratio tracking therefore has the explicit one-factor answer below.
        LeadingEventSampler shadowSource;cg::Random shadowRandom(shadowSource);
        one::Flight shadow(f,start,settings,saved);
        const double transmittance=shadow.transmittance(shortEnd,shadowRandom);
        close(transmittance,1-acceptance,1e-11,
              "refined shadow intervals preserve the exact one-event ratio weight");
        require(shadow.work().candidates==1&&shadowSource.calls()==2,
                "shadow refinements preserve one residual exponential clock");
        compareCandidates(shadow.snapshot(),saved);

        LeadingEventSampler continuedSource;cg::Random continuedRandom(continuedSource);
        one::Flight first(f,start,settings,saved);
        const double firstWeight=first.transmittance(begin+3/maximum,continuedRandom);
        const auto cut=first.snapshot();
        one::Flight continued(f,start,settings,cut);
        const double secondWeight=continued.transmittance(shortEnd,continuedRandom);
        close(firstWeight*secondWeight,transmittance,1e-11,
              "transparent continuation keeps the refined shadow clock and weight");
        require(continued.work().candidates==0&&continuedSource.calls()==2,
                "restoring the refined residual clock does not draw a new threshold");
        compareCandidates(continued.snapshot(),shadow.snapshot());
        std::cout<<(surface?"surface":"first-hit")<<" rare history bound="<<maximum
                 <<" first-proposal bound queries="<<distance.work().intervals<<"\n";
    }
}

void nearBirthRefinementChecks() {
    // A constant mean with conservative, width-dependent interval envelopes.
    // Such loose certificates are valid: shrinking a wide cell range can make
    // its recomputed bound many orders smaller. Jumping directly to 8/M then
    // makes almost-zero-depth microsteps without approaching the cell face.
    // Ordinary bisection must still handle this near-birth, no-history case.
    class LooseCellMean final : public cg::MeanField {
    public:
        const char *typeName() const override { return "loose-near-birth-test-cell"; }
        cg::MeanJet evaluate(const V &) const override { return {0,V::Zero()}; }
        cg::BoundsSummary bounds(const cg::Bounds3 &) const override { return {0,0,true}; }
        std::optional<V> affineGradient() const override { return V::Zero(); }
        void appendRayBreakpoints(const V &,const V &,double lo,double hi,
                                  std::vector<double> &out) const override {
            if(lo<2e-4&&2e-4<hi) out.push_back(2e-4);
        }
        cg::MeanRayBounds rayBounds(const V &origin,const V &direction,double lo,double hi,
                                   const Tungsten::SdfRayQueryOptions &options={}) const override {
            auto result=cg::MeanField::rayBounds(origin,direction,lo,hi,options);
            if(lo>0) {
                const double slopeSlack=1e6*(hi-lo),valueSlack=slopeSlack*(hi-lo);
                result.minimumValue-=valueSlack;result.maximumValue+=valueSlack;
                result.minimumDerivative-=slopeSlack;result.maximumDerivative+=slopeSlack;
            }
            return result;
        }
    };
    const auto f=field(std::make_shared<LooseCellMean>(),.02,25*M::Identity());
    sh::Start start;start.kind=sh::Start::SurfaceValueGradient;start.gradient=.5*V::UnitZ();
    auto settings=majorantVariants().back();
    settings.profile=true;settings.profileDetail=true;settings.maxWork=4096;
    constexpr double end=3e-4;
    one::Flight certificate(f,start,settings);
    const double wide=certificate.diagnosticMajorant(1e-4,2e-4);
    require(wide*1e-4>1e6,"near-birth fixture starts from a loose but finite certified enclosure");
    for(double t:{1e-5,5e-5,1e-4,2e-4,end})
        close(sh::LocalLaw(f,start,t,{},settings.probability).hazard(),0,0,
              "outgoing near-birth fixture has negligible true crossing probability");

    LargeOpticalDepthSampler distanceSource;cg::Random distanceRandom(distanceSource);
    one::Flight distance(f,start,settings);
    const auto result=distance.sampleDistance(end,distanceRandom);
    require(!result.hit&&result.distance==end,"near-birth tracking reaches the requested prefix");
    require(!distance.snapshot().selected&&distance.work().selections==0,
            "near-birth regression does not introduce an internal history point");
    require(distance.work().intervals<1024&&distance.work().eventSplits>0,
            "near-birth binary tightening avoids unbounded empty microsteps");
    LargeOpticalDepthSampler shadowSource;cg::Random shadowRandom(shadowSource);
    one::Flight shadow(f,start,settings);
    close(shadow.transmittance(end,shadowRandom),1,0,
          "near-birth shadow preserves the negligible-crossing limit");
    require(shadow.work().intervals<1024,"near-birth shadow tightening has bounded work");
    compareCandidates(distance.snapshot(),shadow.snapshot());
    std::cout<<"near-birth no-history bound queries="<<distance.work().intervals<<"\n";
}

#if OPENVDB_AVAILABLE
std::shared_ptr<Tungsten::GpisSdfGrid> macroTestGrid(int width) {
    using namespace Tungsten;
    auto grid=std::make_shared<GpisSdfGrid>();
    grid->data.origin=Vec3i(0);grid->data.dimensions=Vec3i(8,8,90);
    grid->data.clampMin=Vec3f(.25f);grid->data.clampMax=Vec3f(6.75f,6.75f,88.75f);
    grid->worldToIndex=20*M::Identity();grid->translation=V::Zero();
    grid->scale=1;grid->offset=0;grid->data.values.resize(8*8*90);
    for(int z=0;z<90;++z) for(int y=0;y<8;++y) for(int x=0;x<8;++x) {
        const double t=z/20.;
        grid->data.values[(z*8+y)*8+x]=float(.35+.14*std::sin(4*t)+.025*std::cos(11*t)+.002*x);
    }
    grid->initializeMagnitudeBounds();grid->buildMacrocells(width);
    return grid;
}

sh::Start macroTestStart(bool surface) {
    sh::Start start;start.origin=V(.083,.107,.071);
    start.gradient=V::UnitZ();
    if(surface) start.kind=sh::Start::SurfaceValueGradient;
    return start;
}

one::Settings macroTestSettings(int width) {
    auto settings=majorantVariants().back();
    settings.profile=true;settings.profileDetail=true;
    settings.macrocellSize=width;settings.macrocellMaxExpectedEvents=8;
    return settings;
}

double macroOpticalReference(const cg::GPSSField &f,const sh::Start &start,double end,int perCell,
                             const one::Settings &settings={}) {
    // Known axis-aligned grid faces plus history changes, independent of the
    // macro traversal and its proposal rates. Within each interval the mean
    // is affine and this composite midpoint reference resolves the hazard.
    std::vector<double> boundaries{0,end};
    for(int node=2;node<90;++node) {
        const double t=node/20.-start.origin.z();
        if(t>0&&t<end) boundaries.push_back(t);
    }
    for(int group=1;group*.5<end;++group) boundaries.push_back(group*.5);
    std::sort(boundaries.begin(),boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(),boundaries.end()),boundaries.end());
    one::Flight reference(f,start,settings);
    double optical=0;
    for(std::size_t j=1;j<boundaries.size();++j) {
        const double a=boundaries[j-1],b=boundaries[j];
        for(int i=0;i<perCell;++i) {
            const double t=a+(b-a)*(i+.5)/perCell;
            optical+=(b-a)/perCell*reference.law(t).hazard();
        }
    }
    return optical;
}

void macroRoutingChecks() {
    constexpr double end=3.25;
    for(int width:{2,4}) {
        const auto f=field(macroTestGrid(width));
        for(bool surface:{false,true}) {
            const auto start=macroTestStart(surface);
            const auto settings=macroTestSettings(width);
            auto disabled=settings;disabled.macrocellSize=0;
            LargeOpticalDepthSampler sourceA,sourceB;
            cg::Random rngA(sourceA),rngB(sourceB);
            one::Flight coarse(f,start,settings),fine(f,start,disabled),reference(f,start);
            const auto coarseHit=coarse.sampleDistance(end,rngA),fineHit=fine.sampleDistance(end,rngB);
            require(!coarseHit.hit&&!fineHit.hit&&coarse.work().candidates==0&&fine.work().candidates==0,
                    "macro routing fixture traverses the complete requested prefix");
            require(coarse.work().macroAccepted>0,"real grid uses accepted macro bounds");
            require(coarse.work().processedIntervals<fine.work().processedIntervals,
                    "macro bounds reduce tracking intervals across fine cells");
            reference.selectHistory(end);
            compareCandidates(coarse.snapshot(),reference.snapshot());
            compareCandidates(fine.snapshot(),reference.snapshot());
            require(coarse.work().cells==reference.work().cells,
                    "macro acceleration retains the canonical fine-cell history collection");
            checkSelectionReference(f,start,settings,coarse);

            // Renderer cuts are deliberately inside fine cells, macro blocks,
            // and selection groups. Rebuild the flight at every transparent cut.
            LargeOpticalDepthSampler sourceC;cg::Random rngC(sourceC);
            std::unique_ptr<one::Flight> split=std::make_unique<one::Flight>(f,start,settings);
            for(double cut:{.137,.509,.873,1.237,2.001,end}) {
                const auto result=split->sampleDistance(cut,rngC);
                require(!result.hit,"large-threshold transparent continuation escapes");
                one::Flight geometry(f,start);geometry.selectHistory(cut);
                compareCandidates(split->snapshot(),geometry.snapshot());
                const auto saved=split->snapshot();
                split=std::make_unique<one::Flight>(f,start,settings,saved);
                close(saved.age,cut,0,"macro continuation leaves its source snapshot immutable");
            }
            compareCandidates(split->snapshot(),reference.snapshot());

            auto restrictive=settings;restrictive.macrocellMaxExpectedEvents=1e-12;
            LargeOpticalDepthSampler sourceD;cg::Random rngD(sourceD);
            one::Flight fallback(f,start,restrictive);
            require(!fallback.sampleDistance(end,rngD).hit,"rejected macro bounds retain a working fine fallback");
            require(fallback.work().macroRejected>0&&fallback.work().macroRetrySkips>0,
                    "rejected macro window is remembered across later fine cells");
            require(fallback.work().macroAttempts<fallback.work().cells,
                    "rejected macro queries are not repeated for every crossed fine cell");
            compareCandidates(fallback.snapshot(),reference.snapshot());

            bool hitChecked=false,shadowChecked=false;
            for(int seed=1;seed<=32&&(!hitChecked||!shadowChecked);++seed) {
                if(!hitChecked) {
                    cg::Random random(811+seed);one::Flight path(f,start,settings);
                    const auto hit=path.sampleDistance(end,random);
                    if(hit.hit&&path.work().macroAccepted>0) {
                        one::Flight geometry(f,start);const auto exact=geometry.law(hit.distance);
                        compareCandidates(path.snapshot(),geometry.snapshot());
                        require(bool(hit.law),"macro collision retains its accepted local law");
                        close(hit.law->hazard(),exact.hazard(),2e-10*(1+exact.hazard()),
                              "macro hit and canonical history evaluate the same hazard");
                        const double speed=hit.law->sampleSpeed(.5);
                        close(exact.speedCdf(speed),.5,3e-4,
                              "macro collision speed uses the causal selected history");
                        hitChecked=true;
                    }
                }
                if(!shadowChecked) {
                    auto roulette=settings;roulette.shadowRouletteThreshold=.99;
                    cg::Random random(1213+seed);one::Flight shadow(f,start,roulette);
                    const double tr=shadow.transmittance(end,random);
                    if(shadow.work().rouletteTerminations>0&&shadow.work().macroAccepted>0) {
                        require(tr==0&&shadow.snapshot().age<end,"shadow roulette ends inside the requested prefix");
                        one::Flight geometry(f,start);geometry.selectHistory(shadow.snapshot().age);
                        compareCandidates(shadow.snapshot(),geometry.snapshot());
                        shadowChecked=true;
                    }
                }
            }
            require(hitChecked&&shadowChecked,"macro fixture exercises early collision and shadow termination");
        }
    }
}

void macroTrackingChecks() {
    const auto f=field(macroTestGrid(4));
    constexpr double end=3.25;constexpr int count=400;
    auto settings=macroTestSettings(4);settings.profile=false;settings.profileDetail=false;
    for(bool surface:{false,true}) {
        const auto start=macroTestStart(surface);
        const double coarse=macroOpticalReference(f,start,end,8),fine=macroOpticalReference(f,start,end,16);
        close(coarse,fine,5e-4,"independent cellwise hazard integral converges for the macro fixture");
        const double expected=std::exp(-fine);
        cg::Random random(surface?75187:25819);
        int escaped=0;double sum=0,square=0;
        for(int i=0;i<count;++i) {
            one::Flight path(f,start,settings);one::TrackResult hit;
            if(i%2==0) {
                hit=path.sampleDistance(.873,random);
                if(!hit.hit) {
                    one::Flight resumed(f,start,settings,path.snapshot());
                    hit=resumed.sampleDistance(end,random);
                }
            } else hit=path.sampleDistance(end,random);
            escaped+=!hit.hit;
            one::Flight shadow(f,start,settings);
            const double tr=shadow.transmittance(end,random);
            require(tr>=0&&tr<=1&&std::isfinite(tr),"macro ratio tracking has a finite valid weight");
            sum+=tr;square+=tr*tr;
        }
        close(escaped/double(count),expected,6*std::sqrt(expected*(1-expected)/count)+.002,
              "macro distance samples preserve independently integrated transmittance");
        const double variance=std::max(0.0,(square-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+.002,
              "macro shadow estimates preserve independently integrated transmittance");
        std::cout<<(surface?"surface":"first-hit")<<" macro expected T="<<expected
                 <<" delta T="<<escaped/double(count)<<" ratio T="<<sum/count<<"\n";
    }
}

void samePruningSnapshot(const one::Snapshot &a,const one::Snapshot &b) {
    compareCandidates(a,b);
    require(a.age==b.age&&a.fragmentBegin==b.fragmentBegin&&
            a.macroRetryAt==b.macroRetryAt&&a.macroUnavailable==b.macroUnavailable,
            "candidate pruning preserves exact traversal and macro retry state");
    const auto same=[](const one::Candidate &x,const one::Candidate &y) {
        return x.distance==y.distance&&x.z==y.z&&x.inverseMills==y.inverseMills;
    };
    if(a.current) {
        require(same(*a.current,*b.current),"pruning preserves the exact current-group representative");
        require(b.current->distance<=b.age,"pruning cannot introduce an unpassed candidate");
    }
    for(std::size_t i=0;i<a.completed.size();++i)
        require(same(a.completed[i],b.completed[i]),"pruning preserves every exact completed representative");
}

std::shared_ptr<Tungsten::GpisSdfGrid> pruningTestGrid(int width,double offset,double slope) {
    auto grid=macroTestGrid(width);
    for(int z=0;z<90;++z) for(int y=0;y<8;++y) for(int x=0;x<8;++x)
        grid->data.values[(z*8+y)*8+x]=float(offset+slope*z/20.);
    grid->initializeMagnitudeBounds();grid->buildMacrocells(width);
    return grid;
}

void pruningSelectionCase(const cg::GPSSField &f,const sh::Start &start,int width,double end,
                          bool expectPruning=false,bool expectNoPruning=false) {
    const auto oldSettings=macroTestSettings(width);
    auto newSettings=oldSettings;newSettings.candidateMacroPruning=true;
    one::Flight original(f,start,oldSettings),pruned(f,start,newSettings);
    std::vector<double> prefixes{1e-8,1e-5,.005,.137,.509,.873,1.237,2.001,end};
    const double step=original.lengthScale()*oldSettings.selectionStepOverLength;
    for(int group=1;group*step<end;++group) {
        const double boundary=group*step;
        prefixes.push_back(std::nextafter(boundary,0.));prefixes.push_back(boundary);
        prefixes.push_back(std::nextafter(boundary,std::numeric_limits<double>::infinity()));
    }
    std::sort(prefixes.begin(),prefixes.end());
    prefixes.erase(std::unique(prefixes.begin(),prefixes.end()),prefixes.end());
    for(double t:prefixes) if(t<=end) {
        require(original.selectHistory(t)==pruned.selectHistory(t),
                "pruning retains selected history across partial cells and group boundaries");
        samePruningSnapshot(original.snapshot(),pruned.snapshot());
    }
    require(pruned.work().cells==original.work().cells,
            "candidate-only pruning preserves canonical fine-fragment traversal");
    if(expectPruning) {
        require(pruned.work().candidateMacroPrunedFragments>0&&
                pruned.work().candidateStatistics<original.work().candidateStatistics,
                "strictly higher macro regions avoid candidate statistics without changing representatives");
    }
    if(expectNoPruning)
        require(pruned.work().candidateMacroPrunedFragments==0,
                "equal-height candidates are not discarded by a non-strict lower-bound comparison");
    std::cout<<(start.kind==sh::Start::SurfaceValueGradient?"surface":"first-hit")
             <<" pruning width="<<width<<" fragments="<<pruned.work().candidateMacroPrunedFragments
             <<" candidate statistics "<<original.work().candidateStatistics
             <<" -> "<<pruned.work().candidateStatistics<<"\n";
}

void pruningGeometryChecks() {
    for(int width:{2,4}) for(bool surface:{false,true}) {
        const auto start=macroTestStart(surface);
        // Once the lower first part of a group is retained, a later increasing
        // macro block can be certified irrelevant. Surface conditioning is
        // deliberately still active near birth; successful pruning occurs
        // farther along this same flight.
        const auto increasing=field(pruningTestGrid(width,-.25,.4));
        pruningSelectionCase(increasing,start,width,4.2,true);
        const auto varying=field(macroTestGrid(width));
        pruningSelectionCase(varying,start,width,3.25);
        const auto flat=field(pruningTestGrid(width,.5,0));
        pruningSelectionCase(flat,start,width,3.25,false,!surface);

        // Negative lower bounds require the small standard-deviation endpoint
        // when bounding z. Large birth corrections and a non-unit directional
        // precision also exercise signs/scales absent from a positive flat grid.
        M metric;metric<<.7,.05,.1, .05,1.2,-.08, .1,-.08,2.25;
        auto oblique=start;oblique.direction=V(.03,.02,1).normalized();
        oblique.gradient=8*oblique.direction+V(.2,-.1,0);
        const auto negative=field(pruningTestGrid(width,-3,.25),.7,metric);
        pruningSelectionCase(negative,oblique,width,3.75);
    }
}

void pruningTrackingChecks() {
    constexpr double end=4.2;
    for(int width:{2,4}) for(bool surface:{false,true}) {
        const auto f=field(pruningTestGrid(width,-.25,.4));
        const auto start=macroTestStart(surface);
        auto oldSettings=macroTestSettings(width);oldSettings.shadowRouletteThreshold=.99;
        auto newSettings=oldSettings;newSettings.candidateMacroPruning=true;
        bool checkedSpeed=false;std::uint64_t prunedFragments=0;
        for(int seed=0;seed<24;++seed) {
            cg::Random oldRandom(5903+seed),newRandom(5903+seed);
            auto original=std::make_unique<one::Flight>(f,start,oldSettings);
            auto pruned=std::make_unique<one::Flight>(f,start,newSettings);
            one::TrackResult oldHit,newHit;
            for(double cut:{.873,end}) {
                oldHit=original->sampleDistance(cut,oldRandom);
                newHit=pruned->sampleDistance(cut,newRandom);
                require(oldHit.hit==newHit.hit&&oldHit.distance==newHit.distance,
                        "candidate pruning preserves the exact same-seed collision sample");
                samePruningSnapshot(original->snapshot(),pruned->snapshot());
                require(original->work().candidates==pruned->work().candidates&&
                        original->work().processedIntervals==pruned->work().processedIntervals,
                        "candidate pruning does not alter the proposal process");
                prunedFragments+=pruned->work().candidateMacroPrunedFragments;
                if(oldHit.hit) break;
                original=std::make_unique<one::Flight>(f,start,oldSettings,original->snapshot());
                pruned=std::make_unique<one::Flight>(f,start,newSettings,pruned->snapshot());
            }
            if(oldHit.hit&&!checkedSpeed) {
                require(oldHit.law&&newHit.law,"same-seed collision retains both velocity laws");
                close(oldHit.law->hazard(),newHit.law->hazard(),0,"pruning preserves collision hazard exactly");
                for(double speed:{.2,.8,1.7})
                    close(oldHit.law->speedCdf(speed),newHit.law->speedCdf(speed),0,
                          "pruning preserves the collision velocity CDF exactly");
                checkedSpeed=true;
            }
            require(oldRandom.uniform01()==newRandom.uniform01(),
                    "candidate pruning does not consume or reorder path random draws");

            cg::Random oldShadowRandom(7907+seed),newShadowRandom(7907+seed);
            auto oldShadow=std::make_unique<one::Flight>(f,start,oldSettings);
            auto newShadow=std::make_unique<one::Flight>(f,start,newSettings);
            for(double cut:{.873,end}) {
                const double a=oldShadow->transmittance(cut,oldShadowRandom);
                const double b=newShadow->transmittance(cut,newShadowRandom);
                close(a,b,0,"candidate pruning preserves exact same-seed shadow estimates");
                samePruningSnapshot(oldShadow->snapshot(),newShadow->snapshot());
                require(oldShadow->work().rouletteTerminations==newShadow->work().rouletteTerminations,
                        "candidate pruning preserves shadow termination decisions");
                prunedFragments+=newShadow->work().candidateMacroPrunedFragments;
                if(a==0) break;
                oldShadow=std::make_unique<one::Flight>(f,start,oldSettings,oldShadow->snapshot());
                newShadow=std::make_unique<one::Flight>(f,start,newSettings,newShadow->snapshot());
            }
            require(oldShadowRandom.uniform01()==newShadowRandom.uniform01(),
                    "candidate pruning does not consume or reorder shadow random draws");
        }
        require(checkedSpeed&&prunedFragments>0,
                "same-seed pruning fixture exercises both actual pruning and collision laws");
    }
}

class CandidateRecordingGrid final : public Tungsten::GpisSdfGrid {
public:
    struct Counts { std::uint64_t fineBoundaries=0,macroBoundaries=0,stationaryQueries=0,cursorCreates=0; };
    mutable Counts counts;
    std::unique_ptr<cg::MeanRayCursor> createRayCursor(const V &o,const V &w) const override {
        ++counts.cursorCreates;
        class Cursor final : public cg::MeanRayCursor {
            std::unique_ptr<cg::MeanRayCursor> _inner;
            Counts &_counts;
        public:
            Cursor(std::unique_ptr<cg::MeanRayCursor> inner,Counts &counts)
                :_inner(std::move(inner)),_counts(counts) {}
            double nextBoundary(double a,double b) override {
                ++_counts.fineBoundaries;return _inner->nextBoundary(a,b);
            }
            bool nextMacroBoundary(double a,double &b,int width) override {
                ++_counts.macroBoundaries;return _inner->nextMacroBoundary(a,b,width);
            }
            bool appendStationaryPoints(double a,double b,std::vector<double> &points) override {
                ++_counts.stationaryQueries;return _inner->appendStationaryPoints(a,b,points);
            }
            cg::MeanRaySample sample(double t) override { return _inner->sample(t); }
            cg::MeanJet evaluate(double t) override { return _inner->evaluate(t); }
            double valueDifference(double t,double *error=nullptr) override { return _inner->valueDifference(t,error); }
            cg::MeanRayBounds rayBounds(double a,double b,const Tungsten::SdfRayQueryOptions &options) override {
                return _inner->rayBounds(a,b,options);
            }
            bool coarseRayBounds(double a,double &b,cg::MeanRayBounds &bounds) override {
                return _inner->coarseRayBounds(a,b,bounds);
            }
            Statistics statistics() const override { return _inner->statistics(); }
        };
        return std::make_unique<Cursor>(GpisSdfGrid::createRayCursor(o,w),counts);
    }
};

one::Settings macroCandidateSettings(int width) {
    auto settings=macroTestSettings(0);
    settings.candidateCollection=one::CandidateCollection::Macro;
    settings.candidateMacrocellSize=width;
    return settings;
}

one::Snapshot macroCandidateReference(const cg::GPSSField &f,const sh::Start &start,
                                     const one::Settings &settings,double end) {
    // Independent oracle: intersect the ray with the known grid's integer
    // macro planes and clamp planes, then add fixed history-group boundaries.
    // The renderer's cursor and its random/bound subdivision are not used.
    const auto *grid=dynamic_cast<const Tungsten::GpisSdfGrid *>(f.mean.get());
    require(grid!=nullptr,"macro candidate oracle needs the real test grid");
    const V q=grid->worldToIndex*start.origin+grid->translation;
    const V v=grid->worldToIndex*start.direction;
    const double ell=1/std::sqrt(start.direction.dot(f.kernel.precision()*start.direction));
    const double step=settings.selectionStepOverLength*ell;
    std::vector<double> boundaries{0};
    auto append=[&](double t) {if(t>0&&t<=end) boundaries.push_back(t);};
    for(int axis=0;axis<3;++axis) if(v[axis]!=0) {
        const double lo=grid->data.clampMin[axis],hi=grid->data.clampMax[axis];
        append((lo-q[axis])/v[axis]);append((hi-q[axis])/v[axis]);
        for(int plane=0;plane<=96;plane+=settings.candidateMacrocellSize)
            if(plane>lo&&plane<hi) append((plane-q[axis])/v[axis]);
    }
    for(int group=1;group*step<=end;++group) append(group*step);
    std::sort(boundaries.begin(),boundaries.end());
    boundaries.erase(std::unique(boundaries.begin(),boundaries.end()),boundaries.end());
    one::Snapshot expected;
    for(std::size_t i=1;i<boundaries.size();++i) {
        const double a=boundaries[i-1],b=boundaries[i],groupEnd=(expected.group+1)*step;
        for(double u:{a,a+.5*(b-a),b}) if(u>0&&u>=expected.group*step&&u<groupEnd) {
            const double z=sh::standardizedMean(f,start,u);
            const one::Candidate candidate{u,z,cg::normalPdfOverCdf(z)};
            if(!expected.current||z<expected.current->z||
               (z==expected.current->z&&u<expected.current->distance)) expected.current=candidate;
        }
        expected.fragmentBegin=b;
        if(b==groupEnd) {
            if(expected.current) expected.completed.push_back(*expected.current);
            ++expected.group;expected.current.reset();expected.selected.reset();
            const double oldest=(expected.group>std::uint64_t(settings.historyGroups)
                ? expected.group-settings.historyGroups:0)*step;
            expected.completed.erase(std::remove_if(expected.completed.begin(),expected.completed.end(),
                [&](const one::Candidate &c){return c.distance<oldest;}),expected.completed.end());
            const double t=expected.group*step;
            double best=-1;
            auto score=[&](double u) {
                const double value=sh::historyCandidateStatistics(f,start,t,u).score;
                if(value>best||(value==best&&(!expected.selected||u<*expected.selected))) {
                    best=value;expected.selected=u;
                }
            };
            for(const auto &candidate:expected.completed) score(candidate.distance);
            score(t-std::min(ell,t/2));
        }
    }
    expected.age=end;
    return expected;
}

void macroCandidateBoundaryChecks() {
    const auto grid=macroTestGrid(0); // geometry traversal requires no baked bounds
    auto reverse=grid->createRayCursor(V(.2,.2,.4),-V::UnitZ());
    double end=1;
    require(reverse->nextMacroBoundary(0,end,4),"reverse ray supports pure macro geometry");
    close(end,.2,1e-14,"reverse ray on a macro face advances to the next lower face");
    double next=1;
    require(reverse->nextMacroBoundary(end,next,4),"reverse ray advances from the second face");
    close(next,.3875,1e-14,"macro geometry is clipped at the grid clamp boundary");
    end=1;
    require(reverse->nextMacroBoundary(next,end,4),"clamped tail still provides macro geometry");
    close(end,1,0,"constant clamped tail has no artificial outside-grid faces");
    require(reverse->statistics().cellBuilds==0,"pure macro geometry builds no fine-cell polynomials");
}

void macroCandidateGeometryChecks() {
    macroCandidateBoundaryChecks();
    // This decorator deliberately supplies no clone capability. The evaluator
    // must retain the fresh-cursor fallback for other MeanField implementations.
    const auto fallbackGrid=std::make_shared<CandidateRecordingGrid>();
    static_cast<Tungsten::GpisSdfGrid &>(*fallbackGrid)=*macroTestGrid(0);
    const auto fallbackField=field(fallbackGrid);
    for(bool surface:{false,true}) {
        const auto birth=macroTestStart(surface);
        const std::vector<double> positions={1e-10,.001,.002,.13,1.,.001,.4};
        std::vector<double> expected;
        for(double t:positions) expected.push_back(sh::standardizedMean(fallbackField,birth,t));
        fallbackGrid->counts={};
        const sh::HistoryCandidateEvaluator cached(fallbackField,birth);
        for(std::size_t i=0;i<positions.size();++i)
            require(cached.standardizedMean(positions[i])==expected[i],
                    "non-cloneable cursor fallback preserves origin-first candidate evaluation");
        require(fallbackGrid->counts.cursorCreates==positions.size()+1,
                "non-cloneable cursor uses a fresh initialized cursor for each candidate");
    }
    for(int width:{2,4}) for(bool surface:{false,true}) {
        auto grid=std::make_shared<CandidateRecordingGrid>();
        static_cast<Tungsten::GpisSdfGrid &>(*grid)=*macroTestGrid(0);
        // A deep value away from the ray belongs to the same macro block.
        // It must never substitute for a field value at an on-ray sample.
        for(int z=0;z<90;++z) for(int y=0;y<8;++y) grid->data.values[(z*8+y)*8]-=9;
        grid->initializeMagnitudeBounds();
        const auto f=field(grid);const auto start=macroTestStart(surface);
        const auto settings=macroCandidateSettings(width);
        one::Settings fineSettings=settings;fineSettings.candidateCollection=one::CandidateCollection::Fine;
        one::Flight fine(f,start,fineSettings);
        grid->counts={};fine.selectHistory(3.25);const auto fineCounts=grid->counts;
        one::Flight macro(f,start,settings);
        grid->counts={};macro.selectHistory(3.25);const auto macroCounts=grid->counts;
        require(fineCounts.fineBoundaries>0&&fineCounts.stationaryQueries>0,
                "fine reference actually traverses interpolation cells");
        require(macroCounts.fineBoundaries==0&&macroCounts.stationaryQueries==0&&macroCounts.macroBoundaries>0,
                "macro candidate collection skips fine DDA and stationary-point queries entirely");
        require(macro.work().cells<fine.work().cells,
                "macro candidate model completes fewer canonical candidate fragments");
        compareCandidates(macro.snapshot(),macroCandidateReference(f,start,settings,3.25));
        std::cout<<(surface?"surface":"first-hit")<<" macro candidates width="<<width
                 <<" fine boundaries "<<fineCounts.fineBoundaries<<" -> "<<macroCounts.fineBoundaries
                 <<", stationary queries "<<fineCounts.stationaryQueries<<" -> "<<macroCounts.stationaryQueries
                 <<", fragments "<<fine.work().cells<<" -> "<<macro.work().cells<<"\n";

        one::Flight split(f,start,settings);
        for(double t:{1e-6,.071,.117,.137,.249,.499,.5,.509,.873,1.237,2.001,3.25}) {
            split.selectHistory(t);
            compareCandidates(split.snapshot(),macroCandidateReference(f,start,settings,t));
            if(split.snapshot().current)
                require(split.snapshot().current->distance<=split.snapshot().fragmentBegin,
                        "partly passed macro fragment never adds a future candidate");
        }
        compareCandidates(split.snapshot(),macro.snapshot());
        one::Flight prefix(f,start,settings);prefix.selectHistory(.873);
        one::Flight resumed(f,start,settings,prefix.snapshot());resumed.selectHistory(3.25);
        compareCandidates(resumed.snapshot(),macro.snapshot());
    }
}

void macroCandidateTrackingChecks() {
    const auto f=field(macroTestGrid(2));
    constexpr double end=3.25;
    for(bool surface:{false,true}) {
        const auto start=macroTestStart(surface);
        auto settings=macroCandidateSettings(4);
        const auto expectedHistory=macroCandidateReference(f,start,settings,end);
        for(int route=0;route<3;++route) {
            // Candidate blocks are four cells wide, independently of either
            // fine tracking or the two-cell certificates used for proposals.
            settings.macrocellSize=route==0 ? 0:2;
            settings.macrocellMaxExpectedEvents=route==2 ? 1e-12:8;
            LargeOpticalDepthSampler source;cg::Random random(source);
            auto flight=std::make_unique<one::Flight>(f,start,settings);
            for(double cut:{.137,.509,.873,1.237,2.001,end}) {
                require(!flight->sampleDistance(cut,random).hit,"macro candidate continuation reaches every transparent cut");
                compareCandidates(flight->snapshot(),macroCandidateReference(f,start,settings,cut));
                flight=std::make_unique<one::Flight>(f,start,settings,flight->snapshot());
            }
            compareCandidates(flight->snapshot(),expectedHistory);
            LargeOpticalDepthSampler wholeSource;cg::Random wholeRandom(wholeSource);
            one::Flight whole(f,start,settings);
            require(!whole.sampleDistance(end,wholeRandom).hit,"macro candidate routing fixture escapes");
            compareCandidates(whole.snapshot(),expectedHistory);
            if(route==0) require(whole.work().fineBoundaryQueries>0,
                                "fine proposal traversal is exercised independently of macro candidates");
            if(route==1) require(whole.work().macroAccepted>0,"macro candidates can use accepted smaller tracking blocks");
            if(route==2) require(whole.work().macroRejected>0&&whole.work().fineBoundaryQueries>0,
                                "rejected tracking blocks fall back without adding fine-cell candidates");
            require(whole.work().macroCandidateFragments==whole.work().cells,
                    "all completed candidate fragments use the macro sampling rule");

            bool hitChecked=false,shadowChecked=false;
            for(int seed=1;seed<=32&&(!hitChecked||!shadowChecked);++seed) {
                if(!hitChecked) {
                    cg::Random rng(5183+seed);one::Flight path(f,start,settings);
                    const auto hit=path.sampleDistance(end,rng);
                    if(hit.hit) {
                        compareCandidates(path.snapshot(),macroCandidateReference(f,start,settings,hit.distance));
                        const auto selected=path.snapshot().selected;
                        const sh::LocalLaw exact(f,start,hit.distance,
                            selected?std::vector<double>{*selected}:std::vector<double>{});
                        close(hit.law->hazard(),exact.hazard(),1e-10*(1+exact.hazard()),
                              "macro candidate distance uses the direct same-history local law");
                        const double speed=hit.law->sampleSpeed(.5);
                        close(exact.speedCdf(speed),.5,3e-4,
                              "macro candidate collision velocity uses the accepted history");
                        hitChecked=true;
                    }
                }
                if(!shadowChecked) {
                    auto roulette=settings;roulette.shadowRouletteThreshold=.99;
                    cg::Random rng(6163+seed);one::Flight shadow(f,start,roulette);
                    shadow.transmittance(end,rng);
                    if(shadow.work().rouletteTerminations) {
                        compareCandidates(shadow.snapshot(),macroCandidateReference(f,start,settings,shadow.snapshot().age));
                        shadowChecked=true;
                    }
                }
            }
            require(hitChecked&&shadowChecked,"macro candidate fixture exercises collision and early shadow termination");
        }
        settings.profile=false;settings.profileDetail=false;settings.macrocellMaxExpectedEvents=8;
        const double a=macroOpticalReference(f,start,end,8,settings);
        const double b=macroOpticalReference(f,start,end,16,settings);
        close(a,b,5e-4,"independent hazard quadrature converges for the macro candidate model");
        const double expected=std::exp(-b);
        constexpr int count=300;
        cg::Random random(surface?87151:91079);int escaped=0;double sum=0,square=0;
        for(int i=0;i<count;++i) {
            one::Flight path(f,start,settings);escaped+=!path.sampleDistance(end,random).hit;
            one::Flight shadow(f,start,settings);const double tr=shadow.transmittance(end,random);
            require(tr>=0&&tr<=1&&std::isfinite(tr),"macro candidate ratio weight is finite and nonnegative");
            sum+=tr;square+=tr*tr;
        }
        close(escaped/double(count),expected,6*std::sqrt(expected*(1-expected)/count)+.002,
              "macro candidate collision samples match their independent transmittance model");
        const double variance=std::max(0.,(square-sum*sum/count)/(count-1));
        close(sum/count,expected,6*std::sqrt(variance/count)+.002,
              "macro candidate shadows match the same independent transmittance model");
        std::cout<<(surface?"surface":"first-hit")<<" macro candidate expected T="<<expected
                 <<" delta T="<<escaped/double(count)<<" ratio T="<<sum/count<<"\n";
    }
}
#endif

std::shared_ptr<Tungsten::SingleHistoryGpisMedium> makeMedium(const std::string &material,
                                                            const std::string &options="") {
    using namespace Tungsten;
    const std::string json=R"({"type":"single_history_gpis","max_bounces":8,
      "gaussian_process":{"type":"standard","mean":{"type":"homogeneous","offset":0.4},
      "covariance":{"type":"squared_exponential","sigma":1,"lengthScale":1}},
      "single_history":{"selection_step_over_length":0.5,"history_groups":8,
        "shadow_roulette_threshold":0)"+options+R"(},
      "material":{"type":")"+material+R"(","albedo":[0.2,0.5,0.8]}})";
    JsonDocument document(Path("single-history-medium-test.json"),json);
    Scene scene;
    auto medium=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
    require(bool(medium),"single-history medium is registered independently");
    medium->loadResources();medium->prepareForRender();
    return medium;
}

void majorantChecks() {
    majorantModelChecks();
    majorantCoverageChecks();
    majorantTrackingChecks();
    largeHistoryBoundRefinementChecks();
    nearBirthRefinementChecks();
    using namespace Tungsten;
    auto medium=makeMedium("mirror",R"(,"normalized_bound_only":true,
        "far_bound_start_scale":4,"tighten_expected_events":8)");
    const auto &settings=medium->flightSettings();
    require(settings.normalizedBoundOnly&&settings.farBoundStartScale==4&&
            settings.tightenExpectedEvents==8,"JSON enables all three bound optimizations");
    rapidjson::Document arena;
    const auto json=medium->toJson(arena.GetAllocator());
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    JsonDocument document(Path("single-history-majorant-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto restored=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
    require(restored&&restored->flightSettings().normalizedBoundOnly&&
            restored->flightSettings().farBoundStartScale==4&&
            restored->flightSettings().tightenExpectedEvents==8,
            "bound settings survive a complete JSON roundtrip");
    medium->teardownAfterRender();
    auto legacy=makeMedium("mirror");
    require(!legacy->flightSettings().normalizedBoundOnly&&
            legacy->flightSettings().farBoundStartScale==0&&
            legacy->flightSettings().tightenExpectedEvents==0,
            "omitting bound settings preserves the existing defaults");
    legacy->teardownAfterRender();
    std::cout<<"Majorant coverage, model preservation, tracking and JSON checks passed\n";
}

void macroChecks() {
#if OPENVDB_AVAILABLE
    macroRoutingChecks();
    macroTrackingChecks();
#endif
    // Analytic means do not provide grid macro certificates. Enabling the
    // option must leave their ordinary conditional/positive-origin path usable.
    const auto f=field(std::make_shared<cg::ConstantMean>(.4));
    for(bool surface:{false,true}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();
        one::Settings settings;settings.macrocellSize=4;settings.shadowRouletteThreshold=0;
        settings.profile=true;settings.profileDetail=true;
        LargeOpticalDepthSampler source;cg::Random random(source);
        one::Flight path(f,start,settings),reference(f,start);
        require(!path.sampleDistance(2.75,random).hit,"unsupported analytic macro query falls back to fine tracking");
        require(path.work().macroAccepted==0,
                "analytic means without coarse certificates retain ordinary tracking");
        reference.selectHistory(2.75);compareCandidates(path.snapshot(),reference.snapshot());
    }
    for(int invalid:{-2,1,3,128}) {
        one::Settings settings;settings.macrocellSize=invalid;
        bool rejected=false;try {settings.validate();} catch(const std::exception &) {rejected=true;}
        require(rejected,"macrocell size validation rejects unsupported widths");
    }
    for(double invalid:{-1.,0.,std::numeric_limits<double>::infinity(),
                        std::numeric_limits<double>::quiet_NaN()}) {
        one::Settings settings;settings.macrocellMaxExpectedEvents=invalid;
        bool rejected=false;try {settings.validate();} catch(const std::exception &) {rejected=true;}
        require(rejected,"macrocell event budget validation rejects nonpositive or nonfinite values");
    }
    using namespace Tungsten;
    auto medium=makeMedium("mirror",R"(,"macrocell_size":4,"macrocell_max_expected_events":0.75)");
    require(medium->flightSettings().macrocellSize==4&&
            medium->flightSettings().macrocellMaxExpectedEvents==.75,"JSON enables macro tracking controls");
    rapidjson::Document arena;
    const auto json=medium->toJson(arena.GetAllocator());
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    JsonDocument document(Path("single-history-macro-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto restored=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
    require(restored&&restored->flightSettings().macrocellSize==4&&
            restored->flightSettings().macrocellMaxExpectedEvents==.75,
            "macro tracking controls survive a complete JSON roundtrip");
    medium->teardownAfterRender();
    auto legacy=makeMedium("mirror");
    require(legacy->flightSettings().macrocellSize==0&&
            legacy->flightSettings().macrocellMaxExpectedEvents==8,
            "omitted macro controls retain the disabled default");
    legacy->teardownAfterRender();
    std::cout<<"Macro routing, history, continuation, tracking and JSON checks passed\n";
}

void pruningChecks() {
#if OPENVDB_AVAILABLE
    pruningGeometryChecks();
    pruningTrackingChecks();
#endif
    {
        one::Settings settings;settings.candidateMacroPruning=true;
        bool rejected=false;try {settings.validate();} catch(const std::exception &) {rejected=true;}
        require(rejected,"candidate macro pruning requires an enabled macrocell size");
    }
    for(auto method:{one::TrackingMethod::Inversion,one::TrackingMethod::Regular}) {
        one::Settings settings;settings.candidateMacroPruning=true;settings.macrocellSize=4;
        settings.trackingMethod=method;
        bool rejected=false;try {settings.validate();} catch(const std::exception &) {rejected=true;}
        require(rejected,"candidate macro pruning is explicitly restricted to delta tracking");
    }
    using namespace Tungsten;
    auto medium=makeMedium("mirror",R"(,"macrocell_size":4,"candidate_macro_pruning":true)");
    require(medium->flightSettings().candidateMacroPruning,"JSON enables conservative candidate macro pruning");
    rapidjson::Document arena;
    const auto json=medium->toJson(arena.GetAllocator());
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    JsonDocument document(Path("single-history-pruning-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto restored=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
    require(restored&&restored->flightSettings().candidateMacroPruning&&
            restored->flightSettings().macrocellSize==4,
            "candidate pruning and its macrocell prerequisite survive JSON roundtrip");
    medium->teardownAfterRender();
    auto legacy=makeMedium("mirror");
    require(!legacy->flightSettings().candidateMacroPruning,"candidate pruning remains opt-in");
    legacy->teardownAfterRender();
    std::cout<<"Conservative candidate pruning, same-seed transport and JSON checks passed\n";
}

void macroCandidateChecks() {
#if OPENVDB_AVAILABLE
    macroCandidateGeometryChecks();
    macroCandidateTrackingChecks();
#endif
    one::Settings settings;settings.candidateCollection=one::CandidateCollection::Macro;
    settings.validate(); // Candidate macro geometry is independent of tracking acceleration.
    const auto analytic=field(std::make_shared<cg::ConstantMean>(.4));
    bool unsupported=false;
    try {one::Flight flight(analytic,sh::Start{},settings);flight.selectHistory(.5);}
    catch(const std::exception &) {unsupported=true;}
    require(unsupported,"unsupported macro candidate geometry fails explicitly instead of substituting fine candidates");
    for(auto method:{one::TrackingMethod::Inversion,one::TrackingMethod::Regular}) {
        auto invalid=settings;invalid.trackingMethod=method;bool rejected=false;
        try {invalid.validate();} catch(const std::exception &) {rejected=true;}
        require(rejected,"macro candidate model is explicitly restricted to supported delta tracking");
    }
    auto conflicting=settings;conflicting.macrocellSize=4;conflicting.candidateMacroPruning=true;
    bool rejected=false;try {conflicting.validate();} catch(const std::exception &) {rejected=true;}
    require(rejected,"macro candidate sampling and exact fine-candidate pruning cannot be mixed");
    for(int width:{0,1,3,128}) {
        auto invalid=settings;invalid.candidateMacrocellSize=width;rejected=false;
        try {invalid.validate();} catch(const std::exception &) {rejected=true;}
        require(rejected,"candidate macro width is validated independently of tracking block width");
    }
    using namespace Tungsten;
    auto medium=makeMedium("mirror",R"(,"candidate_collection":"macro","candidate_macrocell_size":8)");
    require(medium->flightSettings().candidateCollection==one::CandidateCollection::Macro&&
            medium->flightSettings().candidateMacrocellSize==8&&medium->flightSettings().macrocellSize==0,
            "JSON enables macro candidates independently of tracking blocks");
    rapidjson::Document arena;const auto json=medium->toJson(arena.GetAllocator());
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);json.Accept(writer);
    JsonDocument document(Path("single-history-macro-candidates-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto restored=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
    require(restored&&restored->flightSettings().candidateCollection==one::CandidateCollection::Macro&&
            restored->flightSettings().candidateMacrocellSize==8&&restored->flightSettings().macrocellSize==0,
            "macro candidate selection and its independent width survive JSON roundtrip");
    medium->teardownAfterRender();
    auto legacy=makeMedium("mirror");
    require(legacy->flightSettings().candidateCollection==one::CandidateCollection::Fine&&
            legacy->flightSettings().candidateMacrocellSize==4,"fine candidate collection remains the default");
    legacy->teardownAfterRender();
    std::cout<<"Macro candidate geometry, causal history, transport and JSON checks passed\n";
}

void inversionMediumChecks() {
    using namespace Tungsten;
    auto medium=makeMedium("mirror",R"(,"tracking_method":"inversion",
        "inversion_max_step_over_length":0.2,"inversion_mean_range_over_sigma":0.3,
        "inversion_absolute_tolerance":0.00002,"inversion_relative_tolerance":0.002,
        "inversion_distance_tolerance_over_length":0.00004)");
    const auto &settings=medium->flightSettings();
    require(settings.trackingMethod==one::TrackingMethod::Inversion,"JSON enables the inversion backend");
    rapidjson::Document arena;
    const auto json=medium->toJson(arena.GetAllocator());
    const auto &options=json["single_history"];
    require(std::string(options["tracking_method"].GetString())=="inversion","tracking backend survives serialization");
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    JsonDocument restoredDocument(Path("single-history-inversion-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto restored=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(restoredDocument));
    require(restored&&restored->flightSettings().trackingMethod==settings.trackingMethod,
            "tracking backend survives a complete JSON roundtrip");
    const auto &roundtrip=restored->flightSettings();
    close(roundtrip.inversionMaxStepOverLength,.2,0,"inversion maximum step roundtrip");
    close(roundtrip.inversionMeanRangeOverSigma,.3,0,"inversion mean-range roundtrip");
    close(roundtrip.inversionAbsoluteTolerance,2e-5,0,"inversion absolute tolerance roundtrip");
    close(roundtrip.inversionRelativeTolerance,.002,0,"inversion relative tolerance roundtrip");
    close(roundtrip.inversionDistanceToleranceOverLength,4e-5,0,"inversion distance tolerance roundtrip");

    IndependentPathSampler sampler(8497,false);sampler.startPath(0,0);
    Medium::MediumState state{};MediumSample sample{};
    const Ray firstRay(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,.001f);
    bool escaped=false;
    for(int attempt=0;attempt<32&&!escaped;++attempt) {
        state.reset();require(medium->sampleDistance(sampler,firstRay,state,sample),"valid inversion first segment");
        escaped=sample.exited;
    }
    require(escaped,"inversion medium crosses a transparent boundary");
    const auto saved=std::dynamic_pointer_cast<const SingleHistoryPathContext>(state.pathContext);
    require(saved&&saved->snapshot.inversionRemaining>0,"medium retains its inversion threshold on continuation");
    const double remaining=saved->snapshot.inversionRemaining;
    auto shadowState=state;
    const Ray shadowRay(sample.p,Vec3f(0.f,0.f,1.f),0,.003f);
    const double tr=medium->transmittance(sampler,shadowRay,false,false,&shadowState).x();
    require(tr>0&&tr<=1,"inversion medium returns a finite shadow transmittance");
    const auto shadow=std::dynamic_pointer_cast<const SingleHistoryPathContext>(shadowState.pathContext);
    require(shadow&&shadow->snapshot.inversionRemaining<0,
            "shadow branch does not inherit a path's sampled collision threshold");
    require(saved->snapshot.inversionRemaining==remaining&&state.pathContext==saved,
            "shadow inversion leaves the incoming path snapshot unchanged");
    medium->teardownAfterRender();
    auto defaultMedium=makeMedium("mirror");
    require(defaultMedium->flightSettings().trackingMethod==one::TrackingMethod::Delta,
            "omitted tracking_method preserves the delta default");
    defaultMedium->teardownAfterRender();
    std::cout<<"Inversion JSON, medium continuation and shadow independence checks passed\n";
}

void regularMediumChecks() {
    using namespace Tungsten;
    auto medium=makeMedium("mirror",R"(,"tracking_method":"regular","regular_subdivisions":3)");
    require(medium->flightSettings().trackingMethod==one::TrackingMethod::Regular&&
            medium->flightSettings().regularSubdivisions==3,"JSON enables regular tracking and its subdivision count");
    rapidjson::Document arena;
    const auto json=medium->toJson(arena.GetAllocator());
    const auto &options=json["single_history"];
    require(std::string(options["tracking_method"].GetString())=="regular"&&
            options["regular_subdivisions"].GetInt()==3,"regular settings survive serialization");
    rapidjson::StringBuffer buffer;rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    json.Accept(writer);
    JsonDocument document(Path("single-history-regular-roundtrip.json"),buffer.GetString());
    Scene scene;
    const auto restored=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(document));
    require(restored&&restored->flightSettings().trackingMethod==one::TrackingMethod::Regular&&
            restored->flightSettings().regularSubdivisions==3,"regular settings survive a complete JSON roundtrip");
    medium->teardownAfterRender();
    std::cout<<"Regular medium JSON checks passed\n";
}

void mediumChecks() {
    using namespace Tungsten;
    for(const std::string material:{"mirror","lambert"}) {
        auto medium=makeMedium(material);
        IndependentPathSampler sampler(8497,false);sampler.startPath(0,0);
        Medium::MediumState state{};MediumSample sample{};
        const Ray shortRay(Vec3f(0.f),Vec3f(0.f,0.f,1.f),.001f,.002f);
        bool escaped=false;
        for(int attempt=0;attempt<32&&!escaped;++attempt) {
            state.reset();
            require(medium->sampleDistance(sampler,shortRay,state,sample),"valid short first flight");
            escaped=sample.exited;
        }
        require(escaped,"first flight crosses a transparent boundary");
        const auto next=std::dynamic_pointer_cast<const SingleHistoryPathContext>(state.pathContext);
        require(next&&next->start.kind==sh::Start::ExteriorPositiveOnly,
                "camera continuation retains a positive-only origin");
        close(next->snapshot.age,double(.002f),1e-12,"boundary keeps accumulated physical flight age");
        require(next->start.origin==V::Zero(),"boundary keeps the original camera observation");
        const auto saved=state.pathContext;
        auto shadow=state;
        const Ray shadowRay(sample.p,Vec3f(0.f,0.f,1.f),0,.003f);
        const double tr=medium->transmittance(sampler,shadowRay,false,false,&shadow).x();
        require(tr>=0&&tr<=1&&state.pathContext==saved,"shadow uses an independent immutable snapshot");
        const auto shadowEnd=std::dynamic_pointer_cast<const SingleHistoryPathContext>(shadow.pathContext);
        require(bool(shadowEnd),"shadow produces a continuation context");
        close(shadowEnd->snapshot.age,double(.002f)+double(.003f),1e-12,
              "straight shadow continues from the original birth");
        auto turn=state;
        const Ray turned(sample.p,Vec3f(1.f,0.f,0.f),0,.003f);
        medium->transmittance(sampler,turned,false,false,&turn);
        const auto turnEnd=std::dynamic_pointer_cast<const SingleHistoryPathContext>(turn.pathContext);
        require(turnEnd&&turnEnd->start.kind==sh::Start::ExteriorPositiveOnly,
                "ordinary geometric turn does not invent a zero GP value");
        close(turnEnd->snapshot.age,double(.003f),1e-12,"ordinary turn resets the collinear history");

        const Ray incoming(Vec3f(0.f),Vec3f(0.f,0.f,1.f),0,3);
        bool collided=false;
        for(int attempt=0;attempt<32&&!collided;++attempt) {
            state.reset();
            require(medium->sampleDistance(sampler,incoming,state,sample),"valid full first flight");
            collided=!sample.exited;
        }
        require(collided,"first GP collision is sampled");
        PhaseSample scatter;
        require(sample.phase->sample(sampler,incoming.dir(),sample,scatter),"history-aware phase sample");
        const auto successor=std::dynamic_pointer_cast<const SingleHistoryPathContext>(scatter.pathContext);
        require(successor&&successor->start.kind==sh::Start::SurfaceValueGradient,
                "collision successor uses a zero-value known-gradient birth");
        const V outgoing(scatter.w.x(),scatter.w.y(),scatter.w.z());
        require(outgoing.dot(successor->start.gradient)>0,"outgoing direction enters the empty side");
        require(std::abs(successor->start.gradient.norm()-1)>1e-5,
                "successor retains the unnormalized sampled gradient");
        require(successor->snapshot.age==0&&successor->snapshot.completed.empty(),
                "real GP collision starts a fresh history cache");
        std::shared_ptr<const MediumPathContext> connection;
        const Vec3f value=sample.phase->evalConnection(sampler,incoming.dir(),scatter.w,sample,connection);
        require(value.max()>0&&bool(connection),"NEE uses the collision-conditioned phase");
        const auto light=std::dynamic_pointer_cast<const SingleHistoryPathContext>(connection);
        require(light&&light->start.kind==sh::Start::SurfaceValueGradient,
                "NEE creates a surface-conditioned flight");
        if(material=="lambert")
            require(light->start.gradient==successor->start.gradient,
                    "Lambert branches share the observed full gradient");
        auto secondState=state;secondState.pathContext=successor;
        const Ray reflected(sample.p,scatter.w,0,.1f);
        require(medium->sampleDistance(sampler,reflected,secondState,sample),
                "conditioned subsequent flight is supported");
        medium->teardownAfterRender();
    }
    std::cout<<"Medium first-hit, continuation, scattering and NEE checks passed\n";
}

void imageCheck(const char *path) {
    using namespace Tungsten;
    int width=0,height=0;
    const auto pixels=ImageIO::loadHdr(Path(path),TexelConversion::REQUEST_RGB,width,height);
    require(bool(pixels)&&width>0&&height>0,"load single-history smoke render");
    double sum=0,minimum=1e30,maximum=0;
    for(int i=0;i<3*width*height;++i) {
        const double x=pixels[i];
        require(x>=0&&std::isfinite(x),"smoke image is finite and nonnegative");
        sum+=x;minimum=std::min(minimum,x);maximum=std::max(maximum,x);
    }
    require(sum/(3*width*height)>.02&&maximum-minimum>.2,"smoke image has lit geometry");
    std::cout<<"Image "<<width<<"x"<<height<<" mean="<<sum/(3*width*height)
             <<" range="<<minimum<<","<<maximum<<"\n";
}

void benchmark(int count) {
    require(count>0&&count<=100000,"benchmark count must be in [1,100000]");
    const auto f=field(std::make_shared<cg::ConstantMean>(.4));
    for(bool surface:{false,true}) for(bool shadow:{false,true}) {
        sh::Start start;if(surface) start.kind=sh::Start::SurfaceValueGradient;
        start.gradient=V::UnitZ();one::Settings settings;settings.profile=true;
        cg::Random rng(7341);one::Work work;int hits=0;double sum=0;
        const auto begin=std::chrono::steady_clock::now();
        for(int i=0;i<count;++i) {
            one::Flight flight(f,start,settings);
            if(shadow) sum+=flight.transmittance(3,rng);
            else {const auto hit=flight.sampleDistance(3,rng);hits+=hit.hit;sum+=hit.distance;}
            const auto &w=flight.work();
            work.intervals+=w.intervals;work.candidates+=w.candidates;
            work.selections+=w.selections;work.cells+=w.cells;
            work.selectionSeconds+=w.selectionSeconds;work.majorantSeconds+=w.majorantSeconds;
            work.probabilitySeconds+=w.probabilitySeconds;
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
        std::cout<<std::setprecision(10)<<"{\"birth\":\""<<(surface?"surface":"first_hit")
                 <<"\",\"operation\":\""<<(shadow?"ratio_shadow":"delta_distance")
                 <<"\",\"flights\":"<<count<<",\"seconds\":"<<seconds
                 <<",\"selection_seconds\":"<<work.selectionSeconds
                 <<",\"majorant_seconds\":"<<work.majorantSeconds
                 <<",\"probability_seconds\":"<<work.probabilitySeconds
                 <<",\"candidates\":"<<work.candidates<<",\"intervals\":"<<work.intervals
                 <<",\"selections\":"<<work.selections<<",\"cells\":"<<work.cells
                 <<",\"hits\":"<<hits<<",\"mean_result\":"<<sum/count<<"}\n";
    }
}
} // namespace

int main(int argc,char **argv) {
    try {
        const std::string mode=argc>1?argv[1]:"all";
        if(mode=="narrow_reference_export") {
            if(argc<3) throw std::runtime_error("narrow_reference_export requires an output JSON path");
            SingleHistoryReferenceExport::runNarrow(argv[2]);return 0;
        }
        if(mode=="reference_export") {
            if(argc<3) throw std::runtime_error("reference_export requires an output JSON path");
            SingleHistoryReferenceExport::run(argv[2]);return 0;
        }
        if(mode=="benchmark") {benchmark(argc>2?std::stoi(argv[2]):64);return 0;}
        if(mode=="image"&&argc>2) {imageCheck(argv[2]);return 0;}
        if(mode=="all"||mode=="local") localChecks();
        if(mode=="all"||mode=="selector") selectorChecks();
        if(mode=="all"||mode=="tracking") trackingChecks();
        if(mode=="all"||mode=="majorant") majorantChecks();
        if(mode=="all"||mode=="macro") macroChecks();
        if(mode=="all"||mode=="pruning") pruningChecks();
        if(mode=="all"||mode=="macro_candidates") macroCandidateChecks();
        if(mode=="all"||mode=="inversion") {inversionChecks();inversionMediumChecks();}
        if(mode=="all"||mode=="regular") {regularChecks();regularMediumChecks();}
        if(mode=="all"||mode=="medium") mediumChecks();
        if(mode!="all"&&mode!="local"&&mode!="selector"&&mode!="tracking"&&mode!="majorant"&&mode!="macro"&&mode!="pruning"&&mode!="macro_candidates"&&mode!="inversion"&&mode!="regular"&&mode!="medium")
            throw std::runtime_error("Unknown single-history test mode");
        return 0;
    } catch(const std::exception &error) {
        std::cerr<<error.what()<<"\n";
        return 1;
    }
}
