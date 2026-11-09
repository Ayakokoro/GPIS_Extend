#pragma once

#include "math/BivariateNormalCdfTable.hpp"
#include <chrono>
#include <random>

// Included after the parent test's aliases, require/close, makeField, and
// ProfileMean. Every reference below explicitly disables the lookup table.
struct SparseHistoryLutCase {
    const char *name;
    cg::GPSSField field;
    sh::Start start;
    double distance;
    std::vector<double> history;
};

inline std::vector<SparseHistoryLutCase> sparseHistoryLutCases() {
    const auto constant=makeField(std::make_shared<cg::ConstantMean>(.4));
    const auto profile=makeField(std::make_shared<ProfileMean>());
    sh::Start exterior,surface;
    surface.kind=sh::Start::SurfaceValueGradient;surface.gradient=V::UnitZ();
    std::vector<SparseHistoryLutCase> result{
        {"exterior_K1",constant,exterior,2.4,{1.4}},
        {"exterior_K2",constant,exterior,2.4,{.7,1.4}},
        {"surface_K1",profile,surface,2.625,{1.5}},
        {"surface_K2",profile,surface,2.625,{1.,1.5}}
    };
    const auto sphere=makeField(std::make_shared<cg::SphereMean>(V::Zero(),1),.05,.2);
    exterior.origin=V(1.02,0,-2);
    result.push_back({"grazing_exterior_K1",sphere,exterior,1.68928,{1.48928}});
    result.push_back({"grazing_exterior_K2",sphere,exterior,1.68928,{1.28928,1.48928}});
    return result;
}

inline void sparseHistoryLutChecks(const char *tablePath) {
    const auto table=sh::BivariateNormalCdfTable::loadShared(tablePath);
    require(bool(table),"load bivariate normal table");
    sh::ProbabilitySettings reference,lookup;
    reference.relativeTolerance=2e-5;
    reference.absoluteTolerance=1e-11;
    reference.maxSamples=32768;
    lookup.bivariateTable=table;
    lookup.statistics=std::make_shared<sh::ProbabilityStatistics>();
    std::mt19937_64 rng(721648);
    std::uniform_real_distribution<double> mean(-2.5,2.5),correlation(-.9,.9);
    int rawHits=0;
    double maximumProbabilityError=0;
    for(int i=0;i<160;++i) {
        const double a=mean(rng),b=mean(rng),rho=correlation(rng);
        Eigen::Matrix2d covariance;covariance<<1,rho,rho,1;
        const Eigen::Vector2d mu(a,b);
        const double expected=sh::positiveOrthant(mu,covariance,reference);
        const auto estimate=table->lookup(a,b,rho);
        if(estimate) {
            ++rawHits;
            require(std::isfinite(estimate->value)&&std::isfinite(estimate->error)&&estimate->error>=0,
                    "table reports finite value and interpolation error");
            require(estimate->value>=0&&estimate->value<=std::min(cg::normalCdf(a),cg::normalCdf(b))+1e-12,
                    "table respects bivariate probability bounds");
            close(estimate->value,expected,1.05*estimate->error+2e-10,
                  "raw table error estimate encloses independent numerical comparison");
        }
        const double actual=sh::positiveOrthant(mu,covariance,lookup);
        const double error=std::abs(actual-expected);
        maximumProbabilityError=std::max(maximumProbabilityError,error);
        close(actual,expected,std::max(2e-10,2*lookup.relativeTolerance*expected),
              "accepted lookup or fallback meets the probability error budget");
    }
    require(rawHits>0,"random central cases exercise raw table interpolation");
    require(lookup.statistics->tableAccepted.load()>0,"probability wrapper accepts useful table queries");

    // These values are outside the documented table domain. Compare the
    // actual fallback result, including a rare event and both correlation ends.
    const auto acceptedBefore=lookup.statistics->tableAccepted.load();
    for(const auto fixture:{V(-8,.3,.4),V(.4,-.2,.999999),V(.4,-.2,-.999999)}) {
        require(!table->lookup(fixture.x(),fixture.y(),fixture.z()),"tail or near-singular query requests numerical fallback");
        Eigen::Matrix2d covariance;covariance<<1,fixture.z(),fixture.z(),1;
        const Eigen::Vector2d mu=fixture.head<2>();
        const double expected=sh::positiveOrthant(mu,covariance,reference);
        close(sh::positiveOrthant(mu,covariance,lookup),expected,
              std::max(1e-28,2e-6*expected),"tail and near-singular fallback preserves numerical probability");
    }
    require(lookup.statistics->tableAccepted.load()==acceptedBefore,"out-of-domain cases never count as accepted lookups");

    double maximumHazardRelativeError=0,maximumSpeedCdfError=0;
    for(const auto &fixture:sparseHistoryLutCases()) {
        const sh::LocalLaw exact(fixture.field,fixture.start,fixture.distance,fixture.history,reference);
        const sh::LocalLaw fast(fixture.field,fixture.start,fixture.distance,fixture.history,lookup);
        const double expected=exact.hazard(),actual=fast.hazard();
        const double relative=std::abs(actual-expected)/std::max(expected,1e-12);
        maximumHazardRelativeError=std::max(maximumHazardRelativeError,relative);
        close(actual,expected,1e-9+8*lookup.relativeTolerance*std::abs(expected),
              "same-history table hazard agrees with numerical reference");
        const double speedScale=fixture.field.kernel.sigma()*
            std::sqrt(fixture.field.kernel.metric().trace()/3);
        for(double scale:{.5,1.5,3.0,16.0}) {
            const double v=scale*speedScale;
            const double p=exact.speedCdf(v),q=fast.speedCdf(v);
            maximumSpeedCdfError=std::max(maximumSpeedCdfError,std::abs(q-p));
            close(q,p,8*lookup.relativeTolerance+2e-7,
                  "finite-speed CDF agrees with independently normalized reference");
        }
        // A finite far-tail argument catches a wrong normalizer hidden by
        // speedCdf(infinity)'s explicit return-one convention.
        close(fast.speedCdf(32*speedScale),1,8*lookup.relativeTolerance+2e-7,
              "finite far-speed CDF reaches unit mass");
    }
    const auto cases=sparseHistoryLutCases();
    const auto &surfaceOne=cases[2];
    const sh::LocalLaw exactOne(surfaceOne.field,surfaceOne.start,surfaceOne.distance,surfaceOne.history,reference);
    const sh::LocalLaw fastOne(surfaceOne.field,surfaceOne.start,surfaceOne.distance,surfaceOne.history,lookup);
    for(double probability:{.5,.99999}) {
        const double v=fastOne.sampleSpeed(probability);
        close(exactOne.speedCdf(v),probability,8*lookup.relativeTolerance+2e-7,
              "table-enabled inverse speed sampler agrees with numerical target CDF");
    }
    std::cout<<"LUT checks: probability_max_abs="<<maximumProbabilityError
             <<" hazard_max_rel="<<maximumHazardRelativeError
             <<" speed_cdf_max_abs="<<maximumSpeedCdfError
             <<" accepted="<<lookup.statistics->tableAccepted.load()
             <<" attempts="<<lookup.statistics->tableAttempts.load()<<"\n";
}

inline void sparseHistoryLutBenchmark(const char *tablePath) {
    const auto table=sh::BivariateNormalCdfTable::loadShared(tablePath);
    sh::ProbabilitySettings reference,lookup;
    lookup.bivariateTable=table;
    lookup.statistics=std::make_shared<sh::ProbabilityStatistics>();
    std::cout<<"LUT benchmark: same history, geometry, tolerance; load excluded; table_bytes="
             <<table->sizeBytes()<<"\n";
    std::cout<<"case,numerical_ms,lut_ms,hazard_reference,hazard_lut,relative_error,table_accepted,table_attempts\n";
    // Time the actual local-law construction and hazard, not cached repeated
    // hazard() calls. Speed is reported, never asserted as a correctness test.
    for(const auto &fixture:sparseHistoryLutCases()) {
        const auto begin=std::chrono::steady_clock::now();
        const double exact=sh::LocalLaw(fixture.field,fixture.start,fixture.distance,fixture.history,reference).hazard();
        const auto middle=std::chrono::steady_clock::now();
        const auto accepted=lookup.statistics->tableAccepted.load(),attempts=lookup.statistics->tableAttempts.load();
        const double fast=sh::LocalLaw(fixture.field,fixture.start,fixture.distance,fixture.history,lookup).hazard();
        const auto end=std::chrono::steady_clock::now();
        const double exactMs=std::chrono::duration<double,std::milli>(middle-begin).count();
        const double fastMs=std::chrono::duration<double,std::milli>(end-middle).count();
        std::cout<<fixture.name<<','<<exactMs<<','<<fastMs<<','<<exact<<','<<fast<<','
                 <<std::abs(fast-exact)/std::max(1e-12,std::abs(exact))<<','
                 <<lookup.statistics->tableAccepted.load()-accepted<<','
                 <<lookup.statistics->tableAttempts.load()-attempts<<"\n";
    }
}
