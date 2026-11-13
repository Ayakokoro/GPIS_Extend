#pragma once
#include <chrono>
#include <iomanip>

namespace {
void localCovarianceChecks() {
    sh::ProbabilitySettings reference,analytic;
    reference.relativeTolerance=analytic.relativeTolerance=2e-5;
    reference.maxSamples=analytic.maxSamples=32768;
    reference.referenceCovariance=true;
    analytic.statistics=std::make_shared<sh::ProbabilityStatistics>();
    std::size_t hazards=0,cdfs=0;
    double maxHazardRelative=0,maxCdfAbsolute=0;
    const auto compare=[&](const cg::GPSSField &field,const sh::Start &start,double t,
                           const std::vector<double> &history,bool checkCdf) {
        const sh::LocalLaw a(field,start,t,history,analytic),b(field,start,t,history,reference);
        const double actual=a.hazard(),expected=b.hazard();
        maxHazardRelative=std::max(maxHazardRelative,std::abs(actual-expected)/std::max(1e-100,expected));
        close(actual,expected,1e-10+5e-5*std::abs(expected),"analytic covariance changed crossing hazard");
        require(a.meanGradient()==b.meanGradient(),"analytic covariance changed mean gradient");
        ++hazards;
        if(checkCdf && expected>1e-100) {
            const double scale=field.kernel.sigma()*std::sqrt(start.direction.normalized().dot(
                field.kernel.precision()*start.direction.normalized()));
            for(double v:{.5,1.5}) {
                const double x=a.speedCdf(scale*v),y=b.speedCdf(scale*v);
                maxCdfAbsolute=std::max(maxCdfAbsolute,std::abs(x-y));
                close(x,y,5e-5,"analytic covariance changed crossing speed CDF");
                ++cdfs;
            }
        }
    };
    const auto constant=makeField(std::make_shared<cg::ConstantMean>(0));
    sh::Start exterior,surface;
    surface.kind=sh::Start::SurfaceValueGradient;surface.gradient=V::UnitZ();
    for(double q:{.02,.1,.4,1.,2.5,8.,8.00001}) {
        compare(constant,exterior,q,{},false);
        compare(constant,exterior,q,{.6*q},q>=.4 && q<=2.5);
        compare(constant,surface,q,{},false);
        compare(constant,surface,q,{.6*q},q>=.4 && q<=2.5);
    }
    for(double q:{.2,1.6,4.}) {
        compare(constant,exterior,q,{.2*q,.7*q},q==1.6);
        compare(constant,surface,q,{.2*q,.7*q},q==1.6);
        compare(constant,exterior,q,{.7*q,.2*q},false);
        compare(constant,surface,q,{.7*q,.2*q},false);
    }
    for(const auto &fixture:sparseHistoryLutCases())
        compare(fixture.field,fixture.start,fixture.distance,fixture.history,true);
    const auto anisotropic=cg::GPSSField{std::make_shared<cg::PlaneMean>(V(.2,.1,1),.3),
        cg::CovarianceKernel(.7,V(1,2,3).asDiagonal()),{V::Constant(-10),V::Constant(10)}};
    sh::Start oblique=surface;oblique.direction=V(.2,.3,1).normalized();oblique.gradient=V(.2,-.1,.7);
    compare(anisotropic,oblique,.8,{.2,.6},true);
    const auto rare=makeField(std::make_shared<cg::ConstantMean>(-8));
    compare(rare,exterior,8,{2},true);
    // The old projection remains the numerical reference near a singular birth.
    sh::Start weak=surface;weak.gradient=V(0,0,1e-4);
    const auto fallbacksBefore=analytic.statistics->featureCovariances.load();
    compare(constant,weak,1e-4,{5e-5,2.5e-5},true);
    require(analytic.statistics->featureCovariances.load()>fallbacksBefore,"singular surface birth must retain feature fallback");
    // Construction at extremely close and duplicated-to-roundoff positions
    // must not inject jitter or silently drop an observed sign.
    for(bool isSurface:{false,true}) for(double q:{1e-8,1.,8.}) {
        const auto &start=isSurface ? surface:exterior;
        const sh::LocalLaw a(constant,start,q,{std::nextafter(q,0.0)},analytic);
        require(a.meanGradient().allFinite(),"nearly coincident law constructs without regularization");
    }
    require(analytic.statistics->analyticCovariances.load()>30,"checks exercise analytic covariance branch");
    std::cout<<std::setprecision(17)<<"{\"hazard_comparisons\":"<<hazards
        <<",\"speed_cdf_comparisons\":"<<cdfs<<",\"max_hazard_relative_error\":"<<maxHazardRelative
        <<",\"max_cdf_absolute_error\":"<<maxCdfAbsolute
        <<",\"analytic_constructions\":"<<analytic.statistics->analyticCovariances.load()
        <<",\"feature_fallbacks\":"<<analytic.statistics->featureCovariances.load()<<"}\n";
}

double localCovarianceBenchLoop(bool reference,int rounds,double &checksum) {
    const auto field=makeField(std::make_shared<cg::ConstantMean>(.4));
    sh::ProbabilitySettings settings;settings.referenceCovariance=reference;
    const auto begin=std::chrono::steady_clock::now();
    double sum=0;
    for(int i=0;i<rounds;++i) {
        const double q=.3+3.7*(i%251)/250.;
        sh::Start start;
        if(i%2) {start.kind=sh::Start::SurfaceValueGradient;start.gradient=V::UnitZ();}
        const std::vector<double> history=i%3==0 ? std::vector<double>{}:
            i%3==1 ? std::vector<double>{.6*q}:std::vector<double>{.2*q,.7*q};
        const sh::LocalLaw law(field,start,q,history,settings);
        sum+=law.meanGradient().squaredNorm()+q;
    }
    checksum=sum;
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
}
void localCovarianceBenchmark() {
    double oldSum=0,newSum=0;
    localCovarianceBenchLoop(true,1000,oldSum);localCovarianceBenchLoop(false,1000,newSum);
    std::vector<double> oldTimes,newTimes;
    for(int i=0;i<5;++i) {
        if(i%2) {
            newTimes.push_back(localCovarianceBenchLoop(false,20000,newSum));
            oldTimes.push_back(localCovarianceBenchLoop(true,20000,oldSum));
        } else {
            oldTimes.push_back(localCovarianceBenchLoop(true,20000,oldSum));
            newTimes.push_back(localCovarianceBenchLoop(false,20000,newSum));
        }
    }
    std::sort(oldTimes.begin(),oldTimes.end());std::sort(newTimes.begin(),newTimes.end());
    require(oldSum==newSum,"construction benchmark executes identical queries");
    std::cout<<std::setprecision(17)<<"{\"constructions_per_repeat\":20000,\"repeats\":5,\"reference_median_seconds\":"
        <<oldTimes[2]<<",\"analytic_median_seconds\":"<<newTimes[2]
        <<",\"construction_speedup\":"<<oldTimes[2]/newTimes[2]<<"}\n";
}
}
