#pragma once
#include <chrono>
#include <iomanip>
#include <limits>
#include <random>

namespace {
// Frozen pre-optimization algorithm: independent numerical reference for the
// scalar covariance path, including reorthogonalization near a singular birth.
class LegacyHistoryScorer {
    double q;
    int count;
    Eigen::VectorXd current,speed;
    Eigen::VectorXd features(double x,bool derivative) const {
        Eigen::VectorXd f=Eigen::VectorXd::Zero(count);
        double previous=0,value=std::exp(-.5*x*x);
        for(int n=0;n<count;++n) {
            if(n) {previous=value;value*=x/std::sqrt(double(n));}
            if(n>=2) f[n]=derivative ? x*value-std::sqrt(double(n))*previous : value;
        }
        return f;
    }
public:
    explicit LegacyHistoryScorer(double distance)
        : q(distance),count(std::max(24,int(std::ceil(q*q+14*q+50)))),
          current(features(q,false)),speed(features(q,true)) {
        current/=current.stableNorm();
        speed-=current.dot(speed)*current;
        speed-=current.dot(speed)*current;
        speed/=speed.stableNorm();
    }
    double score(double u,double mills) const {
        Eigen::VectorXd h=features(u,false);
        h/=h.stableNorm();
        const double r=std::clamp(current.dot(h),0.0,1.0);
        Eigen::VectorXd residual=h-current.dot(h)*current;
        residual-=current.dot(residual)*current;
        const double unexplained=std::clamp(residual.squaredNorm(),0.0,1.0);
        const double gamma=speed.dot(residual);
        const double association=r*r*unexplained+std::min(unexplained,gamma*gamma);
        if(!(association>0)) return 0;
        const long double weighted=static_cast<long double>(mills)*std::sqrt(association);
        return double(std::min(weighted*weighted,static_cast<long double>(std::numeric_limits<double>::max())));
    }
};

void historyScorerChecks() {
    std::size_t comparisons=0,groups=0;
    double maxAbsolute=0,maxRelative=0;
    const auto compare=[&](const sh::HistoryCandidateScorer &actual,const LegacyHistoryScorer &old,
                          double u,double mills) {
        const double a=actual.score(u,mills),b=old.score(u,mills);
        require(a>=0 && std::isfinite(a),"scalar candidate score must be finite and nonnegative");
        const double error=std::abs(a-b);
        maxAbsolute=std::max(maxAbsolute,error);
        if(b>1e-250) maxRelative=std::max(maxRelative,error/b);
        close(a,b,2e-9*std::abs(b)+1e-280,"scalar scorer differs from legacy feature projection");
        ++comparisons;
    };
    // Tiny births, nearly coincident points, the scalar/vector transition,
    // the production minimum ages, and the existing far-analytic transition.
    for(double q:{1e-60,1e-8,1e-4,.005,.01,.02,.03,.049,.05,.1,.4,1.,1.6,4.,8.}) {
        const sh::HistoryCandidateScorer actual(q,true);
        const LegacyHistoryScorer old(q);
        for(double u:{q*.001,q*.1,q*.5,q*.9,q*(1-1e-8),std::nextafter(q,0.0)})
            for(double mills:{0.,1e-100,.01,1.,10.,100.}) compare(actual,old,u,mills);
        // A difficult candidate first must not affect later scalar queries.
        compare(actual,old,q*.4,1.);
    }
    const auto f=makeField(std::make_shared<cg::ConstantMean>(0));
    for(bool surface:{false,true}) {
        sh::Start birth;
        if(surface) {birth.kind=sh::Start::SurfaceValueGradient;birth.gradient=V::UnitZ();}
        for(double q:{.05,1.,8.,8.00001,10.,40.,100.}) for(double fraction:{.1,.5,.99}) {
            const auto expected=sh::historyCandidateStatistics(f,birth,q,q*fraction);
            const double actual=sh::HistoryCandidateScorer(q,surface).score(q*fraction,
                cg::normalPdfOverCdf(expected.standardizedMean));
            close(actual,expected.score,2e-9*std::abs(expected.score)+1e-280,
                  "scorer preserves exterior and far-distance reference policies");
        }
    }
    std::mt19937_64 rng(20261113);
    std::uniform_real_distribution<double> uniform(0,1);
    for(int trial=0;trial<2048;++trial) {
        const double q=std::exp(std::log(.05)+uniform(rng)*std::log(8/.05));
        const sh::HistoryCandidateScorer actual(q,true);
        const LegacyHistoryScorer old(q);
        int winner=-1,oldWinner=-1;
        double best=-1,oldBest=-1;
        for(int i=0;i<12;++i) {
            const double u=.01+(q-.02)*uniform(rng);
            const double mills=cg::normalPdfOverCdf(-8+12*uniform(rng));
            compare(actual,old,u,mills);
            const double a=actual.score(u,mills),b=old.score(u,mills);
            if(a>best) {best=a;winner=i;}
            if(b>oldBest) {oldBest=b;oldWinner=i;}
        }
        require(winner==oldWinner,"analytic score changed selected history candidate");
        ++groups;
    }
    // Equal candidates keep the original strict-'greater than' tie ordering.
    const sh::HistoryCandidateScorer duplicate(1.6,true);
    require(duplicate.score(1.4,2)==duplicate.score(1.4,2),"score deterministic for exact ties");
    bool rejected=false;
    try {duplicate.score(1.6,1);} catch(const cg::NumericError&) {rejected=true;}
    require(rejected,"non-causal candidate must still be rejected");
    std::cout<<std::setprecision(17)<<"{\"scorer_comparisons\":"<<comparisons
             <<",\"identical_winner_groups\":"<<groups<<",\"max_absolute_error\":"<<maxAbsolute
             <<",\"max_relative_error\":"<<maxRelative<<"}\n";
}

template<class Scorer> double historyScorerBenchLoop(int rounds,double &checksum) {
    const auto start=std::chrono::steady_clock::now();
    double sum=0;
    for(int i=0;i<rounds;++i) {
        const double q=.1+3.9*double(i%257)/256;
        const Scorer scorer(q);
        for(int j=1;j<=8;++j) sum+=scorer.score(q*double(j)/9,.5+.125*j);
    }
    checksum=sum;
    return std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();
}
struct ScalarHistoryBench : sh::HistoryCandidateScorer {
    explicit ScalarHistoryBench(double q) : sh::HistoryCandidateScorer(q,true) {}
};
void historyScorerBenchmark() {
    double oldSum=0,newSum=0;
    historyScorerBenchLoop<LegacyHistoryScorer>(1000,oldSum);
    historyScorerBenchLoop<ScalarHistoryBench>(1000,newSum);
    std::vector<double> oldTimes,newTimes;
    for(int repeat=0;repeat<5;++repeat) {
        if(repeat%2) {
            newTimes.push_back(historyScorerBenchLoop<ScalarHistoryBench>(20000,newSum));
            oldTimes.push_back(historyScorerBenchLoop<LegacyHistoryScorer>(20000,oldSum));
        } else {
            oldTimes.push_back(historyScorerBenchLoop<LegacyHistoryScorer>(20000,oldSum));
            newTimes.push_back(historyScorerBenchLoop<ScalarHistoryBench>(20000,newSum));
        }
    }
    close(newSum,oldSum,2e-9*std::abs(oldSum),"benchmark checksum changed");
    std::sort(oldTimes.begin(),oldTimes.end());std::sort(newTimes.begin(),newTimes.end());
    std::cout<<std::setprecision(12)<<"{\"groups\":20000,\"candidates_per_group\":8,\"repeats\":5,"
        <<"\"legacy_median_seconds\":"<<oldTimes[2]<<",\"scalar_median_seconds\":"<<newTimes[2]
        <<",\"speedup\":"<<oldTimes[2]/newTimes[2]<<",\"legacy_checksum\":"<<oldSum
        <<",\"scalar_checksum\":"<<newSum<<"}\n";
}
}
