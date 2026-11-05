#include "math/SparseHistoryFlight.hpp"
#include "math/conditional/mathutility/NumericPolicy.h"
#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>

namespace Tungsten::SparseHistoryGpis {
namespace cg = ConditionalGpis;
namespace {
[[noreturn]] void invalid(const char *message) {
    throw cg::NumericError(cg::NumericStatus::InvalidInput,message);
}
[[noreturn]] void unresolved(const char *message) {
    throw cg::NumericError(cg::NumericStatus::IntegrationNotConverged,message);
}
bool positiveFinite(double x) { return x>0 && std::isfinite(x); }
}

void FlightSettings::validate() const {
    if(historyPoints<0 || historyPoints>2 || !positiveFinite(maxStepOverLength) ||
       !positiveFinite(maxMeanRangeOverSigma) || !positiveFinite(opticalDepthTolerance) ||
       !positiveFinite(distanceTolerance) || maxSubdivisions<16 || maxSubdivisions>1000000 ||
       !positiveFinite(probability.absoluteTolerance) || !positiveFinite(probability.relativeTolerance) ||
       probability.initialSamples<16 || probability.maxSamples<probability.initialSamples)
        invalid("sparse_history_gpis: invalid flight settings");
}

struct Flight::Impl {
    const cg::GPSSField &field;
    Start birth;
    FlightSettings settings;
    double ell, sigma;
    std::unique_ptr<cg::MeanRayCursor> cursor;
    // Candidate construction advances independently of quadrature/backtracking.
    // Even the polynomial anchor (and hence rounding of its roots) must not
    // depend on which integration or normal query happened first.
    std::unique_ptr<cg::MeanRayCursor> candidateCursor;
    struct Candidate { double t=0,z=std::numeric_limits<double>::infinity(); };
    struct Bucket { std::vector<Candidate> candidates; Candidate best, prefixBest; };
    std::vector<Bucket> buckets;
    std::map<double,double> hazards;

    Impl(const cg::GPSSField &f,const Start &s,const FlightSettings &opts)
        :field(f),birth(s),settings(opts) {
        settings.validate();
        if(!field.mean || !birth.origin.allFinite() || !birth.direction.allFinite() ||
           std::abs(birth.direction.squaredNorm()-1)>1e-10)
            invalid("sparse_history_gpis: flight requires a finite origin and unit direction");
        const auto &metric=field.kernel.metric();
        const double precision=metric.trace()/3;
        if(field.kernel.type()!=cg::CovarianceKernelType::SquaredExponential ||
           !positiveFinite(precision) ||
           (metric-precision*cg::Matrix3::Identity()).cwiseAbs().maxCoeff()>1e-10*precision)
            invalid("sparse_history_gpis: fixed history supports isotropic stationary SE only");
        sigma=field.kernel.sigma(); ell=1/std::sqrt(precision);
        if(!positiveFinite(sigma) || !positiveFinite(ell)) invalid("sparse_history_gpis: invalid field scale");
        if(birth.kind==Start::SurfaceValueGradient &&
           (!birth.gradient.allFinite() || !(birth.gradient.dot(birth.direction)>0)))
            throw cg::NumericError(cg::NumericStatus::NonPositiveSurfaceBirth,
                "sparse_history_gpis: collision-born direction must point into the empty side");
        cursor=field.mean->createRayCursor(birth.origin,birth.direction);
        candidateCursor=field.mean->createRayCursor(birth.origin,birth.direction);
    }

    void validateDistance(double t) const {
        if(!(t>=0) || !std::isfinite(t)) invalid("sparse_history_gpis: invalid physical flight distance");
    }
    cg::MeanJet mean(double t) {
        if(cursor) return cursor->evaluate(t);
        const auto p=(birth.origin+t*birth.direction).eval();
        if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
            const auto radial=(p-sphere->center()).eval();
            const double radius=radial.norm();
            if(radius==0) return {-sphere->radius(),cg::Vector3::Zero()};
        }
        return field.mean->evaluate(p);
    }
    cg::MeanRayBounds bounds(double a,double b) {
        if(cursor) return cursor->rayBounds(a,b);
        // The generic sphere bound evaluates its midpoint; at the center the
        // mean is finite but its gradient is undefined. Analytic radial bounds
        // avoid that query and split geometry/integration at closest approach.
        if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
            const auto radial=(birth.origin-sphere->center()).eval();
            const double closest=std::clamp(-radial.dot(birth.direction),a,b);
            cg::MeanRayBounds result;
            result.minimumValue=(radial+closest*birth.direction).norm()-sphere->radius();
            result.maximumValue=std::max((radial+a*birth.direction).norm(),
                                        (radial+b*birth.direction).norm())-sphere->radius();
            result.certified=true;
            return result;
        }
        return field.mean->rayBounds(birth.origin,birth.direction,a,b);
    }
    static Candidate better(Candidate a,Candidate b) {
        if(!(a.t>0)) return b;
        if(!(b.t>0)) return a;
        return b.z<a.z || (b.z==a.z && b.t<a.t) ? b:a;
    }
    Candidate candidate(double t) {
        if(!(t>0)) return {};
        const double z=standardizedMean(field,birth,t);
        if(std::isnan(z)) invalid("sparse_history_gpis: non-finite candidate score");
        return {t,z};
    }
    std::size_t bucketIndex(double t) const {
        const double index=std::floor(t/ell);
        // Explicit work budget, including candidate discovery over long rays.
        if(index>double(settings.maxSubdivisions)*16)
            unresolved("sparse_history_gpis: history bucket budget exhausted");
        return static_cast<std::size_t>(index);
    }
    void fallbackStationary(double begin,double end,std::vector<double> &points) {
        // Generic procedural means only offer bounds/point queries. These
        // deterministic ell/8 probes are a candidate heuristic, not a claim
        // that arbitrary procedural extrema have been enumerated. Grid roots
        // use the exact polynomial cursor below.
        const int steps=std::max(1,int(std::ceil((end-begin)/(ell/8))));
        for(int i=0;i<steps;++i) {
            const double a=begin+(end-begin)*double(i)/steps;
            const double b=begin+(end-begin)*double(i+1)/steps;
            const double da=mean(a).gradient.dot(birth.direction);
            const double db=mean(b).gradient.dot(birth.direction);
            if(da==0) points.push_back(a);
            if(db==0) points.push_back(b);
            if(da*db<0) {
                double lo=a,hi=b;
                for(int j=0;j<48;++j) {
                    const double mid=lo+0.5*(hi-lo);
                    if(!(mid>lo && mid<hi)) break;
                    const double derivative=mean(mid).gradient.dot(birth.direction);
                    if((derivative<0)==(da<0)) lo=mid; else hi=mid;
                }
                points.push_back(lo+0.5*(hi-lo));
            }
        }
    }
    void ensureBuckets(std::size_t last) {
        while(buckets.size()<=last) {
            const auto index=buckets.size();
            const double begin=double(index)*ell,end=double(index+1)*ell;
            if(!(end>begin) || !std::isfinite(end)) unresolved("sparse_history_gpis: bucket resolution exhausted");
            std::vector<double> points{begin,end};
            if(candidateCursor) {
                int cells=0;
                for(double a=begin;a<end;) {
                    const double b=candidateCursor->nextBoundary(a,end);
                    if(!(b>a) || b>end || ++cells>settings.maxSubdivisions*16)
                        unresolved("sparse_history_gpis: voxel candidate traversal exhausted");
                    points.push_back(b);
                    if(!candidateCursor->appendStationaryPoints(a,b,points)) fallbackStationary(a,b,points);
                    a=b;
                }
            } else {
                field.mean->appendRayBreakpoints(birth.origin,birth.direction,begin,end,points);
                if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
                    const double closest=(sphere->center()-birth.origin).dot(birth.direction);
                    if(closest>begin && closest<end) points.push_back(closest);
                } else fallbackStationary(begin,end,points);
            }
            std::sort(points.begin(),points.end());
            points.erase(std::unique(points.begin(),points.end()),points.end());
            Bucket bucket;
            for(double t:points) if(t>0 && t>=begin && t<end) {
                const auto c=candidate(t);
                bucket.candidates.push_back(c);
                bucket.best=better(bucket.best,c);
            }
            bucket.prefixBest=better(index ? buckets.back().prefixBest:Candidate{},bucket.best);
            buckets.push_back(std::move(bucket));
        }
    }
    std::vector<double> select(double t) {
        validateDistance(t);
        if(t==0 || settings.historyPoints==0) return {};
        const auto current=bucketIndex(t);
        ensureBuckets(current);
        const double recent=t-std::min(ell,t/2);
        const auto recentBucket=bucketIndex(recent);
        std::vector<Candidate> pool;
        const auto earliest=current>2 ? current-2:0;
        for(auto i=earliest;i<=current;++i) {
            Candidate best;
            for(const auto c:buckets[i].candidates) if(c.t<t) best=better(best,c);
            if(best.t>0) pool.push_back(best);
        }
        if(earliest>0) {
            const auto old=buckets[earliest-1].prefixBest;
            if(old.t>0 && old.t<t) pool.push_back(old);
        }
        const auto r=candidate(recent);
        if(settings.historyPoints==1) {
            Candidate best=r;
            for(const auto c:pool) best=better(best,c);
            return {best.t};
        }
        Candidate second;
        for(const auto c:pool)
            if(bucketIndex(c.t)!=recentBucket) second=better(second,c);
        std::vector<double> result{recent};
        if(second.t>0 && second.t<t && second.t!=recent) result.push_back(second.t);
        std::sort(result.begin(),result.end());
        return result;
    }
    double hazard(double t) {
        const auto found=hazards.find(t);
        if(found!=hazards.end()) return found->second;
        const double value=LocalLaw(field,birth,t,select(t),settings.probability).hazard();
        if(!(value>=0) || !std::isfinite(value)) invalid("sparse_history_gpis: invalid local hazard");
        hazards.emplace(t,value);
        return value;
    }
    double nextBoundary(double a,double end) {
        double b=std::min(end,a+ell*settings.maxStepOverLength);
        const auto index=bucketIndex(a);
        if(settings.historyPoints>0) ensureBuckets(index);
        const double bucketEnd=double(index+1)*ell;
        if(bucketEnd>a) b=std::min(b,bucketEnd);
        if(cursor) b=cursor->nextBoundary(a,b);
        else {
            std::vector<double> knots;
            field.mean->appendRayBreakpoints(birth.origin,birth.direction,a,b,knots);
            for(double t:knots) if(t>a && t<b) b=t;
            if(const auto *sphere=dynamic_cast<const cg::SphereMean *>(field.mean.get())) {
                const double closest=(sphere->center()-birth.origin).dot(birth.direction);
                if(closest>a && closest<b) b=closest;
            }
        }
        // A geometry candidate enters the prefix here. Integration never
        // straddles such an insertion or a fixed bucket boundary.
        if(settings.historyPoints>0)
            for(const auto c:buckets[index].candidates) if(c.t>a && c.t<b) b=c.t;
        if(!(b>a)) unresolved("sparse_history_gpis: integration cursor made no progress");
        return b;
    }
    double prepareEnd(double a,double b,int &work) {
        for(;;) {
            if(++work>settings.maxSubdivisions) unresolved("sparse_history_gpis: integration work budget exhausted");
            const auto range=bounds(a,b);
            if(!range.certified || !std::isfinite(range.minimumValue) || !std::isfinite(range.maximumValue))
                invalid("sparse_history_gpis: mean must provide finite interval bounds");
            if(range.maximumValue-range.minimumValue<=sigma*settings.maxMeanRangeOverSigma) return b;
            const double midpoint=a+0.5*(b-a);
            if(!(midpoint>a && midpoint<b)) unresolved("sparse_history_gpis: mean range could not be resolved");
            b=midpoint;
        }
    }
    struct Leaf { double a,b,depth; };
    void integrate(double a,double b,double tolerance,int &work,std::vector<Leaf> &leaves,int recursion=0) {
        if(++work>settings.maxSubdivisions || recursion>48)
            unresolved("sparse_history_gpis: optical depth quadrature exhausted");
        const double h=b-a,mid=a+0.5*h;
        if(!(mid>a && mid<b)) unresolved("sparse_history_gpis: optical depth reached floating point resolution");
        const double coarse=h*hazard(mid);
        const double fine=0.5*h*(hazard(a+0.25*h)+hazard(a+0.75*h));
        const double error=std::abs(fine-coarse)/3;
        const double threshold=std::max(tolerance,settings.probability.relativeTolerance*std::abs(fine));
        if(error<=threshold) {
            leaves.push_back({a,b,std::max(0.0,fine+(fine-coarse)/3)});
            return;
        }
        integrate(a,mid,tolerance/2,work,leaves,recursion+1);
        integrate(mid,b,tolerance/2,work,leaves,recursion+1);
    }
    double intervalDepth(double a,double b,double tolerance,int &work) {
        if(a==b) return 0;
        std::vector<Leaf> leaves;
        integrate(a,b,tolerance,work,leaves);
        double result=0;
        for(const auto &leaf:leaves) result+=leaf.depth;
        return result;
    }
    DistanceSample traverse(double begin,double end,double target) {
        validateDistance(begin); validateDistance(end);
        if(end<begin || !(target>=0) || std::isnan(target)) invalid("sparse_history_gpis: invalid integration interval or target");
        if(begin==end) return {false,end,0};
        if(target==0) return {true,begin,0};
        double total=0;
        int work=0;
        for(double a=begin;a<end;) {
            const double b=prepareEnd(a,nextBoundary(a,end),work);
            std::vector<Leaf> leaves;
            const double tolerance=settings.opticalDepthTolerance*(b-a)/(end-begin);
            integrate(a,b,tolerance,work,leaves);
            for(const auto &leaf:leaves) {
                if(total+leaf.depth>=target) {
                    const double remaining=target-total;
                    double lo=leaf.a,hi=leaf.b;
                    // Invert the same local hazard; the selected history is
                    // recomputed at the eventual collision, never frozen at a
                    // quadrature midpoint. One caller-owned exponential is used.
                    while(hi-lo>settings.distanceTolerance) {
                        const double mid=lo+0.5*(hi-lo);
                        if(!(mid>lo && mid<hi)) break;
                        const double partial=intervalDepth(leaf.a,mid,
                            settings.opticalDepthTolerance*0.25,work);
                        if(partial<remaining) lo=mid; else hi=mid;
                    }
                    return {true,lo+0.5*(hi-lo),target};
                }
                total+=leaf.depth;
            }
            a=b;
        }
        return {false,end,total};
    }
};

Flight::Flight(const cg::GPSSField &field,const Start &start,const FlightSettings &settings)
    :_impl(std::make_unique<Impl>(field,start,settings)) {}
Flight::~Flight()=default;
const Start &Flight::start() const { return _impl->birth; }
double Flight::lengthScale() const { return _impl->ell; }
std::vector<double> Flight::selectHistory(double distance) { return _impl->select(distance); }
LocalLaw Flight::law(double distance) {
    return LocalLaw(_impl->field,_impl->birth,distance,_impl->select(distance),_impl->settings.probability);
}
Eigen::Vector3d Flight::meanGradient(double distance) {
    _impl->validateDistance(distance);
    return _impl->mean(distance).gradient;
}
double Flight::opticalDepth(double begin,double end) {
    return _impl->traverse(begin,end,std::numeric_limits<double>::infinity()).traversedOpticalDepth;
}
DistanceSample Flight::sampleDistance(double begin,double end,double targetOpticalDepth) {
    return _impl->traverse(begin,end,targetOpticalDepth);
}

} // namespace Tungsten::SparseHistoryGpis
