#include "LocalEllipsoid.hpp"
#include "math/conditional/transport/MajorantOpticalDepth.h"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace Tungsten::LocalEllipsoid {
namespace {
namespace cg = ConditionalGpis;
constexpr double pi = cg::kPi;

void validateJet(const SdfJet &jet) {
    if (!std::isfinite(jet.value) || !jet.gradient.allFinite())
        throw std::runtime_error("poisson_ellipsoid_local: nonfinite SDF value/gradient");
}
double finiteExp(double logarithm) {
    const double value = std::exp(logarithm);
    if (!std::isfinite(value))
        throw std::runtime_error("poisson_ellipsoid_local: rate overflow; check density, particle size and decay_length");
    return value;
}
double checkedAcceptance(double logarithm) {
    if (!std::isfinite(logarithm) || logarithm > 1e-9)
        throw std::runtime_error("poisson_ellipsoid_local: local collision intensity exceeds its certified majorant");
    return std::exp(std::min(0.0, logarithm));
}
double sumWeights(const std::vector<double> &weights) {
    double sum = 0;
    for (double value : weights) sum += value;
    if (!std::isfinite(sum)) throw std::runtime_error("poisson_ellipsoid_local: nonfinite total candidate rate");
    return sum;
}
std::size_t choose(const std::vector<double> &weights, double sum, cg::Random &rng) {
    const double threshold = rng.openUniform01()*sum;
    double cumulative = 0;
    std::size_t last = 0;
    for (std::size_t i=0; i<weights.size(); ++i) {
        if (weights[i] == 0) continue;
        last = i;
        cumulative += weights[i];
        if (threshold < cumulative) return i;
    }
    return last;
}
// Gauss-Legendre integration in the sphere hemisphere cosine, on [0,1].
std::vector<std::pair<double,double>> quadrature(int order) {
    if (order < 2 || order > 512) throw std::invalid_argument("ellipsoid quadrature order must be in [2,512]");
    std::vector<std::pair<double,double>> nodes(order);
    for (int i=0; i<(order+1)/2; ++i) {
        double x=std::cos(pi*(i+0.75)/(order+0.5)), derivative=0;
        for (int iteration=0; iteration<64; ++iteration) {
            double p=1, previous=0;
            for (int n=1; n<=order; ++n) {
                const double next=((2*n-1)*x*p-(n-1)*previous)/n;
                previous=p; p=next;
            }
            derivative=order*(x*p-previous)/(x*x-1);
            const double step=p/derivative;
            x-=step;
            if (std::abs(step)<4*std::numeric_limits<double>::epsilon()) break;
        }
        const double weight=1/((1-x*x)*derivative*derivative);
        nodes[i]={0.5*(1-x),weight};
        nodes[order-1-i]={0.5*(1+x),weight};
    }
    return nodes;
}
}

Grain::Grain(const V &axes, const V &degrees, double numberDensity) : density(numberDensity) {
    if (!axes.allFinite() || axes.minCoeff()<=0 || !degrees.allFinite() ||
        !std::isfinite(density) || density<0)
        throw std::invalid_argument("ellipsoid semiaxes must be positive; density must be nonnegative and all parameters finite");
    const V angle=degrees*(pi/180);
    const M rotation=(Eigen::AngleAxisd(angle.z(),V::UnitZ())*
        Eigen::AngleAxisd(angle.y(),V::UnitY())*Eigen::AngleAxisd(angle.x(),V::UnitX())).toRotationMatrix();
    transform=rotation*axes.asDiagonal();
    inverse=axes.cwiseInverse().asDiagonal()*rotation.transpose();
    determinant=axes.prod();
    radius=axes.maxCoeff();
    normalRadius=transform.row(2).norm();
    maximumArea=pi*determinant/axes.minCoeff();
    if (!transform.allFinite() || !inverse.allFinite() || !(determinant>0) ||
        !std::isfinite(determinant) || !(maximumArea>0) || !std::isfinite(maximumArea))
        throw std::invalid_argument("ellipsoid dimensions exceed numerical range");
}

Model::Model(std::vector<Grain> grains, double decayLength, bool sdfFrame)
    : grains_(std::move(grains)),decayLength_(decayLength),sdfFrame_(sdfFrame) {
    if (grains_.empty() || !(decayLength_>0) || !std::isfinite(decayLength_))
        throw std::invalid_argument("ellipsoid model needs at least one type and a finite positive decay_length");
}

M Model::frame(const V &gradient) const {
    if (!sdfFrame_ || gradient.squaredNorm()==0) return M::Identity();
    const V n=gradient.normalized();
    const V reference=std::abs(n.x())<0.9 ? V::UnitX().eval() : V::UnitY().eval();
    const V x=(reference-n.dot(reference)*n).normalized();
    M result;
    result.col(0)=x; result.col(1)=n.cross(x); result.col(2)=n;
    return result;
}

V Model::Projection::spherePoint(double cosine, double azimuth) const {
    const double sine=std::sqrt(std::max(0.0,1-cosine*cosine));
    return -cosine*incoming+sine*(std::cos(azimuth)*tangent+std::sin(azimuth)*bitangent);
}

Model::Projection Model::projection(std::size_t type,const V &gradient,const V &w) const {
    const auto &grain=grains_.at(type);
    const M rotation=frame(gradient);
    const M inverse=grain.inverse*rotation.transpose();
    Projection p;
    p.transform=rotation*grain.transform;
    p.normalTransform=inverse.transpose();
    const V transformed=inverse*w;
    const double speed=transformed.norm();
    p.incoming=transformed/speed;
    cg::orthonormalComplement(p.incoming,p.tangent,p.bitangent);
    p.area=pi*grain.determinant*speed;
    return p;
}

double Model::typeExtinction(const SdfJet &jet,const V &travel,std::size_t type,int order) const {
    validateJet(jet);
    const auto &grain=grains_.at(type);
    if (grain.density==0) return 0;
    const auto p=projection(type,jet.gradient,cg::normalizedOrThrow(travel));
    const auto nodes=quadrature(order);
    const int azimuths=2*order;
    double mean=0;
    for (const auto &node:nodes) for (int j=0;j<azimuths;++j) {
        const V q=p.transform*p.spherePoint(node.first,2*pi*(j+0.5)/azimuths);
        mean+=2*node.first*node.second/azimuths*
            finiteExp((-jet.value+jet.gradient.dot(q))/decayLength_);
    }
    const double result=grain.density*p.area*mean;
    if (!std::isfinite(result)) throw std::runtime_error("ellipsoid extinction exceeds numerical range");
    return result;
}

double Model::extinction(const SdfJet &jet,const V &travel,int order) const {
    double result=0;
    for (std::size_t i=0;i<grains_.size();++i) result+=typeExtinction(jet,travel,i,order);
    if (!std::isfinite(result)) throw std::runtime_error("ellipsoid extinction exceeds numerical range");
    return result;
}

double Model::normalIntensity(const SdfJet &jet,const V &travel,std::size_t type,const V &normal) const {
    validateJet(jet);
    const V w=cg::normalizedOrThrow(travel),m=cg::normalizedOrThrow(normal);
    const auto &grain=grains_.at(type);
    const double cosine=-w.dot(m);
    if (cosine<=0 || grain.density==0) return 0;
    const M transform=frame(jet.gradient)*grain.transform;
    const V transformed=transform.transpose()*m;
    const double support=transformed.norm();
    const V q=transform*transformed/support;
    // dA/dOmega = det(Q)/(m.Q.m)^2.
    const double jacobian=std::pow(grain.determinant/(support*support),2);
    return finiteExp(std::log(grain.density)+std::log(cosine)+std::log(jacobian)+
                     (-jet.value+jet.gradient.dot(q))/decayLength_);
}

double Model::directionIntensity(const SdfJet &jet,const V &travel,const V &outgoing,std::size_t type) const {
    validateJet(jet);
    const V w=cg::normalizedOrThrow(travel),o=cg::normalizedOrThrow(outgoing);
    const V difference=o-w;
    if (difference.squaredNorm()==0) return 0; // Degenerate grazing half-vector.
    const V m=difference.normalized();
    const auto &grain=grains_.at(type);
    const M transform=frame(jet.gradient)*grain.transform;
    const V transformed=transform.transpose()*m;
    const double support=transformed.norm();
    const V q=transform*transformed/support;
    // Cancel the incident cosine against the reflection Jacobian analytically.
    // This avoids a numerically unstable normalIntensity()/(4*cosine).
    return finiteExp(std::log(grain.density)-std::log(4.0)+
        2*(std::log(grain.determinant)-2*std::log(support))+
        (-jet.value+jet.gradient.dot(q))/decayLength_);
}

double Model::visibleMajorant(const SdfJet &jet,const V &travel) const {
    validateJet(jet);
    const V w=cg::normalizedOrThrow(travel);
    double result=0;
    for (std::size_t i=0;i<grains_.size();++i) {
        const auto &g=grains_[i];
        if (g.density==0) continue;
        const auto p=projection(i,jet.gradient,w);
        const V a=p.transform.transpose()*jet.gradient;
        const double along=a.dot(p.incoming);
        // Maximize a.u on the visible sphere hemisphere u.incoming <= 0.
        const double height=along<=0 ? a.norm() : (a-along*p.incoming).norm();
        result+=finiteExp(std::log(g.density)+std::log(p.area)+
                          (-jet.value+height)/decayLength_+1e-12);
    }
    if (!std::isfinite(result)) throw std::runtime_error("ellipsoid visible majorant overflow");
    return result;
}

double Model::misWeight(double sampledDensity,double otherDensity) {
    if (!(sampledDensity>=0) || !(otherDensity>=0) ||
        !std::isfinite(sampledDensity) || !std::isfinite(otherDensity))
        throw std::runtime_error("ellipsoid MIS needs finite nonnegative densities");
    const double scale=std::max(sampledDensity,otherDensity);
    if (scale==0) return 0;
    const double a=sampledDensity/scale,b=otherDensity/scale;
    return a*a/(a*a+b*b);
}

NormalSample Model::sampleNormal(const SdfJet &jet,const V &travel,cg::Random &rng) const {
    validateJet(jet);
    const V w=cg::normalizedOrThrow(travel);
    std::vector<Projection> projections;
    std::vector<double> logWeights, weights;
    double largest=-std::numeric_limits<double>::infinity();
    for (std::size_t i=0;i<grains_.size();++i) {
        projections.push_back(projection(i,jet.gradient,w));
        const auto &p=projections.back();
        const double height=(p.transform.transpose()*jet.gradient).norm()/decayLength_;
        const double logWeight=grains_[i].density>0 ? std::log(grains_[i].density)+std::log(p.area)+height :
            -std::numeric_limits<double>::infinity();
        logWeights.push_back(logWeight); largest=std::max(largest,logWeight);
    }
    if (!std::isfinite(largest)) throw std::runtime_error("cannot sample a normal from a zero-density ellipsoid model");
    for (double value:logWeights) weights.push_back(std::exp(value-largest));
    const double total=sumWeights(weights);
    for (;;) {
        const std::size_t i=choose(weights,total,rng);
        const auto &p=projections[i];
        const V u=p.spherePoint(std::sqrt(rng.openUniform01()),2*pi*rng.openUniform01());
        const V a=p.transform.transpose()*jet.gradient;
        if (rng.openUniform01()<checkedAcceptance((a.dot(u)-a.norm())/decayLength_))
            return {i,(p.normalTransform*u).normalized()};
    }
}

double Model::track(const cg::MeanField &sdf,const V &origin,const V &travel,
    double begin,double end,double maximumStep,cg::Random &rng,bool collision,Collision &result,
    TrackingStatistics *statistics,const CandidateCallback &onCandidate) const {
    const V w=cg::normalizedOrThrow(travel);
    if (!origin.allFinite() || !(begin>=0) || !std::isfinite(begin) || !(end>=0) || !std::isfinite(end) ||
        !(maximumStep>0) || !std::isfinite(maximumStep))
        throw std::invalid_argument("ellipsoid tracking needs a finite ray interval and positive tracking_step");
    result=Collision{}; result.distance=end; result.position=origin+end*w;
    if (end<=begin) return 1;
    auto cursor=sdf.createRayCursor(origin,w);
    const auto global=sdf.globalMagnitudeBounds();
    cg::MajorantOpticalDepth clock;
    double weight=1;
    std::vector<double> rates(grains_.size()),logRates(grains_.size());
    const auto updateRates=[&](double minimumValue,double gradientBound,bool strict=true) {
        double total=0;
        for (std::size_t i=0;i<grains_.size();++i) {
            const auto &g=grains_[i];
            // In an SDF frame, F^T gradient = (0,0,|gradient|), even as
            // F rotates along the ray. Include the particle's local rotation.
            const double extent=sdfFrame_ ? g.normalRadius : g.radius;
            logRates[i]=g.density>0 ? std::log(g.density)+std::log(g.maximumArea)+
                (-minimumValue+gradientBound*extent)/decayLength_+1e-12 :
                -std::numeric_limits<double>::infinity();
            rates[i]=strict ? finiteExp(logRates[i]) : std::exp(logRates[i]);
            total+=rates[i];
        }
        if (strict && !std::isfinite(total))
            throw std::runtime_error("poisson_ellipsoid_local: nonfinite total candidate rate");
        return total;
    };
    double coarseRetry=begin;
    while (begin<end) {
        double limit=std::min(end,begin+maximumStep);
        bool coarse=false;
        double total=0;
        // Sparse blocks need no cell-polynomial construction unless an event
        // actually lands there. This still samples their nonzero extinction;
        // it never treats a small probability as vacuum.
        if (cursor && global && begin>=coarseRetry) {
            double coarseLimit=limit;
            cg::MeanRayBounds block;
            if (cursor->coarseRayBounds(begin,coarseLimit,block) && block.certified &&
                coarseLimit>begin && std::isfinite(block.minimumValue) &&
                global->maximumGradientNorm>=0 && std::isfinite(global->maximumGradientNorm)) {
                total=updateRates(block.minimumValue,global->maximumGradientNorm,false);
                if (total*(coarseLimit-begin)<=1) {
                    limit=coarseLimit;
                    coarse=true;
                    if (statistics) ++statistics->coarseIntervals;
                } else coarseRetry=coarseLimit;
            }
        }
        if (!coarse) {
            if (cursor) limit=cursor->nextBoundary(begin,limit);
            if (!(limit>begin)) throw std::runtime_error("ellipsoid tracking interval cannot advance");
            const V a=origin+begin*w,b=origin+limit*w;
            const double pad=128*std::numeric_limits<double>::epsilon()*(1+a.norm()+b.norm());
            cg::BoundsSummary spatial;
            if (!global) spatial=sdf.bounds({a.cwiseMin(b)-V::Constant(pad),a.cwiseMax(b)+V::Constant(pad)});
            if (!global && !spatial.certified) throw std::runtime_error("ellipsoid tracking requires certified SDF gradient bounds");
            const double gradientBound=global ? global->maximumGradientNorm : spatial.maximumGradientNorm;
            SdfRayQueryOptions options; options.computeBeginDerivative=false;
            auto bounds=cursor ? cursor->rayBounds(begin,limit,options) : sdf.rayBounds(origin,w,begin,limit,options);
            if (!bounds.certified || !std::isfinite(bounds.minimumValue) ||
                !(gradientBound>=0) || !std::isfinite(gradientBound))
                throw std::runtime_error("ellipsoid tracking requires finite certified SDF bounds");
            total=updateRates(bounds.minimumValue,gradientBound);
            // Refine costly intervals spanning several decay lengths. Keep the
            // same optical-depth clock across every subdivision and block.
            const double slope=std::max(std::abs(bounds.minimumDerivative),std::abs(bounds.maximumDerivative));
            if (total*(limit-begin)>1 && std::isfinite(slope) && slope*(limit-begin)>decayLength_) {
                const double refined=begin+decayLength_/slope;
                if (refined>begin && refined<limit) {
                    limit=refined;
                    bounds=cursor ? cursor->rayBounds(begin,limit,options) : sdf.rayBounds(origin,w,begin,limit,options);
                    if (!bounds.certified || !std::isfinite(bounds.minimumValue))
                        throw std::runtime_error("ellipsoid tracking requires finite certified SDF bounds");
                    total=updateRates(bounds.minimumValue,gradientBound);
                }
            }
        }
        if (statistics) ++statistics->intervals;
        double t=begin,candidate;
        while (clock.advance(rng,total,t,limit,candidate)) {
            if (statistics) ++statistics->candidates;
            t=candidate;
            const V position=origin+t*w;
            // Re-evaluate value AND gradient at this candidate. A grid cursor
            // evaluates its exact trilinear polynomial, not a frozen plane.
            const SdfJet jet=cursor ? cursor->evaluate(t) : sdf.evaluate(position);
            validateJet(jet);
            double directRate=0;
            if (collision && onCandidate) {
                // Choose a light-query intensity near the local extinction.
                // A cosine hemisphere has E[u]=-2b/3 and
                // Cov[u]=I/4-7bb^T/36. A capped second-cumulant estimate
                // is cheap even for a thin layer. This is ONLY a proposal:
                // it need not bound extinction, and it cancels in the light
                // estimator. Real collision acceptance still uses total.
                for (std::size_t i=0;i<grains_.size();++i) {
                    const auto &g=grains_[i];
                    if (g.density==0) continue;
                    const auto p=projection(i,jet.gradient,w);
                    const V a=p.transform.transpose()*jet.gradient/decayLength_;
                    const double along=a.dot(p.incoming);
                    const double maximum=along<=0 ? a.norm() : (a-along*p.incoming).norm();
                    const double variance=std::max(0.0,a.squaredNorm()/4-7*along*along/36);
                    const double moment=std::min(maximum,-2*along/3+variance/2);
                    directRate+=finiteExp(std::log(g.density)+std::log(p.area)-jet.value/decayLength_+moment);
                }
                directRate=std::min(total,directRate);
                // Score BEFORE acceptance, including candidates later rejected.
                if (directRate>0 && rng.openUniform01()<directRate/total) {
                    if (statistics) ++statistics->directCandidates;
                    onCandidate({position,jet,directRate});
                }
            }
            const std::size_t i=choose(rates,total,rng);
            const auto p=projection(i,jet.gradient,w);
            const V u=p.spherePoint(std::sqrt(rng.openUniform01()),2*pi*rng.openUniform01());
            const V q=p.transform*u;
            const double acceptance=checkedAcceptance(std::log(grains_[i].density)+std::log(p.area)+
                (-jet.value+jet.gradient.dot(q))/decayLength_-logRates[i]);
            if (collision) {
                if (rng.openUniform01()<acceptance) {
                    if (statistics) ++statistics->collisions;
                    result.hit=true; result.distance=t; result.position=position;
                    result.jet=jet; result.directRate=directRate;
                    result.type=i; result.normal=(p.normalTransform*u).normalized();
                    return 1;
                }
            } else {
                weight*=1-acceptance;
                if (weight<0.1) {
                    if (rng.openUniform01()>=weight/0.1) return 0;
                    weight=0.1;
                }
            }
        }
        begin=limit;
    }
    return weight;
}

} // namespace Tungsten::LocalEllipsoid
