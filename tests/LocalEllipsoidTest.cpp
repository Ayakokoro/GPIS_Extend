#include "math/LocalEllipsoid.hpp"
#include "math/GammaDensityMultiplier.hpp"
#include "media/PoissonEllipsoidMedium.hpp"
#include "media/GpisSdfGrid.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "integrators/TraceBase.hpp"
#include "thread/ThreadUtils.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

using namespace Tungsten;
namespace le=LocalEllipsoid;
namespace cg=ConditionalGpis;
using V=Eigen::Vector3d;
namespace {
constexpr double pi=cg::kPi;
void require(bool condition,const char *message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double a,double b,double tolerance,const char *message) {
    if (!(std::abs(a-b)<=tolerance)) {
        std::cerr<<message<<": "<<a<<" vs "<<b<<" (tolerance "<<tolerance<<")\n";
        throw std::runtime_error(message);
    }
}
double factor(double eta) {
    return eta==0 ? 1 : 2*((eta-1)*std::exp(eta)+1)/(eta*eta);
}
Vec3d vec(const V &p) { return Vec3d(p.x(),p.y(),p.z()); }

void geometryAndNormals() {
    le::Model sphere({le::Grain(V::Constant(0.2),V::Zero(),3)},0.5);
    const SdfJet uniform{0,V::Zero()};
    close(sphere.extinction(uniform,V(1,2,3)),3*pi*0.04,1e-12,"sphere projected area");
    le::Model flat({le::Grain(V(0.3,0.2,0.1),V::Zero(),2)},0.5,false);
    for (double gradient:{0.5,1.0,2.0}) for (double sign:{-1.0,1.0}) {
        const SdfJet plane{0.15,V(0,0,gradient)};
        const double reference=2*std::exp(-0.15/0.5)*pi*0.3*0.2*factor(-sign*gradient*0.1/0.5);
        close(flat.extinction(plane,V(0,0,sign)),reference,1e-12,"weighted vertical projected area with raw gradient");
    }
    le::Model mixture({le::Grain(V::Constant(0.1),V::Zero(),2),
                       le::Grain(V::Constant(0.2),V::Zero(),2)},0.5);
    cg::Random rng(8128);
    int second=0; double cosine=0;
    constexpr int count=24000;
    for (int i=0;i<count;++i) {
        const auto normal=mixture.sampleNormal(uniform,V(0,0,-1),rng);
        second+=normal.type==1;
        cosine+=normal.normal.z();
        require(normal.normal.z()>0,"normal must face the incoming ray");
    }
    close(double(second)/count,0.8,0.015,"type weights are density times projected area");
    close(cosine/count,2.0/3,0.012,"sphere visible-normal cosine law");

    le::Model tilted({le::Grain(V(0.27,0.13,0.18),V(30,-20,42),4),
                      le::Grain(V(0.12,0.23,0.16),V(-10,50,15),3)},0.45,true);
    const SdfJet jet{0.12,V(0.4,-0.3,1.2)};
    const V w=V(0.3,0.2,-1).normalized();
    const double extinction=tilted.extinction(jet,w,64);
    double mass=0,typeMass=0; V reference=V::Zero(),sampled=V::Zero();
    constexpr int nz=220,nphi=440;
    // Independent integral in normal solid angle, not the sphere-projection
    // coordinates used by extinction() and by the candidate sampler.
    for (int z=0;z<nz;++z) for (int p=0;p<nphi;++p) {
        const double c=-1+2.0*(z+0.5)/nz,azimuth=2*pi*(p+0.5)/nphi;
        const double s=std::sqrt(1-c*c);
        const V m(s*std::cos(azimuth),s*std::sin(azimuth),c);
        for (std::size_t type=0;type<2;++type) {
            const double contribution=tilted.normalIntensity(jet,w,type,m)*4*pi/(nz*nphi);
            mass+=contribution; reference+=contribution*m;
            if (type==1) typeMass+=contribution;
        }
    }
    close(mass,extinction,2e-4*extinction,"normal Jacobian integrates to weighted projected area");
    second=0;
    for (int i=0;i<count;++i) {
        const auto sample=tilted.sampleNormal(jet,w,rng);
        sampled+=sample.normal; second+=sample.type==1;
    }
    require((sampled/count-reference/mass).norm()<0.022,"tilted mixture normals vs independent integral");
    close(double(second)/count,typeMass/mass,0.018,"weighted mixture type histogram");
    std::cout<<"geometry: analytic projections, gradient scale, mixture weights and normal histograms passed\n";
}

void planeFlights() {
    cg::PlaneMean plane(V::UnitZ(),0);
    le::Model model({le::Grain(V::Constant(0.1),V::Zero(),20)},0.5);
    const V origin(0,0,0.4),w(0,0,-1);
    const double k0=20*pi*0.01*factor(0.2)*std::exp(-0.4/0.5);
    constexpr double end=1;
    const double expected=std::exp(-k0*0.5*std::expm1(end/0.5));
    constexpr int count=18000;
    for (double step:{0.05,0.5}) {
        cg::Random rng(step<0.1 ? 5211:5212);
        int exits=0; double shadows=0;
        for (int i=0;i<count;++i) {
            le::Collision hit;
            model.track(plane,origin,w,0,end,step,rng,true,hit);
            exits+=!hit.hit;
            if (hit.hit) {
                require(hit.distance>=0 && hit.distance<end,"collision before endpoint");
                require(hit.normal.dot(w)<0,"sampled collision faces incoming ray");
            }
            shadows+=model.track(plane,origin,w,0,end,step,rng,false,hit);
        }
        const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
        close(double(exits)/count,expected,tolerance,"plane free-flight survival");
        close(shadows/count,expected,tolerance,"plane marked ratio tracking");
        std::cout<<"plane step="<<step<<": survival="<<double(exits)/count
                 <<", shadow="<<shadows/count<<", analytic="<<expected<<"\n";
    }
}

V sphereDirection(double z,double azimuth) {
    const double r=std::sqrt(std::max(0.0,1-z*z));
    return V(r*std::cos(azimuth),r*std::sin(azimuth),z);
}

void angularScattering() {
    for (bool sdfFrame:{false,true}) {
        le::Model model({le::Grain(V(0.27,0.13,0.18),V(30,-20,42),4),
                         le::Grain(V(0.12,0.23,0.16),V(-10,50,15),3)},0.45,sdfFrame);
        const SdfJet jet{0.12,V(0.4,-0.3,1.2)};
        const V w=V(0.3,0.2,-1).normalized();
        double mass[2]={0,0};
        constexpr int nz=220,nphi=440;
        for (int z=0;z<nz;++z) for (int p=0;p<nphi;++p) {
            const V outgoing=sphereDirection(-1+2.0*(z+0.5)/nz,2*pi*(p+0.5)/nphi);
            const V normal=(outgoing-w).normalized();
            for (std::size_t type=0;type<2;++type) {
                const double rate=model.directionIntensity(jet,w,outgoing,type);
                const double reference=model.normalIntensity(jet,w,type,normal)/(4*-w.dot(normal));
                close(rate,reference,1e-10*std::max(1.0,rate),"reflection solid-angle Jacobian");
                mass[type]+=rate*4*pi/(nz*nphi);
            }
        }
        for (std::size_t type=0;type<2;++type) {
            const double extinction=model.typeExtinction(jet,w,type,64);
            close(mass[type],extinction,3e-4*extinction,"angular rate integrates to type extinction");
        }
        for (const V &direction:{w,V(-w),V::UnitX().eval()})
            require(model.visibleMajorant(jet,direction)>=model.extinction(jet,direction,64),
                    "visible-hemisphere bound covers extinction");
    }
    close(le::Model::misWeight(1e300,2e300),0.2,1e-15,"MIS avoids squaring overflow");
    close(le::Model::misWeight(0,1),0,0,"zero-density MIS branch");
    std::cout<<"scattering: angular Jacobian, integrated rates, visible bounds and stable MIS passed\n";
}

void candidateLighting() {
    cg::PlaneMean plane(V::UnitZ(),0);
    le::Model model({le::Grain(V(0.22,0.11,0.07),V(20,-15,35),8),
                     le::Grain(V(0.10,0.17,0.13),V(-25,35,-10),6)},0.35);
    const V origin(0,0,0.35),w=V(0.15,0.05,-1).normalized();
    const SdfJet initial=plane.evaluate(origin);
    const double albedo[2]={0.35,0.9};
    const auto lighting=[](const V &outgoing) { return 1+0.6*outgoing.x()+0.2*outgoing.z(); };
    const double k0=model.extinction(initial,w,64),end=0.75;
    const double slope=-w.z()/model.decayLength();
    const double survival=std::exp(-k0*std::expm1(slope*end)/slope);
    double angularEnergy=0;
    constexpr int nz=180,nphi=360;
    for (int z=0;z<nz;++z) for (int p=0;p<nphi;++p) {
        const V outgoing=sphereDirection(-1+2.0*(z+0.5)/nz,2*pi*(p+0.5)/nphi);
        for (std::size_t type=0;type<2;++type)
            angularEnergy+=albedo[type]*model.directionIntensity(initial,w,outgoing,type)*
                lighting(outgoing)*4*pi/(nz*nphi);
    }
    const double expected=(1-survival)*angularEnergy/k0;
    constexpr int count=30000;
    constexpr double lightPdf=1/(4*pi);
    for (double step:{0.04,0.5}) {
        cg::Random rng(step<0.1 ? 47181:47182);
        le::TrackingStatistics stats;
        double sum=0,squareSum=0,neeSum=0,neeSquareSum=0;
        int hits=0,flightsWithNullScores=0;
        for (int j=0;j<count;++j) {
            double estimate=0,neeOnly=0;
            int scores=0;
            const auto score=[&](const le::Candidate &candidate) {
                ++scores;
                const V outgoing=sphereDirection(2*rng.openUniform01()-1,2*pi*rng.openUniform01());
                double density=0,energy=0;
                for (std::size_t type=0;type<2;++type) {
                    const double rate=model.directionIntensity(candidate.jet,w,outgoing,type)/candidate.rate;
                    density+=rate; energy+=albedo[type]*rate;
                }
                const double value=energy*lighting(outgoing)/lightPdf;
                neeOnly+=value;
                estimate+=value*le::Model::misWeight(lightPdf,density);
            };
            le::Collision hit;
            model.track(plane,origin,w,0,end,step,rng,true,hit,&stats,score);
            flightsWithNullScores+=scores>int(hit.hit);
            if (hit.hit) {
                ++hits;
                const V outgoing=w-2*w.dot(hit.normal)*hit.normal;
                double density=0;
                for (std::size_t type=0;type<2;++type)
                    density+=model.directionIntensity(hit.jet,w,outgoing,type)/hit.directRate;
                estimate+=albedo[hit.type]*lighting(outgoing)*le::Model::misWeight(density,lightPdf);
            }
            sum+=estimate; squareSum+=estimate*estimate;
            neeSum+=neeOnly; neeSquareSum+=neeOnly*neeOnly;
        }
        const auto tolerance=[&](double total,double squares) {
            return 6*std::sqrt(std::max(0.0,squares/count-std::pow(total/count,2))/count)+0.0005;
        };
        close(sum/count,expected,tolerance(sum,squareSum),"candidate NEE plus real-collision MIS energy");
        close(neeSum/count,expected,tolerance(neeSum,neeSquareSum),"candidate NEE includes rejected events");
        close(double(hits)/count,1-survival,0.02,"candidate lighting preserves collision distribution");
        require(flightsWithNullScores>0,"fixture needs null candidate lighting contributions");
        std::cout<<"candidate lighting step="<<step<<": MIS="<<sum/count<<", NEE="<<neeSum/count
                 <<", reference="<<expected<<", candidates="<<stats.directCandidates<<"\n";
    }
}

void thinLayerFlights() {
    cg::PlaneMean plane(V::UnitZ(),0);
    constexpr double ell=0.001;
    constexpr int count=12000;
    for (bool sdfFrame:{true,false}) for (double tilt:{0.0,55.0}) {
        le::Model model({le::Grain(V(0.007,0.007,0.004),V(tilt,0,0),200000)},ell,sdfFrame);
        const V w(0,0,-1);
        const double k=model.extinction({0,V::UnitZ()},w,64);
        const double targetHeight=ell*std::log(k*ell);
        const V origin(0,0,targetHeight+6*ell);
        const double expected=std::exp(-1+std::exp(-6.0));
        cg::Random rng(5314+int(tilt)+int(sdfFrame));
        le::TrackingStatistics stats;
        int exits=0; double shadows=0;
        for (int i=0;i<count;++i) {
            le::Collision hit;
            model.track(plane,origin,w,0,6*ell,0.01,rng,true,hit,&stats);
            exits+=!hit.hit;
            shadows+=model.track(plane,origin,w,0,6*ell,0.01,rng,false,hit);
        }
        close(double(exits)/count,expected,0.028,"thin ellipsoid layer survival");
        close(shadows/count,expected,0.028,"thin ellipsoid layer ratio tracking");
        if (sdfFrame && tilt==0)
            require(double(stats.candidates)/count<12,"thin SDF-aligned layer must avoid the largest-axis majorant");
        std::cout<<"thin frame="<<(sdfFrame ? "sdf":"world")<<", tilt="<<tilt
                 <<": survival="<<double(exits)/count<<", shadow="<<shadows/count
                 <<", reference="<<expected<<", candidates/flight="<<double(stats.candidates)/count<<"\n";
    }
}

class CurvedField final : public cg::MeanField {
public:
    mutable int queries=0;
    const char *typeName() const override { return "curved_test"; }
    SdfJet evaluate(const V &p) const override {
        ++queries;
        return {0.2-0.8*p.z()*p.z(),V(0,0,-1.6*p.z())};
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &box) const override {
        const double z=std::max(std::abs(box.minimum.z()),std::abs(box.maximum.z()));
        return {0.2-0.8*z*z,1.6*z,true};
    }
};

void curvedFlights() {
    CurvedField field;
    const double radius=0.1,density=15,ell=0.5;
    le::Model model({le::Grain(V::Constant(radius),V::Zero(),density)},ell);
    // Along +Z, gradient = -1.6z. The center-height correction therefore
    // favors the bottom of a sphere. This rate changes in both height and
    // gradient magnitude along the same ray.
    double optical=0;
    constexpr int integration=4000;
    for (int i=0;i<integration;++i) {
        const double z=(i+0.5)/integration;
        optical+=density*pi*radius*radius*std::exp(-(0.2-0.8*z*z)/ell)*
                 factor(1.6*z*radius/ell)/integration;
    }
    const double expected=std::exp(-optical);
    const double frozen=std::exp(-density*pi*radius*radius*std::exp(-0.2/ell));
    require(std::abs(expected-frozen)>0.15,"curved fixture must distinguish updating from a frozen birth plane");
    cg::Random rng(17312);
    constexpr int count=22000;
    int exits=0; double shadows=0;
    for (int i=0;i<count;++i) {
        le::Collision hit;
        model.track(field,V::Zero(),V::UnitZ(),0,1,0.2,rng,true,hit);
        exits+=!hit.hit;
        shadows+=model.track(field,V::Zero(),V::UnitZ(),0,1,0.2,rng,false,hit);
    }
    const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
    close(double(exits)/count,expected,tolerance,"curved SDF flight uses new local plane at each candidate");
    close(shadows/count,expected,tolerance,"curved SDF shadow uses the same changing local law");
    require(field.queries>count,"SDF must be queried inside flights");
    std::cout<<"curved: survival="<<double(exits)/count<<", shadow="<<shadows/count
             <<", reference="<<expected<<", frozen-plane="<<frozen<<"\n";
}

std::shared_ptr<PoissonEllipsoidMedium> parse(const std::string &text) {
    JsonDocument doc(Path("ellipsoid-test.json"),text); Scene scene;
    auto result=std::dynamic_pointer_cast<PoissonEllipsoidMedium>(scene.fetchMedium(doc));
    require(bool(result),"ellipsoid factory registration");
    result->loadResources(); result->prepareForRender(); return result;
}
std::string mediumJson(const std::string &sdf) {
    return R"({"name":"ellipsoids","type":"poisson_ellipsoid_local","sdf":)"+sdf+
        R"(,"decay_length":0.5,"tracking_step":0.12,"frame":"sdf","max_bounces":128,
        "particles":[{"semiaxes":[0.12,0.08,0.05],"rotation_degrees":[15,0,30],"density":12,
                      "material":{"type":"mirror","albedo":1}},
                     {"semiaxes":[0.07,0.09,0.08],"density":8,
                      "material":{"type":"conductor","force_unit_fresnel":true}}]})";
}

std::string gaussianConfig(const std::string &sdf) {
    auto text=mediumJson(sdf);
    const std::string old=R"("decay_length":0.5)";
    text.replace(text.find(old),old.size(),
        R"("density_profile":{"type":"gaussian","stddev":0.15,"center_height":0.12})");
    return text;
}

void gaussianNormals() {
    const auto profile=le::DensityProfile::gaussian(0.14,0.06);
    close(profile.logMaximum(-1,1),0,0,"Gaussian bound includes an interior peak");
    close(profile.logMaximum(0.2,1),-0.5,1e-14,"Gaussian bound above peak");
    close(profile.logMaximum(-1,-0.08),-0.5,1e-14,"Gaussian bound below peak");
    const std::vector<le::Grain> grains={le::Grain(V(0.19,0.14,0.09),V(15,-20,30),10),
                                       le::Grain(V(0.11,0.17,0.22),V(-10,15,0),6)};
    const le::Model model(grains,profile,false);
    const V w=V(0.2,-0.1,-1).normalized();
    cg::Random rng(18571);
    V means[2]; int heightIndex=0;
    for (double height:{-0.22,0.38}) {
        const SdfJet jet{height,V(0.3,-0.2,1.7)};
        double mass=0,secondMass=0; V moment=V::Zero();
        constexpr int nz=180,nphi=360;
        // Independent solid-angle integral of the ellipsoid Gauss-map
        // Jacobian and Gaussian CENTER density, including raw gradient scale.
        for (int z=0;z<nz;++z) for (int p=0;p<nphi;++p) {
            const V m=sphereDirection(-1+2.0*(z+0.5)/nz,2*pi*(p+0.5)/nphi);
            const double cosine=std::max(0.0,-w.dot(m));
            for (std::size_t i=0;i<grains.size();++i) {
                const auto &grain=grains[i];
                const le::M shape=grain.transform*grain.transform.transpose();
                const double support2=m.dot(shape*m);
                const V q=shape*m/std::sqrt(support2);
                const double h=(height-jet.gradient.dot(q)-0.06)/0.14;
                const double rate=grain.density*cosine*grain.determinant*grain.determinant/
                    (support2*support2)*std::exp(-0.5*h*h);
                const double contribution=rate*4*pi/(nz*nphi);
                mass+=contribution; moment+=contribution*m;
                if (i==1) secondMass+=contribution;
                if (p%31==0 && z%13==0) {
                    close(model.normalIntensity(jet,w,i,m),rate,1e-11,"Gaussian normal intensity uses center height");
                    if (cosine>0) {
                        const V outgoing=w+2*cosine*m;
                        close(model.directionIntensity(jet,w,outgoing,i),rate/(4*cosine),1e-10,
                              "Gaussian scattering has the matching reflection Jacobian");
                    }
                }
            }
        }
        close(model.extinction(jet,w,64),mass,1e-3*mass,"Gaussian extinction vs independent angular integral");
        for (const V &direction:{w,V(-w),V::UnitX().eval()})
            require(model.visibleMajorant(jet,direction)>=model.extinction(jet,direction,64),
                    "Gaussian visible density bound covers either side of the peak");
        constexpr int count=18000;
        V sampled=V::Zero(); int second=0;
        for (int j=0;j<count;++j) {
            const auto sample=model.sampleNormal(jet,w,rng);
            sampled+=sample.normal; second+=sample.type==1;
        }
        means[heightIndex++]=sampled/count;
        require((sampled/count-moment/mass).norm()<0.025,"Gaussian conditional normals vs independent integral");
        close(double(second)/count,secondMass/mass,0.025,"Gaussian height-dependent mixture weights");
    }
    require((means[0]-means[1]).norm()>0.1,"Gaussian normal law must change with height");
    std::cout<<"Gaussian normals: both sides, shifted peak, raw gradient, type weights and scattering passed\n";
}

void gaussianFlights() {
    const cg::PlaneMean plane(V::UnitZ(),0);
    constexpr double sigma=0.08,center=0.13,halfSpan=0.8;
    const le::Model model({le::Grain(V::Constant(0.08),V::Zero(),150),
                           le::Grain(V::Constant(0.12),V::Zero(),60)},
                          le::DensityProfile::gaussian(sigma,center));
    // Integrating through the whole plane eliminates every finite-particle
    // height shift: optical depth = projected area * peak density * sqrt(2pi)*sigma.
    // The finite endpoints are more than 8 sigma beyond all possible centers.
    const double optical=std::sqrt(2*pi)*sigma*pi*(150*0.08*0.08+60*0.12*0.12);
    const double expected=std::exp(-optical);
    constexpr int count=10000;
    constexpr double lightPdf=1/(4*pi);
    for (double sign:{-1.0,1.0}) for (double step:{0.03,2.0}) {
        cg::Random rng(27933+int(sign)+int(step*100));
        const V origin(0,0,center-sign*halfSpan),w(0,0,sign);
        int exits=0; double shadow=0,nee=0,nee2=0,mis=0,mis2=0;
        le::TrackingStatistics stats;
        for (int i=0;i<count;++i) {
            double neeEstimate=0,misEstimate=0;
            const auto score=[&](const le::Candidate &candidate) {
                const V outgoing=sphereDirection(2*rng.openUniform01()-1,2*pi*rng.openUniform01());
                double density=0;
                for (std::size_t type=0;type<2;++type)
                    density+=model.directionIntensity(candidate.jet,w,outgoing,type)/candidate.rate;
                const double value=density/lightPdf;
                neeEstimate+=value;
                misEstimate+=value*le::Model::misWeight(lightPdf,density);
            };
            le::Collision hit;
            model.track(plane,origin,w,0,2*halfSpan,step,rng,true,hit,&stats,score);
            exits+=!hit.hit;
            if (hit.hit) {
                const V outgoing=w-2*w.dot(hit.normal)*hit.normal;
                double density=0;
                for (std::size_t type=0;type<2;++type)
                    density+=model.directionIntensity(hit.jet,w,outgoing,type)/hit.directRate;
                misEstimate+=le::Model::misWeight(density,lightPdf);
            }
            nee+=neeEstimate; nee2+=neeEstimate*neeEstimate;
            mis+=misEstimate; mis2+=misEstimate*misEstimate;
            shadow+=model.track(plane,origin,w,0,2*halfSpan,step,rng,false,hit);
        }
        const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
        close(double(exits)/count,expected,tolerance,"Gaussian shell survival in both directions");
        close(shadow/count,expected,tolerance,"Gaussian shell ratio tracking");
        const auto energyTolerance=[&](double sum,double square) {
            return 6*std::sqrt(std::max(0.0,square/count-std::pow(sum/count,2))/count)+0.001;
        };
        close(nee/count,1-expected,energyTolerance(nee,nee2),"Gaussian candidate NEE energy");
        close(mis/count,1-expected,energyTolerance(mis,mis2),"Gaussian candidate NEE and collision MIS energy");
        require(stats.directCandidates>0,"Gaussian direct-light candidates are generated");
        std::cout<<"Gaussian plane sign="<<sign<<", step="<<step<<": survival="<<double(exits)/count
                 <<", shadow="<<shadow/count<<", reference="<<expected<<", MIS="<<mis/count<<"\n";
    }
    // A very thin layer with a huge finite center height should simply have
    // zero representable density here, rather than generate NaN acceptances.
    const le::Model tail({le::Grain(V::Constant(0.1),V::Zero(),1)},le::DensityProfile::gaussian(1e-100));
    cg::ConstantMean farField(1e100); cg::Random rng(1); le::Collision hit;
    close(tail.track(farField,V::Zero(),V::UnitZ(),0,1,0.2,rng,true,hit),1,0,"Gaussian underflow tail is transparent");
    require(!hit.hit,"Gaussian tail cannot create a spurious collision");
}

void gaussianCurvedFlights() {
    CurvedField field;
    constexpr double radius=0.1,density=60,sigma=0.07,center=-0.3;
    const le::Model model({le::Grain(V::Constant(radius),V::Zero(),density)},
                          le::DensityProfile::gaussian(sigma,center));
    double optical=0;
    constexpr int steps=800,cosines=800;
    for (int s=0;s<steps;++s) for (int c=0;c<cosines;++c) {
        const double z=(s+0.5)/steps,u=(c+0.5)/cosines;
        // Entry point q.z=-radius*u; gradient.z=-1.6*z.
        const double h=(0.2-0.8*z*z-1.6*z*radius*u-center)/sigma;
        optical+=density*pi*radius*radius*2*u*std::exp(-0.5*h*h)/(steps*cosines);
    }
    const double expected=std::exp(-optical);
    const double frozen=std::exp(-density*pi*radius*radius*std::exp(-0.5*std::pow((0.2-center)/sigma,2)));
    require(std::abs(expected-frozen)>0.1,"Gaussian curved fixture distinguishes a frozen plane");
    constexpr int count=12000;
    cg::Random rng(48519); int exits=0; double shadow=0;
    for (int i=0;i<count;++i) {
        le::Collision hit;
        model.track(field,V::Zero(),V::UnitZ(),0,1,0.4,rng,true,hit);
        exits+=!hit.hit;
        shadow+=model.track(field,V::Zero(),V::UnitZ(),0,1,0.4,rng,false,hit);
    }
    const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
    close(double(exits)/count,expected,tolerance,"Gaussian curved free flight queries current SDF");
    close(shadow/count,expected,tolerance,"Gaussian curved shadow queries current SDF");
    require(field.queries>count,"Gaussian flights query SDF inside the ray");
    std::cout<<"Gaussian curved: survival="<<double(exits)/count<<", shadow="<<shadow/count
             <<", reference="<<expected<<", frozen="<<frozen<<"\n";
}

void gaussianJson() {
    auto config=gaussianConfig(R"({"type":"homogeneous","offset":0.12})");
    auto medium=parse(config);
    require(!medium->hasDensityMultiplier(),"Gaussian profile uses ordinary Poisson flights by default");
    double areaRate=0;
    for (const auto &grain:medium->model().grains())
        areaRate+=grain.density*pi*grain.determinant*(grain.inverse*V::UnitZ()).norm();
    close(medium->extinction(Vec3d(0.0),Vec3d(0.0,0.0,1.0)),areaRate,1e-12,"Gaussian density means peak density");
    rapidjson::Document document;
    auto encoded=medium->toJson(document.GetAllocator());
    require(!encoded.HasMember("decay_length") && encoded.HasMember("density_profile"),"Gaussian JSON writes only its profile");
    close(encoded["density_profile"]["stddev"].GetDouble(),0.15,0,"Gaussian JSON width");
    close(encoded["density_profile"]["center_height"].GetDouble(),0.12,0,"Gaussian JSON center");
    rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    encoded.Accept(writer);
    const auto copy=parse(buffer.GetString());
    close(copy->extinction(Vec3d(0.0),Vec3d(0.0,0.0,1.0)),areaRate,1e-12,"Gaussian JSON round trip");
    const std::string step=R"("tracking_step":0.12,)";
    auto defaults=config; defaults.erase(defaults.find(step),step.size());
    const std::string center=R"(,"center_height":0.12)";
    defaults.erase(defaults.find(center),center.size());
    auto defaultMedium=parse(defaults);
    auto defaultJson=defaultMedium->toJson(document.GetAllocator());
    close(defaultJson["tracking_step"].GetDouble(),0.15,0,"Gaussian default tracking step is stddev");
    close(defaultJson["density_profile"]["center_height"].GetDouble(),0,0,"Gaussian default center is zero");
    for (const std::string &replacement:{std::string(R"("stddev":0)"),std::string(R"("stddev":-0.1)"),
                                         std::string(R"("width":0.1)")}) {
        auto bad=config;
        const std::string width=R"("stddev":0.15)";
        bad.replace(bad.find(width),width.size(),replacement);
        bool rejected=false;
        try { parse(bad); } catch (const std::exception &) { rejected=true; }
        require(rejected,"Gaussian invalid or missing stddev rejected");
    }
    auto ambiguous=config; ambiguous.insert(ambiguous.find(R"("density_profile")"),R"("decay_length":0.5,)");
    bool rejected=false;
    try { parse(ambiguous); } catch (const std::exception &) { rejected=true; }
    require(rejected,"Gaussian does not silently ignore exponential decay_length");
    for (double invalid:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()}) {
        rejected=false;
        try { le::DensityProfile::gaussian(invalid); } catch (const std::exception &) { rejected=true; }
        require(rejected,"Gaussian scale must be positive and finite");
    }
    std::cout<<"Gaussian JSON: peak density, defaults, round trip and invalid parameters passed\n";
}

void gammaFlights() {
    const cg::ConstantMean field(0);
    const V w(0,0,-1);
    const std::vector<le::Grain> grains={le::Grain(V::Constant(0.2),V::Zero(),4),
                                       le::Grain(V::Constant(0.3),V::Zero(),2)};
    const le::Model model(grains,0.5);
    const double k=pi*(4*0.04+2*0.09),distance=1.4;
    for (double scale:{0.25,1.0,3.0}) {
        auto scaledGrains=grains;
        for (auto &grain:scaledGrains) grain.density*=scale;
        const le::Model scaled(scaledGrains,0.5);
        cg::Random a(8911),b(8911);
        for (int i=0;i<500;++i) {
            le::Collision x,y;
            model.track(field,V::Zero(),w,0,distance,0.2,a,true,x,nullptr,{},scale);
            scaled.track(field,V::Zero(),w,0,distance,0.2,b,true,y);
            require(x.hit==y.hit,"fixed multiplier vs explicitly scaled densities");
            close(x.distance,y.distance,1e-12,"scaled free-flight distance");
            require(x.type==y.type && (x.normal-y.normal).norm()<1e-12,"scaled type and normal");
        }
    }
    constexpr int count=16000;
    for (double alpha:{2.0,8.0,1e6}) for (bool collisionOrigin:{false,true}) {
        cg::Random rng(91337+int(alpha)+int(collisionOrigin));
        int exits=0,hits=0,second=0;
        double shadows=0,cosines=0,scales=0;
        for (int i=0;i<count;++i) {
            const double scale=le::sampleGammaDensityMultiplier(rng,alpha,collisionOrigin);
            scales+=scale;
            le::Collision hit;
            model.track(field,V::Zero(),w,0,distance,0.2,rng,true,hit,nullptr,{},scale);
            exits+=!hit.hit;
            if (hit.hit) { ++hits; second+=hit.type==1; cosines+=hit.normal.z(); }
            shadows+=model.track(field,V::Zero(),w,0,distance,0.2,rng,false,hit,nullptr,{},
                le::sampleGammaDensityMultiplier(rng,alpha,collisionOrigin));
        }
        const double shape=alpha+int(collisionOrigin);
        const double expected=std::exp(-shape*std::log1p(k*distance/alpha));
        const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
        close(double(exits)/count,expected,tolerance,"Gamma flight survival");
        close(shadows/count,expected,tolerance,"Gamma ratio tracking survival");
        close(scales/count,shape/alpha,6*std::sqrt(shape/count)/alpha,"Gamma multiplier mean");
        close(double(second)/hits,0.18/0.34,0.025,"Gamma common scale preserves type mixture");
        close(cosines/hits,2.0/3,0.02,"Gamma common scale preserves conditional normals");
        std::cout<<"gamma alpha="<<alpha<<", collision origin="<<collisionOrigin
                 <<": survival="<<double(exits)/count<<", shadow="<<shadows/count
                 <<", reference="<<expected<<"\n";
    }
}

std::string gammaConfig() {
    return R"({"type":"poisson_ellipsoid_local","sdf":{"type":"homogeneous","offset":0},
        "decay_length":0.5,"tracking_step":0.3,"frame":"world",
        "density_multiplier":{"type":"gamma_renewal","alpha":2},
        "particles":[{"semiaxes":[0.2,0.2,0.2],"density":4,"material":{"type":"mirror","albedo":0.35}},
                     {"semiaxes":[0.3,0.3,0.3],"density":2,"material":{"type":"mirror","albedo":0.9}}]})";
}

void gammaStateAndJson() {
    auto medium=parse(gammaConfig());
    const double distance=1.4,k=pi*0.34;
    const Ray whole(Vec3f(0.f),Vec3f(0.f,0.f,-1.f),0,float(distance));
    const Ray first(Vec3f(0.f),Vec3f(0.f,0.f,-1.f),0,float(distance/2));
    const Ray second(Vec3f(0.f,0.f,-float(distance/2)),Vec3f(0.f,0.f,-1.f),0,float(distance/2));
    constexpr int count=16000;
    IndependentPathSampler rng(67123,false);
    for (auto origin:{PoissonEllipsoidMedium::FlightOrigin::External,
                      PoissonEllipsoidMedium::FlightOrigin::ParticleCollision}) {
        const bool particle=origin==PoissonEllipsoidMedium::FlightOrigin::ParticleCollision;
        const double expected=std::pow(1+k*distance/2,-(particle ? 3.0 : 2.0));
        int fullExits=0,splitExits=0,hits=0;
        double fullShadow=0,splitShadow=0,afterCollision=0;
        for (int i=0;i<count;++i) {
            rng.startPath(particle,i);
            Medium::MediumState state{}; state.reset(); medium->beginFlight(state,origin);
            MediumSample hit{};
            require(medium->sampleDistance(rng,whole,state,hit),"Gamma medium flight");
            fullExits+=hit.exited;
            if (!hit.exited) {
                ++hits;
                const auto parent=state.pathContext;
                auto branch=state;
                afterCollision+=medium->transmittance(rng,whole,false,true,&branch).x();
                require(state.pathContext==parent && branch.pathContext!=parent,
                    "collision shadow samples a new scale without mutating parent");
            }
            state.reset(); medium->beginFlight(state,origin);
            require(medium->sampleDistance(rng,first,state,hit),"Gamma split flight first");
            if (hit.exited) {
                const auto context=state.pathContext;
                require(medium->sampleDistance(rng,second,state,hit),"Gamma split flight second");
                splitExits+=hit.exited;
                if (hit.exited) require(state.pathContext==context,"continued flight keeps its scale");
            }
            state.reset(); medium->beginFlight(state,origin);
            fullShadow+=medium->transmittance(rng,whole,!particle,true,&state).x();
            state.reset(); medium->beginFlight(state,origin);
            const double a=medium->transmittance(rng,first,!particle,true,&state).x();
            const auto parent=state.pathContext;
            // A transparent boundary changes the legacy surface flag, but
            // must not renew the flight or lose its particle-origin law.
            const double b=medium->transmittance(rng,second,true,true,&state).x();
            require(state.pathContext==parent,"split shadow keeps a single multiplier");
            splitShadow+=a*b;
            auto branch=state;
            medium->beginFlight(branch,PoissonEllipsoidMedium::FlightOrigin::ParticleCollision);
            require(state.pathContext==parent && branch.pathContext!=parent,"virtual collision forks state");
        }
        const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
        close(double(fullExits)/count,expected,tolerance,"Gamma medium API survival");
        close(double(splitExits)/count,expected,tolerance,"Gamma split distance survival");
        close(fullShadow/count,expected,tolerance,"Gamma medium API shadow");
        close(splitShadow/count,expected,tolerance,"Gamma split shadow survival");
        const double collisionExpected=std::pow(1+k*distance/2,-3.0);
        close(afterCollision/hits,collisionExpected,6*std::sqrt(collisionExpected*(1-collisionExpected)/hits)+0.001,
            "real collision renews with the density-weighted Gamma law");
        std::cout<<"gamma state origin="<<particle<<": full="<<double(fullExits)/count
                 <<", split="<<double(splitExits)/count<<", shadow="<<splitShadow/count<<"\n";
    }
    rapidjson::Document document;
    auto encoded=medium->toJson(document.GetAllocator());
    require(encoded.HasMember("density_multiplier") && encoded["density_multiplier"]["alpha"].GetDouble()==2,
        "Gamma JSON retains parameters");
    rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer); encoded.Accept(writer);
    auto copy=parse(buffer.GetString());
    require(copy->hasDensityMultiplier(),"Gamma JSON round trip enables model");
    auto offText=gammaConfig();
    const auto pos=offText.find(R"("type":"gamma_renewal","alpha":2)");
    offText.replace(pos,std::string(R"("type":"gamma_renewal","alpha":2)").size(),R"("type":"none")");
    auto off=parse(offText);
    offText.erase(offText.find(R"("density_multiplier")"),std::string(R"("density_multiplier":{"type":"none"},)").size());
    auto omitted=parse(offText);
    IndependentPathSampler a(152,false),b(152,false);
    for (int i=0;i<500;++i) {
        a.startPath(0,i); b.startPath(0,i);
        Medium::MediumState x{},y{}; x.reset(); y.reset(); MediumSample xs{},ys{};
        off->sampleDistance(a,whole,x,xs); omitted->sampleDistance(b,whole,y,ys);
        require(xs.exited==ys.exited && xs.t==ys.t && (xs.exited || xs.aniso==ys.aniso) && a.next1D()==b.next1D(),
            "disabled multiplier preserves samples and random sequence");
        a.startPath(0,i); b.startPath(0,i); x.reset(); y.reset();
        medium->sampleDistance(a,whole,x,xs); copy->sampleDistance(b,whole,y,ys);
        require(xs.exited==ys.exited && xs.t==ys.t && a.next1D()==b.next1D(),"Gamma round-trip samples");
    }
    for (const std::string invalid:{"0","1","-2"}) {
        auto text=gammaConfig(); text.replace(text.find(R"("alpha":2)"),9,R"("alpha":)"+invalid);
        bool rejected=false; try { parse(text); } catch (const std::exception &) { rejected=true; }
        require(rejected,"invalid Gamma parameter rejected");
    }
}

void gammaCandidateLighting() {
    auto config=gammaConfig();
    config.replace(config.find("[0.2,0.2,0.2]"),13,"[0.2,0.13,0.1]");
    config.replace(config.find("[0.3,0.3,0.3]"),13,"[0.12,0.18,0.15]");
    auto medium=parse(config);
    const auto &model=medium->model();
    const V travel=V(0.4,0.2,-1).normalized();
    const auto extinction=[&](const V &w) {
        double k=0;
        for (const auto &g:model.grains()) k+=g.density*pi*g.determinant*(g.inverse*w).norm();
        return k;
    };
    const auto light=[](const V &w) { return 1+0.6*w.x()+0.2*w.z(); };
    const double length=2.0,shadowLength=0.8,k=extinction(travel),lightPdf=1/(4*pi);
    const Ray ray(Vec3f(0.f),Vec3f(vec(travel)),0,float(length));
    double angular=0;
    constexpr int nz=180,np=360;
    for (int z=0;z<nz;++z) for (int p=0;p<np;++p) {
        const V n=sphereDirection(-1+2.0*(z+0.5)/nz,2*pi*(p+0.5)/np);
        const V outgoing=travel-2*travel.dot(n)*n;
        const double shadow=std::pow(1+extinction(outgoing)*shadowLength/2,-3.0);
        for (std::size_t type=0;type<2;++type)
            angular+=model.normalIntensity({0,V::Zero()},travel,type,n)*
                medium->reflectance(type,-travel.dot(n)).x()*light(outgoing)*shadow*4*pi/(nz*np);
    }
    constexpr int count=26000;
    IndependentPathSampler rng(16171,false);
    for (bool particle:{false,true}) {
        const double expected=(1-std::pow(1+k*length/2,-(particle ? 3.0:2.0)))*angular/k;
        double sums[3]={},squares[3]={}; int nullScores=0;
        for (int i=0;i<count;++i) {
            rng.startPath(particle,i);
            Medium::MediumState state{}; state.reset();
            medium->beginFlight(state,particle ? PoissonEllipsoidMedium::FlightOrigin::ParticleCollision :
                PoissonEllipsoidMedium::FlightOrigin::External);
            double estimates[3]={}; int scores=0;
            const auto score=[&](const le::Candidate &candidate) {
                ++scores;
                const V outgoing=sphereDirection(2*rng.next1D()-1,2*pi*rng.next1D());
                double density;
                const Vec3f f=medium->candidateScattering(candidate,ray.dir(),Vec3f(vec(outgoing)),density);
                auto shadowState=state; shadowState.advance();
                medium->beginFlight(shadowState,PoissonEllipsoidMedium::FlightOrigin::ParticleCollision);
                const Ray shadow(Vec3f(vec(candidate.position)),Vec3f(vec(outgoing)),0,float(shadowLength));
                const double value=f.x()*light(outgoing)*medium->transmittance(rng,shadow,false,true,&shadowState).x()/lightPdf;
                estimates[0]+=value*le::Model::misWeight(lightPdf,density);
                estimates[1]+=value;
                close(candidate.rate,candidate.baseRate*candidate.densityScale,1e-12,"physical light proposal rate");
            };
            MediumSample hit{};
            require(medium->sampleDistance(rng,ray,state,hit,score),"Gamma candidate lighting flight");
            nullScores+=scores>int(!hit.exited);
            if (!hit.exited) {
                PhaseSample phase{}; require(hit.phase->sample(rng,ray.dir(),hit,phase),"Gamma phase sample");
                auto shadowState=state;
                const Ray shadow(hit.p,phase.w,0,float(shadowLength));
                const double value=phase.weight.x()*light(V(phase.w.x(),phase.w.y(),phase.w.z()))*
                    medium->transmittance(rng,shadow,false,true,&shadowState).x();
                estimates[0]+=value*le::Model::misWeight(medium->candidateDirectionPdf(hit,ray.dir(),phase.w),lightPdf);
                estimates[2]+=value;
            }
            for (int j=0;j<3;++j) { sums[j]+=estimates[j]; squares[j]+=estimates[j]*estimates[j]; }
        }
        for (int j=0;j<3;++j)
            close(sums[j]/count,expected,6*std::sqrt(std::max(0.0,squares[j]/count-std::pow(sums[j]/count,2))/count)+0.0008,
                "Gamma candidate MIS / NEE / phase with renewed shadow vs independent integral");
        require(nullScores>0,"Gamma NEE must include rejected candidates");
        std::cout<<"gamma lighting origin="<<particle<<": MIS="<<sums[0]/count<<", NEE="<<sums[1]/count
                 <<", phase="<<sums[2]/count<<", reference="<<expected<<"\n";
    }
}
SdfJet trilinear(const V &p) {
    return {0.1+0.4*p.z()+0.3*p.x()*p.z()+0.15*p.y()*p.z()+0.1*p.x()*p.y()*p.z(),
        V(0.3*p.z()+0.1*p.y()*p.z(),0.15*p.z()+0.1*p.x()*p.z(),
          0.4+0.3*p.x()+0.15*p.y()+0.1*p.x()*p.y())};
}
class BoundaryProbe : public TraceBase {
public:
    explicit BoundaryProbe(TraceableScene *scene) : TraceBase(scene,TraceSettings{},0) {}
    using TraceBase::intersectScene;
};

void boundaryExits(const std::filesystem::path &sceneFile) {
    if (!ThreadUtils::pool) ThreadUtils::startThreads(1);
    if (!EmbreeUtil::getDevice()) EmbreeUtil::initDevice();
    std::unique_ptr<Scene> scene(Scene::load(Path(sceneFile.generic_string())));
    scene->loadResources();
    std::unique_ptr<TraceableScene> traceable(scene->makeTraceable());
    BoundaryProbe probe(traceable.get());
    const auto *medium=scene->media().front().get();
    // Enter near a box corner: the exit through its adjacent face is closer
    // than the geometry epsilon and therefore absent from ordinary BVH hits.
    Ray shortExit(Vec3f(0.9f-1e-4f,0.9f,0.f),Vec3f(1.f,-1.f,0.f).normalized(),5e-4f);
    IntersectionTemporary data; IntersectionInfo info;
    require(probe.intersectScene(shortExit,medium,data,info),"recover a process-box exit inside nearT");
    require(shortExit.farT()>0 && shortExit.farT()<shortExit.nearT(),"retain the short boundary distance");
    require(info.Ng.x()>0.99f && info.primitive->selectMedium(medium,false)==nullptr,
            "the recovered exit updates the medium at the correct face");
    Ray zeroExit(Vec3f(0.9f,0.f,0.f),Vec3f(1.f,0.f,0.f),0.f);
    require(probe.intersectScene(zeroExit,medium,data,info) && zeroExit.farT()==0,
            "recover a rounded zero-distance process-box exit");
    IndependentPathSampler rng(651,false); rng.startPath(0,0);
    Medium::MediumState state{}; state.reset(); MediumSample sample{};
    require(medium->sampleDistance(rng,zeroExit,state,sample) && sample.exited,
            "zero-distance boundary exit must transition without a collision");
    std::cout<<"boundary: short and rounded-zero exits passed\n";
}

void rendererAndGrid(const std::filesystem::path &output) {
    openvdb::initialize();
    auto grid=openvdb::FloatGrid::create(1.0f);
    grid->setName("sdf"); grid->setGridClass(openvdb::GRID_LEVEL_SET);
    grid->setTransform(openvdb::math::Transform::createLinearTransform(0.2));
    auto accessor=grid->getAccessor();
    for (int z=-6;z<=6;++z) for (int y=-6;y<=6;++y) for (int x=-6;x<=6;++x)
        accessor.setValue(openvdb::Coord(x,y,z),float(trilinear(V(0.2*x,0.2*y,0.2*z)).value));
    const auto file=output/"field.vdb";
    openvdb::io::File writer(file.string()); writer.write(openvdb::GridPtrVec{grid}); writer.close();
    const std::string sdf=R"({"type":"tabulated","grid":{"type":"vdb","file":")"+
        file.generic_string()+R"(","density_name":"sdf","emission_name":"","density_scale":1,
        "normalize_size":false,"request_sdf":false,"interpolate":"linear","integration_method":"raymarching"}})";
    const std::string config=mediumJson(sdf);
    auto medium=parse(config);
    for (const V &p:{V(0.13,-0.24,0.37),V(-0.51,0.23,-0.12),V(0.62,0.31,0.54)}) {
        const auto actual=medium->localPlane(vec(p)),reference=trilinear(p);
        close(actual.value,reference.value,2e-7,"trilinear SDF value is queried in world units");
        require((actual.gradient-reference.gradient).norm()<2e-6,"trilinear gradient is recomputed, including its magnitude");
        close(medium->extinction(vec(p),Vec3d(0.0,0.0,-1.0)),
            medium->model().extinction(reference,V(0,0,-1)),1e-6,"VDB local extinction");
    }
    rapidjson::Document document;
    auto encoded=medium->toJson(document.GetAllocator());
    rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> jsonWriter(buffer);
    encoded.Accept(jsonWriter);
    auto roundtrip=parse(buffer.GetString());
    close(roundtrip->extinction(Vec3d(0.0),Vec3d(0.0,0.0,-1.0)),medium->extinction(Vec3d(0.0),Vec3d(0.0,0.0,-1.0)),
          1e-12,"ellipsoid JSON round trip");

    // Reference optical depth uses analytic trilinear jets, independently of
    // the VDB point queries, cursor, majorants, and marked tracking.
    const V origin(-0.3,0.1,0.7),direction=V(0.7,0.1,-1).normalized();
    constexpr double distance=1.1;
    double optical=0;
    constexpr int intervals=240;
    for (int i=0;i<intervals;++i)
        optical+=medium->model().extinction(trilinear(origin+distance*(i+0.5)/intervals*direction),direction,24)*distance/intervals;
    const double expected=std::exp(-optical);
    // Check the sparse-block path against the same independent optical depth,
    // including its nonzero collision probability, with and without blocks.
    JsonDocument fieldDocument(Path("ellipsoid-field.json"),sdf);
    Scene fieldScene;
    auto fieldMean=std::dynamic_pointer_cast<TabulatedMean>(fieldScene.fetchMeanFunction(fieldDocument));
    require(bool(fieldMean),"tabulated fixture mean");
    fieldMean->loadResources();
    auto fieldGrid=std::dynamic_pointer_cast<VdbGrid>(fieldMean->grid());
    require(bool(fieldGrid),"VDB fixture grid");
    GpisSdfGrid field;
    field.initialize(*fieldMean,*fieldGrid);
    for (int blockSize:{0,8}) {
        field.buildMacrocells(blockSize);
        cg::Random random(17983+blockSize);
        le::TrackingStatistics stats;
        constexpr int flights=6000;
        int escapes=0; double transmission=0;
        for (int i=0;i<flights;++i) {
            le::Collision hit;
            medium->model().track(field,origin,direction,0,distance,0.12,random,true,hit,&stats);
            escapes+=!hit.hit;
            transmission+=medium->model().track(field,origin,direction,0,distance,0.12,random,false,hit);
        }
        close(double(escapes)/flights,expected,0.04,"sparse-block survival keeps nonzero extinction");
        close(transmission/flights,expected,0.04,"sparse-block shadow transmission");
        if (blockSize) require(stats.coarseIntervals>0 && stats.collisions>0,"sparse blocks must sample collisions");
        std::cout<<"blocks="<<blockSize<<": survival="<<double(escapes)/flights
                 <<", shadow="<<transmission/flights<<", reference="<<expected
                 <<", coarse intervals="<<stats.coarseIntervals<<"\n";
    }
    // Gaussian density needs the UPPER SDF bound as well as the lower one,
    // including when a coarse sparse block straddles the density peak.
    const auto gaussian=parse(gaussianConfig(sdf));
    double gaussianOptical=0;
    for (int i=0;i<intervals;++i)
        gaussianOptical+=gaussian->model().extinction(trilinear(origin+distance*(i+0.5)/intervals*direction),
                                                     direction,24)*distance/intervals;
    const double gaussianExpected=std::exp(-gaussianOptical);
    for (int blockSize:{0,8}) {
        field.buildMacrocells(blockSize);
        cg::Random random(48763+blockSize);
        le::TrackingStatistics stats;
        constexpr int flights=7000;
        int escapes=0; double transmission=0;
        for (int i=0;i<flights;++i) {
            le::Collision hit;
            gaussian->model().track(field,origin,direction,0,distance,0.3,random,true,hit,&stats);
            escapes+=!hit.hit;
            transmission+=gaussian->model().track(field,origin,direction,0,distance,0.3,random,false,hit);
        }
        const double tolerance=6*std::sqrt(gaussianExpected*(1-gaussianExpected)/flights)+0.001;
        close(double(escapes)/flights,gaussianExpected,tolerance,"Gaussian sparse-block survival vs analytic trilinear field");
        close(transmission/flights,gaussianExpected,tolerance,"Gaussian sparse-block transmittance");
        if (blockSize) require(stats.coarseIntervals>0 && stats.collisions>0,"Gaussian sparse blocks sample nonzero density");
        std::cout<<"Gaussian blocks="<<blockSize<<": survival="<<double(escapes)/flights
                 <<", shadow="<<transmission/flights<<", reference="<<gaussianExpected<<"\n";
    }
    IndependentPathSampler rng(19471,false);
    constexpr int count=14000;
    int exits=0,hits=0; double shadows=0;
    // Non-unit direction also checks conversion between ray t and distance.
    const Ray ray(Vec3f(vec(origin)),Vec3f(vec(2*direction)),0,float(distance/2));
    for (int i=0;i<count;++i) {
        rng.startPath(0,i); Medium::MediumState state{}; state.reset(); MediumSample sample{};
        require(medium->sampleDistance(rng,ray,state,sample),"renderer distance sample");
        exits+=sample.exited;
        if (!sample.exited) {
            ++hits;
            PhaseSample bounce{};
            require(sample.phase->isSpecular(),"sampled particle has a smooth micro-BRDF");
            require(sample.phase->sample(rng,ray.dir(),sample,bounce),"renderer particle reflection");
            require(sample.aniso.dot(Vec3d(bounce.w))>0,"bounce leaves the sampled particle surface");
            close(bounce.weight.x(),1,0,"unit mirror/conductor reflection weight");
            require(sample.eventContext && !state.pathContext,"collision context is local, without path history");
        }
        shadows+=medium->transmittance(rng,ray,false,true,&state).x();
    }
    require(hits>0,"fixture must contain collisions");
    const double tolerance=6*std::sqrt(expected*(1-expected)/count)+0.001;
    close(double(exits)/count,expected,tolerance,"VDB first-flight survival vs spatial angular quadrature");
    close(shadows/count,expected,tolerance,"VDB shadow transmittance vs spatial angular quadrature");
    Medium::MediumState state{}; state.reset(); MediumSample empty{};
    const Ray zero(Vec3f(0.0f),Vec3f(0.0f,0.0f,1.0f),0,0);
    require(medium->sampleDistance(rng,zero,state,empty) && empty.exited,"empty interval exits");
    close(medium->transmittance(rng,zero,false,false,&state).x(),1,0,"empty interval is transparent");
    bool rejected=false;
    try { medium->pdf(rng,ray,false,false); } catch (const std::exception &) { rejected=true; }
    require(rejected,"unsupported BDPT cannot silently use a fake density");

    std::ofstream scene(output/"scene.json");
    scene<<R"({"media":[)"<<config<<R"(],
      "bsdfs":[{"name":"boundary","type":"forward","albedo":1}],
      "primitives":[
        {"type":"cube","bsdf":"boundary","int_medium":"ellipsoids","transform":{"scale":[1.8,1.8,1.8]}},
        {"type":"infinite_sphere","emission":1,"sample":true}],
      "camera":{"type":"pinhole","resolution":[16,16],"fov":28,"reconstruction_filter":"tent",
        "transform":{"position":[0,-3,0.4],"look_at":[0,0,0],"up":[0,0,1]}},
      "integrator":{"type":"path_tracer","max_bounces":128,"min_bounces":0,"enable_light_sampling":true,
        "enable_volume_light_sampling":true,"use_cuda_gpis":false},
      "renderer":{"spp":128,"spp_step":128,"adaptive_sampling":false,"output_file":"ellipsoid.png",
        "hdr_output_file":"ellipsoid.exr","overwrite_output_files":true}})";
    scene.close();
    boundaryExits(output/"scene.json");
    std::cout<<"VDB: survival="<<double(exits)/count<<", shadow="<<shadows/count<<", reference="<<expected
             <<"; reflection, JSON and empty-segment checks passed\n";
}

void validation() {
    auto materials=parse(R"({"type":"poisson_ellipsoid_local","sdf":{"type":"homogeneous"},"decay_length":1,
        "particles":[{"semiaxes":[1,1,1],"density":1,
                      "material":{"type":"mirror","albedo":[0.2,0.4,0.6]}},
                     {"semiaxes":[1,1,1],"density":1,
                      "material":{"type":"conductor","eta":[0.2,0.9,1.1],"k":[3.9,2.5,2.2]}}]})");
    const V eta(0.2,0.9,1.1),k(3.9,2.5,2.2);
    for (int channel=0;channel<3;++channel) {
        close(materials->reflectance(0,0.3)[channel],0.2*(channel+1),3e-8,"mirror color weight");
        const double n=eta[channel],a=k[channel];
        close(materials->reflectance(1,1)[channel],((n-1)*(n-1)+a*a)/((n+1)*(n+1)+a*a),
              5e-8,"conductor normal-incidence Fresnel");
        close(materials->reflectance(1,0)[channel],1,0,"conductor grazing Fresnel");
    }
    for (const std::string &sdf:{std::string(R"({"type":"linear","direction":[0,0,1]})"),
                              std::string(R"({"type":"spherical","radius":0.5})"),
                              std::string(R"({"type":"homogeneous","offset":0.1})")}) {
        auto medium=parse(mediumJson(sdf));
        require(medium->extinction(Vec3d(0.1,0.2,0.3),Vec3d(0.0,0.0,-1.0))>0,"analytic SDF backend");
    }
    auto bad=mediumJson(R"({"type":"homogeneous"})");
    bad.replace(bad.find("\"decay_length\":0.5"),18,"\"decay_length\":-1");
    bool rejected=false;
    try { parse(bad); } catch (const std::exception &) { rejected=true; }
    require(rejected,"negative decay length rejected");
    le::Model vacuum({le::Grain(V::Ones(),V::Zero(),0)},1);
    cg::Random rng(1); cg::ConstantMean field(0); le::Collision hit;
    close(vacuum.track(field,V::Zero(),V::UnitZ(),0,1,0.2,rng,true,hit),1,0,"zero-density tracking");
    require(!hit.hit,"zero density cannot collide");
}

void imageCheck(const char *path) {
    int width,height;
    auto data=ImageIO::loadHdr(Path(path),TexelConversion::REQUEST_RGB,width,height);
    require(bool(data) && width>0 && height>0,"read ellipsoid image");
    double sum=0;
    for (int i=0;i<width*height*3;++i) {
        require(std::isfinite(data[i]) && data[i]>=0,"finite nonnegative image");
        sum+=data[i];
    }
    const double mean=sum/(width*height*3);
    close(mean,1,0.035,"ellipsoid white furnace energy");
    std::cout<<"ellipsoid white furnace mean="<<mean<<"\n";
}
}

int main(int argc,char **argv) {
    std::cout.setf(std::ios::unitbuf);
    try {
        if (argc==3 && std::string(argv[1])=="image") { imageCheck(argv[2]); return 0; }
        require(argc==2,"expected fixture output directory");
        const auto output=std::filesystem::absolute(argv[1]); std::filesystem::create_directories(output);
        geometryAndNormals(); angularScattering(); planeFlights(); candidateLighting();
        thinLayerFlights(); curvedFlights(); validation(); gammaFlights(); gammaStateAndJson(); gammaCandidateLighting();
        gaussianNormals(); gaussianFlights(); gaussianCurvedFlights(); gaussianJson();
        rendererAndGrid(output);
        std::cout<<"local ellipsoid checks passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<"\n"; return 1; }
}
