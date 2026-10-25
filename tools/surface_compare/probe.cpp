// Diagnostic paths use the production medium and phase samplers. No renderer
// behavior is changed. Grid exports evaluate the production Gaussian NDF.
#include "media/ClassicLocalMedium.hpp"
#include "media/GlobalConditionalMedium.hpp"
#include "cameras/PinholeCamera.hpp"
#include "samplerecords/PositionSample.hpp"
#include "samplerecords/DirectionSample.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <nlohmann/json.hpp>
#include <openvdb/openvdb.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <array>
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <vector>

using namespace Tungsten;
using J=nlohmann::json;
using V=Eigen::Vector3d;
using M=Eigen::Matrix3d;
constexpr double Pi=3.14159265358979323846;
J read(const char *path) { std::ifstream f(path); if(!f) throw std::runtime_error("input missing"); return J::parse(f); }
void save(const char *path,const J &j) { std::ofstream f(path); f<<j.dump(); if(!f) throw std::runtime_error("output failed"); }
J jsonV(const V &v) { return J::array({v.x(),v.y(),v.z()}); }
V vector(const J &j) { return {j[0].get<double>(),j[1].get<double>(),j[2].get<double>()}; }
J jsonM(const M &m) { J j=J::array(); for(int i=0;i<3;++i) j.push_back(jsonV(m.row(i).transpose())); return j; }
M matrix(const J &j) { M m;for(int i=0;i<3;++i) m.row(i)=vector(j[i]).transpose();return m; }
V eigen(const Vec3f &p) { return {p.x(),p.y(),p.z()}; }
V eigen(const Vec3d &p) { return {p.x(),p.y(),p.z()}; }
Vec3d vecD(const V &p) { return {p.x(),p.y(),p.z()}; }
Vec3f vecF(const V &p) { return Vec3f(float(p.x()),float(p.y()),float(p.z())); }
bool interval(const V &o,const V &w,const V &lo,const V &hi,double &a,double &b) {
    a=0;b=1e30;
    for(int i=0;i<3;++i) {
        if(w[i]==0) {if(o[i]<lo[i]||o[i]>hi[i]) return false;continue;}
        double x=(lo[i]-o[i])/w[i],y=(hi[i]-o[i])/w[i];
        if(x>y) std::swap(x,y);a=std::max(a,x);b=std::min(b,y);
    }
    return b>a;
}
struct Law {
    V mean; ClassicGlobal::GaussianNdf ndf;
    explicit Law(const J &j):mean(vector(j["mean"])),ndf(matrix(j["covariance"])) {}
};
M frame(const J &r) {
    V z=vector(r["laws"][1]["mean"]).normalized(),w=vector(r["wi"]);
    V x=w-z*w.dot(z);if(x.norm()<1e-8) x=z.unitOrthogonal();else x.normalize();
    M f;f.col(0)=x;f.col(1)=z.cross(x);f.col(2)=z;return f;
}
J evaluate(const J &r,int nz,int np,bool grids) {
    std::array<Law,3> laws{Law(r["laws"][0]),Law(r["laws"][1]),Law(r["laws"][2])};
    const V w=vector(r["wi"]); const M basis=frame(r);
    std::array<double,3> area,mass{},dMass{};
    std::array<std::vector<double>,3> ds,ps;
    for(int i=0;i<3;++i) {
        area[i]=laws[i].ndf.projectedArea(laws[i].mean,w);
        if(!(area[i]>0) || !std::isfinite(area[i]))
            throw std::runtime_error("invalid projected area");
    }
    const double cell=4*Pi/(nz*np);double cg=0,lg=0;
    for(int z=0;z<nz;++z) for(int p=0;p<np;++p) {
        const double c=-1+2*(z+.5)/nz,s=std::sqrt(1-c*c),phi=-Pi+2*Pi*(p+.5)/np;
        const V n=basis*V(s*std::cos(phi),s*std::sin(phi),c);
        double pdf[3];
        for(int i=0;i<3;++i) {
            double d=std::exp(laws[i].ndf.logD(laws[i].mean,n));
            pdf[i]=std::max(-w.dot(n),0.)*d/area[i];
            if(!std::isfinite(d)||!std::isfinite(pdf[i])) throw std::runtime_error("invalid NDF");
            dMass[i]+=d*cell;mass[i]+=pdf[i]*cell;
            if(grids) {ds[i].push_back(d);ps[i].push_back(pdf[i]);}
        }
        cg+=.5*std::abs(pdf[2]-pdf[1])*cell;lg+=.5*std::abs(pdf[0]-pdf[1])*cell;
    }
    J result={{"nz",nz},{"nphi",np},{"frame",jsonM(basis)},{"projected_area",area},
        {"vndf_integral",mass},{"ndf_integral",dMass},{"tv_conditional_global",cg},{"tv_local_global",lg}};
    if(grids) {result["ndf"]=ds;result["vndf"]=ps;}
    return result;
}
void scan(const J &spec,const char *out) {
    Scene scene;std::array<std::shared_ptr<Medium>,3> media;
    for(int i=0;i<3;++i) {
        JsonDocument d(Path("surface-probe.json"),spec["media"][i].dump());
        media[i]=scene.fetchMedium(d);media[i]->loadResources();media[i]->prepareForRender();
    }
    auto local=std::dynamic_pointer_cast<ClassicLocalMedium>(media[0]);
    auto global=std::dynamic_pointer_cast<ClassicGlobalMedium>(media[1]);
    auto conditional=std::dynamic_pointer_cast<GlobalConditionalMedium>(media[2]);
    if(!local||!global||!conditional) throw std::runtime_error("expected local/global/conditional order");
    JsonDocument camDoc(Path("probe-camera.json"),spec["camera"].dump());
    PinholeCamera camera;camera.fromJson(camDoc,scene);camera.prepareForRender();
    const V lo=vector(spec["box_min"]),hi=vector(spec["box_max"]);
    const int width=spec["camera"]["resolution"][0],height=spec["camera"]["resolution"][1];
    const int stride=spec.value("stride",6),samples=spec.value("samples",4);
    IndependentPathSampler rng(spec.value("seed",20261024),false);
    J records=J::array();int paths=0;
    for(int y=stride/2;y<height;y+=stride) for(int x=stride/2;x<width;x+=stride)
    for(int s=0;s<samples;++s) {
        rng.startPath(x+y*width,s);PositionSample pos;DirectionSample dir;
        camera.samplePosition(rng,pos);camera.sampleDirection(rng,pos,Vec2u(x,y),dir);
        V origin=eigen(pos.p),direction=eigen(dir.d).normalized();double a,b;
        if(!interval(origin,direction,lo,hi,a,b)) continue;
        origin+=a*direction;
        Vec3f floatOrigin=vecF(origin),floatDirection=vecF(direction).normalized();
        Medium::MediumState state{};state.reset();
        J vertices=J::array({jsonV(eigen(pos.p)),jsonV(eigen(floatOrigin))});
        double throughput=1;
        for(int bounce=0;bounce<10;++bounce) {
            if(!interval(eigen(floatOrigin),eigen(floatDirection),lo,hi,a,b)) break;
            Ray ray(floatOrigin,floatDirection,0,float(b));MediumSample hit{};
            if(!conditional->sampleDistance(rng,ray,state,hit)||hit.exited) break;
            auto e=std::dynamic_pointer_cast<const ConditionalEventContext>(hit.eventContext);
            auto c=std::dynamic_pointer_cast<const ConditionalPathContext>(state.pathContext);
            const V p=e->position,w=eigen(floatDirection).normalized();
            const V ml=local->localMeanGradient(vecD(p)),mg=eigen(global->meanGradient(vecD(p)));
            const double age=(p-c->birth.birthPosition).norm();
            vertices.push_back(jsonV(p));
            J r={{"pixel",{x,y}},{"sample",s},{"bounce",bounce},{"position",jsonV(p)},
                {"wi",jsonV(w)},{"age",age},{"throughput",throughput},
                {"birth_position",jsonV(c->birth.birthPosition)},{"birth_value",c->birth.birthValue},
                {"birth_gradient",jsonV(c->birth.birthGradient)},{"vertices",vertices},
                {"mean_sdf",global->meanValue(vecD(p))},{"mean_gradient_norm",mg.norm()},
                {"laws",J::array({
                    {{"name","classic_local"},{"mean",jsonV(ml)},{"covariance",jsonM(local->ndf().covariance())}},
                    {{"name","classic_global"},{"mean",jsonV(mg)},{"covariance",jsonM(global->ndf().covariance())}},
                    {{"name","global_conditional"},{"mean",jsonV(e->gradient.mean)},{"covariance",jsonM(e->gradient.covariance)}}})}};
            // Exclude numerically singular/tiny-age examples from selection.
            if(age/spec.at("length_scale").get<double>()>.15 && mg.norm()>.1) {
                r["coarse"]=evaluate(r,40,64,false);records.push_back(r);
            }
            PhaseSample phase;
            if(!hit.phase->sample(rng,floatDirection,hit,phase)) break;
            throughput*=phase.weight.avg();
            state.pathContext=phase.pathContext;floatOrigin=hit.p;floatDirection=phase.w;
        }
        if(++paths%200==0) std::cout<<paths<<" paths, "<<records.size()<<" records\n"<<std::flush;
    }
    save(out,J{{"spec",spec},{"paths",paths},{"records",records},
        {"discarded_surface_paths",conditional->discardedSurfacePaths()},
        {"sampling","Production camera / medium / phase calls; diagnostic paths omit direct-light RNG draws."}});
    std::cout<<"Saved "<<records.size()<<" conditional collision records\n";
}
#include "ensemble.hpp"

int main(int argc,char **argv) {
    try {
        openvdb::initialize();
        if(argc!=4) throw std::runtime_error("surface_probe scan|grid|ensemble input.json output.json");
        const J input=read(argv[2]);
        if(std::string(argv[1])=="scan") scan(input,argv[3]);
        else if(std::string(argv[1])=="ensemble") SurfaceEnsemble::run(input,argv[3]);
        else if(std::string(argv[1])=="grid") {
            const int nz=input.value("grid_nz",256),np=input.value("grid_nphi",512);
            if(nz<16 || np<16 || nz>2048 || np>4096)
                throw std::runtime_error("grid_nz/grid_nphi outside supported range");
            J result=input;result["distribution"]=evaluate(input,nz,np,true);
            for(double mass:result["distribution"]["vndf_integral"])
                if(std::abs(mass-1)>.005) throw std::runtime_error("VNDF grid not resolved; increase grid_nz/grid_nphi");
            save(argv[3],result);
        } else throw std::runtime_error("unknown command: expected scan, grid or ensemble");
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
