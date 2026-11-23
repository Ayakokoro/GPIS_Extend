#include "media/ClassicLocalMedium.hpp"
#include "media/ClassicLocalGrid.hpp"
#include "math/conditional/transport/ConditionalSegment.h"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include "io/ImageIO.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include <nanovdb/tools/GridBuilder.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <cmath>

using namespace Tungsten;
using V=Eigen::Vector3d;
namespace {
constexpr double pi=3.14159265358979323846;
void require(bool condition,const char *message) { if (!condition) throw std::runtime_error(message); }
void close(double actual,double expected,double tolerance,const char *message) {
    if (!(std::abs(actual-expected)<=tolerance)) {
        std::cerr<<message<<": "<<actual<<" vs "<<expected<<"\n";
        throw std::runtime_error(message);
    }
}
double area(const V &mean,const V &w,double s=1) {
    const double a=mean.dot(w);
    return s*std::exp(-0.5*a*a/(s*s))/std::sqrt(2*pi)-a*0.5*std::erfc(a/(s*std::sqrt(2.0)));
}
double density(const V &p) { return 0.8+0.03*(p.x()+0.4)/0.25+0.01*(p.y()-0.2)/0.25; }

void fixture(const std::filesystem::path &path,bool negative=false,bool flat=false,bool twoBands=false) {
    nanovdb::tools::build::Grid<float> sdf(0.0f,"sdf"),rho(0.0f,"density"),alpha(0.0f,"alpha");
    for (auto *grid:{&sdf,&rho,&alpha}) grid->setTransform(0.25,nanovdb::Vec3d(-0.4,0.2,0.1));
    auto &s=sdf.getAccessor(); auto &d=rho.getAccessor(); auto &a=alpha.getAccessor();
    for (int k=-3; k<=8; ++k) for (int j=-3; j<=8; ++j) for (int i=-3; i<=8; ++i) {
        const nanovdb::Coord index(i,j,k);
        const V p(-0.4+0.25*i,0.2+0.25*j,0.1+0.25*k);
        s.setValue(index,float(flat ? 2 : 0.3*p.x()+2*p.z()));
        const double rho=twoBands && i!=-2 && i!=5 ? 0 : density(p);
        d.setValue(index,float(negative && i==1 && j==1 && k==1 ? -1 : rho));
        a.setValue(index,j<=-2 ? 0.0f : j<=1 ? 0.5f : 16.0f);
    }
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles;
    handles.emplace_back(nanovdb::tools::createNanoGrid(sdf));
    handles.emplace_back(nanovdb::tools::createNanoGrid(rho));
    handles.emplace_back(nanovdb::tools::createNanoGrid(alpha));
    nanovdb::io::writeGrids<nanovdb::HostBuffer,std::vector>(path.generic_string(),handles);
}
std::string mediumJson(const std::filesystem::path &path,int width=4,const std::string &transform="{}",
        const std::string &covariance=R"({"type":"squared_exponential","sigma":1,"lengthScale":1})",
        const std::string &alphaName="") {
    return R"({"name":"local","type":"classic_local","field":{"file":")"+path.generic_string()+
        R"(","transform":)"+transform+R"(,"macrocell_size":)"+std::to_string(width)+
        (alphaName.empty() ? "" : ",\"alpha_grid\":\""+alphaName+"\"")+"}"+
        (covariance.empty() ? "" : ",\"covariance\":"+covariance)+
        R"(,"material":{"type":"mirror","albedo":1},"max_bounces":64})";
}
std::shared_ptr<ClassicLocalMedium> parse(const std::string &json) {
    JsonDocument doc(Path("classic-local-test.json"),json); Scene scene;
    auto medium=std::dynamic_pointer_cast<ClassicLocalMedium>(scene.fetchMedium(doc));
    require(bool(medium),"classic_local factory registration");
    medium->loadResources(); medium->prepareForRender(); return medium;
}
void gridChecks(const std::filesystem::path &file) {
    int intervals=0;
    for (int width:{1,4,8,0}) {
        ClassicLocalGrid grid(file.generic_string(),"sdf","density",Mat4f(),width ? width : 8,width ? 0 : 64);
        if (!width) {
            for (int sign:{-1,1}) {
                ClassicLocalGrid::Cursor axis(grid,V(-3*sign,0,0),sign*V::UnitX(),0,7);
                ClassicLocalGrid::Segment block;
                int blocks=0;
                while (axis.next(block)) {
                    if (blocks>0 && blocks<59)
                        close(block.end-block.begin,13*.25/64,1e-12,"fixed-resolution block spacing");
                    ++blocks;
                }
                // The SDF clips off the first source voxel of the 13-voxel density extent.
                require(blocks==60,"fixed-resolution DDA coverage in both directions");
            }
        }
        for (const V &o:{V(-3,0,0),V(0,0,3),V(0.2,0.4,0.7),V(3,-3,2)})
            for (const V &w:{V(1,0.2,0.3),V(-0.4,0.2,-1),V(0,0,-2),V(0,1,0)}) {
                ClassicLocalGrid::Cursor cursor(grid,o,w,0,7);
                ClassicLocalGrid::Segment segment;
                double previous=0;
                while (cursor.next(segment)) {
                    require(segment.begin>=previous && segment.end>segment.begin && segment.end<=7,"DDA coverage/order");
                    for (int i=0; i<=40; ++i) {
                        const double t=segment.begin+(segment.end-segment.begin)*i/40;
                        const double rho=grid.density(o+t*w);
                        require(grid.hasDensity(o+t*w)==(rho>0),"support predicate matches interpolated density");
                        require(rho>=0 && rho<=segment.densityMaximum,"density majorant misses trilinear support");
                    }
                    previous=segment.end; ++intervals;
                }
            }
        for (const V &p:{V(0,0,0),V(0.5,0.4,1.0)}) {
            close(grid.density(p),density(p),1e-7,"baked density interpolation");
            require((grid.meanGradient(p)-V(0.3,0,2)).norm()<2e-6,"trilinear SDF gradient");
        }
        close(grid.density(V(100,0,0)),0,0,"outside field is vacuum");
    }
    const Mat4f transform(1,0,0,0.2f, 0,1,0,-0.3f, 0,0,-1,0.6f, 0,0,0,1);
    ClassicLocalGrid transformed(file.generic_string(),"sdf","density",transform,4);
    const V point(0.2,-0.3,0.6);
    close(transformed.density(point),density(V::Zero()),1e-7,"reflected/translated density");
    require((transformed.meanGradient(point)-V(0.3,0,-2)).norm()<2e-6,"inverse-transpose gradient transform");
    std::cout<<"grid: "<<intervals<<" DDA intervals, point/gradient/halo/background and reflection checks passed\n";
}
void conditionalBandChecks(const std::filesystem::path &file) {
    namespace cg=Tungsten::ConditionalGpis;
    fixture(file,false,false,true);
    ClassicLocalGrid band(file.generic_string(),"sdf","density",Mat4f(),1);
    auto spatial=parse(mediumJson(file,1,"{}","","alpha"));
    close(spatial->extinction(Vec3d(0,0.8,0.4),Vec3d(0.0,1.0,0.0)),0,0,"high alpha in the density gap is vacuum");
    for(int i=-24;i<=40;++i) {
        const V point(-.4+i*.0625,.2,.1);
        require(band.hasDensity(point)==(band.density(point)>0),"support near isolated active nodes and faces");
    }
    const V origin(-3,.2,.1), direction=V::UnitX();
    cg::GPSSField field{std::make_shared<cg::ConstantMean>(.3),cg::CovarianceKernel(1,Eigen::Matrix3d::Identity()),
        {V::Constant(-100),V::Constant(100)}};
    cg::FlightState birth;
    birth.birthPosition=origin-.5*direction; birth.birthValue=.8;
    birth.birthGradient=V::Zero(); birth.direction=direction; birth.age=0;
    cg::ConditionalTrackingSettings settings;
    settings.shadowRouletteThreshold=0;
    cg::ConditionalSegment full(field,birth,origin,direction,6,settings);
    settings.classicLocalBand=&band;
    cg::ConditionalSegment masked(field,birth,origin,direction,6,settings);
    // A lone nonzero node extends support one interpolation cell on each side.
    // The two baked sheets have different densities, but both retain the full conditional rate.
    double depth=0;
    for (double center:{2.1,3.85}) {
        for (double delta:{-.2,0.,.2}) {
            const double t=center+delta;
            require(full.hazard(t)>0,"expected nonzero underlying conditional rate");
            close(masked.hazard(t),full.hazard(t),0,"conditional rate on interpolated support");
        }
        constexpr int steps=256;
        for (int j=0;j<steps;++j) depth+=full.hazard(center-.25+.5*(j+.5)/steps)*(.5/steps);
    }
    for (double t:{0.,1.,2.5,3.,4.3,6.}) close(masked.hazard(t),0,0,"conditional rate outside both sheets");
    const double expected=std::exp(-depth);
    cg::Random rng(5972);
    for (bool collision:{true,false}) {
        double sum=0;
        constexpr int count=4096;
        for (int i=0;i<count;++i) {
            double hit;
            const double tr=masked.track(rng,collision,hit);
            sum+=tr;
            if (collision && tr==0) require(band.density(origin+hit*direction)>0,"hit outside baked support");
        }
        close(sum/count,expected,6*std::sqrt(expected*(1-expected)/count),"disconnected conditional band transmission");
    }
}
void transportChecks(const std::filesystem::path &file) {
    const V normal=V(0.3,0,2).normalized();
    const Ray ray(Vec3f(-0.1f,0.2f,1.4f),Vec3f(0.2f,0.0f,-2.0f),0.1f,0.6f);
    const V origin(ray.pos().x(),ray.pos().y(),ray.pos().z()),direction(ray.dir().x(),ray.dir().y(),ray.dir().z());
    const double expected=std::exp(-0.5*(density(origin+ray.nearT()*direction)+density(origin+ray.farT()*direction))*
        (ray.farT()-ray.nearT())*direction.norm()*area(normal,direction.normalized()));
    for (int width:{1,4,8}) {
        auto medium=parse(mediumJson(file,width));
        require((medium->localMeanGradient(Vec3d(0.0))-normal).norm()<2e-6,"local mean gradient is normalized");
        require((medium->ndf().covariance()-Eigen::Matrix3d::Identity()).norm()<1e-12,"stationary covariance ignores alpha grid");
        for (const V &w:{V(0,0,-1),V(0,0,1),V(1,0,0)})
            close(medium->extinction(Vec3d(0.0),Vec3d(w.x(),w.y(),w.z())),density(V::Zero())*area(normal,w),1e-6,
                  "local extinction matches independent Gaussian projected area");
        IndependentPathSampler rng(1823+width,false);
        constexpr int count=24000;
        int escapes=0; double tr=0;
        for (int i=0; i<count; ++i) {
            rng.startPath(0,i); Medium::MediumState state{}; state.reset(); MediumSample sample{};
            require(medium->sampleDistance(rng,ray,state,sample),"local free flight failed");
            require(sample.t>=ray.nearT() && sample.t<=ray.farT(),"finite geometry endpoint");
            close(sample.weight.x(),1,0,"delta tracking throughput");
            escapes+=sample.exited; tr+=medium->transmittance(rng,ray,false,true,&state).x();
        }
        const double tolerance=6*std::sqrt(expected*(1-expected)/count)+1e-4;
        close(double(escapes)/count,expected,tolerance,"delta survival vs analytic integral");
        close(tr/count,expected,tolerance,"ratio transmittance vs analytic integral");
        std::cout<<"width "<<width<<": survival="<<double(escapes)/count<<", ratio="<<tr/count<<", reference="<<expected<<"\n";
        Medium::MediumState state{}; state.reset(); MediumSample sample{};
        const Ray empty(Vec3f(0.0f),Vec3f(0.0f,0.0f,1.0f),0.01f,0.001f);
        medium->sampleDistance(rng,empty,state,sample);
        require(sample.exited && state.bounce==0,"boundary within geometry epsilon");
        close(medium->transmittance(rng,empty,true,true,&state).x(),1,0,"empty segment transmittance");
        rapidjson::Document doc; auto json=medium->toJson(doc.GetAllocator());
        rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer); json.Accept(writer);
        auto copy=parse(buffer.GetString());
        close(copy->extinction(Vec3d(0.0),Vec3d(0.0,0.0,-1.0)),
            medium->extinction(Vec3d(0.0),Vec3d(0.0,0.0,-1.0)),0,"local JSON round trip");
    }
    auto medium=parse(mediumJson(file));
    const auto phase=medium->phaseFunction(Vec3f(0.0f)); MediumSample event{}; event.p=Vec3f(0.0f);
    IndependentPathSampler rng(1911,false);
    for (const Vec3f &wi:{Vec3f(0.0f,0.0f,-1.0f),Vec3f(0.0f,0.0f,1.0f),Vec3f(1.0f,0.0f,0.0f)}) {
        double mass=0; V integrated=V::Zero(),sampled=V::Zero();
        constexpr int nz=100,nphi=160,n=18000;
        for (int z=0; z<nz; ++z) for (int p=0; p<nphi; ++p) {
            const double c=-1+2.0*(z+0.5)/nz,phi=2*pi*(p+0.5)/nphi,r=std::sqrt(1-c*c);
            const Vec3f wo(float(r*std::cos(phi)),float(r*std::sin(phi)),float(c));
            const double probability=phase->pdf(wi,wo,event)*4*pi/(nz*nphi);
            mass+=probability; integrated+=probability*V(wo.x(),wo.y(),wo.z());
        }
        close(mass,1,0.003,"local phase PDF normalization");
        for (int i=0; i<n; ++i) {
            rng.startPath(1,i); PhaseSample sample;
            require(phase->sample(rng,wi,event,sample),"local VNDF sample");
            sampled+=V(sample.w.x(),sample.w.y(),sample.w.z());
            close(sample.weight.x(),1,0,"white mirror weight");
            close(sample.pdf,phase->pdf(wi,sample.w,event),1e-6*std::max<double>(1,sample.pdf),"sampled PDF matches MIS evaluation");
            close(phase->eval(wi,sample.w,event).x()/phase->pdf(wi,sample.w,event),1,1e-6,"phase eval/pdf consistency for MIS");
        }
        require((sampled/n-integrated).norm()<0.025,"local VNDF distribution vs independent phase integration");
    }
    bool rejected=false;
    try { medium->pdf(rng,ray,false,true); } catch (const std::exception &) { rejected=true; }
    require(rejected,"unsupported BDPT PDF must be explicit");
    std::cout<<"phase: normalization, VNDF distribution, white energy and MIS eval/pdf passed\n";
}
void spatialAlphaChecks(const std::filesystem::path &file) {
    const V normal=V(0.3,0,2).normalized();
    auto medium=parse(mediumJson(file,1,"{}","","alpha"));
    auto phase=medium->phaseFunction(Vec3f(0.0f));
    IndependentPathSampler rng(2486,false);
    const Vec3f wi(float(-normal.x()),0,float(-normal.z()));
    MediumSample event{};
    for (double y:{0.45,0.575,0.7}) {
        const double alpha=0.5+15.5*(y-0.45)/0.25;
        const Vec3d point(0,y,0.4);
        close(medium->extinction(point,Vec3d(0.0,1.0,0.0)),density(V(0,y,0.4))*alpha/(2*std::sqrt(pi)),
            1e-6,"tangent extinction interpolates alpha before forming gradient variance");
        event.p=Vec3f(point);
        const ClassicGlobal::GaussianNdf reference(0.5*alpha*alpha*Eigen::Matrix3d::Identity());
        // Axial incidence makes the reference phase independent of azimuth.
        double referenceCosine=0;
        constexpr int bins=512, draws=2048;
        for (int j=0;j<bins;++j) {
            const double c=-1+2.0*(j+0.5)/bins;
            const V wo=c*normal+std::sqrt(1-c*c)*V::UnitY();
            const double pdf=reference.phasePdf(normal,-normal,wo);
            referenceCosine+=4*pi/bins*pdf*c;
            if (j%32==0) {
                const Vec3f out(float(wo.x()),float(wo.y()),float(wo.z()));
                close(phase->pdf(wi,out,event),pdf,2e-5,"phase PDF uses the collision-point alpha");
                close(phase->eval(wi,out,event).x(),pdf,2e-5,"white mirror evaluation uses local phase PDF");
            }
        }
        double sampledCosine=0;
        for (int i=0;i<draws;++i) {
            rng.startPath(2,i); PhaseSample sample;
            require(phase->sample(rng,wi,event,sample),"spatial visible-normal sample");
            sampledCosine+=V(sample.w.x(),sample.w.y(),sample.w.z()).dot(normal);
            close(sample.pdf,phase->pdf(wi,sample.w,event),1e-6,"spatial sampled and evaluated PDF");
        }
        close(sampledCosine/draws,referenceCosine,0.06,"visible-normal sampling responds to local alpha");
    }
    event.p=Vec3f(0,-0.4f,0.4f);
    PhaseSample zeroSample;
    close(medium->extinction(Vec3d(event.p),Vec3d(wi)),0,0,"positive density with zero alpha has zero extinction");
    close(phase->pdf(wi,-wi,event),0,0,"zero-alpha phase PDF");
    require(!phase->sample(rng,wi,event,zeroSample),"zero-alpha phase has no samples");

    const V origin(0,-0.39,0.4);
    for (int resolution:{0,32}) {
        ClassicLocalGrid grid(file.generic_string(),"sdf","density",Mat4f(),1,resolution,"alpha");
        close(grid.alpha(V(0,0.575,0.4)),8.25,1e-8,"trilinear roughness at half voxel");
        ClassicLocalGrid::Cursor cursor(grid,origin,V::UnitY(),0,1.5);
        ClassicLocalGrid::Segment block;
        int blocks=0;
        while (cursor.next(block)) {
            for (int i=0;i<=8;++i) {
                const V p=origin+(block.begin+(block.end-block.begin)*i/8)*V::UnitY();
                const double a=grid.alpha(p);
                const double hazard=a>0 ? grid.density(p)*area(normal,-normal,a/std::sqrt(2.0)) : 0;
                require(hazard<=block.spatialExtinctionMaximum*(1+1e-10),"macrocell bound misses interpolated roughness");
            }
            ++blocks;
        }
        require(blocks>1,"spatial majorant traverses multiple macrocells");
    }
    // Density and alpha are linear along this tangent ray; Simpson integrates their product exactly.
    const double tau=0.25/(12*std::sqrt(pi))*(0.5*density(V(0,0.45,0.4))+
        4*8.25*density(V(0,0.575,0.4))+16*density(V(0,0.7,0.4)));
    const double expected=std::exp(-tau);
    const Ray ray(Vec3f(0,0.45f,0.4f),Vec3f(0.0f,1.0f,0.0f),0,0.25f);
    constexpr int count=8192;
    int escaped=0; double transmittance=0;
    for (int i=0;i<count;++i) {
        rng.startPath(3,i); Medium::MediumState state{}; state.reset(); MediumSample sample{};
        require(medium->sampleDistance(rng,ray,state,sample),"spatial free flight failed");
        escaped+=sample.exited;
        transmittance+=medium->transmittance(rng,ray,false,true,&state).x();
    }
    const double tolerance=6*std::sqrt(expected*(1-expected)/count);
    close(double(escaped)/count,expected,tolerance,"spatial free flight vs analytic transmittance");
    close(transmittance/count,expected,tolerance,"spatial ratio tracking vs analytic transmittance");
    rapidjson::Document doc; auto saved=medium->toJson(doc.GetAllocator());
    require(!saved.HasMember("covariance") && std::string(saved["field"]["alpha_grid"].GetString())=="alpha",
        "serialize the chosen roughness source");
    rapidjson::StringBuffer buffer; rapidjson::Writer<rapidjson::StringBuffer> writer(buffer); saved.Accept(writer);
    auto copy=parse(buffer.GetString());
    const Vec3d midpoint(0,0.575,0.4),tangent(0.0,1.0,0.0);
    close(copy->extinction(midpoint,tangent),medium->extinction(midpoint,tangent),0,"spatial-alpha scene round trip");
    auto reflected=parse(mediumJson(file,1,R"({"scale":[1,-1,1]})","","alpha"));
    close(reflected->extinction(Vec3d(0,-midpoint.y(),midpoint.z()),-tangent),medium->extinction(midpoint,tangent),
        1e-6,"spatial roughness follows the field transform");
}
void validationChecks(const std::filesystem::path &file,const std::filesystem::path &output) {
    auto rejects=[](const std::string &json) { try { parse(json); } catch (const std::exception &) { return true; } return false; };
    require(rejects(mediumJson(file,0)),"invalid macrocell size");
    auto fixed=mediumJson(file);
    fixed.replace(fixed.find("\"macrocell_size\":4"),std::string("\"macrocell_size\":4").size(),"\"macrocell_resolution\":7");
    auto fixedMedium=parse(fixed);
    rapidjson::Document doc;
    auto saved=fixedMedium->toJson(doc.GetAllocator());
    require(saved["field"]["macrocell_resolution"].GetInt()==7 && !saved["field"].HasMember("macrocell_size"),
        "preserve fixed-resolution layout in serialized scenes");
    fixed.insert(fixed.find("\"macrocell_resolution\""),"\"macrocell_size\":4,");
    require(rejects(fixed),"ambiguous macrocell layout");
    auto alpha=mediumJson(file); alpha.insert(alpha.find("\"macrocell_size\""),"\"alpha_grid\":\"alpha\",");
    require(rejects(alpha),"two roughness sources are ambiguous");
    require(rejects(mediumJson(file,4,"{}","")),"missing roughness source");
    require(rejects(mediumJson(file,4,"{}","","absent")),"missing named alpha grid");
    require(rejects(mediumJson(file,4,"{}","","sdf")),"negative alpha rejected");
    require(rejects(mediumJson(file,4,"{}",R"({"type":"squared_exponential","sigma":0,"lengthScale":1})")),
        "degenerate kernel is rejected");
    require(rejects(mediumJson(file,4,"{}",R"({"type":"nonstationary","cov":{"type":"squared_exponential","sigma":1,"lengthScale":1}})")),
        "non-stationary kernel is rejected");
    fixture(output/"negative.nvdb",true);
    require(rejects(mediumJson(output/"negative.nvdb")),"negative density rejected");
    fixture(output/"flat.nvdb",false,true);
    auto flat=parse(mediumJson(output/"flat.nvdb"));
    require(flat->localMeanGradient(Vec3d(0.0)).norm()<1e-12,"zero SDF gradient remains isotropic");
    const auto flipped=parse(mediumJson(file,4,R"({"scale":[1,1,-1]})"));
    require(flipped->localMeanGradient(Vec3d(0.0)).z()<0,"medium uses reflected gradient");
}
void renderFixture(const std::filesystem::path &file,const std::filesystem::path &output) {
    std::ofstream mesh(output/"mirror.obj");
    mesh<<"v -0.6 -0.3 0.4\nv 0.6 -0.3 0.4\nv 0 0.6 0.4\nf 1 2 3\n"; mesh.close();
    std::ofstream scene(output/"scene.json");
    scene<<R"({"media":[)"<<mediumJson(file)<<R"(],
      "bsdfs":[{"name":"boundary","type":"forward","albedo":1}],
      "primitives":[
        {"type":"cube","bsdf":"boundary","int_medium":"local","transform":{"position":[0,0.5,0.5],"scale":[5,5,5]}},
        {"type":"mesh","file":")"<<(output/"mirror.obj").generic_string()<<R"(","bsdf":{"type":"mirror","albedo":1}},
        {"type":"infinite_sphere","emission":1,"sample":true}],
      "camera":{"type":"pinhole","resolution":[16,16],"fov":25,"reconstruction_filter":"tent",
        "transform":{"position":[0,-6,1.2],"look_at":[0,0.4,0.7],"up":[0,0,1]}},
      "integrator":{"type":"path_tracer","max_bounces":128,"min_bounces":0,"enable_light_sampling":true,
        "enable_volume_light_sampling":true,"use_cuda_gpis":false},
      "renderer":{"spp":128,"spp_step":128,"adaptive_sampling":false,"output_file":"local.png",
        "hdr_output_file":"local.exr","overwrite_output_files":true}})";
}
void imageCheck(const char *file) {
    int width,height; auto data=ImageIO::loadHdr(Path(file),TexelConversion::REQUEST_RGB,width,height);
    require(bool(data) && width>0 && height>0,"read furnace image"); double sum=0;
    for (int i=0; i<width*height*3; ++i) { require(std::isfinite(data[i]) && data[i]>=0,"finite nonnegative furnace"); sum+=data[i]; }
    close(sum/(width*height*3),1,0.025,"local white furnace energy");
    std::cout<<"classic_local furnace mean="<<sum/(width*height*3)<<"\n";
}
}
int main(int argc,char **argv) {
    try {
        if (argc==3 && std::string(argv[1])=="image") { imageCheck(argv[2]); return 0; }
        require(argc==2,"expected fixture output directory");
        const auto output=std::filesystem::absolute(argv[1]); std::filesystem::create_directories(output);
        const auto file=output/"field.nvdb"; fixture(file);
        gridChecks(file); transportChecks(file); spatialAlphaChecks(file); validationChecks(file,output); renderFixture(file,output);
        conditionalBandChecks(output/"two-bands.nvdb");
        std::cout<<"classic_local checks passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<"\n"; return 1; }
}
