#include "media/ClassicLocalMedium.hpp"
#include "media/ClassicLocalGrid.hpp"
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

void fixture(const std::filesystem::path &path,bool negative=false,bool flat=false) {
    nanovdb::tools::build::Grid<float> sdf(0.0f,"sdf"),rho(0.0f,"density"),alpha(0.0f,"alpha");
    for (auto *grid:{&sdf,&rho,&alpha}) grid->setTransform(0.25,nanovdb::Vec3d(-0.4,0.2,0.1));
    auto &s=sdf.getAccessor(); auto &d=rho.getAccessor(); auto &a=alpha.getAccessor();
    for (int k=-3; k<=8; ++k) for (int j=-3; j<=8; ++j) for (int i=-3; i<=8; ++i) {
        const nanovdb::Coord index(i,j,k);
        const V p(-0.4+0.25*i,0.2+0.25*j,0.1+0.25*k);
        s.setValue(index,float(flat ? 2 : 0.3*p.x()+2*p.z()));
        d.setValue(index,float(negative && i==1 && j==1 && k==1 ? -1 : density(p)));
        // Deliberately varying: stationary mode must never use this channel.
        a.setValue(index,float(0.001+0.1*(i+3)));
    }
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles;
    handles.emplace_back(nanovdb::tools::createNanoGrid(sdf));
    handles.emplace_back(nanovdb::tools::createNanoGrid(rho));
    handles.emplace_back(nanovdb::tools::createNanoGrid(alpha));
    nanovdb::io::writeGrids<nanovdb::HostBuffer,std::vector>(path.generic_string(),handles);
}
std::string mediumJson(const std::filesystem::path &path,int width=4,const std::string &transform="{}",
        const std::string &covariance=R"({"type":"squared_exponential","sigma":1,"lengthScale":1})") {
    return R"({"name":"local","type":"classic_local","field":{"file":")"+path.generic_string()+
        R"(","transform":)"+transform+R"(,"macrocell_size":)"+std::to_string(width)+
        R"(},"covariance":)"+covariance+R"(,"material":{"type":"mirror","albedo":1},"max_bounces":64})";
}
std::shared_ptr<ClassicLocalMedium> parse(const std::string &json) {
    JsonDocument doc(Path("classic-local-test.json"),json); Scene scene;
    auto medium=std::dynamic_pointer_cast<ClassicLocalMedium>(scene.fetchMedium(doc));
    require(bool(medium),"classic_local factory registration");
    medium->loadResources(); medium->prepareForRender(); return medium;
}
void gridChecks(const std::filesystem::path &file) {
    int intervals=0;
    for (int width:{1,4,8}) {
        ClassicLocalGrid grid(file.generic_string(),"sdf","density",Mat4f(),width);
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
            close(sample.pdf,phase->pdf(wi,sample.w,event),1e-6*std::max(1.0f,sample.pdf),"sampled PDF matches MIS evaluation");
            close(phase->eval(wi,sample.w,event).x()/phase->pdf(wi,sample.w,event),1,1e-6,"phase eval/pdf consistency for MIS");
        }
        require((sampled/n-integrated).norm()<0.025,"local VNDF distribution vs independent phase integration");
    }
    bool rejected=false;
    try { medium->pdf(rng,ray,false,true); } catch (const std::exception &) { rejected=true; }
    require(rejected,"unsupported BDPT PDF must be explicit");
    std::cout<<"phase: normalization, VNDF distribution, white energy and MIS eval/pdf passed\n";
}
void validationChecks(const std::filesystem::path &file,const std::filesystem::path &output) {
    auto rejects=[](const std::string &json) { try { parse(json); } catch (const std::exception &) { return true; } return false; };
    require(rejects(mediumJson(file,0)),"invalid macrocell size");
    auto alpha=mediumJson(file); alpha.insert(alpha.find("\"macrocell_size\""),"\"alpha_grid\":\"alpha\",");
    require(rejects(alpha),"spatial alpha is unsupported");
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
        gridChecks(file); transportChecks(file); validationChecks(file,output); renderFixture(file,output);
        std::cout<<"stationary classic_local checks passed\n"; return 0;
    } catch (const std::exception &error) { std::cerr<<error.what()<<"\n"; return 1; }
}
