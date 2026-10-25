// Bake the original stationary shaderball's procedural knob for classic_local.
// Use the renderer's shared CSG implementation and native world coordinates.
#include "math/SdfFunctions.hpp"
#include "math/conditional/mathutility/Gaussian1D.h"
#include <openvdb/openvdb.h>
#include <openvdb/tree/LeafManager.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>
#include <nlohmann/json.hpp>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>

using Json=nlohmann::json;
using V=Eigen::Vector3d;
namespace fs=std::filesystem;
void require(bool b,const char *message) { if(!b) throw std::runtime_error(message); }
double density(double mean,double sigma) {
    return Tungsten::ConditionalGpis::normalPdfOverCdf(mean/sigma)/sigma;
}
struct Errors {
    size_t count=0; double squared=0,maximum=0,gradientSquared=0,densitySquared=0;
    void add(double difference,double gradientError,double densityError) {
        ++count;squared+=difference*difference;maximum=std::max(maximum,std::abs(difference));
        gradientSquared+=gradientError;densitySquared+=densityError*densityError;
    }
    Json json() const {
        return {{"samples",count},{"value_rmse",std::sqrt(squared/std::max(size_t(1),count))},
            {"value_max_abs",maximum},{"gradient_vector_rmse",std::sqrt(gradientSquared/std::max(size_t(1),count))},
            {"density_rmse",std::sqrt(densitySquared/std::max(size_t(1),count))}};
    }
};
int main(int argc,char **argv) {
    try {
        require(argc==4,"Usage: gpis_shaderball_bake source.json field.nvdb spacing");
        const fs::path input=fs::absolute(argv[1]),output=fs::absolute(argv[2]);
        require(!fs::exists(output),"Field already exists");
        std::ifstream stream(input);const Json scene=Json::parse(stream);
        const auto &medium=scene.at("media").at(0), &gp=medium.at("gaussian_process");
        const auto &mean=gp.at("mean"), &transform=mean.at("transform");
        require(mean.at("type")=="procedural" && mean.at("func")=="knob","Expected the procedural knob");
        require(transform.at("scale")==1.0 && transform.at("rotation")==Json::array({0,0,0}),"Unsupported mean transform");
        V translation;for(int a=0;a<3;++a) translation[a]=double(float(transform.at("position")[a].get<double>()));
        const auto evaluate=[&](const V &p) {
            return Tungsten::SdfFunctions::evaluateJet(Tungsten::SdfFunctions::Function::Knob,p-translation);
        };
        const double sigma=gp.at("covariance").at("sigma"),dx=std::stod(argv[3]);
        require(std::isfinite(dx) && dx>0 && sigma>0,"Invalid spacing or sigma");
        V lo=V::Constant(1e30),hi=-lo;
        int boundaries=0;
        for(const auto &primitive:scene.at("primitives")) {
            if(primitive.value("int_medium",std::string())!=medium.at("name").get<std::string>()) continue;
            const auto &t=primitive.at("transform");
            require(primitive.at("type")=="cube" && t.at("rotation")==Json::array({0,0,0}) && t.at("scale").is_number(),"Expected an axis-aligned scalar-scale process cube");
            const float half=.5f*t.at("scale").get<float>();
            for(int a=0;a<3;++a) {
                const float center=t.at("position")[a].get<float>();
                lo[a]=std::min(lo[a],double(center-half));hi[a]=std::max(hi[a],double(center+half));
            }
            ++boundaries;
        }
        require(boundaries==1,"Expected one process cube");
        constexpr int halo=4;
        const V origin=lo-V::Constant(halo*dx);
        Eigen::Vector3i dims;size_t count=1;
        for(int a=0;a<3;++a) { dims[a]=int(std::ceil((hi[a]-lo[a])/dx))+2*halo+1;count*=dims[a]; }
        require(count<=250000000,"Field too large; increase spacing");
        std::cout<<"Sampling native knob, "<<dims.transpose()<<", "<<count<<" nodes"<<std::endl;
        openvdb::initialize();
        auto mapping=openvdb::math::Transform::createLinearTransform(dx);
        mapping->postTranslate(openvdb::Vec3d(origin.x(),origin.y(),origin.z()));
        const openvdb::CoordBBox box(openvdb::Coord(0),openvdb::Coord(dims.x()-1,dims.y()-1,dims.z()-1));
        auto sdf=openvdb::FloatGrid::create(float(6*sigma));sdf->setName("sdf");sdf->setTransform(mapping->copy());
        sdf->tree().denseFill(box,sdf->background(),true);
        {
            openvdb::tree::LeafManager<openvdb::FloatTree> leaves(sdf->tree());
            leaves.foreach([&](openvdb::FloatTree::LeafNodeType &leaf,size_t) {
                for(auto it=leaf.beginValueOn();it;++it) {
                    const auto c=it.getCoord();const V p=origin+dx*V(c.x(),c.y(),c.z());
                    const double v=evaluate(p).value;require(std::isfinite(v),"Nonfinite SDF sample");it.setValue(float(v));
                }
            });
        }
        require(sdf->activeVoxelCount()==count,"Incomplete SDF domain");
        std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles;
        handles.push_back(nanovdb::tools::createNanoGrid(*sdf));
        auto rho=openvdb::FloatGrid::create(0.f);rho->setName("density");rho->setTransform(mapping->copy());
        rho->tree().denseFill(box,0.f,true);
        std::cout<<"Sampling density"<<std::endl;
        {
            openvdb::tree::LeafManager<openvdb::FloatTree> leaves(rho->tree());
            leaves.foreach([&](openvdb::FloatTree::LeafNodeType &leaf,size_t) {
                const auto acc=sdf->getConstAccessor();
                for(auto it=leaf.beginValueOn();it;++it) {
                    const double v=density(acc.getValue(it.getCoord()),sigma);
                    require(std::isfinite(v) && v>=0,"Invalid density");it.setValue(float(v));
                }
            });
        }
        handles.push_back(nanovdb::tools::createNanoGrid(*rho));
        sdf.reset();rho.reset();
        fs::create_directories(output.parent_path());
        std::cout<<"Writing "<<output.string()<<std::endl;
        nanovdb::io::writeGrids<nanovdb::HostBuffer,std::vector>(output.string(),handles);handles.clear();
        auto loaded=nanovdb::io::readGrid<nanovdb::HostBuffer>(output.string(),"sdf");
        auto loadedDensity=nanovdb::io::readGrid<nanovdb::HostBuffer>(output.string(),"density");
        const auto *grid=loaded.grid<float>(),*rhoGrid=loadedDensity.grid<float>();
        require(grid && rhoGrid && grid->activeVoxelCount()==count && rhoGrid->activeVoxelCount()==count,"Persisted grid coverage mismatch");
        std::mt19937 random(17429);std::uniform_real_distribution<double> uniform(0,1);
        for(int i=0;i<4096;++i) {
            Eigen::Vector3i ijk;for(int a=0;a<3;++a) ijk[a]=std::uniform_int_distribution<int>(0,dims[a]-1)(random);
            const nanovdb::Coord c(ijk.x(),ijk.y(),ijk.z());const V p=origin+dx*ijk.cast<double>();
            const float expected=float(evaluate(p).value),actual=grid->tree().getValue(c);
            require(actual==expected,"Saved node differs from renderer geometry");
            require(rhoGrid->tree().getValue(c)==float(density(expected,sigma)),"Saved density mismatch");
            const auto pos=grid->indexToWorld(nanovdb::Vec3d(c.x(),c.y(),c.z()));
            for(int a=0;a<3;++a) require(std::abs(pos[a]-p[a])<1e-12,"World placement mismatch");
        }
        Errors all,near;
        for(int sample=0;sample<100000;++sample) {
            V p;for(int a=0;a<3;++a) p[a]=lo[a]+uniform(random)*(hi[a]-lo[a]);
            const auto ref=evaluate(p);const V index=(p-origin)/dx;
            Eigen::Vector3i base;V f;
            for(int a=0;a<3;++a) { base[a]=int(std::floor(index[a]));f[a]=index[a]-base[a]; }
            double value=0,d=0;V gradient=V::Zero();
            for(int bits=0;bits<8;++bits) {
                V w;Eigen::Vector3i c=base;
                for(int a=0;a<3;++a) { const bool upper=(bits>>a)&1;w[a]=upper?f[a]:1-f[a];c[a]+=upper; }
                const nanovdb::Coord coord(c.x(),c.y(),c.z());const double v=grid->tree().getValue(coord);
                value+=w.prod()*v;d+=w.prod()*rhoGrid->tree().getValue(coord);
                for(int a=0;a<3;++a) gradient[a]+=(((bits>>a)&1)?1:-1)*w[(a+1)%3]*w[(a+2)%3]*v/dx;
            }
            const double e=value-ref.value,g=(gradient-ref.gradient).squaredNorm(),de=d-density(ref.value,sigma);
            require(std::isfinite(e) && std::isfinite(g) && std::isfinite(de),"Invalid interpolation query");
            all.add(e,g,de);if(std::abs(ref.value)<3*sigma) near.add(e,g,de);
        }
        const auto vector=[](const V &v){return Json::array({v.x(),v.y(),v.z()});};
        Json meta={{"source_scene",input.string()},{"source_mean",mean},{"sigma",sigma},{"dx",dx},
            {"world_min",vector(lo)},{"world_max",vector(hi)},{"origin",vector(origin)},
            {"dimensions",{dims.x(),dims.y(),dims.z()}},{"active_nodes",count},{"halo_voxels",halo},
            {"saved_node_checks",4096},{"uniform_interpolation_error",all.json()},{"near_surface_interpolation_error",near.json()},
            {"note","Native procedural knob samples, no axis reflection, half-voxel shift, narrow-band clamp, reinitialization, or geometry substitution. Classic_local interpolates density and normalizes its interpolated gradient."}};
        std::ofstream metadata(output.string()+".json");metadata<<meta.dump(2)<<'\n';
        require(bool(metadata),"Cannot write metadata");
        std::cout<<"Verified field; near-surface interpolation: "<<near.json().dump()<<std::endl;
        return 0;
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
