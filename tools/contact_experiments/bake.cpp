// Bake analytic SDFs once into identical OpenVDB / NanoVDB float samples.
// No scene-level field transform or narrow-band clamping is involved.
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <openvdb/tree/LeafManager.h>
#include <nanovdb/tools/CreateNanoGrid.h>
#include <nanovdb/io/IO.h>
#include <nlohmann/json.hpp>
#include "math/conditional/mathutility/Gaussian1D.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

using Json = nlohmann::json;
namespace fs = std::filesystem;
using Vec = std::array<double, 3>;
constexpr double Pi = 3.14159265358979323846;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Json readJson(const fs::path& path) {
    std::ifstream in(path);
    require(bool(in), "cannot open " + path.string());
    return Json::parse(in);
}

void writeJson(const fs::path& path, const Json& data) {
    std::ofstream out(path);
    out << data.dump(2) << '\n';
    require(bool(out), "cannot write " + path.string());
}

struct Shape {
    std::string type;
    double radius = 0.25, gap = 0.02, angle = 90;
    Vec lo{}, hi{};
    std::vector<std::array<double, 2>> polygon;
    // A smooth meridian is densely sampled once. Distance to its revolved
    // polyline is a genuine signed distance (not a scaled radial implicit).
    using Point2 = std::array<double,2>;
    struct Node { Point2 lo,hi; int begin,end,left=-1,right=-1; };
    std::vector<Point2> meridian;
    std::vector<Node> nodes;
    static double ballRadius(double theta) {
        return .5-.07*std::exp(-std::pow((theta-1.25)/.16,2))
                  -.13*std::exp(-std::pow(theta/.28,2));
    }
    int buildNode(int begin,int end) {
        Node n{{1e30,1e30},{-1e30,-1e30},begin,end};
        for(int i=begin;i<=end;++i) for(int k=0;k<2;++k) {
            n.lo[k]=std::min(n.lo[k],meridian[i][k]);
            n.hi[k]=std::max(n.hi[k],meridian[i][k]);
        }
        int index=int(nodes.size()); nodes.push_back(n);
        if(end-begin>8) {
            const int middle=(begin+end)/2;
            const int left=buildNode(begin,middle),right=buildNode(middle,end);
            nodes[index].left=left; nodes[index].right=right;
        }
        return index;
    }
    double boxDistance(int index,const Point2 &p) const {
        double d=0;
        for(int k=0;k<2;++k) {
            const double v=std::max({nodes[index].lo[k]-p[k],p[k]-nodes[index].hi[k],0.});
            d+=v*v;
        }
        return d;
    }
    void nearest(int index,const Point2 &p,double &best) const {
        const auto &n=nodes[index];
        if(boxDistance(index,p)>=best) return;
        if(n.left>=0) {
            int a=n.left,b=n.right;
            if(boxDistance(a,p)>boxDistance(b,p)) std::swap(a,b);
            nearest(a,p,best); nearest(b,p,best);
        } else for(int i=n.begin;i<n.end;++i) {
            const auto &a=meridian[i],&b=meridian[i+1];
            const double x=b[0]-a[0],y=b[1]-a[1];
            const double t=std::clamp(((p[0]-a[0])*x+(p[1]-a[1])*y)/(x*x+y*y),0.,1.);
            best=std::min(best,std::pow(p[0]-a[0]-t*x,2)+std::pow(p[1]-a[1]-t*y,2));
        }
    }

    explicit Shape(const Json& spec) : type(spec.at("shape")) {
        if (type == "two_spheres") {
            gap = spec.at("gap");
            require(std::isfinite(gap) && gap >= 0 && gap <= 0.5,
                    "gap must lie in [0, 0.5]; overlapping spheres need a different SDF");
            lo = {-2 * radius - gap / 2, -radius, -radius};
            hi = { 2 * radius + gap / 2,  radius,  radius};
        } else if (type == "v_groove") {
            angle = spec.at("opening_degrees");
            require(std::isfinite(angle) && angle >= 15 && angle <= 120,
                    "opening_degrees must lie in [15, 120]");
            // A finite 1 x 0.4 x 1 block, with a 0.25-deep notch, Y up.
            const double a = 0.25 * std::tan(angle * Pi / 360);
            polygon = {{-.5,-.2}, {.5,-.2}, {.5,.2}, {a,.2},
                       {0,-.05}, {-a,.2}, {-.5,.2}};
            lo = {-.5,-.2,-.5};
            hi = { .5, .2, .5};
        } else if(type=="plane") {
            const double extent=spec.value("half_extent",1.5);
            require(std::isfinite(extent) && extent>0,"plane half_extent must be positive");
            lo={-extent,0,-extent}; hi={extent,0,extent};
        } else if(type=="rounded_shaderball") {
            constexpr int segments=2048;
            for(int i=0;i<=segments;++i) {
                const double theta=Pi*i/segments,r=ballRadius(theta);
                meridian.push_back({r*std::sin(theta),r*std::cos(theta)});
            }
            meridian.front()[0]=meridian.back()[0]=0;
            buildNode(0,segments);
            lo={-.5,-.5,-.5}; hi={.5,.5,.5};
        } else throw std::runtime_error("unknown shape: " + type);
    }

    double sdf(const Vec& p) const {
        if(type=="plane") return p[1];
        if(type=="rounded_shaderball") {
            const Point2 q{std::hypot(p[0],p[2]),p[1]};
            double best=1e30; nearest(0,q,best);
            const double theta=std::atan2(q[0],q[1]);
            const double index=theta/Pi*(meridian.size()-1);
            const int i=std::min(int(index),int(meridian.size())-2);
            const auto &a=meridian[i],&b=meridian[i+1];
            const double ux=std::sin(theta),uy=std::cos(theta);
            const double ex=b[0]-a[0],ey=b[1]-a[1];
            const double radial=(a[0]*ey-a[1]*ex)/(ux*ey-uy*ex);
            return (std::hypot(q[0],q[1])<radial?-1.:1.)*std::sqrt(best);
        }
        if (type == "two_spheres") {
            const double cx = radius + gap / 2;
            const auto sphere = [&](double center) {
                return std::hypot(std::hypot(p[0] - center, p[1]), p[2]) - radius;
            };
            return std::min(sphere(-cx), sphere(cx));
        }
        // Exact distance to a closed concave polygon, then exact finite extrusion.
        double squared = std::numeric_limits<double>::infinity();
        bool inside = false;
        for (std::size_t i = 0, j = polygon.size()-1; i < polygon.size(); j = i++) {
            const auto& a = polygon[j];
            const auto& b = polygon[i];
            const double ex = b[0]-a[0], ey = b[1]-a[1];
            const double vx = p[0]-a[0], vy = p[1]-a[1];
            const double t = std::clamp((vx*ex + vy*ey)/(ex*ex + ey*ey), 0.0, 1.0);
            squared = std::min(squared, (vx-t*ex)*(vx-t*ex) + (vy-t*ey)*(vy-t*ey));
            if ((a[1] > p[1]) != (b[1] > p[1]) &&
                p[0] < a[0] + (p[1]-a[1])*ex/ey) inside = !inside;
        }
        const double d = (inside ? -1 : 1) * std::sqrt(squared);
        const double cap = std::abs(p[2]) - .5;
        return std::min(std::max(d, cap), 0.0) + std::hypot(std::max(d, 0.0), std::max(cap, 0.0));
    }

    Json probes() const {
        Json result = Json::array();
        auto check = [&](Vec p, double expected) {
            const double actual = sdf(p);
            require(std::abs(actual - expected) < 1e-12, "analytic geometry probe failed");
            result.push_back({{"position", p}, {"expected", expected}, {"actual", actual}});
        };
        if(type=="plane") {
            check({0,0,0},0); check({.3,.02,-.4},.02); check({-.2,-.03,.5},-.03);
        } else if(type=="rounded_shaderball") {
            check({0,ballRadius(0),0},0);
            check({0,-.5,0},0);
            check({0,-.6,0},.1);
            for(int i:{200,600,900,1500}) check({meridian[i][0],meridian[i][1],0},0);
        } else if (type == "two_spheres") {
            check({0,0,0}, gap/2);
            check({radius+gap/2,0,0}, -radius);
            check({-radius-gap/2,0,0}, -radius);
            check({2*radius+gap/2,0,0}, 0);
        } else {
            check({0,-.05,0}, 0);
            check({0,-.075,0}, -.025);
            check({0,.075,0}, .125*std::sin(angle*Pi/360));
            check({0,-.125,.6}, .1);
            check({.6,-.125,0}, .1);
        }
        return result;
    }
};

void bake(const Json& spec, const Json& settings, const fs::path& directory) {
    const Shape shape(spec);
    const std::string name = spec.at("name");
    require(fs::path(name).filename().string() == name && name != "." && name != "..",
            "name must be a filename component");
    const double dx = settings.at("voxel_size"), sigma = settings.at("sigma");
    const double ell = settings.at("length_scale"), pad = settings.at("padding");
    for (double value : {dx, sigma, ell, pad})
        require(std::isfinite(value) && value > 0, "bake settings must be positive and finite");
    const Vec spacing=settings.value("voxel_spacing",Vec{dx,dx,dx});
    for(double value:spacing)
        require(std::isfinite(value) && value>0,"voxel spacing must be positive and finite");
    require(pad >= 4*sigma, "padding must be at least 4 sigma");
    const double alpha = std::sqrt(2.0) * sigma / ell;
    constexpr int halo = 4;
    Vec boxLo{}, boxHi{}, origin{};
    std::array<int, 3> dims{};
    std::uint64_t count = 1;
    for (int i=0; i<3; ++i) {
        // Keep the process box fixed when refining the grid. Extra nodes keep
        // VDB's boundary clamp and NanoVDB's domain boundary outside the box.
        boxLo[i] = shape.lo[i] - pad;
        boxHi[i] = shape.hi[i] + pad;
        origin[i] = boxLo[i] - halo*spacing[i];
        const double n = std::ceil((boxHi[i]-boxLo[i])/spacing[i]) + 2*halo + 1;
        require(n >= 2 && n <= 4096, "invalid grid dimensions; increase voxel_size");
        dims[i] = int(n);
        count *= dims[i];
    }
    require(count <= 250000000, "bake exceeds 250 million nodes; increase voxel_size");
    const openvdb::CoordBBox bbox(openvdb::Coord(0), openvdb::Coord(dims[0]-1,dims[1]-1,dims[2]-1));
    auto transform = openvdb::math::Transform::createLinearTransform(1.0);
    transform->postScale(openvdb::Vec3d(spacing[0],spacing[1],spacing[2]));
    transform->postTranslate(openvdb::Vec3d(origin[0],origin[1],origin[2]));
    auto makeGrid = [&](const char* gridName, float background) {
        auto grid = openvdb::FloatGrid::create(background);
        grid->setName(gridName);
        grid->setTransform(transform->copy());
        return grid;
    };
    auto sdf = makeGrid("sdf", float(6*sigma));
    sdf->setGridClass(openvdb::GRID_LEVEL_SET);
    sdf->tree().denseFill(bbox, sdf->background(), true);
    std::cout << name << ": " << dims[0] << 'x' << dims[1] << 'x' << dims[2]
              << " (" << count << " nodes), sampling SDF" << std::endl;
    openvdb::tree::LeafManager<openvdb::FloatTree> leaves(sdf->tree());
    leaves.foreach([&](openvdb::FloatTree::LeafNodeType& leaf, std::size_t) {
        for (auto it = leaf.beginValueOn(); it; ++it) {
            const auto p = it.getCoord();
            it.setValue(float(shape.sdf({origin[0]+spacing[0]*p.x(), origin[1]+spacing[1]*p.y(), origin[2]+spacing[2]*p.z()})));
        }
    });
    require(sdf->activeVoxelCount() == count, "SDF is not fully active");
    const auto nvdbPath = directory/(name+".nvdb");
    const auto vdbPath = directory/(name+"_sdf.vdb");
    openvdb::io::File(vdbPath.string()).write(openvdb::GridPtrVec{sdf});
    std::vector<nanovdb::GridHandle<nanovdb::HostBuffer>> handles;
    handles.push_back(nanovdb::tools::createNanoGrid(*sdf));

    std::cout << name << ": sampling density and writing grids" << std::endl;
    auto density = makeGrid("density", 0.f);
    density->tree().denseFill(bbox, 0.f, true);
    openvdb::tree::LeafManager<openvdb::FloatTree> densityLeaves(density->tree());
    densityLeaves.foreach([&](openvdb::FloatTree::LeafNodeType& leaf, std::size_t) {
        const auto acc = sdf->getConstAccessor();
        for (auto it = leaf.beginValueOn(); it; ++it) {
            const double value = Tungsten::ConditionalGpis::normalPdfOverCdf(acc.getValue(it.getCoord()) / sigma) / sigma;
            require(std::isfinite(value) && value <= std::numeric_limits<float>::max(), "nonfinite density");
            it.setValue(float(value));
        }
    });
    handles.push_back(nanovdb::tools::createNanoGrid(*density));
    auto alphaGrid = makeGrid("alpha", float(alpha));
    alphaGrid->fill(bbox, float(alpha), true);
    handles.push_back(nanovdb::tools::createNanoGrid(*alphaGrid));
    handles.push_back(nanovdb::tools::createNanoGrid(*makeGrid("sigma", float(sigma))));
    nanovdb::io::writeGrids<nanovdb::HostBuffer, std::vector>(nvdbPath.string(), handles);

    // Check persisted files, including coordinates, activation and density.
    auto loaded = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdbPath.string(), "sdf");
    const auto* nano = loaded.grid<float>();
    require(nano && nano->activeVoxelCount() == count, "NanoVDB full-domain check failed");
    auto loadedDensity = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdbPath.string(), "density");
    const auto* nanoDensity = loadedDensity.grid<float>();
    require(nanoDensity && nanoDensity->activeVoxelCount() == count, "density coverage check failed");
    auto loadedSigma = nanovdb::io::readGrid<nanovdb::HostBuffer>(nvdbPath.string(), "sigma");
    require(loadedSigma.grid<float>() &&
            loadedSigma.grid<float>()->tree().root().background() == float(sigma), "sigma grid check failed");
    openvdb::io::File file(vdbPath.string());
    file.open();
    auto vdb = openvdb::gridPtrCast<openvdb::FloatGrid>(file.readGrid("sdf"));
    file.close();
    require(vdb && vdb->activeVoxelCount() == count, "OpenVDB full-domain check failed");
    std::mt19937 rng(17429);
    double maxError = 0;
    for (int n=0; n<4096; ++n) {
        Vec world{};
        std::array<int,3> ijk{};
        for (int a=0; a<3; ++a) {
            ijk[a] = std::uniform_int_distribution<int>(0,dims[a]-1)(rng);
            world[a] = origin[a] + spacing[a]*ijk[a];
        }
        const openvdb::Coord o(ijk[0],ijk[1],ijk[2]);
        const nanovdb::Coord c(ijk[0],ijk[1],ijk[2]);
        const float expected = float(shape.sdf(world));
        const float actual = nano->tree().getValue(c);
        require(actual == expected && vdb->tree().getValue(o) == expected,
                "saved SDF samples differ between formats or analytic SDF");
        require(nanoDensity->tree().getValue(c) == float(Tungsten::ConditionalGpis::normalPdfOverCdf(expected/sigma)/sigma),
                "saved density does not match SDF and sigma at sample nodes");
        const auto pos = nano->indexToWorld(nanovdb::Vec3d(c.x(),c.y(),c.z()));
        const auto vpos = vdb->indexToWorld(o);
        for (int a=0; a<3; ++a)
            require(std::abs(pos[a]-world[a]) < 1e-12 && std::abs(vpos[a]-world[a]) < 1e-12,
                    "saved grid map differs from baked coordinates");
        maxError = std::max(maxError, std::abs(double(actual)-shape.sdf(world)));
    }
    const auto* ng = handles[0].grid<float>();
    const double sdfMin = ng->tree().root().minimum(), sdfMax = ng->tree().root().maximum();
    const double gradientBound = std::sqrt(3.0)*(1+4*std::numeric_limits<float>::epsilon()*
                         std::max(std::abs(sdfMin),std::abs(sdfMax))/
                         std::min({spacing[0],spacing[1],spacing[2]}));
    Json metadata = {
        {"version",2}, {"sigma",sigma}, {"alpha",alpha}, {"dx",dx}, {"voxel_spacing",spacing}, {"origin",origin},
        {"coverage","full_domain"}, {"sign_mode","analytic"},
        {"grids",{"sdf","density","alpha","sigma"}}, {"voxels",dims}, {"resolution",dims},
        {"mesh",name}, {"band",pad/sigma},
        {"transform",{{"scale",1.0},{"translation",{0,0,0}}}},
        {"background",{{"sdf",6*sigma},{"density",0},{"alpha",alpha}}},
        {"measured",{{"sdf_min",sdfMin},{"sdf_max",sdfMax},
                     {"density_max",handles[1].grid<float>()->tree().root().maximum()},
                     {"maximum_gradient_norm",gradientBound}}},
        {"geometry",spec}, {"geometry_bounds",{{"min",shape.lo},{"max",shape.hi}}},
        {"process_box",{{"min",boxLo},{"max",boxHi}}}, {"halo_voxels",halo},
        {"length_scale",ell}, {"polygon_xy",shape.polygon},
        {"verification",{{"saved_node_checks",4096},{"analytic_max_float_error",maxError},
                          {"sdf_formats_identical",true},{"density_verified",true},
                          {"analytic_probes",shape.probes()}}},
        {"note",shape.type=="plane"
            ? "Mean is y everywhere, not a finite solid slab. The process box truncates the stochastic half-space. Tangential coarsening is exact for this mean and density; only normal density interpolation needs refinement."
            : shape.type=="rounded_shaderball"
            ? "Signed distance at nodes; trilinear between nodes. Rounded shaderball uses a 2048-segment smooth meridian approximation. Both formats contain identical SDF samples."
            : "Exact analytic SDF at nodes; trilinear between nodes. Bounds apply inside the full grid."
              " Both spheres share one field; V groove is a finite extruded closed polygon."}
    };
    writeJson(nvdbPath.string()+".json", metadata);
    std::cout << name << ": verified and saved " << nvdbPath.string() << std::endl;
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Usage: bake_contact_fields request.json");
        const auto request = readJson(argv[1]);
        const fs::path directory = fs::absolute(request.at("output_directory").get<std::string>());
        fs::create_directories(directory);
        openvdb::initialize();
        for (const auto& spec : request.at("cases")) bake(spec, request, directory);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Contact bake failed: " << e.what() << std::endl;
        return 1;
    }
}
