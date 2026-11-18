// Full medium-segment prior means for selected pixels in the slab experiment.
// Reuse the production camera, kernel and mean; do not invent GP observations.
#include "media/SingleHistoryGpisMedium.hpp"
#include "math/SdfFunctions.hpp"
#include "cameras/PinholeCamera.hpp"
#include "samplerecords/PositionSample.hpp"
#include "samplerecords/DirectionSample.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <nlohmann/json.hpp>
#include <openvdb/openvdb.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace Tungsten;
using J=nlohmann::json;
using V=Eigen::Vector3d;
J read(const std::string &p) {
    std::ifstream f(p);if(!f)throw std::runtime_error("Missing input: "+p);
    return J::parse(f);
}
J json(const V &v) {return J::array({v.x(),v.y(),v.z()});}
V vector(const J &v) {return {v[0].get<double>(),v[1].get<double>(),v[2].get<double>()};}
V eigen(const Vec3f &v) {return {v.x(),v.y(),v.z()};}
bool interval(const V &o,const V &w,const V &lo,const V &hi,double &a,double &b) {
    a=0;b=1e30;
    for(int i=0;i<3;++i) {
        if(w[i]==0) {if(o[i]<lo[i]||o[i]>hi[i])return false;continue;}
        double x=(lo[i]-o[i])/w[i],y=(hi[i]-o[i])/w[i];if(x>y)std::swap(x,y);
        a=std::max(a,x);b=std::min(b,y);
    }
    return b>a;
}
int main(int argc,char **argv) {
    try {
        if(argc!=3)throw std::runtime_error("Usage: slabs_ray_profiles spec.json raw.json");
        openvdb::initialize();const J spec=read(argv[1]);
        const std::string scenePath=spec.at("scene");const J source=read(scenePath);
        const J definition=source.at("media").at(0);
        const J mean=definition.at("gaussian_process").at("mean");
        if(mean!=J{{"type","procedural"},{"func","three_slabs"}})
            throw std::runtime_error("Expected untransformed three_slabs mean");
        Scene scene;JsonDocument md(Path(scenePath),definition.dump());
        auto medium=std::dynamic_pointer_cast<SingleHistoryGpisMedium>(scene.fetchMedium(md));
        if(!medium)throw std::runtime_error("Expected original single_history_gpis scene");
        medium->loadResources();medium->prepareForRender();const auto &field=medium->field();
        JsonDocument cd(Path(scenePath),source.at("camera").dump());
        PinholeCamera camera;camera.fromJson(cd,scene);camera.prepareForRender();
        J boundary;
        for(const auto &p:source.at("primitives"))
            if(p.value("name","")=="shared_process_bounds")boundary=p;
        if(boundary.is_null()||boundary.at("type")!="cube")throw std::runtime_error("Missing process box");
        const auto &transform=boundary.at("transform");
        if(transform.contains("rotation"))throw std::runtime_error("Expected axis-aligned process box");
        const V center=vector(transform.at("position"));
        const V half=vector(transform.at("scale"))*.5;
        const V lo=center-half,hi=center+half;
        const V slabHalf(.6,.01,.6);
        const std::array<V,3> centers{{V(.18,.02,0),V(-.18,-.02,0),V(.18,-.06,0)}};
        const unsigned seed=spec.at("seed"),sample=spec.value("sample",0u);
        const int n=spec.value("interval_count",6656);
        if(n<64||n>262144)throw std::runtime_error("Invalid profile resolution");
        J records=J::array();
        for(const auto &choice:spec.at("rays")) {
            const unsigned x=choice.at("pixel")[0],y=choice.at("pixel")[1];
            if(x>=camera.resolution().x()||y>=camera.resolution().y())throw std::runtime_error("Pixel out of range");
            IndependentPathSampler rng(seed,false);rng.startPath(x+y*camera.resolution().x(),sample);
            PositionSample position;DirectionSample direction;
            if(!camera.samplePosition(rng,position)||!camera.sampleDirection(rng,position,Vec2u(x,y),direction))
                throw std::runtime_error("Camera sample failed");
            const V cameraOrigin=eigen(position.p),w=eigen(direction.d).normalized();
            double a,b;if(!interval(cameraOrigin,w,lo,hi,a,b))throw std::runtime_error("Ray misses process box");
            // This is the double-precision geometric section of the sampled
            // camera line, not a replay of float boundary intersection rounding.
            const V origin=cameraOrigin+a*w;const double length=b-a;
            const V endpoint=origin+length*w;
            const auto start=field.mean->evaluate(origin),end=field.mean->evaluate(endpoint);
            const double sigma=field.kernel.sigma(),variance=sigma*sigma;
            const double gamma=std::sqrt(w.dot(field.kernel.metric()*w)),ell=1/gamma;
            const double configuredEll=definition.at("gaussian_process").at("covariance").at("lengthScale");
            const double configuredSigma=definition.at("gaussian_process").at("covariance").at("sigma");
            J samples=J::array();double maxError=0;
            for(int i=0;i<=n;++i) {
                const double s=length*i/n;const V p=origin+s*w;
                const double value=field.mean->evaluate(p).value;
                double reference=1e30;
                for(const auto &c:centers)reference=std::min(reference,((p-c).cwiseAbs()-slabHalf).maxCoeff());
                maxError=std::max(maxError,std::abs(value-reference));
                if(!std::isfinite(value))throw std::runtime_error("Nonfinite mean");
                samples.push_back(J::array({s,s/configuredEll,value,value/configuredSigma}));
            }
            if(maxError>1e-12)throw std::runtime_error("Slab formula differs from production mean");
            J intersections=J::array();
            for(int i=0;i<3;++i) {
                double begin,finish;
                if(interval(origin,w,centers[i]-slabHalf,centers[i]+slabHalf,begin,finish)&&begin<length) {
                    finish=std::min(finish,length);const double mid=.5*(begin+finish);
                    intersections.push_back({{"slab",i+1},{"layout",i==1?"B":"A"},
                        {"begin",begin},{"end",finish},{"midpoint",mid},
                        {"midpoint_mean",field.mean->evaluate(origin+mid*w).value}});
                }
            }
            if(intersections.size()!=choice.at("expected_layers").get<size_t>())
                throw std::runtime_error("Selected pixel has the wrong number of intersected slabs");
            std::vector<double> knots{0,length};
            SdfFunctions::appendHistoryCandidatePoints(SdfFunctions::Function::ThreeSlabs,origin,w,0,length,knots);
            std::sort(knots.begin(),knots.end());knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
            J critical=J::array();
            for(double s:knots)critical.push_back({{"s",s},{"mean",field.mean->evaluate(origin+s*w).value}});
            const double groupStep=medium->flightSettings().selectionStepOverLength*ell;
            J groupBoundaries=J::array();
            for(int i=1;i*groupStep<length;++i)groupBoundaries.push_back(i*groupStep);
            J record={
                {"schema_version",1},
                {"units",{{"positions_and_distances","scene world units"},
                    {"sdf_values_and_sigma","scene world units"},
                    {"gradients","derivative of field value with respect to world position"}}},
                {"ray",{{"parameterization","x(s) = origin + s * direction, 0 <= s <= t"},
                    {"origin",json(origin)},{"direction",json(w)},{"t",length},
                    {"t_over_ell",length/configuredEll},{"endpoint_position",json(endpoint)}}},
                {"kernel",{{"type","squared_exponential"},
                    {"configured",{{"length_scale_ell",configuredEll},{"sigma",configuredSigma}}},
                    {"configured_spatial_covariance_formula","k(x,y) = sigma^2 * exp(-||x-y||^2 / (2 * ell^2))"},
                    {"effective_along_ray",{{"sigma",sigma},{"variance",variance},{"gamma",gamma},
                        {"length_scale_ell_along_ray",ell},
                        {"covariance_formula","k(s1,s2) = variance * exp(-0.5 * gamma^2 * (s1-s2)^2)"}}}}},
                {"start",{{"position",json(origin)},
                    {"sdf_mean",{{"value",start.value},{"gradient",json(start.gradient)},
                        {"directional_derivative",w.dot(start.gradient)},
                        {"definition","m(0), grad m(0), direction dot grad m(0); deterministic prior mean, not a GP observation."}}},
                    {"gp_observation",{{"f0",nullptr},{"gradient",nullptr},{"g0",nullptr},
                        {"definition","No observed birth value or gradient. The original single_history_gpis exterior start conditions only F(0)>0."},
                        {"kind","exterior_positive_only"},{"condition","F(0) > 0"}}}}},
                {"endpoint",{{"position",json(endpoint)},{"conditioned_gp_field_value",nullptr},
                    {"sdf_mean_value",end.value},{"sdf_mean_gradient",json(end.gradient)},
                    {"sdf_mean_directional_derivative",w.dot(end.gradient)},
                    {"note","Medium exit, not a sampled GP collision. F(t)=0 is not imposed; profile continues through all mean slabs regardless of random absorption."}}},
                {"sdf_curve",{{"definition","m(s) = production three_slabs mean(origin + s * direction); GP prior mean."},
                    {"sampling","Uniform production mean evaluations, no interpolation or resampling. Normalized columns use configured ell and sigma, as in F1-ray-input.json."},
                    {"domain",{0,length}},{"interval_count",n},{"sample_count",n+1},{"step",length/n},
                    {"columns",{"distance_s","distance_over_ell","sdf_mean","sdf_mean_over_sigma"}},
                    {"samples",samples}}},
                {"mean_field_scene_definition",mean},
                {"provenance",{{"case",choice},{"scene",scenePath},{"medium_type",definition.at("type")},
                    {"seed",seed},{"sample",sample},{"camera_origin",json(cameraOrigin)},
                    {"camera_direction_float32",json(eigen(direction.d))},{"camera_entry_distance",a},
                    {"pixel_convention","zero-based (x,y), top-left origin"},
                    {"ray_selection","Production PinholeCamera + IndependentPathSampler, seed/pixel/sample as recorded; direction normalized in double and clipped analytically to scene bounds. Not a transport event replay or a pixel-average path."}}},
                {"geometry",{{"centers",{json(centers[0]),json(centers[1]),json(centers[2])}},
                    {"half_size",json(slabHalf)},{"intersected_layers",intersections.size()},
                    {"slab_intersections",intersections},{"piecewise_linear_knots",critical},
                    {"mean_formula","min_i max_j(abs(position_j - slab_center_ij) - half_size_j)"},
                    {"note","This reproduces the renderer max-of-faces field; it is not Euclidean box distance near outside corners. Slab 2 still contributes to the union mean even when the ray misses it."}}},
                {"selection_groups",{{"step",groupStep},{"boundaries",groupBoundaries},
                    {"note","Original single_history selection group boundaries measured from medium entry; shown for alignment only, not a hazard calculation."}}},
                {"checks",{{"max_independent_mean_error",maxError},{"sample_count",n+1},
                    {"direction_norm",w.norm()},{"expected_layer_count_passed",true}}}
            };
            records.push_back(std::move(record));
            std::cout<<choice.at("id")<<": "<<intersections.size()<<" layers, "<<n+1<<" samples, error "<<maxError<<std::endl;
        }
        const auto output=std::filesystem::absolute(argv[2]);std::filesystem::create_directories(output.parent_path());
        std::ofstream f(output);f<<J{{"spec",spec},{"records",records}}.dump();
        if(!f)throw std::runtime_error("Cannot write output");
        medium->teardownAfterRender();return 0;
    }catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
