#include "../tools/single_point_training/SceneField.hpp"
#include <openvdb/openvdb.h>
#include <openvdb/io/File.h>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

namespace fs=std::filesystem;
using Json=nlohmann::json;
using Tungsten::Path;
using Tungsten::SinglePointTraining::loadSceneField;
using V=Eigen::Vector3d;

namespace {
void require(bool condition,const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}
void close(double actual,double expected,const char *message) {
    require(std::abs(actual-expected)<2e-6,message);
}
struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        path=fs::temp_directory_path()/(
            "tungsten-single-point-scene-"+std::to_string(
                std::chrono::high_resolution_clock::now().time_since_epoch().count()));
        fs::create_directories(path/"assets");
    }
    ~TemporaryDirectory() { std::error_code error; fs::remove_all(path,error); }
};
fs::path writeScene(const fs::path &directory,const char *name,const Json &description) {
    const fs::path path=directory/name;
    std::ofstream file(path);
    file<<description.dump(2);
    if (!file) throw std::runtime_error("cannot write test scene");
    return path;
}
template<class F> void rejected(F action,const std::string &expected) {
    try { action(); } catch(const std::exception &error) {
        require(std::string(error.what()).find(expected)!=std::string::npos,
            std::string("unexpected failure: ")+error.what());
        return;
    }
    throw std::runtime_error("expected rejection: "+expected);
}
} // namespace

int main() try {
    TemporaryDirectory temporary;
    openvdb::initialize();
    auto grid=openvdb::FloatGrid::create(10.0f);
    grid->setName("sdf");
    grid->setTransform(openvdb::math::Transform::createLinearTransform(0.1));
    auto accessor=grid->getAccessor();
    for (int z=-6;z<=6;++z) for(int y=-6;y<=6;++y) for(int x=-6;x<=6;++x)
        accessor.setValue(openvdb::Coord(x,y,z),float(0.1*x+0.25));
    openvdb::io::File volume((temporary.path/"assets"/"plane.vdb").string());
    openvdb::GridPtrVec grids; grids.push_back(grid); volume.write(grids); volume.close();

    const Json gp={{"type","global_conditional"},{"name","gpis"},
        {"gaussian_process",{{"type","standard"},
            {"mean",{{"type","tabulated"},{"offset",0.1},{"scale",2.0},
                {"grid",{{"type","vdb"},{"file","assets/plane.vdb"},
                    {"density_name","sdf"},{"emission_name",""},{"normalize_size",false},{"interpolate","linear"},
                    {"request_sdf",false},{"integration_method","raymarching"},
                    {"sampling_method","raymarching"}}}}},
            {"covariance",{{"type","squared_exponential"},{"sigma",0.02},{"lengthScale",0.2}}}}}};
    Json scene={{"media",Json::array({gp})},
        {"camera",{{"type","pinhole"},{"medium","gpis"}}},
        {"bsdfs",Json::array({{{"type","lambert"},{"name","unused_material"},
            {"albedo","does-not-exist.png"}}})},
        {"primitives",Json::array({{{"type","cube"},{"int_medium","gpis"},
            {"bsdf","unused_material"}}})}};
    auto validPath=writeScene(temporary.path,"valid.json",scene);
    auto loaded=loadSceneField(Path(validPath.string()));
    require(loaded.field.activeDomain.valid(),"VDB inferred domain is invalid");
    auto jet=loaded.mean->evaluate(V::Zero());
    close(jet.value,0.7,"VDB value/offset/scale changed");
    close(jet.gradient.x(),2.0,"VDB exact trilinear gradient changed");
    close(jet.gradient.y(),0.0,"VDB y gradient changed");
    close(loaded.field.kernel.sigma(),0.02,"scene covariance sigma changed");
    require(loaded.metadata.at("gp_field_count")==1,"named references double-counted");
    require(fs::equivalent(fs::path(loaded.metadata.at("sdf_asset").get<std::string>()),
        temporary.path/"assets"/"plane.vdb"),"scene-relative SDF path not resolved");
    require(!loaded.metadata.at("loaded_model_weights").get<bool>(),"unexpected weights load");

    const fs::path relaxedPath=temporary.path/"commented.json";
    {
        std::string text=scene.dump(2);
        text.pop_back();
        std::ofstream file(relaxedPath);
        file<<"// Tungsten scene syntax includes comments and trailing commas.\n"
            <<text<<",\n\"unused_optional_number\": NaN,\n}\n";
    }
    auto relaxedLoaded=loadSceneField(Path(relaxedPath.string()));
    close(relaxedLoaded.mean->evaluate(V::Zero()).value,0.7,"Tungsten JSON syntax changed");

    Json inlineOnly={{"camera",{{"type","pinhole"},{"medium",gp}}}};
    auto inlinePath=writeScene(temporary.path,"inline.json",inlineOnly);
    auto inlineLoaded=loadSceneField(Path(inlinePath.string()));
    close(inlineLoaded.mean->evaluate(V::Zero()).value,0.7,"inline GP SDF differs");

    Json second=scene; second["camera"]["medium"]=gp;
    auto secondPath=writeScene(temporary.path,"second.json",second);
    rejected([&]{loadSceneField(Path(secondPath.string()));},"found 2");
    auto emptyPath=writeScene(temporary.path,"empty.json",
        Json{{"camera",{{"type","pinhole"}}}});
    rejected([&]{loadSceneField(Path(emptyPath.string()));},"found 0");

    Json anisotropic=scene;
    anisotropic["media"][0]["gaussian_process"]["covariance"]["aniso"]=Json::array({1.0,2.0,1.0});
    auto anisotropicPath=writeScene(temporary.path,"anisotropic.json",anisotropic);
    rejected([&]{loadSceneField(Path(anisotropicPath.string()));},"isotropic SE");

    Json legacy=scene;
    legacy["media"][0]["type"]="function_space_gaussian_process";
    legacy["media"][0]["correlation_context"]="global";
    legacy["media"][0]["intersect_method"]="gp_discrete";
    legacy["media"][0]["normal_method"]="conditioned_gaussian";
    legacy["media"][0]["gaussian_process"]["mean"]["grid"]["request_sdf"]=true;
    auto legacyPath=writeScene(temporary.path,"request_rebake.json",legacy);
    rejected([&]{loadSceneField(Path(legacyPath.string()));},"request_sdf");

    std::cout<<"single-point scene adapter checks passed\n";
    return 0;
} catch(const std::exception &error) {
    std::cerr<<error.what()<<'\n';
    return 1;
}
