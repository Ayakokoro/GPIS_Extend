#include "SceneField.hpp"

#include "io/DirectoryChange.hpp"
#include "io/JsonDocument.hpp"
#include "math/GPFunctions.hpp"
#include "math/conditional/gpss/SdfMeanField.hpp"
#include "media/GpisSdfGrid.hpp"
#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
#include "media/NeuralSdfGrid.hpp"
#include <nanovdb/HostBuffer.h>
#include <nanovdb/io/IO.h>
#endif
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <stdexcept>
#include <unordered_set>

namespace Tungsten::SinglePointTraining {
namespace {
namespace cg = ConditionalGpis;
using Json = nlohmann::json;
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;

Json describe(const JsonSerializable &object) {
    rapidjson::Document owner;
    auto value = object.toJson(owner.GetAllocator());
    rapidjson::StringBuffer buffer;
    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
    value.Accept(writer);
    return Json::parse(buffer.GetString());
}

bool gpMedium(const Json &medium) {
    static const std::set<std::string> types{
        "gaussian_process", "function_space_gaussian_process", "weight_space_gaussian_process",
        "sparse_conv_noise", "renewal_plus_fp_gradient", "classic_global",
        "classic_global_fp_gradient", "first_hit_gpis", "classic_local",
        "global_conditional", "global_conditional_neural", "sparse_history_gpis", "single_history_gpis",
        "neural_causal_renewal", "poisson_ellipsoid_local"};
    return types.count(medium.value("type", std::string())) != 0;
}

Json vectorJson(const V &v) { return Json::array({v.x(), v.y(), v.z()}); }
Json boundsJson(const cg::Bounds3 &b) {
    return {{"minimum",vectorJson(b.minimum)},{"maximum",vectorJson(b.maximum)}};
}

cg::Bounds3 mappedBounds(const cg::Bounds3 &bounds, const M &linear, const V &translation) {
    cg::Bounds3 result{V::Constant(std::numeric_limits<double>::infinity()),
        V::Constant(-std::numeric_limits<double>::infinity())};
    for (int corner=0; corner<8; ++corner) {
        V p;
        for (int axis=0; axis<3; ++axis)
            p[axis] = (corner & (1 << axis)) ? bounds.maximum[axis] : bounds.minimum[axis];
        p = linear*p + translation;
        result.minimum = result.minimum.cwiseMin(p);
        result.maximum = result.maximum.cwiseMax(p);
    }
    return result;
}

// Matches classic_local's field.transform convention: transform coordinates,
// preserve SDF scalar values, and apply the inverse-transpose to gradients.
class TransformedMean final : public cg::MeanField {
    std::shared_ptr<const cg::MeanField> _base;
    M _inverse;
    V _offset;
    double _gradientScale, _voxelScale;
public:
    TransformedMean(std::shared_ptr<const cg::MeanField> base, const M &linear, const V &offset)
        : _base(std::move(base)), _inverse(linear.inverse()), _offset(-_inverse*offset) {
        Eigen::JacobiSVD<M> decomposition(linear);
        _voxelScale=decomposition.singularValues().minCoeff();
        _gradientScale=1.0/_voxelScale;
        if (!_inverse.allFinite() || !(_voxelScale>0) || !std::isfinite(_gradientScale))
            throw std::runtime_error("training SDF: singular/nonfinite field transform");
    }
    const char *typeName() const override { return "transformed_nanovdb"; }
    cg::MeanJet evaluate(const V &p) const override {
        auto jet=_base->evaluate(_inverse*p+_offset);
        jet.gradient=_inverse.transpose()*jet.gradient;
        return jet;
    }
    cg::MeanJet evaluateInCell(const V &p,const V &interior) const override {
        auto jet=_base->evaluateInCell(_inverse*p+_offset,_inverse*interior+_offset);
        jet.gradient=_inverse.transpose()*jet.gradient;
        return jet;
    }
    cg::BoundsSummary bounds(const cg::Bounds3 &domain) const override {
        auto result=_base->bounds(mappedBounds(domain,_inverse,_offset));
        result.maximumGradientNorm*= _gradientScale;
        return result;
    }
    double voxelSizeHint() const override { return _voxelScale*_base->voxelSizeHint(); }
    void appendRayBreakpoints(const V &o,const V &w,double a,double b,
        std::vector<double> &knots) const override {
        _base->appendRayBreakpoints(_inverse*o+_offset,_inverse*w,a,b,knots);
    }
    void requireFullRayCoverage(const V &o,const V &w,double length) const override {
        _base->requireFullRayCoverage(_inverse*o+_offset,_inverse*w,length);
    }
};

cg::CovarianceKernel loadKernel(Scene &scene, const Json &description, const Path &source,
    bool nativeDouble) {
    if (description.value("type", std::string()) != "squared_exponential")
        throw std::runtime_error("training SDF requires a stationary squared_exponential covariance");
    if (nativeDouble) {
        // NeuralRenewalMedium uses these doubles directly, rather than the
        // float legacy CovarianceFunction representation.
        const double sigma=description.at("sigma").get<double>();
        const double ell=description.at("lengthScale").get<double>();
        if (!(sigma>0 && ell>0) || !std::isfinite(sigma) || !std::isfinite(ell))
            throw std::runtime_error("training SDF requires finite positive sigma and lengthScale");
        return cg::CovarianceKernel::fromCorrelationLengths(sigma,V::Constant(ell));
    }
    JsonDocument config(source,description.dump());
    auto covariance=scene.fetchCovarianceFunction(config);
    if (!dynamic_cast<SquaredExponentialCovariance *>(covariance.get()))
        throw std::runtime_error("training SDF covariance is not stationary SE");
    covariance->loadResources();
    const Vec3d zero(0.0);
    const double variance=(*covariance)(Derivative::None,Derivative::None,zero,zero,zero,zero);
    M gradientCovariance;
    for (int i=0; i<3; ++i) for (int j=0; j<3; ++j) {
        Vec3d a(0.0),b(0.0); a[i]=1; b[j]=1;
        gradientCovariance(i,j)=(*covariance)(Derivative::First,Derivative::First,zero,zero,a,b);
    }
    return cg::CovarianceKernel(std::sqrt(variance),gradientCovariance/variance);
}

#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
void validateNeuralGridMap(const std::string &file) {
    // NeuralSdfGrid represents native nodes as origin + dx * index. Refuse
    // native maps that this existing reader cannot represent; field.transform
    // remains supported separately by TransformedMean.
    const auto handle=nanovdb::io::readGrid<nanovdb::HostBuffer>(file,"sdf");
    const auto *grid=handle.grid<float>();
    if (!grid) throw std::runtime_error("training NanoVDB sdf must be a float grid");
    const auto nativeOrigin=grid->map().applyMap(nanovdb::Vec3d(0.0));
    const V origin(nativeOrigin[0],nativeOrigin[1],nativeOrigin[2]);
    const double dx=grid->voxelSize()[0];
    M linear;
    for (int axis=0;axis<3;++axis) {
        nanovdb::Vec3d index(0.0); index[axis]=1.0;
        const auto p=grid->map().applyMap(index);
        linear.col(axis)=V(p[0],p[1],p[2])-origin;
    }
    const double tolerance=64*std::numeric_limits<double>::epsilon()*
        std::max(origin.norm(),std::abs(dx));
    if (!(dx>0) || !std::isfinite(dx) || !linear.allFinite() ||
        (linear-dx*M::Identity()).cwiseAbs().maxCoeff()>tolerance)
        throw std::runtime_error("training NanoVDB native map must be axis-aligned with uniform voxel size; field.transform is supported separately");
}
#endif
} // namespace

SceneField loadSceneField(const Path &sceneFile,
    const std::optional<cg::Bounds3> &domainOverride) {
    const Path source=sceneFile.absolute();
    if (domainOverride && !domainOverride->valid())
        throw std::invalid_argument("training SDF domain override must be a finite nonempty box");
    SceneField result;
    result.scene.reset(Scene::load(source));
    // Scene::fromJson keeps inline camera/primitive media outside media().
    // References to a named medium must not count as additional fields.
    std::unordered_set<const Medium *> seen;
    std::vector<Json> fields;
    auto inspect=[&](const std::shared_ptr<Medium> &medium) {
        if (!medium || !seen.insert(medium.get()).second) return;
        Json description=describe(*medium);
        if (gpMedium(description)) fields.push_back(std::move(description));
    };
    for (const auto &medium:result.scene->media()) inspect(medium);
    if (result.scene->camera()) inspect(result.scene->camera()->medium());
    for (const auto &primitive:result.scene->primitives()) {
        inspect(primitive->intMedium()); inspect(primitive->extMedium());
    }
    if (fields.size()!=1)
        throw std::runtime_error("training scene must contain exactly one GPIS SDF medium; found "+
            std::to_string(fields.size())+" distinct GPIS media (including inline references)");
    const Json description=fields.front();
    const std::string type=description.at("type").get<std::string>();
    DirectoryChange directory(source.parent());
    // Keep Tungsten's scene syntax, including comments and trailing commas.
    const JsonDocument original(source);
    cg::Bounds3 inferredDomain;
    bool haveDomain=false;
    std::string asset,loader;
    Json covariance;
    if (description.contains("field")) {
        if (type!="classic_local" && type!="neural_causal_renewal")
            throw std::runtime_error("unsupported GPIS field medium for training: "+type);
        const auto &field=description.at("field");
        const std::string gridName=field.value("sdf_grid",std::string("sdf"));
        if (gridName!="sdf")
            throw std::runtime_error("training NanoVDB reader currently requires sdf_grid=sdf");
        asset=result.scene->fetchResource(field.at("file").get<std::string>())->absolute().asString();
#ifdef TUNGSTEN_ENABLE_NEURAL_RENEWAL
        validateNeuralGridMap(asset);
        auto grid=NeuralSdfGrid::open(std::filesystem::path(asset));
        result.mean=grid;
        inferredDomain=grid->activeNodeBounds();
        if (field.contains("transform")) {
            JsonDocument config(source,field.dump());
            Mat4f transform; config["transform"].get(transform);
            Eigen::Matrix4d full;
            for (int i=0;i<4;++i) for(int j=0;j<4;++j) full(i,j)=transform(i,j);
            if (!full.allFinite() || !full.row(3).isApprox(Eigen::RowVector4d(0,0,0,1),0))
                throw std::runtime_error("training SDF field transform must be a finite affine transform");
            const M linear=full.topLeftCorner<3,3>();
            const V translation=full.topRightCorner<3,1>();
            if (!linear.isApprox(M::Identity(),0) || !translation.isZero(0)) {
                result.mean=std::make_shared<TransformedMean>(grid,linear,translation);
                inferredDomain=mappedBounds(inferredDomain,linear,translation);
            }
        }
        loader="NeuralSdfGrid (existing full-domain sdf grid only)";
        haveDomain=true;
#else
        throw std::runtime_error("NanoVDB training requires TUNGSTEN_ENABLE_NEURAL_RENEWAL=ON");
#endif
        covariance=description.at("covariance");
    } else if (description.contains("gaussian_process")) {
        const auto &gp=description.at("gaussian_process");
        if (gp.value("type",std::string("standard"))!="standard" ||
            gp.value("embed_cov",false) || gp.contains("conditioning_data"))
            throw std::runtime_error("training accepts one unconditioned standard GP, not CSG/embedded/conditioned processes");
        Json meanDescription=gp.at("mean");
        if (meanDescription.value("volume",false))
            throw std::runtime_error("training requires an SDF mean, not a volume-density mean");
        if (meanDescription.contains("grid")) {
            auto &grid=meanDescription.at("grid");
            if (grid.value("request_sdf",false) ||
                grid.value("integration_method",std::string())=="residual_ratio")
                throw std::runtime_error("training reads existing SDF data: disable request_sdf and residual_ratio grid baking");
            // VdbGrid::toJson does not preserve request_sdf. Check the original
            // selected asset as well so a render-time SDF conversion cannot be
            // silently interpreted as an already baked SDF.
            if (grid.contains("file")) {
                const auto selected=result.scene->fetchResource(grid.at("file").get<std::string>())->absolute().normalize();
                const auto check=[&](const auto &self,JsonPtr node)->void {
                    if (node.isObject()) {
                        const auto nodeType=node["type"],file=node["file"],request=node["request_sdf"];
                        if (nodeType.isString() && nodeType.cast<std::string>()=="vdb" &&
                            file && request && request.cast<bool>()) {
                            const auto candidate=result.scene->fetchResource(file)->absolute().normalize();
                            if (candidate==selected)
                                throw std::runtime_error("training SDF asset requests request_sdf=true; supply an already baked SDF");
                        }
                        for (const auto &member:node) self(self,member.second);
                    } else if (node.isArray()) {
                        for (unsigned i=0;i<node.size();++i) self(self,node[i]);
                    }
                };
                check(check,original);
            }
            // Gradients come from the exact trilinear interpolant below.
            // Loading an auxiliary finite-difference gradient grid is unnecessary.
            grid["request_gradient"]=false;
        }
        JsonDocument config(source,meanDescription.dump());
        auto mean=result.scene->fetchMeanFunction(config);
        mean->loadResources();
        if (auto tabulated=dynamic_cast<TabulatedMean *>(mean.get())) {
#if OPENVDB_AVAILABLE
            auto vdb=dynamic_cast<VdbGrid *>(tabulated->grid().get());
            if (!vdb) throw std::runtime_error("training tabulated SDF currently requires Tungsten's VDB grid");
            auto grid=std::make_shared<GpisSdfGrid>();
            grid->initialize(*tabulated,*vdb);
            result.mean=grid;
            cg::Bounds3 indexBounds;
            for (int i=0;i<3;++i) {
                indexBounds.minimum[i]=grid->data.clampMin[i];
                indexBounds.maximum[i]=grid->data.clampMax[i];
            }
            const M inverse=grid->worldToIndex.inverse();
            inferredDomain=mappedBounds(indexBounds,inverse,-inverse*grid->translation);
            haveDomain=true;
            const Json gridDescription=describe(*tabulated->grid());
            asset=result.scene->fetchResource(gridDescription.at("file").get<std::string>())->absolute().asString();
            loader="GpisSdfGrid / VdbGrid (same trilinear SDF, transform, scale, offset and clamp as rendering)";
#else
            throw std::runtime_error("VDB training requires OpenVDB support");
#endif
        } else if (auto sphere=dynamic_cast<SphericalMean *>(mean.get())) {
            const auto c=sphere->center();
            result.mean=std::make_shared<cg::SphereMean>(V(c.x(),c.y(),c.z()),sphere->radius());
            loader="SphereMean";
        } else if (mean->sdfQueries()) {
            result.mean=std::make_shared<cg::SdfMeanField>(std::move(mean));
            loader="SdfMeanField / existing Tungsten SDF queries";
        } else {
            throw std::runtime_error("selected GPIS mean has no supported SDF query representation");
        }
        covariance=gp.at("covariance");
    } else {
        throw std::runtime_error("selected GPIS medium has no supported SDF description: "+type);
    }
    if (!haveDomain && !domainOverride)
        throw std::runtime_error("procedural training SDF requires an explicit finite training domain");
    result.field.mean=result.mean;
    result.field.kernel=loadKernel(*result.scene,covariance,source,type=="neural_causal_renewal");
    const double precision=result.field.kernel.metric().trace()/3.0;
    if (!(precision>0) || !result.field.kernel.metric().isApprox(precision*M::Identity(),1e-6))
        throw std::runtime_error("single-point training currently requires isotropic SE covariance with one scalar lengthScale");
    result.field.activeDomain=domainOverride ? *domainOverride : inferredDomain;
    result.field.validate();
    Json metric=Json::array();
    for (int i=0;i<3;++i) metric.push_back(Json::array({result.field.kernel.metric()(i,0),
        result.field.kernel.metric()(i,1),result.field.kernel.metric()(i,2)}));
    result.metadata={{"scene",source.asString()},{"source_medium",description},
        {"source_medium_type",type},{"source_medium_name",description.value("name",std::string())},
        {"gp_field_count",1},{"sdf_asset",asset},{"loader",loader},
        {"domain",boundsJson(result.field.activeDomain)},{"domain_override",bool(domainOverride)},
        {"covariance",covariance},{"resolved_sigma",result.field.kernel.sigma()},
        {"resolved_metric",metric},{"rebaked",false},{"loaded_model_weights",false}};
    if (description.contains("field")) {
        result.metadata["sdf_support"]="NanoVDB float grid named sdf; full active-node domain; axis-aligned uniform native voxel map; affine field.transform supported";
        result.metadata["ignored_auxiliary_grids"]=Json::array({"density","alpha","sigma"});
    }
    if (haveDomain) result.metadata["native_sdf_domain"]=boundsJson(inferredDomain);
    return result;
}

} // namespace Tungsten::SinglePointTraining
