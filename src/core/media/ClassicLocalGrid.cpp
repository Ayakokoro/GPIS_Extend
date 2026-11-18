#include "ClassicLocalGrid.hpp"
#include "io/JsonObject.hpp"
#include "io/Scene.hpp"
#include "thread/ThreadPool.hpp"
#include "thread/ThreadUtils.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>
#if TUNGSTEN_NANOVDB_AVAILABLE
#include <nanovdb/NanoVDB.h>
#include <nanovdb/GridHandle.h>
#include <nanovdb/HostBuffer.h>
#include <nanovdb/io/IO.h>
#endif

namespace Tungsten {
void ClassicLocalField::fromJson(JsonPtr field,const Scene &scene) {
    file=scene.fetchResource(field.getRequiredMember("file"));
    field.getField("sdf_grid",sdfName); field.getField("density_grid",densityName);
    field.getField("transform",transform); field.getField("macrocell_size",macrocellSize);
    if (field.getField("macrocell_resolution",macrocellResolution)) {
        if (field["macrocell_size"] || macrocellResolution<=0)
            field.parseError("classic_local: macrocell_resolution must be positive and excludes macrocell_size");
    }
    if (field["alpha_grid"] || field["use_alpha_grid"])
        field.parseError("classic_local stationary mode takes gradient covariance from the kernel, not an alpha grid");
    if (sdfName.empty() || densityName.empty() || macrocellSize<1 || macrocellSize>64)
        field.parseError("classic_local: nonempty grid names and macrocell_size in [1,64] are required");
}
rapidjson::Value ClassicLocalField::toJson(rapidjson::Document::AllocatorType &allocator) const {
    JsonObject result{allocator,"file",*file,"sdf_grid",sdfName,"density_grid",densityName,"transform",transform};
    if (macrocellResolution) result.add("macrocell_resolution",macrocellResolution);
    else result.add("macrocell_size",macrocellSize);
    return result;
}
std::unique_ptr<ClassicLocalGrid> ClassicLocalField::load() const {
    return std::make_unique<ClassicLocalGrid>(file->absolute().asString(),sdfName,densityName,transform,macrocellSize,macrocellResolution);
}
#if TUNGSTEN_NANOVDB_AVAILABLE
namespace {
using V = Eigen::Vector3d;
using M = Eigen::Matrix3d;
constexpr double guard = 2048*std::numeric_limits<double>::epsilon();

struct GridView {
    nanovdb::GridHandle<nanovdb::HostBuffer> handle;
    const nanovdb::FloatGrid *grid = nullptr;
    M worldToIndex;
    V translation;
    Eigen::Vector3i minimum, maximum;
    double background;

    GridView(const std::string &file, const std::string &name, const M &worldToField, const V &fieldOffset)
        : handle(nanovdb::io::readGrid<nanovdb::HostBuffer>(file, name)) {
        grid = handle.grid<float>();
        if (!grid) throw std::runtime_error("classic_local: missing or non-float NVDB grid '"+name+"'");
        const auto origin = grid->indexToWorld(nanovdb::Vec3d(0));
        M indexToField;
        for (int a=0; a<3; ++a) {
            nanovdb::Vec3d unit(0); unit[a]=1;
            const auto point=grid->indexToWorld(unit);
            for (int b=0; b<3; ++b) indexToField(b,a)=point[b]-origin[b];
        }
        if (!indexToField.allFinite() || indexToField.determinant()==0)
            throw std::runtime_error("classic_local: invalid NVDB transform");
        const M fieldToIndex=indexToField.inverse();
        worldToIndex=fieldToIndex*worldToField;
        translation=fieldToIndex*(fieldOffset-V(origin[0],origin[1],origin[2]));
        background=grid->tree().root().background();
        if (!worldToIndex.allFinite() || !translation.allFinite() || !std::isfinite(background))
            throw std::runtime_error("classic_local: nonfinite grid transform/background");
        const auto box=grid->indexBBox();
        for (int a=0; a<3; ++a) {
            minimum[a]=box.min()[a]; maximum[a]=box.max()[a];
            if (minimum[a]>maximum[a] || minimum[a]<=std::numeric_limits<int>::min()+2 ||
                maximum[a]>=std::numeric_limits<int>::max()-2)
                throw std::runtime_error("classic_local: empty or invalid NVDB index bounds");
        }
    }

    double sample(const V &world, V *gradient=nullptr, bool supportOnly=false) const {
        const V q=worldToIndex*world+translation;
        if (!q.allFinite()) throw std::runtime_error("classic_local: nonfinite grid query");
        if (gradient) gradient->setZero();
        if ((q.array()<(minimum.cast<double>().array()-1)).any() ||
            (q.array()>(maximum.cast<double>().array()+1)).any()) return background;
        const Eigen::Vector3i base=q.array().floor().cast<int>();
        const V fraction=q-base.cast<double>();
        auto accessor=grid->getAccessor();
        if(supportOnly && (base.array()>=minimum.array()).all() && (base.array()<maximum.array()).all()) {
            // Density nodes are finite and nonnegative. The nearest corner
            // contributes at least 1/8 of its value even on voxel faces.
            const nanovdb::Coord nearest(base[0]+(fraction.x()>=.5),
                                         base[1]+(fraction.y()>=.5),
                                         base[2]+(fraction.z()>=.5));
            if(accessor.getValue(nearest)>0) return 1;
        }
        double value=0,corners[2][2][2]; V g=V::Zero();
        for (int z=0; z<2; ++z) for (int y=0; y<2; ++y) for (int x=0; x<2; ++x) {
            const double node=accessor.getValue(nanovdb::Coord(base[0]+x,base[1]+y,base[2]+z));
            if (!std::isfinite(node)) throw std::runtime_error("classic_local: nonfinite grid value");
            corners[x][y][z]=node;
            const double wx=x ? fraction.x() : 1-fraction.x();
            const double wy=y ? fraction.y() : 1-fraction.y();
            const double wz=z ? fraction.z() : 1-fraction.z();
            value+=node*wx*wy*wz;
        }
        if (gradient) {
            // Interpolated edge differences give the same trilinear derivative
            // without turning a constant field's cancellation error into a normal.
            for (int j=0; j<2; ++j) for (int k=0; k<2; ++k) {
                const double wx=j ? fraction.x() : 1-fraction.x();
                const double wy=j ? fraction.y() : 1-fraction.y();
                const double wz=k ? fraction.z() : 1-fraction.z();
                const double yk=k ? fraction.y() : 1-fraction.y();
                g.x()+=(corners[1][j][k]-corners[0][j][k])*wy*wz;
                g.y()+=(corners[j][1][k]-corners[j][0][k])*wx*wz;
                g.z()+=(corners[j][k][1]-corners[j][k][0])*wx*yk;
            }
            *gradient=worldToIndex.transpose()*g;
        }
        return value;
    }
};
}

struct ClassicLocalGrid::Impl {
    GridView sdf, density;
    V width;
    Eigen::Vector3i first, last, dimensions;
    std::vector<double> maxima;
    double densityAt(const V &world,bool supportOnly) const {
        // Match macrofacet's NanoVdbMean active domain: [minNode,maxNode+1].
        const V q=sdf.worldToIndex*world+sdf.translation;
        if ((q.array()<sdf.minimum.cast<double>().array()).any() ||
            (q.array()>(sdf.maximum.cast<double>().array()+1)).any()) return 0;
        return density.sample(world,nullptr,supportOnly);
    }
    double face(int axis, int index) const {
        return index==dimensions[axis] ? double(last[axis]) : first[axis]+index*width[axis];
    }
    Impl(const std::string &file, const std::string &sdfName, const std::string &densityName,
         const M &worldToField, const V &offset, int size, int resolution)
        : sdf(file,sdfName,worldToField,offset), density(file,densityName,worldToField,offset) {
        if (density.background!=0) throw std::runtime_error("classic_local: density background must be zero");
        // Include interpolation support between active nodes and vacuum nodes.
        first=density.minimum-Eigen::Vector3i::Ones();
        last=density.maximum+Eigen::Vector3i::Ones();
        size_t count=1;
        for (int a=0; a<3; ++a) {
            const int64_t extent=int64_t(last[a])-first[a];
            const int64_t n=resolution ? resolution : (extent+size-1)/size;
            if (n>64000000/int64_t(count)) throw std::runtime_error("classic_local: density majorant grid is too large");
            dimensions[a]=int(n);
            width[a]=resolution ? double(extent)/resolution : double(size);
            count*=size_t(n);
        }
        maxima.resize(count);
        const uint32 depth=uint32(dimensions.z());
        const auto workers=ThreadUtils::pool ? std::max(1u,ThreadUtils::pool->threadCount()) : 1u;
        ThreadUtils::parallelFor(0,depth,std::min(workers,depth),[&](uint32 z) {
            auto accessor=density.grid->getAccessor();
            for (int y=0; y<dimensions.y(); ++y)
            for (int x=0; x<dimensions.x(); ++x) {
                const Eigen::Vector3i cell(x,y,int(z));
                Eigen::Vector3i lo,hi;
                for (int a=0; a<3; ++a) {
                    // One-cell halo covers cell selection at rounded DDA faces.
                    lo[a]=std::max(density.minimum[a],int(std::floor(face(a,cell[a])))-1);
                    hi[a]=std::min(density.maximum[a],int(std::ceil(face(a,cell[a]+1)))+1);
                }
                double maximum=0;
                for (int k=lo.z(); k<=hi.z(); ++k) for (int j=lo.y(); j<=hi.y(); ++j)
                    for (int i=lo.x(); i<=hi.x(); ++i) {
                        const double value=accessor.getValue(nanovdb::Coord(i,j,k));
                        if (!(value>=0) || !std::isfinite(value))
                            throw std::runtime_error("classic_local: density must be finite and nonnegative");
                        maximum=std::max(maximum,value);
                    }
                maxima[(size_t(z)*dimensions.y()+y)*dimensions.x()+x]=maximum==0 ? 0 :
                    std::nextafter(maximum*(1+guard),std::numeric_limits<double>::infinity());
            }
        });
    }
};

ClassicLocalGrid::ClassicLocalGrid(const std::string &file, const std::string &sdfName,
        const std::string &densityName, const Mat4f &fieldToWorld, int size, int resolution) {
    if (size<1 || size>64) throw std::invalid_argument("classic_local: macrocell_size must be in [1,64]");
    if (resolution<0) throw std::invalid_argument("classic_local: negative macrocell_resolution");
    Eigen::Matrix4d transform;
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) transform(i,j)=fieldToWorld(i,j);
    if (!transform.allFinite() || transform.row(3)!=Eigen::RowVector4d(0,0,0,1) ||
        transform.topLeftCorner<3,3>().determinant()==0)
        throw std::invalid_argument("classic_local: field transform must be finite, affine and invertible");
    const Eigen::Matrix4d inverse=transform.inverse();
    _impl=std::make_unique<Impl>(file,sdfName,densityName,inverse.topLeftCorner<3,3>(),
                               inverse.topRightCorner<3,1>(),size,resolution);
}
ClassicLocalGrid::~ClassicLocalGrid()=default;
double ClassicLocalGrid::density(const V &world) const {
    return _impl->densityAt(world,false);
}
bool ClassicLocalGrid::hasDensity(const V &world) const { return _impl->densityAt(world,true)!=0; }
V ClassicLocalGrid::meanGradient(const V &world) const {
    V gradient; _impl->sdf.sample(world,&gradient); return gradient;
}

ClassicLocalGrid::Cursor::Cursor(const ClassicLocalGrid &grid,const V &origin,
        const V &direction,double begin,double end)
    : _grid(grid._impl.get()), _origin(_grid->density.worldToIndex*origin+_grid->density.translation),
      _direction(_grid->density.worldToIndex*direction), _begin(begin), _end(end) {
    if (!_origin.allFinite() || !_direction.allFinite())
        throw std::runtime_error("classic_local: nonfinite grid ray");
    for (int a=0; a<3; ++a) {
        if (_direction[a]==0) {
            if (_origin[a]<_grid->first[a] || _origin[a]>_grid->last[a]) _end=_begin;
        } else {
            const double t0=(_grid->first[a]-_origin[a])/_direction[a];
            const double t1=(_grid->last[a]-_origin[a])/_direction[a];
            _begin=std::max(_begin,std::min(t0,t1)); _end=std::min(_end,std::max(t0,t1));
        }
    }
    const V sdfOrigin=_grid->sdf.worldToIndex*origin+_grid->sdf.translation;
    const V sdfDirection=_grid->sdf.worldToIndex*direction;
    for (int a=0; a<3; ++a) {
        const double low=_grid->sdf.minimum[a],high=double(_grid->sdf.maximum[a])+1;
        if (sdfDirection[a]==0) {
            if (sdfOrigin[a]<low || sdfOrigin[a]>high) _end=_begin;
        } else {
            const double t0=(low-sdfOrigin[a])/sdfDirection[a],t1=(high-sdfOrigin[a])/sdfDirection[a];
            _begin=std::max(_begin,std::min(t0,t1)); _end=std::min(_end,std::max(t0,t1));
        }
    }
    // Halo certificates require transform roundoff to stay within one cell.
    const V error=64*std::numeric_limits<double>::epsilon()*
        (V::Ones()+_grid->density.worldToIndex.cwiseAbs()*origin.cwiseAbs()+
         _grid->density.translation.cwiseAbs()+std::max(std::abs(_begin),std::abs(_end))*
         (_grid->density.worldToIndex.cwiseAbs()*direction.cwiseAbs()));
    if (_begin<_end && (!error.allFinite() || error.maxCoeff()>=0.25))
        throw std::runtime_error("classic_local: grid ray transform lacks precision");
    if (!(_begin<_end)) return;
    for (int a=0; a<3; ++a) {
        const double q=_origin[a]+_begin*_direction[a];
        const double u=std::clamp((q-_grid->first[a])/_grid->width[a],0.0,double(_grid->dimensions[a]-1));
        int index=int(std::floor(u));
        _nextFace[a]=_end;
        if (_direction[a]!=0) {
            const auto crossing=[&](int face) {
                return (_grid->face(a,face)-_origin[a])/_direction[a];
            };
            if (_direction[a]>0) {
                while (index+1<_grid->dimensions[a] && crossing(index+1)<=_begin) ++index;
                while (index>0 && crossing(index)>_begin) --index;
                _nextFace[a]=crossing(index+1);
            } else {
                while (index>0 && crossing(index)<=_begin) --index;
                while (index+1<_grid->dimensions[a] && crossing(index+1)>_begin) ++index;
                _nextFace[a]=crossing(index);
            }
        }
        _cell[a]=index;
    }
}

bool ClassicLocalGrid::Cursor::next(Segment &segment) {
    if (!(_begin<_end)) return false;
    const double end=std::min(_end,_nextFace.minCoeff());
    if (!(end>_begin)) throw std::runtime_error("classic_local: density DDA failed to advance");
    const size_t index=(size_t(_cell.z())*_grid->dimensions.y()+_cell.y())*_grid->dimensions.x()+_cell.x();
    segment={_begin,end,_grid->maxima[index]};
    if (end<_end) for (int a=0;a<3;++a) {
        // Recompute absolute crossings, including all axes tied at an edge.
        // Adding a constant delta instead would accumulate rounding error.
        while (_nextFace[a]<=end) {
            _cell[a]+=_direction[a]>0 ? 1 : -1;
            if (_cell[a]<0 || _cell[a]>=_grid->dimensions[a])
                throw std::runtime_error("classic_local: density DDA crossed outside clipped domain");
            const int face=_cell[a]+(_direction[a]>0 ? 1 : 0);
            _nextFace[a]=(_grid->face(a,face)-_origin[a])/_direction[a];
        }
    }
    _begin=end; return true;
}
void ClassicLocalGrid::Cursor::appendVoxelBoundaries(std::vector<double> &knots) const {
    if (!(_end>_begin)) return;
    knots.push_back(_begin);
    knots.push_back(_end);
    for (int a=0;a<3;++a) {
        if (_direction[a]==0) continue;
        const double p=_origin[a]+_begin*_direction[a], q=_origin[a]+_end*_direction[a];
        const int first=std::max(_grid->first[a],int(std::ceil(std::min(p,q))));
        const int last=std::min(_grid->last[a],int(std::floor(std::max(p,q))));
        for (int face=first;face<=last;++face) {
            const double t=(face-_origin[a])/_direction[a];
            if (t>_begin && t<_end) knots.push_back(t);
        }
    }
}
#else
struct ClassicLocalGrid::Impl {};
ClassicLocalGrid::ClassicLocalGrid(const std::string &,const std::string &,const std::string &,const Mat4f &,int,int) {
    throw std::runtime_error("classic_local requires a build with NanoVDB support");
}
ClassicLocalGrid::~ClassicLocalGrid()=default;
double ClassicLocalGrid::density(const Eigen::Vector3d &) const { throw std::runtime_error("NanoVDB unavailable"); }
bool ClassicLocalGrid::hasDensity(const Eigen::Vector3d &) const { throw std::runtime_error("NanoVDB unavailable"); }
Eigen::Vector3d ClassicLocalGrid::meanGradient(const Eigen::Vector3d &) const { throw std::runtime_error("NanoVDB unavailable"); }
ClassicLocalGrid::Cursor::Cursor(const ClassicLocalGrid &,const Eigen::Vector3d &,const Eigen::Vector3d &,double,double) {
    throw std::runtime_error("NanoVDB unavailable");
}
bool ClassicLocalGrid::Cursor::next(Segment &) { return false; }
void ClassicLocalGrid::Cursor::appendVoxelBoundaries(std::vector<double> &) const {}
#endif
bool ClassicLocalGrid::Cursor::nextOccupiedRange(double &begin,double &end) {
    Segment block;
    while (next(block)) {
        if (block.densityMaximum==0) continue;
        begin=block.begin;
        end=block.end;
        while (next(block) && block.densityMaximum>0) end=block.end;
        return true;
    }
    return false;
}
}
