#include "ClassicLocalGrid.hpp"
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

    double sample(const V &world, V *gradient=nullptr) const {
        const V q=worldToIndex*world+translation;
        if (!q.allFinite()) throw std::runtime_error("classic_local: nonfinite grid query");
        if (gradient) gradient->setZero();
        if ((q.array()<(minimum.cast<double>().array()-1)).any() ||
            (q.array()>(maximum.cast<double>().array()+1)).any()) return background;
        const Eigen::Vector3i base=q.array().floor().cast<int>();
        const V fraction=q-base.cast<double>();
        auto accessor=grid->getAccessor();
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
    int width;
    Eigen::Vector3i first, last, dimensions;
    std::vector<double> maxima;
    Impl(const std::string &file, const std::string &sdfName, const std::string &densityName,
         const M &worldToField, const V &offset, int size)
        : sdf(file,sdfName,worldToField,offset), density(file,densityName,worldToField,offset), width(size) {
        if (density.background!=0) throw std::runtime_error("classic_local: density background must be zero");
        // Include interpolation support between active nodes and vacuum nodes.
        first=density.minimum-Eigen::Vector3i::Ones();
        last=density.maximum+Eigen::Vector3i::Ones();
        for (int a=0; a<3; ++a) dimensions[a]=int((int64_t(last[a])-first[a]+width-1)/width);
        const size_t count=size_t(dimensions.x())*dimensions.y()*dimensions.z();
        if (!count || count>64000000) throw std::runtime_error("classic_local: density majorant grid is too large");
        maxima.resize(count);
        auto accessor=density.grid->getAccessor();
        for (int z=0; z<dimensions.z(); ++z) for (int y=0; y<dimensions.y(); ++y)
            for (int x=0; x<dimensions.x(); ++x) {
                const Eigen::Vector3i cell(x,y,z);
                Eigen::Vector3i lo,hi;
                for (int a=0; a<3; ++a) {
                    // One-cell halo covers cell selection at rounded DDA faces.
                    lo[a]=std::max(density.minimum[a],first[a]+cell[a]*width-1);
                    hi[a]=std::min(density.maximum[a],first[a]+(cell[a]+1)*width+1);
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
    }
};

ClassicLocalGrid::ClassicLocalGrid(const std::string &file, const std::string &sdfName,
        const std::string &densityName, const Mat4f &fieldToWorld, int size) {
    if (size<1 || size>64) throw std::invalid_argument("classic_local: macrocell_size must be in [1,64]");
    Eigen::Matrix4d transform;
    for (int i=0; i<4; ++i) for (int j=0; j<4; ++j) transform(i,j)=fieldToWorld(i,j);
    if (!transform.allFinite() || transform.row(3)!=Eigen::RowVector4d(0,0,0,1) ||
        transform.topLeftCorner<3,3>().determinant()==0)
        throw std::invalid_argument("classic_local: field transform must be finite, affine and invertible");
    const Eigen::Matrix4d inverse=transform.inverse();
    _impl=std::make_unique<Impl>(file,sdfName,densityName,inverse.topLeftCorner<3,3>(),
                               inverse.topRightCorner<3,1>(),size);
}
ClassicLocalGrid::~ClassicLocalGrid()=default;
double ClassicLocalGrid::density(const V &world) const {
    // Match macrofacet's NanoVdbMean active domain: [minNode,maxNode+1].
    // Density interpolation support alone can extend one cell beyond it.
    const V q=_impl->sdf.worldToIndex*world+_impl->sdf.translation;
    if ((q.array()<_impl->sdf.minimum.cast<double>().array()).any() ||
        (q.array()>(_impl->sdf.maximum.cast<double>().array()+1)).any()) return 0;
    return _impl->density.sample(world);
}
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
}

bool ClassicLocalGrid::Cursor::next(Segment &segment) {
    if (!(_begin<_end)) return false;
    int cell[3]; double end=_end;
    for (int a=0; a<3; ++a) {
        const double q=_origin[a]+_begin*_direction[a];
        const double u=std::clamp((q-_grid->first[a])/_grid->width,0.0,double(_grid->dimensions[a]-1));
        int index=int(std::floor(u));
        if (_direction[a]!=0) {
            const auto crossing=[&](int face) {
                const double plane=face==_grid->dimensions[a] ? _grid->last[a] :
                    double(_grid->first[a])+double(face)*_grid->width;
                return (plane-_origin[a])/_direction[a];
            };
            if (_direction[a]>0) {
                while (index+1<_grid->dimensions[a] && crossing(index+1)<=_begin) ++index;
                while (index>0 && crossing(index)>_begin) --index;
                end=std::min(end,crossing(index+1));
            } else {
                while (index>0 && crossing(index)<=_begin) --index;
                while (index+1<_grid->dimensions[a] && crossing(index+1)>_begin) ++index;
                end=std::min(end,crossing(index));
            }
        }
        cell[a]=index;
    }
    if (!(end>_begin)) throw std::runtime_error("classic_local: density DDA failed to advance");
    segment={_begin,end,_grid->maxima[(size_t(cell[2])*_grid->dimensions.y()+cell[1])*_grid->dimensions.x()+cell[0]]};
    _begin=end; return true;
}
#else
struct ClassicLocalGrid::Impl {};
ClassicLocalGrid::ClassicLocalGrid(const std::string &,const std::string &,const std::string &,const Mat4f &,int) {
    throw std::runtime_error("classic_local requires a build with NanoVDB support");
}
ClassicLocalGrid::~ClassicLocalGrid()=default;
double ClassicLocalGrid::density(const Eigen::Vector3d &) const { throw std::runtime_error("NanoVDB unavailable"); }
Eigen::Vector3d ClassicLocalGrid::meanGradient(const Eigen::Vector3d &) const { throw std::runtime_error("NanoVDB unavailable"); }
ClassicLocalGrid::Cursor::Cursor(const ClassicLocalGrid &,const Eigen::Vector3d &,const Eigen::Vector3d &,double,double) {
    throw std::runtime_error("NanoVDB unavailable");
}
bool ClassicLocalGrid::Cursor::next(Segment &) { return false; }
#endif
}
