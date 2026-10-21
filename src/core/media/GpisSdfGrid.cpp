#include "GpisSdfGrid.hpp"
#include "math/conditional/transport/ConditionalMajorantBounds.h"
#include "math/conditional/mathutility/NumericPolicy.h"
#include <array>

namespace Tungsten {
#if OPENVDB_AVAILABLE
namespace cg = ConditionalGpis;

void GpisSdfGrid::buildMacrocells(int cellSize)
{
    if(cellSize!=0 && (cellSize<2 || cellSize>64 || (cellSize&(cellSize-1))))
        throw std::invalid_argument("macrocell size must be 0 or a power of two from 2 to 64");
    macrocellSize=0;
    macrocells.clear();
    macrocellDimensions=Vec3i(0);
    if(cellSize==0) return;
    Vec3i last;
    for(int a=0;a<3;++a) {
        macrocellOrigin[a]=int(std::floor(data.clampMin[a]));
        last[a]=int(std::floor(data.clampMax[a]));
        if(macrocellOrigin[a]<data.origin[a] || last[a]+1>=data.origin[a]+data.dimensions[a])
            throw std::invalid_argument("SDF clamp lacks interpolation node coverage");
        macrocellDimensions[a]=(last[a]-macrocellOrigin[a])/cellSize+1;
    }
    macrocells.resize(size_t(macrocellDimensions.x())*macrocellDimensions.y()*macrocellDimensions.z());
    for(int z=0;z<macrocellDimensions.z();++z)
        for(int y=0;y<macrocellDimensions.y();++y)
            for(int x=0;x<macrocellDimensions.x();++x) {
                Macrocell &block=macrocells[(size_t(z)*macrocellDimensions.y()+y)*macrocellDimensions.x()+x];
                const Vec3i index(x,y,z);
                Vec3i firstNode,lastNode;
                for(int a=0;a<3;++a) {
                    // Include one neighbouring interpolation cell on each face.
                    // This covers gradients selected by rounded boundary queries.
                    firstNode[a]=std::max(macrocellOrigin[a],macrocellOrigin[a]+index[a]*cellSize-1);
                    lastNode[a]=std::min(last[a],macrocellOrigin[a]+(index[a]+1)*cellSize)+1;
                }
                double lower=std::numeric_limits<double>::infinity(),upper=-lower;
                block.minimumGradient.setConstant(lower);
                block.maximumGradient.setConstant(upper);
                block.maximumNodeMagnitude=0;
                for(int k=firstNode.z();k<=lastNode.z();++k)
                    for(int j=firstNode.y();j<=lastNode.y();++j)
                        for(int i=firstNode.x();i<=lastNode.x();++i) {
                            const Vec3i node(i,j,k),local=node-data.origin;
                            const double value=at(local.x(),local.y(),local.z());
                            lower=std::min(lower,value);upper=std::max(upper,value);
                            block.maximumNodeMagnitude=std::max(block.maximumNodeMagnitude,std::abs(value));
                            for(int a=0;a<3;++a) if(node[a]<lastNode[a]) {
                                Vec3i next=local;++next[a];
                                const double edge=at(next.x(),next.y(),next.z())-value;
                                block.minimumGradient[a]=std::min(block.minimumGradient[a],edge);
                                block.maximumGradient[a]=std::max(block.maximumGradient[a],edge);
                            }
                        }
                block.minimumValue=std::min((lower+offset)*scale,(upper+offset)*scale);
                block.maximumValue=std::max((lower+offset)*scale,(upper+offset)*scale);
            }
    macrocellSize=cellSize;
}

namespace {
namespace rb = cg::MajorantBounds;

// One mutable cursor per conditional flight, never stored in the shared grid.
// Each axis retains its previous/next crossing. A seek rebuilds the containing
// cell directly, so delayed refinement does not need a history of visited cells.
class GridRayCursor final : public cg::MeanRayCursor {
    const GpisSdfGrid &_grid;
    Eigen::Vector3d _origin,_direction,_q,_v,_positionScale,_directionScale;
    struct Axis {
        double base=0,mask=0;
        double previous=-std::numeric_limits<double>::infinity();
        double next=std::numeric_limits<double>::infinity();
    };
    std::array<Axis,3> _axes;
    bool _valid=false,_usable=false,_constant=false;
    double _begin=0,_end=0,_anchor=0,_rate=1,_sumScaled=0;
    std::array<double,4> _value{};
    std::array<rb::Interval,4> _certificate;
    std::array<Eigen::Vector3d,3> _gradient;
    struct Endpoint {
        double t=std::numeric_limits<double>::quiet_NaN();
        cg::MeanJet jet;
    };
    std::array<Endpoint,2> _endpoints;
    Statistics _stats;
    static constexpr double roundoff=2048*std::numeric_limits<double>::epsilon();

    void locateAxis(int a,double t) {
        auto &axis=_axes[a];
        axis=Axis{};
        const double low=_grid.data.clampMin[a],high=_grid.data.clampMax[a];
        const double q=_q[a]+t*_v[a];
        if(_v[a]==0) {
            axis.base=std::floor(std::clamp(q,low,high));
            axis.mask=q<low || q>high ? 0 : 1;
            return;
        }
        const auto crossing=[&](double plane) { return (plane-_q[a])/_v[a]; };
        const bool positive=_v[a]>0;
        const double entry=crossing(positive ? low : high);
        const double exit=crossing(positive ? high : low);
        if(t<entry) {
            axis.base=std::floor(positive ? low : high);
            axis.next=entry;
            return;
        }
        if(t>=exit) {
            axis.base=std::floor(positive ? high : low);
            axis.previous=exit;
            return;
        }
        axis.mask=1;
        double base=std::floor(std::clamp(q,low,high));
        // Decide ties by crossing times, not by rounded q+t*v. In particular,
        // a negative ray starting on a face must enter the cell below it.
        if(positive) {
            while(base+1<high && crossing(base+1)<=t) ++base;
            while(base>low && crossing(base)>t) --base;
            axis.previous=crossing(std::max(base,low));
            axis.next=crossing(std::min(base+1,high));
        } else {
            while(base>low && crossing(base)<=t) --base;
            while(base+1<high && crossing(base+1)>t) ++base;
            axis.previous=crossing(std::min(base+1,high));
            axis.next=crossing(std::max(base,low));
        }
        axis.base=base;
    }

    Eigen::Vector3d coordinateError(double t) const {
        return (64*std::numeric_limits<double>::epsilon()*
            (Eigen::Vector3d::Ones()+_positionScale+
             (std::abs(t)+std::abs(_anchor))*_directionScale)).eval();
    }
    double valueError(double t) const {
        return 4*_sumScaled*coordinateError(t).sum()+
            roundoff*(_sumScaled+std::abs(_grid.offset*_grid.scale)+1e-300);
    }
    double derivativeError(double t) const {
        return 4*_sumScaled*coordinateError(t).sum()*_v.cwiseAbs().sum()+
            roundoff*(_sumScaled*_directionScale.sum()+1e-300);
    }
    double curvatureError(double t) const {
        const double speed=_v.cwiseAbs().sum();
        return 8*_sumScaled*(coordinateError(t).sum()+roundoff)*speed*speed;
    }
    bool ambiguous(double t) const {
        if(!_usable) return true;
        const Eigen::Vector3d error=coordinateError(t);
        for(int a=0;a<3;++a) {
            const double q=_q[a]+t*_v[a];
            if(std::abs(q-_grid.data.clampMin[a])<=error[a] ||
               std::abs(q-_grid.data.clampMax[a])<=error[a]) return true;
            if(_axes[a].mask &&
               (q-_axes[a].base<=error[a] || _axes[a].base+1-q<=error[a])) return true;
        }
        return false;
    }

    void prepare(double t) {
        if(!(t>=0) || !std::isfinite(t)) throw std::invalid_argument("invalid grid ray age");
        if(_valid && t>=_begin && t<_end) return;
        for(int a=0;a<3;++a)
            if(!_valid || t<_begin || _axes[a].next<=t) locateAxis(a,t);
        _begin=std::max({_axes[0].previous,_axes[1].previous,_axes[2].previous});
        _end=std::min({_axes[0].next,_axes[1].next,_axes[2].next});
        if(!(_begin<=t && _end>t))
            throw cg::NumericError(cg::NumericStatus::InvalidMajorant,"grid cursor cannot advance");
        _valid=true;
        _anchor=t;
        ++_stats.cellBuilds;
        cg::Profile::count(cg::Profile::Counter::GridCellBuilds);

        Eigen::Vector3d fraction,mask;
        int cell[3];
        for(int a=0;a<3;++a) {
            cell[a]=int(_axes[a].base)-_grid.data.origin[a];
            mask[a]=_axes[a].mask;
            fraction[a]=std::clamp(std::clamp(_q[a]+t*_v[a],
                double(_grid.data.clampMin[a]),double(_grid.data.clampMax[a]))-_axes[a].base,0.0,1.0);
        }
        const Eigen::Vector3d active=_v.cwiseProduct(mask);
        _rate=active.cwiseAbs().maxCoeff();
        _constant=_rate==0;
        if(_constant) _rate=1;
        const Eigen::Vector3d step=active/_rate;
        _value.fill(0);
        std::array<double,4> magnitude{};
        for(auto &g:_gradient) g.setZero();
        double sum=0;
        for(int z=0;z<2;++z) for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
            const double c=_grid.at(cell[0]+x,cell[1]+y,cell[2]+z);
            sum+=std::abs(c);
            const double sx=x ? 1:-1,sy=y ? 1:-1,sz=z ? 1:-1;
            const double ax=x ? fraction.x():1-fraction.x(),dx=sx*step.x();
            const double ay=y ? fraction.y():1-fraction.y(),dy=sy*step.y();
            const double az=z ? fraction.z():1-fraction.z(),dz=sz*step.z();
            const double yz0=ay*az,yz1=ay*dz+dy*az,yz2=dy*dz;
            _value[0]+=c*ax*yz0;
            _value[1]+=c*(dx*yz0+ax*yz1);
            _value[2]+=c*(dx*yz1+ax*yz2);
            _value[3]+=c*dx*yz2;
            const double by1=ay*std::abs(dz)+std::abs(dy)*az;
            magnitude[0]+=std::abs(c)*ax*yz0;
            magnitude[1]+=std::abs(c)*(std::abs(dx)*yz0+ax*by1);
            magnitude[2]+=std::abs(c)*(std::abs(dx)*by1+ax*std::abs(yz2));
            magnitude[3]+=std::abs(c*dx*yz2);
            _gradient[0]+=c*Eigen::Vector3d(sx*yz0,sy*ax*az,sz*ax*ay);
            _gradient[1]+=c*Eigen::Vector3d(sx*yz1,sy*(ax*dz+dx*az),sz*(ax*dy+dx*ay));
            _gradient[2]+=c*Eigen::Vector3d(sx*yz2,sy*dx*dz,sz*dx*dy);
        }
        _sumScaled=std::abs(_grid.scale)*sum;
        _value[0]+=_grid.offset;
        magnitude[0]+=std::abs(_grid.offset);
        _usable=std::isfinite(_rate) && coordinateError(t).maxCoeff()<1e-5;
        for(int k=0;k<4;++k) {
            _value[k]*=_grid.scale;
            // Normalized fractions and steps have magnitude <=1. The absolute
            // expansion sums bound cancellation in all eight basis products;
            // this allowance exceeds gamma_n for their double operations.
            const double error=roundoff*std::abs(_grid.scale)*magnitude[k];
            _certificate[k]=magnitude[k]==0 ? rb::Interval(0) :
                rb::Interval(rb::down(_value[k]-error),rb::up(_value[k]+error));
            _usable&=std::isfinite(_certificate[k].lower) && std::isfinite(_certificate[k].upper);
        }
        for(auto &g:_gradient)
            g=(_grid.scale*_grid.worldToIndex.transpose()*g.cwiseProduct(mask)).eval();
    }

    cg::MeanJet reference(double t) {
        for(const auto &e:_endpoints) if(e.t==t) return e.jet;
        ++_stats.pointFallbacks;
        cg::Profile::count(cg::Profile::Counter::GridPointFallbacks);
        const auto jet=_grid.evaluate(_origin+t*_direction);
        _endpoints[1]=_endpoints[0];
        _endpoints[0]={t,jet};
        return jet;
    }
    cg::MeanRaySample inCell(double t) {
        if(ambiguous(t)) {
            const auto jet=reference(t);
            return {jet.value,jet.gradient.dot(_direction)};
        }
        const double s=(t-_anchor)*_rate;
        return {_value[0]+s*(_value[1]+s*(_value[2]+s*_value[3])),
            _rate*(_value[1]+s*(2*_value[2]+s*3*_value[3]))};
    }
    cg::MeanRayBounds fallback(double begin,double end,const SdfRayQueryOptions &options) {
        ++_stats.boundFallbacks;
        cg::Profile::count(cg::Profile::Counter::GridBoundFallbacks);
        return _grid.rayBounds(_origin,_direction,begin,end,options);
    }
public:
    GridRayCursor(const GpisSdfGrid &grid,const Eigen::Vector3d &origin,const Eigen::Vector3d &direction)
        : _grid(grid),_origin(origin),_direction(direction),
          _q(grid.worldToIndex*origin+grid.translation),_v(grid.worldToIndex*direction),
          _positionScale(grid.worldToIndex.cwiseAbs()*origin.cwiseAbs()+
                         grid.translation.cwiseAbs()+_q.cwiseAbs()),
          _directionScale(grid.worldToIndex.cwiseAbs()*direction.cwiseAbs()+_v.cwiseAbs()) {
        if(!_q.allFinite() || !_v.allFinite())
            throw std::invalid_argument("non-finite grid ray transform");
    }
    Statistics statistics() const override { return _stats; }
    double nextBoundary(double begin,double end) override {
        prepare(begin);
        return std::min(end,_end);
    }
    bool coarseRayBounds(double begin,double &end,cg::MeanRayBounds &result) override {
        if(!_grid.macrocellSize || !(begin>=0 && end>begin) || !std::isfinite(end) ||
           coordinateError(end).maxCoeff()>=1e-5) return false;
        const int width=_grid.macrocellSize;
        int index[3];
        double clipped=end;
        for(int a=0;a<3;++a) {
            const double low=_grid.data.clampMin[a],high=_grid.data.clampMax[a];
            const double first=_grid.macrocellOrigin[a];
            const int count=_grid.macrocellDimensions[a];
            const double q=_q[a]+begin*_v[a];
            int cell=std::clamp(int(std::floor((std::clamp(q,low,high)-first)/width)),0,count-1);
            if(_v[a]!=0) {
                const auto crossing=[&](double plane) { return (plane-_q[a])/_v[a]; };
                const bool positive=_v[a]>0;
                const double entry=crossing(positive ? low : high),exit=crossing(positive ? high : low);
                if(begin<entry) clipped=std::min(clipped,entry);
                else if(begin<exit) {
                    if(positive) {
                        while(cell+1<count && crossing(first+(cell+1)*width)<=begin) ++cell;
                        while(cell>0 && crossing(first+cell*width)>begin) --cell;
                        clipped=std::min(clipped,std::min(exit,crossing(first+(cell+1)*width)));
                    } else {
                        while(cell>0 && crossing(first+cell*width)<=begin) --cell;
                        while(cell+1<count && crossing(first+(cell+1)*width)>begin) ++cell;
                        clipped=std::min(clipped,std::min(exit,crossing(first+cell*width)));
                    }
                }
            }
            index[a]=cell;
        }
        const double middle=0.5*begin+0.5*clipped;
        if(!(middle>begin && middle<clipped)) return false;
        const auto &block=_grid.macrocell(index[0],index[1],index[2]);
        const auto error=coordinateError(clipped);
        double derivativeMin=0,derivativeMax=0;
        for(int a=0;a<3;++a) {
            const double q0=_q[a]+begin*_v[a],q1=_q[a]+clipped*_v[a];
            const double low=std::min(q0,q1)-error[a],high=std::max(q0,q1)+error[a];
            double lower=block.minimumGradient[a],upper=block.maximumGradient[a];
            if(high<_grid.data.clampMin[a] || low>_grid.data.clampMax[a]) lower=upper=0;
            else if(low<=_grid.data.clampMin[a] || high>=_grid.data.clampMax[a]) {
                lower=std::min(0.0,lower);upper=std::max(0.0,upper);
            }
            const double coefficient=_grid.scale*_v[a];
            derivativeMin+=std::min(coefficient*lower,coefficient*upper);
            derivativeMax+=std::max(coefficient*lower,coefficient*upper);
        }
        // Node extrema bound the entire trilinear interpolant; edge extrema
        // bound each index derivative. The halo absorbs coordinate/cell-choice
        // uncertainty, while these allowances cover interpolation and transforms.
        const double valueError=roundoff*std::abs(_grid.scale)*
            (block.maximumNodeMagnitude+std::abs(_grid.offset)+1e-300);
        const double slopeError=roundoff*std::abs(_grid.scale)*block.maximumNodeMagnitude*
            (_directionScale.sum()+1e-300);
        result=cg::MeanRayBounds{};
        result.minimumValue=rb::down(block.minimumValue-valueError);
        result.maximumValue=rb::up(block.maximumValue+valueError);
        result.minimumDerivative=rb::down(derivativeMin-slopeError);
        result.maximumDerivative=rb::up(derivativeMax+slopeError);
        const Eigen::Vector3d point=_origin+middle*_direction;
        result.midpointValue=_grid.evaluate(Vec3d(point.x(),point.y(),point.z()),nullptr);
        // The midpoint is evaluated through the world transform. Its rounded
        // index coordinates can differ from the exact ray even inside the halo.
        const auto gradientMagnitude=block.minimumGradient.cwiseAbs().cwiseMax(block.maximumGradient.cwiseAbs());
        result.midpointError=rb::up(valueError+std::abs(_grid.scale)*gradientMagnitude.dot(error));
        result.certified=true;
        end=clipped;
        ++_stats.coarseQueries;
        cg::Profile::count(cg::Profile::Counter::GridMacroQueries);
        return true;
    }
    cg::MeanRaySample sample(double t) override {
        prepare(t);
        return inCell(t);
    }
    cg::MeanJet evaluate(double t) override {
        prepare(t);
        if(ambiguous(t)) return reference(t);
        const double s=(t-_anchor)*_rate;
        return {_value[0]+s*(_value[1]+s*(_value[2]+s*_value[3])),
                (_gradient[0]+s*(_gradient[1]+s*_gradient[2])).eval()};
    }
    double valueDifference(double t,double *error=nullptr) override {
        if(error) *error=0;
        prepare(t);
        if(_usable && _begin<=0 && t<_end) {
            if(_constant) return 0;
            const double s0=-_anchor*_rate,delta=t*_rate;
            if(std::abs(s0)<=2 && std::abs(delta)<=2) {
                if(error) {
                    double coefficientError=0;
                    const double weights[]={0,1,std::abs(2*s0+delta),
                        std::abs(3*s0*s0+3*s0*delta+delta*delta)};
                    for(int k=1;k<4;++k)
                        coefficientError+=weights[k]*std::max(_value[k]-_certificate[k].lower,
                                                              _certificate[k].upper-_value[k]);
                    *error=rb::up(std::abs(delta)*coefficientError+std::abs(t)*derivativeError(t));
                }
                return delta*(_value[1]+_value[2]*(2*s0+delta)+
                    _value[3]*(3*s0*s0+3*s0*delta+delta*delta));
            }
        }
        return _grid.valueDifference(_origin,t*_direction);
    }
    cg::MeanRayBounds rayBounds(double begin,double end,const SdfRayQueryOptions &options) override {
        if(!(begin>=0 && end>begin) || !std::isfinite(end))
            throw std::invalid_argument("invalid grid ray interval");
        const double middle=0.5*begin+0.5*end;
        if(!(middle>begin && middle<end)) return fallback(begin,end,options);
        cg::MeanRayBounds result;
        result.minimumValue=result.minimumDerivative=std::numeric_limits<double>::infinity();
        result.maximumValue=result.maximumDerivative=-result.minimumValue;
        result.maximumSecondDerivative=0;
        result.certified=true;
        int cells=0;
        for(double lo=begin;lo<end;) {
            prepare(lo);
            if(!_usable) return fallback(begin,end,options);
            const double hi=std::min(end,_end);
            if(!(hi>lo)) return fallback(begin,end,options);
            ++cells;
            rb::Interval values,derivatives,curvature;
            double polynomialValueError=0,polynomialDerivativeError=0,polynomialCurvatureError=0;
            if(_constant) {
                values=_certificate[0];
                derivatives=curvature=rb::Interval(0);
            } else {
                const double sl=(lo-_anchor)*_rate,sr=(hi-_anchor)*_rate;
                constexpr double parameterError=8*std::numeric_limits<double>::epsilon();
                const double left=rb::down(sl-parameterError*std::abs(sl));
                const double right=rb::up(sr+parameterError*std::abs(sr));
                const double width=right-left,s=std::max(std::abs(left),std::abs(right));
                const auto value=[&](double u) {
                    return _value[0]+u*(_value[1]+u*(_value[2]+u*_value[3]));
                };
                const auto slope=[&](double u) {
                    return _value[1]+u*(2*_value[2]+u*3*_value[3]);
                };
                double e[4],m[4];
                for(int k=0;k<4;++k) {
                    e[k]=std::max(_value[k]-_certificate[k].lower,_certificate[k].upper-_value[k]);
                    m[k]=std::abs(_value[k]);
                }
                // Bound coefficient uncertainty pointwise. The extra absolute
                // Horner norms cover the arithmetic of the Bernstein controls
                // as well (width <= 2*s). Only the final limits need outward
                // rounding, avoiding per-operation nextafter in this hot path.
                polynomialValueError=e[0]+s*(e[1]+s*(e[2]+s*e[3]))+
                    roundoff*(m[0]+s*(m[1]+s*(m[2]+s*m[3])));
                polynomialDerivativeError=e[1]+s*(2*e[2]+s*3*e[3])+
                    roundoff*(m[1]+s*(2*m[2]+s*3*m[3]));
                polynomialCurvatureError=2*e[2]+6*s*e[3]+roundoff*(2*m[2]+6*s*m[3]);
                const double first=value(left),last=value(right);
                const double firstSlope=slope(left),lastSlope=slope(right);
                const double firstCurvature=2*_value[2]+6*left*_value[3];
                const double lastCurvature=2*_value[2]+6*right*_value[3];
                const double middleSlope=firstSlope+0.5*width*firstCurvature;
                const double minimumSlope=std::min({firstSlope,middleSlope,lastSlope});
                const double maximumSlope=std::max({firstSlope,middleSlope,lastSlope});
                values={std::min(first,last),std::max(first,last)};
                if(minimumSlope<polynomialDerivativeError && maximumSlope>-polynomialDerivativeError) {
                    const double control1=first+(width/3)*firstSlope;
                    const double control2=last-(width/3)*lastSlope;
                    values=rb::hull(values,{std::min(control1,control2),std::max(control1,control2)});
                }
                derivatives={_rate*minimumSlope,_rate*maximumSlope};
                curvature={_rate*_rate*std::min(firstCurvature,lastCurvature),
                           _rate*_rate*std::max(firstCurvature,lastCurvature)};
                polynomialDerivativeError*=_rate;
                polynomialCurvatureError*=_rate*_rate;
            }
            if(cells==1 && options.computeBeginDerivative) {
                const double s=(begin-_anchor)*_rate;
                result.beginDerivative=_rate*(_value[1]+s*(2*_value[2]+s*3*_value[3]));
            }
            // Endpoints may select a neighbouring cell after world-coordinate
            // rounding. Preserve the reference point query in that narrow zone.
            for(double t:{lo,hi}) {
                const auto p=inCell(t);
                values=rb::hull(values,rb::Interval(p.value));
                derivatives=rb::hull(derivatives,rb::Interval(p.derivative));
            }
            const double valuePad=valueError(hi)+polynomialValueError;
            const double derivativePad=derivativeError(hi)+polynomialDerivativeError;
            result.minimumValue=std::min(result.minimumValue,rb::down(values.lower-valuePad));
            result.maximumValue=std::max(result.maximumValue,rb::up(values.upper+valuePad));
            result.minimumDerivative=std::min(result.minimumDerivative,rb::down(derivatives.lower-derivativePad));
            result.maximumDerivative=std::max(result.maximumDerivative,rb::up(derivatives.upper+derivativePad));
            result.maximumSecondDerivative=std::max(result.maximumSecondDerivative,
                rb::up(std::max(std::abs(curvature.lower),std::abs(curvature.upper))+
                    curvatureError(hi)+polynomialCurvatureError));
            if(!options.midpoint && middle>=lo && middle<=hi) {
                result.midpointValue=inCell(middle).value;
                result.midpointError=ambiguous(middle) ? 0 : valueError(middle);
            }
            lo=hi;
        }
        if(cells>1) result.maximumSecondDerivative=std::numeric_limits<double>::infinity();
        if(options.midpoint) {
            result.midpointValue=_grid.evaluate(*options.midpoint).value;
            result.midpointError=0;
        }
        return result;
    }
};
}

std::unique_ptr<cg::MeanRayCursor> GpisSdfGrid::createRayCursor(
    const Eigen::Vector3d &origin,const Eigen::Vector3d &direction) const
{
    return std::make_unique<GridRayCursor>(*this,origin,direction);
}

cg::MeanJet GpisSdfGrid::evaluate(const Eigen::Vector3d &p) const
{
    Vec3d g;
    double f = evaluate(Vec3d(p.x(), p.y(), p.z()), &g);
    return {f, Eigen::Vector3d(g.x(), g.y(), g.z())};
}

cg::BoundsSummary GpisSdfGrid::bounds(const cg::Bounds3 &) const
{
    double minimum = std::numeric_limits<double>::infinity();
    for (float v : data.values) minimum = std::min(minimum, (v + offset)*scale);
    return {minimum, gradientBound, true};
}

void GpisSdfGrid::appendRayBreakpoints(const Eigen::Vector3d &o, const Eigen::Vector3d &w,
                                     double begin, double end, std::vector<double> &knots) const
{
    const Eigen::Vector3d q = worldToIndex*o + translation, v = worldToIndex*w;
    for (int a = 0; a < 3; ++a) {
        if (v[a] == 0.0) continue;
        const double lo = std::max(double(data.clampMin[a]), std::min(q[a]+begin*v[a], q[a]+end*v[a]));
        const double hi = std::min(double(data.clampMax[a]), std::max(q[a]+begin*v[a], q[a]+end*v[a]));
        auto add = [&](double x) {
            const double t = (x-q[a])/v[a];
            if (t > begin && t < end) knots.push_back(t);
        };
        for (double x = std::ceil(lo); x <= hi; x += 1.0) add(x);
        add(data.clampMin[a]); add(data.clampMax[a]);
    }
}

double GpisSdfGrid::valueDifference(const Eigen::Vector3d &p, const Eigen::Vector3d &d) const
{
    Eigen::Vector3d q = worldToIndex*p + translation, dq = worldToIndex*d;
    Eigen::Vector3d f, df;
    int cell[3];
    for (int a = 0; a < 3; ++a) {
        const double x = std::clamp(q[a], double(data.clampMin[a]), double(data.clampMax[a]));
        const double y = std::clamp(q[a]+dq[a], double(data.clampMin[a]), double(data.clampMax[a]));
        if (std::floor(x) != std::floor(y)) return evaluate(p+d).value-evaluate(p).value;
        cell[a] = int(std::floor(x))-data.origin[a];
        f[a] = x-std::floor(x);
        df[a] = (q[a] == x && q[a]+dq[a] == y) ? dq[a] : y-x;
    }
    double result = 0.0;
    for (int z = 0; z < 2; ++z) for (int y = 0; y < 2; ++y) for (int x = 0; x < 2; ++x) {
        const double wx = x ? f.x() : 1-f.x(), dx = x ? df.x() : -df.x();
        const double wy = y ? f.y() : 1-f.y(), dy = y ? df.y() : -df.y();
        const double wz = z ? f.z() : 1-f.z(), dz = z ? df.z() : -df.z();
        result += at(cell[0]+x,cell[1]+y,cell[2]+z)*
            (dx*wy*wz+(wx+dx)*dy*wz+(wx+dx)*(wy+dy)*dz);
    }
    return scale*result;
}

cg::MeanRayBounds GpisSdfGrid::rayBounds(const Eigen::Vector3d &o, const Eigen::Vector3d &w,
                                       double begin, double end,const SdfRayQueryOptions &options) const
{
    std::vector<double> knots{begin, end};
    appendRayBreakpoints(o,w,begin,end,knots);
    std::sort(knots.begin(),knots.end());
    knots.erase(std::unique(knots.begin(),knots.end()),knots.end());
    cg::MeanRayBounds result;
    result.minimumValue = result.minimumDerivative = std::numeric_limits<double>::infinity();
    result.maximumValue = result.maximumDerivative = -result.minimumValue;
    result.maximumSecondDerivative = knots.size() == 2 ? 0.0 : std::numeric_limits<double>::infinity();
    result.certified = true;
    const Eigen::Vector3d q = worldToIndex*o + translation, v = worldToIndex*w;
    // Match evaluate()'s world-point transform, rather than reconstructing the
    // index-space midpoint as q+t*v: they can select different cells at a face.
    const Eigen::Vector3d midpoint = options.midpoint.value_or(o+(0.5*begin+0.5*end)*w);
    const Eigen::Vector3d midpointIndex = worldToIndex*midpoint + translation;
    Eigen::Vector3d midpointFraction;
    int midpointCell[3];
    for (int a = 0; a < 3; ++a) {
        const double x = std::clamp(midpointIndex[a],double(data.clampMin[a]),double(data.clampMax[a]));
        const double base = std::floor(x);
        midpointCell[a] = int(base)-data.origin[a];
        midpointFraction[a] = x-base;
    }
    for (size_t i = 1; i < knots.size(); ++i) {
        Eigen::Vector3d lo, hi, dir = v, fbegin;
        int cell[3];
        for (int a = 0; a < 3; ++a) {
            const double mid = q[a]+(0.5*knots[i-1]+0.5*knots[i])*v[a];
            const double base = std::floor(std::clamp(mid,double(data.clampMin[a]),double(data.clampMax[a])));
            cell[a] = int(base)-data.origin[a];
            if (mid < data.clampMin[a] || mid > data.clampMax[a]) dir[a] = 0.0;
            const double x = std::clamp(q[a]+knots[i-1]*v[a],double(data.clampMin[a]),double(data.clampMax[a]))-base;
            const double y = std::clamp(q[a]+knots[i]*v[a],double(data.clampMin[a]),double(data.clampMax[a]))-base;
            lo[a] = std::clamp(std::min(x,y),0.0,1.0);
            hi[a] = std::clamp(std::max(x,y),0.0,1.0);
            fbegin[a] = std::clamp(x,0.0,1.0);
        }
        // All box corners, the directional curvature and the cached midpoint
        // use this same trilinear cell. Fetch its eight samples only once.
        double corners[2][2][2];
        for (int z=0;z<2;++z) for (int y=0;y<2;++y) for (int x=0;x<2;++x)
            corners[x][y][z] = at(cell[0]+x,cell[1]+y,cell[2]+z);
        auto jet = [&](const Eigen::Vector3d &f) {
            double val = 0, slope = 0;
            for (int z=0;z<2;++z) for (int y=0;y<2;++y) for (int x=0;x<2;++x) {
                double c = corners[x][y][z];
                double a = x ? f.x() : 1-f.x(), b = y ? f.y() : 1-f.y(), d = z ? f.z() : 1-f.z();
                val += c*a*b*d;
                slope += c*((x ? 1:-1)*dir.x()*b*d+a*(y ? 1:-1)*dir.y()*d+a*b*(z ? 1:-1)*dir.z());
            }
            return std::make_pair((val+offset)*scale,slope*scale);
        };
        if (i==1 && options.computeBeginDerivative) result.beginDerivative = jet(fbegin).second;
        for (int mask=0;mask<8;++mask) {
            Eigen::Vector3d f;
            for (int a=0;a<3;++a) f[a]=(mask&(1<<a)) ? hi[a]:lo[a];
            const auto j=jet(f);
            result.minimumValue=std::min(result.minimumValue,j.first);
            result.maximumValue=std::max(result.maximumValue,j.first);
            result.minimumDerivative=std::min(result.minimumDerivative,j.second);
            result.maximumDerivative=std::max(result.maximumDerivative,j.second);
        }
        if (cell[0]==midpointCell[0] && cell[1]==midpointCell[1] && cell[2]==midpointCell[2])
            result.midpointValue=jet(midpointFraction).first;
        // Distinct axis crossings can round to adjacent doubles. Such an
        // interval has no representable interior, and floor/clamp at either
        // endpoint may select its neighbouring cell. Enclose those actual
        // query derivatives too; retain the forward derivative for the Taylor
        // envelope at birth.
        for (double t : {knots[i-1], knots[i]}) {
            const auto endpoint=evaluate(o+t*w);
            result.minimumValue=std::min(result.minimumValue,endpoint.value);
            result.maximumValue=std::max(result.maximumValue,endpoint.value);
            result.minimumDerivative=std::min(result.minimumDerivative,endpoint.gradient.dot(w));
            result.maximumDerivative=std::max(result.maximumDerivative,endpoint.gradient.dot(w));
        }
        double second=0;
        for (int a=0;a<3;++a) for (int b=a+1;b<3;++b) {
            const int other=3-a-b;
            double mixed=0;
            for (int k=0;k<2;++k) {
                int c[3]={0,0,0}; c[other]=k;
                double diff=0;
                for (int x=0;x<2;++x) for (int y=0;y<2;++y) {
                    c[a]=x; c[b]=y;
                    diff+=(x==y ? 1:-1)*corners[c[0]][c[1]][c[2]];
                }
                mixed=std::max(mixed,std::abs(diff));
            }
            second+=2*std::abs(scale*dir[a]*dir[b])*mixed;
        }
        result.maximumSecondDerivative=std::max(result.maximumSecondDerivative,second*(1+1e-11));
    }
    // A supplied midpoint or a rounded cell-face query may lie outside the
    // visited cells. Keep the ordinary point query in that case. Both paths
    // use evaluate()'s interpolation order, so midpointError remains zero.
    if (!std::isfinite(result.midpointValue)) result.midpointValue=evaluate(midpoint).value;
    const double error=512*std::numeric_limits<double>::epsilon()*
        (1+q.cwiseAbs().maxCoeff()+v.norm()*end)*
        (std::max(std::abs(result.minimumValue),std::abs(result.maximumValue))+gradientBound+1e-30);
    result.minimumValue-=error; result.maximumValue+=error;
    result.minimumDerivative-=error*(1+w.norm()); result.maximumDerivative+=error*(1+w.norm());
    return result;
}
#endif
}
