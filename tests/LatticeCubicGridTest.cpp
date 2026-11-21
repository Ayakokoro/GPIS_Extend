#include "media/GpisSdfCubicRay.hpp"
#include "media/GpisSdfGrid.hpp"
#include <iostream>

#if OPENVDB_AVAILABLE
using namespace Tungsten;
using V = Eigen::Vector3d;

static void require(bool ok,const char *message)
{
    if (!ok) throw std::runtime_error(message);
}

static void checkRay(const GpisSdfGrid &grid,const V &q,const V &v)
{
    const V o=grid.worldToIndex.inverse()*(q-grid.translation),w=grid.worldToIndex.inverse()*v;
    GpisSdfCubicRay ray(grid,o,w);
    std::vector<double> faces{0,40};
    grid.appendRayBreakpoints(o,w,0,40,faces);
    std::sort(faces.begin(),faces.end());
    faces.erase(std::unique(faces.begin(),faces.end()),faces.end());
    for (double t=0;t<40;) {
        const double limit=std::min(t+0.73,40.0);
        const double expected=std::min(limit,*std::upper_bound(faces.begin(),faces.end(),t));
        std::array<double,4> c;
        const double end=ray.polynomial(t,limit,c);
        require(end==expected && end>t,"ray must clip at the first forward face or clamp");
        for (int i=0;i<=12;++i) {
            const double u=i/12.0;
            const double actual=c[0]+u*(c[1]+u*(c[2]+u*c[3]));
            const double reference=grid.evaluateValue(o+(t+u*(end-t))*w);
            require(std::abs(actual-reference)<1e-11*(1+std::abs(reference)),
                "ray cubic differs from pointwise trilinear interpolation");
        }
        t=end;
    }
    for (double t:{0.,14.,2.,20.,14.,0.}) {
        for (double bound:{0.,3.,1000.}) {
            const double end=ray.skipPositive(t,40,bound);
            GpisSdfCubicRay fresh(grid,o,w);
            require(end==fresh.skipPositive(t,40,bound),"cached hierarchy differs after a backwards seek");
            require(end>=t && end<=40,"skip outside query interval");
            if (end>t)
                for (int i=0;i<=128;++i)
                    require(grid.evaluateValue(o+(t+(end-t)*i/128)*w)>bound,
                        "hierarchy skipped a candidate surface");
        }
    }
}
#endif

int main()
{
#if OPENVDB_AVAILABLE
    GpisSdfGrid grid;
    grid.data.origin=Vec3i(-13,-4,-1);
    grid.data.dimensions=Vec3i(22,20,18);
    grid.data.clampMin=Vec3f(-12.5f,-3.75f,-0.5f);
    grid.data.clampMax=Vec3f(5.75f,12.75f,15.75f);
    grid.scale=-0.6;
    grid.offset=0.15;
    grid.worldToIndex << 1.25,0.125,0., 0.,1.,0.25, 0.,0.,0.5;
    grid.translation=V(-2,1,4);
    for (int z=0;z<18;++z) for (int y=0;y<20;++y) for (int x=0;x<22;++x) {
        // One negative mean node can hide between the eight outer block corners.
        const double value=x==11 && y==9 && z==8 ? 20 : -8+0.01*x*y-0.002*x*y*z;
        grid.data.values.push_back(float(value));
    }
    const V start(-25,5,7),crossing(1,0,0);
    for (int size:{0,2,8}) {
        grid.buildMacrocells(size,true);
        for (const V &q:{start,V(15,5,7),V(-12.5,-3.75,-0.5),V(-8,4,8),V(5.75,12.75,15.75)}) {
            for (const V &v:{crossing,V(-1,0,0),V(0,1,0),V(1,1,1),V(-0.75,0.25,-0.5),V::Zero().eval()})
                checkRay(grid,q,v);
        }
        const V o=grid.worldToIndex.inverse()*(start-grid.translation),w=grid.worldToIndex.inverse()*crossing;
        GpisSdfCubicRay ray(grid,o,w);
        require((ray.skipPositive(0,40,0)>0)==(size!=0),"hierarchy should skip strictly positive exterior");
        require(ray.skipPositive(23,40,0)==23,"hierarchy missed an interior negative node");
    }
    std::cout<<"SDF ray cubics, clamp/face traversal, random seeks and hierarchy bounds passed\n";
#endif
    return 0;
}
