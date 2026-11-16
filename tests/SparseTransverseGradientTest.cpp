#include "math/SparseConvolutionNoise.hpp"
#include "io/JsonDocument.hpp"
#include "io/Scene.hpp"
#include <array>
#include <iostream>
#include <stdexcept>

using namespace Tungsten;
static void require(bool x, const char* message) { if (!x) throw std::runtime_error(message); }
static void close(double a, double b, double tolerance, const char* message) {
    if (!(std::abs(a-b) <= tolerance)) {
        std::cerr << message << ": " << a << " vs " << b << '\n';
        throw std::runtime_error(message);
    }
}
static std::shared_ptr<GaussianProcess> process(double sigma, double ell) {
    JsonDocument doc(Path("transverse-test.json"), R"({"type":"standard","mean":{"type":"homogeneous","offset":0},
        "covariance":{"type":"squared_exponential","sigma":)"+std::to_string(sigma)+
        R"(,"lengthScale":)"+std::to_string(ell)+"}}");
    Scene scene;
    auto gp=scene.fetchGaussianProcess(doc)->get_gaussian_process(); gp->loadResources();
    return gp;
}
int main() {
    try {
        int cases=0,pdfs=0;
        for (auto param : {std::pair<double,double>{.04,.05},{.02,.2},{.08,.4}}) {
            auto gp=process(param.first,param.second);
            for (Vec3f rd : {Vec3f(0.f,0.f,1.f),Vec3f(1.f,2.f,3.f).normalized()}) {
                TangentFrame frame(rd);
                Vec3f origin(0.f), g0=frame.toGlobal(Vec3f(.3f,-.2f,-.8f));
                SparseConvolutionNoiseRealization noise(gp,GPCorrelationContext::RenewalPlus,
                    17429,10,false,false,true,SparseConv1DSamplingScheme::MIS,true,false,0);
                RayInfo info{}; info.pixelSampleSegment=Vec4u(3u,7u,1u,2u); info.sceneSeed=29;
                UniformSampler uniform;
                Vec2d mu; double var;
                require(noise.seTransverseGradientMoments(origin,rd,true,mu,var),"SE path selected");
                close(mu.lengthSq(),0,0,"unconditioned mean"); close(var,.5,0,"unconditioned variance");
                noise.conditioning(origin,rd,0,g0,info,uniform);
                require(noise.samplingScheme(origin)==SparseConv1DSamplingScheme::UNI,"zero-distance delta transport");
                auto recovered=noise.evaluateGradient(origin,0,rd,info,uniform);
                require((recovered-g0).length()<2e-6,"birth gradient recovery");
                for (double q : {0.,1e-7,.1,.7,1.,3.,12.}) {
                    Vec3f p=rd*float(q*param.second);
                    noise.seTransverseGradientMoments(p,rd,true,mu,var);
                    const double rho=std::exp(-.5*q*q);
                    close(var,.5*(-std::expm1(-q*q)),3e-7,"scaled transverse variance");
                    Vec3f actual=gp->_cov->transformGradLocaltoWorld(frame.toGlobal(Vec3f(float(mu.x()),float(mu.y()),0.f)),1)*float(param.first);
                    Vec3f expected=frame.toGlobal(Vec3f(.3f,-.2f,0.f))*float(rho);
                    require((actual-expected).length()<1e-6,"scaled transverse mean");
                    // Independent check against the complete value + 3-gradient GP block.
                    std::array<Vec3d,4> points{Vec3d(origin),Vec3d(origin),Vec3d(origin),Vec3d(origin)};
                    std::array<Derivative,4> d{Derivative::None,Derivative::First,Derivative::First,Derivative::First};
                    std::array<Vec3d,4> dirs{Vec3d(),Vec3d(frame.tangent),Vec3d(frame.bitangent),Vec3d(rd)};
                    Vec3d query(p), tangent(frame.tangent); Derivative first=Derivative::First;
                    const auto K=gp->cov_sym(points.data(),d.data(),dirs.data(),Vec3d(rd),4);
                    const auto cross=gp->cov(&query,points.data(),&first,d.data(),&tangent,dirs.data(),Vec3d(rd),1,4);
                    const double prior=(*gp->_cov)(first,first,query,query,tangent,tangent);
                    const double denseVar=prior-(cross*K.ldlt().solve(cross.transpose()))(0,0);
                    const double scale=2*param.first*param.first/(param.second*param.second);
                    close(var*scale,denseVar,3e-6,"full GP conditional variance");
                    ++cases;
                }
                // Exercise real sampler and its NEE density, not just moment helpers.
                constexpr int trials=12000; const float t=float(.7*param.second); const Vec3f p=rd*t;
                noise.seTransverseGradientMoments(p,rd,true,mu,var);
                const double v=var*2*param.first*param.first/(param.second*param.second);
                const Vec2d expected(std::exp(-.245)*.3,std::exp(-.245)*-.2);
                Vec2d sum(0.),square(0.);
                for (int i=0;i<trials;++i) {
                    info.pixelSampleSegment=Vec4u(3u,7u,unsigned(i),2u);
                    noise.conditioning(origin,rd,0,g0,info,uniform);
                    const Vec3f g=noise.evaluateGradient(p,t,rd,info,uniform);
                    const Vec3f local=frame.toLocal(g);
                    const Vec2d residual(double(local.x())-expected.x(),double(local.y())-expected.y());
                    sum+=residual; square+=residual*residual;
                    if (i<300 && local.z()<-.02f) {
                        const Vec3f n=g.normalized();
                        const double cosine=std::abs(n.dot(rd));
                        const double density=std::exp(-residual.lengthSq()/(2*v))/(2*M_PI*v);
                        const double expectedPdf=density*g.lengthSq()/(4*cosine*cosine);
                        const double pdf=noise.neePDF(rd,n,p,t,info);
                        close(pdf,expectedPdf,2e-4*std::max(1.,expectedPdf),"NEE PDF vs sampled conditional Gaussian");
                        ++pdfs;
                    }
                }
                for(int axis=0;axis<2;++axis) {
                    close(sum[axis]/trials,0,.035*std::sqrt(v),"empirical conditional mean");
                    close(square[axis]/trials,v,.05*v,"empirical conditional variance");
                }
            }
        }
        if (pdfs<=100) std::cerr << "NEE PDFs checked: " << pdfs << '\n';
        require(pdfs>100,"NEE density coverage");
        std::cout << "PASS: " << cases << " dense-GP/near-origin cases, 72000 samples, " << pdfs << " NEE PDFs\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
