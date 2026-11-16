#pragma once

namespace {
void gaussLegendre5Checks()
{
    sh::ProbabilitySettings fixed;
    fixed.gaussLegendre5=true;
    // An unattainable numerical tolerance must not select another CDF method.
    fixed.absoluteTolerance=1e-15;fixed.relativeTolerance=1e-14;
    fixed.initialSamples=fixed.maxSamples=16;
    Eigen::Matrix2d covariance;covariance<<1,.8001,.8001,1;
    const Eigen::Vector2d mean(-.2,0);
    const double approximate=sh::positiveOrthant(mean,covariance,fixed);
    close(approximate,.35323253140519545,2e-13,"fixed five-point high-correlation fixture");
    const double reference=sh::positiveOrthant(mean,covariance);
    close(reference,.3538015282322997,2e-12,"independent bivariate reference fixture");
    require(std::abs(approximate-reference)>5e-4,"fixed CDF does not silently fall back to the reference");
    const Eigen::Vector2d scale(.02,.7);
    close(sh::positiveOrthant(mean.cwiseProduct(scale),scale.asDiagonal()*covariance*scale.asDiagonal(),fixed),
          approximate,2e-14,"CDF standardizes conditional variances before fixed quadrature");
    for(int n=1;n<=4;++n) {
        Eigen::VectorXd means(n);double expected=1;
        for(int i=0;i<n;++i) {means[i]=.3*i-.6;expected*=cg::normalCdf(means[i]);}
        close(sh::positiveOrthant(means,Eigen::MatrixXd::Identity(n,n),fixed),expected,2e-14,
              "fixed recursive CDF respects independent probability product");
    }
    for(double rho:{-1.,-.999999,-.95,-.3,0.,.5,.8,.8001,.95,.999999,1.})
        for(double a:{-8.,-.2,0.,.7,8.}) for(double b:{-3.,0.,.4,8.}) {
            covariance<<1,rho,rho,1;
            const double value=sh::positiveOrthant(Eigen::Vector2d(a,b),covariance,fixed);
            require(std::isfinite(value) && value>=0 &&
                    value<=std::min(cg::normalCdf(a),cg::normalCdf(b))+2e-15,
                    "fixed CDF remains within probability bounds at tails and singular limits");
            close(value,sh::positiveOrthant(Eigen::Vector2d(b,a),covariance,fixed),2e-14,
                  "fixed CDF is invariant to variable permutation");
        }
    const auto field=makeField(std::make_shared<cg::PlaneMean>(V(0.2,0.1,1),.3),.7,.8);
    sh::Start surface;surface.kind=sh::Start::SurfaceValueGradient;surface.gradient=V(.2,-.1,.7);
    for(double t:{.03,.2,.8,1.7}) {
        const double hazard=sh::LocalLaw(field,surface,t,{.5*t},fixed).hazard();
        require(std::isfinite(hazard) && hazard>=0,"fixed CDF supports value-gradient conditioned local laws");
    }
    std::cout<<"Five-point CDF checks passed; fixture absolute error="<<std::abs(approximate-reference)<<"\n";
}
}
