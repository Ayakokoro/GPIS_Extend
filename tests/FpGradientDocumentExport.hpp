#pragma once

#include "math/FirstPassageGradient.hpp"
#include "sampling/IndependentPathSampler.hpp"
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <vector>

// Export the production law, including both RNG adapters, for an independent
// Python validation against section 8 of GPIS_Gradient_Distribution(1).pdf.
// This diagnostic does not change the renderer or implement the reference GP.
namespace FpGradientDocumentExport {
namespace fp = Tungsten::FirstPassageGradient;
namespace cg = Tungsten::ConditionalGpis;

struct Case {
    std::string id, kind;
    double t, f, g, sigma, gamma;
    std::vector<double> parameters;
    std::function<double(double)> mean, derivative;
};

inline std::vector<Case> cases() {
    const double rows[][6] = {
        {.6,.25,-.4,.3,1,1}, {.8,.30,-.5,.2,1,1},
        {1.2,.20,-1,.5,1,1}, {1.5,.40,-.6,.5,1,1},
        {1.8,.60,.8,-.2,1.2,.9}, {2,.50,-.5,.5,1,1},
        {2,.20,-.8,-.3,1,1}, {2.5,.60,.2,.5,1,1},
        {3,.50,0,-.4,1,1}, {3.5,.30,.5,-.5,1,1},
        {4,1,-.5,-.3,1,1}, {5,.50,.5,0,1,1}
    };
    std::vector<Case> result;
    for(int i=0;i<12;++i) {
        const auto &r=rows[i]; const double slope=r[3];
        result.push_back({"L"+std::string(i<9 ? "0" : "")+std::to_string(i+1),"linear",
            r[0],r[1],r[2],r[4],r[5],{slope},
            [slope](double s){ return slope*s; },[slope](double){ return slope; }});
    }
    auto sphere=[&](const char *id,double t,double f,double g,double base,double sign,double center,double radius) {
        result.push_back({id,"sqrt",t,f,g,1,1,{base,sign,center,radius},
            [=](double s){ return base+sign*std::hypot(s-center,radius); },
            [=](double s){ return sign*(s-center)/std::hypot(s-center,radius); }});
    };
    sphere("N01",2,.25,-.7,-1,1,.8,1.02);
    sphere("N02",2.2,.30,-.5,-1,1,.9,.92);
    sphere("N03",2.5,.50,.2,2,-1,1,1.2);
    result.push_back({"N04","sin",2.2,.30,-.4,1,1,{.3,-.15,-.25,1.8},
        [](double s){ return .3-.15*s-.25*std::sin(1.8*s); },
        [](double s){ return -.15-.45*std::cos(1.8*s); }});
    result.push_back({"N05","quadratic",3,.40,.3,1,1,{.6,-.1,-.12},
        [](double s){ return .6-.1*s-.12*s*s; },
        [](double s){ return -.1-.24*s; }});
    result.push_back({"N06","cos",3.2,.35,-.7,1,1,{.25,-.08,.1,2.5},
        [](double s){ return .25-.08*s+.1*std::cos(2.5*s); },
        [](double s){ return -.08-.25*std::sin(2.5*s); }});
    return result;
}

inline void run(const std::string &directory,int count) {
    if(count<100) throw std::invalid_argument("document export needs at least 100 samples");
    const std::filesystem::path root(directory);
    std::filesystem::create_directories(root);
    std::ofstream out(root/"production.json");
    if(!out) throw std::runtime_error("cannot create document export");
    out<<std::setprecision(17)<<"{\"samples_per_rng\":"<<count<<",\"cases\":[";
    bool first=true;
    for(const auto &c:cases()) {
        const fp::Input input{c.sigma,c.gamma,c.t,c.f,c.g,c.mean(0),c.derivative(0),c.derivative(c.t),c.mean};
        const fp::Law law(input);
        if(!first) out<<','; first=false;
        out<<"{\"id\":\""<<c.id<<"\",\"mean_kind\":\""<<c.kind<<"\",\"parameters\":[";
        for(std::size_t i=0;i<c.parameters.size();++i) out<<(i ? "," : "")<<c.parameters[i];
        out<<"],\"t\":"<<c.t<<",\"f0\":"<<c.f<<",\"g0\":"<<c.g
           <<",\"sigma\":"<<c.sigma<<",\"gamma\":"<<c.gamma
           <<",\"gaussian_mean\":"<<law.gaussianMean()<<",\"gaussian_sd\":"<<law.gaussianStddev()
           <<",\"p\":"<<law.probitIntercept()<<",\"q\":"<<law.probitSlope()
           <<",\"approximation\":\"single_anchor_fixed5_snr\",\"anchor_selection\":\"fixed5_snr\",\"anchor_distance\":"<<law.anchorDistance()
           <<",\"anchor_p\":["<<law.probitIntercept()
           <<"],\"anchor_q\":["<<law.probitSlope()<<"],\"grid\":[";
        const double lo=std::min(0.,law.gaussianMean())-12*law.gaussianStddev();
        for(int i=0;i<=256;++i) {
            const double v=lo*(1-double(i)/256);
            out<<(i ? "," : "")<<'['<<v<<','<<law.pdf(v)<<','<<law.cdf(v)<<']';
        }
        out<<"],\"samples\":[";
        for(int adapter=0;adapter<2;++adapter) {
            const auto seed=std::uint64_t(20261108+97*int(c.id[0])+std::stoi(c.id.substr(1)));
            cg::Random engine(seed);
            Tungsten::IndependentPathSampler path(seed,false); path.startPath(0,0);
            cg::Random rendering(path);
            cg::Random &rng=adapter ? rendering : engine;
            const std::string name=c.id+(adapter ? ".renderer.f64" : ".mt19937.f64");
            std::ofstream data(root/name,std::ios::binary);
            if(!data) throw std::runtime_error("cannot create sample export");
            int fallbacks=0; double sum=0,sum2=0;
            for(int i=0;i<count;++i) {
                bool used=false; const double v=law.sample(rng,&used);
                if(!(v<0) || !std::isfinite(v)) throw std::runtime_error("invalid exported gradient");
                data.write(reinterpret_cast<const char *>(&v),sizeof(v));
                sum+=v; sum2+=v*v; fallbacks+=used;
            }
            if(!data) throw std::runtime_error("sample export write failed");
            out<<(adapter ? "," : "")<<"{\"rng\":\""<<(adapter ? "renderer" : "mt19937_64")
               <<"\",\"file\":\""<<name<<"\",\"seed\":"<<seed<<",\"count\":"<<count
               <<",\"mean\":"<<sum/count<<",\"second_moment\":"<<sum2/count
               <<",\"fallbacks\":"<<fallbacks<<'}';
        }
        out<<"]}"; out.flush();
        std::cout<<"exported "<<c.id<<", "<<count<<" samples per RNG\n";
    }
    out<<"]}\n";
    if(!out) throw std::runtime_error("document export write failed");
}
}
