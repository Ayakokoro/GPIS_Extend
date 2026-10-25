// Fixed camera-ray ensembles, using the same medium/phase code as rendering.
// These are collision-conditioned normal histograms, not fixed-point NDFs.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <mutex>
#include <thread>

namespace SurfaceEnsemble {
using Count = std::uint64_t;
struct Histogram {
    Count attempted=0, hits=0, escaped=0, distanceFailed=0, phaseFailed=0;
    std::vector<Count> normals, theta, incidence;
    std::array<std::vector<Count>,2> split;
    V normalSum=V::Zero(), positionSum=V::Zero();
    double minGradientAgreement=1;
    Histogram(int nz,int np,int na):normals(nz*np),theta(na),incidence(na) {
        for(auto &s:split) s.resize(nz*np);
    }
    void merge(const Histogram &h) {
        attempted+=h.attempted;hits+=h.hits;escaped+=h.escaped;
        distanceFailed+=h.distanceFailed;phaseFailed+=h.phaseFailed;
        normalSum+=h.normalSum;positionSum+=h.positionSum;
        minGradientAgreement=std::min(minGradientAgreement,h.minGradientAgreement);
        for(size_t i=0;i<normals.size();++i) {
            normals[i]+=h.normals[i];
            for(int s=0;s<2;++s) split[s][i]+=h.split[s][i];
        }
        for(size_t i=0;i<theta.size();++i) {theta[i]+=h.theta[i];incidence[i]+=h.incidence[i];}
    }
    J json(Count paths) const {
        return {{"attempted",attempted},{"hits",hits},{"escaped",escaped},
            {"distance_failed",distanceFailed},{"phase_failed",phaseFailed},
            {"hit_probability",double(hits)/paths},{"normal_counts",normals},
            {"theta_counts",theta},{"incidence_counts",incidence},{"split_counts",split},
            {"mean_normal",jsonV(hits?V(normalSum/double(hits)):V::Zero())},
            {"mean_position",jsonV(hits?V(positionSum/double(hits)):V::Zero())},
            {"min_gpis_gradient_agreement",minGradientAgreement}};
    }
};

int bin(double x,int size) {return std::clamp(int(x*size),0,size-1);}

void run(const J &spec,const char *out) {
    const int samples=spec.value("samples",200000),threads=spec.value("threads",24);
    const int nz=spec.value("nz",48),np=spec.value("nphi",96),na=spec.value("angle_bins",90);
    if(samples<1||threads<1||threads>256||nz<4||nz>512||np<4||np>1024||na<4||na>360)
        throw std::runtime_error("invalid ensemble dimensions");
    const J &record=spec.at("reference_ray");
    const V lo=vector(spec.at("box_min")),hi=vector(spec.at("box_max"));
    const V cameraOrigin=vector(record.at("vertices")[0]);
    const Vec3f initialDirection=vecF(vector(record.at("wi"))).normalized();
    const M basis=frame(record);
    double a,b;
    if(!interval(cameraOrigin,eigen(initialDirection),lo,hi,a,b))
        throw std::runtime_error("selected ray misses process box");
    const Vec3f initialOrigin=vecF(cameraOrigin+a*eigen(initialDirection));
    const unsigned x=record.at("pixel")[0],y=record.at("pixel")[1];
    const unsigned width=record.at("diagnostic_resolution")[0],seed=spec.value("seed",20261024u);
    J results=J::array();
    Scene scene;
    for(size_t mode=0;mode<spec.at("media").size();++mode) {
        JsonDocument doc(Path("ensemble-medium.json"),spec["media"][mode].dump());
        auto medium=scene.fetchMedium(doc);
        medium->loadResources();medium->prepareForRender();
        const bool isGpis=spec["media"][mode]["type"]=="sparse_conv_noise";
        const std::string name=spec.at("modes")[mode];
        const auto started=std::chrono::steady_clock::now();
        std::atomic<int> next{0};std::atomic<bool> failed{false};
        std::exception_ptr error;std::mutex errorMutex;
        std::vector<std::array<Histogram,2>> stats;
        for(int t=0;t<threads;++t) stats.push_back({Histogram(nz,np,na),Histogram(nz,np,na)});
        auto worker=[&](int thread) {
            try {
                IndependentPathSampler rng(seed,false);
                for(int index;(index=next.fetch_add(1))<samples&&!failed.load();) {
                    rng.startPath(x+y*width,index);
                    Medium::MediumState state{};
                    state.info.pixelSampleSegment=Vec4u(x,y,unsigned(index),0u);
                    state.info.sceneSeed=seed;state.info.t=0;state.reset();
                    Vec3f origin=initialOrigin,direction=initialDirection;
                    for(int bounce=0;bounce<2;++bounce) {
                        auto &h=stats[thread][bounce];++h.attempted;
                        double nearT,farT;
                        if(!interval(eigen(origin),eigen(direction),lo,hi,nearT,farT)) {++h.escaped;break;}
                        state.info.pixelSampleSegment.w()=bounce+1;
                        Ray ray(origin,direction,0,float(farT));MediumSample hit{};
                        if(!medium->sampleDistance(rng,ray,state,hit)) {++h.distanceFailed;break;}
                        if(hit.exited) {++h.escaped;break;}
                        PhaseSample phase{};
                        if(!hit.phase->sample(rng,direction,hit,phase)) {++h.phaseFailed;break;}
                        const V wi=eigen(direction).normalized(),wo=eigen(phase.w).normalized();
                        V n=wo-wi;
                        if(!n.allFinite()||n.norm()<1e-9)
                            throw std::runtime_error("invalid mirror half-vector");
                        n.normalize(); // outward-facing normal: dot(wi,n) <= 0
                        if(n.dot(wi)>1e-6) throw std::runtime_error("mirror normal faces away");
                        if(isGpis) {
                            const double agreement=n.dot(eigen(hit.aniso).normalized());
                            h.minGradientAgreement=std::min(h.minGradientAgreement,agreement);
                            if(!std::isfinite(agreement)||agreement<.9999)
                                throw std::runtime_error("GPIS reflection disagrees with realized field gradient");
                        }
                        const V local=basis.transpose()*n;
                        int iz=bin((local.z()+1)/2,nz);
                        int ip=bin((std::atan2(local.y(),local.x())+Pi)/(2*Pi),np);
                        ++h.normals[iz*np+ip];++h.split[index%2][iz*np+ip];++h.hits;
                        ++h.theta[bin(std::acos(std::clamp(local.z(),-1.,1.))/Pi,na)];
                        ++h.incidence[bin(std::acos(std::clamp(-n.dot(wi),0.,1.))/(Pi/2),na)];
                        h.normalSum+=n;h.positionSum+=eigen(hit.p);
                        if(phase.pathContext) state.pathContext=phase.pathContext;
                        origin=hit.p;direction=phase.w;
                    }
                }
            } catch(...) {
                failed.store(true);std::lock_guard<std::mutex> lock(errorMutex);
                if(!error) error=std::current_exception();
            }
        };
        std::cout<<"Sampling "<<name<<": "<<samples<<" fixed-ray paths\n"<<std::flush;
        std::vector<std::thread> workers;
        for(int t=0;t<threads;++t) workers.emplace_back(worker,t);
        for(auto &w:workers) w.join();
        if(error) std::rethrow_exception(error);
        std::array<Histogram,2> total{Histogram(nz,np,na),Histogram(nz,np,na)};
        for(const auto &s:stats) for(int k=0;k<2;++k) total[k].merge(s[k]);
        for(int k=0;k<2;++k) {
            const auto &h=total[k];
            if(h.attempted!=h.hits+h.escaped+h.distanceFailed+h.phaseFailed)
                throw std::runtime_error("ensemble event accounting failed");
            if(h.attempted!=(k?total[0].hits:Count(samples)))
                throw std::runtime_error("ensemble continuation accounting failed");
        }
        const double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
        J entry={{"mode",name},{"seconds",seconds},{"collisions",{total[0].json(samples),total[1].json(samples)}}};
        if(auto c=std::dynamic_pointer_cast<GlobalConditionalMedium>(medium))
            entry["discarded_surface_paths"]=c->discardedSurfacePaths();
        results.push_back(entry);
        std::cout<<"Done "<<name<<": hits "<<total[0].hits<<", "<<total[1].hits<<" in "<<seconds<<" sec\n"<<std::flush;
        medium->teardownAfterRender();
    }
    save(out,J{{"spec",spec},{"frame",jsonM(basis)},{"nz",nz},{"nphi",np},{"angle_bins",na},
        {"results",results},{"sampling","Fixed camera ray; production medium and mirror phase samplers; "
        "independent realizations per path, GPIS field retained along each path. First two collisions, "
        "no NEE RNG draws or roulette. Failed paths are counted, never retried. Histograms are conditional "
        "on recorded collisions, not NDFs at a fixed point. Collision positions and second directions vary."}});
}
}
