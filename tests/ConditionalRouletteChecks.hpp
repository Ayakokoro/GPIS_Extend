#pragma once

inline void shadowRouletteChecks()
{
    auto f=field();f.mean=std::make_shared<cg::ConstantMean>(0);
    f.kernel=cg::CovarianceKernel(1,M::Identity());
    cg::FlightState birth;birth.birthGradient=V::UnitZ();
    cg::ConditionalTrackingSettings on;
    on.majorantMode=cg::ConditionalMajorantMode::TwoConstants;
    on.constantBirthLength=4;
    auto off=on;off.shadowRouletteThreshold=0;
    const double length=12;
    const cg::ConditionalSegment enabled(f,birth,4*V::UnitZ(),V::UnitZ(),length,on);
    const cg::ConditionalSegment disabled(f,birth,4*V::UnitZ(),V::UnitZ(),length,off);

    // Script a sequence of candidates down to the first roulette decision,
    // then force either termination or survival followed by an empty tail.
    const double maximum=enabled.majorant(0,length);
    const float stepUniform=0.9f,escapeUniform=std::numeric_limits<float>::min();
    check(maximum*length < -std::log(double(escapeUniform)),"roulette fixture can escape with one draw");
    double position=0,weight=1;
    std::vector<float> draws;
    while(weight>=on.shadowRouletteThreshold) {
        position+=-std::log(double(stepUniform))/maximum;
        check(position<length && draws.size()<1000,"roulette fixture reaches a low positive weight");
        draws.push_back(stepUniform);
        weight*=1-enabled.hazard(position)/maximum;
    }
    check(weight>0,"roulette fixture does not rely on an exactly zero ratio weight");
    const auto run=[&](const cg::ConditionalSegment &flight,std::vector<float> values,
                       cg::ConditionalTrackingWork &work) {
        const size_t expectedDraws=values.size();
        OpticalDepthSampler source(std::move(values));cg::Random rng(source);double hit;
        const double result=flight.track(rng,false,hit,&work);
        check(source.calls==expectedDraws,
            "roulette consumes only its decision draws and no tail draws on termination");
        return result;
    };
    cg::ConditionalTrackingWork killedWork,survivedWork,offWork;
    auto killed=draws;killed.push_back(std::nextafter(1.0f,0.0f));
    close(run(enabled,killed,killedWork),0,0,"roulette terminates a low-weight shadow");
    check(killedWork.distance<length && killedWork.candidates==draws.size(),
        "roulette termination skips the remaining shadow interval");
    auto survived=draws;survived.push_back(escapeUniform);survived.push_back(escapeUniform);
    close(run(enabled,survived,survivedWork),0.1,0,"roulette survivor receives compensating weight");
    auto noRoulette=draws;noRoulette.push_back(escapeUniform);
    close(run(disabled,noRoulette,offWork),weight,1e-12,"disabled roulette retains the original ratio weight");
    check(survivedWork.distance==length && offWork.distance==length,"surviving shadows reach their endpoint");

    // Collision samples must retain exactly the same random stream and hits.
    cg::Random collisionOn(918),collisionOff(918);
    for(int i=0;i<128;++i) {
        double a,b;
        const double wa=enabled.track(collisionOn,true,a),wb=disabled.track(collisionOff,true,b);
        check(wa==wb && a==b,"shadow roulette leaves collision tracking unchanged");
    }

    // Check low-transmittance expectation against an independent hazard integral,
    // using both a zero-valued collision birth and a positive exterior birth.
    for(double f0:{0.0,0.3}) {
        birth.birthValue=f0;
        const cg::ConditionalSegment reference(f,birth,0.4*V::UnitZ(),V::UnitZ(),length,off);
        double optical=0;
        for(int i=0;i<16000;++i) optical+=reference.hazard(length*(i+0.5)/16000)*length/16000;
        const double expected=std::exp(-optical);
        check(expected<0.1,"roulette survival fixture exercises an opaque flight");
        for(double threshold:{0.0,0.1,0.5}) {
            auto settings=on;settings.shadowRouletteThreshold=threshold;
            settings.majorantMode=cg::ConditionalMajorantMode::Adaptive;
            settings.stepScale=2;settings.ageScale=0.5;
            const cg::ConditionalSegment flight(f,birth,0.4*V::UnitZ(),V::UnitZ(),length,settings);
            cg::Random rng(2731);double sum=0,sum2=0;int early=0;
            constexpr int n=12000;
            for(int i=0;i<n;++i) {
                cg::ConditionalTrackingWork work;double hit;
                const double tr=flight.track(rng,false,hit,&work);
                sum+=tr;sum2+=tr*tr;early+=work.distance<length;
            }
            const double variance=std::max(0.0,(sum2-sum*sum/n)/(n-1));
            close(sum/n,expected,6*std::sqrt(variance/n)+1e-4,
                "roulette preserves independently integrated conditional transmittance");
            if(threshold>0) check(early>n/2,"roulette ends most opaque shadow walks early");
        }
    }
}
