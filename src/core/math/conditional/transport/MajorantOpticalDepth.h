#pragma once
#include "math/conditional/core/Random.h"
#include "math/conditional/mathutility/NumericPolicy.h"

namespace Tungsten::ConditionalGpis {

// One exponential clock for one track() call. Consume intervals in ray order;
// a changed majorant or an adaptive subdivision does not reset the clock.
class MajorantOpticalDepth {
    double remaining_ = -1.0; // negative means the next event needs a new draw
public:
    // maximum >= 0 and begin < end. On success, candidate is in [begin,end);
    // otherwise the residual optical depth is retained for the next interval.
    bool advance(Random &rng, double maximum, double begin, double end, double &candidate) {
        if (maximum == 0.0) return false;
        if (remaining_ < 0.0) remaining_ = -std::log(rng.openUniform01());
        // An exact boundary event belongs to the next positive-rate interval.
        // Zero-rate intervals consume neither optical depth nor random draws.
        if (remaining_ == 0.0) {
            candidate = begin;
            remaining_ = -1.0;
            return true;
        }
        const double opticalDepth = maximum * (end - begin);
        if (remaining_ >= opticalDepth) {
            remaining_ -= opticalDepth;
            return false;
        }
        const double next = begin + remaining_ / maximum;
        if (next >= end) {
            // A candidate just before the face can round onto it. Carry that
            // event to the next interval instead of drawing a second threshold.
            remaining_ = 0.0;
            return false;
        }
        if (!(next > begin))
            throw NumericError(NumericStatus::NeedHigherPrecision, "conditional event cannot advance");
        candidate = next;
        remaining_ = -1.0;
        return true;
    }
};

}
