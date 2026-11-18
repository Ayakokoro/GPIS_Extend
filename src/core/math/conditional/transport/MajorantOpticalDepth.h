#pragma once
#include "math/conditional/core/Random.h"
#include "math/conditional/mathutility/NumericPolicy.h"

namespace Tungsten::ConditionalGpis {

// One exponential clock for one track() call. Consume intervals in ray order;
// a changed majorant or an adaptive subdivision does not reset the clock.
class MajorantOpticalDepth {
    double remaining_ = -1.0; // negative means the next event needs a new draw
public:
    // Same residual clock for a nonconstant intensity. The caller supplies
    // its integral on [begin,end] and the inverse of that cumulative integral.
    template<typename Inverse>
    bool advanceIntegrated(Random &rng,double begin,double end,double depth,
                           const Inverse &inverse,double &candidate) {
        if(!(depth>=0) || !std::isfinite(depth))
            throw NumericError(NumericStatus::InvalidMajorant,"invalid integrated majorant");
        if(depth==0) return false;
        if(remaining_<0) remaining_=-std::log(rng.openUniform01());
        if(remaining_==0) { candidate=begin;remaining_=-1;return true; }
        if(remaining_>=depth) { remaining_-=depth;return false; }
        const double next=inverse(remaining_);
        if(next>=end) { remaining_=0;return false; }
        if(!(next>begin) || !std::isfinite(next))
            throw NumericError(NumericStatus::NeedHigherPrecision,"integrated conditional event cannot advance");
        candidate=next;remaining_=-1;
        return true;
    }
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

    // Keep the clock in interval-local coordinates. Distinct events may round
    // to the same ray position while their offsets and optical depths advance.
    bool advanceFromOffset(Random &rng, double maximum, double begin, double end,
                           double &offset, double &candidate) {
        if (!advance(rng, maximum, offset, end - begin, offset))
            return false;
        candidate = begin + offset;
        if (candidate >= end)
            candidate = std::nextafter(end, begin);
        return true;
    }
};

}
