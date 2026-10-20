#ifndef INTTYPES_HPP_
#define INTTYPES_HPP_

#include <cstdint>

namespace Tungsten {

typedef std::uint8_t  uint8;
typedef std::uint16_t uint16;
typedef std::uint32_t uint32;
typedef std::uint64_t uint64;

// Not a standard C++ type. On glibc it leaks in from <sys/types.h> via
// _DEFAULT_SOURCE, which is why the code compiles on Linux but not on MSVC.
typedef unsigned int uint;

typedef std::int8_t  int8;
typedef std::int16_t int16;
typedef std::int32_t int32;
typedef std::int64_t int64;

}

#endif /* INTTYPES_HPP_ */
