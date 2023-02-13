#pragma once

#include <iostream>
#include <cstdlib>
#include <stdint.h>

namespace clustering {

// -----------------------------------------------------------------------------
//  marco & typedef
// -----------------------------------------------------------------------------
#define SQR(x)              ((x) * (x))
#define DEBUG_INFO

typedef uint8_t  u08;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef float    f32;
typedef double   f64;

// -----------------------------------------------------------------------------
//  general constants
// -----------------------------------------------------------------------------
const f32 E            = 2.7182818F;
const f32 PI           = 3.141592654F;
const f32 FLOAT_ERROR  = 1e-6F;

const f32 MAX_FLOAT    = 3.402823466e+38F;
const f32 MIN_FLOAT    = -MAX_FLOAT;
const u32 UINT32_PRIME = 4294967291U; // uint32 prime (2^32-5)
const u32 MAX_UINT32   = 4294967295U; // 2^32-1
const int MAX_INT      = 2147483647;  // 2^31-1
const int MIN_INT      = -MAX_INT;

const int RANDOM_SEED  = 666;         // random seed
const int MAX_ITER     = 10;          // maximum iteration

} // end namespace clustering
