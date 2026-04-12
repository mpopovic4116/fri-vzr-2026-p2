#ifndef FLOAT_ALIAS_H
#define FLOAT_ALIAS_H

#if FEAT_PRECISION == 64
typedef double fhost;
typedef double fcuda;
#elif FEAT_PRECISION == 32
typedef float fhost;
typedef float fcuda;
#elif FEAT_PRECISION == 16
#include <cuda_fp16.h>
typedef _Float16 fhost;
typedef _Float16 fcuda;
#else
#error The precision= compile flag must be 64, 32 or 16
#endif

#endif
