#ifndef FLOAT_ALIAS_H
#define FLOAT_ALIAS_H

#if FEAT_PRECISION == 64
typedef double fhost;
typedef double fcuda;
#define fhost_fmax fmax
#define fcuda_fmax fmax
#define fhost_fmin fmin
#define fcuda_fmin fmin
#elif FEAT_PRECISION == 32
typedef float fhost;
typedef float fcuda;
#define fhost_fmax fmaxf
#define fcuda_fmax fmaxf
#define fhost_fmin fminf
#define fcuda_fmin fminf
#elif FEAT_PRECISION == 16
#include <cuda_fp16.h>
typedef _Float16 fhost;
typedef _Float16 fcuda;
#error TODO
#else
#error The precision= compile flag must be 64, 32 or 16
#endif

#endif
