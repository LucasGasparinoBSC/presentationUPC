#ifndef KERNELS_CUH
#define KERNELS_CUH

#include <cuda.h>
#include <stdint.h>

double hostDotProduct(float *x, float *y, uint32_t N);
__global__ void naiveDotProduct(float *x, float *y, double *s, uint32_t N);
__global__ void optDotProduct(float *x, float *y, double *s, uint32_t N);

#endif // KERNELS_CUH