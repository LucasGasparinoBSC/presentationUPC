#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

void cpuVecAdd(int n, float* a, float *b, float *c);
__global__ void gpuVecAdd(int n, float* a, float *b, float *c);

#endif // KERNELS_H