#ifndef KERNELS_H
#define KERNELS_H

#include <cuda.h>

void matmulCPU(float *A, float *B, float *C, int N);
__global__ void matmul0(float *A, float *B, float *C, int N);
__global__ void tiledMatmul(float *A, float *B, float *C, int N);

#endif // KERNELS_H