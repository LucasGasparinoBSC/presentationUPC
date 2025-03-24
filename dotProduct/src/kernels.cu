#include "kernels.cuh"

double hostDotProduct(float *x, float *y, uint32_t N)
{
    double s = 0.0;
    for (uint32_t i = 0; i < N; i++)
    {
        s += (double) (x[i] * y[i]);
    }
    return s;
}

__global__ void naiveDotProduct(float *x, float *y, double *result, uint32_t N)
{
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < N)
    {
        atomicAdd(result, (double)(x[tid] * y[tid]));
    }
}

// Tiled dot product
__global__ void optDotProduct(float *x, float *y, double *s, uint32_t N)
{

    // Global and local access indexes
    uint32_t gID = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t lID = threadIdx.x;

    // Buffer on shared memory
    // SMEM expects a block with 128 threads. Can that be dynamic??
    __shared__ double buffer[128];

    // Scatter operation (from global to local)
    double tmp = 0.0;
    while (gID < N)
    {
        tmp += (double)(x[gID] * y[gID]);
        gID += blockDim.x * gridDim.x;
    }
    buffer[lID] = tmp;
    __syncthreads();

    // Accumulate on the cache
    uint32_t i = blockDim.x / 2;
    while (i != 0)
    {
        if (lID < i)
        {
          buffer[lID] += buffer[lID + i];
        }
        __syncthreads();
        i /= 2;
    }

    // Atomic accumulation
    if (lID == 0)
    {
        atomicAdd(s, buffer[0]);
    }
}
