#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <nvToolsExt.h>
#include <chrono>
#include "kernels.cuh"
#include "nvtx_macro.cuh"

int main()
{
    // Define datasets
    const uint32_t nruns=10;
    uint32_t N = 100*(1 << 20); // 100M elements
    float h = 6.28f / (float) N;
    float *x  = (float *) malloc(N * sizeof(float));
    float *y1 = (float *) malloc(N * sizeof(float));
    float *y2 = (float *) malloc(N * sizeof(float));

    for ( uint32_t i = 0; i < N; i++ )
    {
        x[i] = 0.0f + (float)i * h;
        y1[i] = sin(x[i]);
        y2[i] = cos(x[i]);
    }

    // Host timing
    double refResult;
    clock_t start, end;
    double tAvg = 0.0;

    PUSH_RANGE("Host timing", 0);
    for (uint32_t iter = 0; iter < nruns; iter++)
    {
        start = clock();
        refResult = hostDotProduct(y1, y2, N);
        end = clock();
        tAvg += (double)(end - start) / CLOCKS_PER_SEC;
    }
    POP_RANGE;
    tAvg = tAvg / (double)nruns;
    printf("Host time: %f seconds\n", tAvg);
    printf("Host result: %f\n", refResult);

    // Device arrays
    float *d_y1;
    float *d_y2;
    double *gpuRes;
    double *hostRes = (double *) malloc(sizeof(double));
    cudaMalloc((void **)&d_y1, N * sizeof(float));
    cudaMalloc((void **)&d_y2, N * sizeof(float));
    cudaMalloc((void **)&gpuRes, sizeof(double));
    cudaMemset(gpuRes, 0, sizeof(double));
    cudaMemcpy(d_y1, y1, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y2, y2, N * sizeof(float), cudaMemcpyHostToDevice);

    // Naive kernel runs
    dim3 block(128, 1, 1);
    dim3 grid((N + block.x - 1) / block.x, 1, 1);

    PUSH_RANGE("Naive kernel", 0);
    for ( uint32_t iter = 0; iter < nruns; iter++)
    {
        cudaMemset(gpuRes, 0, sizeof(double));
        naiveDotProduct<<< grid , block >>>(d_y1, d_y2, gpuRes, N);
    }
    POP_RANGE;
    cudaMemcpy(hostRes, gpuRes, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Naive kernel result: %f\n", *hostRes);

    // Zero the host result
    memset(hostRes, 0, sizeof(double));

    // Tiled kernel runs
    dim3 blockOpt(128, 1, 1);
    dim3 gridOpt((N + block.x - 1) / block.x, 1, 1);

    PUSH_RANGE("Tiled kernel", 0);
    for ( uint32_t iter = 0; iter < nruns; iter++)
    {
        cudaMemset(gpuRes, 0, sizeof(double));
        optDotProduct<<< gridOpt , blockOpt >>>(d_y1, d_y2, gpuRes, N);
    }
    POP_RANGE;
    
    cudaMemcpy(hostRes, gpuRes, sizeof(double), cudaMemcpyDeviceToHost);
    printf("Naive kernel result: %f\n", *hostRes);


    return 0;
}