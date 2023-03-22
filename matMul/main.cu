#include <stdio.h>
#include "kernels.cuh"

int main()
{
    // Set matrix row size
    int nrow = 4096;
    printf("Total memory used: %f MB\n", 3*nrow*nrow*sizeof(float)/1024.0f/1024.0f);

    // Allocate n*n matrices
    float *a = (float*)malloc(nrow*nrow*sizeof(float));
    float *b = (float*)malloc(nrow*nrow*sizeof(float));
    float *c = (float*)malloc(nrow*nrow*sizeof(float));

    // Fill the matrices
    for (int i = 0; i < nrow; i++)
    {
        for (int j = 0; j < nrow; j++)
        {
            a[i*nrow+j] = 1.0f;
            b[i*nrow+j] = 2.0f;
            c[i*nrow+j] = 0.0f;
        }
    }

    // Create device copies of the host matrices
    float *d_a, *d_b, *d_c;
    cudaMalloc((void**)&d_a, nrow*nrow*sizeof(float));
    cudaMalloc((void**)&d_b, nrow*nrow*sizeof(float));
    cudaMalloc((void**)&d_c, nrow*nrow*sizeof(float));
    cudaMemcpy(d_a, a, nrow*nrow*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, nrow*nrow*sizeof(float), cudaMemcpyHostToDevice);

    // Configure a grid for kernel0
    int ntx = 16;
    int nty = 16;
    int nBlocksX = (nrow+ntx-1)/ntx;
    int nBlocksY = (nrow+nty-1)/nty;
    dim3 block(ntx, nty);
    dim3 grid(nBlocksX, nBlocksY);
    printf("Block size: %d x %d\n", ntx, nty);
    printf("Grid size: %d x %d\n", nBlocksX, nBlocksY);

    // Launch kernel0 10 times
    for (int i = 0; i < 10; i++)
    {
        matmul0<<<grid, block>>>(d_a, d_b, d_c, nrow);
    }

    // Copy the result back to the host
    cudaMemcpy(c, d_c, nrow*nrow*sizeof(float), cudaMemcpyDeviceToHost);
    printf("c[0] = %f\n", c[0]);

    // Call the tiled matmul kernel 10 times
    for (int i = 0; i < 10; i++)
    {
        tiledMatmul<<<grid, block,ntx*nty*sizeof(float)>>>(d_a, d_b, d_c, nrow);
    }

    // Copy the result back to the host
    cudaMemcpy(c, d_c, nrow*nrow*sizeof(float), cudaMemcpyDeviceToHost);
    printf("c[0] = %f\n", c[0]);
    return 0;
}