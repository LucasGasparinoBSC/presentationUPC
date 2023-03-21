#include <stdlib.h>
#include <stdio.h>
#include "kernels.h"

int main(void)
{
    // Definne array  size
    int n = 512*512;

    // Allocate host arrrays
    float *h_a = (float *)malloc(n*sizeof(float));
    float *h_b = (float *)malloc(n*sizeof(float));
    float *h_c = (float *)malloc(n*sizeof(float));

    // Fill arrays a and b
    for (int i = 0; i < n; i++)
    {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }

    // Call cpuVecAdd
    cpuVecAdd(n, h_a, h_b, h_c);
    printf("h_c[0] = %f\n", h_c[0]);

    // Zero array h_c
    for (int i = 0; i < n; i++)
    {
        h_c[i] = 0.0f;
    }

    // Allocate device arrrays
    float *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, n*sizeof(float));
    cudaMalloc((void **)&d_b, n*sizeof(float));
    cudaMalloc((void **)&d_c, n*sizeof(float));

    // Copy host arrays to device
    cudaMemcpy(d_a, h_a, n*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(float), cudaMemcpyHostToDevice);

    // Kernel exec grid: given a number of threadds per block, must launch enough blocks to cover all the data
    int tbp_x = 128;
    int numTB_x = (n + tbp_x - 1) / tbp_x; // Guarantees 1 extra block if n is not a multiple of tbp_x
    dim3 block(tbp_x,1,1);
    dim3 grid(numTB_x,1,1);

    // Call gpuVecAdd
    gpuVecAdd<<<grid,block>>>(n, d_a, d_b, d_c);

    // Copy device array to host
    cudaMemcpy(h_c, d_c, n*sizeof(float), cudaMemcpyDeviceToHost);
    printf("h_c[0] = %f\n", h_c[0]);

    // Exit the code
    return 0;
}