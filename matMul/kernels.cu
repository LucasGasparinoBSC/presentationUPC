#include "kernels.cuh"
#include <stdio.h>

void matmulCPU(float *A, float *B, float *C, int N) {
    printf("Launching CPU matmul\n");
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/**
 * @brief Non-optimal matrix multiplication kernel.
 *
 * @param A (float) Input matrix A
 * @param B (float) Input matrix B
 * @param C (float) Output matrix C
 * @param N (int) Row size of the matrices
 */
__global__ void matmul0(float *A, float *B, float *C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N) {
        float sum = 0;
        for (int k = 0; k < N; k++) {
            sum += A[i * N + k] * B[k * N + j];
        }
        C[i * N + j] = sum;
    }
}

__global__ void tiledMatmul(float *A, float *B, float *C, int N)
{
    // Create shared memory blocks
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];
    // Get block indices
    int ib = blockIdx.x;
    int jb = blockIdx.y;
    // Get thread indices
    int it = threadIdx.x;
    int jt = threadIdx.y;
    // Get global indices
    int i = ib * 16 + it;
    int j = jb * 16 + jt;
    // Check if indices are valid
    if (i < N && j < N) {
        float sum = 0;
        // Loop over tiles
        for (int t = 0; t < N / 16; t++) {
            // Load tiles into shared memory
            As[it][jt] = A[i * N + t * 16 + jt];   // Load a chunk of A
            Bs[it][jt] = B[(t * 16 + it) * N + j]; // Load a chunk of B^T
            // Synchronize threads
            __syncthreads();
            // Compute partial sum
            for (int k = 0; k < 16; k++) {
                sum += As[it][k] * Bs[k][jt];
            }
            // Synchronize threads
            __syncthreads();
        }
        // Write result
        C[i * N + j] = sum;
    }
}