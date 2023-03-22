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

__global__ void tiledMatmul(float *A, float *B, float *C, int N) {
    // Create the empty dynamic shared memory arrays
    extern __shared__ float Atile[];
    extern __shared__ float Btile[];
    // Define tile sizes
    int bx = blockDim.x;
    int by = blockDim.y;
    // Get the thread indexes
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    // Define global indexes
    int i = bx * blockIdx.x + tx; // Row
    int j = by * blockIdx.y + ty; // Column
    // Get grid dimensions (blocks in the grid, per direction)
    int gridDimX = gridDim.x;
    int gridDimY = gridDim.y;
    if (i < N && j < N) {
        // Set an initial temp to 0
        float sum = 0.0f;
        // Compute
        for (int k = 0; k < gridDimX; k++) {
            // Load global chunks into shared memory
            Atile[tx * bx + ty] = A[i * N + k * by + ty];
            Btile[ty * by + tx] = B[j + N*(k*bx+tx)]; // local B already transposed
            __syncthreads();
            // Compute the partial sum
            for (int kk = 0; kk < bx; kk++) {
                sum += Atile[kk*bx+tx] * Btile[kk*bx+ty];
            }
            __syncthreads();
            // Write the partial sum to global memory
        }
        C[i * N + j] = sum;
    }
}