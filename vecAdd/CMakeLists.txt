cmake_minimum_required(VERSION 3.15)

project(cudaKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_VERSION 17)

add_executable(cudaKernels main.cu kernels.cu)
set_target_properties(cudaKernels PROPERTIES CUDA_ARCHITECTURES "61")