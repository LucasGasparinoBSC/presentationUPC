cmake_minimum_required(VERSION 3.15)

project(cudaKernels LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_VERSION 17)
set(CMAKE_CUDA_FLAGS "-O3")

add_executable(cudaKernels main.cu kernels.cu)
target_link_libraries(cudaKernels nvToolsExt)
set_target_properties(cudaKernels PROPERTIES CUDA_ARCHITECTURES "86")
