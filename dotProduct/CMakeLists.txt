cmake_minimum_required(VERSION 3.15)

project(gpu_dot LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD_VERSION 17)
set(CMAKE_CUDA_FLAGS "-O3 -lineinfo -res-usage")

file(GLOB_RECURSE SRC_FILES src/*.cu)
set(HEADER_DIR src)
file(GLOB_RECURSE HEADER_FILES src/*.cuh)

include_directories(${HEADER_DIR})
add_executable(gpu_dot ${SRC_FILES} ${HEADER_FILES})
target_link_libraries(gpu_dot nvToolsExt)
set_target_properties(gpu_dot PROPERTIES CUDA_ARCHITECTURES "61")