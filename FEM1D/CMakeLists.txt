cmake_minimum_required(VERSION 3.15)

project(FEM1D LANGUAGES Fortran)

set(CMAKE_Fortran_FLAGS "-cpp -D_USE_NVTX -fast -gpu=cc61,managed,lineinfo -cuda -acc -Minfo=all")

add_executable(FEM1D mod_nvtx.f90 fem1d.f90)
target_link_libraries(FEM1D nvToolsExt)