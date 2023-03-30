#!/bin/bash

# This script is used to build the project

# If no build directory exists, create one annd enter it
if [ ! -d "build" ]; then
    mkdir -p build && cd build
else
    cd build && rm -rf *
fi

# Run CMake using the nvfortran compiler, then build the project
cmake -DCMAKE_Fortran_COMPILER=nvfortran -DCMAKE_BUILD_TYPE=Release ..
make

# Return to the root directory and copy the executable
cd ..
cp build/FEM1D .