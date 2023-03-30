# Build and run instructions

## Prerequisites

* [CMake](https://cmake.org/) 3.15 or newer
* [NVIDIA drivers](https://www.nvidia.com/Download/index.aspx) Check mode and OS language
* [CUDA](https://developer.nvidia.com/cuda-toolkit) driver-dependent
* [NVHPC](https://developer.nvidia.com/hpc-sdk) 22.11 or newer

Make sure that CUDA and the NVHPC compilers are in your PATH, as well as loading the libraries into LD_LIBRARY_PATH. Check the [CUDA documentation](https://docs.nvidia.com/cuda/) and [NVHPC documentation](https://docs.nvidia.com/hpc-sdk/compilers/index.html) for more information. Project only builds on Linux!!

## Building

Please use the included `buildScript.sh` to build the project. It will create a build directory and build the project there, or reuse an existing build directory. It will copy the `FEM1D` executable into the root folder.

## Running

Simply execute the `FFEM1D` executable in a terminal as follllows:

```bash
./FEM1D
```

To modify parameters, the `fem1d.f90` file must be modified. All modifiable parameters are inside the `program` structure.

## Profiling

Use [Nsight Systems](https://developer.nvidia.com/nsight-systems) to profile the code using the following command:

```bash
nsys profile --stats=true --force-overwrite=true --trace=cuda,nvtx  --cuda-memory-usage=true --output=nsys-report ./FEM1D
```

The `output` name can be changed to whatever you want. A `.nsys-rep` file will be generated that can be opened with:

```bash
nsys-ui nsys-report.nsys-rep
```

For kernel profiling an analysis, the [Nsight Compute](https://developer.nvidia.com/nsight-compute) tool can be used. This requires a GPU with a compute capability of 7.0 or higher. A proper command to assess the performance of the GPU kernel will be posted here later on.