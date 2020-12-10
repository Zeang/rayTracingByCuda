#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <thrust/version.h>
#include <iostream>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cmath>

// square<T> computes the square of a number f(x) -> x*x
template <typename T>
struct square
{
    __host__ __device__
    T operator()(const T& x) const
    {
        return x * x;
    }
};

int main(void)
{
    // initialize host array
    float x[4] = { 1.0, 2.0, 3.0, 4.0 };

    // transfer to device
    thrust::device_vector<float> d_x(x, x + 4);

    // setup arguments
    square<float>        unary_op;
    thrust::plus<float> binary_op;
    float init = 0;

    // compute norm
    float norm = std::sqrt(thrust::transform_reduce(d_x.begin(), d_x.end(), unary_op, init, binary_op));

    std::cout << norm << std::endl;

    return 0;
}
//int main() {
//    int deviceCount;
//    cudaGetDeviceCount(&deviceCount);
//
//    int dev;
//    for (dev = 0; dev < deviceCount; dev++)
//    {
//        int driver_version(0), runtime_version(0);
//        cudaDeviceProp deviceProp;
//        cudaGetDeviceProperties(&deviceProp, dev);
//        if (dev == 0)
//            if (deviceProp.minor = 9999 && deviceProp.major == 9999)
//                printf("\n");
//        printf("\nDevice%d:\"%s\"\n", dev, deviceProp.name);
//        cudaDriverGetVersion(&driver_version);
//        printf("CUDA驱动版本:                                   %d.%d\n", driver_version / 1000, (driver_version % 1000) / 10);
//        cudaRuntimeGetVersion(&runtime_version);
//        printf("CUDA运行时版本:                                 %d.%d\n", runtime_version / 1000, (runtime_version % 1000) / 10);
//        printf("设备计算能力:                                   %d.%d\n", deviceProp.major, deviceProp.minor);
//        printf("Total amount of Global Memory:                  %u bytes\n", deviceProp.totalGlobalMem);
//        printf("Number of SMs:                                  %d\n", deviceProp.multiProcessorCount);
//        printf("Total amount of Constant Memory:                %u bytes\n", deviceProp.totalConstMem);
//        printf("Total amount of Shared Memory per block:        %u bytes\n", deviceProp.sharedMemPerBlock);
//        printf("Total number of registers available per block:  %d\n", deviceProp.regsPerBlock);
//        printf("Warp size:                                      %d\n", deviceProp.warpSize);
//        printf("Maximum number of threads per SM:               %d\n", deviceProp.maxThreadsPerMultiProcessor);
//        printf("Maximum number of threads per block:            %d\n", deviceProp.maxThreadsPerBlock);
//        printf("Maximum size of each dimension of a block:      %d x %d x %d\n", deviceProp.maxThreadsDim[0],
//            deviceProp.maxThreadsDim[1],
//            deviceProp.maxThreadsDim[2]);
//        printf("Maximum size of each dimension of a grid:       %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
//        printf("Maximum memory pitch:                           %u bytes\n", deviceProp.memPitch);
//        printf("Texture alignmemt:                              %u bytes\n", deviceProp.texturePitchAlignment);
//        printf("Clock rate:                                     %.2f GHz\n", deviceProp.clockRate * 1e-6f);
//        printf("Memory Clock rate:                              %.0f MHz\n", deviceProp.memoryClockRate * 1e-3f);
//        printf("Memory Bus Width:                               %d-bit\n", deviceProp.memoryBusWidth);
//    }
//
//    int major = THRUST_MAJOR_VERSION;
//    int minor = THRUST_MINOR_VERSION;
//    std::cout << "Thrust v" << major << "." << minor << std::endl;
//
//    return 0;
//}