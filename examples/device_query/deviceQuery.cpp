// std::system includes
#include <memory>
#include <iostream>
#include <stdio.h>

// CUDA-C includes
#include <cuda.h>
#include <cuda_runtime.h>

//#include <helper_cuda.h>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    // This function call returns 0 if there are no CUDA capable devices.
    if (deviceCount == 0)
    {
        printf("There are no available device(s) that support CUDA\n");
    }
    else
    {
        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
    }
	
    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;



        cudaGetDeviceProperties(&deviceProp, dev);

        printf("\nDevice %d: %s \n", dev, deviceProp.name);
        printf("\nMaxThreadsPerBlock: %d \n", deviceProp.maxThreadsPerBlock);
        printf("\nMaxThreadDim (%d,%d,%d)  \n", deviceProp.maxThreadsDim[0],deviceProp.maxThreadsDim[1],deviceProp.maxThreadsDim[2]);
        printf("\nMaxGridSize (%d,%d,%d)  \n", deviceProp.maxGridSize[0],deviceProp.maxGridSize[1],deviceProp.maxGridSize[2]);

    }

    // finish
    // cudaDeviceReset causes the driver to clean up all state. While 
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();

	char ch;
	std::cin >> ch;

    exit(EXIT_SUCCESS);
}
