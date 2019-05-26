#include <stdio.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;

__global__ void GaussJordanStep1(float* A, float* b, int size, int i)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row > i)
    {
        float ratio = A[size * row + i] / A[size * i + i];
        A[size * row + col] -= A[size * i + col] * ratio;
        if (col == 0)
            b[row] -= b[i] * ratio;
    } // end if (row>i)
}

__global__ void GaussJordanStep2(float* A, float* b, int size, int i)
{
    int row = blockIdx.x;
    int col = threadIdx.x;

    if (row < i)
    {
        float ratio = A[size * row + i] / A[size * i + i];
        A[size * row + col] -= A[size * i + col] * ratio;
        if (col == 0)
            b[row] -= b[i] * ratio;
    } // end if (row>i)
}


void savetofile(float* A, string s, int height, int width)
{
    std::ofstream plik;
    plik.open(s.c_str());

    for (int j = 0; j < height; j++)
    { // row
        for (int i = 0; i < width; i++)
        { // column
            plik << A[j * width + i] << "\t";
        }
        plik << endl;
    }
    plik.close();
}


int main()
{
    int n = 3;
    // creating input
    float* A = new float[n * n];
    float* b = new float[n];

    for (int i = 0; i < n; i++) // row
        for (int j = 0; j < n; j++) // column
            A[i * n + j] = (float)rand() / RAND_MAX + 0.5; // zero handling is not implemented...

    savetofile(A, "A.txt", n, n);


    for (int i = 0; i < n; i++) // row
        b[i] = (float)rand() / RAND_MAX + 0.5;

    savetofile(b, "b.txt", n, 1);


    // For time handling
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaError_t err;


    cudaEventRecord(start, 0);

    // Copy to memory
    float *gpuA, *gpuB;


    // memory allocation
    err = cudaMalloc((void**)&gpuA, sizeof(float) * n * n);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    err = cudaMalloc((void**)&gpuB, sizeof(float) * n);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }


    // copy data from GPU to CPU
    err = cudaMemcpy(gpuA, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    err = cudaMemcpy(gpuB, b, sizeof(float) * n, cudaMemcpyHostToDevice);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }


    for (int i = 0; i < (n - 1); i++)
        GaussJordanStep1<<<n, n>>>(gpuA, gpuB, n, i);
    for (int i = n - 1; i > 0; i--)
        GaussJordanStep2<<<n, n>>>(gpuA, gpuB, n, i);

    err = cudaMemcpy(A, gpuA, sizeof(float) * n * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    err = cudaMemcpy(b, gpuB, sizeof(float) * n, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess)
    {
        cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
    }

    cudaFree(gpuA);
    cudaFree(gpuB);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << "Cuda Time - linear solver: " << time << "ms\n";

    savetofile(b, "resB.txt", n, 1);
    savetofile(A, "resA.txt", n, n);


    for (int i = 0; i < n; i++)
        b[i] /= A[i * n + i];
    savetofile(b, "res.txt", n, 1);

    delete[] b;
    delete[] A;
}
