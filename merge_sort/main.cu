#include <iostream>
#include <vector>
#include <iterator>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <thread>
#include <cassert>

#pragma comment(lib, "cuda.lib")
#pragma comment(lib, "cudart.lib")
#include <cuda.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <cublas_v2.h>

using namespace std;
using Iterator = vector<float>::iterator;

// exclusive upper bound
//shifts the range [from..to) by one to the right
__host__ __device__
void shift(float v[], int from, int to)
{
  float prev = v[from];
  for (int i = from+1; i < to; i++)
  {
    float cur = v[i];
    v[i] = prev;
    prev = cur;
  }
}

// exclusive upper bound
__host__ __device__
void insertion_sort(float v[], int from, int to)
{
  // the ith element has to be inserted now
  for (int i = from + 1; i < to; i++)
  {
    for (int j = from; j < i; j++)
    {
      if (v[i] < v[j])
      {
        float to_insert = v[i];
        shift(v, j, i+1);

        v[j] = to_insert;
        break;
      }
    }
  }
}

// exclusive upper bounds
// Preconditions: both [from..sep) and [sep..to) are sorted ranges
__host__ __device__
void in_place_merge(float v[], int from, int sep, int to)
{
  // one of the ranges is empty
  if (sep - from <= 0 || to - sep <= 0)
  {
    return;
  }

  if (v[from] <= v[sep])
  {
    in_place_merge(v, from+1, sep, to);
  }
  else
  {
    float temp = v[sep];
    shift(v, from, sep + 1);
    v[from] = temp;
    in_place_merge(v, from + 1, sep + 1, to);
  }
}

void simple_merge(float v[], int from, int sep, int to)
{
  if (sep - from == 0 || to - sep == 0)
    return;

  vector<float> temp_v;
  std::copy(v + from, v + sep, back_inserter(temp_v));

  int i = from;
  int j = sep;

  int cur = from;

  // both ranges have at least one element
  while (sep - i > 0 && to - j > 0)
  {
    if (temp_v[i-from] <= v[j])
    {
      v[cur] = temp_v[i-from];
      i++;
    }
    else
    {
      v[cur] = v[j];
      j++;
    }
    cur++;
  }

  // cout << i - from << " " << sep << " " << cur << endl;
  std::copy(temp_v.begin() + i - from, temp_v.begin() + sep - from, v + cur);
}

// exclusive upper bound
void merge_sort_cpu_seq(float v[], int from, int to, int range = 5)
{
  // sort some short ranges using insertion sort
  for (int i = from; i < to; i += range)
  {
    int hi = min(i+range, to);
    insertion_sort(v, i, hi);
  }

  for (; range < to; range *= 2)
  {
    for (int i = from; i < to; i += 2*range)
    {
      int mid = min(i +   range, to);
      int hi  = min(i + 2*range, to);

      simple_merge(v, i, mid, hi);
    }
  }
}

// exclusive upper bound
void merge_sort_cpu_par(float v[], int from, int to, int range = 5)
{
  std::vector<std::thread> workers;

  // sort some short ranges using insertion sort
  for (int i = from; i < to; i += range)
  {
    auto f = [v, range, to](int i)
    {
      int hi = min(i+range, to);
      insertion_sort(v, i, hi);
    };
    workers.push_back(std::thread(f,i));
  }
  for (auto& w : workers)
    w.join();

  for (; range < to; range *= 2)
  {
    std::vector<std::thread> workers;
    for (int i = from; i < to; i += 2*range)
    {
      auto f = [range, to, v](int i)
      {
        int mid = min(i +   range, to);
        int hi  = min(i + 2*range, to);

        simple_merge(v, i, mid, hi);
      };
      workers.push_back(std::thread(f, i));
    }

    for (auto& w : workers)
      w.join();
  }
}

__global__
void isort_range(float v[], int range, int to)
{
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * range;
  int lo = min(i, to);
  int hi = min(lo + range, to);
  insertion_sort(v, lo, hi);
}

__global__
void merge_ranges(float v[], int range, int to)
{
  int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2*range;
  int lo  = min(i, to);
  int mid = min(lo +   range, to);
  int hi  = min(lo + 2*range, to);
  in_place_merge(v, lo, mid, hi);
}

__host__
float* merge_sort_pure_gpu(float v[], int from, int to, int range = 5, int range_limit = 512)
{
  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int n = to - from;

  int range_count = n % range == 0 ? n/range : n/range + 1;

  int thread_count = deviceProp.maxThreadsPerBlock;
  int block_count  = range_count % thread_count == 0 ? range_count/thread_count : range_count/thread_count + 1;

  // sort some short ranges using insertion sort
  isort_range<<<block_count, thread_count>>>(v, range, to);

  for (; range < to; range *= 2)
  {
    range_count = n % range == 0 ? n/(2*range) : n/(2*range) + 1;
    block_count = range_count % thread_count == 0 ? range_count/thread_count : range_count/thread_count + 1;
    merge_ranges<<<block_count, thread_count>>>(v, range, to);
  }

  // copy data from GPU to CPU
  float* v2 = new float[n];
  cudaError_t  err = cudaMemcpy(v2, v, sizeof(float) * n, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
      cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
  }

  return v2;
}

// exclusive upper bound
// range ~ initial range size
// range_limit ~ there must be at least this much GPU threads, or switches to CPU
// NOTE: (to - from) / range_limit = range' (ie.: range size at switch)
__host__
float* merge_sort_hybrid_seq(float v[], int from, int to, int range = 5, int range_limit = 512)
{
  // some crude heuristic
  if (range_limit == 0)
  {
    int maximal_workload_for_gpu_thread = 180;
    range_limit = (to - from) / maximal_workload_for_gpu_thread;
  }

  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int n = to - from;

  int range_count = n % range == 0 ? n/range : n/range + 1;

  int thread_count = deviceProp.maxThreadsPerBlock;
  int block_count  = range_count % thread_count == 0 ? range_count/thread_count : range_count/thread_count + 1;

  // sort some short ranges using insertion sort
  isort_range<<<block_count, thread_count>>>(v, range, to);

  for (; range < to && range_limit < range_count; range *= 2)
  {
    range_count = n % range == 0 ? n/(2*range) : n/(2*range) + 1;
    block_count = range_count % thread_count == 0 ? range_count/thread_count : range_count/thread_count + 1;
    merge_ranges<<<block_count, thread_count>>>(v, range, to);
  }

  // sequential from here (CPU)

  // copy data from GPU to CPU
  float* v2 = new float[n];
  cudaError_t  err = cudaMemcpy(v2, v, sizeof(float) * n, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
      cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
  }

  for (; range < to; range *= 2)
  {
    for (int i = from; i < to; i += 2*range)
    {
      int mid = min(i +   range, to);
      int hi  = min(i + 2*range, to);

      simple_merge(v2, i, mid, hi);
    }
  }

  return v2;
}

// exclusive upper bound
// range ~ initial range size
// range_limit ~ there must be at least this many GPU threads, or execution switches to CPU
// NOTE: (to - from) / range_limit = range' (ie.: range size at switch, minimal workload of a CPU thread)
__host__
float* merge_sort_hybrid_par(float v[], int from, int to, int range = 5, int range_limit = 0)
{
  // some crude heuristic
  if (range_limit == 0)
  {
    int minimal_workload_for_cpu_thread = 180;
    range_limit = (to - from) / minimal_workload_for_cpu_thread;
  }

  cudaSetDevice(0);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, 0);

  int n = to - from;

  int range_count = n % range == 0 ? n/range : n/range + 1;

  int thread_count = deviceProp.maxThreadsPerBlock;
  int block_count  = range_count % thread_count == 0 ? range_count/thread_count : range_count/thread_count + 1;

  // sort some short ranges using insertion sort
  isort_range<<<block_count, thread_count>>>(v, range, to);

  for (; range < to && range_limit < range_count; range *= 2)
  {
    range_count = n % range == 0 ? n/(2*range) : n/(2*range) + 1;
    block_count = range_count % thread_count == 0 ? range_count/thread_count : range_count/thread_count + 1;
    merge_ranges<<<block_count, thread_count>>>(v, range, to);
  }

  // sequential from here (CPU)

  // copy data from GPU to CPU
  float* v2 = new float[n];
  cudaError_t  err = cudaMemcpy(v2, v, sizeof(float) * n, cudaMemcpyDeviceToHost);
  if (err != cudaSuccess)
  {
      cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
  }

  for (; range < to; range *= 2)
  {
    std::vector<std::thread> workers;
    for (int i = from; i < to; i += 2*range)
    {
      auto f = [range, to, v2](int i)
      {
        int mid = min(i +   range, to);
        int hi  = min(i + 2*range, to);

        simple_merge(v2, i, mid, hi);
      };
      workers.push_back(std::thread(f, i));
    }

    for (auto& w : workers)
      w.join();
  }

  return v2;
}

// exclusive upper bound
__host__
bool is_sorted(float v[], int from, int to)
{
  for (int i = from; i < to - 1; i++)
  {
    if (v[i] > v[i + 1])
      return false;
  }
  return true;
}

template<typename F>
void execute_cpu_benchmark(F f, float v[], int n)
{
  float* vCPU = new float[n];
  std::copy(v, v + n, vCPU);

  auto start = std::chrono::steady_clock::now();

  f(vCPU, n);

  auto finish = std::chrono::steady_clock::now();
  double elapsed_ms =
    std::chrono::duration_cast< chrono::duration<double> >(finish - start).count() * 1000;

  cout << "Correctness: " << is_sorted(vCPU, 0, n) << endl;
  cout << "Time: " << elapsed_ms << endl;
  cout << "-------" << endl;
}

float* copy_to_gpu_mem(float* toGPU, int n)
{
  float* pGPU;
  cudaError_t err;
  err = cudaMalloc((void**)&pGPU, sizeof(float) * n);
  if (err != cudaSuccess)
  {
      cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
  }

  // copy data from CPU to GPU
  err = cudaMemcpy(pGPU, toGPU, sizeof(float) * n, cudaMemcpyHostToDevice);
  if (err != cudaSuccess)
  {
      cout << cudaGetErrorString(err) << " in " << __FILE__ << " at line " << __LINE__ << endl;
  }

  return pGPU;
}

template<typename F>
std::pair<float, float*> measure_gpu(F f, float vGPU[], int n)
{
  float time;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);
  float* result = f(vGPU, n);
  cudaEventRecord(stop, 0);

  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return {time, result};
}

template<typename F>
void execute_gpu_benchmark(F f, float v[], int n)
{
  float* vGPU = copy_to_gpu_mem(v, n);

  auto p = measure_gpu(f, vGPU, n);
  float time = p.first;
  float* vGPUResult = p.second;

  cudaFree(vGPU);

  cout << "Correctness: " << is_sorted(vGPUResult, 0, n) << endl;
  cout << "Time: " << time << endl;
  cout << "-------" << endl;
}

// ./main.out n b
// n ~ length of input
// b ~ 0 -> no presort, 1 -> presort input
int main(int argc, char** argv)
{
  // float v[] = {103, 1, 43, 7, 59, 3, 83, 19, 5, 2, 71, 23, 31, 13, 17, 19};
  // int n = sizeof(v)/sizeof(float);


  int n = argc >= 2 ? atoi(argv[1]) : 5000;
  float* v    = new float[n];
  for (int i = 0; i < n; i++)
  {
    v[i] = (float)rand() / RAND_MAX + 0.5;
  }

  if (argc == 3 && atoi(argv[2]) > 0)
  { //initial sort
    float* vGPU = copy_to_gpu_mem(v, n);
    v = merge_sort_hybrid_par(vGPU, 0, n, 5, 0);
    assert(is_sorted(v,0,n));
  }

  auto ms_cpu_seq = [](float* vCPU, int n)
  {
    merge_sort_cpu_seq(vCPU, 0, n);
  };
  cout << "CPU Sequential" << endl;
  execute_cpu_benchmark(ms_cpu_seq, v, n);

  auto ms_cpu_par = [](float* vCPU, int n)
  {
    merge_sort_cpu_par(vCPU, 0, n, 400);
  };
  cout << "CPU Parallel" << endl;
  execute_cpu_benchmark(ms_cpu_par, v, n);

  auto ms_pure_gpu = [](float* vGPU, int n)
  {
    return merge_sort_pure_gpu(vGPU, 0, n);
  };
  cout << "Pure GPU" << endl;
  execute_gpu_benchmark(ms_pure_gpu, v, n);

  auto ms_hseq = [](float* vGPU, int n)
  {
    return merge_sort_hybrid_seq(vGPU, 0, n, 5, 0);
  };
  cout << "Hybrid Sequential" << endl;
  execute_gpu_benchmark(ms_hseq, v, n);

  auto ms_hpar = [](float* vGPU, int n)
  {
    return merge_sort_hybrid_par(vGPU, 0, n, 5, 0);
  };
  cout << "Hybrid Parallel" << endl;
  execute_gpu_benchmark(ms_hpar, v, n);


  // std::copy(v, v + n, std::ostream_iterator<float>(std::cout, " "));
  // cout << std::endl;


  return 0;
}
