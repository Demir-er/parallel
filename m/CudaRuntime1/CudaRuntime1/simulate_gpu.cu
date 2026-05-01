#include "simulate_gpu.h"

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/count.h>
#include <thrust/transform_reduce.h>
#include <cstdio>
#include <ctime>
#include <cmath>

#if __has_include(<thrust/nth_element.h>)
  #include <thrust/nth_element.h>
  #define HAVE_THRUST_NTH_ELEMENT 1
#else
  #define HAVE_THRUST_NTH_ELEMENT 0
#endif

// Macro for early return on CUDA errors
#define CUDA_CHECK_RET(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return false; } } while(0)

// Kernel: generate portfolio P/L per sample using per-thread curand (Philox)
__global__ static void generate_portfolio_pl_kernel(std::size_t N,
                                                    float mu0, float mu1,
                                                    float L00, float /*L01*/, float L10, float L11,
                                                    float w0, float w1,
                                                    float* outPL,
                                                    std::uint64_t seed)
{
    std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    curandStatePhilox4_32_10_t state;
    curand_init(seed, static_cast<unsigned long long>(tid), 0, &state);

    for (std::size_t i = tid; i < N; i += stride) {
        float z0 = curand_normal(&state);
        float z1 = curand_normal(&state);
        float x0 = mu0 + L00 * z0;
        float x1 = mu1 + L10 * z0 + L11 * z1;
        outPL[i] = w0 * x0 + w1 * x1;
    }
}

// Functor: return value if <= q, else 0 (for transform_reduce)
struct LeQValue {
    float q;
    __host__ __device__ LeQValue(float _q=0.0f) : q(_q) {}
    __host__ __device__ float operator()(const float &x) const {
        return (x <= q) ? x : 0.0f;
    }
};

// Predicate: x <= q
struct LeQPred {
    float q;
    __host__ __device__ LeQPred(float _q=0.0f) : q(_q) {}
    __host__ __device__ bool operator()(const float &x) const {
        return x <= q;
    }
};

extern "C" bool simulateReturnsGPU(std::size_t N,
                                   const float mu[2],
                                   const float L[4],
                                   const float weights[2],
                                   std::uint64_t seed,
                                   float alpha,
                                   float* outVaR,
                                   float* outCVaR,
                                   float* outKernelMs)
{
    if (N == 0 || !mu || !L || !weights || !outVaR || !outCVaR) return false;
    if (!(alpha > 0.0f && alpha < 1.0f)) return false;

    // Allocate device array for portfolio P/L
    float* d_pl = nullptr;
    CUDA_CHECK_RET(cudaMalloc(reinterpret_cast<void**>(&d_pl), sizeof(float) * N));

    // Launch config
    constexpr int block = 256;
    long long grid_ll = (static_cast<long long>(N) + block - 1) / block;
    int grid = static_cast<int>(grid_ll > 65535 ? 65535 : grid_ll);

    // Events for timing kernel, selection/sort and total
    cudaEvent_t evKernelStart, evKernelStop, evSelectStart, evSelectStop;
    CUDA_CHECK_RET(cudaEventCreate(&evKernelStart));
    CUDA_CHECK_RET(cudaEventCreate(&evKernelStop));
    CUDA_CHECK_RET(cudaEventCreate(&evSelectStart));
    CUDA_CHECK_RET(cudaEventCreate(&evSelectStop));

    std::uint64_t useSeed = seed ? seed : static_cast<std::uint64_t>(time(nullptr));

    // Launch generation kernel and time it
    CUDA_CHECK_RET(cudaEventRecord(evKernelStart, 0));
    generate_portfolio_pl_kernel<<<grid, block>>>(N,
                                                  mu[0], mu[1],
                                                  L[0], L[1], L[2], L[3],
                                                  weights[0], weights[1],
                                                  d_pl,
                                                  useSeed);
    CUDA_CHECK_RET(cudaGetLastError());
    CUDA_CHECK_RET(cudaEventRecord(evKernelStop, 0));
    CUDA_CHECK_RET(cudaEventSynchronize(evKernelStop));

    float kernelMs = 0.0f;
    CUDA_CHECK_RET(cudaEventElapsedTime(&kernelMs, evKernelStart, evKernelStop));

    // Selection / sort phase: find quantile element
    thrust::device_ptr<float> dev_ptr(d_pl);

    // determine index for lower tail (1 - alpha)
    double tailFrac = 1.0 - static_cast<double>(alpha);
    std::size_t raw = static_cast<std::size_t>(tailFrac * static_cast<double>(N));
    std::size_t idx = (raw == 0) ? 0 : ((raw - 1 >= N) ? (N - 1) : raw - 1);

    CUDA_CHECK_RET(cudaEventRecord(evSelectStart, 0));

#if HAVE_THRUST_NTH_ELEMENT
    // Use nth_element if available (expected linear time)
    thrust::nth_element(thrust::device, dev_ptr, dev_ptr + idx, dev_ptr + N);
#else
    // Fallback: full sort (portable). Slower, but correct.
    thrust::sort(thrust::device, dev_ptr, dev_ptr + N);
#endif

    CUDA_CHECK_RET(cudaEventRecord(evSelectStop, 0));
    CUDA_CHECK_RET(cudaEventSynchronize(evSelectStop));

    float selectMs = 0.0f;
    CUDA_CHECK_RET(cudaEventElapsedTime(&selectMs, evSelectStart, evSelectStop));

    // Copy VaR value from device
    CUDA_CHECK_RET(cudaMemcpy(outVaR, d_pl + idx, sizeof(float), cudaMemcpyDeviceToHost));

    // Compute CVaR = mean of values <= q on device
    float q = *outVaR;
    float sum = thrust::transform_reduce(thrust::device, dev_ptr, dev_ptr + N, LeQValue(q), 0.0f, thrust::plus<float>());
    std::size_t count = thrust::count_if(thrust::device, dev_ptr, dev_ptr + N, LeQPred(q));

    if (count == 0) {
        *outCVaR = q;
    } else {
        *outCVaR = sum / static_cast<float>(count);
    }

    // total ms: kernel + select
    float totalMs = kernelMs + selectMs;
    if (outKernelMs) *outKernelMs = totalMs;

    // Optional verbose prints for diagnostics (can be removed)
    // printf("simulateReturnsGPU: kernel ms=%.3f, select ms=%.3f, total ms=%.3f\n", kernelMs, selectMs, totalMs);

    // Cleanup
    CUDA_CHECK_RET(cudaFree(d_pl));
    CUDA_CHECK_RET(cudaEventDestroy(evKernelStart));
    CUDA_CHECK_RET(cudaEventDestroy(evKernelStop));
    CUDA_CHECK_RET(cudaEventDestroy(evSelectStart));
    CUDA_CHECK_RET(cudaEventDestroy(evSelectStop));
    return true;
}
