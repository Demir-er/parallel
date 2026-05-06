// File: simulate_gpu.cu

#include "simulate_gpu.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 
#include <curand_kernel.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/reduce.h>
#include <thrust/count.h> 
#include <thrust/execution_policy.h>
#include <thrust/transform_reduce.h>
#include <cstdio>

// Macro for early return on CUDA errors
#define CUDA_CHECK_RET(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
    printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); return false; } } while(0)

// Define maximum asset limit to prevent register spilling 
#define MAX_ASSETS 256 

__global__ void generate_portfolio_pl_kernel(std::size_t N, int numAssets, const float* mu, const float* L, const float* weights, float* outPL, std::uint64_t seed) {
    std::size_t tid = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;

    // High-performance Philox generator setup
    curandStatePhilox4_32_10_t state;
    curand_init(seed, tid, 0, &state);

    for (std::size_t i = tid; i < N; i += stride) {
        float z[MAX_ASSETS];
        int limit = (numAssets < MAX_ASSETS) ? numAssets : MAX_ASSETS;

        // Generate random numbers from standard normal distribution for N assets
        for (int j = 0; j < limit; ++j) {
            z[j] = curand_normal(&state);
        }

        // Calculation of Portfolio Profit-Loss (P/L) Value (P = mu + L * Z)
        float pl = 0.0f;
        for (int j = 0; j < limit; ++j) {
            float val = mu[j];
            for (int k = 0; k <= j; ++k) {
                val += L[j * numAssets + k] * z[k];
            }
            pl += weights[j] * val;
        }
        outPL[i] = pl;
    }
}

// Functor for CVaR filtering
struct LeQValue {
    float q;
    __host__ __device__ LeQValue(float _q = 0.0f) : q(_q) {}
    __host__ __device__ float operator()(const float& x) const { return (x <= q) ? x : 0.0f; }
};

// Functor to check if the scenario is profitable
struct IsPositive {
    __host__ __device__ bool operator()(const float& x) const { return x > 0.0f; }
};

extern "C" bool simulateReturnsGPU(std::size_t N, int numAssets, const float* mu, const float* L, const float* weights, std::uint64_t seed, float alpha, float* outVaR, float* outCVaR, float* outExpectedReturn, float* outProbProfit, float* outOptimistic, float* outKernelMs) {
    if (N == 0 || !mu || !L || !weights || !outVaR || !outCVaR) return false;
    if (!(alpha > 0.0f && alpha < 1.0f)) return false;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // GPU Memory Allocation for dynamic N-Dimensional arrays
    float* d_mu, * d_L, * d_weights, * d_pl;
    CUDA_CHECK_RET(cudaMalloc(&d_mu, numAssets * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&d_L, numAssets * numAssets * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&d_weights, numAssets * sizeof(float)));
    CUDA_CHECK_RET(cudaMalloc(&d_pl, N * sizeof(float)));

    // Copy parameters from CPU to GPU
    CUDA_CHECK_RET(cudaMemcpy(d_mu, mu, numAssets * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RET(cudaMemcpy(d_L, L, numAssets * numAssets * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_RET(cudaMemcpy(d_weights, weights, numAssets * sizeof(float), cudaMemcpyHostToDevice));

    cudaEventRecord(start);

    // Kernel Launch Parameters
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    if (numBlocks > 1024) numBlocks = 1024;

    generate_portfolio_pl_kernel << <numBlocks, blockSize >> > (N, numAssets, d_mu, d_L, d_weights, d_pl, seed);
    CUDA_CHECK_RET(cudaGetLastError());

    // Sort the P/L values using Thrust
    thrust::device_ptr<float> dev_ptr(d_pl);
    thrust::sort(thrust::device, dev_ptr, dev_ptr + N);

    // 1. VaR & CVaR Calculations (Risk/Loss)
    double tailFrac = 1.0 - static_cast<double>(alpha);
    std::size_t raw = static_cast<std::size_t>(tailFrac * static_cast<double>(N));
    std::size_t target_idx = (raw == 0) ? 0 : ((raw - 1 >= N) ? (N - 1) : (raw - 1));

    float vaR_val = 0.0f;
    CUDA_CHECK_RET(cudaMemcpy(&vaR_val, d_pl + target_idx, sizeof(float), cudaMemcpyDeviceToHost));
    *outVaR = -vaR_val;

    float sumCVaR = thrust::transform_reduce(thrust::device, dev_ptr, dev_ptr + target_idx + 1, LeQValue(vaR_val), 0.0f, thrust::plus<float>());
    *outCVaR = -(sumCVaR / (target_idx + 1));

    // 2. Expected Return (Mean of all Scenarios)
    float sumAll = thrust::reduce(thrust::device, dev_ptr, dev_ptr + N, 0.0f, thrust::plus<float>());
    *outExpectedReturn = sumAll / static_cast<float>(N);

    // 3. Probability of Profit (Win Rate)
    int positiveCount = thrust::count_if(thrust::device, dev_ptr, dev_ptr + N, IsPositive());
    *outProbProfit = (static_cast<float>(positiveCount) / static_cast<float>(N)) * 100.0f;

    // 4. Optimistic Scenario (Best tail boundary)
    std::size_t opt_raw = static_cast<std::size_t>(static_cast<double>(alpha) * static_cast<double>(N));
    std::size_t opt_idx = (opt_raw == 0) ? 0 : ((opt_raw >= N) ? (N - 1) : opt_raw);
    float opt_val = 0.0f;
    CUDA_CHECK_RET(cudaMemcpy(&opt_val, d_pl + opt_idx, sizeof(float), cudaMemcpyDeviceToHost));
    *outOptimistic = opt_val;

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(outKernelMs, start, stop);

    // Cleanup
    cudaFree(d_mu);
    cudaFree(d_L);
    cudaFree(d_weights);
    cudaFree(d_pl);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return true;
}