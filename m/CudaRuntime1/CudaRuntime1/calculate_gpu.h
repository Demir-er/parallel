// File: simulate_gpu.h

#ifndef SIMULATE_GPU_H
#define SIMULATE_GPU_H

#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

    // GPU-based simulation function declaration (N-Dimensional with Profit Projections)
    bool simulateReturnsGPU(std::size_t N,
        int numAssets,
        const float* mu,
        const float* L,
        const float* weights,
        std::uint64_t seed,
        float alpha,
        float* outVaR,
        float* outCVaR,
        float* outExpectedReturn, // NEW: Mean return
        float* outProbProfit,     // NEW: Win rate percentage
        float* outOptimistic,     // NEW: Optimistic tail return
        float* outKernelMs);

#ifdef __cplusplus
}
#endif

#endif