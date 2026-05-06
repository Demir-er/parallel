// File: simulate_gpu.h
#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

    bool simulateReturnsGPU(std::size_t N,
        int numAssets,
        const float* mu,
        const float* L,
        const float* weights,
        std::uint64_t seed,
        float alpha,
        float* outVaR,
        float* outCVaR,
        float* outExpectedReturn,
        float* outProbProfit,
        float* outOptimistic,
        float* outKernelMs);

#ifdef __cplusplus
}
#endif