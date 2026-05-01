#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

bool simulateReturnsGPU(std::size_t N,
                        const float mu[2],
                        const float L[4],
                        const float weights[2],
                        std::uint64_t seed,
                        float alpha,
                        float* outVaR,
                        float* outCVaR,
                        float* outKernelMs);

#ifdef __cplusplus
}
#endif
