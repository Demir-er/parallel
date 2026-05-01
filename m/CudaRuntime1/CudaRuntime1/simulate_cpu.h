#pragma once
#include <cstddef>
#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Simulate N correlated samples on GPU using mu (2) and lower-triangular L (4).
// Compute portfolio P/L = w0 * x0 + w1 * x1 per sample, then compute VaR and CVaR
// at confidence level alpha (e.g. 0.95).
//
// Returns true on success.
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
