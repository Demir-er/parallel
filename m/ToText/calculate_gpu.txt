#include "calculate_gpu.h"
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include <cuda_runtime.h>

// GPU version of calculateVaR
float calculateVaRGPU(float* d_returns, int numPaths, float confidenceLevel) {
    // Create a Thrust device vector from raw device pointer
    thrust::device_ptr<float> d_ptr = thrust::device_pointer_cast(d_returns);

    // Sort the device vector using the device execution policy
    thrust::sort(thrust::device, d_ptr, d_ptr + numPaths);

    // Calculate index for confidence level
    int index = static_cast<int>((1.0 - confidenceLevel) * numPaths);

    // Retrieve the value at the index
    return -d_ptr[index];
}
