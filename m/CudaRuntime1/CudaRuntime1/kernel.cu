#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel (or fallback to CPU).        
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!\n");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
// If CUDA is unavailable or the driver is incompatible, perform the addition on CPU.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Verify there is at least one CUDA device and the driver is compatible.
    int deviceCount = 0;
    cudaStatus = cudaGetDeviceCount(&deviceCount);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaGetDeviceCount failed: %s\n", cudaGetErrorString(cudaStatus));
        fprintf(stderr, "Falling back to CPU implementation.\n");
        for (unsigned int i = 0; i < size; ++i) c[i] = a[i] + b[i];
        return cudaSuccess;
    }
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA-capable devices detected. Falling back to CPU.\n");
        for (unsigned int i = 0; i < size; ++i) c[i] = a[i] + b[i];
        return cudaSuccess;
    }

    // Choose which GPU to run on, change this on a multi-GPU system.
    int device = 0;
    cudaStatus = cudaSetDevice(device);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice(%d) failed: %s\n", device, cudaGetErrorString(cudaStatus));
        fprintf(stderr, "Falling back to CPU implementation.\n");
        for (unsigned int i = 0; i < size; ++i) c[i] = a[i] + b[i];
        return cudaSuccess;
    }

    // Print device name for diagnostics.
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, device) == cudaSuccess) {
        printf("Using device %d: %s\n", device, prop.name);
    }

    // Allocate GPU buffers for three vectors (two input, one output).
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_c failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_a failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc dev_b failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_a failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_b failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy dev_c->host failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // Reset device for profiling/tracing completeness. Preserve the CUDA error if it occurs.
    cudaError_t resetStatus = cudaDeviceReset();
    if (resetStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed: %s\n", cudaGetErrorString(resetStatus));
        // prefer returning the reset error to match prior behavior
        cudaStatus = resetStatus;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
