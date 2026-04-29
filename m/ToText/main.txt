#include <iostream>
#include <chrono>
#include <vector>
#include <fstream> // For file output
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "simulate_cpu.h"
#include "simulate_gpu.h"
#include "calculate_cpu.h"
#include "calculate_gpu.h"
#include "output.h"

int main() {
    // Define the list of numPaths values
    std::vector<int> numPathsList = {10000, 50000, 100000, 1000000, 10000000};
    float meanReturn = 0.0005;
    float stdDev = 0.02;
    float confidenceLevel = 0.95;

    // Open a file to store the results
    std::ofstream outFile("results.txt");
    if (!outFile) {
        std::cerr << "Error: Could not open results.txt for writing." << std::endl;
        return 1;
    }

    // Write the header to the file
    outFile << "numPaths,CPU_VaR,GPU_VaR,CPU_Time,GPU_Time\n";

    // Loop over each numPaths value
    for (int numPaths : numPathsList) {
        std::cout << "Running simulation for numPaths = " << numPaths << "...\n";

        // CPU simulation
        auto cpu_start = std::chrono::high_resolution_clock::now();
        std::vector<float> cpuReturns = simulateReturnsCPU(numPaths, meanReturn, stdDev);
        float cpuVaR = calculateVaR(cpuReturns, confidenceLevel);
        auto cpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> cpu_elapsed = cpu_end - cpu_start;

        // GPU simulation
        auto gpu_start = std::chrono::high_resolution_clock::now();

        // Allocate GPU memory
        float* d_returns;
        curandState* d_states;
        cudaMalloc(&d_returns, numPaths * sizeof(float));
        cudaMalloc(&d_states, numPaths * sizeof(curandState));

        // Perform GPU computation
        simulateReturnsGPU(numPaths, meanReturn, stdDev, d_returns, d_states);
        float gpuVaR = calculateVaRGPU(d_returns, numPaths, confidenceLevel);

        // Free GPU memory
        cudaFree(d_states);
        cudaFree(d_returns);

        auto gpu_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> gpu_elapsed = gpu_end - gpu_start;

        // Output results to the console
        outputResults(cpuVaR, gpuVaR, cpu_elapsed.count(), gpu_elapsed.count());

        // Save results to the file
        outFile << numPaths << "," << cpuVaR << "," << gpuVaR << "," << cpu_elapsed.count() << "," << gpu_elapsed.count() << "\n";
    }

    // Close the output file
    outFile.close();

    return 0;
}
