#include "output.h"
#include <iostream>

void outputResults(float cpuVaR, float gpuVaR, double cpuTime, double gpuTime) {
    std::cout << "CPU VaR (95% Confidence Level): " << cpuVaR << std::endl;
    std::cout << "GPU VaR (95% Confidence Level): " << gpuVaR << std::endl;

    std::cout << "CPU Execution Time: " << cpuTime << " seconds" << std::endl;
    std::cout << "GPU Execution Time: " << gpuTime << " seconds" << std::endl;
}
