#include <iostream>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>

// Include your project headers
 #include "simulate_cpu.h"
 #include "simulate_gpu.h"
 #include "calculate_cpu.h"
 #include "calculate_gpu.h"
 #include "output.h"

// 1. Function to read Close prices from CSV (Handling quotes and comma decimals)
std::vector<float> readClosePrices(const std::string& filename) {
    std::vector<float> prices;
    std::ifstream file(filename);
    std::string line;

    // Check if the file is opened successfully
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return prices;
    }

    // Skip the header line (Date,Open,High,Low,Close,Volume)
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;

        // Parse the line using double quotes ('"') as the delimiter
        while (std::getline(ss, token, '"')) {
            tokens.push_back(token);
        }

        // When splitting by quotes, the "Close" price is typically at index 7
        // Example: 29.04.2024 18:10:00,"59,5","62,25","58,65","62,05",67954497
        if (tokens.size() >= 8) {
            std::string closeStr = tokens[1]; // FIXED: Added [1] to get the specific element

            // Replace Turkish decimal comma with dot (e.g., 62,05 -> 62.05)
            std::replace(closeStr.begin(), closeStr.end(), ',', '.');

            // Convert string to float and add to the vector
            prices.push_back(std::stof(closeStr));
        }
    }
    return prices;
}

// 2. Function to calculate daily returns: (P_t - P_{t-1}) / P_{t-1}
std::vector<float> calculateReturns(const std::vector<float>& prices) {
    std::vector<float> returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        returns.push_back((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
}

// 3. Function to calculate mean return (Mu)
float calculateMean(const std::vector<float>& returns) {
    float sum = 0.0f;
    for (float r : returns) {
        sum += r;
    }
    return sum / returns.size();
}

// 4. Function to calculate covariance between two assets
float calculateCovariance(const std::vector<float>& returnsA, const std::vector<float>& returnsB, float meanA, float meanB) {
    float cov = 0.0f;
    // Find the minimum size to prevent out-of-bounds errors if files differ slightly
    size_t minSize = std::min(returnsA.size(), returnsB.size());
    for (size_t i = 0; i < minSize; ++i) {
        cov += (returnsA[i] - meanA) * (returnsB[i] - meanB);
    }
    // Divide by (N-1) for sample covariance
    return cov / (minSize - 1);
}

// 5. Function for 2x2 Cholesky Decomposition
// FIXED: L is now a pointer to a float array of size 4
void cholesky2x2(float cov00, float cov01, float cov11, float* L) {
    L[0] = std::sqrt(cov00);                  // L00
    L[1] = 0.0f;                              // L01
    L[2] = cov01 / L[0];                      // L10
    L[3] = std::sqrt(cov11 - (L[2] * L[2]));  // L11
}

int main() {
    // --- STEP 1: READ DATA & PREPARE PORTFOLIO ON CPU ---

    // Read CSV files (Make sure "aselsan2year.csv" and "thy.csv" exist in your project folder)
    std::vector<float> aselsanPrices = readClosePrices("aselsan2year.csv");
    std::vector<float> thyPrices = readClosePrices("thy.csv"); // Change this to your second CSV file name

    if (aselsanPrices.empty() || thyPrices.empty()) {
        std::cerr << "Stopping program due to missing CSV files." << std::endl;
        return -1;
    }

    // Print reading test to verify comma-to-dot conversion
    std::cout << "--- CSV DATA READING TEST ---" << std::endl;
    for (int i = 0; i < 5 && i < aselsanPrices.size(); i++) {
        std::cout << "Day " << i + 1 << " Aselsan Close Price: " << aselsanPrices[i] << std::endl;
    }

    // Calculate returns
    std::vector<float> aselsanReturns = calculateReturns(aselsanPrices);
    std::vector<float> thyReturns = calculateReturns(thyPrices);

    // Calculate means (Mu)
    float meanAselsan = calculateMean(aselsanReturns);
    float meanThy = calculateMean(thyReturns);

    // 4. Calculate covariance matrix elements
    float cov00 = calculateCovariance(aselsanReturns, aselsanReturns, meanAselsan, meanAselsan); // Variance of Aselsan
    float cov01 = calculateCovariance(aselsanReturns, thyReturns, meanAselsan, meanThy);         // Covariance Aselsan-THY
    float cov11 = calculateCovariance(thyReturns, thyReturns, meanThy, meanThy);                 // Variance of THY

    // 5. Perform Cholesky Decomposition
    // FIXED: Declare L as an array of 4 elements to hold the 2x2 matrix
    float L[4];
    cholesky2x2(cov00, cov01, cov11, L);

    // Group means (Mu) into a single array for easier GPU transfer
    float muHost[5] = { meanAselsan, meanThy };

    // Print calculated portfolio metrics
    std::cout << "\n--- PORTFOLIO METRICS READY ---" << std::endl;
    std::cout << "Aselsan Mean Return (Mu): " << muHost << std::endl;
    std::cout << "THY Mean Return (Mu): " << muHost[6] << std::endl;
    std::cout << "Covariance (Aselsan-THY): " << cov01 << std::endl;
    std::cout << "Cholesky L00: " << L << std::endl;
    std::cout << "Cholesky L10: " << L[5] << std::endl;
    std::cout << "Cholesky L11: " << L[7] << std::endl;

    // --- STEP 2: ALLOCATE GPU MEMORY & LAUNCH SIMULATION ---
    std::cout << "\nReady for GPU Simulation. Allocating Device Memory..." << std::endl;

    // Declare device pointers
    float* d_mu, * d_L;

    // Allocate memory on the GPU using cudaMalloc
    cudaMalloc((void**)&d_mu, 2 * sizeof(float));
    cudaMalloc((void**)&d_L, 4 * sizeof(float));

    // Transfer calculated Mu and Cholesky Matrix (L) from CPU (Host) to GPU (Device)
    cudaMemcpy(d_mu, muHost, 2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_L, L, 4 * sizeof(float), cudaMemcpyHostToDevice);

    std::cout << "Data successfully copied to GPU!" << std::endl;

    /*
       TODO Next:
       1. Call the updated simulateReturnsGPU kernel to run Monte Carlo.
       2. Use cudaFree to release d_mu and d_L when simulation is done.
    */

    return 0;
}