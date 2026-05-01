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
bool readClosePrices(const std::string& filename,
                     std::vector<float>& prices,
                     std::vector<std::string>& dates)
{
    prices.clear();
    dates.clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        return false;
    }

    std::string line;
    // skip header
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::vector<std::string> fields;
        std::string cur;
        bool inQuotes = false;

        for (char c : line) {
            if (c == '"') {
                inQuotes = !inQuotes;
                continue;
            }
            if (c == ',' && !inQuotes) {
                fields.push_back(cur);
                cur.clear();
            } else {
                cur.push_back(c);
            }
        }
        fields.push_back(cur);

        // Expect header: Date,Open,High,Low,Close,Volume -> Close index = 4
        if (fields.size() <= 4) continue;

        std::string dateStr = fields[0];
        std::string closeStr = fields[4];

        // trim helpers
        auto trim = [](std::string &s) {
            size_t a = s.find_first_not_of(" \t\r\n");
            if (a == std::string::npos) { s.clear(); return; }
            size_t b = s.find_last_not_of(" \t\r\n");
            s = s.substr(a, b - a + 1);
        };
        trim(dateStr);
        trim(closeStr);

        // replace decimal comma with dot
        std::replace(closeStr.begin(), closeStr.end(), ',', '.');

        try {
            float val = std::stof(closeStr);
            dates.push_back(dateStr);
            prices.push_back(val);
        } catch (const std::exception&) {
            // skip malformed line
        }
    }

    return !prices.empty();
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
    if (returns.empty()) return 0.0f;
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
    if (minSize < 2) return 0.0f;
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
    L[3] = std::sqrt(std::max(0.0f, cov11 - (L[2] * L[2])));  // L11
}

int main() {
    // --- STEP 1: READ DATA & PREPARE PORTFOLIO ON CPU ---

    // Read CSV files
    std::vector<float> aselsanPrices;
    std::vector<std::string> aselsanDates;
    std::vector<float> thyPrices;
    std::vector<std::string> thyDates;

    if (!readClosePrices("aselsan2year.csv", aselsanPrices, aselsanDates) ||
        !readClosePrices("thy.csv", thyPrices, thyDates)) {
        std::cerr << "Stopping program due to missing or malformed CSV files." << std::endl;
        return -1;
    }

    // Print reading test to verify comma-to-dot conversion and show real dates
    std::cout << "--- CSV DATA READING TEST ---" << std::endl;

    const size_t printN = 5;

    // Aselsan: first up to printN
    std::cout << "\nAselsan - First " << std::min(printN, aselsanPrices.size()) << " rows:" << std::endl;
    for (size_t i = 0; i < printN && i < aselsanPrices.size(); ++i) {
        std::cout << aselsanDates[i] << " Aselsan Close Price: " << aselsanPrices[i] << std::endl;
    }

    // Aselsan: last up to printN
    std::cout << "\nAselsan - Last " << std::min(printN, aselsanPrices.size()) << " rows:" << std::endl;
    size_t startA = (aselsanPrices.size() > printN) ? (aselsanPrices.size() - printN) : 0;
    for (size_t i = startA; i < aselsanPrices.size(); ++i) {
        std::cout << aselsanDates[i] << " Aselsan Close Price: " << aselsanPrices[i] << std::endl;
    }

    // THY: first up to printN
    std::cout << "\nTHY - First " << std::min(printN, thyPrices.size()) << " rows:" << std::endl;
    for (size_t i = 0; i < printN && i < thyPrices.size(); ++i) {
        std::cout << thyDates[i] << " THY Close Price: " << thyPrices[i] << std::endl;
    }

    // THY: last up to printN
    std::cout << "\nTHY - Last " << std::min(printN, thyPrices.size()) << " rows:" << std::endl;
    size_t startT = (thyPrices.size() > printN) ? (thyPrices.size() - printN) : 0;
    for (size_t i = startT; i < thyPrices.size(); ++i) {
        std::cout << thyDates[i] << " THY Close Price: " << thyPrices[i] << std::endl;
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
    float muHost[2] = { meanAselsan, meanThy };

    // Print calculated portfolio metrics
    std::cout << "\n--- PORTFOLIO METRICS READY ---" << std::endl;
    std::cout << "Aselsan Mean Return (Mu): " << muHost[0] << std::endl;
    std::cout << "THY Mean Return (Mu): " << muHost[1] << std::endl;
    std::cout << "Covariance (Aselsan-THY): " << cov01 << std::endl;
    std::cout << "Cholesky L00: " << L[0] << std::endl;
    std::cout << "Cholesky L10: " << L[2] << std::endl;
    std::cout << "Cholesky L11: " << L[3] << std::endl;

    // --- STEP 2: ALLOCATE GPU MEMORY & LAUNCH SIMULATION ---
    std::cout << "\nReady for GPU Simulation. Allocating Device Memory..." << std::endl;

    // Declare device pointers
    float* d_mu = nullptr;
    float* d_L = nullptr;

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