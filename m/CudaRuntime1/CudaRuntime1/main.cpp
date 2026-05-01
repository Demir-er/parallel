// CudaRuntime1\main.cpp
// Enhanced: Logger, date alignment, CPU/GPU timing scaffolding, and robust CUDA error checks.
// Targets C++17.

#include <iostream>
#include <array>
#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cmath>
#include <chrono>
#include <random>
#include <iomanip>
#include <ctime>
#include <unordered_map>
#include <mutex>

#include <cuda_runtime.h> // CUDA runtime

// Include your project headers
#include "simulate_cpu.h"
#include "simulate_gpu.h"
#include "calculate_cpu.h"
#include "calculate_gpu.h"
#include "output.h"

// Simple Logger that writes to both stdout and an append log file with timestamps.
// Thread-safe for future extension.
class Logger {
    std::ofstream file_;
    std::mutex mtx_;
    bool fileOk_ = false;

    static std::string nowTimestamp() {
        auto t = std::chrono::system_clock::now();
        std::time_t tt = std::chrono::system_clock::to_time_t(t);
        std::tm tm{};
#if defined(_MSC_VER)
        localtime_s(&tm, &tt);
#else
        localtime_r(&tt, &tm);
#endif
        std::ostringstream oss;
        oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
        return oss.str();
    }

public:
    Logger(const std::string& filename = "results.txt") {
        file_.open(filename, std::ios::app);
        fileOk_ = file_.good();
        // Add run header
        std::ostringstream oss;
        oss << "----- Run at " << nowTimestamp() << " -----\n";
        if (fileOk_) file_ << oss.str();
        std::cout << oss.str();
    }

    ~Logger() {
        std::ostringstream oss;
        oss << "----- End run -----\n\n";
        if (fileOk_) file_ << oss.str();
        std::cout << oss.str();
        if (fileOk_) file_.flush();
    }

    template<typename... Args>
    void info(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        std::string s = oss.str();
        std::lock_guard<std::mutex> lk(mtx_);
        std::cout << s;
        if (fileOk_) file_ << nowTimestamp() << " INFO: " << s;
    }

    template<typename... Args>
    void infoLine(Args&&... args) {
        info(std::forward<Args>(args)..., "\n");
    }

    template<typename... Args>
    void errorLine(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        std::string s = oss.str();
        std::lock_guard<std::mutex> lk(mtx_);
        std::cerr << s << std::endl;
        if (fileOk_) file_ << nowTimestamp() << " ERROR: " << s << "\n";
    }
};

// CUDA error checking helper
inline bool checkCuda(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

// Read Close prices and dates from CSV (handles quoted fields and comma decimals)
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
    if (!std::getline(file, line)) return false;

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

// Calculate daily returns: (P_t - P_{t-1}) / P_{t-1}
std::vector<float> calculateReturns(const std::vector<float>& prices) {
    std::vector<float> returns;
    if (prices.size() < 2) return returns;
    for (size_t i = 1; i < prices.size(); ++i) {
        returns.push_back((prices[i] - prices[i - 1]) / prices[i - 1]);
    }
    return returns;
}

// Mean of returns (guard empty)
float calculateMean(const std::vector<float>& returns) {
    if (returns.empty()) return 0.0f;
    float sum = 0.0f;
    for (float r : returns) sum += r;
    return sum / returns.size();
}

// Sample covariance (returnsA and returnsB must be same length; uses N-1)
float calculateCovariance(const std::vector<float>& returnsA,
                          const std::vector<float>& returnsB,
                          float meanA, float meanB) {
    size_t n = std::min(returnsA.size(), returnsB.size());
    if (n < 2) return 0.0f;
    float cov = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cov += (returnsA[i] - meanA) * (returnsB[i] - meanB);
    }
    return cov / (static_cast<float>(n) - 1.0f);
}

// 2x2 Cholesky with jitter/diagnostics. L output arranged row-major lower-triangular:
// L[0] = L00, L[1] = L01 (should be 0), L[2] = L10, L[3] = L11
// Returns true if successful (possibly after jitter), false otherwise.
bool cholesky2x2(float cov00, float cov01, float cov11, float* L, float jitter = 1e-10f) {
    // Ensure diagonal non-negative
    if (cov00 <= 0.0f) {
        if (cov00 > -jitter) cov00 = 0.0f; else return false;
    }
    L[0] = std::sqrt(std::max(0.0f, cov00));
    L[1] = 0.0f;

    if (L[0] == 0.0f) {
        // If L00 is zero, cov01 must be zero too for valid factorization
        if (std::fabs(cov01) > 1e-12f) {
            // Try regularizing diagonal
            cov00 += jitter;
            L[0] = std::sqrt(cov00);
            if (L[0] == 0.0f) return false;
        } else {
            L[2] = 0.0f;
            L[3] = std::sqrt(std::max(0.0f, cov11));
            return true;
        }
    }

    L[2] = cov01 / L[0];

    float tmp = cov11 - (L[2] * L[2]);
    if (tmp < 0.0f) {
        // If tiny negative due to rounding, clamp to zero; otherwise try jitter
        if (tmp > -1e-8f) tmp = 0.0f;
        else {
            // try adding jitter to cov11
            cov11 += jitter;
            tmp = cov11 - (L[2] * L[2]);
            if (tmp < 0.0f) return false;
        }
    }
    L[3] = std::sqrt(tmp);
    return true;
}

// Generate N correlated samples on CPU using mu (2) and lower-triangular L (2x2)
// Returns pair: generated samples (vector of pairs) and boolean success
bool generateCorrelatedSamplesCPU(size_t N, const float mu[2], const float L[4], std::vector<std::array<float,2>>& outSamples) {
    outSamples.clear();
    outSamples.reserve(N);

    std::mt19937_64 rng((unsigned)std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        float z0 = nd(rng);
        float z1 = nd(rng);
        // x = mu + L * z  (lower-triangular multiplication)
        float x0 = mu[0] + L[0] * z0;
        float x1 = mu[1] + L[2] * z0 + L[3] * z1;
        outSamples.push_back({x0, x1});
    }
    return true;
}

// Compute sample mean and covariance of generated samples
void computeSampleStats(const std::vector<std::array<float,2>>& samples,
                        float& mean0, float& mean1,
                        float& cov00, float& cov01, float& cov11) {
    mean0 = mean1 = cov00 = cov01 = cov11 = 0.0f;
    size_t n = samples.size();
    if (n == 0) return;

    for (const auto &s : samples) {
        mean0 += s[0];
        mean1 += s[1];
    }
    mean0 /= n;
    mean1 /= n;

    if (n < 2) return;
    for (const auto &s : samples) {
        float a = s[0] - mean0;
        float b = s[1] - mean1;
        cov00 += a * a;
        cov01 += a * b;
        cov11 += b * b;
    }
    float denom = static_cast<float>(n - 1);
    cov00 /= denom;
    cov01 /= denom;
    cov11 /= denom;
}

// Align two series by date (inner join). Preserves order of the first series.
void alignByDate(const std::vector<std::string>& datesA, const std::vector<float>& pricesA,
                 const std::vector<std::string>& datesB, const std::vector<float>& pricesB,
                 std::vector<float>& outA, std::vector<float>& outB, std::vector<std::string>& outDates) {
    outA.clear();
    outB.clear();
    outDates.clear();
    std::unordered_map<std::string, float> mapB;
    mapB.reserve(datesB.size());
    for (size_t i = 0; i < datesB.size(); ++i) {
        mapB[datesB[i]] = pricesB[i];
    }
    for (size_t i = 0; i < datesA.size(); ++i) {
        auto it = mapB.find(datesA[i]);
        if (it != mapB.end()) {
            outA.push_back(pricesA[i]);
            outB.push_back(it->second);
            outDates.push_back(datesA[i]);
        }
    }
}

int main() {
    Logger logger("CudaRuntime1_results.txt");

    // --- READ CSV DATA ---
    std::vector<float> aselsanPrices;
    std::vector<std::string> aselsanDates;
    std::vector<float> thyPrices;
    std::vector<std::string> thyDates;

    logger.infoLine("Reading CSV files...");

    if (!readClosePrices("aselsan2year.csv", aselsanPrices, aselsanDates)) {
        logger.errorLine("Stopping: could not read aselsan2year.csv");
        return -1;
    }
    if (!readClosePrices("thy.csv", thyPrices, thyDates)) {
        logger.errorLine("Stopping: could not read thy.csv");
        return -1;
    }

    logger.infoLine("Rows read: Aselsan=", aselsanPrices.size(), ", THY=", thyPrices.size());

    // --- ALIGN BY DATE ---
    std::vector<float> aPricesAligned, tPricesAligned;
    std::vector<std::string> alignedDates;
    alignByDate(aselsanDates, aselsanPrices, thyDates, thyPrices, aPricesAligned, tPricesAligned, alignedDates);
    logger.infoLine("Aligned rows: ", alignedDates.size());

    if (alignedDates.size() < 2) {
        logger.errorLine("Not enough aligned data to proceed.");
        return -1;
    }

    // Print first/last few rows (logged)
    const size_t printN = 5;
    logger.infoLine("\n--- ALIGNED CSV DATA SAMPLE ---");
    logger.infoLine("Aselsan - First rows:");
    for (size_t i = 0; i < printN && i < aPricesAligned.size(); ++i) {
        logger.infoLine(alignedDates[i], " Aselsan Close Price: ", aPricesAligned[i]);
    }
    logger.infoLine("Aselsan - Last rows:");
    size_t startA = (aPricesAligned.size() > printN) ? (aPricesAligned.size() - printN) : 0;
    for (size_t i = startA; i < aPricesAligned.size(); ++i) {
        logger.infoLine(alignedDates[i], " Aselsan Close Price: ", aPricesAligned[i]);
    }

    logger.infoLine("THY - First rows:");
    for (size_t i = 0; i < printN && i < tPricesAligned.size(); ++i) {
        logger.infoLine(alignedDates[i], " THY Close Price: ", tPricesAligned[i]);
    }
    logger.infoLine("THY - Last rows:");
    size_t startT = (tPricesAligned.size() > printN) ? (tPricesAligned.size() - printN) : 0;
    for (size_t i = startT; i < tPricesAligned.size(); ++i) {
        logger.infoLine(alignedDates[i], " THY Close Price: ", tPricesAligned[i]);
    }

    // --- COMPUTE RETURNS & COVARIANCE ---
    auto aselsanReturns = calculateReturns(aPricesAligned);
    auto thyReturns = calculateReturns(tPricesAligned);

    if (aselsanReturns.empty() || thyReturns.empty()) {
        logger.errorLine("Not enough data to compute returns.");
        return -1;
    }

    float meanAselsan = calculateMean(aselsanReturns);
    float meanThy = calculateMean(thyReturns);
    float cov00 = calculateCovariance(aselsanReturns, aselsanReturns, meanAselsan, meanAselsan);
    float cov01 = calculateCovariance(aselsanReturns, thyReturns, meanAselsan, meanThy);
    float cov11 = calculateCovariance(thyReturns, thyReturns, meanThy, meanThy);

    logger.infoLine("\nComputed sample covariance matrix (from returns):");
    logger.infoLine("[ ", cov00, "  ", cov01, " ]");
    logger.infoLine("[ ", cov01, "  ", cov11, " ]");

    // --- CHOLESKY ---
    float L[4] = {0,0,0,0};
    bool ok = cholesky2x2(cov00, cov01, cov11, L);
    if (!ok) {
        logger.errorLine("Cholesky factorization failed for covariance matrix. Consider increasing regularization.");
        return -1;
    }
    logger.infoLine("\nCholesky L (lower-triangular):");
    logger.infoLine(L[0], "  0");
    logger.infoLine(L[2], "  ", L[3]);

    // --- CPU SAMPLING TO VERIFY L ---
    float mu[2] = { meanAselsan, meanThy };
    constexpr size_t Nsamples = 20000;
    std::vector<std::array<float,2>> samples;

    logger.infoLine("\nGenerating samples on CPU (N=", Nsamples, ")...");
    auto tStartCPU = std::chrono::high_resolution_clock::now();
    generateCorrelatedSamplesCPU(Nsamples, mu, L, samples);
    auto tEndCPU = std::chrono::high_resolution_clock::now();
    auto cpuMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(tEndCPU - tStartCPU).count();

    logger.infoLine("CPU generation time: ", cpuMs, " ms (", (Nsamples / (cpuMs/1000.0)), " samples/sec)");

    float sMean0, sMean1, sCov00, sCov01, sCov11;
    computeSampleStats(samples, sMean0, sMean1, sCov00, sCov01, sCov11);

    logger.infoLine("\nVerification from generated samples (", Nsamples, "):");
    logger.infoLine("Sample means: [", sMean0, ", ", sMean1, "]");
    logger.infoLine("Sample covariance:");
    logger.infoLine("[ ", sCov00, "  ", sCov01, " ]");
    logger.infoLine("[ ", sCov01, "  ", sCov11, " ]");

    logger.infoLine("\nOriginal estimated covariance vs sample covariance difference (orig - sample):");
    logger.infoLine("[ ", (cov00 - sCov00), "  ", (cov01 - sCov01), " ]");
    logger.infoLine("[ ", (cov01 - sCov01), "  ", (cov11 - sCov11), " ]");

    // --- GPU PREP & TIMING (basic) ---
    logger.infoLine("\nReady for GPU Simulation. Allocating Device Memory and copying parameters...");

    float muHost[2] = { meanAselsan, meanThy };
    float* d_mu = nullptr;
    float* d_L = nullptr;

    cudaError_t err;

    // Time allocation & copy using CUDA events
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    err = cudaEventRecord(startEvent, 0);
    if (err != cudaSuccess) logger.errorLine("cudaEventRecord failed start: ", cudaGetErrorString(err));

    err = cudaMalloc((void**)&d_mu, 2 * sizeof(float));
    if (!checkCuda(err, "cudaMalloc d_mu")) return -1;
    err = cudaMalloc((void**)&d_L, 4 * sizeof(float));
    if (!checkCuda(err, "cudaMalloc d_L")) {
        cudaFree(d_mu);
        return -1;
    }

    err = cudaMemcpy(d_mu, muHost, 2 * sizeof(float), cudaMemcpyHostToDevice);
    if (!checkCuda(err, "cudaMemcpy d_mu")) {
        cudaFree(d_mu); cudaFree(d_L); return -1;
    }
    err = cudaMemcpy(d_L, L, 4 * sizeof(float), cudaMemcpyHostToDevice);
    if (!checkCuda(err, "cudaMemcpy d_L")) {
        cudaFree(d_mu); cudaFree(d_L); return -1;
    }

    err = cudaEventRecord(stopEvent, 0);
    if (err != cudaSuccess) logger.errorLine("cudaEventRecord failed stop: ", cudaGetErrorString(err));
    err = cudaEventSynchronize(stopEvent);
    if (err != cudaSuccess) logger.errorLine("cudaEventSynchronize failed: ", cudaGetErrorString(err));

    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, startEvent, stopEvent);
    logger.infoLine("GPU alloc+copy time (measured by events): ", milliseconds, " ms");

    logger.infoLine("Data successfully copied to GPU!");

    // TODO: Launch GPU kernels (simulateReturnsGPU / calculateVaRGPU).
    // When implemented: use cudaEvent timing around kernel launches, repeat multiple runs, and compute medians.

    // Cleanup GPU memory
    cudaFree(d_mu);
    cudaFree(d_L);
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);

    logger.infoLine("\nSummary:");
    logger.infoLine(" - Aligned rows used: ", alignedDates.size());
    logger.infoLine(" - CPU sample generation time (ms): ", cpuMs);
    logger.infoLine(" - GPU parameter upload time (ms): ", milliseconds);

    // End main (Logger destructor writes run footer)
    return 0;
}