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

#include <cuda_runtime.h>

#include "simulate_gpu.h"
#include "calculate_cpu.h"

// -----------------------------
// Operator overload to print std::vector
// -----------------------------
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i] << (i == v.size() - 1 ? "" : ", ");
    }
    os << "]";
    return os;
}

// -----------------------------
// Simple Logger
// -----------------------------
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
    explicit Logger(const std::string& filename = "CudaRuntime1_results.txt") {
        file_.open(filename, std::ios::app);
        fileOk_ = file_.good();
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
    void infoLine(Args&&... args) {
        std::ostringstream oss;
        (oss << ... << std::forward<Args>(args));
        std::string s = oss.str();
        std::lock_guard<std::mutex> lk(mtx_);
        std::cout << s << "\n";
        if (fileOk_) file_ << nowTimestamp() << " INFO: " << s << "\n";
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

// -----------------------------
// CSV Row struct & parser 
// -----------------------------
struct CsvRow {
    std::string date;
    float open = 0.0f;
    float high = 0.0f;
    float low = 0.0f;
    float close = 0.0f;
    float adjClose = 0.0f;
    long long volume = 0;
    bool hasAdj = false;
    bool valid = false;
};

static void trimInPlace(std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) { s.clear(); return; }
    size_t b = s.find_last_not_of(" \t\r\n");
    s = s.substr(a, b - a + 1);
}

bool readCsvFull(const std::string& filename, std::vector<CsvRow>& outRows, bool& headerHasAdj) {
    outRows.clear();
    headerHasAdj = false;

    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    if (!std::getline(file, line)) return false;

    std::vector<std::string> headerFields;
    {
        std::string cur;
        bool inQuotes = false;
        for (char c : line) {
            if (c == '"') { inQuotes = !inQuotes; continue; }
            if (c == ',' && !inQuotes) { headerFields.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        headerFields.push_back(cur);
    }

    std::unordered_map<std::string, int> idx;
    for (size_t i = 0; i < headerFields.size(); ++i) {
        std::string h = headerFields[i];
        trimInPlace(h);
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return std::tolower(c); });
        idx[h] = static_cast<int>(i);
    }

    auto findIndex = [&](const std::vector<std::string>& candidates)->int {
        for (auto& c : candidates) {
            auto it = idx.find(c);
            if (it != idx.end()) return it->second;
        }
        return -1;
        };
    int dateIdx = findIndex({ "date" });
    int openIdx = findIndex({ "open" });
    int highIdx = findIndex({ "high" });
    int lowIdx = findIndex({ "low" });
    int closeIdx = findIndex({ "close" });
    int adjIdx = findIndex({ "adj close", "adj_close", "adjusted close", "adjusted_close", "adjclose" });
    int volumeIdx = findIndex({ "volume", "vol" });

    headerHasAdj = (adjIdx >= 0);

    while (std::getline(file, line)) {
        std::vector<std::string> fields;
        std::string cur;
        bool inQuotes = false;
        for (char c : line) {
            if (c == '"') { inQuotes = !inQuotes; continue; }
            if (c == ',' && !inQuotes) { fields.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        fields.push_back(cur);

        CsvRow row;
        if (dateIdx >= 0 && dateIdx < (int)fields.size()) {
            row.date = fields[dateIdx];
            trimInPlace(row.date);
        }
        else {
            continue;
        }

        auto parseFloat = [&](int idxField, float& outVal) -> bool {
            if (idxField < 0 || idxField >= (int)fields.size()) return false;
            std::string s = fields[idxField];
            trimInPlace(s);
            if (s.empty()) return false;
            std::replace(s.begin(), s.end(), ',', '.');
            try {
                outVal = std::stof(s);
                return true;
            }
            catch (...) {
                return false;
            }
            };
        auto parseLong = [&](int idxField, long long& outVal) -> bool {
            if (idxField < 0 || idxField >= (int)fields.size()) return false;
            std::string s = fields[idxField];
            trimInPlace(s);
            if (s.empty()) return false;
            try {
                outVal = std::stoll(s);
                return true;
            }
            catch (...) {
                return false;
            }
            };

        bool any = false;
        if (parseFloat(openIdx, row.open)) any = true;
        if (parseFloat(highIdx, row.high)) any = true;
        if (parseFloat(lowIdx, row.low)) any = true;
        if (parseFloat(closeIdx, row.close)) any = true;
        if (parseFloat(adjIdx, row.adjClose)) { row.hasAdj = true; any = true; }
        if (parseLong(volumeIdx, row.volume)) any = true;

        if (!row.date.empty() && (row.hasAdj || row.close != 0.0f || any)) {
            row.valid = true;
            outRows.push_back(row);
        }
    }
    return !outRows.empty();
}

void alignByDateRows(const std::vector<CsvRow>& A, bool A_hasAdj,
    const std::vector<CsvRow>& B, bool B_hasAdj,
    std::vector<float>& outA, std::vector<float>& outB, std::vector<std::string>& outDates) {
    outA.clear(); outB.clear(); outDates.clear();
    std::unordered_map<std::string, float> mapB;
    mapB.reserve(B.size());
    for (const auto& r : B) {
        if (!r.valid) continue;
        float price = (B_hasAdj && r.hasAdj) ? r.adjClose : r.close;
        mapB[r.date] = price;
    }
    for (const auto& r : A) {
        if (!r.valid) continue;
        auto it = mapB.find(r.date);
        if (it != mapB.end()) {
            float priceA = (A_hasAdj && r.hasAdj) ? r.adjClose : r.close;
            outA.push_back(priceA);
            outB.push_back(it->second);
            outDates.push_back(r.date);
        }
    }
}

std::vector<float> calculateLogReturns(const std::vector<float>& prices) {
    std::vector<float> returns;
    if (prices.size() < 2) return returns;
    returns.reserve(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        float p0 = prices[i - 1];
        float p1 = prices[i];
        if (p0 <= 0.0f || p1 <= 0.0f) continue;
        returns.push_back(std::log(p1 / p0));
    }
    return returns;
}

float calculateMean(const std::vector<float>& returns) {
    if (returns.empty()) return 0.0f;
    float sum = 0.0f;
    for (float r : returns) sum += r;
    return sum / returns.size();
}

// NEW: N-Dimensional Dynamic Covariance Matrix Calculator
std::vector<float> calculateCovarianceMatrix(const std::vector<std::vector<float>>& returnsList) {
    int numAssets = returnsList.size();
    if (numAssets == 0) return {};
    int numReturns = returnsList.size(); // fixed out-of-bounds risk

    std::vector<float> covMatrix(numAssets * numAssets, 0.0f);
    std::vector<float> means(numAssets, 0.0f);

    for (int i = 0; i < numAssets; ++i) {
        means[i] = calculateMean(returnsList[i]);
    }

    for (int i = 0; i < numAssets; ++i) {
        for (int j = 0; j < numAssets; ++j) {
            float cov = 0.0f;
            for (int k = 0; k < numReturns; ++k) {
                cov += (returnsList[i][k] - means[i]) * (returnsList[j][k] - means[j]);
            }
            covMatrix[i * numAssets + j] = cov / (numReturns - 1.0f);
        }
    }
    return covMatrix;
}

// NEW: N-Dimensional Dynamic Cholesky Decomposition
bool choleskyDecomp(int n, const std::vector<float>& cov, std::vector<float>& L, float jitter = 1e-9f) {
    L.assign(n * n, 0.0f);
    std::vector<float> tempCov = cov;

    // Add a small jitter to diagonal elements for stability
    for (int i = 0; i < n; ++i) tempCov[i * n + i] += jitter;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                float val = tempCov[i * n + i] - sum;
                if (val <= 0.0f) return false; // Fails if not positive definite
                L[i * n + j] = std::sqrt(val);
            }
            else {
                L[i * n + j] = (tempCov[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return true;
}

// NEW: N-Dimensional CPU Random Number and Correlation Generator
bool generateCorrelatedSamplesCPU(size_t N_samples, int numAssets, const std::vector<float>& mu, const std::vector<float>& L, std::vector<float>& outSamples, uint64_t seed = 0) {
    outSamples.assign(N_samples * numAssets, 0.0f);
    std::mt19937_64 rng(seed ? seed : std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (size_t i = 0; i < N_samples; ++i) {
        std::vector<float> z(numAssets);
        for (int j = 0; j < numAssets; ++j) z[j] = nd(rng);

        for (int j = 0; j < numAssets; ++j) {
            float val = mu[j];
            for (int k = 0; k <= j; ++k) {
                val += L[j * numAssets + k] * z[k];
            }
            outSamples[i * numAssets + j] = val;
        }
    }
    return true;
}

// -----------------------------
// MAIN
// -----------------------------
int main(int argc, char** argv) {
    Logger logger("CudaRuntime1_results.txt");

    // Default CLI parameters
    size_t Ngpu = 2 * 1000 * 1000;
    double alpha = 0.95;
    uint64_t seed = 0;
    float w0 = 1.0f, w1 = 1.0f;

    if (argc >= 2) Ngpu = static_cast<size_t>(std::stoull(argv[1])); // Fixed from std::stoull(argv)
    if (argc >= 3) alpha = std::stod(argv[2]);
    if (argc >= 4) seed = static_cast<uint64_t>(std::stoull(argv[3]));
    if (argc >= 6) { w0 = std::stof(argv[4]); w1 = std::stof(argv[5]); }

    // ---------------------------------------------------------
        // INITIALIZATION REPORT
        // ---------------------------------------------------------
    logger.infoLine("\n======================================================");
    logger.infoLine("         INITIALIZING MONTE CARLO VaR ENGINE          ");
    logger.infoLine("======================================================");
    logger.infoLine("[+] Configuration Settings");
    logger.infoLine("    - GPU Target Paths : ", Ngpu);
    logger.infoLine("    - Confidence Level : ", alpha * 100, "%");
    logger.infoLine("    - Random Seed      : ", (seed == 0 ? "0 (Time-Based/Random)" : std::to_string(seed)));
    logger.infoLine("    - Asset Weights    : [", w0, ", ", w1, "]\n");

    // Read CSV files
    std::vector<CsvRow> rowsA, rowsB;
    bool aHasAdj = false, bHasAdj = false;
    if (!readCsvFull("aselsan.csv", rowsA, aHasAdj)) {
        logger.errorLine("Failed to read aselsan.csv"); return -1;
    }
    if (!readCsvFull("thy.csv", rowsB, bHasAdj)) {
        logger.errorLine("Failed to read thy.csv"); return -1;
    }

    logger.infoLine("[+] Data Processing (Historical CSVs)");
    logger.infoLine("    - Raw Rows Read    : Aselsan = ", rowsA.size(), " | THY = ", rowsB.size());

    // Align by date
    std::vector<float> aPricesAligned, tPricesAligned;
    std::vector<std::string> alignedDates;
    alignByDateRows(rowsA, aHasAdj, rowsB, bHasAdj, aPricesAligned, tPricesAligned, alignedDates);

    if (alignedDates.size() < 2) { logger.errorLine("Not enough aligned rows"); return -1; }
    logger.infoLine("    - Aligned Dates    : ", alignedDates.size(), " valid trading days matched\n");

    // Preparation for N-Asset architecture
    std::vector<std::vector<float>> allReturns;
    allReturns.push_back(calculateLogReturns(aPricesAligned));
    allReturns.push_back(calculateLogReturns(tPricesAligned));
    int numAssets = allReturns.size();

    std::vector<float> mu(numAssets, 0.0f);
    for (int i = 0; i < numAssets; ++i) {
        mu[i] = calculateMean(allReturns[i]);
    }

    std::vector<float> weights = { w0, w1 };

    // Dynamic Covariance and Cholesky 
    std::vector<float> covMatrix = calculateCovarianceMatrix(allReturns);
    std::vector<float> L;
    if (!choleskyDecomp(numAssets, covMatrix, L)) { logger.errorLine("Cholesky failed"); return -1; }

    logger.infoLine("[+] Statistical Models (Matrix Generation)");
    logger.infoLine("    - Covariance Matrix: ", covMatrix);
    logger.infoLine("    - Cholesky (L)     : ", L, "\n");

    // ---------------------------------------------------------
    // CPU STOPWATCH START
    // ---------------------------------------------------------
    size_t Ncpu = 20000;
    logger.infoLine("[+] Hardware Execution Phase");
    logger.infoLine("    -> Launching CPU reference calculations (N=", Ncpu, ") ...");

    auto t0 = std::chrono::high_resolution_clock::now();

    std::vector<float> samples;
    generateCorrelatedSamplesCPU(Ncpu, numAssets, mu, L, samples, seed);

    std::vector<float> cpuPL(Ncpu, 0.0f);
    for (size_t i = 0; i < Ncpu; ++i) {
        float pl = 0.0f;
        for (int j = 0; j < numAssets; ++j) {
            pl += weights[j] * samples[i * numAssets + j];
        }
        cpuPL[i] = pl;
    }

    float cpuVaR = calculateVaR(cpuPL, static_cast<float>(alpha));
    float cpuCVaR = calculateCVaR(cpuPL, static_cast<float>(alpha));

    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    // ---------------------------------------------------------

    // GPU Monte Carlo VaR Call
    float gpuVaR = 0.0f, gpuCVaR = 0.0f, gpuMs = 0.0f;

    // NEW VARIABLES FOR PROFIT PROJECTIONS
    float expReturn = 0.0f, probProfit = 0.0f, optGain = 0.0f;

    bool ok = simulateReturnsGPU(Ngpu, numAssets, mu.data(), L.data(), weights.data(), seed, static_cast<float>(alpha), &gpuVaR, &gpuCVaR, &expReturn, &probProfit, &optGain, &gpuMs);

    if (!ok) {
        logger.errorLine("GPU simulation failed");
    }
    else {
        auto rel = [](double a, double b)->double {
            if (b == 0.0) return std::abs(a - b);
            return std::abs((a - b) / b) * 100.0;
            };

        double varDiff = rel(cpuVaR, gpuVaR);
        double cvarDiff = rel(cpuCVaR, gpuCVaR);

        double workloadMultiplier = static_cast<double>(Ngpu) / static_cast<double>(Ncpu);
        double projectedCpuMs = cpuMs * workloadMultiplier;
        double speedup = projectedCpuMs / gpuMs;

        // ---------------------------------------------------------
        // PROFESSIONAL PERFORMANCE & PROFIT REPORT
        // ---------------------------------------------------------
        logger.infoLine("\n======================================================");
        logger.infoLine("       MONTE CARLO SIMULATION PERFORMANCE REPORT      ");
        logger.infoLine("======================================================");
        logger.infoLine("1. SIMULATION PARAMETERS");
        logger.infoLine("   - Portfolio Assets : ", numAssets);
        logger.infoLine("   - Historical Days  : ", alignedDates.size());
        logger.infoLine("   - CPU Workload     : ", Ncpu, " paths");
        logger.infoLine("   - GPU Workload     : ", Ngpu, " paths (", workloadMultiplier, "x heavier)");
        logger.infoLine("   - Confidence Level : ", alpha * 100, "%");
        logger.infoLine("------------------------------------------------------");
        logger.infoLine("2. RISK METRICS (DEFENSE: What could I lose?)");
        logger.infoLine("   [CPU] VaR: ", cpuVaR, "  |  CVaR: ", cpuCVaR);
        logger.infoLine("   [GPU] VaR: ", gpuVaR, "  |  CVaR: ", gpuCVaR);
        logger.infoLine("   [Diff ] VaR: ", varDiff, "%  |  CVaR: ", cvarDiff, "%");
        logger.infoLine("------------------------------------------------------");

        // NEW SECTION: PROFIT PROJECTIONS
        logger.infoLine("3. PROFIT & GROWTH PROJECTIONS (OFFENSE: What could I win?)");
        logger.infoLine("   [MEAN ] Expected Average Return : ", expReturn * 100.0f, " %");
        logger.infoLine("   [WIN %] Probability of Profit   : ", probProfit, " %");
        logger.infoLine("   [PEAK ] Optimistic Peak Gain    : ", optGain * 100.0f, " % (Best ", (1.0 - alpha) * 100, "% of scenarios)");
        logger.infoLine("------------------------------------------------------");

        logger.infoLine("4. PERFORMANCE & ACCELERATION");
        logger.infoLine("   [CPU] Measured Time (", Ncpu, ") : ", cpuMs, " ms");
        logger.infoLine("   [GPU] Measured Time (", Ngpu, ") : ", gpuMs, " ms");
        logger.infoLine("   [Proj ] Projected CPU Time for ", Ngpu, " : ~", projectedCpuMs, " ms");
        logger.infoLine("   [WIN  ] GPU is ", speedup, " TIMES FASTER than CPU!");
        logger.infoLine("======================================================\n");
    }

    return 0;
}
