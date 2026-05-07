// =====================================================================
// GPU-Accelerated Monte Carlo Simulation for Portfolio Risk Assessment
// =====================================================================

#include <iostream>
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

#include "calculate_cpu.h"
#include "simulate_gpu.h"
#include "main.h"

// ---------------------------------------------------------
// Utility: Vector Print Operator
// ---------------------------------------------------------
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        os << v[i] << (i == v.size() - 1 ? "" : ", ");
    }
    os << "]";
    return os;
}

// ---------------------------------------------------------
// Utility: Thread-Safe Logger
// ---------------------------------------------------------
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
    explicit Logger(const std::string& filename = "Simulation_Results.txt") {
        file_.open(filename, std::ios::app);
        fileOk_ = file_.good();
        std::ostringstream oss;
        oss << "\n----- Run at " << nowTimestamp() << " -----\n";
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

// ---------------------------------------------------------
// Data Structures & CSV Parser
// ---------------------------------------------------------
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
    std::string cur;
    bool inQuotes = false;
    for (char c : line) {
        if (c == '"') { inQuotes = !inQuotes; continue; }
        if (c == ',' && !inQuotes) { headerFields.push_back(cur); cur.clear(); }
        else cur.push_back(c);
    }
    headerFields.push_back(cur);

    std::unordered_map<std::string, int> idx;
    for (size_t i = 0; i < headerFields.size(); ++i) {
        std::string h = headerFields[i];
        trimInPlace(h);
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c) { return std::tolower(c); });
        idx[h] = static_cast<int>(i);
    }

    auto findIndex = [&](const std::vector<std::string>& candidates)->int {
        for (const auto& c : candidates) {
            auto it = idx.find(c);
            if (it != idx.end()) return it->second;
        }
        return -1;
        };

    int dateIdx = findIndex({ "date" });
    int closeIdx = findIndex({ "close" });
    int adjIdx = findIndex({ "adj close", "adj_close", "adjusted close" });

    headerHasAdj = (adjIdx >= 0);

    while (std::getline(file, line)) {
        std::vector<std::string> fields;
        cur.clear();
        inQuotes = false;
        for (char c : line) {
            if (c == '"') { inQuotes = !inQuotes; continue; }
            if (c == ',' && !inQuotes) { fields.push_back(cur); cur.clear(); }
            else cur.push_back(c);
        }
        fields.push_back(cur);

        CsvRow row;
        if (dateIdx >= 0 && dateIdx < static_cast<int>(fields.size())) {
            row.date = fields[dateIdx];
            trimInPlace(row.date);
        }
        else {
            continue;
        }

        auto parseFloat = [&](int idxField, float& outVal) -> bool {
            if (idxField < 0 || idxField >= static_cast<int>(fields.size())) return false;
            std::string s = fields[idxField];
            trimInPlace(s);
            if (s.empty()) return false;
            std::replace(s.begin(), s.end(), ',', '.');
            try { outVal = std::stof(s); return true; }
            catch (...) { return false; }
            };

        bool anyValidData = false;
        if (parseFloat(closeIdx, row.close)) anyValidData = true;
        if (parseFloat(adjIdx, row.adjClose)) { row.hasAdj = true; anyValidData = true; }

        if (!row.date.empty() && anyValidData) {
            row.valid = true;
            outRows.push_back(row);
        }
    }
    return !outRows.empty();
}

// ---------------------------------------------------------
// N-Dimensional Data Alignment
// ---------------------------------------------------------
void alignMultipleByDate(const std::vector<std::vector<CsvRow>>& allRows,
    const std::vector<bool>& hasAdj,
    std::vector<std::vector<float>>& outPrices,
    std::vector<std::string>& outDates) {
    outPrices.clear();
    outDates.clear();

    if (allRows.empty()) return;

    int numAssets = allRows.size();
    outPrices.resize(numAssets);

    std::unordered_map<std::string, int> dateCounts;
    for (int i = 0; i < numAssets; ++i) {
        for (const auto& r : allRows[i]) {
            if (r.valid) dateCounts[r.date]++;
        }
    }

    // Isolate common trading days using the reference asset (index 0)
    for (const auto& r : allRows[0]) {
        if (!r.valid) continue;
        if (dateCounts[r.date] == numAssets) {
            outDates.push_back(r.date);
        }
    }

    for (int i = 0; i < numAssets; ++i) {
        std::unordered_map<std::string, float> priceMap;
        for (const auto& r : allRows[i]) {
            if (r.valid) {
                priceMap[r.date] = (hasAdj[i] && r.hasAdj) ? r.adjClose : r.close;
            }
        }
        for (const auto& date : outDates) {
            outPrices[i].push_back(priceMap[date]);
        }
    }
}

// ---------------------------------------------------------
// Statistical Methods & Matrices
// ---------------------------------------------------------
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

std::vector<float> calculateCovarianceMatrix(const std::vector<std::vector<float>>& returnsList) {
    int numAssets = returnsList.size();
    if (numAssets == 0) return {};
    int numReturns = returnsList.size();

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

bool choleskyDecomp(int n, const std::vector<float>& cov, std::vector<float>& L, float jitter = 1e-9f) {
    L.assign(n * n, 0.0f);
    std::vector<float> tempCov = cov;

    for (int i = 0; i < n; ++i) tempCov[i * n + i] += jitter;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j <= i; j++) {
            float sum = 0.0f;
            for (int k = 0; k < j; k++) {
                sum += L[i * n + k] * L[j * n + k];
            }
            if (i == j) {
                float val = tempCov[i * n + i] - sum;
                if (val <= 0.0f) return false;
                L[i * n + j] = std::sqrt(val);
            }
            else {
                L[i * n + j] = (tempCov[i * n + j] - sum) / L[j * n + j];
            }
        }
    }
    return true;
}

// ---------------------------------------------------------
// Dynamic Portfolio Optimizer (Mean-CVaR Strategy)
// ---------------------------------------------------------
struct OptimizationResult {
    std::vector<float> weights;
    float expectedReturn;
    float var;
    float cvar;
    float score;
};

OptimizationResult optimizePortfolioWeights(const std::vector<float>& samples, int numSamples, int numAssets, const std::vector<float>& mu, float alpha) {
    int numSimulations = 10000;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dist(0.01f, 1.0f);

    OptimizationResult bestResult;
    bestResult.score = -999999.0f;
    bestResult.weights.assign(numAssets, 1.0f / numAssets);

    for (int p = 0; p < numSimulations; ++p) {
        std::vector<float> w(numAssets);
        float sum = 0.0f;
        for (int i = 0; i < numAssets; ++i) { w[i] = dist(gen); sum += w[i]; }

        float expRet = 0.0f;
        for (int i = 0; i < numAssets; ++i) { w[i] /= sum; expRet += w[i] * mu[i]; }

        std::vector<float> pl(numSamples, 0.0f);
        for (int i = 0; i < numSamples; ++i) {
            float val = 0.0f;
            for (int j = 0; j < numAssets; ++j) {
                val += w[j] * samples[i * numAssets + j];
            }
            pl[i] = val;
        }

        float currentVaR = calculateVaR(pl, alpha);
        float currentCVaR = calculateCVaR(pl, alpha);
        float score = expRet / (currentCVaR > 0.0001f ? currentCVaR : 0.0001f);

        if (score > bestResult.score && expRet > 0.0f) {
            bestResult.score = score;
            bestResult.weights = w;
            bestResult.var = currentVaR;
            bestResult.cvar = currentCVaR;
            bestResult.expectedReturn = expRet;
        }
    }
    return bestResult;
}

// ---------------------------------------------------------
// CPU Reference Generator
// ---------------------------------------------------------
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

// ---------------------------------------------------------
// Main Execution
// ---------------------------------------------------------
int main(int argc, char** argv) {
    Logger logger("Simulation_Results.txt");

    size_t Ngpu = 2000000;
    double alpha = 0.95;
    uint64_t seed = 0;

    std::vector<std::string> fileNames = { "aselsan.csv", "thy.csv", "tuprs.csv", "pgsus.csv" };
    int numAssets = fileNames.size();

    logger.infoLine("\n======================================================");
    logger.infoLine("         INITIALIZING N-DIMENSIONAL VaR ENGINE        ");
    logger.infoLine("======================================================");
    logger.infoLine("[+] Configuration Settings");
    logger.infoLine("    - GPU Target Paths : ", Ngpu);
    logger.infoLine("    - Confidence Level : ", alpha * 100, "%");
    logger.infoLine("    - Random Seed      : ", (seed == 0 ? "0 (Time-Based/Random)" : std::to_string(seed)));
    logger.infoLine("    - Asset Portfolio  : ", numAssets, " total assets\n");

    std::vector<std::vector<CsvRow>> allRows(numAssets);
    std::vector<bool> hasAdj(numAssets, false);

    logger.infoLine("[+] Data Processing (Historical CSVs)");
    for (int i = 0; i < numAssets; ++i) {
        bool tempHasAdj = false;

        if (!readCsvFull(fileNames[i], allRows[i], tempHasAdj)) {
            logger.errorLine("Failed to read ", fileNames[i]);
            return -1;
        }

        hasAdj[i] = tempHasAdj;
        logger.infoLine("    - Loaded ", fileNames[i], " -> ", allRows[i].size(), " rows");
    }

    std::vector<std::vector<float>> alignedPrices;
    std::vector<std::string> alignedDates;
    alignMultipleByDate(allRows, hasAdj, alignedPrices, alignedDates);

    if (alignedDates.size() < 2) {
        logger.errorLine("Not enough aligned trading days.");
        return -1;
    }
    logger.infoLine("    - Aligned Dates    : ", alignedDates.size(), " common valid trading days matched\n");

    std::vector<std::vector<float>> allReturns(numAssets);
    std::vector<float> mu(numAssets, 0.0f);
    for (int i = 0; i < numAssets; ++i) {
        allReturns[i] = calculateLogReturns(alignedPrices[i]);
        mu[i] = calculateMean(allReturns[i]);
    }

    std::vector<float> covMatrix = calculateCovarianceMatrix(allReturns);
    std::vector<float> L;
    if (!choleskyDecomp(numAssets, covMatrix, L)) {
        logger.errorLine("Cholesky decomposition failed.");
        return -1;
    }

    logger.infoLine("[+] Statistical Models (Matrix Generation)");
    logger.infoLine("    - Covariance Matrix Generated (", numAssets, "x", numAssets, " dimensions)");
    logger.infoLine("    - Cholesky (L) Matrix Generated (", numAssets, "x", numAssets, " dimensions)\n");

    size_t Ncpu = 20000;
    std::vector<float> equalWeights(numAssets, 1.0f / numAssets);

    logger.infoLine("[+] Hardware Execution Phase");
    logger.infoLine("    -> Launching CPU reference calculations (N=", Ncpu, ") ...");

    auto t0 = std::chrono::high_resolution_clock::now();
    std::vector<float> samples;
    generateCorrelatedSamplesCPU(Ncpu, numAssets, mu, L, samples, seed);

    std::vector<float> cpuPL(Ncpu, 0.0f);
    for (size_t i = 0; i < Ncpu; ++i) {
        float pl = 0.0f;
        for (int j = 0; j < numAssets; ++j) pl += equalWeights[j] * samples[i * numAssets + j];
        cpuPL[i] = pl;
    }

    float cpuVaR = calculateVaR(cpuPL, static_cast<float>(alpha));
    float cpuCVaR = calculateCVaR(cpuPL, static_cast<float>(alpha));
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    logger.infoLine("    -> Launching Dynamic Portfolio Optimizer (10,000 combinations) ...");
    auto optT0 = std::chrono::high_resolution_clock::now();
    OptimizationResult optRes = optimizePortfolioWeights(samples, Ncpu, numAssets, mu, static_cast<float>(alpha));
    auto optT1 = std::chrono::high_resolution_clock::now();
    double optMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(optT1 - optT0).count();

    float baselineExpRet = 0.0f;
    for (int i = 0; i < numAssets; ++i) baselineExpRet += equalWeights[i] * mu[i];
    float baselineScore = baselineExpRet / (cpuCVaR > 0.0001f ? cpuCVaR : 0.0001f);
    double scoreImprovement = ((optRes.score - baselineScore) / baselineScore) * 100.0;

    float gpuVaR = 0.0f, gpuCVaR = 0.0f, expReturn = 0.0f, probProfit = 0.0f, optGain = 0.0f, gpuMs = 0.0f;

    bool ok = simulateReturnsGPU(Ngpu, numAssets, mu.data(), L.data(), equalWeights.data(), seed, static_cast<float>(alpha),
        &gpuVaR, &gpuCVaR, &expReturn, &probProfit, &optGain, &gpuMs);

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

        logger.infoLine("\n======================================================");
        logger.infoLine("       MONTE CARLO SIMULATION PERFORMANCE REPORT      ");
        logger.infoLine("======================================================");
        logger.infoLine("1. SIMULATION PARAMETERS");
        logger.infoLine("   - Portfolio Assets : ", numAssets, " (Equally Weighted)");
        logger.infoLine("   - Historical Days  : ", alignedDates.size(), " active trading days");
        logger.infoLine("   - CPU Workload     : ", Ncpu, " paths");
        logger.infoLine("   - GPU Workload     : ", Ngpu, " paths (", workloadMultiplier, "x heavier)");
        logger.infoLine("   - Confidence Level : ", alpha * 100, "%");
        logger.infoLine("------------------------------------------------------");

        logger.infoLine("2. RISK METRICS (DEFENSE: What could I lose?)");
        logger.infoLine("   [CPU] VaR: ", cpuVaR, "  |  CVaR: ", cpuCVaR);
        logger.infoLine("   [GPU] VaR: ", gpuVaR, "  |  CVaR: ", gpuCVaR);
        logger.infoLine("   [Diff ] VaR: ", varDiff, "%  |  CVaR: ", cvarDiff, "%");
        logger.infoLine("------------------------------------------------------");

        logger.infoLine("3. PROFIT & GROWTH PROJECTIONS (OFFENSE: What could I win?)");
        logger.infoLine("   [MEAN ] Expected Average Return : ", expReturn * 100.0f, " %");
        logger.infoLine("   [WIN %] Probability of Profit   : ", probProfit, " %");
        logger.infoLine("   [PEAK ] Optimistic Peak Gain    : ", optGain * 100.0f, " % (Best ", (1.0 - alpha) * 100, "% of scenarios)");
        logger.infoLine("------------------------------------------------------");

        logger.infoLine("4. CURRENT HARDWARE PERFORMANCE");
        logger.infoLine("   [CPU] Measured Time (", Ncpu, ") : ", cpuMs, " ms");
        logger.infoLine("   [GPU] Measured Time (", Ngpu, ") : ", gpuMs, " ms");
        logger.infoLine("   [Proj ] Projected CPU Time for ", Ngpu, " : ~", projectedCpuMs, " ms");
        logger.infoLine("------------------------------------------------------");

        logger.infoLine("5. EVOLUTION OF OPTIMIZATIONS (For 2M Paths)");
        double histCpuUnopt = 3182.99;
        double histGpuDebug = 248.49;
        logger.infoLine("   [Stage 1] Baseline CPU (Single-Thread)     : ~", histCpuUnopt, " ms");
        logger.infoLine("   [Stage 2] Baseline GPU (No Optimizations)  :  ", histGpuDebug, " ms");
        logger.infoLine("   [Stage 3] Advanced GPU (Release+FastMath)  :  ", gpuMs, " ms\n");
        logger.infoLine("   >> STEP-BY-STEP IMPROVEMENTS:");
        logger.infoLine("      1. Hardware Shift (Stage 1 -> 2) : GPU reduced time by ", (histCpuUnopt / histGpuDebug), "x");
        logger.infoLine("      2. Software Tuning (Stage 2 -> 3): FastMath & Unrolling improved GPU by ", (histGpuDebug / gpuMs), "x");
        logger.infoLine("      ==================================================");
        logger.infoLine("      => TOTAL SPEEDUP (Stage 1 to 3)  : System is now ", (histCpuUnopt / gpuMs), "x FASTER!");
        logger.infoLine("======================================================\n");

        logger.infoLine("6. DYNAMIC PORTFOLIO OPTIMIZATION (Mean-CVaR Strategy)");
        logger.infoLine("   [TIME ] Optimizer ran in : ", optMs, " ms (Tested 10,000 allocations)");
        logger.infoLine("   ---------------------------------------------------");
        logger.infoLine("   >> BASELINE (Equal-Weight) -> Return: ", baselineExpRet * 100.0f, "% | CVaR: ", cpuCVaR, " | Score: ", baselineScore);
        logger.infoLine("   >> OPTIMIZED PORTFOLIO     -> Return: ", optRes.expectedReturn * 100.0f, "% | CVaR: ", optRes.cvar, " | Score: ", optRes.score);
        logger.infoLine("   ---------------------------------------------------");
        logger.infoLine("   [WIN  ] Risk-Adjusted Efficiency Improved by : +", scoreImprovement, " %\n");
        logger.infoLine("   >> OPTIMAL ASSET ALLOCATION:");
        for (int i = 0; i < numAssets; ++i) {
            logger.infoLine("      - ", fileNames[i], " : % ", optRes.weights[i] * 100.0f);
        }
        logger.infoLine("======================================================\n");
    }

    return 0;
}