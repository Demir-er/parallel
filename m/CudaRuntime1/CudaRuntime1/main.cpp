// File: CudaRuntime1/main.cpp
// Upgraded: full-CSV parser (prefers Adj Close), log-returns, logger, alignment, CPU sampling,
// GPU Monte-Carlo VaR integration. Targets C++17.

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

#include "simulate_cpu.h"
#include "simulate_gpu.h"
#include "calculate_cpu.h"
#include "calculate_gpu.h"
#include "output.h"

// -----------------------------
// Simple Logger (console + append file)
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
// CUDA error checking helper
// -----------------------------
inline bool checkCuda(cudaError_t err, const char* msg = "") {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error: " << msg << " : " << cudaGetErrorString(err) << std::endl;
        return false;
    }
    return true;
}

// -----------------------------
// CSV Row struct & parser that detects Adj Close
// -----------------------------
struct CsvRow {
    std::string date;
    float open = 0.0f;
    float high = 0.0f;
    float low = 0.0f;
    float close = 0.0f;
    float adjClose = 0.0f;
    long long volume = 0;
    bool hasAdj = false; // indicates whether adjClose is valid for this row
    bool valid = false;  // indicates row parsed successfully
};

// Trim helper
static void trimInPlace(std::string &s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) { s.clear(); return; }
    size_t b = s.find_last_not_of(" \t\r\n");
    s = s.substr(a, b - a + 1);
}

// Parse CSV into vector<CsvRow>. Detect if header contains "Adj Close".
// Returns true if at least one row parsed.
bool readCsvFull(const std::string& filename, std::vector<CsvRow>& outRows, bool& headerHasAdj) {
    outRows.clear();
    headerHasAdj = false;

    std::ifstream file(filename);
    if (!file.is_open()) return false;

    std::string line;
    if (!std::getline(file, line)) return false;
    // parse header to find column indices
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
    // map header names (lowercase) to indices
    std::unordered_map<std::string, int> idx;
    for (size_t i = 0; i < headerFields.size(); ++i) {
        std::string h = headerFields[i];
        trimInPlace(h);
        std::transform(h.begin(), h.end(), h.begin(), [](unsigned char c){ return std::tolower(c); });
        idx[h] = static_cast<int>(i);
    }
    // Common names: "date", "open", "high", "low", "close", "adj close" or "adj_close" or "adjusted close", "volume"
    auto findIndex = [&](const std::vector<std::string>& candidates)->int {
        for (auto &c : candidates) {
            auto it = idx.find(c);
            if (it != idx.end()) return it->second;
        }
        return -1;
    };
    int dateIdx = findIndex({"date"});
    int openIdx = findIndex({"open"});
    int highIdx = findIndex({"high"});
    int lowIdx = findIndex({"low"});
    int closeIdx = findIndex({"close"});
    int adjIdx = findIndex({"adj close", "adj_close", "adjusted close", "adjusted_close", "adjclose"});
    int volumeIdx = findIndex({"volume", "vol"});

    headerHasAdj = (adjIdx >= 0);

    // parse remaining lines
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
        } else {
            continue; // cannot parse row without date
        }

        auto parseFloat = [&](int idxField, float &outVal) -> bool {
            if (idxField < 0 || idxField >= (int)fields.size()) return false;
            std::string s = fields[idxField];
            trimInPlace(s);
            if (s.empty()) return false;
            // Replace comma decimals with dot
            std::replace(s.begin(), s.end(), ',', '.');
            try {
                outVal = std::stof(s);
                return true;
            } catch (...) {
                return false;
            }
        };
        auto parseLong = [&](int idxField, long long &outVal) -> bool {
            if (idxField < 0 || idxField >= (int)fields.size()) return false;
            std::string s = fields[idxField];
            trimInPlace(s);
            if (s.empty()) return false;
            try {
                outVal = std::stoll(s);
                return true;
            } catch (...) {
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

        // Mark valid if we parsed at least date and a price (close or adj)
        if (!row.date.empty() && (row.hasAdj || row.close != 0.0f || any)) {
            row.valid = true;
            outRows.push_back(row);
        }
    }

    return !outRows.empty();
}

// -----------------------------
// Align two series by date (inner join). Preserves order of the first series.
// For each pair pushes chosen price (adj if present else close).
// -----------------------------
void alignByDateRows(const std::vector<CsvRow>& A, bool A_hasAdj,
                     const std::vector<CsvRow>& B, bool B_hasAdj,
                     std::vector<float>& outA, std::vector<float>& outB, std::vector<std::string>& outDates) {
    outA.clear();
    outB.clear();
    outDates.clear();
    std::unordered_map<std::string, float> mapB;
    mapB.reserve(B.size());
    for (const auto &r : B) {
        if (!r.valid) continue;
        float price = (B_hasAdj && r.hasAdj) ? r.adjClose : r.close;
        mapB[r.date] = price;
    }
    for (const auto &r : A) {
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

// -----------------------------
// Returns: log-returns preferred
// -----------------------------
std::vector<float> calculateLogReturns(const std::vector<float>& prices) {
    std::vector<float> returns;
    if (prices.size() < 2) return returns;
    returns.reserve(prices.size() - 1);
    for (size_t i = 1; i < prices.size(); ++i) {
        float p0 = prices[i - 1];
        float p1 = prices[i];
        if (p0 <= 0.0f || p1 <= 0.0f) {
            // skip invalid / zero price points
            continue;
        }
        returns.push_back(std::log(p1 / p0));
    }
    return returns;
}

// -----------------------------
// Statistical helpers (mean, covariance, cholesky, sampling, stats)
// -----------------------------
float calculateMean(const std::vector<float>& returns) {
    if (returns.empty()) return 0.0f;
    float sum = 0.0f;
    for (float r : returns) sum += r;
    return sum / returns.size();
}

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

bool cholesky2x2(float cov00, float cov01, float cov11, float* L, float jitter = 1e-10f) {
    if (cov00 <= 0.0f) {
        if (cov00 > -jitter) cov00 = 0.0f; else return false;
    }
    L[0] = std::sqrt(std::max(0.0f, cov00));
    L[1] = 0.0f;

    if (L[0] == 0.0f) {
        if (std::fabs(cov01) > 1e-12f) {
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
        if (tmp > -1e-8f) tmp = 0.0f;
        else {
            cov11 += jitter;
            tmp = cov11 - (L[2] * L[2]);
            if (tmp < 0.0f) return false;
        }
    }
    L[3] = std::sqrt(tmp);
    return true;
}

bool generateCorrelatedSamplesCPU(size_t N, const float mu[2], const float L[4], std::vector<std::array<float,2>>& outSamples, uint64_t seed = 0) {
    outSamples.clear();
    outSamples.reserve(N);

    std::mt19937_64 rng;
    if (seed) rng.seed(static_cast<unsigned long long>(seed));
    else rng.seed((unsigned long long)std::chrono::high_resolution_clock::now().time_since_epoch().count());

    std::normal_distribution<float> nd(0.0f, 1.0f);

    for (size_t i = 0; i < N; ++i) {
        float z0 = nd(rng);
        float z1 = nd(rng);
        float x0 = mu[0] + L[0] * z0;
        float x1 = mu[1] + L[2] * z0 + L[3] * z1;
        outSamples.push_back({x0, x1});
    }
    return true;
}

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

// -----------------------------
// MAIN
// -----------------------------
int main(int argc, char** argv) {
    Logger logger("CudaRuntime1_results.txt");

    // CLI defaults
    size_t Ngpu = 2 * 1000 * 1000;
    double alpha = 0.95;
    uint64_t seed = 0; // 0 = time-based
    float w0 = 1.0f, w1 = 1.0f;

    // Simple CLI parsing (order: Ngpu alpha seed w0 w1)
    if (argc >= 2) Ngpu = static_cast<size_t>(std::stoull(argv[1]));
    if (argc >= 3) alpha = std::stod(argv[2]);
    if (argc >= 4) seed = static_cast<uint64_t>(std::stoull(argv[3]));
    if (argc >= 6) { w0 = std::stof(argv[4]); w1 = std::stof(argv[5]); }

    logger.infoLine("CLI: Ngpu=", Ngpu, " alpha=", alpha, " seed=", seed, " weights=[", w0, ",", w1, "]");

    // Read CSVs (reuse your readCsvFull)
    std::vector<CsvRow> rowsA, rowsB;
    bool aHasAdj=false, bHasAdj=false;
    if (!readCsvFull("aselsan2year.csv", rowsA, aHasAdj)) {
        logger.errorLine("Failed to read aselsan2year.csv"); return -1;
    }
    if (!readCsvFull("thy.csv", rowsB, bHasAdj)) {
        logger.errorLine("Failed to read thy.csv"); return -1;
    }
    logger.infoLine("Read rows: Aselsan=", rowsA.size(), " THY=", rowsB.size());

    // Align and get prices vectors
    std::vector<float> aPricesAligned, tPricesAligned;
    std::vector<std::string> alignedDates;
    alignByDateRows(rowsA, aHasAdj, rowsB, bHasAdj, aPricesAligned, tPricesAligned, alignedDates);
    logger.infoLine("Aligned rows: ", alignedDates.size());
    if (alignedDates.size() < 2) { logger.errorLine("Not enough aligned rows"); return -1; }

    // compute log returns
    auto returnsA = calculateLogReturns(aPricesAligned);
    auto returnsB = calculateLogReturns(tPricesAligned);
    if (returnsA.empty() || returnsB.empty()) { logger.errorLine("No returns"); return -1; }

    float meanA = calculateMean(returnsA);
    float meanB = calculateMean(returnsB);
    float cov00 = calculateCovariance(returnsA, returnsA, meanA, meanA);
    float cov01 = calculateCovariance(returnsA, returnsB, meanA, meanB);
    float cov11 = calculateCovariance(returnsB, returnsB, meanB, meanB);

    logger.infoLine("Cov matrix: [", cov00, " ", cov01, "] [", cov01, " ", cov11, "]");

    float L[4] = {0,0,0,0};
    if (!cholesky2x2(cov00, cov01, cov11, L)) { logger.errorLine("Cholesky failed"); return -1; }
    logger.infoLine("Cholesky L: ", L[0], ", ", L[2], ", ", L[3]);

    // CPU sampling (reproducible if seed provided)
    float mu[2] = { meanA, meanB };
    constexpr size_t Ncpu = 20000;
    std::vector<std::array<float,2>> samples;
    logger.infoLine("Generating CPU samples (N=", Ncpu, ") seed=", seed);
    auto t0 = std::chrono::high_resolution_clock::now();
    generateCorrelatedSamplesCPU(Ncpu, mu, L, samples, seed);
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpuMs = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();
    logger.infoLine("CPU generation time (ms): ", cpuMs);

    // CPU portfolio P/L vector
    std::vector<float> cpuPL;
    cpuPL.reserve(samples.size());
    for (const auto &s : samples) cpuPL.push_back(w0 * s[0] + w1 * s[1]);

    float cpuVaR = calculateVaR(cpuPL, static_cast<float>(alpha));
    float cpuCVaR = calculateCVaR(cpuPL, static_cast<float>(alpha));
    logger.infoLine("CPU VaR=", cpuVaR, " CVaR=", cpuCVaR);

    // GPU Monte Carlo VaR
    logger.infoLine("Launching GPU Monte Carlo VaR: Ngpu=", Ngpu);
    float gpuVaR = 0.0f, gpuCVaR = 0.0f, gpuMs = 0.0f;
    float weights[2] = { w0, w1 };
    bool ok = simulateReturnsGPU(Ngpu, mu, L, weights, seed, static_cast<float>(alpha), &gpuVaR, &gpuCVaR, &gpuMs);
    if (!ok) {
        logger.errorLine("GPU simulation failed");
    } else {
        logger.infoLine("GPU kernel+select time (ms) = ", gpuMs);
        logger.infoLine("GPU VaR=", gpuVaR, " GPU CVaR=", gpuCVaR);
    }

    // Compare CPU vs GPU (relative diff)
    if (ok) {
        auto rel = [](double a, double b)->double {
            if (b == 0.0) return std::abs(a - b);
            return std::abs((a - b) / b);
        };
        logger.infoLine("Comparison CPU vs GPU: VaR rel diff=", rel(cpuVaR, gpuVaR), " CVaR rel diff=", rel(cpuCVaR, gpuCVaR));
    }

    logger.infoLine("Summary: aligned rows=", alignedDates.size(), " CPU ms=", cpuMs, " GPU ms=", gpuMs);
    return 0;
}