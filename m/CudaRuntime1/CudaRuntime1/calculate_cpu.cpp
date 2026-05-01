#include "calculate_cpu.h"
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>

// CPU version of calculateVaR and CVaR using nth_element (no full sort)
float calculateVaR(const std::vector<float>& values, float confidenceLevel) {
    if (values.empty()) return 0.0f;
    size_t n = values.size();

    double tailFrac = 1.0 - static_cast<double>(confidenceLevel);
    size_t raw = static_cast<size_t>(tailFrac * static_cast<double>(n));
    size_t idx = (raw == 0) ? 0 : ((raw - 1 >= n) ? (n - 1) : (raw - 1));

    std::vector<float> tmp = values; // nth_element is destructive
    std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(idx), tmp.end());
    float q = tmp[idx];

    // Preserve previous sign convention: return negative loss value (as in earlier code)
    return -q;
}

float calculateCVaR(const std::vector<float>& values, float confidenceLevel) {
    if (values.empty()) return 0.0f;
    size_t n = values.size();

    double tailFrac = 1.0 - static_cast<double>(confidenceLevel);
    size_t raw = static_cast<size_t>(tailFrac * static_cast<double>(n));
    size_t idx = (raw == 0) ? 0 : ((raw - 1 >= n) ? (n - 1) : (raw - 1));

    std::vector<float> tmp = values;
    std::nth_element(tmp.begin(), tmp.begin() + static_cast<std::ptrdiff_t>(idx), tmp.end());
    float q = tmp[idx];

    double sum = 0.0;
    size_t count = 0;
    for (float v : tmp) {
        if (v <= q) { sum += v; ++count; }
    }
    if (count == 0) return -q;
    return static_cast<float>(-(sum / static_cast<double>(count)));
}
