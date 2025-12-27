/**
 * @file main.cpp
 * @brief High-Performance Matrix Multiplication Benchmark Suite
 *
 * This benchmark framework evaluates various GEMM (General Matrix Multiply)
 * implementations including hand-optimized SIMD kernels and OpenBLAS wrappers.
 *
 * Features:
 *   - AVX2/AVX-512 optimized kernels
 *   - Multi-threaded parallel implementations
 *   - Statistical benchmarking with warmup and multiple iterations
 *   - Correctness verification against reference implementation
 *
 * Usage:
 *   ./matmul [M N K] [options]
 *   ./matmul 4096           # Square matrix 4096x4096
 *   ./matmul 1024 2048 512  # M=1024, N=2048, K=512
 *
 * Options:
 *   -v, --verify      Enable correctness verification
 *   -k, --kernel      Filter kernels by name (substring match)
 *   -r, --rounds      Number of benchmark rounds (default: 5)
 *   -w, --warmup      Number of warmup rounds (default: 2)
 */

#include "matmul.hpp"
#include "utils.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <string>
#include <vector>

#include <immintrin.h> // AVX intrinsics & _mm_malloc/_mm_free

// ============================================================================
//                              ANSI Color Codes
// ============================================================================

namespace color {
constexpr const char *RESET = "\033[0m";
constexpr const char *BOLD = "\033[1m";
constexpr const char *DIM = "\033[2m";

constexpr const char *RED = "\033[31m";
constexpr const char *GREEN = "\033[32m";
constexpr const char *YELLOW = "\033[33m";
constexpr const char *BLUE = "\033[34m";
constexpr const char *MAGENTA = "\033[35m";
constexpr const char *CYAN = "\033[36m";
constexpr const char *WHITE = "\033[37m";
} // namespace color

// ============================================================================
//                         Aligned Memory Allocator
// ============================================================================
/**
 * @brief RAII wrapper for 64-byte aligned memory allocation
 *
 * AVX-512 instructions require 64-byte alignment for optimal performance.
 * Standard std::vector only guarantees alignof(T) alignment, which is
 * insufficient for SIMD operations.
 *
 * @tparam T Element type (typically float or double)
 */
template <typename T>
struct AlignedBuffer {
    T *data = nullptr;
    size_t size = 0;

    explicit AlignedBuffer(size_t n) : size(n) {
        data = static_cast<T *>(_mm_malloc(n * sizeof(T), 64));
        if (!data) {
            throw std::runtime_error("AlignedBuffer: allocation failed");
        }
        std::memset(data, 0, n * sizeof(T)); // Zero-initialize
    }

    ~AlignedBuffer() {
        if (data) {
            _mm_free(data);
        }
    }

    // Non-copyable, movable
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;
    AlignedBuffer(AlignedBuffer &&other) noexcept
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }

    T *get() { return data; }
    const T *get() const { return data; }
    T &operator[](size_t i) { return data[i]; }
};

// ============================================================================
//                           Benchmark Configuration
// ============================================================================

struct BenchConfig {
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;

    int warmup_rounds = 2; // Warmup iterations (not measured)
    int bench_rounds = 5;  // Measured iterations
    bool verify = false;

    std::vector<std::string> target_kernels; // Empty = run all kernels
};

/**
 * @brief Check if a kernel name matches any of the filters
 * @param name   Kernel name to check
 * @param filters  List of filter strings (substring match)
 * @return true if filters is empty OR name matches any filter
 */
bool matches_filter(const std::string &name, const std::vector<std::string> &filters) {
    if (filters.empty()) {
        return true; // No filter = run all
    }
    for (const auto &f : filters) {
        if (name.find(f) != std::string::npos) {
            return true;
        }
    }
    return false;
}

/**
 * @brief Split a comma-separated string into tokens
 */
std::vector<std::string> split_string(const std::string &s, char delim = ',') {
    std::vector<std::string> tokens;
    std::string token;
    for (char c : s) {
        if (c == delim) {
            if (!token.empty()) {
                tokens.push_back(token);
                token.clear();
            }
        } else {
            token += c;
        }
    }
    if (!token.empty()) {
        tokens.push_back(token);
    }
    return tokens;
}

// ============================================================================
//                           Statistics Utilities
// ============================================================================

struct BenchStats {
    double min_time;
    double max_time;
    double avg_time;
    double median_time;
    double stddev;
    double gflops_peak; // Based on min_time
    double gflops_avg;  // Based on avg_time
};

/**
 * @brief Calculate comprehensive statistics from timing data
 */
BenchStats calculate_stats(std::vector<double> &times, double flops) {
    BenchStats stats{};
    if (times.empty()) {
        return stats;
    }

    std::sort(times.begin(), times.end());

    stats.min_time = times.front();
    stats.max_time = times.back();

    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    stats.avg_time = sum / times.size();

    // Median
    size_t mid = times.size() / 2;
    stats.median_time = (times.size() % 2 == 0)
                          ? (times[mid - 1] + times[mid]) / 2.0
                          : times[mid];

    // Standard deviation
    double sq_sum = 0.0;
    for (double t : times) {
        sq_sum += (t - stats.avg_time) * (t - stats.avg_time);
    }
    stats.stddev = std::sqrt(sq_sum / times.size());

    // GFLOPS (higher is better, so use min_time for peak)
    stats.gflops_peak = flops / (stats.min_time * 1e9);
    stats.gflops_avg = flops / (stats.avg_time * 1e9);

    return stats;
}

// ============================================================================
//                              Output Formatting
// ============================================================================

void print_header_line(int width = 78) {
    std::cout << color::CYAN << std::string(width, '=') << color::RESET << "\n";
}

void print_section_line(int width = 78) {
    std::cout << color::DIM << std::string(width, '-') << color::RESET << "\n";
}

void print_banner(const BenchConfig &cfg) {
    print_header_line();
    std::cout << color::BOLD << color::CYAN
              << "  ╔╦╗╔═╗╔╦╗╔╦╗╦ ╦╦    ╔╗ ╔═╗╔╗╔╔═╗╦ ╦╔╦╗╔═╗╦═╗╦╔═\n"
              << "  ║║║╠═╣ ║ ║║║║ ║║    ╠╩╗║╣ ║║║║  ╠═╣║║║╠═╣╠╦╝╠╩╗\n"
              << "  ╩ ╩╩ ╩ ╩ ╩ ╩╚═╝╩═╝  ╚═╝╚═╝╝╚╝╚═╝╩ ╩╩ ╩╩ ╩╩╚═╩ ╩\n"
              << color::RESET;
    print_header_line();

    std::cout << color::WHITE << "  Configuration:\n"
              << color::RESET;
    std::cout << "    • Matrix Dims    : " << color::YELLOW
              << "M=" << cfg.M << ", N=" << cfg.N << ", K=" << cfg.K
              << color::RESET << "\n";
    std::cout << "    • Total FLOPs    : " << color::YELLOW
              << std::fixed << std::setprecision(2)
              << (2.0 * cfg.M * cfg.N * cfg.K / 1e9) << " GFLOP"
              << color::RESET << "\n";
    std::cout << "    • Warmup Rounds  : " << cfg.warmup_rounds << "\n";
    std::cout << "    • Bench Rounds   : " << cfg.bench_rounds << "\n";
    std::cout << "    • Verification   : "
              << (cfg.verify ? (std::string(color::GREEN) + "ON")
                             : (std::string(color::DIM) + "OFF"))
              << color::RESET << "\n";
    if (!cfg.target_kernels.empty()) {
        std::cout << "    • Kernel Filter  : " << color::MAGENTA;
        for (size_t i = 0; i < cfg.target_kernels.size(); ++i) {
            if (i > 0) {
                std::cout << ", ";
            }
            std::cout << cfg.target_kernels[i];
        }
        std::cout << color::RESET << "\n";
    }
    print_header_line();
    std::cout << "\n";
}

void print_table_header() {
    std::cout << color::BOLD
              << std::left << std::setw(22) << "  Kernel"
              << std::right << std::setw(12) << "Peak GFLOPS"
              << std::setw(12) << "Avg GFLOPS"
              << std::setw(10) << "Min(s)"
              << std::setw(10) << "Avg(s)"
              << std::setw(10) << "Stddev"
              << color::RESET << "\n";
    print_section_line();
}

void print_result_row(const std::string &name, const BenchStats &stats,
                      bool verified, bool verify_enabled) {
    // Color based on performance
    const char *perf_color = color::WHITE;
    if (stats.gflops_peak > 1000) {
        perf_color = color::GREEN;
    } else if (stats.gflops_peak > 500) {
        perf_color = color::YELLOW;
    } else {
        perf_color = color::RED;
    }

    std::cout << std::left << std::setw(22) << ("  " + name)
              << perf_color
              << std::right << std::fixed << std::setprecision(2)
              << std::setw(12) << stats.gflops_peak
              << std::setw(12) << stats.gflops_avg
              << color::RESET
              << std::setprecision(4)
              << std::setw(10) << stats.min_time
              << std::setw(10) << stats.avg_time
              << std::setw(10) << stats.stddev;

    if (verify_enabled) {
        std::cout << "  " << (verified ? (std::string(color::GREEN) + "✓ PASS") : (std::string(color::RED) + "✗ FAIL"))
                  << color::RESET;
    }
    std::cout << "\n";
}

void print_summary(const std::map<std::string, BenchStats> &results) {
    if (results.empty()) {
        return;
    }

    // Find best performer
    auto best = std::max_element(results.begin(), results.end(),
                                 [](const auto &a, const auto &b) {
                                     return a.second.gflops_peak < b.second.gflops_peak;
                                 });

    std::cout << "\n";
    print_header_line();
    std::cout << color::BOLD << "  Summary\n"
              << color::RESET;
    print_section_line();
    std::cout << "  🏆 Best Kernel: " << color::GREEN << color::BOLD
              << best->first << color::RESET
              << " @ " << color::YELLOW << std::fixed << std::setprecision(2)
              << best->second.gflops_peak << " GFLOPS" << color::RESET << "\n";
    print_header_line();
}

// ============================================================================
//                              Kernel Runner
// ============================================================================

using KernelFunc = std::function<void(float *, float *, float *, size_t, size_t, size_t)>;

/**
 * @brief Run a single kernel benchmark with warmup and multiple iterations
 *
 * @param name     Kernel display name
 * @param kernel   Function to benchmark
 * @param A, B, C  Matrix buffers
 * @param C_ref    Reference result for verification
 * @param config   Benchmark configuration
 * @return BenchStats containing timing statistics
 */
std::pair<BenchStats, bool> run_benchmark(
    const std::string &name,
    KernelFunc kernel,
    float *A, float *B, float *C, const float *C_ref,
    const BenchConfig &config) {
    const double flops = 2.0 * config.M * config.N * config.K;
    std::vector<double> times;
    times.reserve(config.bench_rounds);

    // ---- Warmup Phase ----
    // Purpose: Stabilize CPU frequency (Turbo Boost), warm up caches,
    // and load code into instruction cache
    for (int i = 0; i < config.warmup_rounds; ++i) {
        kernel(A, B, C, config.M, config.N, config.K);
    }

    // ---- Measurement Phase ----
    for (int i = 0; i < config.bench_rounds; ++i) {
        // Clear output matrix to ensure fair comparison
        std::memset(C, 0, config.M * config.N * sizeof(float));

        auto start = std::chrono::high_resolution_clock::now();
        kernel(A, B, C, config.M, config.N, config.K);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        times.push_back(elapsed.count());
    }

    // ---- Calculate Statistics ----
    BenchStats stats = calculate_stats(times, flops);

    // ---- Verification (if enabled) ----
    bool verified = true;
    if (config.verify) {
        verified = verify_matrix(C_ref, C, config.M * config.N);
    }

    return {stats, verified};
}

// ============================================================================
//                               Main Entry
// ============================================================================

void print_usage(const char *prog) {
    std::cout << "Usage: " << prog << " [M N K] [options]\n"
              << "Options:\n"
              << "  -v, --verify       Enable correctness verification\n"
              << "  -k, --kernel NAME  Filter kernels by name (can be used multiple times,\n"
              << "                     or use comma-separated list: -k core,blas)\n"
              << "  -r, --rounds N     Benchmark rounds (default: 5)\n"
              << "  -w, --warmup N     Warmup rounds (default: 2)\n"
              << "  -h, --help         Show this help\n"
              << "\nExamples:\n"
              << "  " << prog << " 4096 -v                    # 4096³ matrix, verify results\n"
              << "  " << prog << " 2048 -k core -k blas       # Run core and blas kernels\n"
              << "  " << prog << " 1024 -k v5,blas_row -r 10  # Multiple filters, 10 rounds\n";
}

int main(int argc, char **argv) {
    BenchConfig config;

    // ---- Argument Parsing ----
    std::vector<int> dims;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        } else if (arg == "-v" || arg == "--verify") {
            config.verify = true;
        } else if (arg == "-k" || arg == "--kernel") {
            if (++i < argc) {
                // Support comma-separated values: -k core,blas,ref
                auto tokens = split_string(argv[i], ',');
                for (const auto &t : tokens) {
                    config.target_kernels.push_back(t);
                }
            }
        } else if (arg == "-r" || arg == "--rounds") {
            if (++i < argc) {
                config.bench_rounds = std::max(1, std::stoi(argv[i]));
            }
        } else if (arg == "-w" || arg == "--warmup") {
            if (++i < argc) {
                config.warmup_rounds = std::max(0, std::stoi(argv[i]));
            }
        } else {
            try {
                dims.push_back(std::stoi(arg));
            } catch (...) {
                std::cerr << color::YELLOW << "Warning: " << color::RESET
                          << "Ignoring unknown argument: " << arg << "\n";
            }
        }
    }

    // Set matrix dimensions
    if (dims.size() == 1) {
        config.M = config.N = config.K = dims[0];
    } else if (dims.size() >= 3) {
        config.M = dims[0];
        config.N = dims[1];
        config.K = dims[2];
    }

    // ---- Print Banner ----
    print_banner(config);

    // ---- Register Kernels ----
    std::map<std::string, KernelFunc> kernels;

    // Core implementations (A × Bᵀ, Row-Major)
    kernels["core_v1_avx2"] = core::matmul_v1_avx2;
    kernels["core_v2_avx512"] = core::matmul_v2_avx512;
    kernels["core_v3_block"] = core::matmul_v3_avx512_block;
    kernels["core_v4_cache"] = core::matmul_v4_cache_opt;
    kernels["core_v5_parallel"] = core::matmul_v5_parallel;

    // Reference implementations (A × B, Column-Major logic)
    kernels["ref_col_cache"] = ref::matmul_ref_col_cache;
    kernels["ref_col_unroll"] = ref::matmul_ref_col_unroll;
    kernels["ref_col_parallel"] = ref::matmul_ref_col_parallel;

    // OpenBLAS wrappers
    kernels["blas_row_ABT"] = wrapper::openblas_matmul_row_ABT;
    kernels["blas_col_AB"] = wrapper::openblas_matmul_col_AB;

    // ---- Allocate Matrices ----
    std::cout << color::DIM << "Allocating matrices..." << color::RESET << "\n";
    AlignedBuffer<float> A(config.M * config.K);
    AlignedBuffer<float> B(config.K * config.N);
    AlignedBuffer<float> C(config.M * config.N);
    AlignedBuffer<float> C_ref(config.M * config.N);

    // ---- Initialize Data ----
    std::cout << color::DIM << "Initializing with random data..." << color::RESET << "\n";
    randomize_matrix(A.get(), config.M * config.K);
    randomize_matrix(B.get(), config.K * config.N);

    // ---- Compute Reference (if verification enabled) ----
    if (config.verify) {
        std::cout << color::DIM << "Computing reference result..." << color::RESET << "\n";
        matmul_ref(A.get(), B.get(), C_ref.get(), config.M, config.N, config.K);
    }

    std::cout << "\n";

    // ---- Run Benchmarks ----
    print_table_header();

    std::map<std::string, BenchStats> all_results;

    for (const auto &[name, kernel] : kernels) {
        // Apply kernel filter (supports multiple filters)
        if (!matches_filter(name, config.target_kernels)) {
            continue;
        }

        auto [stats, verified] = run_benchmark(
            name, kernel, A.get(), B.get(), C.get(), C_ref.get(), config);

        print_result_row(name, stats, verified, config.verify);
        all_results[name] = stats;
    }

    // ---- Print Summary ----
    print_summary(all_results);

    return 0;
}