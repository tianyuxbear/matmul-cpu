#include "matmul.hpp"
#include "utils.hpp"

#include <chrono>
#include <cstring>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

// For _mm_malloc / _mm_free
#include <immintrin.h>

// =============================================================================
// Infrastructure: Aligned Memory Management
// =============================================================================
// AVX-512 requires 64-byte alignment. std::vector does not guarantee this.
template <typename T>
struct AlignedBuffer {
    T *data = nullptr;
    size_t size = 0;

    AlignedBuffer(size_t n) : size(n) {
        // Allocate 64-byte aligned memory
        data = static_cast<T *>(_mm_malloc(n * sizeof(T), 64));
        if (!data) {
            throw std::runtime_error("Memory allocation failed");
        }
        // Initialize to zero to avoid NaN issues
        std::memset(data, 0, n * sizeof(T));
    }

    ~AlignedBuffer() {
        if (data) {
            _mm_free(data);
        }
    }

    // Disable copy to prevent double-free, allow move
    AlignedBuffer(const AlignedBuffer &) = delete;
    AlignedBuffer &operator=(const AlignedBuffer &) = delete;

    T *get() { return data; }
    const T *get() const { return data; }
    T &operator[](size_t i) { return data[i]; }
};

// =============================================================================
// Benchmarking Framework
// =============================================================================

// Define a generic function pointer type that matches all kernels.
// Note: We use non-const float* for inputs in the signature to accommodate
// OpenBLAS/Ref signatures, but Core kernels can still accept them.
using KernelFunc = std::function<void(float *, float *, float *, size_t, size_t, size_t)>;

struct BenchConfig {
    size_t M = 1024;
    size_t N = 1024;
    size_t K = 1024;
    bool verify = false;
    std::string target_kernel = ""; // Empty means run all
};

void run_test(const std::string &name, KernelFunc kernel,
              float *A, float *B, float *C, const float *C_ref,
              const BenchConfig &config) {

    // Warmup (optional, purely to load code into I-Cache)
    // kernel(A, B, C, config.M, config.N, config.K);

    auto start = std::chrono::high_resolution_clock::now();

    kernel(A, B, C, config.M, config.N, config.K);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;

    double gflops = (2.0 * config.M * config.N * config.K) / (diff.count() * 1e9);

    // Print Results
    std::cout << std::left << std::setw(25) << name
              << "| Time: " << std::fixed << std::setprecision(4) << diff.count() << "s "
              << "| GFLOPS: " << std::setw(8) << gflops
              << std::endl;

    if (config.verify) {
        // Note: Verification might fail if layouts (Row vs Col) don't match
        // the reference implementation's logic.
        bool passed = verify_matrix(C_ref, C, config.M * config.N);
        std::cout << "verify" << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    }
    std::cout << std::endl;
}

// =============================================================================
// Main Entry
// =============================================================================

int main(int argc, char **argv) {
    BenchConfig config;

    // --- 1. Argument Parsing ---
    std::vector<int> dims;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--verify" || arg == "-v") {
            config.verify = true;
        } else if (arg == "--kernel" || arg == "-k") {
            if (i + 1 < argc) {
                config.target_kernel = argv[++i];
            } else {
                std::cerr << "Error: -k requires a kernel name." << std::endl;
                return 1;
            }
        } else {
            try {
                dims.push_back(std::stoi(arg));
            } catch (...) {
                std::cerr << "Ignoring unknown argument: " << arg << std::endl;
            }
        }
    }

    if (dims.size() == 1) {
        config.M = config.N = config.K = dims[0];
    } else if (dims.size() == 3) {
        config.M = dims[0];
        config.N = dims[1];
        config.K = dims[2];
    }

    std::cout << "========================================================" << std::endl;
    std::cout << " MATMUL Benchmark" << std::endl;
    std::cout << " Dims: M=" << config.M << ", N=" << config.N << ", K=" << config.K << std::endl;
    std::cout << " Verify: " << (config.verify ? "ON" : "OFF") << std::endl;
    if (!config.target_kernel.empty()) {
        std::cout << " Filter: " << config.target_kernel << std::endl;
    }
    std::cout << "========================================================" << std::endl;

    // --- 2. Kernel Registration ---
    // Using a map to organize benchmarks.
    // Key: Display Name, Value: Function Pointer
    std::map<std::string, KernelFunc> kernels;

    // Part 1: Core (A * B^T)
    kernels["core_v1_avx2"] = core::matmul_v1_avx2;
    kernels["core_v2_avx512"] = core::matmul_v2_avx512;
    kernels["core_v3_block"] = core::matmul_v3_avx512_block;
    kernels["core_v4_cache"] = core::matmul_v4_cache_opt;
    kernels["core_v5_parallel"] = core::matmul_v5_parallel;

    // Part 2: Reference (A * B, Col-Major logic)
    // Note: To test these fairly against Core, input data layout might need handling,
    // but for raw GFLOPS speed, we just pass the pointers.
    kernels["ref_col_cache"] = ref::matmul_ref_col_cache;
    kernels["ref_col_unroll"] = ref::matmul_ref_col_unroll;
    kernels["ref_col_parallel"] = ref::matmul_ref_col_parallel;

    // Part 3: OpenBLAS
    kernels["blas_row_ABT"] = wrapper::openblas_matmul_row_ABT;
    kernels["blas_col_AB"] = wrapper::openblas_matmul_col_AB;

    // --- 3. Data Preparation ---
    // Use aligned buffers for AVX-512 safety
    AlignedBuffer<float> A(config.M * config.K);
    AlignedBuffer<float> B(config.K * config.N); // or N*K for ABT, size is same
    AlignedBuffer<float> C(config.M * config.N);
    AlignedBuffer<float> C_ref(config.M * config.N);

    // Initialize Data
    randomize_matrix(A.get(), config.M * config.K);
    randomize_matrix(B.get(), config.K * config.N);

    // Calculate Reference (Only if verification is needed)
    // Warning: verify_matrix checks C vs C_ref.
    // Ensure your Reference Impl matches the Logic (A*B vs A*B^T) of the kernel being tested.
    if (config.verify) {
        std::cout << "Calculating Golden Reference (Naive)..." << std::endl;
        // Assuming utils::matmul_ref is the absolute truth (A * B)
        // If your core kernels are A * B^T, this verification will FAIL unless B is symmetric or pre-transposed.
        // For now, we assume standard usage.
        matmul_ref(A.get(), B.get(), C_ref.get(), config.M, config.N, config.K);
    }

    // --- 4. Execution Loop ---
    for (const auto &[name, kernel] : kernels) {
        // Filter logic: if target is set, only run matching kernels
        if (!config.target_kernel.empty() && name.find(config.target_kernel) == std::string::npos) {
            continue;
        }

        run_test(name, kernel, A.get(), B.get(), C.get(), C_ref.get(), config);
    }

    return 0;
}