#include "utils.hpp"

#include <cmath>    // std::abs
#include <iostream> // std::cerr, std::cout
#include <random>   // std::mt19937, std::uniform_real_distribution

// Fills buffer with random floats in range [-1.0, 1.0]
void randomize_matrix(float *matrix, size_t elements) {
    static std::mt19937 gen(42);
    static std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (size_t i = 0; i < elements; ++i) {
        matrix[i] = dist(gen);
    }
}

/* Naive reference implementation for correctness testing (single-threaded)
 * Operation: C = A * B^T + C
 * * Dimensions:
 * C: [M, N]
 * A: [M, K]
 * B: [N, K]
 * * Layout:
 * Matrices A, B, and C are stored in row-major order.
 */
void matmul_ref(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                // A[i][k] * B[j][k] (Accessing B as N*K row-major)
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] += sum;
        }
    }
}

/**
 * Check if two float matrices are equal within a tolerance.
 * @param ref       Reference matrix (expected values)
 * @param ans       Actual output matrix
 * @param elements  Total number of elements to compare (M * N)
 * @param tol       Tolerance for floating-point comparison (default: 1e-5f)
 * @return          true if all elements match within tolerance, false otherwise
 */
bool verify_matrix(const float *ref, const float *ans, size_t elements, float tol) {
    for (size_t i = 0; i < elements; ++i) {
        float diff = std::abs(ref[i] - ans[i]);
        if (diff > tol) {
            std::cerr << "Mismatch at " << i << ": ref=" << ref[i]
                      << ", ans=" << ans[i] << ", diff=" << diff << std::endl;
            return false;
        }
    }
    return true;
}