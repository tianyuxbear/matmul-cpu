#pragma once

#include <cstddef>

// Use uppercase 'MIN' to avoid naming collisions with std::min or system headers.
#define MIN(x, y) ((x) < (y) ? (x) : (y))

// Branch prediction hints for the compiler.
// likely(x):   Optimizes assuming 'x' is usually true (keeps code in hot path).
// unlikely(x): Optimizes assuming 'x' is usually false (moves code to cold section).
#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

// Fills buffer with random floats in range [-1.0, 1.0]
void randomize_matrix(float *matrix, size_t elements);

/* Naive reference implementation for correctness testing (single-threaded)
 * Operation: C = A * B^T + C
 * * Dimensions:
 * C: [M, N]
 * A: [M, K]
 * B: [N, K]
 * * Layout:
 * Matrices A, B, and C are stored in row-major order.
 */
void matmul_ref(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

/**
 * Check if two float matrices are equal within a tolerance.
 * @param ref       Reference matrix (expected values)
 * @param ans       Actual output matrix
 * @param elements  Total number of elements to compare (M * N)
 * @param tol       Tolerance for floating-point comparison (default: 1e-5f)
 * @return          true if all elements match within tolerance, false otherwise
 */
bool verify_matrix(const float *ref, const float *ans, size_t elements, float tol = 5e-4f);