#include "utils.h"
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <ctime>

double wall_time() {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

void random_matrix(float *matrix, const size_t M, const size_t N) {
    for (int i = 0; i < M * N; i++) {
        matrix[i] = (float)rand() / RAND_MAX - 0.5f;
    }
}

/* Naive reference implementation for correctness testing (single-threaded)
 * C = A * B^T + C
 * C: [M, N]
 * A: [M, K]
 * B: [N, K]
 */
void matmul_ref(const float *A, const float *B, float *C, const size_t M,
                const size_t N, const size_t K) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[j * K + k];
            }
            C[i * N + j] += sum;
        }
    }
}

/**
 * Check if two float matrices (MxN) are equal within a tolerance.
 * @param ref  Reference matrix (expected values), size M*N
 * @param ans  Actual output matrix, size M*N
 * @param M    Number of rows
 * @param N    Number of columns
 * @param tol  Tolerance for floating-point comparison (default: 1e-5f)
 * @return     0 if all elements match within tolerance, 1 otherwise
 */
void matrix_cmp(const float *ref, const float *ans, const size_t M, const size_t N, const float tol) {
    const size_t total = M * N;
    for (size_t i = 0; i < total; ++i) {
        if (fabsf(ref[i] - ans[i]) > tol) {
            std::printf("Mismatch at %lu: %f vs %f (diff = %f)\n",
                        i, ref[i], ans[i], fabsf(ref[i] - ans[i]));
            return;
        }
    }
    std::printf("PASS\n");
}
