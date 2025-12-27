#include "matmul.hpp"
#include <immintrin.h>

namespace core {

/**
 * @brief Helper: Horizontal sum of a 256-bit AVX register.
 * Reduces 8 floats in __m256 to a single float.
 */
static inline float hsum_avx2(__m256 v) {
    // 1. Split 256-bit into two 128-bit halves and add them: V[0..3] + V[4..7]
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(lo, hi);

    // 2. Shuffle and add to reduce 128-bit lane
    // sum128 = [A, B, C, D]
    __m128 shuf = _mm_movehdup_ps(sum128);  // [B, B, D, D]
    __m128 sums = _mm_add_ps(sum128, shuf); // [A+B, 2B, C+D, 2D]

    shuf = _mm_movehl_ps(shuf, sums); // [C+D, 2D, D, D]
    sums = _mm_add_ss(sums, shuf);    // [A+B+C+D, ...]

    return _mm_cvtss_f32(sums);
}

// ==================== AVX2 Implementation (v1) ==================== //
// Logic: Dot Product (Inner Product)
// Formula: C = A * B^T + C
// Access Pattern: A [Row-Major] vs B [Row-Major] (conceptually B is transposed)
// Optimization: Vectorized dot product using FMA
void matmul_v1_avx2(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {

            // 1. Vectorized loop for the bulk of K
            __m256 sum_vec = _mm256_setzero_ps();
            size_t k = 0;

            // Unrolling by 8 (one AVX2 register width)
            for (; k + 7 < K; k += 8) {
                // Load 8 floats from A and B (unaligned)
                __m256 vec_a = _mm256_loadu_ps(&A[i * K + k]);
                __m256 vec_b = _mm256_loadu_ps(&B[j * K + k]);

                // FMA: sum_vec += vec_a * vec_b
                sum_vec = _mm256_fmadd_ps(vec_a, vec_b, sum_vec);
            }

            // 2. Reduce vector to scalar
            float current_sum = hsum_avx2(sum_vec);

            // 3. Handle remaining elements (Tail case)
            for (; k < K; ++k) {
                current_sum += A[i * K + k] * B[j * K + k];
            }

            // 4. Accumulate result to C
            C[i * N + j] += current_sum;
        }
    }
}

} // namespace core