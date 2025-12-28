#include "matmul.hpp"
#include <immintrin.h>

namespace core {

// ==================== AVX-512 Implementation (v2) ==================== //
// Logic:   Dot Product (Inner Product)
// Formula: C = A * B^T + C
// Layout:  A, B, C are Row-Major.
//          Since B is treated as transposed, we access B sequentially (row-wise),
//          which is cache-friendly and ideal for SIMD.
// Width:   Process 16 floats (512 bits) per iteration.
void matmul_v2_avx512(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {

    for (size_t i = 0; i < M; ++i) {
        for (size_t j = 0; j < N; ++j) {

            // 1. Initialize accumulator (ZMM register) to zero
            __m512 sum_vec = _mm512_setzero_ps();

            size_t k = 0;
            // 2. Main Vector Loop: Unroll by 16 (width of AVX-512)
            for (; k + 15 < K; k += 16) {
                // Load 16 floats from A and B
                // Note: using loadu (unaligned) is safe and efficient on modern CPUs
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);

                // Fused Multiply-Add: sum_vec += a_vec * b_vec
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            // 3. Horizontal Reduction
            // Sums up all 16 elements inside the ZMM register to a single float.
            // Note: _mm512_reduce_add_ps is convenient but can be a latency bottleneck
            // compared to keeping data in registers (optimized in later blocked versions).
            float current_sum = _mm512_reduce_add_ps(sum_vec);

            // 4. Handle Tail (Remaining elements < 16)
            // Scalar fallback is simple and effective here.
            for (; k < K; ++k) {
                current_sum += A[i * K + k] * B[j * K + k];
            }

            // 5. Update C
            C[i * N + j] += current_sum;
        }
    }
}

} // namespace core