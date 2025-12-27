#include "matmul.hpp"
#include <immintrin.h>

namespace core {

// Tile dimensions for the micro-kernel
#define MR 16
#define NR 16

/**
 * @brief Micro-kernel: Computes a 16x16 sub-block of C.
 * * Logic:
 * Iterates through a 16x16 tile. For each element C[row, col],
 * calculates the dot product of A[row] and B[col] using AVX-512.
 * * Note:
 * This is an intermediate "Block + Dot Product" implementation.
 * It improves locality compared to naive versions but still performs
 * heavy reductions per element.
 * @param A_ptr  Pointer to the start of the row-strip in A.
 * @param B_ptr  Pointer to the start of the row-strip in B (conceptually transposed).
 * @param C_tile Pointer to the top-left corner of the 16x16 tile in C.
 * @param N      Leading dimension of C (stride).
 * @param K      K dimension (accumulate loop length).
 */
static inline void micro_kernel_16x16(const float *A_ptr, const float *B_ptr, float *C_tile,
                                      size_t N, size_t K) {
    // Iterate over 16 rows of the tile
    for (int r = 0; r < MR; ++r) {
        // Iterate over 16 columns of the tile
        for (int c = 0; c < NR; ++c) {

            __m512 sum_vec = _mm512_setzero_ps();
            size_t k = 0;

            // Vectorized Dot Product Loop
            for (; k + 15 < K; k += 16) {
                // Load 16 elements from A and B
                // A_ptr points to the block start, offset by current local row 'r'
                __m512 a_vec = _mm512_loadu_ps(&A_ptr[r * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B_ptr[c * K + k]);

                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            // Reduce vector to scalar and handle accumulation
            float sum = _mm512_reduce_add_ps(sum_vec);

            // Tail handling for K (Scalar fallback)
            for (; k < K; ++k) {
                sum += A_ptr[r * K + k] * B_ptr[c * K + k];
            }

            // Update C matrix (Row-Major: stride is N)
            C_tile[r * N + c] += sum;
        }
    }
}

// ==================== AVX-512 Blocked Impl (v3) ==================== //
// Logic:   Tiled GEMM with 16x16 blocking.
// Formula: C = A * B^T + C
// Layout:  All Row-Major.
void matmul_v3_avx512_block(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {
    // Main loop: Iterate over tiles of size MR x NR
    for (size_t i = 0; i < M; i += MR) {
        for (size_t j = 0; j < N; j += NR) {

            // Boundary checks can be added here if M, N are not multiples of 16.
            // Assuming M, N % 16 == 0 for this version.

            // Invoke the micro-kernel for the current tile
            // &A[i * K] points to the start of the i-th row-strip
            // &B[j * K] points to the start of the j-th row-strip (in B's coordinate)
            micro_kernel_16x16(&A[i * K], &B[j * K], &C[i * N + j], N, K);
        }
    }
}

} // namespace core