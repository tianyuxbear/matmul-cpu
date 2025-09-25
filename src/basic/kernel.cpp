#include "matmul.h"
#include <immintrin.h>

#define min(x, y) ((x) < (y) ? (x) : (y))
#define MR 16
#define NR 16

void micro_kernel(float *A_start, float *B_start, float *C_start, int M, int N,
                  int K) {
    for (int i = 0; i < MR; ++i) {
        for (int j = 0; j < NR; ++j) {
            __m512 sum_vec = _mm512_setzero_ps();
            int k = 0;

            for (; k + 15 < K; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A_start[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B_start[j * K + k]);
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);
            C_start[i * N + j] += sum;
        }
    }
}

// ==================== AVX512 kernel impl ==================== //
// C = A * B^T + C, all row-major
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul_avx512_kernel(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i += MR) {
        for (int j = 0; j < N; j += NR) {
            micro_kernel(&A[i * K], &B[j * K], &C[i * N + j], M, N, K);
        }
    }
}