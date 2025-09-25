#include "matmul.h"
#include <immintrin.h>

// ==================== AVX2 impl ==================== //
// C = A * B^T + C, all row-major
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul_avx2(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;

            for (; k + 7 < K; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(&A[i * K + k]);
                __m256 b_vec = _mm256_loadu_ps(&B[j * K + k]);
                sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
            }
            __m128 low = _mm256_castps256_ps128(sum_vec);
            __m128 high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_128 = _mm_add_ps(low, high);

            __m128 shuf = _mm_shuffle_ps(sum_128, sum_128, _MM_SHUFFLE(2, 3, 0, 1));
            __m128 sums = _mm_add_ps(sum_128, shuf);
            shuf = _mm_movehl_ps(shuf, sums);
            sums = _mm_add_ss(sums, shuf);

            float sum = _mm_cvtss_f32(sums);

            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }

            C[i * N + j] += sum;
        }
    }
}

// ==================== AVX512 impl ==================== //
// C = A * B^T + C, all row-major
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul_avx512(float *A, float *B, float *C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            __m512 sum_vec = _mm512_setzero_ps();

            int k = 0;
            for (; k + 15 < K; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(&A[i * K + k]);
                __m512 b_vec = _mm512_loadu_ps(&B[j * K + k]);
                sum_vec = _mm512_fmadd_ps(a_vec, b_vec, sum_vec);
            }

            float sum = _mm512_reduce_add_ps(sum_vec);
            for (; k < K; k++) {
                sum += A[i * K + k] * B[j * K + k];
            }

            C[i * N + j] += sum;
        }
    }
}