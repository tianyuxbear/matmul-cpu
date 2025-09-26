#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>

#define min(x, y) ((x) < (y) ? (x) : (y))

#define MC 840
#define NC 1024
#define KC 384

#define MR 14
#define NR 32

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

__mmask16 create_mask(int nr) {
    nr = (nr < 0) ? 0 : (nr > 16) ? 16
                                  : nr;
    return _cvtu32_mask16((1u << nr) - 1);
}

static void pack_panelA(float *A, float *blockA_packed, int mr, int kc, int K) {
    for (int p = 0; p < kc; ++p) {
        for (int i = 0; i < mr; ++i) {
            *blockA_packed++ = A[i * K + p];
        }
        for (int i = mr; i < MR; ++i) {
            *blockA_packed++ = 0;
        }
    }
}

static void pack_blockA(float *A, float *blockA_packed, int mc, int kc, int K) {
    for (int i = 0; i < mc; i += MR) {
        int mr = min(MR, mc - i);
        pack_panelA(&A[i * K], &blockA_packed[i * kc], mr, kc, K);
    }
}

static void pack_panelB(float *B, float *blockB_packed, int nr, int kc, int K) {

    for (int p = 0; p < kc; ++p) {
        for (int j = 0; j < nr; ++j) {
            *blockB_packed++ = B[j * K + p];
        }
        for (int j = nr; j < NR; ++j) {
            *blockB_packed++ = 0;
        }
    }
}
static void pack_blockB(float *B, float *blockB_packed, int nc, int kc, int K) {
    for (int j = 0; j < nc; j += NR) {
        int nr = min(NR, nc - j);
        pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
    }
}

inline void load_accum(float *C, __m512 C_accum[MR][2], int N, int mr) {
    for (int i = 0; i < mr; ++i) {
        C_accum[i][0] = _mm512_loadu_ps(&C[i * N]);
        C_accum[i][1] = _mm512_loadu_ps(&C[i * N + 16]);
    }
}

inline void maskload_accum(float *C, __m512 C_accum[MR][2], int N, int mr, __mmask16 packed_mask_0, __mmask16 packed_mask_1) {
    for (int i = 0; i < mr; ++i) {
        C_accum[i][0] = _mm512_maskz_loadu_ps(packed_mask_0, &C[i * N]);
        C_accum[i][1] = _mm512_maskz_loadu_ps(packed_mask_1, &C[i * N + 16]);
    }
}

inline void store_accum(float *C, __m512 C_accum[MR][2], int N, int mr) {
    for (int i = 0; i < mr; ++i) {
        _mm512_storeu_ps(&C[i * N], C_accum[i][0]);
        _mm512_storeu_ps(&C[i * N + 16], C_accum[i][1]);
    }
}

inline void maskstore_accum(float *C, __m512 C_accum[MR][2], int N, int mr, __mmask16 packed_mask_0, __mmask16 packed_mask_1) {
    for (int i = 0; i < mr; ++i) {
        _mm512_mask_storeu_ps(&C[i * N], packed_mask_0, C_accum[i][0]);
        _mm512_mask_storeu_ps(&C[i * N + 16], packed_mask_1, C_accum[i][1]);
    }
}

inline void fma_loop(float *blockA_packed, float *blockB_packed,
                     __m512 C_accum[MR][2], __m512 a_packedFloat16,
                     __m512 b0_packedFloat16, __m512 b1_packedFloat16, int kc) {
    for (int p = 0; p < kc; ++p) {
        b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

#define UNROLL_FMA(i)                                                                  \
    a_packedFloat16 = _mm512_set1_ps(blockA_packed[i]);                                \
    C_accum[i][0] = _mm512_fmadd_ps(a_packedFloat16, b0_packedFloat16, C_accum[i][0]); \
    C_accum[i][1] = _mm512_fmadd_ps(a_packedFloat16, b1_packedFloat16, C_accum[i][1]);

        UNROLL_FMA(0)
        UNROLL_FMA(1)
        UNROLL_FMA(2)
        UNROLL_FMA(3)
        UNROLL_FMA(4)
        UNROLL_FMA(5)
        UNROLL_FMA(6)
        UNROLL_FMA(7)
        UNROLL_FMA(8)
        UNROLL_FMA(9)
        UNROLL_FMA(10)
        UNROLL_FMA(11)
        UNROLL_FMA(12)
        UNROLL_FMA(13)

#undef UNROLL_FMA

        blockA_packed += MR;
        blockB_packed += NR;
    }
}
static inline void micro_kernel(float *blockA_packed, float *blockB_packed,
                                float *C, int mr, int nr, int kc, int N) {
    __m512 C_accum[MR][2];
    __m512 a_packedFloat16 = {};
    __m512 b0_packedFloat16 = {};
    __m512 b1_packedFloat16 = {};
    __mmask16 packed_mask_0 = {};
    __mmask16 packed_mask_1 = {};

    if (nr == NR) {
        load_accum(C, C_accum, N, mr);
        fma_loop(blockA_packed, blockB_packed, C_accum, a_packedFloat16,
                 b0_packedFloat16, b1_packedFloat16, kc);
        store_accum(C, C_accum, N, mr);
    } else {
        packed_mask_0 = create_mask(nr);
        packed_mask_1 = create_mask(nr - 16);
        maskload_accum(C, C_accum, N, mr, packed_mask_0, packed_mask_1);
        fma_loop(blockA_packed, blockB_packed, C_accum, a_packedFloat16,
                 b0_packedFloat16, b1_packedFloat16, kc);
        maskstore_accum(C, C_accum, N, mr, packed_mask_0, packed_mask_1);
    }
}

// ==================== Scratch impl ==================== //
// C = A * B^T + C, all row-major
// C: [M, N]
// A: [M, K]
// B: [N, K]
void matmul_avx512_optimized(float *A, float *B, float *C, int M, int N,
                             int K) {
    for (int j = 0; j < N; j += NC) {
        int nc = min(NC, N - j);
        for (int p = 0; p < K; p += KC) {
            int kc = min(KC, K - p);
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
            for (int i = 0; i < M; i += MC) {
                int mc = min(MC, M - i);
                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, K);
                for (int ir = 0; ir < mc; ir += MR) {
                    for (int jr = 0; jr < nc; jr += NR) {
                        int mr = min(MR, mc - ir);
                        int nr = min(NR, nc - jr);
                        micro_kernel(&blockA_packed[kc * ir], &blockB_packed[kc * jr],
                                     &C[(i + ir) * N + (j + jr)], mr, nr, kc, N);
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * K * sizeof(float));
    float *C = (float *)aligned_alloc(32, M * N * sizeof(float));
    memset(C, 0, M * N * sizeof(float));

    random_matrix(A, M, K);
    random_matrix(B, N, K);

    double t1{}, t2{}, time{};
    t1 = wall_time();
    matmul_avx512_optimized(A, B, C, M, N, K);
    t2 = wall_time();
    time = t2 - t1;
    printf("Scratch:  %.6f s,  Perf: %.2f GFLOPS\n", time, FLOPs / (time * 1e9));

#ifdef CHECK
    float *C_ref = (float *)aligned_alloc(32, M * N * sizeof(float));
    memset(C_ref, 0, M * N * sizeof(float));
    matmul_ref(A, B, C_ref, M, N, K);
    matrix_cmp(C_ref, C, M, N);
    free(C_ref);
#endif

    free(A);
    free(B);
    free(C);
    return 0;
}
