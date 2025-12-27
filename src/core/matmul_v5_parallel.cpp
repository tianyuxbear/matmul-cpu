#include "matmul.hpp"
#include "utils.hpp"

#include <immintrin.h>
#include <omp.h>

namespace core {

namespace { // Anonymous namespace for internal linkage

// =========================================================================
// Tuning Parameters (AVX-512 + OpenMP)
// =========================================================================

// Hardcoded thread count for this specific kernel configuration.
// In a production library, this should be dynamic or queried from omp_get_max_threads().
constexpr int NTHREADS = 24;

// Register Blocking Dimensions
constexpr int MR = 14;
constexpr int NR = 32;

// Cache Blocking Dimensions (Tuned for L2/L3)
// Note: MC and NC are scaled by NTHREADS to ensure enough work for all threads.
constexpr int MC = MR * NTHREADS * 5;
constexpr int NC = NR * NTHREADS * 30;
constexpr int KC = 512;

// =========================================================================
// OpenMP Macros
// =========================================================================
// Helper to stringify the thread count for the pragma
#define PRAGMA_STR(x) _Pragma(#x)

// We strictly enforce num_threads to match our blocking logic.
// Usage: PARALLEL_FOR_LOOP
#define PARALLEL_FOR_LOOP PRAGMA_STR(omp parallel for num_threads(NTHREADS))

// =========================================================================
// Global Static Buffers (Aligned)
// =========================================================================
// CAUTION: Using static buffers makes this function NOT thread-safe across
// multiple concurrent calls to matmul_v5_parallel from the host application.
alignas(64) static float blockA_packed[MC * KC];
alignas(64) static float blockB_packed[NC * KC];

// =========================================================================
// Helper Functions
// =========================================================================

static inline __mmask16 create_mask(int remaining) {
    if (remaining <= 0) {
        return 0;
    }
    if (remaining >= 16) {
        return 0xFFFF;
    }
    return (1u << remaining) - 1;
}

/**
 * @brief Packs a panel of Matrix A (Single Threaded Logic).
 * Called by pack_blockA inside a parallel loop.
 */
static inline void pack_panelA(const float *A, float *packed_ptr, int mr, int kc, int K_stride) {
    for (int k = 0; k < kc; ++k) {
        for (int i = 0; i < mr; ++i) {
            *packed_ptr++ = A[i * K_stride + k];
        }
        for (int i = mr; i < MR; ++i) {
            *packed_ptr++ = 0.0f;
        }
    }
}

/**
 * @brief Packs a block of Matrix A in PARALLEL.
 * Splitting the packing workload speeds up the pre-processing phase.
 */
static inline void pack_blockA(const float *A, float *packed_buffer, int mc, int kc, int K_stride) {
    PARALLEL_FOR_LOOP
    for (int i = 0; i < mc; i += MR) {
        int mr = MIN(MR, mc - i);
        // Each thread packs a distinct panel, no data race on packed_buffer.
        pack_panelA(&A[i * K_stride], &packed_buffer[i * kc], mr, kc, K_stride);
    }
}

/**
 * @brief Packs a panel of Matrix B (Single Threaded Logic).
 */
static inline void pack_panelB(const float *B, float *packed_ptr, int nr, int kc, int K_stride) {
    for (int k = 0; k < kc; ++k) {
        for (int j = 0; j < nr; ++j) {
            *packed_ptr++ = B[j * K_stride + k];
        }
        for (int j = nr; j < NR; ++j) {
            *packed_ptr++ = 0.0f;
        }
    }
}

/**
 * @brief Packs a block of Matrix B in PARALLEL.
 */
static inline void pack_blockB(const float *B, float *packed_buffer, int nc, int kc, int K_stride) {
    PARALLEL_FOR_LOOP
    for (int j = 0; j < nc; j += NR) {
        int nr = MIN(NR, nc - j);
        pack_panelB(&B[j * K_stride], &packed_buffer[j * kc], nr, kc, K_stride);
    }
}

// =========================================================================
// Register Accumulation Helpers
// =========================================================================

static inline void load_accum(float *C, __m512 C_accum[MR][2], int N_stride, int mr) {
    for (int i = 0; i < mr; ++i) {
        C_accum[i][0] = _mm512_loadu_ps(&C[i * N_stride]);
        C_accum[i][1] = _mm512_loadu_ps(&C[i * N_stride + 16]);
    }
}

static inline void maskload_accum(float *C, __m512 C_accum[MR][2], int N_stride, int mr,
                                  __mmask16 mask0, __mmask16 mask1) {
    for (int i = 0; i < mr; ++i) {
        C_accum[i][0] = _mm512_maskz_loadu_ps(mask0, &C[i * N_stride]);
        C_accum[i][1] = _mm512_maskz_loadu_ps(mask1, &C[i * N_stride + 16]);
    }
}

static inline void store_accum(float *C, __m512 C_accum[MR][2], int N_stride, int mr) {
    for (int i = 0; i < mr; ++i) {
        _mm512_storeu_ps(&C[i * N_stride], C_accum[i][0]);
        _mm512_storeu_ps(&C[i * N_stride + 16], C_accum[i][1]);
    }
}

static inline void maskstore_accum(float *C, __m512 C_accum[MR][2], int N_stride, int mr,
                                   __mmask16 mask0, __mmask16 mask1) {
    for (int i = 0; i < mr; ++i) {
        _mm512_mask_storeu_ps(&C[i * N_stride], mask0, C_accum[i][0]);
        _mm512_mask_storeu_ps(&C[i * N_stride + 16], mask1, C_accum[i][1]);
    }
}

/**
 * @brief Core AVX-512 FMA Loop.
 * Unrolled to maximize pipeline throughput.
 */
static inline void fma_loop(float *packedA, float *packedB,
                            __m512 C_accum[MR][2],
                            __m512 *va, __m512 *vb0, __m512 *vb1, int kc) {
    for (int k = 0; k < kc; ++k) {
        *vb0 = _mm512_loadu_ps(packedB);
        *vb1 = _mm512_loadu_ps(packedB + 16);

#define UNROLL_FMA(i)                                          \
    *va = _mm512_set1_ps(packedA[i]);                          \
    C_accum[i][0] = _mm512_fmadd_ps(*va, *vb0, C_accum[i][0]); \
    C_accum[i][1] = _mm512_fmadd_ps(*va, *vb1, C_accum[i][1]);

        UNROLL_FMA(0);
        UNROLL_FMA(1);
        UNROLL_FMA(2);
        UNROLL_FMA(3);
        UNROLL_FMA(4);
        UNROLL_FMA(5);
        UNROLL_FMA(6);
        UNROLL_FMA(7);
        UNROLL_FMA(8);
        UNROLL_FMA(9);
        UNROLL_FMA(10);
        UNROLL_FMA(11);
        UNROLL_FMA(12);
        UNROLL_FMA(13);

#undef UNROLL_FMA

        packedA += MR;
        packedB += NR;
    }
}

/**
 * @brief Micro-Kernel (Thread-Safe context).
 * Computes a tile of C using pre-packed A and B buffers.
 */
static inline void micro_kernel(float *packedA, float *packedB, float *C,
                                int mr, int nr, int kc, int N_stride) {
    __m512 C_accum[MR][2];
    __m512 a_reg, b0_reg, b1_reg;

    if (likely(nr == NR)) {
        load_accum(C, C_accum, N_stride, mr);
        fma_loop(packedA, packedB, C_accum, &a_reg, &b0_reg, &b1_reg, kc);
        store_accum(C, C_accum, N_stride, mr);
    } else {
        __mmask16 mask0 = create_mask(nr);
        __mmask16 mask1 = create_mask(nr - 16);

        maskload_accum(C, C_accum, N_stride, mr, mask0, mask1);
        fma_loop(packedA, packedB, C_accum, &a_reg, &b0_reg, &b1_reg, kc);
        maskstore_accum(C, C_accum, N_stride, mr, mask0, mask1);
    }
}

} // anonymous namespace

// =============================================================================
// Public API Implementation
// =============================================================================

// Optimization: Cache Blocking + Packing + Register Blocking + OpenMP Parallelism
// C = A * B^T + C
// Layout: Row-Major
void matmul_v5_parallel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {

    // 1. Loop over N (L3 Cache Blocking for B)
    for (size_t j = 0; j < N; j += NC) {
        int nc = MIN(N - j, NC);

        // 2. Loop over K (Reduction Dimension)
        for (size_t p = 0; p < K; p += KC) {
            int kc = MIN(K - p, KC);

            // Pack B in PARALLEL
            // Threads work on disjoint parts of blockB_packed -> Thread Safe.
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

            // 3. Loop over M (L2 Cache Blocking for A)
            for (size_t i = 0; i < M; i += MC) {
                int mc = MIN(M - i, MC);

                // Pack A in PARALLEL
                // Threads work on disjoint parts of blockA_packed -> Thread Safe.
                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, K);

                // 4. Parallel Compute Loop
                // Threads divide the columns (jr) of the current block.
                // All threads read from the shared blockA_packed and blockB_packed (Read-Only).
                // Threads write to disjoint parts of C (Write-Safe).
                PARALLEL_FOR_LOOP
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = MIN(NR, nc - jr);

                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = MIN(MR, mc - ir);

                        micro_kernel(
                            &blockA_packed[ir * kc],
                            &blockB_packed[jr * kc],
                            &C[(i + ir) * N + (j + jr)],
                            mr, nr, kc, N);
                    }
                }
            }
        }
    }
}

} // namespace core