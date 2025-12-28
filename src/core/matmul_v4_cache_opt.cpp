#include "matmul.hpp"
#include "utils.hpp"

#include <immintrin.h>

namespace core {

// Internal implementation details (hidden from the linker)
namespace {

// =========================================================================
// Tuning Parameters & Constants
// =========================================================================
// Register Blocking: 14x32
// MR = 14 rows of A (accumulated in registers)
// NR = 32 cols of B (accumulated in registers, width of 2 ZMMs)
constexpr int MR = 14;
constexpr int NR = 32;

// Cache Blocking (L2/L3 Cache Tiling)
constexpr int MC = 840;  // Block size for M
constexpr int NC = 1024; // Block size for N
constexpr int KC = 384;  // Block size for K (Reduction dimension)

// =========================================================================
// Global Buffers (Thread-Local / Static)
// =========================================================================
// Packed buffers ensure contiguous memory access during the FMA loop.
// Alignment is critical for AVX-512 loads.
alignas(64) static float blockA_packed[MC * KC];
alignas(64) static float blockB_packed[NC * KC];

// =========================================================================
// Helper Functions
// =========================================================================

/**
 * @brief Creates a mask for AVX-512 operations based on remaining elements.
 */
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
 * @brief Packs a panel of Matrix A into contiguous memory.
 * Layout: [KC, MR] (Row-Major within the block).
 * Goal:   Optimized for broadcasting A elements in the FMA loop.
 */
static inline void pack_panelA(const float *A, float *packed_ptr, int mr, int kc, int K_stride) {
    for (int k = 0; k < kc; ++k) {
        for (int i = 0; i < mr; ++i) {
            *packed_ptr++ = A[i * K_stride + k];
        }
        // Zero-padding if mr < MR
        for (int i = mr; i < MR; ++i) {
            *packed_ptr++ = 0.0f;
        }
    }
}

/**
 * @brief Block-level wrapper for packing A.
 */
static inline void pack_blockA(const float *A, float *packed_buffer, int mc, int kc, int K_stride) {
    for (int i = 0; i < mc; i += MR) {
        int mr = MIN(MR, mc - i);
        pack_panelA(&A[i * K_stride], &packed_buffer[i * kc], mr, kc, K_stride);
    }
}

/**
 * @brief Packs a panel of Matrix B into contiguous memory.
 * Layout: [KC, NR] (Row-Major within the block).
 * Goal:   Optimized for sequential vector loads (ZMM) in the FMA loop.
 */
static inline void pack_panelB(const float *B, float *packed_ptr, int nr, int kc, int K_stride) {
    for (int k = 0; k < kc; ++k) {
        for (int j = 0; j < nr; ++j) {
            *packed_ptr++ = B[j * K_stride + k];
        }
        // Zero-padding if nr < NR
        for (int j = nr; j < NR; ++j) {
            *packed_ptr++ = 0.0f;
        }
    }
}

/**
 * @brief Block-level wrapper for packing B.
 */
static inline void pack_blockB(const float *B, float *packed_buffer, int nc, int kc, int K_stride) {
    for (int j = 0; j < nc; j += NR) {
        int nr = MIN(NR, nc - j);
        pack_panelB(&B[j * K_stride], &packed_buffer[j * kc], nr, kc, K_stride);
    }
}

// =========================================================================
// Micro-Kernel Helpers (Load / Store / FMA)
// =========================================================================

/**
 * @brief Loads C accumulator registers from memory.
 * Each row 'i' has 2 ZMM registers (32 floats total).
 */
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
 * @brief The Core FMA Loop.
 * Computes 14x32 block dot product over the K-dimension (size kc).
 * Uses explicit loop unrolling for maximum pipeline utilization.
 */
static inline void fma_loop(float *packedA, float *packedB,
                            __m512 C_accum[MR][2],
                            __m512 *va, __m512 *vb0, __m512 *vb1, int kc) {

    for (int k = 0; k < kc; ++k) {
        // Load 32 elements of B (2 ZMMs)
        // Since B is packed, these are sequential loads.
        *vb0 = _mm512_loadu_ps(packedB);
        *vb1 = _mm512_loadu_ps(packedB + 16);

// Broadcast A elements and multiply-add
// Unrolling MR (14) times
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

        // Advance pointers
        // A moves by MR (14 floats), B moves by NR (32 floats)
        packedA += MR;
        packedB += NR;
    }
}

/**
 * @brief The Micro-Kernel.
 * Manages register allocation, accumulation, and boundary masking.
 */
static inline void micro_kernel(float *packedA, float *packedB, float *C,
                                int mr, int nr, int kc, int N_stride) {

    __m512 C_accum[MR][2];        // 28 Registers used for Accumulation
    __m512 a_reg, b0_reg, b1_reg; // 3 Registers for operands
    // Total regs used: ~31 (fits in 32 ZMMs)

    if (likely(nr == NR)) {
        load_accum(C, C_accum, N_stride, mr);
        fma_loop(packedA, packedB, C_accum, &a_reg, &b0_reg, &b1_reg, kc);
        store_accum(C, C_accum, N_stride, mr);
    } else {
        // Boundary handling with masks
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

// Optimization: Cache Blocking + Packing + Register Blocking (AVX-512)
// C = A * B^T + C
// Layout: C[M, N], A[M, K], B[N, K] (Row-Major)
void matmul_v4_cache_opt(const float *A, const float *B, float *C, size_t M, size_t N, size_t K) {

    // 1. Loop over N (L3 Cache Blocking for B)
    for (size_t j = 0; j < N; j += NC) {
        int nc = MIN(N - j, NC);

        // 2. Loop over K (Reduction Dimension)
        for (size_t p = 0; p < K; p += KC) {
            int kc = MIN(K - p, KC);

            // Pack Block B into contiguous memory (L2 Cache friendly)
            pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

            // 3. Loop over M (L2 Cache Blocking for A)
            for (size_t i = 0; i < M; i += MC) {
                int mc = MIN(M - i, MC);

                // Pack Block A into contiguous memory
                pack_blockA(&A[i * K + p], blockA_packed, mc, kc, K);

                // 4. Micro-Kernel Loops (Register Blocking)
                // Iterate over the packed blocks
                for (int jr = 0; jr < nc; jr += NR) {
                    int nr = MIN(NR, nc - jr);

                    for (int ir = 0; ir < mc; ir += MR) {
                        int mr = MIN(MR, mc - ir);

                        // Compute 14x32 block
                        micro_kernel(
                            &blockA_packed[ir * kc], // A ptr moves by KC * MR elements
                            &blockB_packed[jr * kc], // B ptr moves by KC * NR elements
                            &C[(i + ir) * N + (j + jr)],
                            mr, nr, kc, N);
                    }
                }
            }
        }
    }
}

} // namespace core