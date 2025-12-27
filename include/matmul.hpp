#pragma once
#include <cstddef>

/**
 * =============================================================================
 * Part 1: Core Implementation (My Work)
 * -----------------------------------------------------------------------------
 * Layout:    Row-Major (Standard C++)
 * Operation: C = A * B^T + C
 * A: [M x K], B: [N x K], C: [M x N]
 * =============================================================================
 */
namespace core {

// Step 1: SIMD optimization using AVX2 intrinsics (256-bit)
void matmul_v1_avx2(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

// Step 2: Upgrade to AVX-512 intrinsics (512-bit)
void matmul_v2_avx512(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

// Step 3: AVX-512 with Basic Blocking
void matmul_v3_avx512_block(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

// Step 4: Advanced Cache Optimization (L1/L2/L3 Tiling + Packing/Reordering)
void matmul_v4_cache_opt(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

// Step 5: Multi-threading based on v4 (Cache Optimization + OpenMP/Thread Pool)
void matmul_v5_parallel(const float *A, const float *B, float *C, size_t M, size_t N, size_t K);

} // namespace core

/**
 * =============================================================================
 * Part 2: Reference Implementation (Copied/Studied)
 * -----------------------------------------------------------------------------
 * Layout:    Column-Major (Fortran Style) !!!
 * Operation: C = A * B
 * =============================================================================
 */
namespace ref {

// Ref 1: Basic Cache Blocking optimization (Column-Major)
void matmul_ref_col_cache(float *A, float *B, float *C, int M, int N, int K);

// Ref 2: Micro-kernel Loop Unrolling (Column-Major)
void matmul_ref_col_unroll(float *A, float *B, float *C, int M, int N, int K);

// Ref 3: Cache Blocking + Parallelism (Column-Major)
void matmul_ref_col_parallel(float *A, float *B, float *C, int M, int N, int K);

} // namespace ref

/**
 * =============================================================================
 * Part 3: External Library Wrappers (OpenBLAS)
 * -----------------------------------------------------------------------------
 * Comparison benchmarks against industry standards.
 * =============================================================================
 */
namespace wrapper {

/**
 * @brief Row-Major Wrapper with Transposed B
 * Formula: C = A * B^T + C  (Accumulates!)
 * Layout:  Row-Major
 * Details: Calls cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, ...)
 * Note:    Ideally suited for SIMD where B is naturally accessed sequentially.
 */
void openblas_matmul_row_ABT(float *A, float *B, float *C, size_t M, size_t N, size_t K);

/**
 * @brief Pure Column-Major Wrapper
 * Formula: C = A * B
 * Layout:  Column-Major
 * Details: Calls cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, ...)
 * Note:    Direct comparison for the 'ref' namespace implementations.
 */
void openblas_matmul_col_AB(float *A, float *B, float *C, size_t M, size_t N, size_t K);

} // namespace wrapper