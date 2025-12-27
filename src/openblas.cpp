#include "matmul.hpp"
#include <cblas.h>

namespace wrapper {

/**
 * @brief Row-Major Wrapper with Transposed B.
 * Formula: C = 1.0 * A * B^T + 1.0 * C (Accumulate)
 * Layout:  Row-Major
 * Input:   A [M x K], B [N x K] (Physically), C [M x N]
 */
void openblas_matmul_row_ABT(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    // cblas_sgemm parameter mapping:
    // 1. Layout: CblasRowMajor
    // 2. TransA: CblasNoTrans (A is used as is)
    // 3. TransB: CblasTrans   (B is physically [N x K], treated as transposed [K x N])
    // 4. M, N, K: Matrix dimensions
    // 5. alpha: 1.0f
    // 6. lda: Stride of A. Row-Major [M x K] -> K
    // 7. ldb: Stride of B. Row-Major [N x K] -> K (Physical row length)
    // 8. beta: 1.0f (Accumulate result into C)
    // 9. ldc: Stride of C. Row-Major [M x N] -> N

    cblas_sgemm(CblasRowMajor,
                CblasNoTrans, CblasTrans,
                M, N, K,
                1.0f,  // alpha
                A, K,  // lda
                B, K,  // ldb
                1.0f,  // beta
                C, N); // ldc
}

/**
 * @brief Pure Column-Major Wrapper (Standard GEMM).
 * Formula: C = 1.0 * A * B + 0.0 * C (Overwrite)
 * Layout:  Column-Major
 * Input:   A [M x K], B [K x N], C [M x N] (Physically Col-Major)
 */
void openblas_matmul_col_AB(float *A, float *B, float *C, size_t M, size_t N, size_t K) {
    // cblas_sgemm parameter mapping:
    // 1. Layout: CblasColMajor
    // 2. TransA/B: CblasNoTrans (Direct multiplication)
    // 3. lda: Stride of A. Col-Major [M x K] -> M
    // 4. ldb: Stride of B. Col-Major [K x N] -> K
    // 5. beta: 0.0f (Overwrite mode, typical for reference checks)
    // 6. ldc: Stride of C. Col-Major [M x N] -> M

    cblas_sgemm(CblasColMajor,
                CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0f,  // alpha
                A, M,  // lda
                B, K,  // ldb
                0.0f,  // beta (Overwrite)
                C, M); // ldc
}

} // namespace wrapper