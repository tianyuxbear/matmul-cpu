#include <cblas.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

#include "utils.h"

int main(int argc, char **argv) {
    float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * K * sizeof(float));
    float *C = (float *)aligned_alloc(32, M * N * sizeof(float));
    memset(C, 0, M * N * sizeof(float));

    random_matrix(A, M, K);
    random_matrix(B, N, K);

    openblas_set_num_threads(1);

    double t1{}, t2{}, time{};
    t1 = wall_time();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, M, N, K, 1.0f, A, K, B,
                K, 0.0f, C, N);
    t2 = wall_time();
    time = t2 - t1;
    printf("OpenBLAS:  %.6f s,  Perf: %.2f GFLOPS\n", time, FLOPs / (time * 1e9));

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
