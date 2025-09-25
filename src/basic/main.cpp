#include "matmul.h"
#include "utils.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

int main(int argc, char **argv) {
    float *A = (float *)aligned_alloc(32, M * K * sizeof(float));
    float *B = (float *)aligned_alloc(32, N * K * sizeof(float));
    float *C1 = (float *)aligned_alloc(32, M * N * sizeof(float));
    float *C2 = (float *)aligned_alloc(32, M * N * sizeof(float));
    float *C3 = (float *)aligned_alloc(32, M * N * sizeof(float));
    float *C4 = (float *)aligned_alloc(32, M * N * sizeof(float));
    memset(C1, 0, M * N * sizeof(float));
    memset(C2, 0, M * N * sizeof(float));
    memset(C3, 0, M * N * sizeof(float));
    memset(C4, 0, M * N * sizeof(float));

    random_matrix(A, M, K);
    random_matrix(B, N, K);

    double t1{}, t2{}, time{};

    // Naive
    t1 = wall_time();
    matmul_ref(A, B, C1, M, N, K);
    t2 = wall_time();
    time = t2 - t1;
    printf("Naive: %.6f s, Perf: %.2f GFLOPS\n", time, FLOPs / (time * 1e9));

    // AVX2
    t1 = wall_time();
    matmul_avx2(A, B, C2, M, N, K);
    t2 = wall_time();
    time = t2 - t1;
    printf("AVX2: %.6f s, Perf: %.2f GFLOPS\n", time, FLOPs / (time * 1e9));
    matrix_cmp(C1, C2, M, N);

    // AVX512
    t1 = wall_time();
    matmul_avx512(A, B, C3, M, N, K);
    t2 = wall_time();
    time = t2 - t1;
    printf("AVX512: %.6f s, Perf: %.2f GFLOPS\n", time, FLOPs / (time * 1e9));
    matrix_cmp(C1, C2, M, N);

    // AVX512_Kernel
    t1 = wall_time();
    matmul_avx512_kernel(A, B, C4, M, N, K);
    t2 = wall_time();
    time = t2 - t1;
    printf("AVX512_Kernel: %.6f s, Perf: %.2f GFLOPS\n", time, FLOPs / (time * 1e9));
    matrix_cmp(C1, C2, M, N);

    free(A);
    free(B);
    free(C1);
    free(C2);
    free(C3);
    free(C4);
    return 0;
}
