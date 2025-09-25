#pragma once
#include <cstddef>

#define CHECK
// #undef CHECK

const size_t M = 4096;
const size_t N = 4096;
const size_t K = 4096;
const size_t FLOPs = 2 * M * N * K;

double wall_time();
void random_matrix(float *matrix, const size_t M, const size_t N);
void matmul_ref(const float *A, const float *B, float *C, const size_t M, const size_t N, const size_t K);
void matrix_cmp(const float *ref, const float *ans, const size_t M, const size_t N, const float tol = 1e-4f);