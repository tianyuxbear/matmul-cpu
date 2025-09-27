#include <immintrin.h>

#define MR 14
#define NR 32

void micro_kernel_unroll(float *blockA_packed, float *blockB_packed,
                                float *C, int mr, int nr, int kc, int N);