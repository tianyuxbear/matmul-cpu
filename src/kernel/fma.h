#include "kernel.h"

// fma_loop_00
inline void fma_loop_00(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_01
inline void fma_loop_01(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_02
inline void fma_loop_02(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_03
inline void fma_loop_03(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_04
inline void fma_loop_04(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_05
inline void fma_loop_05(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_06
inline void fma_loop_06(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_07
inline void fma_loop_07(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_08
inline void fma_loop_08(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[8]);
        *C_accum_08_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_08_0);
        *C_accum_08_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_08_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_09
inline void fma_loop_09(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                        __m512 *C_accum_09_0, __m512 *C_accum_09_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[8]);
        *C_accum_08_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_08_0);
        *C_accum_08_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_08_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[9]);
        *C_accum_09_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_09_0);
        *C_accum_09_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_09_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_10
inline void fma_loop_10(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                        __m512 *C_accum_09_0, __m512 *C_accum_09_1,
                        __m512 *C_accum_10_0, __m512 *C_accum_10_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[8]);
        *C_accum_08_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_08_0);
        *C_accum_08_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_08_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[9]);
        *C_accum_09_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_09_0);
        *C_accum_09_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_09_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[10]);
        *C_accum_10_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_10_0);
        *C_accum_10_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_10_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_11
inline void fma_loop_11(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                        __m512 *C_accum_09_0, __m512 *C_accum_09_1,
                        __m512 *C_accum_10_0, __m512 *C_accum_10_1,
                        __m512 *C_accum_11_0, __m512 *C_accum_11_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[8]);
        *C_accum_08_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_08_0);
        *C_accum_08_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_08_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[9]);
        *C_accum_09_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_09_0);
        *C_accum_09_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_09_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[10]);
        *C_accum_10_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_10_0);
        *C_accum_10_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_10_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[11]);
        *C_accum_11_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_11_0);
        *C_accum_11_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_11_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_12
inline void fma_loop_12(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                        __m512 *C_accum_09_0, __m512 *C_accum_09_1,
                        __m512 *C_accum_10_0, __m512 *C_accum_10_1,
                        __m512 *C_accum_11_0, __m512 *C_accum_11_1,
                        __m512 *C_accum_12_0, __m512 *C_accum_12_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[8]);
        *C_accum_08_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_08_0);
        *C_accum_08_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_08_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[9]);
        *C_accum_09_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_09_0);
        *C_accum_09_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_09_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[10]);
        *C_accum_10_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_10_0);
        *C_accum_10_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_10_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[11]);
        *C_accum_11_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_11_0);
        *C_accum_11_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_11_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[12]);
        *C_accum_12_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_12_0);
        *C_accum_12_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_12_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}

// fma_loop_13
inline void fma_loop_13(float *blockA_packed, float *blockB_packed,
                        __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                        __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                        __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                        __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                        __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                        __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                        __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                        __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                        __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                        __m512 *C_accum_09_0, __m512 *C_accum_09_1,
                        __m512 *C_accum_10_0, __m512 *C_accum_10_1,
                        __m512 *C_accum_11_0, __m512 *C_accum_11_1,
                        __m512 *C_accum_12_0, __m512 *C_accum_12_1,
                        __m512 *C_accum_13_0, __m512 *C_accum_13_1,
                        __m512 *a_packedFloat16,
                        __m512 *b0_packedFloat16, __m512 *b1_packedFloat16,
                        int kc) {
    for (int p = 0; p < kc; ++p) {
        *b0_packedFloat16 = _mm512_loadu_ps(blockB_packed);
        *b1_packedFloat16 = _mm512_loadu_ps(blockB_packed + 16);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[0]);
        *C_accum_00_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_00_0);
        *C_accum_00_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_00_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[1]);
        *C_accum_01_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_01_0);
        *C_accum_01_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_01_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[2]);
        *C_accum_02_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_02_0);
        *C_accum_02_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_02_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[3]);
        *C_accum_03_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_03_0);
        *C_accum_03_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_03_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[4]);
        *C_accum_04_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_04_0);
        *C_accum_04_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_04_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[5]);
        *C_accum_05_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_05_0);
        *C_accum_05_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_05_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[6]);
        *C_accum_06_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_06_0);
        *C_accum_06_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_06_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[7]);
        *C_accum_07_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_07_0);
        *C_accum_07_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_07_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[8]);
        *C_accum_08_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_08_0);
        *C_accum_08_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_08_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[9]);
        *C_accum_09_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_09_0);
        *C_accum_09_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_09_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[10]);
        *C_accum_10_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_10_0);
        *C_accum_10_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_10_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[11]);
        *C_accum_11_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_11_0);
        *C_accum_11_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_11_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[12]);
        *C_accum_12_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_12_0);
        *C_accum_12_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_12_1);

        *a_packedFloat16 = _mm512_set1_ps(blockA_packed[13]);
        *C_accum_13_0 = _mm512_fmadd_ps(*a_packedFloat16, *b0_packedFloat16, *C_accum_13_0);
        *C_accum_13_1 = _mm512_fmadd_ps(*a_packedFloat16, *b1_packedFloat16, *C_accum_13_1);

        blockA_packed += MR;
        blockB_packed += NR;
    }
}