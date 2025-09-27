#include "kernel.h"

// maskstore_accum_00
inline void maskstore_accum_00(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
}

// maskstore_accum_01
inline void maskstore_accum_01(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
}

// maskstore_accum_02
inline void maskstore_accum_02(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
}

// maskstore_accum_03
inline void maskstore_accum_03(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
}

// maskstore_accum_04
inline void maskstore_accum_04(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                               __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
}

// maskstore_accum_05
inline void maskstore_accum_05(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                               __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                               __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
}

// maskstore_accum_06
inline void maskstore_accum_06(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                               __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                               __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                               __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
}

// maskstore_accum_07
inline void maskstore_accum_07(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                               __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                               __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                               __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                               __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
}

// maskstore_accum_08
inline void maskstore_accum_08(float *C,
                               __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                               __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                               __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                               __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                               __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                               __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                               __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                               __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                               __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
    _mm512_mask_storeu_ps(&C[8 * N], packed_mask_0, *C_accum_08_0);
    _mm512_mask_storeu_ps(&C[8 * N + 16], packed_mask_1, *C_accum_08_1);
}

// maskstore_accum_09
inline void maskstore_accum_09(float *C,
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
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
    _mm512_mask_storeu_ps(&C[8 * N], packed_mask_0, *C_accum_08_0);
    _mm512_mask_storeu_ps(&C[8 * N + 16], packed_mask_1, *C_accum_08_1);
    _mm512_mask_storeu_ps(&C[9 * N], packed_mask_0, *C_accum_09_0);
    _mm512_mask_storeu_ps(&C[9 * N + 16], packed_mask_1, *C_accum_09_1);
}

// maskstore_accum_10
inline void maskstore_accum_10(float *C,
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
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
    _mm512_mask_storeu_ps(&C[8 * N], packed_mask_0, *C_accum_08_0);
    _mm512_mask_storeu_ps(&C[8 * N + 16], packed_mask_1, *C_accum_08_1);
    _mm512_mask_storeu_ps(&C[9 * N], packed_mask_0, *C_accum_09_0);
    _mm512_mask_storeu_ps(&C[9 * N + 16], packed_mask_1, *C_accum_09_1);
    _mm512_mask_storeu_ps(&C[10 * N], packed_mask_0, *C_accum_10_0);
    _mm512_mask_storeu_ps(&C[10 * N + 16], packed_mask_1, *C_accum_10_1);
}

// maskstore_accum_11
inline void maskstore_accum_11(float *C,
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
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
    _mm512_mask_storeu_ps(&C[8 * N], packed_mask_0, *C_accum_08_0);
    _mm512_mask_storeu_ps(&C[8 * N + 16], packed_mask_1, *C_accum_08_1);
    _mm512_mask_storeu_ps(&C[9 * N], packed_mask_0, *C_accum_09_0);
    _mm512_mask_storeu_ps(&C[9 * N + 16], packed_mask_1, *C_accum_09_1);
    _mm512_mask_storeu_ps(&C[10 * N], packed_mask_0, *C_accum_10_0);
    _mm512_mask_storeu_ps(&C[10 * N + 16], packed_mask_1, *C_accum_10_1);
    _mm512_mask_storeu_ps(&C[11 * N], packed_mask_0, *C_accum_11_0);
    _mm512_mask_storeu_ps(&C[11 * N + 16], packed_mask_1, *C_accum_11_1);
}

// maskstore_accum_12
inline void maskstore_accum_12(float *C,
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
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
    _mm512_mask_storeu_ps(&C[8 * N], packed_mask_0, *C_accum_08_0);
    _mm512_mask_storeu_ps(&C[8 * N + 16], packed_mask_1, *C_accum_08_1);
    _mm512_mask_storeu_ps(&C[9 * N], packed_mask_0, *C_accum_09_0);
    _mm512_mask_storeu_ps(&C[9 * N + 16], packed_mask_1, *C_accum_09_1);
    _mm512_mask_storeu_ps(&C[10 * N], packed_mask_0, *C_accum_10_0);
    _mm512_mask_storeu_ps(&C[10 * N + 16], packed_mask_1, *C_accum_10_1);
    _mm512_mask_storeu_ps(&C[11 * N], packed_mask_0, *C_accum_11_0);
    _mm512_mask_storeu_ps(&C[11 * N + 16], packed_mask_1, *C_accum_11_1);
    _mm512_mask_storeu_ps(&C[12 * N], packed_mask_0, *C_accum_12_0);
    _mm512_mask_storeu_ps(&C[12 * N + 16], packed_mask_1, *C_accum_12_1);
}

// maskstore_accum_13
inline void maskstore_accum_13(float *C,
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
                               __mmask16 packed_mask_0, __mmask16 packed_mask_1,
                               int N) {
    _mm512_mask_storeu_ps(C, packed_mask_0, *C_accum_00_0);
    _mm512_mask_storeu_ps(&C[16], packed_mask_1, *C_accum_00_1);
    _mm512_mask_storeu_ps(&C[N], packed_mask_0, *C_accum_01_0);
    _mm512_mask_storeu_ps(&C[N + 16], packed_mask_1, *C_accum_01_1);
    _mm512_mask_storeu_ps(&C[2 * N], packed_mask_0, *C_accum_02_0);
    _mm512_mask_storeu_ps(&C[2 * N + 16], packed_mask_1, *C_accum_02_1);
    _mm512_mask_storeu_ps(&C[3 * N], packed_mask_0, *C_accum_03_0);
    _mm512_mask_storeu_ps(&C[3 * N + 16], packed_mask_1, *C_accum_03_1);
    _mm512_mask_storeu_ps(&C[4 * N], packed_mask_0, *C_accum_04_0);
    _mm512_mask_storeu_ps(&C[4 * N + 16], packed_mask_1, *C_accum_04_1);
    _mm512_mask_storeu_ps(&C[5 * N], packed_mask_0, *C_accum_05_0);
    _mm512_mask_storeu_ps(&C[5 * N + 16], packed_mask_1, *C_accum_05_1);
    _mm512_mask_storeu_ps(&C[6 * N], packed_mask_0, *C_accum_06_0);
    _mm512_mask_storeu_ps(&C[6 * N + 16], packed_mask_1, *C_accum_06_1);
    _mm512_mask_storeu_ps(&C[7 * N], packed_mask_0, *C_accum_07_0);
    _mm512_mask_storeu_ps(&C[7 * N + 16], packed_mask_1, *C_accum_07_1);
    _mm512_mask_storeu_ps(&C[8 * N], packed_mask_0, *C_accum_08_0);
    _mm512_mask_storeu_ps(&C[8 * N + 16], packed_mask_1, *C_accum_08_1);
    _mm512_mask_storeu_ps(&C[9 * N], packed_mask_0, *C_accum_09_0);
    _mm512_mask_storeu_ps(&C[9 * N + 16], packed_mask_1, *C_accum_09_1);
    _mm512_mask_storeu_ps(&C[10 * N], packed_mask_0, *C_accum_10_0);
    _mm512_mask_storeu_ps(&C[10 * N + 16], packed_mask_1, *C_accum_10_1);
    _mm512_mask_storeu_ps(&C[11 * N], packed_mask_0, *C_accum_11_0);
    _mm512_mask_storeu_ps(&C[11 * N + 16], packed_mask_1, *C_accum_11_1);
    _mm512_mask_storeu_ps(&C[12 * N], packed_mask_0, *C_accum_12_0);
    _mm512_mask_storeu_ps(&C[12 * N + 16], packed_mask_1, *C_accum_12_1);
    _mm512_mask_storeu_ps(&C[13 * N], packed_mask_0, *C_accum_13_0);
    _mm512_mask_storeu_ps(&C[13 * N + 16], packed_mask_1, *C_accum_13_1);
}
