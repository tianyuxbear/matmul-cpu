#include "kernel.h"

inline void store_accum_00(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
}

inline void store_accum_01(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
}

inline void store_accum_02(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
}

inline void store_accum_03(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
}

inline void store_accum_04(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                           __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
}

inline void store_accum_05(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                           __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                           __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
}

inline void store_accum_06(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                           __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                           __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                           __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
}

inline void store_accum_07(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                           __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                           __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                           __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                           __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
}

inline void store_accum_08(float *C,
                           __m512 *C_accum_00_0, __m512 *C_accum_00_1,
                           __m512 *C_accum_01_0, __m512 *C_accum_01_1,
                           __m512 *C_accum_02_0, __m512 *C_accum_02_1,
                           __m512 *C_accum_03_0, __m512 *C_accum_03_1,
                           __m512 *C_accum_04_0, __m512 *C_accum_04_1,
                           __m512 *C_accum_05_0, __m512 *C_accum_05_1,
                           __m512 *C_accum_06_0, __m512 *C_accum_06_1,
                           __m512 *C_accum_07_0, __m512 *C_accum_07_1,
                           __m512 *C_accum_08_0, __m512 *C_accum_08_1,
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
    _mm512_storeu_ps(&C[8 * N], *C_accum_08_0);
    _mm512_storeu_ps(&C[8 * N + 16], *C_accum_08_1);
}

inline void store_accum_09(float *C,
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
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
    _mm512_storeu_ps(&C[8 * N], *C_accum_08_0);
    _mm512_storeu_ps(&C[8 * N + 16], *C_accum_08_1);
    _mm512_storeu_ps(&C[9 * N], *C_accum_09_0);
    _mm512_storeu_ps(&C[9 * N + 16], *C_accum_09_1);
}

inline void store_accum_10(float *C,
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
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
    _mm512_storeu_ps(&C[8 * N], *C_accum_08_0);
    _mm512_storeu_ps(&C[8 * N + 16], *C_accum_08_1);
    _mm512_storeu_ps(&C[9 * N], *C_accum_09_0);
    _mm512_storeu_ps(&C[9 * N + 16], *C_accum_09_1);
    _mm512_storeu_ps(&C[10 * N], *C_accum_10_0);
    _mm512_storeu_ps(&C[10 * N + 16], *C_accum_10_1);
}

inline void store_accum_11(float *C,
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
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
    _mm512_storeu_ps(&C[8 * N], *C_accum_08_0);
    _mm512_storeu_ps(&C[8 * N + 16], *C_accum_08_1);
    _mm512_storeu_ps(&C[9 * N], *C_accum_09_0);
    _mm512_storeu_ps(&C[9 * N + 16], *C_accum_09_1);
    _mm512_storeu_ps(&C[10 * N], *C_accum_10_0);
    _mm512_storeu_ps(&C[10 * N + 16], *C_accum_10_1);
    _mm512_storeu_ps(&C[11 * N], *C_accum_11_0);
    _mm512_storeu_ps(&C[11 * N + 16], *C_accum_11_1);
}

inline void store_accum_12(float *C,
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
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
    _mm512_storeu_ps(&C[8 * N], *C_accum_08_0);
    _mm512_storeu_ps(&C[8 * N + 16], *C_accum_08_1);
    _mm512_storeu_ps(&C[9 * N], *C_accum_09_0);
    _mm512_storeu_ps(&C[9 * N + 16], *C_accum_09_1);
    _mm512_storeu_ps(&C[10 * N], *C_accum_10_0);
    _mm512_storeu_ps(&C[10 * N + 16], *C_accum_10_1);
    _mm512_storeu_ps(&C[11 * N], *C_accum_11_0);
    _mm512_storeu_ps(&C[11 * N + 16], *C_accum_11_1);
    _mm512_storeu_ps(&C[12 * N], *C_accum_12_0);
    _mm512_storeu_ps(&C[12 * N + 16], *C_accum_12_1);
}

inline void store_accum_13(float *C,
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
                           int N) {
    _mm512_storeu_ps(C, *C_accum_00_0);
    _mm512_storeu_ps(&C[16], *C_accum_00_1);
    _mm512_storeu_ps(&C[N], *C_accum_01_0);
    _mm512_storeu_ps(&C[N + 16], *C_accum_01_1);
    _mm512_storeu_ps(&C[2 * N], *C_accum_02_0);
    _mm512_storeu_ps(&C[2 * N + 16], *C_accum_02_1);
    _mm512_storeu_ps(&C[3 * N], *C_accum_03_0);
    _mm512_storeu_ps(&C[3 * N + 16], *C_accum_03_1);
    _mm512_storeu_ps(&C[4 * N], *C_accum_04_0);
    _mm512_storeu_ps(&C[4 * N + 16], *C_accum_04_1);
    _mm512_storeu_ps(&C[5 * N], *C_accum_05_0);
    _mm512_storeu_ps(&C[5 * N + 16], *C_accum_05_1);
    _mm512_storeu_ps(&C[6 * N], *C_accum_06_0);
    _mm512_storeu_ps(&C[6 * N + 16], *C_accum_06_1);
    _mm512_storeu_ps(&C[7 * N], *C_accum_07_0);
    _mm512_storeu_ps(&C[7 * N + 16], *C_accum_07_1);
    _mm512_storeu_ps(&C[8 * N], *C_accum_08_0);
    _mm512_storeu_ps(&C[8 * N + 16], *C_accum_08_1);
    _mm512_storeu_ps(&C[9 * N], *C_accum_09_0);
    _mm512_storeu_ps(&C[9 * N + 16], *C_accum_09_1);
    _mm512_storeu_ps(&C[10 * N], *C_accum_10_0);
    _mm512_storeu_ps(&C[10 * N + 16], *C_accum_10_1);
    _mm512_storeu_ps(&C[11 * N], *C_accum_11_0);
    _mm512_storeu_ps(&C[11 * N + 16], *C_accum_11_1);
    _mm512_storeu_ps(&C[12 * N], *C_accum_12_0);
    _mm512_storeu_ps(&C[12 * N + 16], *C_accum_12_1);
    _mm512_storeu_ps(&C[13 * N], *C_accum_13_0);
    _mm512_storeu_ps(&C[13 * N + 16], *C_accum_13_1);
}