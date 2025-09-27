#include "kernel.h"
#include "fma.h"
#include "load.h"
#include "maskload.h"
#include "maskstore.h"
#include "store.h"

inline __mmask16 create_mask(int nr) {
    nr = (nr < 0) ? 0 : (nr > 16) ? 16
                                  : nr;
    return _cvtu32_mask16((1u << nr) - 1);
}

void micro_kernel_unroll(float *blockA_packed, float *blockB_packed,
                         float *C, int mr, int nr, int kc, int N) {
    __m512 C_accum_00_0 = {};
    __m512 C_accum_00_1 = {};
    __m512 C_accum_01_0 = {};
    __m512 C_accum_01_1 = {};
    __m512 C_accum_02_0 = {};
    __m512 C_accum_02_1 = {};
    __m512 C_accum_03_0 = {};
    __m512 C_accum_03_1 = {};
    __m512 C_accum_04_0 = {};
    __m512 C_accum_04_1 = {};
    __m512 C_accum_05_0 = {};
    __m512 C_accum_05_1 = {};
    __m512 C_accum_06_0 = {};
    __m512 C_accum_06_1 = {};
    __m512 C_accum_07_0 = {};
    __m512 C_accum_07_1 = {};
    __m512 C_accum_08_0 = {};
    __m512 C_accum_08_1 = {};
    __m512 C_accum_09_0 = {};
    __m512 C_accum_09_1 = {};
    __m512 C_accum_10_0 = {};
    __m512 C_accum_10_1 = {};
    __m512 C_accum_11_0 = {};
    __m512 C_accum_11_1 = {};
    __m512 C_accum_12_0 = {};
    __m512 C_accum_12_1 = {};
    __m512 C_accum_13_0 = {};
    __m512 C_accum_13_1 = {};
    __m512 a_packedFloat16 = {};
    __m512 b0_packedFloat16 = {};
    __m512 b1_packedFloat16 = {};
    __mmask16 packed_mask_0 = {};
    __mmask16 packed_mask_1 = {};

    if (nr == NR) {
        switch (mr) {
        case 1:
            load_accum_00(C,
                          &C_accum_00_0, &C_accum_00_1,
                          N);
            fma_loop_00(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_00(C,
                           &C_accum_00_0, &C_accum_00_1,
                           N);
            break;

        case 2:
            load_accum_01(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          N);
            fma_loop_01(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_01(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           N);
            break;

        case 3:
            load_accum_02(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          N);
            fma_loop_02(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_02(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           N);
            break;

        case 4:
            load_accum_03(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          N);
            fma_loop_03(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_03(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           N);
            break;

        case 5:
            load_accum_04(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          N);
            fma_loop_04(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_04(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           N);
            break;

        case 6:
            load_accum_05(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          N);
            fma_loop_05(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_05(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           N);
            break;

        case 7:
            load_accum_06(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          N);
            fma_loop_06(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_06(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           N);
            break;

        case 8:
            load_accum_07(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          N);
            fma_loop_07(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_07(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           N);
            break;

        case 9:
            load_accum_08(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          &C_accum_08_0, &C_accum_08_1,
                          N);
            fma_loop_08(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_08(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           &C_accum_08_0, &C_accum_08_1,
                           N);
            break;

        case 10:
            load_accum_09(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          &C_accum_08_0, &C_accum_08_1,
                          &C_accum_09_0, &C_accum_09_1,
                          N);
            fma_loop_09(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_09(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           &C_accum_08_0, &C_accum_08_1,
                           &C_accum_09_0, &C_accum_09_1,
                           N);
            break;

        case 11:
            load_accum_10(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          &C_accum_08_0, &C_accum_08_1,
                          &C_accum_09_0, &C_accum_09_1,
                          &C_accum_10_0, &C_accum_10_1,
                          N);
            fma_loop_10(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_10(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           &C_accum_08_0, &C_accum_08_1,
                           &C_accum_09_0, &C_accum_09_1,
                           &C_accum_10_0, &C_accum_10_1,
                           N);
            break;

        case 12:
            load_accum_11(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          &C_accum_08_0, &C_accum_08_1,
                          &C_accum_09_0, &C_accum_09_1,
                          &C_accum_10_0, &C_accum_10_1,
                          &C_accum_11_0, &C_accum_11_1,
                          N);
            fma_loop_11(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &C_accum_11_0, &C_accum_11_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_11(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           &C_accum_08_0, &C_accum_08_1,
                           &C_accum_09_0, &C_accum_09_1,
                           &C_accum_10_0, &C_accum_10_1,
                           &C_accum_11_0, &C_accum_11_1,
                           N);
            break;

        case 13:
            load_accum_12(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          &C_accum_08_0, &C_accum_08_1,
                          &C_accum_09_0, &C_accum_09_1,
                          &C_accum_10_0, &C_accum_10_1,
                          &C_accum_11_0, &C_accum_11_1,
                          &C_accum_12_0, &C_accum_12_1,
                          N);
            fma_loop_12(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &C_accum_11_0, &C_accum_11_1,
                        &C_accum_12_0, &C_accum_12_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_12(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           &C_accum_08_0, &C_accum_08_1,
                           &C_accum_09_0, &C_accum_09_1,
                           &C_accum_10_0, &C_accum_10_1,
                           &C_accum_11_0, &C_accum_11_1,
                           &C_accum_12_0, &C_accum_12_1,
                           N);
            break;

        case 14:
            load_accum_13(C,
                          &C_accum_00_0, &C_accum_00_1,
                          &C_accum_01_0, &C_accum_01_1,
                          &C_accum_02_0, &C_accum_02_1,
                          &C_accum_03_0, &C_accum_03_1,
                          &C_accum_04_0, &C_accum_04_1,
                          &C_accum_05_0, &C_accum_05_1,
                          &C_accum_06_0, &C_accum_06_1,
                          &C_accum_07_0, &C_accum_07_1,
                          &C_accum_08_0, &C_accum_08_1,
                          &C_accum_09_0, &C_accum_09_1,
                          &C_accum_10_0, &C_accum_10_1,
                          &C_accum_11_0, &C_accum_11_1,
                          &C_accum_12_0, &C_accum_12_1,
                          &C_accum_13_0, &C_accum_13_1,
                          N);
            fma_loop_13(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &C_accum_11_0, &C_accum_11_1,
                        &C_accum_12_0, &C_accum_12_1,
                        &C_accum_13_0, &C_accum_13_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            store_accum_13(C,
                           &C_accum_00_0, &C_accum_00_1,
                           &C_accum_01_0, &C_accum_01_1,
                           &C_accum_02_0, &C_accum_02_1,
                           &C_accum_03_0, &C_accum_03_1,
                           &C_accum_04_0, &C_accum_04_1,
                           &C_accum_05_0, &C_accum_05_1,
                           &C_accum_06_0, &C_accum_06_1,
                           &C_accum_07_0, &C_accum_07_1,
                           &C_accum_08_0, &C_accum_08_1,
                           &C_accum_09_0, &C_accum_09_1,
                           &C_accum_10_0, &C_accum_10_1,
                           &C_accum_11_0, &C_accum_11_1,
                           &C_accum_12_0, &C_accum_12_1,
                           &C_accum_13_0, &C_accum_13_1,
                           N);
            break;
        }

    } else {
        packed_mask_0 = create_mask(nr);
        packed_mask_1 = create_mask(nr - 16);
        switch (mr) {
        case 1:
            maskload_accum_00(C,
                              &C_accum_00_0, &C_accum_00_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_00(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_00(C,
                               &C_accum_00_0, &C_accum_00_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 2:
            maskload_accum_01(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_01(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_01(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 3:
            maskload_accum_02(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_02(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_02(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 4:
            maskload_accum_03(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_03(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_03(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 5:
            maskload_accum_04(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_04(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_04(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 6:
            maskload_accum_05(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_05(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_05(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 7:
            maskload_accum_06(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_06(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_06(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 8:
            maskload_accum_07(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_07(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_07(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 9:
            maskload_accum_08(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              &C_accum_08_0, &C_accum_08_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_08(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_08(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               &C_accum_08_0, &C_accum_08_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 10:
            maskload_accum_09(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              &C_accum_08_0, &C_accum_08_1,
                              &C_accum_09_0, &C_accum_09_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_09(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_09(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               &C_accum_08_0, &C_accum_08_1,
                               &C_accum_09_0, &C_accum_09_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 11:
            maskload_accum_10(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              &C_accum_08_0, &C_accum_08_1,
                              &C_accum_09_0, &C_accum_09_1,
                              &C_accum_10_0, &C_accum_10_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_10(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_10(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               &C_accum_08_0, &C_accum_08_1,
                               &C_accum_09_0, &C_accum_09_1,
                               &C_accum_10_0, &C_accum_10_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 12:
            maskload_accum_11(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              &C_accum_08_0, &C_accum_08_1,
                              &C_accum_09_0, &C_accum_09_1,
                              &C_accum_10_0, &C_accum_10_1,
                              &C_accum_11_0, &C_accum_11_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_11(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &C_accum_11_0, &C_accum_11_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_11(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               &C_accum_08_0, &C_accum_08_1,
                               &C_accum_09_0, &C_accum_09_1,
                               &C_accum_10_0, &C_accum_10_1,
                               &C_accum_11_0, &C_accum_11_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 13:
            maskload_accum_12(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              &C_accum_08_0, &C_accum_08_1,
                              &C_accum_09_0, &C_accum_09_1,
                              &C_accum_10_0, &C_accum_10_1,
                              &C_accum_11_0, &C_accum_11_1,
                              &C_accum_12_0, &C_accum_12_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_12(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &C_accum_11_0, &C_accum_11_1,
                        &C_accum_12_0, &C_accum_12_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_12(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               &C_accum_08_0, &C_accum_08_1,
                               &C_accum_09_0, &C_accum_09_1,
                               &C_accum_10_0, &C_accum_10_1,
                               &C_accum_11_0, &C_accum_11_1,
                               &C_accum_12_0, &C_accum_12_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;

        case 14:
            maskload_accum_13(C,
                              &C_accum_00_0, &C_accum_00_1,
                              &C_accum_01_0, &C_accum_01_1,
                              &C_accum_02_0, &C_accum_02_1,
                              &C_accum_03_0, &C_accum_03_1,
                              &C_accum_04_0, &C_accum_04_1,
                              &C_accum_05_0, &C_accum_05_1,
                              &C_accum_06_0, &C_accum_06_1,
                              &C_accum_07_0, &C_accum_07_1,
                              &C_accum_08_0, &C_accum_08_1,
                              &C_accum_09_0, &C_accum_09_1,
                              &C_accum_10_0, &C_accum_10_1,
                              &C_accum_11_0, &C_accum_11_1,
                              &C_accum_12_0, &C_accum_12_1,
                              &C_accum_13_0, &C_accum_13_1,
                              packed_mask_0, packed_mask_1,
                              N);
            fma_loop_13(blockA_packed, blockB_packed,
                        &C_accum_00_0, &C_accum_00_1,
                        &C_accum_01_0, &C_accum_01_1,
                        &C_accum_02_0, &C_accum_02_1,
                        &C_accum_03_0, &C_accum_03_1,
                        &C_accum_04_0, &C_accum_04_1,
                        &C_accum_05_0, &C_accum_05_1,
                        &C_accum_06_0, &C_accum_06_1,
                        &C_accum_07_0, &C_accum_07_1,
                        &C_accum_08_0, &C_accum_08_1,
                        &C_accum_09_0, &C_accum_09_1,
                        &C_accum_10_0, &C_accum_10_1,
                        &C_accum_11_0, &C_accum_11_1,
                        &C_accum_12_0, &C_accum_12_1,
                        &C_accum_13_0, &C_accum_13_1,
                        &a_packedFloat16,
                        &b0_packedFloat16, &b1_packedFloat16,
                        kc);
            maskstore_accum_13(C,
                               &C_accum_00_0, &C_accum_00_1,
                               &C_accum_01_0, &C_accum_01_1,
                               &C_accum_02_0, &C_accum_02_1,
                               &C_accum_03_0, &C_accum_03_1,
                               &C_accum_04_0, &C_accum_04_1,
                               &C_accum_05_0, &C_accum_05_1,
                               &C_accum_06_0, &C_accum_06_1,
                               &C_accum_07_0, &C_accum_07_1,
                               &C_accum_08_0, &C_accum_08_1,
                               &C_accum_09_0, &C_accum_09_1,
                               &C_accum_10_0, &C_accum_10_1,
                               &C_accum_11_0, &C_accum_11_1,
                               &C_accum_12_0, &C_accum_12_1,
                               &C_accum_13_0, &C_accum_13_1,
                               packed_mask_0, packed_mask_1,
                               N);
            break;
        }
    }
}