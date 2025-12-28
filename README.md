# High-Performance CPU GEMM

A high-performance CPU matrix multiplication (GEMM) implementation optimized with AVX-512, targeting the following operation:

$$
C = AB^T + C, \quad A \in \mathbb{R}^{M \times K}, \quad B \in \mathbb{R}^{N \times K}, \quad C \in \mathbb{R}^{M \times N}
$$

All matrices are stored in **row-major** order.

## Motivation

When developing CPU backend operators for inference engines, I needed an optimized GEMM kernel. While OpenBLAS provides excellent performance, there are practical reasons to implement a custom solution:

- **Binary Size**: OpenBLAS adds significant overhead to the final executable
- **Thread Pool Conflicts**: OpenBLAS maintains its own thread pool, which can conflict with OpenMP used in other operators
- **Educational Value**: Implementing GEMM from scratch provides deep insights into CPU architecture optimization

## Test Environment
```
CPU:             Intel Xeon Silver 4310 @ 2.10GHz (Turbo: 3.30GHz)
Sockets:         2
Cores/Socket:    12
Physical Cores:  24

Cache (per core):
  L1d:           48 KiB
  L2:            1.25 MiB
  L3:            18 MiB (shared)

ISA Extensions:  AVX, AVX2, AVX-512 (F/DQ/BW/VL/VNNI/VBMI/VBMI2)
```

## Theoretical Peak Performance

With AVX-512 support and 2 FMA units per core:
```
Peak FLOPS = Frequency × FMA_Units × Ops_per_FMA × SIMD_Width
           = 2.2 GHz × 2 × 2 × 16
           = 140.8 GFLOPS/core (base)
           = 211.2 GFLOPS/core (turbo)
```

## Benchmark Results

**Configuration**: M = N = K = 4096, 2 warmup rounds, 10 benchmark rounds

### Baseline Implementations

| Kernel | Description | Peak GFLOPS | Avg GFLOPS |
|--------|-------------|-------------|------------|
| `core_v1_avx2` | AVX2 vectorized inner loop | 6.42 | 6.35 |
| `core_v2_avx512` | AVX-512 vectorized inner loop | 7.12 | 7.07 |
| `core_v3_block` | Simple 2D cache blocking | 19.74 | 19.73 |

> **Note**: AVX-512 shows minimal improvement over AVX2 in naive implementations because the kernel is **memory-bound** — increasing compute throughput without optimizing memory access patterns yields negligible gains.

### Optimized Implementations

Optimization techniques applied:
- Multi-level cache blocking (L1/L2/L3)
- Data packing for sequential memory access
- Register blocking with 14×32 micro-kernel
- Loop unrolling for instruction-level parallelism

#### Single-Core Performance

| Kernel | Peak GFLOPS | Avg GFLOPS | % Peak | vs OpenBLAS |
|--------|-------------|------------|--------|-------------|
| `core_v4_cache` | 110.74 | 109.37 | 78.7% | 82.4% |
| OpenBLAS | 132.88 | 132.62 | 94.3% | — |

#### Multi-Core Performance (24 threads)

| Kernel | Peak GFLOPS | Avg GFLOPS |
|--------|-------------|------------|
| `core_v5_parallel` | 2010.61 | 1909.40 |
| OpenBLAS | 1914.91 | 1796.69 |

The custom implementation achieves **~106% of OpenBLAS** performance at this matrix size with 24 threads.

## Performance Scaling

Performance comparison across matrix sizes (M = N = K from 256 to 8960, step 256).

### My Implementation ($C = AB^T + C$, Row-Major, AVX-512)

**Peak GFLOPS:**

![Peak GFLOPS](https://cdn.jsdelivr.net/gh/tianyuxbear/images/blog/core_blas_comparison_peak.png)

**Average GFLOPS:**

![Average GFLOPS](https://cdn.jsdelivr.net/gh/tianyuxbear/images/blog/core_blas_comparison_avg.png)

### Reference Implementation ($C = AB$, Column-Major, AVX2)

I also benchmarked the reference implementation from [salykova/matmul-cpu](https://salykova.github.io/matmul-cpu). Since it only uses AVX2, I disabled AVX-512 in OpenBLAS (`OPENBLAS_CORETYPE=Haswell`) for a fair comparison.

**Peak GFLOPS:**

![ref_blas_comparison_peak](https://cdn.jsdelivr.net/gh/tianyuxbear/images/blog/ref_blas_comparison_peak.png)

**Average GFLOPS:**

![ref_blas_comparison_avg](https://cdn.jsdelivr.net/gh/tianyuxbear/images/blog/ref_blas_comparison_avg.png)

> **Observation**: The original author claimed their implementation outperforms both Intel MKL and OpenBLAS. However, in my testing, the reference implementation consistently underperforms OpenBLAS. The reason for this discrepancy is unclear — it may be related to differences in CPU microarchitecture, compiler versions, or other environmental factors.

## Notes on Loop Unrolling

The reference implementation uses micro-kernel loop unrolling as an optimization technique. I tested this on their column-major implementation:

| Kernel | Description | Peak GFLOPS | Avg GFLOPS |
|--------|-------------|-------------|------------|
| `ref_col_cache` | Cache blocking (column-major) | 61.63 | 61.32 |
| `ref_col_unroll` | + Loop unrolling | 72.82 | 72.74 |

Loop unrolling provides a **~18% speedup** in the reference implementation.

However, when I attempted to apply the same technique to my row-major AVX-512 implementation, the results were surprisingly poor:
```bash
git checkout unroll

# Without unrolling
xmake run scratch
# Output: Scratch:  1.208882 s,  Perf: 113.69 GFLOPS

# With unrolling
xmake run scratch_unroll
# Output: Scratch Unroll:  8.487463 s,  Perf: 16.19 GFLOPS
```

The unrolled version is **~7x slower** than the non-unrolled version. The exact cause is unclear, but possible reasons include:

- **Register spilling**: Excessive unrolling may exceed available registers, causing spills to memory
- **Instruction cache pressure**: Larger code size may cause I-cache misses
- **Compiler optimization interference**: Manual unrolling may conflict with compiler's own optimization strategies
- **Different micro-architectural behavior**: AVX-512 execution characteristics differ from AVX2

This remains an open question for further investigation.

## Quick Start

### Build
```bash
xmake build
```

### Run Benchmarks
```bash
# Baseline kernels
xmake run main -k core_v1,core_v2,core_v3 -r 10 4096

# Optimized single-core
xmake run main -k core_v4 -r 10 4096

# Optimized multi-core
xmake run main -k core_v5 -r 10 4096

# OpenBLAS comparison (single-core)
OPENBLAS_NUM_THREADS=1 xmake run main -k blas_row -r 10 4096

# OpenBLAS comparison (multi-core)
OPENBLAS_NUM_THREADS=24 xmake run main -k blas_row -r 10 4096

# Reference implementation (column-major)
xmake run main -k ref_col_cache,ref_col_unroll -r 10 4096
```

## Acknowledgments

This implementation is inspired by [salykova/matmul-cpu](https://salykova.github.io/matmul-cpu), with modifications for:
- Row-major storage layout (original uses column-major)
- $C = AB^T + C$ operation (original computes $C = AB$)
- AVX-512 vectorization (original uses AVX2)

## References

- [Anatomy of High-Performance Matrix Multiplication (Goto & Van De Geijn)](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- [BLIS: A Framework for Rapidly Instantiating BLAS Functionality](https://github.com/flame/blis)
- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/)