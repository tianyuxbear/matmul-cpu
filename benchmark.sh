#!/bin/bash

# =============================================================================
# Benchmark Script for Matrix Operations
# 
# This script runs performance benchmarks with N ranging from 256 to 8960
# with a step size of 256.
#
# Usage:
#   ./benchmark.sh [option]
#
# Options:
#   core_v5          - Run core_v5 benchmark only
#   blas_row         - Run blas_row benchmark only
#   blas_col         - Run blas_col benchmark only
#   ref_col_parallel - Run ref_col_parallel benchmark only
#   all              - Run all benchmarks (default)
#   help             - Show this help message
# =============================================================================

# Configuration
START_N=256
END_N=8960
STEP=256
REPEAT=10
OPENBLAS_THREADS=24

# Generic function to run a single kernel benchmark
# Usage: run_kernel <kernel_name> [use_openblas_threads]
run_kernel() {
    local kernel=$1
    local use_openblas=${2:-false}

    echo "========== Running $kernel benchmark =========="
    
    # Set OpenBLAS threads if required
    if [ "$use_openblas" = true ]; then
        export OPENBLAS_NUM_THREADS=$OPENBLAS_THREADS
        echo "[INFO] OPENBLAS_NUM_THREADS=$OPENBLAS_THREADS"
    fi

    for N in $(seq $START_N $STEP $END_N); do
        echo "[$kernel] N=$N"
        xmake run main -k $kernel -r $REPEAT $N
    done

    echo "========== $kernel benchmark completed =========="
}

# Wrapper functions for each kernel
run_core_v5() {
    run_kernel "core_v5" false
}

run_blas_row() {
    run_kernel "blas_row" true
}

run_blas_col() {
    export OPENBLAS_CORETYPE=Haswell 
    export OPENBLAS_VERBOSE=2
    run_kernel "blas_col" true
}

run_ref_col_parallel() {
    run_kernel "ref_col_parallel" false
}

# Run all benchmarks
run_all() {
    echo "Starting all benchmarks..."
    echo "N range: $START_N to $END_N (step: $STEP)"
    echo ""

    run_core_v5
    echo ""
    run_blas_row
    echo ""
    run_blas_col
    echo ""
    run_ref_col_parallel

    echo ""
    echo "All benchmarks completed!"
}

# Display help message
show_help() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  core_v5          - Run core_v5 benchmark only"
    echo "  blas_row         - Run blas_row benchmark only"
    echo "  blas_col         - Run blas_col benchmark only"
    echo "  ref_col_parallel - Run ref_col_parallel benchmark only"
    echo "  all              - Run all benchmarks (default)"
    echo "  help             - Show this help message"
    echo ""
    echo "Configuration:"
    echo "  N range:         $START_N to $END_N (step: $STEP)"
    echo "  Repeat:          $REPEAT times per test"
    echo "  OpenBLAS threads: $OPENBLAS_THREADS (for blas_row, blas_col)"
}

# Main entry point
main() {
    local option="${1:-all}"
    
    case "$option" in
        core_v5)
            run_core_v5
            ;;
        blas_row)
            run_blas_row
            ;;
        blas_col)
            run_blas_col
            ;;
        ref_col_parallel)
            run_ref_col_parallel
            ;;
        all)
            run_all
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            echo "Error: Unknown option '$option'"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Execute main function with command line arguments
main "$@"