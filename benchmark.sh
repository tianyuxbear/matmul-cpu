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
#   core    - Run core_v5 benchmark only
#   blas    - Run blas_row benchmark only
#   all     - Run both benchmarks (default)
#   help    - Show this help message
# =============================================================================

# Configuration
START_N=256
END_N=8960
STEP=256
REPEAT=10

# Run core_v5 benchmark
run_core_v5() {
    echo "========== Running core_v5 benchmark =========="
    for N in $(seq $START_N $STEP $END_N); do
        echo "[core_v5] N=$N"
        xmake run main -k core_v5 -r $REPEAT $N
    done
    echo "========== core_v5 benchmark completed =========="
}

# Run blas_row benchmark
run_blas_row() {
    export OPENBLAS_NUM_THREADS=24
    echo "========== Running blas_row benchmark =========="
    for N in $(seq $START_N $STEP $END_N); do
        echo "[blas_row] N=$N"
        xmake run main -k blas_row -r $REPEAT $N
    done
    echo "========== blas_row benchmark completed =========="
}

# Display help message
show_help() {
    echo "Usage: $0 [option]"
    echo ""
    echo "Options:"
    echo "  core    - Run core_v5 benchmark only"
    echo "  blas    - Run blas_row benchmark only"
    echo "  all     - Run both benchmarks (default)"
    echo "  help    - Show this help message"
    echo ""
    echo "Configuration:"
    echo "  N range: $START_N to $END_N (step: $STEP)"
    echo "  Repeat:  $REPEAT times per test"
}

# Main entry point
main() {
    local option="${1:-all}"
    
    case "$option" in
        core)
            run_core_v5
            ;;
        blas)
            run_blas_row
            ;;
        all)
            echo "Starting all benchmarks..."
            echo "N range: $START_N to $END_N (step: $STEP)"
            echo ""
            run_core_v5
            echo ""
            run_blas_row
            echo ""
            echo "All benchmarks completed!"
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