# Hybrid MPI/OpenMP/OpenACC Matrix Benchmark — Refactored Layout

This is a logical multi-file refactor of the original single-file program. The computational logic was preserved, while related code was moved into modules with a shared header.

## Layout

```text
include/hybrid_common.h   Shared includes, constants, types, globals, and function declarations
src/main.c                Program entry point and command-line parsing
src/platform.c            Timers, formatting, cache/memory helpers, dtype helpers, allocation, CPU thread setup
src/timing.c              Timeline helpers, MPI clock sync, trace setup, transfer-size accounting
src/cpu_kernels.c         CPU naive kernels, optional CBLAS dispatch, checksums, panel packing
src/gpu_kernels.c         OpenACC/CUDA kernels and tiled GPU execution helpers
src/worker.c              Worker timing/performance aggregation and pthread worker entry points
src/benchmarks.c          N/4 calibration benchmarks and fixed-split panel benchmark helpers
src/partition.c           Variant/config predicates and CPU/GPU row partitioning policy
src/reporting.c           Worker summaries, timeline printing, matrix initialization, memory sizing
src/pthreads_mode.c       Single-process pthreads execution mode
src/mpi_utils.c           MPI datatype and large-count collective helpers
src/mpi_workers.c         MPI fixed-split worker execution, role helpers, benchmark summaries
src/mpi_modes.c           MPI full-B/panel-B execution and policy dispatch
```

## Build

The Makefile keeps the same dependencies expected by the original program: MPI, OpenMP, OpenACC, and optionally CBLAS/CUBLAS.

```bash
make
```

With CBLAS:

```bash
make USE_CBLAS=1 CBLAS_LIBS="-lopenblas"
```

With CUBLAS:

```bash
make USE_CUBLAS=1 CUDA_HOME=/usr/local/cuda
```

Clean:

```bash
make clean
```

## Notes

- Functions were externalized through `include/hybrid_common.h` so the modules compile as separate translation units.
- The original `static` file-scope function boundaries were removed where needed for cross-module linkage.
- The `_POSIX_C_SOURCE` definition remains in the common header before system includes.
- Because this program depends on HPC toolchains, syntax/build validation should be done on the same cluster/compiler stack used for the original version.

## No-panel DGEMM/SGEMM pthreads-vs-MPI benchmark driver

This package includes an updated version of `hybrid15_nopanel_dgemm_sgemm_compare_scatter_gather.sh`.

The driver now supports both source layouts:

1. the refactored multi-file project root in this ZIP; and
2. the original monolithic `hybrid15*.c` source files.

From the project root, run:

```bash
./hybrid15_nopanel_dgemm_sgemm_compare_scatter_gather.sh
```

Useful overrides:

```bash
# Run a small smoke test before the full 10000..80000 sweep
SIZE_START=1000 SIZE_END=1000 SIZE_STEP=1000 TIMEOUT_S=300 ./hybrid15_nopanel_dgemm_sgemm_compare_scatter_gather.sh

# Point at a different refactored project root or single C file
SRC=/path/to/project-or-file ./hybrid15_nopanel_dgemm_sgemm_compare_scatter_gather.sh

# Reuse an already-built executable
KEEP_EXE=1 EXE=hybrid15 ./hybrid15_nopanel_dgemm_sgemm_compare_scatter_gather.sh
```

The script still enforces `MPI_NP=3`, keeps `tile=N` for every size, and writes the same log/CSV outputs as the uploaded driver: `matmul_results15.log`, `results15.csv`, panel CSVs, timeline CSVs, tile-timeline CSVs, and raw key/value CSVs.
