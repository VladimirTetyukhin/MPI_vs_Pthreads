#include "hybrid_common.h"

/* Program entry point and command-line parsing. */

int main(int argc, char **argv) {
    int mode        = (argc > 1 ? atoi(argv[1]) : 0);
    int N           = (argc > 2 ? atoi(argv[2]) : 1024);
    int use_cpu     = (argc > 3 ? atoi(argv[3]) : 1);
    int use_gpu0    = (argc > 4 ? atoi(argv[4]) : 1);
    int use_gpu1    = (argc > 5 ? atoi(argv[5]) : 1);
    int cpu_blas    = (argc > 6 ? atoi(argv[6]) : 0);
    int gpu_blas    = (argc > 7 ? atoi(argv[7]) : 0);
    int cpu_threads = (argc > 8 ? atoi(argv[8]) : 0);
    int tile        = (argc > 9 ? atoi(argv[9]) : 2048);
    int prec        = (argc > 10 ? atoi(argv[10]) : 0);
    const char *variant = (argc > 11 ? argv[11] : "variant");
    const char *config_name = (argc > 12 ? argv[12] : "config");
    if (N <= 0) N = 1024;
    if (tile <= 0) tile = 2048;
    dtype_t dt = (prec == 1) ? DT_REAL_FLOAT : DT_REAL_DOUBLE;
    if (streq(variant, "dgemm")) dt = DT_REAL_DOUBLE;
    if (streq(variant, "sgemm")) dt = DT_REAL_FLOAT;
#ifdef USE_CBLAS
    if (variant_requests_blas_gemm(variant)) cpu_blas = 1;
#else
    if (variant_requests_blas_gemm(variant)) {
        fprintf(stderr, "WARNING: variant=%s requested DGEMM/SGEMM, but this binary was not compiled with -DUSE_CBLAS; falling back to NAIVE CPU kernel.\n", variant);
        cpu_blas = 0;
    }
#endif
#ifdef USE_CUBLAS
    if (variant_requests_blas_gemm(variant)) gpu_blas = 1;
#else
    if (variant_requests_blas_gemm(variant) && (use_gpu0 || use_gpu1)) {
        fprintf(stderr, "WARNING: variant=%s requested GPU GEMM, but this binary was not compiled with -DUSE_CUBLAS; falling back to OPENACC_NAIVE GPU kernel.\n", variant);
        gpu_blas = 0;
    }
#endif
    if (mode == 1) {
        g_mpi_init_start_abs = tnow();
        MPI_Init(&argc, &argv);
        MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
        g_mpi_init_end_abs = tnow();
        int rc = run_mpi_three_process_mode(N, use_cpu, use_gpu0, use_gpu1,
                                            cpu_blas, gpu_blas, cpu_threads, tile,
                                            dt, variant, config_name);
        MPI_Finalize();
        return rc;
    }
    return run_pthreads_mode(N, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile, dt, variant, config_name);
}
