#ifndef HYBRID_COMMON_H
#define HYBRID_COMMON_H

#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <omp.h>
#include <sys/time.h>
#include <math.h>
#include <openacc.h>
#include <mpi.h>
#include <limits.h>
#include <time.h>

#ifdef USE_CBLAS
#include <cblas.h>
#endif

#ifdef USE_CUBLAS
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

#define CPU_MAX_GHZ 3.6
#define L1D_PER_CORE_B (48ULL * 1024ULL)
#define L2_PER_CORE_B  (1280ULL * 1024ULL)
#define L3_SHARED_B    (48ULL * 1024ULL * 1024ULL)

#define MPI_DEFAULT_FALLBACK_PANEL_TILE 5000
#define MPI_DGEMM_DIRECT_PANEL_TILE 5000
#define MPI_SGEMM_DIRECT_PANEL_TILE 10000
#define MPI_DGEMM_DIRECT_PANEL_N_THRESHOLD 34000
#define MPI_SGEMM_DIRECT_PANEL_N_THRESHOLD 42000

typedef enum { DT_REAL_DOUBLE=0, DT_REAL_FLOAT=1 } dtype_t;


typedef struct {
    int wid;
    int enabled;
    int gpu_dev;
    int N;
    int r0;
    int r1;
    int cpu_blas;
    int gpu_blas;
    int tile;
    dtype_t dt;
    void *A;
    void *B;
    void *C;
    double init_time;
    double gemm_time;
    double copy_time;
    unsigned long long copy_bytes;
    unsigned long long copyin_bytes;
    unsigned long long copyout_bytes;
    double cleanup_time;
    double total_time;
    long long rows_done;
    double flops;
    double gflops_gemm;
    double gflops_effective;
    double rows_per_s_eff;
    double copy_bw_gbs;
    double t_ref;
    double t_init_start_abs;
    double t_init_end_abs;
    double t_gemm_start_abs;
    double t_gemm_end_abs;
    double t_copy_start_abs;
    double t_copy_end_abs;
    double t_cleanup_start_abs;
    double t_cleanup_end_abs;
    double t_copyin_start_abs;
    double t_copyin_end_abs;
    double t_copyout_start_abs;
    double t_copyout_end_abs;
    double t_init_start_rel;
    double t_init_end_rel;
    double t_gemm_start_rel;
    double t_gemm_end_rel;
    double t_copy_start_rel;
    double t_copy_end_rel;
    double t_cleanup_start_rel;
    double t_cleanup_end_rel;
    double t_copyin_start_rel;
    double t_copyin_end_rel;
    double t_copyout_start_rel;
    double t_copyout_end_rel;
    double t_compute_start_rel;
    double t_compute_end_rel;
    double t_dev_start_rel;
    double t_dev_end_rel;
    int trace_gpu_tiles;
    int trace_rank;
    int trace_panel_idx;
    int trace_panel_j0;
    int trace_panel_jb;
    double trace_root_abs;
    double trace_clock_offset_to_root;
    char trace_prefix[32];
    char trace_mode[32];
    int err;
    char status[64];
} worker_t;

typedef struct {
    int wid;
    int enabled;
    int gpu_dev;
    int K;
    int cols;
    int r0;
    int r1;
    int cpu_blas;
    int gpu_blas;
    int tile;
    dtype_t dt;
    void *A;
    void *B;
    void *C;
    int c_ld;
    int c_col0;
    double t_ref;
    double init_time;
    double gemm_time;
    double copy_time;
    unsigned long long copy_bytes;
    unsigned long long copyin_bytes;
    unsigned long long copyout_bytes;
    double cleanup_time;
    double total_time;
    long long rows_done;
    double t_init_start_abs;
    double t_init_end_abs;
    double t_gemm_start_abs;
    double t_gemm_end_abs;
    double t_copy_start_abs;
    double t_copy_end_abs;
    double t_cleanup_start_abs;
    double t_cleanup_end_abs;
    double t_copyin_start_abs;
    double t_copyin_end_abs;
    double t_copyout_start_abs;
    double t_copyout_end_abs;
    double t_init_start_rel;
    double t_init_end_rel;
    double t_gemm_start_rel;
    double t_gemm_end_rel;
    double t_copy_start_rel;
    double t_copy_end_rel;
    double t_cleanup_start_rel;
    double t_cleanup_end_rel;
    double t_copyin_start_rel;
    double t_copyin_end_rel;
    double t_copyout_start_rel;
    double t_copyout_end_rel;
    double t_compute_start_rel;
    double t_compute_end_rel;
    double t_dev_start_rel;
    double t_dev_end_rel;
    int trace_gpu_tiles;
    int trace_rank;
    int trace_panel_idx;
    int trace_panel_j0;
    int trace_panel_jb;
    double trace_root_abs;
    double trace_clock_offset_to_root;
    char trace_prefix[32];
    char trace_mode[32];
    int err;
    char status[64];
} panel_worker_t;


typedef struct {
    int rank;
    int panel_idx;
    int j0;
    int jb;
    int rows;
    double bcast_start_rel;
    double bcast_end_rel;
    double scatter_start_rel;
    double scatter_end_rel;
    double gather_start_rel;
    double gather_end_rel;
    unsigned long long bcast_bytes;
    unsigned long long scatter_bytes;
    unsigned long long gather_bytes;
} mpi_timeline_comm_t;

typedef struct {
    double init_start_abs;
    double init_end_abs;
    double copyin_start_abs;
    double copyin_end_abs;
    double gemm_start_abs;
    double gemm_end_abs;
    double copyout_start_abs;
    double copyout_end_abs;
    double cleanup_start_abs;
    double cleanup_end_abs;
} gpu_step_times_t;

typedef struct {
    int enabled;
    const char *prefix;
    const char *mode;
    int rank;
    const char *dev;
    int panel_idx;
    int panel_j0;
    int panel_jb;
    int rows;
    double root_abs;
    double clock_offset_to_root;
} timeline_trace_t;



extern double g_mpi_init_start_abs;
extern double g_mpi_init_end_abs;

void reset_gpu_step_times(gpu_step_times_t *ts);
void mark_time_span(double *dst_start, double *dst_end, double src_start, double src_end);
double rel_or_missing(double abs_t, double t_ref);
double timeline_rel_value(double x);
double interval_or_zero(double start, double end);
double nonneg_or_zero(double x);
int valid_positive_rel_span(double start_rel, double end_rel);
double timeline_span_start_value(double start_rel, double end_rel);
double timeline_span_end_value(double start_rel, double end_rel);
double rel_to_trace_root(double abs_t, const timeline_trace_t *tr);
void trace_gpu_tile_event(const timeline_trace_t *tr, const char *lane, const char *phase,
    int tile_idx, int tile_j0, int tile_jb, int k0, int kb,
    double start_abs, double end_abs, unsigned long long bytes);
double mpi_rel_to_root(double local_abs, double root_abs, double clock_offset_to_root);
double mpi_sync_clock_offset_to_root(int rank);
void add_offset_if_valid(double *x, double offset);
void shift_worker_abs_to_root_clock(worker_t *w, double offset, double root_abs);
void shift_panel_worker_abs_to_root_clock(panel_worker_t *w, double offset, double root_abs);
void set_worker_trace(worker_t *w, const char *prefix, const char *mode, int rank,
    int panel_idx, int panel_j0, int panel_jb, double root_abs, double offset);
void set_panel_worker_trace(panel_worker_t *w, const char *prefix, const char *mode, int rank,
    int panel_idx, int panel_j0, int panel_jb, double root_abs, double offset);
void init_worker_timing_fields(worker_t *w);
void init_panel_worker_timing_fields(panel_worker_t *w);
void close_gpu_init_if_open(gpu_step_times_t *times, double *t_init);
int ceil_div_int(int a, int b);
void gpu_full_transfer_bytes(int N, int rows, int tile, dtype_t dt,
    unsigned long long *copyin_b,
    unsigned long long *copyout_b,
    unsigned long long *total_b);
void gpu_panel_transfer_bytes(int K, int cols, int rows, int tile, dtype_t dt,
    unsigned long long *copyin_b,
    unsigned long long *copyout_b,
    unsigned long long *total_b);
double rel_reduce_finalize_min(double value);
double rel_reduce_finalize_max(double value);
double mpi_init_elapsed_local(void);
void mpi_reduce_init_timing(double *max_init_s, double *global_start_abs, double *global_end_abs);
double clock_sec(clockid_t clk);
void init_tnow_base(void);
double tnow(void);
double sane(double x);
void fmt_ts_iso(double t, char *out, size_t out_sz);
double bytes_to_mib(unsigned long long b);
unsigned long long bytes_mat_NN_elems(int N, size_t elem_bytes);
double cache_N_limit(unsigned long long cache_bytes, size_t elem_bytes);
const char* cache_fit_label_for_B(int N, size_t elem_bytes);
void print_cache_speed_estimates(void);
void print_cache_fit_report(int N, size_t elem_bytes);
void configure_cpu_threads(int cpu_threads);
const char* dtype_name(dtype_t dt);
size_t dtype_size(dtype_t dt);
unsigned long long mat_bytes_rows(int rows, int N, size_t elem_bytes);
unsigned long long mat_bytes_NN(int N, size_t elem_bytes);
unsigned long long mat_elems_rows(int rows, int N);
unsigned long long mat_elems_NN(int N);
void *malloc_aligned64(size_t sz);
int choose_cpu_block(dtype_t dt, int N);
long long checksum_sample_elems(long long n);
double checksum_real_double(const double *C, long long n);
double checksum_real_float(const float *C, long long n);
void cpu_naive_d(int N, int rows, const double *Ablk, const double *B, double *Cblk);
void cpu_naive_s(int N, int rows, const float *Ablk, const float *B, float *Cblk);
void cpu_blas_real(int N, int rows, dtype_t dt, const void *Ablk, const void *B, void *Cblk);
void gpu_openacc_panel_d(int rows, int jb, int kb, int accumulate, const double *A, const double *B, double *C);
void gpu_openacc_panel_s(int rows, int jb, int kb, int accumulate, const float *A, const float *B, float *C);
#ifdef USE_CUBLAS
int cublas_gemm_real_panel(cublasHandle_t h, dtype_t dt, int rows, int jb, int kb, int accumulate, const void *Ablk, const void *B, void *Cblk);
#endif
void pack_A_panel_bytes(const char *src, int rows, int N, int k0, int kb, size_t esz, char *dst);
void pack_B_panel_bytes(const char *src, int N, int k0, int kb, int j0, int jb, size_t esz, char *dst);
void unpack_C_panel_bytes(const char *src, int rows, int N, int j0, int jb, size_t esz, char *dst);
void cpu_naive_panel_d_ldc(int K, int cols, int rows, const double *Ablk, const double *Bpanel, double *Cblk, int ldc, int c_col0);
void cpu_naive_panel_s_ldc(int K, int cols, int rows, const float *Ablk, const float *Bpanel, float *Cblk, int ldc, int c_col0);
void cpu_blas_real_panel_ldc(int K, int cols, int rows, dtype_t dt, const void *Ablk, const void *Bpanel, void *Cblk, int ldc, int c_col0);
void cpu_naive_panel_d(int K, int cols, int rows, const double *Ablk, const double *Bpanel, double *Cblk);
void cpu_naive_panel_s(int K, int cols, int rows, const float *Ablk, const float *Bpanel, float *Cblk);
void cpu_blas_real_panel(int K, int cols, int rows, dtype_t dt, const void *Ablk, const void *Bpanel, void *Cblk);
void pack_B_rect_panel_bytes(const char *src, int ld_src, int k0, int kb, int j0, int jb, size_t esz, char *dst);
int gpu_run_panel_tiled(int dev, int K, int cols, int rows, int tile, int gpu_blas, dtype_t dt,
    const void *Ablk, const void *Bpanel, void *Cpanel,
    double *t_init, double *t_gemm, double *t_copy, double *t_cleanup,
    double *t_copyin, double *t_copyout,
    char *status, size_t status_sz,
    gpu_step_times_t *times,
    const timeline_trace_t *trace);
int gpu_run_panel_tiled_strided(int dev, int K, int cols, int rows, int tile, int gpu_blas, dtype_t dt,
    const void *Ablk, const void *Bpanel, void *Cfull,
    int ldc, int c_col0,
    double *t_init, double *t_gemm, double *t_copy, double *t_cleanup,
    double *t_copyin, double *t_copyout,
    char *status, size_t status_sz,
    gpu_step_times_t *times,
    const timeline_trace_t *trace);
int choose_gpu_tile(int rows, int N, size_t esz, int tile_req);
int gpu_run_tiled(int dev, int N, int rows, int tile, int gpu_blas, dtype_t dt,
    const void *Ablk, const void *B, void *Cblk,
    double *t_init, double *t_gemm, double *t_copy, double *t_cleanup,
    double *t_copyin, double *t_copyout,
    char *status, size_t status_sz,
    gpu_step_times_t *times,
    const timeline_trace_t *trace);
void finalize_worker_perf(worker_t *w);
void worker_phase_sums(const worker_t w[3],
    double *init_s, double *copy_s, double *compute_s,
    double *cleanup_s, double *worker_sum_s,
    double *worker_span_s);
void apply_gpu_timeline_worker(worker_t *w, const gpu_step_times_t *gt);
void apply_gpu_timeline_panel_worker(panel_worker_t *w, const gpu_step_times_t *gt);
void finalize_panel_worker_times(panel_worker_t *w);
void print_panel_dev_timing_line(const char *tag, int rank, int panel_idx, int j0, int jb, int rows_total, const panel_worker_t *w);
void *worker_main(void *arg);
void init_empty_panel_workers(panel_worker_t out_w[3], const char *status, int err_code);
void merge_rel_span(double *dst_start, double *dst_end, double src_start, double src_end);
void accumulate_panel_worker(worker_t *dst, const panel_worker_t *src, int N_full);
void finalize_panel_worker_accum(worker_t *w);
void *panel_worker_main(void *arg);
int n4_rows_for_bench(int N);
void print_bench_n4_speed(const char *mode, int N, int cols, int rows_target,
    const int bench_rows[3], const double rates[3],
    double cpu_s, double g0_s, double g1_s,
    double cpu_gflops, double g0_gflops, double g1_gflops);
int run_n4_pthreads_benchmark_full(int N, int M, void *A, void *B, void *C,
    int use_cpu, int use_gpu0, int use_gpu1,
    int gpu0_dev, int gpu1_dev,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt,
    const char *variant,
    int bench_rows_out[3], double bench_rate_out[3]);
int run_n4_pthreads_benchmark_panel(int N, int cols, int M, void *A, void *Bpanel, void *Cpanel,
    int C_ld, int C_col0,
    int use_cpu, int use_gpu0, int use_gpu1,
    int gpu0_dev, int gpu1_dev,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt,
    const char *variant,
    int bench_rows_out[3], double bench_rate_out[3]);
int run_fixed_split_buffers_panel(int N, int cols, int M, void *A, void *Bpanel, void *Cpanel,
    int C_ld, int C_col0,
    int use_cpu, int use_gpu0, int use_gpu1,
    int gpu0_dev, int gpu1_dev,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt,
    const char *variant, const double bench_rates_in[3],
    panel_worker_t out_w[3], double *t_total_out,
    double t_ref_base, int panel_idx_for_trace);
void bench_cpu_rows(int N, int rows_bench, int cpu_blas, dtype_t dt, void *A, void *B, double *t_gemm, int *ok);
double compute_checksum(dtype_t dt, const void *C, long long n);
double read_c00(dtype_t dt, const void *C);
int validate_partition(int M, int cpu_r0, int cpu_r1, int g0_r0, int g0_r1, int g1_r0, int g1_r1);
int streq(const char *a, const char *b);
int variant_requests_blas_gemm(const char *variant);
int mpi_direct_panel_policy(int N, dtype_t dt, const char *variant, int *tile_out);
int is_naive_variant(const char *variant);
int config_force_nopanel(const char *config_name);
int config_force_panel(const char *config_name);
int resolve_panel_b_mode(const char *config_name, int auto_mode);
double positive_or_zero(double x);
int choose_rows_rate_partition(int rows_total,
    int use_cpu, int use_gpu0, int use_gpu1,
    const double rates[3],
    int *cpu_r0, int *cpu_r1,
    int *g0_r0, int *g0_r1,
    int *g1_r0, int *g1_r1,
    double *cpu_share_out);
int choose_rows_partition_for_variant(int N_global, int rows_total,
    int use_cpu, int use_gpu0, int use_gpu1,
    const char *variant, const double rates[3],
    int prefer_shared_blas_partition,
    int *cpu_r0, int *cpu_r1,
    int *g0_r0, int *g0_r1,
    int *g1_r0, int *g1_r1,
    double *cpu_share_out,
    int *used_shared_blas_partition_out);
double worker_partition_rate(const worker_t *w);
double panel_worker_partition_rate(const panel_worker_t *w);
void init_empty_workers(worker_t out_w[3], int N, const char *status, int err_code);
void print_worker_line(const worker_t *w, int M);
const char *dev_name_lower_from_wid(int wid);
void print_timeline_event_line(const char *prefix, const char *mode, int rank,
    const char *lane, const char *dev,
    int panel_idx, int j0, int jb, int rows,
    const char *phase,
    double start_rel, double end_rel,
    double root_abs,
    unsigned long long bytes);
void print_timeline_tile_event_line(const char *prefix, const char *mode, int rank,
    const char *lane, const char *dev,
    int panel_idx, int panel_j0, int panel_jb,
    int tile_idx, int tile_j0, int tile_jb,
    int k0, int kb, int rows,
    const char *phase,
    double start_rel, double end_rel,
    double root_abs,
    unsigned long long bytes);
void print_worker_timeline_events(const char *prefix, const char *mode, int rank,
    int panel_idx, int j0, int jb,
    const worker_t *w);
void print_panel_worker_timeline_events(const char *prefix, const char *mode, int rank,
    int panel_idx, int j0, int jb,
    const panel_worker_t *w);
void print_mpi_comm_timeline_events(const mpi_timeline_comm_t ev[3], double root_abs);
void print_part_percent_done(const worker_t w[3], int M);
void print_device_speeds(const worker_t w[3]);
int fill_mats(int N, dtype_t dt, void *A, void *B, void *C);
int fill_A_C_only(int N, dtype_t dt, void *A, void *C);
int fill_B_only(int N, dtype_t dt, void *B);
int fill_A_C_rows(int N, int global_row0, int rows, dtype_t dt, void *A, void *C);
int fill_A_rows_only(int N, int global_row0, int rows, dtype_t dt, void *A);
void fill_B_panel_formula(int N, int j0, int jb, dtype_t dt, void *Bpanel);
unsigned long long read_memtotal_bytes(void);
unsigned long long read_mem_reserve_bytes_env(void);
unsigned long long usable_memtotal_bytes(void);
int requested_panel_cols_limit(int N, int tile_req);
int clamp_panel_cols_ull(unsigned long long cols, int N, int tile_req);
int choose_pthreads_panel_cols(int N, size_t esz, int tile_req);
int choose_mpi_panel_cols(int N, size_t esz, int tile_req, int root_rows);
void print_run_summary(const char *mode_name, const char *variant, const char *config_name, int N, int np, dtype_t dt,
    int use_cpu, int use_gpu0, int use_gpu1, int cpu_blas, int gpu_blas, int cpu_threads, int tile,
    const char *cache_fit, const char *status, double total_s, double total_gflops, double c00, double checksum,
    int bench_rows[3], double bench_rates[3]);
int run_pthreads_mode(int N, int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int cpu_threads, int tile,
    dtype_t dt, const char *variant, const char *config_name);
int too_large_for_mpi_count_elems(unsigned long long elems);
void make_abs_from_root(double root_abs, double rel, char *out, size_t out_sz);
void make_abs_from_positive_rel_span(double root_abs, double start_rel, double end_rel,
    char *start_out, size_t start_out_sz,
    char *end_out, size_t end_out_sz);
int mpi_dtype_setup(dtype_t dt, MPI_Datatype *mpi_dt);
int mpi_type_size_bytes(MPI_Datatype dt);
int mpi_bcast_large(void *buf, unsigned long long elems, MPI_Datatype dt, int root, MPI_Comm comm);
int mpi_scatter_rows_large(const void *sendbuf, const int *counts_rows, const int *displs_rows,
    int N, MPI_Datatype dt, void *recvbuf, int rank, int root, MPI_Comm comm);
int mpi_gather_rows_large(const void *sendbuf, int send_rows, int N, MPI_Datatype dt,
    void *recvbuf, const int *counts_rows, const int *displs_rows,
    int rank, int root, MPI_Comm comm);
int run_fixed_split_buffers(int N, int M, void *A, void *B, void *C,
    int use_cpu, int use_gpu0, int use_gpu1,
    int gpu0_dev, int gpu1_dev,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt,
    const char *variant,
    worker_t out_w[3], double *t_total_out,
    char *policy_status, size_t policy_status_sz,
    int bench_rows_out[3], double bench_rate_out[3],
    double *n4_bench_wall_s_out,
    double t_ref_base);
const char *mpi_three_role_name(int role);
int mpi_three_role_gpu_dev(int role);
int mpi_three_choose_counts(int N, int use_cpu, int use_gpu0, int use_gpu1,
    const char *variant, const double bench_rates[3],
    int counts_rows[3], int displs_rows[3],
    int global_r0[3], int global_r1[3],
    double *cpu_share_out,
    int *used_shared_blas_partition_out);
int mpi_three_gpu_available_for_role(int role, char *status, size_t status_sz);
void mpi_three_run_one_full_worker(int role, int N, int local_rows,
    void *A_local, void *B, void *C_local,
    int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt,
    double root_abs, double clock_offset_to_root,
    worker_t out_w[3]);
int run_n4_mpi_benchmark_full(int N, int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt, const char *variant,
    int rank, int size, MPI_Datatype mpi_dt,
    int bench_rows_out[3], double bench_rate_out[3]);
int run_n4_mpi_benchmark_panel(int N, int cols,
    int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt, const char *variant,
    int rank, int size, MPI_Datatype mpi_dt,
    int bench_rows_out[3], double bench_rate_out[3]);
int run_n4_pthreads_partition_benchmark_root(int N, int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int tile, dtype_t dt, const char *variant,
    int rank, int size,
    int bench_rows_out[3], double bench_rate_out[3]);
void mpi_three_print_summary_root(int N, int size, dtype_t dt,
    int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int cpu_threads, int tile,
    const char *variant, const char *config_name,
    const char *policy, const char *cache_fit,
    double total_with_init, double total_no_init,
    double bcast_s, double scatter_s, double compute_s,
    double gather_s, double comm_s,
    double root_abs, double mpi_init_s,
    double c00, double chk,
    worker_t gathered[3],
    int global_r0[3], int global_r1[3]);
int run_mpi_three_process_fullB(int N, int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int cpu_threads, int tile,
    dtype_t dt, const char *variant, const char *config_name,
    int rank, int size, MPI_Datatype mpi_dt,
    int counts_rows[3], int displs_rows[3],
    int global_r0[3], int global_r1[3], double cpu_share,
    double t_mode0_local);
int run_mpi_three_process_panelB(int N, int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int cpu_threads, int tile,
    dtype_t dt, const char *variant, const char *config_name,
    int rank, int size, MPI_Datatype mpi_dt,
    int counts_rows[3], int displs_rows[3],
    int global_r0[3], int global_r1[3], double cpu_share,
    double t_mode0_local);
int run_mpi_three_process_mode(int N, int use_cpu, int use_gpu0, int use_gpu1,
    int cpu_blas, int gpu_blas, int cpu_threads, int tile,
    dtype_t dt, const char *variant, const char *config_name);
int should_use_mpi_panel_b(int N, int size, size_t esz, unsigned long long elemsB_full);

#endif /* HYBRID_COMMON_H */
