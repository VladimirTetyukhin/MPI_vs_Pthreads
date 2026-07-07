#include "hybrid_common.h"

/* Timing primitives, relative span helpers, MPI clock synchronization, trace setup, transfer-size accounting. */

void reset_gpu_step_times(gpu_step_times_t *ts) {
    if (!ts) return;
    memset(ts, 0, sizeof(*ts));
}

void mark_time_span(double *dst_start, double *dst_end, double src_start, double src_end) {
    if (!(src_end > src_start) || !isfinite(src_start) || !isfinite(src_end)) return;
    if (!(*dst_end > *dst_start)) {
        *dst_start = src_start;
        *dst_end = src_end;
        return;
    }
    if (src_start < *dst_start) *dst_start = src_start;
    if (src_end > *dst_end) *dst_end = src_end;
}

double tnow(void);
size_t dtype_size(dtype_t dt);
int choose_gpu_tile(int rows, int N, size_t esz, int tile_req);
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

double rel_or_missing(double abs_t, double t_ref) {
    return (abs_t > 0.0 && isfinite(abs_t) && t_ref > 0.0 && isfinite(t_ref)) ? (abs_t - t_ref) : -1.0;
}

double timeline_rel_value(double x) {
    return (isfinite(x) && x >= 0.0) ? x : -1.0;
}

double interval_or_zero(double start, double end) {
    return (isfinite(start) && isfinite(end) && end > start) ? (end - start) : 0.0;
}

double nonneg_or_zero(double x) {
    return (isfinite(x) && x > 0.0) ? x : 0.0;
}

int valid_positive_rel_span(double start_rel, double end_rel) {
    return isfinite(start_rel) && isfinite(end_rel) && start_rel >= 0.0 && end_rel > start_rel;
}

double timeline_span_start_value(double start_rel, double end_rel) {
    return valid_positive_rel_span(start_rel, end_rel) ? start_rel : -1.0;
}

double timeline_span_end_value(double start_rel, double end_rel) {
    return valid_positive_rel_span(start_rel, end_rel) ? end_rel : -1.0;
}

double rel_to_trace_root(double abs_t, const timeline_trace_t *tr) {
    if (!tr || !(tr->root_abs > 0.0) || !isfinite(abs_t) || !isfinite(tr->root_abs) || !isfinite(tr->clock_offset_to_root)) return -1.0;
    return abs_t + tr->clock_offset_to_root - tr->root_abs;
}

void trace_gpu_tile_event(const timeline_trace_t *tr, const char *lane, const char *phase,
                                 int tile_idx, int tile_j0, int tile_jb, int k0, int kb,
                                 double start_abs, double end_abs, unsigned long long bytes) {
    if (!tr || !tr->enabled) return;
    double sr = rel_to_trace_root(start_abs, tr);
    double er = rel_to_trace_root(end_abs, tr);
    print_timeline_tile_event_line(tr->prefix, tr->mode, tr->rank, lane, tr->dev,
                                   tr->panel_idx, tr->panel_j0, tr->panel_jb,
                                   tile_idx, tile_j0, tile_jb, k0, kb, tr->rows, phase,
                                   sr, er, tr->root_abs, bytes);
}

double mpi_rel_to_root(double local_abs, double root_abs, double clock_offset_to_root) {
    return (isfinite(local_abs) && isfinite(root_abs) && root_abs > 0.0 && isfinite(clock_offset_to_root))
        ? (local_abs + clock_offset_to_root - root_abs) : -1.0;
}

double mpi_sync_clock_offset_to_root(int rank) {
    MPI_Barrier(MPI_COMM_WORLD);
    double local_sync_abs = tnow();
    double root_sync_abs = local_sync_abs;
    MPI_Bcast(&root_sync_abs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    return (rank == 0) ? 0.0 : (root_sync_abs - local_sync_abs);
}

void add_offset_if_valid(double *x, double offset) {
    if (x && *x > 0.0 && isfinite(*x) && isfinite(offset)) *x += offset;
}

void shift_worker_abs_to_root_clock(worker_t *w, double offset, double root_abs) {
    if (!w) return;
    add_offset_if_valid(&w->t_init_start_abs, offset);
    add_offset_if_valid(&w->t_init_end_abs, offset);
    add_offset_if_valid(&w->t_gemm_start_abs, offset);
    add_offset_if_valid(&w->t_gemm_end_abs, offset);
    add_offset_if_valid(&w->t_copy_start_abs, offset);
    add_offset_if_valid(&w->t_copy_end_abs, offset);
    add_offset_if_valid(&w->t_cleanup_start_abs, offset);
    add_offset_if_valid(&w->t_cleanup_end_abs, offset);
    add_offset_if_valid(&w->t_copyin_start_abs, offset);
    add_offset_if_valid(&w->t_copyin_end_abs, offset);
    add_offset_if_valid(&w->t_copyout_start_abs, offset);
    add_offset_if_valid(&w->t_copyout_end_abs, offset);
    if (root_abs > 0.0 && isfinite(root_abs)) w->t_ref = root_abs;
}

void shift_panel_worker_abs_to_root_clock(panel_worker_t *w, double offset, double root_abs) {
    if (!w) return;
    add_offset_if_valid(&w->t_init_start_abs, offset);
    add_offset_if_valid(&w->t_init_end_abs, offset);
    add_offset_if_valid(&w->t_gemm_start_abs, offset);
    add_offset_if_valid(&w->t_gemm_end_abs, offset);
    add_offset_if_valid(&w->t_copy_start_abs, offset);
    add_offset_if_valid(&w->t_copy_end_abs, offset);
    add_offset_if_valid(&w->t_cleanup_start_abs, offset);
    add_offset_if_valid(&w->t_cleanup_end_abs, offset);
    add_offset_if_valid(&w->t_copyin_start_abs, offset);
    add_offset_if_valid(&w->t_copyin_end_abs, offset);
    add_offset_if_valid(&w->t_copyout_start_abs, offset);
    add_offset_if_valid(&w->t_copyout_end_abs, offset);
    if (root_abs > 0.0 && isfinite(root_abs)) w->t_ref = root_abs;
}

void set_worker_trace(worker_t *w, const char *prefix, const char *mode, int rank,
                             int panel_idx, int panel_j0, int panel_jb, double root_abs, double offset) {
    if (!w) return;
    w->trace_gpu_tiles = 1;
    w->trace_rank = rank;
    w->trace_panel_idx = panel_idx;
    w->trace_panel_j0 = panel_j0;
    w->trace_panel_jb = panel_jb;
    w->trace_root_abs = root_abs;
    w->trace_clock_offset_to_root = offset;
    snprintf(w->trace_prefix, sizeof(w->trace_prefix), "%s", prefix ? prefix : "TIMELINE_TILE_EVENT");
    snprintf(w->trace_mode, sizeof(w->trace_mode), "%s", mode ? mode : "unknown");
}

void set_panel_worker_trace(panel_worker_t *w, const char *prefix, const char *mode, int rank,
                                   int panel_idx, int panel_j0, int panel_jb, double root_abs, double offset) {
    if (!w) return;
    w->trace_gpu_tiles = 1;
    w->trace_rank = rank;
    w->trace_panel_idx = panel_idx;
    w->trace_panel_j0 = panel_j0;
    w->trace_panel_jb = panel_jb;
    w->trace_root_abs = root_abs;
    w->trace_clock_offset_to_root = offset;
    snprintf(w->trace_prefix, sizeof(w->trace_prefix), "%s", prefix ? prefix : "TIMELINE_TILE_EVENT");
    snprintf(w->trace_mode, sizeof(w->trace_mode), "%s", mode ? mode : "unknown");
}

void init_worker_timing_fields(worker_t *w) {
    if (!w) return;
    w->init_time = 0.0;
    w->gemm_time = 0.0;
    w->copy_time = 0.0;
    w->copy_bytes = 0ULL;
    w->copyin_bytes = 0ULL;
    w->copyout_bytes = 0ULL;
    w->cleanup_time = 0.0;
    w->total_time = 0.0;
    w->t_ref = 0.0;
    w->t_init_start_abs = w->t_init_end_abs = 0.0;
    w->t_gemm_start_abs = w->t_gemm_end_abs = 0.0;
    w->t_copy_start_abs = w->t_copy_end_abs = 0.0;
    w->t_cleanup_start_abs = w->t_cleanup_end_abs = 0.0;
    w->t_copyin_start_abs = w->t_copyin_end_abs = 0.0;
    w->t_copyout_start_abs = w->t_copyout_end_abs = 0.0;
    w->t_init_start_rel = w->t_init_end_rel = -1.0;
    w->t_gemm_start_rel = w->t_gemm_end_rel = -1.0;
    w->t_copy_start_rel = w->t_copy_end_rel = -1.0;
    w->t_cleanup_start_rel = w->t_cleanup_end_rel = -1.0;
    w->t_copyin_start_rel = w->t_copyin_end_rel = -1.0;
    w->t_copyout_start_rel = w->t_copyout_end_rel = -1.0;
    w->t_compute_start_rel = w->t_compute_end_rel = -1.0;
    w->t_dev_start_rel = w->t_dev_end_rel = -1.0;
}

void init_panel_worker_timing_fields(panel_worker_t *w) {
    if (!w) return;
    w->init_time = 0.0;
    w->gemm_time = 0.0;
    w->copy_time = 0.0;
    w->copy_bytes = 0ULL;
    w->copyin_bytes = 0ULL;
    w->copyout_bytes = 0ULL;
    w->cleanup_time = 0.0;
    w->total_time = 0.0;
    w->t_ref = 0.0;
    w->t_init_start_abs = w->t_init_end_abs = 0.0;
    w->t_gemm_start_abs = w->t_gemm_end_abs = 0.0;
    w->t_copy_start_abs = w->t_copy_end_abs = 0.0;
    w->t_cleanup_start_abs = w->t_cleanup_end_abs = 0.0;
    w->t_copyin_start_abs = w->t_copyin_end_abs = 0.0;
    w->t_copyout_start_abs = w->t_copyout_end_abs = 0.0;
    w->t_init_start_rel = w->t_init_end_rel = -1.0;
    w->t_gemm_start_rel = w->t_gemm_end_rel = -1.0;
    w->t_copy_start_rel = w->t_copy_end_rel = -1.0;
    w->t_cleanup_start_rel = w->t_cleanup_end_rel = -1.0;
    w->t_copyin_start_rel = w->t_copyin_end_rel = -1.0;
    w->t_copyout_start_rel = w->t_copyout_end_rel = -1.0;
    w->t_compute_start_rel = w->t_compute_end_rel = -1.0;
    w->t_dev_start_rel = w->t_dev_end_rel = -1.0;
}

void close_gpu_init_if_open(gpu_step_times_t *times, double *t_init) {
    if (!times || !(times->init_start_abs > 0.0) || times->init_end_abs > times->init_start_abs) return;
    double ti1 = tnow();
    times->init_end_abs = ti1;
    if (t_init) *t_init += ti1 - times->init_start_abs;
}

int ceil_div_int(int a, int b) {
    return (a <= 0 || b <= 0) ? 0 : ((a + b - 1) / b);
}

void gpu_full_transfer_bytes(int N, int rows, int tile, dtype_t dt,
                                    unsigned long long *copyin_b,
                                    unsigned long long *copyout_b,
                                    unsigned long long *total_b) {
    size_t esz = dtype_size(dt);
    int tt = choose_gpu_tile(rows, N, esz, tile);
    int n_j = ceil_div_int(N, tt);
    unsigned long long copyin = ((unsigned long long)rows * (unsigned long long)N * (unsigned long long)n_j
                                + (unsigned long long)N * (unsigned long long)N) * (unsigned long long)esz;
    unsigned long long copyout = (unsigned long long)rows * (unsigned long long)N * (unsigned long long)esz;
    if (copyin_b) *copyin_b = copyin;
    if (copyout_b) *copyout_b = copyout;
    if (total_b) *total_b = copyin + copyout;
}

void gpu_panel_transfer_bytes(int K, int cols, int rows, int tile, dtype_t dt,
                                     unsigned long long *copyin_b,
                                     unsigned long long *copyout_b,
                                     unsigned long long *total_b) {
    /*
     * Panel mode now does exactly one GPU GEMM per MPI B/C panel.
     * Therefore each panel worker copies the whole local A block and the whole
     * current B panel once, computes once, then copies the completed C panel
     * back once.  Do not count extra K-depth tile copies here.
     */
    (void)tile;
    size_t esz = dtype_size(dt);
    unsigned long long copyin = ((unsigned long long)rows * (unsigned long long)K
                                + (unsigned long long)K * (unsigned long long)cols)
                                * (unsigned long long)esz;
    unsigned long long copyout = (unsigned long long)rows * (unsigned long long)cols
                                 * (unsigned long long)esz;
    if (copyin_b) *copyin_b = copyin;
    if (copyout_b) *copyout_b = copyout;
    if (total_b) *total_b = copyin + copyout;
}

double rel_reduce_finalize_min(double value) {
    return (isfinite(value) && value < 1e299) ? value : -1.0;
}

double rel_reduce_finalize_max(double value) {
    return (isfinite(value) && value > -1e299) ? value : -1.0;
}


double g_mpi_init_start_abs = -1.0;
double g_mpi_init_end_abs = -1.0;

double mpi_init_elapsed_local(void) {
    return (g_mpi_init_end_abs > g_mpi_init_start_abs && isfinite(g_mpi_init_start_abs) && isfinite(g_mpi_init_end_abs))
        ? (g_mpi_init_end_abs - g_mpi_init_start_abs) : 0.0;
}

void mpi_reduce_init_timing(double *max_init_s, double *global_start_abs, double *global_end_abs) {
    double local_init_s = mpi_init_elapsed_local();
    double start_in = (isfinite(g_mpi_init_start_abs) && g_mpi_init_start_abs > 0.0) ? g_mpi_init_start_abs : 1e300;
    double end_in = (isfinite(g_mpi_init_end_abs) && g_mpi_init_end_abs > 0.0) ? g_mpi_init_end_abs : -1e300;
    double max_s = 0.0, min_start = 1e300, max_end = -1e300;
    MPI_Reduce(&local_init_s, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&start_in, &min_start, 1, MPI_DOUBLE, MPI_MIN, 0, MPI_COMM_WORLD);
    MPI_Reduce(&end_in, &max_end, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (max_init_s) *max_init_s = max_s;
    if (global_start_abs) *global_start_abs = rel_reduce_finalize_min(min_start);
    if (global_end_abs) *global_end_abs = rel_reduce_finalize_max(max_end);
}

int choose_gpu_tile(int rows, int N, size_t esz, int tile_req);
int is_naive_variant(const char *variant);
int variant_requests_blas_gemm(const char *variant);
int config_force_nopanel(const char *config_name);
int config_force_panel(const char *config_name);
int mpi_direct_panel_policy(int N, dtype_t dt, const char *variant, int *tile_out);
int resolve_panel_b_mode(const char *config_name, int auto_mode);
int should_use_mpi_panel_b(int N, int size, size_t esz, unsigned long long elemsB_full);
void make_abs_from_root(double root_abs, double rel, char *out, size_t out_sz);
void make_abs_from_positive_rel_span(double root_abs, double start_rel, double end_rel, char *start_out, size_t start_out_sz, char *end_out, size_t end_out_sz);
int validate_partition(int M, int cpu_r0, int cpu_r1, int g0_r0, int g0_r1, int g1_r0, int g1_r1);
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

