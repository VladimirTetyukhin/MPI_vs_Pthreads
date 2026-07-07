#include "hybrid_common.h"

/* Worker summaries, timeline output, data initialization, memory sizing, and run summaries. */

void init_empty_workers(worker_t out_w[3], int N, const char *status, int err_code) {
    if (!out_w) return;
    memset(out_w, 0, 3 * sizeof(worker_t));
    for (int i = 0; i < 3; i++) {
        init_worker_timing_fields(&out_w[i]);
        out_w[i].wid = i;
        out_w[i].N = N;
        out_w[i].enabled = 0;
        out_w[i].gpu_dev = (i == 0) ? -1 : (i - 1);
        out_w[i].r0 = 0;
        out_w[i].r1 = 0;
        out_w[i].err = err_code;
        snprintf(out_w[i].status, sizeof(out_w[i].status), "%s", status);
    }
}

void print_worker_line(const worker_t *w, int M) {
    long long rows_assigned = (long long)(w->r1 - w->r0);
    double share_assigned = (M > 0) ? (100.0 * (double)rows_assigned / (double)M) : 0.0;
    double share_done = (M > 0) ? (100.0 * (double)w->rows_done / (double)M) : 0.0;
    const char *name = (w->wid == 0) ? "CPU" : (w->wid == 1 ? "GPU0" : "GPU1");
    char i0[64], i1[64], g0[64], g1[64], cm0[64], cm1[64], c0[64], c1[64], x0[64], x1[64], ci0[64], ci1[64], co0[64], co1[64];
    double init_sr = timeline_span_start_value(w->t_init_start_rel, w->t_init_end_rel);
    double init_er = timeline_span_end_value(w->t_init_start_rel, w->t_init_end_rel);
    double compute_sr = timeline_span_start_value(w->t_compute_start_rel, w->t_compute_end_rel);
    double compute_er = timeline_span_end_value(w->t_compute_start_rel, w->t_compute_end_rel);
    double gemm_sr = timeline_span_start_value(w->t_gemm_start_rel, w->t_gemm_end_rel);
    double gemm_er = timeline_span_end_value(w->t_gemm_start_rel, w->t_gemm_end_rel);
    double copy_sr = timeline_span_start_value(w->t_copy_start_rel, w->t_copy_end_rel);
    double copy_er = timeline_span_end_value(w->t_copy_start_rel, w->t_copy_end_rel);
    double copyin_sr = timeline_span_start_value(w->t_copyin_start_rel, w->t_copyin_end_rel);
    double copyin_er = timeline_span_end_value(w->t_copyin_start_rel, w->t_copyin_end_rel);
    double copyout_sr = timeline_span_start_value(w->t_copyout_start_rel, w->t_copyout_end_rel);
    double copyout_er = timeline_span_end_value(w->t_copyout_start_rel, w->t_copyout_end_rel);
    double cleanup_sr = timeline_span_start_value(w->t_cleanup_start_rel, w->t_cleanup_end_rel);
    double cleanup_er = timeline_span_end_value(w->t_cleanup_start_rel, w->t_cleanup_end_rel);
    make_abs_from_positive_rel_span(w->t_ref, w->t_init_start_rel, w->t_init_end_rel, i0, sizeof(i0), i1, sizeof(i1));
    make_abs_from_positive_rel_span(w->t_ref, w->t_gemm_start_rel, w->t_gemm_end_rel, g0, sizeof(g0), g1, sizeof(g1));
    make_abs_from_positive_rel_span(w->t_ref, w->t_compute_start_rel, w->t_compute_end_rel, cm0, sizeof(cm0), cm1, sizeof(cm1));
    make_abs_from_positive_rel_span(w->t_ref, w->t_copy_start_rel, w->t_copy_end_rel, c0, sizeof(c0), c1, sizeof(c1));
    make_abs_from_positive_rel_span(w->t_ref, w->t_cleanup_start_rel, w->t_cleanup_end_rel, x0, sizeof(x0), x1, sizeof(x1));
    make_abs_from_positive_rel_span(w->t_ref, w->t_copyin_start_rel, w->t_copyin_end_rel, ci0, sizeof(ci0), ci1, sizeof(ci1));
    make_abs_from_positive_rel_span(w->t_ref, w->t_copyout_start_rel, w->t_copyout_end_rel, co0, sizeof(co0), co1, sizeof(co1));
    printf("DEV %s enabled=%d rows_assigned=%lld rows_done=%lld share_assigned=%.2f%% share_done=%.2f%% r0=%d r1=%d status=%s err=%d init=%.6f gemm=%.6f copy=%.6f cleanup=%.6f total=%.6f gflops_gemm=%.3f gflops_eff=%.3f rows_per_s=%.3f copy_bw_gbs=%.3f copy_bytes=%llu copyin_bytes=%llu copyout_bytes=%llu dev_start_rel=%.9f dev_end_rel=%.9f init_start_abs=%s init_end_abs=%s compute_start_abs=%s compute_end_abs=%s copy_start_abs=%s copy_end_abs=%s copyin_start_abs=%s copyin_end_abs=%s copyout_start_abs=%s copyout_end_abs=%s cleanup_start_abs=%s cleanup_end_abs=%s init_start_rel=%.9f init_end_rel=%.9f compute_start_rel=%.9f compute_end_rel=%.9f gemm_start_rel=%.9f gemm_end_rel=%.9f copy_start_rel=%.9f copy_end_rel=%.9f copyin_start_rel=%.9f copyin_end_rel=%.9f copyout_start_rel=%.9f copyout_end_rel=%.9f cleanup_start_rel=%.9f cleanup_end_rel=%.9f\n",
           name, w->enabled, rows_assigned, w->rows_done, share_assigned, share_done, w->r0, w->r1, w->status, w->err,
           sane(w->init_time), sane(w->gemm_time), sane(w->copy_time), sane(w->cleanup_time), sane(w->total_time),
           sane(w->gflops_gemm), sane(w->gflops_effective), sane(w->rows_per_s_eff), sane(w->copy_bw_gbs),
           w->copy_bytes, w->copyin_bytes, w->copyout_bytes,
           timeline_rel_value(w->t_dev_start_rel), timeline_rel_value(w->t_dev_end_rel),
           i0, i1, cm0, cm1, c0, c1, ci0, ci1, co0, co1, x0, x1,
           init_sr, init_er,
           compute_sr, compute_er,
           gemm_sr, gemm_er,
           copy_sr, copy_er,
           copyin_sr, copyin_er,
           copyout_sr, copyout_er,
           cleanup_sr, cleanup_er);
}


const char *dev_name_lower_from_wid(int wid) {
    return (wid == 0) ? "cpu" : ((wid == 1) ? "gpu0" : "gpu1");
}

void print_timeline_event_line(const char *prefix, const char *mode, int rank,
                                      const char *lane, const char *dev,
                                      int panel_idx, int j0, int jb, int rows,
                                      const char *phase,
                                      double start_rel, double end_rel,
                                      double root_abs,
                                      unsigned long long bytes) {
    if (!valid_positive_rel_span(start_rel, end_rel)) return;
    char s_abs[64], e_abs[64];
    make_abs_from_root(root_abs, start_rel, s_abs, sizeof(s_abs));
    make_abs_from_root(root_abs, end_rel, e_abs, sizeof(e_abs));
    printf("%s mode=%s rank=%d lane=%s dev=%s panel_idx=%d j0=%d jb=%d rows=%d phase=%s start_abs=%s end_abs=%s start_rel=%.9f end_rel=%.9f duration=%.9f bytes=%llu\n",
           prefix ? prefix : "TIMELINE_EVENT",
           mode ? mode : "unknown",
           rank,
           lane ? lane : "unknown",
           dev ? dev : "unknown",
           panel_idx, j0, jb, rows,
           phase ? phase : "unknown",
           s_abs, e_abs,
           timeline_rel_value(start_rel), timeline_rel_value(end_rel), nonneg_or_zero(end_rel - start_rel), bytes);
}

void print_timeline_tile_event_line(const char *prefix, const char *mode, int rank,
                                           const char *lane, const char *dev,
                                           int panel_idx, int panel_j0, int panel_jb,
                                           int tile_idx, int tile_j0, int tile_jb,
                                           int k0, int kb, int rows,
                                           const char *phase,
                                           double start_rel, double end_rel,
                                           double root_abs,
                                           unsigned long long bytes) {
    if (!valid_positive_rel_span(start_rel, end_rel)) return;
    char s_abs[64], e_abs[64];
    make_abs_from_root(root_abs, start_rel, s_abs, sizeof(s_abs));
    make_abs_from_root(root_abs, end_rel, e_abs, sizeof(e_abs));
    printf("%s mode=%s rank=%d lane=%s dev=%s panel_idx=%d panel_j0=%d panel_jb=%d tile_idx=%d tile_j0=%d tile_jb=%d k0=%d kb=%d rows=%d phase=%s start_abs=%s end_abs=%s start_rel=%.9f end_rel=%.9f duration=%.9f bytes=%llu\n",
           prefix ? prefix : "TIMELINE_TILE_EVENT",
           mode ? mode : "unknown",
           rank,
           lane ? lane : "unknown",
           dev ? dev : "unknown",
           panel_idx, panel_j0, panel_jb, tile_idx, tile_j0, tile_jb, k0, kb, rows,
           phase ? phase : "unknown",
           s_abs, e_abs,
           timeline_rel_value(start_rel), timeline_rel_value(end_rel), nonneg_or_zero(end_rel - start_rel), bytes);
}

void print_worker_timeline_events(const char *prefix, const char *mode, int rank,
                                         int panel_idx, int j0, int jb,
                                         const worker_t *w) {
    if (!w) return;
    const char *dev = dev_name_lower_from_wid(w->wid);
    int rows = (w->r1 > w->r0) ? (w->r1 - w->r0) : (int)((w->rows_done > 0) ? w->rows_done : 0);
    print_timeline_event_line(prefix, mode, rank, "device", dev, panel_idx, j0, jb, rows,
                              "init", w->t_init_start_rel, w->t_init_end_rel, w->t_ref, 0ULL);
    print_timeline_event_line(prefix, mode, rank, "comm", dev, panel_idx, j0, jb, rows,
                              "copyin", w->t_copyin_start_rel, w->t_copyin_end_rel, w->t_ref, w->copyin_bytes);
    print_timeline_event_line(prefix, mode, rank, "device", dev, panel_idx, j0, jb, rows,
                              "gemm", w->t_gemm_start_rel, w->t_gemm_end_rel, w->t_ref, 0ULL);
    print_timeline_event_line(prefix, mode, rank, "comm", dev, panel_idx, j0, jb, rows,
                              "copyout", w->t_copyout_start_rel, w->t_copyout_end_rel, w->t_ref, w->copyout_bytes);
    print_timeline_event_line(prefix, mode, rank, "device", dev, panel_idx, j0, jb, rows,
                              "cleanup", w->t_cleanup_start_rel, w->t_cleanup_end_rel, w->t_ref, 0ULL);
}

void print_panel_worker_timeline_events(const char *prefix, const char *mode, int rank,
                                               int panel_idx, int j0, int jb,
                                               const panel_worker_t *w) {
    if (!w) return;
    const char *dev = dev_name_lower_from_wid(w->wid);
    int rows = (w->r1 > w->r0) ? (w->r1 - w->r0) : (int)((w->rows_done > 0) ? w->rows_done : 0);
    print_timeline_event_line(prefix, mode, rank, "device", dev, panel_idx, j0, jb, rows,
                              "init", w->t_init_start_rel, w->t_init_end_rel, w->t_ref, 0ULL);
    print_timeline_event_line(prefix, mode, rank, "comm", dev, panel_idx, j0, jb, rows,
                              "copyin", w->t_copyin_start_rel, w->t_copyin_end_rel, w->t_ref, w->copyin_bytes);
    print_timeline_event_line(prefix, mode, rank, "device", dev, panel_idx, j0, jb, rows,
                              "gemm", w->t_gemm_start_rel, w->t_gemm_end_rel, w->t_ref, 0ULL);
    print_timeline_event_line(prefix, mode, rank, "comm", dev, panel_idx, j0, jb, rows,
                              "copyout", w->t_copyout_start_rel, w->t_copyout_end_rel, w->t_ref, w->copyout_bytes);
    print_timeline_event_line(prefix, mode, rank, "device", dev, panel_idx, j0, jb, rows,
                              "cleanup", w->t_cleanup_start_rel, w->t_cleanup_end_rel, w->t_ref, 0ULL);
}

void print_mpi_comm_timeline_events(const mpi_timeline_comm_t ev[3], double root_abs) {
    if (!ev) return;
    for (int r = 0; r < 3; r++) {
        char dev[16];
        snprintf(dev, sizeof(dev), "%s", dev_name_lower_from_wid(r));
        print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", ev[r].rank, "comm", dev,
                                  ev[r].panel_idx, ev[r].j0, ev[r].jb, ev[r].rows,
                                  "mpi_bcast_B", ev[r].bcast_start_rel, ev[r].bcast_end_rel,
                                  root_abs, ev[r].bcast_bytes);
        print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", ev[r].rank, "comm", dev,
                                  ev[r].panel_idx, ev[r].j0, ev[r].jb, ev[r].rows,
                                  "mpi_scatter_A", ev[r].scatter_start_rel, ev[r].scatter_end_rel,
                                  root_abs, ev[r].scatter_bytes);
        print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", ev[r].rank, "comm", dev,
                                  ev[r].panel_idx, ev[r].j0, ev[r].jb, ev[r].rows,
                                  "mpi_gather_C", ev[r].gather_start_rel, ev[r].gather_end_rel,
                                  root_abs, ev[r].gather_bytes);
    }
}

void print_part_percent_done(const worker_t w[3], int M) {
    double cpu_pct = 0.0, g0_pct = 0.0, g1_pct = 0.0;
    double cpu_ass = 0.0, g0_ass = 0.0, g1_ass = 0.0;
    if (M > 0) {
        cpu_pct = 100.0 * (double)w[0].rows_done / (double)M;
        g0_pct  = 100.0 * (double)w[1].rows_done / (double)M;
        g1_pct  = 100.0 * (double)w[2].rows_done / (double)M;
        cpu_ass = 100.0 * (double)(w[0].r1 - w[0].r0) / (double)M;
        g0_ass  = 100.0 * (double)(w[1].r1 - w[1].r0) / (double)M;
        g1_ass  = 100.0 * (double)(w[2].r1 - w[2].r0) / (double)M;
    }
    printf("PART_PCT_DONE cpu=%.2f gpu0=%.2f gpu1=%.2f\n", sane(cpu_pct), sane(g0_pct), sane(g1_pct));
    printf("PART_PCT_ASSIGNED cpu=%.2f gpu0=%.2f gpu1=%.2f\n", sane(cpu_ass), sane(g0_ass), sane(g1_ass));
}

void print_device_speeds(const worker_t w[3]) {
    printf("SPEED cpu_gflops_eff=%.3f gpu0_gflops_eff=%.3f gpu1_gflops_eff=%.3f cpu_gflops_gemm=%.3f gpu0_gflops_gemm=%.3f gpu1_gflops_gemm=%.3f cpu_rows_s=%.3f gpu0_rows_s=%.3f gpu1_rows_s=%.3f gpu0_copy_bw_gbs=%.3f gpu1_copy_bw_gbs=%.3f\n",
           sane(w[0].gflops_effective), sane(w[1].gflops_effective), sane(w[2].gflops_effective),
           sane(w[0].gflops_gemm), sane(w[1].gflops_gemm), sane(w[2].gflops_gemm),
           sane(w[0].rows_per_s_eff), sane(w[1].rows_per_s_eff), sane(w[2].rows_per_s_eff),
           sane(w[1].copy_bw_gbs), sane(w[2].copy_bw_gbs));
}

int fill_mats(int N, dtype_t dt, void *A, void *B, void *C) {
    if (dt == DT_REAL_DOUBLE) {
        double *Ad = (double*)A, *Bd = (double*)B, *Cd = (double*)C;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            Ad[(size_t)i*N + j] = (double)(i + j);
            Bd[(size_t)i*N + j] = (double)(i - j);
            Cd[(size_t)i*N + j] = 0.0;
        }
        return 0;
    }
    float *As = (float*)A, *Bs = (float*)B, *Cs = (float*)C;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        As[(size_t)i*N + j] = (float)(i + j);
        Bs[(size_t)i*N + j] = (float)(i - j);
        Cs[(size_t)i*N + j] = 0.0f;
    }
    return 0;
}


int fill_A_C_only(int N, dtype_t dt, void *A, void *C) {
    if (dt == DT_REAL_DOUBLE) {
        double *Ad = (double*)A, *Cd = (double*)C;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            Ad[(size_t)i*N + j] = (double)(i + j);
            Cd[(size_t)i*N + j] = 0.0;
        }
        return 0;
    }
    float *As = (float*)A, *Cs = (float*)C;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        As[(size_t)i*N + j] = (float)(i + j);
        Cs[(size_t)i*N + j] = 0.0f;
    }
    return 0;
}

int fill_B_only(int N, dtype_t dt, void *B) {
    if (dt == DT_REAL_DOUBLE) {
        double *Bd = (double*)B;
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
            Bd[(size_t)i * (size_t)N + (size_t)j] = (double)(i - j);
        }
        return 0;
    }
    float *Bs = (float*)B;
    for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) {
        Bs[(size_t)i * (size_t)N + (size_t)j] = (float)(i - j);
    }
    return 0;
}

int fill_A_C_rows(int N, int global_row0, int rows, dtype_t dt, void *A, void *C) {
    if (rows <= 0) return 0;
    if (dt == DT_REAL_DOUBLE) {
        double *Ad = (double*)A, *Cd = (double*)C;
        for (int ii = 0; ii < rows; ii++) {
            int gi = global_row0 + ii;
            for (int j = 0; j < N; j++) {
                Ad[(size_t)ii * (size_t)N + (size_t)j] = (double)(gi + j);
                Cd[(size_t)ii * (size_t)N + (size_t)j] = 0.0;
            }
        }
        return 0;
    }
    float *As = (float*)A, *Cs = (float*)C;
    for (int ii = 0; ii < rows; ii++) {
        int gi = global_row0 + ii;
        for (int j = 0; j < N; j++) {
            As[(size_t)ii * (size_t)N + (size_t)j] = (float)(gi + j);
            Cs[(size_t)ii * (size_t)N + (size_t)j] = 0.0f;
        }
    }
    return 0;
}


int fill_A_rows_only(int N, int global_row0, int rows, dtype_t dt, void *A) {
    if (rows <= 0) return 0;
    if (dt == DT_REAL_DOUBLE) {
        double *Ad = (double*)A;
        for (int ii = 0; ii < rows; ii++) {
            int gi = global_row0 + ii;
            for (int j = 0; j < N; j++) {
                Ad[(size_t)ii * (size_t)N + (size_t)j] = (double)(gi + j);
            }
        }
        return 0;
    }
    float *As = (float*)A;
    for (int ii = 0; ii < rows; ii++) {
        int gi = global_row0 + ii;
        for (int j = 0; j < N; j++) {
            As[(size_t)ii * (size_t)N + (size_t)j] = (float)(gi + j);
        }
    }
    return 0;
}


void fill_B_panel_formula(int N, int j0, int jb, dtype_t dt, void *Bpanel) {
    if (dt == DT_REAL_DOUBLE) {
        double *Bd = (double*)Bpanel;
        for (int k = 0; k < N; k++) {
            for (int j = 0; j < jb; j++) {
                Bd[(size_t)k * (size_t)jb + (size_t)j] = (double)(k - (j0 + j));
            }
        }
        return;
    }
    float *Bs = (float*)Bpanel;
    for (int k = 0; k < N; k++) {
        for (int j = 0; j < jb; j++) {
            Bs[(size_t)k * (size_t)jb + (size_t)j] = (float)(k - (j0 + j));
        }
    }
}

unsigned long long read_memtotal_bytes(void) {
    FILE *fp = fopen("/proc/meminfo", "r");
    if (!fp) return 0ULL;
    char line[256];
    unsigned long long kb = 0ULL;
    while (fgets(line, sizeof(line), fp)) {
        if (sscanf(line, "MemTotal: %llu kB", &kb) == 1) {
            fclose(fp);
            return kb * 1024ULL;
        }
    }
    fclose(fp);
    return 0ULL;
}

unsigned long long read_mem_reserve_bytes_env(void) {
    const char *s = getenv("HYBRID15_MEM_RESERVE_MB");
    if (!s || !*s) return 0ULL;
    char *end = NULL;
    unsigned long long mb = strtoull(s, &end, 10);
    if (end == s) return 0ULL;
    return mb * 1024ULL * 1024ULL;
}

unsigned long long usable_memtotal_bytes(void) {
    unsigned long long memtotal = read_memtotal_bytes();
    if (memtotal == 0ULL) return 0ULL;
    unsigned long long reserve = read_mem_reserve_bytes_env();
    if (reserve >= memtotal) return 0ULL;
    return memtotal - reserve;
}

int requested_panel_cols_limit(int N, int tile_req) {
    if (N <= 0) return 1;
    if (tile_req > 0 && tile_req < N) return tile_req;
    return N;
}

int clamp_panel_cols_ull(unsigned long long cols, int N, int tile_req) {
    unsigned long long max_cols = (unsigned long long)requested_panel_cols_limit(N, tile_req);
    if (cols < 1ULL) cols = 1ULL;
    if (cols > max_cols) cols = max_cols;
    if (cols > (unsigned long long)N) cols = (unsigned long long)N;
    return (int)cols;
}

int choose_pthreads_panel_cols(int N, size_t esz, int tile_req) {
    (void)esz;
    (void)tile_req;
    if (N <= 0) return 1;

    /*
     * Full-compute mode: do NOT clamp CPU/pthreads panel width by a fixed memory
     * budget or by host MemTotal. If panel mode is selected, use exactly one
     * full-width B panel. Allocation failure is reported honestly instead of
     * silently shrinking the computation into multiple panels.
     */
    return N;
}

int choose_mpi_panel_cols(int N, size_t esz, int tile_req, int root_rows) {
    (void)esz;
    (void)root_rows;
    if (N <= 0) return 1;

    /*
     * MPI panel-B mode: use the requested tile as the B-panel width.
     * Large direct MPI GEMM passes 5000 for DGEMM and 10000 for SGEMM;
     * fallback panel-B passes the default fallback tile.
     */
    int cols = (tile_req > 0) ? tile_req : 2048;
    if (cols < 1) cols = 1;
    if (cols > N) cols = N;
    return cols;
}
void print_run_summary(const char *mode_name, const char *variant, const char *config_name, int N, int np, dtype_t dt,
                              int use_cpu, int use_gpu0, int use_gpu1, int cpu_blas, int gpu_blas, int cpu_threads, int tile,
                              const char *cache_fit, const char *status, double total_s, double total_gflops, double c00, double checksum,
                              int bench_rows[3], double bench_rates[3]) {
    printf("RUN_SUMMARY mode=%s variant=%s config=%s N=%d np=%d dtype=%s cpu_impl=%s gpu_impl=%s use_cpu=%d use_gpu0=%d use_gpu1=%d cpu_threads=%d tile=%d cache_fit_B=%s status=%s total_s=%.6f total_gflops=%.3f c00=%.6f checksum_sample=%.6e bench_cpu_rows=%d bench_gpu0_rows=%d bench_gpu1_rows=%d bench_cpu_rate=%.6f bench_gpu0_rate=%.6f bench_gpu1_rate=%.6f\n",
           mode_name, variant, config_name, N, np, dtype_name(dt), cpu_blas ? "BLAS" : "NAIVE", gpu_blas ? "CUBLAS" : "OPENACC_NAIVE",
           use_cpu, use_gpu0, use_gpu1, cpu_threads, tile, cache_fit, status, sane(total_s), sane(total_gflops), sane(c00), sane(checksum),
           bench_rows[0], bench_rows[1], bench_rows[2], sane(bench_rates[0]), sane(bench_rates[1]), sane(bench_rates[2]));
}

