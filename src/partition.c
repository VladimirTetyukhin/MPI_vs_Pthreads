#include "hybrid_common.h"

/* Run configuration predicates and CPU/GPU row partitioning policy. */

void bench_cpu_rows(int N, int rows_bench, int cpu_blas, dtype_t dt, void *A, void *B, double *t_gemm, int *ok) {
    *ok = 0;
    if (rows_bench <= 0) { *t_gemm = 1e9; return; }
    size_t esz = dtype_size(dt);
    void *Ctmp = malloc_aligned64((size_t)rows_bench * (size_t)N * esz);
    if (!Ctmp) { *t_gemm = 1e9; return; }
    double t0 = tnow();
    if (cpu_blas) cpu_blas_real(N, rows_bench, dt, A, B, Ctmp);
    else {
        if (dt == DT_REAL_DOUBLE) cpu_naive_d(N, rows_bench, (const double*)A, (const double*)B, (double*)Ctmp);
        else cpu_naive_s(N, rows_bench, (const float*)A, (const float*)B, (float*)Ctmp);
    }
    double t1 = tnow();
    *t_gemm = t1 - t0;
    *ok = 1;
    free(Ctmp);
}


double compute_checksum(dtype_t dt, const void *C, long long n) {
    return dt == DT_REAL_DOUBLE ? checksum_real_double((const double*)C, n) : checksum_real_float((const float*)C, n);
}

double read_c00(dtype_t dt, const void *C) {
    return dt == DT_REAL_DOUBLE ? ((const double*)C)[0] : (double)((const float*)C)[0];
}

int validate_partition(int M, int cpu_r0, int cpu_r1, int g0_r0, int g0_r1, int g1_r0, int g1_r1) {
    if (M < 0) return -1;
    if (cpu_r0 < 0 || cpu_r1 < cpu_r0 || cpu_r1 > M) return -1;
    if (g0_r0 < 0 || g0_r1 < g0_r0 || g0_r1 > M) return -1;
    if (g1_r0 < 0 || g1_r1 < g1_r0 || g1_r1 > M) return -1;
    int prev = 0;
    int starts[3] = { cpu_r0, g0_r0, g1_r0 };
    int ends[3] = { cpu_r1, g0_r1, g1_r1 };
    for (int i = 0; i < 3; i++) {
        if (ends[i] <= starts[i]) continue;
        if (starts[i] != prev) return -1;
        prev = ends[i];
    }
    return prev == M ? 0 : -1;
}


int streq(const char *a, const char *b) {
    return a && b && strcmp(a, b) == 0;
}

int variant_requests_blas_gemm(const char *variant) {
    /*
     * dgemm/sgemm variants request library GEMM kernels in both modes.
     * CPU workers use cblas_dgemm/cblas_sgemm; GPU workers use cuBLAS when
     * this binary is compiled with USE_CUBLAS.
     */
    return streq(variant, "dgemm") || streq(variant, "sgemm");
}

int mpi_direct_panel_policy(int N, dtype_t dt, const char *variant, int *tile_out) {
    /*
     * Large MPI BLAS-GEMM runs use panel-B immediately, but they still use
     * the normal benchmark-rate partitioner.  The tile/panel_cols size is
     * dtype-specific:
     *   DGEMM: N >= 34000 -> 5000 columns
     *   SGEMM: N >= 42000 -> 10000 columns
     */
    if (tile_out) *tile_out = 0;
    if (streq(variant, "dgemm") || (variant_requests_blas_gemm(variant) && dt == DT_REAL_DOUBLE)) {
        if (N >= MPI_DGEMM_DIRECT_PANEL_N_THRESHOLD) {
            if (tile_out) *tile_out = MPI_DGEMM_DIRECT_PANEL_TILE;
            return MPI_DGEMM_DIRECT_PANEL_N_THRESHOLD;
        }
        return 0;
    }
    if (streq(variant, "sgemm") || (variant_requests_blas_gemm(variant) && dt == DT_REAL_FLOAT)) {
        if (N >= MPI_SGEMM_DIRECT_PANEL_N_THRESHOLD) {
            if (tile_out) *tile_out = MPI_SGEMM_DIRECT_PANEL_TILE;
            return MPI_SGEMM_DIRECT_PANEL_N_THRESHOLD;
        }
        return 0;
    }
    return 0;
}


int is_naive_variant(const char *variant) {
    return streq(variant, "naive_d") || streq(variant, "naive_s") ||
           streq(variant, "dgemm_naive") || streq(variant, "sgemm_naive");
}

int config_force_nopanel(const char *config_name) {
    if (!config_name || !*config_name) return 0;
    return strstr(config_name, "nopanel") != NULL ||
           strstr(config_name, "no_panel") != NULL ||
           strstr(config_name, "fullb") != NULL ||
           strstr(config_name, "full_b") != NULL;
}

int config_force_panel(const char *config_name) {
    if (!config_name || !*config_name) return 0;
    if (config_force_nopanel(config_name)) return 0;
    return strstr(config_name, "panel") != NULL ||
           strstr(config_name, "panelb") != NULL ||
           strstr(config_name, "panel_b") != NULL;
}

int resolve_panel_b_mode(const char *config_name, int auto_mode) {
    if (config_force_nopanel(config_name)) return 0;
    if (config_force_panel(config_name)) return 1;
    return auto_mode ? 1 : 0;
}

double positive_or_zero(double x) {
    return (isfinite(x) && x > 0.0) ? x : 0.0;
}

int choose_rows_rate_partition(int rows_total,
                                      int use_cpu, int use_gpu0, int use_gpu1,
                                      const double rates[3],
                                      int *cpu_r0, int *cpu_r1,
                                      int *g0_r0, int *g0_r1,
                                      int *g1_r0, int *g1_r1,
                                      double *cpu_share_out) {
    *cpu_r0 = *cpu_r1 = *g0_r0 = *g0_r1 = *g1_r0 = *g1_r1 = 0;
    if (cpu_share_out) *cpu_share_out = 0.0;
    if (rows_total <= 0 || !rates) return -1;

    int enabled[3] = { use_cpu ? 1 : 0, use_gpu0 ? 1 : 0, use_gpu1 ? 1 : 0 };
    double rr[3] = {
        enabled[0] ? positive_or_zero(rates[0]) : 0.0,
        enabled[1] ? positive_or_zero(rates[1]) : 0.0,
        enabled[2] ? positive_or_zero(rates[2]) : 0.0
    };
    double sum = rr[0] + rr[1] + rr[2];
    if (!(sum > 0.0) || !isfinite(sum)) return -1;

    double exact[3] = {0.0, 0.0, 0.0};
    int cnt[3] = {0, 0, 0};
    int active = 0;
    for (int i = 0; i < 3; i++) {
        if (rr[i] > 0.0) {
            active++;
            exact[i] = (double)rows_total * rr[i] / sum;
            cnt[i] = (int)floor(exact[i]);
        }
    }

    if (active <= 0) return -1;
    if (rows_total >= active) {
        for (int i = 0; i < 3; i++) {
            if (rr[i] > 0.0 && cnt[i] < 1) cnt[i] = 1;
        }
    }

    int assigned = cnt[0] + cnt[1] + cnt[2];
    while (assigned > rows_total) {
        int best = -1;
        int best_cnt = -1;
        for (int i = 0; i < 3; i++) {
            int min_cnt = (rows_total >= active && rr[i] > 0.0) ? 1 : 0;
            if (cnt[i] > min_cnt && cnt[i] > best_cnt) {
                best = i;
                best_cnt = cnt[i];
            }
        }
        if (best < 0) break;
        cnt[best]--;
        assigned--;
    }
    while (assigned < rows_total) {
        int best = -1;
        double best_frac = -1.0;
        for (int i = 0; i < 3; i++) {
            if (rr[i] <= 0.0) continue;
            double frac = exact[i] - floor(exact[i]);
            if (frac > best_frac || (frac == best_frac && rr[i] > (best >= 0 ? rr[best] : 0.0))) {
                best = i;
                best_frac = frac;
            }
        }
        if (best < 0) break;
        cnt[best]++;
        assigned++;
    }

    int pos = 0;
    *cpu_r0 = pos; *cpu_r1 = pos + cnt[0]; pos = *cpu_r1;
    *g0_r0 = pos;  *g0_r1 = pos + cnt[1];  pos = *g0_r1;
    *g1_r0 = pos;  *g1_r1 = pos + cnt[2];
    if (cpu_share_out) *cpu_share_out = (rows_total > 0) ? ((double)cnt[0] / (double)rows_total) : 0.0;
    return (pos + cnt[2] == rows_total) ? 0 : -1;
}

int choose_rows_partition_for_variant(int N_global, int rows_total,
                                             int use_cpu, int use_gpu0, int use_gpu1,
                                             const char *variant, const double rates[3],
                                             int prefer_shared_blas_partition,
                                             int *cpu_r0, int *cpu_r1,
                                             int *g0_r0, int *g0_r1,
                                             int *g1_r0, int *g1_r1,
                                             double *cpu_share_out,
                                             int *used_shared_blas_partition_out) {
    (void)N_global;
    (void)prefer_shared_blas_partition;
    if (used_shared_blas_partition_out) *used_shared_blas_partition_out = 0;

    /*
     * One partition policy for every variant, including DGEMM/SGEMM:
     * use measured N/4 rows/second rates from the benchmark and assign
     * rows proportionally. There is no hard-coded BLAS CPU-share formula.
     */
    if (is_naive_variant(variant)) use_cpu = 0;
    return choose_rows_rate_partition(rows_total, use_cpu, use_gpu0, use_gpu1, rates,
                                      cpu_r0, cpu_r1, g0_r0, g0_r1, g1_r0, g1_r1,
                                      cpu_share_out);
}

double worker_partition_rate(const worker_t *w) {
    if (!w || !w->enabled || w->err != 0 || w->rows_done <= 0) return 0.0;
    double eff_s = (w->wid == 0) ? w->gemm_time : (w->gemm_time + w->copy_time);
    if (!(eff_s > 0.0) || !isfinite(eff_s)) eff_s = w->total_time;
    return (eff_s > 0.0 && isfinite(eff_s)) ? ((double)w->rows_done / eff_s) : 0.0;
}

double panel_worker_partition_rate(const panel_worker_t *w) {
    if (!w || !w->enabled || w->err != 0 || w->rows_done <= 0) return 0.0;
    double eff_s = (w->wid == 0) ? w->gemm_time : (w->gemm_time + w->copy_time);
    if (!(eff_s > 0.0) || !isfinite(eff_s)) eff_s = w->total_time;
    return (eff_s > 0.0 && isfinite(eff_s)) ? ((double)w->rows_done / eff_s) : 0.0;
}

