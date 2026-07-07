#include "hybrid_common.h"

/* N/4 calibration benchmarks and fixed-split panel benchmark support. */

int n4_rows_for_bench(int N) {
    int rb = N / 4;
    return rb > 0 ? rb : 1;
}

void print_bench_n4_speed(const char *mode, int N, int cols, int rows_target,
                                 const int bench_rows[3], const double rates[3],
                                 double cpu_s, double g0_s, double g1_s,
                                 double cpu_gflops, double g0_gflops, double g1_gflops) {
    printf("BENCH_N4_SPEED mode=%s N=%d cols=%d bench_rows_target=%d rows_cpu=%d rows_gpu0=%d rows_gpu1=%d cpu_rows_s=%.6f gpu0_rows_s=%.6f gpu1_rows_s=%.6f cpu_s=%.6f gpu0_s=%.6f gpu1_s=%.6f cpu_gflops_eff=%.3f gpu0_gflops_eff=%.3f gpu1_gflops_eff=%.3f\n",
           mode ? mode : "unknown", N, cols, rows_target,
           bench_rows ? bench_rows[0] : 0, bench_rows ? bench_rows[1] : 0, bench_rows ? bench_rows[2] : 0,
           rates ? sane(rates[0]) : 0.0, rates ? sane(rates[1]) : 0.0, rates ? sane(rates[2]) : 0.0,
           sane(cpu_s), sane(g0_s), sane(g1_s),
           sane(cpu_gflops), sane(g0_gflops), sane(g1_gflops));
}

int run_n4_pthreads_benchmark_full(int N, int M, void *A, void *B, void *C,
                                          int use_cpu, int use_gpu0, int use_gpu1,
                                          int gpu0_dev, int gpu1_dev,
                                          int cpu_blas, int gpu_blas, int tile, dtype_t dt,
                                          const char *variant,
                                          int bench_rows_out[3], double bench_rate_out[3]) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    if (bench_rows_out) bench_rows_out[0] = bench_rows_out[1] = bench_rows_out[2] = 0;
    if (bench_rate_out) bench_rate_out[0] = bench_rate_out[1] = bench_rate_out[2] = 0.0;
    if (M <= 0 || N <= 0) return -1;
    if (is_naive_variant(variant)) use_cpu = 0;

    int target = n4_rows_for_bench(N);
    int enabled[3] = { use_cpu ? 1 : 0, use_gpu0 ? 1 : 0, use_gpu1 ? 1 : 0 };
    int pos = 0;
    int r0[3] = {0,0,0};
    int r1[3] = {0,0,0};
    for (int i = 0; i < 3; i++) {
        r0[i] = pos;
        int rows = enabled[i] ? target : 0;
        if (pos + rows > M) rows = M - pos;
        if (rows < 0) rows = 0;
        r1[i] = pos + rows;
        pos = r1[i];
    }

    worker_t w[3];
    pthread_t th[3];
    int started[3] = {0,0,0};
    double t_ref = tnow();
    memset(w, 0, sizeof(w));
    w[0] = (worker_t){ .wid=0, .enabled=(enabled[0] && r1[0] > r0[0]), .gpu_dev=-1,       .N=N, .r0=r0[0], .r1=r1[0], .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=B, .C=C, .t_ref=t_ref };
    w[1] = (worker_t){ .wid=1, .enabled=(enabled[1] && r1[1] > r0[1]), .gpu_dev=gpu0_dev, .N=N, .r0=r0[1], .r1=r1[1], .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=B, .C=C, .t_ref=t_ref };
    w[2] = (worker_t){ .wid=2, .enabled=(enabled[2] && r1[2] > r0[2]), .gpu_dev=gpu1_dev, .N=N, .r0=r0[2], .r1=r1[2], .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=B, .C=C, .t_ref=t_ref };
    for (int i = 0; i < 3; i++) {
        int keep_wid = w[i].wid, keep_enabled = w[i].enabled, keep_gpu_dev = w[i].gpu_dev;
        int keep_N = w[i].N, keep_r0 = w[i].r0, keep_r1 = w[i].r1;
        int keep_cpu_blas = w[i].cpu_blas, keep_gpu_blas = w[i].gpu_blas, keep_tile = w[i].tile;
        dtype_t keep_dt = w[i].dt;
        void *keep_A = w[i].A, *keep_B = w[i].B, *keep_C = w[i].C;
        double keep_ref = w[i].t_ref;
        init_worker_timing_fields(&w[i]);
        w[i].wid = keep_wid;
        w[i].enabled = keep_enabled;
        w[i].gpu_dev = keep_gpu_dev;
        w[i].N = keep_N;
        w[i].r0 = keep_r0;
        w[i].r1 = keep_r1;
        w[i].cpu_blas = keep_cpu_blas;
        w[i].gpu_blas = keep_gpu_blas;
        w[i].tile = keep_tile;
        w[i].dt = keep_dt;
        w[i].A = keep_A;
        w[i].B = keep_B;
        w[i].C = keep_C;
        w[i].t_ref = keep_ref;
        snprintf(w[i].status, sizeof(w[i].status), "BENCH_UNSET");
    }

    if (w[1].enabled && pthread_create(&th[1], NULL, worker_main, &w[1]) == 0) started[1] = 1;
    else if (w[1].enabled) { w[1].enabled = 0; w[1].err = -410; snprintf(w[1].status, sizeof(w[1].status), "BENCH_THREAD_CREATE_FAIL"); }
    if (w[2].enabled && pthread_create(&th[2], NULL, worker_main, &w[2]) == 0) started[2] = 1;
    else if (w[2].enabled) { w[2].enabled = 0; w[2].err = -411; snprintf(w[2].status, sizeof(w[2].status), "BENCH_THREAD_CREATE_FAIL"); }
    if (w[0].enabled && pthread_create(&th[0], NULL, worker_main, &w[0]) == 0) started[0] = 1;
    else if (w[0].enabled) { w[0].enabled = 0; w[0].err = -412; snprintf(w[0].status, sizeof(w[0].status), "BENCH_THREAD_CREATE_FAIL"); }

    for (int i = 0; i < 3; i++) if (started[i]) pthread_join(th[i], NULL);

    int rows_done[3] = {0,0,0};
    double rates[3] = {0.0,0.0,0.0};
    int ok_rates = 0;
    for (int i = 0; i < 3; i++) {
        rows_done[i] = (int)w[i].rows_done;
        rates[i] = worker_partition_rate(&w[i]);
        if (rates[i] > 0.0) ok_rates++;
    }
    if (bench_rows_out) { bench_rows_out[0] = rows_done[0]; bench_rows_out[1] = rows_done[1]; bench_rows_out[2] = rows_done[2]; }
    if (bench_rate_out) { bench_rate_out[0] = rates[0]; bench_rate_out[1] = rates[1]; bench_rate_out[2] = rates[2]; }
    print_bench_n4_speed("pthreads", N, N, target, rows_done, rates,
                         w[0].gemm_time, w[1].gemm_time + w[1].copy_time, w[2].gemm_time + w[2].copy_time,
                         w[0].gflops_effective, w[1].gflops_effective, w[2].gflops_effective);
    return ok_rates > 0 ? 0 : -1;
}

int run_n4_pthreads_benchmark_panel(int N, int cols, int M, void *A, void *Bpanel, void *Cpanel,
                                           int C_ld, int C_col0,
                                           int use_cpu, int use_gpu0, int use_gpu1,
                                           int gpu0_dev, int gpu1_dev,
                                           int cpu_blas, int gpu_blas, int tile, dtype_t dt,
                                           const char *variant,
                                           int bench_rows_out[3], double bench_rate_out[3]) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    if (bench_rows_out) bench_rows_out[0] = bench_rows_out[1] = bench_rows_out[2] = 0;
    if (bench_rate_out) bench_rate_out[0] = bench_rate_out[1] = bench_rate_out[2] = 0.0;
    if (M <= 0 || N <= 0 || cols <= 0) return -1;
    if (is_naive_variant(variant)) use_cpu = 0;

    int target = n4_rows_for_bench(N);
    int enabled[3] = { use_cpu ? 1 : 0, use_gpu0 ? 1 : 0, use_gpu1 ? 1 : 0 };
    int pos = 0;
    int r0[3] = {0,0,0};
    int r1[3] = {0,0,0};
    for (int i = 0; i < 3; i++) {
        r0[i] = pos;
        int rows = enabled[i] ? target : 0;
        if (pos + rows > M) rows = M - pos;
        if (rows < 0) rows = 0;
        r1[i] = pos + rows;
        pos = r1[i];
    }

    panel_worker_t w[3];
    pthread_t th[3];
    int started[3] = {0,0,0};
    double t_ref = tnow();
    memset(w, 0, sizeof(w));
    w[0] = (panel_worker_t){ .wid=0, .enabled=(enabled[0] && r1[0] > r0[0]), .gpu_dev=-1,       .K=N, .cols=cols, .r0=r0[0], .r1=r1[0], .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=Bpanel, .C=Cpanel, .c_ld=C_ld, .c_col0=C_col0, .t_ref=t_ref };
    w[1] = (panel_worker_t){ .wid=1, .enabled=(enabled[1] && r1[1] > r0[1]), .gpu_dev=gpu0_dev, .K=N, .cols=cols, .r0=r0[1], .r1=r1[1], .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=Bpanel, .C=Cpanel, .c_ld=C_ld, .c_col0=C_col0, .t_ref=t_ref };
    w[2] = (panel_worker_t){ .wid=2, .enabled=(enabled[2] && r1[2] > r0[2]), .gpu_dev=gpu1_dev, .K=N, .cols=cols, .r0=r0[2], .r1=r1[2], .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=Bpanel, .C=Cpanel, .c_ld=C_ld, .c_col0=C_col0, .t_ref=t_ref };
    for (int i = 0; i < 3; i++) {
        double keep_ref = w[i].t_ref;
        init_panel_worker_timing_fields(&w[i]);
        w[i].t_ref = keep_ref;
        snprintf(w[i].status, sizeof(w[i].status), "BENCH_UNSET");
    }

    if (w[1].enabled && pthread_create(&th[1], NULL, panel_worker_main, &w[1]) == 0) started[1] = 1;
    else if (w[1].enabled) { w[1].enabled = 0; w[1].err = -430; snprintf(w[1].status, sizeof(w[1].status), "BENCH_THREAD_CREATE_FAIL"); }
    if (w[2].enabled && pthread_create(&th[2], NULL, panel_worker_main, &w[2]) == 0) started[2] = 1;
    else if (w[2].enabled) { w[2].enabled = 0; w[2].err = -431; snprintf(w[2].status, sizeof(w[2].status), "BENCH_THREAD_CREATE_FAIL"); }
    if (w[0].enabled && pthread_create(&th[0], NULL, panel_worker_main, &w[0]) == 0) started[0] = 1;
    else if (w[0].enabled) { w[0].enabled = 0; w[0].err = -432; snprintf(w[0].status, sizeof(w[0].status), "BENCH_THREAD_CREATE_FAIL"); }

    for (int i = 0; i < 3; i++) if (started[i]) pthread_join(th[i], NULL);

    int rows_done[3] = {0,0,0};
    double rates[3] = {0.0,0.0,0.0};
    int ok_rates = 0;
    for (int i = 0; i < 3; i++) {
        finalize_panel_worker_times(&w[i]);
        rows_done[i] = (int)w[i].rows_done;
        rates[i] = panel_worker_partition_rate(&w[i]);
        if (rates[i] > 0.0) ok_rates++;
    }
    double gflops_eff[3] = {0.0, 0.0, 0.0};
    for (int i = 0; i < 3; i++) {
        double eff_s = (i == 0) ? w[i].gemm_time : (w[i].gemm_time + w[i].copy_time);
        double flops = 2.0 * (double)rows_done[i] * (double)N * (double)cols;
        gflops_eff[i] = (eff_s > 0.0) ? ((flops / 1e9) / eff_s) : 0.0;
    }
    if (bench_rows_out) { bench_rows_out[0] = rows_done[0]; bench_rows_out[1] = rows_done[1]; bench_rows_out[2] = rows_done[2]; }
    if (bench_rate_out) { bench_rate_out[0] = rates[0]; bench_rate_out[1] = rates[1]; bench_rate_out[2] = rates[2]; }
    print_bench_n4_speed("pthreads_panel", N, cols, target, rows_done, rates,
                         w[0].gemm_time, w[1].gemm_time + w[1].copy_time, w[2].gemm_time + w[2].copy_time,
                         gflops_eff[0], gflops_eff[1], gflops_eff[2]);
    return ok_rates > 0 ? 0 : -1;
}

int run_fixed_split_buffers_panel(int N, int cols, int M, void *A, void *Bpanel, void *Cpanel,
                                         int C_ld, int C_col0,
                                         int use_cpu, int use_gpu0, int use_gpu1,
                                         int gpu0_dev, int gpu1_dev,
                                         int cpu_blas, int gpu_blas, int tile, dtype_t dt,
                                         const char *variant, const double bench_rates_in[3],
                                         panel_worker_t out_w[3], double *t_total_out,
                                         double t_ref_base, int panel_idx_for_trace) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu < 1) use_gpu0 = 0;
    if (ngpu < 2) use_gpu1 = 0;
    if (is_naive_variant(variant)) use_cpu = 0;
    if (!use_cpu && !use_gpu0 && !use_gpu1) {
        if (t_total_out) *t_total_out = 0.0;
        init_empty_panel_workers(out_w, "SKIPPED", -500);
        return -500;
    }
    int cpu_r0, cpu_r1, g0_r0, g0_r1, g1_r0, g1_r1;
    double cpu_share = 0.0;
    int used_shared_blas_partition = 0;
    int part_rc = choose_rows_partition_for_variant(N, M, use_cpu, use_gpu0, use_gpu1, variant, bench_rates_in,
                                                    1,
                                                    &cpu_r0, &cpu_r1, &g0_r0, &g0_r1, &g1_r0, &g1_r1, &cpu_share,
                                                    &used_shared_blas_partition);
    (void)used_shared_blas_partition;
    if (part_rc != 0) {
        if (t_total_out) *t_total_out = 0.0;
        init_empty_panel_workers(out_w, "NO_RATE_PARTITION", -502);
        return -502;
    }
    if (validate_partition(M, cpu_r0, cpu_r1, g0_r0, g0_r1, g1_r0, g1_r1) != 0) {
        if (t_total_out) *t_total_out = 0.0;
        init_empty_panel_workers(out_w, "INVALID_PARTITION", -501);
        return -501;
    }
    panel_worker_t w[3];
    pthread_t th[3];
    int started[3] = {0,0,0};
    int wanted[3] = { use_cpu, use_gpu0, use_gpu1 };
    double ts = tnow();
    double t_ref_use = (t_ref_base > 0.0) ? t_ref_base : ts;
    memset(w, 0, sizeof(w));
    w[0] = (panel_worker_t){ .wid=0, .enabled=use_cpu,  .gpu_dev=-1,       .K=N, .cols=cols, .r0=cpu_r0, .r1=cpu_r1, .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=Bpanel, .C=Cpanel, .c_ld=C_ld, .c_col0=C_col0, .t_ref=t_ref_use };
    w[1] = (panel_worker_t){ .wid=1, .enabled=use_gpu0, .gpu_dev=gpu0_dev, .K=N, .cols=cols, .r0=g0_r0,  .r1=g0_r1,  .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=Bpanel, .C=Cpanel, .c_ld=C_ld, .c_col0=C_col0, .t_ref=t_ref_use };
    w[2] = (panel_worker_t){ .wid=2, .enabled=use_gpu1, .gpu_dev=gpu1_dev, .K=N, .cols=cols, .r0=g1_r0,  .r1=g1_r1,  .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=Bpanel, .C=Cpanel, .c_ld=C_ld, .c_col0=C_col0, .t_ref=t_ref_use };
    int trace_panel_idx = panel_idx_for_trace;
    for (int i = 0; i < 3; i++) {
        double keep_ref = w[i].t_ref;
        init_panel_worker_timing_fields(&w[i]);
        w[i].t_ref = keep_ref;
        set_panel_worker_trace(&w[i], "PTHREADS_TIMELINE_TILE_EVENT", "pthreads", 0, trace_panel_idx, C_col0, cols, t_ref_use, 0.0);
    }

    if (pthread_create(&th[1], NULL, panel_worker_main, &w[1]) == 0) started[1] = 1; else { w[1].enabled = 0; w[1].err = -430; snprintf(w[1].status, sizeof(w[1].status), "THREAD_CREATE_FAIL"); }
    if (pthread_create(&th[2], NULL, panel_worker_main, &w[2]) == 0) started[2] = 1; else { w[2].enabled = 0; w[2].err = -431; snprintf(w[2].status, sizeof(w[2].status), "THREAD_CREATE_FAIL"); }
    if (pthread_create(&th[0], NULL, panel_worker_main, &w[0]) == 0) started[0] = 1; else { w[0].enabled = 0; w[0].err = -432; snprintf(w[0].status, sizeof(w[0].status), "THREAD_CREATE_FAIL"); }

    if (started[0]) pthread_join(th[0], NULL);
    if (started[1]) pthread_join(th[1], NULL);
    if (started[2]) pthread_join(th[2], NULL);
    double te = tnow();

    int create_fail = 0;
    for (int i = 0; i < 3; i++) {
        if (wanted[i] && !started[i]) create_fail = 1;
        finalize_panel_worker_times(&w[i]);
    }
    if (out_w) { out_w[0] = w[0]; out_w[1] = w[1]; out_w[2] = w[2]; }
    if (t_total_out) *t_total_out = te - ts;
    return create_fail ? -430 : 0;
}

