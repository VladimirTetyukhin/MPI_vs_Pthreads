#include "hybrid_common.h"

/* Single-process pthreads execution mode. */

int run_pthreads_mode(int N, int use_cpu, int use_gpu0, int use_gpu1,
                             int cpu_blas, int gpu_blas, int cpu_threads, int tile,
                             dtype_t dt, const char *variant, const char *config_name) {
    double t_mode0 = tnow();
    configure_cpu_threads(cpu_threads);
    if (is_naive_variant(variant)) use_cpu = 0;
    size_t esz = dtype_size(dt);
    int use_panel_b = resolve_panel_b_mode(config_name, 1);

    if (!use_panel_b) {
        unsigned long long bytesA = mat_bytes_NN(N, esz);
        unsigned long long bytesB = mat_bytes_NN(N, esz);
        unsigned long long bytesC = mat_bytes_NN(N, esz);
        double t_alloc0 = tnow();
        void *A = malloc_aligned64((size_t)bytesA);
        void *B = malloc_aligned64((size_t)bytesB);
        void *C = malloc_aligned64((size_t)bytesC);
        double t_alloc1 = tnow();
        if (!A || !B || !C) {
            printf("MODE=PTHREADS N=%d dtype=%s STATUS=FAILED reason=ALLOC_FAIL bytesA=%llu bytesB=%llu bytesC=%llu panel_b=0 panel_cols=%d\n",
                   N, dtype_name(dt), bytesA, bytesB, bytesC, N);
            free(A); free(B); free(C);
            return 2;
        }

        double t_fill0 = tnow();
        if (fill_mats(N, dt, A, B, C) != 0) {
            printf("MODE=PTHREADS N=%d dtype=%s STATUS=FAILED reason=FILL_FAIL panel_b=0 panel_cols=%d\n", N, dtype_name(dt), N);
            free(A); free(B); free(C);
            return 2;
        }
        double t_fill1 = tnow();

        int ngpu = acc_get_num_devices(acc_device_nvidia);
        if (ngpu < 1) use_gpu0 = 0;
        if (ngpu < 2) use_gpu1 = 0;
        char ts_now[64];
        fmt_ts_iso(tnow(), ts_now, sizeof(ts_now));
        printf("MODE=PTHREADS N=%d dtype=%s cpu_impl=%s gpu_impl=%s use_cpu=%d use_gpu0=%d use_gpu1=%d ngpu=%d cpu_threads=%d tile=%d cache_fit_B=%s ts=%s variant=%s config=%s panel_b=0 panel_cols=%d\n",
               N, dtype_name(dt), cpu_blas ? "BLAS" : "NAIVE", gpu_blas ? "CUBLAS" : "OPENACC_NAIVE",
               use_cpu, use_gpu0, use_gpu1, ngpu, (cpu_threads > 0 ? cpu_threads : omp_get_max_threads()), tile,
               cache_fit_label_for_B(N, esz), ts_now, variant, config_name, N);
        print_cache_fit_report(N, esz);
        print_cache_speed_estimates();

        worker_t w[3];
        init_empty_workers(w, N, "UNSET", 0);
        double t_pthreads_total = 0.0;
        char policy[64] = "UNSET";
        int bench_rows[3] = {0,0,0};
        double bench_rates[3] = {0,0,0};
        double n4_bench_wall_s = 0.0;
        int rc_exec = run_fixed_split_buffers(N, N, A, B, C,
                                              use_cpu, use_gpu0, use_gpu1,
                                              use_gpu0 ? 0 : -1, use_gpu1 ? 1 : -1,
                                              cpu_blas, gpu_blas, tile, dt,
                                              variant, w, &t_pthreads_total,
                                              policy, sizeof(policy),
                                              bench_rows, bench_rates, &n4_bench_wall_s, t_mode0);

        print_worker_line(&w[0], N);
        print_worker_line(&w[1], N);
        print_worker_line(&w[2], N);
        print_part_percent_done(w, N);
        print_device_speeds(w);

        double init_start_rel = -1.0, init_end_rel = -1.0;
        double copy_start_rel = -1.0, copy_end_rel = -1.0;
        double compute_start_rel = -1.0, compute_end_rel = -1.0;
        double dev_cleanup_start_rel = -1.0, dev_cleanup_end_rel = -1.0;
        int any_enabled = 0;
        int all_ok = 1;
        for (int i = 0; i < 3; i++) {
            if (!w[i].enabled) continue;
            any_enabled = 1;
            if (w[i].err != 0) all_ok = 0;
            merge_rel_span(&init_start_rel, &init_end_rel, w[i].t_init_start_rel, w[i].t_init_end_rel);
            merge_rel_span(&copy_start_rel, &copy_end_rel, w[i].t_copy_start_rel, w[i].t_copy_end_rel);
            merge_rel_span(&compute_start_rel, &compute_end_rel, w[i].t_compute_start_rel, w[i].t_compute_end_rel);
            merge_rel_span(&dev_cleanup_start_rel, &dev_cleanup_end_rel, w[i].t_cleanup_start_rel, w[i].t_cleanup_end_rel);
        }

        double c00 = read_c00(dt, C);
        double chk = compute_checksum(dt, C, (long long)N*(long long)N);
        const char *status = (!any_enabled) ? "FAILED" : ((rc_exec == 0 && all_ok) ? "OK" : "DEGRADED");

        double t_cleanup0 = tnow();
        free(A); free(B); free(C);
        double t_cleanup1 = tnow();
        double t_full_wall_total = t_cleanup1 - t_mode0;
        double t_full_total = nonneg_or_zero(t_full_wall_total - n4_bench_wall_s);
        double total_flops = 2.0 * (double)N * (double)N * (double)N;
        double gflops_total = (t_full_total > 0.0) ? ((total_flops / 1e9) / t_full_total) : 0.0;

        char full0[64], full1[64], a0[64], a1[64], f0[64], f1[64], pb0[64], pb1[64], pi0[64], pi1[64], pc0[64], pc1[64], pg0[64], pg1[64], px0[64], px1[64], cl0[64], cl1[64];
        fmt_ts_iso(t_mode0, full0, sizeof(full0));
        fmt_ts_iso(t_cleanup1, full1, sizeof(full1));
        fmt_ts_iso(t_alloc0, a0, sizeof(a0));
        fmt_ts_iso(t_alloc1, a1, sizeof(a1));
        fmt_ts_iso(t_fill0, f0, sizeof(f0));
        fmt_ts_iso(t_fill1, f1, sizeof(f1));
        make_abs_from_root(t_mode0, -1.0, pb0, sizeof(pb0));
        make_abs_from_root(t_mode0, -1.0, pb1, sizeof(pb1));
        make_abs_from_root(t_mode0, init_start_rel, pi0, sizeof(pi0));
        make_abs_from_root(t_mode0, init_end_rel, pi1, sizeof(pi1));
        make_abs_from_root(t_mode0, copy_start_rel, pc0, sizeof(pc0));
        make_abs_from_root(t_mode0, copy_end_rel, pc1, sizeof(pc1));
        make_abs_from_root(t_mode0, compute_start_rel, pg0, sizeof(pg0));
        make_abs_from_root(t_mode0, compute_end_rel, pg1, sizeof(pg1));
        make_abs_from_root(t_mode0, dev_cleanup_start_rel, px0, sizeof(px0));
        make_abs_from_root(t_mode0, dev_cleanup_end_rel, px1, sizeof(px1));
        fmt_ts_iso(t_cleanup0, cl0, sizeof(cl0));
        fmt_ts_iso(t_cleanup1, cl1, sizeof(cl1));
        printf("PTHREADS_TIMELINE full_start_abs=%s full_end_abs=%s full_start_rel=0.000000 full_end_rel=%.6f alloc_start_abs=%s alloc_end_abs=%s alloc_start_rel=%.6f alloc_end_rel=%.6f fill_ac_start_abs=%s fill_ac_end_abs=%s fill_ac_start_rel=%.6f fill_ac_end_rel=%.6f panel_fill_start_abs=%s panel_fill_end_abs=%s panel_fill_start_rel=%.6f panel_fill_end_rel=%.6f init_start_abs=%s init_end_abs=%s init_start_rel=%.6f init_end_rel=%.6f copy_start_abs=%s copy_end_abs=%s copy_start_rel=%.6f copy_end_rel=%.6f compute_start_abs=%s compute_end_abs=%s compute_start_rel=%.6f compute_end_rel=%.6f dev_cleanup_start_abs=%s dev_cleanup_end_abs=%s dev_cleanup_start_rel=%.6f dev_cleanup_end_rel=%.6f host_cleanup_start_abs=%s host_cleanup_end_abs=%s host_cleanup_start_rel=%.6f host_cleanup_end_rel=%.6f\n",
               full0, full1, sane(t_full_wall_total),
               a0, a1, sane(t_alloc0 - t_mode0), sane(t_alloc1 - t_mode0),
               f0, f1, sane(t_fill0 - t_mode0), sane(t_fill1 - t_mode0),
               pb0, pb1, timeline_rel_value(-1.0), timeline_rel_value(-1.0),
               pi0, pi1, timeline_rel_value(init_start_rel), timeline_rel_value(init_end_rel),
               pc0, pc1, timeline_rel_value(copy_start_rel), timeline_rel_value(copy_end_rel),
               pg0, pg1, timeline_rel_value(compute_start_rel), timeline_rel_value(compute_end_rel),
               px0, px1, timeline_rel_value(dev_cleanup_start_rel), timeline_rel_value(dev_cleanup_end_rel),
               cl0, cl1, sane(t_cleanup0 - t_mode0), sane(t_cleanup1 - t_mode0));
        print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", -1, 0, N, N, "alloc", t_alloc0 - t_mode0, t_alloc1 - t_mode0, t_mode0, bytesA + bytesB + bytesC);
        print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", -1, 0, N, N, "fill_A_B_C", t_fill0 - t_mode0, t_fill1 - t_mode0, t_mode0, bytesA + bytesB + bytesC);
        for (int d = 0; d < 3; d++) print_worker_timeline_events("PTHREADS_TIMELINE_EVENT", "pthreads", 0, -1, 0, N, &w[d]);
        print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", -1, 0, N, N, "host_cleanup", t_cleanup0 - t_mode0, t_cleanup1 - t_mode0, t_mode0, bytesA + bytesB + bytesC);
        double alloc_s = interval_or_zero(t_alloc0 - t_mode0, t_alloc1 - t_mode0);
        double fill_ac_s = interval_or_zero(t_fill0 - t_mode0, t_fill1 - t_mode0);
        double panel_fill_s = 0.0;
        double host_cleanup_s = interval_or_zero(t_cleanup0 - t_mode0, t_cleanup1 - t_mode0);
        double init_sum_s = 0.0, copy_sum_s = 0.0, compute_sum_s = 0.0, dev_cleanup_sum_s = 0.0;
        double worker_sum_s = 0.0, worker_span_s = 0.0;
        worker_phase_sums(w, &init_sum_s, &copy_sum_s, &compute_sum_s, &dev_cleanup_sum_s, &worker_sum_s, &worker_span_s);
        double init_span_s = interval_or_zero(init_start_rel, init_end_rel);
        double copy_span_s = interval_or_zero(copy_start_rel, copy_end_rel);
        double compute_span_s = interval_or_zero(compute_start_rel, compute_end_rel);
        double dev_cleanup_span_s = interval_or_zero(dev_cleanup_start_rel, dev_cleanup_end_rel);
        double other_wall_s = nonneg_or_zero(t_full_total - (alloc_s + fill_ac_s + panel_fill_s + worker_span_s + host_cleanup_s));
        printf("PTHREADS_TOTAL_TIMING start_abs=%s end_abs=%s start_rel=0.000000 end_rel=%.6f wall_total_s=%.6f n4_calibration_s=%.6f n4_excluded_from_N_benchmark=1 alloc_s=%.6f fill_ac_s=%.6f panel_fill_s=%.6f init_s=%.6f copy_s=%.6f compute_s=%.6f dev_cleanup_s=%.6f host_cleanup_s=%.6f worker_total_s=%.6f worker_sum_s=%.6f worker_span_s=%.6f total_s=%.6f other_s=%.6f init_span_s=%.6f copy_span_s=%.6f compute_span_s=%.6f dev_cleanup_span_s=%.6f panel_fill_span_s=%.6f timing_note=phase_s_are_sum_durations_spans_are_timeline_extents\n",
               full0, full1, sane(t_full_wall_total), sane(t_full_wall_total), sane(n4_bench_wall_s),
               sane(alloc_s), sane(fill_ac_s), sane(panel_fill_s),
               sane(init_sum_s), sane(copy_sum_s), sane(compute_sum_s), sane(dev_cleanup_sum_s),
               sane(host_cleanup_s), sane(t_pthreads_total), sane(worker_sum_s), sane(worker_span_s),
               sane(t_full_total), sane(other_wall_s),
               sane(init_span_s), sane(copy_span_s), sane(compute_span_s), sane(dev_cleanup_span_s), 0.0);
    printf("CALENDAR total=%.6f wall_total=%.6f n4_calibration=%.6f pthreads_total=%.6f policy=%s panel_cols=%d n4_excluded_from_N_benchmark=1\n",
               sane(t_full_total), sane(t_full_wall_total), sane(n4_bench_wall_s), sane(t_pthreads_total), policy, N);
        printf("TOTAL total_s=%.6f wall_total_s=%.6f n4_calibration_s=%.6f pthreads_total_s=%.6f TOTAL_GFLOPS=%.3f STATUS=%s C[0]=%.6f CHECKSUM_SAMPLE=%.6e n4_excluded_from_N_benchmark=1\n",
               sane(t_full_total), sane(t_full_wall_total), sane(n4_bench_wall_s), sane(t_pthreads_total), sane(gflops_total), status, sane(c00), sane(chk));
        print_run_summary("pthreads", variant, config_name, N, 1, dt, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile,
                          cache_fit_label_for_B(N, esz), status, t_full_total, gflops_total, c00, chk, bench_rows, bench_rates);
        return strcmp(status, "OK") == 0 ? 0 : 3;
    }

    int panel_cols = choose_pthreads_panel_cols(N, esz, tile);
    if (panel_cols < 1) panel_cols = 1;
    if (panel_cols > N) panel_cols = N;
    unsigned long long bytesA = mat_bytes_NN(N, esz);
    unsigned long long bytesC = mat_bytes_NN(N, esz);
    unsigned long long bytesBpanel = (unsigned long long)N * (unsigned long long)panel_cols * (unsigned long long)esz;

    double t_alloc0 = tnow();
    void *A = malloc_aligned64((size_t)bytesA);
    void *C = malloc_aligned64((size_t)bytesC);
    void *B_panel = malloc_aligned64((size_t)bytesBpanel);
    double t_alloc1 = tnow();
    if (!A || !C || !B_panel) {
        printf("MODE=PTHREADS N=%d dtype=%s STATUS=FAILED reason=ALLOC_FAIL bytesA=%llu bytesB_panel=%llu bytesC=%llu panel_b=1 panel_cols=%d\n",
               N, dtype_name(dt), bytesA, bytesBpanel, bytesC, panel_cols);
        free(A); free(C); free(B_panel);
        return 2;
    }

    double t_fill_ac0 = tnow();
    if (fill_A_C_only(N, dt, A, C) != 0) {
        printf("MODE=PTHREADS N=%d dtype=%s STATUS=FAILED reason=FILL_FAIL panel_b=1 panel_cols=%d\n", N, dtype_name(dt), panel_cols);
        free(A); free(C); free(B_panel);
        return 2;
    }
    double t_fill_ac1 = tnow();

    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu < 1) use_gpu0 = 0;
    if (ngpu < 2) use_gpu1 = 0;
    char ts_now[64];
    fmt_ts_iso(tnow(), ts_now, sizeof(ts_now));
    printf("MODE=PTHREADS N=%d dtype=%s cpu_impl=%s gpu_impl=%s use_cpu=%d use_gpu0=%d use_gpu1=%d ngpu=%d cpu_threads=%d tile=%d cache_fit_B=%s ts=%s variant=%s config=%s panel_b=1 panel_cols=%d\n",
           N, dtype_name(dt), cpu_blas ? "BLAS" : "NAIVE", gpu_blas ? "CUBLAS" : "OPENACC_NAIVE",
           use_cpu, use_gpu0, use_gpu1, ngpu, (cpu_threads > 0 ? cpu_threads : omp_get_max_threads()), tile,
           cache_fit_label_for_B(N, esz), ts_now, variant, config_name, panel_cols);
    print_cache_fit_report(N, esz);
    print_cache_speed_estimates();
    printf("MEM_BYTES A=%.0fMiB C=%.0fMiB B_panel=%.0fMiB total=%.0fMiB\n",
           bytes_to_mib(bytesA), bytes_to_mib(bytesC), bytes_to_mib(bytesBpanel), bytes_to_mib(bytesA + bytesC + bytesBpanel));

    worker_t w[3];
    init_empty_workers(w, N, "UNSET", 0);
    double t_pthreads_total = 0.0;
    char policy[64] = "UNSET";
    snprintf(policy, sizeof(policy), "BENCH_N4_RATE_PARTITION_PTHREADS_PANEL_B(panel_cols=%d)", panel_cols);
    int bench_rows[3] = {0,0,0};
    double bench_rates[3] = {0,0,0};
    double n4_bench_wall_s = 0.0;
    {
        int bench_cols = (panel_cols < N) ? panel_cols : N;
        if (bench_cols > 0) {
            fill_B_panel_formula(N, 0, bench_cols, dt, B_panel);
            double n4_bench_t0 = tnow();
            int brc = run_n4_pthreads_benchmark_panel(N, bench_cols, N, A, B_panel, C,
                                                       N, 0, use_cpu, use_gpu0, use_gpu1,
                                                       use_gpu0 ? 0 : -1, use_gpu1 ? 1 : -1,
                                                       cpu_blas, gpu_blas, tile, dt, variant,
                                                       bench_rows, bench_rates);
            double n4_bench_t1 = tnow();
            n4_bench_wall_s = nonneg_or_zero(n4_bench_t1 - n4_bench_t0);
            printf("PTHREADS_N4_CALIBRATION mode=pthreads_panel N=%d cols=%d bench_rows_target=%d wall_s=%.6f excluded_from_N_benchmark=1 status=%s\n",
                   N, bench_cols, n4_rows_for_bench(N), sane(n4_bench_wall_s), (brc == 0) ? "OK" : "FAILED");
            if (brc != 0) snprintf(policy, sizeof(policy), "BENCH_N4_RATE_PARTITION_PTHREADS_PANEL_B_FAILED_NO_EQUAL_FALLBACK(panel_cols=%d)", panel_cols);
        }
    }
    double panel_fill_start_rel = -1.0, panel_fill_end_rel = -1.0;
    double panel_fill_total = 0.0;
    double panel_init_start_rel = -1.0, panel_init_end_rel = -1.0;
    double panel_copy_start_rel = -1.0, panel_copy_end_rel = -1.0;
    double panel_compute_start_rel = -1.0, panel_compute_end_rel = -1.0;
    double panel_cleanup_start_rel = -1.0, panel_cleanup_end_rel = -1.0;
    int rc_exec = 0;
    double prev_panel_end_rel = -1.0;

    for (int j0 = 0, panel_idx = 0; j0 < N; j0 += panel_cols, panel_idx++) {
        int jb = (j0 + panel_cols <= N) ? panel_cols : (N - j0);
        double t_pf0 = tnow();
        fill_B_panel_formula(N, j0, jb, dt, B_panel);
        double t_pf1 = tnow();
        panel_fill_total += nonneg_or_zero(t_pf1 - t_pf0);
        merge_rel_span(&panel_fill_start_rel, &panel_fill_end_rel, t_pf0 - t_mode0, t_pf1 - t_mode0);

        panel_worker_t pw[3];
        init_empty_panel_workers(pw, "UNSET", 0);
        double t_panel_total = 0.0;
        int rc_panel = run_fixed_split_buffers_panel(N, jb, N, A, B_panel, C,
                                                     N, j0,
                                                     use_cpu, use_gpu0, use_gpu1,
                                                     use_gpu0 ? 0 : -1, use_gpu1 ? 1 : -1,
                                                     cpu_blas, gpu_blas, tile, dt,
                                                     variant, bench_rates, pw, &t_panel_total, t_mode0, panel_idx);
        t_pthreads_total += t_panel_total;

        double p_init_s = -1.0, p_init_e = -1.0;
        double p_copy_s = -1.0, p_copy_e = -1.0;
        double p_comp_s = -1.0, p_comp_e = -1.0;
        double p_clean_s = -1.0, p_clean_e = -1.0;
        int panel_ok = (rc_panel == 0);
        for (int d = 0; d < 3; d++) {
            accumulate_panel_worker(&w[d], &pw[d], N);
            merge_rel_span(&p_init_s, &p_init_e, pw[d].t_init_start_rel, pw[d].t_init_end_rel);
            merge_rel_span(&p_copy_s, &p_copy_e, pw[d].t_copy_start_rel, pw[d].t_copy_end_rel);
            merge_rel_span(&p_comp_s, &p_comp_e, pw[d].t_compute_start_rel, pw[d].t_compute_end_rel);
            merge_rel_span(&p_clean_s, &p_clean_e, pw[d].t_cleanup_start_rel, pw[d].t_cleanup_end_rel);
            if (pw[d].enabled && pw[d].err != 0) panel_ok = 0;
        }
        merge_rel_span(&panel_init_start_rel, &panel_init_end_rel, p_init_s, p_init_e);
        merge_rel_span(&panel_copy_start_rel, &panel_copy_end_rel, p_copy_s, p_copy_e);
        merge_rel_span(&panel_compute_start_rel, &panel_compute_end_rel, p_comp_s, p_comp_e);
        merge_rel_span(&panel_cleanup_start_rel, &panel_cleanup_end_rel, p_clean_s, p_clean_e);

        double fill_s_rel = t_pf0 - t_mode0;
        double fill_e_rel = t_pf1 - t_mode0;
        double local_start_rel = -1.0, local_end_rel = -1.0;
        merge_rel_span(&local_start_rel, &local_end_rel, p_init_s, p_init_e);
        merge_rel_span(&local_start_rel, &local_end_rel, p_copy_s, p_copy_e);
        merge_rel_span(&local_start_rel, &local_end_rel, p_comp_s, p_comp_e);
        double pre_s = -1.0, pre_e = -1.0;
        if (local_start_rel > fill_e_rel) { pre_s = fill_e_rel; pre_e = local_start_rel; }
        double extra_s = -1.0, extra_e = -1.0;
        double main_end_rel = -1.0;
        if (p_init_e > main_end_rel) main_end_rel = p_init_e;
        if (p_copy_e > main_end_rel) main_end_rel = p_copy_e;
        if (p_comp_e > main_end_rel) main_end_rel = p_comp_e;
        if (p_clean_s > main_end_rel) { extra_s = main_end_rel; extra_e = p_clean_s; }

        double gap_start_rel = -1.0, gap_end_rel = -1.0, gap_before_panel_s = 0.0;
        if (prev_panel_end_rel >= 0.0 && fill_s_rel > prev_panel_end_rel) {
            gap_start_rel = prev_panel_end_rel;
            gap_end_rel = fill_s_rel;
            gap_before_panel_s = gap_end_rel - gap_start_rel;
        }
        char pf0_abs[64], pf1_abs[64], pb0_abs[64], pb1_abs[64], ps0_abs[64], ps1_abs[64], pp0_abs[64], pp1_abs[64], pi0_abs[64], pi1_abs[64], pc0_abs[64], pc1_abs[64], pg0_abs[64], pg1_abs[64], pe0_abs[64], pe1_abs[64], gh0_abs[64], gh1_abs[64], px0_abs[64], px1_abs[64], prevp_abs[64], gap0_abs[64], gap1_abs[64];
        make_abs_from_root(t_mode0, fill_s_rel, pf0_abs, sizeof(pf0_abs));
        make_abs_from_root(t_mode0, fill_e_rel, pf1_abs, sizeof(pf1_abs));
        make_abs_from_root(t_mode0, prev_panel_end_rel, prevp_abs, sizeof(prevp_abs));
        make_abs_from_root(t_mode0, gap_start_rel, gap0_abs, sizeof(gap0_abs));
        make_abs_from_root(t_mode0, gap_end_rel, gap1_abs, sizeof(gap1_abs));
        snprintf(pb0_abs, sizeof(pb0_abs), "NA");
        snprintf(pb1_abs, sizeof(pb1_abs), "NA");
        snprintf(ps0_abs, sizeof(ps0_abs), "NA");
        snprintf(ps1_abs, sizeof(ps1_abs), "NA");
        make_abs_from_root(t_mode0, pre_s, pp0_abs, sizeof(pp0_abs));
        make_abs_from_root(t_mode0, pre_e, pp1_abs, sizeof(pp1_abs));
        make_abs_from_root(t_mode0, p_init_s, pi0_abs, sizeof(pi0_abs));
        make_abs_from_root(t_mode0, p_init_e, pi1_abs, sizeof(pi1_abs));
        make_abs_from_root(t_mode0, p_copy_s, pc0_abs, sizeof(pc0_abs));
        make_abs_from_root(t_mode0, p_copy_e, pc1_abs, sizeof(pc1_abs));
        make_abs_from_root(t_mode0, p_comp_s, pg0_abs, sizeof(pg0_abs));
        make_abs_from_root(t_mode0, p_comp_e, pg1_abs, sizeof(pg1_abs));
        make_abs_from_root(t_mode0, extra_s, pe0_abs, sizeof(pe0_abs));
        make_abs_from_root(t_mode0, extra_e, pe1_abs, sizeof(pe1_abs));
        snprintf(gh0_abs, sizeof(gh0_abs), "NA");
        snprintf(gh1_abs, sizeof(gh1_abs), "NA");
        make_abs_from_root(t_mode0, p_clean_s, px0_abs, sizeof(px0_abs));
        make_abs_from_root(t_mode0, p_clean_e, px1_abs, sizeof(px1_abs));
        double panel_start_rel = fill_s_rel;
        double panel_end_rel = fill_e_rel;
        if (pre_s >= 0.0 && pre_s < panel_start_rel) panel_start_rel = pre_s;
        if (p_init_s >= 0.0 && p_init_s < panel_start_rel) panel_start_rel = p_init_s;
        if (p_copy_s >= 0.0 && p_copy_s < panel_start_rel) panel_start_rel = p_copy_s;
        if (p_comp_s >= 0.0 && p_comp_s < panel_start_rel) panel_start_rel = p_comp_s;
        if (extra_s >= 0.0 && extra_s < panel_start_rel) panel_start_rel = extra_s;
        if (p_clean_s >= 0.0 && p_clean_s < panel_start_rel) panel_start_rel = p_clean_s;
        if (pre_e > panel_end_rel) panel_end_rel = pre_e;
        if (p_init_e > panel_end_rel) panel_end_rel = p_init_e;
        if (p_copy_e > panel_end_rel) panel_end_rel = p_copy_e;
        if (p_comp_e > panel_end_rel) panel_end_rel = p_comp_e;
        if (extra_e > panel_end_rel) panel_end_rel = extra_e;
        if (p_clean_e > panel_end_rel) panel_end_rel = p_clean_e;
        printf("PTHREADS_PANEL_TIMING panel_idx=%d j0=%d jb=%d prev_panel_end_abs=%s prev_panel_end_rel=%.6f gap_start_abs=%s gap_end_abs=%s gap_start_rel=%.6f gap_end_rel=%.6f gap_before_panel_s=%.6f fill_start_abs=%s fill_end_abs=%s fill_start_rel=%.6f fill_end_rel=%.6f bcast_start_abs=%s bcast_end_abs=%s bcast_start_rel=%.6f bcast_end_rel=%.6f scatter_start_abs=%s scatter_end_abs=%s scatter_start_rel=%.6f scatter_end_rel=%.6f precompute_start_abs=%s precompute_end_abs=%s precompute_start_rel=%.6f precompute_end_rel=%.6f init_start_abs=%s init_end_abs=%s init_start_rel=%.6f init_end_rel=%.6f copy_start_abs=%s copy_end_abs=%s copy_start_rel=%.6f copy_end_rel=%.6f compute_start_abs=%s compute_end_abs=%s compute_start_rel=%.6f compute_end_rel=%.6f extra_calc_start_abs=%s extra_calc_end_abs=%s extra_calc_start_rel=%.6f extra_calc_end_rel=%.6f gather_start_abs=%s gather_end_abs=%s gather_start_rel=%.6f gather_end_rel=%.6f cleanup_start_abs=%s cleanup_end_abs=%s cleanup_start_rel=%.6f cleanup_end_rel=%.6f panel_total_s=%.6f\n",
               panel_idx, j0, jb,
               prevp_abs, timeline_rel_value(prev_panel_end_rel),
               gap0_abs, gap1_abs, timeline_rel_value(gap_start_rel), timeline_rel_value(gap_end_rel), sane(gap_before_panel_s),
               pf0_abs, pf1_abs, sane(fill_s_rel), sane(fill_e_rel),
               pb0_abs, pb1_abs, -1.0, -1.0,
               ps0_abs, ps1_abs, -1.0, -1.0,
               pp0_abs, pp1_abs, timeline_rel_value(pre_s), timeline_rel_value(pre_e),
               pi0_abs, pi1_abs, timeline_rel_value(p_init_s), timeline_rel_value(p_init_e),
               pc0_abs, pc1_abs, timeline_rel_value(p_copy_s), timeline_rel_value(p_copy_e),
               pg0_abs, pg1_abs, timeline_rel_value(p_comp_s), timeline_rel_value(p_comp_e),
               pe0_abs, pe1_abs, timeline_rel_value(extra_s), timeline_rel_value(extra_e),
               gh0_abs, gh1_abs, -1.0, -1.0,
               px0_abs, px1_abs, timeline_rel_value(p_clean_s), timeline_rel_value(p_clean_e),
               sane(panel_end_rel - panel_start_rel));
        print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", panel_idx, j0, jb, N, "panel_fill_B", fill_s_rel, fill_e_rel, t_mode0, (unsigned long long)N * (unsigned long long)jb * (unsigned long long)esz);
        print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", panel_idx, j0, jb, N, "precompute", pre_s, pre_e, t_mode0, 0ULL);
        print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", panel_idx, j0, jb, N, "extra_calc", extra_s, extra_e, t_mode0, 0ULL);
        prev_panel_end_rel = panel_end_rel;
        for (int d = 0; d < 3; d++) {
            print_panel_dev_timing_line("PTHREADS_PANEL_DEV_TIMING", 0, panel_idx, j0, jb, N, &pw[d]);
            print_panel_worker_timeline_events("PTHREADS_TIMELINE_EVENT", "pthreads", 0, panel_idx, j0, jb, &pw[d]);
        }
        fflush(stdout);

        if (!panel_ok) {
            rc_exec = (rc_panel != 0) ? rc_panel : -430;
            break;
        }
    }

    for (int d = 0; d < 3; d++) finalize_panel_worker_accum(&w[d]);
    print_worker_line(&w[0], N);
    print_worker_line(&w[1], N);
    print_worker_line(&w[2], N);
    print_part_percent_done(w, N);
    print_device_speeds(w);

    double c00 = read_c00(dt, C);
    double chk = compute_checksum(dt, C, (long long)N*(long long)N);
    int any_enabled = (w[0].enabled || w[1].enabled || w[2].enabled);
    int all_ok = 1;
    for (int i = 0; i < 3; i++) if (w[i].enabled && w[i].err != 0) all_ok = 0;
    const char *status = (!any_enabled) ? "FAILED" : ((rc_exec == 0 && all_ok) ? "OK" : "DEGRADED");

    double t_cleanup0 = tnow();
    free(A); free(C); free(B_panel);
    double t_cleanup1 = tnow();
    double t_full_wall_total = t_cleanup1 - t_mode0;
    double t_full_total = nonneg_or_zero(t_full_wall_total - n4_bench_wall_s);
    double total_flops = 2.0 * (double)N * (double)N * (double)N;
    double gflops_total = (t_full_total > 0.0) ? ((total_flops / 1e9) / t_full_total) : 0.0;

    char full0[64], full1[64], a0[64], a1[64], f0[64], f1[64], pb0[64], pb1[64], pi0[64], pi1[64], pc0[64], pc1[64], pg0[64], pg1[64], px0[64], px1[64], cl0[64], cl1[64];
    fmt_ts_iso(t_mode0, full0, sizeof(full0));
    fmt_ts_iso(t_cleanup1, full1, sizeof(full1));
    fmt_ts_iso(t_alloc0, a0, sizeof(a0));
    fmt_ts_iso(t_alloc1, a1, sizeof(a1));
    fmt_ts_iso(t_fill_ac0, f0, sizeof(f0));
    fmt_ts_iso(t_fill_ac1, f1, sizeof(f1));
    make_abs_from_root(t_mode0, panel_fill_start_rel, pb0, sizeof(pb0));
    make_abs_from_root(t_mode0, panel_fill_end_rel, pb1, sizeof(pb1));
    make_abs_from_root(t_mode0, panel_init_start_rel, pi0, sizeof(pi0));
    make_abs_from_root(t_mode0, panel_init_end_rel, pi1, sizeof(pi1));
    make_abs_from_root(t_mode0, panel_copy_start_rel, pc0, sizeof(pc0));
    make_abs_from_root(t_mode0, panel_copy_end_rel, pc1, sizeof(pc1));
    make_abs_from_root(t_mode0, panel_compute_start_rel, pg0, sizeof(pg0));
    make_abs_from_root(t_mode0, panel_compute_end_rel, pg1, sizeof(pg1));
    make_abs_from_root(t_mode0, panel_cleanup_start_rel, px0, sizeof(px0));
    make_abs_from_root(t_mode0, panel_cleanup_end_rel, px1, sizeof(px1));
    fmt_ts_iso(t_cleanup0, cl0, sizeof(cl0));
    fmt_ts_iso(t_cleanup1, cl1, sizeof(cl1));
    printf("PTHREADS_TIMELINE full_start_abs=%s full_end_abs=%s full_start_rel=0.000000 full_end_rel=%.6f alloc_start_abs=%s alloc_end_abs=%s alloc_start_rel=%.6f alloc_end_rel=%.6f fill_ac_start_abs=%s fill_ac_end_abs=%s fill_ac_start_rel=%.6f fill_ac_end_rel=%.6f panel_fill_start_abs=%s panel_fill_end_abs=%s panel_fill_start_rel=%.6f panel_fill_end_rel=%.6f init_start_abs=%s init_end_abs=%s init_start_rel=%.6f init_end_rel=%.6f copy_start_abs=%s copy_end_abs=%s copy_start_rel=%.6f copy_end_rel=%.6f compute_start_abs=%s compute_end_abs=%s compute_start_rel=%.6f compute_end_rel=%.6f dev_cleanup_start_abs=%s dev_cleanup_end_abs=%s dev_cleanup_start_rel=%.6f dev_cleanup_end_rel=%.6f host_cleanup_start_abs=%s host_cleanup_end_abs=%s host_cleanup_start_rel=%.6f host_cleanup_end_rel=%.6f\n",
           full0, full1, sane(t_full_wall_total),
           a0, a1, sane(t_alloc0 - t_mode0), sane(t_alloc1 - t_mode0),
           f0, f1, sane(t_fill_ac0 - t_mode0), sane(t_fill_ac1 - t_mode0),
           pb0, pb1, timeline_rel_value(panel_fill_start_rel), timeline_rel_value(panel_fill_end_rel),
           pi0, pi1, timeline_rel_value(panel_init_start_rel), timeline_rel_value(panel_init_end_rel),
           pc0, pc1, timeline_rel_value(panel_copy_start_rel), timeline_rel_value(panel_copy_end_rel),
           pg0, pg1, timeline_rel_value(panel_compute_start_rel), timeline_rel_value(panel_compute_end_rel),
           px0, px1, timeline_rel_value(panel_cleanup_start_rel), timeline_rel_value(panel_cleanup_end_rel),
           cl0, cl1, sane(t_cleanup0 - t_mode0), sane(t_cleanup1 - t_mode0));
    print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", -1, 0, N, N, "alloc", t_alloc0 - t_mode0, t_alloc1 - t_mode0, t_mode0, bytesA + bytesBpanel + bytesC);
    print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", -1, 0, N, N, "fill_A_C", t_fill_ac0 - t_mode0, t_fill_ac1 - t_mode0, t_mode0, bytesA + bytesC);
    print_timeline_event_line("PTHREADS_TIMELINE_EVENT", "pthreads", 0, "host", "host", -1, 0, N, N, "host_cleanup", t_cleanup0 - t_mode0, t_cleanup1 - t_mode0, t_mode0, bytesA + bytesBpanel + bytesC);
    double alloc_s = interval_or_zero(t_alloc0 - t_mode0, t_alloc1 - t_mode0);
    double fill_ac_s = interval_or_zero(t_fill_ac0 - t_mode0, t_fill_ac1 - t_mode0);
    double panel_fill_s = panel_fill_total;
    double host_cleanup_s = interval_or_zero(t_cleanup0 - t_mode0, t_cleanup1 - t_mode0);
    double init_sum_s = 0.0, copy_sum_s = 0.0, compute_sum_s = 0.0, dev_cleanup_sum_s = 0.0;
    double worker_sum_s = 0.0, worker_span_s = 0.0;
    worker_phase_sums(w, &init_sum_s, &copy_sum_s, &compute_sum_s, &dev_cleanup_sum_s, &worker_sum_s, &worker_span_s);
    double init_span_s = interval_or_zero(panel_init_start_rel, panel_init_end_rel);
    double copy_span_s = interval_or_zero(panel_copy_start_rel, panel_copy_end_rel);
    double compute_span_s = interval_or_zero(panel_compute_start_rel, panel_compute_end_rel);
    double dev_cleanup_span_s = interval_or_zero(panel_cleanup_start_rel, panel_cleanup_end_rel);
    double panel_fill_span_s = interval_or_zero(panel_fill_start_rel, panel_fill_end_rel);
    double other_wall_s = nonneg_or_zero(t_full_total - (alloc_s + fill_ac_s + panel_fill_s + worker_span_s + host_cleanup_s));
    printf("PTHREADS_TOTAL_TIMING start_abs=%s end_abs=%s start_rel=0.000000 end_rel=%.6f wall_total_s=%.6f n4_calibration_s=%.6f n4_excluded_from_N_benchmark=1 alloc_s=%.6f fill_ac_s=%.6f panel_fill_s=%.6f init_s=%.6f copy_s=%.6f compute_s=%.6f dev_cleanup_s=%.6f host_cleanup_s=%.6f worker_total_s=%.6f worker_sum_s=%.6f worker_span_s=%.6f total_s=%.6f other_s=%.6f init_span_s=%.6f copy_span_s=%.6f compute_span_s=%.6f dev_cleanup_span_s=%.6f panel_fill_span_s=%.6f timing_note=phase_s_are_sum_durations_spans_are_timeline_extents\n",
           full0, full1, sane(t_full_wall_total), sane(t_full_wall_total), sane(n4_bench_wall_s),
           sane(alloc_s), sane(fill_ac_s), sane(panel_fill_s),
           sane(init_sum_s), sane(copy_sum_s), sane(compute_sum_s), sane(dev_cleanup_sum_s),
           sane(host_cleanup_s), sane(t_pthreads_total), sane(worker_sum_s), sane(worker_span_s),
           sane(t_full_total), sane(other_wall_s),
           sane(init_span_s), sane(copy_span_s), sane(compute_span_s), sane(dev_cleanup_span_s), sane(panel_fill_span_s));
    printf("CALENDAR total=%.6f wall_total=%.6f n4_calibration=%.6f pthreads_total=%.6f policy=%s panel_cols=%d n4_excluded_from_N_benchmark=1\n",
           sane(t_full_total), sane(t_full_wall_total), sane(n4_bench_wall_s), sane(t_pthreads_total), policy, panel_cols);
    printf("TOTAL total_s=%.6f wall_total_s=%.6f n4_calibration_s=%.6f pthreads_total_s=%.6f TOTAL_GFLOPS=%.3f STATUS=%s C[0]=%.6f CHECKSUM_SAMPLE=%.6e n4_excluded_from_N_benchmark=1\n",
           sane(t_full_total), sane(t_full_wall_total), sane(n4_bench_wall_s), sane(t_pthreads_total), sane(gflops_total), status, sane(c00), sane(chk));
    print_run_summary("pthreads", variant, config_name, N, 1, dt, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile,
                      cache_fit_label_for_B(N, esz), status, t_full_total, gflops_total, c00, chk, bench_rows, bench_rates);
    return strcmp(status, "OK") == 0 ? 0 : 3;
}

