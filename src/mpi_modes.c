#include "hybrid_common.h"

/* Three-process MPI full-B/panel-B execution modes and mode dispatch policy. */

int run_mpi_three_process_fullB(int N, int use_cpu, int use_gpu0, int use_gpu1,
                                       int cpu_blas, int gpu_blas, int cpu_threads, int tile,
                                       dtype_t dt, const char *variant, const char *config_name,
                                       int rank, int size, MPI_Datatype mpi_dt,
                                       int counts_rows[3], int displs_rows[3],
                                       int global_r0[3], int global_r1[3], double cpu_share,
                                       double t_mode0_local) {
    size_t esz = dtype_size(dt);
    int local_rows = counts_rows[rank];
    unsigned long long elemsB_full = mat_elems_NN(N);
    unsigned long long elemsA_local = mat_elems_rows(local_rows, N);
    unsigned long long elemsC_local = mat_elems_rows(local_rows, N);

    double t_alloc0 = tnow();
    void *B = malloc_aligned64((size_t)elemsB_full * esz);
    void *A_local = malloc_aligned64((size_t)elemsA_local * esz);
    void *C_local = malloc_aligned64((size_t)elemsC_local * esz);
    void *A_full = NULL;
    void *C_full = NULL;
    if (rank == 0) {
        A_full = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
        C_full = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
    }
    double t_alloc1 = tnow();

    int alloc_fail = (!B || !A_local || !C_local || (rank == 0 && (!A_full || !C_full))) ? 1 : 0;
    int alloc_fail_any = 0;
    MPI_Allreduce(&alloc_fail, &alloc_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (alloc_fail_any) {
        if (rank == 0) printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=ALLOC_FAIL panel_b=0 scatter_gather=1\n", N, size, dtype_name(dt));
        free(B); free(A_local); free(C_local);
        if (rank == 0) { free(A_full); free(C_full); }
        return 2;
    }

    /* No-panel scatter/gather mode:
     *   rank 0 owns full B, A_full, C_full; all ranks own A_local/C_local.
     *   B is broadcast once, A_full rows are scattered, C_local rows are gathered.
     * This preserves tile=N/no-panel compute while forcing real MPI scatter/gather timings. */
    int fill_fail = 0;
    double t_fill0 = tnow();
    double t_fill1 = t_fill0;
    if (rank == 0) {
        fill_fail |= fill_B_only(N, dt, B);
        fill_fail |= fill_A_C_only(N, dt, A_full, C_full);
        t_fill1 = tnow();
    }

    int fill_fail_any = 0;
    MPI_Allreduce(&fill_fail, &fill_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (fill_fail_any != 0) {
        if (rank == 0) printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=FILL_FAIL panel_b=0 scatter_gather=1\n", N, size, dtype_name(dt));
        free(B); free(A_local); free(C_local);
        if (rank == 0) { free(A_full); free(C_full); }
        return 2;
    }

    if (rank == 0) {
        int ngpu_root = acc_get_num_devices(acc_device_nvidia);
        char ts_now[64];
        fmt_ts_iso(tnow(), ts_now, sizeof(ts_now));
        printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s cpu_impl=%s gpu_impl=%s use_cpu=%d use_gpu0=%d use_gpu1=%d ngpu_root=%d cpu_threads=%d tile=%d cache_fit_B=%s ts=%s variant=%s config=%s panel_b=0 scatter_gather=1\n",
               N, size, dtype_name(dt), cpu_blas ? "BLAS" : "NAIVE", gpu_blas ? "CUBLAS" : "OPENACC_NAIVE", use_cpu, use_gpu0, use_gpu1,
               ngpu_root, (cpu_threads > 0 ? cpu_threads : omp_get_max_threads()), tile,
               cache_fit_label_for_B(N, esz), ts_now, variant, config_name);
        printf("MPI_ROLE_MAP rank0=CPU rank1=GPU0 rank2=GPU1 gpu0_device=%d gpu1_device=%d\n",
               mpi_three_role_gpu_dev(1), mpi_three_role_gpu_dev(2));
        print_cache_fit_report(N, esz);
        print_cache_speed_estimates();
        printf("MEM_BYTES_ROOT A_full=%.0fMiB B=%.0fMiB C_full=%.0fMiB A_local=%.0fMiB C_local=%.0fMiB total=%.0fMiB no_A_full_C_full=0 scatter_gather=1\n",
               bytes_to_mib(mat_bytes_NN(N, esz)),
               bytes_to_mib(mat_bytes_NN(N, esz)),
               bytes_to_mib(mat_bytes_NN(N, esz)),
               bytes_to_mib(mat_bytes_rows(local_rows, N, esz)),
               bytes_to_mib(mat_bytes_rows(local_rows, N, esz)),
               bytes_to_mib(mat_bytes_NN(N, esz) * 3ULL + mat_bytes_rows(local_rows, N, esz) * 2ULL));
        printf("PARTITION_FINAL mode=mpi3proc variant=%s cpu_share=%.6f rows_cpu=%d rows_gpu0=%d rows_gpu1=%d\n",
               variant ? variant : "variant", sane(cpu_share), counts_rows[0], counts_rows[1], counts_rows[2]);
    }

    double root_abs = (rank == 0) ? t_mode0_local : 0.0;
    MPI_Bcast(&root_abs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double clock_offset_to_root = mpi_sync_clock_offset_to_root(rank);
    if (rank == 0) printf("MPI_CLOCK_SYNC method=barrier_bcast_root_offset applied=1 root_abs=%.6f\n", root_abs);
    unsigned long long alloc_bytes_local = ((unsigned long long)elemsB_full + (unsigned long long)elemsA_local + (unsigned long long)elemsC_local) * (unsigned long long)esz;
    if (rank == 0) alloc_bytes_local += 2ULL * mat_bytes_NN(N, esz);
    print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", rank, "host", dev_name_lower_from_wid(rank),
                              -1, 0, N, local_rows, "alloc",
                              mpi_rel_to_root(t_alloc0, root_abs, clock_offset_to_root),
                              mpi_rel_to_root(t_alloc1, root_abs, clock_offset_to_root),
                              root_abs, alloc_bytes_local);
    print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", rank, "host", dev_name_lower_from_wid(rank),
                              -1, 0, N, local_rows, rank == 0 ? "fill_B_A_full_C_full" : "fill_none_before_scatter",
                              mpi_rel_to_root(t_fill0, root_abs, clock_offset_to_root),
                              mpi_rel_to_root(t_fill1, root_abs, clock_offset_to_root),
                              root_abs,
                              (rank == 0) ? (3ULL * mat_bytes_NN(N, esz)) : 0ULL);
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_bc0 = tnow();
    int bcast_rc = mpi_bcast_large(B, elemsB_full, mpi_dt, 0, MPI_COMM_WORLD);
    double t_bc1 = tnow();
    if (bcast_rc != MPI_SUCCESS) {
        printf("RANK=%d MODE=MPI_3_PROCESS N=%d dtype=%s STATUS=FAILED reason=MPI_BCAST_FAIL\n", rank, N, dtype_name(dt));
        free(B); free(A_local); free(C_local);
        if (rank == 0) { free(A_full); free(C_full); }
        return 5;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_sc0 = tnow();
    int scatter_rc = mpi_scatter_rows_large(A_full, counts_rows, displs_rows, N, mpi_dt, A_local, rank, 0, MPI_COMM_WORLD);
    double t_sc1 = tnow();
    int scatter_fail = (scatter_rc != MPI_SUCCESS) ? 1 : 0;
    int scatter_fail_any = 0;
    MPI_Allreduce(&scatter_fail, &scatter_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (scatter_fail_any) {
        printf("RANK=%d MODE=MPI_3_PROCESS N=%d dtype=%s STATUS=FAILED reason=MPI_SCATTER_FAIL\n", rank, N, dtype_name(dt));
        free(B); free(A_local); free(C_local);
        if (rank == 0) { free(A_full); free(C_full); }
        return 5;
    }
    memset(C_local, 0, (size_t)elemsC_local * esz);

    MPI_Barrier(MPI_COMM_WORLD);
    worker_t wloc[3];
    mpi_three_run_one_full_worker(rank, N, local_rows, A_local, B, C_local,
                                  use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, tile, dt,
                                  root_abs, clock_offset_to_root, wloc);
    MPI_Barrier(MPI_COMM_WORLD);

    int role_enabled = (rank == 0) ? use_cpu : ((rank == 1) ? use_gpu0 : use_gpu1);
    int local_worker_fail = 0;
    if (role_enabled && local_rows > 0) {
        const worker_t *lw = &wloc[rank];
        if (lw->err != 0 || lw->rows_done != (long long)local_rows) local_worker_fail = 1;
    }
    int worker_fail_any = 0;
    MPI_Allreduce(&local_worker_fail, &worker_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_g0 = tnow();
    int gather_rc = mpi_gather_rows_large(C_local, local_rows, N, mpi_dt, C_full, counts_rows, displs_rows, rank, 0, MPI_COMM_WORLD);
    double t_g1 = tnow();
    int gather_fail = (gather_rc != MPI_SUCCESS) ? 1 : 0;
    int gather_fail_any = 0;
    MPI_Allreduce(&gather_fail, &gather_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (gather_fail_any) {
        printf("RANK=%d MODE=MPI_3_PROCESS N=%d dtype=%s STATUS=FAILED reason=MPI_GATHER_FAIL\n", rank, N, dtype_name(dt));
        free(B); free(A_local); free(C_local);
        if (rank == 0) { free(A_full); free(C_full); }
        return 5;
    }

    mpi_timeline_comm_t comm_local;
    memset(&comm_local, 0, sizeof(comm_local));
    comm_local.rank = rank;
    comm_local.panel_idx = -1;
    comm_local.j0 = 0;
    comm_local.jb = N;
    comm_local.rows = local_rows;
    comm_local.bcast_start_rel = mpi_rel_to_root(t_bc0, root_abs, clock_offset_to_root);
    comm_local.bcast_end_rel = mpi_rel_to_root(t_bc1, root_abs, clock_offset_to_root);
    comm_local.scatter_start_rel = mpi_rel_to_root(t_sc0, root_abs, clock_offset_to_root);
    comm_local.scatter_end_rel = mpi_rel_to_root(t_sc1, root_abs, clock_offset_to_root);
    comm_local.gather_start_rel = mpi_rel_to_root(t_g0, root_abs, clock_offset_to_root);
    comm_local.gather_end_rel = mpi_rel_to_root(t_g1, root_abs, clock_offset_to_root);
    comm_local.bcast_bytes = mat_bytes_NN(N, esz);
    comm_local.scatter_bytes = mat_bytes_rows(local_rows, N, esz);
    comm_local.gather_bytes = mat_bytes_rows(local_rows, N, esz);

    double local_bcast = t_bc1 - t_bc0;
    double local_scatter = t_sc1 - t_sc0;
    double local_compute = wloc[rank].total_time;
    double local_gather = t_g1 - t_g0;
    double local_full = mpi_rel_to_root(t_g1, root_abs, clock_offset_to_root);
    double max_bcast = 0.0, max_scatter = 0.0, max_compute = 0.0, max_gather = 0.0, max_full = 0.0;
    MPI_Reduce(&local_bcast, &max_bcast, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_scatter, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gather, &max_gather, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_full, &max_full, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    worker_t one = wloc[rank];
    worker_t gathered[3];
    MPI_Gather(&one, (int)sizeof(worker_t), MPI_BYTE,
               gathered, (int)sizeof(worker_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    mpi_timeline_comm_t comm_gathered[3];
    MPI_Gather(&comm_local, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE,
               comm_gathered, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    double max_mpi_init = 0.0, global_mpi_init_start_abs = -1.0, global_mpi_init_end_abs = -1.0;
    mpi_reduce_init_timing(&max_mpi_init, &global_mpi_init_start_abs, &global_mpi_init_end_abs);

    if (rank == 0) {
        double c00 = read_c00(dt, C_full);
        double chk = compute_checksum(dt, C_full, checksum_sample_elems((long long)N * (long long)N));
        double comm = max_bcast + max_scatter + max_gather;
        double total_with_init = max_full + max_mpi_init;
        g_mpi_init_start_abs = global_mpi_init_start_abs;
        g_mpi_init_end_abs = global_mpi_init_end_abs;
        print_mpi_comm_timeline_events(comm_gathered, root_abs);
        mpi_three_print_summary_root(N, size, dt, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile,
                                     variant, config_name, "MPI_3_PROCESS_FULL_B_SCATTER_GATHER", cache_fit_label_for_B(N, esz),
                                     total_with_init, max_full, max_bcast, max_scatter, max_compute, max_gather, comm,
                                     root_abs, max_mpi_init, c00, chk, gathered, global_r0, global_r1);
    }

    double t_free0 = tnow();
    free(B); free(A_local); free(C_local);
    if (rank == 0) { free(A_full); free(C_full); }
    double t_free1 = tnow();
    print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", rank, "host", dev_name_lower_from_wid(rank),
                              -1, 0, N, local_rows, "host_cleanup",
                              mpi_rel_to_root(t_free0, root_abs, clock_offset_to_root),
                              mpi_rel_to_root(t_free1, root_abs, clock_offset_to_root),
                              root_abs, alloc_bytes_local);
    fflush(stdout);
    return worker_fail_any ? 3 : 0;
}

int run_mpi_three_process_panelB(int N, int use_cpu, int use_gpu0, int use_gpu1,
                                        int cpu_blas, int gpu_blas, int cpu_threads, int tile,
                                        dtype_t dt, const char *variant, const char *config_name,
                                        int rank, int size, MPI_Datatype mpi_dt,
                                        int counts_rows[3], int displs_rows[3],
                                        int global_r0[3], int global_r1[3], double cpu_share,
                                        double t_mode0_local) {
    size_t esz = dtype_size(dt);
    int local_rows = counts_rows[rank];
    int panel_cols = choose_mpi_panel_cols(N, esz, tile, counts_rows[0]);
    if (panel_cols <= 0) panel_cols = 1;
    if (panel_cols > N) panel_cols = N;

    unsigned long long elemsA_local = mat_elems_rows(local_rows, N);
    unsigned long long elemsC_local = mat_elems_rows(local_rows, N);
    unsigned long long elemsB_panel = (unsigned long long)N * (unsigned long long)panel_cols;

    double t_alloc0 = tnow();
    void *A_local = malloc_aligned64((size_t)elemsA_local * esz);
    void *C_local = malloc_aligned64((size_t)elemsC_local * esz);
    void *B_panel = malloc_aligned64((size_t)elemsB_panel * esz);
    void *A_full = NULL;
    void *C_full = NULL;
    if (rank == 0) {
        A_full = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
        C_full = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
    }
    double t_alloc1 = tnow();

    int alloc_fail = (!A_local || !C_local || !B_panel || (rank == 0 && (!A_full || !C_full))) ? 1 : 0;
    int alloc_fail_any = 0;
    MPI_Allreduce(&alloc_fail, &alloc_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (alloc_fail_any) {
        if (rank == 0) printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=ALLOC_FAIL panel_b=1 panel_cols=%d\n", N, size, dtype_name(dt), panel_cols);
        free(A_local); free(C_local); free(B_panel); if (rank == 0) { free(A_full); free(C_full); }
        return 2;
    }

    int fill_fail = 0;
    double t_fill_ac0 = -1.0, t_fill_ac1 = -1.0;
    if (rank == 0) {
        t_fill_ac0 = tnow();
        fill_fail = fill_A_C_only(N, dt, A_full, C_full);
        t_fill_ac1 = tnow();
    }
    MPI_Bcast(&fill_fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (fill_fail != 0) {
        if (rank == 0) printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=FILL_FAIL panel_b=1\n", N, size, dtype_name(dt));
        free(A_local); free(C_local); free(B_panel); if (rank == 0) { free(A_full); free(C_full); }
        return 2;
    }

    if (rank == 0) {
        int ngpu_root = acc_get_num_devices(acc_device_nvidia);
        char ts_now[64];
        fmt_ts_iso(tnow(), ts_now, sizeof(ts_now));
        printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s cpu_impl=%s gpu_impl=%s use_cpu=%d use_gpu0=%d use_gpu1=%d ngpu_root=%d cpu_threads=%d tile=%d cache_fit_B=panel_B ts=%s variant=%s config=%s panel_b=1 panel_cols=%d\n",
               N, size, dtype_name(dt), cpu_blas ? "BLAS" : "NAIVE", gpu_blas ? "CUBLAS" : "OPENACC_NAIVE", use_cpu, use_gpu0, use_gpu1,
               ngpu_root, (cpu_threads > 0 ? cpu_threads : omp_get_max_threads()), tile,
               ts_now, variant, config_name, panel_cols);
        printf("MPI_ROLE_MAP rank0=CPU rank1=GPU0 rank2=GPU1 gpu0_device=%d gpu1_device=%d\n",
               mpi_three_role_gpu_dev(1), mpi_three_role_gpu_dev(2));
        print_cache_fit_report(N, esz);
        print_cache_speed_estimates();
        printf("MEM_BYTES_ROOT A=%.0fMiB B_panel=%.0fMiB C=%.0fMiB total_base=%.0fMiB\n",
               bytes_to_mib(mat_bytes_NN(N, esz)),
               bytes_to_mib((unsigned long long)N * (unsigned long long)panel_cols * (unsigned long long)esz),
               bytes_to_mib(mat_bytes_NN(N, esz)), bytes_to_mib(2ULL * mat_bytes_NN(N, esz)));
        printf("PARTITION_FINAL mode=mpi3proc variant=%s cpu_share=%.6f rows_cpu=%d rows_gpu0=%d rows_gpu1=%d\n",
               variant ? variant : "variant", sane(cpu_share), counts_rows[0], counts_rows[1], counts_rows[2]);
    }

    double root_abs = (rank == 0) ? t_mode0_local : 0.0;
    MPI_Bcast(&root_abs, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    double clock_offset_to_root = mpi_sync_clock_offset_to_root(rank);
    if (rank == 0) printf("MPI_CLOCK_SYNC method=barrier_bcast_root_offset applied=1 root_abs=%.6f\n", root_abs);
    unsigned long long alloc_bytes_local = ((unsigned long long)elemsA_local + (unsigned long long)elemsC_local + elemsB_panel) * (unsigned long long)esz;
    if (rank == 0) alloc_bytes_local += 2ULL * mat_bytes_NN(N, esz);
    print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", rank, "host", dev_name_lower_from_wid(rank),
                              -1, 0, N, local_rows, "alloc",
                              mpi_rel_to_root(t_alloc0, root_abs, clock_offset_to_root),
                              mpi_rel_to_root(t_alloc1, root_abs, clock_offset_to_root),
                              root_abs, alloc_bytes_local);
    if (rank == 0) {
        print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", 0, "host", "host",
                                  -1, 0, N, N, "fill_A_C",
                                  t_fill_ac0 - root_abs, t_fill_ac1 - root_abs,
                                  root_abs, 2ULL * mat_bytes_NN(N, esz));
    }
    fflush(stdout);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_sc0 = tnow();
    int scatter_rc = mpi_scatter_rows_large(A_full, counts_rows, displs_rows, N, mpi_dt, A_local, rank, 0, MPI_COMM_WORLD);
    double t_sc1 = tnow();
    if (scatter_rc != MPI_SUCCESS) {
        printf("RANK=%d MODE=MPI_3_PROCESS N=%d dtype=%s STATUS=FAILED reason=MPI_SCATTER_FAIL\n", rank, N, dtype_name(dt));
        free(A_local); free(C_local); free(B_panel); if (rank == 0) { free(A_full); free(C_full); }
        return 5;
    }
    memset(C_local, 0, (size_t)elemsC_local * esz);

    mpi_timeline_comm_t scatter_ev_local;
    memset(&scatter_ev_local, 0, sizeof(scatter_ev_local));
    scatter_ev_local.rank = rank;
    scatter_ev_local.panel_idx = -1;
    scatter_ev_local.j0 = 0;
    scatter_ev_local.jb = N;
    scatter_ev_local.rows = local_rows;
    scatter_ev_local.bcast_start_rel = scatter_ev_local.bcast_end_rel = -1.0;
    scatter_ev_local.scatter_start_rel = mpi_rel_to_root(t_sc0, root_abs, clock_offset_to_root);
    scatter_ev_local.scatter_end_rel = mpi_rel_to_root(t_sc1, root_abs, clock_offset_to_root);
    scatter_ev_local.gather_start_rel = scatter_ev_local.gather_end_rel = -1.0;
    scatter_ev_local.scatter_bytes = mat_bytes_rows(local_rows, N, esz);
    mpi_timeline_comm_t scatter_ev_gathered[3];
    MPI_Gather(&scatter_ev_local, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE,
               scatter_ev_gathered, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) print_mpi_comm_timeline_events(scatter_ev_gathered, root_abs);

    worker_t wacc[3];
    init_empty_workers(wacc, N, "SKIPPED", 0);
    int role_enabled = (rank == 0) ? use_cpu : ((rank == 1) ? use_gpu0 : use_gpu1);
    int device_ok = 1;
    if (rank != 0 && role_enabled && local_rows > 0) {
        char why[64];
        if (!mpi_three_gpu_available_for_role(rank, why, sizeof(why))) {
            device_ok = 0;
            worker_t *w = &wacc[rank];
            memset(w, 0, sizeof(*w));
            init_worker_timing_fields(w);
            w->wid = rank;
            w->enabled = 1;
            w->gpu_dev = mpi_three_role_gpu_dev(rank);
            w->N = N;
            w->r0 = 0;
            w->r1 = local_rows;
            w->dt = dt;
            w->tile = tile;
            w->t_ref = root_abs;
            w->err = -621;
            snprintf(w->status, sizeof(w->status), "%s", why);
            finalize_worker_perf(w);
        } else {
        }
    }

    double bcast_total = 0.0;
    double prev_panel_end_rel = -1.0;
    MPI_Barrier(MPI_COMM_WORLD);
    for (int j0 = 0, panel_idx = 0; j0 < N; j0 += panel_cols, panel_idx++) {
        int jb = (j0 + panel_cols <= N) ? panel_cols : (N - j0);
        double t_pf0 = -1.0, t_pf1 = -1.0;
        if (rank == 0) {
            t_pf0 = tnow();
            fill_B_panel_formula(N, j0, jb, dt, B_panel);
            t_pf1 = tnow();
        }
        MPI_Barrier(MPI_COMM_WORLD);
        double t_bc0 = tnow();
        int bcast_rc = mpi_bcast_large(B_panel, (unsigned long long)N * (unsigned long long)jb, mpi_dt, 0, MPI_COMM_WORLD);
        double t_bc1 = tnow();
        bcast_total += (t_bc1 - t_bc0);
        if (bcast_rc != MPI_SUCCESS) {
            printf("RANK=%d MODE=MPI_3_PROCESS N=%d dtype=%s STATUS=FAILED reason=MPI_BCAST_PANEL_FAIL panel_idx=%d\n", rank, N, dtype_name(dt), panel_idx);
            free(A_local); free(C_local); free(B_panel); if (rank == 0) { free(A_full); free(C_full); }
            return 5;
        }

        panel_worker_t pw_event;
        memset(&pw_event, 0, sizeof(pw_event));
        init_panel_worker_timing_fields(&pw_event);
        pw_event.wid = rank;
        pw_event.enabled = (role_enabled && device_ok && local_rows > 0) ? 1 : 0;
        pw_event.gpu_dev = (rank == 0) ? -1 : mpi_three_role_gpu_dev(rank);
        pw_event.K = N;
        pw_event.cols = jb;
        pw_event.r0 = 0;
        pw_event.r1 = local_rows;
        pw_event.cpu_blas = cpu_blas;
        pw_event.gpu_blas = (rank == 0) ? 0 : gpu_blas;
        pw_event.tile = tile;
        pw_event.dt = dt;
        pw_event.A = A_local;
        pw_event.B = B_panel;
        pw_event.C = C_local;
        pw_event.c_ld = N;
        pw_event.c_col0 = j0;
        pw_event.t_ref = root_abs - clock_offset_to_root;
        set_panel_worker_trace(&pw_event, "MPI_TIMELINE_TILE_EVENT", "mpi3proc", rank, panel_idx, j0, jb, root_abs, clock_offset_to_root);
        snprintf(pw_event.status, sizeof(pw_event.status), "%s", pw_event.enabled ? "UNRUN" : "SKIPPED");
        if (pw_event.enabled) {
            panel_worker_main(&pw_event);
            shift_panel_worker_abs_to_root_clock(&pw_event, clock_offset_to_root, root_abs);
            accumulate_panel_worker(&wacc[rank], &pw_event, N);
        }
        mpi_timeline_comm_t panel_comm_local;
        memset(&panel_comm_local, 0, sizeof(panel_comm_local));
        panel_comm_local.rank = rank;
        panel_comm_local.panel_idx = panel_idx;
        panel_comm_local.j0 = j0;
        panel_comm_local.jb = jb;
        panel_comm_local.rows = local_rows;
        panel_comm_local.bcast_start_rel = mpi_rel_to_root(t_bc0, root_abs, clock_offset_to_root);
        panel_comm_local.bcast_end_rel = mpi_rel_to_root(t_bc1, root_abs, clock_offset_to_root);
        panel_comm_local.scatter_start_rel = panel_comm_local.scatter_end_rel = -1.0;
        panel_comm_local.gather_start_rel = panel_comm_local.gather_end_rel = -1.0;
        panel_comm_local.bcast_bytes = (unsigned long long)N * (unsigned long long)jb * (unsigned long long)esz;
        mpi_timeline_comm_t panel_comm_gathered[3];
        panel_worker_t panel_pw_gathered[3];
        MPI_Gather(&panel_comm_local, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE,
                   panel_comm_gathered, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        MPI_Gather(&pw_event, (int)sizeof(panel_worker_t), MPI_BYTE,
                   panel_pw_gathered, (int)sizeof(panel_worker_t), MPI_BYTE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            double fill_s_rel = t_pf0 - root_abs;
            double fill_e_rel = t_pf1 - root_abs;
            print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", 0, "host", "host",
                                      panel_idx, j0, jb, N, "panel_fill_B",
                                      fill_s_rel, fill_e_rel, root_abs,
                                      (unsigned long long)N * (unsigned long long)jb * (unsigned long long)esz);
            print_mpi_comm_timeline_events(panel_comm_gathered, root_abs);

            double bcast_s = -1.0, bcast_e = -1.0;
            double p_init_s = -1.0, p_init_e = -1.0;
            double p_copy_s = -1.0, p_copy_e = -1.0;
            double p_comp_s = -1.0, p_comp_e = -1.0;
            double p_clean_s = -1.0, p_clean_e = -1.0;
            for (int rr = 0; rr < 3; rr++) {
                merge_rel_span(&bcast_s, &bcast_e, panel_comm_gathered[rr].bcast_start_rel, panel_comm_gathered[rr].bcast_end_rel);
                merge_rel_span(&p_init_s, &p_init_e, panel_pw_gathered[rr].t_init_start_rel, panel_pw_gathered[rr].t_init_end_rel);
                merge_rel_span(&p_copy_s, &p_copy_e, panel_pw_gathered[rr].t_copy_start_rel, panel_pw_gathered[rr].t_copy_end_rel);
                merge_rel_span(&p_comp_s, &p_comp_e, panel_pw_gathered[rr].t_compute_start_rel, panel_pw_gathered[rr].t_compute_end_rel);
                merge_rel_span(&p_clean_s, &p_clean_e, panel_pw_gathered[rr].t_cleanup_start_rel, panel_pw_gathered[rr].t_cleanup_end_rel);
            }

            double local_start_rel = -1.0, local_end_rel = -1.0;
            merge_rel_span(&local_start_rel, &local_end_rel, fill_s_rel, fill_e_rel);
            merge_rel_span(&local_start_rel, &local_end_rel, bcast_s, bcast_e);
            merge_rel_span(&local_start_rel, &local_end_rel, p_init_s, p_init_e);
            merge_rel_span(&local_start_rel, &local_end_rel, p_copy_s, p_copy_e);
            merge_rel_span(&local_start_rel, &local_end_rel, p_comp_s, p_comp_e);
            merge_rel_span(&local_start_rel, &local_end_rel, p_clean_s, p_clean_e);

            double gap_start_rel = -1.0, gap_end_rel = -1.0, gap_before_panel_s = 0.0;
            if (prev_panel_end_rel >= 0.0 && local_start_rel > prev_panel_end_rel) {
                gap_start_rel = prev_panel_end_rel;
                gap_end_rel = local_start_rel;
                gap_before_panel_s = gap_end_rel - gap_start_rel;
            }

            char prevp_abs[64], gap0_abs[64], gap1_abs[64], pf0_abs[64], pf1_abs[64];
            char bc0_abs[64], bc1_abs[64], sc0_abs[64], sc1_abs[64], pi0_abs[64], pi1_abs[64];
            char pc0_abs[64], pc1_abs[64], pg0_abs[64], pg1_abs[64], gh0_abs[64], gh1_abs[64], px0_abs[64], px1_abs[64];
            make_abs_from_root(root_abs, prev_panel_end_rel, prevp_abs, sizeof(prevp_abs));
            make_abs_from_root(root_abs, gap_start_rel, gap0_abs, sizeof(gap0_abs));
            make_abs_from_root(root_abs, gap_end_rel, gap1_abs, sizeof(gap1_abs));
            make_abs_from_root(root_abs, fill_s_rel, pf0_abs, sizeof(pf0_abs));
            make_abs_from_root(root_abs, fill_e_rel, pf1_abs, sizeof(pf1_abs));
            make_abs_from_root(root_abs, bcast_s, bc0_abs, sizeof(bc0_abs));
            make_abs_from_root(root_abs, bcast_e, bc1_abs, sizeof(bc1_abs));
            snprintf(sc0_abs, sizeof(sc0_abs), "NA"); snprintf(sc1_abs, sizeof(sc1_abs), "NA");
            make_abs_from_root(root_abs, p_init_s, pi0_abs, sizeof(pi0_abs));
            make_abs_from_root(root_abs, p_init_e, pi1_abs, sizeof(pi1_abs));
            make_abs_from_root(root_abs, p_copy_s, pc0_abs, sizeof(pc0_abs));
            make_abs_from_root(root_abs, p_copy_e, pc1_abs, sizeof(pc1_abs));
            make_abs_from_root(root_abs, p_comp_s, pg0_abs, sizeof(pg0_abs));
            make_abs_from_root(root_abs, p_comp_e, pg1_abs, sizeof(pg1_abs));
            snprintf(gh0_abs, sizeof(gh0_abs), "NA"); snprintf(gh1_abs, sizeof(gh1_abs), "NA");
            make_abs_from_root(root_abs, p_clean_s, px0_abs, sizeof(px0_abs));
            make_abs_from_root(root_abs, p_clean_e, px1_abs, sizeof(px1_abs));

            printf("MPI_PANEL_TIMING panel_idx=%d j0=%d jb=%d prev_panel_end_abs=%s prev_panel_end_rel=%.6f gap_start_abs=%s gap_end_abs=%s gap_start_rel=%.6f gap_end_rel=%.6f gap_before_panel_s=%.6f fill_start_abs=%s fill_end_abs=%s fill_start_rel=%.6f fill_end_rel=%.6f bcast_start_abs=%s bcast_end_abs=%s bcast_start_rel=%.6f bcast_end_rel=%.6f scatter_start_abs=%s scatter_end_abs=%s scatter_start_rel=%.6f scatter_end_rel=%.6f init_start_abs=%s init_end_abs=%s init_start_rel=%.6f init_end_rel=%.6f copy_start_abs=%s copy_end_abs=%s copy_start_rel=%.6f copy_end_rel=%.6f compute_start_abs=%s compute_end_abs=%s compute_start_rel=%.6f compute_end_rel=%.6f gather_start_abs=%s gather_end_abs=%s gather_start_rel=%.6f gather_end_rel=%.6f cleanup_start_abs=%s cleanup_end_abs=%s cleanup_start_rel=%.6f cleanup_end_rel=%.6f panel_total_s=%.6f\n",
                   panel_idx, j0, jb,
                   prevp_abs, timeline_rel_value(prev_panel_end_rel),
                   gap0_abs, gap1_abs, timeline_rel_value(gap_start_rel), timeline_rel_value(gap_end_rel), sane(gap_before_panel_s),
                   pf0_abs, pf1_abs, timeline_rel_value(fill_s_rel), timeline_rel_value(fill_e_rel),
                   bc0_abs, bc1_abs, timeline_rel_value(bcast_s), timeline_rel_value(bcast_e),
                   sc0_abs, sc1_abs, -1.0, -1.0,
                   pi0_abs, pi1_abs, timeline_rel_value(p_init_s), timeline_rel_value(p_init_e),
                   pc0_abs, pc1_abs, timeline_rel_value(p_copy_s), timeline_rel_value(p_copy_e),
                   pg0_abs, pg1_abs, timeline_rel_value(p_comp_s), timeline_rel_value(p_comp_e),
                   gh0_abs, gh1_abs, -1.0, -1.0,
                   px0_abs, px1_abs, timeline_rel_value(p_clean_s), timeline_rel_value(p_clean_e),
                   interval_or_zero(local_start_rel, local_end_rel));
            prev_panel_end_rel = local_end_rel;

            for (int rr = 0; rr < 3; rr++) {
                print_panel_dev_timing_line("MPI_PANEL_DEV_TIMING", rr, panel_idx, j0, jb, N, &panel_pw_gathered[rr]);
                print_panel_worker_timeline_events("MPI_TIMELINE_EVENT", "mpi3proc", rr, panel_idx, j0, jb, &panel_pw_gathered[rr]);
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (role_enabled && device_ok && local_rows > 0) finalize_panel_worker_accum(&wacc[rank]);
    MPI_Barrier(MPI_COMM_WORLD);

    int local_worker_fail = 0;
    if (role_enabled && local_rows > 0) {
        const worker_t *lw = &wacc[rank];
        if (!device_ok || lw->err != 0 || lw->rows_done != (long long)local_rows) local_worker_fail = 1;
    }
    int worker_fail_any = 0;
    MPI_Allreduce(&local_worker_fail, &worker_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_g0 = tnow();
    int gather_rc = mpi_gather_rows_large(C_local, local_rows, N, mpi_dt, C_full, counts_rows, displs_rows, rank, 0, MPI_COMM_WORLD);
    double t_g1 = tnow();
    if (gather_rc != MPI_SUCCESS) {
        printf("RANK=%d MODE=MPI_3_PROCESS N=%d dtype=%s STATUS=FAILED reason=MPI_GATHER_FAIL\n", rank, N, dtype_name(dt));
        free(A_local); free(C_local); free(B_panel); if (rank == 0) { free(A_full); free(C_full); }
        return 5;
    }

    mpi_timeline_comm_t gather_ev_local;
    memset(&gather_ev_local, 0, sizeof(gather_ev_local));
    gather_ev_local.rank = rank;
    gather_ev_local.panel_idx = -1;
    gather_ev_local.j0 = 0;
    gather_ev_local.jb = N;
    gather_ev_local.rows = local_rows;
    gather_ev_local.bcast_start_rel = gather_ev_local.bcast_end_rel = -1.0;
    gather_ev_local.scatter_start_rel = gather_ev_local.scatter_end_rel = -1.0;
    gather_ev_local.gather_start_rel = mpi_rel_to_root(t_g0, root_abs, clock_offset_to_root);
    gather_ev_local.gather_end_rel = mpi_rel_to_root(t_g1, root_abs, clock_offset_to_root);
    gather_ev_local.gather_bytes = mat_bytes_rows(local_rows, N, esz);
    mpi_timeline_comm_t gather_ev_gathered[3];
    MPI_Gather(&gather_ev_local, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE,
               gather_ev_gathered, (int)sizeof(mpi_timeline_comm_t), MPI_BYTE, 0, MPI_COMM_WORLD);
    if (rank == 0) print_mpi_comm_timeline_events(gather_ev_gathered, root_abs);

    double local_scatter = t_sc1 - t_sc0;
    double local_compute = wacc[rank].total_time;
    double local_gather = t_g1 - t_g0;
    double local_full = mpi_rel_to_root(t_g1, root_abs, clock_offset_to_root);
    double max_bcast = 0.0, max_scatter = 0.0, max_compute = 0.0, max_gather = 0.0, max_full = 0.0;
    MPI_Reduce(&bcast_total, &max_bcast, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_scatter, &max_scatter, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_compute, &max_compute, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_gather, &max_gather, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&local_full, &max_full, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    worker_t one = wacc[rank];
    worker_t gathered[3];
    MPI_Gather(&one, (int)sizeof(worker_t), MPI_BYTE,
               gathered, (int)sizeof(worker_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    double max_mpi_init = 0.0, global_mpi_init_start_abs = -1.0, global_mpi_init_end_abs = -1.0;
    mpi_reduce_init_timing(&max_mpi_init, &global_mpi_init_start_abs, &global_mpi_init_end_abs);

    if (rank == 0) {
        double c00 = read_c00(dt, C_full);
        double chk = compute_checksum(dt, C_full, (long long)N * (long long)N);
        double comm = max_bcast + max_scatter + max_gather;
        double total_with_init = max_full + max_mpi_init;
        g_mpi_init_start_abs = global_mpi_init_start_abs;
        g_mpi_init_end_abs = global_mpi_init_end_abs;
        char policy[128];
        snprintf(policy, sizeof(policy), "MPI_3_PROCESS_PANEL_B(panel_cols=%d)", panel_cols);
        mpi_three_print_summary_root(N, size, dt, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile,
                                     variant, config_name, policy, "panel_B",
                                     total_with_init, max_full, max_bcast, max_scatter, max_compute, max_gather, comm,
                                     root_abs, max_mpi_init, c00, chk, gathered, global_r0, global_r1);
    }

    double t_free0 = tnow();
    free(A_local); free(C_local); free(B_panel); if (rank == 0) { free(A_full); free(C_full); }
    double t_free1 = tnow();
    print_timeline_event_line("MPI_TIMELINE_EVENT", "mpi3proc", rank, "host", dev_name_lower_from_wid(rank),
                              -1, 0, N, local_rows, "host_cleanup",
                              mpi_rel_to_root(t_free0, root_abs, clock_offset_to_root),
                              mpi_rel_to_root(t_free1, root_abs, clock_offset_to_root),
                              root_abs, alloc_bytes_local);
    fflush(stdout);
    return worker_fail_any ? 3 : 0;
}

int run_mpi_three_process_mode(int N, int use_cpu, int use_gpu0, int use_gpu1,
                                      int cpu_blas, int gpu_blas, int cpu_threads, int tile,
                                      dtype_t dt, const char *variant, const char *config_name) {
    double t_mode0_local = tnow();
    configure_cpu_threads(cpu_threads);
    if (is_naive_variant(variant)) use_cpu = 0;

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 3) {
        if (rank == 0) {
            printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=REQUIRES_EXACTLY_3_MPI_PROCESSES expected_launch='mpirun -np 3 ./program 1 ...'\n",
                   N, size, dtype_name(dt));
        }
        return 2;
    }

    MPI_Datatype mpi_dt = MPI_BYTE;
    if (mpi_dtype_setup(dt, &mpi_dt) != 0) {
        if (rank == 0) printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=MPI_DTYPE_FAIL\n", N, size, dtype_name(dt));
        return 2;
    }

    int direct_panel_tile = 0;
    int direct_panel_threshold = mpi_direct_panel_policy(N, dt, variant, &direct_panel_tile);
    int direct_panel_requested = (direct_panel_threshold > 0 && direct_panel_tile > 0);
    if (direct_panel_requested) {
        tile = direct_panel_tile;
        if (rank == 0) {
            printf("MPI_DIRECT_PANEL_POLICY variant=%s dtype=%s N=%d threshold=%d action=panel_B tile=%d panel_cols=%d calibration=mpi3proc_panel_rates partition=standard_rate_based equal_partition=disabled\n",
                   variant ? variant : "variant", dtype_name(dt), N, direct_panel_threshold, direct_panel_tile, direct_panel_tile);
            fflush(stdout);
        }
    }

    int bench_rows[3] = {0,0,0};
    double bench_rates[3] = {0.0,0.0,0.0};
    int bench_used_pthreads_partition = 0;
    int bench_rc = -1;
    if (direct_panel_requested) {
        /*
         * Direct panel-B still uses standard rate-based partitioning.
         * Calibrate with the same panel width that the run will use, avoiding
         * full-B replication while keeping the measured-rate row split.
         */
        int bench_cols = (direct_panel_tile < N) ? direct_panel_tile : N;
        double n4_bench_t0 = tnow();
        bench_rc = run_n4_mpi_benchmark_panel(N, bench_cols, use_cpu, use_gpu0, use_gpu1,
                                              cpu_blas, gpu_blas, direct_panel_tile, dt, variant,
                                              rank, size, mpi_dt,
                                              bench_rows, bench_rates);
        double n4_bench_t1 = tnow();
        if (rank == 0) {
            printf("MPI_N4_CALIBRATION mode=mpi3proc source=mpi3proc_panel_rates N=%d cols=%d bench_rows_target=%d wall_s=%.6f excluded_from_N_benchmark=1 status=%s partition=standard_rate_based equal_partition=disabled\n",
                   N, bench_cols, n4_rows_for_bench(N), sane(nonneg_or_zero(n4_bench_t1 - n4_bench_t0)),
                   (bench_rc == 0) ? "OK" : "FAILED");
        }
    } else {
        double n4_bench_t0 = tnow();
        bench_rc = run_n4_pthreads_partition_benchmark_root(N, use_cpu, use_gpu0, use_gpu1,
                                                            cpu_blas, gpu_blas, tile, dt, variant,
                                                            rank, size,
                                                            bench_rows, bench_rates);
        double n4_bench_t1 = tnow();
        if (bench_rc == 0) {
            bench_used_pthreads_partition = 1;
            if (rank == 0) {
                printf("MPI_N4_CALIBRATION mode=mpi3proc source=pthreads_partition N=%d bench_rows_target=%d wall_s=%.6f excluded_from_N_benchmark=1 status=OK\n",
                       N, n4_rows_for_bench(N), sane(nonneg_or_zero(n4_bench_t1 - n4_bench_t0)));
            }
        } else {
            if (rank == 0) {
                printf("MPI_N4_CALIBRATION mode=mpi3proc source=pthreads_partition N=%d bench_rows_target=%d wall_s=%.6f excluded_from_N_benchmark=1 status=FAILED fallback=mpi3proc_rates\n",
                       N, n4_rows_for_bench(N), sane(nonneg_or_zero(n4_bench_t1 - n4_bench_t0)));
            }
            n4_bench_t0 = tnow();
            bench_rc = run_n4_mpi_benchmark_full(N, use_cpu, use_gpu0, use_gpu1,
                                                 cpu_blas, gpu_blas, tile, dt, variant,
                                                 rank, size, mpi_dt,
                                                 bench_rows, bench_rates);
            n4_bench_t1 = tnow();
            if (rank == 0) {
                printf("MPI_N4_CALIBRATION mode=mpi3proc source=mpi3proc_rates N=%d bench_rows_target=%d wall_s=%.6f excluded_from_N_benchmark=1 status=%s\n",
                       N, n4_rows_for_bench(N), sane(nonneg_or_zero(n4_bench_t1 - n4_bench_t0)), (bench_rc == 0) ? "OK" : "FAILED");
            }
        }
    }

    int counts_rows[3] = {0,0,0};
    int displs_rows[3] = {0,0,0};
    int global_r0[3] = {0,0,0};
    int global_r1[3] = {0,0,0};
    double cpu_share = 0.0;
    int used_shared_blas_partition = 0;
    int part_rc = mpi_three_choose_counts(N, use_cpu, use_gpu0, use_gpu1, variant, bench_rates,
                                          counts_rows, displs_rows, global_r0, global_r1, &cpu_share,
                                          &used_shared_blas_partition);
    (void)used_shared_blas_partition;
    const char *partition_policy =
        (bench_rc == 0 && part_rc == 0)
            ? (direct_panel_requested ? "BENCH_N4_RATE_PARTITION_MPI_PANEL"
                                      : (bench_used_pthreads_partition ? "BENCH_N4_RATE_PARTITION_MPI_MATCH_PTHREADS" : "BENCH_N4_RATE_PARTITION_MPI"))
            : "BENCH_N4_RATE_PARTITION_MPI_FAILED_NO_EQUAL_FALLBACK";
    double sum_rates = bench_rates[0] + bench_rates[1] + bench_rates[2];
    if (rank == 0) {
        printf("PARTITION_RATE mode=mpi3proc variant=%s policy=%s cpu_share=%.6f rows_cpu=%d rows_gpu0=%d rows_gpu1=%d rate_cpu=%.6f rate_gpu0=%.6f rate_gpu1=%.6f target_rate_time_s=%.6f\n",
               variant ? variant : "variant", partition_policy, sane(cpu_share),
               counts_rows[0], counts_rows[1], counts_rows[2],
               sane(bench_rates[0]), sane(bench_rates[1]), sane(bench_rates[2]),
               (sum_rates > 0.0) ? ((double)N / sum_rates) : 0.0);
    }
    t_mode0_local = tnow();
    if (validate_partition(N, global_r0[0], global_r1[0], global_r0[1], global_r1[1], global_r0[2], global_r1[2]) != 0) {
        if (rank == 0) printf("MODE=MPI_3_PROCESS N=%d ranks=%d dtype=%s STATUS=FAILED reason=INVALID_ROLE_PARTITION\n", N, size, dtype_name(dt));
        return 2;
    }

    if (rank == 0) {
        printf("MPI_RANK_ROLE rank=0 role=CPU compute=CPU_ONLY\n");
        printf("MPI_RANK_ROLE rank=1 role=GPU0 compute=%s\n", gpu_blas ? "CUBLAS" : "OPENACC_ONLY");
        printf("MPI_RANK_ROLE rank=2 role=GPU1 compute=%s\n", gpu_blas ? "CUBLAS" : "OPENACC_ONLY");
    }
    printf("RANK=%d ROLE=%s rows=%d r0=%d r1=%d\n", rank, mpi_three_role_name(rank), counts_rows[rank], global_r0[rank], global_r1[rank]);
    fflush(stdout);

    /*
     * MPI policy:
     *   1) For large BLAS-GEMM MPI runs, force panel-B immediately:
     *        DGEMM: N >= 34000 -> tile/panel_cols = 5000
     *        SGEMM: N >= 42000 -> tile/panel_cols = 10000
     *      This skips the full-B attempt, but still runs panel-B N/4
     *      calibration and uses standard rate-based partitioning.
     *   2) Otherwise, try the original full-B/no-panel scatter-gather run first.
     *   3) Only if that run returns a failure code, retry with panel-B tiling.
     *   4) The automatic fallback tile is fixed at 5000, independent of the
     *      command-line tile used for the first full-B attempt.
     *
     * A config containing "panel" still forces panel-B immediately. A config
     * containing "strict_nopanel" or "no_fallback" disables only the retry after
     * a failed full-B attempt; the large-GEMM direct panel policy above wins.
     * Note: if the OS OOM-kills an MPI rank, the process cannot catch that and
     * cannot enter the fallback path. Clean malloc/MPI/worker failures can retry.
     */
    const int fallback_tile = MPI_DEFAULT_FALLBACK_PANEL_TILE;
    int force_panel = config_force_panel(config_name);
    int disable_fallback = (config_name &&
                            (strstr(config_name, "strict_nopanel") != NULL ||
                             strstr(config_name, "no_fallback") != NULL));

    if (direct_panel_requested) {
        if (rank == 0) {
            printf("MPI_DIRECT_PANEL_TRIGGERED variant=%s dtype=%s N=%d threshold=%d action=panel_B tile=%d panel_cols=%d skip_full_B=1 partition=standard_rate_based equal_partition=disabled\n",
                   variant ? variant : "variant", dtype_name(dt), N, direct_panel_threshold, direct_panel_tile, direct_panel_tile);
            fflush(stdout);
        }
        return run_mpi_three_process_panelB(N, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, direct_panel_tile,
                                           dt, variant, config_name, rank, size, mpi_dt,
                                           counts_rows, displs_rows, global_r0, global_r1, cpu_share, t_mode0_local);
    }

    if (force_panel) {
        return run_mpi_three_process_panelB(N, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile,
                                           dt, variant, config_name, rank, size, mpi_dt,
                                           counts_rows, displs_rows, global_r0, global_r1, cpu_share, t_mode0_local);
    }

    int rc_full = run_mpi_three_process_fullB(N, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, tile,
                                             dt, variant, config_name, rank, size, mpi_dt,
                                             counts_rows, displs_rows, global_r0, global_r1, cpu_share, t_mode0_local);
    if (rc_full == 0 || disable_fallback) return rc_full;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("MPI_FALLBACK_TRIGGERED from=full_B rc=%d action=panel_B fallback_tile=%d fallback_panel_cols=%d\n",
               rc_full, fallback_tile, fallback_tile);
        fflush(stdout);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    t_mode0_local = tnow();
    return run_mpi_three_process_panelB(N, use_cpu, use_gpu0, use_gpu1, cpu_blas, gpu_blas, cpu_threads, fallback_tile,
                                       dt, variant, config_name, rank, size, mpi_dt,
                                       counts_rows, displs_rows, global_r0, global_r1, cpu_share, t_mode0_local);
}


int should_use_mpi_panel_b(int N, int size, size_t esz, unsigned long long elemsB_full) {
    (void)size;
    /*
     * Use panel-B mode for genuinely large matrices regardless of rank count.
     * For size==1 this is still useful because it reduces peak B/C working-set size
     * and emits per-panel timing records for the plotting/CSV pipeline.
     */
    if (too_large_for_mpi_count_elems(elemsB_full)) return 1;
    if (mat_bytes_NN(N, esz) > (unsigned long long)(2ULL * 1024ULL * 1024ULL * 1024ULL)) return 1;
    return 0;
}


