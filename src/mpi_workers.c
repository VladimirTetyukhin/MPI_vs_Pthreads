#include "hybrid_common.h"

/* MPI fixed-split worker execution, role helpers, and MPI benchmark summaries. */

int run_fixed_split_buffers(int N, int M, void *A, void *B, void *C,
                                   int use_cpu, int use_gpu0, int use_gpu1,
                                   int gpu0_dev, int gpu1_dev,
                                   int cpu_blas, int gpu_blas, int tile, dtype_t dt,
                                   const char *variant,
                                   worker_t out_w[3], double *t_total_out,
                                   char *policy_status, size_t policy_status_sz,
                                   int bench_rows_out[3], double bench_rate_out[3],
                                   double *n4_bench_wall_s_out,
                                   double t_ref_base) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    snprintf(policy_status, policy_status_sz, "BENCH_N4_RATE_PARTITION_PTHREADS");
    if (bench_rows_out) { bench_rows_out[0] = bench_rows_out[1] = bench_rows_out[2] = 0; }
    if (bench_rate_out) { bench_rate_out[0] = bench_rate_out[1] = bench_rate_out[2] = 0.0; }
    if (n4_bench_wall_s_out) *n4_bench_wall_s_out = 0.0;
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu < 1) use_gpu0 = 0;
    if (ngpu < 2) use_gpu1 = 0;
    if (is_naive_variant(variant)) use_cpu = 0;
    if (!use_cpu && !use_gpu0 && !use_gpu1) {
        snprintf(policy_status, policy_status_sz, "NO_ENABLED_DEVICE");
        if (t_total_out) *t_total_out = 0.0;
        init_empty_workers(out_w, N, "SKIPPED", -500);
        return -500;
    }
    double bench_rates_local[3] = {0.0, 0.0, 0.0};
    int bench_rows_local[3] = {0, 0, 0};
    double n4_bench_t0 = tnow();
    int bench_rc = run_n4_pthreads_benchmark_full(N, M, A, B, C,
                                                  use_cpu, use_gpu0, use_gpu1,
                                                  gpu0_dev, gpu1_dev,
                                                  cpu_blas, gpu_blas, tile, dt, variant,
                                                  bench_rows_local, bench_rates_local);
    double n4_bench_t1 = tnow();
    double n4_bench_wall_s = nonneg_or_zero(n4_bench_t1 - n4_bench_t0);
    if (n4_bench_wall_s_out) *n4_bench_wall_s_out = n4_bench_wall_s;
    printf("PTHREADS_N4_CALIBRATION mode=pthreads N=%d bench_rows_target=%d wall_s=%.6f excluded_from_N_benchmark=1 status=%s\n",
           N, n4_rows_for_bench(N), sane(n4_bench_wall_s), (bench_rc == 0) ? "OK" : "FAILED");
    if (bench_rows_out) { bench_rows_out[0] = bench_rows_local[0]; bench_rows_out[1] = bench_rows_local[1]; bench_rows_out[2] = bench_rows_local[2]; }
    if (bench_rate_out) { bench_rate_out[0] = bench_rates_local[0]; bench_rate_out[1] = bench_rates_local[1]; bench_rate_out[2] = bench_rates_local[2]; }

    int cpu_r0, cpu_r1, g0_r0, g0_r1, g1_r0, g1_r1;
    double cpu_share = 0.0;
    int used_shared_blas_partition = 0;
    int part_rc = choose_rows_partition_for_variant(N, M, use_cpu, use_gpu0, use_gpu1, variant, bench_rates_local,
                                                    1,
                                                    &cpu_r0, &cpu_r1, &g0_r0, &g0_r1, &g1_r0, &g1_r1, &cpu_share,
                                                    &used_shared_blas_partition);
    (void)used_shared_blas_partition;
    if (part_rc != 0) {
        snprintf(policy_status, policy_status_sz, "BENCH_N4_RATE_PARTITION_PTHREADS_FAILED_NO_EQUAL_FALLBACK");
        if (t_total_out) *t_total_out = 0.0;
        init_empty_workers(out_w, N, "NO_RATE_PARTITION", -502);
        return -502;
    }
    if (validate_partition(M, cpu_r0, cpu_r1, g0_r0, g0_r1, g1_r0, g1_r1) != 0) {
        snprintf(policy_status, policy_status_sz, "INVALID_FIXED_PARTITION");
        if (t_total_out) *t_total_out = 0.0;
        init_empty_workers(out_w, N, "INVALID_PARTITION", -501);
        return -501;
    }
    printf("PARTITION_RATE mode=pthreads variant=%s policy=%s cpu_share=%.6f rows_cpu=%d rows_gpu0=%d rows_gpu1=%d rate_cpu=%.6f rate_gpu0=%.6f rate_gpu1=%.6f target_rate_time_s=%.6f\n",
           variant ? variant : "variant", policy_status ? policy_status : "BENCH_N4_RATE_PARTITION_PTHREADS",
           sane(cpu_share), cpu_r1 - cpu_r0, g0_r1 - g0_r0, g1_r1 - g1_r0,
           sane(bench_rates_local[0]), sane(bench_rates_local[1]), sane(bench_rates_local[2]),
           (bench_rates_local[0] + bench_rates_local[1] + bench_rates_local[2] > 0.0) ? ((double)M / (bench_rates_local[0] + bench_rates_local[1] + bench_rates_local[2])) : 0.0);
    worker_t w[3];
    pthread_t th[3];
    int started[3] = {0,0,0};
    int wanted[3] = { use_cpu, use_gpu0, use_gpu1 };
    double ts = tnow();
    double t_ref_use = (t_ref_base > 0.0) ? t_ref_base : ts;
    memset(w, 0, sizeof(w));
    w[0] = (worker_t){ .wid=0, .enabled=use_cpu,  .gpu_dev=-1,        .N=N, .r0=cpu_r0, .r1=cpu_r1, .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=B, .C=C, .t_ref=t_ref_use };
    w[1] = (worker_t){ .wid=1, .enabled=use_gpu0, .gpu_dev=gpu0_dev, .N=N, .r0=g0_r0,  .r1=g0_r1,  .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=B, .C=C, .t_ref=t_ref_use };
    w[2] = (worker_t){ .wid=2, .enabled=use_gpu1, .gpu_dev=gpu1_dev, .N=N, .r0=g1_r0,  .r1=g1_r1,  .cpu_blas=cpu_blas, .gpu_blas=gpu_blas, .tile=tile, .dt=dt, .A=A, .B=B, .C=C, .t_ref=t_ref_use };
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
        set_worker_trace(&w[i], "PTHREADS_TIMELINE_TILE_EVENT", "pthreads", 0, -1, 0, N, t_ref_use, 0.0);
        snprintf(w[i].status, sizeof(w[i].status), "UNSET");
    }

    if (pthread_create(&th[1], NULL, worker_main, &w[1]) == 0) started[1] = 1; else { w[1].enabled = 0; w[1].err = -410; snprintf(w[1].status, sizeof(w[1].status), "THREAD_CREATE_FAIL"); }
    if (pthread_create(&th[2], NULL, worker_main, &w[2]) == 0) started[2] = 1; else { w[2].enabled = 0; w[2].err = -411; snprintf(w[2].status, sizeof(w[2].status), "THREAD_CREATE_FAIL"); }
    if (pthread_create(&th[0], NULL, worker_main, &w[0]) == 0) started[0] = 1; else { w[0].enabled = 0; w[0].err = -412; snprintf(w[0].status, sizeof(w[0].status), "THREAD_CREATE_FAIL"); }

    if (started[0]) pthread_join(th[0], NULL);
    if (started[1]) pthread_join(th[1], NULL);
    if (started[2]) pthread_join(th[2], NULL);
    double te = tnow();

    int create_fail = 0;
    for (int i = 0; i < 3; i++) {
        if (wanted[i] && !started[i]) create_fail = 1;
    }
    if (create_fail) snprintf(policy_status, policy_status_sz, "THREAD_CREATE_FAIL");
    if (out_w) { out_w[0] = w[0]; out_w[1] = w[1]; out_w[2] = w[2]; }
    if (t_total_out) *t_total_out = te - ts;
    return create_fail ? -410 : 0;
}


/*
 * True 3-process MPI mode:
 *   rank 0 -> CPU only
 *   rank 1 -> GPU0 only, OpenACC only
 *   rank 2 -> GPU1 only, OpenACC only
 *
 * This replaces the old MPI-hybrid behavior where every MPI rank created
 * pthread workers for CPU/GPU0/GPU1. In this path there are no pthread
 * workers inside MPI ranks; each MPI rank owns exactly one compute role.
 */
const char *mpi_three_role_name(int role) {
    return (role == 0) ? "CPU" : ((role == 1) ? "GPU0" : ((role == 2) ? "GPU1" : "INVALID"));
}

int mpi_three_role_gpu_dev(int role) {
    const char *env = (role == 1) ? getenv("MPI_GPU0_DEVICE") : ((role == 2) ? getenv("MPI_GPU1_DEVICE") : NULL);
    if (env && *env) {
        char *endp = NULL;
        long v = strtol(env, &endp, 10);
        if (endp && endp != env && v >= 0 && v <= INT_MAX) return (int)v;
    }
    return role - 1;
}

int mpi_three_choose_counts(int N, int use_cpu, int use_gpu0, int use_gpu1,
                                   const char *variant, const double bench_rates[3],
                                   int counts_rows[3], int displs_rows[3],
                                   int global_r0[3], int global_r1[3],
                                   double *cpu_share_out,
                                   int *used_shared_blas_partition_out) {
    int cpu_r0 = 0, cpu_r1 = 0, g0_r0 = 0, g0_r1 = 0, g1_r0 = 0, g1_r1 = 0;
    double cpu_share = 0.0;
    int used_shared_blas_partition = 0;
    int part_rc = choose_rows_partition_for_variant(N, N, use_cpu, use_gpu0, use_gpu1, variant, bench_rates,
                                                    1,
                                                    &cpu_r0, &cpu_r1, &g0_r0, &g0_r1, &g1_r0, &g1_r1, &cpu_share,
                                                    &used_shared_blas_partition);
    if (part_rc != 0) {
        /* No equal fallback for MPI: require valid benchmark rates for standard rate-based partitioning. */
        return part_rc;
    }
    counts_rows[0] = cpu_r1 - cpu_r0;
    counts_rows[1] = g0_r1 - g0_r0;
    counts_rows[2] = g1_r1 - g1_r0;
    displs_rows[0] = cpu_r0;
    displs_rows[1] = g0_r0;
    displs_rows[2] = g1_r0;
    global_r0[0] = cpu_r0; global_r1[0] = cpu_r1;
    global_r0[1] = g0_r0;  global_r1[1] = g0_r1;
    global_r0[2] = g1_r0;  global_r1[2] = g1_r1;
    if (cpu_share_out) *cpu_share_out = cpu_share;
    if (used_shared_blas_partition_out) *used_shared_blas_partition_out = used_shared_blas_partition;
    return part_rc;
}

int mpi_three_gpu_available_for_role(int role, char *status, size_t status_sz) {
    if (role == 0) return 1;
    int dev = mpi_three_role_gpu_dev(role);
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu <= dev || dev < 0) {
        snprintf(status, status_sz, "GPU_DEVICE_UNAVAILABLE(dev=%d,ngpu=%d)", dev, ngpu);
        return 0;
    }
    return 1;
}

void mpi_three_run_one_full_worker(int role, int N, int local_rows,
                                          void *A_local, void *B, void *C_local,
                                          int use_cpu, int use_gpu0, int use_gpu1,
                                          int cpu_blas, int gpu_blas, int tile, dtype_t dt,
                                          double root_abs, double clock_offset_to_root,
                                          worker_t out_w[3]) {
    init_empty_workers(out_w, N, "SKIPPED", 0);
    int role_enabled = (role == 0) ? use_cpu : ((role == 1) ? use_gpu0 : use_gpu1);
    if (!role_enabled || local_rows <= 0) return;

    worker_t *w = &out_w[role];
    memset(w, 0, sizeof(*w));
    init_worker_timing_fields(w);
    w->wid = role;
    w->enabled = 1;
    w->gpu_dev = (role == 0) ? -1 : mpi_three_role_gpu_dev(role);
    w->N = N;
    w->r0 = 0;
    w->r1 = local_rows;
    w->cpu_blas = cpu_blas;
    w->gpu_blas = (role == 0) ? 0 : gpu_blas;
    w->tile = tile;
    w->dt = dt;
    w->A = A_local;
    w->B = B;
    w->C = C_local;
    w->t_ref = root_abs - clock_offset_to_root;
    set_worker_trace(w, "MPI_TIMELINE_TILE_EVENT", "mpi3proc", role, -1, 0, N, root_abs, clock_offset_to_root);
    snprintf(w->status, sizeof(w->status), "OK");

    if (role != 0) {
        char why[64];
        if (!mpi_three_gpu_available_for_role(role, why, sizeof(why))) {
            w->err = -620;
            snprintf(w->status, sizeof(w->status), "%s", why);
            w->rows_done = 0;
            finalize_worker_perf(w);
            return;
        }
    }

    worker_main(w);
    shift_worker_abs_to_root_clock(w, clock_offset_to_root, root_abs);
}


int run_n4_mpi_benchmark_full(int N, int use_cpu, int use_gpu0, int use_gpu1,
                                     int cpu_blas, int gpu_blas, int tile, dtype_t dt, const char *variant,
                                     int rank, int size, MPI_Datatype mpi_dt,
                                     int bench_rows_out[3], double bench_rate_out[3]) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
#endif
    if (bench_rows_out) bench_rows_out[0] = bench_rows_out[1] = bench_rows_out[2] = 0;
    if (bench_rate_out) bench_rate_out[0] = bench_rate_out[1] = bench_rate_out[2] = 0.0;
    if (N <= 0 || size != 3) return -1;
    if (is_naive_variant(variant)) use_cpu = 0;

    int role_enabled = (rank == 0) ? use_cpu : ((rank == 1) ? use_gpu0 : use_gpu1);
    int target = n4_rows_for_bench(N);
    int local_rows = role_enabled ? target : 0;
    size_t esz = dtype_size(dt);
    unsigned long long elemsB = mat_elems_NN(N);
    unsigned long long elemsA = mat_elems_rows(local_rows, N);
    unsigned long long elemsC = mat_elems_rows(local_rows, N);

    void *B = malloc_aligned64((size_t)elemsB * esz);
    void *A_local = local_rows > 0 ? malloc_aligned64((size_t)elemsA * esz) : NULL;
    void *C_local = local_rows > 0 ? malloc_aligned64((size_t)elemsC * esz) : NULL;
    int alloc_fail = (!B || (local_rows > 0 && (!A_local || !C_local))) ? 1 : 0;
    int alloc_fail_any = 0;
    MPI_Allreduce(&alloc_fail, &alloc_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (alloc_fail_any) {
        if (rank == 0) printf("BENCH_N4_SPEED mode=mpi3proc N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=ALLOC_FAIL\n", N, N, target);
        free(B); free(A_local); free(C_local);
        return -1;
    }

    int fill_fail = 0;
    if (rank == 0) fill_fail |= fill_B_only(N, dt, B);
    if (local_rows > 0) fill_fail |= fill_A_C_rows(N, rank * target, local_rows, dt, A_local, C_local);
    int fill_fail_any = 0;
    MPI_Allreduce(&fill_fail, &fill_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (fill_fail_any) {
        if (rank == 0) printf("BENCH_N4_SPEED mode=mpi3proc N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=FILL_FAIL\n", N, N, target);
        free(B); free(A_local); free(C_local);
        return -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int bcast_rc = mpi_bcast_large(B, elemsB, mpi_dt, 0, MPI_COMM_WORLD);
    int bcast_fail = (bcast_rc != MPI_SUCCESS) ? 1 : 0;
    int bcast_fail_any = 0;
    MPI_Allreduce(&bcast_fail, &bcast_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (bcast_fail_any) {
        if (rank == 0) printf("BENCH_N4_SPEED mode=mpi3proc N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=MPI_BCAST_FAIL\n", N, N, target);
        free(B); free(A_local); free(C_local);
        return -1;
    }

    worker_t wloc[3];
    init_empty_workers(wloc, N, "SKIPPED", 0);
    if (role_enabled && local_rows > 0) {
        worker_t *w = &wloc[rank];
        memset(w, 0, sizeof(*w));
        init_worker_timing_fields(w);
        w->wid = rank;
        w->enabled = 1;
        w->gpu_dev = (rank == 0) ? -1 : mpi_three_role_gpu_dev(rank);
        w->N = N;
        w->r0 = 0;
        w->r1 = local_rows;
        w->cpu_blas = cpu_blas;
        w->gpu_blas = (rank == 0) ? 0 : gpu_blas;
        w->tile = tile;
        w->dt = dt;
        w->A = A_local;
        w->B = B;
        w->C = C_local;
        w->t_ref = tnow();
        snprintf(w->status, sizeof(w->status), "BENCH_UNSET");
        if (rank != 0) {
            char why[64];
            if (!mpi_three_gpu_available_for_role(rank, why, sizeof(why))) {
                w->err = -620;
                snprintf(w->status, sizeof(w->status), "%s", why);
                w->rows_done = 0;
                finalize_worker_perf(w);
            } else {
                worker_main(w);
            }
        } else {
            worker_main(w);
        }
    }

    worker_t one = wloc[rank];
    worker_t gathered[3];
    MPI_Gather(&one, (int)sizeof(worker_t), MPI_BYTE,
               gathered, (int)sizeof(worker_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    int rows_done[3] = {0,0,0};
    double rates[3] = {0.0,0.0,0.0};
    int ok_rates = 0;
    if (rank == 0) {
        for (int i = 0; i < 3; i++) {
            rows_done[i] = (int)gathered[i].rows_done;
            rates[i] = worker_partition_rate(&gathered[i]);
            if (rates[i] > 0.0) ok_rates++;
        }
        print_bench_n4_speed("mpi3proc", N, N, target, rows_done, rates,
                             gathered[0].gemm_time, gathered[1].gemm_time + gathered[1].copy_time, gathered[2].gemm_time + gathered[2].copy_time,
                             gathered[0].gflops_effective, gathered[1].gflops_effective, gathered[2].gflops_effective);
    }
    MPI_Bcast(rows_done, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rates, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ok_rates, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (bench_rows_out) { bench_rows_out[0] = rows_done[0]; bench_rows_out[1] = rows_done[1]; bench_rows_out[2] = rows_done[2]; }
    if (bench_rate_out) { bench_rate_out[0] = rates[0]; bench_rate_out[1] = rates[1]; bench_rate_out[2] = rates[2]; }

    free(B); free(A_local); free(C_local);
    return ok_rates > 0 ? 0 : -1;
}

int run_n4_mpi_benchmark_panel(int N, int cols,
                                      int use_cpu, int use_gpu0, int use_gpu1,
                                      int cpu_blas, int gpu_blas, int tile, dtype_t dt, const char *variant,
                                      int rank, int size, MPI_Datatype mpi_dt,
                                      int bench_rows_out[3], double bench_rate_out[3]) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
#endif
    if (bench_rows_out) bench_rows_out[0] = bench_rows_out[1] = bench_rows_out[2] = 0;
    if (bench_rate_out) bench_rate_out[0] = bench_rate_out[1] = bench_rate_out[2] = 0.0;
    if (N <= 0 || cols <= 0 || size != 3) return -1;
    if (cols > N) cols = N;
    if (is_naive_variant(variant)) use_cpu = 0;

    int role_enabled = (rank == 0) ? use_cpu : ((rank == 1) ? use_gpu0 : use_gpu1);
    int target = n4_rows_for_bench(N);
    int local_rows = role_enabled ? target : 0;
    size_t esz = dtype_size(dt);
    unsigned long long elemsB = (unsigned long long)N * (unsigned long long)cols;
    unsigned long long elemsA = mat_elems_rows(local_rows, N);
    unsigned long long elemsC = (unsigned long long)local_rows * (unsigned long long)cols;

    void *B_panel = malloc_aligned64((size_t)elemsB * esz);
    void *A_local = local_rows > 0 ? malloc_aligned64((size_t)elemsA * esz) : NULL;
    void *C_panel = local_rows > 0 ? malloc_aligned64((size_t)elemsC * esz) : NULL;
    int alloc_fail = (!B_panel || (local_rows > 0 && (!A_local || !C_panel))) ? 1 : 0;
    int alloc_fail_any = 0;
    MPI_Allreduce(&alloc_fail, &alloc_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (alloc_fail_any) {
        if (rank == 0) printf("BENCH_N4_SPEED mode=mpi3proc_panel N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=ALLOC_FAIL\n", N, cols, target);
        free(B_panel); free(A_local); free(C_panel);
        return -1;
    }

    int fill_fail = 0;
    if (rank == 0) fill_B_panel_formula(N, 0, cols, dt, B_panel);
    if (local_rows > 0) {
        fill_fail |= fill_A_rows_only(N, rank * target, local_rows, dt, A_local);
        memset(C_panel, 0, (size_t)elemsC * esz);
    }
    int fill_fail_any = 0;
    MPI_Allreduce(&fill_fail, &fill_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (fill_fail_any) {
        if (rank == 0) printf("BENCH_N4_SPEED mode=mpi3proc_panel N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=FILL_FAIL\n", N, cols, target);
        free(B_panel); free(A_local); free(C_panel);
        return -1;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    int bcast_rc = mpi_bcast_large(B_panel, elemsB, mpi_dt, 0, MPI_COMM_WORLD);
    int bcast_fail = (bcast_rc != MPI_SUCCESS) ? 1 : 0;
    int bcast_fail_any = 0;
    MPI_Allreduce(&bcast_fail, &bcast_fail_any, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (bcast_fail_any) {
        if (rank == 0) printf("BENCH_N4_SPEED mode=mpi3proc_panel N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=MPI_BCAST_FAIL\n", N, cols, target);
        free(B_panel); free(A_local); free(C_panel);
        return -1;
    }

    panel_worker_t wloc[3];
    init_empty_panel_workers(wloc, "SKIPPED", 0);
    if (role_enabled && local_rows > 0) {
        panel_worker_t *w = &wloc[rank];
        memset(w, 0, sizeof(*w));
        init_panel_worker_timing_fields(w);
        w->wid = rank;
        w->enabled = 1;
        w->gpu_dev = (rank == 0) ? -1 : mpi_three_role_gpu_dev(rank);
        w->K = N;
        w->cols = cols;
        w->r0 = 0;
        w->r1 = local_rows;
        w->cpu_blas = cpu_blas;
        w->gpu_blas = (rank == 0) ? 0 : gpu_blas;
        w->tile = tile;
        w->dt = dt;
        w->A = A_local;
        w->B = B_panel;
        w->C = C_panel;
        w->c_ld = cols;
        w->c_col0 = 0;
        w->t_ref = tnow();
        snprintf(w->status, sizeof(w->status), "BENCH_UNSET");
        if (rank != 0) {
            char why[64];
            if (!mpi_three_gpu_available_for_role(rank, why, sizeof(why))) {
                w->err = -620;
                snprintf(w->status, sizeof(w->status), "%s", why);
                w->rows_done = 0;
                finalize_panel_worker_times(w);
            } else {
                panel_worker_main(w);
            }
        } else {
            panel_worker_main(w);
        }
    }

    panel_worker_t one = wloc[rank];
    panel_worker_t gathered[3];
    MPI_Gather(&one, (int)sizeof(panel_worker_t), MPI_BYTE,
               gathered, (int)sizeof(panel_worker_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    int rows_done[3] = {0,0,0};
    double rates[3] = {0.0,0.0,0.0};
    int ok_rates = 0;
    if (rank == 0) {
        double gflops_eff[3] = {0.0, 0.0, 0.0};
        double elapsed[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < 3; i++) {
            rows_done[i] = (int)gathered[i].rows_done;
            rates[i] = panel_worker_partition_rate(&gathered[i]);
            if (rates[i] > 0.0) ok_rates++;
            elapsed[i] = (i == 0) ? gathered[i].gemm_time : (gathered[i].gemm_time + gathered[i].copy_time);
            double flops = 2.0 * (double)rows_done[i] * (double)N * (double)cols;
            gflops_eff[i] = (elapsed[i] > 0.0) ? ((flops / 1e9) / elapsed[i]) : 0.0;
        }
        print_bench_n4_speed("mpi3proc_panel", N, cols, target, rows_done, rates,
                             elapsed[0], elapsed[1], elapsed[2],
                             gflops_eff[0], gflops_eff[1], gflops_eff[2]);
    }
    MPI_Bcast(rows_done, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rates, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&ok_rates, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (bench_rows_out) { bench_rows_out[0] = rows_done[0]; bench_rows_out[1] = rows_done[1]; bench_rows_out[2] = rows_done[2]; }
    if (bench_rate_out) { bench_rate_out[0] = rates[0]; bench_rate_out[1] = rates[1]; bench_rate_out[2] = rates[2]; }

    free(B_panel); free(A_local); free(C_panel);
    return ok_rates > 0 ? 0 : -1;
}


int run_n4_pthreads_partition_benchmark_root(int N, int use_cpu, int use_gpu0, int use_gpu1,
                                                   int cpu_blas, int gpu_blas, int tile, dtype_t dt, const char *variant,
                                                   int rank, int size,
                                                   int bench_rows_out[3], double bench_rate_out[3]) {
#ifndef USE_CBLAS
    cpu_blas = 0;
#endif
#ifndef USE_CUBLAS
    gpu_blas = 0;
#endif
    int rows_done[3] = {0, 0, 0};
    double rates[3] = {0.0, 0.0, 0.0};
    int rc = -1;

    if (bench_rows_out) { bench_rows_out[0] = bench_rows_out[1] = bench_rows_out[2] = 0; }
    if (bench_rate_out) { bench_rate_out[0] = bench_rate_out[1] = bench_rate_out[2] = 0.0; }
    if (N <= 0 || size != 3) return -1;
    if (is_naive_variant(variant)) use_cpu = 0;

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        size_t esz = dtype_size(dt);
        void *A = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
        void *B = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
        void *C = malloc_aligned64((size_t)mat_bytes_NN(N, esz));
        if (!A || !B || !C) {
            printf("BENCH_N4_SPEED mode=pthreads_for_mpi_partition N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=ALLOC_FAIL\n",
                   N, N, n4_rows_for_bench(N));
            rc = -1;
        } else if (fill_B_only(N, dt, B) != 0 || fill_A_C_only(N, dt, A, C) != 0) {
            printf("BENCH_N4_SPEED mode=pthreads_for_mpi_partition N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=FILL_FAIL\n",
                   N, N, n4_rows_for_bench(N));
            rc = -1;
        } else {
            rc = run_n4_pthreads_benchmark_full(N, N, A, B, C,
                                                use_cpu, use_gpu0, use_gpu1,
                                                use_gpu0 ? mpi_three_role_gpu_dev(1) : -1,
                                                use_gpu1 ? mpi_three_role_gpu_dev(2) : -1,
                                                cpu_blas, gpu_blas, tile, dt, variant,
                                                rows_done, rates);
            if (rc == 0) {
                int missing_enabled_rate =
                    (use_cpu  && !(rates[0] > 0.0)) ||
                    (use_gpu0 && !(rates[1] > 0.0)) ||
                    (use_gpu1 && !(rates[2] > 0.0));
                if (missing_enabled_rate) {
                    printf("BENCH_N4_SPEED mode=pthreads_for_mpi_partition N=%d cols=%d bench_rows_target=%d STATUS=FAILED reason=MISSING_ENABLED_DEVICE_RATE rows_cpu=%d rows_gpu0=%d rows_gpu1=%d rate_cpu=%.6f rate_gpu0=%.6f rate_gpu1=%.6f\n",
                           N, N, n4_rows_for_bench(N), rows_done[0], rows_done[1], rows_done[2],
                           sane(rates[0]), sane(rates[1]), sane(rates[2]));
                    rc = -1;
                }
            }
        }
        free(A);
        free(B);
        free(C);
    }

    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rows_done, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(rates, 3, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Barrier(MPI_COMM_WORLD);

    if (bench_rows_out) { bench_rows_out[0] = rows_done[0]; bench_rows_out[1] = rows_done[1]; bench_rows_out[2] = rows_done[2]; }
    if (bench_rate_out) { bench_rate_out[0] = rates[0]; bench_rate_out[1] = rates[1]; bench_rate_out[2] = rates[2]; }
    return rc;
}

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
                                         int global_r0[3], int global_r1[3]) {
    for (int d = 0; d < 3; d++) {
        gathered[d].r0 = global_r0[d];
        gathered[d].r1 = global_r1[d];
        if (global_r1[d] <= global_r0[d]) {
            gathered[d].enabled = 0;
            gathered[d].rows_done = 0;
            if (gathered[d].err == 0) snprintf(gathered[d].status, sizeof(gathered[d].status), "SKIPPED");
        }
        print_worker_line(&gathered[d], N);
    }
    print_part_percent_done(gathered, N);
    print_device_speeds(gathered);
    for (int d = 0; d < 3; d++) {
        print_worker_timeline_events("MPI_TIMELINE_EVENT", "mpi3proc", d, -1, 0, N, &gathered[d]);
    }

    double total_flops = 2.0 * (double)N * (double)N * (double)N;
    double mpi_gflops = (total_with_init > 0.0) ? ((total_flops / 1e9) / total_with_init) : 0.0;
    int any_enabled = 0;
    int all_ok = 1;
    for (int d = 0; d < 3; d++) {
        if (global_r1[d] > global_r0[d]) any_enabled = 1;
        if (global_r1[d] > global_r0[d] && gathered[d].err != 0) all_ok = 0;
        if (global_r1[d] > global_r0[d] && gathered[d].rows_done <= 0) all_ok = 0;
    }
    const char *status = (!any_enabled) ? "FAILED" : (all_ok ? "OK" : "DEGRADED");
    char full0[64], full1[64], mpi_init0[64], mpi_init1[64];
    fmt_ts_iso(root_abs, full0, sizeof(full0));
    fmt_ts_iso(root_abs + total_no_init, full1, sizeof(full1));
    fmt_ts_iso(g_mpi_init_start_abs, mpi_init0, sizeof(mpi_init0));
    fmt_ts_iso(g_mpi_init_end_abs, mpi_init1, sizeof(mpi_init1));
    printf("MPI_INIT init_start_abs=%s init_end_abs=%s init_s_max=%.6f total_with_init=%.6f\n",
           mpi_init0, mpi_init1, sane(mpi_init_s), sane(total_with_init));
    printf("MPI_TOTAL_TIMING start_abs=%s end_abs=%s start_rel=0.000000 end_rel=%.6f full_start_abs=%s full_end_abs=%s full_start_rel=%.6f full_end_rel=%.6f global_init_start_abs=%s global_init_end_abs=%s global_init_s=%.6f total_ex_init_s=%.6f total_with_init_s=%.6f comm_total_s=%.6f compute_s=%.6f other_s=%.6f timing_note=max_rank_durations_no_cross_rank_clock_subtraction\n",
           mpi_init0, full1, sane(total_with_init),
           full0, full1, 0.0, sane(total_no_init),
           mpi_init0, mpi_init1, sane(mpi_init_s), sane(total_no_init), sane(total_with_init),
           sane(comm_s), sane(compute_s), sane(nonneg_or_zero(total_with_init - (mpi_init_s + comm_s + compute_s))));
    printf("MPI_PART bcast=%.6f scatter=%.6f compute=%.6f gather=%.6f comm_total=%.6f comm_frac=%.6f total=%.6f total_with_init=%.6f process_compute_total=%.6f policy=%s\n",
           sane(bcast_s), sane(scatter_s), sane(compute_s), sane(gather_s), sane(comm_s),
           (total_with_init > 0.0) ? sane(comm_s / total_with_init) : 0.0,
           sane(total_no_init), sane(total_with_init), sane(compute_s), policy ? policy : "MPI_3_PROCESS_ROLES");
    printf("TOTAL total_s=%.6f mpi_compute_s=%.6f TOTAL_GFLOPS=%.3f STATUS=%s C[0]=%.6f CHECKSUM_SAMPLE=%.6e\n",
           sane(total_with_init), sane(compute_s), sane(mpi_gflops), status, sane(c00), sane(chk));
    print_run_summary("mpi3proc", variant, config_name, N, size, dt, use_cpu, use_gpu0, use_gpu1,
                      cpu_blas, gpu_blas, cpu_threads, tile, cache_fit, status, total_with_init,
                      mpi_gflops, c00, chk,
                      (int[3]){0,0,0}, (double[3]){0.0,0.0,0.0});
}

