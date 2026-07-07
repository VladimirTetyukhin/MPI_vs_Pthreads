#include "hybrid_common.h"

/* OpenACC/CUDA GPU kernels and tiled GPU execution helpers. */

int gpu_run_panel_tiled(int dev, int K, int cols, int rows, int tile, int gpu_blas, dtype_t dt,
                               const void *Ablk, const void *Bpanel, void *Cpanel,
                               double *t_init, double *t_gemm, double *t_copy, double *t_cleanup,
                               double *t_copyin, double *t_copyout,
                               char *status, size_t status_sz,
                               gpu_step_times_t *times,
                               const timeline_trace_t *trace) {
    *t_init = 0.0;
    *t_gemm = 0.0;
    *t_copy = 0.0;
    *t_cleanup = 0.0;
    *t_copyin = 0.0;
    *t_copyout = 0.0;
    snprintf(status, status_sz, "OK");
    reset_gpu_step_times(times);
    if (rows <= 0 || cols <= 0 || K <= 0) return 0;
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    /*
     * One-compute-per-panel path:
     * The MPI layer already split the output columns into a B/C panel
     * (``cols`` columns).  Do not split the reduction dimension K again here.
     * Copy the full local A block and the full B panel, launch exactly one
     * GEMM, and copy the completed C panel back.
     */
    (void)tile;
    size_t esz = dtype_size(dt);
    size_t asz = (size_t)rows * (size_t)K * esz;
    size_t bsz = (size_t)K * (size_t)cols * esz;
    size_t csz = (size_t)rows * (size_t)cols * esz;
    char *hA = NULL;
    char *hB = NULL;
    char *hC = NULL;
    void *dA = NULL;
    void *dB = NULL;
    void *dC = NULL;
    const char *Abytes = (const char*)Ablk;
    const char *Bbytes = (const char*)Bpanel;
    char *Cbytes = (char*)Cpanel;
#ifdef USE_CUBLAS
    cublasHandle_t h = 0;
#endif
    int ret = 0;

    hA = (char*)malloc_aligned64(asz);
    hB = (char*)malloc_aligned64(bsz);
    hC = (char*)malloc_aligned64(csz);
    if (!hA || !hB || !hC) {
        snprintf(status, status_sz, "FAIL_HOST_PANEL_ALLOC");
        ret = -320;
        goto cleanup;
    }

    acc_set_device_num(dev, acc_device_nvidia);
    double ti0 = tnow();
    if (times) times->init_start_abs = ti0;
    acc_init(acc_device_nvidia);
    dA = acc_malloc(asz);
    dB = acc_malloc(bsz);
    dC = acc_malloc(csz);
    if (!dA || !dB || !dC) {
        snprintf(status, status_sz, "FAIL_DEV_PANEL_ALLOC");
        ret = -321;
        close_gpu_init_if_open(times, t_init);
        goto cleanup;
    }
#ifdef USE_CUBLAS
    if (gpu_blas) {
        cublasStatus_t st = cublasCreate(&h);
        if (st != CUBLAS_STATUS_SUCCESS) {
            snprintf(status, status_sz, "FAIL_CUBLAS_CREATE");
            ret = -322;
            close_gpu_init_if_open(times, t_init);
            goto cleanup;
        }
    }
#endif
    double ti1 = tnow();
    if (times) times->init_end_abs = ti1;
    *t_init = ti1 - ti0;

    double tc2 = tnow();
    memcpy(hA, Abytes, asz);
    memcpy(hB, Bbytes, bsz);
    acc_memcpy_to_device(dA, hA, asz);
    acc_memcpy_to_device(dB, hB, bsz);
    acc_wait_all();
    double tc3 = tnow();
    if (times) mark_time_span(&times->copyin_start_abs, &times->copyin_end_abs, tc2, tc3);
    *t_copy += tc3 - tc2;
    *t_copyin += tc3 - tc2;
    trace_gpu_tile_event(trace, "comm", "copyin_tile", 0, 0, cols, 0, K, tc2, tc3, (unsigned long long)(asz + bsz));

    double tg0 = tnow();
#ifdef USE_CUBLAS
    if (gpu_blas) {
        int rc = cublas_gemm_real_panel(h, dt, rows, cols, K, 0, dA, dB, dC);
        if (rc != 0) {
            snprintf(status, status_sz, "FAIL_CUBLAS_GEMM");
            ret = rc;
            goto cleanup;
        }
        cudaDeviceSynchronize();
    } else
#endif
    {
        if (dt == DT_REAL_DOUBLE) gpu_openacc_panel_d(rows, cols, K, 0, (const double*)dA, (const double*)dB, (double*)dC);
        else gpu_openacc_panel_s(rows, cols, K, 0, (const float*)dA, (const float*)dB, (float*)dC);
    }
    double tg1 = tnow();
    if (times) mark_time_span(&times->gemm_start_abs, &times->gemm_end_abs, tg0, tg1);
    *t_gemm += tg1 - tg0;
    trace_gpu_tile_event(trace, "device", "gemm_tile", 0, 0, cols, 0, K, tg0, tg1, 0ULL);

    double tc4 = tnow();
    acc_memcpy_from_device(hC, dC, csz);
    acc_wait_all();
    memcpy(Cbytes, hC, csz);
    double tc5 = tnow();
    if (times) mark_time_span(&times->copyout_start_abs, &times->copyout_end_abs, tc4, tc5);
    *t_copy += tc5 - tc4;
    *t_copyout += tc5 - tc4;
    trace_gpu_tile_event(trace, "comm", "copyout_tile", 0, 0, cols, -1, 0, tc4, tc5, (unsigned long long)csz);

cleanup:
    {
        double tx0 = tnow();
        if (times) times->cleanup_start_abs = tx0;
#ifdef USE_CUBLAS
        if (h) cublasDestroy(h);
#endif
        if (dA) acc_free(dA);
        if (dB) acc_free(dB);
        if (dC) acc_free(dC);
        double tx1 = tnow();
        if (times) times->cleanup_end_abs = tx1;
        *t_cleanup += (tx1 - tx0);
    }
    free(hA);
    free(hB);
    free(hC);
    return ret;
}

int gpu_run_panel_tiled_strided(int dev, int K, int cols, int rows, int tile, int gpu_blas, dtype_t dt,
                               const void *Ablk, const void *Bpanel, void *Cfull,
                               int ldc, int c_col0,
                               double *t_init, double *t_gemm, double *t_copy, double *t_cleanup,
                               double *t_copyin, double *t_copyout,
                               char *status, size_t status_sz,
                               gpu_step_times_t *times,
                               const timeline_trace_t *trace) {
    *t_init = 0.0; *t_gemm = 0.0; *t_copy = 0.0; *t_cleanup = 0.0; *t_copyin = 0.0; *t_copyout = 0.0;
    snprintf(status, status_sz, "OK");
    reset_gpu_step_times(times);
    if (rows <= 0 || cols <= 0 || K <= 0) return 0;
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    /*
     * One-compute-per-panel path for strided C output.  The C result is
     * produced as a compact rows x cols panel on the device, then unpacked
     * into the full C leading dimension at c_col0.
     */
    (void)tile;
    size_t esz = dtype_size(dt);
    size_t asz = (size_t)rows * (size_t)K * esz;
    size_t bsz = (size_t)K * (size_t)cols * esz;
    size_t csz = (size_t)rows * (size_t)cols * esz;
    char *hA = NULL, *hB = NULL, *hC = NULL;
    void *dA = NULL, *dB = NULL, *dC = NULL;
    const char *Abytes = (const char*)Ablk;
    const char *Bbytes = (const char*)Bpanel;
    char *Cbytes = (char*)Cfull;
#ifdef USE_CUBLAS
    cublasHandle_t h = 0;
#endif
    int ret = 0;
    hA = (char*)malloc_aligned64(asz); hB = (char*)malloc_aligned64(bsz); hC = (char*)malloc_aligned64(csz);
    if (!hA || !hB || !hC) { snprintf(status, status_sz, "FAIL_HOST_PANEL_ALLOC"); ret = -323; goto cleanup; }
    acc_set_device_num(dev, acc_device_nvidia);
    double ti0 = tnow();
    if (times) times->init_start_abs = ti0;
    acc_init(acc_device_nvidia);
    dA = acc_malloc(asz); dB = acc_malloc(bsz); dC = acc_malloc(csz);
    if (!dA || !dB || !dC) { snprintf(status, status_sz, "FAIL_DEV_PANEL_ALLOC"); ret = -324; close_gpu_init_if_open(times, t_init); goto cleanup; }
#ifdef USE_CUBLAS
    if (gpu_blas) { cublasStatus_t st = cublasCreate(&h); if (st != CUBLAS_STATUS_SUCCESS) { snprintf(status, status_sz, "FAIL_CUBLAS_CREATE"); ret = -325; close_gpu_init_if_open(times, t_init); goto cleanup; } }
#endif
    double ti1 = tnow();
    if (times) times->init_end_abs = ti1;
    *t_init = ti1 - ti0;

    double tc2 = tnow();
    memcpy(hA, Abytes, asz);
    memcpy(hB, Bbytes, bsz);
    acc_memcpy_to_device(dA, hA, asz); acc_memcpy_to_device(dB, hB, bsz); acc_wait_all();
    double tc3 = tnow();
    if (times) mark_time_span(&times->copyin_start_abs, &times->copyin_end_abs, tc2, tc3);
    *t_copy += tc3 - tc2; *t_copyin += tc3 - tc2;
    trace_gpu_tile_event(trace, "comm", "copyin_tile", 0, 0, cols, 0, K, tc2, tc3, (unsigned long long)(asz + bsz));

    double tg0 = tnow();
#ifdef USE_CUBLAS
    if (gpu_blas) {
        int rc = cublas_gemm_real_panel(h, dt, rows, cols, K, 0, dA, dB, dC);
        if (rc != 0) { snprintf(status, status_sz, "FAIL_CUBLAS_GEMM"); ret = rc; goto cleanup; }
        cudaDeviceSynchronize();
    } else
#endif
    {
        if (dt == DT_REAL_DOUBLE) gpu_openacc_panel_d(rows, cols, K, 0, (const double*)dA, (const double*)dB, (double*)dC);
        else gpu_openacc_panel_s(rows, cols, K, 0, (const float*)dA, (const float*)dB, (float*)dC);
    }
    double tg1 = tnow();
    if (times) mark_time_span(&times->gemm_start_abs, &times->gemm_end_abs, tg0, tg1);
    *t_gemm += tg1 - tg0;
    trace_gpu_tile_event(trace, "device", "gemm_tile", 0, 0, cols, 0, K, tg0, tg1, 0ULL);

    double tc4 = tnow();
    acc_memcpy_from_device(hC, dC, csz); acc_wait_all();
    unpack_C_panel_bytes(hC, rows, ldc, c_col0, cols, esz, Cbytes);
    double tc5 = tnow();
    if (times) mark_time_span(&times->copyout_start_abs, &times->copyout_end_abs, tc4, tc5);
    *t_copy += tc5 - tc4; *t_copyout += tc5 - tc4;
    trace_gpu_tile_event(trace, "comm", "copyout_tile", 0, 0, cols, -1, 0, tc4, tc5, (unsigned long long)csz);

cleanup:
    { double tx0 = tnow();
      if (times) times->cleanup_start_abs = tx0;
#ifdef USE_CUBLAS
      if (h) cublasDestroy(h);
#endif
      if (dA) acc_free(dA);
      if (dB) acc_free(dB);
      if (dC) acc_free(dC);
      double tx1 = tnow();
      if (times) times->cleanup_end_abs = tx1;
      *t_cleanup += (tx1 - tx0); }
    free(hA); free(hB); free(hC); return ret;
}

int choose_gpu_tile(int rows, int N, size_t esz, int tile_req) {
    /*
     * Full-compute mode: do NOT clamp GPU tile size by any fixed memory budget.
     * The benchmark script passes tile=N, and with this function that now
     * means one full j tile and one full k tile for non-panel runs.
     *
     * Memory consequence per GPU worker/rank in the full tile=N path is now:
     *   device: rows*N*esz + N*N*esz + rows*N*esz bytes
     *   host: no extra hA/hB/hC staging buffers; copies use A/B/C directly
     * If device allocations do not fit, the run fails with FAIL_DEV_PANEL_ALLOC
     * instead of silently falling back to smaller GPU tiles.
     */
    (void)rows;
    (void)esz;
    if (N <= 0) return 1;

    unsigned long long t = (tile_req > 0) ? (unsigned long long)tile_req : (unsigned long long)N;
    if (t > (unsigned long long)N) t = (unsigned long long)N;
    if (t < 1ULL) t = 1ULL;
    return (int)t;
}

int gpu_run_tiled(int dev, int N, int rows, int tile, int gpu_blas, dtype_t dt,
                         const void *Ablk, const void *B, void *Cblk,
                         double *t_init, double *t_gemm, double *t_copy, double *t_cleanup,
                         double *t_copyin, double *t_copyout,
                         char *status, size_t status_sz,
                         gpu_step_times_t *times,
                         const timeline_trace_t *trace) {
    *t_init = 0.0;
    *t_gemm = 0.0;
    *t_copy = 0.0;
    *t_cleanup = 0.0;
    *t_copyin = 0.0;
    *t_copyout = 0.0;
    snprintf(status, status_sz, "OK");
    reset_gpu_step_times(times);
    if (rows <= 0) return 0;
#ifndef USE_CUBLAS
    gpu_blas = 0;
    (void)gpu_blas;
#endif
    size_t esz = dtype_size(dt);
    int tt = choose_gpu_tile(rows, N, esz, tile);
    int full_no_staging = (tt >= N);
    size_t maxA = (size_t)rows * (size_t)tt * esz;
    size_t maxB = (size_t)tt * (size_t)tt * esz;
    size_t maxC = (size_t)rows * (size_t)tt * esz;
    char *hA = NULL;
    char *hB = NULL;
    char *hC = NULL;
    void *dA = NULL;
    void *dB = NULL;
    void *dC = NULL;
    const char *Abytes = (const char*)Ablk;
    const char *Bbytes = (const char*)B;
    char *Cbytes = (char*)Cblk;
#ifdef USE_CUBLAS
    cublasHandle_t h = 0;
#endif
    int ret = 0;

    /* Exact no-tiling fix: when tile=N, Ablk/B/Cblk are already contiguous.
     * Do not allocate hA/hB/hC host staging. Only device buffers remain.
     * The old packed-staging path is kept for non-full tiles. */
    if (!full_no_staging) {
        hA = (char*)malloc_aligned64(maxA);
        hB = (char*)malloc_aligned64(maxB);
        hC = (char*)malloc_aligned64(maxC);
        if (!hA || !hB || !hC) {
            snprintf(status, status_sz, "FAIL_HOST_PANEL_ALLOC");
            ret = -300;
            goto cleanup;
        }
    }

    acc_set_device_num(dev, acc_device_nvidia);
    double ti0 = tnow();
    if (times) times->init_start_abs = ti0;
    acc_init(acc_device_nvidia);
    dA = acc_malloc(maxA);
    dB = acc_malloc(maxB);
    dC = acc_malloc(maxC);
    if (!dA || !dB || !dC) {
        snprintf(status, status_sz, "FAIL_DEV_PANEL_ALLOC");
        ret = -301;
        close_gpu_init_if_open(times, t_init);
        goto cleanup;
    }
#ifdef USE_CUBLAS
    if (gpu_blas) {
        cublasStatus_t st = cublasCreate(&h);
        if (st != CUBLAS_STATUS_SUCCESS) {
            snprintf(status, status_sz, "FAIL_CUBLAS_CREATE");
            ret = -302;
            close_gpu_init_if_open(times, t_init);
            goto cleanup;
        }
    }
#endif
    double ti1 = tnow();
    if (times) times->init_end_abs = ti1;
    *t_init = ti1 - ti0;

    int tile_idx = 0;
    for (int j0 = 0; j0 < N; j0 += tt) {
        int jb = (j0 + tt <= N) ? tt : (N - j0);
        size_t csz = (size_t)rows * (size_t)jb * esz;

        for (int k0 = 0; k0 < N; k0 += tt) {
            int kb = (k0 + tt <= N) ? tt : (N - k0);
            int accumulate = (k0 != 0);
            size_t asz = (size_t)rows * (size_t)kb * esz;
            size_t bsz = (size_t)kb * (size_t)jb * esz;

            double tc2 = tnow();
            if (full_no_staging) {
                acc_memcpy_to_device(dA, Abytes, asz);
                acc_memcpy_to_device(dB, Bbytes, bsz);
            } else {
                pack_A_panel_bytes(Abytes, rows, N, k0, kb, esz, hA);
                pack_B_panel_bytes(Bbytes, N, k0, kb, j0, jb, esz, hB);
                acc_memcpy_to_device(dA, hA, asz);
                acc_memcpy_to_device(dB, hB, bsz);
            }
            acc_wait_all();
            double tc3 = tnow();
            if (times) mark_time_span(&times->copyin_start_abs, &times->copyin_end_abs, tc2, tc3);
            *t_copy += tc3 - tc2;
            *t_copyin += tc3 - tc2;
            trace_gpu_tile_event(trace, "comm", "copyin_tile", tile_idx, j0, jb, k0, kb, tc2, tc3, (unsigned long long)(asz + bsz));

            double tg0 = tnow();
#ifdef USE_CUBLAS
            if (gpu_blas) {
                int rc = cublas_gemm_real_panel(h, dt, rows, jb, kb, accumulate, dA, dB, dC);
                if (rc != 0) {
                    snprintf(status, status_sz, "FAIL_CUBLAS_GEMM");
                    ret = rc;
                    goto cleanup;
                }
                cudaDeviceSynchronize();
            } else
#endif
            {
                if (dt == DT_REAL_DOUBLE) gpu_openacc_panel_d(rows, jb, kb, accumulate, (const double*)dA, (const double*)dB, (double*)dC);
                else gpu_openacc_panel_s(rows, jb, kb, accumulate, (const float*)dA, (const float*)dB, (float*)dC);
            }
            double tg1 = tnow();
            if (times) mark_time_span(&times->gemm_start_abs, &times->gemm_end_abs, tg0, tg1);
            *t_gemm += tg1 - tg0;
            trace_gpu_tile_event(trace, "device", "gemm_tile", tile_idx, j0, jb, k0, kb, tg0, tg1, 0ULL);
            tile_idx++;
        }

        double tc4 = tnow();
        if (full_no_staging) {
            acc_memcpy_from_device(Cbytes, dC, csz);
        } else {
            acc_memcpy_from_device(hC, dC, csz);
            unpack_C_panel_bytes(hC, rows, N, j0, jb, esz, Cbytes);
        }
        acc_wait_all();
        double tc5 = tnow();
        if (times) mark_time_span(&times->copyout_start_abs, &times->copyout_end_abs, tc4, tc5);
        *t_copy += tc5 - tc4;
        *t_copyout += tc5 - tc4;
        trace_gpu_tile_event(trace, "comm", "copyout_tile", tile_idx, j0, jb, -1, 0, tc4, tc5, (unsigned long long)csz);
        tile_idx++;
    }

cleanup:
    {
        double tx0 = tnow();
        if (times) times->cleanup_start_abs = tx0;
#ifdef USE_CUBLAS
        if (h) cublasDestroy(h);
#endif
        if (dA) acc_free(dA);
        if (dB) acc_free(dB);
        if (dC) acc_free(dC);
        double tx1 = tnow();
        if (times) times->cleanup_end_abs = tx1;
        *t_cleanup += (tx1 - tx0);
    }
    free(hA);
    free(hB);
    free(hC);
    return ret;
}

