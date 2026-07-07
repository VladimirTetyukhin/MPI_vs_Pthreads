#include "hybrid_common.h"

/* Worker timing/performance finalization and pthread worker entry points. */

void finalize_worker_perf(worker_t *w) {
    int N = w->N;
    long long rows = w->rows_done;
    w->flops = 2.0 * (double)rows * (double)N * (double)N;
    w->gflops_gemm = (w->gemm_time > 0.0) ? ((w->flops / 1e9) / w->gemm_time) : 0.0;
    if (w->wid == 0) w->gflops_effective = w->gflops_gemm;
    else {
        double t_eff = w->gemm_time + w->copy_time;
        w->gflops_effective = (t_eff > 0.0) ? ((w->flops / 1e9) / t_eff) : 0.0;
    }
    double t_eff2 = (w->wid == 0) ? w->gemm_time : (w->gemm_time + w->copy_time);
    w->rows_per_s_eff = (t_eff2 > 0.0) ? ((double)w->rows_done / t_eff2) : 0.0;
    if (w->wid == 0) w->copy_bw_gbs = 0.0;
    else {
        w->copy_bw_gbs = (w->copy_time > 0.0 && w->copy_bytes > 0ULL) ? ((double)w->copy_bytes / 1e9 / w->copy_time) : 0.0;
    }
    w->gflops_gemm = sane(w->gflops_gemm);
    w->gflops_effective = sane(w->gflops_effective);
    w->rows_per_s_eff = sane(w->rows_per_s_eff);
    w->copy_bw_gbs = sane(w->copy_bw_gbs);
    w->t_dev_start_rel = -1.0;
    w->t_dev_end_rel = -1.0;
    if (w->enabled && (w->rows_done > 0 || w->err != 0)) {
        double s = 1e300, e = -1e300;
        if (w->t_init_end_rel > w->t_init_start_rel) { if (w->t_init_start_rel < s) s = w->t_init_start_rel; if (w->t_init_end_rel > e) e = w->t_init_end_rel; }
        if (w->t_gemm_end_rel > w->t_gemm_start_rel) { if (w->t_gemm_start_rel < s) s = w->t_gemm_start_rel; if (w->t_gemm_end_rel > e) e = w->t_gemm_end_rel; }
        if (w->t_copy_end_rel > w->t_copy_start_rel) { if (w->t_copy_start_rel < s) s = w->t_copy_start_rel; if (w->t_copy_end_rel > e) e = w->t_copy_end_rel; }
        if (w->t_cleanup_end_rel > w->t_cleanup_start_rel) { if (w->t_cleanup_start_rel < s) s = w->t_cleanup_start_rel; if (w->t_cleanup_end_rel > e) e = w->t_cleanup_end_rel; }
        if (e > s && s < 1e299) {
            w->t_dev_start_rel = s;
            w->t_dev_end_rel = e;
        }
    }
}


void merge_rel_span(double *dst_start, double *dst_end, double src_start, double src_end);


void worker_phase_sums(const worker_t w[3],
                              double *init_s, double *copy_s, double *compute_s,
                              double *cleanup_s, double *worker_sum_s,
                              double *worker_span_s) {
    double init_sum = 0.0, copy_sum = 0.0, compute_sum = 0.0, cleanup_sum = 0.0, total_sum = 0.0;
    double dev_start = -1.0, dev_end = -1.0;
    for (int i = 0; i < 3; i++) {
        if (!w[i].enabled && w[i].err == 0 && w[i].rows_done <= 0) continue;
        init_sum += nonneg_or_zero(w[i].init_time);
        copy_sum += nonneg_or_zero(w[i].copy_time);
        compute_sum += nonneg_or_zero(w[i].gemm_time);
        cleanup_sum += nonneg_or_zero(w[i].cleanup_time);
        total_sum += nonneg_or_zero(w[i].total_time);
        merge_rel_span(&dev_start, &dev_end, w[i].t_dev_start_rel, w[i].t_dev_end_rel);
    }
    if (init_s) *init_s = init_sum;
    if (copy_s) *copy_s = copy_sum;
    if (compute_s) *compute_s = compute_sum;
    if (cleanup_s) *cleanup_s = cleanup_sum;
    if (worker_sum_s) *worker_sum_s = total_sum;
    if (worker_span_s) *worker_span_s = interval_or_zero(dev_start, dev_end);
}

void apply_gpu_timeline_worker(worker_t *w, const gpu_step_times_t *gt) {
    if (!w || !gt) return;
    w->t_init_start_abs = gt->init_start_abs;
    w->t_init_end_abs = gt->init_end_abs;
    w->t_copyin_start_abs = gt->copyin_start_abs;
    w->t_copyin_end_abs = gt->copyin_end_abs;
    w->t_gemm_start_abs = gt->gemm_start_abs;
    w->t_gemm_end_abs = gt->gemm_end_abs;
    w->t_copyout_start_abs = gt->copyout_start_abs;
    w->t_copyout_end_abs = gt->copyout_end_abs;
    w->t_cleanup_start_abs = gt->cleanup_start_abs;
    w->t_cleanup_end_abs = gt->cleanup_end_abs;
    w->t_init_start_rel = rel_or_missing(w->t_init_start_abs, w->t_ref);
    w->t_init_end_rel = rel_or_missing(w->t_init_end_abs, w->t_ref);
    w->t_copyin_start_rel = rel_or_missing(w->t_copyin_start_abs, w->t_ref);
    w->t_copyin_end_rel = rel_or_missing(w->t_copyin_end_abs, w->t_ref);
    w->t_gemm_start_rel = rel_or_missing(w->t_gemm_start_abs, w->t_ref);
    w->t_gemm_end_rel = rel_or_missing(w->t_gemm_end_abs, w->t_ref);
    w->t_copyout_start_rel = rel_or_missing(w->t_copyout_start_abs, w->t_ref);
    w->t_copyout_end_rel = rel_or_missing(w->t_copyout_end_abs, w->t_ref);
    w->t_cleanup_start_rel = rel_or_missing(w->t_cleanup_start_abs, w->t_ref);
    w->t_cleanup_end_rel = rel_or_missing(w->t_cleanup_end_abs, w->t_ref);
    if (w->t_gemm_start_abs > 0.0 && w->t_gemm_end_abs > w->t_gemm_start_abs) {
        w->t_compute_start_rel = w->t_gemm_start_rel;
        w->t_compute_end_rel = w->t_gemm_end_rel;
    }
    w->t_copy_start_abs = 0.0;
    w->t_copy_end_abs = 0.0;
    mark_time_span(&w->t_copy_start_abs, &w->t_copy_end_abs, w->t_copyin_start_abs, w->t_copyin_end_abs);
    mark_time_span(&w->t_copy_start_abs, &w->t_copy_end_abs, w->t_copyout_start_abs, w->t_copyout_end_abs);
    w->t_copy_start_rel = rel_or_missing(w->t_copy_start_abs, w->t_ref);
    w->t_copy_end_rel = rel_or_missing(w->t_copy_end_abs, w->t_ref);
}

void apply_gpu_timeline_panel_worker(panel_worker_t *w, const gpu_step_times_t *gt) {
    if (!w || !gt) return;
    w->t_init_start_abs = gt->init_start_abs;
    w->t_init_end_abs = gt->init_end_abs;
    w->t_copyin_start_abs = gt->copyin_start_abs;
    w->t_copyin_end_abs = gt->copyin_end_abs;
    w->t_gemm_start_abs = gt->gemm_start_abs;
    w->t_gemm_end_abs = gt->gemm_end_abs;
    w->t_copyout_start_abs = gt->copyout_start_abs;
    w->t_copyout_end_abs = gt->copyout_end_abs;
    w->t_cleanup_start_abs = gt->cleanup_start_abs;
    w->t_cleanup_end_abs = gt->cleanup_end_abs;
    w->t_init_start_rel = rel_or_missing(gt->init_start_abs, w->t_ref);
    w->t_init_end_rel = rel_or_missing(gt->init_end_abs, w->t_ref);
    w->t_copyin_start_rel = rel_or_missing(gt->copyin_start_abs, w->t_ref);
    w->t_copyin_end_rel = rel_or_missing(gt->copyin_end_abs, w->t_ref);
    w->t_gemm_start_rel = rel_or_missing(gt->gemm_start_abs, w->t_ref);
    w->t_gemm_end_rel = rel_or_missing(gt->gemm_end_abs, w->t_ref);
    w->t_copyout_start_rel = rel_or_missing(gt->copyout_start_abs, w->t_ref);
    w->t_copyout_end_rel = rel_or_missing(gt->copyout_end_abs, w->t_ref);
    w->t_cleanup_start_rel = rel_or_missing(gt->cleanup_start_abs, w->t_ref);
    w->t_cleanup_end_rel = rel_or_missing(gt->cleanup_end_abs, w->t_ref);
    if (gt->gemm_start_abs > 0.0 && gt->gemm_end_abs > gt->gemm_start_abs) {
        w->t_compute_start_rel = w->t_gemm_start_rel;
        w->t_compute_end_rel = w->t_gemm_end_rel;
    }
    w->t_copy_start_abs = 0.0;
    w->t_copy_end_abs = 0.0;
    mark_time_span(&w->t_copy_start_abs, &w->t_copy_end_abs, w->t_copyin_start_abs, w->t_copyin_end_abs);
    mark_time_span(&w->t_copy_start_abs, &w->t_copy_end_abs, w->t_copyout_start_abs, w->t_copyout_end_abs);
    w->t_copy_start_rel = rel_or_missing(w->t_copy_start_abs, w->t_ref);
    w->t_copy_end_rel = rel_or_missing(w->t_copy_end_abs, w->t_ref);
}

void finalize_panel_worker_times(panel_worker_t *w) {
    if (!w) return;
    if (!(w->t_init_start_abs > 0.0) && w->t_init_start_rel >= 0.0 && w->t_ref > 0.0) w->t_init_start_abs = w->t_ref + w->t_init_start_rel;
    if (!(w->t_init_end_abs > 0.0) && w->t_init_end_rel >= 0.0 && w->t_ref > 0.0) w->t_init_end_abs = w->t_ref + w->t_init_end_rel;
    if (!(w->t_gemm_start_abs > 0.0) && w->t_gemm_start_rel >= 0.0 && w->t_ref > 0.0) w->t_gemm_start_abs = w->t_ref + w->t_gemm_start_rel;
    if (!(w->t_gemm_end_abs > 0.0) && w->t_gemm_end_rel >= 0.0 && w->t_ref > 0.0) w->t_gemm_end_abs = w->t_ref + w->t_gemm_end_rel;
    if (!(w->t_copy_start_abs > 0.0) && w->t_copy_start_rel >= 0.0 && w->t_ref > 0.0) w->t_copy_start_abs = w->t_ref + w->t_copy_start_rel;
    if (!(w->t_copy_end_abs > 0.0) && w->t_copy_end_rel >= 0.0 && w->t_ref > 0.0) w->t_copy_end_abs = w->t_ref + w->t_copy_end_rel;
    if (!(w->t_copyin_start_abs > 0.0) && w->t_copyin_start_rel >= 0.0 && w->t_ref > 0.0) w->t_copyin_start_abs = w->t_ref + w->t_copyin_start_rel;
    if (!(w->t_copyin_end_abs > 0.0) && w->t_copyin_end_rel >= 0.0 && w->t_ref > 0.0) w->t_copyin_end_abs = w->t_ref + w->t_copyin_end_rel;
    if (!(w->t_copyout_start_abs > 0.0) && w->t_copyout_start_rel >= 0.0 && w->t_ref > 0.0) w->t_copyout_start_abs = w->t_ref + w->t_copyout_start_rel;
    if (!(w->t_copyout_end_abs > 0.0) && w->t_copyout_end_rel >= 0.0 && w->t_ref > 0.0) w->t_copyout_end_abs = w->t_ref + w->t_copyout_end_rel;
    if (!(w->t_cleanup_start_abs > 0.0) && w->t_cleanup_start_rel >= 0.0 && w->t_ref > 0.0) w->t_cleanup_start_abs = w->t_ref + w->t_cleanup_start_rel;
    if (!(w->t_cleanup_end_abs > 0.0) && w->t_cleanup_end_rel >= 0.0 && w->t_ref > 0.0) w->t_cleanup_end_abs = w->t_ref + w->t_cleanup_end_rel;
    w->t_dev_start_rel = -1.0;
    w->t_dev_end_rel = -1.0;
    if (w->enabled && ((w->r1 > w->r0) || (w->rows_done > 0) || (w->err != 0))) {
        double s = 1e300, e = -1e300;
        if (w->t_init_end_rel > w->t_init_start_rel) { if (w->t_init_start_rel < s) s = w->t_init_start_rel; if (w->t_init_end_rel > e) e = w->t_init_end_rel; }
        if (w->t_gemm_end_rel > w->t_gemm_start_rel) { if (w->t_gemm_start_rel < s) s = w->t_gemm_start_rel; if (w->t_gemm_end_rel > e) e = w->t_gemm_end_rel; }
        if (w->t_copy_end_rel > w->t_copy_start_rel) { if (w->t_copy_start_rel < s) s = w->t_copy_start_rel; if (w->t_copy_end_rel > e) e = w->t_copy_end_rel; }
        if (w->t_cleanup_end_rel > w->t_cleanup_start_rel) { if (w->t_cleanup_start_rel < s) s = w->t_cleanup_start_rel; if (w->t_cleanup_end_rel > e) e = w->t_cleanup_end_rel; }
        if (e > s && s < 1e299) {
            w->t_dev_start_rel = s;
            w->t_dev_end_rel = e;
        }
    }
}

void print_panel_dev_timing_line(const char *tag, int rank, int panel_idx, int j0, int jb, int rows_total, const panel_worker_t *w) {
    char i0_abs[64], i1_abs[64], g0_abs[64], g1_abs[64], cm0_abs[64], cm1_abs[64], c0_abs[64], c1_abs[64], ci0_abs[64], ci1_abs[64], co0_abs[64], co1_abs[64], x0_abs[64], x1_abs[64];
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
    make_abs_from_positive_rel_span(w->t_ref, w->t_init_start_rel, w->t_init_end_rel, i0_abs, sizeof(i0_abs), i1_abs, sizeof(i1_abs));
    make_abs_from_positive_rel_span(w->t_ref, w->t_gemm_start_rel, w->t_gemm_end_rel, g0_abs, sizeof(g0_abs), g1_abs, sizeof(g1_abs));
    make_abs_from_positive_rel_span(w->t_ref, w->t_compute_start_rel, w->t_compute_end_rel, cm0_abs, sizeof(cm0_abs), cm1_abs, sizeof(cm1_abs));
    make_abs_from_positive_rel_span(w->t_ref, w->t_copy_start_rel, w->t_copy_end_rel, c0_abs, sizeof(c0_abs), c1_abs, sizeof(c1_abs));
    make_abs_from_positive_rel_span(w->t_ref, w->t_copyin_start_rel, w->t_copyin_end_rel, ci0_abs, sizeof(ci0_abs), ci1_abs, sizeof(ci1_abs));
    make_abs_from_positive_rel_span(w->t_ref, w->t_copyout_start_rel, w->t_copyout_end_rel, co0_abs, sizeof(co0_abs), co1_abs, sizeof(co1_abs));
    make_abs_from_positive_rel_span(w->t_ref, w->t_cleanup_start_rel, w->t_cleanup_end_rel, x0_abs, sizeof(x0_abs), x1_abs, sizeof(x1_abs));
    const char *dev_name = (w->wid == 0) ? "cpu" : (w->wid == 1 ? "gpu0" : "gpu1");
    long long rows_assigned = (w->r1 > w->r0) ? (long long)(w->r1 - w->r0) : 0LL;
    double share_assigned = (rows_total > 0) ? (100.0 * (double)rows_assigned / (double)rows_total) : 0.0;
    printf("%s rank=%d panel_idx=%d j0=%d jb=%d dev=%s enabled=%d rows_total=%d rows_assigned=%lld rows_done=%lld share_assigned=%.6f r0=%d r1=%d status=%s err=%d init=%.6f gemm=%.6f copy=%.6f cleanup=%.6f total=%.6f copy_bytes=%llu copyin_bytes=%llu copyout_bytes=%llu dev_start_rel=%.9f dev_end_rel=%.9f init_start_abs=%s init_end_abs=%s compute_start_abs=%s compute_end_abs=%s copy_start_abs=%s copy_end_abs=%s copyin_start_abs=%s copyin_end_abs=%s copyout_start_abs=%s copyout_end_abs=%s cleanup_start_abs=%s cleanup_end_abs=%s init_start_rel=%.9f init_end_rel=%.9f compute_start_rel=%.9f compute_end_rel=%.9f gemm_start_rel=%.9f gemm_end_rel=%.9f copy_start_rel=%.9f copy_end_rel=%.9f copyin_start_rel=%.9f copyin_end_rel=%.9f copyout_start_rel=%.9f copyout_end_rel=%.9f cleanup_start_rel=%.9f cleanup_end_rel=%.9f\n",
           tag, rank, panel_idx, j0, jb, dev_name, w->enabled ? 1 : 0, rows_total, rows_assigned, w->rows_done, share_assigned,
           w->r0, w->r1, w->status, w->err,
           sane(w->init_time), sane(w->gemm_time), sane(w->copy_time), sane(w->cleanup_time), sane(w->total_time),
           w->copy_bytes, w->copyin_bytes, w->copyout_bytes,
           timeline_rel_value(w->t_dev_start_rel), timeline_rel_value(w->t_dev_end_rel),
           i0_abs, i1_abs, cm0_abs, cm1_abs, c0_abs, c1_abs, ci0_abs, ci1_abs, co0_abs, co1_abs, x0_abs, x1_abs,
           init_sr, init_er,
           compute_sr, compute_er,
           gemm_sr, gemm_er,
           copy_sr, copy_er,
           copyin_sr, copyin_er,
           copyout_sr, copyout_er,
           cleanup_sr, cleanup_er);
}

void *worker_main(void *arg) {
    worker_t *w = (worker_t*)arg;
    double t_all0 = tnow();
    w->err = 0;
    snprintf(w->status, sizeof(w->status), "OK");
    double saved_t_ref = w->t_ref;
    init_worker_timing_fields(w);
    w->t_ref = saved_t_ref;
    if (!w->enabled || w->r1 <= w->r0) {
        w->init_time = 0.0;
        w->gemm_time = 0.0;
        w->copy_time = 0.0;
        w->cleanup_time = 0.0;
        w->total_time = tnow() - t_all0;
        w->rows_done = 0;
        snprintf(w->status, sizeof(w->status), "SKIPPED");
        finalize_worker_perf(w);
        return NULL;
    }
    int N = w->N;
    int rows = w->r1 - w->r0;
    size_t esz = dtype_size(w->dt);
    void *Aptr = (char*)w->A + (size_t)w->r0 * (size_t)N * esz;
    void *Bptr = w->B;
    void *Cptr = (char*)w->C + (size_t)w->r0 * (size_t)N * esz;
    if (w->wid == 0) {
        w->t_init_start_abs = tnow();
        w->t_init_start_rel = w->t_init_start_abs - w->t_ref;
        w->t_init_end_abs = tnow();
        w->t_init_end_rel = w->t_init_end_abs - w->t_ref;
        w->init_time = w->t_init_end_abs - w->t_init_start_abs;
        w->t_gemm_start_abs = tnow();
        w->t_gemm_start_rel = w->t_gemm_start_abs - w->t_ref;
        if (w->cpu_blas) cpu_blas_real(N, rows, w->dt, Aptr, Bptr, Cptr);
        else {
            if (w->dt == DT_REAL_DOUBLE) cpu_naive_d(N, rows, (const double*)Aptr, (const double*)Bptr, (double*)Cptr);
            else cpu_naive_s(N, rows, (const float*)Aptr, (const float*)Bptr, (float*)Cptr);
        }
        w->t_gemm_end_abs = tnow();
        w->t_gemm_end_rel = w->t_gemm_end_abs - w->t_ref;
        w->t_compute_start_rel = w->t_gemm_start_rel;
        w->t_compute_end_rel = w->t_gemm_end_rel;
        w->gemm_time = w->t_gemm_end_abs - w->t_gemm_start_abs;
        w->copy_time = 0.0;
        w->t_cleanup_start_abs = tnow();
        w->t_cleanup_end_abs = w->t_cleanup_start_abs;
        w->t_cleanup_start_rel = w->t_cleanup_start_abs - w->t_ref;
        w->t_cleanup_end_rel = w->t_cleanup_end_abs - w->t_ref;
        w->cleanup_time = 0.0;
        w->rows_done = (long long)rows;
        w->total_time = tnow() - t_all0;
        finalize_worker_perf(w);
        return NULL;
    }
    gpu_step_times_t gt;
    reset_gpu_step_times(&gt);
    double gi = 0.0, gg = 0.0, gc = 0.0, gx = 0.0, gcin = 0.0, gcout = 0.0;
    char gst[64];
    const char *trace_prefix = w->trace_prefix[0] ? w->trace_prefix : "PTHREADS_TIMELINE_TILE_EVENT";
    const char *trace_mode = w->trace_mode[0] ? w->trace_mode : "pthreads";
    double trace_root_abs = (w->trace_root_abs > 0.0) ? w->trace_root_abs : w->t_ref;
    timeline_trace_t trace = {
        .enabled = w->trace_gpu_tiles,
        .prefix = trace_prefix,
        .mode = trace_mode,
        .rank = w->trace_rank,
        .dev = dev_name_lower_from_wid(w->wid),
        .panel_idx = w->trace_panel_idx,
        .panel_j0 = w->trace_panel_j0,
        .panel_jb = w->trace_panel_jb,
        .rows = rows,
        .root_abs = trace_root_abs,
        .clock_offset_to_root = w->trace_clock_offset_to_root
    };
    int rc = gpu_run_tiled(w->gpu_dev, N, rows, w->tile, w->gpu_blas, w->dt, Aptr, Bptr, Cptr,
                           &gi, &gg, &gc, &gx, &gcin, &gcout, gst, sizeof(gst), &gt, &trace);
    apply_gpu_timeline_worker(w, &gt);
    w->init_time = gi;
    w->gemm_time = gg;
    w->copy_time = gc;
    w->cleanup_time = gx;
    if (rc == 0) gpu_full_transfer_bytes(N, rows, w->tile, w->dt, &w->copyin_bytes, &w->copyout_bytes, &w->copy_bytes);
    if (rc != 0) {
        w->err = rc;
        snprintf(w->status, sizeof(w->status), "%s", gst);
        w->rows_done = 0;
        w->total_time = tnow() - t_all0;
        finalize_worker_perf(w);
        return NULL;
    }
    w->rows_done = (long long)rows;
    w->total_time = tnow() - t_all0;
    finalize_worker_perf(w);
    return NULL;
}


void init_empty_panel_workers(panel_worker_t out_w[3], const char *status, int err_code) {
    for (int i = 0; i < 3; i++) {
        memset(&out_w[i], 0, sizeof(out_w[i]));
        init_panel_worker_timing_fields(&out_w[i]);
        out_w[i].wid = i;
        out_w[i].gpu_dev = (i == 0) ? -1 : (i - 1);
        out_w[i].err = err_code;
        snprintf(out_w[i].status, sizeof(out_w[i].status), "%s", status);
    }
}

void merge_rel_span(double *dst_start, double *dst_end, double src_start, double src_end) {
    if (!(src_end > src_start) || !isfinite(src_start) || !isfinite(src_end)) return;
    if (!(*dst_end > *dst_start)) {
        *dst_start = src_start;
        *dst_end = src_end;
        return;
    }
    if (src_start < *dst_start) *dst_start = src_start;
    if (src_end > *dst_end) *dst_end = src_end;
}

void accumulate_panel_worker(worker_t *dst, const panel_worker_t *src, int N_full) {
    if (!src->enabled && src->err == 0) return;
    if (!dst->enabled) {
        dst->wid = src->wid;
        dst->enabled = src->enabled;
        dst->gpu_dev = src->gpu_dev;
        dst->N = N_full;
        dst->r0 = src->r0;
        dst->r1 = src->r1;
        dst->cpu_blas = src->cpu_blas;
        dst->gpu_blas = src->gpu_blas;
        dst->tile = src->tile;
        dst->dt = src->dt;
        dst->A = src->A;
        dst->B = src->B;
        dst->C = src->C;
        dst->t_ref = src->t_ref;
        dst->err = 0;
        snprintf(dst->status, sizeof(dst->status), "OK");
    }
    dst->enabled = dst->enabled || src->enabled;
    dst->r0 = src->r0;
    dst->r1 = src->r1;
    dst->init_time += src->init_time;
    dst->gemm_time += src->gemm_time;
    dst->copy_time += src->copy_time;
    dst->copy_bytes += src->copy_bytes;
    dst->copyin_bytes += src->copyin_bytes;
    dst->copyout_bytes += src->copyout_bytes;
    dst->cleanup_time += src->cleanup_time;
    dst->total_time += src->total_time;
    merge_rel_span(&dst->t_init_start_rel, &dst->t_init_end_rel, src->t_init_start_rel, src->t_init_end_rel);
    merge_rel_span(&dst->t_gemm_start_rel, &dst->t_gemm_end_rel, src->t_gemm_start_rel, src->t_gemm_end_rel);
    merge_rel_span(&dst->t_copy_start_rel, &dst->t_copy_end_rel, src->t_copy_start_rel, src->t_copy_end_rel);
    merge_rel_span(&dst->t_cleanup_start_rel, &dst->t_cleanup_end_rel, src->t_cleanup_start_rel, src->t_cleanup_end_rel);
    merge_rel_span(&dst->t_copyin_start_rel, &dst->t_copyin_end_rel, src->t_copyin_start_rel, src->t_copyin_end_rel);
    merge_rel_span(&dst->t_copyout_start_rel, &dst->t_copyout_end_rel, src->t_copyout_start_rel, src->t_copyout_end_rel);
    merge_rel_span(&dst->t_compute_start_rel, &dst->t_compute_end_rel, src->t_compute_start_rel, src->t_compute_end_rel);
    if (src->err != 0) {
        dst->err = src->err;
        snprintf(dst->status, sizeof(dst->status), "%s", src->status);
    }
}

void finalize_panel_worker_accum(worker_t *w) {
    if (!w->enabled || w->r1 <= w->r0 || w->err != 0) w->rows_done = 0;
    else w->rows_done = (long long)(w->r1 - w->r0);
    w->t_init_start_abs = (w->t_init_start_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_init_start_rel) : 0.0;
    w->t_init_end_abs = (w->t_init_end_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_init_end_rel) : 0.0;
    w->t_gemm_start_abs = (w->t_gemm_start_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_gemm_start_rel) : 0.0;
    w->t_gemm_end_abs = (w->t_gemm_end_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_gemm_end_rel) : 0.0;
    w->t_copy_start_abs = (w->t_copy_start_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_copy_start_rel) : 0.0;
    w->t_copy_end_abs = (w->t_copy_end_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_copy_end_rel) : 0.0;
    w->t_copyin_start_abs = (w->t_copyin_start_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_copyin_start_rel) : 0.0;
    w->t_copyin_end_abs = (w->t_copyin_end_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_copyin_end_rel) : 0.0;
    w->t_copyout_start_abs = (w->t_copyout_start_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_copyout_start_rel) : 0.0;
    w->t_copyout_end_abs = (w->t_copyout_end_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_copyout_end_rel) : 0.0;
    w->t_cleanup_start_abs = (w->t_cleanup_start_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_cleanup_start_rel) : 0.0;
    w->t_cleanup_end_abs = (w->t_cleanup_end_rel >= 0.0 && w->t_ref > 0.0) ? (w->t_ref + w->t_cleanup_end_rel) : 0.0;
    finalize_worker_perf(w);
    if (!w->enabled || w->r1 <= w->r0) snprintf(w->status, sizeof(w->status), "SKIPPED");
    else if (w->err == 0) snprintf(w->status, sizeof(w->status), "OK");
}

void *panel_worker_main(void *arg) {
    panel_worker_t *w = (panel_worker_t*)arg;
    double t_all0 = tnow();
    w->err = 0;
    snprintf(w->status, sizeof(w->status), "OK");
    double saved_t_ref = w->t_ref;
    init_panel_worker_timing_fields(w);
    w->t_ref = saved_t_ref;
    if (!w->enabled || w->r1 <= w->r0 || w->cols <= 0 || w->K <= 0) {
        w->init_time = 0.0;
        w->gemm_time = 0.0;
        w->copy_time = 0.0;
        w->cleanup_time = 0.0;
        w->total_time = tnow() - t_all0;
        w->rows_done = 0;
        snprintf(w->status, sizeof(w->status), "SKIPPED");
        finalize_panel_worker_times(w);
        return NULL;
    }
    int rows = w->r1 - w->r0;
    size_t esz = dtype_size(w->dt);
    void *Aptr = (char*)w->A + (size_t)w->r0 * (size_t)w->K * esz;
    void *Bptr = w->B;
    int c_ld = (w->c_ld > 0) ? w->c_ld : w->cols;
    int c_col0 = (w->c_col0 >= 0) ? w->c_col0 : 0;
    void *Cptr = (char*)w->C + (size_t)w->r0 * (size_t)c_ld * esz;
    if (w->wid == 0) {
        double ti0 = tnow();
        w->t_init_start_abs = ti0;
        w->t_init_start_rel = ti0 - w->t_ref;
        double ti1 = tnow();
        w->t_init_end_abs = ti1;
        w->t_init_end_rel = ti1 - w->t_ref;
        w->init_time = ti1 - ti0;
        double tg0 = tnow();
        w->t_gemm_start_abs = tg0;
        w->t_gemm_start_rel = tg0 - w->t_ref;
        if (c_ld == w->cols && c_col0 == 0) {
            if (w->cpu_blas) cpu_blas_real_panel(w->K, w->cols, rows, w->dt, Aptr, Bptr, Cptr);
            else {
                if (w->dt == DT_REAL_DOUBLE) cpu_naive_panel_d(w->K, w->cols, rows, (const double*)Aptr, (const double*)Bptr, (double*)Cptr);
                else cpu_naive_panel_s(w->K, w->cols, rows, (const float*)Aptr, (const float*)Bptr, (float*)Cptr);
            }
        } else {
            if (w->cpu_blas) {
                cpu_blas_real_panel_ldc(w->K, w->cols, rows, w->dt, Aptr, Bptr, Cptr, c_ld, c_col0);
            } else {
                if (w->dt == DT_REAL_DOUBLE) {
                    cpu_naive_panel_d_ldc(w->K, w->cols, rows, (const double*)Aptr, (const double*)Bptr, (double*)Cptr, c_ld, c_col0);
                } else {
                    cpu_naive_panel_s_ldc(w->K, w->cols, rows, (const float*)Aptr, (const float*)Bptr, (float*)Cptr, c_ld, c_col0);
                }
            }
        }
        double tg1 = tnow();
        w->t_gemm_end_abs = tg1;
        w->t_gemm_end_rel = tg1 - w->t_ref;
        w->t_compute_start_rel = w->t_gemm_start_rel;
        w->t_compute_end_rel = w->t_gemm_end_rel;
        w->gemm_time = tg1 - tg0;
        w->copy_time = 0.0;
        w->t_cleanup_start_abs = tnow();
        w->t_cleanup_end_abs = w->t_cleanup_start_abs;
        w->t_cleanup_start_rel = w->t_cleanup_start_abs - w->t_ref;
        w->t_cleanup_end_rel = w->t_cleanup_end_abs - w->t_ref;
        w->cleanup_time = 0.0;
        w->rows_done = (long long)rows;
        w->total_time = tnow() - t_all0;
        finalize_panel_worker_times(w);
        return NULL;
    }
    gpu_step_times_t gt;
    reset_gpu_step_times(&gt);
    double gi = 0.0, gg = 0.0, gc = 0.0, gx = 0.0, gcin = 0.0, gcout = 0.0;
    char gst[64];
    const char *trace_prefix = w->trace_prefix[0] ? w->trace_prefix : "PTHREADS_TIMELINE_TILE_EVENT";
    const char *trace_mode = w->trace_mode[0] ? w->trace_mode : "pthreads";
    double trace_root_abs = (w->trace_root_abs > 0.0) ? w->trace_root_abs : w->t_ref;
    timeline_trace_t trace = {
        .enabled = w->trace_gpu_tiles,
        .prefix = trace_prefix,
        .mode = trace_mode,
        .rank = w->trace_rank,
        .dev = dev_name_lower_from_wid(w->wid),
        .panel_idx = w->trace_panel_idx,
        .panel_j0 = w->trace_panel_j0,
        .panel_jb = w->trace_panel_jb,
        .rows = rows,
        .root_abs = trace_root_abs,
        .clock_offset_to_root = w->trace_clock_offset_to_root
    };
    int rc = (c_ld == w->cols && c_col0 == 0)
        ? gpu_run_panel_tiled(w->gpu_dev, w->K, w->cols, rows, w->tile, w->gpu_blas, w->dt, Aptr, Bptr, Cptr,
                              &gi, &gg, &gc, &gx, &gcin, &gcout, gst, sizeof(gst), &gt, &trace)
        : gpu_run_panel_tiled_strided(w->gpu_dev, w->K, w->cols, rows, w->tile, w->gpu_blas, w->dt, Aptr, Bptr, Cptr,
                                      c_ld, c_col0, &gi, &gg, &gc, &gx, &gcin, &gcout, gst, sizeof(gst), &gt, &trace);
    apply_gpu_timeline_panel_worker(w, &gt);
    w->init_time = gi;
    w->gemm_time = gg;
    w->copy_time = gc;
    w->cleanup_time = gx;
    if (rc == 0) gpu_panel_transfer_bytes(w->K, w->cols, rows, w->tile, w->dt, &w->copyin_bytes, &w->copyout_bytes, &w->copy_bytes);
    if (rc != 0) {
        w->err = rc;
        snprintf(w->status, sizeof(w->status), "%s", gst);
        w->rows_done = 0;
    } else {
        w->rows_done = (long long)rows;
    }
    w->total_time = tnow() - t_all0;
    finalize_panel_worker_times(w);
    return NULL;
}


