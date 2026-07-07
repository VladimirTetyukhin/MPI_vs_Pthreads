#include "hybrid_common.h"

/* CPU kernels, BLAS dispatch, checksums, and matrix panel packing helpers. */

#define CHECKSUM_SAMPLE_MAX 1048576LL

long long checksum_sample_elems(long long n) {
    return n < CHECKSUM_SAMPLE_MAX ? n : CHECKSUM_SAMPLE_MAX;
}

double checksum_real_double(const double *C, long long n) {
    double s = 0.0;
    long long m = checksum_sample_elems(n);
    for (long long i = 0; i < m; i++) s += C[i];
    return s;
}

double checksum_real_float(const float *C, long long n) {
    double s = 0.0;
    long long m = checksum_sample_elems(n);
    for (long long i = 0; i < m; i++) s += (double)C[i];
    return s;
}

void cpu_naive_d(int N, int rows, const double *Ablk, const double *B, double *Cblk) {
    if (rows <= 0) return;
    int JB = choose_cpu_block(DT_REAL_DOUBLE, N);
    int KB = JB;
    memset(Cblk, 0, (size_t)rows * (size_t)N * sizeof(double));
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < rows; ii++) {
        const double *Ai = &Ablk[(size_t)ii * N];
        double *Ci = &Cblk[(size_t)ii * N];
        for (int kk = 0; kk < N; kk += KB) {
            int kend = (kk + KB <= N) ? (kk + KB) : N;
            for (int jj = 0; jj < N; jj += JB) {
                int jend = (jj + JB <= N) ? (jj + JB) : N;
                for (int k = kk; k < kend; k++) {
                    double a = Ai[k];
                    const double *Bk = &B[(size_t)k * N + jj];
                    double *Cj = &Ci[jj];
                    int width = jend - jj;
                    int j = 0;
                    for (; j + 3 < width; j += 4) {
                        Cj[j + 0] += a * Bk[j + 0];
                        Cj[j + 1] += a * Bk[j + 1];
                        Cj[j + 2] += a * Bk[j + 2];
                        Cj[j + 3] += a * Bk[j + 3];
                    }
                    for (; j < width; j++) Cj[j] += a * Bk[j];
                }
            }
        }
    }
}

void cpu_naive_s(int N, int rows, const float *Ablk, const float *B, float *Cblk) {
    if (rows <= 0) return;
    int JB = choose_cpu_block(DT_REAL_FLOAT, N);
    int KB = JB;
    memset(Cblk, 0, (size_t)rows * (size_t)N * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < rows; ii++) {
        const float *Ai = &Ablk[(size_t)ii * N];
        float *Ci = &Cblk[(size_t)ii * N];
        for (int kk = 0; kk < N; kk += KB) {
            int kend = (kk + KB <= N) ? (kk + KB) : N;
            for (int jj = 0; jj < N; jj += JB) {
                int jend = (jj + JB <= N) ? (jj + JB) : N;
                for (int k = kk; k < kend; k++) {
                    float a = Ai[k];
                    const float *Bk = &B[(size_t)k * N + jj];
                    float *Cj = &Ci[jj];
                    int width = jend - jj;
                    int j = 0;
                    for (; j + 3 < width; j += 4) {
                        Cj[j + 0] += a * Bk[j + 0];
                        Cj[j + 1] += a * Bk[j + 1];
                        Cj[j + 2] += a * Bk[j + 2];
                        Cj[j + 3] += a * Bk[j + 3];
                    }
                    for (; j < width; j++) Cj[j] += a * Bk[j];
                }
            }
        }
    }
}

void cpu_blas_real(int N, int rows, dtype_t dt, const void *Ablk, const void *B, void *Cblk) {
#ifdef USE_CBLAS
    if (dt == DT_REAL_DOUBLE) {
        const double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, N, N, alpha, (const double*)Ablk, N, (const double*)B, N, beta, (double*)Cblk, N);
        return;
    }
    if (dt == DT_REAL_FLOAT) {
        const float alpha = 1.0f, beta = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, N, N, alpha, (const float*)Ablk, N, (const float*)B, N, beta, (float*)Cblk, N);
        return;
    }
#endif
    if (dt == DT_REAL_DOUBLE) cpu_naive_d(N, rows, (const double*)Ablk, (const double*)B, (double*)Cblk);
    else cpu_naive_s(N, rows, (const float*)Ablk, (const float*)B, (float*)Cblk);
}

void gpu_openacc_panel_d(int rows, int jb, int kb, int accumulate, const double *A, const double *B, double *C) {
#pragma acc parallel loop collapse(2) deviceptr(A,B,C)
    for (int ii = 0; ii < rows; ii++) {
        for (int j = 0; j < jb; j++) {
            double s = 0.0;
#pragma acc loop seq
            for (int k = 0; k < kb; k++) s += A[(size_t)ii * kb + k] * B[(size_t)k * jb + j];
            if (accumulate) C[(size_t)ii * jb + j] += s;
            else            C[(size_t)ii * jb + j]  = s;
        }
    }
    acc_wait_all();
}

void gpu_openacc_panel_s(int rows, int jb, int kb, int accumulate, const float *A, const float *B, float *C) {
#pragma acc parallel loop collapse(2) deviceptr(A,B,C)
    for (int ii = 0; ii < rows; ii++) {
        for (int j = 0; j < jb; j++) {
            float s = 0.0f;
#pragma acc loop seq
            for (int k = 0; k < kb; k++) s += A[(size_t)ii * kb + k] * B[(size_t)k * jb + j];
            if (accumulate) C[(size_t)ii * jb + j] += s;
            else            C[(size_t)ii * jb + j]  = s;
        }
    }
    acc_wait_all();
}

#ifdef USE_CUBLAS
int cublas_gemm_real_panel(cublasHandle_t h, dtype_t dt, int rows, int jb, int kb, int accumulate, const void *Ablk, const void *B, void *Cblk) {
    if (dt == DT_REAL_DOUBLE) {
        const double alpha = 1.0, beta = accumulate ? 1.0 : 0.0;
        cublasStatus_t st = cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, jb, rows, kb, &alpha, (const double*)B, jb, (const double*)Ablk, kb, &beta, (double*)Cblk, jb);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : -10;
    }
    if (dt == DT_REAL_FLOAT) {
        const float alpha = 1.0f, beta = accumulate ? 1.0f : 0.0f;
        cublasStatus_t st = cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, jb, rows, kb, &alpha, (const float*)B, jb, (const float*)Ablk, kb, &beta, (float*)Cblk, jb);
        return st == CUBLAS_STATUS_SUCCESS ? 0 : -11;
    }
    return -12;
}
#endif

void pack_A_panel_bytes(const char *src, int rows, int N, int k0, int kb, size_t esz, char *dst) {
    for (int ii = 0; ii < rows; ii++) {
        memcpy(dst + (size_t)ii * kb * esz,
               src + (((size_t)ii * N + (size_t)k0) * esz),
               (size_t)kb * esz);
    }
}

void pack_B_panel_bytes(const char *src, int N, int k0, int kb, int j0, int jb, size_t esz, char *dst) {
    for (int k = 0; k < kb; k++) {
        memcpy(dst + (size_t)k * jb * esz,
               src + ((((size_t)(k0 + k) * N) + (size_t)j0) * esz),
               (size_t)jb * esz);
    }
}

void unpack_C_panel_bytes(const char *src, int rows, int N, int j0, int jb, size_t esz, char *dst) {
    for (int ii = 0; ii < rows; ii++) {
        memcpy(dst + ((((size_t)ii * N) + (size_t)j0) * esz),
               src + (size_t)ii * jb * esz,
               (size_t)jb * esz);
    }
}

void cpu_naive_panel_d_ldc(int K, int cols, int rows, const double *Ablk, const double *Bpanel, double *Cblk, int ldc, int c_col0) {
    if (rows <= 0 || cols <= 0 || K <= 0) return;
    int JB = choose_cpu_block(DT_REAL_DOUBLE, cols > 0 ? cols : K);
    if (JB < 1) JB = 1;
    for (int ii = 0; ii < rows; ii++) memset(&Cblk[(size_t)ii * (size_t)ldc + (size_t)c_col0], 0, (size_t)cols * sizeof(double));
    for (int ii = 0; ii < rows; ii++) {
        const double *Ai = &Ablk[(size_t)ii * (size_t)K];
        double *Ci = &Cblk[(size_t)ii * (size_t)ldc + (size_t)c_col0];
        for (int k = 0; k < K; k++) {
            double a = Ai[k];
            const double *Bk = &Bpanel[(size_t)k * (size_t)cols];
            for (int jj = 0; jj < cols; jj += JB) {
                int jend = (jj + JB <= cols) ? (jj + JB) : cols;
                for (int j = jj; j < jend; j++) Ci[j] += a * Bk[j];
            }
        }
    }
}

void cpu_naive_panel_s_ldc(int K, int cols, int rows, const float *Ablk, const float *Bpanel, float *Cblk, int ldc, int c_col0) {
    if (rows <= 0 || cols <= 0 || K <= 0) return;
    int JB = choose_cpu_block(DT_REAL_FLOAT, cols > 0 ? cols : K);
    if (JB < 1) JB = 1;
    for (int ii = 0; ii < rows; ii++) memset(&Cblk[(size_t)ii * (size_t)ldc + (size_t)c_col0], 0, (size_t)cols * sizeof(float));
    for (int ii = 0; ii < rows; ii++) {
        const float *Ai = &Ablk[(size_t)ii * (size_t)K];
        float *Ci = &Cblk[(size_t)ii * (size_t)ldc + (size_t)c_col0];
        for (int k = 0; k < K; k++) {
            float a = Ai[k];
            const float *Bk = &Bpanel[(size_t)k * (size_t)cols];
            for (int jj = 0; jj < cols; jj += JB) {
                int jend = (jj + JB <= cols) ? (jj + JB) : cols;
                for (int j = jj; j < jend; j++) Ci[j] += a * Bk[j];
            }
        }
    }
}

void cpu_blas_real_panel_ldc(int K, int cols, int rows, dtype_t dt, const void *Ablk, const void *Bpanel, void *Cblk, int ldc, int c_col0) {
#ifdef USE_CBLAS
    if (dt == DT_REAL_DOUBLE) {
        const double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, K, alpha,
                    (const double*)Ablk, K, (const double*)Bpanel, cols, beta,
                    (double*)Cblk + (size_t)c_col0, ldc);
        return;
    }
    if (dt == DT_REAL_FLOAT) {
        const float alpha = 1.0f, beta = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, K, alpha,
                    (const float*)Ablk, K, (const float*)Bpanel, cols, beta,
                    (float*)Cblk + (size_t)c_col0, ldc);
        return;
    }
#endif
    if (dt == DT_REAL_DOUBLE) cpu_naive_panel_d_ldc(K, cols, rows, (const double*)Ablk, (const double*)Bpanel, (double*)Cblk, ldc, c_col0);
    else cpu_naive_panel_s_ldc(K, cols, rows, (const float*)Ablk, (const float*)Bpanel, (float*)Cblk, ldc, c_col0);
}


void cpu_naive_panel_d(int K, int cols, int rows, const double *Ablk, const double *Bpanel, double *Cblk) {
    if (rows <= 0 || cols <= 0 || K <= 0) return;
    int JB = choose_cpu_block(DT_REAL_DOUBLE, cols > 0 ? cols : K);
    int KB = choose_cpu_block(DT_REAL_DOUBLE, K);
    if (JB < 1) JB = 1;
    if (KB < 1) KB = 1;
    memset(Cblk, 0, (size_t)rows * (size_t)cols * sizeof(double));
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < rows; ii++) {
        const double *Ai = &Ablk[(size_t)ii * (size_t)K];
        double *Ci = &Cblk[(size_t)ii * (size_t)cols];
        for (int kk = 0; kk < K; kk += KB) {
            int kend = (kk + KB <= K) ? (kk + KB) : K;
            for (int jj = 0; jj < cols; jj += JB) {
                int jend = (jj + JB <= cols) ? (jj + JB) : cols;
                for (int k = kk; k < kend; k++) {
                    double a = Ai[k];
                    const double *Bk = &Bpanel[(size_t)k * (size_t)cols + (size_t)jj];
                    double *Cj = &Ci[jj];
                    int width = jend - jj;
                    int j = 0;
                    for (; j + 3 < width; j += 4) {
                        Cj[j + 0] += a * Bk[j + 0];
                        Cj[j + 1] += a * Bk[j + 1];
                        Cj[j + 2] += a * Bk[j + 2];
                        Cj[j + 3] += a * Bk[j + 3];
                    }
                    for (; j < width; j++) Cj[j] += a * Bk[j];
                }
            }
        }
    }
}

void cpu_naive_panel_s(int K, int cols, int rows, const float *Ablk, const float *Bpanel, float *Cblk) {
    if (rows <= 0 || cols <= 0 || K <= 0) return;
    int JB = choose_cpu_block(DT_REAL_FLOAT, cols > 0 ? cols : K);
    int KB = choose_cpu_block(DT_REAL_FLOAT, K);
    if (JB < 1) JB = 1;
    if (KB < 1) KB = 1;
    memset(Cblk, 0, (size_t)rows * (size_t)cols * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int ii = 0; ii < rows; ii++) {
        const float *Ai = &Ablk[(size_t)ii * (size_t)K];
        float *Ci = &Cblk[(size_t)ii * (size_t)cols];
        for (int kk = 0; kk < K; kk += KB) {
            int kend = (kk + KB <= K) ? (kk + KB) : K;
            for (int jj = 0; jj < cols; jj += JB) {
                int jend = (jj + JB <= cols) ? (jj + JB) : cols;
                for (int k = kk; k < kend; k++) {
                    float a = Ai[k];
                    const float *Bk = &Bpanel[(size_t)k * (size_t)cols + (size_t)jj];
                    float *Cj = &Ci[jj];
                    int width = jend - jj;
                    int j = 0;
                    for (; j + 3 < width; j += 4) {
                        Cj[j + 0] += a * Bk[j + 0];
                        Cj[j + 1] += a * Bk[j + 1];
                        Cj[j + 2] += a * Bk[j + 2];
                        Cj[j + 3] += a * Bk[j + 3];
                    }
                    for (; j < width; j++) Cj[j] += a * Bk[j];
                }
            }
        }
    }
}

void cpu_blas_real_panel(int K, int cols, int rows, dtype_t dt, const void *Ablk, const void *Bpanel, void *Cblk) {
#ifdef USE_CBLAS
    if (dt == DT_REAL_DOUBLE) {
        const double alpha = 1.0, beta = 0.0;
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, K, alpha, (const double*)Ablk, K, (const double*)Bpanel, cols, beta, (double*)Cblk, cols);
        return;
    }
    if (dt == DT_REAL_FLOAT) {
        const float alpha = 1.0f, beta = 0.0f;
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows, cols, K, alpha, (const float*)Ablk, K, (const float*)Bpanel, cols, beta, (float*)Cblk, cols);
        return;
    }
#endif
    if (dt == DT_REAL_DOUBLE) cpu_naive_panel_d(K, cols, rows, (const double*)Ablk, (const double*)Bpanel, (double*)Cblk);
    else cpu_naive_panel_s(K, cols, rows, (const float*)Ablk, (const float*)Bpanel, (float*)Cblk);
}

void pack_B_rect_panel_bytes(const char *src, int ld_src, int k0, int kb, int j0, int jb, size_t esz, char *dst) {
    for (int k = 0; k < kb; k++) {
        memcpy(dst + (size_t)k * (size_t)jb * esz,
               src + ((((size_t)(k0 + k) * (size_t)ld_src) + (size_t)j0) * esz),
               (size_t)jb * esz);
    }
}

