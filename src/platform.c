#include "hybrid_common.h"

/* Wall-clock helpers, formatting, cache/memory reporting, dtype utilities, aligned allocation, and CPU thread setup. */

pthread_once_t g_tnow_once = PTHREAD_ONCE_INIT;
double g_tnow_wall0 = 0.0;
double g_tnow_mono0 = 0.0;

double clock_sec(clockid_t clk) {
    struct timespec ts;
    if (clock_gettime(clk, &ts) == 0) {
        return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
    }
    struct timeval tv;
    gettimeofday(&tv, 0);
    return (double)tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

void init_tnow_base(void) {
    g_tnow_wall0 = clock_sec(CLOCK_REALTIME);
    g_tnow_mono0 = clock_sec(CLOCK_MONOTONIC);
}

double tnow(void) {
    pthread_once(&g_tnow_once, init_tnow_base);
    return g_tnow_wall0 + (clock_sec(CLOCK_MONOTONIC) - g_tnow_mono0);
}

double sane(double x) {
    return isfinite(x) ? x : 0.0;
}

void fmt_ts_iso(double t, char *out, size_t out_sz) {
    if (out_sz == 0) return;
    if (!(t > 0.0)) { snprintf(out, out_sz, "NA"); return; }
    time_t sec = (time_t)t;
    double frac = t - (double)sec;
    if (frac < 0.0) frac = 0.0;
    int usec = (int)llround(frac * 1000000.0);
    if (usec >= 1000000) { sec += 1; usec -= 1000000; }
    struct tm tmv;
    localtime_r(&sec, &tmv);
    snprintf(out, out_sz, "%04d-%02d-%02dT%02d:%02d:%02d.%06d",
             tmv.tm_year + 1900, tmv.tm_mon + 1, tmv.tm_mday,
             tmv.tm_hour, tmv.tm_min, tmv.tm_sec, usec);
}

double bytes_to_mib(unsigned long long b) {
    return (double)b / (1024.0 * 1024.0);
}

unsigned long long bytes_mat_NN_elems(int N, size_t elem_bytes) {
    return (unsigned long long)N * (unsigned long long)N * (unsigned long long)elem_bytes;
}

double cache_N_limit(unsigned long long cache_bytes, size_t elem_bytes) {
    double x = (double)cache_bytes / (double)elem_bytes;
    if (x <= 0.0) return 0.0;
    return sqrt(x);
}

const char* cache_fit_label_for_B(int N, size_t elem_bytes) {
    unsigned long long bB = bytes_mat_NN_elems(N, elem_bytes);
    if (bB <= L1D_PER_CORE_B) return "L1D_fit";
    if (bB <= L2_PER_CORE_B)  return "L2_fit";
    if (bB <= L3_SHARED_B)    return "L3_fit";
    return "DRAM";
}

void print_cache_speed_estimates(void) {
    double ghz = CPU_MAX_GHZ;
    double l1_gbs = ghz * 64.0;
    double l2_gbs = ghz * 32.0;
    double l3_gbs = ghz * 16.0;
    double dram_gbs = ghz * 8.0;
    printf("CACHE_SPEED_EST rough@%.1fGHz L1~%.1fGB/s L2~%.1fGB/s L3~%.1fGB/s DRAM~%.1fGB/s\n", ghz, l1_gbs, l2_gbs, l3_gbs, dram_gbs);
}

void print_cache_fit_report(int N, size_t elem_bytes) {
    unsigned long long bA = bytes_mat_NN_elems(N, elem_bytes);
    unsigned long long bB = bytes_mat_NN_elems(N, elem_bytes);
    unsigned long long bC = bytes_mat_NN_elems(N, elem_bytes);
    unsigned long long bABC = bA + bB + bC;
    double nL1 = cache_N_limit(L1D_PER_CORE_B, elem_bytes);
    double nL2 = cache_N_limit(L2_PER_CORE_B, elem_bytes);
    double nL3 = cache_N_limit(L3_SHARED_B, elem_bytes);
    printf("MEM_FOOTPRINT A=%.2fMiB B=%.2fMiB C=%.2fMiB total=%.2fMiB elemB=%zu\n",
           bytes_to_mib(bA), bytes_to_mib(bB), bytes_to_mib(bC), bytes_to_mib(bABC), elem_bytes);
    printf("CACHE_LIMITS_FOR_B L1D_per_core=%lluB(N<=%.0f) L2_per_core=%lluB(N<=%.0f) L3_shared=%lluB(N<=%.0f)\n",
           (unsigned long long)L1D_PER_CORE_B, nL1,
           (unsigned long long)L2_PER_CORE_B, nL2,
           (unsigned long long)L3_SHARED_B, nL3);
    printf("CACHE_FIT_B B_bytes=%llu(%.2fMiB) => %s\n",
           bB, bytes_to_mib(bB), cache_fit_label_for_B(N, elem_bytes));
}

void configure_cpu_threads(int cpu_threads) {
    if (cpu_threads <= 0) cpu_threads = 1;
    omp_set_dynamic(0);
    omp_set_num_threads(cpu_threads);
    char buf[32];
    snprintf(buf, sizeof(buf), "%d", cpu_threads);
    setenv("OMP_NUM_THREADS", buf, 1);
    setenv("OPENBLAS_NUM_THREADS", buf, 1);
    setenv("MKL_NUM_THREADS", buf, 1);
    setenv("BLIS_NUM_THREADS", buf, 1);
    setenv("VECLIB_MAXIMUM_THREADS", buf, 1);
    setenv("OMP_PROC_BIND", "close", 0);
    setenv("OMP_PLACES", "cores", 0);
}

const char* dtype_name(dtype_t dt) {
    return dt == DT_REAL_DOUBLE ? "double" : "float";
}

size_t dtype_size(dtype_t dt) {
    return dt == DT_REAL_DOUBLE ? sizeof(double) : sizeof(float);
}

unsigned long long mat_bytes_rows(int rows, int N, size_t elem_bytes) {
    return (unsigned long long)rows * (unsigned long long)N * (unsigned long long)elem_bytes;
}

unsigned long long mat_bytes_NN(int N, size_t elem_bytes) {
    return (unsigned long long)N * (unsigned long long)N * (unsigned long long)elem_bytes;
}

unsigned long long mat_elems_rows(int rows, int N) {
    return (unsigned long long)rows * (unsigned long long)N;
}

unsigned long long mat_elems_NN(int N) {
    return (unsigned long long)N * (unsigned long long)N;
}

void *malloc_aligned64(size_t sz) {
    void *p = NULL;
    if (sz == 0) sz = 1;
#if defined(_POSIX_C_SOURCE) && (_POSIX_C_SOURCE >= 200112L)
    if (posix_memalign(&p, 64, sz) != 0) p = NULL;
#else
    p = NULL;
#endif
    if (!p) p = malloc(sz);
    return p;
}

int choose_cpu_block(dtype_t dt, int N) {
    const char *env = getenv("CPU_TILE");
    if (env && *env) {
        char *endp = NULL;
        long v = strtol(env, &endp, 10);
        if (endp && endp != env && v > 0) {
            if (v > N) v = N;
            if (v < 8) v = 8;
            return (int)v;
        }
    }
    if (dt == DT_REAL_DOUBLE) {
        if (N >= 32768) return 64;
        if (N >= 16384) return 96;
        return 128;
    }
    if (N >= 32768) return 96;
    if (N >= 16384) return 128;
    return 192;
}


