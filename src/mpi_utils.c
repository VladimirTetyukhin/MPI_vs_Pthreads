#include "hybrid_common.h"

/* MPI datatype helpers and large-count broadcast/scatter/gather utilities. */

int too_large_for_mpi_count_elems(unsigned long long elems) {
    return elems > (unsigned long long)INT_MAX;
}

void make_abs_from_root(double root_abs, double rel, char *out, size_t out_sz) {
    if (out_sz == 0) return;
    if (!(root_abs > 0.0) || !isfinite(root_abs) || !(rel >= 0.0) || !isfinite(rel)) {
        snprintf(out, out_sz, "NA");
        return;
    }
    fmt_ts_iso(root_abs + rel, out, out_sz);
}

void make_abs_from_positive_rel_span(double root_abs, double start_rel, double end_rel,
                                            char *start_out, size_t start_out_sz,
                                            char *end_out, size_t end_out_sz) {
    if (valid_positive_rel_span(start_rel, end_rel)) {
        make_abs_from_root(root_abs, start_rel, start_out, start_out_sz);
        make_abs_from_root(root_abs, end_rel, end_out, end_out_sz);
    } else {
        snprintf(start_out, start_out_sz, "NA");
        snprintf(end_out, end_out_sz, "NA");
    }
}

int mpi_dtype_setup(dtype_t dt, MPI_Datatype *mpi_dt) {
    if (dt == DT_REAL_DOUBLE) { *mpi_dt = MPI_DOUBLE; return 0; }
    if (dt == DT_REAL_FLOAT)  { *mpi_dt = MPI_FLOAT; return 0; }
    return -1;
}

int mpi_type_size_bytes(MPI_Datatype dt) {
    int sz = 0;
    MPI_Type_size(dt, &sz);
    return sz;
}

int mpi_bcast_large(void *buf, unsigned long long elems, MPI_Datatype dt, int root, MPI_Comm comm) {
    int tsz = mpi_type_size_bytes(dt);
    unsigned long long done = 0;
    char *p = (char*)buf;
    while (done < elems) {
        int chunk = (elems - done > (unsigned long long)INT_MAX) ? INT_MAX : (int)(elems - done);
        int rc = MPI_Bcast(p + (size_t)done * (size_t)tsz, chunk, dt, root, comm);
        if (rc != MPI_SUCCESS) return rc;
        done += (unsigned long long)chunk;
    }
    return MPI_SUCCESS;
}

int mpi_scatter_rows_large(const void *sendbuf, const int *counts_rows, const int *displs_rows,
                                  int N, MPI_Datatype dt, void *recvbuf, int rank, int root, MPI_Comm comm) {
    int tsz = mpi_type_size_bytes(dt);
    unsigned long long max_rows_chunk = (N > 0) ? ((unsigned long long)INT_MAX / (unsigned long long)N) : 1ULL;
    if (max_rows_chunk < 1ULL) max_rows_chunk = 1ULL;
    MPI_Status st;
    for (int r = 0; ; r++) {
        int world_size = 0;
        MPI_Comm_size(comm, &world_size);
        if (r >= world_size) break;
        int sent_rows = 0;
        while (sent_rows < counts_rows[r]) {
            int chunk_rows = counts_rows[r] - sent_rows;
            if ((unsigned long long)chunk_rows > max_rows_chunk) chunk_rows = (int)max_rows_chunk;
            int chunk_elems = chunk_rows * N;
            size_t byte_off = ((size_t)displs_rows[r] + (size_t)sent_rows) * (size_t)N * (size_t)tsz;
            if (rank == root) {
                const char *src = (const char*)sendbuf + byte_off;
                if (r == root) {
                    memcpy((char*)recvbuf + (size_t)sent_rows * (size_t)N * (size_t)tsz, src, (size_t)chunk_elems * (size_t)tsz);
                } else {
                    int rc = MPI_Send(src, chunk_elems, dt, r, 700, comm);
                    if (rc != MPI_SUCCESS) return rc;
                }
            } else if (rank == r) {
                int rc = MPI_Recv((char*)recvbuf + (size_t)sent_rows * (size_t)N * (size_t)tsz, chunk_elems, dt, root, 700, comm, &st);
                if (rc != MPI_SUCCESS) return rc;
            }
            sent_rows += chunk_rows;
        }
    }
    return MPI_SUCCESS;
}

int mpi_gather_rows_large(const void *sendbuf, int send_rows, int N, MPI_Datatype dt,
                                 void *recvbuf, const int *counts_rows, const int *displs_rows,
                                 int rank, int root, MPI_Comm comm) {
    int tsz = mpi_type_size_bytes(dt);
    unsigned long long max_rows_chunk = (N > 0) ? ((unsigned long long)INT_MAX / (unsigned long long)N) : 1ULL;
    if (max_rows_chunk < 1ULL) max_rows_chunk = 1ULL;
    MPI_Status st;
    for (int r = 0; ; r++) {
        int world_size = 0;
        MPI_Comm_size(comm, &world_size);
        if (r >= world_size) break;
        int recv_rows = counts_rows[r];
        int done_rows = 0;
        while (done_rows < recv_rows) {
            int chunk_rows = recv_rows - done_rows;
            if ((unsigned long long)chunk_rows > max_rows_chunk) chunk_rows = (int)max_rows_chunk;
            int chunk_elems = chunk_rows * N;
            size_t dst_off = ((size_t)displs_rows[r] + (size_t)done_rows) * (size_t)N * (size_t)tsz;
            if (rank == root) {
                char *dst = (char*)recvbuf + dst_off;
                if (r == root) {
                    memcpy(dst, (const char*)sendbuf + (size_t)done_rows * (size_t)N * (size_t)tsz, (size_t)chunk_elems * (size_t)tsz);
                } else {
                    int rc = MPI_Recv(dst, chunk_elems, dt, r, 701, comm, &st);
                    if (rc != MPI_SUCCESS) return rc;
                }
            } else if (rank == r) {
                int rc = MPI_Send((const char*)sendbuf + (size_t)done_rows * (size_t)N * (size_t)tsz, chunk_elems, dt, root, 701, comm);
                if (rc != MPI_SUCCESS) return rc;
            }
            done_rows += chunk_rows;
        }
    }
    (void)send_rows;
    return MPI_SUCCESS;
}


