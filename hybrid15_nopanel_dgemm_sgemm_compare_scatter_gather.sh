#!/bin/bash
set -euo pipefail

# HYBRID15 C-compatible benchmark matrix
# Tile behavior is hardwired: argv[9] tile=N for each matrix size.
# Use with this refactored HYBRID15 project or a compatible monolithic HYBRID15 source for:
#   - pthreads+GPU full tile=N with no panels
#   - MPI no-panel full-B with real rank-0 A_full/C_full scatter/gather
# Runs exactly these no-panel comparisons:
#   1) pthreads, DGEMM/CUBLAS, no panels
#   2) MPI, DGEMM/CUBLAS MPI GPU ranks, no panels
#   3) pthreads, SGEMM/CUBLAS, no panels
#   4) MPI, SGEMM/CUBLAS MPI GPU ranks, no panels
# Panel and naive GPU runs are intentionally disabled.
# MPI no-panel still uses scatter/gather in the compatible C source/refactored project.
# Sizes: 10000..80000 step 10000.
#
# C argv contract:
#   argv[1]  mode: 0=pthreads, 1=mpi3proc
#   argv[2]  N
#   argv[3]  use_cpu
#   argv[4]  use_gpu0
#   argv[5]  use_gpu1
#   argv[6]  cpu_blas
#   argv[7]  gpu_blas
#   argv[8]  cpu_threads
#   argv[9]  tile
#   argv[10] prec: 0=double, 1=float
#   argv[11] variant
#   argv[12] config_name

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

EXE="${EXE:-hybrid15}"
OBJ="${OBJ:-hybrid15.o}"

LOG="${LOG:-matmul_results15.log}"
CSV="${CSV:-results15.csv}"
PANEL_CSV="${PANEL_CSV:-results15_panels.csv}"
PANEL_DEV_CSV="${PANEL_DEV_CSV:-results15_panel_devices.csv}"
TIMELINE_CSV="${TIMELINE_CSV:-results15_timeline_events.csv}"
TILE_TIMELINE_CSV="${TILE_TIMELINE_CSV:-results15_timeline_tile_events.csv}"
RAW_KV_CSV="${RAW_KV_CSV:-results15_raw_kv.csv}"

TMPDIR="${TMPDIR:-.hybrid15_tmp}"

MPI_INC="${MPI_INC:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/openmpi4/openmpi-4.1.5/include}"
MPI_LIB="${MPI_LIB:-/opt/nvidia/hpc_sdk/Linux_x86_64/25.1/comm_libs/12.6/openmpi4/openmpi-4.1.5/lib}"
BLAS_LIBS="${BLAS_LIBS:--lopenblas}"

CC="${CC:-nvc}"
MPIRUN="${MPIRUN:-mpirun}"
MPIRUN_EXTRA="${MPIRUN_EXTRA:---bind-to none}"

CPU_THREADS="${CPU_THREADS:-1}"
# Tile is always the current matrix size N. This intentionally disables fixed 2048 tiling.
MPI_NP="${MPI_NP:-3}"
TIMEOUT_S="${TIMEOUT_S:-0}"
REPEATS="${REPEATS:-1}"   # repetitions per run spec; default 1 keeps the intended 4-run no-panel matrix per size

SIZE_START="${SIZE_START:-10000}"
SIZE_END="${SIZE_END:-80000}"
SIZE_STEP="${SIZE_STEP:-10000}"

# Set KEEP_EXE=1 if you want to reuse an existing executable.
KEEP_EXE="${KEEP_EXE:-0}"

resolve_src() {
  # Supports either the refactored multi-file project root or the original
  # monolithic .c source.  Set SRC explicitly to override auto-detection.
  local candidates=()
  if [[ -n "${SRC:-}" ]]; then
    candidates+=("$SRC")
  fi
  candidates+=(
    "$SCRIPT_DIR"
    "."
    "hybrid15_nopanel_mpi_scatter_gather.c"
    "hybrid15_timeline_n4_rate_partition_blas_forced.c"
    "hybrid15_timeline_full_nodup_memfix.c"
    "hybrid15_timeline_full_no_256_full_compute.c"
    "hybrid15_timeline_full_patched.c"
    "hybrid15_pthreads_panel_toggle_real_memlimit_nodup_memfix.c"
    "hybrid15_pthreads_panel_toggle_real_memlimit_nodup_fixed.c"
    "hybrid15_pthreads_panel_toggle_real_memlimit_nodup.c"
    "hybrid15_pthreads_panel_toggle_real_memlimit.c"
    "hybrid15_pthreads_panel_toggle_real_final.c"
    "hybrid15_pthreads_panel_toggle_real_final (3).c"
    "hybrid15_pthreads_panel_toggle_real_final (2).c"
    "hybrid15_pthreads_panel_toggle_real_final_fixed.c"
    "hybrid15_pthreads_panel_toggle_real_updated.c"
    "hybrid15_pthreads_panel_toggle_real.c"
    "hybrid15_panel_toggle.c"
    "hybrid15_pthreads_panel_timing_real.c"
    "hybrid15.c"
    "hybrid15 (5).c"
  )

  local c
  for c in "${candidates[@]}"; do
    if [[ -d "$c" && -f "$c/include/hybrid_common.h" && -f "$c/src/main.c" ]]; then
      (cd "$c" && pwd)
      return 0
    fi
    if [[ -f "$c" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
    if [[ -f "$SCRIPT_DIR/$c" ]]; then
      printf '%s\n' "$SCRIPT_DIR/$c"
      return 0
    fi
  done

  shopt -s nullglob
  local globbed=(
    "$SCRIPT_DIR"/hybrid15_timeline_full_no_256_full_compute.c
    "$SCRIPT_DIR"/hybrid15_timeline_full_patched.c
    "$SCRIPT_DIR"/hybrid15*.c
  )
  shopt -u nullglob
  for c in "${globbed[@]}"; do
    if [[ -f "$c" ]]; then
      printf '%s\n' "$c"
      return 0
    fi
  done

  echo "ERROR: no compatible HYBRID15 source found. Set SRC=/path/to/refactored/project-root or SRC=/path/to/hybrid15.c." >&2
  return 1
}

is_refactored_source() {
  [[ -d "$SRC" && -f "$SRC/include/hybrid_common.h" && -f "$SRC/src/main.c" ]]
}

source_description() {
  if is_refactored_source; then
    printf 'refactored source tree: %s\n' "$SRC"
  else
    printf 'single C source: %s\n' "$SRC"
  fi
}

source_grep() {
  local pattern="$1"
  if is_refactored_source; then
    grep -R -Eq --include='*.c' --include='*.h' "$pattern" "$SRC/src" "$SRC/include"
  else
    grep -Eq "$pattern" "$SRC"
  fi
}

SRC="$(resolve_src)" || exit 1

if [[ "$MPI_NP" -ne 3 ]]; then
  echo "ERROR: this C file's MPI mode requires exactly 3 ranks. Set MPI_NP=3." >&2
  exit 2
fi

# Run the executable from ./ by default, but respect EXE paths such as /tmp/hybrid15 or bin/hybrid15.
exe_path() {
  if [[ "$EXE" == */* ]]; then
    printf '%s\n' "$EXE"
  else
    printf './%s\n' "$EXE"
  fi
}

check_c_compatibility() {
  local missing=0

  require_src_pattern() {
    local pattern="$1"
    local description="$2"
    if ! source_grep "$pattern"; then
      echo "ERROR: source is missing expected compatibility marker: $description" >&2
      missing=1
    fi
  }

  # argv contract used by run_and_capture: mode N use_cpu use_gpu0 use_gpu1 cpu_blas gpu_blas cpu_threads tile prec variant config_name.
  require_src_pattern 'int[[:space:]]+mode[[:space:]]*=.*argv\[1\]' 'argv[1] mode'
  require_src_pattern 'int[[:space:]]+N[[:space:]]*=.*argv\[2\]' 'argv[2] N'
  require_src_pattern 'int[[:space:]]+use_cpu[[:space:]]*=.*argv\[3\]' 'argv[3] use_cpu'
  require_src_pattern 'int[[:space:]]+use_gpu0[[:space:]]*=.*argv\[4\]' 'argv[4] use_gpu0'
  require_src_pattern 'int[[:space:]]+use_gpu1[[:space:]]*=.*argv\[5\]' 'argv[5] use_gpu1'
  require_src_pattern 'int[[:space:]]+cpu_blas[[:space:]]*=.*argv\[6\]' 'argv[6] cpu_blas'
  require_src_pattern 'int[[:space:]]+gpu_blas[[:space:]]*=.*argv\[7\]' 'argv[7] gpu_blas'
  require_src_pattern 'int[[:space:]]+cpu_threads[[:space:]]*=.*argv\[8\]' 'argv[8] cpu_threads'
  require_src_pattern 'int[[:space:]]+tile[[:space:]]*=.*argv\[9\]' 'argv[9] tile'
  require_src_pattern 'int[[:space:]]+prec[[:space:]]*=.*argv\[10\]' 'argv[10] precision'
  require_src_pattern 'variant[[:space:]]*=.*argv\[11\]' 'argv[11] variant'
  require_src_pattern 'config_name[[:space:]]*=.*argv\[12\]' 'argv[12] config_name'

  # Output records consumed by the CSV extractors.
  require_src_pattern 'RUN_SUMMARY mode=' 'RUN_SUMMARY line'
  require_src_pattern 'PTHREADS_TOTAL_TIMING' 'PTHREADS_TOTAL_TIMING line'
  require_src_pattern 'MPI_TOTAL_TIMING' 'MPI_TOTAL_TIMING line'
  require_src_pattern 'PTHREADS_TIMELINE_EVENT' 'PTHREADS_TIMELINE_EVENT line'
  require_src_pattern 'MPI_TIMELINE_EVENT' 'MPI_TIMELINE_EVENT line'
  require_src_pattern 'PTHREADS_TIMELINE_TILE_EVENT' 'PTHREADS_TIMELINE_TILE_EVENT line'
  require_src_pattern 'MPI_TIMELINE_TILE_EVENT' 'MPI_TIMELINE_TILE_EVENT line'
  require_src_pattern 'PTHREADS_PANEL_TIMING' 'PTHREADS_PANEL_TIMING line'
  require_src_pattern 'MPI_PANEL_TIMING' 'MPI_PANEL_TIMING line'
  require_src_pattern 'PTHREADS_PANEL_DEV_TIMING' 'PTHREADS_PANEL_DEV_TIMING line'
  require_src_pattern 'MPI_PANEL_DEV_TIMING' 'MPI_PANEL_DEV_TIMING line'

  if [[ "$missing" -ne 0 ]]; then
    echo "ERROR: $SRC does not match the HYBRID15 interface expected by this script." >&2
    exit 3
  fi
}

SIZES=()
for ((n=SIZE_START; n<=SIZE_END; n+=SIZE_STEP)); do
  SIZES+=("$n")
done

tile_for_N() {
  local n="$1"
  printf '%s\n' "$n"
}

csv_quote() {
  local v="${1-}"
  v="${v//\"/\"\"}"
  printf '"%s"' "$v"
}

append_csv_row_to() {
  local file="$1"
  shift
  local first=1 v
  for v in "$@"; do
    if [[ $first -eq 1 ]]; then
      csv_quote "$v" >> "$file"
      first=0
    else
      printf ',' >> "$file"
      csv_quote "$v" >> "$file"
    fi
  done
  printf '\n' >> "$file"
}

write_csv_header() {
  local file="$1"
  shift
  : > "$file"
  append_csv_row_to "$file" "$@"
}

join_header_prefixed() {
  local prefix="$1"
  shift
  local f
  for f in "$@"; do
    printf '%s_%s\n' "$prefix" "$f"
  done
}

summary_base_fields=(
  phase requested_mode N repeat_count variant requested_config dtype requested_np
  cpu_blas_arg gpu_blas_arg use_cpu_arg use_gpu0_arg use_gpu1_arg cpu_threads_arg tile_arg expected_panel
  run_status rc
)

summary_keys=(
  mode variant config n np dtype cpu_impl gpu_impl use_cpu use_gpu0 use_gpu1 cpu_threads tile cache_fit_b status
  total_s total_gflops c00 checksum_sample
  bench_cpu_rows bench_gpu0_rows bench_gpu1_rows bench_cpu_rate bench_gpu0_rate bench_gpu1_rate

  modeinfo_mode modeinfo_n modeinfo_ranks modeinfo_dtype modeinfo_status modeinfo_reason
  modeinfo_bytesa modeinfo_bytesb modeinfo_bytesb_panel modeinfo_bytesc
  modeinfo_panel_b modeinfo_panel_cols modeinfo_cpu_impl modeinfo_gpu_impl
  modeinfo_use_cpu modeinfo_use_gpu0 modeinfo_use_gpu1 modeinfo_ngpu modeinfo_ngpu_root
  modeinfo_cpu_threads modeinfo_tile modeinfo_cache_fit_b modeinfo_ts modeinfo_variant modeinfo_config

  total_total_s total_pthreads_total_s total_mpi_compute_s total_total_gflops total_status total_c_0 total_checksum_sample
  calendar_total calendar_pthreads_total calendar_policy calendar_panel_cols
  partition_mode partition_variant partition_cpu_share partition_rows_cpu partition_rows_gpu0 partition_rows_gpu1

  part_done_cpu part_done_gpu0 part_done_gpu1
  part_assigned_cpu part_assigned_gpu0 part_assigned_gpu1

  speed_cpu_gflops_eff speed_gpu0_gflops_eff speed_gpu1_gflops_eff
  speed_cpu_gflops_gemm speed_gpu0_gflops_gemm speed_gpu1_gflops_gemm
  speed_cpu_rows_s speed_gpu0_rows_s speed_gpu1_rows_s speed_gpu0_copy_bw_gbs speed_gpu1_copy_bw_gbs

  pthreads_full_start_abs pthreads_full_end_abs pthreads_full_start_rel pthreads_full_end_rel
  pthreads_alloc_start_abs pthreads_alloc_end_abs pthreads_alloc_start_rel pthreads_alloc_end_rel
  pthreads_fill_ac_start_abs pthreads_fill_ac_end_abs pthreads_fill_ac_start_rel pthreads_fill_ac_end_rel
  pthreads_panel_fill_start_abs pthreads_panel_fill_end_abs pthreads_panel_fill_start_rel pthreads_panel_fill_end_rel
  pthreads_init_start_abs pthreads_init_end_abs pthreads_init_start_rel pthreads_init_end_rel
  pthreads_copy_start_abs pthreads_copy_end_abs pthreads_copy_start_rel pthreads_copy_end_rel
  pthreads_compute_start_abs pthreads_compute_end_abs pthreads_compute_start_rel pthreads_compute_end_rel
  pthreads_dev_cleanup_start_abs pthreads_dev_cleanup_end_abs pthreads_dev_cleanup_start_rel pthreads_dev_cleanup_end_rel
  pthreads_host_cleanup_start_abs pthreads_host_cleanup_end_abs pthreads_host_cleanup_start_rel pthreads_host_cleanup_end_rel

  pthreads_total_start_abs pthreads_total_end_abs pthreads_total_start_rel pthreads_total_end_rel
  pthreads_total_alloc_s pthreads_total_fill_ac_s pthreads_total_panel_fill_s pthreads_total_init_s
  pthreads_total_copy_s pthreads_total_compute_s pthreads_total_dev_cleanup_s pthreads_total_host_cleanup_s
  pthreads_total_worker_total_s pthreads_total_worker_sum_s pthreads_total_worker_span_s
  pthreads_total_total_s pthreads_total_other_s
  pthreads_total_init_span_s pthreads_total_copy_span_s pthreads_total_compute_span_s
  pthreads_total_dev_cleanup_span_s pthreads_total_panel_fill_span_s pthreads_total_timing_note

  mpi_init_init_start_abs mpi_init_init_end_abs mpi_init_init_s_max mpi_init_total_with_init

  mpi_total_start_abs mpi_total_end_abs mpi_total_start_rel mpi_total_end_rel
  mpi_total_full_start_abs mpi_total_full_end_abs mpi_total_full_start_rel mpi_total_full_end_rel
  mpi_total_global_init_start_abs mpi_total_global_init_end_abs mpi_total_global_init_s
  mpi_total_total_ex_init_s mpi_total_total_with_init_s
  mpi_total_comm_total_s mpi_total_compute_s mpi_total_other_s mpi_total_timing_note

  mpi_part_bcast mpi_part_scatter mpi_part_compute mpi_part_gather
  mpi_part_comm_total mpi_part_comm_frac mpi_part_total mpi_part_total_with_init
  mpi_part_process_compute_total mpi_part_policy
)

dev_fields=(
  enabled rows_assigned rows_done share_assigned share_done r0 r1 status err
  init gemm copy cleanup total gflops_gemm gflops_eff rows_per_s copy_bw_gbs
  copy_bytes copyin_bytes copyout_bytes
  dev_start_rel dev_end_rel
  init_start_abs init_end_abs compute_start_abs compute_end_abs
  copy_start_abs copy_end_abs copyin_start_abs copyin_end_abs copyout_start_abs copyout_end_abs cleanup_start_abs cleanup_end_abs
  init_start_rel init_end_rel compute_start_rel compute_end_rel gemm_start_rel gemm_end_rel
  copy_start_rel copy_end_rel copyin_start_rel copyin_end_rel copyout_start_rel copyout_end_rel cleanup_start_rel cleanup_end_rel
)

panel_fields=(
  panel_idx j0 jb
  prev_panel_end_abs prev_panel_end_rel
  gap_start_abs gap_end_abs gap_start_rel gap_end_rel gap_before_panel_s
  fill_start_abs fill_end_abs fill_start_rel fill_end_rel
  bcast_start_abs bcast_end_abs bcast_start_rel bcast_end_rel
  scatter_start_abs scatter_end_abs scatter_start_rel scatter_end_rel
  precompute_start_abs precompute_end_abs precompute_start_rel precompute_end_rel
  init_start_abs init_end_abs init_start_rel init_end_rel
  copy_start_abs copy_end_abs copy_start_rel copy_end_rel
  compute_start_abs compute_end_abs compute_start_rel compute_end_rel
  extra_calc_start_abs extra_calc_end_abs extra_calc_start_rel extra_calc_end_rel
  gather_start_abs gather_end_abs gather_start_rel gather_end_rel
  cleanup_start_abs cleanup_end_abs cleanup_start_rel cleanup_end_rel
  panel_total_s
)

panel_dev_fields=(
  rank panel_idx j0 jb dev enabled rows_total rows_assigned rows_done share_assigned r0 r1 status err
  init gemm copy cleanup total copy_bytes copyin_bytes copyout_bytes
  dev_start_rel dev_end_rel
  init_start_abs init_end_abs compute_start_abs compute_end_abs
  copy_start_abs copy_end_abs copyin_start_abs copyin_end_abs copyout_start_abs copyout_end_abs cleanup_start_abs cleanup_end_abs
  init_start_rel init_end_rel compute_start_rel compute_end_rel gemm_start_rel gemm_end_rel
  copy_start_rel copy_end_rel copyin_start_rel copyin_end_rel copyout_start_rel copyout_end_rel cleanup_start_rel cleanup_end_rel
)

timeline_fields=(
  c_mode rank lane dev panel_idx j0 jb rows event_phase start_abs end_abs start_rel end_rel duration bytes
)

tile_timeline_fields=(
  c_mode rank lane dev panel_idx panel_j0 panel_jb tile_idx tile_j0 tile_jb k0 kb rows event_phase start_abs end_abs start_rel end_rel duration bytes
)

summary_header=("${summary_base_fields[@]}" "${summary_keys[@]}")
all_summary_keys=("${summary_keys[@]}")
while IFS= read -r f; do summary_header+=("$f"); all_summary_keys+=("$f"); done < <(join_header_prefixed cpu "${dev_fields[@]}")
while IFS= read -r f; do summary_header+=("$f"); all_summary_keys+=("$f"); done < <(join_header_prefixed gpu0 "${dev_fields[@]}")
while IFS= read -r f; do summary_header+=("$f"); all_summary_keys+=("$f"); done < <(join_header_prefixed gpu1 "${dev_fields[@]}")

rm -f "$LOG" "$CSV" "$PANEL_CSV" "$PANEL_DEV_CSV" "$TIMELINE_CSV" "$TILE_TIMELINE_CSV" "$RAW_KV_CSV"
rm -rf "$TMPDIR"
mkdir -p "$TMPDIR"

write_csv_header "$CSV" "${summary_header[@]}"
write_csv_header "$PANEL_CSV" phase requested_mode panel_kind N repeat variant config dtype np "${panel_fields[@]}"
write_csv_header "$PANEL_DEV_CSV" phase requested_mode panel_dev_kind N repeat variant config dtype np "${panel_dev_fields[@]}"
write_csv_header "$TIMELINE_CSV" phase requested_mode event_kind N repeat variant config dtype np "${timeline_fields[@]}"
write_csv_header "$TILE_TIMELINE_CSV" phase requested_mode tile_event_kind N repeat variant config dtype np "${tile_timeline_fields[@]}"
write_csv_header "$RAW_KV_CSV" phase requested_mode N repeat variant config dtype np line_no line_kind key value raw_line

compile_program() {
  local exe_actual="$EXE"
  if [[ "$EXE" != */* ]]; then
    exe_actual="./$EXE"
  fi

  if [[ "$KEEP_EXE" -eq 1 && -x "$exe_actual" ]]; then
    echo "Reusing existing executable: $exe_actual"
    return 0
  fi

  rm -f "$EXE" "$OBJ"

  if is_refactored_source; then
    echo "Compiling HYBRID15 refactored project from $SRC..."
    local build_dir="$TMPDIR/build_objs"
    rm -rf "$build_dir"
    mkdir -p "$build_dir"

    local -a objs=()
    local srcfile base obj
    for srcfile in "$SRC"/src/*.c; do
      base="$(basename "${srcfile%.c}")"
      obj="$build_dir/${base}.o"
      objs+=("$obj")
      "$CC" -O3 -acc -mp -pthread -DUSE_CBLAS -DUSE_CUBLAS -cuda -cudalib=cublas \
        -I"$SRC/include" -I"$MPI_INC" -c "$srcfile" -o "$obj"
    done

    # shellcheck disable=SC2086
    "$CC" -O3 -acc -mp -pthread -cuda -cudalib=cublas \
      -L"$MPI_LIB" "${objs[@]}" -lmpi $BLAS_LIBS -o "$EXE"
    return 0
  fi

  echo "Compiling HYBRID15 from $SRC..."
  "$CC" -O3 -acc -mp -pthread -DUSE_CBLAS -DUSE_CUBLAS -cuda -cudalib=cublas -I"$MPI_INC" -c "$SRC" -o "$OBJ"
  # shellcheck disable=SC2086
  "$CC" -O3 -acc -mp -pthread -cuda -cudalib=cublas -L"$MPI_LIB" "$OBJ" -lmpi $BLAS_LIBS -o "$EXE"
}

kv_norm_note='keys are normalized to lowercase with non-alphanumeric characters converted to underscores'

parse_run_to_kv() {
  local infile="$1"
  local outfile="$2"
  awk '
    function norm(k) {
      k = tolower(k)
      gsub(/[^a-z0-9_]+/, "_", k)
      gsub(/^_+|_+$/, "", k)
      return k
    }
    function put(k, v) {
      k = norm(k)
      if (k != "") print k "=" v
    }
    function emit_pairs(prefix, start_i,    i, p, k, v) {
      for (i = start_i; i <= NF; i++) {
        if (index($i, "=") > 0) {
          p = index($i, "=")
          k = substr($i, 1, p - 1)
          v = substr($i, p + 1)
          put(prefix k, v)
        }
      }
    }
    /^RUN_SUMMARY / { emit_pairs("", 2) }
    /^PART_PCT_DONE / { emit_pairs("part_done_", 2) }
    /^PART_PCT_ASSIGNED / { emit_pairs("part_assigned_", 2) }
    /^SPEED / { emit_pairs("speed_", 2) }
    /^DEV / {
      dev = tolower($2)
      emit_pairs(dev "_", 3)
    }
    /^PTHREADS_TIMELINE / { emit_pairs("pthreads_", 2) }
    /^PTHREADS_TOTAL_TIMING / { emit_pairs("pthreads_total_", 2) }
    /^MPI_INIT / { emit_pairs("mpi_init_", 2) }
    /^MPI_TOTAL_TIMING / { emit_pairs("mpi_total_", 2) }
    /^MPI_PART / { emit_pairs("mpi_part_", 2) }
    /^TOTAL / { emit_pairs("total_", 2) }
    /^CALENDAR / { emit_pairs("calendar_", 2) }
    /^PARTITION_FORMULA / { emit_pairs("partition_", 2) }
    /^MODE=/ { emit_pairs("modeinfo_", 1) }
    /^MEM_BYTES / { emit_pairs("mem_bytes_", 2) }
    /^MEM_BYTES_ROOT / { emit_pairs("mem_bytes_root_", 2) }
    /^MEM_FOOTPRINT / { emit_pairs("mem_footprint_", 2) }
    /^CACHE_SPEED_EST / { emit_pairs("cache_speed_", 2) }
    /^CACHE_LIMITS_FOR_B / { emit_pairs("cache_limits_", 2) }
    /^CACHE_FIT_B / { emit_pairs("cache_fit_", 2) }
    /^MPI_ROLE_MAP / { emit_pairs("mpi_role_map_", 2) }
    /^MPI_CLOCK_SYNC / { emit_pairs("mpi_clock_sync_", 2) }
    /^MPI_RANK_ROLE / { emit_pairs("mpi_rank_role_", 2) }
    /^RANK=/ { emit_pairs("rankinfo_", 1) }
  ' "$infile" > "$outfile"
}

reduced_kv_value() {
  local key="$1"
  shift
  if [[ $# -eq 0 ]]; then
    printf 'NA\n'
    return 0
  fi

  awk -F= -v k="$key" '
    function isnum(x) {
      return x ~ /^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$/
    }
    $1 == k {
      v = substr($0, length(k) + 2)
      if (v == "" || v == "NA") next
      if (first == "") first = v
      if (isnum(v)) { sum += v + 0.0; nnum++ }
      else { nonnum++ }
      nall++
    }
    END {
      if (nall == 0) {
        printf "NA"
      } else if (nonnum == 0 && nnum > 0) {
        printf "%.6f", sum / nnum
      } else {
        printf "%s", first
      }
    }
  ' "$@"
}

append_raw_kv_rows_from_log() {
  local infile="$1" phase="$2" mode="$3" N="$4" repeat_idx="$5" variant="$6" config="$7" dtype="$8" np="$9"
  awk -v out="$RAW_KV_CSV" -v phase="$phase" -v mode="$mode" -v N="$N" -v repeat_idx="$repeat_idx" -v variant="$variant" -v config="$config" -v dtype="$dtype" -v np="$np" '
    function qcsv(s,    q) {
      q = sprintf("%c", 34)
      gsub(q, q q, s)
      return q s q
    }
    function emit(line_no, line_kind, key, value, raw) {
      print qcsv(phase) "," qcsv(mode) "," qcsv(N) "," qcsv(repeat_idx) "," qcsv(variant) "," qcsv(config) "," qcsv(dtype) "," qcsv(np) "," qcsv(line_no) "," qcsv(line_kind) "," qcsv(key) "," qcsv(value) "," qcsv(raw) >> out
    }
    {
      raw = $0
      line_kind = $1
      if ($1 ~ /^MODE=/) line_kind = "MODE"
      if ($1 ~ /^RANK=/) line_kind = "RANK"
      for (i = 1; i <= NF; i++) {
        p = index($i, "=")
        if (p > 0) {
          k = substr($i, 1, p - 1)
          v = substr($i, p + 1)
          emit(NR, line_kind, k, v, raw)
        }
      }
    }
  ' "$infile"
}

append_panel_rows_from_log() {
  local infile="$1" phase="$2" mode="$3" N="$4" repeat_idx="$5" variant="$6" config="$7" dtype="$8" np="$9"
  awk -v out="$PANEL_CSV" -v phase="$phase" -v mode="$mode" -v N="$N" -v repeat_idx="$repeat_idx" -v variant="$variant" -v config="$config" -v dtype="$dtype" -v np="$np" '
    function qcsv(s,    q) { q = sprintf("%c", 34); gsub(q, q q, s); return q s q }
    function get(k) { return ((k in kv && kv[k] != "") ? kv[k] : "NA") }
    function emit(kind,    cols, n, i) {
      n = 0
      cols[++n] = phase; cols[++n] = mode; cols[++n] = kind; cols[++n] = N; cols[++n] = repeat_idx; cols[++n] = variant; cols[++n] = config; cols[++n] = dtype; cols[++n] = np
      cols[++n] = get("panel_idx"); cols[++n] = get("j0"); cols[++n] = get("jb")
      cols[++n] = get("prev_panel_end_abs"); cols[++n] = get("prev_panel_end_rel")
      cols[++n] = get("gap_start_abs"); cols[++n] = get("gap_end_abs"); cols[++n] = get("gap_start_rel"); cols[++n] = get("gap_end_rel"); cols[++n] = get("gap_before_panel_s")
      cols[++n] = get("fill_start_abs"); cols[++n] = get("fill_end_abs"); cols[++n] = get("fill_start_rel"); cols[++n] = get("fill_end_rel")
      cols[++n] = get("bcast_start_abs"); cols[++n] = get("bcast_end_abs"); cols[++n] = get("bcast_start_rel"); cols[++n] = get("bcast_end_rel")
      cols[++n] = get("scatter_start_abs"); cols[++n] = get("scatter_end_abs"); cols[++n] = get("scatter_start_rel"); cols[++n] = get("scatter_end_rel")
      cols[++n] = get("precompute_start_abs"); cols[++n] = get("precompute_end_abs"); cols[++n] = get("precompute_start_rel"); cols[++n] = get("precompute_end_rel")
      cols[++n] = get("init_start_abs"); cols[++n] = get("init_end_abs"); cols[++n] = get("init_start_rel"); cols[++n] = get("init_end_rel")
      cols[++n] = get("copy_start_abs"); cols[++n] = get("copy_end_abs"); cols[++n] = get("copy_start_rel"); cols[++n] = get("copy_end_rel")
      cols[++n] = get("compute_start_abs"); cols[++n] = get("compute_end_abs"); cols[++n] = get("compute_start_rel"); cols[++n] = get("compute_end_rel")
      cols[++n] = get("extra_calc_start_abs"); cols[++n] = get("extra_calc_end_abs"); cols[++n] = get("extra_calc_start_rel"); cols[++n] = get("extra_calc_end_rel")
      cols[++n] = get("gather_start_abs"); cols[++n] = get("gather_end_abs"); cols[++n] = get("gather_start_rel"); cols[++n] = get("gather_end_rel")
      cols[++n] = get("cleanup_start_abs"); cols[++n] = get("cleanup_end_abs"); cols[++n] = get("cleanup_start_rel"); cols[++n] = get("cleanup_end_rel")
      cols[++n] = get("panel_total_s")
      for (i = 1; i <= n; i++) printf "%s%s", (i == 1 ? "" : ","), qcsv(cols[i]) >> out
      printf "\n" >> out
    }
    /^(MPI_PANEL_TIMING|PTHREADS_PANEL_TIMING) / {
      delete kv
      for (i = 2; i <= NF; i++) {
        p = index($i, "=")
        if (p > 0) kv[substr($i, 1, p - 1)] = substr($i, p + 1)
      }
      emit($1)
    }
  ' "$infile"
}

append_panel_dev_rows_from_log() {
  local infile="$1" phase="$2" mode="$3" N="$4" repeat_idx="$5" variant="$6" config="$7" dtype="$8" np="$9"
  awk -v out="$PANEL_DEV_CSV" -v phase="$phase" -v mode="$mode" -v N="$N" -v repeat_idx="$repeat_idx" -v variant="$variant" -v config="$config" -v dtype="$dtype" -v np="$np" '
    function qcsv(s,    q) { q = sprintf("%c", 34); gsub(q, q q, s); return q s q }
    function get(k) { return ((k in kv && kv[k] != "") ? kv[k] : "NA") }
    function emit(kind,    cols, n, i) {
      n = 0
      cols[++n] = phase; cols[++n] = mode; cols[++n] = kind; cols[++n] = N; cols[++n] = repeat_idx; cols[++n] = variant; cols[++n] = config; cols[++n] = dtype; cols[++n] = np
      cols[++n] = get("rank"); cols[++n] = get("panel_idx"); cols[++n] = get("j0"); cols[++n] = get("jb"); cols[++n] = get("dev")
      cols[++n] = get("enabled"); cols[++n] = get("rows_total"); cols[++n] = get("rows_assigned"); cols[++n] = get("rows_done"); cols[++n] = get("share_assigned")
      cols[++n] = get("r0"); cols[++n] = get("r1"); cols[++n] = get("status"); cols[++n] = get("err")
      cols[++n] = get("init"); cols[++n] = get("gemm"); cols[++n] = get("copy"); cols[++n] = get("cleanup"); cols[++n] = get("total")
      cols[++n] = get("copy_bytes"); cols[++n] = get("copyin_bytes"); cols[++n] = get("copyout_bytes")
      cols[++n] = get("dev_start_rel"); cols[++n] = get("dev_end_rel")
      cols[++n] = get("init_start_abs"); cols[++n] = get("init_end_abs"); cols[++n] = get("compute_start_abs"); cols[++n] = get("compute_end_abs")
      cols[++n] = get("copy_start_abs"); cols[++n] = get("copy_end_abs"); cols[++n] = get("copyin_start_abs"); cols[++n] = get("copyin_end_abs"); cols[++n] = get("copyout_start_abs"); cols[++n] = get("copyout_end_abs"); cols[++n] = get("cleanup_start_abs"); cols[++n] = get("cleanup_end_abs")
      cols[++n] = get("init_start_rel"); cols[++n] = get("init_end_rel"); cols[++n] = get("compute_start_rel"); cols[++n] = get("compute_end_rel"); cols[++n] = get("gemm_start_rel"); cols[++n] = get("gemm_end_rel")
      cols[++n] = get("copy_start_rel"); cols[++n] = get("copy_end_rel"); cols[++n] = get("copyin_start_rel"); cols[++n] = get("copyin_end_rel"); cols[++n] = get("copyout_start_rel"); cols[++n] = get("copyout_end_rel"); cols[++n] = get("cleanup_start_rel"); cols[++n] = get("cleanup_end_rel")
      for (i = 1; i <= n; i++) printf "%s%s", (i == 1 ? "" : ","), qcsv(cols[i]) >> out
      printf "\n" >> out
    }
    /^(MPI_PANEL_DEV_TIMING|PTHREADS_PANEL_DEV_TIMING) / {
      delete kv
      for (i = 2; i <= NF; i++) {
        p = index($i, "=")
        if (p > 0) kv[substr($i, 1, p - 1)] = substr($i, p + 1)
      }
      emit($1)
    }
  ' "$infile"
}

append_timeline_rows_from_log() {
  local infile="$1" phase="$2" mode="$3" N="$4" repeat_idx="$5" variant="$6" config="$7" dtype="$8" np="$9"
  awk -v out="$TIMELINE_CSV" -v phase_meta="$phase" -v mode_meta="$mode" -v N="$N" -v repeat_idx="$repeat_idx" -v variant="$variant" -v config="$config" -v dtype="$dtype" -v np="$np" '
    function qcsv(s,    q) { q = sprintf("%c", 34); gsub(q, q q, s); return q s q }
    function get(k) { return ((k in kv && kv[k] != "") ? kv[k] : "NA") }
    function emit(kind,    cols, n, i) {
      n = 0
      cols[++n] = phase_meta; cols[++n] = mode_meta; cols[++n] = kind; cols[++n] = N; cols[++n] = repeat_idx; cols[++n] = variant; cols[++n] = config; cols[++n] = dtype; cols[++n] = np
      cols[++n] = get("mode"); cols[++n] = get("rank"); cols[++n] = get("lane"); cols[++n] = get("dev")
      cols[++n] = get("panel_idx"); cols[++n] = get("j0"); cols[++n] = get("jb"); cols[++n] = get("rows")
      cols[++n] = get("phase"); cols[++n] = get("start_abs"); cols[++n] = get("end_abs"); cols[++n] = get("start_rel"); cols[++n] = get("end_rel"); cols[++n] = get("duration"); cols[++n] = get("bytes")
      for (i = 1; i <= n; i++) printf "%s%s", (i == 1 ? "" : ","), qcsv(cols[i]) >> out
      printf "\n" >> out
    }
    /^(PTHREADS_TIMELINE_EVENT|MPI_TIMELINE_EVENT) / {
      delete kv
      for (i = 2; i <= NF; i++) {
        p = index($i, "=")
        if (p > 0) kv[substr($i, 1, p - 1)] = substr($i, p + 1)
      }
      emit($1)
    }
  ' "$infile"
}

append_tile_timeline_rows_from_log() {
  local infile="$1" phase="$2" mode="$3" N="$4" repeat_idx="$5" variant="$6" config="$7" dtype="$8" np="$9"
  awk -v out="$TILE_TIMELINE_CSV" -v phase_meta="$phase" -v mode_meta="$mode" -v N="$N" -v repeat_idx="$repeat_idx" -v variant="$variant" -v config="$config" -v dtype="$dtype" -v np="$np" '
    function qcsv(s,    q) { q = sprintf("%c", 34); gsub(q, q q, s); return q s q }
    function get(k) { return ((k in kv && kv[k] != "") ? kv[k] : "NA") }
    function emit(kind,    cols, n, i) {
      n = 0
      cols[++n] = phase_meta; cols[++n] = mode_meta; cols[++n] = kind; cols[++n] = N; cols[++n] = repeat_idx; cols[++n] = variant; cols[++n] = config; cols[++n] = dtype; cols[++n] = np
      cols[++n] = get("mode"); cols[++n] = get("rank"); cols[++n] = get("lane"); cols[++n] = get("dev")
      cols[++n] = get("panel_idx"); cols[++n] = get("panel_j0"); cols[++n] = get("panel_jb")
      cols[++n] = get("tile_idx"); cols[++n] = get("tile_j0"); cols[++n] = get("tile_jb")
      cols[++n] = get("k0"); cols[++n] = get("kb"); cols[++n] = get("rows")
      cols[++n] = get("phase"); cols[++n] = get("start_abs"); cols[++n] = get("end_abs"); cols[++n] = get("start_rel"); cols[++n] = get("end_rel"); cols[++n] = get("duration"); cols[++n] = get("bytes")
      for (i = 1; i <= n; i++) printf "%s%s", (i == 1 ? "" : ","), qcsv(cols[i]) >> out
      printf "\n" >> out
    }
    /^(PTHREADS_TIMELINE_TILE_EVENT|MPI_TIMELINE_TILE_EVENT) / {
      delete kv
      for (i = 2; i <= NF; i++) {
        p = index($i, "=")
        if (p > 0) kv[substr($i, 1, p - 1)] = substr($i, p + 1)
      }
      emit($1)
    }
  ' "$infile"
}

panel_outputs_present() {
  local runlog="$1"
  grep -qE '^(PTHREADS_PANEL_TIMING|MPI_PANEL_TIMING) ' "$runlog" &&
  grep -qE '^(PTHREADS_PANEL_DEV_TIMING|MPI_PANEL_DEV_TIMING) ' "$runlog"
}

run_and_capture() {
  local mode_num="$1"
  shift

  local -a cmd
  local exe_cmd
  exe_cmd="$(exe_path)"
  if [[ "$mode_num" == "0" ]]; then
    cmd=("$exe_cmd" "$mode_num" "$@")
  else
    local -a extra=()
    if [[ -n "${MPIRUN_EXTRA// }" ]]; then
      # shellcheck disable=SC2206
      extra=($MPIRUN_EXTRA)
    fi
    cmd=("$MPIRUN" -np "$MPI_NP" "${extra[@]}" "$exe_cmd" "$mode_num" "$@")
  fi

  if [[ "$TIMEOUT_S" -gt 0 ]]; then
    timeout "$TIMEOUT_S" "${cmd[@]}" 2>&1
  else
    "${cmd[@]}" 2>&1
  fi
}

append_summary_row() {
  local phase="$1" requested_mode="$2" N="$3" repeat_count="$4" variant="$5" config="$6" dtype="$7" np="$8"
  local cpu_blas="$9" gpu_blas="${10}" use_cpu="${11}" use_gpu0="${12}" use_gpu1="${13}" tile_arg="${14}" expected_panel="${15}" status="${16}" rc="${17}"
  shift 17
  local -a kvfiles=("$@")

  local keys_joined
  keys_joined="$(printf '%s\t' "${all_summary_keys[@]}")"

  awk -F= \
    -v out="$CSV" \
    -v keys_joined="$keys_joined" \
    -v m1="$phase" -v m2="$requested_mode" -v m3="$N" -v m4="$repeat_count" \
    -v m5="$variant" -v m6="$config" -v m7="$dtype" -v m8="$np" \
    -v m9="$cpu_blas" -v m10="$gpu_blas" -v m11="$use_cpu" -v m12="$use_gpu0" -v m13="$use_gpu1" \
    -v m14="$CPU_THREADS" -v m15="$tile_arg" -v m16="$expected_panel" -v m17="$status" -v m18="$rc" '
      function qcsv(s,    q) { q = sprintf("%c", 34); gsub(q, q q, s); return q s q }
      function isnum(x) {
        return x ~ /^[-+]?(([0-9]+([.][0-9]*)?)|([.][0-9]+))([eE][-+]?[0-9]+)?$/
      }
      BEGIN {
        nk = split(keys_joined, keys, "\t")
        for (i = 1; i <= nk; i++) if (keys[i] != "") wanted[keys[i]] = 1
      }
      {
        k = $1
        if (!(k in wanted)) next
        v = substr($0, length(k) + 2)
        if (v == "" || v == "NA") next
        if (!(k in first)) first[k] = v
        if (isnum(v)) { sum[k] += v + 0.0; nnum[k]++ }
        else { nonnum[k]++ }
        nall[k]++
      }
      END {
        out_line = qcsv(m1)
        for (i = 2; i <= 18; i++) {
          mv = (i == 2 ? m2 : i == 3 ? m3 : i == 4 ? m4 : i == 5 ? m5 : i == 6 ? m6 : i == 7 ? m7 : i == 8 ? m8 : i == 9 ? m9 : i == 10 ? m10 : i == 11 ? m11 : i == 12 ? m12 : i == 13 ? m13 : i == 14 ? m14 : i == 15 ? m15 : i == 16 ? m16 : i == 17 ? m17 : m18)
          out_line = out_line "," qcsv(mv)
        }
        for (i = 1; i <= nk; i++) {
          k = keys[i]
          if (k == "") continue
          if (!(k in nall)) v = "NA"
          else if ((k in nnum) && !(k in nonnum)) v = sprintf("%.6f", sum[k] / nnum[k])
          else v = first[k]
          out_line = out_line "," qcsv(v)
        }
        print out_line >> out
      }
    ' "${kvfiles[@]}"
}
run_series() {
  local phase="$1"
  local requested_mode="$2"
  local mode_num="$3"
  local variant="$4"
  local prec="$5"
  local cpu_blas="$6"
  local gpu_blas="$7"
  local use_cpu="$8"
  local use_gpu0="$9"
  local use_gpu1="${10}"
  local config="${11}"
  local expect_panel="${12}"
  local N="${13}"

  local dtype="double"
  if [[ "$prec" -eq 1 ]]; then dtype="float"; fi

  local np=1
  if [[ "$requested_mode" == "mpi" ]]; then np="$MPI_NP"; fi

  local run_tile
  run_tile="$(tile_for_N "$N")"

  local -a kvfiles=()
  local status_overall="OK"
  local last_rc=0
  local r runlog kvfile out rc

  for ((r=1; r<=REPEATS; r++)); do
    runlog="$TMPDIR/${phase}_${requested_mode}_${config}_${N}_run${r}.log"
    kvfile="$TMPDIR/${phase}_${requested_mode}_${config}_${N}_run${r}.kv"

    set +e
    out="$(run_and_capture "$mode_num" "$N" "$use_cpu" "$use_gpu0" "$use_gpu1" "$cpu_blas" "$gpu_blas" "$CPU_THREADS" "$run_tile" "$prec" "$variant" "$config")"
    rc=$?
    set -e
    last_rc="$rc"

    printf "%s\n" "$out" | tee -a "$LOG" > "$runlog"

    parse_run_to_kv "$runlog" "$kvfile"
    kvfiles+=("$kvfile")

    append_raw_kv_rows_from_log "$runlog" "$phase" "$requested_mode" "$N" "$r" "$variant" "$config" "$dtype" "$np"
    append_panel_rows_from_log "$runlog" "$phase" "$requested_mode" "$N" "$r" "$variant" "$config" "$dtype" "$np"
    append_panel_dev_rows_from_log "$runlog" "$phase" "$requested_mode" "$N" "$r" "$variant" "$config" "$dtype" "$np"
    append_timeline_rows_from_log "$runlog" "$phase" "$requested_mode" "$N" "$r" "$variant" "$config" "$dtype" "$np"
    append_tile_timeline_rows_from_log "$runlog" "$phase" "$requested_mode" "$N" "$r" "$variant" "$config" "$dtype" "$np"

    if [[ "$rc" -ne 0 ]]; then
      status_overall="FAILED_RC_${rc}"
      break
    fi

    local c_status
    c_status="$(reduced_kv_value status "$kvfile")"
    if [[ "$c_status" != "NA" && "$c_status" != "OK" ]]; then
      status_overall="$c_status"
      break
    fi

    if [[ "$expect_panel" -eq 1 ]] && ! panel_outputs_present "$runlog"; then
      status_overall="FAILED_NO_PANEL_OUTPUT"
      last_rc=90
      break
    fi
  done

  append_summary_row "$phase" "$requested_mode" "$N" "${#kvfiles[@]}" "$variant" "$config" "$dtype" "$np" \
    "$cpu_blas" "$gpu_blas" "$use_cpu" "$use_gpu0" "$use_gpu1" "$run_tile" "$expect_panel" "$status_overall" "$last_rc" \
    "${kvfiles[@]}"

  [[ "$status_overall" == "OK" ]]
}

check_c_compatibility
compile_program

{
  echo "==== HYBRID15 Matmul Benchmark: no-panel DGEMM/SGEMM pthreads-vs-MPI matrix, MPI scatter/gather enabled ===="
  echo "Date: $(date)"
  echo "SRC=$(source_description)"
  echo "EXE=$EXE CPU_THREADS=$CPU_THREADS TILE=N MPI_NP=$MPI_NP TIMEOUT_S=$TIMEOUT_S REPEATS=$REPEATS"
  echo "Sizes: ${SIZES[*]}"
  echo "KV parser note: $kv_norm_note"
  echo ""
} | tee "$LOG"

# phase:requested_mode:mode_num:variant:prec:cpu_blas:gpu_blas:use_cpu:use_gpu0:use_gpu1:config:expect_panel
# prec: 0=double/DGEMM, 1=float/SGEMM. gpu_blas=1 requests cuBLAS on GPU lanes/ranks.
RUN_SPECS=(
  "pthreads_nopanel_dgemm:pthreads:0:dgemm:0:1:1:1:1:1:cpu_gpu2_nopanel_scatter_gather:0"
  "mpi_nopanel_dgemm:mpi:1:dgemm:0:1:1:1:1:1:cpu_gpu2_nopanel_scatter_gather:0"
  "pthreads_nopanel_sgemm:pthreads:0:sgemm:1:1:1:1:1:1:cpu_gpu2_nopanel_scatter_gather:0"
  "mpi_nopanel_sgemm:mpi:1:sgemm:1:1:1:1:1:1:cpu_gpu2_nopanel_scatter_gather:0"
)

for N in "${SIZES[@]}"; do
  echo "=== SIZE $N ===" | tee -a "$LOG"
  for spec in "${RUN_SPECS[@]}"; do
    IFS=':' read -r phase requested_mode mode_num variant prec cpu_blas gpu_blas use_cpu use_gpu0 use_gpu1 config expect_panel <<< "$spec"
    echo "--- RUN $phase N=$N tile=$N variant=$variant config=$config ---" | tee -a "$LOG"
    if ! run_series "$phase" "$requested_mode" "$mode_num" "$variant" "$prec" "$cpu_blas" "$gpu_blas" "$use_cpu" "$use_gpu0" "$use_gpu1" "$config" "$expect_panel" "$N"; then
      echo "WARN: $phase N=$N did not finish OK; recorded status in $CSV and continuing." | tee -a "$LOG"
    fi
    echo "" | tee -a "$LOG"
  done
done

echo "Results saved to $LOG"
echo "Summary CSV saved to $CSV"
echo "Panel CSV saved to $PANEL_CSV"
echo "Panel-device CSV saved to $PANEL_DEV_CSV"
echo "Timeline-event CSV saved to $TIMELINE_CSV"
echo "Timeline-tile-event CSV saved to $TILE_TIMELINE_CSV"
echo "Raw key/value CSV saved to $RAW_KV_CSV"
