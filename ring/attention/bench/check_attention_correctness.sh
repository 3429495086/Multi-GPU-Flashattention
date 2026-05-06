#!/bin/bash
# Compare project attention baseline and overlap outputs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

CUDA_HOME=${CUDA_HOME:-/opt/cuda/current}
MPI_HOME=${MPI_HOME:-/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5}
NVCC=${NVCC:-"${CUDA_HOME}/bin/nvcc"}
MPIRUN=${MPIRUN:-"${MPI_HOME}/bin/mpirun"}
CC=${CC:-gcc}

NP=${NP:-2}
SIZE=${SIZE:-262144}
WARMUP=${WARMUP:-1}
ITERS=${ITERS:-1}
BUILD=${BUILD:-1}
ABS_TOL=${ABS_TOL:-1e-3}
REL_TOL=${REL_TOL:-1e-5}
RESULTS_ROOT=${RESULTS_ROOT:-"${SCRIPT_DIR}/results"}
RUN_LABEL=${RUN_LABEL:-"correctness_$(hostname)_$(date +%Y%m%d_%H%M%S)"}
CHECK_DIR="${RESULTS_ROOT}/${RUN_LABEL}/attention_correctness_np${NP}_size${SIZE}"

MPI_INC=${MPI_INC:-"-I${MPI_HOME}/include"}
MPI_LIB=${MPI_LIB:-"-L${MPI_HOME}/lib -lmpi"}

mkdir -p "${CHECK_DIR}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "ERROR: python3 is required to compare binary attention outputs." >&2
  exit 1
fi

build_targets() {
  "${CC}" -O2 -o ../verify_full ../verify_full.c -lm
  "${NVCC}" -O3 -o ring_attention_staged_Isendrecv_gpu_bench \
    ring_attention_staged_Isendrecv_gpu_bench.cu ${MPI_INC} ${MPI_LIB}
  "${NVCC}" -O3 -o ring_attention_staged_Isendrecv_overlap_gpu_bench \
    ring_attention_staged_Isendrecv_overlap_gpu_bench.cu ${MPI_INC} ${MPI_LIB}
  "${NVCC}" -O3 -o ring_attention_cuda_aware_Isendrecv_gpu_bench \
    ring_attention_cuda_aware_Isendrecv_gpu_bench.cu ${MPI_INC} ${MPI_LIB}
  "${NVCC}" -O3 -o ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench \
    ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench.cu ${MPI_INC} ${MPI_LIB}
}

verify_full_reference() {
  local label="$1"
  local output_dir="$2"

  echo "--- full reference check: ${label} ---"
  ../verify_full "${output_dir}" "${NP}" "${SIZE}"
}

run_exe() {
  local exe="$1"
  ATTENTION_DUMP_OUTPUT=1 "${MPIRUN}" -np "${NP}" "${exe}" "${SIZE}" "${WARMUP}" "${ITERS}"
}

collect_outputs() {
  local label="$1"
  local dest="${CHECK_DIR}/${label}"
  local rank=""
  mkdir -p "${dest}"

  for ((rank = 0; rank < NP; rank++)); do
    local src="ring_output_rank${rank}.bin"
    if [ ! -f "${src}" ]; then
      echo "ERROR: missing ${src} after ${label}" >&2
      return 1
    fi
    mv "${src}" "${dest}/${src}"
  done
}

compare_outputs() {
  local label="$1"
  local ref_dir="$2"
  local test_dir="$3"

  python3 - "${label}" "${ref_dir}" "${test_dir}" "${NP}" "${ABS_TOL}" "${REL_TOL}" <<'PY'
import os
import sys
from array import array

label, ref_dir, test_dir = sys.argv[1], sys.argv[2], sys.argv[3]
np = int(sys.argv[4])
abs_tol = float(sys.argv[5])
rel_tol = float(sys.argv[6])

def load_f32(path):
    size = os.path.getsize(path)
    if size % 4 != 0:
        raise RuntimeError(f"{path}: size is not a multiple of float32")
    data = array("f")
    with open(path, "rb") as f:
        data.fromfile(f, size // 4)
    return data

total = 0
sum_abs = 0.0
max_abs = 0.0
max_rel = 0.0
worst_rank = -1

for rank in range(np):
    ref = load_f32(os.path.join(ref_dir, f"ring_output_rank{rank}.bin"))
    test = load_f32(os.path.join(test_dir, f"ring_output_rank{rank}.bin"))
    if len(ref) != len(test):
        raise RuntimeError(f"rank {rank}: length mismatch {len(ref)} vs {len(test)}")

    for a, b in zip(ref, test):
        diff = abs(a - b)
        denom = max(abs(a), abs(b), 1e-12)
        rel = diff / denom
        sum_abs += diff
        total += 1
        if diff > max_abs:
            max_abs = diff
            worst_rank = rank
        if rel > max_rel:
            max_rel = rel

mean_abs = sum_abs / total if total else 0.0
passed = max_abs <= abs_tol or max_rel <= rel_tol
status = "PASS" if passed else "FAIL"

print(
    f"{label}: {status} "
    f"elements={total} max_abs_err={max_abs:.6e} "
    f"mean_abs_err={mean_abs:.6e} max_rel_err={max_rel:.6e} "
    f"worst_rank={worst_rank} abs_tol={abs_tol:.1e} rel_tol={rel_tol:.1e}"
)

sys.exit(0 if passed else 1)
PY
}

run_pair() {
  local label="$1"
  local baseline_exe="$2"
  local overlap_exe="$3"

  echo ""
  echo "===== correctness pair=${label} ====="
  rm -f ring_output_rank*.bin

  echo "--- baseline: ${baseline_exe} ---"
  run_exe "${baseline_exe}"
  collect_outputs "${label}_baseline"
  verify_full_reference "${label}_baseline" "${CHECK_DIR}/${label}_baseline"

  rm -f ring_output_rank*.bin

  echo "--- overlap: ${overlap_exe} ---"
  run_exe "${overlap_exe}"
  collect_outputs "${label}_overlap"
  verify_full_reference "${label}_overlap" "${CHECK_DIR}/${label}_overlap"

  compare_outputs \
    "${label}" \
    "${CHECK_DIR}/${label}_baseline" \
    "${CHECK_DIR}/${label}_overlap"
}

echo "========== Project Attention Correctness Check =========="
echo "Date: $(date)"
echo "Hostname: $(hostname)"
echo "np=${NP} size=${SIZE} warmup=${WARMUP} iters=${ITERS}"
echo "nvcc=${NVCC}"
echo "mpirun=${MPIRUN}"
echo "results=${CHECK_DIR}"

if [ "${BUILD}" = "1" ]; then
  build_targets
fi

run_pair \
  "staged_Isendrecv_vs_overlap" \
  "./ring_attention_staged_Isendrecv_gpu_bench" \
  "./ring_attention_staged_Isendrecv_overlap_gpu_bench"

run_pair \
  "cuda_aware_Isendrecv_vs_overlap" \
  "./ring_attention_cuda_aware_Isendrecv_gpu_bench" \
  "./ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench"

rm -f ring_output_rank*.bin

echo ""
echo "========== Project Correctness Check Done =========="
