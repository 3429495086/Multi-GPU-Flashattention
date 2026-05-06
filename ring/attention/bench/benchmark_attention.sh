#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

MPIRUN=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun
NP=${NP:-2}
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}

# Default sizes are conservative because this benchmark also runs attention.
# Override like: SIZES="262144 1048576 2097152" ./benchmark_attention.sh
SIZES=${SIZES:-"262144 524288 1048576"}

declare -A EXES=(
  [staged]=./ring_attention_staged_gpu_bench
  [staged_Isendrecv]=./ring_attention_staged_Isendrecv_gpu_bench
  [cuda_aware]=./ring_attention_cuda_aware_gpu_bench
  [cuda_aware_Isendrecv]=./ring_attention_cuda_aware_Isendrecv_gpu_bench
  [nccl]=./ring_attention_nccl_gpu_bench
)

echo "========== Ring Attention Benchmark =========="
echo "np=${NP} warmup=${WARMUP} iters=${ITERS}"
echo "sizes=${SIZES}"
echo

for backend in staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl; do
  exe="${EXES[$backend]}"
  if [[ ! -x "${exe}" ]]; then
    echo "skip backend=${backend} (missing executable ${exe})"
    echo
    continue
  fi

  echo "===== backend=${backend} ====="
  for size in ${SIZES}; do
    echo "--- kv_size=${size} bytes ---"
    "${MPIRUN}" -np "${NP}" "${exe}" "${size}" "${WARMUP}" "${ITERS}"
  done
  echo
done

echo "========== Done =========="
