#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

MPIRUN=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun
WARMUP=${WARMUP:-10}
ITERS=${ITERS:-100}
SIZES=(262144 1048576 4194304 16777216)

declare -A EXES=(
  [staged]=./ring_loop_staged
  [staged_Isendrecv]=./ring_loop_staged_Isendrecv
  [cuda_aware]=./ring_loop_cuda_aware
  [cuda_aware_Isendrecv]=./ring_loop_cuda_aware_Isendrecv
  [nccl]=./ring_loop_nccl
)

echo "==========Loop Communication Benchmark =========="
echo "warmup=${WARMUP} iters=${ITERS}"
echo

for backend in staged staged_Isendrecv cuda_aware cuda_aware_Isendrecv nccl; do
  exe="${EXES[$backend]}"
  if [[ ! -x "${exe}" ]]; then
    echo "skip backend=${backend} (missing executable ${exe})"
    echo
    continue
  fi

  echo "===== backend=${backend} ====="
  for size in "${SIZES[@]}"; do
    "${MPIRUN}" -np 2 "${exe}" "${size}" "${WARMUP}" "${ITERS}"
  done
  echo
done

echo "========== Done =========="