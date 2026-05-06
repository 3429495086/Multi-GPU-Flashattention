# Ring Attention Benchmarks

This directory contains CUDA/MPI benchmark implementations for a simple
single-head Ring Attention prototype. The current experimental entry point is
`bench/`.

The `gpu/` directory is kept as an older implementation area, but the current
overlap experiments, correctness checks, and shard-owner debug output are in
`bench/`.

## Files

- `bench/ring_attention_staged_gpu_bench.cu`: staged MPI baseline.
- `bench/ring_attention_staged_Isendrecv_gpu_bench.cu`: staged MPI with `MPI_Isend/Irecv`, non-overlap baseline.
- `bench/ring_attention_staged_Isendrecv_overlap_gpu_bench.cu`: staged MPI with compute/communication overlap.
- `bench/ring_attention_cuda_aware_gpu_bench.cu`: CUDA-aware MPI baseline.
- `bench/ring_attention_cuda_aware_Isendrecv_gpu_bench.cu`: CUDA-aware MPI with `MPI_Isend/Irecv`, non-overlap baseline.
- `bench/ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench.cu`: CUDA-aware MPI with compute/communication overlap.
- `bench/ring_attention_nccl_gpu_bench.cu`: NCCL baseline.
- `bench/check_attention_correctness.sh`: compares non-overlap and overlap outputs and checks both against the full reference.
- `verify_full.c`: CPU full-reference checker for dumped benchmark outputs.
- `common/device_utils.cuh`: local CUDA device selection helper.
- `common/shard_utils.cuh`: Q/KV token shard and ring-owner helper.

## Environment

The commands below use the cluster paths used in the experiments:

```bash
export CUDA_HOME=/opt/cuda/current
export MPI_HOME=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5
export NCCL_HOME=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl
export MPIRUN=$MPI_HOME/bin/mpirun
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$LD_LIBRARY_PATH
```

If your cluster uses different CUDA, MPI, or NCCL paths, update these variables.

## Build Examples

Run commands from `project/ring/attention/bench`.

```bash
cd project/ring/attention/bench
```

CUDA-aware overlap:

```bash
$CUDA_HOME/bin/nvcc -O3 -o ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench \
  ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench.cu \
  -I$MPI_HOME/include -L$MPI_HOME/lib -lmpi
```

Staged overlap:

```bash
$CUDA_HOME/bin/nvcc -O3 -o ring_attention_staged_Isendrecv_overlap_gpu_bench \
  ring_attention_staged_Isendrecv_overlap_gpu_bench.cu \
  -I$MPI_HOME/include -L$MPI_HOME/lib -lmpi
```

NCCL baseline:

```bash
$CUDA_HOME/bin/nvcc -O3 -o ring_attention_nccl_gpu_bench \
  ring_attention_nccl_gpu_bench.cu \
  -I$MPI_HOME/include -L$MPI_HOME/lib -lmpi \
  -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl
```

## Run Examples

Normal benchmark run:

```bash
$MPIRUN -np 2 ./ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench 262144 10 100
```

The arguments are:

```text
<kv_size_bytes> [warmup] [iters]
```

Default benchmark sizes used for attention experiments are:

```text
262144 524288 1048576
```

The larger `4194304` byte attention case is intentionally not used by default
because it is much slower and does not add much comparison value on the current
2 x GTX 1660 Ti server.

## Correctness Check

The correctness script builds the required benchmark executables, runs baseline
and overlap variants, dumps outputs, and verifies each output against
`verify_full.c`.

```bash
cd project/ring/attention/bench

CUDA_HOME=/opt/cuda/current \
MPI_HOME=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5 \
NP=2 SIZE=262144 WARMUP=1 ITERS=1 \
./check_attention_correctness.sh
```

Expected result:

```text
Overall: ALL PASS
staged_Isendrecv_vs_overlap: PASS
cuda_aware_Isendrecv_vs_overlap: PASS
```

This is stronger than only comparing overlap and non-overlap outputs, because
each implementation is also checked against the full reference output.

## Debug Output

Two environment variables control optional output.

`ATTENTION_PRINT_SHARDS=1` prints the local Q/KV token ranges and the KV shard
owner in each ring step. This is for debugging the current data distribution and
future workload-distribution changes.

```bash
ATTENTION_PRINT_SHARDS=1 \
$MPIRUN -np 2 ./ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench 262144 1 1
```

For overlap implementations, the output shows both the KV shard being computed
and the KV shard being received:

```text
step=0 compute_kv_owner=0 ... receiving_kv_owner=1 ...
final compute_kv_owner=1 ...
```

`ATTENTION_DUMP_OUTPUT=1` writes `ring_output_rank*.bin` files for correctness
checking. It is used by `check_attention_correctness.sh` and should not be used
for normal performance measurements.

## Notes

- MPI is initialized with `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`.
- Current attention is single-head and uses a simple GPU kernel as a placeholder
  for future FlashAttention-style kernels.
- Current partitioning is token/KV-shard based: each rank owns a local Q range
  and a local KV shard, then ring exchange circulates KV shards across ranks.
- The overlap variants compute attention on the current KV shard while receiving
  the next KV shard.
