# Ring Communication Loop Benchmarks

This directory contains pure communication benchmarks for ring-style KV shard
exchange. No attention kernel is executed here.

The goal is to measure communication behavior separately from the full Ring
Attention benchmark.

## Backends

- `ring_loop_staged.cu`: GPU -> pinned host -> MPI `Sendrecv` -> pinned host -> GPU.
- `ring_loop_staged_Isendrecv.cu`: staged transfer with MPI `Isend/Irecv`.
- `ring_loop_cuda_aware.cu`: CUDA-aware MPI `Sendrecv` directly on GPU buffers.
- `ring_loop_cuda_aware_Isendrecv.cu`: CUDA-aware MPI `Isend/Irecv`.
- `ring_loop_nccl.cu`: NCCL point-to-point `ncclSend/ncclRecv`.
- `benchmark_comm_loop.sh`: helper script for running all available loop benchmarks.

## Environment

Set paths for the cluster environment:

```bash
export CUDA_HOME=/opt/cuda/current
export MPI_HOME=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5
export NCCL_HOME=/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl
export MPIRUN=$MPI_HOME/bin/mpirun
export LD_LIBRARY_PATH=$MPI_HOME/lib:$NCCL_HOME/lib:$LD_LIBRARY_PATH
```

## Build Examples

Run from `project/ring/loop`.

Staged MPI:

```bash
$CUDA_HOME/bin/nvcc -O3 -o ring_loop_staged ring_loop_staged.cu \
  -I$MPI_HOME/include -L$MPI_HOME/lib -lmpi
```

CUDA-aware MPI:

```bash
$CUDA_HOME/bin/nvcc -O3 -o ring_loop_cuda_aware ring_loop_cuda_aware.cu \
  -I$MPI_HOME/include -L$MPI_HOME/lib -lmpi
```

NCCL:

```bash
$CUDA_HOME/bin/nvcc -O3 -o ring_loop_nccl ring_loop_nccl.cu \
  -I$MPI_HOME/include -L$MPI_HOME/lib -lmpi \
  -I$NCCL_HOME/include -L$NCCL_HOME/lib -lnccl
```

## Run Example

```bash
$MPIRUN -np 2 ./ring_loop_staged 262144 10 100
```

Arguments:

```text
<kv_size_bytes> [warmup] [iters]
```

The helper script runs available executables over several sizes:

```bash
chmod +x benchmark_comm_loop.sh
./benchmark_comm_loop.sh
```

Default sizes in the script:

```text
262144 1048576 4194304 16777216
```

## Output

Each benchmark prints timing and bandwidth-style summary fields such as:

```text
avg_d2h_ms
avg_mpi_ms
avg_h2d_ms
avg_total_ms
total_GBps
mpi_GBps
```

Exact fields differ slightly by backend.

## Notes

- MPI is initialized with `MPI_Init_thread(..., MPI_THREAD_FUNNELED, ...)`.
- These benchmarks are communication-only. Use `../attention/bench` for
  end-to-end Ring Attention measurements.
- Do not commit generated executables such as `ring_loop_staged`,
  `ring_loop_cuda_aware`, or `ring_loop_nccl`.
