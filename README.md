# Multi-GPU FlashAttention

Research prototype for workload modeling, scheduling, and Ring-Attention-style
multi-GPU execution for FlashAttention-like workloads.

The project started from workload modeling and scheduling experiments, and is
now being extended toward a real multi-GPU Ring Attention prototype with MPI,
CUDA-aware MPI, staged MPI, and NCCL communication backends.

## Current Focus

The current active direction is:

- model attention workload distribution;
- compare communication backends for ring-style data exchange;
- implement Ring Attention benchmark variants;
- add compute/communication overlap;
- verify overlap correctness against a full reference implementation;
- make token/KV shard ownership visible for debugging future load balancing.

The most recent experimental work is under:

```text
ring/attention/bench/
```

See `ring/attention/README.md` for build, run, correctness, and shard-debug
instructions.

## Project Structure

```text
project/
├── workload/
├── ring/
│   ├── loop/
│   └── attention/
├── tests/
└── README.md
```

### `workload/`

Workload modeling and scheduling prototype.

Main contents:

- `workload.c`, `workload.h`: attention block generation, mask handling, and cost modeling.
- `scheduler.c`, `scheduler.h`: scheduling algorithms and row/block partitioning logic.
- `demo.c`: command-line demo for scheduling experiments.
- `run_experiments.sh`: helper script for preset experiments.
- `readme`: detailed notes about workload variables and scheduler internals.

This part answers questions such as:

- how many active attention elements each block contains;
- how block-level and row-level work differ;
- how balanced Round-Robin, LPT, row-level LPT, and contiguous DP partitions are.

### `ring/`

Multi-GPU ring communication and Ring Attention experiments.

Main contents:

- `ring/loop/`: pure ring communication benchmarks without attention computation.
- `ring/attention/`: Ring Attention benchmark and correctness code.
- `ring_cuda.cu`, `ring_test.c`, `check_peer.cu`: earlier ring/CUDA experiments and utilities.

See `ring/README.md` for the Ring experiment overview.

### `ring/loop/`

Pure communication benchmarks. These measure only ring data movement with
different backends:

- staged MPI;
- staged MPI with `MPI_Isend/Irecv`;
- CUDA-aware MPI;
- CUDA-aware MPI with `MPI_Isend/Irecv`;
- NCCL point-to-point.

This part is useful for separating communication cost from full attention cost.

See `ring/loop/README.md` for usage.

### `ring/attention/`

Ring Attention benchmark implementations.

Main contents:

- `bench/`: current benchmark entry point.
- `common/`: shared helpers such as local CUDA device selection and shard-owner utilities.
- `verify_full.c`: full reference checker for benchmark outputs.
- `gpu/`: older implementation area kept for reference, not the current experiment entry point.

Current benchmark features:

- staged MPI, CUDA-aware MPI, and NCCL backends;
- blocking and `MPI_Isend/Irecv` variants;
- overlap variants for staged and CUDA-aware MPI;
- optional output dump for correctness checking;
- optional shard-owner debug logging with `ATTENTION_PRINT_SHARDS=1`;
- correctness comparison against a full reference.

See `ring/attention/README.md` for details.

### `tests/`

Python single-GPU FlashAttention timing experiments.

Main contents:

- `benchmark_single_gpu_shapes.py`: benchmarks different attention shapes.
- `single_gpu_baseline.py`: simpler baseline timing script.

This part is separate from the MPI Ring Attention experiments.

## Requirements

For workload modeling:

- GCC or Clang with C11 support.

For Ring Attention / communication benchmarks:

- NVIDIA GPU;
- CUDA toolkit;
- MPI implementation;
- NCCL for NCCL benchmarks.

For Python single-GPU tests:

- Python 3.9+;
- PyTorch with CUDA support;
- `flash-attn`.

## Quick Start

### Workload demo

```bash
cd project/workload
gcc -O2 -std=c11 -o demo demo.c scheduler.c workload.c
./demo --seq 4096 --gpus 2 --mask causal
```

### Ring Attention correctness check

Run on the GPU server from `project/ring/attention/bench`:

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

### Ring Attention shard debug

```bash
cd project/ring/attention/bench

ATTENTION_PRINT_SHARDS=1 \
/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
-np 2 ./ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench 262144 1 1
```

This prints which Q/KV token range each rank owns and which KV shard is computed
or received at each ring step.

## Current Status

Completed:

- attention workload modeling;
- block-level and row-level scheduling baselines;
- pure ring communication benchmarks;
- Ring Attention benchmark variants;
- staged and CUDA-aware overlap benchmarks;
- full-reference correctness checking;
- shard-owner debug output for benchmark files.

In progress / future work:

- clearer connection between workload scheduler output and Ring Attention execution;
- communication-aware scheduling cost terms;
- dynamic load balancing;
- heterogeneous GPU scheduling;
- future replacement of the placeholder attention kernel with a more realistic FlashAttention-style implementation.

## Author

Yuvinci
