# Ring Experiments

This directory contains the multi-GPU ring communication and Ring Attention
experiments.

The purpose of this part of the project is to move from workload modeling toward
real multi-GPU execution and measurement.

## Directory Structure

```text
ring/
├── loop/
├── attention/
├── ring_cuda.cu
├── ring_test.c
└── check_peer.cu
```

## `loop/`

Pure communication benchmarks.

These programs move a KV-sized buffer around a ring without running attention.
They are used to measure communication cost separately from compute cost.

Backends:

- staged MPI;
- staged MPI with `MPI_Isend/Irecv`;
- CUDA-aware MPI;
- CUDA-aware MPI with `MPI_Isend/Irecv`;
- NCCL point-to-point.

See `loop/README.md`.

## `attention/`

Ring Attention benchmarks.

These programs combine ring KV exchange with a simple GPU attention kernel. The
current active experiments are under `attention/bench`.

Main features:

- staged MPI / CUDA-aware MPI / NCCL backends;
- `MPI_Isend/Irecv` variants;
- compute/communication overlap variants;
- correctness checking against a full reference;
- optional shard-owner debug output for token/KV ranges.

See `attention/README.md`.

## Earlier Utility Files

- `ring_cuda.cu`: early CUDA ring prototype.
- `ring_test.c`: early MPI ring test.
- `check_peer.cu`: CUDA peer-access utility.

These are kept as reference utilities. The current benchmark entry points are
`loop/` and `attention/bench/`.
