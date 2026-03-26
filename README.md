# Multi-GPU FlashAttention

A research prototype for workload modeling, scheduling, and Ring-Attention-style multi-GPU execution in FlashAttention, with longer-term directions in dynamic load balancing and heterogeneous GPU scheduling.

This repository started from workload modeling and scheduling experiments, and is now being extended toward a Ring-Attention-style multi-GPU execution prototype with real end-to-end runtime evaluation. A longer-term goal is to study how the scheduler should adapt when runtime load changes or when GPUs have different capabilities.

## Overview

The project studies several questions behind multi-GPU attention:

- How should attention be partitioned into blocks?
- How much work does each block or row actually contain under different masks?
- How balanced are different scheduling strategies across GPUs?
- What kind of partitioning is a better fit for Ring-Attention-style execution?
- How should scheduling change under dynamic load imbalance or heterogeneous GPUs?

At the current stage, the code covers workload construction, simple cost estimation, scheduling experiments, and ongoing work toward a real multi-GPU Ring-style prototype.

## Current Features

- Block-level workload construction for attention matrices
- Support for causal and full attention masks
- Simple linear computation cost model
- Row-level aggregation for coarser scheduling
- Scheduling baselines:
  - Round-Robin
  - LPT (Longest Processing Time)
  - Row-level LPT
  - Contiguous row partitioning with dynamic programming
- Single-GPU benchmark scripts for shape-based timing experiments

## Project Structure

- `workload/`
  - `workload.c`, `workload.h`: block generation, mask handling, and cost modeling
  - `scheduler.c`, `scheduler.h`: scheduling algorithms and partitioning logic
  - `demo.c`: command-line demo for quick scheduling experiments
  - `run_experiments.sh`: helper script for several demo runs

- `tests/`
  - `benchmark_single_gpu_shapes.py`: benchmark different attention shapes on a single GPU
  - `single_gpu_baseline.py`: simple baseline timing script
  - `results/`: saved CSV benchmark outputs

## Requirements

For the C prototype:

- GCC or Clang with C11 support

For the Python benchmark scripts:

- Python 3.9+
- PyTorch with CUDA support
- `flash-attn`
- An NVIDIA GPU with a working CUDA environment

## Build And Run

### Compile the scheduling demo

```bash
cd workload
gcc -O2 -std=c11 -o demo demo.c scheduler.c workload.c
```

### Run a simple scheduling example

```bash
./demo --seq 4096 --gpus 2 --mask causal
```

Example output:

```text
4096   2     causal ... | RR | Block-LPT | Row-LPT | DP | ideal
```

### Run the preset experiment script

```bash
cd workload
sh run_experiments.sh
```

### Run the Python single-GPU benchmark

From the repository root:

```bash
python3 tests/benchmark_single_gpu_shapes.py --device 0
```

Or run the simpler baseline script:

```bash
python3 tests/single_gpu_baseline.py
```

## Project Status

This repository is still in an early research / prototype stage.

Completed so far:

- workload modeling for attention blocks
- masked active-element counting
- baseline scheduling strategies
- row-level task aggregation
- initial benchmark scripts

Current work:

- Ring-Attention-style multi-GPU execution prototype
- real end-to-end runtime measurement on multiple GPUs
- comparison between measured runtime and compute-only predictions from the current model
- cleaner experiment pipeline

Next steps:

- communication backend comparison across MPI, CUDA-aware MPI, and NCCL
- deeper communication-aware cost terms
- dynamic load balancing beyond static partitioning
- heterogeneous GPU scheduling
- tighter integration between scheduling results and execution results

## Notes

- The project started from scheduling logic and is now moving toward real multi-GPU execution.
- Communication cost is still not modeled in detail.
- The repository structure may still change as experiments grow.

## Author

Yuvinci
