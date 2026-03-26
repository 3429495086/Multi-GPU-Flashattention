# Multi-GPU FlashAttention

A research prototype for workload modeling and scheduling in multi-GPU FlashAttention.

This repository is not yet a full multi-GPU FlashAttention runtime. The current focus is on understanding how attention workloads can be partitioned, modeled, and scheduled across GPUs before moving to communication-aware execution.

## Overview

The project studies several questions behind multi-GPU attention:

- How should attention be partitioned into blocks?
- How much work does each block or row actually contain under different masks?
- How balanced are different scheduling strategies across GPUs?
- What kind of partitioning is a better fit for future ring-style or communication-aware execution?

At the current stage, the code mainly covers workload construction, simple cost estimation, and scheduling experiments.

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

Implemented now:

- workload modeling for attention blocks
- masked active-element counting
- baseline scheduling strategies
- row-level task aggregation
- initial benchmark scripts

Still in progress:

- cleaner experiment pipeline
- stronger comparisons between scheduling methods
- better cost modeling beyond pure computation

Planned next:

- communication-aware cost terms
- heterogeneous GPU scheduling
- integration with a real multi-GPU execution backend

## Notes

- The current code focuses on scheduling logic, not kernel optimization.
- Communication cost is not yet modeled in detail.
- The repository structure may still change as experiments grow.

## Author

Yuvinci
