# Workload Modeling And Scheduling

This directory contains the workload modeling and scheduling prototype for
attention computation. It is separate from the MPI/CUDA Ring Attention runtime
benchmarks under `../ring/`.

The goal of this module is to estimate how attention work is distributed across
blocks or rows, then compare several static scheduling strategies across GPUs.

## Files

- `workload.c`, `workload.h`: attention block construction, mask handling, active-element counting, and simple cost modeling.
- `scheduler.c`, `scheduler.h`: scheduling algorithms and partitioning utilities.
- `demo.c`: command-line demo for workload generation and scheduler comparison.
- `run_experiments.sh`: helper script for running preset demo configurations.

## Main Concepts

### `workload_input`

Defines the attention problem shape and block size.

- `seq_q`: total Q sequence length.
- `seq_k`: total K sequence length.
- `block_q`: block size along the Q dimension.
- `block_k`: block size along the K dimension.

This determines how the attention matrix is split into blocks.

### `mask_desc`

Describes which `(q, k)` positions are valid.

- `type`: mask type, currently mainly `MASK_CAUSAL`.

The mask logic is centralized in `mask_value()` so all workload builders use the
same validity rule.

### `cost_model`

Converts active elements into an estimated computation cost.

- `alpha`: fixed per-task overhead.
- `beta`: per-active-element cost.

The current model is intentionally simple and compute-only. Communication cost
is not modeled in detail yet.

## Task Types

### `block_task`

The basic block-level workload unit.

- `row_idx`, `col_idx`: block-grid coordinates.
- `q_begin`, `q_end`: Q token range covered by the block.
- `k_begin`, `k_end`: K token range covered by the block.
- `active`: number of valid attention elements in the block.
- `total`: total number of elements in the block.
- `density`: `active / total`.
- `cost`: estimated computation cost.

`build_tasks()` only emits non-empty blocks, so scheduling is not polluted by
zero-work tasks.

### `row_task`

Row-level workload unit built by aggregating block tasks from the same block
row.

- `row_idx`: block-row index.
- `num_blocks`: number of non-empty blocks in the row.
- `num_cols_needed`: number of KV block columns needed by this row.
- `row_cost`: total compute cost of the row.
- `comm_cost`: communication cost placeholder, currently zero.
- `total_cost`: `row_cost + comm_cost`.

`build_row_tasks()` is useful because row-level or shard-level scheduling is
closer to Ring Attention partitioning than individual block scheduling.

## Scheduling Data Structures

### `gpu_load`

Tracks the current load assigned to one GPU.

- `gpu_id`: GPU index.
- `task_count`: number of assigned tasks.
- `total_cost`: total estimated cost assigned to the GPU.

Schedulers use this to find the currently lightest or heaviest GPU.

### `task_to_gpu[]`

Maps each block task to a GPU.

Example:

```c
task_to_gpu[5] = 2;
```

This means block task 5 is assigned to GPU 2.

### `row_to_gpu[]`

Maps each row task to a GPU. It is the row-level equivalent of `task_to_gpu[]`.

### `indexed_task`

Internal helper used by LPT scheduling.

- `original_idx`: original task index before sorting.
- `cost`: task cost used for sorting.

This allows the scheduler to sort by cost while still writing assignments back
to the original task indices.

### `row_shard`

Represents a contiguous row range assigned to one GPU.

- `gpu_id`: GPU index.
- `row_begin`: first row in the shard.
- `row_end`: one-past-the-end row index.
- `total_cost`: total cost of the shard.

This structure is used by contiguous row partitioning.

## Scheduling Methods

### Round-Robin

Assigns tasks cyclically across GPUs. It is simple but does not consider task
cost.

### LPT

Longest Processing Time first. Tasks are sorted by descending cost, then each
task is assigned to the currently lightest GPU.

`assign_lpt()` is an idealized baseline because it allows arbitrary task
placement and does not enforce communication locality.

### Row-Level LPT

Same idea as LPT, but the scheduling unit is `row_task` instead of `block_task`.
This is coarser and closer to token-row partitioning.

### Move / Swap Rebalancing

`rebalance_move_once()` tries to move one task from the heaviest GPU to a
lighter GPU.

`rebalance_swap_once()` tries to swap tasks between GPUs to reduce imbalance.

These are local repair heuristics.

### Contiguous Row DP

`partition_rows_contiguous()` partitions row tasks into contiguous ranges.

Important internal variables:

- `prefix[i]`: prefix sum of costs for the first `i` rows.
- `dp[g][i]`: best makespan when assigning the first `i` rows to `g` GPUs.
- `cut[g][i]`: split point used to reconstruct the solution.
- `k`: candidate start of the last segment `[k, i)`.

This is useful when task assignment must preserve contiguous token ranges.

## Summary Metrics

- `total_cost`: total estimated work across all GPUs.
- `avg_load`: `total_cost / num_gpus`.
- `makespan`: maximum GPU load.
- `gap_to_avg`: `makespan - avg_load`.
- `imbalance`: `makespan / avg_load`; closer to `1.0` means better balance.

## Build

```bash
gcc -O2 -std=c11 -o demo demo.c scheduler.c workload.c
```

## Run

Example:

```bash
./demo --seq 4096 --gpus 2 --mask causal
```

Run preset experiments:

```bash
sh run_experiments.sh
```

## Relation To Ring Attention

This module does not run MPI or CUDA communication. It provides the workload and
scheduling side of the project.

The Ring Attention benchmark under `../ring/attention/bench` currently uses
equal token/KV shards. Future work can connect this workload scheduler to the
runtime benchmark by using scheduler output to choose Q/KV shard assignments.
