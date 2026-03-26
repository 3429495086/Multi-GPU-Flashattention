#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "workload.h"

// different LPT entry modes
enum lpt_mode {
    LPT_SCAN = 0,
    LPT_HEAP = 1,
    LPT_AUTO = 2
};

struct gpu_load {
    int gpu_id;
    // how many tasks this gpu gets
    int task_count;
    // current summed cost on this gpu
    double total_cost;
};

// reset every gpu load to 0
void init_gpu_loads(struct gpu_load *gpus, int num_gpus);

// fill the mapping with -1 first
void init_task_assignment(int *task_to_gpu, int task_count);

// plain baseline: assign tasks in fixed order
void assign_round_robin(const struct block_task *tasks,
                        int task_count, 
                        struct gpu_load *gpus,
                        int num_gpus,
                        int *task_to_gpu); 

// helpers for current load summary
int find_least_loaded_gpu(const struct gpu_load *gpus, int num_gpus);
int find_most_loaded_gpu(const struct gpu_load *gpus, int num_gpus);
double get_makespan(const struct gpu_load *gpus, int num_gpus);

// idealized LPT baseline
void assign_lpt(const struct block_task *tasks, 
                int task_count,
                struct gpu_load *gpus,
                int num_gpus,
                int *task_to_gpu,
                enum lpt_mode mode);

int rebalance_move_once(const struct block_task *tasks,
                        int task_count,
                        struct gpu_load *gpus,
                        int num_gpus,
                        int *task_to_gpu);

int rebalance_swap_once(const struct block_task *tasks,
                        int task_count,
                        struct gpu_load *gpus,
                        int num_gpus,
                        int *task_to_gpu);

// same idea as block-level LPT, but the unit becomes one row_task
void assign_row_tasks_lpt(const struct row_task *rows,
                          int num_rows,
                          struct gpu_load *gpus,
                          int num_gpus,
                          int *row_to_gpu);

// output of the contiguous-row DP partition
struct row_shard {
    int gpu_id;
    int row_begin;
    // shard covers [row_begin, row_end)
    int row_end;
    double total_cost;
};
 
// split rows into contiguous shards with minimum makespan
int partition_rows_contiguous(const struct row_task *rows,
                              int num_rows,
                              struct row_shard *shards,
                              int num_gpus,
                              int *row_to_gpu);

#endif
