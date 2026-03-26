#include <stdlib.h>
#include <float.h>
#include "scheduler.h"

struct indexed_task {
    // index in the original task array
    int original_idx;
    // copied out for sorting
    double cost;
};

void init_gpu_loads(struct gpu_load *gpus, int num_gpus) {
    for(int i = 0; i < num_gpus; i++) {
        gpus[i].gpu_id = i;
        gpus[i].task_count = 0;
        gpus[i].total_cost = 0.0;
    }
}

void init_task_assignment(int *task_to_gpu, int task_count) {
    for (int i = 0; i < task_count; i++) {
        task_to_gpu[i] = -1;
    }
}

// round-robin baseline: ignore weight and just cycle gpu ids
void assign_round_robin(const struct block_task *tasks,
                        int task_count, 
                        struct gpu_load *gpus,
                        int num_gpus,
                        int *task_to_gpu) {
    init_gpu_loads(gpus, num_gpus);
    init_task_assignment(task_to_gpu, task_count);

    for(int t = 0; t < task_count; t++) {
        int gpu = t % num_gpus;
        gpus[gpu].task_count += 1;
        gpus[gpu].total_cost += tasks[t].cost;
        task_to_gpu[t] = gpu;
    }
}

int find_least_loaded_gpu(const struct gpu_load *gpus, int num_gpus) {
    int least_gpu_id = 0;
    for(int i = 0; i < num_gpus; i++) {
        if(gpus[i].total_cost < gpus[least_gpu_id].total_cost){
            least_gpu_id = i;
        }
    }
    return least_gpu_id;
}

int find_most_loaded_gpu(const struct gpu_load *gpus, int num_gpus) {
    int most_gpu_id = 0;
    for (int i = 1; i < num_gpus; i++) {
        if (gpus[i].total_cost > gpus[most_gpu_id].total_cost) {
            most_gpu_id = i;
        }
    }
    return most_gpu_id;
}

double get_makespan(const struct gpu_load *gpus, int num_gpus) {
    double max_cost = 0.0;
    for (int i = 0; i < num_gpus; i++) {
        if (gpus[i].total_cost > max_cost) {
            max_cost = gpus[i].total_cost;
        }
    }
    return max_cost;
}

static int compare_indexed_tasks_desc(const void *a, const void *b) {
    const struct indexed_task *aa = (const struct indexed_task *)a;
    const struct indexed_task *bb = (const struct indexed_task *)b;
    if (bb->cost > aa->cost) {
        return 1;
    }
    if (bb->cost < aa->cost) {
        return -1;
    }
    return 0;
}

// sort by descending cost, but still remember who each task originally was
static void assign_lpt_scan(const struct block_task *tasks,
                            int task_count, 
                            struct gpu_load *gpus,
                            int num_gpus,
                            int *task_to_gpu) {
    struct indexed_task *sorted = malloc((size_t)task_count * sizeof(struct indexed_task));
    if(sorted == NULL) {
        return;
    }
    for (int i = 0; i < task_count; i++) {
        sorted[i].original_idx = i;
        sorted[i].cost = tasks[i].cost;
    }

    qsort(sorted, (size_t)task_count, sizeof(struct indexed_task), compare_indexed_tasks_desc);

    for(int i = 0; i < task_count; i++) {
        int gpu_least = find_least_loaded_gpu(gpus, num_gpus);
        int idx = sorted[i].original_idx;

        gpus[gpu_least].task_count += 1;
        gpus[gpu_least].total_cost += sorted[i].cost;
        task_to_gpu[idx] = gpu_least;
    }
    free(sorted);
}

// this is the idealized LPT baseline in the paper/code discussion
// HEAP and AUTO are kept as placeholders, but currently still use scan
void assign_lpt(const struct block_task *tasks,
                int task_count,
                struct gpu_load *gpus,
                int num_gpus,
                int *task_to_gpu,
                enum lpt_mode mode) {
    init_gpu_loads(gpus, num_gpus);
    init_task_assignment(task_to_gpu, task_count);
    if (mode == LPT_HEAP) {
        // heap version will be added later
        assign_lpt_scan(tasks, task_count, gpus, num_gpus, task_to_gpu);
        return;
    }
    if (mode == LPT_AUTO) {
        // for now, AUTO just picks the same simple implementation
        assign_lpt_scan(tasks, task_count, gpus, num_gpus, task_to_gpu);
        return;
    }
    assign_lpt_scan(tasks, task_count, gpus, num_gpus, task_to_gpu);
}

// try one best move: take one task away from the heaviest gpu
int rebalance_move_once(const struct block_task *tasks,
                        int task_count,
                        struct gpu_load *gpus,
                        int num_gpus,
                        int *task_to_gpu) {
    int heavy = find_most_loaded_gpu(gpus, num_gpus);
    double old_makespan = get_makespan(gpus, num_gpus);

    int best_task = -1;
    int best_dst = -1;
    double best_makespan = old_makespan;

    for (int t = 0; t < task_count; t++) {
        if (task_to_gpu[t] != heavy) {
            continue;
        }

        for (int dst = 0; dst < num_gpus; dst++) {
            if (dst == heavy) {
                continue;
            }

            // simulate the move and see whether the new makespan gets smaller
            double tmp_makespan = 0.0;
            for (int g = 0; g < num_gpus; g++) {
                double load = gpus[g].total_cost;
                if (g == heavy) {
                    load -= tasks[t].cost;
                } else if (g == dst) {
                    load += tasks[t].cost;
                }
                if (load > tmp_makespan) {
                    tmp_makespan = load;
                }
            }
            if (tmp_makespan < best_makespan) {
                best_makespan = tmp_makespan;
                best_task = t;
                best_dst = dst;
            }
        }
    }
    if (best_task < 0) {
        return 0;
    }

    gpus[heavy].task_count -= 1;
    gpus[heavy].total_cost -= tasks[best_task].cost;
    gpus[best_dst].task_count += 1;
    gpus[best_dst].total_cost += tasks[best_task].cost;
    task_to_gpu[best_task] = best_dst;

    return 1;
}

// if a plain move cannot help, try one swap between heavy gpu and another gpu
int rebalance_swap_once(const struct block_task *tasks,
                        int task_count,
                        struct gpu_load *gpus,
                        int num_gpus,
                        int *task_to_gpu) {
    int heavy = find_most_loaded_gpu(gpus, num_gpus);
    double old_makespan = get_makespan(gpus, num_gpus);

    int best_task = -1;
    int best_dst = -1;
    double best_makespan = old_makespan;

    for (int t = 0; t < task_count; t++) {
        int gpu_task = task_to_gpu[t];
        if (gpu_task != heavy) {
            continue;
        }

        for (int dst = 0; dst < num_gpus; dst++) {
            int gpu_dst = task_to_gpu[dst];
            if (gpu_dst == heavy) {
                continue;
            }

            // simulate swapping task t with task dst
            double tmp_makespan = 0.0;
            for (int g = 0; g < num_gpus; g++) {
                double load = gpus[g].total_cost;
                if (g == gpu_task) {
                    load = load - tasks[t].cost + tasks[dst].cost;
                } else if (g == gpu_dst) {
                    load = load - tasks[dst].cost + tasks[t].cost;
                }
                if (load > tmp_makespan) {
                    tmp_makespan = load;
                }
            }
            if (tmp_makespan < best_makespan) {
                best_makespan = tmp_makespan;
                best_task = t;
                best_dst = dst;
            }
        }
    }
    if (best_task < 0) {
        return 0;
    }

    int gpu_task = task_to_gpu[best_task];
    int gpu_dst = task_to_gpu[best_dst];
    double cost_task = tasks[best_task].cost;
    double cost_dst = tasks[best_dst].cost;
    gpus[gpu_task].total_cost = gpus[gpu_task].total_cost - cost_task + cost_dst;
    gpus[gpu_dst].total_cost = gpus[gpu_dst].total_cost - cost_dst + cost_task;
    task_to_gpu[best_task] = gpu_dst;
    task_to_gpu[best_dst] = gpu_task;

    return 1;
}

// same LPT idea, but the scheduling unit changes from block to row
void assign_row_tasks_lpt(const struct row_task *rows,
                          int num_rows,
                          struct gpu_load *gpus,
                          int num_gpus,
                          int *row_to_gpu) {
    init_gpu_loads(gpus, num_gpus);
    init_task_assignment(row_to_gpu, num_rows);
 
    struct indexed_task *sorted = malloc((size_t)num_rows * sizeof(*sorted));
    if (sorted == NULL) {
        return;
    }
    for (int i = 0; i < num_rows; i++) {
        sorted[i].original_idx = i;
        sorted[i].cost = rows[i].total_cost;
    }

    qsort(sorted, (size_t)num_rows, sizeof(*sorted), compare_indexed_tasks_desc);

    for (int i = 0; i < num_rows; i++) {
        int gpu = find_least_loaded_gpu(gpus, num_gpus);
        int idx = sorted[i].original_idx;
        gpus[gpu].task_count += 1;
        gpus[gpu].total_cost += rows[idx].total_cost;
        row_to_gpu[idx] = gpu;
    }
    free(sorted);
}
 
// DP for contiguous row partition, which is closer to ring-style sharding
int partition_rows_contiguous(const struct row_task *rows,
                              int num_rows,
                              struct row_shard *shards,
                              int num_gpus,
                              int *row_to_gpu) {
    if (num_rows <= 0 || num_gpus <= 0 || num_gpus > num_rows) {
        return -1;
    }
 
    double prefix[num_rows + 1];
    prefix[0] = 0.0;
    for (int i = 0; i < num_rows; i++) {
        prefix[i + 1] = prefix[i] + rows[i].total_cost;
    }
 
    // dp[g][i] = best makespan for first i rows using g gpus
    double dp[num_gpus + 1][num_rows + 1];
    // cut[g][i] stores the start of the last shard in that best answer
    int cut[num_gpus + 1][num_rows + 1];
    for (int g = 0; g <= num_gpus; g++) {
        for (int i = 0; i <= num_rows; i++) {
            dp[g][i] = DBL_MAX;
            cut[g][i] = -1;
        }
    }
 
    dp[0][0] = 0.0;
 
    for (int g = 1; g <= num_gpus; g++) {
        for (int i = 1; i <= num_rows; i++) {
            for (int k = g - 1; k <= i - 1; k++) {
                double segment_cost = prefix[i] - prefix[k];
                double candidate = dp[g - 1][k];

                // last shard is [k, i), so makespan is the worse one of:
                // 1. old answer for first k rows
                // 2. cost of the new shard [k, i)
                if (segment_cost > candidate) {
                    candidate = segment_cost;
                }
                if (candidate < dp[g][i]) {
                    dp[g][i] = candidate;
                    cut[g][i] = k;
                }
            }
        }
    }
 
    // backtrack cut[][] to recover the real shard boundaries
    for (int i = 0; i < num_rows; i++) {
        row_to_gpu[i] = -1;
    }
    int end = num_rows;
    for (int g = num_gpus; g >= 1; g--) {
        int begin = cut[g][end];
        int gpu_id = g - 1;
 
        if (begin < 0) {
            return -1;
        }
        shards[gpu_id].gpu_id = gpu_id;
        shards[gpu_id].row_begin = begin;
        shards[gpu_id].row_end = end;
        shards[gpu_id].total_cost = prefix[end] - prefix[begin];
        for (int r = begin; r < end; r++) {
            row_to_gpu[r] = gpu_id;
        }
 
        end = begin;
    }
 
    return 0;
}
