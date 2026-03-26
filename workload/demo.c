#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "workload.h"
#include "scheduler.h"

#define MAX_TASKS 4096
#define MAX_ROWS  256

static void usage(const char *prog) {
    printf("Usage: %s --seq <N> --gpus <G> [--block <B>] [--mask causal|full] [--alpha <a>] [--beta <b>]\n", prog);
}

int main(int argc, char *argv[]) {
    int seq = 0;
    int num_gpus = 0;
    int block_size = 128;
    int mask_type_val = 0;
    double alpha = 1.21;
    double beta = 0.00000384;

    // parse a few simple command-line args for quick experiments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--seq") == 0 && i + 1 < argc) {
            seq = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--gpus") == 0 && i + 1 < argc) {
            num_gpus = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--block") == 0 && i + 1 < argc) {
            block_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--mask") == 0 && i + 1 < argc) {
            i++;
            if (strcmp(argv[i], "full") == 0) mask_type_val = 1;
        } else if (strcmp(argv[i], "--alpha") == 0 && i + 1 < argc) {
            alpha = atof(argv[++i]);
        } else if (strcmp(argv[i], "--beta") == 0 && i + 1 < argc) {
            beta = atof(argv[++i]);
        }
    }

    if (seq <= 0 || num_gpus <= 0) {
        usage(argv[0]);
        return 1;
    }

    // here we use square attention shapes, so seq_q = seq_k = seq
    struct workload_input op = { .seq_q = seq, .seq_k = seq,
                                 .block_q = block_size, .block_k = block_size };
    struct cost_model model = { .alpha = 0.0, .beta = beta };
    struct mask_desc mask = { .type = mask_type_val };

    struct block_task *tasks = malloc(MAX_TASKS * sizeof(*tasks));
    int count = build_tasks(&op, &mask, &model, tasks, MAX_TASKS);

    const char *mask_name = (mask_type_val == 0) ? "causal" : "full";

    /* total pure computation cost (without alpha) */
    double total_cost = 0.0;
    for (int t = 0; t < count; t++) total_cost += tasks[t].cost;
    /* ideal = one alpha + fair share of computation */
    double ideal = alpha + total_cost / num_gpus;

    /* Round-Robin */
    struct gpu_load *rr_gpus = malloc(num_gpus * sizeof(*rr_gpus));
    int *rr_map = malloc(count * sizeof(*rr_map));
    assign_round_robin(tasks, count, rr_gpus, num_gpus, rr_map);
    double rr_ms = alpha + get_makespan(rr_gpus, num_gpus);
    double rr_imb = rr_ms / ideal;

    /* Block LPT */
    struct gpu_load *blk_gpus = malloc(num_gpus * sizeof(*blk_gpus));
    int *blk_map = malloc(count * sizeof(*blk_map));
    assign_lpt(tasks, count, blk_gpus, num_gpus, blk_map, LPT_SCAN);
    double blk_ms = alpha + get_makespan(blk_gpus, num_gpus);
    double blk_imb = blk_ms / ideal;

    /* Row LPT */
    struct row_task *rows = malloc(MAX_ROWS * sizeof(*rows));
    int row_count = build_row_tasks(&op, tasks, count, rows, MAX_ROWS);

    // now the unit is one whole row, not one single block
    struct gpu_load *row_gpus = malloc(num_gpus * sizeof(*row_gpus));
    int *row_map = malloc(row_count * sizeof(*row_map));
    assign_row_tasks_lpt(rows, row_count, row_gpus, num_gpus, row_map);
    double row_ms = alpha + get_makespan(row_gpus, num_gpus);
    double row_imb = row_ms / ideal;

    /* Contiguous DP */
    double dp_ms = -1.0;
    double dp_imb = -1.0;
    if (num_gpus <= row_count) {
        struct row_shard *shards = malloc(num_gpus * sizeof(*shards));
        int *dp_map = malloc(row_count * sizeof(*dp_map));
        if (partition_rows_contiguous(rows, row_count, shards, num_gpus, dp_map) == 0) {
            double dp_max = 0.0;

            // final DP makespan = heaviest contiguous shard
            for (int g = 0; g < num_gpus; g++) {
                if (shards[g].total_cost > dp_max) dp_max = shards[g].total_cost;
            }
            dp_ms = alpha + dp_max;
            dp_imb = dp_ms / ideal;
        }
        free(shards);
        free(dp_map);
    }

    /* output */
    printf("%-6d %-5d %-6s %-6d | %-10.2f %-6.3f | %-10.2f %-6.3f | %-10.2f %-6.3f | ",
           seq, num_gpus, mask_name, count,
           rr_ms, rr_imb,
           blk_ms, blk_imb,
           row_ms, row_imb);
    if (dp_ms >= 0) {
        printf("%-10.2f %-6.3f | ", dp_ms, dp_imb);
    } else {
        printf("%-10s %-6s | ", "N/A", "N/A");
    }
    printf("%-10.2f\n", ideal);

    free(tasks); free(rr_gpus); free(rr_map);
    free(blk_gpus); free(blk_map);
    free(rows); free(row_gpus); free(row_map);
    return 0;
}
