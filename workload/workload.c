#include "workload.h"

static int min_int(int a, int b) {
    return a < b ? a : b;
}

int num_row_blocks(const struct workload_input *op) {
    return (op->seq_q + op->block_q - 1) / op->block_q;
}

int num_col_blocks(const struct workload_input *op) {
    return (op->seq_k + op->block_k - 1) / op->block_k;
}

int total_blocks(const struct workload_input *op) {
    return num_row_blocks(op) * num_col_blocks(op);
}

// keep all mask rules here, so other code only asks "is (q, k) active?"
int mask_value(const struct mask_desc *mask, int q, int k) {
    if (mask->type == MASK_CAUSAL) {
        return k <= q;
    }
    if (mask->type == MASK_FULL) {
        return 1;
    }
    return 0;
}

// brute-force count for one block range
int count_active_masked(const struct mask_desc *mask,
                        int q_begin, int q_end,
                        int k_begin, int k_end) {
    int active = 0;

    for (int q = q_begin; q < q_end; q++) {
        for (int k = k_begin; k < k_end; k++) {
            active += mask_value(mask, q, k);
        }
    }

    return active;
}

double estimate_cost(const struct cost_model *model, int active) {
    if (active == 0) {
        return 0.0;
    }

    return model->alpha + model->beta * (double)active;
}

// build block tasks and only keep non-empty blocks
int build_tasks(const struct workload_input *op,
                const struct mask_desc *mask,
                const struct cost_model *model,
                struct block_task *tasks,
                int max_tasks) {
    int rows = num_row_blocks(op);
    int cols = num_col_blocks(op);
    int count = 0;

    for (int i = 0; i < rows; i++) {
        int q_begin = i * op->block_q;
        int q_end = min_int(q_begin + op->block_q, op->seq_q);

        for (int j = 0; j < cols; j++) {
            int k_begin = j * op->block_k;
            int k_end = min_int(k_begin + op->block_k, op->seq_k);
            int active = count_active_masked(mask, q_begin, q_end, k_begin, k_end);
            int total = (q_end - q_begin) * (k_end - k_begin);

            // skip empty blocks, because they create no real work
            if (active == 0) {
                continue;
            }

            if (count >= max_tasks) {
                return count;
            }

            tasks[count].row_idx = i;
            tasks[count].col_idx = j;
            tasks[count].q_begin = q_begin;
            tasks[count].q_end = q_end;
            tasks[count].k_begin = k_begin;
            tasks[count].k_end = k_end;
            tasks[count].active = active;
            tasks[count].cost = estimate_cost(model, active);
            tasks[count].total = total;
            tasks[count].density = (double)active / (double)total;
            count++;
        }
    }

    return count;
}

// aggregate all blocks from the same row into one row-level task
int build_row_tasks(const struct workload_input *op,
                    const struct block_task *tasks,
                    int task_count,
                    struct row_task *rows,
                    int max_rows) {
    int num_rows = num_row_blocks(op);
    if (max_rows < num_rows) {
        return -1;
    }

    for (int i = 0; i < num_rows; i++) {
        rows[i].row_idx = i;
        rows[i].num_blocks = 0;
        rows[i].num_cols_needed = 0;
        rows[i].comp_cost = 0.0;
        rows[i].comm_cost = 0.0;
        rows[i].total_cost = 0.0;
    }

    for (int t = 0; t < task_count; t++) {
        int r = tasks[t].row_idx;
        int cols_needed = tasks[t].col_idx + 1;

        rows[r].num_blocks += 1;
        rows[r].comp_cost += tasks[t].cost;

        // for causal mask, this is basically the rightmost non-empty block + 1
        if (cols_needed > rows[r].num_cols_needed) {
            rows[r].num_cols_needed = cols_needed;
        }
    }

    for (int i = 0; i < num_rows; i++) {
        rows[i].total_cost = rows[i].comp_cost + rows[i].comm_cost;
    }

    return num_rows;
}
