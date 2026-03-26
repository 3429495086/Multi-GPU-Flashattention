#ifndef WORKLOAD_H
#define WORKLOAD_H

enum mask_type {
    // causal means q can only look at current and past k
    MASK_CAUSAL = 0,
    MASK_FULL = 1
};

struct mask_desc {
    enum mask_type type;
};

struct workload_input {
    // full Q/K lengths before we cut them into blocks
    int seq_q;
    int seq_k;
    // block size on each axis
    int block_q;
    int block_k;
};

// simple linear cost model: fixed part + active-element part
struct cost_model {
    // alpha = fixed overhead
    double alpha;
    // beta = per active element cost
    double beta;
};

struct block_task {
    // block position in the block grid
    int row_idx;
    int col_idx;
    // real token range covered by this block, [begin, end)
    int q_begin;
    int q_end;
    int k_begin;
    int k_end;
    // how many entries in this block are still valid after mask
    int active;
    // cost = alpha + beta * active
    double cost;
    // number of entries in this block before masking
    int total;
    // density = active / total
    double density;
};

struct row_task{
    int row_idx;
    // how many non-empty blocks belong to this row
    int num_blocks;
    // rightmost needed KV block index + 1
    int num_cols_needed;
    // sum of all block costs in this row
    double comp_cost;
    // communication part, still 0 in the current model
    double comm_cost;
    // total_cost = comp_cost + comm_cost
    double total_cost;
};

int num_row_blocks(const struct workload_input *op);

int num_col_blocks(const struct workload_input *op);

int total_blocks(const struct workload_input *op);

int mask_value(const struct mask_desc *mask, int q, int k);

int count_active_masked(const struct mask_desc *mask,
                        int q_begin, int q_end, 
                        int k_begin, int k_end);

double estimate_cost(const struct cost_model *model, int active);

int build_tasks(const struct workload_input *op,
                const struct mask_desc *mask,
                const struct cost_model *model,
                struct block_task *tasks,
                int max_tasks);

int build_row_tasks(const struct workload_input *op,
                    const struct block_task *tasks,
                    int task_count,
                    struct row_task *rows,
                    int max_rows);
                    

#endif
