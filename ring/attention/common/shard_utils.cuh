#ifndef PROJECT_RING_SHARD_UTILS_CUH
#define PROJECT_RING_SHARD_UTILS_CUH

typedef struct{
    int local_q_tokens;
    int local_kv_tokens;
    int global_q_tokens;
    int global_kv_tokens;
    int q_begin;
    int q_end;
    int kv_begin;
    int kv_end;
}TokenShard;

static inline TokenShard make_token_shard(int rank, int size,
                                          int seq_q, int seq_k) {
    TokenShard s;
    s.local_q_tokens = seq_q;
    s.local_kv_tokens = seq_k;
    s.global_q_tokens = seq_q * size;
    s.global_kv_tokens = seq_k * size;
    s.q_begin = rank * seq_q;
    s.q_end = s.q_begin + seq_q;
    s.kv_begin = rank * seq_k;
    s.kv_end = s.kv_begin + seq_k;
    return s;
}

static inline int ring_current_kv_owner(int rank, int size, int step) {
    return (rank - step + size) % size;
}

static inline int ring_next_kv_owner(int rank, int size, int step) {
    return (rank - step - 1 + size) % size;
}

#endif