/*
 * ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench.cu
 * Ring Attention benchmark — CUDA-aware MPI Isend/Irecv + GPU attention overlap
 *
 * Compile:
nvcc -o ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench \
ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench.cu \
-I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
-L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
-lmpi
 *
 * Run:
/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
-np 2 ./ring_attention_cuda_aware_Isendrecv_overlap_gpu_bench 262144 10 100
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<mpi.h>
#include<cuda_runtime.h>
#include "../common/device_utils.cuh"
#include "../common/shard_utils.cuh"

// CHECK_CUDA: if any CUDA call fails, stop all processes
#define CHECK_CUDA(call) do{ \
    cudaError_t err = (call); \
    if(err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

/*
 * GPU-side partial attention (placeholder for FlashAttention)
 *
 * Output:
 *   Out[i]  = sum_j( exp(score_j - m_i) * V_j )  (UNNORMALIZED)
 *   m[i]    = max over j of score_j
 *   l[i]    = sum_j( exp(score_j - m_i) )
 */
__global__
void gpu_attention_partial_kernel(
    const float *Q, const float *K, const float *V,
    int seq_q, int seq_k, int dim,
    float *Out, float *m_out, float *lout
){
    int i = blockIdx.x;
    int tid = threadIdx.x;
    int blockSize = blockDim.x;
    if(i >= seq_q)
        return;

    float local_max = -FLT_MAX;
    for(int j = tid; j < seq_k; j += blockSize){
        float score = 0.0f;
        for(int d = 0; d < dim; d++)
            score += Q[i * dim + d] * K[j * dim + d];
        if(local_max < score)
            local_max = score;
    }

    __shared__ float smax[256];
    smax[tid] = local_max;
    __syncthreads();
    for(int stride = blockSize / 2; stride > 0; stride /= 2){
        if(tid < stride && smax[tid] < smax[tid + stride])
            smax[tid] = smax[tid + stride];
        __syncthreads();
    }
    float row_max = smax[0];

    if(tid == 0){
        for(int d = 0; d < dim; d++)
            Out[i * dim + d] = 0.0f;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for(int j = tid; j < seq_k; j += blockSize){
        float score = 0.0f;
        for(int d = 0; d < dim; d++)
            score += Q[i * dim + d] * K[j * dim + d];
        float w = expf(score - row_max);
        local_sum += w;
        for(int d = 0; d < dim; d++)
            atomicAdd(&Out[i * dim + d], w * V[j * dim + d]);
    }

    __shared__ float ssum[256];
    ssum[tid] = local_sum;
    __syncthreads();
    for(int s = blockSize / 2; s > 0; s /= 2){
        if(tid < s)
            ssum[tid] += ssum[tid + s];
        __syncthreads();
    }

    if(tid == 0){
        m_out[i] = row_max;
        lout[i] = ssum[0];
    }
}

/*
 * Online softmax merge
 *
 * Both acc_run and acc_new are UNNORMALIZED.
 * acc_run is updated in-place. Division by l_run happens once after all steps.
 */
__global__
void merge_online_softmax(
    float *m_run, float *l_run, float *acc_run,
    const float *m_new, const float *l_new, const float *acc_new,
    int seq_q, int dim
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= seq_q)
        return;
    float m_old = m_run[i];
    float m_max = fmaxf(m_old, m_new[i]);

    float scale_old = expf(m_old - m_max);
    float scale_new = expf(m_new[i] - m_max);

    for(int d = 0; d < dim; d++){
        acc_run[i * dim + d] = scale_old * acc_run[i * dim + d] +
                               scale_new * acc_new[i * dim + d];
    }
    l_run[i] = scale_old * l_run[i] + scale_new * l_new[i];
    m_run[i] = m_max;
}

static bool event_ready(cudaEvent_t event, int rank){
    cudaError_t err = cudaEventQuery(event);
    if(err == cudaSuccess)
        return true;
    if(err == cudaErrorNotReady)
        return false;
    fprintf(stderr, "[Rank %d] CUDA event query failed: %s\n",
            rank, cudaGetErrorString(err));
    MPI_Abort(MPI_COMM_WORLD, 1);
    return false;
}

static void launch_attention_and_merge(
    int rank,
    cudaStream_t compute_stream,
    const float *d_Q, const float *kv,
    int seq_q, int seq_k, int dim,
    float *d_acc_local, float *d_m_local, float *d_l_local,
    float *m_run, float *l_run, float *acc_run,
    cudaEvent_t attn_start, cudaEvent_t attn_end, cudaEvent_t merge_end
){
    CHECK_CUDA(cudaEventRecord(attn_start, compute_stream));
    gpu_attention_partial_kernel<<<seq_q, 256, 0, compute_stream>>>(
        d_Q, kv, kv,
        seq_q, seq_k, dim,
        d_acc_local, d_m_local, d_l_local
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(attn_end, compute_stream));

    merge_online_softmax<<<(seq_q + 255) / 256, 256, 0, compute_stream>>>(
        m_run, l_run, acc_run,
        d_m_local, d_l_local, d_acc_local,
        seq_q, dim
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaEventRecord(merge_end, compute_stream));
}

int main(int argc, char **argv){
    int required = MPI_THREAD_FUNNELED;
    int provided = 0;
    int mpi_err = MPI_Init_thread(&argc, &argv, required, &provided);

    if (mpi_err != MPI_SUCCESS) {
        fprintf(stderr, "MPI_Init_thread failed\n");
        return 1;
    }

    if (provided < required) {
        fprintf(stderr, "MPI thread support too low: required=%d provided=%d\n",
                required, provided);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        printf("MPI thread support: required=%d provided=%d\n",
               required, provided);
    }

    if(size < 2){
        if(rank == 0)
            printf("Need at least 2 processes. Got %d.\n", size);
        MPI_Finalize();
        return 1;
    }

    const int head_dim = 64;
    if(argc < 2){
        if(rank == 0){
            printf("Usage: %s <kv_size_bytes> [warmup>=0] [iters>0]\n"
                   "kv_size_bytes must be a positive multiple of %zu bytes.\n",
                   argv[0], (size_t)(head_dim * sizeof(float)));
        }
        MPI_Finalize();
        return 1;
    }

    long long kv_size_ll = atoll(argv[1]);
    int warmup = (argc > 2) ? atoi(argv[2]) : 10;
    int iters = (argc > 3) ? atoi(argv[3]) : 100;
    if(kv_size_ll <= 0 ||
       (((size_t)kv_size_ll) % (head_dim * sizeof(float))) != 0 ||
       warmup < 0 || iters <= 0){
        if(rank == 0){
            printf("Invalid arguments: kv_size must be a positive multiple of %zu, "
                   "warmup >= 0, iters > 0.\n",
                   (size_t)(head_dim * sizeof(float)));
        }
        MPI_Finalize();
        return 1;
    }
    size_t kv_size_arg = (size_t)kv_size_ll;

    int local_rank = 0;
    int device = select_local_cuda_device(MPI_COMM_WORLD, rank, &local_rank);

    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("[Rank %d local_rank %d] GPU %d: %s (%.0f MB total)\n",
           rank, local_rank, device, prop.name, prop.totalGlobalMem / 1e6);

    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    int seq_k = (int)(kv_size_arg / (head_dim * sizeof(float)));
    int seq_q = seq_k;
    int dim = head_dim;

    TokenShard shard = make_token_shard(rank, size, seq_q, seq_k);
    int print_shards = (getenv("ATTENTION_PRINT_SHARDS") != NULL);

    if(print_shards){
        printf("[Rank %d] Q tokens=[%d,%d) KV tokens=[%d,%d) global_Q=%d global_KV=%d\n",
            rank,
            shard.q_begin, shard.q_end,
            shard.kv_begin, shard.kv_end,
            shard.global_q_tokens, shard.global_kv_tokens);
    }

    size_t kv_size = kv_size_arg;
    int n_floats = seq_k * dim;

    printf("[Rank %d] seq_q=%d, seq_k=%d, dim=%d, KV shard=%.1f KB\n",
           rank, seq_q, seq_k, dim, kv_size / 1024.0);

    float *d_buf_1, *d_buf_2;
    CHECK_CUDA(cudaMalloc(&d_buf_1, kv_size));
    CHECK_CUDA(cudaMalloc(&d_buf_2, kv_size));

    float *h_init = (float *)malloc(kv_size);
    for(int i = 0; i < n_floats; i++)
        h_init[i] = rank * 1000.0f + (float)i;
    CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size,
                          cudaMemcpyHostToDevice));

    float *Q = (float *)malloc(seq_q * dim * sizeof(float));
    for(int i = 0; i < seq_q * dim; i++)
        Q[i] = (rank * 500.0f + (float)(i % dim) * 0.01f) * 0.001f;

    float *m_run, *l_run, *acc_run;
    CHECK_CUDA(cudaMalloc(&m_run, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l_run, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acc_run, seq_q * dim * sizeof(float)));

    float *h_m_init = (float *)malloc(seq_q * sizeof(float));
    for(int i = 0; i < seq_q; i++)
        h_m_init[i] = -FLT_MAX;
    CHECK_CUDA(cudaMemcpy(m_run, h_m_init, seq_q * sizeof(float),
                          cudaMemcpyHostToDevice));

    /*
     * current_kv: the shard to SEND this step
     *             (starts as our own KV)
     * next_kv:    the buffer to RECEIVE into
     *
     * After each step we swap them:
     *   what we received becomes what we send next.
     */
    float *current_kv = d_buf_1;
    float *next_kv = d_buf_2;

    float *d_Q, *d_m_local;
    CHECK_CUDA(cudaMalloc(&d_Q, seq_q * dim * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_m_local, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_Q, Q, seq_q * dim * sizeof(float),
                          cudaMemcpyHostToDevice));

    float *d_l_local;
    CHECK_CUDA(cudaMalloc(&d_l_local, seq_q * sizeof(float)));

    float *d_acc_local;
    CHECK_CUDA(cudaMalloc(&d_acc_local, seq_q * dim * sizeof(float)));

    float *h_acc = (float *)malloc(seq_q * dim * sizeof(float));
    float *h_l = (float *)malloc(seq_q * sizeof(float));

    cudaStream_t compute_stream;
    CHECK_CUDA(cudaStreamCreate(&compute_stream));

    cudaEvent_t attn_start_event, attn_end_event, merge_end_event;
    CHECK_CUDA(cudaEventCreate(&attn_start_event));
    CHECK_CUDA(cudaEventCreate(&attn_end_event));
    CHECK_CUDA(cudaEventCreate(&merge_end_event));

    double sum_total = 0.0;
    double sum_mpi = 0.0;
    double sum_attn = 0.0;
    double sum_merge = 0.0;
    double sum_final = 0.0;

    for(int iter = 0; iter < warmup + iters; iter++){
        CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size,
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(m_run, h_m_init, seq_q * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(l_run, 0, seq_q * sizeof(float)));
        CHECK_CUDA(cudaMemset(acc_run, 0, seq_q * dim * sizeof(float)));
        current_kv = d_buf_1;
        next_kv = d_buf_2;
        int current_kv_owner = rank;

        double local_mpi = 0.0;
        double local_attn = 0.0;
        double local_merge = 0.0;
        double local_final = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_total_start = MPI_Wtime();

        for(int step = 0; step < size - 1; step++){
            double t_mpi_start = MPI_Wtime();
            MPI_Request reqs[2];
            MPI_Irecv(next_kv, n_floats, MPI_FLOAT, left, step,
                      MPI_COMM_WORLD, &reqs[0]);
            MPI_Isend(current_kv, n_floats, MPI_FLOAT, right, step,
                      MPI_COMM_WORLD, &reqs[1]);
            
            int recv_kv_owner = ring_next_kv_owner(rank, size, step);
            if(print_shards && iter == 0){
                printf("[Rank %d] step=%d compute_kv_owner=%d KV tokens=[%d,%d), "
                    "receiving_kv_owner=%d KV tokens=[%d,%d)\n",
                    rank, step,
                    current_kv_owner,
                    current_kv_owner * seq_k,
                    (current_kv_owner + 1) * seq_k,
                    recv_kv_owner,
                    recv_kv_owner * seq_k,
                    (recv_kv_owner + 1) * seq_k);
            }

            launch_attention_and_merge(
                rank, compute_stream,
                d_Q, current_kv,
                seq_q, seq_k, dim,
                d_acc_local, d_m_local, d_l_local,
                m_run, l_run, acc_run,
                attn_start_event, attn_end_event, merge_end_event
            );

            // Record pure MPI completion time; wait for merge separately.
            int mpi_done = 0;
            while(!mpi_done && !event_ready(merge_end_event, rank))
                MPI_Testall(2, reqs, &mpi_done, MPI_STATUSES_IGNORE);
            while(!mpi_done)
                MPI_Testall(2, reqs, &mpi_done, MPI_STATUSES_IGNORE);
            double t_mpi_end = MPI_Wtime();
            CHECK_CUDA(cudaEventSynchronize(merge_end_event));

            float attn_ms = 0.0f;
            float merge_ms = 0.0f;
            CHECK_CUDA(cudaEventElapsedTime(&attn_ms, attn_start_event, attn_end_event));
            CHECK_CUDA(cudaEventElapsedTime(&merge_ms, attn_end_event, merge_end_event));

            local_mpi += (t_mpi_end - t_mpi_start) * 1000.0;
            local_attn += attn_ms;
            local_merge += merge_ms;

            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv = tmp;
            current_kv_owner = recv_kv_owner;
        }

        if(print_shards && iter == 0){
            printf("[Rank %d] final compute_kv_owner=%d KV tokens=[%d,%d)\n",
                rank,
                current_kv_owner,
                current_kv_owner * seq_k,
                (current_kv_owner + 1) * seq_k);
        }

        launch_attention_and_merge(
            rank, compute_stream,
            d_Q, current_kv,
            seq_q, seq_k, dim,
            d_acc_local, d_m_local, d_l_local,
            m_run, l_run, acc_run,
            attn_start_event, attn_end_event, merge_end_event
        );
        CHECK_CUDA(cudaEventSynchronize(merge_end_event));

        float attn_ms = 0.0f;
        float merge_ms = 0.0f;
        CHECK_CUDA(cudaEventElapsedTime(&attn_ms, attn_start_event, attn_end_event));
        CHECK_CUDA(cudaEventElapsedTime(&merge_ms, attn_end_event, merge_end_event));
        local_attn += attn_ms;
        local_merge += merge_ms;

        double tf1 = MPI_Wtime();
        CHECK_CUDA(cudaMemcpy(h_acc, acc_run, seq_q * dim * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_l, l_run, seq_q * sizeof(float),
                              cudaMemcpyDeviceToHost));
        for(int i = 0; i < seq_q; i++)
            for(int d = 0; d < dim; d++)
                h_acc[i * dim + d] /= h_l[i];
        double tf2 = MPI_Wtime();
        local_final += (tf2 - tf1) * 1000.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double local_total = (MPI_Wtime() - t_total_start) * 1000.0;
        double local_vals[5] = {
            local_total, local_mpi, local_attn, local_merge, local_final
        };
        double max_vals[5];
        MPI_Reduce(local_vals, max_vals, 5, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank == 0 && iter >= warmup){
            sum_total += max_vals[0];
            sum_mpi += max_vals[1];
            sum_attn += max_vals[2];
            sum_merge += max_vals[3];
            sum_final += max_vals[4];
        }
    }

    if(rank == 0){
        printf("CUDA-aware Isendrecv OVERLAP bench: warmup=%d iters=%d\n",
               warmup, iters);
        printf("avg_total=%.3f ms avg_CUDA-aware=%.3f "
               "avg_attn=%.3f avg_merge=%.3f avg_final=%.3f\n",
               sum_total / iters, sum_mpi / iters,
               sum_attn / iters, sum_merge / iters,
               sum_final / iters);
    }

    if(getenv("ATTENTION_DUMP_OUTPUT") != NULL){
        char fname[64];
        snprintf(fname, sizeof(fname), "ring_output_rank%d.bin", rank);
        FILE *fp = fopen(fname, "wb");
        if(fp == NULL){
            fprintf(stderr, "[Rank %d] failed to open %s for output\n", rank, fname);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        size_t output_elems = (size_t)seq_q * dim;
        size_t written = fwrite(h_acc, sizeof(float), output_elems, fp);
        fclose(fp);
        if(written != output_elems){
            fprintf(stderr, "[Rank %d] incomplete output dump to %s\n", rank, fname);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("[Rank %d] output dumped to %s\n", rank, fname);
    }

    free(Q);
    free(h_init);
    free(h_m_init);
    free(h_acc);
    free(h_l);
    CHECK_CUDA(cudaFree(m_run));
    CHECK_CUDA(cudaFree(l_run));
    CHECK_CUDA(cudaFree(acc_run));
    CHECK_CUDA(cudaFree(d_buf_1));
    CHECK_CUDA(cudaFree(d_buf_2));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_m_local));
    CHECK_CUDA(cudaFree(d_l_local));
    CHECK_CUDA(cudaFree(d_acc_local));
    CHECK_CUDA(cudaEventDestroy(attn_start_event));
    CHECK_CUDA(cudaEventDestroy(attn_end_event));
    CHECK_CUDA(cudaEventDestroy(merge_end_event));
    CHECK_CUDA(cudaStreamDestroy(compute_stream));

    MPI_Finalize();
    return 0;
}
