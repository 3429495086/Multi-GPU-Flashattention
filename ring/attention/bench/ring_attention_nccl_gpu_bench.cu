/*
 * ring_attention_nccl_gpu.cu
 * Ring Attention Skeleton — NCCL + GPU attention
 *
 * Compile:
nvcc -o ring_attention_nccl_gpu_bench ring_attention_nccl_gpu_bench.cu \
-I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
-L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
-lmpi \
-I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/include \
-L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/lib \
-lnccl
 *
 * Run:
/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
-np 2 ./ring_attention_nccl_gpu_bench 262144 10 100
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<mpi.h>
#include<cuda_runtime.h>
#include<nccl.h>
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

#define CHECK_NCCL(call) do{ \
    ncclResult_t err = (call); \
    if(err != ncclSuccess){ \
        fprintf(stderr, "[Rank %d] NCCL error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, ncclGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

__global__
void gpu_attention_partial_kernel(
    const float *Q, const float *K, const float *V,
    int seq_q, int seq_k, int dim,
    float *Out, float *m_out, float *lout
){
    int i = blockIdx.x;             // which row query
    int tid = threadIdx.x;          // block thread id
    int blockSize = blockDim.x;     // 256
    if(i >= seq_q)
        return;

    // calculate score, find max
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
    __syncthreads();            // wait all threads
    // reduction: find global max
    for(int stride = blockSize / 2; stride > 0; stride /= 2){
        if(tid < stride && smax[tid] < smax[tid + stride]) 
            smax[tid] = smax[tid + stride];
        __syncthreads();
    }
    float row_max = smax[0];

    // clear Out
    if(tid == 0){
        for(int d = 0; d < dim; d++)
            Out[i * dim + d] = 0.0f;
    }
    __syncthreads();

    // calculate both exp sum and weighted V
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

__global__
void merge_online_softmax(
    float *m_run, float *l_run, float *acc_run,
    const float *m_new, const float *l_new, const float *acc_new,
    int seq_q, int dim
){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if( i >= seq_q)
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
       warmup < 0 || iters <= 0) {
        if (rank == 0) {
            printf("Invalid arguments: kv_size must be a positive multiple of %zu, warmup >= 0, iters > 0.\n",
                   (size_t)(head_dim * sizeof(float)));
        }
        MPI_Finalize();
        return 1;
    }
    size_t kv_size_arg = (size_t)kv_size_ll;
    int local_rank = 0;
    int device = select_local_cuda_device(MPI_COMM_WORLD, rank, &local_rank);
    // GPU's information
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("[Rank %d local_rank %d] GPU %d: %s (%.0f MB total)\n",
            rank, local_rank, device, prop.name, prop.totalGlobalMem / 1e6);
    
    // right neighbors
    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    // dimensions
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

    // KV shard size
    size_t kv_size = seq_k * dim * sizeof(float);
    int n_floats = seq_k * dim;

    printf("[Rank %d] seq_q=%d, seq_k=%d, dim=%d, "
           "KV shard=%.1f KB\n",
           rank, seq_q, seq_k, dim, kv_size / 1024.0);

    // NCCL-initialization
    ncclUniqueId nccl_id;
    if(rank == 0){
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    }
    // use MPI tell all rank
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));


    // allocate two GPU buffers
    float *d_buf_1, *d_buf_2;
    CHECK_CUDA(cudaMalloc(&d_buf_1, kv_size));
    CHECK_CUDA(cudaMalloc(&d_buf_2, kv_size));
    
    float *h_init = (float *)malloc(kv_size);
    for(int i = 0; i < n_floats; i++)
        h_init[i] = rank * 1000.0f + (float)i;
    CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size,
                          cudaMemcpyHostToDevice));

    // Q stays local on CPU
    float *Q = (float *)malloc(seq_q * dim * sizeof(float));
    for(int i = 0; i < seq_q * dim; i++)
        Q[i] = (rank * 500.0f + (float)(i % dim) * 0.01f) * 0.001f; // for ex

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

    double sum_total = 0.0;
    double sum_nccl = 0.0;
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

        double local_nccl = 0.0;
        double local_attn = 0.0;
        double local_merge = 0.0;
        double local_final = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_total_start = MPI_Wtime();
        
        // step 0: attention on own KV
        if(print_shards && iter == 0){
            printf("[Rank %d] step=own compute_kv_owner=%d KV tokens=[%d,%d)\n",
                rank, rank, shard.kv_begin, shard.kv_end);
        }

        double ta = MPI_Wtime();
        gpu_attention_partial_kernel<<<seq_q, 256>>>(
            d_Q, d_buf_1, d_buf_1,
            seq_q, seq_k, dim,
            d_acc_local, d_m_local, d_l_local
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        double tb = MPI_Wtime();
        local_attn += (tb - ta) * 1000.0;

        ta = MPI_Wtime();
        merge_online_softmax<<<(seq_q + 255) / 256, 256>>>(
            m_run, l_run, acc_run, 
            d_m_local, d_l_local, d_acc_local,
            seq_q, dim
        );
        CHECK_CUDA(cudaGetLastError());
        CHECK_CUDA(cudaDeviceSynchronize());
        tb = MPI_Wtime();
        local_merge += (tb - ta) * 1000.0;

        // RING LOOP: size - 1 steps
        for(int step = 0; step < size - 1; step++){
            double t1 = MPI_Wtime();

            // NCCL
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclSend(current_kv, n_floats, ncclFloat,
                                right, nccl_comm, stream));
            CHECK_NCCL(ncclRecv(next_kv, n_floats, ncclFloat,
                                left, nccl_comm, stream));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(stream));

            double t2 = MPI_Wtime();

            int recv_kv_owner = ring_next_kv_owner(rank, size, step);
            if(print_shards && iter == 0){
                printf("[Rank %d] step=%d received_kv_owner=%d KV tokens=[%d,%d)\n",
                    rank, step,
                    recv_kv_owner,
                    recv_kv_owner * seq_k,
                    (recv_kv_owner + 1) * seq_k);
            }

            gpu_attention_partial_kernel<<<seq_q, 256>>>(
                d_Q, next_kv, next_kv,
                seq_q, seq_k, dim,
                d_acc_local, d_m_local, d_l_local
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            double t3 = MPI_Wtime();

            merge_online_softmax<<<(seq_q + 255) / 256, 256>>>(
                m_run, l_run, acc_run, 
                d_m_local, d_l_local, d_acc_local,
                seq_q, dim
            );
            CHECK_CUDA(cudaGetLastError());
            CHECK_CUDA(cudaDeviceSynchronize());
            double t4 = MPI_Wtime();

            local_nccl += (t2 - t1) * 1000.0;
            local_attn += (t3 - t2) * 1000.0;
            local_merge += (t4 - t3) * 1000.0;

            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv    = tmp;
        }
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
            local_total, local_nccl,
            local_attn, local_merge, local_final
        };
        double max_vals[5];
        MPI_Reduce(local_vals, max_vals, 5, MPI_DOUBLE,
                   MPI_MAX, 0, MPI_COMM_WORLD);

        if(rank == 0 && iter >= warmup){
            sum_total += max_vals[0];
            sum_nccl += max_vals[1];
            sum_attn += max_vals[2];
            sum_merge += max_vals[3];
            sum_final += max_vals[4];
        }
    }

    if(rank == 0){
        printf("nccl bench: warmup=%d iters=%d\n", warmup, iters);
        printf("avg_total=%.3f ms avg_NCCL=%.3f "
               "avg_attn=%.3f avg_merge=%.3f avg_final=%.3f\n",
               sum_total / iters, sum_nccl / iters,
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

    // clean up
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
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_NCCL(ncclCommDestroy(nccl_comm));


    MPI_Finalize();
    return 0;

}
