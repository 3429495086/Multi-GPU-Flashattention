/*
 * ring_attention_cuda_aware.cu
 * Ring Attention Skeleton — CUDA + MPI (aware)
 *
 * Compile:
 *   nvcc -o ring_attention_cuda_aware ring_attention_cuda_aware.cu \
 *   -I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
 *   -L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
 *   -lmpi
 *
 * Run:
 *   /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
 *   -np 2 ./ring_attention_cuda_aware
 */
#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include<float.h>
#include<mpi.h>
#include<cuda_runtime.h>

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
 * CPU-side partial attention (placeholder for FlashAttention)
 *
 * Output:
 *   Out[i]  = sum_j( exp(score_j - m_i) * V_j )  (UNNORMALIZED)
 *   m[i]    = max over j of score_j
 *   l[i]    = sum_j( exp(score_j - m_i) )
 */
void cpu_attention_partial(
    const float *Q, int seq_q,
    const float *K, const float *V, int seq_k,
    int dim,
    float *Out, float *m_out, float *l_out
){
    // allocate scores buffer once, reuse for every row
    float *scores = (float *)malloc(seq_k * sizeof(float));
    for(int i = 0; i < seq_q; i++){
        // compute scores and find row max
        float row_max = -FLT_MAX;
        for(int j = 0; j < seq_k; j++){
            float s = 0.0f;
            for(int d = 0; d < dim; d++)
                s += Q[i * dim + d] * K[j * dim + d];
            scores[j] = s;
            if(s > row_max)
                row_max = s;
        }
        // exp, sum, weighted V
        float row_sum = 0.0f;
        for(int d = 0; d < dim; d++){
            Out[i * dim + d] = 0.0f;
        }
        for(int j = 0; j < seq_k; j++){
            float w = expf(scores[j] - row_max);
            row_sum += w;
            for(int d = 0; d < dim; d++){
                Out[i * dim + d] += w * V[j * dim + d];
            }
        }
        m_out[i] = row_max;
        l_out[i] = row_sum;
        // Out[i] is UNNORMALIZED: sum(w * V), NOT divided by l
    }
    free(scores);
}

/*
 * Online softmax merge — FIXED VERSION
 *
 * Both acc_run and acc_new are UNNORMALIZED.
 * acc_run is updated in-place. NO division here.
 * Division by l_run happens once after all steps.
 */
void merge_online_softmax(
    float *m_run, float *l_run, float *acc_run,
    const float *m_new, const float *l_new, const float *acc_new,
    int seq_q, int dim
){
    for(int i = 0; i < seq_q; i++){
        float m_old = m_run[i];
        float m_max = fmaxf(m_old, m_new[i]);

        float scale_old = expf(m_old - m_max);
        float scale_new = expf(m_new[i] - m_max);

        //merge unnormalized accumulators
        for(int d = 0; d < dim; d++){
            acc_run[i * dim + d] = scale_old * acc_run[i * dim + d] +
                                   scale_new * acc_new[i * dim + d];
        }
        l_run[i] = scale_old * l_run[i] + scale_new * l_new[i];
        m_run[i] = m_max;
    }
}

int main(int argc, char **argv){
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if(size < 2){
        if(rank == 0)
            printf("Need at least 2 processes. Got %d.\n", size);
        MPI_Finalize();
        return 1;
    }
    // bind each rank to their GPUs
    CHECK_CUDA(cudaSetDevice(rank));

    int device;
    CHECK_CUDA(cudaGetDevice(&device));
    // GPU's information
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("[Rank %d] GPU %d: %s (%.0f MB total)\n",
            rank, device, prop.name, prop.totalGlobalMem / 1e6);
    
    // right neighbors
    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;

    // dimensions
    int seq_per_gpu = 128;  //demo.c(block size)
    int num_heads = 8;      //benchmark_single_gpu_shapes.py(num-head)
    int head_dim = 64;      //benchmark_single_gpu_shapes.py(head-dim)
    int seq_q = seq_per_gpu * num_heads;
    int seq_k = seq_q;
    int dim = head_dim;

    // KV shard size
    size_t kv_size = seq_k * dim * sizeof(float);
    int n_floats = seq_k * dim;

    printf("[Rank %d] seq_q=%d, seq_k=%d, dim=%d, "
           "KV shard=%.1f KB\n",
           rank, seq_q, seq_k, dim, kv_size / 1024.0);

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

    float *m_run = (float *)malloc(seq_q * sizeof(float));
    float *l_run = (float *)malloc(seq_q * sizeof(float));
    float *acc_run = (float *)malloc(seq_q * dim * sizeof(float));

    for(int i = 0; i < seq_q; i++){
        m_run[i] = -FLT_MAX;
        l_run[i] = 0.0f;
    }
    for (int i = 0; i < seq_q * dim; i++) {
        acc_run[i] = 0.0f;
    }
    // tmp
    float *m_local = (float *)malloc(seq_q * sizeof(float));
    float *l_local = (float *)malloc(seq_q * sizeof(float));
    float *acc_local = (float *)malloc(seq_q * dim * sizeof(float));

    // host staging buffers (pinned memory)
    float *h_send, *h_recv;
    CHECK_CUDA(cudaMallocHost(&h_send, kv_size));
    CHECK_CUDA(cudaMallocHost(&h_recv, kv_size));
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
    
    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();

    // step 0: attention on own KV
    cpu_attention_partial(Q, seq_q, h_init, h_init, seq_k, dim,
                          acc_local, m_local, l_local);
    merge_online_softmax(m_run, l_run, acc_run,
                         m_local, l_local, acc_local,
                         seq_q, dim);
    printf("[Rank %d] step 0: attention on own KV\n", rank);

    free(h_init);

    // RING LOOP: size - 1 steps (CUDA-aware MPI)
    for(int step = 0; step < size - 1; step++){
        double t1 = MPI_Wtime();

        // GPU pointers directly to MPI — no manual cudaMemcpy
        MPI_Status status;
        MPI_Sendrecv(current_kv, n_floats, MPI_FLOAT, right, step,
                     next_kv,    n_floats, MPI_FLOAT, left,  step,
                     MPI_COMM_WORLD, &status);

        double t2 = MPI_Wtime();

        // still need host copy for CPU attention (temporary)
        CHECK_CUDA(cudaMemcpy(h_recv, next_kv, kv_size,
                              cudaMemcpyDeviceToHost));

        double t3 = MPI_Wtime();

        cpu_attention_partial(Q, seq_q, h_recv, h_recv, seq_k, dim,
                              acc_local, m_local, l_local);
        double t4 = MPI_Wtime();

        merge_online_softmax(m_run, l_run, acc_run,
                             m_local, l_local, acc_local,
                             seq_q, dim);
        double t5 = MPI_Wtime();

        int source = (rank - step - 1 + size) % size;
        printf("[Rank %d] step %d: KV from rank %d | "
               "mpi=%.3f D2H=%.3f attn=%.3f merge=%.3f "
               "total=%.3f ms\n",
               rank, step + 1, source,
               (t2-t1)*1000, (t3-t2)*1000,
               (t4-t3)*1000, (t5-t4)*1000, (t5-t1)*1000);

        float *tmp = current_kv;
        current_kv = next_kv;
        next_kv    = tmp;
    }

    // FINAL: normalize acc by l to get output
    for (int i = 0; i < seq_q; i++) {
        for (int d = 0; d < dim; d++) {
            acc_run[i * dim + d] /= l_run[i];
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = (MPI_Wtime() - t_total_start) * 1000.0;

    printf("[Rank %d] ring attention done in %.3f ms\n",
           rank, t_total);

    // dump full output for verification
    char fname[64];
    sprintf(fname, "ring_output_rank%d.bin", rank);
    FILE *fp = fopen(fname, "wb");
    fwrite(acc_run, sizeof(float), seq_q * dim, fp);
    fclose(fp);
    printf("[Rank %d] output dumped to %s\n", rank, fname);

    // clean up
    free(Q);
    free(m_run);
    free(l_run);
    free(acc_run);
    free(m_local);   
    free(l_local);
    free(acc_local);
    CHECK_CUDA(cudaFree(d_buf_1));
    CHECK_CUDA(cudaFree(d_buf_2));
    CHECK_CUDA(cudaFreeHost(h_send));
    CHECK_CUDA(cudaFreeHost(h_recv));

    MPI_Finalize();
    return 0;

}