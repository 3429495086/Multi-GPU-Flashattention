/*
 * ring_attention_cuda_aware_gpu.cu
 * Ring Attention Skeleton — CUDA-aware MPI + GPU attention
 *
 * Compile:
nvcc -o ring_attention_cuda_aware_gpu ring_attention_cuda_aware_gpu.cu \
-I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
-L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
-lmpi
 *
 * Run:
/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
-np 2 ./ring_attention_cuda_aware_gpu
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

/*
 * Online softmax merge — FIXED VERSION
 *
 * Both acc_run and acc_new are UNNORMALIZED.
 * acc_run is updated in-place. NO division here.
 * Division by l_run happens once after all steps.
 */
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

    float *m_run, *l_run, *acc_run;
    CHECK_CUDA(cudaMalloc(&m_run, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&l_run, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&acc_run, seq_q * dim * sizeof(float)));

    float *h_m_init = (float *)malloc(seq_q * sizeof(float));
    for(int i = 0; i < seq_q; i++)
        h_m_init[i] = -FLT_MAX;
    CHECK_CUDA(cudaMemcpy(m_run, h_m_init, seq_q * sizeof(float),
                          cudaMemcpyHostToDevice));
    free(h_m_init);
    CHECK_CUDA(cudaMemset(l_run, 0, seq_q * sizeof(float)));
    CHECK_CUDA(cudaMemset(acc_run, 0, seq_q * dim * sizeof(float)));

    // float *h_send, *h_recv;
    // CHECK_CUDA(cudaMallocHost(&h_send, kv_size));
    // CHECK_CUDA(cudaMallocHost(&h_recv, kv_size));
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

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total_start = MPI_Wtime();
    
    // step 0: attention on own KV
    gpu_attention_partial_kernel<<<seq_q, 256>>>(
        d_Q, d_buf_1, d_buf_1,
        seq_q, seq_k, dim,
        d_acc_local, d_m_local, d_l_local
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    merge_online_softmax<<<(seq_q + 255) / 256, 256>>>(
        m_run, l_run, acc_run, 
        d_m_local, d_l_local, d_acc_local,
        seq_q, dim
    );
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    printf("[Rank %d] step 0: attention on own KV\n", rank);

    free(h_init);

    // RING LOOP: size - 1 steps
    for(int step = 0; step < size - 1; step++){
        double t1 = MPI_Wtime();
        // // GPU -> Host
        // CHECK_CUDA(cudaMemcpy(h_send, current_kv, kv_size,
        //                       cudaMemcpyDeviceToHost));
        // double t2 = MPI_Wtime();

        // MPI: send right, receive left
        MPI_Status status;
        MPI_Sendrecv(current_kv, n_floats, MPI_FLOAT, right, step,
                     next_kv, n_floats, MPI_FLOAT, left, step,
                     MPI_COMM_WORLD, &status);
        double t2 = MPI_Wtime();

        // Host -> GPU
        // CHECK_CUDA(cudaMemcpy(next_kv, h_recv, kv_size,
        //                       cudaMemcpyHostToDevice));
        // double t4 = MPI_Wtime();

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

        int source = (rank - step - 1 + size) % size;
        printf("[Rank %d] step %d: KV from rank %d | "
               "MPI=%.3f "
               "attn=%.3f merge=%.3f total=%.3f ms\n",
               rank, step + 1, source,
               (t2-t1)*1000,
               (t3-t2)*1000, (t4-t3)*1000, (t4-t1)*1000);

        float *tmp = current_kv;
        current_kv = next_kv;
        next_kv    = tmp;
    }

    // FINAL: normalize acc by l to get output
    float *h_acc = (float *)malloc(seq_q * dim * sizeof(float));
    float *h_l = (float *)malloc(seq_q * sizeof(float));
    CHECK_CUDA(cudaMemcpy(h_acc, acc_run, seq_q * dim * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_l, l_run, seq_q * sizeof(float),
                          cudaMemcpyDeviceToHost));
    for(int i = 0; i < seq_q; i++)
        for(int d = 0; d < dim; d++)
            h_acc[i * dim + d] /= h_l[i];

    MPI_Barrier(MPI_COMM_WORLD);
    double t_total = (MPI_Wtime() - t_total_start) * 1000.0;

    printf("[Rank %d] ring attention done in %.3f ms\n",
           rank, t_total);

    // dump full output for verification
    char fname[64];
    sprintf(fname, "ring_output_rank%d.bin", rank);
    FILE *fp = fopen(fname, "wb");
    fwrite(h_acc, sizeof(float), seq_q * dim, fp);
    fclose(fp);
    printf("[Rank %d] output dumped to %s\n", rank, fname);

    // clean up
    free(Q);
    free(h_acc);
    free(h_l);
    CHECK_CUDA(cudaFree(m_run));
    CHECK_CUDA(cudaFree(l_run));
    CHECK_CUDA(cudaFree(acc_run));
        // free(m_local);   
        // free(l_local);
        // free(acc_local);
    CHECK_CUDA(cudaFree(d_buf_1));
    CHECK_CUDA(cudaFree(d_buf_2));
        // CHECK_CUDA(cudaFreeHost(h_send));
        // CHECK_CUDA(cudaFreeHost(h_recv));
    CHECK_CUDA(cudaFree(d_Q));
    CHECK_CUDA(cudaFree(d_m_local));
    CHECK_CUDA(cudaFree(d_l_local));
    CHECK_CUDA(cudaFree(d_acc_local));


    MPI_Finalize();
    return 0;

}