/*
 * ring_loop_nccl.cu
 * Pure communication benchmark using NCCL point-to-point:
 *   device buffer -> ncclSend/ncclRecv -> device buffer
 *
 * Compile:
nvcc -o ring_loop_nccl ring_loop_nccl.cu \
 *   -I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
 *   -L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
 *   -lmpi \
 *   -I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/include \
 *   -L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/nccl/lib \
 *   -lnccl
 *
 * Run:
 *   /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
 *   -np 2 ./ring_loop_nccl 262144 10 100
 */


#include<stdio.h>
#include<stdlib.h>
#include<mpi.h>
#include<cuda_runtime.h>
#include<nccl.h>

#define CHECK_CUDA(call)do{ \
    cudaError err = (call); \
    if( err != cudaSuccess){ \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
                MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

#define CHECK_NCCL(call) do { \
    ncclResult_t err = (call); \
    if (err != ncclSuccess) { \
        fprintf(stderr, "[Rank %d] NCCL error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, ncclGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

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

    if (argc < 2) {
        if (rank == 0)
            printf("Usage: mpirun -np N ./ring_nccl <kv_size_bytes>\n");
        MPI_Finalize();
        return 1;
    }

    size_t kv_size = (size_t)atol(argv[1]);
    int warmup = (argc > 2) ? atoi(argv[2]) : 10;
    int iters = (argc > 3) ? atoi(argv[3]) : 100;
    if (kv_size == 0 || (kv_size % sizeof(float)) != 0 || warmup < 0 || iters <= 0) {
        if (rank == 0) {
            printf("Invalid arguments: kv_size must be a positive multiple of %zu, warmup >= 0, iters > 0.\n",
                   sizeof(float));
        }
        MPI_Finalize();
        return 1;
    }
    int n_floats = (int)(kv_size / sizeof(float));

    CHECK_CUDA(cudaSetDevice(rank));

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if (device_count <= 0) {
        if (rank == 0) printf("No CUDA devices found.\n");
        MPI_Finalize();
        return 1;
    }

    int device = rank;
    CHECK_CUDA(cudaGetDevice(&device));

    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("[Rank %d] GPU %d: %s (%.0f MB total)\n",
            rank, device, prop.name, prop.totalGlobalMem / 1e6);
    
    int right = (rank + 1) % size;
    int left = (rank - 1 + size) % size;
    printf("[Rank %d] KV shard: %zu bytes (%.1f KB), neighbors: left=%d right=%d\n",
            rank, kv_size, kv_size / 1024.0, left, right);
    
    // initialize NCCL
    ncclUniqueId nccl_id;
    if(rank == 0) 
        CHECK_NCCL(ncclGetUniqueId(&nccl_id));
    MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);

    ncclComm_t nccl_comm;
    CHECK_NCCL(ncclCommInitRank(&nccl_comm, size, nccl_id, rank));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    // GPU buffers
    float *d_buf_1, *d_buf_2;
    CHECK_CUDA(cudaMalloc(&d_buf_1, kv_size));
    CHECK_CUDA(cudaMalloc(&d_buf_2, kv_size));

    // fill with rank-specific data
    float *h_init = (float*)malloc(kv_size);
    if(h_init == NULL){
        fprintf(stderr, "[Rank %d] malloc(h_init) failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for(int i = 0; i < n_floats; i++)
        h_init[i] = rank * 1000.0f + (float)i;

    // One untimed correctness check before benchmarking.
    CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_buf_2, 0, kv_size));
    {
        float *current_kv = d_buf_1;
        float *next_kv = d_buf_2;
        MPI_Barrier(MPI_COMM_WORLD);
        for(int step = 0; step < size - 1; step++){
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclSend(current_kv, n_floats, ncclFloat,
                                right, nccl_comm, stream));
            CHECK_NCCL(ncclRecv(next_kv, n_floats, ncclFloat,
                                left, nccl_comm, stream));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(stream));

            float h_check = -1.0f;
            CHECK_CUDA(cudaMemcpy(&h_check, next_kv, sizeof(float), cudaMemcpyDeviceToHost));
            int expected = (rank - step - 1 + size) % size;
            if (h_check != expected * 1000.0f) {
                fprintf(stderr,
                        "[Rank %d] NCCL precheck mismatch at step %d: got %.0f expect %.0f\n",
                        rank, step, h_check, expected * 1000.0f);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }

            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv = tmp;
        }
    }

    double sum_nccl = 0.0;
    double sum_total = 0.0;

     for (int rep = 0; rep < warmup + iters; rep++) {
        CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_buf_2, 0, kv_size));

        float *current_kv = d_buf_1;
        float *next_kv = d_buf_2;

        double local_nccl = 0.0;

        MPI_Barrier(MPI_COMM_WORLD);
        double t_ring_start = MPI_Wtime();

        // RING LOOP
        for(int step = 0; step < size - 1; step++){
            double t1 = MPI_Wtime();
            CHECK_NCCL(ncclGroupStart());
            CHECK_NCCL(ncclSend(current_kv, n_floats, ncclFloat,
                                right, nccl_comm, stream));
            CHECK_NCCL(ncclRecv(next_kv, n_floats, ncclFloat,
                                left, nccl_comm, stream));
            CHECK_NCCL(ncclGroupEnd());
            CHECK_CUDA(cudaStreamSynchronize(stream));
            double t2 = MPI_Wtime();

            local_nccl += (t2 - t1) * 1000.0;

            // online softmax
            // swap
            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv = tmp;
        }
        
        double local_vals[2] = {
            local_nccl,
            (MPI_Wtime() - t_ring_start) * 1000.0
        };
        double max_vals[2];
        MPI_Reduce(local_vals, max_vals, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0 && rep >= warmup) {
            sum_nccl += max_vals[0];
            sum_total += max_vals[1];
        }
    }
    
    if (rank == 0) {
        double avg_nccl = sum_nccl / iters;
        double avg_total = sum_total / iters;
        double payload_bytes = (double)(size - 1) * kv_size;
        double total_gbps = payload_bytes / (avg_total / 1e3) / 1e9;
        double nccl_gbps = payload_bytes / (avg_nccl / 1e3) / 1e9;

        printf("RESULT backend=nccl bytes=%zu ranks=%d warmup=%d iters=%d "
               "avg_nccl_ms=%.3f avg_total_ms=%.3f total_GBps=%.3f nccl_GBps=%.3f\n",
               kv_size, size, warmup, iters,
               avg_nccl, avg_total, total_gbps, nccl_gbps);
    }

    free(h_init);
    CHECK_NCCL(ncclCommDestroy(nccl_comm));
    CHECK_CUDA(cudaStreamDestroy(stream));
    CHECK_CUDA(cudaFree(d_buf_1));
    CHECK_CUDA(cudaFree(d_buf_2));

    MPI_Finalize();
    return 0;
}