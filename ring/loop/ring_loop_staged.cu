/*
 * ring_loop_staged.cu
 * Pure communication benchmark using staged MPI:
 *   GPU -> host pinned buffer -> MPI_Sendrecv -> host pinned buffer -> GPU
 *
 * Compile:
 *   nvcc -o ring_loop_staged ring_loop_staged.cu \
 *   -I/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/include \
 *   -L/nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/lib \
 *   -lmpi
 *
 * Run:
 *   /nethome/nvidia/hpc_sdk/Linux_x86_64/24.3/comm_libs/12.3/openmpi4/openmpi-4.1.5/bin/mpirun \
 *   -np 2 ./ring_loop_staged 262144 10 100
 */


#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

// CHECK_CUDA: if any CUDA call fails, stop all processes 
#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

//

int main(int argc, char **argv) {
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

    if(argc < 2){
        if(rank == 0)
            printf("Usage: mpirun -np N ./ring_loop_staged <kv_size_bytes> [warmup] [iters]\n");
        MPI_Finalize();
        return 1;
    }

    size_t kv_size  = (size_t)atol(argv[1]);
    int warmup = (argc > 2) ? atoi(argv[2]) : 10;
    int iters = (argc > 3) ? atoi(argv[3]) : 100;
    if(kv_size == 0 || (kv_size % sizeof(float)) != 0 || warmup < 0 || iters <= 0) {
        if (rank == 0) {
            printf("Invalid arguments: kv_size must be a positive multiple of %zu, warmup >= 0, iters > 0.\n",
                   sizeof(float));
        }
        MPI_Finalize();
        return 1;
    }
    int n_floats = (int)(kv_size / sizeof(float));

    int device_count = 0;
    CHECK_CUDA(cudaGetDeviceCount(&device_count));
    if(device_count <= 0){
        if(rank == 0)
            printf("No CUDA devices found.\n");
        MPI_Finalize();
        return 1;
    }
    // bind each rank to its GPU
    int device = rank;
    CHECK_CUDA(cudaSetDevice(device));

    // GPU's information
    struct cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, device));
    printf("[Rank %d] GPU %d: %s (%.0f MB total)\n",
           rank, device, prop.name, prop.totalGlobalMem / 1e6);
    
    // ring neighbors
    int right = (rank + 1) % size;
    int left  = (rank - 1 + size) % size;
    printf("[Rank %d] KV shard: %zu bytes (%.1f KB), neighbors: left=%d right=%d\n",
           rank, kv_size, kv_size / 1024.0, left, right);

    // allocate two GPU buffers
    float *d_buf_1, *d_buf_2;
    CHECK_CUDA(cudaMalloc(&d_buf_1, kv_size));
    CHECK_CUDA(cudaMalloc(&d_buf_2, kv_size));

    // fill d_buf_1 with this rank's KV data
    float *h_init = (float *)malloc(kv_size);
    if(h_init == NULL){
        fprintf(stderr, "[Rank %d] malloc(h_init) failed\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    for (int i = 0; i < n_floats; i++)
        h_init[i] = rank * 1000.0f + (float)i;

    // host staging buffers (pinned memory)
    float *h_send, *h_recv;
    CHECK_CUDA(cudaMallocHost(&h_send, kv_size));
    CHECK_CUDA(cudaMallocHost(&h_recv, kv_size));

    // One untimed correctness check before benchmarking
    /*
     * current_kv: the shard to SEND this step
     *             (starts as our own KV)
     * next_kv:    the buffer to RECEIVE into
     *
     * After each step we swap them:
     *   what we received becomes what we send next.
     */
    CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_buf_2, 0, kv_size));
    {
        float *current_kv = d_buf_1;
        float *next_kv = d_buf_2;
        MPI_Barrier(MPI_COMM_WORLD);
        for(int step = 0; step < size - 1; step++){
            CHECK_CUDA(cudaMemcpy(h_send, current_kv, kv_size,
                                  cudaMemcpyDeviceToHost));
            MPI_Sendrecv(h_send, n_floats, MPI_FLOAT, right, step,
                         h_recv, n_floats, MPI_FLOAT, left,  step,
                         MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int expected = (rank - step - 1 + size) % size;
            if (h_recv[0] != expected * 1000.0f) {
                fprintf(stderr,
                        "[Rank %d] staged precheck mismatch at step %d: got %.0f expect %.0f\n",
                        rank, step, h_recv[0], expected * 1000.0f);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            CHECK_CUDA(cudaMemcpy(next_kv, h_recv, kv_size, cudaMemcpyHostToDevice));
            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv = tmp;
        }
    }
    
    double sum_d2h = 0.0;
    double sum_mpi = 0.0;
    double sum_h2d = 0.0;
    double sum_total = 0.0;

    for (int rep = 0; rep < warmup + iters; rep++) {
        CHECK_CUDA(cudaMemcpy(d_buf_1, h_init, kv_size, cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemset(d_buf_2, 0, kv_size));

        float *current_kv = d_buf_1;
        float *next_kv = d_buf_2;

        double local_d2h = 0.0;
        double local_mpi = 0.0;
        double local_h2d = 0.0;
    
        MPI_Barrier(MPI_COMM_WORLD);
        double t_ring_start = MPI_Wtime();

        // RING LOOP: size-1
        for (int step = 0; step < size - 1; step++) {
            double t1 = MPI_Wtime();

            // GPU -> Host
            CHECK_CUDA(cudaMemcpy(h_send, current_kv, kv_size, 
                                  cudaMemcpyDeviceToHost));
            double t2 = MPI_Wtime();
            
            // MPI: send right, receive left
            MPI_Status status;
            MPI_Sendrecv(h_send, n_floats, MPI_FLOAT, right, step,
                        h_recv, n_floats, MPI_FLOAT, left,  step,
                        MPI_COMM_WORLD, &status);
            double t3 = MPI_Wtime();

            // Host -> GPU
            CHECK_CUDA(cudaMemcpy(next_kv, h_recv, kv_size,
                                cudaMemcpyHostToDevice));
            double t4 = MPI_Wtime();

            local_d2h += (t2 - t1) * 1000.0;
            local_mpi += (t3 - t2) * 1000.0;
            local_h2d += (t4 - t3) * 1000.0;
            
            /*
            * ★ TODO: Ring Attention 计算插入点
            *
            * compute_partial_attention(
            *     Q_local, next_kv, &output, &lse
            * );
            */

            float *tmp = current_kv;
            current_kv = next_kv;
            next_kv    = tmp;
        }

        double local_vals[4] = {
            local_d2h,
            local_mpi,
            local_h2d,
            (MPI_Wtime() - t_ring_start) * 1000.0
        };
        double max_vals[4];
        MPI_Reduce(local_vals, max_vals, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

        if (rank == 0 && rep >= warmup) {
            sum_d2h += max_vals[0];
            sum_mpi += max_vals[1];
            sum_h2d += max_vals[2];
            sum_total += max_vals[3];
        }

    }
    if (rank == 0) {
        double avg_d2h = sum_d2h / iters;
        double avg_mpi = sum_mpi / iters;
        double avg_h2d = sum_h2d / iters;
        double avg_total = sum_total / iters;
        double payload_bytes = (double)(size - 1) * kv_size;
        double total_gbps = payload_bytes / (avg_total / 1e3) / 1e9;
        double mpi_gbps = payload_bytes / (avg_mpi / 1e3) / 1e9;

        printf("RESULT backend=staged bytes=%zu ranks=%d warmup=%d iters=%d "
               "avg_d2h_ms=%.3f avg_mpi_ms=%.3f avg_h2d_ms=%.3f avg_total_ms=%.3f "
               "total_GBps=%.3f mpi_GBps=%.3f\n",
               kv_size, size, warmup, iters,
               avg_d2h, avg_mpi, avg_h2d, avg_total, total_gbps, mpi_gbps);
    }

    // cleanup 
    free(h_init);
    CHECK_CUDA(cudaFree(d_buf_1));
    CHECK_CUDA(cudaFree(d_buf_2));
    CHECK_CUDA(cudaFreeHost(h_send));
    CHECK_CUDA(cudaFreeHost(h_recv));

    MPI_Finalize();
    return 0;
}