#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "[Rank %d] CUDA error at %s:%d: %s\n", \
                rank, __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != 2) {
        if (rank == 0) printf("This test expects exactly 2 processes.\n");
        MPI_Finalize();
        return 1;
    }

    CHECK_CUDA(cudaSetDevice(rank));

    int device;
    cudaGetDevice(&device);
    struct cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    printf("[Rank %d] bound to GPU %d: %s (%.0f MB free)\n",
           rank, device, prop.name, prop.totalGlobalMem / 1e6);

    int right = (rank + 1) % size;
    int left  = (rank - 1 + size) % size;

    size_t kv_size = 128 * 8 * 64 * sizeof(float);
    printf("[Rank %d] KV shard size = %zu bytes (%.1f KB)\n",
           rank, kv_size, kv_size / 1024.0);

    float *d_send, *d_recv;
    CHECK_CUDA(cudaMalloc(&d_send, kv_size));
    CHECK_CUDA(cudaMalloc(&d_recv, kv_size));

    int n_floats = (int)(kv_size / sizeof(float));
    float *h_init = (float *)malloc(kv_size);
    for (int i = 0; i < n_floats; i++) {
        h_init[i] = rank * 1000.0f + (float)i;
    }
    CHECK_CUDA(cudaMemcpy(d_send, h_init, kv_size, cudaMemcpyHostToDevice));
    free(h_init);

    float *h_send = (float *)malloc(kv_size);
    float *h_recv = (float *)malloc(kv_size);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

    CHECK_CUDA(cudaMemcpy(h_send, d_send, kv_size, cudaMemcpyDeviceToHost));

    MPI_Status status;
    MPI_Sendrecv(h_send, n_floats, MPI_FLOAT, right, 0,
                 h_recv, n_floats, MPI_FLOAT, left,  0,
                 MPI_COMM_WORLD, &status);

    CHECK_CUDA(cudaMemcpy(d_recv, h_recv, kv_size, cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    double t_end = MPI_Wtime();

    printf("[Rank %d] ring step done in %.3f ms (GPU->host->MPI->host->GPU)\n",
           rank, (t_end - t_start) * 1000.0);

    float *h_check = (float *)malloc(kv_size);
    CHECK_CUDA(cudaMemcpy(h_check, d_recv, kv_size, cudaMemcpyDeviceToHost));
    printf("[Rank %d] received from rank %d: first 4 values = [%.0f, %.0f, %.0f, %.0f]\n",
           rank, left, h_check[0], h_check[1], h_check[2], h_check[3]);
    free(h_check);

    CHECK_CUDA(cudaFree(d_send));
    CHECK_CUDA(cudaFree(d_recv));
    free(h_send);
    free(h_recv);

    MPI_Finalize();
    return 0;
}

