#ifndef PROJECT_RING_DEVICE_UTILS_CUH
#define PROJECT_RING_DEVICE_UTILS_CUH

#include<cuda_runtime.h>
#include<mpi.h>
#include<stdio.h>
#include<stdlib.h>

static int read_local_rank_from_env(void){
    const char *names[] = {
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MV2_COMM_WORLD_LOCAL_RANK",
        "SLURM_LOCALID",
        "MPI_LOCALRANKID",
        "PMI_LOCAL_RANK",
        "PALS_LOCAL_RANK",
        NULL
    };

    for(int i = 0; names[i] != NULL; i++){
        const char *value = getenv(names[i]);
        if(value != NULL && value[0] != '\0'){
            return atoi(value);
        }
    }
    return -1;
}

static int get_local_rank(MPI_Comm comm){
    int env_rank = read_local_rank_from_env();
    if(env_rank >= 0){
        return env_rank;
    }

    MPI_Comm local_comm;
    int local_rank = 0;
    MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    MPI_Comm_rank(local_comm, &local_rank);
    MPI_Comm_free(&local_comm);
    return local_rank;
}

static int select_local_cuda_device(MPI_Comm comm, int rank, int *local_rank_out){
    int local_rank = get_local_rank(comm);
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if(err != cudaSuccess){
        fprintf(stderr, "[Rank %d] cudaGetDeviceCount failed: %s\n",
                rank, cudaGetErrorString(err));
        MPI_Abort(comm, 1);
    }
    if(device_count <= 0){
        if (rank == 0) {
            printf("No CUDA devices found.\n");
        }
        MPI_Abort(comm, 1);
    }
    if (local_rank >= device_count && getenv("ALLOW_GPU_OVERSUBSCRIBE") == NULL) {
        fprintf(stderr,
                "[Rank %d local_rank %d] only %d CUDA device(s) are visible. "
                "Reduce ranks per node or set ALLOW_GPU_OVERSUBSCRIBE=1.\n",
                rank, local_rank, device_count);
        MPI_Abort(comm, 1);
    }

    int device = local_rank % device_count;
    err = cudaSetDevice(device);
    if(err != cudaSuccess){
        fprintf(stderr,
                "[Rank %d local_rank %d] cudaSetDevice(%d) failed: %s\n",
                rank, local_rank, device, cudaGetErrorString(err));
        MPI_Abort(comm, 1);
    }

    if (local_rank_out != NULL) {
        *local_rank_out = local_rank;
    }
    return device;
}

#endif
