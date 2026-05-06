#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * Compile: mpicc -o ring_test ring_test.c
 * Run:     mpirun -np 2 ./ring_test
 */

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) {
            printf("Need at least 2 processes. Got %d.\n", size);
        }
        MPI_Finalize();
        return 1;
    }

    // Ring neighbors
    int right = (rank + 1) % size;
    int left  = (rank - 1 + size) % size;

    printf("[Rank %d] neighbors: left=%d, right=%d\n", rank, left, right);

    // Each process has a small buffer to send
    int buf_size = 4;
    double *send_buf = malloc(buf_size * sizeof(double));
    double *recv_buf = malloc(buf_size * sizeof(double));

    //Fill send buffer with rank-specific values
    for (int i = 0; i < buf_size; i++) {
        send_buf[i] = rank * 100.0 + i;
    }

    //Ring step: send to right, receive from left
    MPI_Status status;
    double t_start = MPI_Wtime();

    MPI_Sendrecv(send_buf, buf_size, MPI_DOUBLE, right, 0,
                 recv_buf, buf_size, MPI_DOUBLE, left,  0,
                 MPI_COMM_WORLD, &status);

    double t_end = MPI_Wtime();

    printf("[Rank %d] sent to %d, received from %d in %.3f ms\n",
           rank, right, left, (t_end - t_start) * 1000.0);
    printf("[Rank %d] recv_buf = [%.0f, %.0f, %.0f, %.0f]\n",
           rank, recv_buf[0], recv_buf[1], recv_buf[2], recv_buf[3]);

    free(send_buf);
    free(recv_buf);

    MPI_Finalize();
    return 0;
}