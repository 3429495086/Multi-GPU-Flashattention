#include <stdio.h>
#include <cuda_runtime.h>
int main() {
    int count;
    cudaGetDeviceCount(&count);
    for (int i = 0; i < count; i++)
        for (int j = 0; j < count; j++) {
            if (i == j) continue;
            int can;
            cudaDeviceCanAccessPeer(&can, i, j);
            printf("GPU %d -> GPU %d: %s\n",
                   i, j, can ? "YES" : "NO");
        }
    return 0;
}
