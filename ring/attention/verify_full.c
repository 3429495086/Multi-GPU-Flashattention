/*
 * verify_full.c
 * Full element-wise verification of Ring Attention output
 *
 * Reads binary dump from ring code, computes reference,
 * compares all values, prints summary only.
 *
 * Compile: gcc -O2 -o verify_full verify_full.c -lm
 * Run:     ./verify_full [output_dir] [num_gpus] [kv_size_bytes]
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>

void full_attention(
    const float *Q, int seq_q,
    const float *K, const float *V, int total_k,
    int dim, float *output
) {
    float *scores = malloc(total_k * sizeof(float));
    for (int i = 0; i < seq_q; i++) {
        float row_max = -FLT_MAX;
        for (int j = 0; j < total_k; j++) {
            float s = 0.0f;
            for (int d = 0; d < dim; d++)
                s += Q[i * dim + d] * K[j * dim + d];
            scores[j] = s;
            if (s > row_max) row_max = s;
        }
        float row_sum = 0.0f;
        for (int d = 0; d < dim; d++)
            output[i * dim + d] = 0.0f;
        for (int j = 0; j < total_k; j++) {
            float w = expf(scores[j] - row_max);
            row_sum += w;
            for (int d = 0; d < dim; d++)
                output[i * dim + d] += w * V[j * dim + d];
        }
        for (int d = 0; d < dim; d++)
            output[i * dim + d] /= row_sum;
    }
    free(scores);
}

int main(int argc, char **argv) {
    const char *output_dir = (argc > 1) ? argv[1] : "bench";
    int num_gpus = (argc > 2) ? atoi(argv[2]) : 2;
    long long kv_size_bytes = (argc > 3) ? atoll(argv[3]) : 262144;
    int head_dim = 64;

    if (num_gpus <= 0 ||
        kv_size_bytes <= 0 ||
        (kv_size_bytes % (head_dim * (long long)sizeof(float))) != 0) {
        fprintf(stderr,
                "Usage: %s [output_dir] [num_gpus] [kv_size_bytes]\n"
                "kv_size_bytes must be a positive multiple of %zu bytes.\n",
                argv[0], (size_t)(head_dim * sizeof(float)));
        return 1;
    }

    int seq_k = (int)(kv_size_bytes / (head_dim * (long long)sizeof(float)));
    int seq_q = seq_k;
    int dim = head_dim;
    int total_k = seq_k * num_gpus;
    int total_elems = seq_q * dim;

    // build full KV (same init as ring code)
    float *K_full = malloc(total_k * dim * sizeof(float));
    float *V_full = malloc(total_k * dim * sizeof(float));
    for (int r = 0; r < num_gpus; r++) {
        for (int i = 0; i < seq_k * dim; i++) {
            float val = r * 1000.0f + (float)i;
            K_full[r * seq_k * dim + i] = val;
            V_full[r * seq_k * dim + i] = val;
        }
    }

    float *Q      = malloc(total_elems * sizeof(float));
    float *ref    = malloc(total_elems * sizeof(float));
    float *ring   = malloc(total_elems * sizeof(float));

    int all_pass = 1;

    for (int r = 0; r < num_gpus; r++) {
        // build Q for this rank 
        for (int i = 0; i < total_elems; i++)
            Q[i] = (r * 500.0f + (float)(i % dim) * 0.01f)
                   * 0.001f;

        // compute reference
        full_attention(Q, seq_q, K_full, V_full,
                       total_k, dim, ref);

        // read ring output
        char fname[512];
        snprintf(fname, sizeof(fname), "%s/ring_output_rank%d.bin",
                 output_dir, r);
        FILE *fp = fopen(fname, "rb");
        if (!fp) {
            printf("=== Rank %d === SKIP (file not found)\n"
                   "  Run ring code first to generate %s\n\n",
                   r, fname);
            all_pass = 0;
            continue;
        }
        size_t read = fread(ring, sizeof(float), total_elems, fp);
        fclose(fp);
        if ((int)read != total_elems) {
            printf("=== Rank %d === FAIL "
                   "(read %zu of %d floats)\n\n",
                   r, read, total_elems);
            all_pass = 0;
            continue;
        }

        // full element-wise comparison
        float max_abs = 0.0f;
        float max_rel = 0.0f;
        int   worst_idx = 0;
        int   nan_count = 0;

        for (int i = 0; i < total_elems; i++) {
            if (isnan(ring[i]) || isinf(ring[i])) {
                nan_count++;
                continue;
            }
            float abs_err = fabsf(ring[i] - ref[i]);
            float rel_err = (fabsf(ref[i]) > 1e-6f)
                          ? abs_err / fabsf(ref[i]) : 0.0f;
            if (abs_err > max_abs) {
                max_abs = abs_err;
                worst_idx = i;
            }
            if (rel_err > max_rel)
                max_rel = rel_err;
        }

        float atol = 0.05f;
        float rtol = 1e-5f;
        int pass = (nan_count == 0) && (max_abs < atol || max_rel < rtol);
        if (!pass) all_pass = 0;

        printf("=== Rank %d === %s\n", r,
               pass ? "PASS" : "FAIL");
        printf("  Elements compared : %d\n", total_elems);
        printf("  Max absolute error: %.6f\n", max_abs);
        printf("  Max relative error: %.8f\n", max_rel);
        printf("  Worst index       : %d (ref=%.4f, ring=%.4f)\n",
               worst_idx, ref[worst_idx], ring[worst_idx]);
        printf("  NaN/Inf count     : %d\n\n", nan_count);
    }

    printf("============================\n");
    printf("Overall: %s\n", all_pass ? "ALL PASS" : "SOME FAILED");

    free(Q); free(ref); free(ring);
    free(K_full); free(V_full);
    return all_pass ? 0 : 1;
}
