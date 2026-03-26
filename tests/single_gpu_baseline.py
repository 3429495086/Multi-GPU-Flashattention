import torch
import time
from flash_attn.flash_attn_interface import flash_attn_unpadded_func


def run_test(seq_len, device=0):
    torch.cuda.set_device(device)

    # keep this baseline setting simple and fixed
    batch_size = 2
    num_heads = 8
    head_dim = 64

    # flash-attn unpadded interface uses flattened q/k/v
    q = torch.randn(batch_size * seq_len, num_heads, head_dim,
                    dtype=torch.float16, device=f"cuda:{device}")
    k = torch.randn(batch_size * seq_len, num_heads, head_dim,
                    dtype=torch.float16, device=f"cuda:{device}")
    v = torch.randn(batch_size * seq_len, num_heads, head_dim,
                    dtype=torch.float16, device=f"cuda:{device}")

    cu_seqlens = torch.arange(
        0, (batch_size + 1) * seq_len, seq_len,
        dtype=torch.int32, device=f"cuda:{device}"
    )

    # warmup first, so later timing is more stable
    for _ in range(3):
        out = flash_attn_unpadded_func(
            q, k, v, cu_seqlens, cu_seqlens,
            seq_len, seq_len, 0.0, causal=True
        )
    torch.cuda.synchronize()

    # use a simple average over 10 runs
    start = time.time()
    for _ in range(10):
        out = flash_attn_unpadded_func(
            q, k, v, cu_seqlens, cu_seqlens,
            seq_len, seq_len, 0.0, causal=True
        )
    torch.cuda.synchronize()

    elapsed = (time.time() - start) / 10
    print(f"seq_len={seq_len}, time={elapsed * 1000:.2f} ms")
    return elapsed


if __name__ == "__main__":
    print("=== Single-GPU FlashAttention baseline ===")
    for seq_len in [512, 1024, 2048]:
        run_test(seq_len)
