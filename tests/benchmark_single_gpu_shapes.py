import argparse
import csv
from pathlib import Path
from statistics import mean, pstdev

import torch

from flash_attn.flash_attn_interface import flash_attn_unpadded_func


# some common square / rectangular shapes for quick benchmark
DEFAULT_SHAPES = [
    "256x256",
    "512x512",
    "1024x1024",
    "2048x2048",
    "4096x4096",
    "1024x64",
    "64x1024",
]


def parse_shape(shape_text):
    # expect input like "1024x64"
    parts = shape_text.lower().split("x")
    if len(parts) != 2:
        raise ValueError(f"Invalid shape '{shape_text}', expected AxB.")
    seq_q = int(parts[0])
    seq_k = int(parts[1])
    if seq_q <= 0 or seq_k <= 0:
        raise ValueError(f"Invalid shape '{shape_text}', dimensions must be positive.")
    return seq_q, seq_k


def causal_active_elements(seq_q, seq_k):
    # count how many attention entries are still valid under causal mask
    if seq_q <= seq_k:
        return seq_q * (seq_q + 1) // 2

    # if q is longer, extra rows can already see all available k positions
    triangular = seq_k * (seq_k + 1) // 2
    tail_rows = seq_q - seq_k
    return triangular + tail_rows * seq_k


def attention_active_elements(seq_q, seq_k, causal):
    return causal_active_elements(seq_q, seq_k) if causal else seq_q * seq_k


def make_inputs(seq_q, seq_k, batch_size, num_heads, head_dim, dtype, device):
    # flash_attn_unpadded_func uses flattened q/k/v plus cumulative lengths
    total_q = batch_size * seq_q
    total_k = batch_size * seq_k

    q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device=device)
    k = torch.randn(total_k, num_heads, head_dim, dtype=dtype, device=device)
    v = torch.randn(total_k, num_heads, head_dim, dtype=dtype, device=device)
    cu_seqlens_q = torch.arange(
        0, (batch_size + 1) * seq_q, seq_q, dtype=torch.int32, device=device
    )
    cu_seqlens_k = torch.arange(
        0, (batch_size + 1) * seq_k, seq_k, dtype=torch.int32, device=device
    )

    return q, k, v, cu_seqlens_q, cu_seqlens_k


def benchmark_shape(seq_q, seq_k, args):
    device = f"cuda:{args.device}"
    dtype = getattr(torch, args.dtype)

    torch.cuda.set_device(args.device)

    # build inputs once, then only time the actual kernel call
    q, k, v, cu_seqlens_q, cu_seqlens_k = make_inputs(
        seq_q, seq_k, args.batch_size, args.num_heads, args.head_dim, dtype, device
    )

    with torch.inference_mode():
        # warmup first, so first-call overhead does not affect timing
        for _ in range(args.warmup):
            flash_attn_unpadded_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seq_q,
                seq_k,
                0.0,
                causal=args.causal,
            )
        torch.cuda.synchronize(args.device)

        # reset here so warmup allocations do not count into peak memory
        torch.cuda.reset_peak_memory_stats(args.device)
        times_ms = []
        for _ in range(args.repeats):
            # cuda events are more suitable than time.time() for gpu kernels
            # (because if use time.time(), more like CPU-wait time)
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            flash_attn_unpadded_func(
                q,
                k,
                v,
                cu_seqlens_q,
                cu_seqlens_k,
                seq_q,
                seq_k,
                0.0,
                causal=args.causal,
            )
            end.record()
            end.synchronize()
            times_ms.append(start.elapsed_time(end))

    peak_mem_mb = torch.cuda.max_memory_allocated(args.device) / (1024 ** 2)
    # active count is computed analytically from shape + mask type
    active_per_seq = attention_active_elements(seq_q, seq_k, args.causal)
    total_active = active_per_seq * args.batch_size

    return {
        "seq_q": seq_q,
        "seq_k": seq_k,
        "batch_size": args.batch_size,
        "num_heads": args.num_heads,
        "head_dim": args.head_dim,
        "dtype": args.dtype,
        "causal": args.causal,
        "active_per_seq": active_per_seq,
        "total_active": total_active,
        "avg_ms": mean(times_ms),
        "std_ms": pstdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min_ms": min(times_ms),
        "max_ms": max(times_ms),
        "peak_mem_mb": peak_mem_mb,
    }


def print_results(rows):
    # print a small table for terminal viewing
    header = (
        f"{'shape':>12} {'active/seq':>12} {'avg_ms':>10} "
        f"{'std_ms':>10} {'min_ms':>10} {'max_ms':>10} {'peak_mem(MB)':>14}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        shape = f"{row['seq_q']}x{row['seq_k']}"
        print(
            f"{shape:>12} {row['active_per_seq']:>12d} {row['avg_ms']:>10.3f} "
            f"{row['std_ms']:>10.3f} {row['min_ms']:>10.3f} "
            f"{row['max_ms']:>10.3f} {row['peak_mem_mb']:>14.1f}"
        )


def write_csv(rows, csv_path):
    # also save everything in csv for later plotting / report tables
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seq_q",
        "seq_k",
        "batch_size",
        "num_heads",
        "head_dim",
        "dtype",
        "causal",
        "active_per_seq",
        "total_active",
        "avg_ms",
        "std_ms",
        "min_ms",
        "max_ms",
        "peak_mem_mb",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Single-GPU FlashAttention benchmark for different matrix shapes."
    )
    parser.add_argument("--device", type=int, default=0, help="CUDA device index.")
    parser.add_argument(
        "--shapes",
        nargs="+",
        default=DEFAULT_SHAPES,
        help="Shapes to benchmark, e.g. 512x512 1024x64.",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Benchmark causal attention. Use --no-causal for full attention.",
    )
    parser.add_argument(
        "--csv-out",
        type=Path,
        default=Path("results/single_gpu_shapes.csv"),
        help="Where to write CSV results.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    rows = []
    for shape_text in args.shapes:
        seq_q, seq_k = parse_shape(shape_text)
        try:
            row = benchmark_shape(seq_q, seq_k, args)
        except RuntimeError as exc:
            # e.g. OOM: skip this shape and continue with the rest
            print(f"Skipping {shape_text}: {exc}")
            torch.cuda.empty_cache()
            continue
        rows.append(row)

        # clear cache between shapes to reduce cross-shape interference
        torch.cuda.empty_cache()

    print_results(rows)
    write_csv(rows, args.csv_out)
    print(f"\nCSV saved to: {args.csv_out}")


if __name__ == "__main__":
    main()
