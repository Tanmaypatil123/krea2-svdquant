"""KernelIDE B200 Triton smoke/benchmark for packed W4A16 linear.

This is the first real SVDQuant building block:
  X_fp16/bf16 [M, K] x packed W_int4 [N, K] -> Y_fp16 [M, N]

It validates against PyTorch dequantized linear and prints timing for Krea2-like
projection shapes. Submit with:
  kernelide submit scripts/kernelide_w4a16_linear_triton.py --language triton --gpu B200 --timeout 180
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def w4a16_linear_kernel(
    x_ptr,
    w_packed_ptr,
    w_scale_ptr,
    y_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        x = tl.load(
            x_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )

        # Packed signed int4 uses storage nibble = q + 8, q in [-8, 7].
        packed_k = k // 2
        packed = tl.load(
            w_packed_ptr + offs_n[:, None] * (K // 2) + packed_k[None, :],
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0,
        )
        low = packed & 0x0F
        high = (packed >> 4) & 0x0F
        nibble = tl.where((k[None, :] & 1) == 0, low, high)
        q = nibble.to(tl.int16) - 8

        scale = tl.load(
            w_scale_ptr + offs_n[:, None] * (K // GROUP_SIZE) + (k[None, :] // GROUP_SIZE),
            mask=(offs_n[:, None] < N) & (k[None, :] < K),
            other=0.0,
        )
        # tl.dot requires same dtype operands. Convert dequantized weights to fp16;
        # accumulation remains fp32. This is the W4A16 Tensor Core path.
        w = (q.to(tl.float32) * scale).to(tl.float32).to(tl.float16)
        acc += tl.dot(x, tl.trans(w), input_precision="tf32")

    tl.store(
        y_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def quantize_pack_weight(w: torch.Tensor, group_size: int):
    assert w.ndim == 2
    n, k = w.shape
    assert k % group_size == 0
    grouped = w.float().view(n, k // group_size, group_size)
    scales = grouped.abs().amax(dim=-1).clamp_min(1e-8) / 7.0
    q = torch.round(grouped / scales.unsqueeze(-1)).clamp(-8, 7).to(torch.int8).view(n, k)
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    packed = u[:, 0::2] | (u[:, 1::2] << 4)
    return packed.contiguous(), scales.contiguous(), q


def dequant_weight(q: torch.Tensor, scales: torch.Tensor, group_size: int):
    n, k = q.shape
    return (q.view(n, k // group_size, group_size).float() * scales.unsqueeze(-1)).view(n, k)


def run_case(m: int, n: int, k: int, group_size: int, dtype: torch.dtype, warmup: int, iters: int):
    torch.manual_seed(0)
    x = torch.randn((m, k), device="cuda", dtype=dtype) / (k**0.5)
    w = torch.randn((n, k), device="cuda", dtype=dtype) / (k**0.5)
    w_packed, scales, q = quantize_pack_weight(w, group_size)
    y = torch.empty((m, n), device="cuda", dtype=torch.float32)

    grid = (triton.cdiv(m, 16), triton.cdiv(n, 32))
    kernel_kwargs = dict(
        M=m,
        N=n,
        K=k,
        GROUP_SIZE=group_size,
        BLOCK_M=16,
        BLOCK_N=32,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )

    # Correctness against dequantized PyTorch reference.
    w_deq = dequant_weight(q, scales, group_size).to(dtype)
    ref = x.float() @ w_deq.float().t()
    w4a16_linear_kernel[grid](x, w_packed, scales, y, **kernel_kwargs)
    torch.cuda.synchronize()
    diff = (y - ref).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()
    rel_l2 = (torch.linalg.vector_norm(y - ref) / torch.linalg.vector_norm(ref)).item()

    for _ in range(warmup):
        w4a16_linear_kernel[grid](x, w_packed, scales, y, **kernel_kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        w4a16_linear_kernel[grid](x, w_packed, scales, y, **kernel_kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    # Count dense-equivalent FMA FLOPs, useful for comparing progress across kernels.
    tflops = (2.0 * m * n * k) / elapsed / 1e12
    print(
        f"case M={m} N={n} K={k} dtype={str(dtype).replace('torch.', '')} "
        f"time_ms={elapsed*1e3:.3f} dense_equiv_tflops={tflops:.2f} "
        f"max_err={max_err:.3e} mean_err={mean_err:.3e} rel_l2={rel_l2:.3e}"
    )
    assert rel_l2 < 3e-3 or max_err < 2e-2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--group-size", type=int, default=128)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    args = ap.parse_args()

    print(f"GPU={torch.cuda.get_device_name()} SM={torch.cuda.get_device_capability()} triton={triton.__version__}")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    run_case(args.m, args.n, args.k, args.group_size, dtype, args.warmup, args.iters)
    # A Krea2 MLP-ish projection, kept moderate so KernelIDE smoke stays quick.
    if args.n != 16384:
        run_case(args.m, 16384, args.k, args.group_size, dtype, max(2, args.warmup // 2), max(5, args.iters // 4))
    print("PASS packed W4A16 Triton linear smoke")


if __name__ == "__main__":
    main()
