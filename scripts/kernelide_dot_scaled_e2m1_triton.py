"""KernelIDE B200 Triton tl.dot_scaled smoke/benchmark for FP16 x E2M1.

This probes the Blackwell fast path we want for the next SVDQuant generation.
Unlike the scalar-unpack W4A16 baseline, this uses Triton's scaled-dot interface
with RHS packed FP4/e2m1 data. It is not yet a calibrated Krea2 quantizer; it is
a B200 kernel capability + shape benchmark.

Submit:
  kernelide submit scripts/kernelide_dot_scaled_e2m1_triton.py --language triton --gpu B200 --timeout 180
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def fp16_e2m1_dot_scaled_kernel(
    a_ptr,
    b_e2m1_ptr,
    c_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    offs_k_packed = tl.arange(0, BLOCK_K // 2)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k0 in range(0, K, BLOCK_K):
        k = k0 + offs_k
        k_packed = (k0 // 2) + offs_k_packed
        a = tl.load(
            a_ptr + offs_m[:, None] * K + k[None, :],
            mask=(offs_m[:, None] < M) & (k[None, :] < K),
            other=0.0,
        )
        # For rhs_format="e2m1" with rhs_k_pack=True, the physical K dimension is K/2.
        b = tl.load(
            b_e2m1_ptr + k_packed[:, None] * N + offs_n[None, :],
            mask=(k_packed[:, None] < (K // 2)) & (offs_n[None, :] < N),
            other=0,
        )
        acc = tl.dot_scaled(
            a,
            None,
            "fp16",
            b,
            None,
            "e2m1",
            acc=acc,
            rhs_k_pack=True,
        )
    tl.store(
        c_ptr + offs_m[:, None] * N + offs_n[None, :],
        acc,
        mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
    )


def run_case(m: int, n: int, k: int, warmup: int, iters: int):
    a = torch.randn((m, k), device="cuda", dtype=torch.float16)
    # Random e2m1 bytes; quantizer/calibrated scales come next.
    b = torch.randint(0, 256, (k // 2, n), device="cuda", dtype=torch.uint8)
    c = torch.empty((m, n), device="cuda", dtype=torch.float32)
    grid = (triton.cdiv(m, 128), triton.cdiv(n, 64))
    kwargs = dict(M=m, N=n, K=k, BLOCK_M=128, BLOCK_N=64, BLOCK_K=128, num_warps=4, num_stages=3)
    fp16_e2m1_dot_scaled_kernel[grid](a, b, c, **kwargs)
    torch.cuda.synchronize()
    assert torch.isfinite(c).all()
    for _ in range(warmup):
        fp16_e2m1_dot_scaled_kernel[grid](a, b, c, **kwargs)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fp16_e2m1_dot_scaled_kernel[grid](a, b, c, **kwargs)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / iters
    tflops = (2.0 * m * n * k) / elapsed / 1e12
    print(
        f"case M={m} N={n} K={k} time_ms={elapsed*1e3:.3f} "
        f"dense_equiv_tflops={tflops:.2f} mean_abs={c.float().abs().mean().item():.3e}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=512)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--n", type=int, default=4096)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()
    print(f"GPU={torch.cuda.get_device_name()} SM={torch.cuda.get_device_capability()} triton={triton.__version__}")
    run_case(args.m, args.n, args.k, args.warmup, args.iters)
    if args.n != 16384:
        run_case(args.m, 16384, args.k, max(2, args.warmup // 2), max(5, args.iters // 4))
    print("PASS tl.dot_scaled fp16 x e2m1 B200 smoke")


if __name__ == "__main__":
    main()
