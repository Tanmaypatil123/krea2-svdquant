"""Generic Triton dynamic activation INT4 quantization smoke test.

No B200/Blackwell-specific instructions. This is a normal Triton kernel intended
for generic NVIDIA GPUs as the first W4A4 building block:

  X_fp16 [M, K] -> X_int4_packed [M, K/2] + row_scale [M]

Submit examples:
  kernelide submit scripts/kernelide_activation_quant_triton.py --language triton --gpu H100 --timeout 120
  kernelide submit scripts/kernelide_activation_quant_triton.py --language triton --gpu A100-80GB --timeout 120
"""

from __future__ import annotations

import argparse
import time

import torch
import triton
import triton.language as tl


@triton.jit
def rowwise_int4_quant_kernel(
    x_ptr,
    packed_ptr,
    scale_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_K)
    x = tl.load(x_ptr + row * K + offs, mask=offs < K, other=0.0).to(tl.float32)
    absmax = tl.max(tl.abs(x), axis=0)
    scale = tl.maximum(absmax / 7.0, 1.0e-8)
    pair = tl.arange(0, BLOCK_K // 2)
    offs0 = pair * 2
    offs1 = offs0 + 1
    x0 = tl.load(x_ptr + row * K + offs0, mask=offs0 < K, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + row * K + offs1, mask=offs1 < K, other=0.0).to(tl.float32)
    s0 = x0 / scale
    s1 = x1 / scale
    r0 = tl.where(s0 >= 0.0, tl.floor(s0 + 0.5), tl.ceil(s0 - 0.5))
    r1 = tl.where(s1 >= 0.0, tl.floor(s1 + 0.5), tl.ceil(s1 - 0.5))
    q0 = tl.clamp(r0, -8, 7).to(tl.int16)
    q1 = tl.clamp(r1, -8, 7).to(tl.int16)
    lo = (q0 + 8).to(tl.uint8)
    hi = (q1 + 8).to(tl.uint8)
    packed = lo | (hi << 4)
    tl.store(packed_ptr + row * (K // 2) + pair, packed, mask=pair < (K // 2))
    tl.store(scale_ptr + row, scale)


@triton.jit
def rowwise_int4_dequant_kernel(
    packed_ptr,
    scale_ptr,
    y_ptr,
    M: tl.constexpr,
    K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    row = tl.program_id(0)
    pair = tl.arange(0, BLOCK_K // 2)
    packed = tl.load(packed_ptr + row * (K // 2) + pair, mask=pair < (K // 2), other=0)
    lo = ((packed & 0x0F).to(tl.int16) - 8).to(tl.float32)
    hi = (((packed >> 4) & 0x0F).to(tl.int16) - 8).to(tl.float32)
    scale = tl.load(scale_ptr + row).to(tl.float32)
    tl.store(y_ptr + row * K + pair * 2, lo * scale, mask=(pair * 2) < K)
    tl.store(y_ptr + row * K + pair * 2 + 1, hi * scale, mask=(pair * 2 + 1) < K)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=1024)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    assert args.k % 2 == 0
    # This smoke kernel handles one row per program with BLOCK_K=next_power_of_2(K).
    block_k = triton.next_power_of_2(args.k)
    assert block_k <= 8192, "increase test carefully for larger K"

    x = torch.randn((args.m, args.k), device="cuda", dtype=torch.float16)
    packed = torch.empty((args.m, args.k // 2), device="cuda", dtype=torch.uint8)
    scales = torch.empty((args.m,), device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    grid = (args.m,)
    rowwise_int4_quant_kernel[grid](x, packed, scales, args.m, args.k, BLOCK_K=block_k, num_warps=8)
    rowwise_int4_dequant_kernel[grid](packed, scales, y, args.m, args.k, BLOCK_K=block_k, num_warps=8)
    torch.cuda.synchronize()

    ref_scale = x.float().abs().amax(dim=1).clamp_min(1e-8) / 7.0
    ref_q = torch.round(x.float() / ref_scale[:, None]).clamp(-8, 7)
    ref_y = ref_q * ref_scale[:, None]
    diff = (y.float() - ref_y).abs()
    max_err = diff.max().item()
    mean_err = diff.mean().item()

    for _ in range(args.warmup):
        rowwise_int4_quant_kernel[grid](x, packed, scales, args.m, args.k, BLOCK_K=block_k, num_warps=8)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.iters):
        rowwise_int4_quant_kernel[grid](x, packed, scales, args.m, args.k, BLOCK_K=block_k, num_warps=8)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / args.iters
    gbps = (args.m * args.k * 2 + args.m * args.k / 2 + args.m * 4) / elapsed / 1e9

    print(f"GPU={torch.cuda.get_device_name()} SM={torch.cuda.get_device_capability()} triton={triton.__version__}")
    print(
        f"rowwise_int4_quant M={args.m} K={args.k} time_ms={elapsed*1e3:.3f} "
        f"effective_GBps={gbps:.2f} max_err={max_err:.3e} mean_err={mean_err:.3e}"
    )
    # Rounding mode can differ from torch.round at exact .5 ties; require bounded
    # dequant error instead of bit-exact equality against PyTorch's tie-to-even.
    assert mean_err < 1.0e-3 and max_err < 1.0
    print("PASS generic Triton activation INT4 quant smoke")


if __name__ == "__main__":
    main()
