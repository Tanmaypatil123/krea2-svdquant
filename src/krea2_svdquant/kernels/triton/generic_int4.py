from __future__ import annotations

import torch
import torch.nn.functional as F

from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import guard for CPU dev machines
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _w4a16_residual_matmul_kernel(
        X,
        WQ_PACKED,
        WS,
        SMOOTH,
        Y,
        M: tl.constexpr,
        K: tl.constexpr,
        N: tl.constexpr,
        GROUP: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Fused residual W4A16 matmul for SVDQuant.

        Computes ``Y = (X / smooth) @ dequant(WQ_PACKED, WS).T`` without
        materializing dequantized weights. WQ is signed int4 packed along K as
        uint8 nibbles with zero-point 8, shape [N, padded_K // 2]. WS is per
        output row / K-group scale, shape [N, padded_K // GROUP].
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        packed_stride = tl.cdiv(K, 2)
        scale_stride = tl.cdiv(K, GROUP)

        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(
                X + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            smooth = tl.load(SMOOTH + k, mask=k < K, other=1.0).to(tl.float32)
            x = x / smooth[None, :]

            packed = tl.load(
                WQ_PACKED + offs_n[:, None] * packed_stride + (k[None, :] // 2),
                mask=(offs_n[:, None] < N) & (k[None, :] < K),
                other=0,
            )
            lo = packed & 0x0F
            hi = (packed >> 4) & 0x0F
            q_u = tl.where((k[None, :] & 1) == 0, lo, hi)
            q = q_u.to(tl.float32) - 8.0

            scales = tl.load(
                WS + offs_n[:, None] * scale_stride + (k[None, :] // GROUP),
                mask=(offs_n[:, None] < N) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            w = q * scales
            acc += tl.dot(x, tl.trans(w), out_dtype=tl.float32)

        tl.store(
            Y + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )


def _flatten_for_linear(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
    if x.ndim < 2:
        raise ValueError(f"expected [..., K] activation, got shape {tuple(x.shape)}")
    orig = tuple(x.shape[:-1])
    return x.reshape(-1, x.shape[-1]).contiguous(), orig


def _residual_triton(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    if triton is None or not x.is_cuda:
        raise RuntimeError("Triton CUDA backend is unavailable")
    if not state.qweight_packed:
        raise RuntimeError("Triton backend expects packed uint8 qweight")

    x2d, out_prefix = _flatten_for_linear(x)
    m, k = x2d.shape
    n, packed_k = state.qweight.shape
    padded_k = int(state.padded_in_features or packed_k * 2)
    if k > padded_k:
        raise ValueError(f"activation K={k} exceeds packed weight K={padded_k}")
    # Krea2 checkpoint K is group-aligned; use logical padded K so packed/scales
    # indexing matches converter layout. Pad activation with zeros only for rare
    # non-padded inputs.
    if k != padded_k:
        x_work = torch.zeros((m, padded_k), device=x.device, dtype=x.dtype)
        x_work[:, :k] = x2d
    else:
        x_work = x2d

    qweight = state.qweight.to(device=x.device)
    scales = state.weight_scales.to(device=x.device)
    smooth = state.smooth_scale.to(device=x.device, dtype=torch.float32)
    if smooth.numel() != padded_k:
        smooth_work = torch.ones((padded_k,), device=x.device, dtype=torch.float32)
        smooth_work[: smooth.numel()] = smooth
        smooth = smooth_work

    y = torch.empty((m, n), device=x.device, dtype=x.dtype)
    # Conservative tiles: portable on Ada/Hopper/Blackwell. This scalar-unpack
    # kernel prioritizes low memory and correctness; B200 speed path should use
    # dot_scaled/Gluon once calibrated FP4 layout is ready.
    grid = (triton.cdiv(m, 16), triton.cdiv(n, 64))
    _w4a16_residual_matmul_kernel[grid](
        x_work,
        qweight,
        scales,
        smooth,
        y,
        M=m,
        K=padded_k,
        N=n,
        GROUP=state.group_size,
        BLOCK_M=16,
        BLOCK_N=64,
        BLOCK_K=64,
        num_warps=4,
        num_stages=3,
    )
    return y.reshape(*out_prefix, n)


def svdquant_linear_triton_generic(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    """Portable fused SVDQuant runtime for CUDA GPUs.

    The residual branch uses a Triton kernel that fuses smooth scaling, INT4
    unpack, dequantization, and matmul. The low-rank branch remains PyTorch GEMM
    because it is already compact (rank 64/128) and avoids a large dequantized
    weight materialization. Falls back to the PyTorch reference if Triton is not
    available or the checkpoint is not packed.
    """
    try:
        y = _residual_triton(x, state)
        smooth = state.smooth_scale.to(device=x.device, dtype=x.dtype)
        x_hat = x / smooth
        l1 = state.l1.to(device=x.device, dtype=x.dtype)
        l2 = state.l2.to(device=x.device, dtype=x.dtype)
        y.add_(F.linear(F.linear(x_hat, l2), l1))
        if state.bias is not None:
            y.add_(state.bias.to(device=x.device, dtype=x.dtype))
        return y
    except Exception:
        return svdquant_linear_sim(x, state)
