from __future__ import annotations

import torch

from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - import guard for CPU dev machines
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _dequant_matmul_kernel(
        X, WQ, WS, L1, L2, SMOOTH, Y,
        M: tl.constexpr, K: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
        GROUP: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ):
        # Development kernel placeholder: currently only a shape/smoke target.
        # Production path will unpack WQ int4 and fuse low-rank add. Keep this file
        # importable while the PyTorch simulation remains the correctness reference.
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k0 in range(0, K, BLOCK_K):
            offs_k = k0 + tl.arange(0, BLOCK_K)
            x = tl.load(X + offs_m[:, None] * K + offs_k[None, :], mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)
            # Placeholder loads W as if already dequantized contiguous BF16 when used for experiments.
            w = tl.load(WQ + offs_n[:, None] * K + offs_k[None, :], mask=(offs_n[:, None] < N) & (offs_k[None, :] < K), other=0.0)
            acc += tl.dot(x, tl.trans(w))
        tl.store(Y + offs_m[:, None] * N + offs_n[None, :], acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


def svdquant_linear_triton_generic(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    """Generic backend entry point.

    For now this deliberately falls back to the PyTorch simulation. The Triton JIT
    kernel above is kept as a scaffold/smoke target while we finalize the exact int4
    memory layout and activation quantization fusion.
    """
    return svdquant_linear_sim(x, state)
