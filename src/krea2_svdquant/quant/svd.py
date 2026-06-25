from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn.functional as F

from .pack import dequantize_symmetric_int4, quantize_symmetric_int4


@dataclass
class SVDQuantLinearState:
    smooth_scale: torch.Tensor
    qweight: torch.Tensor
    weight_scales: torch.Tensor
    l1: torch.Tensor
    l2: torch.Tensor
    bias: torch.Tensor | None
    group_size: int
    original_shape: tuple[int, int]


def compute_smooth_scale(x_absmax: torch.Tensor, weight: torch.Tensor, min_value=0.25, max_value=4.0):
    """Compute SmoothQuant-like migration scale for a linear weight [out, in]."""
    w_absmax = weight.abs().amax(dim=0).clamp_min(1e-8)
    s = torch.sqrt(x_absmax.float().clamp_min(1e-8) / w_absmax.float())
    return s.clamp(min_value, max_value).to(weight.device, dtype=weight.dtype)


def svd_lowrank(
    weight: torch.Tensor,
    rank: int,
    *,
    oversample: int = 8,
    niter: int = 4,
    device: str | torch.device | None = None,
    exact: bool = False,
):
    """Return L1 [out, rank], L2 [rank, in] for weight [out, in].

    Uses randomized/truncated SVD by default, which is dramatically cheaper than a
    full ``torch.linalg.svd`` for the huge Krea2 linears (e.g. 6144x16384) while
    recovering the leading ``rank`` singular directions to high accuracy.

    Args:
        weight: 2D weight tensor [out_features, in_features].
        rank: target low-rank dimension. Clamped to ``min(out, in)``.
        oversample: extra probing columns for the randomized range finder. A small
            oversample (5-10) markedly improves accuracy of the top singular values.
        niter: number of power/subspace iterations. More iterations sharpen the
            approximation when the spectrum decays slowly; 2-4 is usually enough.
        device: device to run the decomposition on (e.g. ``"cuda"``). Defaults to the
            weight's own device. The returned tensors are moved back to the weight's
            original device/dtype.
        exact: force a full ``torch.linalg.svd`` (useful for tiny layers or to sanity
            check the randomized path). Also used automatically for very small matrices
            where randomized SVD has no benefit.
    """
    if weight.ndim != 2:
        raise ValueError("svd_lowrank expects a 2D [out, in] weight")
    orig_device, orig_dtype = weight.device, weight.dtype
    compute_device = torch.device(device) if device is not None else orig_device
    w = weight.to(device=compute_device, dtype=torch.float32)

    min_dim = min(w.shape)
    rank = max(1, min(int(rank), min_dim))
    # q is the number of columns the randomized range finder probes with.
    q = min(rank + max(0, int(oversample)), min_dim)

    if exact or min_dim <= max(q, 32):
        u, s, vh = torch.linalg.svd(w, full_matrices=False)
        u_r, s_r, vh_r = u[:, :rank], s[:rank], vh[:rank, :]
    else:
        # torch.svd_lowrank performs randomized SVD: U [out, q], S [q], V [in, q].
        u, s, v = torch.svd_lowrank(w, q=q, niter=max(0, int(niter)))
        u_r, s_r, vh_r = u[:, :rank], s[:rank], v[:, :rank].transpose(-2, -1)

    l1 = u_r * s_r.unsqueeze(0)
    l2 = vh_r
    return (
        l1.to(device=orig_device, dtype=orig_dtype).contiguous(),
        l2.to(device=orig_device, dtype=orig_dtype).contiguous(),
    )


def quantize_linear_from_samples(
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    x_samples: torch.Tensor,
    rank: int = 32,
    group_size: int = 128,
    smooth_min: float = 0.25,
    smooth_max: float = 4.0,
) -> SVDQuantLinearState:
    """Build a simulated SVDQuant state from a BF16/FP16/FP32 linear weight."""
    if weight.ndim != 2:
        raise ValueError("linear weight must be [out_features, in_features]")
    x_absmax = x_samples.float().abs().amax(dim=0)
    smooth = compute_smooth_scale(x_absmax, weight, smooth_min, smooth_max)
    migrated = weight * smooth.unsqueeze(0)
    l1, l2 = svd_lowrank(migrated, rank)
    residual = migrated.float() - (l1.float() @ l2.float())
    qweight, weight_scales = quantize_symmetric_int4(residual, group_size=group_size, dim=1)
    return SVDQuantLinearState(
        smooth_scale=smooth.detach().cpu(),
        qweight=qweight.detach().cpu(),
        weight_scales=weight_scales.detach().cpu(),
        l1=l1.detach().cpu(),
        l2=l2.detach().cpu(),
        bias=None if bias is None else bias.detach().cpu(),
        group_size=group_size,
        original_shape=tuple(weight.shape),
    )


def svdquant_linear_sim(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    """Reference PyTorch simulation for correctness/quality checks."""
    smooth = state.smooth_scale.to(device=x.device, dtype=x.dtype)
    x_hat = x / smooth
    qweight = state.qweight.to(device=x.device)
    scales = state.weight_scales.to(device=x.device)
    l1 = state.l1.to(device=x.device, dtype=x.dtype)
    l2 = state.l2.to(device=x.device, dtype=x.dtype)

    out_chunk = int(os.environ.get("KREA2_SVDQ_OUT_CHUNK", "0") or 0)
    if out_chunk > 0 and qweight.shape[0] > out_chunk:
        # Consumer-GPU safety path: dequantize and matmul only a slice of output
        # channels at a time. This is slower than the normal PyTorch reference, but
        # it avoids transient full-size dequantized weights and large GEMM
        # workspaces, which is useful until the fused Triton/Gluon kernels land.
        lowrank_mid = F.linear(x_hat, l2)
        chunks = []
        for start in range(0, qweight.shape[0], out_chunk):
            end = min(start + out_chunk, qweight.shape[0])
            w_res = dequantize_symmetric_int4(
                qweight[start:end],
                scales[start:end],
                group_size=state.group_size,
                dim=1,
                dtype=x.dtype,
            )
            w_res = w_res[..., : state.original_shape[1]]
            y_part = F.linear(x_hat, w_res) + F.linear(lowrank_mid, l1[start:end])
            if state.bias is not None:
                y_part = y_part + state.bias[start:end].to(device=x.device, dtype=x.dtype)
            chunks.append(y_part)
        return torch.cat(chunks, dim=-1)

    w_res = dequantize_symmetric_int4(qweight, scales, group_size=state.group_size, dim=1, dtype=x.dtype)
    w_res = w_res[..., : state.original_shape[1]]
    y_res = F.linear(x_hat, w_res)
    y_lr = F.linear(F.linear(x_hat, l2), l1)
    y = y_res + y_lr
    if state.bias is not None:
        y = y + state.bias.to(device=x.device, dtype=x.dtype)
    return y
