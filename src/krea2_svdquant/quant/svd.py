from __future__ import annotations

from dataclasses import dataclass

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


def svd_lowrank(weight: torch.Tensor, rank: int):
    """Return L1 [out, rank], L2 [rank, in] for weight [out, in]."""
    u, s, vh = torch.linalg.svd(weight.float(), full_matrices=False)
    u_r = u[:, :rank]
    s_r = s[:rank]
    vh_r = vh[:rank, :]
    l1 = u_r * s_r.unsqueeze(0)
    l2 = vh_r
    return l1.to(weight.dtype), l2.to(weight.dtype)


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
    w_res = dequantize_symmetric_int4(qweight, scales, group_size=state.group_size, dim=1)
    w_res = w_res[..., : state.original_shape[1]].to(dtype=x.dtype)
    y_res = F.linear(x_hat, w_res)
    l1 = state.l1.to(device=x.device, dtype=x.dtype)
    l2 = state.l2.to(device=x.device, dtype=x.dtype)
    y_lr = F.linear(F.linear(x_hat, l2), l1)
    y = y_res + y_lr
    if state.bias is not None:
        y = y + state.bias.to(device=x.device, dtype=x.dtype)
    return y
