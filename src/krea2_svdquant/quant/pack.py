from __future__ import annotations

import torch


def quantize_symmetric_int4(x: torch.Tensor, group_size: int = 128, dim: int = -1):
    """Groupwise symmetric int4 quantization.

    Returns int8 values in [-8, 7] and fp32 scales. Packing is separate so tests can
    compare dequantization without nibble layout concerns.
    """
    if x.shape[dim] % group_size != 0:
        pad = group_size - (x.shape[dim] % group_size)
        pad_shape = list(x.shape)
        pad_shape[dim] = pad
        x = torch.cat([x, torch.zeros(pad_shape, dtype=x.dtype, device=x.device)], dim=dim)
    moved = x.movedim(dim, -1).contiguous()
    orig = moved.shape
    grouped = moved.reshape(*orig[:-1], orig[-1] // group_size, group_size)
    scales = grouped.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / 7.0
    q = torch.round(grouped.float() / scales).clamp(-8, 7).to(torch.int8)
    q = q.reshape(*orig).movedim(-1, dim).contiguous()
    return q, scales.squeeze(-1)


def dequantize_symmetric_int4(
    q: torch.Tensor,
    scales: torch.Tensor,
    group_size: int = 128,
    dim: int = -1,
    dtype: torch.dtype | None = None,
):
    """Dequantize signed int4 values using group scales.

    ``dtype`` lets runtime paths materialize the dequantized weight directly in
    bf16/fp16 instead of briefly allocating a full fp32 copy of huge Krea2 matrices.
    That keeps peak VRAM much closer to consumer-GPU targets.
    """
    out_dtype = dtype or torch.float32
    moved = q.movedim(dim, -1).contiguous()
    orig = moved.shape
    grouped = moved.reshape(*orig[:-1], orig[-1] // group_size, group_size).to(out_dtype)
    y = grouped * scales.unsqueeze(-1).to(out_dtype)
    return y.reshape(*orig).movedim(-1, dim).contiguous()


def pack_int4(q: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 [-8, 7] values into uint8 nibbles along the last dim."""
    if q.dtype != torch.int8:
        raise TypeError("q must be torch.int8")
    if q.shape[-1] % 2 != 0:
        q = torch.cat([q, torch.zeros_like(q[..., :1])], dim=-1)
    u = (q.to(torch.int16) + 8).to(torch.uint8)
    lo = u[..., 0::2]
    hi = u[..., 1::2]
    return lo | (hi << 4)


def unpack_int4(packed: torch.Tensor, values_last_dim: int | None = None) -> torch.Tensor:
    """Unpack uint8 nibbles into signed int8 [-8, 7] values along last dim."""
    if packed.dtype != torch.uint8:
        raise TypeError("packed must be torch.uint8")
    lo = (packed & 0x0F).to(torch.int16) - 8
    hi = ((packed >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.empty(*packed.shape[:-1], packed.shape[-1] * 2, dtype=torch.int8, device=packed.device)
    out[..., 0::2] = lo.to(torch.int8)
    out[..., 1::2] = hi.to(torch.int8)
    if values_last_dim is not None:
        out = out[..., :values_last_dim]
    return out
