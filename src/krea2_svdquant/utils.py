from __future__ import annotations

import torch


def get_gpu_memory(device: int | str | torch.device | None = None) -> float:
    """Return total GPU memory in GiB for API parity with Nunchaku-style examples."""
    if not torch.cuda.is_available():
        return 0.0
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.total_memory / 2**30


def get_precision(device: int | str | torch.device | None = None) -> str:
    """Return a good default precision string for the current GPU.

    Krea2 normally runs well in bf16 on modern CUDA GPUs. For older GPUs without
    bf16 support, return fp16.
    """
    if not torch.cuda.is_available():
        return "fp32"
    if device is None:
        device = torch.cuda.current_device()
    major, _minor = torch.cuda.get_device_capability(device)
    return "bf16" if major >= 8 else "fp16"


def get_torch_dtype(precision: str | None = None) -> torch.dtype:
    if precision is None:
        precision = get_precision()
    precision = precision.lower()
    if precision in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if precision in {"fp16", "float16", "half"}:
        return torch.float16
    if precision in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"unsupported precision: {precision}")
