from __future__ import annotations

import importlib

import torch

from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim


def _load_gluon():
    """Load optional Gluon backend when present in the runtime image.

    KernelIDE currently exposes Triton/CUDA/CUTLASS/etc. as languages, not a separate
    Gluon language. This hook keeps the B200 Gluon path explicit without making the
    package impossible to import on normal machines.
    """
    for name in ("gluon", "triton._C.libtriton.gluon", "triton.language.extra.gluon"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise ImportError("No supported Gluon module found in this environment")


def svdquant_linear_gluon_blackwell(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    """B200/SM120 Gluon backend entry point.

    TODO once Gluon API is available in KernelIDE image:
    1. fused quantize + low-rank down kernel
    2. fused int4/NVFP4 matmul + low-rank up kernel
    3. compare against Triton Blackwell path
    """
    _load_gluon()
    return svdquant_linear_sim(x, state)
