from __future__ import annotations

import importlib

import torch

from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim


def _load_gluon():
    """Load Triton Experimental Gluon.

    KernelIDE runs Gluon through the Triton language image. The correct import path
    on the B200 image is `triton.experimental.gluon`, with layouts and Blackwell
    helpers under `triton.experimental.gluon.language.nvidia.blackwell`.
    """
    for name in (
        "triton.experimental.gluon",
        "triton.experimental.gluon.language",
        "triton.experimental.gluon.language.nvidia.blackwell",
    ):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    raise ImportError("Triton Experimental Gluon was not found in this environment")


def svdquant_linear_gluon_blackwell(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    """B200/SM100/SM120 Gluon backend entry point.

    TODO:
    1. fused quantize + low-rank down kernel
    2. fused int4/NVFP4 matmul + low-rank up kernel
    3. compare against Triton Blackwell path
    """
    _load_gluon()
    return svdquant_linear_sim(x, state)
