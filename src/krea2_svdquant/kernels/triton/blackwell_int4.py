from __future__ import annotations

import torch

from krea2_svdquant.kernels.triton.generic_int4 import svdquant_linear_triton_generic
from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim

try:
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _blackwell_feature_probe_kernel(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < N
        x = tl.load(X + offs, mask=mask, other=0.0)
        # B200/SM100 tuning area: dot_scaled/Gluon FP4 path will replace the
        # portable scalar-unpack residual kernel after calibrated E2M1/NVFP4
        # checkpoint export is implemented.
        tl.store(Y + offs, x + 1.0, mask=mask)


def svdquant_linear_triton_blackwell(x: torch.Tensor, state: SVDQuantLinearState) -> torch.Tensor:
    """Blackwell-specific backend entry point.

    Today this routes through the portable fused W4A16 residual kernel. That gives
    the real memory win immediately on RTX PRO 6000/B200-family GPUs while keeping
    correctness tied to the existing INT4 checkpoint. The next Blackwell speed path
    is a separate calibrated FP4/E2M1 export plus ``tl.dot_scaled``/Gluon TCGEN05.
    """
    try:
        return svdquant_linear_triton_generic(x, state)
    except Exception:
        return svdquant_linear_sim(x, state)
