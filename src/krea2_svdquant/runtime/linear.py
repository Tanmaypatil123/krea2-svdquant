from __future__ import annotations

import torch
from torch import nn

from krea2_svdquant.config import BackendKind
from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim


def detect_sm() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def is_blackwell_or_newer() -> bool:
    sm = detect_sm()
    # KernelIDE reports NVIDIA B200 as capability (10, 0) / SM100. Some RTX/GB20x
    # Blackwell parts are expected to appear as SM120. Treat both families as the
    # Blackwell path; older Ada/Hopper remain on the generic path.
    return sm is not None and sm[0] >= 10


class SVDQuantLinear(nn.Module):
    """Runtime wrapper for simulated and kernel-backed SVDQuant linear layers."""

    def __init__(self, state: SVDQuantLinearState, backend: BackendKind | str = BackendKind.AUTO):
        super().__init__()
        self.state = state
        self.backend = BackendKind(backend)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.backend
        if backend is BackendKind.AUTO:
            # Blackwell/B200 uses the specialized path. Every other CUDA GPU uses
            # the portable normal-Triton path by default.
            backend = BackendKind.TRITON_BLACKWELL if is_blackwell_or_newer() else BackendKind.TRITON_GENERIC

        # Kernel backends are wired as explicit imports so missing Triton/Gluon gives a clear fallback.
        if backend is BackendKind.TRITON_GENERIC:
            try:
                from krea2_svdquant.kernels.triton.generic_int4 import svdquant_linear_triton_generic

                return svdquant_linear_triton_generic(x, self.state)
            except Exception:
                return svdquant_linear_sim(x, self.state)
        if backend is BackendKind.TRITON_BLACKWELL:
            try:
                from krea2_svdquant.kernels.triton.blackwell_int4 import svdquant_linear_triton_blackwell

                return svdquant_linear_triton_blackwell(x, self.state)
            except Exception:
                return svdquant_linear_sim(x, self.state)
        if backend is BackendKind.GLUON_BLACKWELL:
            try:
                from krea2_svdquant.kernels.gluon.blackwell_fused_svdquant import svdquant_linear_gluon_blackwell

                return svdquant_linear_gluon_blackwell(x, self.state)
            except Exception:
                return svdquant_linear_sim(x, self.state)
        return svdquant_linear_sim(x, self.state)
