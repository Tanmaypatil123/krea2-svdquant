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
    """Runtime wrapper for simulated and kernel-backed SVDQuant linear layers.

    The quantized tensors are registered as buffers instead of being kept only in a
    Python dataclass. That makes normal PyTorch module moves work:
    ``pipe.to("cuda")`` places qweight/scales/low-rank tensors on the GPU once,
    rather than copying huge weights from CPU on every forward pass.
    """

    def __init__(self, state: SVDQuantLinearState, backend: BackendKind | str = BackendKind.AUTO):
        super().__init__()
        self.backend = BackendKind(backend)
        self.group_size = int(state.group_size)
        self.original_shape = tuple(int(v) for v in state.original_shape)
        self.qweight_packed = bool(state.qweight_packed)
        self.padded_in_features = state.padded_in_features
        self.register_buffer("smooth_scale", state.smooth_scale.contiguous(), persistent=True)
        self.register_buffer("qweight", state.qweight.contiguous(), persistent=True)
        self.register_buffer("weight_scales", state.weight_scales.contiguous(), persistent=True)
        self.register_buffer("l1", state.l1.contiguous(), persistent=True)
        self.register_buffer("l2", state.l2.contiguous(), persistent=True)
        if state.bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", state.bias.contiguous(), persistent=True)

    @property
    def state(self) -> SVDQuantLinearState:
        return SVDQuantLinearState(
            smooth_scale=self.smooth_scale,
            qweight=self.qweight,
            weight_scales=self.weight_scales,
            l1=self.l1,
            l2=self.l2,
            bias=self.bias,
            group_size=self.group_size,
            original_shape=self.original_shape,
            qweight_packed=self.qweight_packed,
            padded_in_features=self.padded_in_features,
        )

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
