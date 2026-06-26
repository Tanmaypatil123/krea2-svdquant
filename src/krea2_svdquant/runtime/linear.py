from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from krea2_svdquant.config import BackendKind
from krea2_svdquant.quant.svd import SVDQuantLinearState, svdquant_linear_sim


class SVDQuantLoRAAdapter(nn.Module):
    """Inference-only LoRA branch attached to a quantized SVDQuant linear."""

    def __init__(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        *,
        scale: float = 1.0,
        network_alpha: float | None = None,
    ) -> None:
        super().__init__()
        if down_weight.ndim != 2 or up_weight.ndim != 2:
            raise ValueError("LoRA down/up weights must be rank-2 tensors")
        rank = int(down_weight.shape[0])
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if int(up_weight.shape[1]) != rank:
            raise ValueError(
                f"LoRA shape mismatch: down={tuple(down_weight.shape)} up={tuple(up_weight.shape)}"
            )
        self.rank = rank
        self.scale = float(scale)
        self.network_alpha = float(network_alpha) if network_alpha is not None else float(rank)
        self.register_buffer("down_weight", down_weight.contiguous(), persistent=True)
        self.register_buffer("up_weight", up_weight.contiguous(), persistent=True)

    @property
    def multiplier(self) -> float:
        return self.scale * (self.network_alpha / self.rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        down = self.down_weight.to(dtype=x.dtype)
        up = self.up_weight.to(dtype=x.dtype)
        return F.linear(F.linear(x, down), up) * self.multiplier


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

    Quantized tensors are buffers so normal module moves and offload policies move
    qweight/scales/low-rank tensors with the module. Optional LoRA adapters are
    inference-only side branches added on top of the SVDQuant approximation.
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
        self.lora_adapters = nn.ModuleList()

    def add_lora_adapter(
        self,
        down_weight: torch.Tensor,
        up_weight: torch.Tensor,
        *,
        scale: float = 1.0,
        network_alpha: float | None = None,
    ) -> None:
        """Attach an inference LoRA adapter to this quantized linear."""
        expected_out, expected_in = self.original_shape
        if tuple(down_weight.shape[1:]) != (expected_in,):
            raise ValueError(
                f"LoRA down input mismatch for SVDQuantLinear: expected {expected_in}, "
                f"got {tuple(down_weight.shape)}"
            )
        if int(up_weight.shape[0]) != expected_out:
            raise ValueError(
                f"LoRA up output mismatch for SVDQuantLinear: expected {expected_out}, "
                f"got {tuple(up_weight.shape)}"
            )
        adapter = SVDQuantLoRAAdapter(
            down_weight, up_weight, scale=scale, network_alpha=network_alpha
        )
        self.lora_adapters.append(adapter)

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

    def _forward_base(self, x: torch.Tensor) -> torch.Tensor:
        backend = self.backend
        if backend is BackendKind.AUTO:
            backend = BackendKind.TRITON_BLACKWELL if is_blackwell_or_newer() else BackendKind.TRITON_GENERIC

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self._forward_base(x)
        if self.lora_adapters:
            for adapter in self.lora_adapters:
                y = y + adapter(x)
        return y
