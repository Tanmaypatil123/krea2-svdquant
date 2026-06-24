from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class BackendKind(str, Enum):
    AUTO = "auto"
    PYTORCH_SIM = "pytorch_sim"
    TRITON_GENERIC = "triton_generic"
    TRITON_BLACKWELL = "triton_blackwell"
    GLUON_BLACKWELL = "gluon_blackwell"


@dataclass(slots=True)
class SVDQuantConfig:
    weight_bits: int = 4
    activation_bits: int = 4
    weight_group_size: int = 128
    activation_group_size: int = 128
    default_rank: int = 32
    mlp_rank: int = 64
    attn_rank: int = 32
    smooth_min: float = 0.25
    smooth_max: float = 4.0
    lowrank_dtype: str = "bfloat16"
    output_dtype: str = "bfloat16"
    target_modules: tuple[str, ...] = field(
        default_factory=lambda: (
            "attn.to_q",
            "attn.to_k",
            "attn.to_v",
            "attn.to_gate",
            "attn.to_out.0",
            "ff.gate",
            "ff.up",
            "ff.down",
        )
    )

    def rank_for(self, module_name: str) -> int:
        if ".ff." in module_name or module_name.endswith(("ff.gate", "ff.up", "ff.down")):
            return self.mlp_rank
        if ".attn." in module_name or ".to_" in module_name:
            return self.attn_rank
        return self.default_rank
