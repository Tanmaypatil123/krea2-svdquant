from __future__ import annotations

import types

import torch
from safetensors.torch import save_file
from torch import nn

from krea2_svdquant.quant.svd import SVDQuantLinearState
from krea2_svdquant.runtime.linear import SVDQuantLinear
from krea2_svdquant.runtime.lora import load_svdquant_lora_adapters


def _svdq_linear(in_features: int = 4, out_features: int = 3) -> SVDQuantLinear:
    state = SVDQuantLinearState(
        smooth_scale=torch.ones(in_features),
        qweight=torch.zeros(out_features, in_features // 2, dtype=torch.uint8),
        weight_scales=torch.ones(out_features, 1),
        l1=torch.zeros(out_features, 1),
        l2=torch.zeros(1, in_features),
        bias=None,
        group_size=in_features,
        original_shape=(out_features, in_features),
        qweight_packed=True,
        padded_in_features=in_features,
    )
    return SVDQuantLinear(state, backend="pytorch_sim")


class _ToyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        block = nn.Module()
        block.attn = nn.Module()
        block.attn.to_q = _svdq_linear()
        self.transformer_blocks = nn.ModuleList([block])


def test_load_lora_adapter_matches_transformer_prefixed_keys(tmp_path):
    model = _ToyTransformer()
    down = torch.arange(8, dtype=torch.float32).reshape(2, 4) / 10
    up = torch.arange(6, dtype=torch.float32).reshape(3, 2) / 10
    lora_file = tmp_path / "pytorch_lora_weights.safetensors"
    save_file(
        {
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight": down,
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight": up,
            "transformer.transformer_blocks.0.attn.to_q.alpha": torch.tensor(4.0),
        },
        lora_file,
    )

    report = load_svdquant_lora_adapters(model, lora_file, scale=0.5)

    assert report["loaded"] == ["transformer_blocks.0.attn.to_q"]
    target = model.transformer_blocks[0].attn.to_q
    assert len(target.lora_adapters) == 1

    # Isolate the LoRA path from the SVDQuant approximation for a deterministic unit test.
    target._forward_base = types.MethodType(lambda self, x: torch.zeros(x.shape[:-1] + (3,)), target)
    x = torch.randn(2, 4)
    expected = torch.nn.functional.linear(torch.nn.functional.linear(x, down), up) * (0.5 * 4.0 / 2)
    torch.testing.assert_close(target(x), expected)


def test_load_lora_adapter_supports_flat_kohya_style_prefix(tmp_path):
    model = _ToyTransformer()
    lora_file = tmp_path / "flat.safetensors"
    save_file(
        {
            "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": torch.zeros(2, 4),
            "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": torch.zeros(3, 2),
        },
        lora_file,
    )

    report = load_svdquant_lora_adapters(model, lora_file)

    assert report["loaded"] == ["transformer_blocks.0.attn.to_q"]
