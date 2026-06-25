from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn

from krea2_svdquant.config import BackendKind
from krea2_svdquant.quant.svd import SVDQuantLinearState
from krea2_svdquant.runtime.linear import SVDQuantLinear
from krea2_svdquant.runtime.replace import replace_module


def _layer_key(layer_name: str) -> str:
    return layer_name.replace(".", "__")


def load_svdquant_transformer(
    transformer: nn.Module,
    checkpoint_dir: str | Path,
    *,
    backend: BackendKind | str = BackendKind.AUTO,
    strict: bool = True,
) -> dict:
    """Load a transformer-only SVDQuant checkpoint into a Krea2 transformer.

    This is the intended runtime flow:

    1. Load the full base Krea pipeline from Hugging Face normally.
    2. Call this function on `pipe.transformer`.
    3. Only quantized transformer linear modules are replaced; text encoder, VAE,
       scheduler, tokenizer, pipeline class, and unquantized transformer modules stay
       from the base model.

    Checkpoint layout:

    ```text
    checkpoint_dir/
      svdquant_config.json
      transformer_svdquant.safetensors
    ```
    """
    checkpoint_dir = Path(checkpoint_dir)
    config_path = checkpoint_dir / "svdquant_config.json"
    tensor_path = checkpoint_dir / "transformer_svdquant.safetensors"
    if not config_path.exists():
        raise FileNotFoundError(f"missing {config_path}")
    if not tensor_path.exists():
        raise FileNotFoundError(f"missing {tensor_path}")

    config = json.loads(config_path.read_text())
    tensors = load_file(str(tensor_path), device="cpu")
    replaced: list[str] = []
    missing: list[str] = []

    for layer_name, layer_meta in config.get("layers", {}).items():
        key = _layer_key(layer_name)
        required = [
            f"{key}.qweight_packed",
            f"{key}.weight_scales",
            f"{key}.smooth_scale",
            f"{key}.l1",
            f"{key}.l2",
        ]
        absent = [name for name in required if name not in tensors]
        if absent:
            missing.extend(absent)
            if strict:
                raise KeyError(f"missing tensors for {layer_name}: {absent}")
            continue

        shape = tuple(layer_meta["shape"])
        out_features, in_features = int(shape[0]), int(shape[1])
        qweight_packed = tensors[f"{key}.qweight_packed"]
        # Keep qweight packed in memory. The runtime unpacks per-layer/per-chunk,
        # which saves several GiB versus expanding every INT4 weight to int8 at load.
        padded_in = qweight_packed.shape[1] * 2

        bias_name = f"{key}.bias"
        state = SVDQuantLinearState(
            smooth_scale=tensors[f"{key}.smooth_scale"],
            qweight=qweight_packed,
            weight_scales=tensors[f"{key}.weight_scales"],
            l1=tensors[f"{key}.l1"],
            l2=tensors[f"{key}.l2"],
            bias=tensors[bias_name] if bias_name in tensors else None,
            group_size=int(layer_meta.get("group_size", config.get("group_size", 128))),
            original_shape=(out_features, in_features),
            qweight_packed=True,
            padded_in_features=padded_in,
        )
        replace_module(transformer, layer_name, SVDQuantLinear(state, backend=backend))
        replaced.append(layer_name)

    if strict and missing:
        raise KeyError(f"missing tensors: {missing}")
    config["_load_report"] = {"replaced": replaced, "missing": missing, "checkpoint_dir": str(checkpoint_dir)}
    return config


def save_transformer_checkpoint_readme(checkpoint_dir: str | Path, base_model: str) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    text = f"""# Krea2 SVDQuant transformer checkpoint

This is a transformer-only SVDQuant checkpoint for `{base_model}`.

Load the full base pipeline from Hugging Face, then replace only the transformer:

```python
import torch
from diffusers import Krea2Pipeline
from krea2_svdquant.runtime.load import load_svdquant_transformer

pipe = Krea2Pipeline.from_pretrained("{base_model}", torch_dtype=torch.bfloat16)
load_svdquant_transformer(pipe.transformer, "{checkpoint_dir.name}", backend="auto")
pipe.to("cuda")
```

The checkpoint intentionally does not include text encoder, tokenizer, scheduler, or VAE.
"""
    (checkpoint_dir / "README.md").write_text(text)
