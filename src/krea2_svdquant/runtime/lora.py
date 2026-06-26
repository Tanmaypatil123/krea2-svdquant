from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from krea2_svdquant.runtime.linear import SVDQuantLinear

_COMMON_WEIGHT_NAMES = (
    "pytorch_lora_weights.safetensors",
    "adapter_model.safetensors",
    "lora.safetensors",
    "pytorch_lora_weights.bin",
    "adapter_model.bin",
)

_DOWN_SUFFIXES = (
    ".lora_A.weight",
    ".lora.down.weight",
    ".lora_linear_layer.down.weight",
    ".lora_down.weight",
    ".down.weight",
)
_UP_SUFFIXES = (
    ".lora_B.weight",
    ".lora.up.weight",
    ".lora_linear_layer.up.weight",
    ".lora_up.weight",
    ".up.weight",
)
_ALPHA_SUFFIXES = (".alpha", ".network_alpha", ".lora_alpha")


def _resolve_lora_file(lora_path_or_repo_id: str | Path, weight_name: str | None = None) -> Path:
    path = Path(lora_path_or_repo_id).expanduser()
    if path.is_file():
        return path
    if path.is_dir():
        if weight_name is not None:
            candidate = path / weight_name
            if not candidate.exists():
                raise FileNotFoundError(candidate)
            return candidate
        for name in _COMMON_WEIGHT_NAMES:
            candidate = path / name
            if candidate.exists():
                return candidate
        safetensors = sorted(path.glob("*.safetensors"))
        if len(safetensors) == 1:
            return safetensors[0]
        raise FileNotFoundError(
            f"could not infer LoRA weight file in {path}; pass --lora-weight-name"
        )

    repo_id = str(lora_path_or_repo_id)
    if "/" not in repo_id or repo_id.startswith((".", "/", "~")):
        raise FileNotFoundError(path)
    try:
        from huggingface_hub import hf_hub_download
    except Exception as exc:  # pragma: no cover - dependency error path
        raise ImportError("huggingface_hub is required to load LoRA weights from the Hub") from exc

    names = (weight_name,) if weight_name is not None else _COMMON_WEIGHT_NAMES
    last_error: Exception | None = None
    for name in names:
        try:
            return Path(hf_hub_download(repo_id=repo_id, filename=name, repo_type="model"))
        except Exception as exc:  # pragma: no cover - network-dependent fallback
            last_error = exc
    raise FileNotFoundError(f"could not find a LoRA weight file in Hub repo {repo_id}") from last_error


def _load_lora_state_dict(path: Path) -> dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        return load_file(str(path), device="cpu")
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"unsupported LoRA checkpoint format: {path}")
    return {str(k): v for k, v in state.items() if torch.is_tensor(v)}


def _strip_known_prefixes(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for prefix in (
            "transformer.",
            "unet.",
            "base_model.model.",
            "model.",
            "module.",
            "diffusion_model.",
        ):
            if name.startswith(prefix):
                name = name[len(prefix) :]
                changed = True
    return name


def _module_matches(prefix: str, module_name: str) -> bool:
    prefix = _strip_known_prefixes(prefix)
    if prefix == module_name or prefix.endswith(f".{module_name}"):
        return True
    # Some LoRA exporters flatten module names with underscores, e.g.
    # lora_unet_transformer_blocks_0_attn_to_q. This covers the common case.
    flat_prefix = prefix.replace(".", "_")
    flat_module = module_name.replace(".", "_")
    return flat_prefix == flat_module or flat_prefix.endswith(f"_{flat_module}")


def _split_lora_pairs(state_dict: dict[str, torch.Tensor]) -> dict[str, dict[str, Any]]:
    pairs: dict[str, dict[str, Any]] = defaultdict(dict)
    for key, tensor in state_dict.items():
        for suffix in _DOWN_SUFFIXES:
            if key.endswith(suffix):
                pairs[key[: -len(suffix)]]["down"] = tensor
                break
        for suffix in _UP_SUFFIXES:
            if key.endswith(suffix):
                pairs[key[: -len(suffix)]]["up"] = tensor
                break
        for suffix in _ALPHA_SUFFIXES:
            if key.endswith(suffix):
                alpha = tensor.item() if tensor.numel() == 1 else None
                pairs[key[: -len(suffix)]]["alpha"] = alpha
                break
    return pairs


def load_svdquant_lora_adapters(
    transformer: torch.nn.Module,
    lora_path_or_repo_id: str | Path,
    *,
    weight_name: str | None = None,
    scale: float = 1.0,
    strict: bool = True,
) -> dict[str, Any]:
    """Load LoRA weights into SVDQuantLinear modules on an already-quantized transformer.

    Diffusers/PEFT injects adapters into regular Linear modules. Krea2 SVDQuant
    replaces those linears, so this helper attaches an equivalent inference-only
    LoRA branch directly to each matching ``SVDQuantLinear`` module.
    """
    lora_file = _resolve_lora_file(lora_path_or_repo_id, weight_name=weight_name)
    state_dict = _load_lora_state_dict(lora_file)
    pairs = _split_lora_pairs(state_dict)
    modules = {name: module for name, module in transformer.named_modules() if isinstance(module, SVDQuantLinear)}

    loaded: list[str] = []
    unmatched: list[str] = []
    incomplete: list[str] = []

    for prefix, parts in pairs.items():
        if "down" not in parts or "up" not in parts:
            incomplete.append(prefix)
            continue
        matches = [name for name in modules if _module_matches(prefix, name)]
        if not matches:
            unmatched.append(prefix)
            continue
        if len(matches) > 1:
            raise ValueError(f"LoRA prefix {prefix!r} matched multiple modules: {matches}")
        name = matches[0]
        modules[name].add_lora_adapter(
            parts["down"],
            parts["up"],
            scale=scale,
            network_alpha=parts.get("alpha"),
        )
        loaded.append(name)

    if strict and not loaded:
        raise ValueError(
            f"no LoRA tensors from {lora_file} matched SVDQuant transformer layers; "
            "check that this is a Krea2 transformer LoRA or pass the right weight_name"
        )
    return {
        "lora_file": str(lora_file),
        "loaded": loaded,
        "unmatched": unmatched,
        "incomplete": incomplete,
    }
