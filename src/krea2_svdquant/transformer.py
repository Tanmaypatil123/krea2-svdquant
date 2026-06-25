from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import torch

from krea2_svdquant.config import BackendKind
from krea2_svdquant.runtime.load import load_svdquant_transformer


def _is_local_dir(path_or_repo_id: str | Path) -> bool:
    return Path(path_or_repo_id).expanduser().exists()


def _snapshot_or_local(path_or_repo_id: str | Path, revision: str | None = None) -> Path:
    """Resolve a local directory or download a HF snapshot containing SVDQuant files."""
    if _is_local_dir(path_or_repo_id):
        return Path(path_or_repo_id).expanduser()

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - dependency error path
        raise ImportError("huggingface_hub is required to load SVDQuant checkpoints from the Hub") from exc

    local = snapshot_download(
        repo_id=str(path_or_repo_id),
        revision=revision,
        allow_patterns=["svdquant_config.json", "transformer_svdquant.safetensors", "README.md"],
    )
    return Path(local)


def _load_base_krea2_transformer(base_model: str, torch_dtype: torch.dtype | None = None, **kwargs: Any):
    """Load only the official Krea2 transformer when possible.

    Falls back to loading the pipeline's transformer if the direct model class is not
    exported by the installed Diffusers version.
    """
    try:
        from diffusers import Krea2Transformer2DModel

        return Krea2Transformer2DModel.from_pretrained(
            base_model,
            subfolder="transformer",
            torch_dtype=torch_dtype,
            **kwargs,
        )
    except Exception:
        from diffusers import Krea2Pipeline

        pipe = Krea2Pipeline.from_pretrained(base_model, torch_dtype=torch_dtype)
        transformer = pipe.transformer
        # Break references quickly; caller only wants the transformer.
        del pipe
        return transformer


class Krea2SVDQuantTransformer2DModel:
    """Nunchaku-style factory for Krea2 transformer-only SVDQuant checkpoints.

    This class intentionally returns an official Diffusers `Krea2Transformer2DModel`
    instance with selected linear layers replaced by `SVDQuantLinear` modules. That
    means the returned object can be passed directly to `Krea2Pipeline.from_pretrained`:

    ```python
    import torch
    from diffusers import Krea2Pipeline
    from krea2_svdquant import Krea2SVDQuantTransformer2DModel

    transformer = Krea2SVDQuantTransformer2DModel.from_pretrained(
        "your-org/krea2-turbo-svdquant-transformer",
        torch_dtype=torch.bfloat16,
    )

    pipe = Krea2Pipeline.from_pretrained(
        "krea/Krea-2-Turbo",
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    )
    ```
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | Path,
        *,
        base_model: str | None = None,
        revision: str | None = None,
        torch_dtype: torch.dtype | None = torch.bfloat16,
        backend: BackendKind | str = BackendKind.AUTO,
        strict: bool = True,
        **base_transformer_kwargs: Any,
    ):
        checkpoint_dir = _snapshot_or_local(pretrained_model_name_or_path, revision=revision)
        config_path = checkpoint_dir / "svdquant_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"missing {config_path}; expected a transformer-only SVDQuant checkpoint")
        config = json.loads(config_path.read_text())
        resolved_base_model = base_model or config.get("base_model") or "krea/Krea-2-Turbo"

        transformer = _load_base_krea2_transformer(
            resolved_base_model,
            torch_dtype=torch_dtype,
            **base_transformer_kwargs,
        )
        load_svdquant_transformer(transformer, checkpoint_dir, backend=backend, strict=strict)
        return transformer

    @classmethod
    def from_single_file(
        cls,
        safetensors_path: str | Path,
        *,
        config_path: str | Path,
        base_model: str = "krea/Krea-2-Turbo",
        torch_dtype: torch.dtype | None = torch.bfloat16,
        backend: BackendKind | str = BackendKind.AUTO,
        strict: bool = True,
        **base_transformer_kwargs: Any,
    ):
        """Load from a standalone safetensors file plus config file.

        This creates a temporary checkpoint directory with the standard layout, so
        runtime code remains identical to `from_pretrained`.
        """
        safetensors_path = Path(safetensors_path)
        config_path = Path(config_path)
        if not safetensors_path.exists():
            raise FileNotFoundError(safetensors_path)
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            (td_path / "svdquant_config.json").write_text(config_path.read_text())
            # Avoid copying huge files where possible by hardlinking; fallback to copy.
            target = td_path / "transformer_svdquant.safetensors"
            try:
                target.hardlink_to(safetensors_path)
            except Exception:
                import shutil

                shutil.copy2(safetensors_path, target)
            return cls.from_pretrained(
                td_path,
                base_model=base_model,
                torch_dtype=torch_dtype,
                backend=backend,
                strict=strict,
                **base_transformer_kwargs,
            )
