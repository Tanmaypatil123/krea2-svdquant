from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(slots=True)
class OptimizationReport:
    backend: str
    applied: list[str]
    skipped: list[str]


def _try_import(name: str):
    try:
        return __import__(name, fromlist=["*"])
    except Exception:
        return None


def apply_generic_optimizations(pipe: Any, *, compile_transformer: bool = False) -> OptimizationReport:
    """Conservative path for non-Blackwell GPUs.

    Intended for T4/L4/A10/A100/L40S/H100/H200 and also as a safe fallback on any
    CUDA GPU. It avoids Blackwell-only FP4/NVFP4 assumptions.
    """
    applied: list[str] = []
    skipped: list[str] = []

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        applied.append("tf32_matmul_enabled")
    else:
        skipped.append("cuda_unavailable")

    if hasattr(pipe, "enable_attention_slicing"):
        # Useful for lower VRAM; can be disabled in benchmark scripts if it hurts latency.
        pipe.enable_attention_slicing("auto")
        applied.append("attention_slicing_auto")

    if hasattr(pipe, "vae") and hasattr(pipe.vae, "enable_slicing"):
        pipe.vae.enable_slicing()
        applied.append("vae_slicing")

    if compile_transformer and hasattr(torch, "compile") and hasattr(pipe, "transformer"):
        try:
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=False)
            applied.append("torch_compile_transformer")
        except Exception as exc:  # compile often fails on dynamic diffusers graphs
            skipped.append(f"torch_compile_transformer:{type(exc).__name__}")

    return OptimizationReport("generic", applied, skipped)


def apply_blackwell_optimizations(
    pipe: Any,
    *,
    torchao_fp8: bool = True,
    compile_transformer: bool = False,
    prefer_custom_kernels: bool = True,
) -> OptimizationReport:
    """B200/SM100/SM120 path.

    This path is for Blackwell-family GPUs. It enables generic safe flags, then tries
    optional Blackwell-friendly optimizations:
    - TorchAO FP8 transformer quantization when torchao is installed.
    - torch.compile if requested.
    - placeholder hook for our SVDQuant Triton/Gluon kernels.
    """
    base = apply_generic_optimizations(pipe, compile_transformer=False)
    applied = list(base.applied)
    skipped = list(base.skipped)

    if torchao_fp8 and hasattr(pipe, "transformer"):
        torchao = _try_import("torchao.quantization")
        if torchao is None:
            skipped.append("torchao_fp8:torchao_unavailable")
        else:
            try:
                # Attribute access avoids hard dependency at import time.
                cfg = getattr(torchao, "Float8WeightOnlyConfig")()
                getattr(torchao, "quantize_")(pipe.transformer, cfg)
                applied.append("torchao_float8_weight_only_transformer")
            except Exception as exc:
                skipped.append(f"torchao_fp8:{type(exc).__name__}")

    if prefer_custom_kernels:
        # Actual replacement is done by SVDQuantLinear/backend selector after conversion.
        applied.append("prefer_svdquant_blackwell_backend")

    if compile_transformer and hasattr(torch, "compile") and hasattr(pipe, "transformer"):
        try:
            pipe.transformer = torch.compile(pipe.transformer, mode="max-autotune", fullgraph=False)
            applied.append("torch_compile_transformer")
        except Exception as exc:
            skipped.append(f"torch_compile_transformer:{type(exc).__name__}")

    return OptimizationReport("blackwell", applied, skipped)


def cache_prompt_and_offload_text_encoder(pipe: Any, prompt: str, *, max_sequence_length: int = 128):
    """Cache Krea2 prompt embeddings, then offload/remove the text encoder.

    This mirrors the low-VRAM Krea2 pattern: encode prompt once, move text encoder
    away, and call the pipeline with prompt_embeds/prompt_embeds_mask.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt,
        device=device,
        max_sequence_length=max_sequence_length,
    )
    if hasattr(pipe, "text_encoder") and pipe.text_encoder is not None:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return prompt_embeds, prompt_embeds_mask
