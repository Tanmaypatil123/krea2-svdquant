# Inference optimization paths

## Generic non-Blackwell path

Target GPUs: T4, L4, A10, A100, L40S, H100, H200, and unknown CUDA GPUs.

Default choices:

- BF16 pipeline where supported.
- TF32 matmul enabled for fp32 fallback ops.
- VAE slicing for memory safety.
- Optional attention slicing for low VRAM.
- Optional `torch.compile` only when benchmarking confirms it helps.
- Generic Triton SVDQuant kernels after conversion.

Command:

```bash
python scripts/infer_optimized.py --backend generic --prompt "..."
```

## Blackwell path: B200 / SM100 / SM120

Target GPUs: B200 as exposed by KernelIDE and RTX/GB20x Blackwell-family GPUs.

Default choices:

- Generic safe optimizations.
- Prefer SVDQuant Blackwell backend selector.
- Try TorchAO FP8 weight-only transformer optimization if `torchao` exists.
- Keep a separate Triton Blackwell file for `tl.dot_scaled`, FP4/NVFP4 experiments.
- Keep a separate Gluon file. KernelIDE B200 exposes Gluon under `triton.experimental.gluon` and Blackwell helpers under `triton.experimental.gluon.language.nvidia.blackwell`.

Command:

```bash
python scripts/infer_optimized.py --backend blackwell --prompt "..." --cache-prompt-offload-text
```

## Prompt embedding cache + text encoder offload

For Krea2, a useful memory optimization is:

1. encode prompt on CUDA,
2. move/remove text encoder,
3. denoise with `prompt_embeds` and `prompt_embeds_mask`.

This is implemented in:

```text
src/krea2_svdquant/inference/optimizations.py
```

## KernelIDE results captured during repo setup

Triton smoke on B200:

```text
GPU=NVIDIA B200 SM=100 max_err=0.000e+00
PASS kernelide Triton smoke
Status: SUCCESS
```

Gluon smoke on B200:

```text
GPU=NVIDIA B200 SM=(10, 0)
max_err=0.000e+00
PASS kernelide Gluon smoke
```
