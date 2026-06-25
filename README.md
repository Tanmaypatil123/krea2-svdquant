# krea2-svdquant

SVDQuant-style quantization and inference runtime scaffold for [`krea/Krea-2-Turbo`](https://huggingface.co/krea/Krea-2-Turbo).

## Aesthetic samples

Generated with the uploaded [`Patil/krea-turbo-svdquant`](https://huggingface.co/Patil/krea-turbo-svdquant) checkpoint using the low-VRAM 768px path (`--low-vram --cpu-offload model --block-offload --num-blocks-on-gpu 1 --out-chunk 1024 --vae-tiling --vae-slicing`).

![Krea Turbo SVDQuant aesthetic sample grid](https://huggingface.co/Patil/krea-turbo-svdquant/resolve/main/assets/contact_sheet.jpg)

| Glasshouse cafe | Coastal bedroom | Lavender portrait | Neon bookstore |
|---|---|---|---|
| ![](https://huggingface.co/Patil/krea-turbo-svdquant/resolve/main/assets/glasshouse_cafe.png) | ![](https://huggingface.co/Patil/krea-turbo-svdquant/resolve/main/assets/coastal_bedroom.png) | ![](https://huggingface.co/Patil/krea-turbo-svdquant/resolve/main/assets/lavender_portrait.png) | ![](https://huggingface.co/Patil/krea-turbo-svdquant/resolve/main/assets/neon_bookstore.png) |


This repository is intentionally set up as a research/engineering base for building our own Nunchaku-like path:

- W4A4 residual branch
- BF16/FP16 low-rank SVD branch
- Triton kernels for generic NVIDIA GPUs
- Blackwell/B200/SM120-specific optimized path using Triton + optional Gluon hooks
- Diffusers-compatible replacement of Krea2 transformer linear layers
- KernelIDE smoke-test scripts

> Status: working transformer-only SVDQuant checkpoint/runtime. The practical low-VRAM path uses packed INT4 qweights, prompt-embedding/text-encoder offload, transformer block offload, VAE tiling/slicing, and chunked SVDQuant linear execution. Experimental Triton fused kernels are included for correctness and future optimization.

## Architecture target

Krea-2-Turbo uses `Krea2Pipeline` with a `Krea2Transformer2DModel` denoiser:

```text
num_layers: 28
hidden size: 6144
attention heads: 48
kv heads: 12
head dim: 128
MLP intermediate: 16384
```

Primary quantization targets:

```text
transformer_blocks.*.attn.to_q
transformer_blocks.*.attn.to_k
transformer_blocks.*.attn.to_v
transformer_blocks.*.attn.to_gate
transformer_blocks.*.attn.to_out.0
transformer_blocks.*.ff.gate
transformer_blocks.*.ff.up
transformer_blocks.*.ff.down
```

## Method

For each linear layer:

```text
Y = X @ W.T
```

We approximate:

```text
Y ≈ int4_gemm(dynamic_int4(X / s), int4(W_res)) + ((X / s) @ L2.T) @ L1.T
```

Where:

```text
W_migrated = W * s
W_migrated ≈ W_res + L1 @ L2
```

`s` is the activation smoothing/migration scale. `L1/L2` are BF16 low-rank tensors from SVD. `W_res` is stored as groupwise packed INT4.

## Quick low-VRAM inference

Install the repo, then run directly from the Hugging Face checkpoint:

```bash
python scripts/infer_svdquant_transformer.py \
  --svdquant-transformer Patil/krea-turbo-svdquant \
  --backend pytorch_sim \
  --low-vram \
  --cpu-offload model \
  --block-offload \
  --num-blocks-on-gpu 1 \
  --out-chunk 1024 \
  --vae-tiling \
  --vae-slicing \
  --height 768 \
  --width 768 \
  --steps 8 \
  --prompt "a serene glasshouse cafe at golden hour, rain on windows, soft cinematic lighting" \
  --out outputs/krea_svdquant.png
```

Measured RunPod RTX PRO 6000 Blackwell examples:

```text
768x768:  peak≈11.88GiB, ~9.2s, 8 steps
1024x1024: peak≈16.96GiB, ~17.4s with PyTorch chunked backend, 8 steps
```

## Install

```bash
uv venv
. .venv/bin/activate
uv pip install -e '.[dev,kernels]'
```

For Krea2 inference, use a CUDA machine with enough VRAM and install PyTorch matching the CUDA stack.

## Main scripts

```bash
python scripts/baseline_infer.py --prompt "a cinematic robot doctor, gentle lighting"
python scripts/collect_calib.py --out calib_cache/krea2_small --max-prompts 16
python scripts/convert_simulated.py --calib calib_cache/krea2_small --out quantized_models/krea2-svdq-sim
python scripts/infer_svdquant_transformer.py --svdquant-transformer quantized_models/krea2-svdq-sim --prompt "a cinematic robot doctor"
python scripts/bench_linear.py --backend auto --m 4096 --k 6144 --n 16384
```

The intended SVDQuant workflow is transformer-only: load the full base Krea pipeline from Hugging Face, then replace only `pipe.transformer` from the SVDQuant checkpoint. Text encoder, tokenizer, scheduler, VAE, and unquantized transformer modules stay from the base HF model.

## Nunchaku-style API for Krea2 Turbo

```python
import torch
from diffusers import Krea2Pipeline
from krea2_svdquant import Krea2SVDQuantTransformer2DModel
from krea2_svdquant.utils import get_gpu_memory, get_precision, get_torch_dtype

torch_dtype = get_torch_dtype(get_precision())

transformer = Krea2SVDQuantTransformer2DModel.from_pretrained(
    "your-org/krea2-turbo-svdquant-transformer",
    torch_dtype=torch_dtype,
)

pipeline = Krea2Pipeline.from_pretrained(
    "krea/Krea-2-Turbo",
    transformer=transformer,
    torch_dtype=torch_dtype,
)

if get_gpu_memory() > 18:
    pipeline.enable_model_cpu_offload()
else:
    pipeline.enable_sequential_cpu_offload()

image = pipeline(
    "a tiny robot doctor holding a glowing flower, cinematic",
    num_inference_steps=8,
    guidance_scale=0.0,
).images[0]
```

Full example: `examples/krea2_svdquant_api.py`.

## Development order

1. Make the simulated PyTorch `SVDQuantLinearSim` produce good layer/block/full-image quality.
2. Add packed W4A16 Triton kernel.
3. Add dynamic activation A4 quantization.
4. Fuse activation quantization + low-rank down projection.
5. Fuse INT4 GEMM + low-rank up projection.
6. Specialize Blackwell path with B200 KernelIDE measurements.

## Important quality note

Do not optimize kernels before the simulation path preserves Krea2 image quality. Text rendering and attention projections are fragile; q/k may need higher rank or BF16 escape hatches.
