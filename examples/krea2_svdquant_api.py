from __future__ import annotations

import argparse
import torch
from diffusers import Krea2Pipeline

from krea2_svdquant import Krea2SVDQuantTransformer2DModel
from krea2_svdquant.utils import get_gpu_memory, get_precision, get_torch_dtype


def main():
    parser = argparse.ArgumentParser(description="Nunchaku-style Krea2 SVDQuant API example")
    parser.add_argument("--base-model", default="krea/Krea-2-Turbo")
    parser.add_argument("--svdquant-transformer", required=True)
    parser.add_argument("--lora", action="append", default=[], help="Optional LoRA file/dir/HF repo; repeatable.")
    parser.add_argument("--lora-weight-name", default=None)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--prompt", default="a tiny robot doctor holding a glowing flower, cinematic")
    parser.add_argument("--out", default="outputs/krea2_svdquant_api.png")
    args = parser.parse_args()

    precision = get_precision()
    torch_dtype = get_torch_dtype(precision)
    print(f"gpu_memory_gib={get_gpu_memory():.2f} precision={precision}")

    transformer = Krea2SVDQuantTransformer2DModel.from_pretrained(
        args.svdquant_transformer,
        torch_dtype=torch_dtype,
        lora_weights=args.lora or None,
        lora_weight_name=args.lora_weight_name,
        lora_scale=args.lora_scale,
    )

    pipeline = Krea2Pipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=torch_dtype,
    )

    if get_gpu_memory() > 18:
        pipeline.enable_model_cpu_offload()
    else:
        # Keep this simple for now; Krea2 transformer per-layer offload helper can be
        # added once our runtime modules support fine-grained offload policies.
        pipeline.enable_sequential_cpu_offload()

    image = pipeline(
        args.prompt,
        num_inference_steps=8,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(12345),
    ).images[0]
    image.save(args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
