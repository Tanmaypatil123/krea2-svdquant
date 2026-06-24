from __future__ import annotations

import argparse
import time

import torch
from diffusers import Krea2Pipeline

from krea2_svdquant.runtime.load import load_svdquant_transformer


def main():
    ap = argparse.ArgumentParser(description="Run Krea2 with a transformer-only SVDQuant checkpoint.")
    ap.add_argument("--base-model", default="krea/Krea-2-Turbo")
    ap.add_argument("--svdquant-transformer", required=True, help="Directory containing svdquant_config.json and transformer_svdquant.safetensors")
    ap.add_argument("--backend", default="auto", choices=["auto", "pytorch_sim", "triton_generic", "triton_blackwell", "gluon_blackwell"])
    ap.add_argument("--prompt", default="a gentle medical robot, cinematic soft lighting")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--out", default="outputs/svdquant.png")
    args = ap.parse_args()

    pipe = Krea2Pipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    report = load_svdquant_transformer(pipe.transformer, args.svdquant_transformer, backend=args.backend)
    print(f"loaded_svdquant_layers={len(report['_load_report']['replaced'])}")
    pipe.to("cuda")

    torch.cuda.reset_peak_memory_stats()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    start = time.perf_counter()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=generator,
    ).images[0]
    torch.cuda.synchronize()
    print(f"seconds={time.perf_counter() - start:.3f}")
    print(f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")
    image.save(args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
