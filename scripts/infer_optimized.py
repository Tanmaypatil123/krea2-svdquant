from __future__ import annotations

import argparse
import time

import torch
from diffusers import Krea2Pipeline

from krea2_svdquant.inference import (
    apply_blackwell_optimizations,
    apply_generic_optimizations,
    cache_prompt_and_offload_text_encoder,
)
from krea2_svdquant.runtime.linear import is_blackwell_or_newer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="krea/Krea-2-Turbo")
    ap.add_argument("--prompt", default="a gentle robot doctor, cinematic soft light")
    ap.add_argument("--backend", choices=["auto", "generic", "blackwell"], default="auto")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--cache-prompt-offload-text", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--no-torchao-fp8", action="store_true")
    ap.add_argument("--out", default="outputs/optimized.png")
    args = ap.parse_args()

    pipe = Krea2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    backend = args.backend
    if backend == "auto":
        backend = "blackwell" if is_blackwell_or_newer() else "generic"

    if backend == "blackwell":
        report = apply_blackwell_optimizations(
            pipe,
            torchao_fp8=not args.no_torchao_fp8,
            compile_transformer=args.compile,
        )
    else:
        report = apply_generic_optimizations(pipe, compile_transformer=args.compile)
    print(report)

    kwargs = dict(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator(device="cuda").manual_seed(args.seed),
    )
    if args.cache_prompt_offload_text:
        prompt_embeds, prompt_embeds_mask = cache_prompt_and_offload_text_encoder(pipe, args.prompt)
        kwargs.update(prompt=None, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask)
    else:
        kwargs["prompt"] = args.prompt

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    image = pipe(**kwargs).images[0]
    torch.cuda.synchronize()
    print(f"backend={backend} seconds={time.perf_counter() - t0:.3f}")
    print(f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")
    image.save(args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
