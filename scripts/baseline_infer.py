from __future__ import annotations

import argparse
import time

import torch
from diffusers import Krea2Pipeline


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="krea/Krea-2-Turbo")
    p.add_argument("--prompt", default="a gentle medical robot in soft cinematic lighting")
    p.add_argument("--height", type=int, default=1024)
    p.add_argument("--width", type=int, default=1024)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--out", default="outputs/baseline.png")
    args = p.parse_args()

    pipe = Krea2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    torch.cuda.reset_peak_memory_stats()
    gen = torch.Generator(device="cuda").manual_seed(args.seed)
    t0 = time.perf_counter()
    image = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=gen,
    ).images[0]
    torch.cuda.synchronize()
    print(f"generation_seconds={time.perf_counter() - t0:.3f}")
    print(f"peak_gib={torch.cuda.max_memory_allocated() / 2**30:.2f}")
    image.save(args.out)
    print(f"saved={args.out}")


if __name__ == "__main__":
    main()
