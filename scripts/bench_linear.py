from __future__ import annotations

import argparse
import time

import torch

from krea2_svdquant.quant.svd import quantize_linear_from_samples, svdquant_linear_sim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=int, default=4096)
    ap.add_argument("--k", type=int, default=6144)
    ap.add_argument("--n", type=int, default=16384)
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--backend", default="sim")
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    x = torch.randn(args.m, args.k, device=dev, dtype=dtype)
    w = torch.randn(args.n, args.k, device=dev, dtype=dtype) / (args.k**0.5)
    state = quantize_linear_from_samples(w.cpu(), None, x[: min(args.m, 256)].cpu(), rank=args.rank)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = svdquant_linear_sim(x, state)
    if dev == "cuda":
        torch.cuda.synchronize()
    print(f"device={dev} shape=({args.m},{args.k})x({args.n},{args.k}) rank={args.rank}")
    print(f"seconds={time.perf_counter() - t0:.4f} output_norm={y.float().norm().item():.4f}")


if __name__ == "__main__":
    main()
