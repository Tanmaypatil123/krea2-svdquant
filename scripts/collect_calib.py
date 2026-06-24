from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import Krea2Pipeline

from krea2_svdquant.runtime.replace import iter_named_linears

DEFAULT_PROMPTS = [
    "a detailed portrait photograph with soft skin texture",
    "a poster with the words KREA TURBO in clean typography",
    "a cinematic robot doctor, gentle light, high detail",
    "a macro photo of a harvest mouse on green leaves",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="krea/Krea-2-Turbo")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-prompts", type=int, default=16)
    ap.add_argument("--max-tokens-per-layer", type=int, default=8192)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pipe = Krea2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    stats: dict[str, torch.Tensor] = {}
    handles = []

    def make_hook(name):
        def hook(_module, inputs):
            x = inputs[0].detach().reshape(-1, inputs[0].shape[-1]).float()
            absmax = x.abs().amax(dim=0).cpu()
            stats[name] = absmax if name not in stats else torch.maximum(stats[name], absmax)
        return hook

    for name, mod in iter_named_linears(pipe.transformer):
        # v1 checkpoint is transformer-block-only. Keep text_fusion/txt_in/final
        # from the base HF model for quality and LoRA compatibility.
        if not name.startswith("transformer_blocks."):
            continue
        handles.append(mod.register_forward_pre_hook(make_hook(name)))

    prompts = (DEFAULT_PROMPTS * ((args.max_prompts + len(DEFAULT_PROMPTS) - 1) // len(DEFAULT_PROMPTS)))[: args.max_prompts]
    for i, prompt in enumerate(prompts):
        print(f"calib {i+1}/{len(prompts)}: {prompt}")
        _ = pipe(prompt, num_inference_steps=1, guidance_scale=0.0, height=1024, width=1024).images[0]

    for h in handles:
        h.remove()
    torch.save(stats, out / "activation_absmax.pt")
    (out / "metadata.json").write_text(json.dumps({"model": args.model, "prompts": prompts}, indent=2))
    print(f"saved calibration stats to {out}")


if __name__ == "__main__":
    main()
