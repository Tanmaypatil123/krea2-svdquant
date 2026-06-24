from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from diffusers import Krea2Pipeline
from safetensors.torch import save_file

from krea2_svdquant.config import SVDQuantConfig
from krea2_svdquant.quant.svd import compute_smooth_scale, svd_lowrank
from krea2_svdquant.quant.pack import quantize_symmetric_int4, pack_int4
from krea2_svdquant.runtime.replace import iter_named_linears


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="krea/Krea-2-Turbo")
    ap.add_argument("--calib", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--group-size", type=int, default=128)
    args = ap.parse_args()
    cfg = SVDQuantConfig(weight_group_size=args.group_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    absmax = torch.load(Path(args.calib) / "activation_absmax.pt", map_location="cpu")
    pipe = Krea2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16, transformer=None)
    transformer = Krea2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).transformer.cpu()

    tensors = {}
    meta = {"base_model": args.model, "format": "krea2-svdquant-v1", "layers": {}}
    for name, mod in iter_named_linears(transformer):
        if name not in absmax:
            print(f"skip {name}: no calibration stats")
            continue
        rank = cfg.rank_for(name)
        w = mod.weight.detach().to(torch.bfloat16)
        smooth = compute_smooth_scale(absmax[name], w, cfg.smooth_min, cfg.smooth_max).cpu()
        migrated = w.cpu() * smooth.unsqueeze(0)
        l1, l2 = svd_lowrank(migrated, rank)
        residual = migrated.float() - (l1.float() @ l2.float())
        q, scales = quantize_symmetric_int4(residual, group_size=args.group_size, dim=1)
        key = name.replace(".", "__")
        tensors[f"{key}.qweight_packed"] = pack_int4(q)
        tensors[f"{key}.weight_scales"] = scales.to(torch.float16)
        tensors[f"{key}.smooth_scale"] = smooth.to(torch.bfloat16)
        tensors[f"{key}.l1"] = l1.to(torch.bfloat16)
        tensors[f"{key}.l2"] = l2.to(torch.bfloat16)
        if mod.bias is not None:
            tensors[f"{key}.bias"] = mod.bias.detach().cpu().to(torch.bfloat16)
        meta["layers"][name] = {"rank": rank, "group_size": args.group_size, "shape": list(w.shape)}
        print(f"converted {name} rank={rank} shape={tuple(w.shape)}")

    save_file(tensors, out / "transformer_svdquant.safetensors")
    (out / "svdquant_config.json").write_text(json.dumps(meta, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
