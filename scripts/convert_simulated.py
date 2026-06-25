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
from krea2_svdquant.runtime.load import save_transformer_checkpoint_readme
from krea2_svdquant.runtime.replace import iter_named_linears


def build_argparser() -> argparse.ArgumentParser:
    cfg = SVDQuantConfig()
    ap = argparse.ArgumentParser(
        prog="convert_simulated.py",
        description=(
            "Create a transformer-only SVDQuant checkpoint for Krea-2-Turbo.\n\n"
            "Calibration is OPTIONAL: pass --calib to migrate activations with "
            "SmoothQuant-style scales; without it (or for layers missing stats) the "
            "converter falls back to identity smoothing and still produces a usable "
            "low-rank + INT4 checkpoint."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g_io = ap.add_argument_group("inputs / outputs")
    g_io.add_argument("--model", default="krea/Krea-2-Turbo", help="Base HF model id or local path.")
    g_io.add_argument(
        "--calib",
        default=None,
        help="Optional calibration dir containing activation_absmax.pt. Omit to convert "
        "without calibration (identity activation smoothing).",
    )
    g_io.add_argument(
        "--require-calib",
        action="store_true",
        help="Skip layers that have no calibration stats instead of falling back to "
        "identity smoothing. Requires --calib.",
    )
    g_io.add_argument("--out", required=True, help="Output checkpoint directory.")

    g_q = ap.add_argument_group("quantization")
    g_q.add_argument("--group-size", type=int, default=cfg.weight_group_size, help="INT4 weight group size.")
    g_q.add_argument("--rank", type=int, default=cfg.default_rank, help="Default low-rank dim.")
    g_q.add_argument("--attn-rank", type=int, default=cfg.attn_rank, help="Rank override for attention projections.")
    g_q.add_argument("--mlp-rank", type=int, default=cfg.mlp_rank, help="Rank override for MLP/ff projections.")

    g_quality = ap.add_argument_group("quality preservation")
    g_quality.add_argument(
        "--skip-first-blocks", type=int, default=0, help="Leave the first N transformer blocks in base BF16."
    )
    g_quality.add_argument(
        "--skip-last-blocks", type=int, default=0, help="Leave the last N transformer blocks in base BF16."
    )
    g_quality.add_argument(
        "--skip-qk",
        action="store_true",
        help="Do not quantize attn.to_q / attn.to_k (these are fragile for text/structure).",
    )
    g_quality.add_argument(
        "--max-layers",
        type=int,
        default=0,
        help="Smoke mode: convert at most N layers (0 = all).",
    )

    g_svd = ap.add_argument_group("SVD (randomized/truncated)")
    g_svd.add_argument("--svd-oversample", type=int, default=cfg.svd_oversample, help="Randomized SVD oversampling.")
    g_svd.add_argument("--svd-niter", type=int, default=cfg.svd_niter, help="Randomized SVD power iterations.")
    g_svd.add_argument(
        "--svd-device",
        default=None,
        help="Device to run SVD on, e.g. 'cuda' to accelerate huge linears. Defaults to weight device (cpu).",
    )
    g_svd.add_argument(
        "--svd-exact",
        action="store_true",
        help="Force full torch.linalg.svd instead of the randomized path (slow; reference only).",
    )
    return ap


def _block_index(name: str) -> int | None:
    """Return the transformer block index for a 'transformer_blocks.<i>.*' name."""
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "transformer_blocks" and parts[1].isdigit():
        return int(parts[1])
    return None


def _rank_for(name: str, args) -> int:
    if ".ff." in name or name.endswith(("ff.gate", "ff.up", "ff.down")):
        return args.mlp_rank
    if ".attn." in name or ".to_" in name:
        return args.attn_rank
    return args.rank


def main():
    args = build_argparser().parse_args()
    if args.require_calib and not args.calib:
        raise SystemExit("--require-calib needs --calib")

    cfg = SVDQuantConfig(weight_group_size=args.group_size)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    absmax: dict[str, torch.Tensor] = {}
    if args.calib:
        calib_path = Path(args.calib) / "activation_absmax.pt"
        if not calib_path.exists():
            raise SystemExit(f"--calib given but {calib_path} not found")
        absmax = torch.load(calib_path, map_location="cpu")
        print(f"loaded calibration stats for {len(absmax)} layers from {calib_path}")
    else:
        print("no --calib: converting with identity activation smoothing")

    transformer = Krea2Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).transformer.cpu()

    # Gather quantization targets first so we can resolve last-block skipping.
    targets = [
        (name, mod)
        for name, mod in iter_named_linears(transformer)
        if name.startswith("transformer_blocks.")
    ]
    block_ids = sorted({b for b in (_block_index(n) for n, _ in targets) if b is not None})
    num_blocks = (max(block_ids) + 1) if block_ids else 0
    skip_block_set = set(block_ids[: args.skip_first_blocks]) | set(
        block_ids[len(block_ids) - args.skip_last_blocks :] if args.skip_last_blocks else []
    )

    tensors: dict[str, torch.Tensor] = {}
    meta = {
        "base_model": args.model,
        "format": "krea2-svdquant-transformer-v1",
        "target_component": "transformer",
        "transformer_class": "Krea2Transformer2DModel",
        "runtime_target": "w4a16_plus_bf16_lowrank",
        "weight_bits": 4,
        "activation_bits": 16,
        "group_size": args.group_size,
        "calibrated": bool(absmax),
        "num_blocks": num_blocks,
        "options": {
            "rank": args.rank,
            "attn_rank": args.attn_rank,
            "mlp_rank": args.mlp_rank,
            "skip_first_blocks": args.skip_first_blocks,
            "skip_last_blocks": args.skip_last_blocks,
            "skip_qk": args.skip_qk,
            "max_layers": args.max_layers,
            "svd_oversample": args.svd_oversample,
            "svd_niter": args.svd_niter,
            "svd_exact": args.svd_exact,
        },
        "layers": {},
    }

    converted = 0
    for name, mod in targets:
        if args.max_layers and converted >= args.max_layers:
            print(f"reached --max-layers={args.max_layers}; stopping")
            break

        block = _block_index(name)
        if block is not None and block in skip_block_set:
            print(f"skip {name}: block {block} kept in base BF16")
            continue
        if args.skip_qk and name.endswith(("attn.to_q", "attn.to_k")):
            print(f"skip {name}: --skip-qk")
            continue

        calibrated = name in absmax
        if not calibrated and args.require_calib:
            print(f"skip {name}: no calibration stats (--require-calib)")
            continue

        rank = _rank_for(name, args)
        w = mod.weight.detach().to(torch.bfloat16)
        if calibrated:
            smooth = compute_smooth_scale(absmax[name], w, cfg.smooth_min, cfg.smooth_max).cpu()
        else:
            # Identity smoothing: no activation migration, still SVD + INT4 the weight.
            smooth = torch.ones(w.shape[1], dtype=w.dtype)
        migrated = w.cpu() * smooth.unsqueeze(0)
        l1, l2 = svd_lowrank(
            migrated,
            rank,
            oversample=args.svd_oversample,
            niter=args.svd_niter,
            device=args.svd_device,
            exact=args.svd_exact,
        )
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
        meta["layers"][name] = {
            "rank": rank,
            "group_size": args.group_size,
            "shape": list(w.shape),
            "calibrated": calibrated,
        }
        converted += 1
        print(f"converted {name} rank={rank} shape={tuple(w.shape)} calibrated={calibrated}")

    save_file(tensors, out / "transformer_svdquant.safetensors")
    (out / "svdquant_config.json").write_text(json.dumps(meta, indent=2))
    save_transformer_checkpoint_readme(out, args.model)
    print(f"converted {converted} layers; saved to {out}")


if __name__ == "__main__":
    main()
