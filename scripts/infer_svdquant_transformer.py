from __future__ import annotations

import argparse
import gc
import os
import time
from pathlib import Path

import torch
from diffusers import Krea2Pipeline

from krea2_svdquant.runtime.block_offload import enable_block_offload
from krea2_svdquant.runtime.load import load_svdquant_transformer
from krea2_svdquant.utils import report_cuda_memory


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="infer_svdquant_transformer.py",
        description=(
            "Run Krea2 with a transformer-only SVDQuant checkpoint.\n\n"
            "With --low-vram the script encodes the prompt once, offloads/removes the "
            "text encoder, and runs generation with prompt_embeds. VRAM (allocated / "
            "reserved / peak) is reported at load, encode, offload, and generate stages."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--base-model", default="krea/Krea-2-Turbo", help="Base HF model id or local path.")
    ap.add_argument(
        "--svdquant-transformer",
        required=True,
        help="Directory with svdquant_config.json and transformer_svdquant.safetensors.",
    )
    ap.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "pytorch_sim", "triton_generic", "triton_blackwell", "gluon_blackwell"],
        help="SVDQuant linear backend.",
    )
    ap.add_argument("--prompt", default="a gentle medical robot, cinematic soft lighting")
    ap.add_argument("--height", type=int, default=1024)
    ap.add_argument("--width", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument(
        "--low-vram",
        action="store_true",
        help="Encode the prompt, then offload/remove the text encoder before generation.",
    )
    ap.add_argument(
        "--cpu-offload",
        default="none",
        choices=["none", "model", "sequential"],
        help=(
            "Optional Diffusers/Accelerate CPU offload mode. 'model' keeps one model "
            "component on GPU at a time; 'sequential' offloads submodules more "
            "aggressively but is slower. Use with --low-vram for consumer GPUs."
        ),
    )
    ap.add_argument(
        "--out-chunk",
        type=int,
        default=0,
        help=(
            "Chunk output channels per SVDQuant linear in the PyTorch reference runtime. "
            "Example: 2048 lowers peak VRAM for 1024px consumer tests at the cost of "
            "slower generation. 0 disables chunking."
        ),
    )
    ap.add_argument(
        "--max-sequence-length",
        type=int,
        default=128,
        help="Max prompt token length for the prompt-embedding path.",
    )
    ap.add_argument("--vae-tiling", action="store_true", help="Enable VAE tiling to reduce decode peak VRAM.")
    ap.add_argument("--vae-slicing", action="store_true", help="Enable VAE slicing to reduce decode peak VRAM.")
    ap.add_argument(
        "--block-offload",
        action="store_true",
        help=(
            "Keep transformer_blocks on CPU and stream each block to the CUDA device "
            "only for its forward pass. Non-block transformer modules stay on CUDA. "
            "Drastically lowers transformer VRAM for consumer GPUs at the cost of "
            "per-step host<->device copies. Combine with --low-vram and --out-chunk."
        ),
    )
    ap.add_argument(
        "--num-blocks-on-gpu",
        type=int,
        default=1,
        help="With --block-offload, how many transformer blocks to keep resident on GPU.",
    )
    ap.add_argument(
        "--pin-block-memory",
        action="store_true",
        help=(
            "With --block-offload, pin the CPU block weights so host->device copies are "
            "asynchronous and faster. Uses extra (page-locked) host RAM."
        ),
    )
    ap.add_argument("--out", default="outputs/svdquant.png")
    return ap


def _encode_and_offload(pipe, prompt: str, device, max_sequence_length: int):
    """Encode prompt embeddings, then move/remove the text encoder for low VRAM."""
    prompt_embeds, prompt_embeds_mask = pipe.encode_prompt(
        prompt,
        device=device,
        max_sequence_length=max_sequence_length,
    )
    report_cuda_memory("encode")

    if getattr(pipe, "text_encoder", None) is not None:
        pipe.text_encoder.to("cpu")
        pipe.text_encoder = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    report_cuda_memory("offload")
    return prompt_embeds, prompt_embeds_mask


def main():
    args = build_argparser().parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.out_chunk > 0:
        os.environ["KREA2_SVDQ_OUT_CHUNK"] = str(args.out_chunk)
        print(f"svdq_out_chunk={args.out_chunk}")

    pipe = Krea2Pipeline.from_pretrained(args.base_model, torch_dtype=torch.bfloat16)
    report = load_svdquant_transformer(pipe.transformer, args.svdquant_transformer, backend=args.backend)
    print(f"loaded_svdquant_layers={len(report['_load_report']['replaced'])}")

    offloader = None
    if args.block_offload:
        # Block offload owns transformer placement: never run a whole-component
        # CPU offload over the transformer, and never transformer.to(cuda) the
        # blocks. Exclude the transformer from any Diffusers offload, move the
        # remaining pipeline components to the device, then stream blocks.
        if args.cpu_offload in ("model", "sequential"):
            exclude = list(getattr(pipe, "_exclude_from_cpu_offload", []) or [])
            if "transformer" not in exclude:
                exclude.append("transformer")
            pipe._exclude_from_cpu_offload = exclude
            if args.cpu_offload == "model":
                pipe.enable_model_cpu_offload(device=device)
            else:
                pipe.enable_sequential_cpu_offload(device=device)
            print(f"cpu_offload={args.cpu_offload} (transformer excluded)")
        else:
            for name, comp in pipe.components.items():
                if name != "transformer" and isinstance(comp, torch.nn.Module):
                    comp.to(device)
        offloader = enable_block_offload(
            pipe.transformer,
            device,
            num_blocks_on_gpu=args.num_blocks_on_gpu,
            pin_memory=args.pin_block_memory,
        )
        print(
            f"block_offload=on blocks={offloader.num_blocks} "
            f"num_blocks_on_gpu={offloader.num_blocks_on_gpu} "
            f"pin_memory={args.pin_block_memory} containers={offloader.attr_names}"
        )
    elif args.cpu_offload == "model":
        pipe.enable_model_cpu_offload(device=device)
        print("cpu_offload=model")
    elif args.cpu_offload == "sequential":
        pipe.enable_sequential_cpu_offload(device=device)
        print("cpu_offload=sequential")
    else:
        pipe.to(device)
        print("cpu_offload=none")
    if args.vae_tiling:
        pipe.vae.enable_tiling()
        print("vae_tiling=on")
    if args.vae_slicing:
        pipe.vae.enable_slicing()
        print("vae_slicing=on")
    report_cuda_memory("load")

    kwargs = dict(
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        guidance_scale=0.0,
        generator=torch.Generator(device=device).manual_seed(args.seed),
    )
    if args.low_vram:
        prompt_embeds, prompt_embeds_mask = _encode_and_offload(
            pipe, args.prompt, device, args.max_sequence_length
        )
        kwargs.update(prompt=None, prompt_embeds=prompt_embeds, prompt_embeds_mask=prompt_embeds_mask)
    else:
        kwargs["prompt"] = args.prompt

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    image = pipe(**kwargs).images[0]
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    print(f"seconds={time.perf_counter() - start:.3f}")
    report_cuda_memory("generate")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    print(f"saved={out_path}")


if __name__ == "__main__":
    main()
