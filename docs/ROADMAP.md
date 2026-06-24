# Roadmap

## V0: simulation correctness

- Per-layer calibration activation absmax collection.
- SmoothQuant-style activation outlier migration.
- SVD low-rank residual extraction.
- Groupwise INT4 residual weight quantization.
- Full Krea2 image generation with PyTorch simulated quantized linears.

## V1: generic Triton

- Packed INT4 unpack + W4A16 matmul.
- Dynamic A4 quantization kernel.
- W4A4 GEMM for Krea2 shapes.
- Correctness tests against PyTorch simulation.

## V2: Blackwell/B200

- KernelIDE B200 benchmark harness.
- SM100/B200 and SM120 tuned tile sizes.
- Triton `tl.dot_scaled` / FP4/NVFP4 experiments where applicable.
- Optional Gluon fused kernels once the runtime package/API is available.
- Calibrated E2M1/MXFP4 quantizer for Krea2 weights/activations so the verified `tl.dot_scaled` speed path becomes an image-quality-preserving inference path.

## V3: fused runtime

- Fuse activation quantization with low-rank down projection.
- Fuse INT4 GEMM with low-rank up projection.
- Avoid extra reads/writes of BF16 activations and outputs.

## V4: packaging

- Safetensors model format.
- HF repo upload scripts.
- Diffusers pipeline loader.
- ComfyUI / FastAPI integration.
