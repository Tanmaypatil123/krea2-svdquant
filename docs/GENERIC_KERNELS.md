# Generic non-Blackwell kernels

These kernels are the portable fallback path for NVIDIA GPUs that are **not** B200/Blackwell-family.

Target examples:

```text
T4 / SM75
A10 / Ampere
A100 / SM80
RTX 3090 / SM86
RTX 4090 / Ada SM89
L4 / Ada SM89
L40S / Ada SM89
H100 / SM90
H200 / SM90
```

Do **not** use Blackwell-only assumptions here:

```text
no Gluon TCGEN05
no Blackwell tensor memory
no B200-specific layouts
no mandatory tl.dot_scaled/e2m1/MXFP4 path
```

Allowed generic Triton features:

```text
tl.load / tl.store
tl.arange
tl.max / reductions
bit operations for int4 pack/unpack
normal tl.dot with fp16/bf16 operands
standard program grids and masks
```

## Current generic scripts

### Activation INT4 quantization

```bash
kernelide submit scripts/kernelide_activation_quant_triton.py --language triton --gpu H100 --timeout 120
```

This is a normal Triton rowwise quantization kernel:

```text
X_fp16 [M, K] -> packed X_int4 [M, K/2] + row_scale [M]
```

Verified on H100:

```text
rowwise_int4_quant M=1024 K=6144 time_ms=0.018 effective_GBps=880.05
PASS generic Triton activation INT4 quant smoke
```

### Packed W4A16 linear baseline

```bash
kernelide submit scripts/kernelide_w4a16_linear_triton.py --language triton --gpu H100 --timeout 180
```

This is a portable correctness baseline:

```text
X_fp16 [M, K] @ W_int4 [N, K] -> Y_fp32 [M, N]
```

It should compile on non-Blackwell NVIDIA GPUs supported by Triton. It is intentionally simple and not yet the final speed path.

## Runtime selection rule

Use Blackwell path only for B200/Blackwell-family GPUs:

```python
if sm_major >= 10:
    backend = "blackwell"
else:
    backend = "generic"
```

Generic does not mean slow forever. It means portable first. Optimization should happen within normal Triton/CUTLASS-compatible constraints for each GPU family.

## Next generic optimization work

1. Improve W4A16 tile sizes and memory layout.
2. Add autotune configs for SM75/80/86/89/90.
3. Add optional FP8 path for Hopper/Ada where it is faster.
4. Fuse activation quantization with W4A4 GEMM for a true portable W4A4 path.
5. Keep q/k or first/last layers BF16 if image quality requires it.
