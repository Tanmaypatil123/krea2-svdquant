# B200 kernel notes

## Verified environment

KernelIDE B200 reports:

```text
GPU=NVIDIA B200
SM=(10, 0)
Triton=3.7.0
Torch=2.12.0+cu130
```

Working Gluon import path:

```python
from triton.experimental import gluon
import triton.experimental.gluon.language as gl
import triton.experimental.gluon.language.nvidia.blackwell as bw
```

Submit Gluon kernels through KernelIDE as Triton:

```bash
kernelide submit scripts/kernelide_smoke_gluon.py --language triton --gpu B200 --timeout 120
```

## Current real kernel baseline

Script:

```bash
kernelide submit scripts/kernelide_w4a16_linear_triton.py --language triton --gpu B200 --timeout 180
```

Result from the first packed W4A16 baseline:

```text
GPU=NVIDIA B200 SM=(10, 0) triton=3.7.0
case M=512 N=4096 K=6144 dtype=float16 time_ms=6.621 dense_equiv_tflops=3.89 max_err=6.706e-07 mean_err=7.357e-08 rel_l2=7.340e-06
case M=512 N=16384 K=6144 dtype=float16 time_ms=26.191 dense_equiv_tflops=3.94 max_err=6.706e-07 mean_err=7.349e-08 rel_l2=7.337e-06
PASS packed W4A16 Triton linear smoke
```

Interpretation: this kernel is correct and uses packed INT4 weights, but it is intentionally naive and not a speedup kernel yet. It scalar-unpacks/dequantizes inside the K loop and then uses normal `tl.dot`. It is a correctness/regression baseline.

## B200 speed path

Do **not** optimize the naive scalar-unpack kernel too far. The B200 path should move to one of these:

### 1. Triton `tl.dot_scaled`

B200 Triton 3.7 exposes:

```python
tl.dot_scaled(lhs, lhs_scale, lhs_format, rhs, rhs_scale, rhs_format, ...)
```

Supported formats include:

```text
e2m1  # packed FP4 / MXFP4-like
_e4m3 / e5m2 FP8
bf16 / fp16
```

For packed FP4, inputs are `uint8` with two FP4 values per byte. Scales use microscaling shapes:

```text
lhs_scale: [M, K // 32]
rhs_scale: [N, K // 32]
```

This is the likely fastest Triton path for Blackwell-family kernels.

Verified speed smoke:

```bash
kernelide submit scripts/kernelide_dot_scaled_e2m1_triton.py --language triton --gpu B200 --timeout 180
```

Result:

```text
GPU=NVIDIA B200 SM=(10, 0) triton=3.7.0
case M=512 N=4096 K=6144 time_ms=0.072 dense_equiv_tflops=358.56 mean_abs=1.832e+02
case M=512 N=16384 K=6144 time_ms=0.258 dense_equiv_tflops=398.90 mean_abs=1.829e+02
PASS tl.dot_scaled fp16 x e2m1 B200 smoke
```

This is a raw capability benchmark with random e2m1 bytes, not a calibrated Krea2 quality test yet. It shows why the B200 path should target `tl.dot_scaled`/TCGEN05 rather than scalar int4 unpack.

### 2. Gluon Blackwell TCGEN05 path

Gluon exposes Blackwell helpers:

```python
import triton.experimental.gluon.language.nvidia.blackwell as bw
bw.tcgen05_mma
bw.tcgen05_mma_scaled
```

`tcgen05_mma_scaled` supports:

```text
e2m1, e4m3, e5m2
```

This is the lower-level Blackwell path to use once the data layout is fixed.

## Next implementation target

Convert the speed smoke into a calibrated kernel:

```text
scripts/kernelide_mxfp4_dot_scaled_triton.py
```

Goal:

```text
A: fp16/bf16 activations, initially as `lhs_format="fp16"` or converted to e2m1 later
B: packed e2m1 weights with uint8 scales
Compute: tl.dot_scaled(...)
Validate against PyTorch dequant reference
Benchmark Krea2 shapes: K=6144, N=4096/16384
```

After that works, integrate into:

```text
src/krea2_svdquant/kernels/triton/blackwell_int4.py
```

Then add Gluon TCGEN05 implementation with explicit layouts.

## Generic non-Blackwell Triton kernels

These intentionally avoid B200-only `dot_scaled`/TCGEN05 features.

Packed W4A16 linear on H100:

```bash
kernelide submit scripts/kernelide_w4a16_linear_triton.py --language triton --gpu H100 --timeout 180
```

Result:

```text
GPU=NVIDIA H100 80GB HBM3 SM=(9, 0) triton=3.7.0
case M=512 N=4096 K=6144 dtype=float16 time_ms=7.707 dense_equiv_tflops=3.34 rel_l2=7.288e-06
case M=512 N=16384 K=6144 dtype=float16 time_ms=29.728 dense_equiv_tflops=3.47 rel_l2=7.339e-06
PASS packed W4A16 Triton linear smoke
```

Generic rowwise activation INT4 quantization on H100:

```bash
kernelide submit scripts/kernelide_activation_quant_triton.py --language triton --gpu H100 --timeout 120
```

Result:

```text
GPU=NVIDIA H100 80GB HBM3 SM=(9, 0) triton=3.7.0
rowwise_int4_quant M=1024 K=6144 time_ms=0.018 effective_GBps=880.05 max_err=7.263e-01 mean_err=2.099e-04
PASS generic Triton activation INT4 quant smoke
```

## Claude Code status

Attempted to ask Claude Code with:

```bash
claude -p '...' --model opus --effort high
```

but the local Claude Code CLI returned:

```text
Failed to authenticate. API Error: 401 Invalid authentication credentials
```

`claude auth status --text` still reports Pro login, so this is a Claude Code/API auth issue, not a missing install. Re-run after refreshing Claude auth.
