"""KernelIDE B200 smoke test for Triton Experimental Gluon.

Submit as Triton language for now:
  kernelide submit scripts/kernelide_smoke_gluon.py --language triton --gpu B200 --timeout 120
"""

import torch
import triton
from triton.experimental import gluon
import triton.experimental.gluon.language as gl


@gluon.jit
def add_kernel(x, y, z, n: gl.constexpr, BLOCK: gl.constexpr, layout: gl.constexpr):
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=layout)
    mask = offs < n
    a = gl.load(x + offs, mask=mask, other=0.0)
    b = gl.load(y + offs, mask=mask, other=0.0)
    gl.store(z + offs, a + b, mask=mask)


def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    # Gluon needs an explicit distributed layout; auto layout fails for this simple
    # arange/load/store kernel on the current B200 image.
    layout = gl.BlockedLayout([1], [32], [4], [0])
    add_kernel[(triton.cdiv(n, 256),)](x, y, z, n, BLOCK=256, layout=layout)
    torch.cuda.synchronize()
    max_err = (z - (x + y)).abs().max().item()
    print(f"GPU={torch.cuda.get_device_name()} SM={torch.cuda.get_device_capability()} max_err={max_err:.3e}")
    assert max_err < 1e-6
    print("PASS kernelide Gluon smoke")


if __name__ == "__main__":
    main()
