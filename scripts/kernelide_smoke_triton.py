import torch
import triton
import triton.language as tl


@triton.jit
def add_kernel(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    a = tl.load(x + offs, mask=mask, other=0.0)
    b = tl.load(y + offs, mask=mask, other=0.0)
    tl.store(z + offs, a + b, mask=mask)


def main():
    n = 1 << 20
    x = torch.randn(n, device="cuda", dtype=torch.float32)
    y = torch.randn(n, device="cuda", dtype=torch.float32)
    z = torch.empty_like(x)
    add_kernel[(triton.cdiv(n, 256),)](x, y, z, n, BLOCK=256)
    torch.cuda.synchronize()
    max_err = (z - (x + y)).abs().max().item()
    name = torch.cuda.get_device_name()
    sm = torch.cuda.get_device_capability()
    print(f"GPU={name} SM={sm[0]}{sm[1]} max_err={max_err:.3e}")
    assert max_err < 1e-6
    print("PASS kernelide Triton smoke")


if __name__ == "__main__":
    main()
