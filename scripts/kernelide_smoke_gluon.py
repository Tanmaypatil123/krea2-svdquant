"""KernelIDE B200 smoke probe for optional Gluon availability.

Submit as Triton language for now:
  kernelide submit scripts/kernelide_smoke_gluon.py --language triton --gpu B200 --timeout 120
"""

import importlib

import torch


def main():
    print(f"GPU={torch.cuda.get_device_name()} SM={torch.cuda.get_device_capability()}")
    candidates = ["gluon", "triton._C.libtriton.gluon", "triton.language.extra.gluon"]
    found = None
    for name in candidates:
        try:
            mod = importlib.import_module(name)
            found = (name, mod)
            break
        except Exception as exc:
            print(f"missing {name}: {type(exc).__name__}: {exc}")
    if found is None:
        print("PASS probe: Gluon module not present yet; Triton Blackwell path remains active.")
        return
    print(f"PASS probe: found Gluon module {found[0]} -> {found[1]}")


if __name__ == "__main__":
    main()
