from __future__ import annotations

from collections.abc import Iterable

from torch import nn

from krea2_svdquant.config import SVDQuantConfig


def iter_named_linears(root: nn.Module, target_suffixes: Iterable[str] | None = None):
    suffixes = tuple(target_suffixes or SVDQuantConfig().target_modules)
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and any(name.endswith(s) for s in suffixes):
            yield name, module


def get_parent_module(root: nn.Module, dotted_name: str):
    parts = dotted_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = parent[int(p)] if p.isdigit() else getattr(parent, p)
    return parent, parts[-1]


def replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module):
    parent, leaf = get_parent_module(root, dotted_name)
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)
