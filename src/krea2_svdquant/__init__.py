"""Krea2 SVDQuant research runtime."""

from .config import BackendKind, SVDQuantConfig

__all__ = ["BackendKind", "SVDQuantConfig", "Krea2SVDQuantTransformer2DModel"]


def __getattr__(name: str):
    if name == "Krea2SVDQuantTransformer2DModel":
        from .transformer import Krea2SVDQuantTransformer2DModel

        return Krea2SVDQuantTransformer2DModel
    raise AttributeError(name)
