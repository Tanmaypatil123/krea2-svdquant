from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Sequence

import torch
from torch import nn

# Block-container attribute names tried in order. Krea2 / Flux-style transformers
# expose dual- and single-stream stacks; both are large and worth offloading.
_DEFAULT_BLOCK_ATTRS: tuple[str, ...] = ("transformer_blocks", "single_transformer_blocks")


def _first_tensor_device(args: tuple, kwargs: dict) -> torch.device | None:
    """Return the device of the first CUDA tensor found in a forward call."""
    for value in list(args) + list(kwargs.values()):
        if isinstance(value, torch.Tensor) and value.is_cuda:
            return value.device
    return None


class _BlockSwap:
    """Owns the CPU master copy of one transformer block and swaps it on/off GPU.

    Block weights are read-only during inference, so we never copy data back from
    GPU. ``to_gpu`` points each parameter/buffer at a fresh device tensor and
    ``to_cpu`` simply restores the cached CPU tensor, letting the GPU copy free. The
    cached CPU tensors can be pinned once so host->device copies run asynchronously.
    """

    def __init__(self, block: nn.Module, *, pin_memory: bool):
        self.block = block
        self.on_gpu = False
        block.to("cpu")

        pin = pin_memory and torch.cuda.is_available()
        self._cpu_params: list[tuple[nn.Parameter, torch.Tensor]] = []
        for param in block.parameters(recurse=True):
            tensor = param.data
            if pin and not tensor.is_pinned():
                tensor = tensor.pin_memory()
                param.data = tensor
            self._cpu_params.append((param, tensor))

        self._cpu_buffers: list[tuple[nn.Module, str, torch.Tensor]] = []
        for module in block.modules():
            for name, buf in module._buffers.items():
                if buf is None:
                    continue
                tensor = buf
                if pin and not tensor.is_pinned():
                    tensor = tensor.pin_memory()
                    module._buffers[name] = tensor
                self._cpu_buffers.append((module, name, tensor))

    def to_gpu(self, device: torch.device, *, non_blocking: bool) -> None:
        if self.on_gpu:
            return
        for param, cpu_tensor in self._cpu_params:
            param.data = cpu_tensor.to(device, non_blocking=non_blocking)
        for module, name, cpu_tensor in self._cpu_buffers:
            module._buffers[name] = cpu_tensor.to(device, non_blocking=non_blocking)
        self.on_gpu = True

    def to_cpu(self) -> None:
        if not self.on_gpu:
            return
        for param, cpu_tensor in self._cpu_params:
            param.data = cpu_tensor
        for module, name, cpu_tensor in self._cpu_buffers:
            module._buffers[name] = cpu_tensor
        self.on_gpu = False


class TransformerBlockOffloader:
    """Keep transformer blocks on CPU and stream them through CUDA per forward.

    Non-block transformer modules (embedders, norms, projections) are placed on the
    CUDA device once. Each block is moved to the device only while it runs and
    evicted back to CPU afterwards, so peak block VRAM is bounded by
    ``num_blocks_on_gpu`` (+1 transiently while the next block is prefetched).

    This deliberately avoids ``transformer.to(cuda)`` for the block stacks, so the
    full-precision block weights never have to fit on the GPU at once.
    """

    def __init__(
        self,
        transformer: nn.Module,
        device: torch.device | str,
        *,
        num_blocks_on_gpu: int = 1,
        pin_memory: bool = False,
        block_attrs: Sequence[str] | None = None,
    ):
        self.transformer = transformer
        self.device = torch.device(device)
        self.num_blocks_on_gpu = max(1, int(num_blocks_on_gpu))
        self.pin_memory = bool(pin_memory)
        self._handles: list[torch.utils.hooks.RemovableHandle] = []
        self._swaps: dict[int, _BlockSwap] = {}
        self._resident: deque[int] = deque()
        self._containers = self._find_containers(block_attrs)
        if not self._containers:
            raise ValueError(
                "no transformer block containers found; expected one of "
                f"{block_attrs or _DEFAULT_BLOCK_ATTRS}"
            )

    @property
    def num_blocks(self) -> int:
        return sum(len(blocks) for _, blocks in self._containers)

    @property
    def attr_names(self) -> list[str]:
        return [attr for attr, _ in self._containers]

    def _find_containers(self, block_attrs: Sequence[str] | None) -> list[tuple[str, nn.ModuleList]]:
        names = tuple(block_attrs) if block_attrs else _DEFAULT_BLOCK_ATTRS
        found: list[tuple[str, nn.ModuleList]] = []
        for attr in names:
            container = getattr(self.transformer, attr, None)
            if isinstance(container, nn.ModuleList) and len(container) > 0:
                found.append((attr, container))
        return found

    def _iter_blocks(self) -> Iterable[nn.Module]:
        for _, container in self._containers:
            yield from container

    def _move_non_blocks_to_device(self) -> None:
        """Move every transformer submodule except the block stacks to the device.

        The block containers are temporarily detached so ``Module.to`` skips them,
        avoiding a full-transformer GPU materialization before offloading.
        """
        saved: list[tuple[str, nn.ModuleList]] = []
        for attr, container in self._containers:
            saved.append((attr, container))
            setattr(self.transformer, attr, None)
        try:
            self.transformer.to(self.device)
        finally:
            for attr, container in saved:
                setattr(self.transformer, attr, container)

    def install(self) -> "TransformerBlockOffloader":
        self._move_non_blocks_to_device()
        for block in self._iter_blocks():
            self._swaps[id(block)] = _BlockSwap(block, pin_memory=self.pin_memory)
            pre = block.register_forward_pre_hook(self._pre_hook, with_kwargs=True)
            self._handles.append(pre)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return self

    def _pre_hook(self, block: nn.Module, args: tuple, kwargs: dict):
        device = _first_tensor_device(args, kwargs) or self.device
        swap = self._swaps[id(block)]
        swap.to_gpu(device, non_blocking=self.pin_memory)
        key = id(block)
        if key in self._resident:
            self._resident.remove(key)
        self._resident.append(key)
        # Evict least-recently-used blocks beyond the resident window, but never the
        # block that is about to run.
        while len(self._resident) > self.num_blocks_on_gpu:
            old = self._resident.popleft()
            if old == key:
                self._resident.append(old)
                break
            self._swaps[old].to_cpu()
        return None

    def offload_all(self) -> None:
        """Return every block to CPU and drop the resident window."""
        for swap in self._swaps.values():
            swap.to_cpu()
        self._resident.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def remove(self) -> None:
        """Remove hooks and offload blocks back to CPU."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        self.offload_all()


def enable_block_offload(
    transformer: nn.Module,
    device: torch.device | str,
    *,
    num_blocks_on_gpu: int = 1,
    pin_memory: bool = False,
    block_attrs: Sequence[str] | None = None,
) -> TransformerBlockOffloader:
    """Install transformer-block CPU offload and return the live offloader.

    Keeps ``transformer_blocks`` (and ``single_transformer_blocks`` when present) on
    CPU, streaming each block onto ``device`` only for its forward pass. Non-block
    modules stay resident on ``device``. Compatible with the SVDQuant packed/chunked
    runtime: ``SVDQuantLinear`` buffers move with their owning block.
    """
    offloader = TransformerBlockOffloader(
        transformer,
        device,
        num_blocks_on_gpu=num_blocks_on_gpu,
        pin_memory=pin_memory,
        block_attrs=block_attrs,
    )
    return offloader.install()
