import torch
from torch import nn

from krea2_svdquant.runtime.block_offload import TransformerBlockOffloader, enable_block_offload


class _Block(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.register_buffer("scale", torch.ones(dim))

    def forward(self, x):
        return self.lin(x) * self.scale


class _Toy(nn.Module):
    def __init__(self, dim: int, num_blocks: int):
        super().__init__()
        self.embed = nn.Linear(dim, dim)
        self.transformer_blocks = nn.ModuleList([_Block(dim) for _ in range(num_blocks)])
        self.norm_out = nn.Linear(dim, dim)

    def forward(self, x):
        x = self.embed(x)
        for block in self.transformer_blocks:
            x = block(x)
        return self.norm_out(x)


def _make_pair(dim=8, num_blocks=5):
    torch.manual_seed(0)
    ref = _Toy(dim, num_blocks)
    swapped = _Toy(dim, num_blocks)
    swapped.load_state_dict(ref.state_dict())
    return ref, swapped


def test_block_offload_matches_reference_cpu():
    ref, swapped = _make_pair()
    x = torch.randn(2, 8)
    expected = ref(x)

    off = enable_block_offload(swapped, "cpu", num_blocks_on_gpu=2)
    assert off.num_blocks == 5
    assert off.attr_names == ["transformer_blocks"]

    out = swapped(x)
    assert torch.allclose(expected, out, atol=1e-6)


def test_resident_window_bounded_and_cleanup():
    _, swapped = _make_pair(num_blocks=6)
    off = TransformerBlockOffloader(swapped, "cpu", num_blocks_on_gpu=2).install()
    swapped(torch.randn(1, 8))
    # The resident window is bounded by num_blocks_on_gpu after a full pass.
    assert len(off._resident) <= 2
    off.remove()
    assert len(off._resident) == 0
    assert len(off._handles) == 0
    # Every block is back on CPU after teardown.
    assert all(p.device.type == "cpu" for p in swapped.transformer_blocks.parameters())


def test_missing_container_raises():
    bare = nn.Linear(4, 4)
    try:
        TransformerBlockOffloader(bare, "cpu")
    except ValueError:
        return
    raise AssertionError("expected ValueError for missing block container")
