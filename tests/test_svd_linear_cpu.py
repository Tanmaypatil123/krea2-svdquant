import torch

from krea2_svdquant.quant.svd import quantize_linear_from_samples, svdquant_linear_sim


def test_svdquant_linear_sim_shape():
    torch.manual_seed(0)
    x = torch.randn(8, 32)
    w = torch.randn(16, 32) * 0.02
    state = quantize_linear_from_samples(w, None, x, rank=4, group_size=16)
    y = svdquant_linear_sim(x, state)
    assert y.shape == (8, 16)
    assert torch.isfinite(y).all()
