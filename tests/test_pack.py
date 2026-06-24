import torch

from krea2_svdquant.quant.pack import pack_int4, unpack_int4, quantize_symmetric_int4, dequantize_symmetric_int4


def test_pack_roundtrip():
    q = torch.arange(-8, 8, dtype=torch.int8).repeat(3, 1)
    packed = pack_int4(q)
    out = unpack_int4(packed, q.shape[-1])
    assert torch.equal(q, out)


def test_quant_dequant_shape():
    x = torch.randn(7, 130)
    q, s = quantize_symmetric_int4(x, group_size=64, dim=1)
    y = dequantize_symmetric_int4(q, s, group_size=64, dim=1)
    assert y.shape[-1] == 192
    assert torch.isfinite(y).all()
