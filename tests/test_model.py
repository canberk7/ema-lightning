import torch
import torch.nn as nn

from ema_lightning.decoder import Decoder
from ema_lightning.model import masked_group_norm


def test_masked_group_norm_matches_group_norm_without_padding():
    norm = nn.GroupNorm(1, 8)
    norm.weight.data.normal_()
    norm.bias.data.normal_()
    x = torch.randn(3, 8, 11)
    assert torch.allclose(masked_group_norm(x, norm, torch.ones(3, 1, 11)), norm(x), atol=1e-5)


def test_text_stage_is_exact_under_padding(model):
    ids = [torch.randint(2, 40, (n,)) for n in (7, 13)]
    batch = torch.zeros(2, 20, dtype=torch.long)
    for i, row in enumerate(ids):
        batch[i, :row.numel()] = row
    h, dur = model.text_stage(batch, batch != 0)
    for i, row in enumerate(ids):
        h1, d1 = model.text_stage(row[None], torch.ones(1, row.numel(), dtype=torch.bool))
        assert torch.allclose(h[i, :row.numel()], h1[0], atol=1e-5)
        assert torch.allclose(dur[i, :row.numel()], d1[0], atol=1e-5)
        assert (dur[i, row.numel():] == 0).all()


def test_decoder_is_exact_under_padding():
    torch.manual_seed(0)
    decoder = Decoder(ch=64, rb_kernels=(3, 5), rb_dilations=((1, 3), (1, 3))).eval()
    z = torch.randn(1, 64, 20)
    padded = torch.cat([z, torch.randn(1, 64, 13)], -1)
    exact = decoder(z)
    masked = decoder(padded, torch.tensor([20]))[:, :exact.shape[1]]
    assert torch.allclose(masked, exact, atol=1e-6)
