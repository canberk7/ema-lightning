"""EMA Lightning's decoder: 64-dim latents at 25 Hz in, 48 kHz audio out.

With `lengths`, everything past each row's real length is held at zero after every layer, so a window
padded to a fixed shape decodes exactly like the same window unpadded.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _keep(x, length):
    return x if length is None else x * (torch.arange(x.shape[-1], device=x.device) < length[:, None])[:, None]


class ResBlock(nn.Module):
    def __init__(self, ch, k, dilations):
        super().__init__()
        self.convs1 = nn.ModuleList(nn.Conv1d(ch, ch, k, dilation=d, padding=(k * d - d) // 2) for d in dilations)
        self.convs2 = nn.ModuleList(nn.Conv1d(ch, ch, k, padding=(k - 1) // 2) for _ in dilations)

    def forward(self, x, length=None):
        for c1, c2 in zip(self.convs1, self.convs2, strict=True):
            y = _keep(F.leaky_relu(c1(F.leaky_relu(x, 0.1)), 0.1), length)
            x = _keep(x + c2(y), length)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_dim=64, ch=256, rates=(8, 6, 5, 2, 2, 2), kernels=(16, 12, 10, 4, 4, 4),
                 rb_kernels=(3, 5, 9), rb_dilations=((1, 3, 5),) * 3):
        super().__init__()
        self.hop, self.nk = math.prod(rates), len(rb_kernels)
        self.pre = nn.Conv1d(latent_dim, ch, 7, padding=3)
        self.ups = nn.ModuleList(nn.ConvTranspose1d(ch >> i, ch >> (i + 1), k, r, padding=(k - r) // 2)
                                 for i, (r, k) in enumerate(zip(rates, kernels, strict=True)))
        self.blocks = nn.ModuleList(ResBlock(ch >> (i + 1), k, d)
                                    for i in range(len(rates)) for k, d in zip(rb_kernels, rb_dilations, strict=True))
        self.post = nn.Conv1d(ch >> len(rates), 1, 7, padding=3)

    def forward(self, z, lengths=None):
        """z: [B, latent_dim, T]; lengths: real frames per row, or None when nothing is padded."""
        frames = z.shape[-1]
        length = lengths
        x = _keep(self.pre(_keep(z, length)), length)
        for i, up in enumerate(self.ups):
            x = up(F.leaky_relu(x, 0.1))
            if length is not None:
                length = (length - 1) * up.stride[0] - 2 * up.padding[0] + up.kernel_size[0]
            x = _keep(x, length)
            x = sum(b(x, length) for b in self.blocks[i * self.nk:(i + 1) * self.nk]) / self.nk
        return torch.tanh(self.post(F.leaky_relu(x)))[:, 0, :frames * self.hop]


def load_decoder(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, sd = ck.get("cfg", {}), ck["G"]
    for key in [k for k in sd if k.endswith("weight_g")]:
        g, v = sd.pop(key), sd.pop(key[:-1] + "v")
        sd[key[:-2]] = g * v / v.norm(dim=tuple(range(1, v.dim())), keepdim=True)
    keys = ("latent_dim", "ch", "rates", "kernels", "rb_kernels", "rb_dilations")
    model = Decoder(**{k: cfg[k] for k in keys if k in cfg})
    model.load_state_dict(sd)
    return model.to(device).float().eval().requires_grad_(False)
