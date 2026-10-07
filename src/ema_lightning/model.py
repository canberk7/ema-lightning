"""EMA Lightning's acoustic model: Turkish letters in, 64-dim latents at 25 Hz out, in four steps.

Both stages take padded batches with masks and return exactly what each item would get alone, so the
plain path, the batched path and the recorded CUDA graphs all run these same functions.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def rope(pos, dim):
    inv = 1.0 / (10000.0 ** (torch.arange(dim // 2, device=pos.device).float() / (dim // 2)))
    ang = pos[..., None].float() * inv
    ang = torch.cat([ang, ang], -1)
    return ang.cos(), ang.sin()


def apply_rope(x, cos, sin):
    a, b = x.chunk(2, -1)
    return x * cos + torch.cat([-b, a], -1) * sin


class SwiGLU(nn.Module):
    def __init__(self, d, mult=4):
        super().__init__()
        hidden = int(d * mult * 2 / 3)
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Attention(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)

    def forward(self, x, cos, sin, mask):
        B, T, D = x.shape
        q, k, v = (t.view(B, T, self.h, self.dh).transpose(1, 2) for t in self.qkv(x).chunk(3, -1))
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask[:, None, None, :])
        return self.proj(x.transpose(1, 2).reshape(B, T, D))


class ConvNeXtBlock(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.dw = nn.Conv1d(d, d, 7, padding=3, groups=d)
        self.norm = nn.LayerNorm(d)
        self.pw1 = nn.Linear(d, 2 * d)
        self.pw2 = nn.Linear(2 * d, d)
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x, mask):
        y = self.norm(self.dw((x * mask[..., None]).transpose(1, 2)).transpose(1, 2))  # padding reads as zeros
        return x + self.gamma * self.pw2(F.gelu(self.pw1(y)))


class TextBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d)

    def forward(self, x, cos, sin, mask):
        x = x + self.attn(self.n1(x), cos, sin, mask)
        return x + self.ff(self.n2(x))


class TextEncoder(nn.Module):
    def __init__(self, n_vocab, d, n_conv, n_attn, heads):
        super().__init__()
        self.dh = d // heads
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.conv = nn.ModuleList(ConvNeXtBlock(d) for _ in range(n_conv))
        self.attn = nn.ModuleList(TextBlock(d, heads) for _ in range(n_attn))
        self.norm = nn.LayerNorm(d)

    def forward(self, ids, mask):
        x = self.emb(ids)
        for block in self.conv:
            x = block(x, mask)
        cos, sin = rope(torch.arange(ids.shape[1], device=ids.device), self.dh)
        for block in self.attn:
            x = block(x, cos, sin, mask)
        return self.norm(x)


def masked_group_norm(x, norm, m):
    """GroupNorm(1, C) with statistics over real positions only."""
    count = m.sum((1, 2), keepdim=True) * x.shape[1]
    mean = (x * m).sum((1, 2), keepdim=True) / count
    var = (((x - mean) * m) ** 2).sum((1, 2), keepdim=True) / count
    return (x - mean) / torch.sqrt(var + norm.eps) * norm.weight[None, :, None] + norm.bias[None, :, None]


class Duration(nn.Module):
    def __init__(self, d, hidden):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d, hidden, 3, padding=1), nn.SiLU(), nn.GroupNorm(1, hidden), nn.Identity(),
            nn.Conv1d(hidden, hidden, 3, padding=1), nn.SiLU(), nn.GroupNorm(1, hidden))
        self.out = nn.Linear(hidden, 1)

    def forward(self, h, mask):
        m = mask[:, None].float()
        x = masked_group_norm(F.silu(self.net[0](h.transpose(1, 2) * m)), self.net[2], m) * m
        x = masked_group_norm(F.silu(self.net[4](x)), self.net[6], m)
        log_d = self.out(x.transpose(1, 2)).squeeze(-1).masked_fill(~mask, 0.0)
        return torch.expm1(log_d.clamp(max=6.0)).clamp(min=1e-3) * mask.float()


class Aligner(nn.Module):
    def __init__(self, d, heads, pos_scale, lookback, lookahead, letter_pos):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.pos_scale, self.lookback, self.lookahead, self.letter_pos = pos_scale, lookback, lookahead, letter_pos
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.frame_q = nn.Parameter(torch.zeros(1, 1, d))
        self.log_sigma = nn.Parameter(torch.zeros(heads))
        self.bias_w = nn.Parameter(torch.zeros(heads))
        self.log_temp = nn.Parameter(torch.zeros(()))

    def forward(self, h, cw, cp, fw, fp, mask, n_words):
        B, L, D = h.shape
        T = fw.shape[1]
        if self.letter_pos:
            c, f = cw.clamp(min=0), fw.clamp(min=0)
            wlen = torch.zeros(B, n_words, device=h.device).scatter_add_(1, c, (cw >= 0).float()).clamp(min=1.0)
            woff = wlen.cumsum(-1) - wlen
            cg = woff.gather(1, c) + cp.float() * wlen.gather(1, c)
            fg = woff.gather(1, f) + fp.float() * wlen.gather(1, f)
        else:
            cg, fg = cw.clamp(min=0).float() + cp, fw.float() + fp
        q = self.q(self.frame_q.expand(B, T, D)).view(B, T, self.h, self.dh).transpose(1, 2)
        k = self.k(h).view(B, L, self.h, self.dh).transpose(1, 2)
        v = self.v(h).view(B, L, self.h, self.dh).transpose(1, 2)
        q = apply_rope(q, *(t[:, None] for t in rope(fg * self.pos_scale, self.dh)))
        k = apply_rope(k, *(t[:, None] for t in rope(cg * self.pos_scale, self.dh)))
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.dh) * self.log_temp.exp()
        sig2 = (self.log_sigma.exp() ** 2).view(1, self.h, 1, 1)
        dist2 = ((fg[:, :, None] - cg[:, None, :]) ** 2)[:, None]
        logits = logits - self.bias_w.view(1, self.h, 1, 1) * dist2 / (2 * sig2)
        rel = cw[:, None, :] - fw[:, :, None]
        allow = (rel >= -self.lookback) & (rel <= self.lookahead) & mask[:, None, :] & (cw[:, None, :] >= 0)
        attn = logits.masked_fill(~allow[:, None], -1e4).softmax(-1) * allow[:, None]
        return self.o((attn @ v).transpose(1, 2).reshape(B, T, D))


class DiTBlock(nn.Module):
    def __init__(self, d, heads, ff_mult, shared):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.n2 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn, self.ff = Attention(d, heads), SwiGLU(d, ff_mult)
        if shared:
            self.ada_offset = nn.Parameter(torch.zeros(6, d))
        else:
            self.ada = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))

    def forward(self, x, c, cos, sin, mask):
        p = (c + self.ada_offset[None]).unbind(1) if hasattr(self, "ada_offset") else self.ada(c).chunk(6, -1)
        sa, ga, aa, sf, gf, af = (t.unsqueeze(1) for t in p)
        x = x + aa * self.attn(self.n1(x) * (1 + ga) + sa, cos, sin, mask)
        return x + af * self.ff(self.n2(x) * (1 + gf) + sf)


class TimestepEmbed(nn.Module):
    def __init__(self, d, freq=256):
        super().__init__()
        self.freq = freq
        self.mlp = nn.Sequential(nn.Linear(freq, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        half = self.freq // 2
        a = t[:, None] * 1000.0 * torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)[None]
        return self.mlp(torch.cat([a.cos(), a.sin()], -1))


class Acoustic(nn.Module):
    def __init__(self, cfg, vocab, shared):
        super().__init__()
        d, heads = cfg["d"], cfg["n_heads"]
        self.d, self.dh, self.latent_dim, self.shared = d, d // heads, cfg.get("latent_dim", 64), shared
        self.vocab = list(vocab)
        self.stoi = {ch: i for i, ch in enumerate(self.vocab)}
        self.times = [float(t) for t in cfg["distilled"]["times"]]
        self.text = TextEncoder(len(vocab), d, cfg.get("text_conv", 4), cfg.get("text_attn", 2), heads)
        self.chardur = Duration(d, cfg.get("dur_hidden", 256))
        self.aligner = Aligner(d, cfg.get("align_heads", 4), cfg.get("pos_scale", 24.0), cfg.get("lookback", 1),
                               cfg.get("lookahead", 1), bool(cfg.get("letter_pos", False)))
        self.in_proj = nn.Linear(self.latent_dim, d)
        self.t_embed = TimestepEmbed(d)
        if shared:
            self.ada_shared = nn.Sequential(nn.SiLU(), nn.Linear(d, 6 * d))
        self.blocks = nn.ModuleList(DiTBlock(d, heads, cfg.get("ff_mult", 4), shared) for _ in range(cfg["n_layers"]))
        self.norm_out = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Sequential(nn.SiLU(), nn.Linear(d, 2 * d))
        self.out_proj = nn.Linear(d, self.latent_dim)

    def backbone(self, x, cond, t, mask, cos, sin):
        c = self.t_embed(t)
        bc = self.ada_shared(c).view(-1, 6, self.d) if self.shared else c
        x = self.in_proj(x) + cond
        for block in self.blocks:
            x = block(x, bc, cos, sin, mask)
        s, g = self.ada_out(c).chunk(2, -1)
        return self.out_proj(self.norm_out(x) * (1 + g.unsqueeze(1)) + s.unsqueeze(1))

    def text_stage(self, ids, mask):
        """Letters to text features and per-letter durations in frames."""
        h = self.text(ids, mask)
        return h, self.chardur(h, mask)

    def sound_stage(self, h, dur, mask, cw, wstart, fw, fp, fmask, noise):
        """Text features and a frame timeline to latents, in len(self.times) steps from the given noise."""
        c = dur.clamp(min=1e-4) * mask
        done = c.cumsum(-1)
        before = done - c
        word = cw.clamp(min=0)
        total = torch.zeros_like(c).scatter_add_(1, word, c).gather(1, word)
        cp = ((done - before.gather(1, wstart) - 0.5 * c) / total.clamp(min=1e-8)).clamp(0.0, 1.0) * mask
        cond = self.aligner(h, cw, cp, fw, fp, mask, cw.shape[1])
        cos, sin = rope(torch.arange(fw.shape[1], device=h.device), self.dh)
        x = noise[:, 0]
        for k, t in enumerate(self.times):
            step = torch.full((x.shape[0],), t, device=x.device)
            x1 = x + (1 - t) * self.backbone(x, cond, step, fmask, cos, sin)
            if k + 1 < len(self.times):
                x = (1 - self.times[k + 1]) * noise[:, k + 1] + self.times[k + 1] * x1
        return x1


def load_acoustic(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=True)
    sd = ck["ema"]
    shared = bool(ck["cfg"].get("shared_ada", any(k.startswith("ada_shared") for k in sd)))
    model = Acoustic(ck["cfg"], ck["vocab"], shared)
    model.load_state_dict(sd)
    return model.to(device).float().eval().requires_grad_(False)
