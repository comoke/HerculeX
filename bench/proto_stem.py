"""Prototype: the depth stem through MPSGraph convolutions (PyTorch MPS) instead of the hand-written Metal kernel.
Speed of several formulations and how far their features are from the stock kernel's. Run from a checkout root."""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import torch
from torch.nn import functional as F

from ink.model import models
from ink.model.reader import lane_reader
from ink.model.tiles import scaled

cache = Path(sys.argv[1])
reader = lane_reader(models.chosen().weights)
stem, sync = reader.stem, torch.mps.synchronize
CL = torch.channels_last


def weights_of(fixed):
    """Folded conv weights from the kernel's own (rounded, norm-folded) half weights, so both paths start equal."""
    w1, b1, w2, b2, att = fixed[1]
    w1 = w1.reshape(3, 3, 3, 16).permute(3, 0, 1, 2).contiguous()  # [out, dz, dy, dx]
    w2 = w2.reshape(3, 3, 3, 16, 16).permute(4, 0, 3, 1, 2).contiguous()  # [out, dz, in, dy, dx]
    return w1, b1, w2, b2, att


def stem_a(block, fixed, table, bias, layout=None):
    """Stacked: depth neighbours as channels, one conv per layer (48 -> 16)."""
    w1, b1, w2, b2, att = weights_of(fixed)
    z, _, _ = block.shape
    x = table[block.long()]  # [z, h, w] half
    xp = F.pad(x, (0, 0, 0, 0, 1, 1))
    s1 = torch.stack([xp[k : k + z] for k in range(3)], 1)  # [z, 3, h, w]
    k2 = w2.reshape(16, 48, 3, 3)
    if layout:
        s1, w1, k2 = (
            s1.contiguous(memory_format=layout),
            w1.contiguous(memory_format=layout),
            k2.contiguous(memory_format=layout),
        )
    a = F.leaky_relu(F.conv2d(s1, w1, b1, padding=1), 0.01)
    ap = F.pad(a, (0, 0, 0, 0, 0, 0, 1, 1))
    s2 = torch.cat([ap[k : k + z] for k in range(3)], 1)  # [z, 48, h, w]
    b = F.leaky_relu(F.conv2d(s2, k2, b2, padding=1), 0.01)
    return pooled(b, att, bias)


def stem_b(block, fixed, table, bias, layout=None):
    """Fan-out: one conv 16 -> 48 over each layer, then the three depth taps added with a shift."""
    w1, b1, w2, b2, att = weights_of(fixed)
    z, _, _ = block.shape
    x = table[block.long()][:, None]  # [z, 1, h, w]
    k1 = w1.reshape(48, 1, 3, 3) if False else w1.permute(1, 0, 2, 3).reshape(48, 1, 3, 3)  # [dz*16+out, 1, 3, 3]
    k2 = w2.permute(1, 0, 2, 3, 4).reshape(48, 16, 3, 3)  # [dz*16+out, in, 3, 3]
    if layout:
        x, k1, k2 = (
            x.contiguous(memory_format=layout),
            k1.contiguous(memory_format=layout),
            k2.contiguous(memory_format=layout),
        )

    def taps(y):
        y = F.pad(y, (0, 0, 0, 0, 0, 0, 1, 1))  # depth padded: index z+1 is layer z
        return y[0:z, 0:16] + y[1 : z + 1, 16:32] + y[2 : z + 2, 32:48]

    a = F.leaky_relu(taps(F.conv2d(x, k1, None, padding=1)) + b1.reshape(1, 16, 1, 1), 0.01)
    b = F.leaky_relu(taps(F.conv2d(a, k2, None, padding=1)) + b2.reshape(1, 16, 1, 1), 0.01)
    return pooled(b, att, bias)


def stem_c(block, fixed, table, bias, layout=None):
    """conv3d."""
    w1, b1, w2, b2, att = weights_of(fixed)
    x = table[block.long()][None, None]  # [1, 1, z, h, w]
    a = F.leaky_relu(F.conv3d(x, w1[:, None], b1, padding=1), 0.01)
    b = F.leaky_relu(F.conv3d(a, w2.permute(0, 2, 1, 3, 4).contiguous(), b2, padding=1), 0.01)
    return pooled(b[0].permute(1, 0, 2, 3), att, bias)


def pooled(b, att, bias):
    """[z, 16, h, w] -> [1, 32, h, w]: attention over depth, and the peak."""
    v = b.float()
    logit = (v * att.float().reshape(1, 16, 1, 1)).sum(1, keepdim=True) + bias
    share = torch.softmax(logit, 0)
    return torch.cat([(v * share).sum(0), v.amax(0)])[None].half()


pieces = [pickle.load(open(f, "rb")) for f in sorted(cache.glob("*.pkl"))[:12:3]]
with torch.inference_mode():
    rows = []
    for piece in pieces:
        (values, coverage), _ = piece["stacks"][-1]
        measure = next(m for _, m in piece["stacks"] if m is not None)
        if values.shape[1] < 512 or values.shape[2] < 512:
            continue
        fixed = stem.statistics_of(*measure)
        table = scaled(torch.arange(256, dtype=torch.int32, device="mps"), fixed[0])
        y, x = (values.shape[1] - 512) // 2, (values.shape[2] - 512) // 2
        block_np = np.ascontiguousarray(values[:, y : y + 512, x : x + 512])
        want = next(iter(stem.tiles(block_np, fixed, 0))).float().cpu().numpy()
        block = torch.from_numpy(block_np).to("mps")
        rows.append((block, fixed, table, want))
    bias = stem.attention_bias
    for name, fn, layout in (
        ("A stacked 48->16, NCHW", stem_a, None),
        ("A stacked 48->16, channels_last", stem_a, CL),
        ("B fan-out 16->48 + shifted add, NCHW", stem_b, None),
        ("B fan-out 16->48 + shifted add, channels_last", stem_b, CL),
        ("C conv3d", stem_c, None),
    ):
        block, fixed, table, want = rows[0]
        for _ in range(3):
            fn(block, fixed, table, bias, layout)
        sync()
        t = time.perf_counter()
        for k in range(20):
            out = fn(*rows[k % len(rows)][:3], bias, layout)
        sync()
        ms = 1000 * (time.perf_counter() - t) / 20
        worst, mean, rel = 0.0, 0.0, 0.0
        for block, fixed, table, want in rows:
            got = fn(block, fixed, table, bias, layout).float().cpu().numpy()
            d = np.abs(got - want)
            worst, mean = max(worst, float(d.max())), mean + float(d.mean()) / len(rows)
            rel = max(rel, float(d.max() / np.abs(want).max()))
        print(
            f"{name:48s} {ms:6.2f} ms/tile | vs stock kernel features: max abs {worst:.4f} mean abs {mean:.6f} (features up to {np.abs(want).max():.1f})",
            flush=True,
        )
