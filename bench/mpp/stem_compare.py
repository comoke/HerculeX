"""The tensor-op stem against the one-pass kernel on real tiles: feature agreement and time per tile. Run in the m5 checkout."""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import torch

from ink.model import models, stock_model
from ink.model.reader_apple.tile_reader import DepthStem, TensorStem
from ink.model.tiles import STEM_CHANNELS

cache = Path(sys.argv[1])
weights = models.chosen().weights
old = DepthStem(stock_model.taught_stem(weights, STEM_CHANNELS))
new = TensorStem(stock_model.taught_stem(weights, STEM_CHANNELS))
sync = torch.mps.synchronize
worst, total, count, same, n = 0.0, 0.0, 0, 0, 0
with torch.inference_mode():
    blocks = []
    for f in sorted(cache.glob("*.pkl"))[:16:2]:
        piece = pickle.load(open(f, "rb"))
        (values, coverage), _ = piece["stacks"][-1]
        measure = next(m for _, m in piece["stacks"] if m is not None)
        fixed = old.statistics_of(*measure)
        fixed_new = new.statistics_of(*measure)
        h, w = values.shape[1:]
        cases = []
        if h >= 512 and w >= 512:
            cases.append((np.ascontiguousarray(values[:, (h - 512) // 2 :][:, :512, (w - 512) // 2 :][:, :, :512]), 0))
        if h >= 512 and w >= 792:
            cases.append((np.ascontiguousarray(values[:, :512, :792]), 280))
        th, tw = min(512, h // 32 * 32), min(512, w // 32 * 32)
        if (th, tw) != (512, 512) and th and tw:
            cases.append((np.ascontiguousarray(values[:, :th, :tw]), 0))
        for block, gap in cases:
            a = [t.float().cpu().numpy() for t in old.tiles(block, fixed, gap)]
            b = [t.float().cpu().numpy() for t in new.tiles(block, fixed_new, gap)]
            for x, y in zip(a, b):
                d = np.abs(x - y)
                worst, total, count = max(worst, float(d.max())), total + float(d.sum()), count + d.size
                same += int((x == y).sum())
                n += 1
            if gap == 0 and block.shape[1:] == (512, 512):
                blocks.append((block, fixed, fixed_new))
    print(
        f"{n} tiles (singles, pairs, small): max abs diff {worst:.5f}, mean abs diff {total / count:.7f}, identical elements {same / count:.5f}",
        flush=True,
    )
    for name, stem, k in (("one-pass kernel (stock)", old, 1), ("tensor-op stem", new, 2)):
        for _ in range(3):
            list(stem.tiles(blocks[0][0], blocks[0][k], 0))
        sync()
        t = time.perf_counter()
        for i in range(40):
            block = blocks[i % len(blocks)]
            list(stem.tiles(block[0], block[k], 0))
        sync()
        print(f"{name:26s} {1000 * (time.perf_counter() - t) / 40:6.2f} ms per 512 tile", flush=True)
