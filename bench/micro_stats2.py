"""statistics_of in the checkout it is run from, per call, plus its pieces."""
import os, pickle, sys, time
from pathlib import Path
sys.path.insert(0, os.getcwd())
import torch
from torch.nn import functional as F
from ink.model import models
from ink.model.reader import lane_reader
from ink.model.tiles import scaled
cache = Path(sys.argv[1])
piece = pickle.load(open(sorted(cache.glob("*.pkl"))[0], "rb"))
tiles, scale = next(m for _, m in piece["stacks"] if m is not None)
reader = lane_reader(models.chosen().weights)
stem, sync = reader.stem, torch.mps.synchronize
def clock(name, fn, n=20):
    for _ in range(3): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): out = fn()
    sync(); print(f"{1000 * (time.perf_counter() - t) / n:8.2f} ms  {name}", flush=True)
    return out
with torch.inference_mode():
    clock("whole statistics_of", lambda: stem.statistics_of(tiles, scale))
    f = stem.module.features
    image = scaled(torch.from_numpy(tiles).to("mps"), scale)
    b, d, h, w = image.shape
    def stack1():
        p = F.pad(image, (0, 0, 0, 0, 1, 1))
        return torch.stack([p[:, k:k+d] for k in range(3)], 2).reshape(-1, 3, h, w)
    s1 = clock("pad + stack 1", stack1)
    first = clock("conv 1", lambda: F.conv2d(s1, f[0].weight, padding=1))
    v = clock("float()", lambda: first.float())
    clock("mean [0,2,3]", lambda: v.mean([0, 2, 3]))
    clock("var [0,2,3]", lambda: v.var([0, 2, 3], unbiased=False))
    clock("var_mean", lambda: torch.var_mean(v, dim=(0, 2, 3), unbiased=False))
    clock("sum + sum of squares", lambda: (v.sum([0, 2, 3]), (v * v).sum([0, 2, 3])))
    a = clock("scale, shift, leaky", lambda: F.leaky_relu(first * 1.5 + 0.1, 0.01))
    def stack2():
        p = F.pad(a.reshape(b, d, -1, h, w), (0, 0, 0, 0, 0, 0, 1, 1))
        return torch.cat([p[:, k:k+d] for k in range(3)], 2).reshape(b * d, -1, h, w)
    s2 = clock("pad + cat 2", stack2)
    clock("conv 2", lambda: F.conv2d(s2, f[3].weight, padding=1))
    s2c = s2.contiguous(memory_format=torch.channels_last); w2c = f[3].weight.contiguous(memory_format=torch.channels_last)
    clock("conv 2 channels_last", lambda: F.conv2d(s2c, w2c, padding=1))
