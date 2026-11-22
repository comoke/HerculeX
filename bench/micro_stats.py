"""Where statistics_of's 40 ms goes. Run from a checkout root."""
import os, pickle, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, os.getcwd())
import torch
from ink.model import models
from ink.model.reader import lane_reader
from ink.model.reader_apple.tile_reader import stem_of
from ink.model.tiles import scaled

cache = Path(sys.argv[1])
piece = pickle.load(open(sorted(cache.glob("*.pkl"))[0], "rb"))
measure = next(m for _, m in piece["stacks"] if m is not None)
reader = lane_reader(models.chosen().weights)
stem, sync = reader.stem, torch.mps.synchronize
tiles, scale = measure
print("measure tiles", tiles.shape, tiles.dtype, flush=True)
T = {}
def clock(name, fn, n=20):
    for _ in range(3): fn()
    sync(); t = time.perf_counter()
    for _ in range(n): out = fn()
    sync(); T[name] = 1000 * (time.perf_counter() - t) / n
    return out
with torch.inference_mode():
    f = stem.module.features
    clock("whole statistics_of", lambda: stem.statistics_of(tiles, scale))
    image = clock("upload + scaled", lambda: scaled(torch.from_numpy(tiles).to("mps"), scale)[:, None])
    first = clock("conv 1 (folded, fp16, [136,3,128,128])", lambda: f[0](image))
    clock("measure 1 (float, mean, var)", lambda: stem.norms[0].measure(first))
    a = clock("norm 1 + leaky", lambda: f[2](stem.norms[0](first)))
    second = clock("conv 2 (folded, [136,48,128,128])", lambda: f[3](a))
    clock("measure 2", lambda: stem.norms[1].measure(second))
with torch.no_grad():
    clock("stem_of (fold weights, 5 tensors)", lambda: stem_of(stem.module))
for k, v in T.items():
    print(f"{v:8.2f} ms  {k}")
