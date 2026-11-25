"""Where the stem's 11.5 ms per tile goes: GPU compute or per-dispatch overhead. Run from a checkout root."""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import torch

from ink.model import models
from ink.model.reader import lane_reader

cache = Path(sys.argv[1])
piece = pickle.load(open(min(cache.glob("*.pkl")), "rb"))
(values, coverage), _ = piece["stacks"][-1]
measure = next(m for _, m in piece["stacks"] if m is not None)
print("stack", values.shape, flush=True)
reader = lane_reader(models.chosen().weights)
stem = reader.stem
sync = torch.mps.synchronize
with torch.inference_mode():
    fixed = stem.statistics_of(*measure)
    sync()
    block = np.ascontiguousarray(values[:, :512, :512])
    for _ in range(3):
        list(stem.tiles(block, fixed, 0))
    sync()

    def bench(label, fn, n):
        sync()
        t = time.perf_counter()
        fn()
        sync()
        dt = time.perf_counter() - t
        print(f"{label:70s} {1000 * dt / n:7.2f} ms per tile", flush=True)

    n = 40
    bench(
        "single tile: upload + dispatch + sync each (as profiled)",
        lambda: [(list(stem.tiles(block, fixed, 0)), sync()) for _ in range(n)],
        n,
    )
    bench(
        "single tile: upload + dispatch, one sync at the end",
        lambda: [list(stem.tiles(block, fixed, 0)) for _ in range(n)],
        n,
    )
    image = torch.from_numpy(block).to("mps")[None]
    table = stem.table
    bench(
        "dispatch only (image already on GPU), one sync at the end",
        lambda: [stem.features(stem.raw_kernel, image, fixed, table) for _ in range(n)],
        n,
    )
    for batch in (2, 4, 8):
        images = image.repeat(batch, 1, 1, 1).contiguous()
        sync()
        bench(
            f"batch {batch} in one dispatch, one sync at the end",
            lambda images=images, batch=batch: [
                stem.features(stem.raw_kernel, images, fixed, table) for _ in range(n // batch)
            ],
            n // batch * batch,
        )
    f = stem.features(stem.raw_kernel, image, fixed, table)
    sync()
    bench("features .cpu().numpy() alone", lambda: [f.cpu().numpy() for _ in range(n)], n)
    bench(
        "upload alone (torch.from_numpy(block).to(mps))",
        lambda: [torch.from_numpy(block).to("mps") for _ in range(n)],
        n,
    )
    bench("statistics_of", lambda: [stem.statistics_of(*measure) for _ in range(10)], 10)
    # pair path: one wide dispatch + two edge fixes
    wide = np.ascontiguousarray(values[:, :512, : 512 + 280]) if values.shape[2] >= 792 else None
    if wide is not None:
        bench(
            "pair (792 wide + 2 edge kernels), per tile",
            lambda: [list(stem.tiles(wide, fixed, 280)) for _ in range(n // 2)],
            n,
        )
