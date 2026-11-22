"""Reader-side profile on captured stacks (see profile_sampler.py). Run from a checkout root:
runtime/python/bin/python BENCH/profile_reader.py CACHE MODE [REF_DIR] [LIMIT]
MODE pipelined: the checkout's own TileReader.maps over every stack, wall time; with REF_DIR, maps are saved (first run)
or compared against the saved ones (later runs).  MODE serial: the same work one tile at a time with sync points, per op."""

import collections
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
cache, mode = Path(sys.argv[1]), sys.argv[2]
ref = Path(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[3] != "-" else None
limit = int(sys.argv[4]) if len(sys.argv) > 4 else None

from ink.model import models, tiles  # noqa: E402
from ink.model.reader import lane_reader  # noqa: E402

files = sorted(cache.glob("*.pkl"))[:limit]
t0 = time.perf_counter()
pieces = [pickle.load(open(f, "rb")) for f in files]
print(f"loaded {len(pieces)} pieces in {time.perf_counter() - t0:.1f} s", flush=True)


def stacks_of(pieces):
    for piece in pieces:
        for stack, measure in piece["stacks"]:
            yield tiles.PreparedStack(*stack), measure


count = sum(len(p["stacks"]) for p in pieces)
t0 = time.perf_counter()
reader = lane_reader(models.chosen().weights)
print(f"reader ready in {time.perf_counter() - t0:.1f} s", flush=True)

if mode == "pipelined":
    import threading

    spans, lock = [], threading.Lock()
    if hasattr(reader, "gpu_engine") and os.environ.get("HX_LANE_LOG"):
        inner = reader.ink_of

        def logged(body, features):
            t = time.perf_counter()
            out = inner(body, features)
            with lock:
                spans.append(("G" if body is not reader.engine else "E", t, time.perf_counter()))
            return out

        reader.ink_of = logged
        stem_tiles = reader.stem.tiles

        def timed_tiles(*a):
            t = time.perf_counter()
            for item in stem_tiles(*a):
                item = item.cpu().numpy()
                with lock:
                    spans.append(("S", t, time.perf_counter()))
                yield __import__("torch").from_numpy(item)
                t = time.perf_counter()

        reader.stem.tiles = timed_tiles
    names = [f"{f.stem}-{k}" for f, p in zip(files, pieces) for k in range(len(p["stacks"]))]
    worst, worst_r, n, px = 0.0, 1.0, 0, 0
    sums = np.zeros(5)
    t0 = time.perf_counter()
    maps = []
    for name, read in zip(names, reader.maps(stacks_of(pieces))):
        if isinstance(read, Exception):
            raise read
        maps.append((name, read))
    wall = time.perf_counter() - t0
    for name, read in maps:
        px += read.size
        if ref is None:
            continue
        path = ref / f"{name}.npy"
        if path.exists():
            want = np.load(path)
            d = float(np.abs(read - want).max())
            worst = max(worst, d)
            a, b = read.ravel().astype(np.float64), want.ravel().astype(np.float64)
            sums += [a.sum(), b.sum(), (a * a).sum(), (b * b).sum(), (a * b).sum()]
            n += a.size
            if a.std() > 0 and b.std() > 0:
                worst_r = min(worst_r, float(np.corrcoef(a, b)[0, 1]))
        else:
            ref.mkdir(parents=True, exist_ok=True)
            np.save(path, read)
    out = {"mode": mode, "pieces": len(pieces), "stacks": count, "wall_s": wall, "Mpx": px / 1e6}
    for lane in "ESG":
        mine = [(a, b) for name, a, b in spans if name == lane]
        if mine:
            out[f"lane_{lane}"] = {"calls": len(mine), "busy_s": round(sum(b - a for a, b in mine), 2), "mean_ms": round(1000 * sum(b - a for a, b in mine) / len(mine), 2)}
    if n:
        sa, sb, saa, sbb, sab = sums
        r = (sab - sa * sb / n) / np.sqrt((saa - sa * sa / n) * (sbb - sb * sb / n))
        out |= {"max_abs_diff": worst, "pearson_all_pixels": float(r), "worst_map_pearson": worst_r, "compared_px": n}
    print(json.dumps(out))
    sys.exit(0)

# serial, instrumented
import torch  # noqa: E402

from ink.model.reader_apple.neural_engine import SIDE  # noqa: E402
from ink.model.reader_apple.tile_reader import tile_groups  # noqa: E402
from ink.model.tiles import Sums, tile_starts  # noqa: E402

T, N = collections.defaultdict(float), collections.Counter()


class clock:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        self.t = time.perf_counter()

    def __exit__(self, *_):
        T[self.name] += time.perf_counter() - self.t
        N[self.name] += 1


sync = torch.mps.synchronize
stem = reader.stem
wall0 = time.perf_counter()
with torch.inference_mode():
    for stack, measure in stacks_of(pieces):
        _, height, width = stack.shape
        with clock("Sums alloc (2 float maps + hann)"):
            sums = Sums(height, width)
        if measure is not None:
            with clock("stem statistics_of (norm stats on MPS)"):
                reader.fixed = stem.statistics_of(*measure)
                sync()
        ty, tx = sums.tile_y, sums.tile_x
        whole, coverage = stack
        on_engine = (ty, tx) == (SIDE, SIDE)
        pair = on_engine and height >= ty
        xs = tile_starts(width, tx)
        for y in tile_starts(height, ty):
            with clock("coverage any() per tile"):
                active = [x for x in xs if coverage[y : y + ty, x : x + tx].any()]
            N["tiles skipped (no coverage)"] += len(xs) - len(active)
            for group in tile_groups(active, pair):
                x, last = group[0], group[-1]
                with clock("block slice/pad"):
                    block = whole[:, y : y + ty, x : last + tx]
                    h, w = block.shape[1:]
                    if h < ty or w < tx:
                        block = np.pad(block, ((0, 0), (0, ty - h), (0, tx - w)))
                    block = np.ascontiguousarray(block, dtype=np.uint8)
                N["groups pair" if len(group) == 2 else "groups single"] += 1
                gen = stem.tiles(block, reader.fixed, last - x)
                for at in group:
                    with clock("stem Metal kernel (upload+dispatch+GPU wait)"):
                        features = next(gen)
                        sync()
                    place = (y, at, h, min(w, tx))
                    if on_engine:
                        with clock("features .cpu().numpy() (16 MB fp16 copy)"):
                            f = features.cpu().numpy()
                        with clock("Neural Engine body call"):
                            logits = reader.engine(f)
                        with clock("sigmoid (numpy exp)"):
                            read = 1 / (1 + np.exp(-logits))
                        N["tiles on engine"] += 1
                    else:
                        with clock("torch body on MPS (small tiles) + sigmoid + copy"):
                            read = torch.sigmoid(reader.body(features)["ink"].float())[0, 0].cpu().numpy()
                        N["tiles on torch body"] += 1
                        N[f"  torch tile {ty}x{tx}"] += 1
                    with clock("Sums.add (hann blend)"):
                        sums.add(*place, read)
                for _ in gen:
                    pass
        with clock("Sums.result (divide)"):
            sums.result()
wall = time.perf_counter() - wall0
print(f"\nserial reader wall {wall:.1f} s, {len(pieces)} pieces, {count} stacks")
for k in sorted(T, key=T.get, reverse=True):
    print(f"{T[k]:9.2f} s  {100 * T[k] / wall:5.1f} %  n={N[k]:6d}  {1000 * T[k] / max(N[k], 1):8.2f} ms each  {k}")
print({k: v for k, v in N.items() if k not in T})
(cache / "reader_profile_serial.json").write_text(json.dumps({"wall": wall, "ops": dict(T), "counts": dict(N)}, indent=1))
