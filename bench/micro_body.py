"""How fast each compute unit runs the reader's 2-D body on a 512 tile, alone and with concurrent calls.
Run from a checkout root: runtime/python/bin/python BENCH/micro_body.py [CALLS]"""

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.getcwd())
from ink.model import models  # noqa: E402
from ink.model.reader_apple.neural_engine import CHANNELS, SIDE, EngineBody  # noqa: E402

calls = int(sys.argv[1]) if len(sys.argv) > 1 else 40
weights = models.chosen().weights
rng = np.random.default_rng(0)
features = [rng.standard_normal((1, CHANNELS, SIDE, SIDE)).astype(np.float16) for _ in range(4)]
UNITS = {"cpuAndNeuralEngine": 3, "cpuAndGPU": 1, "all": 2, "cpuOnly": 0}
outputs = {}
for name, units in UNITS.items():
    if name == "cpuOnly" and "--cpu" not in sys.argv:
        continue
    t = time.perf_counter()
    body = EngineBody(weights, units=units)
    opened = time.perf_counter() - t
    t = time.perf_counter()
    outputs[name] = body(features[0])
    first = time.perf_counter() - t
    for _ in range(3):
        body(features[0])
    line = f"{name:20s} open {opened:5.1f} s, first call {first * 1000:7.1f} ms |"
    for workers in (1, 2, 3, 4):
        n = calls if name != "cpuOnly" else 6
        with ThreadPoolExecutor(workers) as pool:
            t = time.perf_counter()
            list(pool.map(lambda k: body(features[k % 4]), range(n)))
            dt = time.perf_counter() - t
        line += f"  {workers} at once: {1000 * dt / n:6.2f} ms/tile"
    print(line, flush=True)
base = outputs["cpuAndNeuralEngine"]
for name, out in outputs.items():
    d = np.abs(out - base)
    print(f"{name:20s} vs engine logits: max abs diff {d.max():.4f}, mean {d.mean():.5f}, r {np.corrcoef(out.ravel(), base.ravel())[0, 1]:.8f}")
