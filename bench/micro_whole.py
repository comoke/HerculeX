"""Stock's whole-network Core ML model (stem + body, EngineReader's) timed on each compute unit. Run from a checkout root."""

import ctypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.getcwd())
from ink.model import models, stock_model
from ink.model.reader_apple import neural_engine as ne
from ink.model.tiles import DEPTH, STEM_CHANNELS

weights = models.chosen().weights
path = ne.package_of(weights).with_name(ne.package_of(weights).stem + "-whole.mlpackage")
if not path.exists():
    t = time.perf_counter()
    net = (
        stock_model.InkModel(names=tuple(stock_model.CHECKPOINTS)[:1], weights=[weights], device="cpu")
        .models[0]
        .model.model.requires_grad_(False)
    )
    ne.compile_whole(net.depth_fusion.float().eval(), net.network.float().eval(), path)
    print(f"converted in {time.perf_counter() - t:.0f} s", flush=True)
calls = ne.engine_calls()
rng = np.random.default_rng(0)
image = rng.standard_normal((1, DEPTH, ne.SIDE, ne.SIDE)).astype(np.float16)
norms = np.concatenate([np.ones(16), np.zeros(16), np.ones(16), np.zeros(16)]).astype(np.float16)
for name, units in {"cpuAndNeuralEngine": 3, "cpuAndGPU": 1, "all": 2}.items():
    for special in (0, 1):
        t = time.perf_counter()
        engine = calls.hx_engine_open(str(path).encode(), str(path.with_suffix(".mlmodelc")).encode(), units, special)
        opened = time.perf_counter() - t

        def run(_, engine=engine):
            logits = np.empty((ne.SIDE, ne.SIDE), dtype=np.float16)
            failed = calls.hx_engine_whole(
                ctypes.c_void_p(engine),
                image.ctypes.data,
                DEPTH,
                norms.ctypes.data,
                STEM_CHANNELS,
                ne.SIDE,
                logits.ctypes.data,
            )
            assert not failed, failed
            return logits

        t = time.perf_counter()
        run(0)
        first = time.perf_counter() - t
        for _ in range(3):
            run(0)
        line = f"whole {name:20s} specialized={special} open {opened:5.1f} s first {first:6.2f} s |"
        for workers in (1, 2, 3):
            n = 30
            with ThreadPoolExecutor(workers) as pool:
                t = time.perf_counter()
                list(pool.map(run, range(n)))
                dt = time.perf_counter() - t
            line += f"  {workers} at once: {1000 * dt / n:6.2f} ms/tile"
        print(line, flush=True)
