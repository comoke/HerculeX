"""The body on real stem features: speed of PyTorch MPS (NCHW vs channels_last, fp16) and how close each path's ink
probabilities are to the float32 network on CPU. Run from a checkout root."""

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.getcwd())
import torch  # noqa: E402

from ink.model import models  # noqa: E402
from ink.model.reader import lane_reader  # noqa: E402
from ink.model.reader_apple.neural_engine import EngineBody  # noqa: E402

cache = Path(sys.argv[1])
weights = models.chosen().weights
reader = lane_reader(weights)
stem, sync = reader.stem, torch.mps.synchronize
feats = []
with torch.inference_mode():
    for f in sorted(cache.glob("*.pkl"))[:12:3]:
        piece = pickle.load(open(f, "rb"))
        (values, coverage), _ = piece["stacks"][-1]
        measure = next(m for _, m in piece["stacks"] if m is not None)
        fixed = stem.statistics_of(*measure)
        if values.shape[1] >= 512 and values.shape[2] >= 512:
            y, x = (values.shape[1] - 512) // 2, (values.shape[2] - 512) // 2
            block = np.ascontiguousarray(values[:, y : y + 512, x : x + 512])
            feats.append(next(iter(stem.tiles(block, fixed, 0))).cpu().numpy())
print(len(feats), "real feature tiles", flush=True)

sig = lambda z: 1 / (1 + np.exp(-z.astype(np.float32)))  # noqa: E731
body = reader.body
with torch.inference_mode():
    gold_net = __import__("copy").deepcopy(body).float().cpu().eval()
    t = time.perf_counter()
    gold = [sig(gold_net(torch.from_numpy(f.astype(np.float32)))["ink"][0, 0].numpy()) for f in feats]
    print(f"float32 CPU network: {1000 * (time.perf_counter() - t) / len(feats):.0f} ms per tile", flush=True)
    results = {}
    for name, units in {"Core ML Neural Engine (stock)": 3, "Core ML GPU": 1}.items():
        engine = EngineBody(weights, units=units)
        engine(feats[0])
        results[name] = [sig(engine(f)) for f in feats]
    x = [torch.from_numpy(f).to("mps") for f in feats]

    def timed(label, net, inputs, n=20):
        for _ in range(3):
            net(inputs[0])["ink"]
        sync()
        t = time.perf_counter()
        for k in range(n):
            out = net(inputs[k % len(inputs)])["ink"]
        sync()
        print(f"{label:55s} {1000 * (time.perf_counter() - t) / n:7.2f} ms per tile", flush=True)

    timed("torch MPS body fp16 NCHW (stock small-tile path)", body, x)
    results["torch MPS fp16 NCHW"] = [sig(body(v)["ink"].float()[0, 0].cpu().numpy()) for v in x]
    body_cl = __import__("copy").deepcopy(body).to(memory_format=torch.channels_last)
    xc = [v.contiguous(memory_format=torch.channels_last) for v in x]
    timed("torch MPS body fp16 channels_last", body_cl, xc)
    results["torch MPS fp16 channels_last"] = [sig(body_cl(v)["ink"].float()[0, 0].cpu().numpy()) for v in xc]
    body32 = __import__("copy").deepcopy(body).float()
    x32 = [v.float() for v in x]
    timed("torch MPS body fp32 NCHW", body32, x32, n=10)
    results["torch MPS fp32 NCHW"] = [sig(body32(v)["ink"][0, 0].cpu().numpy()) for v in x32]
    for batch in (2, 4):
        xb = [torch.cat(xc[:batch] if len(xc) >= batch else xc * batch)[:batch].contiguous(memory_format=torch.channels_last)]
        for _ in range(2):
            body_cl(xb[0])["ink"]
        sync()
        t = time.perf_counter()
        for _ in range(10):
            body_cl(xb[0])["ink"]
        sync()
        print(f"{'torch MPS body fp16 channels_last, batch ' + str(batch):55s} {1000 * (time.perf_counter() - t) / 10 / batch:7.2f} ms per tile", flush=True)

g = np.concatenate([m.ravel() for m in gold])
print("\nagreement of ink probabilities with the float32 CPU network (and with stock's engine):")
stock = np.concatenate([m.ravel() for m in results["Core ML Neural Engine (stock)"]])
for name, maps in results.items():
    v = np.concatenate([m.ravel() for m in maps])
    print(
        f"{name:32s} vs fp32: max abs {np.abs(v - g).max():.4f} mean abs {np.abs(v - g).mean():.5f} r {np.corrcoef(v, g)[0, 1]:.7f}"
        f" | vs stock engine: max abs {np.abs(v - stock).max():.4f} r {np.corrcoef(v, stock)[0, 1]:.7f}"
    )
